import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from nselib import capital_market

# --- Page Layout Configuration ---
st.set_page_config(page_title="RISHI's Multi-Asset Dashboard", page_icon="📊", layout="wide")

def safe_price(val):
    try:
        return float(str(val).replace(',', '').strip())
    except:
        return 0.0

# --- Automated Lot Size Fetcher (For Futures Mode) ---
@st.cache_data(ttl=3600)
def get_fno_lot_sizes():
    """Fetches live F&O stock list from NSE and maps symbols to lot sizes."""
    try:
        df = capital_market.fno_equity_list()
        df.columns = [str(col).strip().upper() for col in df.columns]
        if 'SYMBOL' in df.columns and 'LOT SIZE' in df.columns:
            return dict(zip(df['SYMBOL'], pd.to_numeric(df['LOT SIZE'], errors='coerce').fillna(250).astype(int)))
    except Exception:
        pass
    return {}

def lookup_lot_size(symbol, lot_dict):
    clean_sym = str(symbol).strip().upper()
    return lot_dict.get(clean_sym, 250)

# --- ATR Calculation (For Intraday Cash Mode) ---
def get_atr(symbol, period=14):
    """Fetches last 30 days of daily OHLC and returns 14-period ATR."""
    try:
        ticker = symbol.upper()
        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"
        df = yf.download(ticker, period="30d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < period + 1:
            return None
        df = df.copy()
        df["prev_close"] = df["Close"].shift(1)
        df["tr"] = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["prev_close"]),
                abs(df["Low"]  - df["prev_close"])
            )
        )
        atr = df["tr"].iloc[-period:].mean()
        return round(float(atr), 2)
    except Exception:
        return None

# --- Pre-load Data ---
with st.spinner("Initializing NSE Live Lot Directories..."):
    nse_lot_sizes = get_fno_lot_sizes()

# --- Time Calculation ---
UTC_NOW = datetime.utcnow()
IST_NOW = UTC_NOW + timedelta(hours=5, minutes=30)

# --- SIDEBAR CONTROL: Mood / Segment Selector ---
st.sidebar.header("🕹️ Control Center")
trading_mode = st.sidebar.radio(
    "Choose Your Trading Mode:",
    ["📈 Intraday Cash (Shares)", "🔥 Stock Futures (Lots)"],
    help="Toggle between trading individual equity shares or standardized futures contracts."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Capital & Risk Settings")

if trading_mode == "📈 Intraday Cash (Shares)":
    capital = st.sidebar.number_input("Trading Capital (₹)", value=30000, step=1000)
    leverage = st.sidebar.number_input("MIS Leverage (x)", value=5, min_value=1, max_value=5)
    max_risk = st.sidebar.number_input("Max Risk Per Trade (₹)", value=300, step=10)
    buying_power = capital * leverage
    st.sidebar.info(f"Total Buying Power: **₹{buying_power:,}**")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 ATR Settings")
    atr_period = st.sidebar.number_input("ATR Period (candles)", value=14, min_value=5, max_value=50)
    atr_multiplier = st.sidebar.number_input("ATR Multiplier", value=1.5, min_value=0.5, max_value=5.0, step=0.1)

else:  # Futures Mode
    capital = st.sidebar.number_input("Trading Margin (₹)", value=150000, step=10000)
    sl_pct = st.sidebar.number_input("Stop Loss (%)", value=0.5, step=0.1) / 100
    tgt_pct = st.sidebar.number_input("Profit Target (%)", value=1.0, step=0.1) / 100
    st.sidebar.info(f"Target Risk-Reward: **1 : {round(tgt_pct / sl_pct, 1)}**")

# --- MAIN PAGE HEADER ---
st.title("📊 RISHI's Multi-Asset Momentum Dashboard")
st.caption(f"Engine Mode: **{trading_mode.upper()}** • System Time: {IST_NOW.strftime('%d %b %Y | %H:%M IST')}")
st.markdown("---")

# --- Live Market Feed Section (Shared by both modes) ---
st.subheader("📡 Nifty Live Market Feed")
auto_bullish_stock, auto_bullish_ltp = "no stock", 0.0
auto_bearish_stock, auto_bearish_ltp = "no stock", 0.0

tab_auto, tab_manual = st.tabs(["🤖 Automated Nifty Scanner", "✍️ Manual Entry Override"])

with tab_auto:
    try:
        gainers_df = capital_market.top_gainers_or_losers(to_get='gainers')
        losers_df  = capital_market.top_gainers_or_losers(to_get='loosers')
        auto_bullish_stock = gainers_df.iloc[0]['symbol']
        auto_bullish_ltp   = safe_price(gainers_df.iloc[0]['ltp'])
        auto_bearish_stock = losers_df.iloc[0]['symbol']
        auto_bearish_ltp   = safe_price(losers_df.iloc[0]['ltp'])
        st.success(f"✅ Live NSE Data Connected! Top Gainer: {auto_bullish_stock} | Top Loser: {auto_bearish_stock}")
    except Exception as e:
        st.error("⚠️ NSE Data Feed is busy or closed. Use the 'Manual Entry Override' tab.")

with tab_manual:
    st.caption("Manual override feed controls:")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        man_bullish     = st.text_input("Top Gainer Symbol", value=auto_bullish_stock)
        man_bullish_ltp = st.number_input("Gainer Live Price (₹)", value=auto_bullish_ltp, step=0.05)
    with col_m2:
        man_bearish     = st.text_input("Top Loser Symbol", value=auto_bearish_stock)
        man_bearish_ltp = st.number_input("Loser Live Price (₹)", value=auto_bearish_ltp, step=0.05)

bullish_stock = man_bullish
bullish_ltp   = man_bullish_ltp
bearish_stock = man_bearish
bearish_ltp   = man_bearish_ltp

st.markdown("---")

# ==========================================
# MODULE 1: INTRADAY CASH ENGINE
# ==========================================
if trading_mode == "📈 Intraday Cash (Shares)":
    with st.spinner("Fetching historical daily ATR from Yahoo Finance..."):
        bull_atr = get_atr(bullish_stock, period=atr_period)
        bear_atr = get_atr(bearish_stock, period=atr_period)

    # --- Calculations: Long ---
    long_entry = round(bullish_ltp * 1.002, 2)
    if bull_atr:
        atr_sl_dist = round(bull_atr * atr_multiplier, 2)
        long_sl = round(long_entry - atr_sl_dist, 2)
        long_risk = round(long_entry - long_sl, 2)
        sl_method_long = f"ATR × {atr_multiplier} (₹{atr_sl_dist})"
    else:
        long_sl = round(long_entry * 0.995, 2)
        long_risk = round(long_entry - long_sl, 2)
        sl_method_long = "Fallback: 0.5% Rule"

    long_qty = min(int(max_risk // long_risk), int(buying_power // long_entry)) if long_risk > 0 else 1
    long_target1 = round(long_entry + (long_risk * 2), 2)
    long_target2 = round(long_entry + (long_risk * 3), 2)

    # --- Calculations: Short ---
    short_entry = round(bearish_ltp * 0.998, 2)
    if bear_atr:
        atr_sl_dist_s = round(bear_atr * atr_multiplier, 2)
        short_sl = round(short_entry + atr_sl_dist_s, 2)
        short_risk = round(short_sl - short_entry, 2)
        sl_method_short = f"ATR × {atr_multiplier} (₹{atr_sl_dist_s})"
    else:
        short_sl = round(short_entry * 1.005, 2)
        short_risk = round(short_sl - short_entry, 2)
        sl_method_short = "Fallback: 0.5% Rule"

    short_qty = min(int(max_risk // short_risk), int(buying_power // short_entry)) if short_risk > 0 else 1
    short_target1 = round(short_entry - (short_risk * 2), 2)
    short_target2 = round(short_entry - (short_risk * 3), 2)

    # --- Render: Long Setup ---
    st.success(f"### 📈 INTRADAY CASH LONG: {bullish_stock}")
    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Entry Trigger", f"₹{long_entry}")
    l_col2.metric(f"Stop Loss ({sl_method_long})", f"₹{long_sl}", delta=f"-₹{long_risk}/sh", delta_color="inverse")
    l_col3.metric("Targets (1:2 | 1:3)", f"₹{long_target1} / ₹{long_target2}")
    l_col4.metric("Max Account Loss", f"₹{round(long_risk * long_qty, 2):,}", delta="Calculated Risk", delta_color="inverse")
    st.caption(f"💡 Trade Size: **{long_qty} shares** | Target 1 Profit potential: **₹{round(long_risk * 2 * long_qty, 2):,}** | Margin Utilized: ~₹{round((long_entry * long_qty)/leverage, 2):,}")

    st.markdown("---")

    # --- Render: Short Setup ---
    st.error(f"### 📉 INTRADAY CASH SHORT: {bearish_stock}")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Entry Trigger", f"₹{short_entry}")
    s_col2.metric(f"Stop Loss ({sl_method_short})", f"₹{short_sl}", delta=f"+₹{short_risk}/sh", delta_color="inverse")
    s_col3.metric("Targets (1:2 | 1:3)", f"₹{short_target1} / ₹{short_target2}")
    s_col4.metric("Max Account Loss", f"₹{round(short_risk * short_qty, 2):,}", delta="Calculated Risk", delta_color="inverse")
    st.caption(f"💡 Trade Size: **{short_qty} shares** | Target 1 Profit potential: **₹{round(short_risk * 2 * short_qty, 2):,}** | Margin Utilized: ~₹{round((short_entry * short_qty)/leverage, 2):,}")

# ==========================================
# MODULE 2: STOCK FUTURES ENGINE
# ==========================================
else:
    bull_lot_size = lookup_lot_size(bullish_stock, nse_lot_sizes)
    bear_lot_size = lookup_lot_size(bearish_stock, nse_lot_sizes)

    # --- Calculations: Long Futures ---
    long_entry  = round(bullish_ltp * 1.002, 2)
    long_sl     = round(long_entry * (1 - sl_pct), 2)
    long_risk   = round(long_entry - long_sl, 2)
    long_target = round(long_entry * (1 + tgt_pct), 2)
    long_reward = round(long_target - long_entry, 2)
    max_loss_long   = round(long_risk * bull_lot_size, 2)
    max_profit_long = round(long_reward * bull_lot_size, 2)

    # --- Calculations: Short Futures ---
    short_entry  = round(bearish_ltp * 0.998, 2)
    short_sl     = round(short_entry * (1 + sl_pct), 2)
    short_risk   = round(short_sl - short_entry, 2)
    short_target = round(short_entry * (1 - tgt_pct), 2)
    short_reward = round(short_entry - short_target, 2)
    max_loss_short   = round(short_risk * bear_lot_size, 2)
    max_profit_short = round(short_reward * bear_lot_size, 2)

    # --- Render: Long Futures Setup ---
    st.success(f"### 📈 STOCK FUTURES LONG: {bullish_stock}")
    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Trigger Entry", f"₹{long_entry}")
    l_col2.metric("Stop Loss (0.5%)", f"₹{long_sl}", delta=f"-₹{long_risk}/sh", delta_color="inverse")
    l_col3.metric("Target (1.0%)", f"₹{long_target}", delta=f"+₹{long_reward}/sh")
    l_col4.metric("Max Loss Risk", f"₹{max_loss_long:,}", delta="1 Contract Lot", delta_color="inverse")
    st.caption(f"📋 **Contract Details:** Auto Lot Size: **{bull_lot_size}** units | Potential Profit: **₹{max_profit_long:,}** | Margin Check: Ensure account can accommodate contract span constraints.")

    st.markdown("---")

    # --- Render: Short Futures Setup ---
    st.error(f"### 📉 STOCK FUTURES SHORT: {bearish_stock}")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Trigger Entry", f"₹{short_entry}")
    s_col2.metric("Stop Loss (0.5%)", f"₹{short_sl}", delta=f"+₹{short_risk}/sh", delta_color="inverse")
    s_col3.metric("Target (1.0%)", f"₹{short_target}", delta=f"-₹{short_reward}/sh")
    s_col4.metric("Max Loss Risk", f"₹{max_loss_short:,}", delta="1 Contract Lot", delta_color="inverse")
    st.caption(f"📋 **Contract Details:** Auto Lot Size: **{bear_lot_size}** units | Potential Profit: **₹{max_profit_short:,}** | Margin Check: Ensure account can accommodate contract span constraints.")

# --- Shared Footer Guardrails ---
st.markdown("---")
st.warning("⚠️ **Execution Guardrail:** Always deploy these setups using **SL-Limit (MIS/NRML)** orders directly within your broker terminal. Do not use market orders to avoid execution slippage.")
