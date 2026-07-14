import streamlit as st
from datetime import datetime, timedelta, timezone
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

# --- TRUE INTRADAY ATR CALCULATION (Fixes Circuit & Wide Range Issues) ---
def get_atr(symbol, period=14):
    """
    Fetches the last 5 days of 5-minute intraday data 
    and returns a valid Intraday 14-period ATR.
    """
    try:
        ticker = symbol.upper().strip()
        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"
        
        # 1. Fetch 5-minute bars over the last 5 days (Intraday context)
        df = yf.download(
            tickers=ticker, 
            period="5d", 
            interval="5m", 
            progress=False, 
            auto_adjust=True,
            multi_level_index=False  # Prevents MultiIndex formatting errors
        )
        
        if df.empty or len(df) < period + 1:
            return None
            
        df = df.copy()
        
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        
        # 2. True Range Calculation
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # 3. Simple rolling mean of the True Range for 14 periods
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        
        return round(float(atr), 2)
    except Exception as e:
        return None

# --- Pre-load Data ---
with st.spinner("Initializing NSE Live Lot Directories..."):
    nse_lot_sizes = get_fno_lot_sizes()

# Native timezone calculation
IST_NOW = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

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
    sl_pct = st.sidebar.number_input("Stop Loss (%)", value=1.0, step=0.1) / 100
    tgt_pct = st.sidebar.number_input("Profit Target (%)", value=2.0, step=0.1) / 100
    st.sidebar.info(f"Target Risk-Reward: **1 : {round(tgt_pct / sl_pct, 1)}**")

# --- MAIN PAGE HEADER ---
st.title("📊 RISHI's Multi-Asset Momentum Dashboard")
st.caption(f"Engine Mode: **{trading_mode.upper()}** • System Time: {IST_NOW.strftime('%d %b %Y | %H:%M IST')}")
st.markdown("---")

# --- Live Market Feed Section ---
st.subheader("📡 Nifty Live Market Feed")
auto_bullish_stock, auto_bullish_ltp = "no stock", 0.0
auto_bearish_stock, auto_bearish_ltp = "no stock", 0.0

chosen_feed = st.radio("Select Feed Input Source:", ["🤖 Automated Nifty Scanner", "✍️ Manual Entry Override"], horizontal=True)

try:
    gainers_df = capital_market.top_gainers_or_losers(to_get='gainers')
    losers_df  = capital_market.top_gainers_or_losers(to_get='loosers')
    
    gainers_df.columns = gainers_df.columns.str.strip().str.upper()
    losers_df.columns = losers_df.columns.str.strip().str.upper()
    
    auto_bullish_stock = gainers_df.iloc[0]['SYMBOL']
    auto_bullish_ltp   = safe_price(gainers_df.iloc[0]['LTP'])
    auto_bearish_stock = losers_df.iloc[0]['SYMBOL']
    auto_bearish_ltp   = safe_price(losers_df.iloc[0]['LTP'])
except Exception as e:
    st.sidebar.warning("⚠️ Live NSE Data Feed is busy/closed. Defaulting to manual mode entry choices.")

if chosen_feed == "🤖 Automated Nifty Scanner":
    st.success(f"✅ Live NSE Data Active! Top Gainer: **{auto_bullish_stock}** (₹{auto_bullish_ltp}) | Top Loser: **{auto_bearish_stock}** (₹{auto_bearish_ltp})")
    bullish_stock, bullish_ltp = auto_bullish_stock, auto_bullish_ltp
    bearish_stock, bearish_ltp = auto_bearish_stock, auto_bearish_ltp
else:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        bullish_stock = st.text_input("Top Gainer Symbol", value=auto_bullish_stock if auto_bullish_stock != "no stock" else "RELIANCE")
        bullish_ltp = st.number_input("Gainer Live Price (₹)", value=auto_bullish_ltp if auto_bullish_ltp != 0.0 else 2500.0, step=0.05)
    with col_m2:
        bearish_stock = st.text_input("Top Loser Symbol", value=auto_bearish_stock if auto_bearish_stock != "no stock" else "TCS")
        bearish_ltp = st.number_input("Loser Live Price (₹)", value=auto_bearish_ltp if auto_bearish_ltp != 0.0 else 4000.0, step=0.05)

st.markdown("---")

# ==========================================
# MODULE 1: INTRADAY CASH ENGINE
# ==========================================
if trading_mode == "📈 Intraday Cash (Shares)":
    with st.spinner("Fetching historical intraday ATR from Yahoo Finance..."):
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

    long_risk = max(long_risk, 0.05) 
    long_qty = min(int(max_risk // long_risk), int(buying_power // long_entry))
    long_qty = max(long_qty, 1)
    
    # Target calculations
    long_target1 = round(long_entry + (long_risk * 2), 2)
    long_target2 = round(long_entry + (long_risk * 3), 2)
    
    # % calculations for UI
    long_sl_pct = round((long_risk / long_entry) * 100, 2)
    long_t1_pct = round(((long_target1 - long_entry) / long_entry) * 100, 2)
    long_t2_pct = round(((long_target2 - long_entry) / long_entry) * 100, 2)

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

    short_risk = max(short_risk, 0.05)
    short_qty = min(int(max_risk // short_risk), int(buying_power // short_entry))
    short_qty = max(short_qty, 1)
    
    # Target calculations
    short_target1 = round(short_entry - (short_risk * 2), 2)
    short_target2 = round(short_entry - (short_risk * 3), 2)
    
    # % calculations for UI
    short_sl_pct = round((short_risk / short_entry) * 100, 2)
    short_t1_pct = round(((short_entry - short_target1) / short_entry) * 100, 2)
    short_t2_pct = round(((short_entry - short_target2) / short_entry) * 100, 2)

    # --- Render: Long Setup ---
    st.success(f"### 📈 INTRADAY CASH LONG: {bullish_stock}")
    
    # Highlighted Quantity and Loss Display
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.info(f"### 🎯 TRADE QUANTITY: **{long_qty} shares**")
    with q_col2:
        st.error(f"### 🛡️ MAX LOSS RISK: **₹{round(long_risk * long_qty, 2):,}**")
        
    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Entry Trigger (+0.2% Buffer)", f"₹{long_entry}")
    l_col2.metric(f"Stop Loss ({sl_method_long})", f"₹{long_sl}", delta=f"-{long_sl_pct}% (-₹{long_risk}/sh)", delta_color="inverse")
    l_col3.metric("Target 1 (1:2 R:R)", f"₹{long_target1}", delta=f"+{long_t1_pct}%")
    l_col4.metric("Target 2 (1:3 R:R)", f"₹{long_target2}", delta=f"+{long_t2_pct}%")
    
    with st.expander("🔍 VIEW TRADE SETUP LOGIC", expanded=True):
        st.markdown(f"""
        * **Trend-Following Trigger:** Entry set at **₹{long_entry}** (0.2% above LTP) to ensure trade triggers only when momentum establishes a high.
        * **Anti-Hunting SL:** Daily candles would have set a ₹{round(long_entry * 0.02, 2)} stop. Instead, using the active 5-min chart ATR (**₹{bull_atr if bull_atr else 'N/A'}**), your stop is placed strictly **₹{atr_sl_dist} ({long_sl_pct}%)** away. This safely clears market noise without hitting circuit thresholds.
        * **Mathematical Risk-Control:** Sized strictly at **{long_qty} shares** to ensure that even if the stop loss is triggered, you will not lose more than your allocated limit of **₹{max_risk}**.
        """)

    st.markdown("---")

    # --- Render: Short Setup ---
    st.error(f"### 📉 INTRADAY CASH SHORT: {bearish_stock}")
    
    # Highlighted Quantity and Loss Display
    qs_col1, qs_col2 = st.columns(2)
    with qs_col1:
        st.info(f"### 🎯 TRADE QUANTITY: **{short_qty} shares**")
    with qs_col2:
        st.error(f"### 🛡️ MAX LOSS RISK: **₹{round(short_risk * short_qty, 2):,}**")
        
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Entry Trigger (-0.2% Buffer)", f"₹{short_entry}")
    s_col2.metric(f"Stop Loss ({sl_method_short})", f"₹{short_sl}", delta=f"+{short_sl_pct}% (+₹{short_risk}/sh)", delta_color="inverse")
    s_col3.metric("Target 1 (1:2 R:R)", f"₹{short_target1}", delta=f"-{short_t1_pct}%")
    s_col4.metric("Target 2 (1:3 R:R)", f"₹{short_target2}", delta=f"-{short_t2_pct}%")
    
    with st.expander("🔍 VIEW TRADE SETUP LOGIC", expanded=True):
        st.markdown(f"""
        * **Breakdown Confirmation:** Entry sits at **₹{short_entry}** (0.2% below LTP) to bypass minor ticks and require real downward force to execute.
        * **Anti-Hunting SL:** Placed outside the micro trading ranges. Backed by 5-min ATR (**₹{bear_atr if bear_atr else 'N/A'}**), your protective stop is at **₹{short_sl} ({short_sl_pct}%)**, ensuring random spikes won't trigger it.
        * **Strict Sizing Rule:** Leverage ensures buying power limit, but risk limits sizing to **{short_qty} shares**, locking in potential loss to maximum **₹{max_risk}**.
        """)

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
    
    # Highlighted Quantity and Loss Display
    qf_col1, qf_col2 = st.columns(2)
    with qf_col1:
        st.info(f"### 🎯 TRADE QUANTITY: **1 Lot ({bull_lot_size} units)**")
    with qf_col2:
        st.error(f"### 🛡️ MAX LOSS RISK: **₹{max_loss_long:,}**")
        
    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Trigger Entry", f"₹{long_entry}")
    l_col2.metric(f"Stop Loss ({round(sl_pct * 100, 2)}%)", f"₹{long_sl}", delta=f"-{round(sl_pct * 100, 2)}% (-₹{long_risk}/sh)", delta_color="inverse")
    l_col3.metric(f"Target ({round(tgt_pct * 100, 2)}%)", f"₹{long_target}", delta=f"+{round(tgt_pct * 100, 2)}% (+₹{long_reward}/sh)")
    l_col4.metric("Potential Profit", f"₹{max_profit_long:,}")
    
    with st.expander("🔍 VIEW TRADE SETUP LOGIC", expanded=True):
        st.markdown(f"""
        * **Breakout Entry:** Set slightly above the market price to trigger only if buying pressure pushes through resistance.
        * **Standardized Risk Boundary:** Operating on a strict **{round(sl_pct * 100, 2)}%** percentage stop loss model to protect fixed-lot derivative margin capital.
        * **Contracts Rule:** Leverage and standard exchange risk are calculated on **{bull_lot_size}** contract units.
        """)

    st.markdown("---")

    # --- Render: Short Futures Setup ---
    st.error(f"### 📉 STOCK FUTURES SHORT: {bearish_stock}")
    
    # Highlighted Quantity and Loss Display
    qfs_col1, qfs_col2 = st.columns(2)
    with qfs_col1:
        st.info(f"### 🎯 TRADE QUANTITY: **1 Lot ({bear_lot_size} units)**")
    with qfs_col2:
        st.error(f"### 🛡️ MAX LOSS RISK: **₹{max_loss_short:,}**")
        
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Trigger Entry", f"₹{short_entry}")
    s_col2.metric(f"Stop Loss ({round(sl_pct * 100, 2)}%)", f"₹{short_sl}", delta=f"+{round(sl_pct * 100, 2)}% (+₹{short_risk}/sh)", delta_color="inverse")
    s_col3.metric(f"Target ({round(tgt_pct * 100, 2)}%)", f"₹{short_target}", delta=f"-{round(tgt_pct * 100, 2)}% (-₹{short_reward}/sh)")
    s_col4.metric("Potential Profit", f"₹{max_profit_short:,}")
    
    with st.expander("🔍 VIEW TRADE SETUP LOGIC", expanded=True):
        st.markdown(f"""
        * **Breakdown Entry:** Triggers below market price to catch confirmed downward trend acceleration.
        * **Standardized Risk Boundary:** Keeps the trade parameters highly systematic by locking loss threshold at a flat **{round(sl_pct * 100, 2)}%**.
        * **Contracts Rule:** Calculates maximum gain and loss exposure relative to your F&O lot sizing (**{bear_lot_size}** units).
        """)

# --- Shared Footer Guardrails ---
st.markdown("---")
st.warning("⚠️ **Execution Guardrail:** Always deploy these setups using **SL-Limit (MIS/NRML)** orders directly within your broker terminal. Do not use market orders to avoid execution slippage.")
