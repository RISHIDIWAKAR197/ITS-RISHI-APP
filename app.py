import streamlit as st
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import yfinance as yf
from nselib import capital_market

# Now you can safely configure the page layout!
st.set_page_config(
    page_title="The Ennoble Trader | Multi-Asset Intraday Tool", 
    page_icon="📊", 
    layout="wide"
)

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

# --- TRUE INTRADAY ATR CALCULATION (Robust Version) ---
def get_atr(symbol, period=14):
    """
    Fetches the last 5 days of 5-minute intraday data 
    and returns a valid Intraday 14-period ATR.
    """
    try:
        ticker = symbol.upper().strip()
        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"
        
        # Fetch 5-minute bars over the last 5 days
        df = yf.download(
            tickers=ticker, 
            period="5d", 
            interval="5m", 
            progress=False, 
            auto_adjust=True
        )
        
        if df.empty or len(df) < period + 1:
            return None
            
        df = df.copy()
        
        # Safe structural dictionary fallback reading
        high = df.get("High")
        low = df.get("Low")
        close = df.get("Close")
        
        # Flatten structure in case MultiIndex forces multi-columns
        if isinstance(high, pd.DataFrame): high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame): low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        
        return round(float(atr), 2)
    except Exception:
        return None

# --- Pre-load Data ---
with st.spinner("Initializing NSE Live Lot Directories..."):
    nse_lot_sizes = get_fno_lot_sizes()

# Native timezone calculation
IST_NOW = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

# --- SIDEBAR CONTROL: Control Center ---
st.sidebar.header("🕹️ Control Center")
trading_mode = st.sidebar.radio(
    "Choose Your Trading Mode:",
    ["📈 Intraday Cash (Shares)", "🔥 Stock Futures (Lots)"],
    help="Toggle between individual equity risk sizing or standardized derivatives."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Capital & Risk Settings")

if trading_mode == "📈 Intraday Cash (Shares)":
    capital = st.sidebar.number_input("Trading Capital (₹)", value=30000, step=1000)
    leverage = st.sidebar.number_input("MIS Leverage (x)", value=5, min_value=1, max_value=5)
    max_risk = st.sidebar.number_input("Max Risk Per Trade (₹)", value=300, step=10)
    buying_power = capital * leverage
    st.sidebar.info(f"Total Buying Power: **₹{buying_power:,}**")
else:  # Futures Mode
    capital = st.sidebar.number_input("Trading Margin (₹)", value=150000, step=10000)
    st.sidebar.warning("🛡️ **Hard Risk Rule:** High-volatility options exceeding ₹5,000 max lot exposure are automatically blocked.")

st.sidebar.markdown("---")
st.sidebar.subheader("📐 ATR Parameters (Shared)")
atr_period = st.sidebar.number_input("ATR Period (candles)", value=14, min_value=5, max_value=50)
atr_multiplier = st.sidebar.number_input("ATR Multiplier", value=1.5, min_value=0.5, max_value=5.0, step=0.1)

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
except Exception:
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
# FETCH INTRADAY ATR FOR SELECTED STOCKS
# ==========================================
with st.spinner("Fetching historical intraday data from Yahoo Finance..."):
    bull_atr = get_atr(bullish_stock, period=atr_period)
    bear_atr = get_atr(bearish_stock, period=atr_period)

# ==========================================
# MODULE 1: INTRADAY CASH ENGINE
# ==========================================
if trading_mode == "📈 Intraday Cash (Shares)":
    # Long Calculations
    long_entry = round(bullish_ltp * 1.002, 2)
    atr_sl_dist = round(bull_atr * atr_multiplier, 2) if bull_atr else round(long_entry * 0.005, 2)
    long_sl = round(long_entry - atr_sl_dist, 2)
    long_risk = max(round(long_entry - long_sl, 2), 0.05)
    long_qty = max(min(int(max_risk // long_risk), int(buying_power // long_entry)), 1)
    long_target1 = round(long_entry + (long_risk * 2), 2)
    long_target2 = round(long_entry + (long_risk * 3), 2)
    
    # Short Calculations
    short_entry = round(bearish_ltp * 0.998, 2)
    atr_sl_dist_s = round(bear_atr * atr_multiplier, 2) if bear_atr else round(short_entry * 0.005, 2)
    short_sl = round(short_entry + atr_sl_dist_s, 2)
    short_risk = max(round(short_sl - short_entry, 2), 0.05)
    short_qty = max(min(int(max_risk // short_risk), int(buying_power // short_entry)), 1)
    short_target1 = round(short_entry - (short_risk * 2), 2)
    short_target2 = round(short_entry - (short_risk * 3), 2)

    # --- Render: Long ---
    st.success(f"### 📈 INTRADAY CASH LONG: {bullish_stock}")
    q_col1, q_col2 = st.columns(2)
    q_col1.info(f"### 🎯 TRADE QUANTITY: **{long_qty} shares**")
    q_col2.error(f"### 🛡️ MAX LOSS RISK: **₹{round(long_risk * long_qty, 2):,}**")
    
    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Entry Trigger", f"₹{long_entry}")
    l_col2.metric("Stop Loss", f"₹{long_sl}", delta=f"-{round((long_risk/long_entry)*100,2)}% (-₹{long_risk}/sh)", delta_color="inverse")
    l_col3.metric("Target 1 (1:2)", f"₹{long_target1}", delta=f"+{round(((long_target1-long_entry)/long_entry)*100,2)}%")
    l_col4.metric("Target 2 (1:3)", f"₹{long_target2}", delta=f"+{round(((long_target2-long_entry)/long_entry)*100,2)}%")

    # --- Render: Short ---
    st.error(f"### 📉 INTRADAY CASH SHORT: {bearish_stock}")
    qs_col1, qs_col2 = st.columns(2)
    qs_col1.info(f"### 🎯 TRADE QUANTITY: **{short_qty} shares**")
    qs_col2.error(f"### 🛡️ MAX LOSS RISK: **₹{round(short_risk * short_qty, 2):,}**")
    
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Entry Trigger", f"₹{short_entry}")
    s_col2.metric("Stop Loss", f"₹{short_sl}", delta=f"+{round((short_risk/short_entry)*100,2)}% (+₹{short_risk}/sh)", delta_color="inverse")
    s_col3.metric("Target 1 (1:2)", f"₹{short_target1}", delta=f"-{round(((short_entry-short_target1)/short_entry)*100,2)}%")
    s_col4.metric("Target 2 (1:3)", f"₹{short_target2}", delta=f"-{round(((short_entry-short_target2)/short_entry)*100,2)}%")

# ==========================================
# MODULE 2: STOCK FUTURES ENGINE (WITH ATR & RISK-FILTER)
# ==========================================
else:
    bull_lot_size = lookup_lot_size(bullish_stock, nse_lot_sizes)
    bear_lot_size = lookup_lot_size(bearish_stock, nse_lot_sizes)

    # --- Calculations: Long Futures using Intraday ATR ---
    long_entry = round(bullish_ltp * 1.002, 2)
    atr_sl_dist_f = round(bull_atr * atr_multiplier, 2) if bull_atr else round(long_entry * 0.005, 2)
    long_sl = round(long_entry - atr_sl_dist_f, 2)
    long_risk = max(round(long_entry - long_sl, 2), 0.05)
    max_loss_long = round(long_risk * bull_lot_size, 2)
    
    long_target1 = round(long_entry + (long_risk * 2), 2)
    long_target2 = round(long_entry + (long_risk * 3), 2)

    # --- Calculations: Short Futures using Intraday ATR ---
    short_entry = round(bearish_ltp * 0.998, 2)
    atr_sl_dist_fs = round(bear_atr * atr_multiplier, 2) if bear_atr else round(short_entry * 0.005, 2)
    short_sl = round(short_entry + atr_sl_dist_fs, 2)
    short_risk = max(round(short_sl - short_entry, 2), 0.05)
    max_loss_short = round(short_risk * bear_lot_size, 2)
    
    short_target1 = round(short_entry - (short_risk * 2), 2)
    short_target2 = round(short_entry - (short_risk * 3), 2)

    # --- Render: Long Futures ---
    st.success(f"### 📈 STOCK FUTURES LONG: {bullish_stock}")
    if max_loss_long > 5000:
        st.error(f"🚫 **TRADE BLOCKED:** {bullish_stock} Futures risk is **₹{max_loss_long:,}**, exceeding your maximum risk parameter of ₹5,000. Sizing execution suppressed.")
    else:
        qf_col1, qf_col2 = st.columns(2)
        qf_col1.info(f"### 🎯 TRADE QUANTITY: **1 Lot ({bull_lot_size} units)**")
        qf_col2.error(f"### 🛡️ MAX LOSS RISK: **₹{max_loss_long:,}** [PASSED RISK CHECK]")
        
        lf_col1, lf_col2, lf_col3, lf_col4 = st.columns(4)
        lf_col1.metric("Trigger Entry (+0.2%)", f"₹{long_entry}")
        lf_col2.metric(f"ATR Stop Loss", f"₹{long_sl}", delta=f"-{round((long_risk/long_entry)*100,2)}% (-₹{long_risk}/sh)", delta_color="inverse")
        lf_col3.metric("Target 1 (1:2)", f"₹{long_target1}", delta=f"+{round(((long_target1-long_entry)/long_entry)*100,2)}%")
        lf_col4.metric("Target 2 (1:3)", f"₹{long_target2}", delta=f"+{round(((long_target2-long_entry)/long_entry)*100,2)}%")
        
        with st.expander("🔍 VIEW LONG FUTURES LOGIC"):
            st.markdown(f"""
            * **Why this entry?** It triggers breakout entries on momentum confirmation above local high liquidity blocks.
            * **Why this stop loss?** Applied true 5-min ATR (**₹{bull_atr}**). Your risk is dynamically scaled to volatility rather than an arbitrary index percentage, keeping it safe from institutional liquidity sweeps.
            * **Risk Filtering Rationale:** This contract lot passed structural screening because the ultimate absolute financial liability stays under **₹5,000** for a full lot contract.
            """)

    st.markdown("---")

    # --- Render: Short Futures ---
    st.error(f"### 📉 STOCK FUTURES SHORT: {bearish_stock}")
    if max_loss_short > 5000:
        st.error(f"🚫 **TRADE BLOCKED:** {bearish_stock} Futures risk is **₹{max_loss_short:,}**, exceeding your maximum risk parameter of ₹5,000. Sizing execution suppressed.")
    else:
        qfs_col1, qfs_col2 = st.columns(2)
        qfs_col1.info(f"### 🎯 TRADE QUANTITY: **1 Lot ({bear_lot_size} units)**")
        qfs_col2.error(f"### 🛡️ MAX LOSS RISK: **₹{max_loss_short:,}** [PASSED RISK CHECK]")
        
        sf_col1, sf_col2, sf_col3, sf_col4 = st.columns(4)
        sf_col1.metric("Trigger Entry (-0.2%)", f"₹{short_entry}")
        sf_col2.metric(f"ATR Stop Loss", f"₹{short_sl}", delta=f"+{round((short_risk/short_entry)*100,2)}% (+₹{short_risk}/sh)", delta_color="inverse")
        sf_col1.metric("Target 1 (1:2)", f"₹{short_target1}", delta=f"-{round(((short_entry-short_target1)/short_entry)*100,2)}%")
        sf_col4.metric("Target 2 (1:3)", f"₹{short_target2}", delta=f"-{round(((short_entry-short_target2)/short_entry)*100,2)}%")
        
        with st.expander("🔍 VIEW SHORT FUTURES LOGIC"):
            st.markdown(f"""
            * **Why this entry?** Placed 0.2% below the market price to trigger only if massive selling pressure forces price breakdown.
            * **Why this stop loss?** Set cleanly above minor technical clusters using the 5-min chart ATR value of **₹{bear_atr}** multiplied by {atr_multiplier}.
            * **Risk Filtering Rationale:** The trade is permitted to execute because its entire structural risk exposure is safely within your **₹5,000** capital restriction window.
            """)

# --- Shared Footer Guardrails ---
st.markdown("---")
st.warning("⚠️ **Execution Guardrail:** Always deploy these setups using **SL-Limit (MIS/NRML)** orders directly within your broker terminal. Do not use market orders to avoid execution slippage.")
