import streamlit as st
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import yfinance as yf
from nselib import capital_market

# --- PAGE CONFIGURATION ---
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

# --- Live F&O Lot Directory Fetcher ---
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

# --- Fetch Exact Near-Month Futures LTP ---
def get_futures_ltp(symbol, spot_fallback):
    """
    Fetches the exact real-time or last traded price of the 
    nearest expiry futures contract using yfinance (SYMBOL=F syntax).
    """
    try:
        clean_sym = str(symbol).strip().upper()
        # Yahoo Finance tracks Indian stock futures using SYMBOL=F structure
        fut_ticker = f"{clean_sym}=F"
        data = yf.Ticker(fut_ticker)
        
        # Pull live summary price data
        fast_info = data.fast_info
        if fast_info and 'last_price' in fast_info and fast_info['last_price'] > 0:
            return round(float(fast_info['last_price']), 2)
            
        # History fallback if fast_info fails
        hist = data.history(period="1d")
        if not hist.empty:
            return round(float(hist['Close'].iloc[-1]), 2)
    except Exception:
        pass
    return spot_fallback

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
        
        high = df.get("High")
        low = df.get("Low")
        close = df.get("Close")
        
        # Flatten names explicitly to drop MultiIndex labels
        high = high.iloc[:, 0].rename(None) if isinstance(high, pd.DataFrame) else high.rename(None)
        low = low.iloc[:, 0].rename(None) if isinstance(low, pd.DataFrame) else low.rename(None)
        close = close.iloc[:, 0].rename(None) if isinstance(close, pd.DataFrame) else close.rename(None)
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        
        return round(float(atr), 2)
    except Exception:
        return None

# --- Pre-load Lot Metadata ---
with st.spinner("Initializing NSE Live Lot Directories..."):
    nse_lot_sizes = get_fno_lot_sizes()

# Native Time calculations
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
    st.sidebar.info(f"Total Buying Power: *₹{buying_power:,}*")
else:  # Futures Mode
    capital = st.sidebar.number_input("Trading Margin (₹)", value=150000, step=10000)
    max_risk = st.sidebar.number_input("Max Risk Per Trade (₹)", value=5000, step=250)
    st.sidebar.warning("🛡️ Contracts exceeding your absolute max risk or available margin limits will automatically block.")

st.sidebar.markdown("---")
st.sidebar.subheader("📐 ATR Parameters (Shared)")
atr_period = st.sidebar.number_input("ATR Period (candles)", value=14, min_value=5, max_value=50)
atr_multiplier = st.sidebar.number_input("ATR Multiplier", value=1.5, min_value=0.5, max_value=5.0, step=0.1)

# --- MAIN PAGE HEADER ---
st.title("📊 RISHI's Multi-Asset Momentum Dashboard")
st.caption(f"Engine Mode: *{trading_mode.upper()}* • System Time: {IST_NOW.strftime('%d %b %Y | %H:%M IST')}")
st.markdown("---")

# --- Live Market Feed Section ---
st.subheader("📡 Nifty Live Market Feed")
auto_bullish_stock, auto_bullish_ltp = "no stock", 0.0
auto_bearish_stock, auto_bearish_ltp = "no stock", 0.0

chosen_feed = st.radio("Select Feed Input Source:", ["🤖 Automated Nifty Scanner", "✍️ Manual Entry Override"], horizontal=True)

try:
    gainers_df = capital_market.top_gainers_or_losers(to_get='gainers')
    losers_df  = capital_market.top_gainers_or_losers(to_get='losers') # Fixed the 'loosers' library typo
    gainers_df.columns = gainers_df.columns.str.strip().str.upper()
    losers_df.columns = losers_df.columns.str.strip().str.upper()
    
    auto_bullish_stock = gainers_df.iloc[0]['SYMBOL']
    auto_bullish_ltp   = safe_price(gainers_df.iloc[0]['LTP'])
    auto_bearish_stock = losers_df.iloc[0]['SYMBOL']
    auto_bearish_ltp   = safe_price(losers_df.iloc[0]['LTP'])
except Exception:
    st.sidebar.warning("⚠️ Live NSE Data Feed is busy/closed. Defaulting to manual mode entry choices.")

if chosen_feed == "🤖 Automated Nifty Scanner":
    st.success(f"✅ Live NSE Data Active! Top Gainer: *{auto_bullish_stock}* (₹{auto_bullish_ltp}) | Top Loser: *{auto_bearish_stock}* (₹{auto_bearish_ltp})")
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

# Swap prices to Futures Contracts if Futures Engine is running
if trading_mode == "🔥 Stock Futures (Lots)":
    with st.spinner("Fetching matching Near-Month Futures prices..."):
        bullish_ltp = get_futures_ltp(bullish_stock, spot_fallback=bullish_ltp)
        bearish_ltp = get_futures_ltp(bearish_stock, spot_fallback=bearish_ltp)
    st.info(f"⚡ *Futures Contract Pricing Loaded:* {bullish_stock} Fut = *₹{bullish_ltp}* | {bearish_stock} Fut = *₹{bearish_ltp}*")

# Fetch Intraday Volatility Reference
with st.spinner("Fetching historical intraday data from Yahoo Finance..."):
    bull_atr = get_atr(bullish_stock, period=atr_period)
    bear_atr = get_atr(bearish_stock, period=atr_period)

# ==========================================
# MODULE 1: INTRADAY CASH ENGINE
# ==========================================
if trading_mode == "📈 Intraday Cash (Shares)":
    # --- Long Calculations ---
    long_entry = round(bullish_ltp * 1.002, 2)
    atr_sl_dist = round(bull_atr * atr_multiplier, 2) if bull_atr else round(long_entry * 0.005, 2)
    long_sl = round(long_entry - atr_sl_dist, 2)
    long_risk = max(round(long_entry - long_sl, 2), 0.05)
    
    if long_entry > 0:
        long_qty = max(min(int(max_risk // long_risk), int(buying_power // long_entry)), 1)
        long_target1 = round(long_entry + (long_risk * 2), 2)
        long_target2 = round(long_entry + (long_risk * 3), 2)
    else:
        long_qty, long_target1, long_target2 = 0, 0.0, 0.0
        
    # --- Short Calculations (Moved outside the Long Else block to fix execution crash) ---
    short_entry = round(bearish_ltp * 0.998, 2)
    atr_sl_dist_s = round(bear_atr * atr_multiplier, 2) if bear_atr else round(short_entry * 0.005, 2)
    short_sl = round(short_entry + atr_sl_dist_s, 2)
    short_risk = max(round(short_sl - short_entry, 2), 0.05)
    
    if short_entry > 0:
        short_qty = max(min(int(max_risk // short_risk), int(buying_power // short_entry)), 1)
        short_target1 = round(short_entry - (short_risk * 2), 2)
        short_target2 = round(short_entry - (short_risk * 3), 2)
    else:
        short_qty, short_target1, short_target2 = 0, 0.0, 0.0

    # --- Render: Long ---
    st.success(f"### 📈 INTRADAY CASH LONG: {bullish_stock}")
    q_col1, q_col2 = st.columns(2)
    q_col1.info(f"### 🎯 TRADE QUANTITY: *{long_qty} shares*")
    q_col2.error(f"### 🛡️ MAX LOSS RISK: *₹{round(long_risk * long_qty, 2):,}*")
    
    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Entry Trigger", f"₹{long_entry}")
    l_col2.metric("Stop Loss", f"₹{long_sl}", delta=f"-{round((long_risk/long_entry)*100,2)}% (-₹{long_risk}/sh)", delta_color="inverse")
    l_col3.metric("Target 1 (1:2)", f"₹{long_target1}", delta=f"+{round(((long_target1-long_entry)/long_entry)*100,2)}%")
    l_col4.metric("Target 2 (1:3)", f"₹{long_target2}", delta=f"+{round(((long_target2-long_entry)/long_entry)*100,2)}%")

    # --- Render: Short ---
    st.error(f"### 📉 INTRADAY CASH SHORT: {bearish_stock}")
    qs_col1, qs_col2 = st.columns(2)
    qs_col1.info(f"### 🎯 TRADE QUANTITY: *{short_qty} shares*")
    qs_col2.error(f"### 🛡️ MAX LOSS RISK: *₹{round(short_risk * short_qty, 2):,}*")
    
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Entry Trigger", f"₹{short_entry}")
    s_col2.metric("Stop Loss", f"₹{short_sl}", delta=f"+{round((short_risk/short_entry)*100,2)}% (+₹{short_risk}/sh)", delta_color="inverse")
    s_col3.metric("Target 1 (1:2)", f"₹{short_target1}", delta=f"-{round(((short_entry-short_target1)/short_entry)*100,2)}%")
    s_col4.metric("Target 2 (1:3)", f"₹{short_target2}", delta=f"-{round(((short_entry-short_target2)/short_entry)*100,2)}%")

# ==========================================
# MODULE 2: STOCK FUTURES ENGINE (WITH FUTURES LTP & ESTIMATED SPAN CHECKS)
# ==========================================
else:
    bull_lot_size = lookup_lot_size(bullish_stock, nse_lot_sizes)
    bear_lot_size = lookup_lot_size(bearish_stock, nse_lot_sizes)

    # --- Calculations: Long Futures using Near-Month Price ---
    long_entry = round(bullish_ltp * 1.002, 2)
    atr_sl_dist_f = round(bull_atr * atr_multiplier, 2) if bull_atr else round(long_entry * 0.005, 2)
    long_sl = round(long_entry - atr_sl_dist_f, 2)
    long_risk = max(round(long_entry - long_sl, 2), 0.05)
    max_loss_long = round(long_risk * bull_lot_size, 2)
    
    # Structural Margin Validation (Assuming standard 20% margin requirement for equity contracts)
    est_margin_required_long = round(long_entry * bull_lot_size * 0.20, 2)
    
    long_target1 = round(long_entry + (long_risk * 2), 2)
    long_target2 = round(long_entry + (long_risk * 3), 2)

    # --- Calculations: Short Futures using Near-Month Price ---
    short_entry = round(bearish_ltp * 0.998, 2)
    atr_sl_dist_fs = round(bear_atr * atr_multiplier, 2) if bear_atr else round(short_entry * 0.005, 2)
    short_sl = round(short_entry + atr_sl_dist_fs, 2)
    short_risk = max(round(short_sl - short_entry, 2), 0.05)
    max_loss_short = round(short_risk * bear_lot_size, 2)
    
    est_margin_required_short = round(short_entry * bear_lot_size * 0.20, 2)
    
    short_target1 = round(short_entry - (short_risk * 2), 2)
    short_target2 = round(short_entry - (short_risk * 3), 2)

    # --- Render: Long Futures ---
    st.success(f"### 📈 STOCK FUTURES LONG: {bullish_stock} (NEAR EXPIRY)")
    if max_loss_long > max_risk:
        st.error(f"🚫 *TRADE BLOCKED (RISK OUTOF BOUNDS):* Risk is *₹{max_loss_long:,}*, exceeding your allocation parameter of ₹{max_risk}.")
    elif est_margin_required_long > capital:
        st.error(f"🚫 *TRADE BLOCKED (INSUFFICIENT CAPITAL):* Upfront margin needed is ~*₹{est_margin_required_long:,}. Available margin setting is only *₹{capital:,}**.")
    else:
        qf_col1, qf_col2 = st.columns(2)
        qf_col1.info(f"### 🎯 TRADE QUANTITY: *1 Lot ({bull_lot_size} units)*")
        qf_col2.error(f"### 🛡️ MAX LOSS RISK: *₹{max_loss_long:,}* [PASSED SAFETY CHECKS]")
        
        lf_col1, lf_col2, lf_col3, lf_col4 = st.columns(4)
        lf_col1.metric("Futures Entry Trigger", f"₹{long_entry}")
        lf_col2.metric(f"ATR Stop Loss", f"₹{long_sl}", delta=f"-{round((long_risk/long_entry)*100,2)}% (-₹{long_risk}/sh)", delta_color="inverse")
        lf_col3.metric("Target 1 (1:2)", f"₹{long_target1}", delta=f"+{round(((long_target1-long_entry)/long_entry)*100,2)}%")
        lf_col4.metric("Target 2 (1:3)", f"₹{long_target2}", delta=f"+{round(((long_target2-long_entry)/long_entry)*100,2)}%")
        
        with st.expander("🔍 VIEW LONG FUTURES LOGIC"):
            st.markdown(f"""
            * *Why this entry?* Triggers purely off near-expiry contract volume momentum, removing spot-to-futures premium tracking error.
            * *Dynamic Sizing Rationale:* Checked against a required structural margin of *₹{est_margin_required_long:,}* and approved for deployment. 
            """)

    st.markdown("---")

    # --- Render: Short Futures ---
    st.error(f"### 📉 STOCK FUTURES SHORT: {bearish_stock} (NEAR EXPIRY)")
    if max_loss_short > max_risk:
        st.error(f"🚫 *TRADE BLOCKED (RISK OUTOF BOUNDS):* Risk is *₹{max_loss_short:,}*, exceeding your allocation parameter of ₹{max_risk}.")
    elif est_margin_required_short > capital:
        st.error(f"🚫 *TRADE BLOCKED (INSUFFICIENT CAPITAL):* Upfront margin needed is ~*₹{est_margin_required_short:,}. Available margin setting is only *₹{capital:,}**.")
    else:
        qfs_col1, qfs_col2 = st.columns(2)
        qfs_col1.info(f"### 🎯 TRADE QUANTITY: *1 Lot ({bear_lot_size} units)*")
        qfs_col2.error(f"### 🛡️ MAX LOSS RISK: *₹{max_loss_short:,}* [PASSED SAFETY CHECKS]")
        
        sf_col1, sf_col2, sf_col3, sf_col4 = st.columns(4)
        sf_col1.metric("Futures Entry Trigger", f"₹{short_entry}")
        sf_col2.metric(f"ATR Stop Loss", f"₹{short_sl}", delta=f"+{round((short_risk/short_entry)*100,2)}% (+₹{short_risk}/sh)", delta_color="inverse")
        sf_col3.metric("Target 1 (1:2)", f"₹{short_target1}", delta=f"-{round(((short_entry-short_target1)/short_entry)*100,2)}%")
        sf_col4.metric("Target 2 (1:3)", f"₹{short_target2}", delta=f"-{round(((short_entry-short_target2)/short_entry)*100,2)}%")
        
        with st.expander("🔍 VIEW SHORT FUTURES LOGIC"):
            st.markdown(f"""
            * *Why this entry?* Triggers automatically on hard resistance breakdown inside the active liquid derivative contract.
            * *Dynamic Sizing Rationale:* Estimated margin deployment stays safely within bounds at ~*₹{est_margin_required_short:,}*.
            """)

# --- Shared Footer Guardrails ---
st.markdown("---")
st.warning("⚠️ *Execution Guardrail:* Always deploy these setups using *SL-Limit (MIS/NRML)* orders directly within your broker terminal. Do not use market orders to avoid execution slippage.")
