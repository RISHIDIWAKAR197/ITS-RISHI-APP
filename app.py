import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from nselib import capital_market

st.set_page_config(page_title="RISHI's Intraday Dashboard", page_icon="📊", layout="centered")

def safe_price(val):
    try:
        return float(str(val).replace(',', '').strip())
    except:
        return 0.0

# --- ATR Calculation ---
def get_atr(symbol, period=14):
    """
    Fetches last 30 days of daily OHLC from Yahoo Finance and returns
    the 14-period ATR. Symbol should be NSE format e.g. 'RELIANCE.NS'
    Returns None if data cannot be fetched.
    """
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

# --- IST time ---
UTC_NOW = datetime.utcnow()
IST_NOW = UTC_NOW + timedelta(hours=5, minutes=30)

# --- Header ---
st.title("📊 RISHI's Momentum Dashboard")
st.caption(f"Live Market Analysis Engine • System Time: {IST_NOW.strftime('%d %b %Y | %H:%M IST')}")
st.markdown("---")

# --- Strategy Parameters ---
st.subheader("⚙️ Strategy Parameters")
col_cap, col_lev, col_risk = st.columns(3)
with col_cap:
    capital = st.number_input("Trading Capital (₹)", value=30000, step=1000)
with col_lev:
    leverage = st.number_input("MIS Leverage (x)", value=5, min_value=1, max_value=5)
with col_risk:
    max_risk = st.number_input("Max Risk Per Trade (₹)", value=300, step=10)

buying_power = capital * leverage
st.info(f"Total Buying Power Available: **₹{buying_power:,}**")

# --- ATR Multiplier Control ---
st.subheader("📐 ATR Stop Loss Settings")
col_atr1, col_atr2 = st.columns(2)
with col_atr1:
    atr_period = st.number_input("ATR Period (candles)", value=14, min_value=5, max_value=50,
                                  help="Number of daily candles used to calculate ATR. 14 is standard.")
with col_atr2:
    atr_multiplier = st.number_input("ATR Multiplier", value=1.5, min_value=0.5, max_value=5.0, step=0.1,
                                      help="SL = Entry ± (Multiplier × ATR). Lower = tighter SL, Higher = wider SL.")

st.caption("📌 **How ATR SL works:** ATR measures average daily price range. "
           "SL is placed at 1.5× that range from entry, so normal noise won't stop you out. "
           "A higher multiplier gives the trade more breathing room but increases per-share risk.")
st.markdown("---")

# --- Live Market Feed ---
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
    st.caption("If the automated scanner fails, check your broker app's top movers list and type them manually:")
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

# --- Fetch ATR for both stocks ---
with st.spinner("Fetching ATR data from Yahoo Finance..."):
    bull_atr = get_atr(bullish_stock, period=atr_period)
    bear_atr = get_atr(bearish_stock, period=atr_period)

# Show ATR info
col_atr_bull, col_atr_bear = st.columns(2)
with col_atr_bull:
    if bull_atr:
        st.metric(f"ATR ({atr_period}d) — {bullish_stock}", f"₹{bull_atr}",
                  help="Average True Range: typical daily price swing for this stock.")
    else:
        st.warning(f"Could not fetch ATR for {bullish_stock}. Falling back to 0.5% SL.")
with col_atr_bear:
    if bear_atr:
        st.metric(f"ATR ({atr_period}d) — {bearish_stock}", f"₹{bear_atr}",
                  help="Average True Range: typical daily price swing for this stock.")
    else:
        st.warning(f"Could not fetch ATR for {bearish_stock}. Falling back to 0.5% SL.")

st.markdown("---")

# --- LONG calculations ---
long_entry = round(bullish_ltp * 1.002, 2)

if bull_atr:
    atr_sl_distance_long = round(bull_atr * atr_multiplier, 2)
    long_sl   = round(long_entry - atr_sl_distance_long, 2)
    long_risk = round(long_entry - long_sl, 2)
    sl_method_long = f"ATR × {atr_multiplier} = ₹{atr_sl_distance_long} below entry"
else:
    long_sl   = round(long_entry * 0.995, 2)
    long_risk = round(long_entry - long_sl, 2)
    sl_method_long = "Fallback: 0.5% below entry (ATR unavailable)"

long_qty     = min(int(max_risk // long_risk), int(buying_power // long_entry)) if long_risk > 0 else 1
long_target1 = round(long_entry + (long_risk * 2), 2)
long_target2 = round(long_entry + (long_risk * 3), 2)

# --- SHORT calculations ---
short_entry = round(bearish_ltp * 0.998, 2)

if bear_atr:
    atr_sl_distance_short = round(bear_atr * atr_multiplier, 2)
    short_sl   = round(short_entry + atr_sl_distance_short, 2)
    short_risk = round(short_sl - short_entry, 2)
    sl_method_short = f"ATR × {atr_multiplier} = ₹{atr_sl_distance_short} above entry"
else:
    short_sl   = round(short_entry * 1.005, 2)
    short_risk = round(short_sl - short_entry, 2)
    sl_method_short = "Fallback: 0.5% above entry (ATR unavailable)"

short_qty     = min(int(max_risk // short_risk), int(buying_power // short_entry)) if short_risk > 0 else 1
short_target1 = round(short_entry - (short_risk * 2), 2)
short_target2 = round(short_entry - (short_risk * 3), 2)

# --- RENDER: LONG ---
st.success(f"### 📈 LONG SETUP: {bullish_stock}")
st.caption(f"🛑 SL Method: {sl_method_long}")

l_col1, l_col2, l_col3 = st.columns(3)
l_col1.metric("Trigger Entry", f"₹{long_entry}")
l_col2.metric("Stop Loss (ATR)", f"₹{long_sl}", delta=f"-₹{long_risk} risk/share", delta_color="inverse")
l_col3.metric("Quantity", f"{long_qty} shares")

l_col_t1, l_col_t2 = st.columns(2)
l_col_t1.metric("🎯 Target 1 (1:2 RR)", f"₹{long_target1}", delta=f"+₹{round(long_risk*2,2)}")
l_col_t2.metric("🎯 Target 2 (1:3 RR)", f"₹{long_target2}", delta=f"+₹{round(long_risk*3,2)}")
st.caption(f"Max loss if SL hit: **₹{round(long_risk * long_qty, 2)}** | "
           f"Profit at T1: **₹{round(long_risk * 2 * long_qty, 2)}** | "
           f"Approx margin: **₹{round((long_entry * long_qty)/leverage, 2)}**")

st.markdown("---")

# --- RENDER: SHORT ---
st.error(f"### 📉 SHORT SETUP: {bearish_stock}")
st.caption(f"🛑 SL Method: {sl_method_short}")

s_col1, s_col2, s_col3 = st.columns(3)
s_col1.metric("Trigger Entry", f"₹{short_entry}")
s_col2.metric("Stop Loss (ATR)", f"₹{short_sl}", delta=f"+₹{short_risk} risk/share", delta_color="inverse")
s_col3.metric("Quantity", f"{short_qty} shares")

s_col_t1, s_col_t2 = st.columns(2)
s_col_t1.metric("🎯 Target 1 (1:2 RR)", f"₹{short_target1}", delta=f"-₹{round(short_risk*2,2)}")
s_col_t2.metric("🎯 Target 2 (1:3 RR)", f"₹{short_target2}", delta=f"-₹{round(short_risk*3,2)}")
st.caption(f"Max loss if SL hit: **₹{round(short_risk * short_qty, 2)}** | "
           f"Profit at T1: **₹{round(short_risk * 2 * short_qty, 2)}** | "
           f"Approx margin: **₹{round((short_entry * short_qty)/leverage, 2)}**")

st.markdown("---")
st.warning("⚠️ **Execution Guardrail:** Manually configure these setups as **SL-Limit (MIS)** orders directly on your broker platform.")
st.info("ℹ️ ATR is calculated from daily candles via Yahoo Finance (.NS suffix). For intraday precision, consider switching to 15-min ATR once your data source supports it.")
