import streamlit as st
from datetime import datetime, timedelta
import requests
from nselib import capital_market

# Set up browser tab properties
st.set_page_config(page_title="RISHI's Intraday Dashboard", page_icon="📊", layout="centered")

def safe_price(val):
    try:
        return float(str(val).replace(',', '').strip())
    except:
        return 0.0

# Force the server to calculate actual Indian Standard Time (UTC + 5.5 hours)
UTC_NOW = datetime.utcnow()
IST_NOW = UTC_NOW + timedelta(hours=5, minutes=30)

# 🎨 App Branding Header
st.title("📊 RISHI's Momentum Dashboard")
st.caption(f"Live Market Analysis Engine • System Time: {IST_NOW.strftime('%d %b %Y | %H:%M IST')}")
st.markdown("---")

# ⚙️ Risk Parameters Config Card
st.subheader("⚙️ Strategy Parameters")
col_cap, col_lev, col_risk = st.columns(3)

with col_cap:
    capital = st.number_input("Trading Capital (₹)", value=10000, step=1000)
with col_lev:
    leverage = st.number_input("MIS Leverage (x)", value=5, min_value=1, max_value=5)
with col_risk:
    max_risk = st.number_input("Max Risk Per Trade (₹)", value=100, step=10)

buying_power = capital * leverage
st.info(f"Total Buying Power Available: **₹{buying_power:,}**")
st.markdown("---")

# 🚀 Real-time Live execution logic with Manual Overrides
st.subheader("📡 Nifty Live Market Feed")

# Pre-define fallback variables cleanly so they are always available to the manual inputs
auto_bullish_stock, auto_bullish_ltp = "SBIN", 780.00
auto_bearish_stock, auto_bearish_ltp = "WIPRO", 490.00

# Create two tabs: One for Auto, one for Manual fallback
tab_auto, tab_manual = st.tabs(["🤖 Automated Nifty Scanner", "✍️ Manual Entry Override"])

with tab_auto:
    try:
        # Fixed: Using the verified, official nselib core functions
        gainers_df = capital_market.top_gainers_or_losers(to_get='gainers')
        losers_df = capital_market.top_gainers_or_losers(to_get='loosers')
        
        # Extract Top High-Volume Gainer
        auto_bullish_stock = gainers_df.iloc[0]['symbol']
        auto_bullish_ltp = safe_price(gainers_df.iloc[0]['ltp'])
        
        # Extract Top High-Volume Loser
        auto_bearish_stock = losers_df.iloc[0]['symbol']
        auto_bearish_ltp = safe_price(losers_df.iloc[0]['ltp'])
        
        st.success(f"✅ Live NSE Data Connected! Top Gainer: {auto_bullish_stock} | Top Loser: {auto_bearish_stock}")
    except Exception as e:
        st.error("⚠️ NSE Data Feed is busy or closed. Use the 'Manual Entry Override' tab to type a stock manually!")

with tab_manual:
    st.caption("If the automated scanner fails, check your broker app's top movers list and type them manually:")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        man_bullish = st.text_input("Top Gainer Symbol", value=auto_bullish_stock)
        man_bullish_ltp = st.number_input("Gainer Live Price (₹)", value=auto_bullish_ltp, step=0.05)
    with col_m2:
        man_bearish = st.text_input("Top Loser Symbol", value=auto_bearish_stock)
        man_bearish_ltp = st.number_input("Loser Live Price (₹)", value=auto_bearish_ltp, step=0.05)

# Direct data handling
bullish_stock = man_bullish
bullish_ltp = man_bullish_ltp
bearish_stock = man_bearish
bearish_ltp = man_bearish_ltp

st.markdown("---")

# Math Calculations based on inputs
long_entry = round(bullish_ltp * 1.002, 2)
long_sl = round(long_entry * 0.995, 2)
long_risk = round(long_entry - long_sl, 2)
long_qty = min(int(max_risk // long_risk), int(buying_power // long_entry)) if long_risk > 0 else 1
long_target1 = round(long_entry + (long_risk * 2), 2)
long_target2 = round(long_entry + (long_risk * 3), 2)

short_entry = round(bearish_ltp * 0.998, 2)
short_sl = round(short_entry * 1.005, 2)
short_risk = round(short_sl - short_entry, 2)
short_qty = min(int(max_risk // short_risk), int(buying_power // short_entry)) if short_risk > 0 else 1
short_target1 = round(short_entry - (short_risk * 2), 2)
short_target2 = round(short_entry - (short_risk * 3), 2)

# 📈 RENDER SYSTEM 1: LONG TRADES
st.success(f"### 📈 LONG SETUP: {bullish_stock}")
l_col1, l_col2, l_col3 = st.columns(3)
l_col1.metric("Trigger Entry", f"₹{long_entry}")
l_col2.metric("Stop Loss (SL)", f"₹{long_sl}")
l_col3.metric("Exact Quantity", f"{long_qty} Shares")

l_col_t1, l_col_t2 = st.columns(2)
l_col_t1.metric("🎯 Target 1 (1:2 RR)", f"₹{long_target1}")
l_col_t2.metric("🎯 Target 2 (1:3 RR)", f"₹{long_target2}")
st.caption(f"*Approx. margin required:* ₹{round((long_entry * long_qty)/5, 2)}")

st.markdown("---")

# 📉 RENDER SYSTEM 2: SHORT TRADES
st.error(f"### 📉 SHORT SETUP: {bearish_stock}")
s_col1, s_col2, s_col3 = st.columns(3)
s_col1.metric("Trigger Entry", f"₹{short_entry}")
s_col2.metric("Stop Loss (SL)", f"₹{short_sl}")
s_col3.metric("Exact Quantity", f"{short_qty} Shares")

s_col_t1, s_col_t2 = st.columns(2)
s_col_t1.metric("🎯 Target 1 (1:2 RR)", f"₹{short_target1}")
s_col_t2.metric("🎯 Target 2 (1:3 RR)", f"₹{short_target2}")
st.caption(f"*Approx. margin required:* ₹{round((short_entry * short_qty)/5, 2)}")

st.markdown("---")
st.warning("⚠️ **Execution Guardrail:** Manually configure these setups as **SL-Limit (MIS)** orders directly on your broker platform.")
