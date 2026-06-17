import streamlit as st
from datetime import datetime
import requests
from nselib import capital_market

# Set up browser tab properties
st.set_page_config(page_title="RISHI's Intraday Dashboard", page_icon="📊", layout="centered")

def safe_price(val):
    try:
        return float(str(val).replace(',', '').strip())
    except:
        return 0.0

# 🎨 App Branding Header
st.title("📊 Mohit's Momentum Dashboard")
st.caption(f"Live Market Analysis Engine • System Time: {datetime.now().strftime('%d %b %Y | %H:%M IST')}")
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

# 🚀 Real-time Live execution logic triggered on Page Access or Click
try:
    # Fetching data using correct capitalized parameters
    gainers_df = capital_market.get_top_gainers_losers(type='Gainers')
    losers_df = capital_market.get_top_gainers_losers(type='Losers')
    
    bullish_stock = gainers_df.iloc[0]['symbol']
    bullish_ltp = safe_price(gainers_df.iloc[0]['ltp'])
    
    bearish_stock = losers_df.iloc[0]['symbol']
    bearish_ltp = safe_price(losers_df.iloc[0]['ltp'])
except Exception as e:
    st.warning("⚠️ Live NSE Feed busy or closed. Displaying backup fallback data placeholders.")
    bullish_stock, bullish_ltp = "SBIN", 780.00
    bearish_stock, bearish_ltp = "WIPRO", 490.00

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
