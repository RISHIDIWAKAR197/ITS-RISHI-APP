import streamlit as st
import pandas as pd
import numpy as np
import time as time_lib
from datetime import datetime
from strategy import analyze_ticker_signal

# Page Config
st.set_page_config(
    page_title="Intraday 9:30 AM Signal Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Intraday Signal Screener (9:30 AM - 1:30 PM)")
st.caption("Strategy: 15-Min ORB + VWAP Alignment + 9/20 EMA Stack")

# Sidebar Configuration
st.sidebar.header("Control Panel")
watchlist_input = st.sidebar.text_area(
    "Watchlist (Comma Separated)", 
    "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK"
)
watchlist = [s.strip() for s.strip(watchlist_input).split(",")]

refresh_rate = st.sidebar.slider("Auto-Refresh Interval (Seconds)", 2, 30, 5)

# Mock Data Streamer (Replace with your Broker WebSocket/REST API)
def fetch_live_data(ticker):
    times = pd.date_range(start=f"{datetime.today().strftime('%Y-%m-%d')} 09:15:00", periods=4, freq="5min")
    np.random.seed(abs(hash(ticker)) % 1000)
    base_price = 1000 + np.random.randint(-100, 100)
    
    close = base_price + np.cumsum(np.random.normal(1, 2, len(times)))
    high = close + np.random.uniform(0.5, 2.0, len(times))
    low = close - np.random.uniform(0.5, 2.0, len(times))
    open_p = close - np.random.uniform(-1.0, 1.0, len(times))
    volume = np.random.randint(15000, 60000, len(times))
    
    return pd.DataFrame({
        'Timestamp': times,
        'Open': open_p,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })

# Streamlit Fragment ensures only this section refreshes without reloading the full UI
@st.fragment
def render_live_signals():
    signals = []
    
    for ticker in watchlist:
        df = fetch_live_data(ticker)
        result = analyze_ticker_signal(df, ticker)
        signals.append(result)
    
    results_df = pd.DataFrame(signals)
    
    # Visual Highlighting
    def highlight_signals(val):
        if "BUY" in str(val):
            return "background-color: #1e4620; color: #4CAF50; font-weight: bold;"
        elif "SELL" in str(val):
            return "background-color: #4a1e1e; color: #F44336; font-weight: bold;"
        return "color: #888888;"

    styled_df = results_df.style.applymap(highlight_signals, subset=['Signal'])
    
    # Display Table
    st.subheader(f"Live Market Screener — Updated: {datetime.now().strftime('%H:%M:%S')}")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Run the Fragment
render_live_signals()

# Trigger Automatic Rerun Loop
time_lib.sleep(refresh_rate)
st.rerun()
