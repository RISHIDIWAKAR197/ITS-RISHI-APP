import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Intraday 9:30 AM Signal Screener",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Intraday Signal Screener (9:30 AM - 1:30 PM)")
st.caption("Strategy: 15-Min ORB + VWAP Alignment + 9/20 EMA Stack")

# -------------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -------------------------------------------------------------------
st.sidebar.header("Control Panel")
default_symbols = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK"
watchlist_input = st.sidebar.text_area("Watchlist (NSE Symbols)", default_symbols)
watchlist = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

refresh_rate = st.sidebar.slider("Auto-Refresh Interval (Seconds)", 5, 60, 10)

# -------------------------------------------------------------------
# 3. REAL MARKET DATA FETCHING (Yahoo Finance)
# -------------------------------------------------------------------
def fetch_live_data(ticker: str) -> pd.DataFrame:
    """Fetches intraday 5-minute bar data for NSE stocks via yfinance."""
    formatted_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    
    try:
        # Download 5-minute interval data for 1 trading day
        df = yf.download(
            tickers=formatted_ticker, 
            period="1d", 
            interval="5m", 
            progress=False,
            multi_level_index=False
        )
        
        if df.empty:
            return pd.DataFrame()
            
        df = df.reset_index()
        
        # Standardize timestamp column name
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
        elif 'Date' in df.columns:
            df.rename(columns={'Date': 'Timestamp'}, inplace=True)
            
        return df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------------------------
# 4. STRATEGY ENGINE & CALCULATIONS
# -------------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Exponential Moving Averages (9 & 20) and Intraday VWAP."""
    df = df.copy()
    
    # 9 EMA and 20 EMA
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Cumulative Intraday VWAP
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return df

def analyze_ticker_signal(df: pd.DataFrame, ticker: str) -> dict:
    """Evaluates 9:30 AM trading rules against live candles."""
    if df.empty or len(df) < 3:
        return {
            "Ticker": ticker, 
            "Signal": "NO DATA", 
            "Last Price": 0.0, 
            "ORB High": 0.0, 
            "ORB Low": 0.0, 
            "VWAP": 0.0, 
            "EMA 9/20": "N/A", 
            "Suggested SL": 0.0, 
            "Target (1:2)": 0.0
        }
        
    df = calculate_indicators(df)
    
    # Extract Opening Window (09:15 AM to 09:25 AM inclusive)
    df['Time'] = pd.to_datetime(df['Timestamp']).dt.time
    orb_window = df[(df['Time'] >= time(9, 15)) & (df['Time'] <= time(9, 25))]
    
    if orb_window.empty:
        orb_high = df.iloc[:3]['High'].max()
        orb_low = df.iloc[:3]['Low'].min()
    else:
        orb_high = orb_window['High'].max()
        orb_low = orb_window['Low'].min()
    
    # Latest available candle
    latest = df.iloc[-1]
    current_price = float(latest['Close'])
    vwap_val = float(latest['VWAP'])
    ema9_val = float(latest['EMA9'])
    ema20_val = float(latest['EMA20'])
    
    # Strategy conditions
    above_vwap = current_price > vwap_val
    ema_bullish = ema9_val > ema20_val
    orb_breakout = current_price > orb_high
    orb_breakdown = current_price < orb_low
    
    stop_loss = round(float(latest['Low']), 2)
    risk = current_price - stop_loss
    target_1to2 = round(current_price + (risk * 2.0), 2) if risk > 0 else 0.0

    # Decision Logic
    if orb_breakout and above_vwap and ema_bullish:
        signal = "BUY / LONG SETUP"
    elif orb_breakdown and not above_vwap and not ema_bullish:
        signal = "SELL / SHORT SETUP"
    else:
        signal = "NO TRADE / CHOP"

    return {
        "Ticker": ticker,
        "Signal": signal,
        "Last Price": round(current_price, 2),
        "ORB High": round(float(orb_high), 2),
        "ORB Low": round(float(orb_low), 2),
        "VWAP": round(vwap_val, 2),
        "EMA 9/20": "BULLISH" if ema_bullish else "BEARISH",
        "Suggested SL": stop_loss,
        "Target (1:2)": target_1to2
    }

# -------------------------------------------------------------------
# 5. STREAMLIT FRAGMENT (Auto-Refreshing UI)
# -------------------------------------------------------------------
@st.fragment(run_every=refresh_rate)
def render_live_signals():
    signals = []
    
    for ticker in watchlist:
        df = fetch_live_data(ticker)
        result = analyze_ticker_signal(df, ticker)
        signals.append(result)
    
    results_df = pd.DataFrame(signals)
    
    # Cell highlighting function
    def highlight_signals(val):
        if "BUY" in str(val):
            return "background-color: #1e4620; color: #4CAF50; font-weight: bold;"
        elif "SELL" in str(val):
            return "background-color: #4a1e1e; color: #F44336; font-weight: bold;"
        return "color: #888888;"

    # Modern Pandas .map() for styling
    styled_df = results_df.style.map(highlight_signals, subset=['Signal'])
    
    # Display Output
    st.subheader(f"Live Market Screener — Last Check: {datetime.now().strftime('%H:%M:%S IST')}")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Run Dashboard Fragment
render_live_signals()
