import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Professional Intraday Signal Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Intraday Signal Screener (9:30 AM - 1:30 PM)")
st.caption("Strategy: 15-Min ORB + VWAP Alignment + 9/20 EMA + ATR Volatility Protection")

# -------------------------------------------------------------------
# 2. SIDEBAR CONTROLS & RISK MANAGEMENT PARAMETERS
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Screener Controls")

default_symbols = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, LT, SBIN, AXISBANK"
watchlist_input = st.sidebar.text_area("Watchlist (NSE Tickers)", default_symbols, height=120)
watchlist = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

refresh_rate = st.sidebar.slider("Auto-Refresh Interval (Seconds)", 5, 60, 10)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk Management")
total_capital = st.sidebar.number_input("Total Trading Capital (₹)", value=100000, step=10000)
risk_per_trade_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.5)

# -------------------------------------------------------------------
# 3. REAL MARKET DATA FETCHING (Yahoo Finance)
# -------------------------------------------------------------------
def fetch_live_data(ticker: str) -> pd.DataFrame:
    """Fetches real-time 5-minute intraday bar data from NSE."""
    formatted_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    
    try:
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
        
        # Normalize datetime column
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
        elif 'Date' in df.columns:
            df.rename(columns={'Date': 'Timestamp'}, inplace=True)
            
        return df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------------------------
# 4. STRATEGY ENGINE & TECHNICAL CALCULATIONS
# -------------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates EMA 9, EMA 20, Intraday VWAP, and ATR (14)."""
    df = df.copy()
    
    # Exponential Moving Averages
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Volume Weighted Average Price (VWAP)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Average True Range (ATR 14) for Dynamic Volatility SL
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
    
    return df

def analyze_ticker_signal(df: pd.DataFrame, ticker: str, capital: float, max_risk_pct: float) -> dict:
    """Evaluates 9:30 AM trading rules with ATR safety buffers and position sizing."""
    if df.empty or len(df) < 3:
        return {
            "Ticker": ticker, 
            "Signal": "NO DATA", 
            "Last Price": 0.0, 
            "ORB High": 0.0, 
            "ORB Low": 0.0, 
            "VWAP": 0.0, 
            "ATR (Noise)": 0.0,
            "Safe SL": 0.0, 
            "Target (1:2)": 0.0,
            "Max Shares": 0
        }
        
    df = calculate_indicators(df)
    
    # Isolate Opening 15-min Window (09:15 to 09:25)
    df['Time'] = pd.to_datetime(df['Timestamp']).dt.time
    orb_window = df[(df['Time'] >= time(9, 15)) & (df['Time'] <= time(9, 25))]
    
    if orb_window.empty:
        orb_high = float(df.iloc[:3]['High'].max())
        orb_low = float(df.iloc[:3]['Low'].min())
    else:
        orb_high = float(orb_window['High'].max())
        orb_low = float(orb_window['Low'].min())
    
    # Current bar parameters
    latest = df.iloc[-1]
    current_price = float(latest['Close'])
    vwap_val = float(latest['VWAP'])
    atr_val = float(latest['ATR'])
    ema9_val = float(latest['EMA9'])
    ema20_val = float(latest['EMA20'])
    
    # Safe Structural & Volatility Stop Loss Calculation
    atr_stop_loss = current_price - (1.5 * atr_val)
    structural_stop_loss = orb_low
    safe_stop_loss = round(min(atr_stop_loss, structural_stop_loss), 2)
    
    # Risk and Target Math
    risk_per_share = current_price - safe_stop_loss
    target_1to2 = round(current_price + (risk_per_share * 2.0), 2) if risk_per_share > 0 else current_price
    
    # Position Sizing Calculation (Caps risk to user defined %)
    max_capital_risk = capital * (max_risk_pct / 100.0)
    max_shares = int(max_capital_risk / risk_per_share) if risk_per_share > 0 else 0

    # Strategy Execution Rules
    above_vwap = current_price > vwap_val
    ema_bullish = ema9_val > ema20_val
    orb_breakout = current_price > orb_high
    orb_breakdown = current_price < orb_low

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
        "ORB High": round(orb_high, 2),
        "ORB Low": round(orb_low, 2),
        "VWAP": round(vwap_val, 2),
        "ATR (Noise)": round(atr_val, 2),
        "Safe SL": safe_stop_loss,
        "Target (1:2)": target_1to2,
        "Max Shares": max_shares
    }

# -------------------------------------------------------------------
# 5. STREAMLIT FRAGMENT (Live Auto-Refreshing UI)
# -------------------------------------------------------------------
@st.fragment(run_every=refresh_rate)
def render_live_signals():
    signals = []
    
    for ticker in watchlist:
        df = fetch_live_data(ticker)
        result = analyze_ticker_signal(df, ticker, total_capital, risk_per_trade_pct)
        signals.append(result)
    
    results_df = pd.DataFrame(signals)
    
    # Custom DataFrame Color Styling
    def highlight_signals(val):
        if "BUY" in str(val):
            return "background-color: #1e4620; color: #4CAF50; font-weight: bold;"
        elif "SELL" in str(val):
            return "background-color: #4a1e1e; color: #F44336; font-weight: bold;"
        return "color: #888888;"

    styled_df = results_df.style.map(highlight_signals, subset=['Signal'])
    
    # Header & Table Render
    st.subheader(f"Live Market Screener — Updated: {datetime.now().strftime('%H:%M:%S IST')}")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Execute Fragment
render_live_signals()
