import pandas as pd
import numpy as np
from datetime import time

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 9 EMA, 20 EMA, and intraday VWAP."""
    df = df.copy()
    
    # Moving Averages
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Intraday VWAP Calculation
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return df

def analyze_ticker_signal(df: pd.DataFrame, ticker: str) -> dict:
    """Evaluates 9:30 AM ORB + VWAP + EMA strategy for a single ticker."""
    df = calculate_indicators(df)
    
    # 1. Isolate the 9:15 AM - 9:25 AM Opening Range
    df['Time'] = df['Timestamp'].dt.time
    orb_window = df[(df['Time'] >= time(9, 15)) & (df['Time'] <= time(9, 25))]
    
    if orb_window.empty:
        return {"Ticker": ticker, "Status": "WAITING", "Reason": "Insufficient Opening Data"}
        
    orb_high = orb_window['High'].max()
    orb_low = orb_window['Low'].min()
    
    # 2. Get current/latest 5-minute candle
    latest_candle = df.iloc[-1]
    current_price = latest_candle['Close']
    vwap_val = latest_candle['VWAP']
    ema9_val = latest_candle['EMA9']
    ema20_val = latest_candle['EMA20']
    
    # 3. Check Criteria
    above_vwap = current_price > vwap_val
    ema_bullish = ema9_val > ema20_val
    orb_breakout = current_price > orb_high
    
    # Target calculations based on risk
    stop_loss = round(latest_candle['Low'], 2)
    risk = current_price - stop_loss
    target_1to2 = round(current_price + (risk * 2.0), 2) if risk > 0 else 0.0

    # 4. Signal Matrix
    if orb_breakout and above_vwap and ema_bullish:
        signal = "BUY / LONG SETUP"
    elif current_price < orb_low and not above_vwap and not ema_bullish:
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
        "EMA 9/20": "BULLISH" if ema_bullish else "BEARISH",
        "Suggested SL": stop_loss,
        "Target (1:2)": target_1to2
    }
