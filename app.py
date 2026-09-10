import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from google import genai

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Professional Intraday Signal Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Dynamic Intraday Signal Screener + AI Trade Analyst")
st.caption("Strategy: Multi-Bar Breakout + VWAP Alignment + 9/20 EMA + ATR Dynamic Risk Management")

# -------------------------------------------------------------------
# 2. SIDEBAR CONTROLS & RISK MANAGEMENT
# -------------------------------------------------------------------
st.sidebar.header("🔑 API Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Pass your key here or via secrets.toml")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Screener Controls")
default_symbols = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, LT, SBIN, AXISBANK"
watchlist_input = st.sidebar.text_area("Watchlist (NSE Tickers)", default_symbols, height=100)
watchlist = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

refresh_rate = st.sidebar.slider("Auto-Refresh Interval (Seconds)", 10, 120, 30)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk Management")
total_capital = st.sidebar.number_input("Total Trading Capital (₹)", value=100000, step=10000)
risk_per_trade_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.25)

# -------------------------------------------------------------------
# 3. REAL MARKET DATA FETCHING
# -------------------------------------------------------------------
def fetch_live_data(ticker: str) -> pd.DataFrame:
    """Fetches intraday 5m data from Yahoo Finance."""
    formatted_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    
    try:
        df = yf.download(
            tickers=formatted_ticker, 
            period="1d", 
            interval="5m", 
            progress=False,
            multi_level_index=False
        )
        
        if df.empty or len(df) < 5:
            # Fallback to 5 days if market is closed or early pre-open
            df = yf.download(tickers=formatted_ticker, period="5d", interval="5m", progress=False, multi_level_index=False)
            if df.empty:
                return pd.DataFrame()
            
        df = df.reset_index()
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
    df = df.copy()
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, np.nan)
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
    
    return df

def analyze_ticker_signal(df: pd.DataFrame, ticker: str, capital: float, max_risk_pct: float) -> dict:
    if df.empty or len(df) < 5:
        return {"Ticker": ticker, "Signal": "NO DATA", "Last Price": 0.0, "Ref High": 0.0, "Ref Low": 0.0, "VWAP": 0.0, "Safe SL": 0.0, "Target (1:2)": 0.0, "Max Shares": 0}
        
    df = calculate_indicators(df)
    
    # Dynamic Range: uses the first 3 bars (15 mins) of available session data
    ref_window = df.iloc[:3]
    ref_high = float(ref_window['High'].max())
    ref_low = float(ref_window['Low'].min())
    
    latest = df.iloc[-1]
    current_price = float(latest['Close'])
    vwap_val = float(latest['VWAP'])
    atr_val = float(latest['ATR'])
    ema9_val = float(latest['EMA9'])
    ema20_val = float(latest['EMA20'])
    
    above_vwap = current_price > vwap_val
    ema_bullish = ema9_val > ema20_val
    breakout = current_price > ref_high
    breakdown = current_price < ref_low

    max_capital_risk = capital * (max_risk_pct / 100.0)
    
    if breakout and above_vwap and ema_bullish:
        signal = "BUY / LONG"
        safe_stop_loss = round(min(current_price - (1.5 * atr_val), ref_low), 2)
        risk_per_share = current_price - safe_stop_loss
        target_1to2 = round(current_price + (risk_per_share * 2.0), 2) if risk_per_share > 0 else current_price
        max_shares = int(max_capital_risk / risk_per_share) if risk_per_share > 0 else 0
    elif breakdown and not above_vwap and not ema_bullish:
        signal = "SELL / SHORT"
        safe_stop_loss = round(max(current_price + (1.5 * atr_val), ref_high), 2)
        risk_per_share = safe_stop_loss - current_price
        target_1to2 = round(current_price - (risk_per_share * 2.0), 2) if risk_per_share > 0 else current_price
        max_shares = int(max_capital_risk / risk_per_share) if risk_per_share > 0 else 0
    else:
        signal = "NO TRADE / CHOP"
        safe_stop_loss = 0.0
        target_1to2 = 0.0
        max_shares = 0

    return {
        "Ticker": ticker,
        "Signal": signal,
        "Last Price": round(current_price, 2),
        "Ref High": round(ref_high, 2),
        "Ref Low": round(ref_low, 2),
        "VWAP": round(vwap_val, 2),
        "Safe SL": safe_stop_loss,
        "Target (1:2)": target_1to2,
        "Max Shares": max_shares
    }

# -------------------------------------------------------------------
# 5. GEMINI AI ANALYSIS FUNCTION
# -------------------------------------------------------------------
def get_gemini_verdict(api_key: str, active_trades: list) -> str:
    if not api_key:
        return "Please input your Gemini API Key in the sidebar to generate AI trade insights."
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are a disciplined intraday proprietary desk trader. Analyze these screener setups generated right now:
{active_trades}

For each active trade setup:
1. Validate if the risk-to-reward ratio and volatility buffers are realistic.
2. Provide a 2-sentence actionable execution plan (including trail stop advice).
3. State whether the setup has a high or low probability of false breakouts in typical Indian equity market conditions.
Be concise, practical, and objective.
"""
        # Updated to gemini-3.6-flash
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Generation Failed: {str(e)}"
# -------------------------------------------------------------------
# 6. UI RENDERING & AUTO-REFRESH
# -------------------------------------------------------------------
@st.fragment(run_every=refresh_rate)
def render_live_signals():
    signals = []
    for ticker in watchlist:
        df = fetch_live_data(ticker)
        result = analyze_ticker_signal(df, ticker, total_capital, risk_per_trade_pct)
        signals.append(result)
    
    results_df = pd.DataFrame(signals)
    
    def highlight_signals(val):
        if "BUY" in str(val):
            return "background-color: #1e4620; color: #4CAF50; font-weight: bold;"
        elif "SELL" in str(val):
            return "background-color: #4a1e1e; color: #F44336; font-weight: bold;"
        return "color: #888888;"

    styled_df = results_df.style.map(highlight_signals, subset=['Signal'])
    
    st.subheader(f"Live Screener — {datetime.now().strftime('%H:%M:%S')}")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Filter for active candidates to show AI breakdown
    active_candidates = [s for s in signals if s['Signal'] != "NO TRADE / CHOP" and s['Signal'] != "NO DATA"]
    
    st.markdown("---")
    st.subheader("🤖 AI Trade Auditor (Gemini)")
    if active_candidates:
        if st.button("Generate AI Risk & Probability Audit"):
            with st.spinner("Gemini is auditing active setups..."):
                verdict = get_gemini_verdict(gemini_api_key, active_candidates)
                st.markdown(verdict)
    else:
        st.info("No active BUY or SELL setups currently detected. AI audit will be available when a breakout triggers.")

render_live_signals()
