import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import time
from PIL import Image
from google import genai
from google.genai.errors import APIError
from zoneinfo import ZoneInfo

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Professional Intraday Signal Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Dynamic Intraday Signal Screener + AI Chart Assistant")
st.caption("Strategy: Multi-Bar Breakout + VWAP Alignment + 9/20 EMA + ATR Dynamic Risk Management")

# -------------------------------------------------------------------
# 2. INTERNAL API KEY CONFIGURATION
# -------------------------------------------------------------------
try:
    INTERNAL_GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    INTERNAL_GEMINI_API_KEY = "PASTE_YOUR_NEW_KEY_HERE"

# -------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & CHATBOT WITH IMAGE UPLOAD
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Screener Controls")
default_symbols = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, LT, SBIN, AXISBANK"
watchlist_input = st.sidebar.text_area("Watchlist (NSE Tickers)", default_symbols, height=80)
watchlist = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

refresh_rate = st.sidebar.slider("Auto-Refresh Interval (Seconds)", 10, 120, 30)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk Management")
total_capital = st.sidebar.number_input("Total Trading Capital (₹)", value=100000, step=10000)
risk_per_trade_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.25)

st.sidebar.markdown("---")
st.sidebar.header("💬 AI Chart Auditor & Chat")

# Initialize Chat & Analysis States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_verdict" not in st.session_state:
    st.session_state.ai_verdict = None

# Image Uploader in Sidebar
uploaded_chart = st.sidebar.file_uploader("Upload Stock Chart (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded_chart:
    img_preview = Image.open(uploaded_chart)
    st.sidebar.image(img_preview, caption="Active Chart Context", use_container_width=True)

# Chat History Display
for role, message in st.session_state.chat_history:
    with st.sidebar.chat_message(role):
        st.write(message)

# Chat Input in Sidebar
user_chat_prompt = st.sidebar.chat_input("Ask Gemini about this chart/setup...")
if user_chat_prompt:
    st.session_state.chat_history.append(("user", user_chat_prompt))
    with st.sidebar.chat_message("user"):
        st.write(user_chat_prompt)

    # Call Gemini Multimodal with image + query
    with st.sidebar.chat_message("assistant"):
        with st.spinner("Analyzing chart patterns..."):
            try:
                client = genai.Client(api_key=INTERNAL_GEMINI_API_KEY)
                
                content_payload = [
                    "You are an expert intraday technical price-action trader. "
                    "Analyze the user query based on technical indicators, chart patterns, support/resistance, and volume."
                ]
                
                if uploaded_chart:
                    content_payload.append(Image.open(uploaded_chart))
                
                content_payload.append(f"User Query: {user_chat_prompt}")
                
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=content_payload
                )
                ai_reply = response.text
            except Exception as e:
                ai_reply = f"Error generating chart analysis: {str(e)}"
                
            st.write(ai_reply)
            st.session_state.chat_history.append(("assistant", ai_reply))

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# -------------------------------------------------------------------
# 4. MARKET DATA FETCHING
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
# 5. STRATEGY ENGINE & INDICATORS
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
# 6. SCREENER-LEVEL GEMINI AUDIT
# -------------------------------------------------------------------
def get_gemini_verdict(active_trades: list) -> str:
    if not INTERNAL_GEMINI_API_KEY or "PASTE_YOUR" in INTERNAL_GEMINI_API_KEY:
        return "⚠️ Please configure your Gemini API Key in `.streamlit/secrets.toml`."
    
    prompt = f"""
You are a disciplined intraday proprietary desk trader. Analyze these screener setups generated right now:
{active_trades}

For each active trade setup:
1. Validate if the risk-to-reward ratio and volatility buffers are realistic.
2. Provide a 2-sentence actionable execution plan (including trail stop advice).
3. State whether the setup has a high or low probability of false breakouts in typical Indian equity market conditions.
Be concise, practical, and objective.
"""
    models_to_try = ['gemini-3.6-flash', 'gemini-2.5-pro']
    
    try:
        client = genai.Client(api_key=INTERNAL_GEMINI_API_KEY)
    except Exception as e:
        return f"Client initialization failed: {str(e)}"

    for model_name in models_to_try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except APIError as e:
                if getattr(e, 'code', None) == 503 or "503" in str(e):
                    if attempt < max_retries - 1:
                        time.sleep(2 ** (attempt + 1))
                        continue
                break
            except Exception:
                break

    return "⚠️ The Gemini API is currently experiencing high load. Please try again in 15 seconds."

# -------------------------------------------------------------------
# 7. UI RENDERING & AUTO-REFRESH
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
    
    ist_now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime('%I:%M:%S %p IST')
st.subheader(f"Live Screener — {ist_now}")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    active_candidates = [s for s in signals if s['Signal'] not in ("NO TRADE / CHOP", "NO DATA")]
    
    st.markdown("---")
    st.subheader("🤖 AI Trade Auditor (Gemini)")
    if active_candidates:
        if st.button("Generate AI Risk & Probability Audit"):
            with st.spinner("Gemini is auditing active setups..."):
                st.session_state.ai_verdict = get_gemini_verdict(active_candidates)
    else:
        st.info("No active BUY or SELL setups currently detected. AI audit will be available when a breakout triggers.")

    if st.session_state.ai_verdict:
        st.markdown(st.session_state.ai_verdict)

render_live_signals()
