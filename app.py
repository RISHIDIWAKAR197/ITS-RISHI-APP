import time
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from nselib import capital_market

# --- FIXED LOGGING SETUP (Bug 5) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SYSTEM CONSTANTS ---
NSE_RETRIES = 2
NSE_RETRY_DELAY = 1.0

INDUSTRIAL_UNIVERSE = [
    "RELIANCE", "TATASTEEL", "LT", "MARUTI", "M&M", 
    "BHARTARTL", "NTPC", "POWERGRID", "COALINDIA", "ULTRACEMCO", 
    "GRASIM", "JSL", "JINDALSTEL", "HINDALCO", "BEL", "BHEL"
]

# --- SESSION STATE INITIALIZATION ---
if "market_data" not in st.session_state:
    st.session_state.market_data = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "error_logs" not in st.session_state:
    st.session_state.error_logs = {}

# ============================================================
# DATA ACCURACY & MATRICES HELPERS (ADX Corrected - Bug 4)
# ============================================================
def wilders_moving_average(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes a true Welles Wilder Moving Average at every index step."""
    vals = series.values
    out = np.zeros_like(vals)
    if len(vals) < period:
        return pd.Series(out, index=series.index)
    
    # Initialize seed with a simple moving average
    out[period - 1] = np.mean(vals[:period])
    
    # Recurse using Wilder's exact smoothing standard
    for i in range(period, len(vals)):
        out[i] = (out[i - 1] * (period - 1) + vals[i]) / period
        
    return pd.Series(out, index=series.index)

def calculate_vwap_by_session(df: pd.DataFrame) -> pd.Series:
    """Computes accurate intraday VWAP that strictly resets each trading day."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_pv = typical_price * df["Volume"]
    dates = df.index.date
    temp_df = pd.DataFrame({'cum_pv': cum_pv, 'vol': df['Volume'], 'date': dates}, index=df.index)
    grouped = temp_df.groupby('date')
    return grouped['cum_pv'].cumsum() / grouped['vol'].cumsum()

# --- LOW-LEVEL NSE NETWORK CALLER ---
def nse_call(fn, *args, **kwargs):
    last_err = None
    for attempt in range(NSE_RETRIES):
        try:
            result = fn(*args, **kwargs)
            if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                last_err = "Empty response received."
            else:
                return result, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < NSE_RETRIES - 1:
            time.sleep(NSE_RETRY_DELAY)
    return None, last_err

# --- CACHED LOT MATRIX RESILIENCY (Bug 3) ---
@st.cache_data(ttl=86400, show_spinner=False)
def get_fno_lot_sizes():
    """Fetches F&O lot sizes with robust hardcoded network fallbacks."""
    fallback = {
        "RELIANCE": 250, "TATASTEEL": 5500, "LT": 300, "MARUTI": 100, "M&M": 400,
        "BHARTARTL": 950, "NTPC": 1500, "POWERGRID": 3600, "COALINDIA": 1250, 
        "ULTRACEMCO": 100, "GRASIM": 400, "JSL": 1000, "JINDALSTEL": 1250, 
        "HINDALCO": 1400, "BEL": 2850, "BHEL": 5250
    }
    df, err = nse_call(capital_market.fno_equity_list)
    if err or df is None or "SYMBOL" not in df.columns or "LOT SIZE" not in df.columns:
        return fallback
    try:
        df.columns = [str(c).strip().upper() for c in df.columns]
        lots = pd.to_numeric(df["LOT SIZE"], errors="coerce")
        lot_map = dict(zip(df["SYMBOL"].str.strip().str.upper(), lots))
        return {k: int(v) for k, v in lot_map.items() if pd.notna(v)}
    except Exception:
        return fallback

# ============================================================
# QUANT MOMENTUM SCANNER ENGINE
# ============================================================
def fetch_momentum_metrics(symbol: str, benchmark_df: pd.DataFrame):
    """Calculates Technical Indicators with Standard Wilder's Metrics."""
    ticker = f"{symbol.strip().upper()}.NS"
    try:
        df = yf.download(tickers=ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None, "Ticker returned empty execution footprint or insufficient history."
        
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        
        df["VWAP"] = calculate_vwap_by_session(df)
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ROC"] = ((df["Close"] - df["Close"].shift(14)) / df["Close"].shift(14)) * 100
        
        # Standard Wilder's DMI/ADX Logic
        high_diff = df["High"].diff()
        low_diff = df["Low"].diff(1).multiply(-1)
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        
        smoothed_tr = wilders_moving_average(tr, 14)
        smoothed_plus_dm = wilders_moving_average(pd.Series(plus_dm, index=df.index), 14)
        smoothed_minus_dm = wilders_moving_average(pd.Series(minus_dm, index=df.index), 14)
        
        # Standardized Scaling Assured via exact WMA outputs
        df["+DI"] = 100 * (smoothed_plus_dm / np.where(smoothed_tr != 0, smoothed_tr, 1.0))
        df["-DI"] = 100 * (smoothed_minus_dm / np.where(smoothed_tr != 0, smoothed_tr, 1.0))
        
        di_sum = df["+DI"] + df["-DI"]
        di_diff = (df["+DI"] - df["-DI"]).abs()
        
        dx = np.where(di_sum != 0, 100 * (di_diff / di_sum), 0.0)
        df["ADX"] = wilders_moving_average(pd.Series(dx, index=df.index), 14)
        df["ATR"] = smoothed_tr
        
        df = df.join(benchmark_df[['Close']], rsuffix='_BENCH', how='inner')
        if not df.empty:
            stock_perf = (df["Close"] / df["Close"].iloc[0]) - 1
            bench_perf = (df["Close_BENCH"] / df["Close_BENCH"].iloc[0]) - 1
            df["RS"] = stock_perf - bench_perf
        else:
            df["RS"] = 0.0

        return df.iloc[-1].to_dict(), None
    except Exception as e:
        return None, f"Execution Failure: {str(e)}"

def scan_industrial_universe():
    bench = yf.download(tickers="^NSEI", period="5d", interval="5m", progress=False, auto_adjust=True)
    bench.columns = [c[0] if isinstance(c, tuple) else c for c in bench.columns]
    
    results = []
    errors = {}
    
    for sym in INDUSTRIAL_UNIVERSE:
        metrics, err_msg = fetch_momentum_metrics(sym, bench)
        if err_msg:
            errors[sym] = err_msg
        elif metrics:
            metrics["Symbol"] = sym
            results.append(metrics)
            
    st.session_state.error_logs = errors
    if not results:
        return pd.DataFrame()
        
    res_df = pd.DataFrame(results)
    res_df["Score"] = res_df["RS"] * 0.6 + (res_df["ADX"] / 100) * 0.4
    return res_df.sort_values(by="Score", ascending=False)

# ============================================================
# PERFORMANCE BATCHED HISTORICAL BACKTEST ENGINE (Bug 6)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def execute_historical_backtest(days_to_test=5):
    """Performs batched high-speed historic simulation across structural windows."""
    tickers = [f"{s}.NS" for s in INDUSTRIAL_UNIVERSE]
    all_tickers = tickers + ["^NSEI"]
    
    # Batch download all required tickers to prevent rate limits
    data_cluster = yf.download(tickers=all_tickers, period=f"{days_to_test + 4}d", interval="5m", progress=False)
    if data_cluster.empty:
        return []
        
    bench = data_cluster["Close"]["^NSEI"].dropna()
    unique_days = sorted(list(set(bench.index.date)))[-days_to_test:]
    trades = []
    
    for day in unique_days:
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=bench.index.tz)
        day_end = datetime.combine(day, datetime.max.time(), tzinfo=bench.index.tz)
        
        ranking_list = []
        for sym in INDUSTRIAL_UNIVERSE:
            tk = f"{sym}.NS"
            if tk not in data_cluster["Close"].columns:
                continue
                
            s_data = pd.DataFrame({
                "Open": data_cluster["Open"][tk],
                "High": data_cluster["High"][tk],
                "Low": data_cluster["Low"][tk],
                "Close": data_cluster["Close"][tk]
            }).dropna()
            
            historical_slice = s_data[s_data.index < day_start]
            target_session = s_data[(s_data.index >= day_start) & (s_data.index <= day_end)]
            
            if historical_slice.empty or target_session.empty:
                continue
                
            ref_close = historical_slice["Close"].iloc[-1]
            bench_history = bench[bench.index < day_start]
            if bench_history.empty:
                continue
            prev_close_bench = bench_history.iloc[-1]
            
            rs_proxy = (ref_close / historical_slice["Close"].iloc[0]) - (prev_close_bench / bench.iloc[0])
            ranking_list.append({
                "Symbol": sym, 
                "RS": rs_proxy, 
                "Session_Data": target_session, 
                "ATR_Ref": (historical_slice["High"] - historical_slice["Low"]).mean()
            })
            
        if not ranking_list:
            continue
            
        rank_df = pd.DataFrame(ranking_list).sort_values(by="RS", ascending=False)
        
        # Simulate Long Vector
        long_pick = rank_df.iloc[0]
        l_trigger = long_pick["Session_Data"]["Open"].iloc[0] * 1.0015
        l_sl = l_trigger - (long_pick["ATR_Ref"] * 2.0)
        l_t1 = l_trigger + ((l_trigger - l_sl) * 2.0)
        
        long_hit = long_pick["Session_Data"][long_pick["Session_Data"]["High"] >= l_trigger]
        if not long_hit.empty:
            exec_timeline = long_pick["Session_Data"][long_pick["Session_Data"].index >= long_hit.index[0]]
            hit_sl = exec_timeline[exec_timeline["Low"] <= l_sl]
            hit_t1 = exec_timeline[exec_timeline["High"] >= l_t1]
            
            if not hit_sl.empty and (hit_t1.empty or hit_sl.index[0] < hit_t1.index[0]):
                trades.append(-1.0)
            elif not hit_t1.empty:
                trades.append(2.0)
            else:
                trades.append(0.0)
                
    return trades

# ============================================================
# LAYOUT VIEWPORTS & RISK PARSING
# ============================================================
st.sidebar.header("🕹️ Control Center")
trading_mode = st.sidebar.radio("Choose Your Trading Mode:", ["📈 Intraday Cash (Shares)", "🔥 Stock Futures (Lots)"])
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Capital Management Framework")
if trading_mode == "📈 Intraday Cash (Shares)":
    capital = st.sidebar.number_input("Trading Capital (₹)", value=50000, step=5000)
    leverage = st.sidebar.number_input("MIS Leverage (x)", value=5, min_value=1, max_value=5)
    max_risk = st.sidebar.number_input("Max Risk Per Trade (₹)", value=500, step=50)
    buying_power = capital * leverage
    st.sidebar.info(f"Total Buying Power: **₹{buying_power:,}**")
else:
    capital = st.sidebar.number_input("Trading Margin (₹)", value=200000, step=10000)
    max_risk = st.sidebar.number_input("Max Risk Per Trade (₹)", value=5000, step=250)

st.sidebar.subheader("📐 Strategy Exit Bounds")
atr_multiplier = st.sidebar.number_input("ATR Multiplier", value=2.0, min_value=1.0, max_value=4.0, step=0.1)

tab_live, tab_backtest = st.tabs(["📡 Real-Time Momentum Scanner", "🧮 Strategy Backtester Modules"])

with tab_live:
    st.title("🏭 High-Velocity Industrial Engine")
    
    col_run, col_time = st.columns([1, 3])
    with col_run:
        trigger_scan = st.button("🔄 Run Scanner / Sync Terminals", use_container_width=True)
        
    if trigger_scan or st.session_state.market_data is None:
        with st.spinner("Compiling structural market matrix features..."):
            scanned_df = scan_industrial_universe()
            if not scanned_df.empty:
                st.session_state.market_data = scanned_df
                st.session_state.last_scan_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
                
    if st.session_state.error_logs:
        with st.expander("⚠️ Data Pipeline Status Reports (Network Latency Logs)"):
            for sym, log_err in st.session_state.error_logs.items():
                st.error(f"**{sym}**: {log_err}")
                
    if st.session_state.market_data is not None:
        df_display = st.session_state.market_data.copy()
        
        st.subheader("📊 Quant Screen Matrix Dashboard")
        st.dataframe(df_display[["Symbol", "Close", "RS", "ADX", "ROC", "VWAP", "EMA_20"]].style.format({
            "RS": "{:,.4f}", "ADX": "{:,.2f}", "ROC": "{:,.2f}%", "Close": "{:,.2f}", "VWAP": "{:,.2f}", "EMA_20": "{:,.2f}"
        }), use_container_width=True)
        
        bullish_candidate = df_display.iloc[0]
        short_filtered_pool = df_display[(df_display["RS"] < 0) & (df_display["ROC"] < 0)]
        
        def build_setups(row, direction="long"):
            entry = row["Close"]
            atr = row["ATR"] if (pd.notna(row["ATR"]) and row["ATR"] > 0) else entry * 0.005
            
            if direction == "long":
                trig = round(entry * 1.0015, 2)
                sl = round(trig - (atr * atr_multiplier), 2)
                risk = max(trig - sl, 0.05)
                return trig, sl, risk, round(trig + (risk * 2), 2), round(trig + (risk * 3), 2)
            else:
                trig = round(entry * 0.9985, 2)
                sl = round(trig + (atr * atr_multiplier), 2)
                risk = max(sl - trig, 0.05)
                return trig, sl, risk, round(trig - (risk * 2), 2), round(trig - (risk * 3), 2)

        st.markdown("---")
        
        # --- LONG SETUP VECTOR ---
        st.success(f"### 📈 ACCELERATED LONG CHANNEL: {bullish_candidate['Symbol']}")
        l_trig, l_sl, l_risk, l_t1, l_t2 = build_setups(bullish_candidate, "long")
        
        if trading_mode == "📈 Intraday Cash (Shares)":
            qty = max(min(int(max_risk // l_risk), int(buying_power // l_trig)), 1)
            st.info(f"👉 **Allocation Layout:** Buy **{qty} shares** at Trigger. Risk Capital: ₹{round(l_risk * qty, 2)}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Spot Entry Trigger", f"₹{l_trig}")
            c2.metric("Stop Loss", f"₹{l_sl}", delta=f"-{l_risk:.2f}")
            c3.metric("Target 1 (1:2)", f"₹{l_t1}")
            c4.metric("Target 2 (1:3)", f"₹{l_t2}")
        else:
            # --- DERIVATIVES RISK CONTROLS & LOCKOUTS (Bug 2) ---
            lot_dict = get_fno_lot_sizes()
            lot_size_long = lot_dict.get(bullish_candidate['Symbol'], 1)
            max_loss_long = round(l_risk * lot_size_long, 2)
            margin_req_long = round(l_trig * lot_size_long * 0.22, 2)
            
            st.info(f"ℹ️ **Exchange Margin Proxy Disclaimer:** Margin requirement calculated at ~22% rough SPAN+Exposure estimation framework.")
            
            if max_loss_long > max_risk:
                st.error(f"🚫 **EXECUTION LOCKOUT:** Single Lot Risk Exposure (₹{max_loss_long:,}) violates your parameter max risk configuration of ₹{max_risk}.")
            elif margin_req_long > capital:
                st.error(f"🚫 **EXECUTION LOCKOUT:** Estimated Exchange Margin requirement (~₹{margin_req_long:,}) exceeds your system capital settings profile (₹{capital:,}).")
            else:
                st.info(f"👉 **Derivative Setup:** 1 Lot ({lot_size_long} Units). Max Loss Exposure: ₹{max_loss_long}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spot Entry Trigger", f"₹{l_trig}")
                c2.metric("Stop Loss", f"₹{l_sl}", delta=f"-{l_risk:.2f}")
                c3.metric("Target 1 (1:2)", f"₹{l_t1}")
                c4.metric("Target 2 (1:3)", f"₹{l_t2}")
        
        st.markdown("---")
        
        # --- SHORT SETUP VECTOR (Variables Fixed - Bug 1) ---
        st.error("### 📉 DISTRESSED SHORT CHANNEL DISPATCH")
        if short_filtered_pool.empty:
            st.info("ℹ️ **No valid short setups found.** No industrial assets meet the necessary mathematical parameters for short positions (RS < 0 and ROC < 0).")
        else:
            bearish_candidate = short_filtered_pool.iloc[-1]
            st.error(f"Confirmed Short Execution Candidate Detected: **{bearish_candidate['Symbol']}**")
            s_trig, s_sl, s_risk, s_t1, s_t2 = build_setups(bearish_candidate, "short")
            
            if trading_mode == "📈 Intraday Cash (Shares)":
                qty_s = max(min(int(max_risk // s_risk), int(buying_power // s_trig)), 1)
                st.info(f"👉 **Allocation Layout:** Short **{qty_s} shares** at Trigger. Risk Capital: ₹{round(s_risk * qty_s, 2)}")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spot Entry Trigger", f"₹{s_trig}")
                c2.metric("Stop Loss", f"₹{s_sl}", delta=f"+{s_risk:.2f}", delta_color="inverse")
                c3.metric("Target 1 (1:2)", f"₹{s_t1}")
                c4.metric("Target 2 (1:3)", f"₹{s_t2}")
            else:
                # --- SHORT DERIVATIVES RISK CONTROLS & LOCKOUTS (Bug 2) ---
                lot_dict = get_fno_lot_sizes()
                lot_size_short = lot_dict.get(bearish_candidate['Symbol'], 1)
                max_loss_short = round(s_risk * lot_size_short, 2)
                margin_req_short = round(s_trig * lot_size_short * 0.22, 2)
                
                st.info(f"ℹ️ **Exchange Margin Proxy Disclaimer:** Margin requirement calculated at ~22% rough SPAN+Exposure estimation framework.")
                
                if max_loss_short > max_risk:
                    st.error(f"🚫 **EXECUTION LOCKOUT:** Single Lot Risk Exposure (₹{max_loss_short:,}) violates your parameter max risk configuration of ₹{max_risk}.")
                elif margin_req_short > capital:
                    st.error(f"🚫 **EXECUTION LOCKOUT:** Estimated Exchange Margin requirement (~₹{margin_req_short:,}) exceeds your system capital settings profile (₹{capital:,}).")
                else:
                    st.info(f"👉 **Derivative Setup:** 1 Lot ({lot_size_short} Units). Max Loss Exposure: ₹{max_loss_short}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Spot Entry Trigger", f"₹{s_trig}")
                    c2.metric("Stop Loss", f"₹{s_sl}", delta=f"+{s_risk:.2f}", delta_color="inverse")
                    c3.metric("Target 1 (1:2)", f"₹{s_t1}")
                    c4.metric("Target 2 (1:3)", f"₹{s_t2}")

with tab_backtest:
    st.title("🧮 Quantitative Verification Sandbox")
    st.caption("Verifies historic risk metrics over recent trading horizons.")
    
    run_backtest = st.button("🚀 Run Backtest Simulation Engine")
    if run_backtest:
        with st.spinner("Processing high-speed batched simulations..."):
            sample_results = execute_historical_backtest(days_to_test=5)
            if sample_results:
                arr = np.array(sample_results)
                win_rate = (np.sum(arr > 0) / len(arr)) * 100
                st.metric("Model Sample Win Rate", f"{win_rate:.2f}%")
                st.metric("Net Risk Multiplier Alpha Yield (R)", f"{np.sum(arr):.1f} R")
            else:
                st.info("Insufficient baseline data clusters within standard historical tracking windows to run verification maps.")
