import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="The Ennoble Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern dark/light contrast
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #2e344e;
        padding: 18px;
        border-radius: 10px;
        color: #ffffff;
        margin-bottom: 15px;
    }
    .stMetric {
        background: #111422;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Nifty Basket Universe
# ---------------------------------------------------------
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "TITAN.NS", "ADANIENT.NS", "BAJAJFINSV.NS", "TATASTEEL.NS", "M&M.NS",
    "ASIANPAINT.NS", "COALINDIA.NS", "JSWSTEEL.NS", "ULTRACEMCO.NS", "TECHM.NS",
    "WIPRO.NS", "GRASIM.NS", "HINDALCO.NS", "NESTLEIND.NS", "SHRIRAMFIN.NS",
    "CIPLA.NS", "HEROMOTOCO.NS", "DRREDDY.NS", "TATACONSUM.NS", "EICHERMOT.NS",
    "DIVISLAB.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "BPCL.NS", "BRITANNIA.NS",
    "BEL.NS", "TRENT.NS", "ADANIPORTS.NS", "SBILIFE.NS", "HDFCLIFE.NS"
]

# ---------------------------------------------------------
# Sidebar: Capital & Risk Management Settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("🕹️ Control Center")
    st.subheader("⚙️ Capital & Risk Allocation")
    
    capital = st.number_input("Trading Capital (₹)", min_value=5000, value=100000, step=5000)
    mis_leverage = st.slider("Intraday MIS Leverage (x)", min_value=1.0, max_value=5.0, value=5.0, step=0.5)
    max_risk_per_trade = st.number_input("Max Risk Per Trade (₹)", min_value=100, value=1000, step=100)
    
    buying_power = capital * mis_leverage
    st.markdown(f"**Total Buying Power:** `₹{buying_power:,.2f}`")
    st.markdown("---")
    
    st.subheader("🎯 Strategy Rules")
    sl_pct = st.slider("Stop Loss % per trade", min_value=0.2, max_value=3.0, value=0.75, step=0.05) / 100
    rr_ratio = st.slider("Risk-to-Reward Ratio (1 : X)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    target_pct = sl_pct * rr_ratio
    
    st.info(f"🛡️ **Risk:** {sl_pct*100:.2f}% | 🎯 **Target:** {target_pct*100:.2f}% (1:{rr_ratio:.1f})")

# ---------------------------------------------------------
# Helper Functions: Data Fetching & Ranking
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_live_market_data():
    """Fetch current daily performance for Nifty universe."""
    try:
        data = yf.download(NIFTY_50_TICKERS, period="5d", interval="1d", group_by='ticker', progress=False)
        records = []
        for sym in NIFTY_50_TICKERS:
            try:
                df = data[sym].dropna()
                if len(df) >= 2:
                    curr_close = float(df['Close'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2])
                    change_pct = ((curr_close - prev_close) / prev_close) * 100
                    high = float(df['High'].iloc[-1])
                    low = float(df['Low'].iloc[-1])
                    volume = int(df['Volume'].iloc[-1])
                    
                    records.append({
                        "Symbol": sym.replace(".NS", ""),
                        "Ticker": sym,
                        "Price (₹)": round(curr_close, 2),
                        "Change (%)": round(change_pct, 2),
                        "Day High (₹)": round(high, 2),
                        "Day Low (₹)": round(low, 2),
                        "Volume": volume
                    })
            except Exception:
                continue
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# Main UI & Navigation Tabs
# ---------------------------------------------------------
st.title("📊 The Ennoble Trader")
st.caption("Simplified Intraday Momentum Screener & Quantitative Backtesting Platform")

tab_live, tab_backtest = st.tabs(["⚡ Live Strategy & Scanners", "🧪 Enhanced Backtester"])

# =========================================================
# TAB 1: Live Top Gainer & Loser Strategy
# =========================================================
with tab_live:
    with st.spinner("Scanning Nifty Basket for Top Gainers and Losers..."):
        market_df = fetch_live_market_data()

    if market_df.empty:
        st.warning("Unable to fetch live market feeds. Please check your network or try again in a few seconds.")
    else:
        # Sort for Gainers & Losers
        sorted_df = market_df.sort_values(by="Change (%)", ascending=False).reset_index(drop=True)
        top_gainers = sorted_df.head(5)
        top_losers = sorted_df.tail(5).sort_values(by="Change (%)", ascending=True)

        st.subheader("🔥 Top 5 Gainers & Top 5 Losers Today")
        col_g, col_l = st.columns(2)
        
        with col_g:
            st.markdown("### 📈 Top Gainers (Bullish Momentum)")
            st.dataframe(
                top_gainers[["Symbol", "Price (₹)", "Change (%)", "Day High (₹)", "Volume"]],
                use_container_width=True,
                hide_index=True
            )
            
        with col_l:
            st.markdown("### 📉 Top Losers (Bearish Momentum)")
            st.dataframe(
                top_losers[["Symbol", "Price (₹)", "Change (%)", "Day Low (₹)", "Volume"]],
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")
        st.subheader("🎯 Auto-Calculated Intraday Action Setups")

        top_gainer = top_gainers.iloc[0]
        top_loser = top_losers.iloc[0]

        setup_col1, setup_col2 = st.columns(2)

        # Setup 1: Top Gainer Long Breakout
        with setup_col1:
            g_price = top_gainer["Price (₹)"]
            g_entry = round(g_price * 1.0015, 2)  # Entry slightly above current
            g_sl = round(g_entry * (1 - sl_pct), 2)
            g_risk_per_share = g_entry - g_sl
            g_qty = int(min(max_risk_per_trade / g_risk_per_share, buying_power / g_entry))
            g_tgt1 = round(g_entry * (1 + target_pct), 2)
            g_tgt2 = round(g_entry * (1 + target_pct * 1.5), 2)

            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid #00c853;">
                <h3>🟢 LONG SETUP: {top_gainer['Symbol']}</h3>
                <p><b>Condition:</b> Strongest Top Gainer (+{top_gainer['Change (%)']}%)</p>
                <hr style="border-color:#2e344e;">
                <p><b>🎯 Calculated Quantity:</b> {g_qty} shares</p>
                <p><b>🛡️ Max Risk Value:</b> ₹{round(g_qty * g_risk_per_share, 2)}</p>
                <p><b>🚀 Entry Trigger:</b> ₹{g_entry}</p>
                <p><b>🛑 Stop Loss:</b> ₹{g_sl} (-{sl_pct*100:.2f}%)</p>
                <p><b>🎯 Target 1 (1:{rr_ratio:.1f}):</b> ₹{g_tgt1} (+{target_pct*100:.2f}%)</p>
                <p><b>🎯 Target 2 (1:{rr_ratio*1.5:.1f}):</b> ₹{g_tgt2}</p>
            </div>
            """, unsafe_allow_html=True)

        # Setup 2: Top Loser Short Breakdown
        with setup_col2:
            l_price = top_loser["Price (₹)"]
            l_entry = round(l_price * 0.9985, 2)  # Entry slightly below current
            l_sl = round(l_entry * (1 + sl_pct), 2)
            l_risk_per_share = l_sl - l_entry
            l_qty = int(min(max_risk_per_trade / l_risk_per_share, buying_power / l_entry))
            l_tgt1 = round(l_entry * (1 - target_pct), 2)
            l_tgt2 = round(l_entry * (1 - target_pct * 1.5), 2)

            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid #ff3d00;">
                <h3>🔴 SHORT SETUP: {top_loser['Symbol']}</h3>
                <p><b>Condition:</b> Weakest Top Loser ({top_loser['Change (%)']}%)</p>
                <hr style="border-color:#2e344e;">
                <p><b>🎯 Calculated Quantity:</b> {l_qty} shares</p>
                <p><b>🛡️ Max Risk Value:</b> ₹{round(l_qty * l_risk_per_share, 2)}</p>
                <p><b>🚀 Entry Trigger:</b> ₹{l_entry}</p>
                <p><b>🛑 Stop Loss:</b> ₹{l_sl} (+{sl_pct*100:.2f}%)</p>
                <p><b>🎯 Target 1 (1:{rr_ratio:.1f}):</b> ₹{l_tgt1} (-{target_pct*100:.2f}%)</p>
                <p><b>🎯 Target 2 (1:{rr_ratio*1.5:.1f}):</b> ₹{l_tgt2}</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 2: Enhanced Backtesting Engine
# =========================================================
with tab_backtest:
    st.subheader("🧪 Top Gainer/Loser Timing & Risk-Reward Backtester")
    
    # Backtest Configuration Row
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        lookback_days = st.slider("Backtest Window (Days)", min_value=15, max_value=60, value=30)
    with b_col2:
        entry_time_val = st.time_input("Entry Time (IST)", time(9, 30))
    with b_col3:
        exit_time_val = st.time_input("Square-Off Time (IST)", time(15, 15))
    with b_col4:
        max_daily_trades = st.slider("Max Trades per Day", min_value=1, max_value=4, value=2)

    run_btn = st.button("🚀 Run Enhanced Backtest", type="primary")

    if run_btn:
        with st.spinner("Downloading historical 5-minute candle feeds and computing executions..."):
            # Fetch intraday data for representative liquid universe
            sample_universe = NIFTY_50_TICKERS[:15]  # Balanced representative sample for fast execution
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            try:
                hist_data = yf.download(
                    tickers=sample_universe,
                    start=start_date,
                    interval="5m",
                    group_by="ticker",
                    progress=False
                )
            except Exception as e:
                st.error(f"Failed to fetch historical 5m data: {e}")
                hist_data = None

            if hist_data is not None and not hist_data.empty:
                trade_records = []
                
                # Extract unique dates
                all_timestamps = hist_data.index
                unique_dates = sorted(list(set([ts.date() for ts in all_timestamps])))
                
                for d in unique_dates:
                    daily_perf = []
                    for sym in sample_universe:
                        try:
                            df_sym = hist_data[sym].dropna()
                            df_day = df_sym[df_sym.index.date == d]
                            if len(df_day) > 5:
                                open_p = df_day['Open'].iloc[0]
                                entry_df = df_day[df_day.index.time >= entry_time_val]
                                if not entry_df.empty:
                                    entry_bar_price = entry_df['Open'].iloc[0]
                                    change = ((entry_bar_price - open_p) / open_p) * 100
                                    daily_perf.append((sym, change, entry_df))
                        except Exception:
                            continue
                    
                    if not daily_perf:
                        continue
                    
                    # Sort candidates
                    daily_perf.sort(key=lambda x: x[1], reverse=True)
                    
                    trades_today = 0
                    # Trade 1: Long Top Gainer
                    if len(daily_perf) > 0 and trades_today < max_daily_trades:
                        sym, chg, df_candles = daily_perf[0]
                        entry_price = df_candles['Open'].iloc[0]
                        target_price = entry_price * (1 + target_pct)
                        stop_loss_price = entry_price * (1 - sl_pct)
                        risk_per_share = entry_price - stop_loss_price
                        qty = int(min(max_risk_per_trade / risk_per_share, buying_power / entry_price))
                        
                        outcome = "SQUARE_OFF"
                        exit_price = df_candles['Close'].iloc[-1]
                        
                        for idx, row in df_candles.iterrows():
                            if row.name.time() > exit_time_val:
                                exit_price = row['Close']
                                break
                            if row['High'] >= target_price:
                                exit_price = target_price
                                outcome = "TARGET"
                                break
                            if row['Low'] <= stop_loss_price:
                                exit_price = stop_loss_price
                                outcome = "STOP_LOSS"
                                break
                        
                        pnl = (exit_price - entry_price) * qty
                        trade_records.append({
                            "Date": d,
                            "Symbol": sym.replace(".NS", ""),
                            "Type": "LONG",
                            "Entry (₹)": round(entry_price, 2),
                            "Exit (₹)": round(exit_price, 2),
                            "Qty": qty,
                            "PnL (₹)": round(pnl, 2),
                            "Return (%)": round(((exit_price - entry_price) / entry_price) * 100, 2),
                            "Outcome": outcome
                        })
                        trades_today += 1

                    # Trade 2: Short Top Loser
                    if len(daily_perf) > 1 and trades_today < max_daily_trades:
                        sym, chg, df_candles = daily_perf[-1]
                        entry_price = df_candles['Open'].iloc[0]
                        target_price = entry_price * (1 - target_pct)
                        stop_loss_price = entry_price * (1 + sl_pct)
                        risk_per_share = stop_loss_price - entry_price
                        qty = int(min(max_risk_per_trade / risk_per_share, buying_power / entry_price))
                        
                        outcome = "SQUARE_OFF"
                        exit_price = df_candles['Close'].iloc[-1]
                        
                        for idx, row in df_candles.iterrows():
                            if row.name.time() > exit_time_val:
                                exit_price = row['Close']
                                break
                            if row['Low'] <= target_price:
                                exit_price = target_price
                                outcome = "TARGET"
                                break
                            if row['High'] >= stop_loss_price:
                                exit_price = stop_loss_price
                                outcome = "STOP_LOSS"
                                break
                        
                        pnl = (entry_price - exit_price) * qty
                        trade_records.append({
                            "Date": d,
                            "Symbol": sym.replace(".NS", ""),
                            "Type": "SHORT",
                            "Entry (₹)": round(entry_price, 2),
                            "Exit (₹)": round(exit_price, 2),
                            "Qty": qty,
                            "PnL (₹)": round(pnl, 2),
                            "Return (%)": round(((entry_price - exit_price) / entry_price) * 100, 2),
                            "Outcome": outcome
                        })
                        trades_today += 1

                # -------------------------------------------------
                # Display Backtesting Metrics & Charts
                # -------------------------------------------------
                if trade_records:
                    bt_df = pd.DataFrame(trade_records)
                    bt_df['Cumulative PnL'] = bt_df['PnL (₹)'].cumsum()
                    
                    total_trades = len(bt_df)
                    winning_trades = len(bt_df[bt_df['PnL (₹)'] > 0])
                    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                    total_profit = bt_df['PnL (₹)'].sum()
                    max_dd = (bt_df['Cumulative PnL'].cummax() - bt_df['Cumulative PnL']).max()
                    
                    st.markdown("### 📈 Performance Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Trades Executed", total_trades)
                    m2.metric("Win Rate", f"{win_rate:.1f}%")
                    m3.metric("Net Strategy P&L", f"₹{total_profit:,.2f}", delta=f"{total_profit:,.2f}")
                    m4.metric("Max Strategy Drawdown", f"₹{max_dd:,.2f}")

                    # Equity Curve Chart
                    st.markdown("#### 📉 Cumulative Equity Growth")
                    fig = px.line(
                        bt_df,
                        x="Date",
                        y="Cumulative PnL",
                        title="Cumulative Net P&L Curve (₹)",
                        markers=True,
                        template="plotly_dark"
                    )
                    fig.update_layout(xaxis_title="Date", yaxis_title="Net PnL (₹)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed Log Table
                    st.markdown("#### 📜 Executed Trades Log")
                    st.dataframe(bt_df, use_container_width=True)
                else:
                    st.info("No trades matched the criteria during this backtesting window.")
            else:
                st.warning("Historical data was empty or unavailable for the selected period.")

# ---------------------------------------------------------
# Footer Guardrail
# ---------------------------------------------------------
st.markdown("---")
st.caption("⚠️ **Execution Guardrail:** *Intraday momentum strategies require strict discipline. Always place Stop-Loss Limit orders directly in your broker terminal to protect your trading capital.*")
