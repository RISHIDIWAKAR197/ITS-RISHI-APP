import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from nselib import capital_market, derivatives
from nselib.libutil import nse_urlfetch

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="The Ennoble Trader | Multi-Asset Intraday Tool",
    page_icon="📊",
    layout="wide",
)

NSE_RETRIES = 3
NSE_RETRY_DELAY = 1.5  # seconds


# ============================================================
# LOW-LEVEL HELPERS
# ============================================================
def safe_float(val):
    """Convert to float, return None (not 0.0) if it can't be parsed."""
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def nse_call(fn, *args, **kwargs):
    """
    Call an NSE-hitting function with retries.
    Returns (result, error_message). error_message is None on success.
    Never swallows the error silently - always returns what went wrong.
    """
    last_err = None
    for attempt in range(NSE_RETRIES):
        try:
            result = fn(*args, **kwargs)
            if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                last_err = "NSE returned an empty response."
            else:
                return result, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < NSE_RETRIES - 1:
            time.sleep(NSE_RETRY_DELAY)
    return None, last_err


# ============================================================
# DATA FETCHERS (all real NSE/Yahoo data - no hardcoded fallbacks)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_lot_sizes():
    """Live F&O lot-size directory. Returns (dict, error_message)."""
    df, err = nse_call(capital_market.fno_equity_list)
    if err:
        return {}, err
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns or "LOT SIZE" not in df.columns:
        return {}, "Unexpected F&O list format from NSE (columns changed)."
    lots = pd.to_numeric(df["LOT SIZE"], errors="coerce")
    lot_map = dict(zip(df["SYMBOL"].str.strip().str.upper(), lots))
    lot_map = {k: int(v) for k, v in lot_map.items() if pd.notna(v)}
    return lot_map, None


def lookup_lot_size(symbol, lot_dict):
    """Returns the real lot size, or None if unknown. Never guesses."""
    return lot_dict.get(str(symbol).strip().upper())


@st.cache_data(ttl=60, show_spinner=False)
def get_top_movers(direction: str):
    """
    direction: 'gainers' or 'losers' (mapped to NSE's own 'loosers' spelling internally).
    Returns (symbol, ltp, error_message).
    """
    nse_param = "gainers" if direction == "gainers" else "loosers"
    df, err = nse_call(capital_market.top_gainers_or_losers, to_get=nse_param)
    if err:
        return None, None, err
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns or "LTP" not in df.columns:
        return None, None, "Unexpected gainers/losers format from NSE (columns changed)."
    symbol = str(df.iloc[0]["SYMBOL"]).strip().upper()
    ltp = safe_float(df.iloc[0]["LTP"])
    if not symbol or ltp is None or ltp <= 0:
        return None, None, "NSE returned an invalid top-mover record."
    return symbol, ltp, None


API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def nse_api_fetch(url: str, referer: str):
    """
    Standalone JSON-API fetcher (separate from nselib's HTML-oriented
    nse_urlfetch) tuned for NSE's XHR endpoints. On failure, returns
    enough of the raw HTTP response to diagnose *why* (status code +
    body snippet) instead of just a JSON-decode error.
    """
    session = requests.Session()
    session.headers.update(API_HEADERS)
    # Prime cookies from the homepage - required before any /api/ call.
    session.get("https://www.nseindia.com/", timeout=10)
    time.sleep(0.5)
    resp = session.get(url, headers={"Referer": referer}, timeout=10)

    body_snippet = (resp.text or "")[:200].replace("\n", " ").strip()
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from NSE. Body: '{body_snippet}'")
    if not resp.text or not resp.text.strip():
        raise RuntimeError(f"NSE returned an empty body (HTTP {resp.status_code}, likely blocked/rate-limited).")
    try:
        return resp.json()
    except ValueError:
        raise RuntimeError(f"NSE returned non-JSON content. Body starts with: '{body_snippet}'")


@st.cache_data(ttl=15, show_spinner=False)
def get_live_futures_quote(symbol: str):
    """
    Fetches the ACTUAL live near-month futures LTP from NSE's real-time
    quote-derivative endpoint (the same data that powers nseindia.com's
    own quote page). This is a genuine live tick, unlike the historical
    archive used by nselib's future_price_volume_data().
    Returns (ltp, lot_size, expiry_str, error_message).
    """
    clean_sym = str(symbol).strip().upper()
    url = f"https://www.nseindia.com/api/quote-derivative?symbol={clean_sym}"
    referer = f"https://www.nseindia.com/get-quotes/derivatives?symbol={clean_sym}"

    last_err = None
    data = None
    live_attempts = 1  # NSE's quote-derivative endpoint is hard-blocked from this host;
    # retrying repeatedly just adds latency without changing the outcome.
    for attempt in range(live_attempts):
        try:
            data = nse_api_fetch(url, referer)
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            data = None
        if attempt < live_attempts - 1:
            time.sleep(NSE_RETRY_DELAY)

    if data is None:
        return None, None, None, last_err or "No response from NSE live endpoint."

    stocks = data.get("stocks") if isinstance(data, dict) else None
    if not stocks:
        return None, None, None, "Live quote endpoint returned no derivative contracts for this symbol."

    futures_rows = []
    for item in stocks:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata", {}) or {}
        if "future" not in str(meta.get("instrumentType", "")).lower():
            continue
        expiry_parsed = pd.to_datetime(meta.get("expiryDate"), format="%d-%b-%Y", errors="coerce")
        ltp = safe_float(meta.get("lastPrice"))
        trade_info = (item.get("marketDeptOrderBook") or {}).get("tradeInfo") or {}
        lot_size = safe_float(trade_info.get("marketLot")) or safe_float(meta.get("marketLot"))
        if pd.notna(expiry_parsed) and ltp:
            futures_rows.append((expiry_parsed, ltp, lot_size, meta.get("expiryDate")))

    if not futures_rows:
        return None, None, None, "NSE response didn't contain a parsable futures contract (schema may have shifted)."

    futures_rows.sort(key=lambda r: r[0])
    _, ltp, lot_size, expiry_str = futures_rows[0]
    if not ltp or ltp <= 0:
        return None, None, None, "Live futures LTP came back invalid/zero."

    return ltp, int(lot_size) if lot_size else None, expiry_str, None


@st.cache_data(ttl=60, show_spinner=False)
def get_eod_futures_quote(symbol: str):
    """
    Fallback source: NSE's historical/EOD derivatives archive
    (api/historicalOR/foCPV). NOTE - this reflects the last recorded
    session's data, not a real-time tick, so it can lag the live market.
    Used only when get_live_futures_quote() fails.
    Returns (ltp, lot_size, expiry_str, error_message).
    """
    clean_sym = str(symbol).strip().upper()
    df, err = nse_call(
        derivatives.future_price_volume_data,
        symbol=clean_sym,
        instrument="FUTSTK",
        period="1D",
    )
    if err:
        return None, None, None, err

    df.columns = [str(c).strip().upper() for c in df.columns]
    required = {"LAST_TRADED_PRICE", "EXPIRY_DT", "MARKET_LOT"}
    if not required.issubset(df.columns):
        return None, None, None, "Unexpected futures data format from NSE (columns changed)."

    df["EXPIRY_DT_PARSED"] = pd.to_datetime(df["EXPIRY_DT"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["EXPIRY_DT_PARSED"]).sort_values("EXPIRY_DT_PARSED")
    if df.empty:
        return None, None, None, "No active futures contract found for this symbol."

    near = df.iloc[0]
    ltp = safe_float(near["LAST_TRADED_PRICE"])
    lot_size = safe_float(near["MARKET_LOT"])
    expiry = near["EXPIRY_DT_PARSED"].strftime("%d %b %Y")

    if not ltp or ltp <= 0:
        return None, None, None, "Futures contract found but LTP was invalid/zero."

    return ltp, int(lot_size) if lot_size else None, expiry, None


def get_atr(symbol: str, period: int = 14):
    """
    Intraday 14-period ATR from 5-min candles via Yahoo Finance.
    Returns (atr_value, error_message).
    """
    ticker = symbol.upper().strip()
    if not ticker.endswith(".NS"):
        ticker += ".NS"
    try:
        df = yf.download(
            tickers=ticker, period="5d", interval="5m", progress=False, auto_adjust=True
        )
        if df.empty or len(df) < period + 1:
            return None, "Not enough intraday history returned by Yahoo Finance."

        high = df["High"].iloc[:, 0] if isinstance(df["High"], pd.DataFrame) else df["High"]
        low = df["Low"].iloc[:, 0] if isinstance(df["Low"], pd.DataFrame) else df["Low"]
        close = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        if pd.isna(atr):
            return None, "ATR calculation returned NaN."
        return round(float(atr), 2), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("🕹️ Control Center")
trading_mode = st.sidebar.radio(
    "Choose Your Trading Mode:",
    ["📈 Intraday Cash (Shares)", "🔥 Stock Futures (Lots)"],
    help="Toggle between individual equity risk sizing or standardized derivatives.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Capital & Risk Settings")

if trading_mode == "📈 Intraday Cash (Shares)":
    capital = st.sidebar.number_input("Trading Capital (₹)", value=30000, step=1000)
    leverage = st.sidebar.number_input("MIS Leverage (x)", value=5, min_value=1, max_value=5)
    max_risk = st.sidebar.number_input("Max Risk Per Trade (₹)", value=300, step=10)
    buying_power = capital * leverage
    st.sidebar.info(f"Total Buying Power: **₹{buying_power:,}**")
else:
    capital = st.sidebar.number_input("Trading Margin (₹)", value=170000, step=10000)
    max_risk = st.sidebar.number_input("Max Risk Per Trade (₹)", value=5000, step=250)
    st.sidebar.warning(
        "🛡️ Contracts exceeding your absolute max risk or available margin limits will automatically block."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("📐 ATR Parameters (Shared)")
atr_period = st.sidebar.number_input("ATR Period (candles)", value=14, min_value=5, max_value=50)
atr_multiplier = st.sidebar.number_input(
    "ATR Multiplier", value=1.5, min_value=0.5, max_value=5.0, step=0.1
)

# --- HEADER ---
IST_NOW = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
st.title("📊 RISHI's Multi-Asset Momentum Dashboard")
st.caption(
    f"Engine Mode: **{trading_mode.upper()}** • System Time: {IST_NOW.strftime('%d %b %Y | %H:%M IST')}"
)
st.markdown("---")

# ============================================================
# LIVE FEED
# ============================================================
st.subheader("📡 Nifty Live Market Feed")
chosen_feed = st.radio(
    "Select Feed Input Source:",
    ["🤖 Automated Nifty Scanner", "✍️ Manual Entry"],
    horizontal=True,
)

bullish_stock = bullish_ltp = bearish_stock = bearish_ltp = None

if chosen_feed == "🤖 Automated Nifty Scanner":
    with st.spinner("Fetching live NSE gainers/losers..."):
        bullish_stock, bullish_ltp, gain_err = get_top_movers("gainers")
        bearish_stock, bearish_ltp, lose_err = get_top_movers("losers")

    if gain_err or lose_err:
        st.error(
            "🚫 **Live NSE feed unavailable.** NSE India frequently blocks requests from cloud "
            "servers (Streamlit Cloud, AWS, etc.), which is the most common cause of this."
        )
        with st.expander("🔍 Technical details"):
            st.write(f"Gainers error: {gain_err or 'OK'}")
            st.write(f"Losers error: {lose_err or 'OK'}")
        st.info("Switch to **Manual Entry** above and enter real symbols/prices to continue.")
        st.stop()

    st.success(
        f"✅ Live NSE Data Active! Top Gainer: **{bullish_stock}** (₹{bullish_ltp}) | "
        f"Top Loser: **{bearish_stock}** (₹{bearish_ltp})"
    )
else:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        bullish_stock = st.text_input("Top Gainer Symbol", value="", placeholder="e.g. RELIANCE")
        bullish_ltp = st.number_input("Gainer Live Price (₹)", min_value=0.0, value=0.0, step=0.05)
    with col_m2:
        bearish_stock = st.text_input("Top Loser Symbol", value="", placeholder="e.g. TCS")
        bearish_ltp = st.number_input("Loser Live Price (₹)", min_value=0.0, value=0.0, step=0.05)

    if not bullish_stock.strip() or not bearish_stock.strip() or bullish_ltp <= 0 or bearish_ltp <= 0:
        st.info("👆 Enter both symbols and their live prices to generate trade setups.")
        st.stop()

    bullish_stock = bullish_stock.strip().upper()
    bearish_stock = bearish_stock.strip().upper()

st.markdown("---")

# ============================================================
# FUTURES PRICING (real futures data, honest fallback to spot)
# ============================================================
bull_price_tag = bear_price_tag = "Spot"
bull_lot_size = bear_lot_size = None


def resolve_futures_price(symbol, spot_price):
    """
    Tries live quote -> EOD historical quote -> spot, in that order.
    Returns (price, lot_size, tag, debug_lines) where tag is one of
    'Live Futures', 'EOD Futures', or 'Spot (futures unavailable)'.
    """
    debug_lines = []

    ltp, lot_size, expiry, err = get_live_futures_quote(symbol)
    if not err:
        return ltp, lot_size, f"Live Futures ({expiry})", debug_lines
    debug_lines.append(f"Live quote failed: {err}")

    ltp, lot_size, expiry, err = get_eod_futures_quote(symbol)
    if not err:
        return ltp, lot_size, f"EOD Futures ({expiry}, previous session)", debug_lines
    debug_lines.append(f"EOD quote failed: {err}")

    return spot_price, None, "Spot (futures unavailable)", debug_lines


if trading_mode == "🔥 Stock Futures (Lots)":
    with st.spinner("Fetching live near-month futures quotes from NSE..."):
        bull_price, bull_lot_size, bull_price_tag, bull_dbg = resolve_futures_price(
            bullish_stock, bullish_ltp
        )
        bear_price, bear_lot_size, bear_price_tag, bear_dbg = resolve_futures_price(
            bearish_stock, bearish_ltp
        )
        lot_map, lot_err = get_fno_lot_sizes()

    bull_is_futures = bull_price_tag.startswith(("Live", "EOD"))
    bear_is_futures = bear_price_tag.startswith(("Live", "EOD"))

    if bull_is_futures:
        st.info(f"⚡ **{bullish_stock}** — {bull_price_tag} = **₹{bull_price}**")
        if bull_price_tag.startswith("EOD"):
            st.caption("⏳ This is last-recorded-session data, not live. Verify against your broker terminal below.")
    else:
        st.warning(f"⚠️ **{bullish_stock}**: futures data unavailable. Showing **spot price (₹{bull_price})** below.")
        if bull_lot_size is None:
            bull_lot_size = lookup_lot_size(bullish_stock, lot_map)
    bullish_ltp = st.number_input(
        f"{bullish_stock} Futures Price - verify/override (₹)",
        min_value=0.0,
        value=float(bull_price),
        step=0.05,
        key="bull_fut_override",
        help="Pre-filled from NSE. Overwrite with your broker's live LTP if this looks stale.",
    )
    if bull_dbg:
        with st.expander(f"🔍 {bullish_stock} futures fetch details"):
            for line in bull_dbg:
                st.write(line)

    if bear_is_futures:
        st.info(f"⚡ **{bearish_stock}** — {bear_price_tag} = **₹{bear_price}**")
        if bear_price_tag.startswith("EOD"):
            st.caption("⏳ This is last-recorded-session data, not live. Verify against your broker terminal below.")
    else:
        st.warning(f"⚠️ **{bearish_stock}**: futures data unavailable. Showing **spot price (₹{bear_price})** below.")
        if bear_lot_size is None:
            bear_lot_size = lookup_lot_size(bearish_stock, lot_map)
    bearish_ltp = st.number_input(
        f"{bearish_stock} Futures Price - verify/override (₹)",
        min_value=0.0,
        value=float(bear_price),
        step=0.05,
        key="bear_fut_override",
        help="Pre-filled from NSE. Overwrite with your broker's live LTP if this looks stale.",
    )
    if bear_dbg:
        with st.expander(f"🔍 {bearish_stock} futures fetch details"):
            for line in bear_dbg:
                st.write(line)

    if bull_lot_size is None or bear_lot_size is None:
        missing = []
        if bull_lot_size is None:
            missing.append(bullish_stock)
        if bear_lot_size is None:
            missing.append(bearish_stock)
        st.error(
            f"🚫 Could not determine a real lot size for: **{', '.join(missing)}**. "
            "This usually means the symbol isn't currently F&O-enabled, or NSE's lot-size "
            "directory is unreachable. Refusing to guess a lot size - please verify the "
            "symbol and try again."
        )
        st.stop()

# ============================================================
# ATR (shared)
# ============================================================
with st.spinner("Fetching historical intraday data from Yahoo Finance..."):
    bull_atr, bull_atr_err = get_atr(bullish_stock, period=atr_period)
    bear_atr, bear_atr_err = get_atr(bearish_stock, period=atr_period)

atr_note_bull = "" if bull_atr else " *(ATR unavailable — using 0.5% of price as an estimate)*"
atr_note_bear = "" if bear_atr else " *(ATR unavailable — using 0.5% of price as an estimate)*"


# ============================================================
# SHARED CALC HELPERS
# ============================================================
def build_long_setup(entry_price, atr_value):
    entry = round(entry_price * 1.002, 2)
    sl_dist = round(atr_value * atr_multiplier, 2) if atr_value else round(entry * 0.005, 2)
    sl = round(entry - sl_dist, 2)
    risk = max(round(entry - sl, 2), 0.05)
    t1 = round(entry + risk * 2, 2)
    t2 = round(entry + risk * 3, 2)
    return entry, sl, risk, t1, t2


def build_short_setup(entry_price, atr_value):
    entry = round(entry_price * 0.998, 2)
    sl_dist = round(atr_value * atr_multiplier, 2) if atr_value else round(entry * 0.005, 2)
    sl = round(entry + sl_dist, 2)
    risk = max(round(sl - entry, 2), 0.05)
    t1 = round(entry - risk * 2, 2)
    t2 = round(entry - risk * 3, 2)
    return entry, sl, risk, t1, t2


def pct_of(entry, value):
    """Percentage move of `value` relative to `entry`, entry-safe."""
    if not entry:
        return 0.0
    return round(((value - entry) / entry) * 100, 2)


# ============================================================
# MODULE 1: INTRADAY CASH ENGINE
# ============================================================
if trading_mode == "📈 Intraday Cash (Shares)":
    long_entry, long_sl, long_risk, long_t1, long_t2 = build_long_setup(bullish_ltp, bull_atr)
    if long_entry > 0 and long_risk > 0:
        long_qty = max(min(int(max_risk // long_risk), int(buying_power // long_entry)), 1)

        st.success(f"### 📈 INTRADAY CASH LONG: {bullish_stock}{atr_note_bull}")
        q1, q2 = st.columns(2)
        q1.info(f"### 🎯 TRADE QUANTITY: **{long_qty} shares**")
        q2.error(f"### 🛡️ MAX LOSS RISK: **₹{round(long_risk * long_qty, 2):,}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry Trigger", f"₹{long_entry}")
        c2.metric(
            "Stop Loss", f"₹{long_sl}",
            delta=f"{pct_of(long_entry, long_sl)}% (-{long_risk} pts)", delta_color="inverse",
        )
        c3.metric("Target 1 (1:2)", f"₹{long_t1}", delta=f"+{pct_of(long_entry, long_t1)}% (+{round(long_t1 - long_entry, 2)} pts)")
        c4.metric("Target 2 (1:3)", f"₹{long_t2}", delta=f"+{pct_of(long_entry, long_t2)}% (+{round(long_t2 - long_entry, 2)} pts)")
    else:
        st.warning(f"⚠️ Long setup for {bullish_stock} unavailable (invalid price/risk).")

    st.markdown("---")

    short_entry, short_sl, short_risk, short_t1, short_t2 = build_short_setup(bearish_ltp, bear_atr)
    if short_entry > 0 and short_risk > 0:
        short_qty = max(min(int(max_risk // short_risk), int(buying_power // short_entry)), 1)

        st.error(f"### 📉 INTRADAY CASH SHORT: {bearish_stock}{atr_note_bear}")
        q1, q2 = st.columns(2)
        q1.info(f"### 🎯 TRADE QUANTITY: **{short_qty} shares**")
        q2.error(f"### 🛡️ MAX LOSS RISK: **₹{round(short_risk * short_qty, 2):,}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry Trigger", f"₹{short_entry}")
        c2.metric(
            "Stop Loss", f"₹{short_sl}",
            delta=f"+{pct_of(short_entry, short_sl)}% (+{short_risk} pts)", delta_color="inverse",
        )
        c3.metric("Target 1 (1:2)", f"₹{short_t1}", delta=f"{pct_of(short_entry, short_t1)}% ({round(short_t1 - short_entry, 2)} pts)")
        c4.metric("Target 2 (1:3)", f"₹{short_t2}", delta=f"{pct_of(short_entry, short_t2)}% ({round(short_t2 - short_entry, 2)} pts)")
    else:
        st.warning(f"⚠️ Short setup for {bearish_stock} unavailable (invalid price/risk).")

# ============================================================
# MODULE 2: STOCK FUTURES ENGINE
# ============================================================
else:
    long_entry, long_sl, long_risk, long_t1, long_t2 = build_long_setup(bullish_ltp, bull_atr)
    price_tag_bull = bull_price_tag

    st.success(f"### 📈 STOCK FUTURES LONG: {bullish_stock} — {price_tag_bull}{atr_note_bull}")
    if long_entry > 0 and long_risk > 0:
        max_loss_long = round(long_risk * bull_lot_size, 2)
        est_margin_long = round(long_entry * bull_lot_size * 0.20, 2)

        if max_loss_long > max_risk:
            st.error(
                f"🚫 **TRADE BLOCKED (RISK OUT OF BOUNDS):** Risk is **₹{max_loss_long:,}**, "
                f"exceeding your allocation of ₹{max_risk}."
            )
        elif est_margin_long > capital:
            st.error(
                f"🚫 **TRADE BLOCKED (INSUFFICIENT CAPITAL):** Margin needed ~**₹{est_margin_long:,}**, "
                f"available is **₹{capital:,}**."
            )
        else:
            q1, q2 = st.columns(2)
            q1.info(f"### 🎯 TRADE QUANTITY: **1 Lot ({bull_lot_size} units)**")
            q2.error(f"### 🛡️ MAX LOSS RISK: **₹{max_loss_long:,}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{price_tag_bull.split(' (')[0]} Entry Trigger", f"₹{long_entry}")
            c2.metric(
                "ATR Stop Loss", f"₹{long_sl}",
                delta=f"{pct_of(long_entry, long_sl)}% (-{long_risk} pts)", delta_color="inverse",
            )
            c3.metric("Target 1 (1:2)", f"₹{long_t1}", delta=f"+{pct_of(long_entry, long_t1)}% (+{round(long_t1 - long_entry, 2)} pts)")
            c4.metric("Target 2 (1:3)", f"₹{long_t2}", delta=f"+{pct_of(long_entry, long_t2)}% (+{round(long_t2 - long_entry, 2)} pts)")
    else:
        st.warning(f"⚠️ Long futures setup for {bullish_stock} unavailable.")

    st.markdown("---")

    short_entry, short_sl, short_risk, short_t1, short_t2 = build_short_setup(bearish_ltp, bear_atr)
    price_tag_bear = bear_price_tag

    st.error(f"### 📉 STOCK FUTURES SHORT: {bearish_stock} — {price_tag_bear}{atr_note_bear}")
    if short_entry > 0 and short_risk > 0:
        max_loss_short = round(short_risk * bear_lot_size, 2)
        est_margin_short = round(short_entry * bear_lot_size * 0.20, 2)

        if max_loss_short > max_risk:
            st.error(
                f"🚫 **TRADE BLOCKED (RISK OUT OF BOUNDS):** Risk is **₹{max_loss_short:,}**, "
                f"exceeding your allocation of ₹{max_risk}."
            )
        elif est_margin_short > capital:
            st.error(
                f"🚫 **TRADE BLOCKED (INSUFFICIENT CAPITAL):** Margin needed ~**₹{est_margin_short:,}**, "
                f"available is **₹{capital:,}**."
            )
        else:
            q1, q2 = st.columns(2)
            q1.info(f"### 🎯 TRADE QUANTITY: **1 Lot ({bear_lot_size} units)**")
            q2.error(f"### 🛡️ MAX LOSS RISK: **₹{max_loss_short:,}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{price_tag_bear.split(' (')[0]} Entry Trigger", f"₹{short_entry}")
            c2.metric(
                "ATR Stop Loss", f"₹{short_sl}",
                delta=f"+{pct_of(short_entry, short_sl)}% (+{short_risk} pts)", delta_color="inverse",
            )
            c3.metric("Target 1 (1:2)", f"₹{short_t1}", delta=f"{pct_of(short_entry, short_t1)}% ({round(short_t1 - short_entry, 2)} pts)")
            c4.metric("Target 2 (1:3)", f"₹{short_t2}", delta=f"{pct_of(short_entry, short_t2)}% ({round(short_t2 - short_entry, 2)} pts)")
    else:
        st.warning(f"⚠️ Short futures setup for {bearish_stock} unavailable.")

# --- FOOTER ---
st.markdown("---")
st.warning(
    "⚠️ **Execution Guardrail:** Always deploy these setups using **SL-Limit (MIS/NRML)** orders "
    "directly within your broker terminal. Do not use market orders to avoid execution slippage."
)
