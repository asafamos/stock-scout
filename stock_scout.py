# -*- coding: utf-8 -*-
"""
Stock Scout — 2025 (Pro++)
אין באמור ייעוץ השקעות.
"""
from __future__ import annotations
import os, io, time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from dotenv import load_dotenv, find_dotenv

# ========= ENV =========
load_dotenv(find_dotenv(usecwd=True))
for _extra in ["nev", "stock_scout.nev"]:
    try:
        if os.path.exists(_extra): load_dotenv(_extra)
    except Exception:
        pass

# ========= HTTP + Backoff =========
def http_get_retry(url: str, tries: int = 3, backoff: float = 1.7, timeout: int = 14, headers: dict | None = None):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff**i); continue
            return r
        except requests.RequestException:
            time.sleep(backoff**i)
    return None

_last_alpha_call = 0.0
def alpha_throttle(min_gap_seconds: float = 12.0):
    global _last_alpha_call
    now = time.time()
    gap = now - _last_alpha_call
    if gap < min_gap_seconds: time.sleep(min_gap_seconds - gap)
    _last_alpha_call = time.time()

# ========= Status (cache) =========
@st.cache_data(ttl=300)
def check_openai_verbose():
    key = os.getenv("OPENAI_API_KEY")
    if not key: return False, "Missing API key"
    try:
        from openai import OpenAI
        OpenAI(api_key=key).models.list()
        return True, "OK"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=300)
def check_alpha_vantage_verbose():
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key: return False, "Missing API key"
    try:
        r = http_get_retry(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=MSFT&apikey={key}", tries=2, timeout=12)
        if not r: return False, "Timeout/Network"
        j = r.json()
        if "Global Quote" in j: return True, "OK"
        if "Note" in j: return False, "Rate limit"
        if "Information" in j: return False, "Invalid call"
        return False, "Bad/Unknown response"
    except Exception:
        return False, "Exception"

@st.cache_data(ttl=300)
def check_finnhub_verbose():
    key = os.getenv("FINNHUB_API_KEY")
    if not key: return False, "Missing API key"
    try:
        r = http_get_retry(f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}", tries=2, timeout=10)
        if not r: return False, "Timeout/Network"
        j = r.json()
        return ("c" in j), ("OK" if "c" in j else "Bad response")
    except Exception:
        return False, "Exception"

@st.cache_data(ttl=300)
def check_polygon_verbose():
    key = os.getenv("POLYGON_API_KEY")
    if not key: return False, "Missing API key"
    try:
        r = http_get_retry(f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true&apiKey={key}", tries=2, timeout=10)
        if not r: return False, "Timeout/Network"
        j = r.json()
        ok = bool(j.get("resultsCount", 0) > 0 and "results" in j)
        return ok, ("OK" if ok else "Bad response")
    except Exception:
        return False, "Exception"

@st.cache_data(ttl=300)
def check_tiingo_verbose():
    key = os.getenv("TIINGO_API_KEY")
    if not key: return False, "Missing API key"
    try:
        r = http_get_retry(f"https://api.tiingo.com/tiingo/daily/AAPL/prices?token={key}&resampleFreq=daily", tries=2, timeout=10)
        if not r: return False, "Timeout/Network"
        arr = r.json()
        ok = isinstance(arr, list) and len(arr) > 0 and isinstance(arr[-1], dict) and ("close" in arr[-1])
        return ok, ("OK" if ok else "Bad response")
    except Exception:
        return False, "Exception"

@st.cache_data(ttl=300)
def check_fmp_verbose():
    key = os.getenv("FMP_API_KEY")
    if not key: return False, "Missing API key"
    try:
        r = http_get_retry(f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={key}", tries=3, timeout=16, headers={"User-Agent":"StockScout/1.0"})
        if not r: return False, "Timeout/Network"
        j = r.json()
        if isinstance(j, list) and j and isinstance(j[0], dict) and ("price" in j[0]): return True, "OK"
        if isinstance(j, dict) and "Error Message" in j: return False, j.get("Error Message", "Error")
        return False, "Bad/Unknown response"
    except Exception:
        return False, "Exception"

# ========= TA =========
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if isinstance(series, pd.DataFrame): series = series.squeeze(axis=1)
    s = pd.to_numeric(series, errors="coerce")
    d = s.diff()
    gain  = d.clip(lower=0)
    loss  = (-d.clip(upper=0))
    ag = gain.rolling(period, min_periods=period).mean()
    al = loss.rolling(period, min_periods=period).mean()
    rs = ag / (al + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def macd_line(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    macd = ef - es
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    plus_dm = (h.diff()).clip(lower=0); minus_dm = (-l.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0;    minus_dm[minus_dm <= plus_dm] = 0
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(period, min_periods=period).mean()
    plus_di  = 100 * (plus_dm.rolling(period, min_periods=period).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=period).mean() / (atr14 + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    return dx.rolling(period, min_periods=period).mean()

# ========= OpenAI (optional, לצ'אט) =========
def openai_client():
    try:
        from openai import OpenAI
        k = os.getenv("OPENAI_API_KEY")
        return OpenAI(api_key=k) if k else None
    except Exception:
        return None

# ========= External prices =========
def get_alpha_vantage_price(tkr: str) -> float | None:
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key: return None
    if "av_calls" not in st.session_state: st.session_state.av_calls = 0
    alpha_throttle()
    r = http_get_retry(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={tkr}&apikey={key}", tries=3, timeout=16)
    if not r: st.session_state.av_calls += 1; return None
    try:
        j = r.json(); q = j.get("Global Quote", {})
        if "05. price" in q:
            st.session_state.av_calls += 1
            return float(q["05. price"])
    except Exception:
        st.session_state.av_calls += 1
    return None

def get_finnhub_price(tkr: str) -> float | None:
    key = os.getenv("FINNHUB_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://finnhub.io/api/v1/quote?symbol={tkr}&token={key}", tries=2, timeout=12)
    if not r: return None
    try:
        j = r.json();  return float(j["c"]) if "c" in j else None
    except Exception:
        return None

def get_polygon_price(tkr: str) -> float | None:
    key = os.getenv("POLYGON_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://api.polygon.io/v2/aggs/ticker/{tkr}/prev?adjusted=true&apiKey={key}", tries=2, timeout=10)
    if not r: return None
    try:
        j = r.json()
        if j.get("resultsCount", 0) > 0 and "results" in j: return float(j["results"][0]["c"])
    except Exception:
        return None
    return None

def get_tiingo_price(tkr: str) -> float | None:
    key = os.getenv("TIINGO_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://api.tiingo.com/tiingo/daily/{tkr}/prices?token={key}&resampleFreq=daily", tries=2, timeout=10)
    if not r: return None
    try:
        arr = r.json()
        if isinstance(arr, list) and arr: return float(arr[-1].get("close", np.nan))
    except Exception:
        return None
    return None

def get_fmp_price(tkr: str) -> float | None:
    key = os.getenv("FMP_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://financialmodelingprep.com/api/v3/quote/{tkr}?apikey={key}", tries=3, timeout=16, headers={"User-Agent":"StockScout/1.0"})
    if not r: return None
    try:
        j = r.json()
        if isinstance(j, list) and j: return float(j[0].get("price", np.nan))
    except Exception:
        return None
    return None

# ========= Universe =========
def build_universe(limit: int = 350) -> List[str]:
    ok, _ = check_finnhub_verbose()
    if not ok:
        return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","QCOM","ADBE","CRM","NFLX","INTC","ORCL","PANW","SNPS","CDNS","MU","KLAC"]
    key = os.getenv("FINNHUB_API_KEY")
    symbols: List[str] = []
    for mic in ("XNAS","XNYS"):
        r = http_get_retry(f"https://finnhub.io/api/v1/stock/symbol?exchange=US&mic={mic}&token={key}", tries=2, timeout=15)
        if not r: continue
        try:
            for it in r.json():
                s = it.get("symbol",""); typ = it.get("type","")
                if not s or "." in s: continue
                if typ and "Common Stock" not in typ: continue
                symbols.append(s)
        except Exception:
            pass
    symbols = sorted(pd.unique(pd.Series(symbols)))
    if not symbols: return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META"]
    if len(symbols) > limit:
        bins = {};  [bins.setdefault(t[0], []).append(t) for t in symbols]
        per = max(1, int(limit / max(1, len(bins))))
        sampled = []
        for k, arr in sorted(bins.items()): sampled.extend(arr[:per])
        if len(sampled) < limit: sampled.extend([t for t in symbols if t not in sampled][: (limit - len(sampled))])
        symbols = sampled
    return symbols[:limit]

# ========= History =========
@st.cache_data(show_spinner=True, ttl=60*15)
def fetch_history_bulk(tickers: List[str], period_days: int, ma_long: int) -> Dict[str, pd.DataFrame]:
    if not tickers: return {}
    end = datetime.utcnow();  start = end - timedelta(days=period_days)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker', threads=True)
    data: Dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = raw[t].dropna()
                if len(df) >= max(60, int(ma_long) + 10): data[t] = df
            except Exception: continue
    else:
        df = raw.dropna()
        if len(df) >= max(60, int(ma_long) + 10): data[tickers[0]] = df
    return data

# ========= Fundamentals (light) =========
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    out = {"marketCap": np.nan, "debtToEquity": np.nan, "trailingPE": np.nan, "priceToBook": np.nan, "sector": None}
    try:
        tk = yf.Ticker(ticker)
        try:
            fi = getattr(tk, "fast_info", {}) or {}
            mc = fi.get("market_cap")
            if mc: out["marketCap"] = float(mc)
        except Exception: pass
        try:
            inf = tk.info or {}
        except Exception:
            inf = {}
        for k in ("marketCap","debtToEquity","trailingPE","priceToBook","sector"):
            if k in inf and (pd.isna(out.get(k, np.nan)) or out.get(k) is None): out[k] = inf.get(k)
    except Exception:
        pass

    fmp = os.getenv("FMP_API_KEY")
    if fmp:
        try:
            r = http_get_retry(f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp}", tries=2, timeout=10, headers={"User-Agent":"StockScout/1.0"})
            if r:
                arr = r.json()
                if isinstance(arr, list) and arr:
                    row = arr[0]
                    out["sector"] = out.get("sector") or row.get("sector")
                    if pd.isna(out.get("trailingPE", np.nan)) and row.get("pe") is not None: out["trailingPE"] = float(row["pe"])
                    if pd.isna(out.get("priceToBook", np.nan)) and row.get("priceToBook") is not None: out["priceToBook"] = float(row["priceToBook"])
                    if pd.isna(out.get("debtToEquity", np.nan)) and row.get("debtEquityRatio") is not None: out["debtToEquity"] = float(row["debtEquityRatio"])
                    if pd.isna(out.get("marketCap", np.nan)) and row.get("mktCap") is not None: out["marketCap"] = float(row["mktCap"])
        except Exception:
            pass
    return out

# ========= Helpers =========
def t_start(): return time.perf_counter()
def t_end(t0): return time.perf_counter() - t0

# ========= UI =========
st.set_page_config(page_title="Asaf's Stock Scout — 2025", page_icon="📈", layout="wide")
st.markdown("""
<style>
:root{ --badge-bg:#eef2ff; --badge-bd:#c7d2fe; --badge-fg:#1e293b; --pill-bg:#ecfdf5; --pill-bd:#34d399; --pill-fg:#065f46; }
@media (prefers-color-scheme: dark){
  :root{ --badge-bg:#0b1220; --badge-bd:#334155; --badge-fg:#e2e8f0; --pill-bg:#064e3b; --pill-bd:#10b981; --pill-fg:#d1fae5; }
}
body{direction:rtl}
.block-container{padding-top:1rem;padding-bottom:2rem}
h1,h2,h3{text-align:right}
[data-testid="stMarkdownContainer"], label{ text-align:right }
input, textarea{ direction:rtl; text-align:right }
thead tr th{ text-align:right }
.rtl-table th,.rtl-table td{ text-align:right !important }
.badge{display:inline-block;background:var(--badge-bg);border:1px solid var(--badge-bd);color:var(--badge-fg);padding:2px 10px;border-radius:999px;font-weight:600}
.status-buy{background:var(--pill-bg);border:1px solid var(--pill-bd);color:var(--pill-fg);padding:2px 10px;border-radius:999px;font-weight:600}
.recommend-card{backdrop-filter:saturate(1.1)}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Scout — 2025")

# ===== Status =====
openai_ok, openai_reason    = check_openai_verbose()
alpha_ok,  alpha_reason     = check_alpha_vantage_verbose()
finnhub_ok, finnhub_reason  = check_finnhub_verbose()
polygon_ok, polygon_reason  = check_polygon_verbose()
tiingo_ok, tiingo_reason    = check_tiingo_verbose()
fmp_ok,    fmp_reason       = check_fmp_verbose()

status_df = pd.DataFrame({
    "מקור":["OpenAI","Alpha Vantage","Finnhub","Polygon","Tiingo","FMP"],
    "סטטוס":[
        "🟢 מחובר" if openai_ok else "🔴 לא מחובר",
        "🟢 מחובר" if alpha_ok else "🔴 לא מחובר",
        "🟢 מחובר" if finnhub_ok else "🔴 לא מחובר",
        "🟢 מחובר" if polygon_ok else "🔴 לא מחובר",
        "🟢 מחובר" if tiingo_ok else "🔴 לא מחובר",
        "🟢 מחובר" if fmp_ok else "🔴 לא מחובר",
    ],
    "סיבה":[openai_reason, alpha_reason, finnhub_reason, polygon_reason, tiingo_reason, fmp_reason],
    "מפתח מזוהה":[
        "✅" if os.getenv("OPENAI_API_KEY") else "—",
        "✅" if os.getenv("ALPHA_VANTAGE_API_KEY") else "—",
        "✅" if os.getenv("FINNHUB_API_KEY") else "—",
        "✅" if os.getenv("POLYGON_API_KEY") else "—",
        "✅" if os.getenv("TIINGO_API_KEY") else "—",
        "✅" if os.getenv("FMP_API_KEY") else "—",
    ],
})
st.table(status_df.style.set_properties(**{'text-align':'center','direction':'rtl'}))

# ===== Basic settings =====
st.subheader("⚙️ הגדרות בסיס")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    budget_total = st.number_input("סה״כ השקעה ($)", 0.0, 1_000_000.0, 5_000.0, step=500.0)
with col_b:
    min_position = st.number_input("השקעה מינימלית לפוזיציה ($)", 0.0, 50_000.0, 500.0, step=100.0)
with col_c:
    lookback_days = st.number_input("תקופת בדיקה (ימים)", 120, 1500, 400)
with col_d:
    use_external_prices = st.toggle("אימות חיצוני", value=True)

# ===== Advanced =====
with st.expander("מתקדם"):
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        min_price = st.number_input("מחיר מינ׳ $ (מומלץ ≥5$)", 0.0, 5000.0, 5.0)
        ma_short  = st.number_input("MA קצר", 5, 100, 20)
        pullback_low, pullback_high = st.slider("טווח Pullback מהשיא", 0.70, 0.99, (0.85, 0.97))
    with col_m2:
        min_avg_volume = st.number_input("ממוצע נפח 20 ימים מינ׳", 0, 50_000_000, 500_000, step=50_000)
        ma_long   = st.number_input("MA ארוך", 20, 400, 50)
        overext_threshold = st.number_input("Overextension יחס ל-MA ארוך (סף)", 0.05, 0.80, 0.20, step=0.05)
    with col_m3:
        rsi_bounds = st.slider("טווח RSI מועדף", 0, 100, (40, 75))
        top_validate_k = st.number_input("אימות חיצוני ל-Top K", 5, 50, 12)
        universe_limit = st.number_input("גודל יקום מקסימלי לסריקה", 50, 1500, 350, step=50)
        smart_scan = st.checkbox("סריקה חכמה (Finnhub)", value=True if finnhub_ok else False)

    st.divider(); st.markdown("**בקרות סיכון ופונדמנטלי**")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_dollar_volume = st.number_input("מינ׳ דולר-ווליום (Price × Vol20)", 0, 2_000_000_000, 5_000_000, step=500_000)
        max_atr_price_cap = st.number_input("מקס׳ ATR/Price (קשיח)", 0.01, 0.50, 0.08, step=0.01)
        max_overext_cap   = st.number_input("מקס׳ Overextension מול MA_L (קשיח)", 0.05, 1.00, 0.30, step=0.05)
    with col_f2:
        min_marketcap = st.number_input("MarketCap מינ׳ ($)", 0, 5_000_000_000, 2_000_000_000, step=100_000_000)
        max_debt_to_equity = st.number_input("Debt/Equity מקס׳", 0.0, 10.0, 3.0, step=0.1)
        min_sources_required = st.number_input("מס׳ מקורות מחיר מינ׳ (כאשר אימות פעיל)", 1, 5, 2)
    with col_f3:
        earnings_blackout_days = st.number_input("Earnings blackout (ימים ±)", 0, 30, 7)
        blackout_scope_all = st.checkbox("החלה על כל הרשימה (לא רק Top-K)", value=False)
        sector_filter = st.text_input("סינון מגזרים (ריק=ללא), פסיקים", value="")
        value_filter_enabled = st.toggle("סינון Value (P/E 5–40, P/B < 10)", value=True)
        macd_adx_enabled = st.toggle("שקלול MACD/ADX", value=False)

    st.markdown("**משקולות ניקוד (JSON)**")
    default_weights = {"ma":0.22,"mom":0.30,"rsi":0.12,"near_high_bell":0.10,"vol":0.08,"overext":0.08,"pullback":0.05,"risk_reward":0.03,"macd":0.01,"adx":0.01}
    risk_preset = st.selectbox("Preset סיכון", ["Balanced","Conservative","Aggressive"], index=0)
    if risk_preset == "Conservative":
        default_weights.update({"vol":0.12,"overext":0.12,"near_high_bell":0.08,"mom":0.24})
    elif risk_preset == "Aggressive":
        default_weights.update({"vol":0.06,"overext":0.05,"near_high_bell":0.12,"mom":0.36})
    score_weights_raw = st.text_input("משקולות", value=pd.Series(default_weights).to_json(force_ascii=False))

try:
    SCORE_W = pd.Series(pd.read_json(io.StringIO(score_weights_raw), typ="series"))
except Exception:
    SCORE_W = pd.Series(default_weights)

# ========= Earnings helper =========
def get_next_earnings_date(ticker: str):
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            vals = cal.loc["Earnings Date"].values
            if len(vals) > 0:
                dt = pd.to_datetime(str(vals[0]))
                if pd.notna(dt): return dt.to_pydatetime()
    except Exception:
        pass
    return None

# ========= Pipeline =========
if "av_calls" not in st.session_state: st.session_state.av_calls = 0
phase_times: Dict[str, float] = {}

# 1) Universe
t0 = t_start()
universe = build_universe(limit=int(universe_limit)) if smart_scan else build_universe(limit=200)
if len(universe) > universe_limit:
    bins = {}; [bins.setdefault(t[0], []).append(t) for t in universe]
    per = max(1, int(universe_limit / max(1, len(bins))))
    sampled = []
    for k, arr in sorted(bins.items()): sampled.extend(arr[:per])
    if len(sampled) < universe_limit: sampled.extend([t for t in universe if t not in sampled][: (universe_limit - len(sampled))])
    universe = sampled
phase_times["בונה יקום"] = t_end(t0)

# 2) History
t0 = t_start()
data_map = fetch_history_bulk(universe, lookback_days, ma_long=int(ma_long))
phase_times["מוריד נתונים"] = t_end(t0)

# 3) Score + Filters
t0 = t_start()
rows = []
lo_rsi, hi_rsi = rsi_bounds
allowed_sectors = [s.strip() for s in sector_filter.split(",") if s.strip()]
for t, df in data_map.items():
    if df is None or df.empty: continue
    df = df.copy()
    df["MA_S"]  = df["Close"].rolling(int(ma_short)).mean()
    df["MA_L"]  = df["Close"].rolling(int(ma_long)).mean()
    df["RSI"]   = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["Vol20"] = df["Volume"].rolling(20).mean()

    if macd_adx_enabled:
        macd_v, macd_sig, _ = macd_line(df["Close"])
        df["MACD"], df["MACD_SIG"] = macd_v, macd_sig
        df["ADX14"] = adx(df, 14)

    price = float(df["Close"].iloc[-1])
    if not np.isfinite(price) or price < float(min_price): continue

    ma_s = float(df["MA_S"].iloc[-1]); ma_l = float(df["MA_L"].iloc[-1])
    ma_ok = (float(price > ma_s) + float(ma_s > ma_l)) / 2.0 if (np.isfinite(ma_s) and np.isfinite(ma_l)) else 0.0

    rsi_v = float(df["RSI"].iloc[-1])
    if np.isfinite(rsi_v):
        if rsi_v < lo_rsi:   rsi_score = max(0.0, 1 - (lo_rsi - rsi_v) / 20)
        elif rsi_v > hi_rsi: rsi_score = max(0.0, 1 - (rsi_v - hi_rsi) / 20)
        else:                rsi_score = 1.0
    else:
        rsi_score = 0.0

    vol20 = float(df["Vol20"].iloc[-1]); vol_today = float(df["Volume"].iloc[-1])
    if np.isfinite(vol20) and vol20 < float(min_avg_volume): continue
    vol_ok = (min(2.0, vol_today / vol20) / 2.0) if (np.isfinite(vol20) and vol20 > 0 and np.isfinite(vol_today)) else 0.0

    ret_1m = float(df["Close"].pct_change(21).iloc[-1])
    ret_3m = float(df["Close"].pct_change(63).iloc[-1])
    ret_6m = float(df["Close"].pct_change(126).iloc[-1])
    momentum = float(np.nanmean([ret_1m, ret_3m, ret_6m]))

    window_52w = min(len(df), 252); hi_52w = float(df["Close"].tail(window_52w).max()) if window_52w > 0 else np.nan
    if np.isfinite(hi_52w) and hi_52w > 0:
        near_raw = 1.0 - min(1.0, max(0.0, (hi_52w - price) / hi_52w))
        if near_raw >= 0.95: near_score = 0.45
        elif 0.75 <= near_raw <= 0.90: near_score = 1.00
        elif 0.90 < near_raw < 0.95:  near_score = 0.75
        else: near_score = near_raw
    else:
        near_raw, near_score = np.nan, 0.0

    if np.isfinite(ma_l) and ma_l > 0:
        overext_ratio = max(0.0, (price - ma_l) / ma_l)
        overext_score = 1.0 - min(1.0, overext_ratio / max(1e-6, overext_threshold))
    else:
        overext_ratio, overext_score = np.nan, 0.0

    if np.isfinite(hi_52w) and hi_52w > 0:
        ratio_to_high = price / hi_52w
        if pullback_low <= ratio_to_high <= pullback_high: pullback_score = 1.0
        else:
            dist = min(abs(ratio_to_high - pullback_low), abs(ratio_to_high - pullback_high))
            pullback_score = max(0.0, 1.0 - dist * 10)
    else:
        pullback_score = 0.0

    atr14 = float(df["ATR14"].iloc[-1])
    if np.isfinite(atr14) and price > 0:
        vol_rel = atr14 / price; vol_score = 1.0 - min(1.0, vol_rel / 0.05)
    else:
        vol_rel, vol_score = np.nan, 0.0

    dollar_vol = (price * vol20) if (np.isfinite(price) and np.isfinite(vol20)) else 0.0
    if dollar_vol < float(min_dollar_volume): continue
    if np.isfinite(vol_rel) and vol_rel > float(max_atr_price_cap): continue
    if np.isfinite(overext_ratio) and overext_ratio > float(max_overext_cap): continue

    if np.isfinite(hi_52w) and np.isfinite(atr14) and atr14 > 0:
        reward_risk = max(0.0, (hi_52w - price) / atr14); rr_score = min(1.0, reward_risk / 4.0)
    else:
        reward_risk, rr_score = np.nan, 0.0

    macd_score = adx_score = 0.0
    if macd_adx_enabled and "MACD" in df.columns:
        macd_val = float(df["MACD"].iloc[-1]); macd_sig = float(df["MACD_SIG"].iloc[-1])
        macd_score = 1.0 if macd_val > macd_sig else 0.0
    if macd_adx_enabled and "ADX14" in df.columns:
        adx_val = float(df["ADX14"].iloc[-1])
        adx_score = np.clip((adx_val - 15) / 20.0, 0.0, 1.0) if np.isfinite(adx_val) else 0.0

    score = (
        float(SCORE_W.get("ma", 0.22)) * ma_ok +
        float(SCORE_W.get("mom", 0.30)) * np.clip((momentum + 0.5), 0, 1) +
        float(SCORE_W.get("rsi", 0.12)) * rsi_score +
        float(SCORE_W.get("near_high_bell", 0.10)) * near_score +
        float(SCORE_W.get("vol", 0.08)) * (vol_ok if np.isfinite(vol_ok) else 0.0) +
        float(SCORE_W.get("overext", 0.08)) * overext_score +
        float(SCORE_W.get("pullback", 0.05)) * pullback_score +
        float(SCORE_W.get("risk_reward", 0.03)) * rr_score +
        float(SCORE_W.get("macd", 0.01)) * macd_score +
        float(SCORE_W.get("adx", 0.01)) * adx_score
    )

    # fundamentals
    f = fetch_fundamentals(t)
    allowed = (not allowed_sectors) or (str(f.get("sector") or "") in allowed_sectors)
    mc_ok = (isinstance(f.get("marketCap"), (int,float)) and f["marketCap"] >= float(min_marketcap))
    de_val = f.get("debtToEquity", np.nan)
    debt_ok = (not isinstance(de_val, (int,float))) or (np.isfinite(de_val) and de_val <= float(max_debt_to_equity))

    pe = f.get("trailingPE", np.nan); pb = f.get("priceToBook", np.nan)
    value_ok = True
    if value_filter_enabled:
        bad_val = (isinstance(pe, (int,float)) and (pe < 5 or pe > 40)) or (isinstance(pb, (int,float)) and pb >= 10)
        value_ok = not bad_val
        if bad_val: score = max(0.0, score - 0.02)

    rows.append({
        "Ticker": t, "Price_Yahoo": price,
        "Price_Alpha": np.nan, "Price_Finnhub": np.nan,
        "Price_Mean": np.nan, "Price_STD": np.nan,
        "Sources": 1, "Source_List": "🟡Yahoo",
        "Score": round(100 * float(score), 1),
        "RSI": round(rsi_v, 1) if np.isfinite(rsi_v) else np.nan,
        "Mom123": f"{ret_1m:.1%}/{ret_3m:.1%}/{ret_6m:.1%}",
        "MA_OK": ma_ok >= 0.5,
        "Near52w": round(near_raw * 100, 1) if np.isfinite(near_raw) else np.nan,
        "Volx20d": round(vol_today / vol20, 2) if (np.isfinite(vol_today) and np.isfinite(vol20) and vol20>0) else np.nan,
        "OverextRatio": round(overext_ratio, 3) if np.isfinite(overext_ratio) else np.nan,
        "ATR_Price": round(vol_rel, 4) if np.isfinite(vol_rel) else np.nan,
        "RewardRisk": round(reward_risk, 2) if np.isfinite(reward_risk) else np.nan,
        "MarketCap": f.get("marketCap", np.nan), "DebtToEquity": de_val,
        "PE": pe, "PB": pb, "Sector": f.get("sector", None),
        "Value_OK": value_ok, "MC_OK": mc_ok, "Debt_OK": debt_ok, "Sector_OK": allowed
    })

results = pd.DataFrame(rows)
phase_times["מחשב ניקוד"] = t_end(t0)
if results.empty:
    st.warning("אין תוצאות אחרי הסינון. נסה להקל פרמטרים במתקדם."); st.stop()

# סידור
results = results.sort_values(["Score","Price_Yahoo"], ascending=[False, True]).reset_index(drop=True)

# Earnings blackout
def apply_blackout(df: pd.DataFrame, scope_all: bool, days: int, topk: int) -> pd.DataFrame:
    if days <= 0: return df
    target_idx = df.index if scope_all else df.head(int(topk)).index
    now_utc = pd.Timestamp.utcnow().to_pydatetime()
    keep = np.ones(len(df), dtype=bool)
    for idx in target_idx:
        tkr = df.at[idx, "Ticker"]
        dt = get_next_earnings_date(tkr)
        if dt is None: continue
        if abs((dt - now_utc).days) <= int(days):
            keep[idx] = False; df.at[idx, "EarningsNote"] = f"Excluded: earnings within ±{days}d"
    return df[keep].reset_index(drop=True)

results = apply_blackout(results, blackout_scope_all, earnings_blackout_days, top_validate_k)

# External verification (Top-K)
t0 = t_start()
if use_external_prices and (alpha_ok or finnhub_ok or os.getenv("POLYGON_API_KEY") or os.getenv("TIINGO_API_KEY") or os.getenv("FMP_API_KEY")):
    subset_idx = list(results.head(int(top_validate_k)).index)
    for idx in subset_idx:
        r = results.loc[idx]; tkr = r["Ticker"]; py = r["Price_Yahoo"]
        pa = get_alpha_vantage_price(tkr) if alpha_ok else None
        pf = get_finnhub_price(tkr) if finnhub_ok else None
        ppg = get_polygon_price(tkr) if os.getenv("POLYGON_API_KEY") else None
        pti = get_tiingo_price(tkr)  if os.getenv("TIINGO_API_KEY")  else None
        pfmp = get_fmp_price(tkr)    if os.getenv("FMP_API_KEY")     else None
        prices, srcs = [], []
        if py is not None and not np.isnan(py): prices.append(py);  srcs.append("🟡Yahoo")
        if pa is not None: prices.append(pa);   srcs.append("🟣Alpha")
        if pf is not None: prices.append(pf);   srcs.append("🔵Finnhub")
        if ppg is not None:prices.append(ppg);  srcs.append("🟢Polygon")
        if pti is not None:prices.append(pti);  srcs.append("🟠Tiingo")
        if pfmp is not None:prices.append(pfmp);srcs.append("🟤FMP")
        results.loc[idx, ["Price_Alpha","Price_Finnhub","Price_Mean","Price_STD","Sources","Source_List"]] = [
            pa, pf, (float(np.mean(prices)) if prices else np.nan),
            (float(np.std(prices)) if len(prices) > 1 else np.nan),
            len(prices), " · ".join(srcs) if srcs else "—"
        ]
else:
    if use_external_prices: st.info("אימות חיצוני הופעל אך אין מפתחות תקינים. משתמש רק במחיר Yahoo.")
phase_times["מאמת מחירים"] = t_end(t0)

# Buy Candidate
def is_buy_candidate(row) -> bool:
    tech_ok = (
        bool(row.get("MA_OK")) and
        (isinstance(row.get("RSI"), (int,float)) and lo_rsi <= row["RSI"] <= 78) and
        (isinstance(row.get("ATR_Price"), (int,float)) and row["ATR_Price"] <= max_atr_price_cap) and
        (isinstance(row.get("OverextRatio"), (int,float)) and row["OverextRatio"] <= max_overext_cap)
    )
    fund_ok = bool(row.get("MC_OK")) and bool(row.get("Debt_OK")) and (not value_filter_enabled or bool(row.get("Value_OK"))) and bool(row.get("Sector_OK"))
    verify_ok = (not use_external_prices) or (int(row.get("Sources", 1)) >= int(min_sources_required))
    return tech_ok and fund_ok and verify_ok

results["BuyCandidate"] = results.apply(is_buy_candidate, axis=1)

# Allocation
def allocate_budget(df: pd.DataFrame, total: float, min_pos: float) -> pd.DataFrame:
    df = df.copy(); df["סכום קנייה ($)"] = 0.0
    if total <= 0 or df.empty: return df
    df = df.sort_values(["BuyCandidate","Score"], ascending=[False, False]).reset_index(drop=True)
    remaining = float(total); n = len(df)
    cand_mask = df["BuyCandidate"].values.astype(bool)
    order = np.r_[np.where(cand_mask)[0], np.where(~cand_mask)[0]]
    df = df.loc[order].reset_index(drop=True)
    can_min = int(min(len(df), remaining // max(min_pos, 0.0))) if min_pos > 0 else 0
    if min_pos > 0 and can_min > 0:
        df.loc[:can_min-1, "סכום קנייה ($)"] = float(min_pos); remaining -= can_min * float(min_pos)
    if remaining > 0:
        w = df["Score"].clip(lower=0).to_numpy()
        extra = (remaining / len(df)) * np.ones(len(df)) if w.sum() == 0 else remaining * (w / w.sum())
        df["סכום קנייה ($)"] = df["סכום קנייה ($)"].to_numpy() + extra
    if min_pos > 0:
        raw = df["סכום קנייה ($)"].to_numpy()
        raw = np.where(raw < min_pos * 0.9999, 0.0, raw)
        s = raw.sum()
        if s > 0 and abs(s - total) / max(total, 1) > 1e-6: raw = raw * (total / s)
        df["סכום קנייה ($)"] = raw
    df["סכום קנייה ($)"] = df["סכום קנייה ($)"].round(2)
    return df

TOPN = min(15, len(results))
results = allocate_budget(results.head(TOPN).reset_index(drop=True), budget_total, min_position)

# Display helpers
def source_badges(row):
    if isinstance(row.get("Source_List"), str) and row.get("Source_List"): return row["Source_List"]
    badges = []
    if not np.isnan(row.get("Price_Yahoo", np.nan)):  badges.append("🟡Yahoo")
    if not pd.isna(row.get("Price_Finnhub", np.nan)): badges.append("🔵Finnhub")
    if not pd.isna(row.get("Price_Alpha", np.nan)):   badges.append("🟣Alpha")
    return " · ".join(badges) if badges else "—"

results["מקורות מחיר"] = results.apply(source_badges, axis=1)
results["מחיר ממוצע"]  = results["Price_Mean"].round(2)
results["סטיית תקן"]    = results["Price_STD"].round(4)

# Quantities
results["Unit_Price"] = np.where(results["Price_Mean"].notna(), results["Price_Mean"], results["Price_Yahoo"])
results["Unit_Price"] = pd.to_numeric(results["Unit_Price"], errors="coerce")
results["מניות לקנייה"] = np.floor(np.where(results["Unit_Price"] > 0, results["סכום קנייה ($)"] / results["Unit_Price"], 0)).astype(int)
results["עודף ($)"] = np.round(results["סכום קנייה ($)"] - results["מניות לקנייה"] * results["Unit_Price"], 2)

# Timings
st.subheader("⏱️ זמני ביצוע")
times_df = pd.DataFrame([{"שלב": k, "משך (שניות)": round(v, 2)} for k, v in phase_times.items()])
st.table(times_df.style.set_properties(**{'text-align':'center','direction':'rtl'}))
if alpha_ok: st.caption(f"Alpha Vantage — קריאות בסשן זה: {int(st.session_state.av_calls)}")

# Cards
st.subheader("🤖 המלצה עכשיו")
st.caption("מועמדות לקנייה (Buy Candidate) — לא ייעוץ השקעות.")
rec_df = results[results["סכום קנייה ($)"] > 0].copy()

st.markdown("""
<style>
.recommend-card{direction: rtl; text-align: right; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 16px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,.04);}
.recommend-grid{display: grid; grid-template-columns: repeat(5,minmax(0,1fr)); gap: 8px; margin-top: 6px;}
.small{color:#444;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

if rec_df.empty:
    st.info("אין כרגע מניות עם סכום קנייה חיובי. נסה להקל ספים ב'מתקדם'.")
else:
    for _, r in rec_df.head(min(5, len(rec_df))).iterrows():
        mean = r.get("מחיר ממוצע", np.nan); std = r.get("סטיית תקן", np.nan)
        show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
        show_std  = std if not np.isnan(std) else "—"
        sources   = r.get("מקורות מחיר", "—")
        buy_amt   = float(r.get("סכום קנייה ($)", 0.0))
        horizon   = "קצר/בינוני" if r.get("RSI", 50) >= 55 else "בינוני/ארוך"
        rsi_v     = r.get("RSI", np.nan); near52 = r.get("Near52w", np.nan); score = r.get("Score", 0)
        unit_p    = r.get("Unit_Price", np.nan); shares = int(r.get("מניות לקנייה", 0)); leftover = r.get("עודף ($)", 0.0)
        rr        = r.get("RewardRisk", np.nan); atrp = r.get("ATR_Price", np.nan); overx = r.get("OverextRatio", np.nan)

        st.markdown(f"""
        <div class="recommend-card">
          <h3><span class="badge">{r['Ticker']}</span> <span class="status-buy">{'✅ Buy Candidate' if r.get('BuyCandidate') else '⛔ לא קנדידט'}</span></h3>
          <div class="recommend-grid small">
            <div><b>מחיר ממוצע:</b> {np.round(show_mean,2) if not np.isnan(show_mean) else "—"}</div>
            <div><b>סטיית תקן:</b> {show_std}</div>
            <div><b>RSI:</b> {rsi_v if not np.isnan(rsi_v) else '—'}</div>
            <div><b>קרבה לשיא 52ש׳:</b> {near52 if not np.isnan(near52) else '—'}%</div>
            <div><b>ניקוד:</b> {score}</div>
            <div><b>מקורות:</b> {sources}</div>
            <div><b>סכום קנייה מומלץ:</b> ${buy_amt:,.0f}</div>
            <div><b>טווח החזקה:</b> {horizon}</div>
            <div><b>מחיר יחידה:</b> {np.round(unit_p,2) if not np.isnan(unit_p) else '—'}</div>
            <div><b>מניות לקנייה:</b> {shares}</div>
            <div><b>עודף:</b> ${leftover:,.2f}</div>
            <div><b>ATR/Price:</b> {f"{atrp:.3f}" if isinstance(atrp,(int,float)) and np.isfinite(atrp) else "—"}</div>
            <div><b>Overext מעל MA_L:</b> {f"{overx:.3f}" if isinstance(overx,(int,float)) and np.isfinite(overx) else "—"}</div>
            <div><b>Reward/Risk ≈</b> {f"{rr:.2f}R" if isinstance(rr,(int,float)) and np.isfinite(rr) else "—"}</div>
            <div><b>MarketCap:</b> {f"${r.get('MarketCap')/1e9:.1f}B" if isinstance(r.get('MarketCap'),(int,float)) and np.isfinite(r.get('MarketCap')) else "—"}</div>
            <div><b>D/E:</b> {r.get('DebtToEquity') if isinstance(r.get('DebtToEquity'),(int,float)) else "—"}</div>
            <div><b>P/E:</b> {r.get('PE') if isinstance(r.get('PE'),(int,float)) else "—"}</div>
            <div><b>P/B:</b> {r.get('PB') if isinstance(r.get('PB'),(int,float)) else "—"}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# Table
st.subheader("🎯 תוצאות מסוננות ומדורגות")
view_df = rec_df if not rec_df.empty else results
hebrew_cols = {
    "Ticker":"טיקר","Price_Yahoo":"מחיר (Yahoo)","מחיר ממוצע":"מחיר ממוצע","Unit_Price":"מחיר יחידה (חישוב)","סטיית תקן":"סטיית תקן",
    "מקורות מחיר":"מקורות מחיר","Score":"ניקוד","RSI":"RSI","Near52w":"קרבה לשיא 52ש׳ (%)","Volx20d":"נפח/ממוצע 20 ימים",
    "OverextRatio":"Overextension מול MA_L","ATR_Price":"ATR/Price","RewardRisk":"Reward/Risk (≈R)",
    "MarketCap":"שווי שוק ($)","DebtToEquity":"D/E","PE":"P/E","PB":"P/B","Sector":"מגזר",
    "Value_OK":"Value OK","MC_OK":"MarketCap OK","Debt_OK":"Debt OK","Sector_OK":"Sector OK","BuyCandidate":"Buy Candidate",
    "סכום קנייה ($)":"סכום קנייה ($)","מניות לקנייה":"מניות לקנייה","עודף ($)":"עודף ($)"
}
order = ["טיקר","מחיר (Yahoo)","מחיר ממוצע","מחיר יחידה (חישוב)","סטיית תקן","מקורות מחיר","ניקוד","RSI","קרבה לשיא 52ש׳ (%)","נפח/ממוצע 20 ימים",
         "Overextension מול MA_L","ATR/Price","Reward/Risk (≈R)","שווי שוק ($)","D/E","P/E","P/B","מגזר",
         "Value OK","MarketCap OK","Debt OK","Sector OK","Buy Candidate","סכום קנייה ($)","מניות לקנייה","עודף ($)"]

csv_df = view_df.rename(columns=hebrew_cols)
st.download_button("⬇️ הורדת תוצאות ל-CSV", data=csv_df[order].to_csv(index=False).encode("utf-8-sig"),
                   file_name="stock_scout_results.csv", mime="text/csv")
st.dataframe(csv_df[order], use_container_width=True, hide_index=True)

# Charts
st.subheader("🔍 גרף לטיקר נבחר")
choice = st.selectbox("בחר טיקר לצפייה", ["(בחר)"] + view_df["Ticker"].tolist())
if choice and choice != "(בחר)" and choice in data_map:
    dfv = data_map[choice].copy()
    dfv["MA_S"] = dfv["Close"].rolling(int(ma_short)).mean()
    dfv["MA_L"] = dfv["Close"].rolling(int(ma_long)).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfv.index, open=dfv['Open'], high=dfv['High'], low=dfv['Low'], close=dfv['Close'], name='מחיר'))
    fig.add_trace(go.Scatter(x=dfv.index, y=dfv['MA_S'], mode='lines', name=f'MA{int(ma_short)}'))
    fig.add_trace(go.Scatter(x=dfv.index, y=dfv['MA_L'], mode='lines', name=f'MA{int(ma_long)}'))
    fig.update_layout(height=480, xaxis_rangeslider_visible=False, legend_orientation='h')
    st.plotly_chart(fig, use_container_width=True)

    dfv["RSI"] = rsi(dfv["Close"], 14)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dfv.index, y=dfv['RSI'], mode='lines', name='RSI14'))
    fig2.add_hrect(y0=rsi_bounds[0], y1=rsi_bounds[1], fillcolor='LightGreen', opacity=0.2, line_width=0)
    fig2.update_layout(height=220, legend_orientation='h')
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("ℹ️ הערות ומתודולוגיה"):
    st.markdown("""
- היסטורי: **Yahoo Finance** (`yfinance`).
- אימות מחירים (אם פעיל): **Alpha Vantage / Finnhub / Polygon / Tiingo / FMP**.
- ניקוד: MA, מומנטום 1/3/6 חו׳, RSI, Near-High Bell, Overextension, Pullback, ATR/Price, Reward/Risk (אינפורמטיבי), אופציונלי MACD/ADX.
- בקרות: מינ׳ Dollar-Volume, תקרת ATR/Price, תקרת Overextension, MarketCap מינ׳, D/E מקס׳, Value אופציונלי, Blackout סביב דו"חות.
- **Buy Candidate** דורש: טכני ∧ פונדמנטלי ∧ אימות מחיר (≥ N מקורות כשאימות פעיל).
""")
