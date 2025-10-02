# -*- coding: utf-8 -*-
"""
Asaf Stock Scout — סורק מניות 2025 (מצומצם + מדדי זמן + מונה Alpha Vantage + כרטיסי קנייה + צ'אט)
הרצה:
    python3 -m pip install --upgrade pip
    python3 -m pip install streamlit yfinance pandas numpy plotly openai python-dotenv requests
    python3 -m streamlit run stock_scout.py --server.runOnSave true

.env / .nev (אופציונלי):
    OPENAI_API_KEY=...
    ALPHA_VANTAGE_API_KEY=...
    FINNHUB_API_KEY=...
    POLYGON_API_KEY=...      # אופציונלי
    TIINGO_API_KEY=...       # אופציונלי
    FMP_API_KEY=...          # אופציונלי

הערה: אין באמור ייעוץ השקעות.
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

# ========= טעינת ENV =========
load_dotenv(find_dotenv(usecwd=True))
for _extra in ["nev", "stock_scout.nev"]:
    try:
        if os.path.exists(_extra):
            load_dotenv(_extra)
    except Exception:
        pass

# ========= כלי רשת ו-Backoff =========
def http_get_retry(url: str, tries: int = 3, backoff: float = 1.7, timeout: int = 14, headers: dict | None = None):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {})
            if r.headers.get("Content-Type", "").startswith("application/json"):
                try:
                    j = r.json()
                    if isinstance(j, dict) and any(k in j for k in ("Note", "Information", "Error Message", "message", "status")):
                        time.sleep(backoff**i); continue
                except Exception:
                    pass
            return r
        except requests.RequestException:
            time.sleep(backoff**i)
    return None

_last_alpha_call = 0.0
def alpha_throttle(min_gap_seconds: float = 12.0):
    global _last_alpha_call
    now = time.time()
    gap = now - _last_alpha_call
    if gap < min_gap_seconds:
        time.sleep(min_gap_seconds - gap)
    _last_alpha_call = time.time()

# ========= סטטוסים (עם Cache) =========
@st.cache_data(ttl=300)
def check_openai_verbose():
    key = os.getenv("OPENAI_API_KEY")
    if not key: return False, "Missing API key"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        client.models.list()
        return True, "OK"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=300)
def check_alpha_vantage_verbose():
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key: return False, "Missing API key"
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=MSFT&apikey={key}"
        r = http_get_retry(url, tries=2, timeout=12)
        if not r: return False, "Timeout/Network"
        j = r.json()
        if "Global Quote" in j: return True, "OK"
        if "Note" in j:        return False, "Rate limit"
        if "Information" in j: return False, "Invalid call"
        return False, "Bad/Unknown response"
    except Exception:
        return False, "Exception"

@st.cache_data(ttl=300)
def check_finnhub_verbose():
    key = os.getenv("FINNHUB_API_KEY")
    if not key: return False, "Missing API key"
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}"
        r = http_get_retry(url, tries=2, timeout=10)
        if not r: return False, "Timeout/Network"
        j = r.json()
        return ("c" in j), ("OK" if "c" in j else "Bad response")
    except Exception:
        return False, "Exception"

# ---- מקורות חדשים: Polygon / Tiingo / FMP ----
@st.cache_data(ttl=300)
def check_polygon_verbose():
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        return False, "Missing API key"
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true&apiKey={key}"
        r = http_get_retry(url, tries=2, timeout=10)
        if not r: return False, "Timeout/Network"
        j = r.json()
        ok = bool(j.get("resultsCount", 0) > 0 and "results" in j)
        return ok, ("OK" if ok else "Bad response")
    except Exception:
        return False, "Exception"

@st.cache_data(ttl=300)
def check_tiingo_verbose():
    key = os.getenv("TIINGO_API_KEY")
    if not key:
        return False, "Missing API key"
    try:
        url = f"https://api.tiingo.com/tiingo/daily/AAPL/prices?token={key}&resampleFreq=daily"
        r = http_get_retry(url, tries=2, timeout=10)
        if not r: return False, "Timeout/Network"
        arr = r.json()
        ok = isinstance(arr, list) and len(arr) > 0 and isinstance(arr[-1], dict) and ("close" in arr[-1])
        return ok, ("OK" if ok else "Bad response")
    except Exception:
        return False, "Exception"

@st.cache_data(ttl=300)
def check_fmp_verbose():
    key = os.getenv("FMP_API_KEY")
    if not key:
        return False, "Missing API key"
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote-short/AAPL?apikey={key}"
        r = http_get_retry(url, tries=2, timeout=10)
        if not r: return False, "Timeout/Network"
        arr = r.json()
        ok = isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict) and ("price" in arr[0])
        return ok, ("OK" if ok else "Bad response")
    except Exception:
        return False, "Exception"

# ========= פונקציות טכניות =========
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if isinstance(series, pd.DataFrame): series = series.squeeze(axis=1)
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

# ========= OpenAI =========
def openai_client():
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key: return None
        return OpenAI(api_key=key)
    except Exception:
        return None

def ai_recommend(df: pd.DataFrame) -> Optional[str]:
    client = openai_client()
    if not client or df.empty: return None
    top = df.head(12).to_dict(orient="records")
    prompt = "\n".join([
        "נתח את המניות הבאות והחזר המלצה קצרה וברורה:",
        "- 2–4 מניות לטווח קצר (1–3 חודשים)",
        "- 2–4 מניות לטווח ארוך (6–12 חודשים)",
        "כלול התייחסות לרמת הסיכון אם עולה מהנתונים. נקודות תמציתיות בלבד.",
        f"Data: {top}",
    ])
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
        )
        return rsp.choices[0].message.content.strip()
    except Exception:
        return None

# ========= מקורות מחיר =========
def get_alpha_vantage_price(ticker: str) -> float | None:
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key: return None
    if "av_calls" not in st.session_state: st.session_state.av_calls = 0
    alpha_throttle()
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={key}"
    r = http_get_retry(url, tries=3, timeout=16)
    if not r:
        st.session_state.av_calls += 1
        return None
    try:
        data = r.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            st.session_state.av_calls += 1
            return float(data["Global Quote"]["05. price"])
    except Exception:
        st.session_state.av_calls += 1
        return None
    return None

def get_finnhub_price(ticker: str) -> float | None:
    key = os.getenv("FINNHUB_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={key}", tries=2, timeout=12)
    if not r: return None
    try:
        data = r.json()
        return float(data["c"]) if "c" in data else None
    except Exception:
        return None

def get_polygon_price(ticker: str) -> float | None:
    key = os.getenv("POLYGON_API_KEY")
    if not key: return None
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={key}"
    r = http_get_retry(url, tries=2, timeout=10)
    if not r: return None
    try:
        j = r.json()
        if j.get("resultsCount", 0) > 0 and "results" in j:
            return float(j["results"][0]["c"])
    except Exception:
        return None
    return None

def get_tiingo_price(ticker: str) -> float | None:
    key = os.getenv("TIINGO_API_KEY")
    if not key: return None
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?token={key}&resampleFreq=daily"
    r = http_get_retry(url, tries=2, timeout=10)
    if not r: return None
    try:
        arr = r.json()
        if isinstance(arr, list) and arr:
            return float(arr[-1].get("close", np.nan))
    except Exception:
        return None
    return None

def get_fmp_price(ticker: str) -> float | None:
    key = os.getenv("FMP_API_KEY")
    if not key: return None
    url = f"https://financialmodelingprep.com/api/v3/quote-short/{ticker}?apikey={key}"
    r = http_get_retry(url, tries=2, timeout=10)
    if not r: return None
    try:
        arr = r.json()
        if isinstance(arr, list) and arr:
            return float(arr[0].get("price", np.nan))
    except Exception:
        return None
    return None

# ========= יקום סריקה =========
def build_universe(limit: int = 350) -> List[str]:
    ok, _ = check_finnhub_verbose()
    if not ok:
        # fallback קטן כדי שתהיה ריצה גם בלי Finnhub
        return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","QCOM","ADBE","CRM","NFLX","INTC","ORCL","PANW","SNPS","CDNS","MU","KLAC"]
    key = os.getenv("FINNHUB_API_KEY")
    symbols: List[str] = []
    for mic in ("XNAS","XNYS"):
        r = http_get_retry(f"https://finnhub.io/api/v1/stock/symbol?exchange=US&mic={mic}&token={key}", tries=2, timeout=15)
        if not r: continue
        try:
            arr = r.json()
            for it in arr:
                s = it.get("symbol",""); typ = it.get("type","")
                if not s or "." in s: continue
                if typ and "Common Stock" not in typ: continue
                symbols.append(s)
        except Exception:
            pass
    symbols = sorted(pd.unique(pd.Series(symbols)))
    if not symbols:
        return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META"]
    return symbols[:max(100, limit)]

# ========= הורדת נתונים מהירה =========
@st.cache_data(show_spinner=True, ttl=60*15)
def fetch_history_bulk(tickers: List[str], period_days: int, ma_long: int) -> Dict[str, pd.DataFrame]:
    if not tickers: return {}
    end = datetime.utcnow()
    start = end - timedelta(days=period_days)
    data_raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker', threads=True)
    data: Dict[str, pd.DataFrame] = {}
    if isinstance(data_raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = data_raw[t].dropna()
                if len(df) >= max(60, int(ma_long) + 10):
                    data[t] = df
            except Exception:
                continue
    else:
        df = data_raw.dropna()
        if len(df) >= max(60, int(ma_long) + 10):
            data[tickers[0]] = df
    return data

# ========= עזרי זמן =========
def t_start(): return time.perf_counter()
def t_end(t0): return time.perf_counter() - t0

# ========= UI =========
st.set_page_config(page_title="Asaf's Stock Scout — 2025", page_icon="📈", layout="wide")
st.markdown("""
<style>
  body{direction:rtl}
  .block-container{padding-top:1rem;padding-bottom:2rem}
  h1,h2,h3{ text-align:right }
  [data-testid="stMarkdownContainer"], label{ text-align:right }
  input, textarea{ direction:rtl; text-align:right }
  thead tr th{ text-align:right }
  .rtl-table table { direction: rtl; }
  .rtl-table th, .rtl-table td { text-align: right !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Scout — 2025")

# ===== סטטוס מקורות (6) =====
openai_ok, openai_reason    = check_openai_verbose()
alpha_ok,  alpha_reason     = check_alpha_vantage_verbose()
finnhub_ok, finnhub_reason  = check_finnhub_verbose()
polygon_ok, polygon_reason  = check_polygon_verbose()
tiingo_ok, tiingo_reason    = check_tiingo_verbose()
fmp_ok,    fmp_reason       = check_fmp_verbose()

status_df = pd.DataFrame({
    "מקור":   ["OpenAI","Alpha Vantage","Finnhub","Polygon","Tiingo","FMP"],
    "סטטוס":  [
        "🟢 מחובר" if openai_ok else "🔴 לא מחובר",
        "🟢 מחובר" if alpha_ok  else "🔴 לא מחובר",
        "🟢 מחובר" if finnhub_ok else "🔴 לא מחובר",
        "🟢 מחובר" if polygon_ok else "🔴 לא מחובר",
        "🟢 מחובר" if tiingo_ok else "🔴 לא מחובר",
        "🟢 מחובר" if fmp_ok else "🔴 לא מחובר",
    ],
    "סיבה":   [openai_reason, alpha_reason, finnhub_reason, polygon_reason, tiingo_reason, fmp_reason],
    "מפתח מזוהה": [
        "✅" if os.getenv("OPENAI_API_KEY") else "—",
        "✅" if os.getenv("ALPHA_VANTAGE_API_KEY") else "—",
        "✅" if os.getenv("FINNHUB_API_KEY") else "—",
        "✅" if os.getenv("POLYGON_API_KEY") else "—",
        "✅" if os.getenv("TIINGO_API_KEY") else "—",
        "✅" if os.getenv("FMP_API_KEY") else "—",
    ],
})
st.table(status_df.style.set_properties(**{'text-align':'center','direction':'rtl'}))

# ===== הגדרות בסיס =====
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

# ===== מתקדם =====
with st.expander("מתקדם"):
    st.caption("למתקדמים בלבד — אפשר להשאיר כברירת מחדל.")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        min_price = st.number_input("מחיר מינ׳ $", 0.0, 5000.0, 3.0)
        ma_short  = st.number_input("MA קצר", 5, 100, 20)
    with col_m2:
        min_avg_volume = st.number_input("ממוצע נפח 20 ימים מינ׳", 0, 50_000_000, 500_000, step=50_000)
        ma_long   = st.number_input("MA ארוך", 20, 400, 50)
    with col_m3:
        rsi_bounds = st.slider("טווח RSI מועדף", 0, 100, (40, 75))
        top_validate_k = st.number_input("אימות חיצוני ל-Top K", 5, 50, 12)
        universe_limit = st.number_input("גודל יקום מקסימלי לסריקה", 50, 1500, 350, step=50)
        smart_scan = st.checkbox("סריקה חכמה (Finnhub)", value=True if finnhub_ok else False)
    score_weights_raw = st.text_input("משקולות ניקוד (JSON)",
        value='{"ma":0.25,"mom":0.35,"rsi":0.15,"near_high":0.15,"vol":0.10}')
try:
    SCORE_W = pd.Series(pd.read_json(io.StringIO(score_weights_raw), typ="series"))
except Exception:
    SCORE_W = pd.Series({"ma":0.25,"mom":0.35,"rsi":0.15,"near_high":0.15,"vol":0.10})

# ========= צינור ריצה + מדדי זמן =========
if "av_calls" not in st.session_state: st.session_state.av_calls = 0
phase_times = {}

# שלב 1: יקום
t0 = t_start()
universe = build_universe(limit=int(universe_limit)) if smart_scan else build_universe(limit=200)
# דגימה דטרמיניסטית לפי אלפבית לניהול גודל
if len(universe) > universe_limit:
    bins = {}
    for tkr in universe:
        bins.setdefault(tkr[0], []).append(tkr)
    per = max(1, int(universe_limit / max(1, len(bins))))
    sampled = []
    for k, arr in sorted(bins.items()):
        sampled.extend(arr[:per])
    if len(sampled) < universe_limit:
        sampled.extend([t for t in universe if t not in sampled][: (universe_limit - len(sampled))])
    universe = sampled
phase_times["בונה יקום"] = t_end(t0)

# שלב 2: הורדת נתונים
t0 = t_start()
data_map = fetch_history_bulk(universe, lookback_days, ma_long=int(ma_long))
phase_times["מוריד נתונים"] = t_end(t0)

# שלב 3: ניקוד
t0 = t_start()
rows = []
for t, df in data_map.items():
    if df is None or df.empty: continue
    df = df.copy()
    df["MA_S"]  = df["Close"].rolling(int(ma_short)).mean()
    df["MA_L"]  = df["Close"].rolling(int(ma_long)).mean()
    df["RSI"]   = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["Vol20"] = df["Volume"].rolling(20).mean()

    try: price = float(df["Close"].iloc[-1])
    except Exception: price = np.nan
    if (np.isnan(price)) or (not np.isfinite(price)) or (price < float(min_price)): continue

    try:  last_ma_s = float(df["MA_S"].iloc[-1])
    except Exception: last_ma_s = np.nan
    try:  last_ma_l = float(df["MA_L"].iloc[-1])
    except Exception: last_ma_l = np.nan
    ma_ok = (float(price > last_ma_s) + float(last_ma_s > last_ma_l)) / 2.0 if (np.isfinite(last_ma_s) and np.isfinite(last_ma_l)) else 0.0

    try:  rsi_val = float(df["RSI"].iloc[-1])
    except Exception: rsi_val = np.nan
    lo, hi = rsi_bounds
    if np.isfinite(rsi_val):
        if rsi_val < lo:   rsi_score = max(0.0, 1 - (lo - rsi_val) / 20)
        elif rsi_val > hi: rsi_score = max(0.0, 1 - (rsi_val - hi) / 20)
        else:              rsi_score = 1.0
    else:
        rsi_score = 0.0

    try:  vol20     = float(df["Vol20"].iloc[-1])
    except Exception: vol20 = np.nan
    try:  vol_today = float(df["Volume"].iloc[-1])
    except Exception: vol_today = np.nan
    if np.isfinite(vol20) and vol20 < float(min_avg_volume): continue
    vol_ok = (min(2.0, vol_today / vol20) / 2.0) if (np.isfinite(vol20) and vol20 > 0 and np.isfinite(vol_today)) else 0.0

    ret_1m = float(df["Close"].pct_change(21).iloc[-1])
    ret_3m = float(df["Close"].pct_change(63).iloc[-1])
    ret_6m = float(df["Close"].pct_change(126).iloc[-1])
    momentum = float(np.nanmean([ret_1m, ret_3m, ret_6m]))

    window_52w = min(len(df), 252)
    hi_52w = float(df["Close"].tail(window_52w).max()) if window_52w > 0 else np.nan
    # תיקון קטן: max(0, (hi - price)/hi)
    if np.isfinite(hi_52w) and hi_52w > 0:
        near_high = 1.0 - min(1.0, max(0.0, (hi_52w - price) / hi_52w))
    else:
        near_high = np.nan

    score = (
        float(SCORE_W.get("ma", 0.25)) * ma_ok
        + float(SCORE_W.get("mom", 0.35)) * np.clip((momentum + 0.5), 0, 1)
        + float(SCORE_W.get("rsi", 0.15)) * rsi_score
        + float(SCORE_W.get("near_high", 0.15)) * (near_high if np.isfinite(near_high) else 0.0)
        + float(SCORE_W.get("vol", 0.10)) * (vol_ok if np.isfinite(vol_ok) else 0.0)
    )

    rows.append({
        "Ticker": t,
        "Price_Yahoo": price,
        "Price_Alpha": np.nan,
        "Price_Finnhub": np.nan,
        "Price_Mean": np.nan,
        "Price_STD": np.nan,
        "Sources": 1,                 # כרגע רק Yahoo (יעודכן אחרי אימות חיצוני)
        "Source_List": "🟡Yahoo",     # תצוגת מקורות
        "Score": round(100 * float(score), 1),
        "RSI": round(rsi_val, 1) if np.isfinite(rsi_val) else np.nan,
        "Mom123": f"{ret_1m:.1%}/{ret_3m:.1%}/{ret_6m:.1%}",
        "MA_OK": ma_ok >= 0.5,
        "Near52w": round(near_high * 100, 1) if np.isfinite(near_high) else np.nan,
        "Volx20d": round(vol_today / vol20, 2) if (np.isfinite(vol_today) and np.isfinite(vol20) and vol20>0) else np.nan,
    })

results = pd.DataFrame(rows)
phase_times["מחשב ניקוד"] = t_end(t0)

if results.empty:
    st.warning("אין תוצאות אחרי הסינון. נסה להקל פרמטרים במתקדם.")
    st.stop()

# מיון וחיתוך ראשוני
results = results.sort_values(["Score","Price_Yahoo"], ascending=[False, True]).reset_index(drop=True)

# שלב 4: אימות חיצוני
t0 = t_start()
if use_external_prices and (alpha_ok or finnhub_ok or os.getenv("POLYGON_API_KEY") or os.getenv("TIINGO_API_KEY") or os.getenv("FMP_API_KEY")):
    subset = results.head(int(top_validate_k)).copy()
    for idx, r in subset.iterrows():
        tkr = r["Ticker"]
        py = r["Price_Yahoo"]

        pa   = get_alpha_vantage_price(tkr) if alpha_ok else None
        pf   = get_finnhub_price(tkr) if finnhub_ok else None
        ppg  = get_polygon_price(tkr) if os.getenv("POLYGON_API_KEY") else None
        pti  = get_tiingo_price(tkr)  if os.getenv("TIINGO_API_KEY")  else None
        pfmp = get_fmp_price(tkr)     if os.getenv("FMP_API_KEY")     else None

        prices, srcs = [], []
        if py is not None and not np.isnan(py): prices.append(py);  srcs.append("🟡Yahoo")
        if pa is not None: prices.append(pa);   srcs.append("🟣Alpha")
        if pf is not None: prices.append(pf);   srcs.append("🔵Finnhub")
        if ppg is not None:prices.append(ppg);  srcs.append("🟢Polygon")
        if pti is not None:prices.append(pti);  srcs.append("🟠Tiingo")
        if pfmp is not None:prices.append(pfmp);srcs.append("🟤FMP")

        pmean  = float(np.mean(prices)) if prices else np.nan
        pstd   = float(np.std(prices))  if len(prices) > 1 else np.nan
        sources = len(prices)

        results.loc[idx, ["Price_Alpha","Price_Finnhub","Price_Mean","Price_STD","Sources","Source_List"]] = \
            [pa, pf, pmean, pstd, sources, " · ".join(srcs) if srcs else "—"]
else:
    if use_external_prices:
        st.info("אימות חיצוני הופעל אך אין מפתחות תקינים (Alpha/Finnhub/Polygon/Tiingo/FMP). משתמש רק במחיר Yahoo.")
phase_times["מאמת מחירים"] = t_end(t0)

# ===== טווח החזקה =====
def infer_horizon(row) -> str:
    rsi_v = row.get("RSI", np.nan)
    near  = row.get("Near52w", np.nan)
    score = row.get("Score", 0)
    if isinstance(near,(int,float)) and near >= 90 and isinstance(rsi_v,(int,float)) and 55 <= rsi_v <= 75:
        return "קצר (1–3 חודשים)"
    if isinstance(rsi_v,(int,float)) and rsi_v <= 40 and score >= 60:
        return "בינוני (3–6 חודשים)"
    if score >= 70:
        return "בינוני-ארוך (6–12 חודשים)"
    return "ארוך (6–12 חודשים)"

results["טווח החזקה"] = results.apply(infer_horizon, axis=1)

# ===== הקצאת תקציב =====
def allocate_budget(df: pd.DataFrame, total: float, min_pos: float) -> pd.DataFrame:
    """מחלק תקציב:
    1) ממיין לפי Score ונותן לכל מועמדת חזקה min_pos כל עוד יש תקציב.
    2) את היתרה מחלק פרופורציונלית לניקוד.
    3) אפס מתחת לסף (אם הוגדר), איזון סכום כולל ועיגול.
    """
    df = df.copy()
    df["סכום קנייה ($)"] = 0.0
    if total <= 0 or df.empty:
        return df

    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    remaining = float(total)
    n = len(df)
    can_min = int(min(n, remaining // max(min_pos, 0.0))) if min_pos > 0 else 0
    if min_pos > 0 and can_min > 0:
        df.loc[:can_min-1, "סכום קנייה ($)"] = float(min_pos)
        remaining -= can_min * float(min_pos)

    if remaining > 0:
        weights = df["Score"].clip(lower=0).to_numpy()
        if weights.sum() == 0:
            extra = np.array([remaining / n] * n)
        else:
            extra = remaining * (weights / weights.sum())
        df["סכום קנייה ($)"] = df["סכום קנייה ($)"].to_numpy() + extra

    if min_pos > 0:
        raw = df["סכום קנייה ($)"].to_numpy()
        raw = np.where(raw < min_pos * 0.9999, 0.0, raw)
        s = raw.sum()
        if s > 0 and abs(s - total) / max(total, 1) > 1e-6:
            raw = raw * (total / s)
        df["סכום קנייה ($)"] = raw

    df["סכום קנייה ($)"] = df["סכום קנייה ($)"].round(2)
    return df

TOPN = min(15, len(results))
results = results.head(TOPN).reset_index(drop=True)
results = allocate_budget(results, budget_total, min_position)

# ===== תצוגת מקורות מחיר =====
def source_badges(row):
    if isinstance(row.get("Source_List"), str) and row.get("Source_List"):
        return row["Source_List"]
    badges = []
    if not np.isnan(row.get("Price_Yahoo", np.nan)):  badges.append("🟡Yahoo")
    if not pd.isna(row.get("Price_Finnhub", np.nan)): badges.append("🔵Finnhub")
    if not pd.isna(row.get("Price_Alpha", np.nan)):   badges.append("🟣Alpha")
    return " · ".join(badges) if badges else "—"

results["מקורות מחיר"] = results.apply(source_badges, axis=1)
results["מחיר ממוצע"]  = results["Price_Mean"].round(2)
results["סטיית תקן"]    = results["Price_STD"].round(4)

# ===== כמות מניות מומלצת =====
results["Unit_Price"] = np.where(results["Price_Mean"].notna(), results["Price_Mean"], results["Price_Yahoo"])
results["Unit_Price"] = pd.to_numeric(results["Unit_Price"], errors="coerce")
results["מניות לקנייה"] = np.floor(np.where(results["Unit_Price"] > 0, results["סכום קנייה ($)"] / results["Unit_Price"], 0)).astype(int)
results["עודף ($)"] = np.round(results["סכום קנייה ($)"] - results["מניות לקנייה"] * results["Unit_Price"], 2)

# ===== מדדי זמן + מונה Alpha =====
st.subheader("⏱️ זמני ביצוע")
times_df = pd.DataFrame([{"שלב": k, "משך (שניות)": round(v, 2)} for k, v in phase_times.items()])
st.table(times_df.style.set_properties(**{'text-align':'center','direction':'rtl'}))
if alpha_ok:
    st.caption(f"Alpha Vantage — קריאות בסשן זה: {int(st.session_state.av_calls)}  | טיפ: אל תחרוג מ~5 לדקה ו~500 ליום בגרסה החינמית.")

# ===== המלצה (כרטיסים בלבד; רק מניות לקנייה) =====
st.subheader("🤖 המלצה עכשיו")
st.caption("הכרטיסים הבאים הם **המלצות קנייה** בלבד (הוקצה להן תקציב בפועל).")

rec_df = results[results["סכום קנייה ($)"] > 0].copy()
if rec_df.empty:
    st.info("אין כרגע מניות שעוברות את הסף עם סכום קנייה חיובי. נסה להקטין את 'השקעה מינימלית' או להגדיל את התקציב / להקל ספים ב'מתקדם'.")
else:
    st.markdown("""
    <style>
    .recommend-card{
      direction: rtl; text-align: right;
      background: #f9fafb; border: 1px solid #e5e7eb;
      border-radius: 14px; padding: 14px 16px; margin: 10px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,.04);
    }
    .recommend-card h3{
      margin: 0 0 6px 0; font-size: 1.05rem; display:flex; align-items:center; gap:8px;
    }
    .recommend-grid{
      display: grid; grid-template-columns: repeat(4,minmax(0,1fr));
      gap: 8px; margin-top: 6px;
    }
    .badge{display:inline-block;background:#eef2ff;border:1px solid #dbeafe;
      padding:2px 8px;border-radius:999px;font-size:.85rem}
    .status-buy{background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;padding:2px 8px;border-radius:999px;font-size:.8rem}
    .small{color:#444;font-size:.9rem}
    </style>
    """, unsafe_allow_html=True)

    TOPK_RECOMMEND = min(5, len(rec_df))
    for _, r in rec_df.head(TOPK_RECOMMEND).iterrows():
        mean = r.get("מחיר ממוצע", np.nan)
        std  = r.get("סטיית תקן", np.nan)
        show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
        show_std  = std if not np.isnan(std) else "—"
        sources   = r.get("מקורות מחיר", "—")
        buy_amt   = float(r.get("סכום קנייה ($)", 0.0))
        horizon   = r.get("טווח החזקה", "—")
        rsi_v     = r.get("RSI", np.nan)
        near52    = r.get("Near52w", np.nan)
        score     = r.get("Score", 0)
        unit_price = r.get("Unit_Price", np.nan)
        shares     = int(r.get("מניות לקנייה", 0))
        leftover   = r.get("עודף ($)", 0.0)

        show_mean_fmt  = f"{np.round(show_mean, 2)}" if not np.isnan(show_mean) else "—"
        unit_price_fmt = f"{np.round(unit_price, 2)}" if not np.isnan(unit_price) else "—"

        st.markdown(
            f"""
            <div class="recommend-card">
              <h3>
                <span class="badge">{r['Ticker']}</span>
                <span class="status-buy">סטטוס: קנייה</span>
              </h3>
              <div class="recommend-grid small">
                <div><b>מחיר ממוצע:</b> {show_mean_fmt}</div>
                <div><b>סטיית תקן:</b> {show_std}</div>
                <div><b>RSI:</b> {rsi_v if not np.isnan(rsi_v) else '—'}</div>
                <div><b>קרבה לשיא 52ש׳:</b> {near52 if not np.isnan(near52) else '—'}%</div>
                <div><b>ניקוד:</b> {score}</div>
                <div><b>מקורות:</b> {sources}</div>
                <div><b>סכום קנייה מומלץ:</b> ${buy_amt:,.0f}</div>
                <div><b>טווח החזקה מומלץ:</b> {horizon}</div>
                <div><b>מחיר יחידה לחישוב:</b> {unit_price_fmt}</div>
                <div><b>מניות לקנייה:</b> {shares}</div>
                <div><b>עודף לא מנוצל:</b> ${leftover:,.2f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ===== טבלה =====
st.subheader("🎯 תוצאות מסוננות ומדורגות")
view_df_source = rec_df if not rec_df.empty else results

hebrew_cols = {
    "Ticker":"טיקר",
    "Price_Yahoo":"מחיר (Yahoo)",
    "מחיר ממוצע":"מחיר ממוצע",
    "Unit_Price": "מחיר יחידה (חישוב)",
    "סטיית תקן":"סטיית תקן",
    "מקורות מחיר":"מקורות מחיר",
    "Score":"ניקוד",
    "RSI":"RSI",
    "Near52w":"קרבה לשיא 52ש׳ (%)",
    "Volx20d":"נפח/ממוצע 20 ימים",
    "טווח החזקה":"טווח החזקה",
    "סכום קנייה ($)":"סכום קנייה ($)",
    "מניות לקנייה": "מניות לקנייה",
    "עודף ($)": "עודף ($)"
}

show_order = [
    "טיקר","מחיר (Yahoo)","מחיר ממוצע","מחיר יחידה (חישוב)","סטיית תקן","מקורות מחיר",
    "ניקוד","RSI","קרבה לשיא 52ש׳ (%)","נפח/ממוצע 20 ימים",
    "טווח החזקה","סכום קנייה ($)","מניות לקנייה","עודף ($)"
]

df_view = view_df_source.rename(columns=hebrew_cols)
st.markdown('<div class="rtl-table">', unsafe_allow_html=True)
st.dataframe(df_view[show_order], use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# ===== גרף =====
st.subheader("🔍 גרף לטיקר נבחר")
choice = st.selectbox("בחר טיקר לצפייה", ["(בחר)"] + view_df_source["Ticker"].tolist())
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

# ===== צ'אט AI =====
with st.expander("💬 צ'אט עם ה-AI"):
    if openai_ok:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role":"system","content":"ענה בעברית, קצר ותמציתי, על המניות שבתוצאות והרציונל מאחוריהן."},
                {"role":"user","content": f"הנה טבלת מניות: {df_view.head(5).to_dict(orient='records')}"}
            ]
        for msg in st.session_state.chat_history[2:]:
            if msg["role"] == "user":
                st.markdown(f"<div style='background:#e6f0fa;padding:.6em 1em;border-radius:10px;margin:.3em 0;direction:rtl;text-align:right'><b>🧑‍💻 אתה:</b> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#f6f6f6;padding:.6em 1em;border-radius:10px;margin:.3em 0 .8em 0;direction:rtl;text-align:right;box-shadow:0 1px 4px #0001'><b>🤖 AI:</b> {msg['content']}</div>", unsafe_allow_html=True)
        user_q = st.text_input("שאלה (לגבי הממצאים/מניות):", key="ai_chat_input")
        if st.button("שלח"):
            st.session_state.chat_history.append({"role":"user","content":user_q})
            client = openai_client()
            try:
                rsp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.chat_history,
                    temperature=0.4,
                )
                answer = rsp.choices[0].message.content.strip()
                st.session_state.chat_history.append({"role":"assistant","content":answer})
                st.rerun()
            except Exception as e:
                st.error(f"שגיאה מה-AI: {e}")
    else:
        st.info("AI לא מחובר (חסר OPENAI_API_KEY).")

# ===== הערות =====
with st.expander("ℹ️ הערות ומתודולוגיה"):
    st.markdown("""
- הנתונים ההיסטוריים מתקבלים מ-**Yahoo Finance** (`yfinance`) — ללא צורך במפתח.
- אימות מחירים חיצוני (אם הופעל ויש מפתחות): **Alpha Vantage**, **Finnhub**, **Polygon**, **Tiingo**, **FMP** — מוצג כ״מקורות מחיר״ לכל מניה.
- ניקוד משלב: MA, מומנטום (1/3/6 חודשים), RSI בתחום מועדף, קרבה לשיא 52ש׳, ונפח מול ממוצע 20 ימים.
- ההמלצה כוללת **סכום קנייה**, **מס' מניות לקנייה** לכל מניה מתוך התקציב, ו**טווח החזקה** משוער. כל הכרטיסים שמוצגים הם **קנייה**.
- מדדי זמן מציגים משך לכל שלב; מונה Alpha מציג כמה קריאות ביצעת בסשן זה.
- אין באמור ייעוץ השקעות.
""")
