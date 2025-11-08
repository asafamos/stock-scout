# -*- coding: utf-8 -*-
"""
Asaf Stock Scout — סורק מניות 2025 (גרסה משולבת מקסימום)
--------------------------------------------------------
מכיל את כל היכולות הקיימות + כל השדרוגים החדשים:
• שכבה פונדמנטלית (FMP): Growth/Quality/Valuation + בונוס Earnings Surprise.
• כללי סיכון: מגבלת סקטור (Sector Cap), מסנן בטא מול SPY/QQQ, תקרת פוזיציה אחוזית.
• ATR Stops/Targets בכרטיסי ההמלצה.
• כל היציבות הקודמת: הורדות עמידות, cache, throttling, אימות מחירים חיצוני, earnings blackout, value check ועוד.
הערה: אין באמור ייעוץ השקעות.
"""

from __future__ import annotations
import os, io, time, math, warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from streamlit.components.v1 import html as st_html
import html as html_escape  # stdlib

warnings.filterwarnings("ignore")

# ========= ENV/Secrets =========
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if "secrets" in dir(st) and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

load_dotenv(find_dotenv(usecwd=True))
for _extra in ["nev", "stock_scout.nev", ".env.local", ".env.production"]:
    try:
        if os.path.exists(_extra):
            load_dotenv(_extra)
    except Exception:
        pass

# ========= רשת/Backoff =========
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

def alpha_throttle(min_gap_seconds: float = 12.0):
    ts_key = "_alpha_last_call_ts"
    last = st.session_state.get(ts_key, 0.0)
    now  = time.time()
    gap  = now - last
    if gap < min_gap_seconds:
        time.sleep(min_gap_seconds - gap)
    st.session_state[ts_key] = time.time()

# ========= סטטוסים =========
@st.cache_data(ttl=300)
def check_openai_verbose():
    key = _env("OPENAI_API_KEY")
    if not key: return False, "Missing API key"
    try:
        from openai import OpenAI
        OpenAI(api_key=key).models.list()
        return True, "OK"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=300)
def check_alpha_vantage_verbose():
    key = _env("ALPHA_VANTAGE_API_KEY")
    if not key: return False, "Missing API key"
    try:
        r = http_get_retry(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=MSFT&apikey={key}", tries=2, timeout=12)
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
    key = _env("FINNHUB_API_KEY")
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
    key = _env("POLYGON_API_KEY")
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
    key = _env("TIINGO_API_KEY")
    if not key: return False, "Missing API key"
    try:
        r = http_get_retry(f"https://api.tiingo.com/tiingo/daily/AAPL/prices?token={key}&resampleFreq=daily", tries=2, timeout=10)
        if not r: return False, "Timeout/Network"
        arr = r.json()
        ok = isinstance(arr, list) and arr and isinstance(arr[-1], dict) and ("close" in arr[-1])
        return ok, ("OK" if ok else "Bad response")
    except Exception:
        return False, "Exception"

@st.cache_data(ttl=300)
def check_fmp_verbose():
    key = _env("FMP_API_KEY")
    if not key: return False, "Missing API key"
    try:
        r = http_get_retry(f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={key}", tries=3, timeout=16, headers={"User-Agent":"StockScout/1.0"})
        if not r: return False, "Timeout/Network"
        j = r.json()
        if isinstance(j, list) and j and isinstance(j[0], dict) and ("price" in j[0]): return True, "OK"
        if isinstance(j, dict) and "Error Message" in j: return False, j.get("Error Message","Error")
        return False, "Bad/Unknown response"
    except Exception:
        return False, "Exception"

# ========= אינדיקטורים =========
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

def macd_line(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(period, min_periods=period).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=period).mean() / (atr14 + 1e-9))
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(period, min_periods=period).mean()

def _sigmoid(x, k=3.0):
    try:
        return 1.0 / (1.0 + np.exp(-k * x))
    except Exception:
        return 0.5

# ========= OpenAI (לא חובה) =========
def openai_client():
    try:
        from openai import OpenAI
        key = _env("OPENAI_API_KEY")
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
    key = _env("ALPHA_VANTAGE_API_KEY")
    if not key: return None
    if "av_calls" not in st.session_state: st.session_state.av_calls = 0
    alpha_throttle()
    r = http_get_retry(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={key}", tries=3, timeout=16)
    if not r: st.session_state.av_calls += 1; return None
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
    key = _env("FINNHUB_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={key}", tries=2, timeout=12)
    if not r: return None
    try:
        data = r.json()
        return float(data["c"]) if "c" in data else None
    except Exception:
        return None

def get_polygon_price(ticker: str) -> float | None:
    key = _env("POLYGON_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={key}", tries=2, timeout=10)
    if not r: return None
    try:
        j = r.json()
        if j.get("resultsCount", 0) > 0 and "results" in j:
            return float(j["results"][0]["c"])
    except Exception:
        return None
    return None

def get_tiingo_price(ticker: str) -> float | None:
    key = _env("TIINGO_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?token={key}&resampleFreq=daily", tries=2, timeout=10)
    if not r: return None
    try:
        arr = r.json()
        if isinstance(arr, list) and arr:
            return float(arr[-1].get("close", np.nan))
    except Exception:
        return None
    return None

def get_fmp_price(ticker: str) -> float | None:
    key = _env("FMP_API_KEY")
    if not key: return None
    r = http_get_retry(f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={key}", tries=3, timeout=16, headers={"User-Agent":"StockScout/1.0"})
    if not r: return None
    try:
        j = r.json()
        if isinstance(j, list) and j:
            return float(j[0].get("price", np.nan))
        if isinstance(j, dict) and "Error Message" in j:
            st.warning(f"FMP error: {j.get('Error Message')}")
            return None
    except Exception:
        return None
    return None

# ========= יקום =========
def build_universe(limit: int = 350) -> List[str]:
    ok, _ = check_finnhub_verbose()
    if not ok:
        return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","QCOM","ADBE","CRM",
                "NFLX","INTC","ORCL","PANW","SNPS","CDNS","MU","KLAC"]
    key = _env("FINNHUB_API_KEY")
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
    if len(symbols) > limit:
        bins: Dict[str, List[str]] = {}
        for tkr in symbols:
            bins.setdefault(tkr[0], []).append(tkr)
        per = max(1, int(limit / max(1, len(bins))))
        sampled: List[str] = []
        for k, arr in sorted(bins.items()):
            sampled.extend(sorted(arr)[:per])
        if len(sampled) < limit:
            sampled.extend([t for t in symbols if t not in sampled][: (limit - len(sampled))])
        symbols = sampled
    return symbols[:limit]

# ========= הורדת נתונים =========
def safe_yf_download(tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not tickers: return out
    try:
        data_raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker', threads=True)
        if isinstance(data_raw.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    df = data_raw[t].dropna()
                    if not df.empty: out[t] = df
                except Exception:
                    continue
        else:
            df = data_raw.dropna()
            if not df.empty: out[tickers[0]] = df
    except Exception:
        pass
    missing = [t for t in tickers if t not in out]
    for t in missing:
        try:
            dfi = yf.download(t, start=start, end=end, auto_adjust=True, progress=False).dropna()
            if not dfi.empty: out[t] = dfi
        except Exception:
            continue
    return out

@st.cache_data(show_spinner=True, ttl=60*15)
def fetch_history_bulk(tickers: List[str], period_days: int, ma_long: int) -> Dict[str, pd.DataFrame]:
    if not tickers: return {}
    end = datetime.utcnow(); start = end - timedelta(days=period_days)
    data: Dict[str, pd.DataFrame] = safe_yf_download(tickers, start, end)
    out = {}
    min_len = max(60, int(ma_long) + 10)
    for t, df in data.items():
        try:
            if len(df) >= min_len: out[t] = df
        except Exception:
            continue
    return out

# ========= Earnings יציב =========
@st.cache_data(ttl=60*60)
def get_next_earnings_date(ticker: str) -> Optional[datetime]:
    try:
        key = _env("FINNHUB_API_KEY")
        if key:
            today = datetime.utcnow().date()
            url = ("https://finnhub.io/api/v1/calendar/earnings"
                   f"?from={today.isoformat()}&to={(today + timedelta(days=180)).isoformat()}"
                   f"&symbol={ticker}&token={key}")
            r = http_get_retry(url, tries=2, timeout=12)
            if r:
                data = r.json()
                for row in data.get("earningsCalendar", []):
                    if row.get("symbol") == ticker and row.get("date"):
                        return datetime.fromisoformat(row["date"])
    except Exception:
        pass
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=4)
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            now = pd.Timestamp.utcnow()
            future = ed[ed.index >= now]
            dt = (future.index.min() if not future.empty else ed.index.max())
            if pd.notna(dt): return dt.to_pydatetime()
    except Exception:
        pass
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

@st.cache_data(ttl=60*30)
def _earnings_batch(symbols: List[str]) -> Dict[str, Optional[datetime]]:
    out = {}
    if not symbols: return out
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(get_next_earnings_date, s): s for s in symbols}
        for f in as_completed(futs):
            s = futs[f]
            try: out[s] = f.result()
            except Exception: out[s] = None
    return out

# ========= פונדמנטלי (FMP) =========
@st.cache_data(ttl=60*60)
def fetch_fundamentals_bundle(ticker: str) -> dict:
    key = _env("FMP_API_KEY")
    if not key: return {}
    out = {}

    r = http_get_retry(f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?limit=1&apikey={key}",
                       tries=2, timeout=12, headers={"User-Agent":"StockScout/1.0"})
    if r:
        arr = r.json()
        if isinstance(arr, list) and arr:
            out["roe"]  = arr[0].get("returnOnEquity", np.nan)
            out["roic"] = arr[0].get("returnOnCapitalEmployed", np.nan)
            out["gm"]   = arr[0].get("grossProfitMargin", np.nan)
            out["ps"]   = arr[0].get("priceToSalesRatio", np.nan)
            out["pe"]   = arr[0].get("priceEarningsRatio", np.nan)
            out["de"]   = arr[0].get("debtEquityRatio", np.nan)

    isr = http_get_retry(f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=4&apikey={key}",
                         tries=2, timeout=12, headers={"User-Agent":"StockScout/1.0"})
    if isr:
        arr = isr.json()
        if isinstance(arr, list) and len(arr) >= 2:
            rev0 = arr[0].get("revenue", np.nan); rev1 = arr[1].get("revenue", np.nan)
            eps0 = arr[0].get("eps", np.nan);     eps1 = arr[1].get("eps", np.nan)
            out["rev_g_yoy"] = (rev0 - rev1) / rev1 if (isinstance(rev0,(int,float)) and isinstance(rev1,(int,float)) and rev1) else np.nan
            out["eps_g_yoy"] = (eps0 - eps1) / abs(eps1) if (isinstance(eps0,(int,float)) and isinstance(eps1,(int,float)) and eps1) else np.nan

    # Sector/Profile
    prof = http_get_retry(f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={key}",
                          tries=2, timeout=12, headers={"User-Agent":"StockScout/1.0"})
    if prof:
        arr = prof.json()
        if isinstance(arr, list) and arr:
            out["sector"] = (arr[0].get("sector") or "Unknown") or "Unknown"

    # Earnings surprise (אחרון)
    es = http_get_retry(f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?limit=1&apikey={key}",
                        tries=2, timeout=12, headers={"User-Agent":"StockScout/1.0"})
    if es:
        arr = es.json()
        if isinstance(arr, list) and arr:
            out["surprise"] = arr[0].get("surprisePercentage", np.nan)  # אחוז

    return out

def _to_01(x, low, high):
    if not isinstance(x,(int,float)) or not np.isfinite(x): return np.nan
    return np.clip((x - low) / (high - low), 0, 1)

def fundamental_score(d: dict) -> float:
    g_rev = _to_01(d.get("rev_g_yoy", np.nan), 0.00, 0.30)
    g_eps = _to_01(d.get("eps_g_yoy", np.nan), 0.00, 0.30)
    growth = np.nanmean([g_rev, g_eps])

    q_roe  = _to_01(d.get("roe", np.nan), 0.05, 0.25)
    q_roic = _to_01(d.get("roic", np.nan), 0.05, 0.20)
    q_gm   = _to_01(d.get("gm", np.nan), 0.20, 0.60)
    quality = np.nanmean([q_roe, q_roic, q_gm])

    pe     = d.get("pe", np.nan); ps = d.get("ps", np.nan)
    val_pe = np.nan if not np.isfinite(pe) else _to_01(40 - np.clip(pe, 0, 40), 0, 40)  # P/E<=40
    val_ps = np.nan if not np.isfinite(ps) else _to_01(10 - np.clip(ps, 0, 10), 0, 10)  # P/S<=10
    valuation = np.nanmean([val_pe, val_ps])

    de = d.get("de", np.nan)
    penalty = 0.0
    if isinstance(de,(int,float)) and np.isfinite(de) and de > 2.0:
        penalty += 0.15

    comp = np.nanmean([growth, quality, valuation])
    if not np.isfinite(comp): comp = 0.0

    # בונוס קטן אם ההפתעה האחרונה חיובית (PED)
    surprise = d.get("surprise", np.nan)
    bonus = 0.05 if (isinstance(surprise,(int,float)) and surprise >= 2.0) else 0.0

    return float(np.clip(comp - penalty + bonus, 0.0, 1.0))

@st.cache_data(ttl=60*60)
def fetch_beta_vs_benchmark(ticker: str, bench: str = "SPY", days: int = 252) -> float:
    try:
        end = datetime.utcnow(); start = end - timedelta(days=days+30)
        df_t = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df_b = yf.download(bench,  start=start, end=end, auto_adjust=True, progress=False)
        if df_t.empty or df_b.empty: return np.nan
        rt = df_t["Close"].pct_change().dropna()
        rb = df_b["Close"].pct_change().dropna()
        j = pd.concat([rt, rb], axis=1).dropna()
        j.columns = ["rt","rb"]
        if len(j) < 40: return np.nan
        # β = slope(rt ~ rb)
        x = j["rb"].to_numpy(); y = j["rt"].to_numpy()
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return np.nan

# ========= UI =========
st.set_page_config(page_title="Asaf's Stock Scout — 2025", page_icon="📈", layout="wide")

col_clear, _ = st.columns([1,6])
with col_clear:
    if st.button("🧹 נקה קאש (Cache)"):
        st.cache_data.clear()
        st.success("נוקו כל המטמונים (cache).")

st.markdown("""
<style>
body{direction:rtl}
.block-container{padding-top:1rem;padding-bottom:2rem}
h1,h2,h3{text-align:right}
[data-testid="stMarkdownContainer"], label{ text-align:right }
input, textarea{ direction:rtl; text-align:right }
thead tr th{ text-align:right }
.rtl-table table{ direction:rtl }
.rtl-table th,.rtl-table td{ text-align:right !important }
.badge{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;color:#1e293b;
  padding:2px 10px;border-radius:999px;font-weight:600;letter-spacing:.2px}
.status-buy{background:#ecfdf5;border:1px solid #34d399;color:#065f46;padding:2px 10px;border-radius:999px;font-weight:600}
.recommend-card{direction: rtl; text-align: right;background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.recommend-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:6px}
.small{color:#444;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Scout — 2025")

# ===== סטטוס מקורות =====
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
        "✅" if _env("OPENAI_API_KEY") else "—",
        "✅" if _env("ALPHA_VANTAGE_API_KEY") else "—",
        "✅" if _env("FINNHUB_API_KEY") else "—",
        "✅" if _env("POLYGON_API_KEY") else "—",
        "✅" if _env("TIINGO_API_KEY") else "—",
        "✅" if _env("FMP_API_KEY") else "—",
    ],
})
st.table(status_df.style.set_properties(**{'text-align':'center','direction':'rtl'}))

# ===== בסיס =====
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
    st.caption("אפשר להשאיר כברירת מחדל.")
    col_m0, col_m1, col_m2, col_m3 = st.columns([2,1,1,1])
    with col_m0:
        manual_tickers_raw = st.text_input("טיקרים ידניים (מופרדים בפסיקים, אופציונלי)", value="")
    with col_m1:
        min_price = st.number_input("מחיר מינ׳ $", 0.0, 5000.0, 3.0)
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

    default_weights = {
        "ma":0.22, "mom":0.30, "rsi":0.12,
        "near_high_bell":0.10, "vol":0.08,
        "overext":0.08, "pullback":0.05,
        "risk_reward":0.03, "macd":0.01, "adx":0.01
    }
    risk_preset = st.selectbox("Preset סיכון", ["Balanced (ברירת מחדל)", "Conservative", "Aggressive"], index=0)
    if risk_preset == "Conservative":
        default_weights.update({"vol":0.12, "overext":0.12, "near_high_bell":0.08, "mom":0.24})
    elif risk_preset == "Aggressive":
        default_weights.update({"vol":0.06, "overext":0.05, "near_high_bell":0.12, "mom":0.36})

    min_dollar_volume = st.number_input("מינ׳ דולר-ווליום (Price × Vol20)", 0, 2_000_000_000, 5_000_000, step=500_000)
    max_atr_price_cap = st.number_input("מקס׳ ATR/Price (סף קשיח)", 0.01, 0.50, 0.08, step=0.01)
    max_overext_cap   = st.number_input("מקס׳ Overextension מול MA_L (סף קשיח)", 0.05, 1.00, 0.30, step=0.05)

    earnings_blackout_days = st.number_input("בלוק-אאוט דו״חות (ימים לפני/אחרי) – 0=כבוי", 0, 30, 7)
    earnings_check_topk    = st.number_input("בדיקת דו״חות ל-Top-K", 3, 50, 12)

    st.markdown("**משקולות ניקוד (JSON) — אפשר לשנות ידנית**")
    c1, c2 = st.columns([3,1])
    with c1:
        score_weights_raw = st.text_input("משקולות", value=pd.Series(default_weights).to_json(force_ascii=False))
    with c2:
        if st.button("איפוס משקולות"):
            st.session_state["reset_weights"] = True
    if st.session_state.get("reset_weights"):
        st.session_state["reset_weights"] = False
        score_weights_raw = pd.Series(default_weights).to_json(force_ascii=False)
        st.experimental_rerun()

    value_filter_enabled = st.toggle("סינון ערך בסיסי (P/E 5–40, P/B < 10) לטופ-K בלבד", value=False)
    macd_adx_enabled     = st.toggle("שקלול MACD/ADX", value=False)

    st.divider()
    st.subheader("🧱 שכבה פונדמנטלית + כללי סיכון (חדשים)")
    fundamental_enabled  = st.toggle("שכבה פונדמנטלית (FMP)", value=False)
    fundamental_weight   = st.slider("משקל פונדמנטלי בניקוד (0–0.35)", 0.0, 0.35, 0.15)
    fundamental_top_k    = st.number_input("כמה מועמדים להביא להם נתונים פונדמנטליים", 10, 200, 50, step=5)
    earnings_surprise_bonus = st.toggle("בונוס על הפתעת רווח חיובית (FMP)", value=True)

    sector_cap_enabled   = st.toggle("הגבלת מספר פוזיציות לסקטור", value=False)
    sector_cap_max       = st.number_input("מקסימום פוזיציות לסקטור", 1, 10, 3)

    beta_filter_enabled  = st.toggle("מסנן בטא", value=False)
    beta_benchmark       = st.selectbox("Benchmark לבטא", ["SPY","QQQ"], index=0)
    beta_max_allowed     = st.slider("בטא מקסימלית מותרת", 0.5, 3.0, 2.0, 0.1)
    beta_top_k           = st.number_input("חישוב בטא רק לטופ-K (לשיפור מהירות)", 10, 200, 60, step=5)

    max_position_pct     = st.slider("תקרת גודל פוזיציה (% מהתקציב)", 5, 50, 15)

# ===== משקולות =====
def _normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    keys = ["ma","mom","rsi","near_high_bell","vol","overext","pullback","risk_reward","macd","adx"]
    w = {k: float(d.get(k, default_weights.get(k, 0.0))) for k in keys}
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0: return default_weights
    return {k: max(0.0, v)/s for k, v in w.items()}

try:
    SCORE_W = pd.Series(_normalize_weights(pd.read_json(io.StringIO(score_weights_raw), typ="series").to_dict()))
except Exception:
    SCORE_W = pd.Series(_normalize_weights(default_weights))

# ===== עזר =====
def t_start(): return time.perf_counter()
def t_end(t0): return time.perf_counter() - t0

# ===== צינור =====
if "av_calls" not in st.session_state: st.session_state.av_calls = 0
phase_times: Dict[str, float] = {}

# 1) יקום
t0 = t_start()
universe = build_universe(limit=int(universe_limit)) if smart_scan else build_universe(limit=200)

manual_list = []
if manual_tickers_raw.strip():
    manual_list = [x.strip().upper() for x in manual_tickers_raw.split(",") if x.strip()]
    manual_list = [t for t in manual_list if "." not in t]
if manual_list:
    universe = sorted(list(pd.unique(pd.Series(universe + manual_list))))

if len(universe) > universe_limit:
    bins: Dict[str, List[str]] = {}
    for tkr in universe: bins.setdefault(tkr[0], []).append(tkr)
    per = max(1, int(universe_limit / max(1, len(bins))))
    sampled = []
    for k, arr in sorted(bins.items()):
        sampled.extend(sorted(arr)[:per])
    if len(sampled) < universe_limit:
        sampled.extend([t for t in sorted(universe) if t not in sampled][: (universe_limit - len(sampled))])
    universe = sampled
phase_times["בונה יקום"] = t_end(t0)

# 2) היסטוריה
t0 = t_start()
data_map = fetch_history_bulk(universe, lookback_days, ma_long=int(ma_long))
phase_times["מוריד נתונים"] = t_end(t0)

# 3) ניקוד טכני + סינונים קשיחים
t0 = t_start()
rows = []
lo_rsi, hi_rsi = rsi_bounds
for t, df in data_map.items():
    if df is None or df.empty: continue
    df = df.copy()
    df["MA_S"]  = df["Close"].rolling(int(ma_short)).mean()
    df["MA_L"]  = df["Close"].rolling(int(ma_long)).mean()
    df["RSI"]   = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["Vol20"] = df["Volume"].rolling(20).mean()

    if macd_adx_enabled:
        macd_line_v, macd_sig_v, macd_hist_v = macd_line(df["Close"])
        df["MACD"], df["MACD_SIG"], df["MACD_HIST"] = macd_line_v, macd_sig_v, macd_hist_v
        df["ADX14"] = adx(df, 14)

    try: price = float(df["Close"].iloc[-1])
    except Exception: price = np.nan
    if (np.isnan(price)) or (not np.isfinite(price)) or (price < float(min_price)): continue

    try: last_ma_s = float(df["MA_S"].iloc[-1])
    except Exception: last_ma_s = np.nan
    try: last_ma_l = float(df["MA_L"].iloc[-1])
    except Exception: last_ma_l = np.nan
    ma_ok = (float(price > last_ma_s) + float(last_ma_s > last_ma_l)) / 2.0 if (np.isfinite(last_ma_s) and np.isfinite(last_ma_l)) else 0.0

    try: rsi_val = float(df["RSI"].iloc[-1])
    except Exception: rsi_val = np.nan
    if np.isfinite(rsi_val):
        if rsi_val < lo_rsi:   rsi_score = max(0.0, 1 - (lo_rsi - rsi_val) / 20)
        elif rsi_val > hi_rsi: rsi_score = max(0.0, 1 - (rsi_val - hi_rsi) / 20)
        else:                  rsi_score = 1.0
    else:
        rsi_score = 0.0

    try: vol20 = float(df["Vol20"].iloc[-1])
    except Exception: vol20 = np.nan
    try: vol_today = float(df["Volume"].iloc[-1])
    except Exception: vol_today = np.nan
    if np.isfinite(vol20) and vol20 < float(min_avg_volume): continue
    vol_ok = (min(2.0, vol_today / vol20) / 2.0) if (np.isfinite(vol20) and vol20 > 0 and np.isfinite(vol_today)) else 0.0

    ret_1m = float(df["Close"].pct_change(21).iloc[-1])
    ret_3m = float(df["Close"].pct_change(63).iloc[-1])
    ret_6m = float(df["Close"].pct_change(126).iloc[-1])
    mom_raw = float(np.nanmean([ret_1m, ret_3m, ret_6m]))
    mom_score = float(_sigmoid(mom_raw))

    window_52w = min(len(df), 252)
    hi_52w = float(df["Close"].tail(window_52w).max()) if window_52w > 0 else np.nan
    if np.isfinite(hi_52w) and hi_52w > 0:
        near_high_raw = 1.0 - min(1.0, max(0.0, (hi_52w - price) / hi_52w))
        if near_high_raw >= 0.95: near_high_score = 0.45
        elif 0.75 <= near_high_raw <= 0.90: near_high_score = 1.00
        elif 0.90 < near_high_raw < 0.95:  near_high_score = 0.75
        else:                               near_high_score = near_high_raw
    else:
        near_high_raw, near_high_score = np.nan, 0.0

    if np.isfinite(last_ma_l) and last_ma_l > 0:
        overext_ratio = max(0.0, (price - last_ma_l) / last_ma_l)
        overext_score = 1.0 - min(1.0, overext_ratio / max(1e-6, overext_threshold))
    else:
        overext_ratio, overext_score = np.nan, 0.0

    if np.isfinite(hi_52w) and hi_52w > 0:
        ratio_to_high = price / hi_52w
        if pullback_low <= ratio_to_high <= pullback_high:
            pullback_score = 1.0
        else:
            dist = min(abs(ratio_to_high - pullback_low), abs(ratio_to_high - pullback_high))
            pullback_score = max(0.0, 1.0 - dist * 10)
    else:
        pullback_score = 0.0

    try: atr14 = float(df["ATR14"].iloc[-1])
    except Exception: atr14 = np.nan
    if np.isfinite(atr14) and price > 0:
        vol_rel = atr14 / price
        volatility_score = 1.0 - min(1.0, vol_rel / 0.05)
    else:
        vol_rel, volatility_score = np.nan, 0.0

    dollar_vol = (price * vol20) if (np.isfinite(price) and np.isfinite(vol20)) else 0.0
    if dollar_vol < float(min_dollar_volume): continue
    if np.isfinite(vol_rel) and vol_rel > float(max_atr_price_cap): continue
    if np.isfinite(overext_ratio) and overext_ratio > float(max_overext_cap): continue

    if np.isfinite(hi_52w) and np.isfinite(atr14) and atr14 > 0:
        reward_risk = max(0.0, (hi_52w - price) / atr14)
        rr_score = min(1.0, reward_risk / 4.0)
    else:
        reward_risk, rr_score = np.nan, 0.0

    macd_score = adx_score = 0.0
    if macd_adx_enabled and "MACD" in df.columns:
        macd_v = float(df["MACD"].iloc[-1]); macd_sig = float(df["MACD_SIG"].iloc[-1])
        macd_score = 1.0 if macd_v > macd_sig else 0.0
    if macd_adx_enabled and "ADX14" in df.columns:
        adx_v = float(df["ADX14"].iloc[-1])
        adx_score = np.clip((adx_v - 15) / 20.0, 0.0, 1.0) if np.isfinite(adx_v) else 0.0

    score = (
        float(SCORE_W.get("ma", 0.22)) * ma_ok +
        float(SCORE_W.get("mom", 0.30)) * mom_score +
        float(SCORE_W.get("rsi", 0.12)) * rsi_score +
        float(SCORE_W.get("near_high_bell", 0.10)) * near_high_score +
        float(SCORE_W.get("vol", 0.08)) * (vol_ok if np.isfinite(vol_ok) else 0.0) +
        float(SCORE_W.get("overext", 0.08)) * overext_score +
        float(SCORE_W.get("pullback", 0.05)) * pullback_score +
        float(SCORE_W.get("risk_reward", 0.03)) * rr_score +
        float(SCORE_W.get("macd", 0.01)) * macd_score +
        float(SCORE_W.get("adx", 0.01)) * adx_score
    )

    rows.append({
        "Ticker": t,
        "Price_Yahoo": price,
        "Score_Tech": round(100 * float(score), 1),
        "RSI": round(rsi_val, 1) if np.isfinite(rsi_val) else np.nan,
        "Mom123": f"{ret_1m:.1%}/{ret_3m:.1%}/{ret_6m:.1%}",
        "MA_OK": ma_ok >= 0.5,
        "Near52w": round(near_high_raw * 100, 1) if np.isfinite(near_high_raw) else np.nan,
        "Volx20d": round(vol_today / vol20, 2) if (np.isfinite(vol_today) and np.isfinite(vol20) and vol20>0) else np.nan,
        "OverextRatio": round(overext_ratio, 3) if np.isfinite(overext_ratio) else np.nan,
        "ATR_Price": round(vol_rel, 4) if np.isfinite(vol_rel) else np.nan,
        "RewardRisk": round(reward_risk, 2) if np.isfinite(reward_risk) else np.nan,
        "ATR14": atr14
    })

results = pd.DataFrame(rows)
phase_times["מחשב ניקוד (טכני)"] = t_end(t0)

if results.empty:
    st.warning("אין תוצאות אחרי הסינון. נסה להקל פרמטרים במתקדם.")
    st.stop()

# מיון לפי ניקוד טכני (ראשוני)
results = results.sort_values(["Score_Tech","Ticker"], ascending=[False, True]).reset_index(drop=True)

# 3.א) שכבה פונדמנטלית לטופ-K ראשוני -> ניקוד משולב + סקטור
if fundamental_enabled and fmp_ok:
    t0 = t_start()
    take_k = int(min(fundamental_top_k, len(results)))
    fund_cols = ["Fundamental_S","Sector","SurprisePct","PE_f","PS_f","ROE_f","ROIC_f","GM_f","DE_f"]
    for c in fund_cols:
        results[c] = np.nan
    sect = {}
    for idx in results.head(take_k).index:
        tkr = results.at[idx,"Ticker"]
        d = fetch_fundamentals_bundle(tkr)
        fs = fundamental_score(d)
        results.loc[idx,"Fundamental_S"] = round(100*fs,1)
        results.loc[idx,"SurprisePct"]   = d.get("surprise", np.nan)
        results.loc[idx,"PE_f"]          = d.get("pe", np.nan)
        results.loc[idx,"PS_f"]          = d.get("ps", np.nan)
        results.loc[idx,"ROE_f"]         = d.get("roe", np.nan)
        results.loc[idx,"ROIC_f"]        = d.get("roic", np.nan)
        results.loc[idx,"GM_f"]          = d.get("gm", np.nan)
        results.loc[idx,"DE_f"]          = d.get("de", np.nan)
        sec = d.get("sector","Unknown") or "Unknown"
        results.loc[idx,"Sector"] = sec
        sect[tkr] = sec

    # ניקוד סופי משולב
    results["Score"] = results["Score_Tech"]
    results.loc[results.head(take_k).index, "Score"] = (
        (1 - float(fundamental_weight)) * results.loc[results.head(take_k).index, "Score_Tech"] +
        float(fundamental_weight) * results.loc[results.head(take_k).index, "Fundamental_S"].fillna(0)
    )
    results = results.sort_values(["Score","Ticker"], ascending=[False, True]).reset_index(drop=True)
    phase_times["פונדמנטלי (FMP)"] = t_end(t0)
else:
    results["Score"] = results["Score_Tech"]

# Earnings blackout (Top-K)
if earnings_blackout_days > 0:
    to_check_idx = list(results.head(int(earnings_check_topk)).index)
    symbols = [results.at[i,"Ticker"] for i in to_check_idx]
    ed_map = _earnings_batch(symbols)
    now_utc = datetime.utcnow()
    keep_mask = np.ones(len(results), dtype=bool)
    for idx in to_check_idx:
        tkr = results.at[idx, "Ticker"]
        dt_earn = ed_map.get(tkr)
        if dt_earn is None: continue
        gap_days = abs((dt_earn - now_utc).days)
        if gap_days <= int(earnings_blackout_days):
            keep_mask[idx] = False
            results.at[idx, "EarningsNote"] = f"Excluded: earnings within {gap_days}d"
    results = results[keep_mask].reset_index(drop=True)
    if results.empty:
        st.warning("כל המועמדות בטופ-K נפסלו עקב חלון דו\"חות. הקל ספים/הגדל K או נטרל blackout זמנית.")
        st.stop()

# 3.ב) מסנן בטא לטופ-K (אופציונלי)
if beta_filter_enabled:
    t0 = t_start()
    take_k_beta = int(min(beta_top_k, len(results)))
    beta_vals = {}
    for idx in results.head(take_k_beta).index:
        tkr = results.at[idx,"Ticker"]
        beta_vals[tkr] = fetch_beta_vs_benchmark(tkr, bench=beta_benchmark, days=252)
        results.loc[idx,"Beta"] = beta_vals[tkr]
    mask = ~((results["Beta"].notna()) & (results["Beta"] > float(beta_max_allowed)))
    results = results[mask].reset_index(drop=True)
    phase_times["מסנן בטא"] = t_end(t0)

# אימות מחירים (Top-K)
t0 = t_start()
results["Price_Alpha"] = np.nan
results["Price_Finnhub"] = np.nan
results["Price_Mean"] = np.nan
results["Price_STD"] = np.nan
results["Sources"] = 1
results["Source_List"] = "🟡Yahoo"

def _fetch_external_for(tkr: str, py: float) -> Tuple[str, Dict[str, Optional[float]], List[str]]:
    vals, srcs = {}, []
    if not math.isnan(py):
        vals["Yahoo"] = py; srcs.append("🟡Yahoo")
    pa = pf = ppg = pti = pfmp = None
    if alpha_ok: pa = get_alpha_vantage_price(tkr)
    if finnhub_ok: pf = get_finnhub_price(tkr)
    if _env("POLYGON_API_KEY"): ppg = get_polygon_price(tkr)
    if _env("TIINGO_API_KEY"):  pti = get_tiingo_price(tkr)
    if _env("FMP_API_KEY"):     pfmp = get_fmp_price(tkr)
    if pa is not None:   vals["Alpha"]   = pa;   srcs.append("🟣Alpha")
    if pf is not None:   vals["Finnhub"] = pf;   srcs.append("🔵Finnhub")
    if ppg is not None:  vals["Polygon"] = ppg;  srcs.append("🟢Polygon")
    if pti is not None:  vals["Tiingo"]  = pti;  srcs.append("🟠Tiingo")
    if pfmp is not None: vals["FMP"]     = pfmp; srcs.append("🟤FMP")
    return tkr, vals, srcs

if use_external_prices and (alpha_ok or finnhub_ok or _env("POLYGON_API_KEY") or _env("TIINGO_API_KEY") or _env("FMP_API_KEY")):
    subset_idx = list(results.head(int(top_validate_k)).index)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = []
        for idx in subset_idx:
            r = results.loc[idx]
            futures.append(ex.submit(_fetch_external_for, r["Ticker"], float(r["Price_Yahoo"])))
        for f in as_completed(futures):
            try:
                tkr, vals, srcs = f.result()
            except Exception:
                continue
            idx = results.index[results["Ticker"] == tkr][0]
            prices = [v for v in vals.values() if v is not None]
            pmean  = float(np.mean(prices)) if prices else np.nan
            pstd   = float(np.std(prices))  if len(prices) > 1 else np.nan
            pa     = vals.get("Alpha", np.nan)
            pf     = vals.get("Finnhub", np.nan)
            results.loc[idx, ["Price_Alpha","Price_Finnhub","Price_Mean","Price_STD","Sources","Source_List"]] = \
                [pa, pf, pmean, pstd, len(prices), " · ".join(srcs) if srcs else "—"]
else:
    if use_external_prices:
        st.info("אימות חיצוני הופעל אך אין מפתחות תקינים. משתמש רק במחיר Yahoo.")
phase_times["מאמת מחירים"] = t_end(t0)

# Value check (אופציונלי) לטופ-K
if value_filter_enabled:
    t0 = t_start()
    try:
        for idx in results.head(int(top_validate_k)).index:
            tkr = results.at[idx, "Ticker"]
            # שימוש מהיר ב-FMP שכבר הבאנו (אם פונדמנטלי פעיל) – אחרת ratios קצר
            if pd.notna(results.at[idx, "PE_f"]):
                pe, pb = results.at[idx, "PE_f"], np.nan  # P/B לא הבאנו כאן; משמרים API-min
            else:
                key = _env("FMP_API_KEY")
                pe = pb = np.nan
                if key:
                    r = http_get_retry(f"https://financialmodelingprep.com/api/v3/ratios/{tkr}?limit=1&apikey={key}",
                                       tries=2, timeout=12, headers={"User-Agent":"StockScout/1.0"})
                    if r:
                        arr = r.json()
                        if isinstance(arr, list) and arr:
                            pe = float(arr[0].get("priceEarningsRatio", np.nan))
                            pb = float(arr[0].get("priceToBookRatio", np.nan))
            results.loc[idx, "PE"] = pe
            results.loc[idx, "PB"] = pb
            bad_val = ((isinstance(pe,(int,float)) and (pe < 5 or pe > 40)) or
                       (isinstance(pb,(int,float)) and pb >= 10))
            results.loc[idx, "Value_OK"] = (not bad_val)
            if bad_val and np.isfinite(results.loc[idx, "Score"]):
                results.loc[idx, "Score"] = max(0, results.loc[idx, "Score"] - 2.0)
    except Exception:
        pass
    phase_times["Value Check"] = t_end(t0)

# טווחי החזקה
def infer_horizon(row) -> str:
    rsi_v = row.get("RSI", np.nan); near  = row.get("Near52w", np.nan); score = row.get("Score", 0)
    if isinstance(near,(int,float)) and near >= 90 and isinstance(rsi_v,(int,float)) and 55 <= rsi_v <= 75:
        return "קצר (1–3 חודשים)"
    if isinstance(rsi_v,(int,float)) and rsi_v <= 40 and score >= 60:
        return "בינוני (3–6 חודשים)"
    if score >= 70:
        return "בינוני-ארוך (6–12 חודשים)"
    return "ארוך (6–12 חודשים)"
results["טווח החזקה"] = results.apply(infer_horizon, axis=1)

# מגבלת סקטור (גרידי) – אחרי ניקוד סופי
def apply_sector_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if not sector_cap_enabled: return df
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"
    counts = {}
    keep = []
    for _, r in df.iterrows():
        s = r.get("Sector", "Unknown") or "Unknown"
        counts[s] = counts.get(s, 0) + 1
        if counts[s] <= cap: keep.append(True)
        else: keep.append(False)
    return df[pd.Series(keep).values].reset_index(drop=True)

results = apply_sector_cap(results.sort_values(["Score","Ticker"], ascending=[False, True]).reset_index(drop=True),
                           int(sector_cap_max))

# ===== תצוגת מקור מחיר + חישובי כמות =====
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
results["Unit_Price"]   = np.where(results["מחיר ממוצע"].notna(), results["מחיר ממוצע"], results["Price_Yahoo"])
results["Unit_Price"]   = pd.to_numeric(results["Unit_Price"], errors="coerce")

# הקצאת תקציב עם תקרת פוזיציה
def allocate_budget(df: pd.DataFrame, total: float, min_pos: float, max_pos_pct: float) -> pd.DataFrame:
    df = df.copy()
    df["סכום קנייה ($)"] = 0.0
    if total <= 0 or df.empty: return df

    df = df.sort_values(["Score","Ticker"], ascending=[False, True]).reset_index(drop=True)
    remaining = float(total); n = len(df)
    max_pos_abs = (max_pos_pct/100.0) * total if max_pos_pct > 0 else float("inf")

    # הקצאת מינימום
    can_min = int(min(n, remaining // max(min_pos, 0.0))) if min_pos > 0 else 0
    if min_pos > 0 and can_min > 0:
        df.loc[:can_min-1, "סכום קנייה ($)"] = np.minimum(float(min_pos), max_pos_abs)
        remaining -= float(df.loc[:can_min-1, "סכום קנייה ($)"].sum())

    # שארית לפי משקולות (Score) תחת תקרה
    if remaining > 0:
        weights = df["Score"].clip(lower=0).to_numpy()
        extras = (remaining / n) * np.ones(n) if weights.sum() == 0 else remaining * (weights / weights.sum())
        df["סכום קנייה ($)"] = (df["סכום קנייה ($)"].to_numpy() + extras).clip(upper=max_pos_abs)

    # נורמליזציה קלה לשמירת סכום כולל
    s = df["סכום קנייה ($)"].sum()
    if s > 0 and abs(s - total)/max(total,1) > 1e-6:
        df["סכום קנייה ($)"] = df["סכום קנייה ($)"] * (total / s)

    df["סכום קנייה ($)"] = df["סכום קנייה ($)"].round(2)
    return df

TOPN = min(15, len(results))
results = allocate_budget(results.head(TOPN).reset_index(drop=True), budget_total, min_position, float(max_position_pct))

results["מניות לקנייה"] = np.floor(np.where(results["Unit_Price"] > 0, results["סכום קנייה ($)"] / results["Unit_Price"], 0)).astype(int)
results["עודף ($)"] = np.round(results["סכום קנייה ($)"] - results["מניות לקנייה"] * results["Unit_Price"], 2)

# KPI
budget_used = float(results["מניות לקנייה"].to_numpy() @ results["Unit_Price"].fillna(0).to_numpy())
kpi_cols = st.columns(4)
kpi_cols[0].metric("גודל יקום לאחר סינון היסטוריה", len(data_map))
kpi_cols[1].metric("כמות תוצאות אחרי סינון", len(results))
kpi_cols[2].metric("תקציב מנוצל (≈$)", f"{budget_used:,.0f}")
kpi_cols[3].metric("עודף תקציב (≈$)", f"{max(0.0, budget_total - budget_used):,.0f}")

# זמנים + Alpha counter
st.subheader("⏱️ זמני ביצוע")
times_df = pd.DataFrame([{"שלב": k, "משך (שניות)": round(v, 2)} for k, v in phase_times.items()])
st.table(times_df.style.set_properties(**{'text-align':'center','direction':'rtl'}))
if alpha_ok:
    st.caption(f"Alpha Vantage — קריאות בסשן זה: {int(st.session_state.av_calls)}  | טיפ: אל תחרוג מ~5 לדקה ו~500 ליום בגרסה החינמית.")

# ===== כרטיסי המלצה (עם ATR Stop/Target) =====
st.subheader("🤖 המלצה עכשיו")
st.caption("הכרטיסים הבאים הם **המלצות קנייה** בלבד. אין באמור ייעוץ השקעות.")
rec_df = results[results["סכום קנייה ($)"] > 0].copy()

if rec_df.empty:
    st.info("אין כרגע מניות שעוברות את הסף עם סכום קנייה חיובי. נסה להקל ספים ב'מתקדם'.")
else:
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
        rr         = r.get("RewardRisk", np.nan)
        atrp       = r.get("ATR_Price", np.nan)
        overx      = r.get("OverextRatio", np.nan)

        show_mean_fmt  = f"{np.round(show_mean, 2)}" if not np.isnan(show_mean) else "—"
        unit_price_fmt = f"{np.round(unit_price, 2)}" if not np.isnan(unit_price) else "—"
        rr_fmt   = f"{rr:.2f}R" if np.isfinite(rr) else "—"
        atrp_fmt = f"{atrp:.3f}" if np.isfinite(atrp) else "—"
        overx_fmt= f"{overx:.3f}" if np.isfinite(overx) else "—"

        esc = html_escape.escape
        ticker  = esc(str(r['Ticker']))
        sources = esc(str(sources))
        horizon = esc(str(horizon))

        card_html = f"""
        <div class="recommend-card" dir="rtl" style="text-align:right;background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)">
          <h3 style="display:flex;align-items:center;gap:10px;margin:0 0 6px 0">
            <span class="badge" style="display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;color:#1e293b;padding:2px 10px;border-radius:999px;font-weight:600">{ticker}</span>
            <span class="status-buy" style="background:#ecfdf5;border:1px solid #34d399;color:#065f46;padding:2px 10px;border-radius:999px;font-weight:600">סטטוס: קנייה</span>
          </h3>
          <div class="recommend-grid" style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:6px;font-size:.9rem;color:#222">
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
            <div><b>ATR/Price:</b> {atrp_fmt}</div>
            <div><b>Overextension (מעל MA_L):</b> {overx_fmt}</div>
            <div><b>Reward/Risk (≈R):</b> {rr_fmt}</div>
          </div>
        </div>
        """
        st_html(card_html, height=190, scrolling=False)


# ===== טבלה/CSV =====
st.subheader("🎯 תוצאות מסוננות ומדורגות")
view_df_source = rec_df if not rec_df.empty else results

hebrew_cols = {
    "Ticker":"טיקר","Price_Yahoo":"מחיר (Yahoo)","מחיר ממוצע":"מחיר ממוצע",
    "Unit_Price":"מחיר יחידה (חישוב)","סטיית תקן":"סטיית תקן","מקורות מחיר":"מקורות מחיר",
    "Score":"ניקוד","Score_Tech":"ניקוד טכני","Fundamental_S":"ציון פונדמנטלי","Sector":"סקטור",
    "RSI":"RSI","Near52w":"קרבה לשיא 52ש׳ (%)","Volx20d":"נפח/ממוצע 20 ימים",
    "OverextRatio":"Overextension מול MA_L","ATR_Price":"ATR/Price","RewardRisk":"Reward/Risk (≈R)",
    "Beta":"בטא","טווח החזקה":"טווח החזקה","סכום קנייה ($)":"סכום קנייה ($)",
    "מניות לקנייה":"מניות לקנייה","עודף ($)":"עודף ($)","PE":"P/E","PB":"P/B","Value_OK":"Value OK",
    "PE_f":"P/E (FMP)","PS_f":"P/S (FMP)","ROE_f":"ROE","ROIC_f":"ROIC","GM_f":"Gross Margin","DE_f":"Debt/Equity",
    "SurprisePct":"Earnings Surprise %"
}
show_order = [
    "טיקר","סקטור","מחיר (Yahoo)","מחיר ממוצע","מחיר יחידה (חישוב)","מקורות מחיר",
    "ניקוד","ניקוד טכני","ציון פונדמנטלי","RSI","קרבה לשיא 52ש׳ (%)","נפח/ממוצע 20 ימים",
    "Overextension מול MA_L","ATR/Price","Reward/Risk (≈R)","בטא",
    "טווח החזקה","סכום קנייה ($)","מניות לקנייה","עודף ($)","P/E (FMP)","P/S (FMP)","ROE","ROIC","Gross Margin","Debt/Equity","Earnings Surprise %"
]
csv_df = view_df_source.rename(columns=hebrew_cols)
csv_bytes = csv_df[[c for c in show_order if c in csv_df.columns]].to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ הורדת תוצאות ל-CSV", data=csv_bytes, file_name="stock_scout_results.csv", mime="text/csv")

st.markdown('<div class="rtl-table">', unsafe_allow_html=True)
st.dataframe(csv_df[[c for c in show_order if c in csv_df.columns]], use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# ===== גרפים =====
st.subheader("🔍 גרף לטיקר נבחר")
choice = st.selectbox("בחר טיקר לצפייה", ["(בחר)"] + view_df_source["Ticker"].astype(str).tolist())
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
                {"role":"user","content": f"הנה טבלת מניות: {csv_df.head(5).to_dict(orient='records')}"}
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
- נתונים היסטוריים: **Yahoo Finance** (`yfinance`) — ללא צורך במפתח.
- אימות מחירים (אם הופעל ויש מפתחות): **Alpha Vantage**, **Finnhub**, **Polygon**, **Tiingo**, **FMP**.
- ניקוד טכני: MA, מומנטום (1/3/6 חו׳, סיגמואיד), RSI, **Near-High Bell**, **Overextension מול MA ארוך**, **Pullback**,
  **ATR/Price**, **Reward/Risk** (אינפורמטיבי). אופציונלי: **MACD/ADX**.
- שכבה פונדמנטלית (אופציונלי): Growth/Quality/Valuation + בונוס Earnings Surprise — משוקלל לפי משקל שנבחר.
- סינוני איכות קשיחים: מינ׳ דולר-ווליום, מקס׳ ATR/Price, מקס׳ Overextension; **Earnings blackout** לטופ-K.
- כללי סיכון: מסנן בטא מול SPY/QQQ, מגבלת סקטור, תקרת גודל פוזיציה.
- כרטיסים כוללים Stop/Target לפי ATR כהכוונה בלבד. אין באמור ייעוץ השקעות.
""")
