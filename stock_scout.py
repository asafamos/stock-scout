# -*- coding: utf-8 -*-
"""
Asaf Stock Scout — 2025 (Auto Mode, Zero-Input) — FMP-free
-----------------------------------------------------------
• Technical score: MA, Momentum(1/3/6m), RSI band, Near-High bell, Overextension vs MA_L,
  Pullback window, ATR/Price, Reward/Risk, MACD/ADX.
• Fundamentals (Alpha OVERVIEW → Finnhub fallback): Growth (Rev/EPS YoY), Quality (ROE/Margin),
  Valuation (P/E,P/S), Debt/Equity penalty. (Surprise bonus off)
• Risk rules: earnings blackout, sector cap, beta vs SPY/QQQ, min dollar-volume, hard caps.
• External price verification: Alpha/Finnhub/Polygon/Tiingo (mean/std).
• Allocation: min position + max position % of budget.
• RTL UI, recommendation cards, CSV export, quick chart.
הערה: אין באמור ייעוץ השקעות.
"""

from __future__ import annotations
import os, time, warnings
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
import html as html_escape

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
CONFIG = dict(
    BUDGET_TOTAL=5000.0, MIN_POSITION=500.0, MAX_POSITION_PCT=15.0,
    UNIVERSE_LIMIT=350, LOOKBACK_DAYS=400, SMART_SCAN=True,
    MIN_PRICE=3.0, MIN_AVG_VOLUME=500_000, MIN_DOLLAR_VOLUME=5_000_000,
    MA_SHORT=20, MA_LONG=50, RSI_BOUNDS=(40,75), PULLBACK_RANGE=(0.85,0.97),
    OVEREXT_SOFT=0.20, OVEREXT_HARD=0.30, ATR_PRICE_HARD=0.08, USE_MACD_ADX=True,
    WEIGHTS=dict(ma=0.22,mom=0.30,rsi=0.12,near_high_bell=0.10,vol=0.08,overext=0.08,pullback=0.05,risk_reward=0.03,macd=0.01,adx=0.01),
    FUNDAMENTAL_ENABLED=True, FUNDAMENTAL_WEIGHT=0.15, FUNDAMENTAL_TOP_K=15, SURPRISE_BONUS_ON=False,
    EARNINGS_BLACKOUT_DAYS=7, EARNINGS_CHECK_TOPK=12, SECTOR_CAP_ENABLED=True, SECTOR_CAP_MAX=3,
    BETA_FILTER_ENABLED=True, BETA_BENCHMARK="SPY", BETA_MAX_ALLOWED=2.0, BETA_TOP_K=60,
    EXTERNAL_PRICE_VERIFY=True, TOP_VALIDATE_K=12,
    TOPN_RESULTS=15, TOPK_RECOMMEND=5
)

# ==================== ENV/Secrets ====================
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if "secrets" in dir(st) and key in st.secrets: return st.secrets[key]
    except Exception: pass
    return os.getenv(key, default)

load_dotenv(find_dotenv(usecwd=True))
for _extra in ["nev","stock_scout.nev",".env.local",".env.production"]:
    try:
        if os.path.exists(_extra): load_dotenv(_extra)
    except Exception: pass

# ==================== HTTP helpers ====================
def http_get_retry(url: str, tries: int = 2, backoff: float = 1.6, timeout: int = 8, headers: dict | None = None):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {})
            if r.status_code in (429,500,502,503,504):
                time.sleep(min(6, backoff**i)); continue
            return r
        except requests.RequestException:
            time.sleep(min(6, backoff**i))
    return None

def alpha_throttle(min_gap_seconds: float = 12.0):
    ts_key = "_alpha_last_call_ts"
    last = st.session_state.get(ts_key, 0.0); now = time.time()
    if now - last < min_gap_seconds: time.sleep(min_gap_seconds - (now - last))
    st.session_state[ts_key] = time.time()

# ==================== Connectivity checks ====================
@st.cache_data(ttl=300)
def _check_alpha():
    k = _env("ALPHA_VANTAGE_API_KEY")
    if not k: return False, "Missing API key"
    r = http_get_retry(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=MSFT&apikey={k}", tries=1, timeout=8)
    if not r: return False, "Timeout"
    try: j = r.json()
    except Exception: return False, "Bad JSON"
    if "Global Quote" in j: return True, "OK"
    return False, j.get("Note") or j.get("Information") or "Rate-limited"

@st.cache_data(ttl=300)
def _check_finnhub():
    k = _env("FINNHUB_API_KEY")
    if not k: return False, "Missing API key"
    r = http_get_retry(f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={k}", tries=1, timeout=6)
    if not r: return False, "Timeout"
    try: j = r.json()
    except Exception: return False, "Bad JSON"
    return ("c" in j), ("OK" if "c" in j else "Bad response")

@st.cache_data(ttl=300)
def _check_polygon():
    k = _env("POLYGON_API_KEY")
    if not k: return False, "Missing API key"
    r = http_get_retry(f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true&apiKey={k}", tries=1, timeout=6)
    if not r: return False, "Timeout"
    try: j = r.json()
    except Exception: return False, "Bad JSON"
    ok = bool(j.get("resultsCount",0)>0 and "results" in j)
    return ok, ("OK" if ok else "Bad response")

@st.cache_data(ttl=300)
def _check_tiingo():
    k = _env("TIINGO_API_KEY")
    if not k: return False, "Missing API key"
    r = http_get_retry(f"https://api.tiingo.com/tiingo/daily/AAPL/prices?token={k}&resampleFreq=daily", tries=1, timeout=6)
    if not r: return False, "Timeout"
    try: arr = r.json()
    except Exception: return False, "Bad JSON"
    ok = isinstance(arr,list) and arr and isinstance(arr[-1],dict) and ("close" in arr[-1])
    return ok, ("OK" if ok else "Bad response")

# ==================== Indicators ====================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = pd.to_numeric(series.squeeze(), errors="coerce")
    delta = series.diff(); gain = delta.clip(lower=0); loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain/(avg_loss+1e-9)
    return 100 - (100/(1+rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def macd_line(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow; macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = (high.diff()).clip(lower=0); minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0; minus_dm[minus_dm <= plus_dm] = 0
    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(period, min_periods=period).mean()
    plus_di = 100*(plus_dm.rolling(period, min_periods=period).mean()/(atr14+1e-9))
    minus_di= 100*(minus_dm.rolling(period, min_periods=period).mean()/(atr14+1e-9))
    dx = (100*(plus_di-minus_di).abs()/((plus_di+minus_di)+1e-9))
    return dx.rolling(period, min_periods=period).mean()

def _sigmoid(x, k=3.0):
    try: return 1.0/(1.0+np.exp(-k*x))
    except Exception: return 0.5

# ==================== Universe & data ====================
@st.cache_data(ttl=60*15)
def build_universe(limit: int = 350) -> List[str]:
    ok,_ = _check_finnhub()
    if not ok:
        return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","QCOM","ADBE","CRM","NFLX","INTC","ORCL","PANW","SNPS","CDNS","MU","KLAC"]
    key = _env("FINNHUB_API_KEY"); symbols: List[str] = []
    for mic in ("XNAS","XNYS"):
        r = http_get_retry(f"https://finnhub.io/api/v1/stock/symbol?exchange=US&mic={mic}&token={key}", tries=1, timeout=14)
        if not r: continue
        try:
            arr = r.json()
            for it in arr:
                s = it.get("symbol",""); typ = it.get("type","")
                if not s or "." in s: continue
                if typ and "Common Stock" not in typ: continue
                symbols.append(s)
        except Exception: continue
    symbols = sorted(pd.unique(pd.Series(symbols)))
    if not symbols: return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META"]
    if len(symbols) > limit:
        bins: Dict[str,List[str]] = {}
        for tkr in symbols: bins.setdefault(tkr[0],[]).append(tkr)
        per = max(1, int(limit/max(1,len(bins)))); sampled=[]
        for _,arr in sorted(bins.items()): sampled.extend(sorted(arr)[:per])
        if len(sampled)<limit: sampled.extend([t for t in symbols if t not in sampled][: (limit-len(sampled))])
        symbols = sampled
    return symbols[:limit]

def safe_yf_download(tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    out: Dict[str,pd.DataFrame] = {}
    if not tickers: return out
    try:
        data_raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker', threads=True)
        if isinstance(data_raw.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    df = data_raw[t].dropna()
                    if not df.empty: out[t]=df
                except Exception: continue
        else:
            df = data_raw.dropna()
            if not df.empty: out[tickers[0]] = df
    except Exception: pass
    missing = [t for t in tickers if t not in out]
    for t in missing:
        try:
            dfi = yf.download(t, start=start, end=end, auto_adjust=True, progress=False).dropna()
            if not dfi.empty: out[t]=dfi
        except Exception: continue
    return out

@st.cache_data(show_spinner=True, ttl=60*15)
def fetch_history_bulk(tickers: List[str], period_days: int, ma_long: int) -> Dict[str, pd.DataFrame]:
    end = datetime.utcnow(); start = end - timedelta(days=period_days)
    data = safe_yf_download(tickers, start, end)
    out={}; min_len=max(60, int(ma_long)+10)
    for t,df in data.items():
        try:
            if len(df)>=min_len: out[t]=df
        except Exception: continue
    return out

# ==================== Earnings ====================
@st.cache_data(ttl=60*60)
def get_next_earnings_date(ticker: str) -> Optional[datetime]:
    try:
        key=_env("FINNHUB_API_KEY")
        if key:
            today=datetime.utcnow().date()
            url=(f"https://finnhub.io/api/v1/calendar/earnings?from={today.isoformat()}&to={(today+timedelta(days=180)).isoformat()}&symbol={ticker}&token={key}")
            r=http_get_retry(url, tries=1, timeout=10)
            if r:
                data=r.json()
                for row in data.get("earningsCalendar",[]):
                    if row.get("symbol")==ticker and row.get("date"): return datetime.fromisoformat(row["date"])
    except Exception: pass
    try:
        ed=yf.Ticker(ticker).get_earnings_dates(limit=4)
        if isinstance(ed,pd.DataFrame) and not ed.empty:
            now=pd.Timestamp.utcnow(); future=ed[ed.index>=now]
            dt=(future.index.min() if not future.empty else ed.index.max())
            if pd.notna(dt): return dt.to_pydatetime()
    except Exception: pass
    try:
        cal=yf.Ticker(ticker).calendar
        if isinstance(cal,pd.DataFrame) and "Earnings Date" in cal.index:
            vals=cal.loc["Earnings Date"].values
            if len(vals)>0:
                dt=pd.to_datetime(str(vals[0]))
                if pd.notna(dt): return dt.to_pydatetime()
    except Exception: pass
    return None

@st.cache_data(ttl=60*30)
def _earnings_batch(symbols: List[str]) -> Dict[str, Optional[datetime]]:
    out={}
    if not symbols: return out
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(get_next_earnings_date, s): s for s in symbols}
        for f in as_completed(futs):
            s=futs[f]
            try: out[s]=f.result()
            except Exception: out[s]=None
    return out

# ==================== Fundamentals (Alpha → Finnhub) ====================
def _to_01(x, low, high):
    if not isinstance(x,(int,float)) or not np.isfinite(x): return np.nan
    return np.clip((x-low)/(high-low), 0, 1)

@st.cache_data(ttl=60*60)
def fetch_fundamentals_bundle(ticker: str) -> dict:
    out = {}

    # Alpha OVERVIEW – נשתמש רק אם ה־check יצא ירוק
    ak = _env("ALPHA_VANTAGE_API_KEY")
    use_alpha = bool(st.session_state.get("_alpha_ok")) and bool(ak)
    if use_alpha:
        try:
            alpha_throttle(2.0)
            r = http_get_retry(
                f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ak}",
                tries=1, timeout=6
            )
            if r:
                j = r.json()
                if isinstance(j, dict) and j.get("Symbol"):
                    def fnum(k):
                        try:
                            v = float(j.get(k, np.nan))
                            return v if np.isfinite(v) else np.nan
                        except Exception:
                            return np.nan

                    out["roe"]  = fnum("ReturnOnEquityTTM")
                    out["roic"] = np.nan
                    gp = fnum("GrossProfitTTM"); tr = fnum("TotalRevenueTTM")
                    gm_calc = (gp / tr) if (np.isfinite(gp) and np.isfinite(tr) and tr > 0) else np.nan
                    pm = fnum("ProfitMargin")
                    out["gm"] = gm_calc if np.isfinite(gm_calc) else pm
                    out["ps"] = fnum("PriceToSalesTTM")
                    out["pe"] = fnum("PERatio")
                    out["de"] = fnum("DebtToEquityTTM")
                    out["rev_g_yoy"] = fnum("QuarterlyRevenueGrowthYOY")
                    out["eps_g_yoy"] = fnum("QuarterlyEarningsGrowthYOY")
                    out["sector"] = j.get("Sector") or "Unknown"
                    return out
        except Exception:
            pass

    # Finnhub fallback
    fk = _env("FINNHUB_API_KEY")
    if fk:
        try:
            r = http_get_retry(
                f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={fk}",
                tries=1, timeout=10
            )
            if r:
                j = r.json()
                m = j.get("metric", {})

                def fget(*keys):
                    for k in keys:
                        v = m.get(k)
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            return float(v)
                    return np.nan

                out["roe"]  = fget("roeTtm", "roeAnnual")
                out["roic"] = np.nan
                out["gm"]   = fget("grossMarginTTM", "grossMarginAnnual")
                out["ps"]   = fget("psTTM", "priceToSalesTTM")
                out["pe"]   = fget("peBasicExclExtraTTM", "peNormalizedAnnual", "peTTM")

                de = np.nan
                try:
                    total_debt   = fget("totalDebt")
                    total_equity = fget("totalEquity")
                    if np.isfinite(total_debt) and np.isfinite(total_equity) and total_equity != 0:
                        de = total_debt / total_equity
                except Exception:
                    pass
                out["de"]        = de
                out["rev_g_yoy"] = fget("revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy")
                out["eps_g_yoy"] = fget("epsGrowthTTMYoy", "epsGrowthQuarterlyYoy")
                out["sector"]    = _finnhub_sector(ticker, fk)
                return out
        except Exception:
            pass

    return out

def fundamental_score(d: dict) -> float:
    g_rev=_to_01(d.get("rev_g_yoy",np.nan), 0.00, 0.30)
    g_eps=_to_01(d.get("eps_g_yoy",np.nan), 0.00, 0.30)
    growth=np.nanmean([g_rev,g_eps])
    q_roe=_to_01(d.get("roe",np.nan), 0.05, 0.25)
    q_roic=_to_01(d.get("roic",np.nan), 0.05, 0.20)
    q_gm=_to_01(d.get("gm",np.nan), 0.10, 0.60)
    quality=np.nanmean([q_roe,q_roic,q_gm])
    pe=d.get("pe",np.nan); ps=d.get("ps",np.nan)
    val_pe=np.nan if not np.isfinite(pe) else _to_01(40-np.clip(pe,0,40), 0, 40)
    val_ps=np.nan if not np.isfinite(ps) else _to_01(10-np.clip(ps,0,10), 0, 10)
    valuation=np.nanmean([val_pe,val_ps])
    penalty=0.0
    de=d.get("de",np.nan)
    if isinstance(de,(int,float)) and np.isfinite(de) and de>2.0: penalty+=0.15
    comp=np.nanmean([growth,quality,valuation]); comp=0.0 if not np.isfinite(comp) else float(np.clip(comp,0.0,1.0))
    if CONFIG["SURPRISE_BONUS_ON"]:
        surprise=d.get("surprise",np.nan); comp+= (0.05 if (isinstance(surprise,(int,float)) and surprise>=2.0) else 0.0)
    comp-=penalty
    return float(np.clip(comp,0.0,1.0))
  
def _finnhub_sector(ticker: str, token: str) -> str:
    """משיכת סקטור אמיתי מ-Finnhub (profile2), לא metricType הכללי."""
    r = http_get_retry(
        f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={token}",
        tries=1, timeout=8
    )
    if not r:
        return "Unknown"
    try:
        j = r.json()
        return j.get("finnhubIndustry") or j.get("sector") or "Unknown"
    except Exception:
        return "Unknown"

@st.cache_data(ttl=60*60)
def fetch_beta_vs_benchmark(ticker: str, bench: str = "SPY", days: int = 252) -> float:
    try:
        end=datetime.utcnow(); start=end - timedelta(days=days+30)
        df_t=yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df_b=yf.download(bench,  start=start, end=end, auto_adjust=True, progress=False)
        if df_t.empty or df_b.empty: return np.nan
        j=pd.concat([df_t["Close"].pct_change().dropna(), df_b["Close"].pct_change().dropna()], axis=1).dropna()
        j.columns = ["rt", "rb"]
        if len(j) < 40:
            return np.nan

        slope=np.polyfit(j["rb"].to_numpy(), j["rt"].to_numpy(), 1)[0]
        return float(slope)
    except Exception: return np.nan

# ==================== External Prices ====================
def get_alpha_price(ticker: str) -> float | None:
    k = _env("ALPHA_VANTAGE_API_KEY")
    if not k:
        return None
    if st.session_state.get("_alpha_ok", False):
        alpha_throttle()
    r = http_get_retry(
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={k}",
        tries=1, timeout=10
    )
    if not r: return None
    try:
        j=r.json()
        if "Global Quote" in j and "05. price" in j["Global Quote"]:
            return float(j["Global Quote"]["05. price"])
    except Exception: return None
    return None

def get_finnhub_price(ticker: str) -> float | None:
    k = _env("FINNHUB_API_KEY")
    if not k:
        return None
    r = http_get_retry(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={k}", tries=1, timeout=8)
    if not r:
        return None
    try:
        j = r.json()
        return float(j["c"]) if "c" in j else None
    except Exception:
        return None

def get_polygon_price(ticker: str) -> float | None:
    k = _env("POLYGON_API_KEY")
    if not k:
        return None
    r = http_get_retry(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={k}", tries=1, timeout=8)
    if not r:
        return None
    try:
        j = r.json()
        if j.get("resultsCount", 0) > 0 and "results" in j:
            return float(j["results"][0]["c"])
    except Exception:
        return None
    return None

def get_tiingo_price(ticker: str) -> float | None:
    k = _env("TIINGO_API_KEY")
    if not k:
        return None
    r = http_get_retry(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?token={k}&resampleFreq=daily", tries=1, timeout=8)
    if not r:
        return None
    try:
        arr = r.json()
        if isinstance(arr, list) and arr:
            return float(arr[-1].get("close", np.nan))
    except Exception:
        return None
    return None

# ==================== UI ====================
st.set_page_config(page_title="Asaf's Stock Scout — 2025", page_icon="📈", layout="wide")
st.markdown("""
<style>
body{direction:rtl}.block-container{padding-top:1rem;padding-bottom:2rem}
h1,h2,h3{text-align:right}[data-testid="stMarkdownContainer"], label{ text-align:right }
thead tr th{ text-align:right } .rtl-table table{ direction:rtl }
.rtl-table th,.rtl-table td{ text-align:right !important }
.badge{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;color:#1e293b;padding:2px 10px;border-radius:999px;font-weight:600}
.status-buy{background:#ecfdf5;border:1px solid #34d399;color:#065f46;padding:2px 10px;border-radius:999px;font-weight:600}
</style>
""", unsafe_allow_html=True)

# Secrets button
def _mask(s: Optional[str], show_last=4) -> str:
    if not s: return "—"
    s=str(s).strip();  return ("•"*(len(s)-show_last))+s[-show_last:] if len(s)>show_last else ("•"*(len(s)-1))+s[-1]

col_btn,_=st.columns([1,4])
with col_btn:
    if st.button("בדיקת סיקרטים 🔐"):
        st.info(
            f"Alpha: {_mask(_env('ALPHA_VANTAGE_API_KEY'))}\n\n"
            f"Finnhub: {_mask(_env('FINNHUB_API_KEY'))}\n\n"
            f"Polygon: {_mask(_env('POLYGON_API_KEY'))}\n\n"
            f"Tiingo: {_mask(_env('TIINGO_API_KEY'))}"
        )

st.title("📈 Stock Scout — 2025 (Auto)")

# Status table
alpha_ok, alpha_reason = _check_alpha()
finn_ok,  finnh_reason = _check_finnhub()
poly_ok,  poly_reason  = _check_polygon()
tiin_ok,  tiin_reason  = _check_tiingo()
st.session_state["_alpha_ok"]=bool(alpha_ok)
status_df = pd.DataFrame({
    "מקור":["Alpha Vantage","Finnhub","Polygon","Tiingo"],
    "סטטוס":["🟢" if alpha_ok else "🔴","🟢" if finn_ok else "🔴","🟢" if poly_ok else "🔴","🟢" if tiin_ok else "🔴"],
    "סיבה":[alpha_reason,finnh_reason,poly_reason,tiin_reason],
})
st.table(status_df.style.set_properties(**{'text-align':'center','direction':'rtl'}))

# timers
def t_start(): return time.perf_counter()
def t_end(t0): return time.perf_counter() - t0
phase_times: Dict[str,float] = {}
if "av_calls" not in st.session_state: st.session_state.av_calls = 0

# 1) Universe
t0=t_start()
universe = build_universe(limit=CONFIG["UNIVERSE_LIMIT"]) if CONFIG["SMART_SCAN"] else build_universe(limit=200)
phase_times["בונה יקום"]=t_end(t0)

# 2) History
t0=t_start()
data_map = fetch_history_bulk(universe, CONFIG["LOOKBACK_DAYS"], CONFIG["MA_LONG"])
phase_times["מוריד נתונים"]=t_end(t0)

# 3) Technical score + hard filters
t0=t_start()
W=CONFIG["WEIGHTS"]
def _normalize_weights(d: Dict[str,float]) -> Dict[str,float]:
    keys=["ma","mom","rsi","near_high_bell","vol","overext","pullback","risk_reward","macd","adx"]
    w={k: float(d.get(k,0.0)) for k in keys}; s=sum(max(0.0,v) for v in w.values())
    s=1.0 if s<=0 else s;  return {k: max(0.0,v)/s for k,v in w.items()}
W=_normalize_weights(W)

rows=[]; lo_rsi,hi_rsi=CONFIG["RSI_BOUNDS"]
for tkr, df in data_map.items():
    if df is None or df.empty: continue
    df=df.copy()
    df["MA_S"]=df["Close"].rolling(int(CONFIG["MA_SHORT"])).mean()
    df["MA_L"]=df["Close"].rolling(int(CONFIG["MA_LONG"])).mean()
    df["RSI"]=rsi(df["Close"],14); df["ATR14"]=atr(df,14); df["Vol20"]=df["Volume"].rolling(20).mean()

    if CONFIG["USE_MACD_ADX"]:
        m,ms,mh = macd_line(df["Close"]); df["MACD"],df["MACD_SIG"],df["MACD_HIST"]=m,ms,mh
        try:
            adx_val=adx(df,14); 
            if isinstance(adx_val,pd.DataFrame): adx_val=adx_val.iloc[:,0]
            adx_val=pd.to_numeric(adx_val, errors="coerce").reindex(df.index); df.loc[:,"ADX14"]=adx_val.values
        except Exception: df["ADX14"]=np.nan

    price=float(df["Close"].iloc[-1]); 
    if (not np.isfinite(price)) or (price<CONFIG["MIN_PRICE"]): continue

    last_ma_s=float(df["MA_S"].iloc[-1]); last_ma_l=float(df["MA_L"].iloc[-1])
    ma_ok=(float(price>last_ma_s)+float(last_ma_s>last_ma_l))/2.0 if (np.isfinite(last_ma_s) and np.isfinite(last_ma_l)) else 0.0

    rsi_val=float(df["RSI"].iloc[-1])
    if np.isfinite(rsi_val):
        if rsi_val<lo_rsi:   rsi_score=max(0.0, 1-(lo_rsi-rsi_val)/20)
        elif rsi_val>hi_rsi: rsi_score=max(0.0, 1-(rsi_val-hi_rsi)/20)
        else:                rsi_score=1.0
    else: rsi_score=0.0

    vol20=float(df["Vol20"].iloc[-1]); vol_today=float(df["Volume"].iloc[-1])
    if np.isfinite(vol20) and vol20<CONFIG["MIN_AVG_VOLUME"]: continue
    vol_ok=(min(2.0, vol_today/vol20)/2.0) if (np.isfinite(vol20) and vol20>0 and np.isfinite(vol_today)) else 0.0

    ret_1m=float(df["Close"].pct_change(21).iloc[-1])
    ret_3m=float(df["Close"].pct_change(63).iloc[-1])
    ret_6m=float(df["Close"].pct_change(126).iloc[-1])
    mom_score=float(_sigmoid(np.nanmean([ret_1m,ret_3m,ret_6m])))

    window_52w=min(len(df),252); hi_52w=float(df["Close"].tail(window_52w).max())
    if np.isfinite(hi_52w) and hi_52w>0:
        near_high_raw=1.0 - min(1.0, max(0.0,(hi_52w-price)/hi_52w))
        if   near_high_raw>=0.95: near_high_score=0.45
        elif 0.75<=near_high_raw<=0.90: near_high_score=1.00
        elif 0.90<near_high_raw<0.95:  near_high_score=0.75
        else:                           near_high_score=near_high_raw
    else: near_high_raw,near_high_score=np.nan,0.0

    if np.isfinite(last_ma_l) and last_ma_l>0:
        overext_ratio=max(0.0,(price-last_ma_l)/last_ma_l)
        overext_score=1.0 - min(1.0, overext_ratio/max(1e-6, CONFIG["OVEREXT_SOFT"]))
    else: overext_ratio,overext_score=np.nan,0.0

    ratio_to_high=price/hi_52w if (np.isfinite(hi_52w) and hi_52w>0) else np.nan
    if np.isfinite(ratio_to_high):
        lo,hi=CONFIG["PULLBACK_RANGE"]
        if lo<=ratio_to_high<=hi: pullback_score=1.0
        else:
            dist=min(abs(ratio_to_high-lo), abs(ratio_to_high-hi))
            pullback_score=max(0.0, 1.0 - dist*10)
    else: pullback_score=0.0

    atr14=float(df["ATR14"].iloc[-1])
    if np.isfinite(atr14) and price>0:
        vol_rel=atr14/price; volatility_score=1.0 - min(1.0, vol_rel/0.05)
    else: vol_rel,volatility_score=np.nan,0.0

    dollar_vol=(price*vol20) if (np.isfinite(price) and np.isfinite(vol20)) else 0.0
    if dollar_vol<CONFIG["MIN_DOLLAR_VOLUME"]: continue
    if np.isfinite(vol_rel) and vol_rel>CONFIG["ATR_PRICE_HARD"]: continue
    if np.isfinite(overext_ratio) and overext_ratio>CONFIG["OVEREXT_HARD"]: continue

    if np.isfinite(hi_52w) and np.isfinite(atr14) and atr14>0:
        reward_risk=max(0.0,(hi_52w-price)/atr14); rr_score=min(1.0, reward_risk/4.0)
    else: reward_risk,rr_score=np.nan,0.0

    macd_score=0.0; adx_score=0.0
    if CONFIG["USE_MACD_ADX"] and "MACD" in df.columns:
        macd_v=float(df["MACD"].iloc[-1]); macd_sig=float(df["MACD_SIG"].iloc[-1])
        macd_score=1.0 if macd_v>macd_sig else 0.0
    if CONFIG["USE_MACD_ADX"] and "ADX14" in df.columns:
        adx_v=float(df["ADX14"].iloc[-1]) if pd.notna(df["ADX14"].iloc[-1]) else np.nan
        adx_score=np.clip((adx_v-15)/20.0, 0.0, 1.0) if np.isfinite(adx_v) else 0.0

    score=(W["ma"]*ma_ok + W["mom"]*mom_score + W["rsi"]*rsi_score + W["near_high_bell"]*near_high_score +
           W["vol"]*(vol_ok if np.isfinite(vol_ok) else 0.0) + W["overext"]*overext_score +
           W["pullback"]*pullback_score + W["risk_reward"]*rr_score + W["macd"]*macd_score + W["adx"]*adx_score)

    rows.append({
        "Ticker":tkr, "Price_Yahoo":price, "Score_Tech":round(100*float(score),1),
        "RSI":round(rsi_val,1) if np.isfinite(rsi_val) else np.nan,
        "Near52w":round(near_high_raw*100,1) if np.isfinite(near_high_raw) else np.nan,
        "Volx20d":round(vol_today/vol20,2) if (np.isfinite(vol_today) and np.isfinite(vol20) and vol20>0) else np.nan,
        "OverextRatio":round(overext_ratio,3) if np.isfinite(overext_ratio) else np.nan,
        "ATR_Price":round(vol_rel,4) if np.isfinite(vol_rel) else np.nan,
        "RewardRisk":round(reward_risk,2) if np.isfinite(reward_risk) else np.nan,
        "ATR14":atr14
    })

results=pd.DataFrame(rows)
phase_times["מחשב ניקוד (טכני)"]=t_end(t0)
if results.empty:
    st.warning("אין תוצאות אחרי הסינון. ייתכן שהספים קשוחים מדי עבור היקום הנוכחי.")
    st.stop()

results=results.sort_values(["Score_Tech","Ticker"], ascending=[False,True]).reset_index(drop=True)

# 3a) Fundamentals (Top-K) + mix score
if CONFIG["FUNDAMENTAL_ENABLED"] and (alpha_ok or finn_ok):
    t0=t_start()
    take_k=int(min(CONFIG["FUNDAMENTAL_TOP_K"], len(results)))
    for c in ["Fundamental_S","Sector","PE_f","PS_f","ROE_f","ROIC_f","GM_f","DE_f","RevG_f","EPSG_f"]:
        results[c]=np.nan
    for idx in results.head(take_k).index:
        tkr=results.at[idx,"Ticker"]; d=fetch_fundamentals_bundle(tkr); fs=fundamental_score(d)
        results.loc[idx,"Fundamental_S"]=round(100*fs,1)
        results.loc[idx,"PE_f"]=d.get("pe",np.nan); results.loc[idx,"PS_f"]=d.get("ps",np.nan)
        results.loc[idx,"ROE_f"]=d.get("roe",np.nan); results.loc[idx,"ROIC_f"]=d.get("roic",np.nan)
        results.loc[idx,"GM_f"]=d.get("gm",np.nan); results.loc[idx,"DE_f"]=d.get("de",np.nan)
        results.loc[idx,"RevG_f"]=d.get("rev_g_yoy",np.nan); results.loc[idx,"EPSG_f"]=d.get("eps_g_yoy",np.nan)
        results.loc[idx,"Sector"]=(d.get("sector") or "Unknown")
    results["Score"]=results["Score_Tech"]
    results.loc[results.head(take_k).index,"Score"]=(
        (1-float(CONFIG["FUNDAMENTAL_WEIGHT"])) * results.loc[results.head(take_k).index,"Score_Tech"] +
        float(CONFIG["FUNDAMENTAL_WEIGHT"]) * results.loc[results.head(take_k).index,"Fundamental_S"].fillna(0)
    )
    results=results.sort_values(["Score","Ticker"], ascending=[False,True]).reset_index(drop=True)
    phase_times["פונדמנטלי (Alpha/Finnhub)"]=t_end(t0)
else:
    results["Score"]=results["Score_Tech"]

# Earnings blackout
if CONFIG["EARNINGS_BLACKOUT_DAYS"]>0:
    to_check_idx=list(results.head(int(CONFIG["EARNINGS_CHECK_TOPK"])).index)
    symbols=[results.at[i,"Ticker"] for i in to_check_idx]
    ed_map=_earnings_batch(symbols); now_utc=datetime.utcnow()
    keep_mask=np.ones(len(results), dtype=bool)
    for idx in to_check_idx:
        tkr=results.at[idx,"Ticker"]; dt_earn=ed_map.get(tkr)
        if dt_earn is None: continue
        gap_days=abs((dt_earn - now_utc).days)
        if gap_days <= int(CONFIG["EARNINGS_BLACKOUT_DAYS"]):
            keep_mask[idx]=False; results.at[idx,"EarningsNote"]=f"Excluded: earnings within {gap_days}d"
    results=results[keep_mask].reset_index(drop=True)
    if results.empty:
        st.warning("כל המועמדות בטופ-K נפסלו עקב חלון דו\"חות.")
        st.stop()

# 3b) Beta filter
if CONFIG["BETA_FILTER_ENABLED"]:
    t0=t_start()
    take_k_beta=int(min(CONFIG["BETA_TOP_K"], len(results)))
    for idx in results.head(take_k_beta).index:
        tkr=results.at[idx,"Ticker"]
        results.loc[idx,"Beta"]=fetch_beta_vs_benchmark(tkr, bench=CONFIG["BETA_BENCHMARK"], days=252)
    results=results[~((results["Beta"].notna()) & (results["Beta"]>float(CONFIG["BETA_MAX_ALLOWED"])))].reset_index(drop=True)
    phase_times["מסנן בטא"]=t_end(t0)

# External price verification (Top-K)
t0=t_start()
results["Price_Alpha"]=np.nan; results["Price_Finnhub"]=np.nan
results["Price_Mean"]=np.nan;  results["Price_STD"]=np.nan
results["Source_List"]="🟡Yahoo"

def _fetch_external_for(tkr: str, py: float) -> Tuple[str, Dict[str, Optional[float]], List[str]]:
    vals, srcs={}, []
    if np.isfinite(py): vals["Yahoo"]=float(py); srcs.append("🟡Yahoo")
    if alpha_ok:
        p=get_alpha_price(tkr);  (vals.setdefault("Alpha",p), srcs.append("🟣Alpha")) if p is not None else None
        if p is not None: st.session_state.av_calls = st.session_state.get("av_calls",0)+1
    if finn_ok:
        p=get_finnhub_price(tkr); (vals.setdefault("Finnhub",p), srcs.append("🔵Finnhub")) if p is not None else None
    if poly_ok and _env("POLYGON_API_KEY"):
        p=get_polygon_price(tkr); (vals.setdefault("Polygon",p), srcs.append("🟢Polygon")) if p is not None else None
    if tiin_ok and _env("TIINGO_API_KEY"):
        p=get_tiingo_price(tkr);  (vals.setdefault("Tiingo",p),  srcs.append("🟠Tiingo"))  if p is not None else None
    return tkr, vals, srcs

if CONFIG["EXTERNAL_PRICE_VERIFY"] and (alpha_ok or finn_ok or (poly_ok and _env("POLYGON_API_KEY")) or (tiin_ok and _env("TIINGO_API_KEY"))):
    subset_idx=list(results.head(int(CONFIG["TOP_VALIDATE_K"])).index)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures=[ex.submit(_fetch_external_for, results.loc[idx,"Ticker"], float(results.loc[idx,"Price_Yahoo"])) for idx in subset_idx]
        for f in as_completed(futures):
            try: tkr, vals, srcs=f.result()
            except Exception: continue
            idx=results.index[results["Ticker"]==tkr][0]
            prices=[v for v in vals.values() if v is not None]
            pmean=float(np.mean(prices)) if prices else np.nan
            pstd=float(np.std(prices)) if len(prices)>1 else np.nan
            results.loc[idx, ["Price_Alpha","Price_Finnhub","Price_Mean","Price_STD","Source_List"]] = \
                [vals.get("Alpha",np.nan), vals.get("Finnhub",np.nan), pmean, pstd, " · ".join(srcs)]
phase_times["מאמת מחירים"]=t_end(t0)

# Horizon heuristic
def infer_horizon(row)->str:
    rsi_v=row.get("RSI",np.nan); near=row.get("Near52w",np.nan); score=row.get("Score",0)
    if isinstance(near,(int,float)) and near>=90 and isinstance(rsi_v,(int,float)) and 55<=rsi_v<=75: return "קצר (1–3 חודשים)"
    if isinstance(rsi_v,(int,float)) and rsi_v<=40 and score>=60: return "בינוני (3–6 חודשים)"
    if score>=70: return "בינוני-ארוך (6–12 חודשים)"
    return "ארוך (6–12 חודשים)"
results["טווח החזקה"]=results.apply(infer_horizon, axis=1)

# Sector cap
def apply_sector_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if not CONFIG["SECTOR_CAP_ENABLED"]: return df
    if "Sector" not in df.columns: df["Sector"]="Unknown"
    counts={}; keep=[]
    for _,r in df.iterrows():
        s=r.get("Sector","Unknown") or "Unknown"
        counts[s]=counts.get(s,0)+1; keep.append(counts[s]<=cap)
    return df[pd.Series(keep).values].reset_index(drop=True)

results = apply_sector_cap(results.sort_values(["Score","Ticker"], ascending=[False,True]).reset_index(drop=True),
                           int(CONFIG["SECTOR_CAP_MAX"]))

# Source badges & unit price
def source_badges(row):
    s=row.get("Source_List"); return s if isinstance(s,str) and s else "🟡Yahoo"
results["מקורות מחיר"]=results.apply(source_badges, axis=1)
results["מחיר ממוצע"]=results["Price_Mean"].round(2)
results["סטיית תקן"]=results["Price_STD"].round(4)
results["Unit_Price"]=np.where(results["מחיר ממוצע"].notna(), results["מחיר ממוצע"], results["Price_Yahoo"])
results["Unit_Price"]=pd.to_numeric(results["Unit_Price"], errors="coerce")

# Allocation
def allocate_budget(df: pd.DataFrame, total: float, min_pos: float, max_pos_pct: float) -> pd.DataFrame:
    df=df.copy(); df["סכום קנייה ($)"]=0.0
    if total<=0 or df.empty: return df
    df=df.sort_values(["Score","Ticker"], ascending=[False,True]).reset_index(drop=True)
    remaining=float(total); n=len(df); max_pos_abs=(max_pos_pct/100.0)*total if max_pos_pct>0 else float("inf")
    if min_pos>0:
        can_min=int(min(n, remaining//min_pos))
        if can_min>0:
            base=np.full(can_min, min(min_pos, max_pos_abs), dtype=float); df.loc[:can_min-1,"סכום קנייה ($)"]=base; remaining-=float(base.sum())
    if remaining>0:
        weights=df["Score"].clip(lower=0).to_numpy(dtype=float)
        extras=np.full(n, remaining/n, dtype=float) if np.nansum(weights)<=0 else remaining*(np.nan_to_num(weights,nan=0.0)/np.nansum(weights))
        current=df["סכום קנייה ($)"].to_numpy(dtype=float); proposed=current+extras
        if np.isfinite(max_pos_abs): proposed=np.minimum(proposed, max_pos_abs)
        df["סכום קנייה ($)"]=proposed
    s=float(df["סכום קנייה ($)"].sum())
    if s>0 and abs(s-total)/max(total,1)>1e-6: df["סכום קנייה ($)"]=(df["סכום קנייה ($)"].to_numpy(dtype=float)*(total/s))
    df["סכום קנייה ($)"]=df["סכום קנייה ($)"].round(2); return df

TOPN=min(CONFIG["TOPN_RESULTS"], len(results))
results=allocate_budget(results.head(TOPN).reset_index(drop=True), CONFIG["BUDGET_TOTAL"], CONFIG["MIN_POSITION"], float(CONFIG["MAX_POSITION_PCT"]))
results["מניות לקנייה"]=np.floor(np.where(results["Unit_Price"]>0, results["סכום קנייה ($)"]/results["Unit_Price"], 0)).astype(int)
results["עודף ($)"]=np.round(results["סכום קנייה ($)"] - results["מניות לקנייה"]*results["Unit_Price"], 2)

# KPI
budget_used=float(results["מניות לקנייה"].to_numpy() @ results["Unit_Price"].fillna(0).to_numpy())
k0,k1,k2,k3=st.columns(4)
k0.metric("גודל יקום לאחר סינון היסטוריה", len(data_map))
k1.metric("כמות תוצאות אחרי סינון", len(results))
k2.metric("תקציב מנוצל (≈$)", f"{budget_used:,.0f}")
k3.metric("עודף תקציב (≈$)", f"{max(0.0, CONFIG['BUDGET_TOTAL'] - budget_used):,.0f}")

# Timings
st.subheader("⏱️ זמני ביצוע")
times_df=pd.DataFrame([{"שלב":k,"משך (שניות)":round(v,2)} for k,v in phase_times.items()])
st.table(times_df.style.set_properties(**{'text-align':'center','direction':'rtl'}))
if alpha_ok: st.caption(f"Alpha Vantage — קריאות בסשן זה: {int(st.session_state.get('av_calls',0))} (שמור על קצבים).")

# ==================== Recommendation Cards ====================
st.subheader("🤖 המלצה עכשיו")
st.caption("הכרטיסים הבאים הם **המלצות קנייה** בלבד. אין באמור ייעוץ השקעות.")
rec_df=results[results["סכום קנייה ($)"]>0].copy()

CARD_CSS="""
<style>
.card{direction:rtl;text-align:right;background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;
      padding:14px 16px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,.05);font-family:system-ui,-apple-system}
.badge{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;color:#1e293b;padding:2px 10px;border-radius:999px;font-weight:700}
.status-buy{display:inline-block;background:#ecfdf5;border:1px solid #34d399;color:#065f46;padding:2px 10px;border-radius:999px;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:6px;font-size:.92rem;color:#222}
.item b{color:#111}
@media(max-width:1100px){ .grid{grid-template-columns:repeat(2,minmax(0,1fr));} }
</style>
"""

if rec_df.empty:
    st.info("אין כרגע מניות שעוברות את הסף עם סכום קנייה חיובי.")
else:
    for _,r in rec_df.head(CONFIG["TOPK_RECOMMEND"]).iterrows():
        mean=r.get("מחיר ממוצע",np.nan); std=r.get("סטיית תקן",np.nan)
        show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
        show_std  = std if not np.isnan(std) else "—"
        sources   = r.get("מקורות מחיר","—"); buy_amt=float(r.get("סכום קנייה ($)",0.0))
        horizon   = r.get("טווח החזקה","—"); rsi_v=r.get("RSI",np.nan); near52=r.get("Near52w",np.nan)
        score     = r.get("Score",0); unit_price=r.get("Unit_Price",np.nan)
        shares    = int(r.get("מניות לקנייה",0)); leftover=r.get("עודף ($)",0.0)
        rr        = r.get("RewardRisk",np.nan); atrp=r.get("ATR_Price",np.nan); overx=r.get("OverextRatio",np.nan)

        show_mean_fmt=f"{np.round(show_mean,2)}" if not np.isnan(show_mean) else "—"
        unit_price_fmt=f"{np.round(unit_price,2)}" if not np.isnan(unit_price) else "—"
        rr_fmt=f"{rr:.2f}R" if np.isfinite(rr) else "—"
        atrp_fmt=f"{atrp:.3f}" if np.isfinite(atrp) else "—"
        overx_fmt=f"{overx:.3f}" if np.isfinite(overx) else "—"

        esc=html_escape.escape; ticker=esc(str(r['Ticker'])); sources_esc=esc(str(sources))
        card_html=f"""{CARD_CSS}
<div class="card">
  <h3 style="display:flex;align-items:center;gap:10px;margin:0 0 6px 0">
    <span class="badge">{ticker}</span>
    <span class="status-buy">סטטוס: קנייה</span>
  </h3>
  <div class="grid">
    <div class="item"><b>מחיר ממוצע:</b> {show_mean_fmt}</div>
    <div class="item"><b>סטיית תקן:</b> {show_std}</div>
    <div class="item"><b>RSI:</b> {rsi_v if not np.isnan(rsi_v) else '—'}</div>
    <div class="item"><b>קרבה לשיא 52ש׳:</b> {near52 if not np.isnan(near52) else '—'}%</div>
    <div class="item"><b>ניקוד:</b> {int(round(score))}</div>
    <div class="item"><b>מקורות:</b> {sources_esc.replace(' · ','&nbsp;•&nbsp;')}</div>
    <div class="item"><b>סכום קנייה מומלץ:</b> ${buy_amt:,.0f}</div>
    <div class="item"><b>טווח החזקה:</b> {horizon}</div>
    <div class="item"><b>מחיר יחידה:</b> {unit_price_fmt}</div>
    <div class="item"><b>מניות לקנייה:</b> {shares}</div>
    <div class="item"><b>עודף לא מנוצל:</b> ${leftover:,.2f}</div>
    <div class="item"><b>ATR/Price:</b> {atrp_fmt}</div>
    <div class="item"><b>Overextension:</b> {overx_fmt}</div>
    <div class="item"><b>Reward/Risk (≈R):</b> {rr_fmt}</div>
  </div>
</div>
"""
        st_html(card_html, height=210, scrolling=False)

# ==================== Results table + CSV ====================
st.subheader("🎯 תוצאות מסוננות ומדורגות")
view_df_source = rec_df if not rec_df.empty else results

hebrew_cols = {
    "Ticker":"טיקר","Price_Yahoo":"מחיר (Yahoo)","מחיר ממוצע":"מחיר ממוצע","Unit_Price":"מחיר יחידה (חישוב)",
    "סטיית תקן":"סטיית תקן","מקורות מחיר":"מקורות מחיר","Score":"ניקוד","Score_Tech":"ניקוד טכני","Fundamental_S":"ציון פונדמנטלי",
    "Sector":"סקטור","RSI":"RSI","Near52w":"קרבה לשיא 52ש׳ (%)","Volx20d":"נפח/ממוצע 20 ימים",
    "OverextRatio":"Overextension מול MA_L","ATR_Price":"ATR/Price","RewardRisk":"Reward/Risk (≈R)","Beta":"בטא",
    "טווח החזקה":"טווח החזקה","סכום קנייה ($)":"סכום קנייה ($)","מניות לקנייה":"מניות לקנייה","עודף ($)":"עודף ($)",
    "PE_f":"P/E","PS_f":"P/S","ROE_f":"ROE","ROIC_f":"ROIC","GM_f":"Margin","DE_f":"Debt/Equity",
    "RevG_f":"Revenue YoY","EPSG_f":"EPS YoY"
}
show_order = [
    "טיקר","סקטור","מחיר (Yahoo)","מחיר ממוצע","מחיר יחידה (חישוב)","מקורות מחיר",
    "ניקוד","ניקוד טכני","ציון פונדמנטלי","RSI","קרבה לשיא 52ש׳ (%)","נפח/ממוצע 20 ימים",
    "Overextension מול MA_L","ATR/Price","Reward/Risk (≈R)","בטא",
    "טווח החזקה","סכום קנייה ($)","מניות לקנייה","עודף ($)",
    "P/E","P/S","ROE","ROIC","Margin","Debt/Equity","Revenue YoY","EPS YoY"
]
csv_df = view_df_source.rename(columns=hebrew_cols)
csv_bytes = csv_df[[c for c in show_order if c in csv_df.columns]].to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ הורדת תוצאות ל-CSV", data=csv_bytes, file_name="stock_scout_results.csv", mime="text/csv")

st.markdown('<div class="rtl-table">', unsafe_allow_html=True)
st.dataframe(csv_df[[c for c in show_order if c in csv_df.columns]], use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==================== Quick chart ====================
st.subheader("🔍 גרף טיקר מהתוצאות")
choices = ["(בחר)"] + view_df_source["Ticker"].astype(str).tolist()
choice = st.selectbox("בחר טיקר", choices, index=0)
if choice and choice != "(בחר)" and choice in data_map:
    dfv = data_map[choice].copy()
    dfv["MA_S"] = dfv["Close"].rolling(int(CONFIG["MA_SHORT"])).mean()
    dfv["MA_L"] = dfv["Close"].rolling(int(CONFIG["MA_LONG"])).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfv.index, open=dfv['Open'], high=dfv['High'], low=dfv['Low'], close=dfv['Close'], name='מחיר'))
    fig.add_trace(go.Scatter(x=dfv.index, y=dfv['MA_S'], mode='lines', name=f"MA{int(CONFIG['MA_SHORT'])}"))
    fig.add_trace(go.Scatter(x=dfv.index, y=dfv['MA_L'], mode='lines', name=f"MA{int(CONFIG['MA_LONG'])}"))
    fig.update_layout(height=480, xaxis_rangeslider_visible=False, legend_orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    dfv["RSI"] = rsi(dfv["Close"], 14)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dfv.index, y=dfv['RSI'], mode='lines', name='RSI14'))
    fig2.add_hrect(y0=CONFIG["RSI_BOUNDS"][0], y1=CONFIG["RSI_BOUNDS"][1], fillcolor='LightGreen', opacity=0.2, line_width=0)
    fig2.update_layout(height=220, legend_orientation='h')
    st.plotly_chart(fig2, use_container_width=True)

# ==================== Notes ====================
with st.expander("ℹ️ מתודולוגיה (תקציר)"):
    st.markdown("""
- היסטוריה: **Yahoo Finance** (`yfinance`). אימות מחירים: **Alpha**, **Finnhub**, **Polygon**, **Tiingo**.
- ניקוד טכני: MA, מומנטום 1/3/6 חו׳ (Sigmoid), RSI בטווח, **Near-High bell**, **Overextension מול MA_L**, **Pullback**,
  **ATR/Price**, **Reward/Risk**, **MACD/ADX**.
- פונדמנטלי: **Alpha OVERVIEW** עם **Fallback ל-Finnhub** (Growth/Quality/Valuation + ענישת מינוף גבוה).
- חוקים: מינ׳ דולר-ווליום, תקרות ATR/Price ו-Overextension (פסילות קשיחות), **earnings blackout**, **beta filter**, **sector cap**,
  והקצאת תקציב עם תקרת פוזיציה מקסימלית.
""")
