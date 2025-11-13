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
import os
import time
import logging
import warnings
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
from core.portfolio import _normalize_weights
from advanced_filters import (
    compute_advanced_score, 
    should_reject_ticker, 
    fetch_benchmark_data
)

# Core modules
from core.logging_config import setup_logging, get_logger
from core.config import get_config as get_core_config, get_api_keys
from core.scoring.fundamental import compute_fundamental_score_with_breakdown
from core.classification import apply_classification, filter_core_recommendations

warnings.filterwarnings("ignore")

# Setup logging
setup_logging(level=logging.INFO)
logger = get_logger("stock_scout")

# ==================== CONFIG ====================
CONFIG = dict(
    BUDGET_TOTAL=1000.0,
    MIN_POSITION=500.0,
    MAX_POSITION_PCT=15.0,
    UNIVERSE_LIMIT=350,
    LOOKBACK_DAYS=400,
    SMART_SCAN=True,
    MIN_PRICE=3.0,
    MIN_AVG_VOLUME=500_000,
    MIN_DOLLAR_VOLUME=5_000_000,
    MA_SHORT=20,
    MA_LONG=50,
    RSI_BOUNDS=(40, 75),
    PULLBACK_RANGE=(0.85, 0.97),
    OVEREXT_SOFT=0.20,
    OVEREXT_HARD=0.30,
    ATR_PRICE_HARD=0.08,
    USE_MACD_ADX=True,
    WEIGHTS=dict(
        ma=0.22,
        mom=0.30,
        rsi=0.12,
        near_high_bell=0.10,
        vol=0.08,
        overext=0.08,
        pullback=0.05,
        risk_reward=0.03,
        macd=0.01,
        adx=0.01,
    ),
    FUNDAMENTAL_ENABLED=True,
    FUNDAMENTAL_WEIGHT=0.15,
    FUNDAMENTAL_TOP_K=15,
    SURPRISE_BONUS_ON=False,
    EARNINGS_BLACKOUT_DAYS=7,
    EARNINGS_CHECK_TOPK=12,
    SECTOR_CAP_ENABLED=True,
    SECTOR_CAP_MAX=3,
    BETA_FILTER_ENABLED=True,
    BETA_BENCHMARK="SPY",
    BETA_MAX_ALLOWED=2.0,
    BETA_TOP_K=60,
    EXTERNAL_PRICE_VERIFY=True,
    TOP_VALIDATE_K=30,
    TOPN_RESULTS=15,
    TOPK_RECOMMEND=5,
    # ==================== CORE RECOMMENDATION FILTERS ====================
    # These constants control the final filtering for "Core" stock recommendations.
    # Relaxed slightly from previous hard-coded values to yield 3-7 high-quality stocks per run.
    MIN_QUALITY_SCORE_CORE=27.0,      # Minimum fundamental quality score (out of 50). Was ~33, now 27.
    MAX_OVEREXTENSION_CORE=0.10,      # Max allowed overextension ratio. Was ~0.05, now 0.10.
    MAX_ATR_PRICE_CORE=0.08,          # Max ATR/Price ratio (volatility). Was ~0.05, now 0.08.
    RSI_MIN_CORE=45,                  # Minimum RSI for Core stocks. Was ~50, now 45.
    RSI_MAX_CORE=70,                  # Maximum RSI for Core stocks. Was ~75, now 70.
    MIN_RR_CORE=1.5,                  # Minimum Reward/Risk ratio. Was higher or not explicit, now 1.5.
    TARGET_RECOMMENDATIONS_MIN=3,     # Warn if fewer than this many stocks pass filters
    TARGET_RECOMMENDATIONS_MAX=7,     # Show only top N if more than this pass filters
    ENABLE_SIMFIN=True,               # Attempt SimFin fundamentals if API key present
    COVERAGE_WARN_THRESHOLD=0.4,      # Warn if <40% of tickers have ≥3 fundamental fields
    # ENABLE_IEX removed (IEX service no longer available)
    ENABLE_MARKETSTACK=True,          # Marketstack price source (EOD latest)
    ENABLE_NASDAQ_DL=True,            # Nasdaq Data Link price source (experimental)
    ENABLE_EODHD=True,                # EODHD price + fundamentals source
)
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

# ==================== HTTP helpers ====================
import time
import random
import logging
import requests
from typing import Optional

_log = logging.getLogger(__name__)

def http_get_retry(
    url: str,
    tries: int = 4,
    timeout: float = 8.0,
    headers: Optional[dict] = None,
    session: Optional[requests.Session] = None,
    backoff_base: float = 0.5,
    max_backoff: float = 10.0,
) -> Optional[requests.Response]:
    """
    HTTP GET with exponential backoff + full jitter.

    - tries: total attempts (including the first).
    - timeout: requests timeout for each attempt (seconds).
    - session: optional requests.Session to reuse connections.
    - returns requests.Response on status_code == 200, otherwise None after all retries.
    """
    sess = session or requests
    for attempt in range(1, max(1, tries) + 1):
        try:
            resp = sess.get(url, timeout=timeout, headers=headers)
            if resp is not None and resp.status_code == 200:
                return resp
            # treat 429 / 5xx as retryable too
            if resp is not None and resp.status_code in (429,) or (500 <= getattr(resp, "status_code", 0) < 600):
                _log.debug("HTTP %s -> retry attempt %d/%d for %s", getattr(resp, "status_code", None), attempt, tries, url)
            else:
                # non-retryable code (e.g., 400). Return response so caller can inspect.
                return resp
        except requests.RequestException as exc:
            _log.debug("Request exception on attempt %d/%d for %s: %s", attempt, tries, url, exc)

        # if we'll retry, sleep with full jitter
        if attempt < tries:
            backoff = min(max_backoff, backoff_base * (2 ** (attempt - 1)))
            sleep = random.uniform(0, backoff)
            time.sleep(sleep)
    _log.warning("All %d attempts failed for %s", tries, url)
    return None


def alpha_throttle(min_gap_seconds: float = 12.0) -> None:
    ts_key = "_alpha_last_call_ts"
    last = st.session_state.get(ts_key, 0.0)
    now = time.time()
    if now - last < min_gap_seconds:
        time.sleep(min_gap_seconds - (now - last))
    st.session_state[ts_key] = time.time()


# ==================== Connectivity checks ====================
@st.cache_data(ttl=300)
def _check_alpha() -> Tuple[bool, str]:
    k = _env("ALPHA_VANTAGE_API_KEY")
    if not k:
        return False, "Missing API key"
    r = http_get_retry(
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=MSFT&apikey={k}",
        tries=1,
        timeout=8,
    )
    if not r:
        return False, "Timeout"
    try:
        j = r.json()
    except Exception:
        return False, "Bad JSON"
    if "Global Quote" in j:
        return True, "OK"
    return False, j.get("Note") or j.get("Information") or "Rate-limited"


@st.cache_data(ttl=300)
def _check_finnhub() -> Tuple[bool, str]:
    k = _env("FINNHUB_API_KEY")
    if not k:
        return False, "Missing API key"
    r = http_get_retry(
        f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={k}", tries=1, timeout=6
    )
    if not r:
        return False, "Timeout"
    try:
        j = r.json()
    except Exception:
        return False, "Bad JSON"
    return ("c" in j), ("OK" if "c" in j else "Bad response")


@st.cache_data(ttl=300)
def _check_polygon() -> Tuple[bool, str]:
    k = _env("POLYGON_API_KEY")
    if not k:
        return False, "Missing API key"
    r = http_get_retry(
        f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true&apiKey={k}",
        tries=1,
        timeout=6,
    )
    if not r:
        return False, "Timeout"
    try:
        j = r.json()
    except Exception:
        return False, "Bad JSON"
    ok = bool(j.get("resultsCount", 0) > 0 and "results" in j)
    return ok, ("OK" if ok else "Bad response")


@st.cache_data(ttl=300)
def _check_tiingo() -> Tuple[bool, str]:
    k = _env("TIINGO_API_KEY")
    if not k:
        return False, "Missing API key"
    r = http_get_retry(
        f"https://api.tiingo.com/tiingo/daily/AAPL/prices?token={k}&resampleFreq=daily",
        tries=1,
        timeout=6,
    )
    if not r:
        return False, "Timeout"
    try:
        arr = r.json()
    except Exception:
        return False, "Bad JSON"
    ok = (
        isinstance(arr, list)
        and arr
        and isinstance(arr[-1], dict)
        and ("close" in arr[-1])
    )
    return ok, ("OK" if ok else "Bad response")


@st.cache_data(ttl=300)
def _check_fmp() -> Tuple[bool, str]:
    k = _env("FMP_API_KEY")
    if not k:
        return False, "Missing API key"
    r = http_get_retry(
        f"https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey={k}",
        tries=3,
        timeout=12.0,
    )
    if not r:
        return False, "Timeout"
    try:
        j = r.json()
    except Exception:
        return False, "Bad JSON"
    # Check for FMP error responses
    if isinstance(j, dict):
        if "Error Message" in j:
            return False, j["Error Message"]
        if "error" in j:
            return False, str(j.get("error", "Unknown error"))
    ok = isinstance(j, list) and j and isinstance(j[0], dict) and "symbol" in j[0]
    return ok, ("OK" if ok else "Bad response")


# ==================== Indicators ====================
from indicators import rsi, atr, macd_line, adx, _sigmoid


# ==================== Universe & data ====================
@st.cache_data(ttl=60 * 15)
def build_universe(limit: int = 350) -> List[str]:
    ok, _ = _check_finnhub()
    if not ok:
        return [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "AVGO",
            "AMD",
            "QCOM",
            "ADBE",
            "CRM",
            "NFLX",
            "INTC",
            "ORCL",
            "PANW",
            "SNPS",
            "CDNS",
            "MU",
            "KLAC",
        ]
    key = _env("FINNHUB_API_KEY")
    symbols: List[str] = []
    for mic in ("XNAS", "XNYS"):
        r = http_get_retry(
            f"https://finnhub.io/api/v1/stock/symbol?exchange=US&mic={mic}&token={key}",
            tries=1,
            timeout=14,
        )
        if not r:
            continue
        try:
            arr = r.json()
            for it in arr:
                s = it.get("symbol", "")
                typ = it.get("type", "")
                if not s or "." in s:
                    continue
                if typ and "Common Stock" not in typ:
                    continue
                symbols.append(s)
        except Exception:
            continue
    symbols = sorted(pd.unique(pd.Series(symbols)))
    if not symbols:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
    if len(symbols) > limit:
        bins: Dict[str, List[str]] = {}
        for tkr in symbols:
            bins.setdefault(tkr[0], []).append(tkr)
        per = max(1, int(limit / max(1, len(bins))))
        sampled: List[str] = []
        for _, arr in sorted(bins.items()):
            sampled.extend(sorted(arr)[:per])
        if len(sampled) < limit:
            sampled.extend(
                [t for t in symbols if t not in sampled][: (limit - len(sampled))]
            )
        symbols = sampled
    return symbols[:limit]


def safe_yf_download(
    tickers: List[str], start: datetime, end: datetime
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    try:
        data_raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if isinstance(data_raw.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    df = data_raw[t].dropna()
                    if not df.empty:
                        out[t] = df
                except Exception:
                    continue
        else:
            df = data_raw.dropna()
            if not df.empty:
                out[tickers[0]] = df
    except Exception:
        pass
    missing = [t for t in tickers if t not in out]
    for t in missing:
        try:
            dfi = yf.download(
                t, start=start, end=end, auto_adjust=True, progress=False
            ).dropna()
            if not dfi.empty:
                out[t] = dfi
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=True, ttl=60 * 60 * 4)  # 4 hours - history changes slowly
def fetch_history_bulk(
    tickers: List[str], period_days: int, ma_long: int
) -> Dict[str, pd.DataFrame]:
    end = datetime.utcnow()
    start = end - timedelta(days=period_days)
    data = safe_yf_download(tickers, start, end)
    out: Dict[str, pd.DataFrame] = {}
    min_len = max(60, int(ma_long) + 10)
    for t, df in data.items():
        try:
            if len(df) >= min_len:
                out[t] = df
        except Exception:
            continue
    return out


# ==================== Earnings ====================
@st.cache_data(ttl=60 * 60)
def get_next_earnings_date(ticker: str) -> Optional[datetime]:
    try:
        key = _env("FINNHUB_API_KEY")
        if key:
            today = datetime.utcnow().date()
            url = (
                f"https://finnhub.io/api/v1/calendar/earnings?from={today.isoformat()}"
                f"&to={(today + timedelta(days=180)).isoformat()}&symbol={ticker}&token={key}"
            )
            r = http_get_retry(url, tries=1, timeout=10)
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
            dt = future.index.min() if not future.empty else ed.index.max()
            if pd.notna(dt):
                return dt.to_pydatetime()
    except Exception:
        pass

    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            vals = cal.loc["Earnings Date"].values
            if len(vals) > 0:
                dt = pd.to_datetime(str(vals[0]))
                if pd.notna(dt):
                    return dt.to_pydatetime()
    except Exception:
        pass

    return None


@st.cache_data(ttl=60 * 30)
def _earnings_batch(symbols: List[str]) -> Dict[str, Optional[datetime]]:
    out: Dict[str, Optional[datetime]] = {}
    if not symbols:
        return out
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(get_next_earnings_date, s): s for s in symbols}
        for f in as_completed(futs):
            s = futs[f]
            try:
                out[s] = f.result()
            except Exception:
                out[s] = None
    return out


# ==================== Fundamentals (Alpha → Finnhub) ====================
def _to_01(x: float, low: float, high: float) -> float:
    if not isinstance(x, (int, float)) or not np.isfinite(x):
        return np.nan
    return np.clip((x - low) / (high - low), 0, 1)


@st.cache_data(ttl=60 * 60 * 24)  # 24h cache for fundamentals
def fetch_fundamentals_bundle(ticker: str, enable_alpha_smart: bool = False) -> dict:
    """משיכת פונדמנטלים ממספר מקורות ומיזוג לקובץ אחד - **במקביל!**

    במקום sequential (איטי), מריץ את כל המקורות בו-זמנית עם ThreadPoolExecutor.
    חיסכון: 60-70% מזמן הריצה!
    
    Priority merge: FMP → SimFin → Alpha (smart) → Finnhub → EODHD → Tiingo
    enable_alpha_smart: אם True, יפעיל Alpha Vantage (מומלץ רק ל-top picks)

    החזר: dict עם השדות הרלוונטיים + דגלי מקור + _sources (attribution) + Fund_Coverage_Pct.
    """
    merged: dict = {
        "roe": np.nan,
        "roic": np.nan,
        "gm": np.nan,
        "ps": np.nan,
        "pe": np.nan,
        "de": np.nan,
        "rev_g_yoy": np.nan,
        "eps_g_yoy": np.nan,
        "sector": "Unknown",
        "from_fmp": False,
        "from_fmp_full": False,
        "from_alpha": False,
        "from_finnhub": False,
        "from_simfin": False,
        "from_eodhd": False,
        "from_tiingo": False,
        "_sources_used": [],
        "_sources": {},  # Track which provider gave which field
    }

    def _merge(src: dict, flag: str, source_name: str):
        if not src:
            return
        for k in ["roe","roic","gm","ps","pe","de","rev_g_yoy","eps_g_yoy"]:
            v_cur = merged.get(k, np.nan)
            v_new = src.get(k, np.nan)
            if (not np.isfinite(v_cur)) and isinstance(v_new, (int,float)) and np.isfinite(v_new):
                merged[k] = float(v_new)
                merged["_sources"][k] = source_name  # Attribution!
        # sector preference: keep first non-Unknown
        if merged.get("sector", "Unknown") == "Unknown":
            sec = src.get("sector", "Unknown")
            if isinstance(sec, str) and sec:
                merged["sector"] = sec
                merged["_sources"]["sector"] = source_name
        merged[flag] = True
        merged["_sources_used"].append(flag)

    # ========== PARALLEL FETCHING ==========
    with ThreadPoolExecutor(max_workers=7) as ex:
        futures = {}
        
        # FMP (2 endpoints)
        fmp_key = _env("FMP_API_KEY")
        if fmp_key:
            futures['fmp_full'] = ex.submit(_fmp_full_bundle_fetch, ticker, fmp_key)
            futures['fmp_legacy'] = ex.submit(_fmp_metrics_fetch, ticker, fmp_key)
        
        # SimFin
        if CONFIG.get("ENABLE_SIMFIN"):
            sim_key = _env("SIMFIN_API_KEY")
            if sim_key:
                futures['simfin'] = ex.submit(_simfin_fetch, ticker, sim_key)
        
        # Alpha Vantage (smart - only if enabled)
        if enable_alpha_smart and bool(st.session_state.get("_alpha_ok")) and bool(_env("ALPHA_VANTAGE_API_KEY")):
            futures['alpha'] = ex.submit(_alpha_overview_fetch, ticker)
        
        # Finnhub
        futures['finnhub'] = ex.submit(_finnhub_metrics_fetch, ticker)
        
        # EODHD
        if CONFIG.get("ENABLE_EODHD"):
            ek = _env("EODHD_API_KEY") or _env("EODHD_TOKEN")
            if ek:
                futures['eodhd'] = ex.submit(_eodhd_fetch_fundamentals, ticker, ek)
        
        # Tiingo
        tk = _env("TIINGO_API_KEY")
        if tk:
            futures['tiingo'] = ex.submit(_tiingo_fundamentals_fetch, ticker)
        
        # Collect results with timeout
        results = {}
        for source, fut in futures.items():
            try:
                results[source] = fut.result(timeout=15)
            except Exception as e:
                logger.warning(f"Parallel fetch failed for {source}/{ticker}: {e}")
                results[source] = {}
    
    # ========== MERGE IN PRIORITY ORDER ==========
    if results.get('fmp_full'):
        _merge(results['fmp_full'], "from_fmp_full", "FMP")
        merged["from_fmp"] = True
        logger.debug(f"Fundamentals merge: FMP/full ✓ {ticker} fields={results['fmp_full'].get('_fmp_field_count')}")
    
    if results.get('fmp_legacy'):
        _merge(results['fmp_legacy'], "from_fmp", "FMP")
        logger.debug(f"Fundamentals merge: FMP/legacy ✓ {ticker}")
    
    if results.get('simfin'):
        _merge(results['simfin'], "from_simfin", "SimFin")
        logger.debug(f"Fundamentals merge: SimFin ✓ {ticker}")
    
    if results.get('alpha'):
        _merge(results['alpha'], "from_alpha", "Alpha")
        logger.debug(f"Fundamentals merge: Alpha ✓ {ticker}")
    
    if results.get('finnhub'):
        _merge(results['finnhub'], "from_finnhub", "Finnhub")
        logger.debug(f"Fundamentals merge: Finnhub ✓ {ticker}")
    
    if results.get('eodhd'):
        _merge(results['eodhd'], "from_eodhd", "EODHD")
        logger.debug(f"Fundamentals merge: EODHD ✓ {ticker}")
    
    if results.get('tiingo'):
        _merge(results['tiingo'], "from_tiingo", "Tiingo")
        logger.debug(f"Fundamentals merge: Tiingo ✓ {ticker}")

    cov_fields = [merged.get(k) for k in ["roe","roic","gm","ps","pe","de","rev_g_yoy","eps_g_yoy"]]
    merged["Fund_Coverage_Pct"] = float(sum(isinstance(v,(int,float)) and np.isfinite(v) for v in cov_fields))/float(len(cov_fields))
    return merged


def _eodhd_fetch_fundamentals(ticker: str, api_key: str) -> Dict[str, any]:
    """Extracted EODHD fundamentals fetching for parallel execution."""
    try:
        r_eod = http_get_retry(
            f"https://eodhistoricaldata.com/api/fundamentals/{ticker}.US?api_token={api_key}&fmt=json",
            tries=1,
            timeout=10,
        )
        if not r_eod:
            return {}
        
        fj = r_eod.json()
        highlights = fj.get("Highlights", {}) if isinstance(fj, dict) else {}
        valuation = fj.get("Valuation", {}) if isinstance(fj, dict) else {}
        ratios = fj.get("Ratios", {}) if isinstance(fj, dict) else {}
        growth = fj.get("Growth", {}) if isinstance(fj, dict) else {}
        
        def finum(*keys):
            for k in keys:
                v = highlights.get(k) or valuation.get(k) or ratios.get(k) or growth.get(k)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    return float(v)
            return np.nan
        
        return {
            "roe": finum("ReturnOnEquityTTM", "ROE"),
            "roic": np.nan,
            "gm": finum("GrossMarginTTM", "GrossMargin"),
            "ps": finum("PriceToSalesTTM", "PriceToSales"),
            "pe": finum("PERatio", "PE"),
            "de": finum("DebtToEquity", "DebtEquityRatio"),
            "rev_g_yoy": finum("RevenueGrowthTTMYoy", "RevenueGrowth"),
            "eps_g_yoy": finum("EPSGrowthTTMYoy", "EPSGrowth"),
            "sector": fj.get("General", {}).get("Sector", "Unknown") if isinstance(fj.get("General"), dict) else "Unknown",
            "from_eodhd": True,
        }
    except Exception:
        return {}


def _fmp_metrics_fetch(ticker: str, api_key: str) -> Dict[str, any]:
    """
    Fetch fundamental metrics from Financial Modeling Prep (FMP).
    Primary data provider with comprehensive fundamental data.
    """
    try:
        # FMP key-metrics endpoint provides comprehensive ratios
        url = f"https://financialmodelingprep.com/stable/key-metrics?symbol={ticker}&apikey={api_key}"
        r = http_get_retry(url, tries=2, timeout=8)
        if not r:
            return {}
        
        data = r.json()
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}
        
        # Get most recent metrics
        metrics = data[0]
        
        def fget(key, default=np.nan):
            val = metrics.get(key)
            if val is None or val == "None":
                return default
            try:
                return float(val)
            except:
                return default
        
        # Get sector from profile endpoint
        profile_url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
        sector = "Unknown"
        try:
            profile_r = http_get_retry(profile_url, tries=1, timeout=6)
            if profile_r:
                profile_data = profile_r.json()
                if profile_data and isinstance(profile_data, list) and len(profile_data) > 0:
                    sector = profile_data[0].get("sector", "Unknown")
        except:
            pass
        
        return {
            "roe": fget("roe"),
            "roic": fget("roic"),
            "gm": fget("grossProfitMargin"),
            "ps": fget("priceToSalesRatio"),
            "pe": fget("peRatio"),
            "de": fget("debtToEquity"),
            "rev_g_yoy": fget("revenuePerShareGrowth"),
            "eps_g_yoy": fget("netIncomePerShareGrowth"),
            "sector": sector,
        }
    except Exception as e:
        return {}

@st.cache_data(ttl=60*60)
def _fmp_full_bundle_fetch(ticker: str, api_key: str) -> Dict[str, any]:
    """Fetch a richer fundamental bundle from FMP using multiple endpoints (profile, key-metrics, ratios-ttm, financial-growth).

    Returns a dict with unified fields similar to other providers. Falls back silently if endpoints fail.
    """
    try:
        base = "https://financialmodelingprep.com/api/v3"
        endpoints = {
            "profile": f"{base}/profile/{ticker}?apikey={api_key}",
            "key_metrics": f"{base}/key-metrics/{ticker}?period=annual&limit=1&apikey={api_key}",
            "ratios_ttm": f"{base}/ratios-ttm/{ticker}?apikey={api_key}",
            "growth": f"{base}/financial-growth/{ticker}?period=annual&limit=1&apikey={api_key}",
        }

        fetched: Dict[str, any] = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_map = {ex.submit(http_get_retry, url, 2, 8): name for name, url in endpoints.items()}
            for fut in as_completed(fut_map):
                name = fut_map[fut]
                try:
                    resp = fut.result()
                    if resp and resp.status_code == 200:
                        j = resp.json()
                        fetched[name] = j
                except Exception:
                    fetched[name] = None

        def pick_first(obj):
            return obj[0] if isinstance(obj, list) and obj else {}

        profile = pick_first(fetched.get("profile"))
        key_metrics = pick_first(fetched.get("key_metrics"))
        ratios = pick_first(fetched.get("ratios_ttm"))
        growth = pick_first(fetched.get("growth"))

        def ffloat(src, key):
            try:
                v = src.get(key)
                v = float(v)
                return v if np.isfinite(v) else np.nan
            except Exception:
                return np.nan

        out = {
            "roe": ffloat(key_metrics, "roe"),
            "roic": ffloat(key_metrics, "roic"),
            "gm": ffloat(key_metrics, "grossProfitMargin"),
            "ps": ffloat(ratios, "priceToSalesRatioTTM"),
            "pe": ffloat(ratios, "priceEarningsRatioTTM"),
            "de": ffloat(ratios, "debtEquityRatioTTM"),
            "rev_g_yoy": ffloat(growth, "revenueGrowth"),
            "eps_g_yoy": ffloat(growth, "epsGrowth"),
            "sector": profile.get("sector", "Unknown") if isinstance(profile, dict) else "Unknown",
        }

        # Alternate gross margin if missing
        if not np.isfinite(out.get("gm", np.nan)):
            alt_gm = ffloat(ratios, "grossProfitMarginTTM")
            if np.isfinite(alt_gm):
                out["gm"] = alt_gm

        valid_fields = sum(
            1 for k, v in out.items() if k not in ("sector",) and isinstance(v, (int, float)) and np.isfinite(v)
        )
        out["_fmp_field_count"] = valid_fields
        out["from_fmp_full"] = valid_fields >= 3
        return out if valid_fields > 0 else {}
    except Exception:
        return {}


def _alpha_overview_fetch(ticker: str) -> Dict[str, any]:
    """משיכה מ-Alpha OVERVIEW (פשוט ולעניין)."""
    ak = _env("ALPHA_VANTAGE_API_KEY")
    if not ak:
        return {}
    try:
        alpha_throttle(2.0)
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ak}"
        r = http_get_retry(url, tries=1, timeout=6)
        if not r:
            return {}
        j = r.json()
        if not (isinstance(j, dict) and j.get("Symbol")):
            return {}

        def fnum(k):
            try:
                v = float(j.get(k, np.nan))
                return v if np.isfinite(v) else np.nan
            except Exception:
                return np.nan

        gp = fnum("GrossProfitTTM")
        tr = fnum("TotalRevenueTTM")
        gm_calc = (
            (gp / tr) if (np.isfinite(gp) and np.isfinite(tr) and tr > 0) else np.nan
        )
        pm = fnum("ProfitMargin")
        return {
            "roe": fnum("ReturnOnEquityTTM"),
            "roic": np.nan,
            "gm": gm_calc if np.isfinite(gm_calc) else pm,
            "ps": fnum("PriceToSalesTTM"),
            "pe": fnum("PERatio"),
            "de": fnum("DebtToEquityTTM"),
            "rev_g_yoy": fnum("QuarterlyRevenueGrowthYOY"),
            "eps_g_yoy": fnum("QuarterlyEarningsGrowthYOY"),
            "sector": j.get("Sector") or "Unknown",
        }
    except Exception:
        return {}


def _finnhub_metrics_fetch(ticker: str) -> Dict[str, any]:
    """Fallback ל-Finnhub metrics + sector."""
    fk = _env("FINNHUB_API_KEY")
    if not fk:
        return {}
    try:
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={fk}"
        r = http_get_retry(url, tries=1, timeout=10)
        if not r:
            return {}
        j = r.json()
        m = j.get("metric", {})

        def fget(*keys):
            for k in keys:
                v = m.get(k)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    return float(v)
            return np.nan

        de = np.nan
        try:
            total_debt = fget("totalDebt")
            total_equity = fget("totalEquity")
            if (
                np.isfinite(total_debt)
                and np.isfinite(total_equity)
                and total_equity != 0
            ):
                de = total_debt / total_equity
        except Exception:
            pass
        return {
            "roe": fget("roeTtm", "roeAnnual"),
            "roic": np.nan,
            "gm": fget("grossMarginTTM", "grossMarginAnnual"),
            "ps": fget("psTTM", "priceToSalesTTM"),
            "pe": fget("peBasicExclExtraTTM", "peNormalizedAnnual", "peTTM"),
            "de": de,
            "rev_g_yoy": fget("revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy"),
            "eps_g_yoy": fget("epsGrowthTTMYoy", "epsGrowthQuarterlyYoy"),
            "sector": _finnhub_sector(ticker, fk),
        }
    except Exception:
        return {}





def _finnhub_sector(ticker: str, token: str) -> str:
    """משיכת סקטור אמיתי מ-Finnhub (profile2)."""
    r = http_get_retry(
        f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={token}",
        tries=1,
        timeout=8,
    )
    if not r:
        return "Unknown"
    try:
        j = r.json()
        return j.get("finnhubIndustry") or j.get("sector") or "Unknown"
    except Exception:
        return "Unknown"

@st.cache_data(ttl=60 * 60)
def _simfin_fetch(ticker: str, api_key: str) -> Dict[str, any]:
    """Fetch fundamentals from SimFin (basic ratios). Defensive; returns {} if insufficient fields."""
    try:
        url = (
            "https://simfin.com/api/v2/companies/statements/standardised?"
            f"ticker={ticker}&statement=pl&period=FY&limit=1&api-key={api_key}"
        )
        r = http_get_retry(url, tries=2, timeout=10)
        if not r:
            return {}
        j = r.json()
        if not j or not isinstance(j, dict):
            return {}
        def grab(*paths):
            for p in paths:
                v = j.get(p)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    return float(v)
            return np.nan
        roe = grab("roe")
        gp = grab("grossProfit")
        rev = grab("revenue")
        gm = (gp / rev) if (np.isfinite(gp) and np.isfinite(rev) and rev > 0) else np.nan
        ps = grab("priceToSales")
        pe = grab("peRatio")
        de = grab("debtToEquity")
        rev_g = grab("revenueGrowth")
        eps_g = grab("epsGrowth")
        sector = j.get("sector") or "Unknown"
        out = {
            "roe": roe,
            "roic": np.nan,
            "gm": gm,
            "ps": ps,
            "pe": pe,
            "de": de,
            "rev_g_yoy": rev_g,
            "eps_g_yoy": eps_g,
            "sector": sector,
        }
        valid_fields = sum(1 for k,v in out.items() if k != "sector" and isinstance(v,(int,float)) and np.isfinite(v))
        if valid_fields >= 3:
            out["from_simfin"] = True
            out["_simfin_field_count"] = valid_fields
            return out
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60 * 24)
def _tiingo_fundamentals_fetch(ticker: str) -> Dict[str, any]:
    """Fetch fundamentals from Tiingo - P/E, P/B, Market Cap, Dividend Yield."""
    tk = _env("TIINGO_API_KEY")
    if not tk:
        return {}
    try:
        # Tiingo daily price metadata includes fundamental ratios
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}?token={tk}"
        r = http_get_retry(url, tries=2, timeout=8)
        if not r:
            return {}
        j = r.json()
        if not isinstance(j, dict):
            return {}
        
        def fnum(key):
            try:
                v = j.get(key)
                if v is None:
                    return np.nan
                return float(v) if np.isfinite(float(v)) else np.nan
            except:
                return np.nan
        
        # Tiingo provides limited fundamentals in metadata
        # For comprehensive data, use the fundamentals endpoint
        fund_url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/daily?token={tk}"
        fund_r = http_get_retry(fund_url, tries=1, timeout=10)
        
        out = {}
        if fund_r:
            try:
                fund_j = fund_r.json()
                if fund_j and isinstance(fund_j, list) and len(fund_j) > 0:
                    latest = fund_j[0]
                    out = {
                        "pe": fnum("pe") if "pe" in latest else np.nan,
                        "pb": fnum("pb") if "pb" in latest else np.nan,
                        "dividend_yield": fnum("divYield") if "divYield" in latest else np.nan,
                        "market_cap": fnum("marketCap") if "marketCap" in latest else np.nan,
                    }
            except:
                pass
        
        # If fundamentals endpoint failed, try basic metadata
        if not out or all(not np.isfinite(v) for v in out.values() if isinstance(v, float)):
            return {}
        
        valid_count = sum(1 for v in out.values() if isinstance(v, (int, float)) and np.isfinite(v))
        if valid_count >= 1:
            out["from_tiingo"] = True
            out["_tiingo_field_count"] = valid_count
            return out
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60 * 24)
def _tiingo_fundamentals_fetch(ticker: str) -> Dict[str, any]:
    """Fetch fundamentals from Tiingo - P/E, Market Cap."""
    tk = _env("TIINGO_API_KEY")
    if not tk:
        return {}
    try:
        # Tiingo daily metadata includes fundamentals
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}?token={tk}"
        r = http_get_retry(url, tries=2, timeout=8)
        if not r:
            return {}
        j = r.json()
        if not isinstance(j, dict):
            return {}
        
        def fnum(key):
            try:
                v = j.get(key)
                if v is None:
                    return np.nan
                return float(v) if np.isfinite(float(v)) else np.nan
            except:
                return np.nan
        
        # Extract available fundamentals
        out = {}
        if "description" in j:  # has metadata
            out["market_cap"] = fnum("marketCap")
        
        valid_count = sum(1 for v in out.values() if isinstance(v, (int, float)) and np.isfinite(v))
        if valid_count >= 1:
            out["from_tiingo"] = True
            out["_tiingo_field_count"] = valid_count
            return out
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60)
def fetch_beta_vs_benchmark(ticker: str, bench: str = "SPY", days: int = 252) -> float:
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days + 30)
        df_t = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )
        df_b = yf.download(
            bench, start=start, end=end, auto_adjust=True, progress=False
        )
        if df_t.empty or df_b.empty:
            return np.nan
        j = pd.concat(
            [df_t["Close"].pct_change().dropna(), df_b["Close"].pct_change().dropna()],
            axis=1,
        ).dropna()
        j.columns = ["rt", "rb"]
        if len(j) < 40:
            return np.nan
        slope = np.polyfit(j["rb"].to_numpy(), j["rt"].to_numpy(), 1)[0]
        return float(slope)
    except Exception:
        return np.nan


# ==================== External Prices ====================
def get_alpha_price(ticker: str) -> Optional[float]:
    k = _env("ALPHA_VANTAGE_API_KEY")
    if not k:
        return None
    if st.session_state.get("_alpha_ok", False):
        alpha_throttle()
    r = http_get_retry(
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={k}",
        tries=1,
        timeout=10,
    )
    if not r:
        return None
    try:
        j = r.json()
        if "Global Quote" in j and "05. price" in j["Global Quote"]:
            return float(j["Global Quote"]["05. price"])
    except Exception:
        return None
    return None


def get_finnhub_price(ticker: str) -> Optional[float]:
    k = _env("FINNHUB_API_KEY")
    if not k:
        return None
    r = http_get_retry(
        f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={k}", tries=1, timeout=8
    )
    if not r:
        return None
    try:
        j = r.json()
        return float(j["c"]) if "c" in j else None
    except Exception:
        return None


def get_polygon_price(ticker: str) -> Optional[float]:
    k = _env("POLYGON_API_KEY")
    if not k:
        return None
    r = http_get_retry(
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={k}",
        tries=1,
        timeout=8,
    )
    if not r:
        return None
    try:
        j = r.json()
        if j.get("resultsCount", 0) > 0 and "results" in j:
            return float(j["results"][0]["c"])
    except Exception:
        return None
    return None



def get_tiingo_price(ticker: str) -> Optional[float]:
    k = _env("TIINGO_API_KEY")
    if not k:
        return None
    r = http_get_retry(
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?token={k}&resampleFreq=daily",
        tries=1,
        timeout=8,
    )
    if not r:
        return None
    try:
        arr = r.json()
        if isinstance(arr, list) and arr:
            return float(arr[-1].get("close", np.nan))
    except Exception:
        return None
    return None

def get_marketstack_price(ticker: str) -> Optional[float]:
    """Fetch latest end-of-day price from Marketstack."""
    if not CONFIG.get("ENABLE_MARKETSTACK"):
        return None
    mk = _env("MARKETSTACK_API_KEY")
    if not mk:
        return None
    try:
        r = http_get_retry(
            f"http://api.marketstack.com/v1/eod/latest?access_key={mk}&symbols={ticker}",
            tries=1,
            timeout=6,
        )
        if not r:
            return None
        j = r.json()
        data = j.get("data") if isinstance(j, dict) else None
        if isinstance(data, list) and data:
            last = data[0]
            c = last.get("close")
            if isinstance(c, (int, float)) and np.isfinite(c):
                return float(c)
    except Exception:
        return None
    return None

def get_nasdaq_price(ticker: str) -> Optional[float]:
    """Attempt price via Nasdaq Data Link dataset (WIKI legacy). Returns None on failure."""
    if not CONFIG.get("ENABLE_NASDAQ_DL"):
        return None
    nk = _env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY")
    if not nk:
        return None
    try:
        r = http_get_retry(
            f"https://data.nasdaq.com/api/v3/datasets/WIKI/{ticker}.json?api_key={nk}",
            tries=1,
            timeout=6,
        )
        if not r:
            return None
        j = r.json()
        ds = j.get("dataset", {})
        data = ds.get("data")
        if isinstance(data, list) and data:
            row = data[0]
            if len(row) >= 5:
                c = row[4]
                if isinstance(c, (int, float)) and np.isfinite(c):
                    return float(c)
    except Exception:
        return None
    return None

def get_eodhd_price(ticker: str) -> Optional[float]:
    """Fetch real-time (delayed) price from EODHD."""
    if not CONFIG.get("ENABLE_EODHD"):
        return None
    ek = _env("EODHD_API_KEY") or _env("EODHD_TOKEN")
    if not ek:
        return None
    try:
        r = http_get_retry(
            f"https://eodhistoricaldata.com/api/real-time/{ticker}.US?api_token={ek}&fmt=json",
            tries=1,
            timeout=6,
        )
        if not r:
            return None
        j = r.json()
        c = j.get("close") or j.get("Close")
        if isinstance(c, (int, float)) and np.isfinite(c):
            return float(c)
    except Exception:
        return None
    return None


# ==================== UI ====================
st.set_page_config(page_title="Asaf's Stock Scout — 2025", page_icon="📈", layout="wide")
st.markdown(
    """
<style>
body{direction:rtl}.block-container{padding-top:1rem;padding-bottom:2rem}
h1,h2,h3{text-align:right}[data-testid="stMarkdownContainer"], label{ text-align:right }
thead tr th{ text-align:right } .rtl-table table{ direction:rtl }
.rtl-table th,.rtl-table td{ text-align:right !important }
.badge{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;color:#1e293b;padding:2px 10px;border-radius:999px;font-weight:600}
.status-buy{background:#ecfdf5;border:1px solid #34d399;color:#065f46;padding:2px 10px;border-radius:999px;font-weight:600}
</style>
""",
    unsafe_allow_html=True,
)

# Secrets button
def _mask(s: Optional[str], show_last: int = 4) -> str:
    if not s:
        return "—"
    s = str(s).strip()
    return (
        ("•" * (len(s) - show_last)) + s[-show_last:]
        if len(s) > show_last
        else ("•" * (len(s) - 1)) + s[-1]
    )


col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("בדיקת סיקרטים 🔐"):
        st.info(
            f"Alpha: {_mask(_env('ALPHA_VANTAGE_API_KEY'))}\n\n"
            f"Finnhub: {_mask(_env('FINNHUB_API_KEY'))}\n\n"
            f"Polygon: {_mask(_env('POLYGON_API_KEY'))}\n\n"
            f"Tiingo: {_mask(_env('TIINGO_API_KEY'))}\n\n"
            f"FMP: {_mask(_env('FMP_API_KEY'))}"
        )

st.title("📈 Stock Scout — 2025 (Auto)")

# Status table
alpha_ok, alpha_reason = _check_alpha()
finn_ok, finnh_reason = _check_finnhub()
poly_ok, poly_reason = _check_polygon()
tiin_ok, tiin_reason = _check_tiingo()
fmp_ok, fmp_reason = _check_fmp()
st.session_state["_alpha_ok"] = bool(alpha_ok)
simfin_key = _env("SIMFIN_API_KEY") if CONFIG.get("ENABLE_SIMFIN") else None
marketstack_key = _env("MARKETSTACK_API_KEY") if CONFIG.get("ENABLE_MARKETSTACK") else None
nasdaq_key = (_env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY")) if CONFIG.get("ENABLE_NASDAQ_DL") else None
eodhd_key = (_env("EODHD_API_KEY") or _env("EODHD_TOKEN")) if CONFIG.get("ENABLE_EODHD") else None

providers_status = []
def add_provider(name, price_ok, fund_ok, reason):
    providers_status.append({
        "מקור": name,
        "מחיר": "🟢" if price_ok else "🔴",
        "פונדמנטלים": "🟢" if fund_ok else "🔴",
        "סיבה": reason
    })

def _std_reason(name: str, price_ok: bool, fund_ok: bool, has_key: bool, base: bool=False) -> str:
    if base:
        return "Base"
    if price_ok or fund_ok:
        return "OK"
    if not has_key:
        return "No API Key"
    return "Unavailable"

# Yahoo baseline always price only
add_provider("Yahoo", True, False, _std_reason("Yahoo", True, False, True, base=True))
add_provider("FMP", False, fmp_ok, _std_reason("FMP", False, fmp_ok, fmp_ok))
add_provider("Alpha Vantage", alpha_ok, alpha_ok, _std_reason("Alpha Vantage", alpha_ok, alpha_ok, bool(_env("ALPHA_VANTAGE_API_KEY"))))
add_provider("Finnhub", finn_ok, finn_ok, _std_reason("Finnhub", finn_ok, finn_ok, bool(_env("FINNHUB_API_KEY"))))
add_provider("Polygon", poly_ok, False, _std_reason("Polygon", poly_ok, False, bool(_env("POLYGON_API_KEY"))))
add_provider("Tiingo", tiin_ok, False, _std_reason("Tiingo", tiin_ok, False, bool(_env("TIINGO_API_KEY"))))
add_provider("SimFin", False, bool(simfin_key), _std_reason("SimFin", False, bool(simfin_key), bool(simfin_key)))
add_provider("Marketstack", bool(marketstack_key), False, _std_reason("Marketstack", bool(marketstack_key), False, bool(marketstack_key)))
add_provider("NasdaqDL", bool(nasdaq_key), False, _std_reason("NasdaqDL", bool(nasdaq_key), False, bool(nasdaq_key)))
add_provider("EODHD", bool(eodhd_key), bool(eodhd_key), _std_reason("EODHD", bool(eodhd_key), bool(eodhd_key), bool(eodhd_key)))

status_df = pd.DataFrame(providers_status)
st.markdown("### 🔌 סטטוס מקורות")
_provider_css = """
<style>
.provider-table-container {overflow-x:auto;}
.provider-table-container table {width:100%; border-collapse:collapse; table-layout:fixed;}
.provider-table-container th, .provider-table-container td {padding:6px 8px; text-align:center; font-size:14px;}
.provider-table-container th:nth-child(1), .provider-table-container td:nth-child(1){text-align:right; min-width:130px;}
@media (max-width:600px){
    .provider-table-container th, .provider-table-container td {padding:4px 6px; font-size:12px;}
    .provider-table-container th:nth-child(1), .provider-table-container td:nth-child(1){min-width:140px;}
}
</style>
"""
st.markdown(_provider_css, unsafe_allow_html=True)
html_rows = []
html_rows.append("<tr><th>מקור</th><th>מחיר</th><th>פונדמנטלים</th><th>Reason</th></tr>")
for r in providers_status:
        html_rows.append(
                f"<tr><td>{r['מקור']}</td><td>{r['מחיר']}</td><td>{r['פונדמנטלים']}</td><td>{r['סיבה']}</td></tr>"
        )
providers_html = "<div class='provider-table-container'><table>" + "".join(html_rows) + "</table></div>"
st.markdown(providers_html, unsafe_allow_html=True)

# Cache reset button
col_cache, _ = st.columns([1,4])
with col_cache:
    if st.button("🔄 איפוס Cache וטעינה מחדש"):
        st.cache_data.clear()
        st.rerun()

# timers
def t_start() -> float:
    return time.perf_counter()


def t_end(t0: float) -> float:
    return time.perf_counter() - t0


phase_times: Dict[str, float] = {}
if "av_calls" not in st.session_state:
    st.session_state.av_calls = 0

# 1) Universe
t0 = t_start()
universe = (
    build_universe(limit=CONFIG["UNIVERSE_LIMIT"])
    if CONFIG["SMART_SCAN"]
    else build_universe(limit=200)
)
phase_times["בונה יקום"] = t_end(t0)

# 2) History
t0 = t_start()
data_map = fetch_history_bulk(universe, CONFIG["LOOKBACK_DAYS"], CONFIG["MA_LONG"])
phase_times["מוריד נתונים"] = t_end(t0)

# 3) Technical score + hard filters
t0 = t_start()
W = CONFIG["WEIGHTS"]


W = _normalize_weights(W)

rows: List[dict] = []
lo_rsi, hi_rsi = CONFIG["RSI_BOUNDS"]

for tkr, df in data_map.items():
    if df is None or df.empty:
        continue
    df = df.copy()
    df["MA_S"] = df["Close"].rolling(int(CONFIG["MA_SHORT"])).mean()
    df["MA_L"] = df["Close"].rolling(int(CONFIG["MA_LONG"])).mean()
    df["RSI"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["Vol20"] = df["Volume"].rolling(20).mean()

    if CONFIG["USE_MACD_ADX"]:
        m, ms, mh = macd_line(df["Close"])
        df["MACD"], df["MACD_SIG"], df["MACD_HIST"] = m, ms, mh
        try:
            adx_out = adx(df, 14)
            # adx() may return a DataFrame with columns ['ADX','PLUS_DI','MINUS_DI']
            if isinstance(adx_out, pd.DataFrame):
                # use named columns when available, fallback to positional
                if "ADX" in adx_out.columns:
                    adx_series = pd.to_numeric(adx_out["ADX"], errors="coerce").reindex(df.index)
                else:
                    adx_series = pd.to_numeric(adx_out.iloc[:, 0], errors="coerce").reindex(df.index)

                if "PLUS_DI" in adx_out.columns:
                    plus_di = pd.to_numeric(adx_out["PLUS_DI"], errors="coerce").reindex(df.index)
                else:
                    plus_di = pd.to_numeric(adx_out.iloc[:, 1] if adx_out.shape[1] > 1 else pd.Series(np.nan, index=df.index), errors="coerce").reindex(df.index)

                if "MINUS_DI" in adx_out.columns:
                    minus_di = pd.to_numeric(adx_out["MINUS_DI"], errors="coerce").reindex(df.index)
                else:
                    minus_di = pd.to_numeric(adx_out.iloc[:, 2] if adx_out.shape[1] > 2 else pd.Series(np.nan, index=df.index), errors="coerce").reindex(df.index)
            else:
                adx_series = pd.to_numeric(adx_out, errors="coerce").reindex(df.index)
                plus_di = pd.Series(np.nan, index=df.index)
                minus_di = pd.Series(np.nan, index=df.index)

            df.loc[:, "ADX14"] = adx_series.values
            df.loc[:, "PLUS_DI14"] = plus_di.values
            df.loc[:, "MINUS_DI14"] = minus_di.values
        except Exception:
            df["ADX14"] = np.nan
            df["PLUS_DI14"] = np.nan
            df["MINUS_DI14"] = np.nan

    price = float(df["Close"].iloc[-1])
    if (not np.isfinite(price)) or (price < CONFIG["MIN_PRICE"]):
        continue

    last_ma_s = float(df["MA_S"].iloc[-1])
    last_ma_l = float(df["MA_L"].iloc[-1])
    ma_ok = (
        (float(price > last_ma_s) + float(last_ma_s > last_ma_l)) / 2.0
        if (np.isfinite(last_ma_s) and np.isfinite(last_ma_l))
        else 0.0
    )

    rsi_val = float(df["RSI"].iloc[-1])
    if np.isfinite(rsi_val):
        if rsi_val < lo_rsi:
            rsi_score = max(0.0, 1 - (lo_rsi - rsi_val) / 20)
        elif rsi_val > hi_rsi:
            rsi_score = max(0.0, 1 - (rsi_val - hi_rsi) / 20)
        else:
            rsi_score = 1.0
    else:
        rsi_score = 0.0

    vol20 = float(df["Vol20"].iloc[-1])
    vol_today = float(df["Volume"].iloc[-1])
    if np.isfinite(vol20) and vol20 < CONFIG["MIN_AVG_VOLUME"]:
        continue
    vol_ok = (
        (min(2.0, vol_today / vol20) / 2.0)
        if (np.isfinite(vol20) and vol20 > 0 and np.isfinite(vol_today))
        else 0.0
    )

    ret_1m = float(df["Close"].pct_change(21).iloc[-1])
    ret_3m = float(df["Close"].pct_change(63).iloc[-1])
    ret_6m = float(df["Close"].pct_change(126).iloc[-1])
    mom_score = float(_sigmoid(np.nanmean([ret_1m, ret_3m, ret_6m])))

    window_52w = min(len(df), 252)
    hi_52w = float(df["Close"].tail(window_52w).max())
    if np.isfinite(hi_52w) and hi_52w > 0:
        near_high_raw = 1.0 - min(1.0, max(0.0, (hi_52w - price) / hi_52w))
        if near_high_raw >= 0.95:
            near_high_score = 0.45
        elif 0.75 <= near_high_raw <= 0.90:
            near_high_score = 1.00
        elif 0.90 < near_high_raw < 0.95:
            near_high_score = 0.75
        else:
            near_high_score = near_high_raw
    else:
        near_high_raw, near_high_score = np.nan, 0.0

    if np.isfinite(last_ma_l) and last_ma_l > 0:
        overext_ratio = max(0.0, (price - last_ma_l) / last_ma_l)
        overext_score = 1.0 - min(
            1.0, overext_ratio / max(1e-6, CONFIG["OVEREXT_SOFT"])
        )
    else:
        overext_ratio, overext_score = np.nan, 0.0

    ratio_to_high = price / hi_52w if (np.isfinite(hi_52w) and hi_52w > 0) else np.nan
    if np.isfinite(ratio_to_high):
        lo, hi = CONFIG["PULLBACK_RANGE"]
        if lo <= ratio_to_high <= hi:
            pullback_score = 1.0
        else:
            dist = min(abs(ratio_to_high - lo), abs(ratio_to_high - hi))
            pullback_score = max(0.0, 1.0 - dist * 10)
    else:
        pullback_score = 0.0

    atr14 = float(df["ATR14"].iloc[-1])
    if np.isfinite(atr14) and price > 0:
        vol_rel = atr14 / price
        volatility_score = 1.0 - min(1.0, vol_rel / 0.05)
    else:
        vol_rel, volatility_score = np.nan, 0.0

    dollar_vol = (price * vol20) if (np.isfinite(price) and np.isfinite(vol20)) else 0.0
    if dollar_vol < CONFIG["MIN_DOLLAR_VOLUME"]:
        continue
    if np.isfinite(vol_rel) and vol_rel > CONFIG["ATR_PRICE_HARD"]:
        continue
    if np.isfinite(overext_ratio) and overext_ratio > CONFIG["OVEREXT_HARD"]:
        continue

    if np.isfinite(hi_52w) and np.isfinite(atr14) and atr14 > 0:
        reward_risk = max(0.0, (hi_52w - price) / atr14)
        rr_score = min(1.0, reward_risk / 4.0)
    else:
        reward_risk, rr_score = np.nan, 0.0

    macd_score = 0.0
    adx_score = 0.0
    if CONFIG["USE_MACD_ADX"] and "MACD" in df.columns:
        macd_v = float(df["MACD"].iloc[-1])
        macd_sig = float(df["MACD_SIG"].iloc[-1])
        macd_score = 1.0 if macd_v > macd_sig else 0.0
    if CONFIG["USE_MACD_ADX"] and "ADX14" in df.columns:
        adx_v = (
            float(df["ADX14"].iloc[-1]) if pd.notna(df["ADX14"].iloc[-1]) else np.nan
        )
        adx_score = (
            np.clip((adx_v - 15) / 20.0, 0.0, 1.0) if np.isfinite(adx_v) else 0.0
        )

    score = (
        W["ma"] * ma_ok
        + W["mom"] * mom_score
        + W["rsi"] * rsi_score
        + W["near_high_bell"] * near_high_score
        + W["vol"] * (vol_ok if np.isfinite(vol_ok) else 0.0)
        + W["overext"] * overext_score
        + W["pullback"] * pullback_score
        + W["risk_reward"] * rr_score
        + W["macd"] * macd_score
        + W["adx"] * adx_score
    )

    rows.append(
        {
            "Ticker": tkr,
            "Price_Yahoo": price,
            "Score_Tech": round(100 * float(score), 1),
            "RSI": round(rsi_val, 1) if np.isfinite(rsi_val) else np.nan,
            "Near52w": round(near_high_raw * 100, 1)
            if np.isfinite(near_high_raw)
            else np.nan,
            "Volx20d": round(vol_today / vol20, 2)
            if (np.isfinite(vol_today) and np.isfinite(vol20) and vol20 > 0)
            else np.nan,
            "OverextRatio": round(overext_ratio, 3)
            if np.isfinite(overext_ratio)
            else np.nan,
            "ATR_Price": round(vol_rel, 4) if np.isfinite(vol_rel) else np.nan,
            "RewardRisk": round(reward_risk, 2) if np.isfinite(reward_risk) else np.nan,
            "ATR14": atr14,
        }
    )

results = pd.DataFrame(rows)
phase_times["מחשב ניקוד (טכני)"] = t_end(t0)
if results.empty:
    st.warning("אין תוצאות אחרי הסינון. ייתכן שהספים קשוחים מדי עבור היקום הנוכחי.")
    st.stop()

results = results.sort_values(
    ["Score_Tech", "Ticker"], ascending=[False, True]
).reset_index(drop=True)

# 3a) Fundamentals (Top-K) + mix score
fundamental_available = (
    alpha_ok or finn_ok or fmp_ok or bool(simfin_key) or bool(eodhd_key)
)
if CONFIG["FUNDAMENTAL_ENABLED"] and fundamental_available:
    t0 = t_start()
    broaden = max(int(CONFIG.get("FUNDAMENTAL_TOP_K", 15)), int(CONFIG.get("TOPN_RESULTS", 20)) * 3, int(CONFIG.get("TARGET_RECOMMENDATIONS_MAX", 7)) * 3)
    take_k = int(min(max(10, broaden), len(results)))
    for c in [
        "Fundamental_S",
        "Sector",
        "PE_f",
        "PS_f",
        "ROE_f",
        "ROIC_f",
        "GM_f",
        "DE_f",
        "RevG_f",
        "EPSG_f",
    ]:
        results[c] = np.nan
    # Add provider flag columns
    results["Fund_from_FMP"] = False
    results["Fund_from_Alpha"] = False
    results["Fund_from_Finnhub"] = False
    results["Fund_from_SimFin"] = False
    results["Fund_from_EODHD"] = False
    
    for idx in results.head(take_k).index:
        tkr = results.at[idx, "Ticker"]
        rank = list(results.head(take_k).index).index(idx) + 1  # 1-based rank
        # Smart Alpha: enable only for top 15 to respect 25/day rate limit
        use_alpha = (rank <= 15)
        d = fetch_fundamentals_bundle(tkr, enable_alpha_smart=use_alpha)
        
        # Store provider metadata
        results.loc[idx, "Fund_from_FMP"] = d.get("from_fmp", False) or d.get("from_fmp_full", False)
        results.loc[idx, "Fund_from_Alpha"] = d.get("from_alpha", False)
        results.loc[idx, "Fund_from_Finnhub"] = d.get("from_finnhub", False)
        results.loc[idx, "Fund_from_SimFin"] = d.get("from_simfin", False)
        results.loc[idx, "Fund_from_EODHD"] = d.get("from_eodhd", False)
        
        # Get detailed breakdown using new function
        fund_result = compute_fundamental_score_with_breakdown(d)
        results.loc[idx, "Fundamental_S"] = round(fund_result.total, 1)
        
        # Store breakdown scores and labels
        results.loc[idx, "Quality_Score_F"] = round(fund_result.breakdown.quality_score, 1)
        results.loc[idx, "Quality_Label"] = fund_result.breakdown.quality_label
        results.loc[idx, "Growth_Score_F"] = round(fund_result.breakdown.growth_score, 1)
        results.loc[idx, "Growth_Label"] = fund_result.breakdown.growth_label
        results.loc[idx, "Valuation_Score_F"] = round(fund_result.breakdown.valuation_score, 1)
        results.loc[idx, "Valuation_Label"] = fund_result.breakdown.valuation_label
        results.loc[idx, "Leverage_Score_F"] = round(fund_result.breakdown.leverage_score, 1)
        results.loc[idx, "Leverage_Label"] = fund_result.breakdown.leverage_label
        
        # Store provider attribution for transparency
        sources_dict = d.get("_sources", {})
        if sources_dict:
            # Build attribution string: "ROE: FMP | GM: SimFin | PE: Finnhub"
            attrs = [f"{k.upper()}: {v}" for k, v in sources_dict.items() if k != "sector"]
            results.loc[idx, "Fund_Attribution"] = " | ".join(attrs[:5]) if attrs else ""  # Limit to 5 for readability
        else:
            results.loc[idx, "Fund_Attribution"] = ""
        
        # Store raw metrics
        results.loc[idx, "PE_f"] = d.get("pe", np.nan)
        results.loc[idx, "PS_f"] = d.get("ps", np.nan)
        results.loc[idx, "ROE_f"] = d.get("roe", np.nan)
        results.loc[idx, "ROIC_f"] = d.get("roic", np.nan)
        results.loc[idx, "GM_f"] = d.get("gm", np.nan)
        results.loc[idx, "DE_f"] = d.get("de", np.nan)
        results.loc[idx, "RevG_f"] = d.get("rev_g_yoy", np.nan)
        results.loc[idx, "EPSG_f"] = d.get("eps_g_yoy", np.nan)
        results.loc[idx, "Sector"] = d.get("sector") or "Unknown"
    results["Score"] = results["Score_Tech"]
    results.loc[results.head(take_k).index, "Score"] = (
        1 - float(CONFIG["FUNDAMENTAL_WEIGHT"])
    ) * results.loc[results.head(take_k).index, "Score_Tech"] + float(
        CONFIG["FUNDAMENTAL_WEIGHT"]
    ) * results.loc[
        results.head(take_k).index, "Fundamental_S"
    ].fillna(
        0
    )
    results = results.sort_values(
        ["Score", "Ticker"], ascending=[False, True]
    ).reset_index(drop=True)
    phase_times["פונדמנטלי (Alpha/Finnhub)"] = t_end(t0)
else:
    results["Score"] = results["Score_Tech"]

# Earnings blackout
if CONFIG["EARNINGS_BLACKOUT_DAYS"] > 0:
    to_check_idx = list(results.head(int(CONFIG["EARNINGS_CHECK_TOPK"])).index)
    symbols = [results.at[i, "Ticker"] for i in to_check_idx]
    ed_map = _earnings_batch(symbols)
    now_utc = datetime.utcnow()
    keep_mask = np.ones(len(results), dtype=bool)
    for idx in to_check_idx:
        tkr = results.at[idx, "Ticker"]
        dt_earn = ed_map.get(tkr)
        if dt_earn is None:
            continue
        gap_days = abs((dt_earn - now_utc).days)
        if gap_days <= int(CONFIG["EARNINGS_BLACKOUT_DAYS"]):
            keep_mask[idx] = False
            results.at[idx, "EarningsNote"] = f"Excluded: earnings within {gap_days}d"
    results = results[keep_mask].reset_index(drop=True)
    if results.empty:
        st.warning('כל המועמדות בטופ-K נפסלו עקב חלון דו"חות.')
        st.stop()

# 3b) Beta filter
if CONFIG["BETA_FILTER_ENABLED"]:
    t0 = t_start()
    take_k_beta = int(min(CONFIG["BETA_TOP_K"], len(results)))
    for idx in results.head(take_k_beta).index:
        tkr = results.at[idx, "Ticker"]
        results.loc[idx, "Beta"] = fetch_beta_vs_benchmark(
            tkr, bench=CONFIG["BETA_BENCHMARK"], days=252
        )
    results = results[
        ~(
            (results["Beta"].notna())
            & (results["Beta"] > float(CONFIG["BETA_MAX_ALLOWED"]))
        )
    ].reset_index(drop=True)
    phase_times["מסנן בטא"] = t_end(t0)

# 3c) Advanced Filters - NEW!
t0 = t_start()
st.info("🔬 מפעיל סינונים מתקדמים...")

# Fetch benchmark data once
benchmark_df = fetch_benchmark_data(CONFIG["BETA_BENCHMARK"], CONFIG["LOOKBACK_DAYS"])

# Add columns for advanced signals
for col in ["RS_63d", "Volume_Surge", "MA_Aligned", "Quality_Score", 
            "RR_Ratio", "Momentum_Consistency", "High_Confidence"]:
    results[col] = np.nan

advanced_keep_mask = []
logger.info(f"🔬 Running advanced filters on full set ({len(results)} stocks)...")
rejection_reasons = {}

for idx in results.index:
    tkr = results.at[idx, "Ticker"]
    if tkr not in data_map or benchmark_df.empty:
        advanced_keep_mask.append(True)
        continue
    
    df = data_map[tkr]
    base_score = results.at[idx, "Score"]
    
    # Compute advanced score and signals
    enhanced_score, signals = compute_advanced_score(tkr, df, benchmark_df, base_score)
    
    # Check rejection criteria
    should_reject, reason = should_reject_ticker(signals)
    
    # Debug: Log rejections
    if should_reject:
        logger.debug(f"{tkr} rejected: {reason} | signals: {signals}")
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    if should_reject:
        advanced_keep_mask.append(False)
        results.at[idx, "RejectionReason"] = reason
        continue
    
    # Update score and signals
    results.loc[idx, "Score"] = enhanced_score
    results.loc[idx, "RS_63d"] = signals.get("rs_63d", np.nan)
    results.loc[idx, "Volume_Surge"] = signals.get("volume_surge", np.nan)
    results.loc[idx, "MA_Aligned"] = signals.get("ma_aligned", False)
    results.loc[idx, "Quality_Score"] = signals.get("quality_score", 0.0)
    results.loc[idx, "RR_Ratio"] = signals.get("risk_reward_ratio", np.nan)
    results.loc[idx, "Momentum_Consistency"] = signals.get("momentum_consistency", 0.0)
    results.loc[idx, "High_Confidence"] = signals.get("high_confidence", False)
    
    advanced_keep_mask.append(True)

# Extend mask for remaining rows
while len(advanced_keep_mask) < len(results):
    advanced_keep_mask.append(True)

# Log rejection statistics
logger.info(f"📊 Advanced filter results: {sum(advanced_keep_mask)}/{len(advanced_keep_mask)} stocks kept (after rejections)")
if rejection_reasons:
    logger.info(f"Rejection breakdown: {rejection_reasons}")

results = results[advanced_keep_mask].reset_index(drop=True)

# Re-sort by enhanced score
results = results.sort_values(["Score", "Ticker"], ascending=[False, True]).reset_index(drop=True)

phase_times["סינונים מתקדמים"] = t_end(t0)

if results.empty:
    st.warning("כל המניות נפסלו בסינונים המתקדמים. נסה להקל על הקריטריונים.")
    st.stop()

# 3d) Apply risk classification and data quality evaluation
t0 = t_start()
st.info("🔍 מסווג מניות לפי רמת סיכון ואיכות נתונים...")

# Debug: Check what data we have
logger.info(f"Columns available: {results.columns.tolist()}")
logger.info(f"Sample RS_63d values: {results['RS_63d'].head().tolist() if 'RS_63d' in results.columns else 'MISSING'}")
logger.info(f"Sample Volume_Surge values: {results['Volume_Surge'].head().tolist() if 'Volume_Surge' in results.columns else 'MISSING'}")
logger.info(f"Sample RR_Ratio values: {results['RR_Ratio'].head().tolist() if 'RR_Ratio' in results.columns else 'MISSING'}")
logger.info(f"Sample Quality_Score values: {results['Quality_Score'].head().tolist() if 'Quality_Score' in results.columns else 'MISSING'}")
logger.info(f"Sample Fundamental_S values: {results['Fundamental_S'].head().tolist() if 'Fundamental_S' in results.columns else 'MISSING'}")
logger.info(f"Sample Momentum_Consistency values: {results['Momentum_Consistency'].head().tolist() if 'Momentum_Consistency' in results.columns else 'MISSING'}")

results = apply_classification(results)

# Fundamentals coverage summary
if "Fund_Coverage_Pct" in results.columns:
    coverage_vals = results["Fund_Coverage_Pct"].dropna().tolist()
    if coverage_vals:
        avg_cov = float(np.mean(coverage_vals))
        pct_good = float(sum(v >= 0.5 for v in coverage_vals)) / float(len(coverage_vals))
        # Histogram buckets
        buckets = {"0-25%":0, "25-50%":0, "50-75%":0, "75-100%":0}
        for v in coverage_vals:
            if v < 0.25: buckets["0-25%"] += 1
            elif v < 0.50: buckets["25-50%"] += 1
            elif v < 0.75: buckets["50-75%"] += 1
            else: buckets["75-100%"] += 1
        logger.info(f"Fundamentals coverage buckets: {buckets}")
        logger.info(f"Fundamentals coverage: mean={avg_cov:.2f}, >=50% fields for {pct_good*100:.1f}% of tickers")
        warn_thresh = CONFIG.get("COVERAGE_WARN_THRESHOLD", 0.4)
        if avg_cov < warn_thresh:
            st.warning(f"⚠️ כיסוי פונדמנטלי נמוך: ממוצע {avg_cov:.2f} ({pct_good*100:.1f}% עם ≥50% מהשדות). שקול הוספת מקורות או מפתחות.")
        else:
            st.info(f"🧬 כיסוי פונדמנטלי ממוצע: {avg_cov:.2f} | {pct_good*100:.0f}% מהטיקרסים עם ≥50% שדות.")

# Show classification statistics
core_count = len(results[results["Risk_Level"] == "core"])
spec_count = len(results[results["Risk_Level"] == "speculative"])
high_qual = len(results[results["Data_Quality"] == "high"])
med_qual = len(results[results["Data_Quality"] == "medium"])
low_qual = len(results[results["Data_Quality"] == "low"])

st.write(f"**סיווג ראשוני:** {core_count} Core, {spec_count} Speculative | "
         f"**איכות:** {high_qual} גבוהה, {med_qual} בינונית, {low_qual} נמוכה")

# Filter out stocks that shouldn't be displayed (very low quality)
displayable = results[results["Should_Display"]].copy()
hidden_count = len(results) - len(displayable)
if hidden_count > 0:
    st.write(f"🔻 הוסתרו {hidden_count} מניות עם איכות נתונים נמוכה מאוד")
    logger.info(f"Hidden {hidden_count} stocks due to very low data quality")

results = displayable.reset_index(drop=True)

# Apply Core recommendation filters using CONFIG constants
core_before_filter = len(results[results["Risk_Level"] == "core"])
results = filter_core_recommendations(results, CONFIG, adaptive=True)
core_after_filter = len(results)

if core_before_filter > 0:
    st.write(f"🎯 **סינון Core:** {core_before_filter} → {core_after_filter} מניות עברו את הפילטרים המחמירים")

phase_times["סיווג סיכון ואיכות"] = t_end(t0)

if results.empty:
    st.error("❌ **כל המניות נפסלו!**")
    st.write("**סיבות אפשריות:**")
    st.write("- 🔴 אף מניה לא סווגה כ-Core (כולן Speculative)")
    st.write("- 🔴 מניות Core לא עמדו בפילטרים הטכניים (RSI, ATR, Overextension)")
    st.write("- 🔴 איכות נתונים נמוכה מדי (חסרים מחירים/פונדמנטלים)")
    st.write(f"- 🔴 ניתן להקל על הגדרות ב-CONFIG: MIN_QUALITY_SCORE_CORE={CONFIG['MIN_QUALITY_SCORE_CORE']}, "
             f"MAX_ATR_PRICE_CORE={CONFIG['MAX_ATR_PRICE_CORE']}")
    st.stop()
elif "Adaptive_Relaxed" in results.columns and results["Adaptive_Relaxed"].any():
    # Show adaptive relaxation banner
    st.warning("🔄 הופעל מצב ריכוך אוטומטי — מוצגות מועמדות ספקולטיביות עם סינון טכני רופף יותר. מומלץ לבדוק פונדמנטלים לפני החלטה.")
    st.write("קריטריוני ריכוך: איכות ≥ "
             f"{max(CONFIG['MIN_QUALITY_SCORE_CORE']-5,15)}, ATR/Price ≤ {CONFIG['MAX_ATR_PRICE_CORE']+0.04}, "
             f"Overext ≤ {CONFIG['MAX_OVEREXTENSION_CORE']+0.05}, RSI בטווח מורחב, Reward/Risk ≥ {max(CONFIG['MIN_RR_CORE']-0.3,1.0)}")

# Show results count with guidance
results_count = len(results)
target_min = CONFIG.get("TARGET_RECOMMENDATIONS_MIN", 3)
target_max = CONFIG.get("TARGET_RECOMMENDATIONS_MAX", 7)

if results_count < target_min:
    st.warning(f"⚠️ רק {results_count} מניות עברו את הסינונים (יעד: {target_min}-{target_max}). "
               f"הסינונים מחמירים כרגע. שקול להרחיב את ה-thresholds אם אתה רוצה יותר מועמדים.")
elif results_count > target_max:
    st.info(f"📊 {results_count} מניות עברו את הסינונים. מציג את {target_max} המובילות.")
    results = results.head(target_max)
else:
    st.success(f"✅ {results_count} מניות Core איכותיות עברו את כל הסינונים!")

# External price verification (Top-K)
t0 = t_start()
results["Price_Alpha"] = np.nan
results["Price_Finnhub"] = np.nan
 # IEX price column removed
results["Price_Marketstack"] = np.nan
results["Price_NasdaqDL"] = np.nan
results["Price_EODHD"] = np.nan
results["Price_Mean"] = np.nan
results["Price_STD"] = np.nan
results["Source_List"] = "🟡Yahoo"


def _fetch_external_for(
    tkr: str, py: float
) -> Tuple[str, Dict[str, Optional[float]], List[str]]:
    vals: Dict[str, Optional[float]] = {}
    srcs: List[str] = []
    if np.isfinite(py):
        vals["Yahoo"] = float(py)
        srcs.append("🟡Yahoo")
    if alpha_ok:
        p = get_alpha_price(tkr)
        if p is not None:
            vals.setdefault("Alpha", p)
            srcs.append("🟣Alpha")
            st.session_state.av_calls = st.session_state.get("av_calls", 0) + 1
    if finn_ok:
        p = get_finnhub_price(tkr)
        if p is not None:
            vals.setdefault("Finnhub", p)
            srcs.append("🔵Finnhub")
    if poly_ok and _env("POLYGON_API_KEY"):
        p = get_polygon_price(tkr)
        if p is not None:
            vals.setdefault("Polygon", p)
            srcs.append("🟢Polygon")
    if tiin_ok and _env("TIINGO_API_KEY"):
        p = get_tiingo_price(tkr)
        if p is not None:
            vals.setdefault("Tiingo", p)
            srcs.append("🟠Tiingo")
    if CONFIG.get("ENABLE_MARKETSTACK") and _env("MARKETSTACK_API_KEY"):
        p = get_marketstack_price(tkr)
        if p is not None:
            vals.setdefault("Marketstack", p)
            srcs.append("🧩Marketstack")
    if CONFIG.get("ENABLE_NASDAQ_DL") and (_env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY")):
        p = get_nasdaq_price(tkr)
        if p is not None:
            vals.setdefault("NasdaqDL", p)
            srcs.append("🏛NasdaqDL")
    if CONFIG.get("ENABLE_EODHD") and (_env("EODHD_API_KEY") or _env("EODHD_TOKEN")):
        p = get_eodhd_price(tkr)
        if p is not None:
            vals.setdefault("EODHD", p)
            srcs.append("📘EODHD")

# External price verification - run if ANY provider is available
any_price_provider = (
    finn_ok
    or (poly_ok and _env("POLYGON_API_KEY"))
    or (CONFIG.get("ENABLE_MARKETSTACK") and _env("MARKETSTACK_API_KEY"))
    or (CONFIG.get("ENABLE_NASDAQ_DL") and (_env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY")))
    or (CONFIG.get("ENABLE_EODHD") and (_env("EODHD_API_KEY") or _env("EODHD_TOKEN")))
)

if CONFIG["EXTERNAL_PRICE_VERIFY"] and any_price_provider:
    subset_idx = list(results.head(int(CONFIG["TOP_VALIDATE_K"])).index)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [
            ex.submit(
                _fetch_external_for,
                results.loc[idx, "Ticker"],
                float(results.loc[idx, "Price_Yahoo"]),
            )
            for idx in subset_idx
        ]
        for f in as_completed(futures):
            try:
                tkr, vals, srcs = f.result()
            except Exception:
                continue
            idx = results.index[results["Ticker"] == tkr][0]
            prices = [v for v in vals.values() if v is not None]
            pmean = float(np.mean(prices)) if prices else np.nan
            pstd = float(np.std(prices)) if len(prices) > 1 else np.nan
            results.loc[
                idx,
                [
                    "Price_Alpha",
                    "Price_Finnhub",
                    "Price_Marketstack",
                    "Price_NasdaqDL",
                    "Price_EODHD",
                    "Price_Mean",
                    "Price_STD",
                    "Source_List",
                ],
            ] = [
                vals.get("Alpha", np.nan),
                vals.get("Finnhub", np.nan),
                # IEX removed
                vals.get("Marketstack", np.nan),
                vals.get("NasdaqDL", np.nan),
                vals.get("EODHD", np.nan),
                pmean,
                pstd,
                " · ".join(srcs),
            ]
    # Price reliability metric
    results["Price_Reliability"] = np.nan
    for i, row in results.iterrows():
        pmean = row.get("Price_Mean", np.nan)
        pstd = row.get("Price_STD", np.nan)
        providers = str(row.get("Source_List", "")).split(" · ") if isinstance(row.get("Source_List"), str) else []
        providers = [p for p in providers if p]
        count = len(providers)
        if np.isfinite(pmean) and pmean > 0 and np.isfinite(pstd):
            pct_std = pstd / pmean
            base = min(1.0, count / 5.0)
            disp = 1.0 / (1.0 + (pct_std / 2.0))  # lower dispersion → higher reliability
            results.at[i, "Price_Reliability"] = round(base * disp, 4)
        else:
            # fallback: minimal reliability if only Yahoo present
            results.at[i, "Price_Reliability"] = min(1.0, count / 8.0)

    # Fundamental reliability metric
    if "Fund_Coverage_Pct" in results.columns:
        results["Fundamental_Reliability"] = 0.0
        for i, row in results.iterrows():
            cov = row.get("Fund_Coverage_Pct", np.nan)
            if not isinstance(cov, (int, float)) or not np.isfinite(cov):
                results.at[i, "Fundamental_Reliability"] = 0.0
                continue
            # Base score from coverage (0-1)
            base_score = float(cov)
            
            # Boost for premium sources
            boost = 0.0
            if bool(row.get("from_fmp_full")):
                boost += 0.05
            if bool(row.get("from_simfin")):
                boost += 0.03
            
            # Count provider diversity
            provider_count = sum([
                bool(row.get("Fund_from_FMP")),
                bool(row.get("Fund_from_Alpha")),
                bool(row.get("Fund_from_Finnhub")),
                bool(row.get("Fund_from_SimFin")),
                bool(row.get("Fund_from_EODHD"))
            ])
            diversity_bonus = min(0.15, provider_count * 0.03)
            
            results.at[i, "Fundamental_Reliability"] = min(1.0, base_score + boost + diversity_bonus)
    else:
        results["Fundamental_Reliability"] = 0.0

    # Combined reliability score
    if "Price_Reliability" in results.columns and "Fundamental_Reliability" in results.columns:
        results["Reliability_Score"] = (
            0.4 * results["Price_Reliability"].fillna(0) + 0.6 * results["Fundamental_Reliability"].fillna(0)
        ).round(4)
    else:
        results["Reliability_Score"] = np.nan
phase_times["מאמת מחירים"] = t_end(t0)

# Horizon heuristic
def infer_horizon(row: pd.Series) -> str:
    rsi_v = row.get("RSI", np.nan)
    near = row.get("Near52w", np.nan)
    score = row.get("Score", 0)
    if (
        isinstance(near, (int, float))
        and near >= 90
        and isinstance(rsi_v, (int, float))
        and 55 <= rsi_v <= 75
    ):
        return "קצר (1–3 חודשים)"
    if isinstance(rsi_v, (int, float)) and rsi_v <= 40 and score >= 60:
        return "בינוני (3–6 חודשים)"
    if score >= 70:
        return "בינוני-ארוך (6–12 חודשים)"
    return "ארוך (6–12 חודשים)"


results["טווח החזקה"] = results.apply(infer_horizon, axis=1)

# Sector cap
def apply_sector_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if not CONFIG["SECTOR_CAP_ENABLED"]:
        return df
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"
    counts: Dict[str, int] = {}
    keep: List[bool] = []
    for _, r in df.iterrows():
        s = r.get("Sector", "Unknown") or "Unknown"
        counts[s] = counts.get(s, 0) + 1
        keep.append(counts[s] <= cap)
    return df[pd.Series(keep).values].reset_index(drop=True)


results = apply_sector_cap(
    results.sort_values(["Score", "Ticker"], ascending=[False, True]).reset_index(
        drop=True
    ),
    int(CONFIG["SECTOR_CAP_MAX"]),
)

# Source badges & unit price
def source_badges(row: pd.Series) -> str:
    """Build badges for all fundamental + price providers present on the row.

    Fundamental flags (merged dict) use: from_fmp_full, from_fmp, from_simfin, from_eodhd, from_alpha, from_finnhub.
    Price providers parsed from Source_List.
    """
    badges: list[str] = []
    # Fundamentals (order of preference)
    if row.get("from_fmp_full") or row.get("from_fmp"):
        badges.append("🟣FMP")
    if row.get("from_simfin"):
        badges.append("🧪SimFin")
    # IEX removed
    if row.get("from_eodhd"):
        badges.append("📘EODHD")
    if row.get("from_alpha"):
        badges.append("🟣Alpha")
    if row.get("from_finnhub"):
        badges.append("🔵Finnhub")

    # Price providers
    price_sources = row.get("Source_List")
    if isinstance(price_sources, str) and price_sources:
        for provider in price_sources.split(" · "):
            if provider and provider not in badges:
                badges.append(provider)
    if not badges:
        badges.append("🟡Yahoo")
    return " · ".join(badges)


results["מקורות מחיר"] = results.apply(source_badges, axis=1)
results["מחיר ממוצע"] = results["Price_Mean"].round(2)
results["סטיית תקן"] = results["Price_STD"].round(4)
results["Unit_Price"] = np.where(
    results["מחיר ממוצע"].notna(), results["מחיר ממוצע"], results["Price_Yahoo"]
)
results["Unit_Price"] = pd.to_numeric(results["Unit_Price"], errors="coerce")

# Allocation
def allocate_budget(
    df: pd.DataFrame, total: float, min_pos: float, max_pos_pct: float, *, score_col: str = "Score"
) -> pd.DataFrame:
    df = df.copy()
    df["סכום קנייה ($)"] = 0.0
    if total <= 0 or df.empty:
        return df
    df = df.sort_values([score_col, "Ticker"], ascending=[False, True]).reset_index(
        drop=True
    )
    remaining = float(total)
    n = len(df)
    max_pos_abs = (max_pos_pct / 100.0) * total if max_pos_pct > 0 else float("inf")
    if min_pos > 0:
        can_min = int(min(n, remaining // min_pos))
        if can_min > 0:
            base = np.full(can_min, min(min_pos, max_pos_abs), dtype=float)
            df.loc[: can_min - 1, "סכום קנייה ($)"] = base
            remaining -= float(base.sum())
    if remaining > 0:
        weights = df[score_col].clip(lower=0).to_numpy(dtype=float)
        extras = (
            np.full(n, remaining / n, dtype=float)
            if np.nansum(weights) <= 0
            else remaining * (np.nan_to_num(weights, nan=0.0) / np.nansum(weights))
        )
        current = df["סכום קנייה ($)"].to_numpy(dtype=float)
        proposed = current + extras
        if np.isfinite(max_pos_abs):
            proposed = np.minimum(proposed, max_pos_abs)
        df["סכום קנייה ($)"] = proposed
    s = float(df["סכום קנייה ($)"].sum())
    if s > 0 and abs(s - total) / max(total, 1) > 1e-6:
        df["סכום קנייה ($)"] = df["סכום קנייה ($)"].to_numpy(dtype=float) * (total / s)
    df["סכום קנייה ($)"] = df["סכום קנייה ($)"].round(2)
    return df


TOPN = min(CONFIG["TOPN_RESULTS"], len(results))
alloc_df = results.head(TOPN).reset_index(drop=True).copy()
# Build allocation score with risk/reliability tilt
risk_mult_core, risk_mult_spec = 1.2, 0.8
if st.session_state.get("alloc_style_idx", 0) == 1:  # Conservative
    risk_mult_core, risk_mult_spec = 1.5, 0.5
elif st.session_state.get("alloc_style_idx", 0) == 2:  # Aggressive
    risk_mult_core, risk_mult_spec = 1.0, 1.2
if "Risk_Level" in alloc_df.columns:
    alloc_df["_risk_mult"] = np.where(alloc_df["Risk_Level"] == "core", risk_mult_core, risk_mult_spec)
else:
    alloc_df["_risk_mult"] = 1.0
if "Reliability_Score" in alloc_df.columns:
    rel = alloc_df["Reliability_Score"].fillna(0.5).clip(0, 1)
    alloc_df["_rel_mult"] = 0.6 + 0.6 * rel
else:
    alloc_df["_rel_mult"] = 1.0
alloc_df["AllocScore"] = alloc_df["Score"].clip(lower=0) * alloc_df["_risk_mult"] * alloc_df["_rel_mult"]

results = allocate_budget(
    alloc_df,
    float(st.session_state.get("total_budget", CONFIG["BUDGET_TOTAL"])),
    float(st.session_state.get("min_position", CONFIG["MIN_POSITION"])),
    float(st.session_state.get("max_position_pct", CONFIG["MAX_POSITION_PCT"])),
    score_col="AllocScore",
)
results["מניות לקנייה"] = np.floor(
    np.where(
        results["Unit_Price"] > 0, results["סכום קנייה ($)"] / results["Unit_Price"], 0
    )
).astype(int)
results["עודף ($)"] = np.round(
    results["סכום קנייה ($)"] - results["מניות לקנייה"] * results["Unit_Price"], 2
)

# KPI
budget_used = float(
    results["מניות לקנייה"].to_numpy() @ results["Unit_Price"].fillna(0).to_numpy()
)
k0, k1, k2, k3 = st.columns(4)
k0.metric("גודל יקום לאחר סינון היסטוריה", len(data_map))
k1.metric("כמות תוצאות אחרי סינון", len(results))
k2.metric("תקציב מנוצל (≈$)", f"{budget_used:,.0f}")
k3.metric("עודף תקציב (≈$)", f"{max(0.0, float(st.session_state.get('total_budget', CONFIG['BUDGET_TOTAL'])) - budget_used):,.0f}")

# Timings
st.subheader("⏱️ זמני ביצוע")
times_df = pd.DataFrame(
    [{"שלב": k, "משך (שניות)": round(v, 2)} for k, v in phase_times.items()]
)
st.table(times_df.style.set_properties(**{"text-align": "center", "direction": "rtl"}))
if alpha_ok:
    st.caption(
        f"Alpha Vantage — קריאות בסשן זה: {int(st.session_state.get('av_calls', 0))} (שמור על קצבים)."
    )

# ==================== Recommendation Cards ====================
st.subheader("🤖 המלצה עכשיו")
st.caption("הכרטיסים הבאים הם **המלצות קנייה** בלבד. אין באמור ייעוץ השקעות.")

# Sidebar filters
with st.sidebar:
    st.header("🎛️ פילטרים")
    st.caption("התאם אישית את תוצאות הסריקה")

    st.markdown("---")
    st.subheader("💰 Allocation")
    total_budget = st.number_input("Total investment ($)", min_value=0.0, value=float(st.session_state.get("total_budget", CONFIG["BUDGET_TOTAL"])), step=100.0)
    st.session_state["total_budget"] = float(total_budget)
    min_position = st.number_input("Min position ($)", min_value=0.0, value=float(st.session_state.get("min_position", max(50.0, round(float(total_budget) * 0.10)))), step=50.0)
    st.session_state["min_position"] = float(min_position)
    max_position_pct = st.slider("Max position (% of total)", min_value=5.0, max_value=60.0, value=float(st.session_state.get("max_position_pct", CONFIG["MAX_POSITION_PCT"])), step=1.0)
    st.session_state["max_position_pct"] = float(max_position_pct)
    alloc_style = st.selectbox("Allocation style", ["Balanced (core tilt)", "Conservative", "Aggressive"], index=int(st.session_state.get("alloc_style_idx", 0)))
    st.session_state["alloc_style_idx"] = ["Balanced (core tilt)", "Conservative", "Aggressive"].index(alloc_style)

    st.markdown("---")
    
    # Risk level filter
    risk_filter = st.multiselect(
        "רמת סיכון",
        options=["core", "speculative"],
        default=["core", "speculative"],
        format_func=lambda x: "🛡️ ליבה (Core)" if x == "core" else "⚡ ספקולטיבי",
        help="בחר אילו סוגי מניות להציג"
    )
    
    # Data quality filter
    quality_filter = st.multiselect(
        "איכות נתונים מינימלית",
        options=["high", "medium", "low"],
        default=["high", "medium", "low"],
        format_func=lambda x: {"high": "✅ גבוהה (85%+)", "medium": "⚠️ בינונית (60-85%)", "low": "❌ נמוכה (<60%)"}[x],
        help="סנן לפי רמת איכות הנתונים"
    )
    
    # Score range
    if not results.empty and "Score" in results.columns:
        min_score_val = float(results["Score"].min())
        max_score_val = float(results["Score"].max())
        # Only show slider if there's a range
        if max_score_val > min_score_val:
            score_range = st.slider(
                "טווח ניקוד",
                min_value=min_score_val,
                max_value=max_score_val,
                value=(min_score_val, max_score_val),
                help="הצג רק מניות בטווח ניקוד זה"
            )
        else:
            st.caption(f"ניקוד: {min_score_val:.1f} (מניה אחת)")
            score_range = (min_score_val, max_score_val)
    else:
        score_range = (0.0, 100.0)
    
    # Sector filter removed - not useful in sidebar at this stage
    sector_filter = []
    
    # RSI filter
    rsi_max = st.slider(
        "RSI מקסימלי",
        min_value=0,
        max_value=100,
        value=80,
        help="סנן מניות עם RSI גבוה מדי (overbought)"
    )

# Apply filters
rec_df = results[results["סכום קנייה ($)"] > 0].copy()

if not rec_df.empty:
    # Apply risk filter
    if risk_filter and "Risk_Level" in rec_df.columns:
        rec_df = rec_df[rec_df["Risk_Level"].isin(risk_filter)]
    
    # Apply quality filter
    if quality_filter and "Data_Quality" in rec_df.columns:
        rec_df = rec_df[rec_df["Data_Quality"].isin(quality_filter)]
    
    # Apply score range
    if "Score" in rec_df.columns:
        rec_df = rec_df[(rec_df["Score"] >= score_range[0]) & (rec_df["Score"] <= score_range[1])]
    
    # Apply sector filter
    if sector_filter and "Sector" in rec_df.columns:
        rec_df = rec_df[rec_df["Sector"].isin(sector_filter)]
    
    # Apply RSI filter
    if "RSI" in rec_df.columns:
        rec_df = rec_df[(rec_df["RSI"].isna()) | (rec_df["RSI"] <= rsi_max)]

st.info(f"📊 מציג {len(rec_df)} מניות אחרי פילטרים")

rec_df = rec_df.copy()

CARD_CSS = """
<style>
.card{direction:rtl;text-align:right;background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;
    padding:14px 16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.05);font-family:system-ui,-apple-system;
    width:100%;box-sizing:border-box}
.card-core{background:#f0fdf4;border:2px solid #86efac}
.card-speculative{background:#fef3c7;border:2px solid #fbbf24}
.badge{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;color:#1e293b;padding:2px 10px;border-radius:999px;font-weight:700;white-space:nowrap}
.badge-quality-high{background:#dcfce7;border:1px solid #22c55e;color:#166534}
.badge-quality-medium{background:#fef3c7;border:1px solid #fbbf24;color:#92400e}
.badge-quality-low{background:#fee2e2;border:1px solid #f87171;color:#991b1b}
.status-buy{display:inline-block;background:#ecfdf5;border:1px solid #34d399;color:#065f46;padding:2px 10px;border-radius:999px;font-weight:700;white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:6px;font-size:.9rem;color:#222}
.item b{color:#111}
@media(max-width:1200px){ .grid{grid-template-columns:repeat(3,minmax(0,1fr));} }
@media(max-width:900px){ .grid{grid-template-columns:repeat(2,minmax(0,1fr));font-size:.85rem} .card h3{font-size:1rem} }
@media(max-width:520px){ .grid{grid-template-columns:repeat(1,minmax(0,1fr));gap:6px} .card{padding:12px 12px} .badge,.status-buy{padding:2px 8px;font-size:.75rem} }
.flex-wrap{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.section-divider{grid-column:1/-1;border-top:1px solid #e5e7eb;padding-top:8px;margin-top:4px;font-weight:600}
.warning-box{background:#fef3c7;border-left:4px solid #f59e0b;padding:10px;margin:10px 0;border-radius:6px;font-size:0.9em}
</style>
"""

if rec_df.empty:
    st.info("אין כרגע מניות שעוברות את הסף עם סכום קנייה חיובי.")
else:
    # Split into Core and Speculative
    if "Risk_Level" in rec_df.columns:
        core_df = rec_df[rec_df["Risk_Level"] == "core"].head(CONFIG["TOPK_RECOMMEND"])
        spec_df = rec_df[rec_df["Risk_Level"] == "speculative"].head(CONFIG["TOPK_RECOMMEND"])
    else:
        # Fallback if Risk_Level column doesn't exist
        core_df = rec_df.head(CONFIG["TOPK_RECOMMEND"])
        spec_df = pd.DataFrame()
    
    # Display Core recommendations first
    if not core_df.empty:
        st.markdown("### 🛡️ מניות ליבה (Core) - סיכון נמוך יחסית")
        st.caption(f"✅ {len(core_df)} מניות עם איכות נתונים גבוהה ופרופיל סיכון מאוזן")
        
        for _, r in core_df.iterrows():
            mean = r.get("מחיר ממוצע", np.nan)
            std = r.get("סטיית תקן", np.nan)
            show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
            show_std = std if not np.isnan(std) else "לא זמין"
            sources = r.get("מקורות מחיר", "לא זמין")
            buy_amt = float(r.get("סכום קנייה ($)", 0.0))
            horizon = r.get("טווח החזקה", "לא זמין")
            rsi_v = r.get("RSI", np.nan)
            near52 = r.get("Near52w", np.nan)
            score = r.get("Score", 0)
            unit_price = r.get("Unit_Price", np.nan)
            shares = int(r.get("מניות לקנייה", 0))
            leftover = r.get("עודף ($)", 0.0)
            rr = r.get("RewardRisk", np.nan)
            atrp = r.get("ATR_Price", np.nan)
            overx = r.get("OverextRatio", np.nan)
            
            # New advanced signals
            rs_63d = r.get("RS_63d", np.nan)
            vol_surge = r.get("Volume_Surge", np.nan)
            ma_aligned = r.get("MA_Aligned", False)
            quality_score = r.get("Quality_Score", 0.0)
            rr_ratio = r.get("RR_Ratio", np.nan)
            mom_consistency = r.get("Momentum_Consistency", 0.0)
            high_confidence = r.get("High_Confidence", False)
            
            # Classification info
            risk_level = r.get("Risk_Level", "core")
            data_quality = r.get("Data_Quality", "medium")
            confidence_level = r.get("Confidence_Level", "medium")
            warnings = r.get("Classification_Warnings", "")
            
            # Data quality badge
            if data_quality == "high":
                quality_badge_class = "badge-quality-high"
                quality_icon = "✅"
                quality_pct = "85%+"
            elif data_quality == "medium":
                quality_badge_class = "badge-quality-medium"
                quality_icon = "⚠️"
                quality_pct = "60-85%"
            else:
                quality_badge_class = "badge-quality-low"
                quality_icon = "❌"
                quality_pct = "<60%"

            show_mean_fmt = f"{np.round(show_mean, 2)}" if not np.isnan(show_mean) else "לא זמין"
            unit_price_fmt = f"{np.round(unit_price, 2)}" if not np.isnan(unit_price) else "לא זמין"
            rr_fmt = f"{rr:.2f}R" if np.isfinite(rr) else "לא זמין"
            atrp_fmt = f"{atrp:.3f}" if np.isfinite(atrp) else "לא זמין"
            overx_fmt = f"{overx:.3f}" if np.isfinite(overx) else "לא זמין"
            
            # Format new signals
            rs_fmt = f"{rs_63d*100:+.1f}%" if np.isfinite(rs_63d) else "לא זמין"
            vol_surge_fmt = f"{vol_surge:.2f}x" if np.isfinite(vol_surge) else "לא זמין"
            ma_status = "✅ מיושר" if ma_aligned else "⚠️ לא מיושר"
            quality_fmt = f"{quality_score:.0f}/50"
            rr_ratio_fmt = f"{rr_ratio:.2f}" if np.isfinite(rr_ratio) else "לא זמין"
            mom_fmt = f"{mom_consistency*100:.0f}%"
            confidence_badge = f"{confidence_level.upper()}" if confidence_level else "MEDIUM"
            
            # Fundamental breakdown
            qual_score_f = r.get("Quality_Score_F", np.nan)
            qual_label = r.get("Quality_Label", "לא זמין")
            growth_score_f = r.get("Growth_Score_F", np.nan)
            growth_label = r.get("Growth_Label", "לא זמין")
            val_score_f = r.get("Valuation_Score_F", np.nan)
            val_label = r.get("Valuation_Label", "לא זמין")
            lev_score_f = r.get("Leverage_Score_F", np.nan)
            lev_label = r.get("Leverage_Label", "לא זמין")
            
            # Format fundamental scores with labels
            qual_fmt = f"{qual_score_f:.0f} ({qual_label})" if np.isfinite(qual_score_f) else "לא זמין"
            growth_fmt = f"{growth_score_f:.0f} ({growth_label})" if np.isfinite(growth_score_f) else "לא זמין"
            val_fmt = f"{val_score_f:.0f} ({val_label})" if np.isfinite(val_score_f) else "לא זמין"
            lev_fmt = f"{lev_score_f:.0f} ({lev_label})" if np.isfinite(lev_score_f) else "לא זמין"
            
            # Color coding for labels
            def label_color(label, good_vals):
                if label in good_vals:
                    return '#16a34a'  # green
                elif label in ['Medium', 'Fair', 'Moderate']:
                    return '#f59e0b'  # orange
                else:
                    return '#dc2626'  # red
            
            qual_color = label_color(qual_label, ['High'])
            growth_color = label_color(growth_label, ['Fast', 'Moderate'])
            val_color = label_color(val_label, ['Cheap', 'Fair'])
            lev_color = label_color(lev_label, ['Low', 'Medium'])

            esc = html_escape.escape
            ticker = esc(str(r["Ticker"]))
            sources_esc = esc(str(sources))
            
            # Next earnings date
            next_earnings = r.get("NextEarnings", "לא ידוע")
            
            card_html = f"""{CARD_CSS}
<div class="card card-core">
    <h3 class="flex-wrap" style="margin:0 0 6px 0">
        <span class="badge">{ticker}</span>
        <span class="status-buy">🛡️ ליבה</span>
        <span class="{quality_badge_class}">{quality_icon} איכות: {quality_pct}</span>
        <span style="background:#3b82f6;color:white;padding:2px 10px;border-radius:999px;font-size:0.8em;white-space:nowrap">רמת ביטחון: {confidence_badge}</span>
    </h3>
    <div class="grid">
    <div class="item"><b>מחיר ממוצע:</b> {show_mean_fmt}</div>
    <div class="item"><b>סטיית תקן:</b> {show_std}</div>
    <div class="item"><b>RSI:</b> {rsi_v if not np.isnan(rsi_v) else 'לא זמין'}</div>
    <div class="item"><b>קרבה לשיא 52ש׳:</b> {near52 if not np.isnan(near52) else 'לא זמין'}%</div>
    <div class="item"><b>ניקוד:</b> {int(round(score))}</div>
    <div class="item"><b>מקורות:</b> {sources_esc.replace(' · ','&nbsp;•&nbsp;')}</div>
    <div class="item"><b>מהימנות מחיר:</b> {r.get('Price_Reliability', np.nan)}</div>
    <div class="item"><b>מהימנות פונד׳:</b> {r.get('Fundamental_Reliability', np.nan)}</div>
    <div class="item"><b>ציון מהימנות:</b> {r.get('Reliability_Score', np.nan)}</div>
    <div class="item"><b>סכום קנייה מומלץ:</b> ${buy_amt:,.0f}</div>
    <div class="item"><b>טווח החזקה:</b> {horizon}</div>
    <div class="item"><b>מחיר יחידה:</b> {unit_price_fmt}</div>
    <div class="item"><b>מניות לקנייה:</b> {shares}</div>
    <div class="item"><b>עודף לא מנוצל:</b> ${leftover:,.2f}</div>
    <div class="item"><b>📅 Earnings הבא:</b> {next_earnings}</div>
    <div class="section-divider">🔬 אינדיקטורים מתקדמים:</div>
    <div class="item"><b>יחס לשוק (3M):</b> <span style="color:{'#16a34a' if np.isfinite(rs_63d) and rs_63d > 0 else '#dc2626'}">{rs_fmt}</span></div>
    <div class="item"><b>עליית נפח:</b> {vol_surge_fmt}</div>
    <div class="item"><b>יישור ממוצעים:</b> {ma_status}</div>
    <div class="item"><b>ציון איכות:</b> {quality_fmt}</div>
    <div class="item"><b>יחס סיכון/תשואה:</b> {rr_ratio_fmt}</div>
    <div class="item"><b>עקביות מומנטום:</b> {mom_fmt}</div>
    <div class="item"><b>ATR/Price:</b> {atrp_fmt}</div>
    <div class="item"><b>Overextension:</b> {overx_fmt}</div>
    <div class="section-divider">💎 פירוט פונדמנטלי:</div>
    <div class="item"><b>איכות:</b> <span style="color:{qual_color};font-weight:600">{qual_fmt}</span></div>
    <div class="item"><b>צמיחה:</b> <span style="color:{growth_color};font-weight:600">{growth_fmt}</span></div>
    <div class="item"><b>שווי:</b> <span style="color:{val_color};font-weight:600">{val_fmt}</span></div>
    <div class="item"><b>מינוף:</b> <span style="color:{lev_color};font-weight:600">{lev_fmt}</span></div>"""
            
            # Add provider attribution if available (Core cards)
            attribution = r.get("Fund_Attribution", "")
            if attribution:
                card_html += f"""
    <div class="item" style="grid-column:span 5;font-size:0.75em;color:#6b7280;border-top:1px solid #e5e7eb;margin-top:4px;padding-top:4px"><b>📊 מקורות נתונים:</b> {esc(attribution)}</div>"""
            
            card_html += """
  </div>
</div>
"""
            # Use a reasonable height to ensure visibility (auto sizing in Streamlit components needs explicit height)
            st_html(card_html, height=560, scrolling=False)
    
    # Display Speculative recommendations
    if not spec_df.empty:
        st.markdown("### ⚡ מניות ספקולטיביות - פוטנציאל גבוה, סיכון גבוה")
        st.caption(f"⚠️ {len(spec_df)} מניות עם פרופיל סיכון גבוה יותר")
        st.warning("🔔 **אזהרה**: מניות אלו מסווגות כספקולטיביות בשל נתונים חלקיים או גורמי סיכון מוגברים. מתאים למשקיעים מנוסים בלבד.")
        
        for _, r in spec_df.iterrows():
            mean = r.get("מחיר ממוצע", np.nan)
            std = r.get("סטיית תקן", np.nan)
            show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
            show_std = std if not np.isnan(std) else "לא זמין"
            sources = r.get("מקורות מחיר", "לא זמין")
            buy_amt = float(r.get("סכום קנייה ($)", 0.0))
            horizon = r.get("טווח החזקה", "לא זמין")
            rsi_v = r.get("RSI", np.nan)
            near52 = r.get("Near52w", np.nan)
            score = r.get("Score", 0)
            unit_price = r.get("Unit_Price", np.nan)
            shares = int(r.get("מניות לקנייה", 0))
            leftover = r.get("עודף ($)", 0.0)
            rr = r.get("RewardRisk", np.nan)
            atrp = r.get("ATR_Price", np.nan)
            overx = r.get("OverextRatio", np.nan)
            
            rs_63d = r.get("RS_63d", np.nan)
            vol_surge = r.get("Volume_Surge", np.nan)
            ma_aligned = r.get("MA_Aligned", False)
            quality_score = r.get("Quality_Score", 0.0)
            rr_ratio = r.get("RR_Ratio", np.nan)
            mom_consistency = r.get("Momentum_Consistency", 0.0)
            
            risk_level = r.get("Risk_Level", "speculative")
            data_quality = r.get("Data_Quality", "low")
            confidence_level = r.get("Confidence_Level", "low")
            warnings = r.get("Classification_Warnings", "")
            
            if data_quality == "high":
                quality_badge_class = "badge-quality-high"
                quality_icon = "✅"
                quality_pct = "85%+"
            elif data_quality == "medium":
                quality_badge_class = "badge-quality-medium"
                quality_icon = "⚠️"
                quality_pct = "60-85%"
            else:
                quality_badge_class = "badge-quality-low"
                quality_icon = "❌"
                quality_pct = "<60%"
            
            show_mean_fmt = f"{np.round(show_mean, 2)}" if not np.isnan(show_mean) else "לא זמין"
            unit_price_fmt = f"{np.round(unit_price, 2)}" if not np.isnan(unit_price) else "לא זמין"
            rr_fmt = f"{rr:.2f}R" if np.isfinite(rr) else "לא זמין"
            atrp_fmt = f"{atrp:.3f}" if np.isfinite(atrp) else "לא זמין"
            overx_fmt = f"{overx:.3f}" if np.isfinite(overx) else "לא זמין"
            rs_fmt = f"{rs_63d*100:+.1f}%" if np.isfinite(rs_63d) else "לא זמין"
            vol_surge_fmt = f"{vol_surge:.2f}x" if np.isfinite(vol_surge) else "לא זמין"
            ma_status = "✅ מיושר" if ma_aligned else "⚠️ לא מיושר"
            quality_fmt = f"{quality_score:.0f}/50"
            rr_ratio_fmt = f"{rr_ratio:.2f}" if np.isfinite(rr_ratio) else "לא זמין"
            mom_fmt = f"{mom_consistency*100:.0f}%"
            confidence_badge = f"{confidence_level.upper()}" if confidence_level else "LOW"
            
            qual_score_f = r.get("Quality_Score_F", np.nan)
            qual_label = r.get("Quality_Label", "לא זמין")
            growth_score_f = r.get("Growth_Score_F", np.nan)
            growth_label = r.get("Growth_Label", "לא זמין")
            val_score_f = r.get("Valuation_Score_F", np.nan)
            val_label = r.get("Valuation_Label", "לא זמין")
            lev_score_f = r.get("Leverage_Score_F", np.nan)
            lev_label = r.get("Leverage_Label", "לא זמין")
            
            qual_fmt = f"{qual_score_f:.0f} ({qual_label})" if np.isfinite(qual_score_f) else "לא זמין"
            growth_fmt = f"{growth_score_f:.0f} ({growth_label})" if np.isfinite(growth_score_f) else "לא זמין"
            val_fmt = f"{val_score_f:.0f} ({val_label})" if np.isfinite(val_score_f) else "לא זמין"
            lev_fmt = f"{lev_score_f:.0f} ({lev_label})" if np.isfinite(lev_score_f) else "לא זמין"
            
            def label_color(label, good_vals):
                if label in good_vals:
                    return '#16a34a'
                elif label in ['Medium', 'Fair', 'Moderate']:
                    return '#f59e0b'
                else:
                    return '#dc2626'
            
            qual_color = label_color(qual_label, ['High'])
            growth_color = label_color(growth_label, ['Fast', 'Moderate'])
            val_color = label_color(val_label, ['Cheap', 'Fair'])
            lev_color = label_color(lev_label, ['Low', 'Medium'])
            
            esc = html_escape.escape
            ticker = esc(str(r["Ticker"]))
            sources_esc = esc(str(sources))
            next_earnings = r.get("NextEarnings", "לא ידוע")
            warnings_esc = esc(warnings) if warnings else ""
            
            card_html = f"""{CARD_CSS}
<div class="card card-speculative">
    <h3 class="flex-wrap" style="margin:0 0 6px 0">
        <span class="badge">{ticker}</span>
        <span style="background:#f59e0b;color:white;padding:2px 10px;border-radius:999px;font-weight:700;white-space:nowrap">⚡ ספקולטיבי</span>
        <span class="{quality_badge_class}">{quality_icon} איכות: {quality_pct}</span>
        <span style="background:#dc2626;color:white;padding:2px 10px;border-radius:999px;font-size:0.8em;white-space:nowrap">רמת ביטחון: {confidence_badge}</span>
    </h3>
    {f'<div class="warning-box"><b>⚠️ אזהרות:</b> {warnings_esc}</div>' if warnings_esc else ''}
    <div class="grid">
    <div class="item"><b>מחיר ממוצע:</b> {show_mean_fmt}</div>
    <div class="item"><b>סטיית תקן:</b> {show_std}</div>
    <div class="item"><b>RSI:</b> {rsi_v if not np.isnan(rsi_v) else 'לא זמין'}</div>
    <div class="item"><b>קרבה לשיא 52ש׳:</b> {near52 if not np.isnan(near52) else 'לא זמין'}%</div>
    <div class="item"><b>ניקוד:</b> {int(round(score))}</div>
    <div class="item"><b>מקורות:</b> {sources_esc.replace(' · ','&nbsp;•&nbsp;')}</div>
    <div class="item"><b>מהימנות מחיר:</b> {r.get('Price_Reliability', np.nan)}</div>
    <div class="item"><b>מהימנות פונד׳:</b> {r.get('Fundamental_Reliability', np.nan)}</div>
    <div class="item"><b>ציון מהימנות:</b> {r.get('Reliability_Score', np.nan)}</div>
    <div class="item"><b>סכום קנייה מומלץ:</b> ${buy_amt:,.0f}</div>
    <div class="item"><b>טווח החזקה:</b> {horizon}</div>
    <div class="item"><b>מחיר יחידה:</b> {unit_price_fmt}</div>
    <div class="item"><b>מניות לקנייה:</b> {shares}</div>
    <div class="item"><b>עודף לא מנוצל:</b> ${leftover:,.2f}</div>
    <div class="item"><b>📅 Earnings הבא:</b> {next_earnings}</div>
    <div class="section-divider">🔬 אינדיקטורים מתקדמים:</div>
    <div class="item"><b>יחס לשוק (3M):</b> <span style="color:{'#16a34a' if np.isfinite(rs_63d) and rs_63d > 0 else '#dc2626'}">{rs_fmt}</span></div>
    <div class="item"><b>עליית נפח:</b> {vol_surge_fmt}</div>
    <div class="item"><b>יישור ממוצעים:</b> {ma_status}</div>
    <div class="item"><b>ציון איכות:</b> {quality_fmt}</div>
    <div class="item"><b>יחס סיכון/תשואה:</b> {rr_ratio_fmt}</div>
    <div class="item"><b>עקביות מומנטום:</b> {mom_fmt}</div>
    <div class="item"><b>ATR/Price:</b> {atrp_fmt}</div>
    <div class="item"><b>Overextension:</b> {overx_fmt}</div>
    <div class="section-divider">💎 פירוט פונדמנטלי:</div>
    <div class="item"><b>איכות:</b> <span style="color:{qual_color};font-weight:600">{qual_fmt}</span></div>
    <div class="item"><b>צמיחה:</b> <span style="color:{growth_color};font-weight:600">{growth_fmt}</span></div>
    <div class="item"><b>שווי:</b> <span style="color:{val_color};font-weight:600">{val_fmt}</span></div>
    <div class="item"><b>מינוף:</b> <span style="color:{lev_color};font-weight:600">{lev_fmt}</span></div>"""
            
            # Add provider attribution if available (Speculative cards)
            attribution_spec = r.get("Fund_Attribution", "")
            if attribution_spec:
                card_html += f"""
    <div class="item" style="grid-column:span 5;font-size:0.75em;color:#6b7280;border-top:1px solid #e5e7eb;margin-top:4px;padding-top:4px"><b>📊 מקורות נתונים:</b> {esc(attribution_spec)}</div>"""
            
            card_html += """
  </div>
</div>
"""
            st_html(card_html, height=560, scrolling=False)

# ==================== Results table + CSV ====================
st.subheader("🎯 תוצאות מסוננות ומדורגות")
view_df_source = rec_df if not rec_df.empty else results

# Augment with sources count if reliability columns present
if "Reliability_Score" in view_df_source.columns and "Source_List" in view_df_source.columns:
    view_df_source["Sources_Count"] = view_df_source["Source_List"].apply(lambda s: len(str(s).split(" · ")) if isinstance(s, str) and s else 0)

hebrew_cols = {
    "Ticker": "טיקר",
    "Price_Yahoo": "מחיר (Yahoo)",
    "Price_Mean": "מחיר ממוצע",
    "Unit_Price": "מחיר יחידה (חישוב)",
    "Price_STD": "סטיית תקן",
    "Source_List": "מקורות מחיר",
    "Price_Reliability": "מהימנות מחיר",
    "Fundamental_Reliability": "מהימנות פונד׳",
    "Reliability_Score": "ציון מהימנות",
    "Sources_Count": "מספר מקורות",
    # "Price_IEX" removed
    "Price_Marketstack": "מחיר Marketstack",
    "Price_NasdaqDL": "מחיר NasdaqDL",
    "Price_EODHD": "מחיר EODHD",
    "Score": "ניקוד",
    "Score_Tech": "ניקוד טכני",
    "Fundamental_S": "ציון פונדמנטלי",
    "Quality_Score_F": "ציון איכות פונד׳",
    "Quality_Label": "תווית איכות",
    "Growth_Score_F": "ציון צמיחה פונד׳",
    "Growth_Label": "תווית צמיחה",
    "Valuation_Score_F": "ציון שווי פונד׳",
    "Valuation_Label": "תווית שווי",
    "Leverage_Score_F": "ציון מינוף פונד׳",
    "Leverage_Label": "תווית מינוף",
    "Sector": "סקטור",
    "RSI": "RSI",
    "Near52w": "קרבה לשיא 52ש׳ (%)",
    "Volx20d": "נפח/ממוצע 20 ימים",
    "OverextRatio": "Overextension מול MA_L",
    "ATR_Price": "ATR/Price",
    "RewardRisk": "Reward/Risk (≈R)",
    "Beta": "בטא",
    "טווח החזקה": "טווח החזקה",
    "סכום קנייה ($)": "סכום קנייה ($)",
    "מניות לקנייה": "מניות לקנייה",
    "עודף ($)": "עודף ($)",
    "PE_f": "P/E",
    "PS_f": "P/S",
    "ROE_f": "ROE",
    "ROIC_f": "ROIC",
    "GM_f": "Margin",
    "DE_f": "Debt/Equity",
    "RevG_f": "Revenue YoY",
    "EPSG_f": "EPS YoY",
    "RS_63d": "יחס לשוק 3M (%)",
    "Volume_Surge": "עליית נפח (x)",
    "MA_Aligned": "ממוצעים מיושרים",
    "Quality_Score": "ציון איכות",
    "RR_Ratio": "סיכון/תשואה",
    "Momentum_Consistency": "עקביות מומנטום (%)",
    "High_Confidence": "ביטחון גבוה",
    "Risk_Level": "רמת סיכון",
    "Data_Quality": "איכות נתונים",
    "Confidence_Level": "רמת ביטחון",
    "Classification_Warnings": "אזהרות"
}
show_order = [
    "טיקר",
    "סקטור",
    "רמת סיכון",
    "איכות נתונים",
    "רמת ביטחון",
    "ציון מהימנות",
    "מהימנות פונד׳",
    "מהימנות מחיר",
    "מספר מקורות",
    "ניקוד",
    "ציון איכות",
    "מחיר ממוצע",
    "מחיר יחידה (חישוב)",
    "סכום קנייה ($)",
    "מניות לקנייה",
    "עודף ($)",
    "מקורות מחיר",
    "מחיר Marketstack",
    "מחיר NasdaqDL",
    "מחיר EODHD",
    "ניקוד טכני",
    "ציון פונדמנטלי",
    "RSI",
    "יחס לשוק 3M (%)",
    "עליית נפח (x)",
    "ממוצעים מיושרים",
    "סיכון/תשואה",
    "עקביות מומנטום (%)",
    "קרבה לשיא 52ש׳ (%)",
    "נפח/ממוצע 20 ימים",
    "Overextension מול MA_L",
    "ATR/Price",
    "Reward/Risk (≈R)",
    "בטא",
    "טווח החזקה",
    "P/E",
    "P/S",
    "ROE",
    "ROIC",
    "Margin",
    "Debt/Equity",
    "Revenue YoY",
    "EPS YoY",
]
csv_df = view_df_source.rename(columns=hebrew_cols)
# Ensure column names unique after rename (pandas JSON export requires uniqueness)
def _make_unique(names):
    counts = {}
    out = []
    for n in names:
        if n not in counts:
            counts[n] = 1
            out.append(n)
        else:
            counts[n] += 1
            out.append(f"{n}_{counts[n]}")
    return out
csv_df.columns = _make_unique(list(csv_df.columns))
# Build unique ordered columns for export referencing updated names
cols_for_export = []
seen_cols = set()
for c in show_order:
    # select first matching column (since duplicates now suffixed)
    matches = [col for col in csv_df.columns if col == c or col.startswith(f"{c}_")]
    if matches:
        first = matches[0]
        if first not in seen_cols:
            cols_for_export.append(first)
            seen_cols.add(first)
csv_bytes = (
    csv_df[cols_for_export]
    .to_csv(index=False)
    .encode("utf-8-sig")
)

# Download buttons side by side
col_csv, col_json = st.columns(2)
with col_csv:
    st.download_button(
        "📥 הורדת תוצאות ל-CSV",
        data=csv_bytes,
        file_name=f"stock_scout_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
with col_json:
    # JSON export for API/automation
    json_data = csv_df[cols_for_export].to_json(
        orient="records", force_ascii=False, indent=2
    )
    st.download_button(
        "📊 הורדת תוצאות ל-JSON",
        data=json_data.encode("utf-8"),
        file_name=f"stock_scout_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

st.markdown('<div class="rtl-table">', unsafe_allow_html=True)
st.dataframe(
    csv_df[[c for c in show_order if c in csv_df.columns]],
    use_container_width=True,
    hide_index=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ==================== Quick chart ====================
st.subheader("🔍 גרף טיקר מהתוצאות")
choices = ["(בחר)"] + view_df_source["Ticker"].astype(str).tolist()
choice = st.selectbox("בחר טיקר", choices, index=0)
if choice and choice != "(בחר)" and choice in data_map:
    dfv = data_map[choice].copy()
    dfv["MA_S"] = dfv["Close"].rolling(int(CONFIG["MA_SHORT"])).mean()
    dfv["MA_L"] = dfv["Close"].rolling(int(CONFIG["MA_LONG"])).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=dfv.index,
            open=dfv["Open"],
            high=dfv["High"],
            low=dfv["Low"],
            close=dfv["Close"],
            name="מחיר",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dfv.index,
            y=dfv["MA_S"],
            mode="lines",
            name=f"MA{int(CONFIG['MA_SHORT'])}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dfv.index, y=dfv["MA_L"], mode="lines", name=f"MA{int(CONFIG['MA_LONG'])}"
        )
    )
    fig.update_layout(
        height=480, xaxis_rangeslider_visible=False, legend_orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)
    dfv["RSI"] = rsi(dfv["Close"], 14)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dfv.index, y=dfv["RSI"], mode="lines", name="RSI14"))
    fig2.add_hrect(
        y0=CONFIG["RSI_BOUNDS"][0],
        y1=CONFIG["RSI_BOUNDS"][1],
        fillcolor="LightGreen",
        opacity=0.2,
        line_width=0,
    )
    fig2.update_layout(height=220, legend_orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

# ==================== Notes ====================
with st.expander("ℹ️ מתודולוגיה (תקציר)"):
    st.markdown(
        """
- היסטוריה: **Yahoo Finance** (`yfinance`). אימות מחירים: **Alpha**, **Finnhub**, **Polygon**, **Tiingo**.
- ניקוד טכני: MA, מומנטום 1/3/6 חו׳ (Sigmoid), RSI בטווח, **Near-High bell**, **Overextension מול MA_L**, **Pullback**,
  **ATR/Price**, **Reward/Risk**, **MACD/ADX**.
- פונדמנטלי: **Alpha OVERVIEW** עם **Fallback ל-Finnhub** (Growth/Quality/Valuation + ענישת מינוף גבוה).
- חוקים: מינ׳ דולר-ווליום, תקרות ATR/Price ו-Overextension (פסילות קשיחות), **earnings blackout**, **beta filter**, **sector cap**,
  והקצאת תקציב עם תקרת פוזיציה מקסימלית.
"""
    )
