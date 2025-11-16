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
• LTR English UI, recommendation cards, CSV export, quick chart.
Note: This is not investment advice.
"""

from __future__ import annotations
import os
import time
import logging
import warnings
import pickle
from pathlib import Path
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
from card_styles import get_card_css
from core.portfolio import _normalize_weights, allocate_budget
from core.config import get_config
from advanced_filters import compute_advanced_score, should_reject_ticker, fetch_benchmark_data
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters,
    score_with_ml_model,
    compute_technical_score,
    compute_final_score,
)
from core.classification import apply_classification, filter_core_recommendations
from core.scoring.fundamental import compute_fundamental_score_with_breakdown
from indicators import rsi, atr, macd_line, adx, _sigmoid

# Get global configuration
_config_obj = get_config()
CONFIG = {
    # Convert dataclass to dict with all attributes
    **{k: getattr(_config_obj, k) for k in dir(_config_obj) if not k.startswith('_') and not callable(getattr(_config_obj, k))},
    # Additional backwards-compatible keys
    "SMART_SCAN": _config_obj.smart_scan,
    "UNIVERSE_LIMIT": _config_obj.universe_limit,
    "LOOKBACK_DAYS": _config_obj.lookback_days,
    "MA_SHORT": _config_obj.ma_short,
    "MA_LONG": _config_obj.ma_long,
    "RSI_BOUNDS": _config_obj.rsi_bounds,
    "WEIGHTS": _config_obj.weights,
    # Indicator toggles and thresholds
    "USE_MACD_ADX": _config_obj.use_macd_adx,
    "OVEREXT_SOFT": _config_obj.overext_soft,
    "OVEREXT_HARD": _config_obj.overext_hard,
    "PULLBACK_RANGE": _config_obj.pullback_range,
    "ATR_PRICE_HARD": _config_obj.atr_price_hard,
    # Basic price/volume constraints
    "MIN_PRICE": _config_obj.min_price,
    "MIN_AVG_VOLUME": _config_obj.min_avg_volume,
    "MIN_DOLLAR_VOLUME": _config_obj.min_dollar_volume,
    "MIN_MARKET_CAP": _config_obj.min_market_cap,
    "FUNDAMENTAL_ENABLED": _config_obj.fundamental_enabled,
    "FUNDAMENTAL_WEIGHT": _config_obj.fundamental_weight,
    # Earnings / risk filters
    "EARNINGS_BLACKOUT_DAYS": _config_obj.earnings_blackout_days,
    "EARNINGS_CHECK_TOPK": _config_obj.earnings_check_topk,
    "BETA_FILTER_ENABLED": _config_obj.beta_filter_enabled,
    "BETA_TOP_K": _config_obj.beta_top_k,
    "BETA_MAX_ALLOWED": _config_obj.beta_max_allowed,
    "SECTOR_CAP_MAX": _config_obj.sector_cap_max,
    "SECTOR_CAP_ENABLED": _config_obj.sector_cap_enabled,
    # Results sizing
    "TOPN_RESULTS": _config_obj.topn_results,
    "TOPK_RECOMMEND": _config_obj.topk_recommend,
    # Allocation / budget
    "BUDGET_TOTAL": _config_obj.budget_total,
    "MIN_POSITION": _config_obj.min_position,
    "MAX_POSITION_PCT": _config_obj.max_position_pct,
    "EXTERNAL_PRICE_VERIFY": _config_obj.external_price_verify,
    "TOP_VALIDATE_K": _config_obj.top_validate_k,
    "BETA_BENCHMARK": _config_obj.beta_benchmark,
    # Disabled providers (legacy/deprecated/paid):
    "ENABLE_SIMFIN": False,      # SimFin API v2 deprecated, v3 requires paid subscription
    "ENABLE_MARKETSTACK": False,  # Monthly usage limit reached (free tier exhausted)
    "ENABLE_NASDAQ_DL": False,    # API access blocked (403 Forbidden)
    "ENABLE_EODHD": False,        # Requires paid subscription (402 Payment Required)
}

# Load environment variables
warnings.simplefilter("ignore", FutureWarning)

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(usecwd=True))
MODEL_DATA = None
try:
    _base = Path(__file__).parent
    _cand_cal = _base / "model_xgboost_5d_calibrated.pkl"
    _cand_fb = _base / "model_xgboost_5d.pkl"
    _model_path = _cand_cal if _cand_cal.exists() else (_cand_fb if _cand_fb.exists() else None)
    if _model_path is not None:
        with open(_model_path, 'rb') as _f:
            MODEL_DATA = pickle.load(_f)
            try:
                st.info(f"✓ Loaded ML model: {_model_path.name} (features: {len(MODEL_DATA.get('feature_names', []))})")
            except Exception:
                pass
    else:
        st.info("ML model not found; ML scoring will be neutral.")
except Exception as _e:
    st.warning(f"Could not load ML model: {_e}")

# Backwards-compatible ML objects and helpers for existing scoring flow
XGBOOST_MODEL = None
XGBOOST_FEATURES = []
if MODEL_DATA is not None:
    try:
        XGBOOST_MODEL = MODEL_DATA.get('model')
        XGBOOST_FEATURES = MODEL_DATA.get('feature_names', [])
    except Exception:
        XGBOOST_MODEL = None


def score_with_xgboost(row: pd.Series) -> float:
    """Compatibility wrapper used in older code paths. Returns 0.5 when model unavailable."""
    if XGBOOST_MODEL is None:
        return 0.5
    # Build features mapping consistent with training
    features = {}
    for fname in XGBOOST_FEATURES:
        # Use available columns or fallbacks
        if fname in row:
            features[fname] = row.get(fname)
        else:
            # sensible defaults
            if 'RSI' in fname:
                features[fname] = row.get('RSI', 50)
            elif 'ATR' in fname or 'ATR_Pct' in fname:
                features[fname] = row.get('ATR_Pct', 0.05)
            elif 'Vol' in fname:
                features[fname] = row.get('Volx20d', 1.0)
            else:
                features[fname] = row.get(fname, 0.0)

    X = pd.DataFrame([features])[XGBOOST_FEATURES]
    X = X.fillna(X.median())
    try:
        if hasattr(XGBOOST_MODEL, 'predict_proba'):
            return float(XGBOOST_MODEL.predict_proba(X.values)[0][1])
        else:
            return float(XGBOOST_MODEL.predict_proba(X)[0][1])
    except Exception:
        return 0.5


def assign_confidence_tier(prob: float) -> str:
    if not isinstance(prob, (int, float)) or not np.isfinite(prob):
        return "N/A"
    if prob >= 0.50:
        return "High"
    if prob >= 0.30:
        return "Medium"
    return "Low"


# ==================== Environment helper ====================
def _env(key: str) -> Optional[str]:
    """Get environment variable or Streamlit secret (supports nested sections)."""
    # Try Streamlit secrets with nested sections
    try:
        if hasattr(st, 'secrets'):
            secrets_obj = st.secrets
            # direct top-level
            if isinstance(secrets_obj, dict) and key in secrets_obj:
                return secrets_obj[key]
            # common nested containers
            for section in ("api_keys", "keys", "secrets", "tokens"):
                try:
                    container = secrets_obj.get(section) if hasattr(secrets_obj, 'get') else secrets_obj[section]
                    if isinstance(container, dict) and key in container:
                        return container[key]
                except Exception:
                    continue
    except Exception:
        pass
    # Fallback to environment
    return os.getenv(key)


# ==================== HTTP helpers ====================
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
    """HTTP GET with exponential backoff + full jitter."""
    import random
    sess = session or requests
    for attempt in range(1, max(1, tries) + 1):
        try:
            resp = sess.get(url, timeout=timeout, headers=headers)
            if resp is not None and resp.status_code == 200:
                return resp
            if resp is not None and (
                resp.status_code == 429 or (500 <= resp.status_code < 600)
            ):
                _log.debug(f"HTTP {resp.status_code} -> retry attempt {attempt}/{tries}")
            else:
                return resp
        except requests.RequestException as exc:
            _log.debug(f"Request exception on attempt {attempt}/{tries}: {exc}")
        if attempt < tries:
            backoff = min(max_backoff, backoff_base * (2 ** (attempt - 1)))
            sleep_time = random.uniform(0, backoff)
            time.sleep(sleep_time)
    _log.warning(f"All {tries} attempts failed for URL")
    return None


def alpha_throttle(min_gap_seconds: float = 12.0) -> None:
    """Throttle Alpha Vantage calls."""
    ts_key = "_alpha_last_call_ts"
    last = st.session_state.get(ts_key, 0.0)
    now = time.time()
    elapsed = now - last
    if elapsed < min_gap_seconds:
        time.sleep(min_gap_seconds - elapsed)
    st.session_state[ts_key] = time.time()


# --- Build Universe (restored) ---
def build_universe(limit: int) -> List[str]:
    """Fetch S&P 500 tickers (wikipedia) then fallback to common mega-cap list.
    Limit result length to `limit`."""
    try:
        # Wikipedia requires a User-Agent header
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            storage_options={'User-Agent': 'Mozilla/5.0'}
        )
        # S&P500 table is typically the second table (index 1)
        df_sp = tables[1]
        tickers = df_sp['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
        logger.info(f"✓ Loaded {len(tickers)} S&P500 tickers from Wikipedia")
        return tickers[:limit]
    except Exception as e:
        logger.warning(f"Wikipedia S&P500 fetch failed ({e}), using fallback list")
        fallback = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","WMT","UNH","AVGO"]
        return fallback[:limit]


# ==================== Universe & data ====================
def safe_yf_download(
    tickers: List[str], start: datetime, end: datetime
) -> Dict[str, pd.DataFrame]:
    """Download with fallback for single tickers."""
    out: Dict[str, pd.DataFrame] = {}
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker" if len(tickers) > 1 else None,
            threads=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    df = data[ticker].dropna()
                    if not df.empty:
                        out[ticker] = df
                except Exception:
                    continue
        elif not data.empty:
            out[tickers[0]] = data.dropna()
    except Exception:
        pass
    
    # Download missing individually
    missing = [t for t in tickers if t not in out]
    for ticker in missing:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            ).dropna()
            if not df.empty:
                out[ticker] = df
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=True, ttl=60 * 60 * 4)  # 4 hours - history changes slowly
def fetch_history_bulk(
    tickers: List[str], period_days: int, ma_long: int
) -> Dict[str, pd.DataFrame]:
    """Fetch bulk historical data with sufficient lookback for moving averages."""
    end = datetime.utcnow()
    start = end - timedelta(days=period_days + 50)
    data_map = safe_yf_download(tickers, start, end)
    
    # Filter: need at least ma_long + 40 rows (relaxed from +50 to handle weekends/holidays)
    min_rows = ma_long + 40
    filtered = {}
    for tkr, df in data_map.items():
        if len(df) >= min_rows:
            filtered[tkr] = df
    return filtered


# ==================== Earnings ====================
@st.cache_data(ttl=60 * 60)
def get_next_earnings_date(ticker: str) -> Optional[datetime]:
    """Get next earnings date from Finnhub → yfinance fallback."""
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
    """Batch fetch earnings dates in parallel."""
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
        f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={k}",
        tries=1,
        timeout=6,
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
        f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={k}",
        tries=1,
        timeout=6,
    )
    if not r:
        return False, "Timeout"
    try:
        j = r.json()
    except Exception:
        return False, "Bad JSON"
    ok = (
        isinstance(j, dict)
        and "results" in j
        and isinstance(j["results"], list)
        and j["results"]
    )
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


# ==================== Fundamentals (Alpha → Finnhub) ====================
def _to_01(x: float, low: float, high: float) -> float:
    if not isinstance(x, (int, float)) or not np.isfinite(x):
        return np.nan
    return np.clip((x - low) / (high - low), 0, 1)


@st.cache_data(ttl=60 * 60 * 24)  # 24h cache for fundamentals
def fetch_fundamentals_bundle(ticker: str, enable_alpha_smart: bool = False) -> dict:
    """Fetch fundamentals from multiple providers and merge into a single dict (parallel).

    This function runs all configured fundamentals providers in parallel using
    a ThreadPoolExecutor instead of a slow sequential approach. Typical runtime
    savings: ~60-70% per ticker depending on enabled providers.

    Merge priority: FMP → SimFin → Alpha (smart) → Finnhub → EODHD → Tiingo
    `enable_alpha_smart`: if True, uses Alpha Vantage (recommended only for top picks)

    Returns a dict with the merged fields plus source flags, `_sources` attribution
    and `Fund_Coverage_Pct`.
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
        base = "https://financialmodelingprep.com/stable"
        endpoints = {
            "profile": f"{base}/profile?symbol={ticker}&apikey={api_key}",
            "key_metrics": f"{base}/key-metrics?symbol={ticker}&period=annual&limit=1&apikey={api_key}",
            "ratios_ttm": f"{base}/ratios-ttm?symbol={ticker}&apikey={api_key}",
            "growth": f"{base}/financial-growth?symbol={ticker}&period=annual&limit=1&apikey={api_key}",
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
    """Fetch Alpha Vantage OVERVIEW (simple overview fields)."""
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
    """Fallback to Finnhub metrics + sector information."""
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
    """Fetch sector information from Finnhub (profile2 endpoint)."""
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
from design_system import get_modern_css

st.set_page_config(page_title="Asaf's Stock Scout — 2025", page_icon="📈", layout="wide")
st.markdown(get_modern_css(), unsafe_allow_html=True)

st.title("📈 Stock Scout — 2025 (Auto)")

# Relaxed mode toggle: allows looser filters (momentum-first)
RELAXED_MODE = st.checkbox(
    "Relaxed Mode (Momentum-first) — allow looser filters",
    value=False,
    help="When enabled, speculative/relaxed filters are preferred; ML still applies but filters are looser.",
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
        "source": name,
        "price": "🟢" if price_ok else "🔴",
        "fundamentals": "🟢" if fund_ok else "🔴",
        "reason": reason
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
add_provider("SimFin", False, False, "API Deprecated")
add_provider("Marketstack", False, False, "Usage Limit Reached")
add_provider("NasdaqDL", False, False, "Access Blocked")
add_provider("EODHD", False, False, "Paid Subscription Required")

status_df = pd.DataFrame(providers_status)
st.markdown("### 🔌 Data Sources Status")
_provider_css = """
<style>
.provider-table-container {overflow-x:auto;}
.provider-table-container table {width:100%; border-collapse:collapse; table-layout:fixed;}
.provider-table-container th, .provider-table-container td {padding:6px 8px; text-align:center; font-size:14px;}
.provider-table-container th:nth-child(1), .provider-table-container td:nth-child(1){text-align:center; min-width:130px;}
@media (max-width:600px){
    .provider-table-container th, .provider-table-container td {padding:4px 6px; font-size:12px;}
    .provider-table-container th:nth-child(1), .provider-table-container td:nth-child(1){min-width:140px;}
}
@media (max-width:480px){
    /* Stack rows into cards, hide header and Reason column */
    .provider-table-container table, .provider-table-container thead, .provider-table-container tbody, .provider-table-container th, .provider-table-container td, .provider-table-container tr {display:block;}
    .provider-table-container tr {margin:0 0 10px 0; border:1px solid #e5e7eb; border-radius:8px; padding:6px; background:#ffffff;}
    .provider-table-container th {display:none;}
    .provider-table-container td {text-align:right; padding:4px 8px; font-size:13px; border:none;}
    .provider-table-container td:nth-child(4){display:none;} /* hide Reason for compact view */
    .provider-table-container td::before {font-weight:600; display:inline-block; margin-left:4px;}
    .provider-table-container td:nth-child(1)::before {content:'Source: ';}
    .provider-table-container td:nth-child(2)::before {content:'Price: ';}
    .provider-table-container td:nth-child(3)::before {content:'Fund: ';}
}
</style>
"""
st.markdown(_provider_css, unsafe_allow_html=True)
html_rows = []
html_rows.append("<tr><th>Source</th><th>Price</th><th>Fundamentals</th><th>Reason</th></tr>")
for r in providers_status:
        html_rows.append(
                f"<tr><td>{r['source']}</td><td>{r['price']}</td><td>{r['fundamentals']}</td><td>{r['reason']}</td></tr>"
        )
providers_html = "<div class='provider-table-container'><table>" + "".join(html_rows) + "</table></div>"
st.markdown(providers_html, unsafe_allow_html=True)

# Utility buttons row
col_secrets, col_cache, _ = st.columns([1, 1, 3])
with col_secrets:
    if st.button("Check Secrets 🔐"):
        st.info(
            f"Alpha: {_mask(_env('ALPHA_VANTAGE_API_KEY'))}\\n\\n"
            f"Finnhub: {_mask(_env('FINNHUB_API_KEY'))}\\n\\n"
            f"Polygon: {_mask(_env('POLYGON_API_KEY'))}\\n\\n"
            f"Tiingo: {_mask(_env('TIINGO_API_KEY'))}\\n\\n"
            f"FMP: {_mask(_env('FMP_API_KEY'))}"
        )

# Cache reset button
with col_cache:
    if st.button("🔄 Clear Cache & Reload"):
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
phase_times["build_universe"] = t_end(t0)

# 2) History
t0 = t_start()
data_map = fetch_history_bulk(universe, CONFIG["LOOKBACK_DAYS"], CONFIG["MA_LONG"])
phase_times["fetch_history"] = t_end(t0)

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
phase_times["calc_score_technical"] = t_end(t0)
if results.empty:
    st.warning("No results after filtering. Filters may be too strict for the current universe.")
    st.stop()
sort_col = 'Final_Score' if 'Final_Score' in results.columns else 'Score_Tech'
results = results.sort_values([sort_col, "Ticker"], ascending=[False, True]).reset_index(drop=True)

# 3a) Initialize fundamental columns (will populate after advanced_filters)
fundamental_available = (
    alpha_ok or finn_ok or fmp_ok or bool(simfin_key) or bool(eodhd_key)
)
if CONFIG["FUNDAMENTAL_ENABLED"] and fundamental_available:
    # Initialize columns - will fetch data AFTER advanced_filters
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
    results["Score"] = results.get('Final_Score', results["Score_Tech"])  # unified final score or fallback
else:
    results["Score"] = results.get('Final_Score', results["Score_Tech"])

# Earnings blackout
if CONFIG["EARNINGS_BLACKOUT_DAYS"] > 0:
    to_check_idx = list(results.head(int(CONFIG["EARNINGS_CHECK_TOPK"])).index)
    symbols = [results.at[i, "Ticker"] for i in to_check_idx]
    ed_map = _earnings_batch(symbols)
    now_utc = datetime.utcnow().replace(tzinfo=None)
    keep_mask = np.ones(len(results), dtype=bool)
    for idx in to_check_idx:
        tkr = results.at[idx, "Ticker"]
        dt_earn = ed_map.get(tkr)
        if dt_earn is None:
            continue
        # Ensure dt_earn is timezone-naive
        if hasattr(dt_earn, 'tzinfo') and dt_earn.tzinfo is not None:
            dt_earn = dt_earn.replace(tzinfo=None)
        gap_days = abs((dt_earn - now_utc).days)
        if gap_days <= int(CONFIG["EARNINGS_BLACKOUT_DAYS"]):
            keep_mask[idx] = False
            results.at[idx, "EarningsNote"] = f"Excluded: earnings within {gap_days}d"
    results = results[keep_mask].reset_index(drop=True)
    if results.empty:
        st.warning('All top-K candidates were excluded due to the earnings blackout window.')
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
    phase_times["beta_filter"] = t_end(t0)

# 3c) Advanced Filters (dynamic penalty approach)
t0 = t_start()
st.info("🔬 Running advanced filters (dynamic penalties)...")

benchmark_df = fetch_benchmark_data(CONFIG["BETA_BENCHMARK"], CONFIG["LOOKBACK_DAYS"])

adv_cols = [
    "RS_63d", "Volume_Surge", "MA_Aligned", "Quality_Score", "RR_Ratio",
    "Momentum_Consistency", "High_Confidence", "AdvPenalty", "AdvFlags"
]
for col in adv_cols:
    if col in ["MA_Aligned", "High_Confidence"]:
        results[col] = False
    elif col == "AdvFlags":
        results[col] = ""
    else:
        results[col] = np.nan

logger.info(f"🔬 Advanced filters pre-pass on {len(results)} stocks...")

signals_store = []  # (idx, signals, enhanced_score, catastrophic, reason)
for idx in results.index:
    tkr = results.at[idx, "Ticker"]
    if tkr not in data_map or benchmark_df.empty:
        signals_store.append((idx, {}, results.at[idx, "Score"], False, ""))
        continue
    df = data_map[tkr]
    base_score = results.at[idx, "Score"]
    enhanced_score, signals = compute_advanced_score(tkr, df, benchmark_df, base_score)
    catastrophic, reason = should_reject_ticker(signals)
    signals_store.append((idx, signals, enhanced_score, catastrophic, reason))

rs_vals = []
mom_vals = []
rr_vals = []
for (_, sig, _, _, _) in signals_store:
    v_rs = sig.get("rs_63d")
    if isinstance(v_rs, (int, float)) and np.isfinite(v_rs):
        rs_vals.append(v_rs)
    v_mom = sig.get("momentum_consistency")
    if isinstance(v_mom, (int, float)) and np.isfinite(v_mom):
        mom_vals.append(v_mom)
    v_rr = sig.get("risk_reward_ratio")
    if isinstance(v_rr, (int, float)) and np.isfinite(v_rr):
        rr_vals.append(v_rr)

def _q(vals, q, default):
    return float(np.quantile(vals, q)) if vals else default

rs_thresh_dyn = _q(rs_vals, 0.15, -0.15)
mom_thresh_dyn = _q(mom_vals, 0.20, 0.25)
rr_thresh_dyn = _q(rr_vals, 0.25, 0.90)
logger.info(f"Dynamic thresholds -> RS:{rs_thresh_dyn:.3f} MOM:{mom_thresh_dyn:.3f} RR:{rr_thresh_dyn:.3f}")

catastrophic_count = 0
kept = 0
for (idx, sig, enhanced_score, catastrophic, reason) in signals_store:
    flags = []
    penalty = 0.0
    if sig:
        rs_val = sig.get("rs_63d", np.nan)
        mom_val = sig.get("momentum_consistency", 0.0)
        rr_val = sig.get("risk_reward_ratio", np.nan)
        if np.isfinite(rs_val) and rs_val < rs_thresh_dyn:
            penalty += 8.0
            flags.append("LowRS")
        if mom_val < mom_thresh_dyn:
            penalty += 6.0
            flags.append("WeakMomentum")
        if np.isfinite(rr_val) and rr_val < rr_thresh_dyn:
            penalty += 10.0
            flags.append("LowRR")
    if catastrophic:
        catastrophic_count += 1
        results.loc[idx, "RejectionReason"] = reason
    results.loc[idx, "RS_63d"] = sig.get("rs_63d", np.nan)
    results.loc[idx, "Volume_Surge"] = sig.get("volume_surge", np.nan)
    results.loc[idx, "MA_Aligned"] = sig.get("ma_aligned", False)
    results.loc[idx, "Quality_Score"] = sig.get("quality_score", 0.0)
    results.loc[idx, "RR_Ratio"] = sig.get("risk_reward_ratio", np.nan)
    results.loc[idx, "Momentum_Consistency"] = sig.get("momentum_consistency", 0.0)
    results.loc[idx, "High_Confidence"] = sig.get("high_confidence", False)
    results.loc[idx, "AdvPenalty"] = penalty
    results.loc[idx, "AdvFlags"] = ",".join(flags)
    adj_score = max(0.0, enhanced_score - penalty)
    results.loc[idx, "Score"] = adj_score
    if not catastrophic:
        kept += 1

if catastrophic_count == len(signals_store):
    logger.warning("All stocks met catastrophic rejection; overriding to keep all for inspection.")
    kept = len(signals_store)

logger.info(f"Advanced filters dynamic: kept {kept}/{len(signals_store)} catastrophic={catastrophic_count}")

if catastrophic_count > 0 and catastrophic_count < len(signals_store):
    drop_indices = [idx for (idx, _, _, c, _) in signals_store if c]
    results = results[~results.index.isin(drop_indices)].reset_index(drop=True)

results = results.sort_values(["Score", "Ticker"], ascending=[False, True]).reset_index(drop=True)
phase_times["advanced_filters"] = t_end(t0)

if results.empty:
    st.warning("Advanced filters produced empty set even after penalties.")
    st.stop()

# 3d) Fetch Fundamentals for stocks that passed advanced_filters
if CONFIG["FUNDAMENTAL_ENABLED"] and fundamental_available:
    t0 = t_start()
    st.info(f"📊 Fetching fundamentals for {len(results)} stocks that passed advanced filters...")
    
    # Fetch for all stocks that passed advanced_filters (typically 15-25)
    take_k = len(results)
    
    for idx in results.index:
        tkr = results.at[idx, "Ticker"]
        rank = list(results.index).index(idx) + 1  # 1-based rank
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
    
    # Mix technical + fundamental scores
    results["Score"] = (
        1 - float(CONFIG["FUNDAMENTAL_WEIGHT"])
    ) * results["Score_Tech"] + float(
        CONFIG["FUNDAMENTAL_WEIGHT"]
    ) * results["Fundamental_S"].fillna(0)
    
    results = results.sort_values(
        ["Score", "Ticker"], ascending=[False, True]
    ).reset_index(drop=True)
    phase_times["fundamentals_alpha_finnhub"] = t_end(t0)
    logger.info(f"✓ Fetched fundamentals for {take_k} stocks")

# 3e) Apply risk classification and data quality evaluation
t0 = t_start()
st.info("🔍 Classifying stocks by risk level and data quality...")

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
            st.warning(f"⚠️ Low fundamentals coverage: mean {avg_cov:.2f} ({pct_good*100:.1f}% with ≥50% fields). Consider adding sources or API keys.")
        else:
            st.info(f"🧬 Avg fundamentals coverage: {avg_cov:.2f} | {pct_good*100:.0f}% of tickers have ≥50% fields.")

# Show classification statistics
core_count = len(results[results["Risk_Level"] == "core"])
spec_count = len(results[results["Risk_Level"] == "speculative"])
high_qual = len(results[results["Data_Quality"] == "high"])
med_qual = len(results[results["Data_Quality"] == "medium"])
low_qual = len(results[results["Data_Quality"] == "low"])

st.write(f"**Initial classification:** {core_count} Core, {spec_count} Speculative | "
         f"**Quality:** {high_qual} high, {med_qual} medium, {low_qual} low")

# Filter out stocks that shouldn't be displayed (very low quality)
displayable = results[results["Should_Display"]].copy()
hidden_count = len(results) - len(displayable)
if hidden_count > 0:
    st.write(f"🔻 Hidden {hidden_count} stocks due to very low data quality")
    logger.info(f"Hidden {hidden_count} stocks due to very low data quality")

results = displayable.reset_index(drop=True)

# Split and filter Core vs Speculative separately
core_stocks = results[results["Risk_Level"] == "core"].copy()
spec_stocks = results[results["Risk_Level"] == "speculative"].copy()

st.write(f"📊 **Before filtering:** {len(core_stocks)} Core, {len(spec_stocks)} Speculative")

# Filter Core with strict criteria
core_before_filter = len(core_stocks)
core_filtered = filter_core_recommendations(core_stocks, CONFIG, adaptive=True) if not core_stocks.empty else pd.DataFrame()
core_after_filter = len(core_filtered)

if core_before_filter > 0:
    st.write(f"🛡️ **Core filter:** {core_before_filter} → {core_after_filter} passed strict filters")

# Filter Speculative with relaxed criteria (allow higher volatility, missing some fundamentals)
spec_before_filter = len(spec_stocks)
if not spec_stocks.empty:
    # Apply looser technical filters for speculative
    spec_filtered = spec_stocks.copy()
    
    # Filter 1: RSI bounds (wider range)
    if "RSI" in spec_filtered.columns:
        rsi_min = max(CONFIG.get("RSI_MIN_CORE", 40) - 10, 25)
        rsi_max = min(CONFIG.get("RSI_MAX_CORE", 75) + 10, 85)
        spec_filtered = spec_filtered[
            (spec_filtered["RSI"].isna()) | 
            ((spec_filtered["RSI"] >= rsi_min) & (spec_filtered["RSI"] <= rsi_max))
        ]
    
    # Filter 2: Maximum ATR/Price (allow higher volatility)
    if "ATR_Price" in spec_filtered.columns:
        max_atr = CONFIG.get("MAX_ATR_PRICE_CORE", 0.09) + 0.06  # +6% more volatility allowed
        spec_filtered = spec_filtered[
            (spec_filtered["ATR_Price"].isna()) | 
            (spec_filtered["ATR_Price"] <= max_atr)
        ]
    
    # Filter 3: Maximum overextension (allow stronger uptrends)
    if "OverextRatio" in spec_filtered.columns:
        max_overext = CONFIG.get("MAX_OVEREXTENSION_CORE", 0.12) + 0.08
        spec_filtered = spec_filtered[
            (spec_filtered["OverextRatio"].isna()) | 
            (spec_filtered["OverextRatio"] <= max_overext)
        ]
    
    # Filter 4: Minimum reward/risk (more lenient)
    if "RewardRisk" in spec_filtered.columns:
        min_rr = max(CONFIG.get("MIN_RR_CORE", 1.3) - 0.5, 0.8)
        spec_filtered = spec_filtered[
            (spec_filtered["RewardRisk"].isna()) | 
            (spec_filtered["RewardRisk"] >= min_rr)
        ]
    
    spec_filtered = spec_filtered.reset_index(drop=True)
    logger.info(f"Speculative filter: {spec_before_filter} → {len(spec_filtered)}")
else:
    spec_filtered = pd.DataFrame()

spec_after_filter = len(spec_filtered)

# ==================== ML SCORING ====================
if XGBOOST_MODEL is not None:
    logger.info("Applying XGBoost ML scoring...")
    
    # Score Core stocks
    if not core_filtered.empty:
        core_filtered['ML_Probability'] = core_filtered.apply(score_with_xgboost, axis=1)
        core_filtered['ML_Confidence'] = core_filtered['ML_Probability'].apply(assign_confidence_tier)
        core_filtered = core_filtered.sort_values('ML_Probability', ascending=False)
        logger.info(f"Core stocks scored: avg probability {core_filtered['ML_Probability'].mean():.3f}")
    
    # Score Speculative stocks
    if not spec_filtered.empty:
        spec_filtered['ML_Probability'] = spec_filtered.apply(score_with_xgboost, axis=1)
        spec_filtered['ML_Confidence'] = spec_filtered['ML_Probability'].apply(assign_confidence_tier)
        spec_filtered = spec_filtered.sort_values('ML_Probability', ascending=False)
        logger.info(f"Speculative stocks scored: avg probability {spec_filtered['ML_Probability'].mean():.3f}")
else:
    logger.info("ML scoring skipped - model not available")
    if not core_filtered.empty:
        core_filtered['ML_Probability'] = 0.5
        core_filtered['ML_Confidence'] = "N/A"
    if not spec_filtered.empty:
        spec_filtered['ML_Probability'] = 0.5
        spec_filtered['ML_Confidence'] = "N/A"
if spec_before_filter > 0:
    st.write(f"⚡ **Speculative filter:** {spec_before_filter} → {spec_after_filter} passed relaxed filters")

# Combine Core and Speculative
results = pd.concat([core_filtered, spec_filtered], ignore_index=True)

# Sort by score within each risk level
if not results.empty:
    results = results.sort_values(
        ["Risk_Level", "Score"], 
        ascending=[True, False]  # Core first (comes before Spec alphabetically), then by score
    ).reset_index(drop=True)

phase_times["risk_quality_classification"] = t_end(t0)

if results.empty:
    st.error("❌ **All stocks were filtered out!**")
    st.write("**Possible reasons:**")
    st.write("- 🔴 Core stocks failed technical filters (RSI, ATR, Overextension)")
    st.write("- 🔴 Speculative stocks failed relaxed filters (extremely high volatility)")
    st.write("- 🔴 Data quality too low (missing critical metrics)")
    st.write(f"- 🔴 Consider relaxing CONFIG: MIN_QUALITY_SCORE_CORE={CONFIG['MIN_QUALITY_SCORE_CORE']}, "
             f"MAX_ATR_PRICE_CORE={CONFIG['MAX_ATR_PRICE_CORE']}")
    st.stop()

# Show results count with guidance
results_count = len(results)
core_count_final = len(results[results["Risk_Level"] == "core"])
spec_count_final = len(results[results["Risk_Level"] == "speculative"])

st.success(f"✅ **Final recommendations:** {core_count_final} 🛡️ Core + {spec_count_final} ⚡ Speculative = {results_count} total")

# Updated targets: aim for balanced mix
target_min = CONFIG.get("TARGET_RECOMMENDATIONS_MIN", 5)
target_max = CONFIG.get("TARGET_RECOMMENDATIONS_MAX", 12)
target_core_min = 3
target_spec_min = 2

if results_count < target_min:
    st.warning(f"⚠️ Only {results_count} stocks passed filters (target: {target_min}-{target_max}). "
               f"Consider checking market conditions or relaxing filters.")
elif core_count_final < target_core_min:
    st.info(f"ℹ️ Only {core_count_final} Core stocks (target: {target_core_min}+). "
            f"Core stocks meet strict quality criteria. {spec_count_final} Speculative stocks offer higher-risk opportunities.")
elif spec_count_final < target_spec_min and core_count_final >= target_core_min:
    st.info(f"ℹ️ Good Core selection ({core_count_final} stocks). "
            f"Few Speculative opportunities today - market may be in consolidation phase.")

# External price verification (Top-K)
t0 = t_start()
results["Price_Alpha"] = np.nan
results["Price_Finnhub"] = np.nan
 # IEX price column removed
results["Price_Polygon"] = np.nan
results["Price_Tiingo"] = np.nan
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
    # Return collected prices and source badges
    return tkr, vals, srcs

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
                    "Price_Polygon",
                    "Price_Tiingo",
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
                vals.get("Polygon", np.nan),
                vals.get("Tiingo", np.nan),
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

    # Price sources count column
    results["Price_Sources_Count"] = results["Source_List"].apply(lambda s: len(str(s).split(" · ")) if isinstance(s, str) and s else 0)

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

    # Fundamental sources count column (flags)
    fund_flags = ["from_fmp_full", "from_fmp", "from_simfin", "from_eodhd", "from_alpha", "from_finnhub"]
    def _fund_count(row: pd.Series) -> int:
        return int(sum(bool(row.get(f)) for f in fund_flags))
    results["Fundamental_Sources_Count"] = results.apply(_fund_count, axis=1)

    # Combined reliability score
    if "Price_Reliability" in results.columns and "Fundamental_Reliability" in results.columns:
        results["Reliability_Score"] = (
            0.4 * results["Price_Reliability"].fillna(0) + 0.6 * results["Fundamental_Reliability"].fillna(0)
        ).round(4)
    else:
        results["Reliability_Score"] = np.nan
phase_times["price_verification"] = t_end(t0)

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
        return "Short (1-3 months)"
    if isinstance(rsi_v, (int, float)) and rsi_v <= 40 and score >= 60:
        return "Medium (3-6 months)"
    if score >= 70:
        return "Medium-Long (6-12 months)"
    return "Long (>12 months)"


results["Holding_Horizon"] = results.apply(infer_horizon, axis=1)

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


results["Price_Sources"] = results.apply(source_badges, axis=1)
results["Price_Mean_Rounded"] = results["Price_Mean"].round(2)
results["Price_STD_Rounded"] = results["Price_STD"].round(4)
results["Unit_Price"] = np.where(
    results["Price_Mean_Rounded"].notna(), results["Price_Mean_Rounded"], results["Price_Yahoo"]
)
results["Unit_Price"] = pd.to_numeric(results["Unit_Price"], errors="coerce")

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
k0.metric("Universe size after history filtering", len(data_map))
k1.metric("Results after filtering", len(results))
k2.metric("Budget used (≈$)", f"{budget_used:,.0f}")
k3.metric("Remaining budget (≈$)", f"{max(0.0, float(st.session_state.get('total_budget', CONFIG['BUDGET_TOTAL'])) - budget_used):,.0f}")

# Timings
st.subheader("⏱️ Execution Times")
times_df = pd.DataFrame(
    [{"Phase": k, "Duration (s)": round(v, 2)} for k, v in phase_times.items()]
)
st.table(times_df.style.set_properties(**{"text-align": "center"}))
if alpha_ok:
    st.caption(
        f"Alpha Vantage — calls this session: {int(st.session_state.get('av_calls', 0))} (respect rate limits)."
    )

# ==================== Recommendation Cards ====================
st.subheader("🤖 Recommendations Now")
st.caption("These cards are buy recommendations only. This is not investment advice.")

# Sidebar filters
with st.sidebar:
    st.header("🎛️ Filters")
    st.caption("Customize your scan results")

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
        "Risk level",
        options=["core", "speculative"],
        default=["core", "speculative"],
        format_func=lambda x: "🛡️ Core" if x == "core" else "⚡ Speculative",
        help="Choose which types of stocks to show"
    )
    
    # Data quality filter
    quality_filter = st.multiselect(
        "Minimum data quality",
        options=["high", "medium", "low"],
        default=["high", "medium", "low"],
        format_func=lambda x: {"high": "✅ High (85%+)", "medium": "⚠️ Medium (60-85%)", "low": "❌ Low (<60%)"}[x],
        help="Filter by data quality level"
    )
    
    # Score range
    if not results.empty and "Score" in results.columns:
        min_score_val = float(results["Score"].min())
        max_score_val = float(results["Score"].max())
        # Only show slider if there's a range
        if max_score_val > min_score_val:
            score_range = st.slider(
                "Score range",
                min_value=min_score_val,
                max_value=max_score_val,
                value=(min_score_val, max_score_val),
                help="Show only stocks in this score range"
            )
        else:
            st.caption(f"Score: {min_score_val:.1f} (single stock)")
            score_range = (min_score_val, max_score_val)
    else:
        score_range = (0.0, 100.0)
    
    # Sector filter removed - not useful in sidebar at this stage
    sector_filter = []
    
    # RSI filter
    rsi_max = st.slider(
        "Max RSI",
        min_value=0,
        max_value=100,
        value=80,
        help="Filter stocks with too high RSI (overbought)"
    )

    # Developer debug toggle for raw attribution
    show_debug_attr = st.checkbox("🧪 Show raw source attribution (Debug)", value=False, help="Display _sources mapping for developers")
    st.session_state["show_debug_attr"] = show_debug_attr
    compact_mode = st.checkbox("📦 Compact view", value=bool(st.session_state.get("compact_mode", False)), help="Hide indicator/fundamental details to reduce card height")
    st.session_state["compact_mode"] = compact_mode

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

st.info(f"📊 Showing {len(rec_df)} stocks after filters")

rec_df = rec_df.copy()

# CSS now loaded from design_system.py - no need for separate CARD_CSS

def format_rel(val) -> str:
    if not isinstance(val, (int, float)) or not np.isfinite(val):
        return "Not available"
    tier = "High" if val >= 0.75 else ("Medium" if val >= 0.4 else "Low")
    color = "#16a34a" if tier == "High" else ("#f59e0b" if tier == "Medium" else "#dc2626")
    return f"<span style='color:{color};font-weight:600'>{val:.2f} ({tier})</span>"

if rec_df.empty:
    st.info("No stocks currently pass the threshold with a positive buy amount.")
else:
    # Split into Core and Speculative
    if "Risk_Level" in rec_df.columns:
        core_df = rec_df[rec_df["Risk_Level"] == "core"]
        spec_df = rec_df[rec_df["Risk_Level"] == "speculative"]
    else:
        # Fallback if Risk_Level column doesn't exist
        core_df = rec_df
        spec_df = pd.DataFrame()
    
    # Display Core recommendations first
    if not core_df.empty:
        st.markdown("### 🛡️ Core Stocks — Lower Relative Risk")
        st.caption(f"✅ {len(core_df)} stocks with high data quality and balanced risk profile")
        st.markdown("""
<div style='direction:ltr;text-align:left;font-size:0.75em;margin:4px 0 10px 0'>
<b>Reliability legend:</b> <span style='color:#16a34a;font-weight:600'>High ≥ 0.75</span> · <span style='color:#f59e0b;font-weight:600'>Medium 0.40–0.74</span> · <span style='color:#dc2626;font-weight:600'>Low &lt; 0.40</span>
</div>
""", unsafe_allow_html=True)
        
        for _, r in core_df.iterrows():
            mean = r.get("מחיר ממוצע", np.nan)
            std = r.get("סטיית תקן", np.nan)
            show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
            show_std = std if not np.isnan(std) else "N/A"
            sources = r.get("מקורות מחיר", "N/A")
            buy_amt = float(r.get("סכום קנייה ($)", 0.0))
            horizon = r.get("טווח החזקה", "N/A")
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
            
            # ML scoring info
            ml_prob = r.get("ML_Probability", np.nan)
            ml_confidence = r.get("ML_Confidence", "N/A")
            
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
            
            # ML confidence badge
            if np.isfinite(ml_prob):
                if ml_prob >= 0.50:
                    ml_badge_color = "#16a34a"  # green
                    ml_badge_text = "גבוה"
                elif ml_prob >= 0.30:
                    ml_badge_color = "#f59e0b"  # orange
                    ml_badge_text = "בינוני"
                else:
                    ml_badge_color = "#dc2626"  # red
                    ml_badge_text = "נמוך"
                ml_badge_html = f"""<span style='display:inline-block;padding:3px 8px;border-radius:4px;background:{ml_badge_color};color:white;font-weight:bold;font-size:0.85em;margin-left:8px'>ML: {ml_badge_text} ({ml_prob*100:.0f}%)</span>"""
            else:
                ml_badge_html = ""

            show_mean_fmt = f"{show_mean:.2f}" if np.isfinite(show_mean) else "N/A"
            unit_price_fmt = f"{unit_price:.2f}" if np.isfinite(unit_price) else "N/A"
            rr_fmt = f"{rr:.2f}R" if np.isfinite(rr) else "N/A"
            atrp_fmt = f"{atrp:.2f}" if np.isfinite(atrp) else "N/A"
            overx_fmt = f"{overx:.2f}" if np.isfinite(overx) else "N/A"
            near52_fmt = f"{near52:.1f}" if np.isfinite(near52) else "N/A"
            
            # Format new signals
            rs_fmt = f"{rs_63d*100:+.1f}%" if np.isfinite(rs_63d) else "N/A"
            vol_surge_fmt = f"{vol_surge:.2f}x" if np.isfinite(vol_surge) else "N/A"
            ma_status = "✅ Aligned" if ma_aligned else "⚠️ Not aligned"
            quality_fmt = f"{quality_score:.0f}/50"
            rr_ratio_fmt = f"{rr_ratio:.2f}" if np.isfinite(rr_ratio) else "N/A"
            mom_fmt = f"{mom_consistency*100:.0f}%"
            confidence_badge = f"{confidence_level.upper()}" if confidence_level else "MEDIUM"
            
            # Fundamental breakdown
            qual_score_f = r.get("Quality_Score_F", np.nan)
            qual_label = r.get("Quality_Label", "N/A")
            growth_score_f = r.get("Growth_Score_F", np.nan)
            growth_label = r.get("Growth_Label", "N/A")
            val_score_f = r.get("Valuation_Score_F", np.nan)
            val_label = r.get("Valuation_Label", "N/A")
            lev_score_f = r.get("Leverage_Score_F", np.nan)
            lev_label = r.get("Leverage_Label", "N/A")
            
            # Format fundamental scores with labels
            qual_fmt = f"{qual_score_f:.0f} ({qual_label})" if np.isfinite(qual_score_f) else "N/A"
            growth_fmt = f"{growth_score_f:.0f} ({growth_label})" if np.isfinite(growth_score_f) else "N/A"
            val_fmt = f"{val_score_f:.0f} ({val_label})" if np.isfinite(val_score_f) else "N/A"
            lev_fmt = f"{lev_score_f:.0f} ({lev_label})" if np.isfinite(lev_score_f) else "N/A"
            
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
            next_earnings = r.get("NextEarnings", "Unknown")
            
            # Inject CSS for iframe isolation
            price_rel_fmt = format_rel(r.get('Price_Reliability', np.nan))
            fund_rel_fmt = format_rel(r.get('Fundamental_Reliability', np.nan))
            rel_score_fmt = format_rel(r.get('Reliability_Score', np.nan))
            
            card_html = get_card_css() + f"""
<div class="modern-card card-core">
    <h3 class="flex-wrap" style="margin:0 0 6px 0">
        <span class="modern-badge">{ticker}</span>
        <span class="modern-badge badge-success">🛡️ Core</span>
        <span class="modern-badge {quality_badge_class}">{quality_icon} Quality: {quality_pct}</span>
        <span class="modern-badge badge-primary" style="font-size:0.8em">Confidence: {confidence_badge}</span>
        {ml_badge_html}
    </h3>
    <div class="modern-grid">
    <div class="item"><b>Average Price:</b> {show_mean_fmt}</div>
    <div class="item"><b>Std Dev:</b> {show_std}</div>
    <div class="item"><b>RSI:</b> {rsi_v if not np.isnan(rsi_v) else 'N/A'}</div>
    <div class="item"><b>Near 52w High:</b> {near52_fmt}%</div>
    <div class="item"><b>Score:</b> {int(round(score))}</div>

    <div class="item"><b>Sources:</b> {sources_esc.replace(' · ','&nbsp;•&nbsp;')}</div>
    <div class="item"><b># Price Sources:</b> {r.get('Price_Sources_Count', 0)}</div>
    <div class="item"><b># Fund Sources:</b> {r.get('Fundamental_Sources_Count', 0)}</div>
    <div class="item"><b>Price Reliability:</b> {price_rel_fmt}</div>
    <div class="item"><b>Fund Reliability:</b> {fund_rel_fmt}</div>
    <div class="item"><b>Reliability Score:</b> {rel_score_fmt}</div>
    <div class="item"><b>Recommended Buy ($):</b> ${buy_amt:,.0f}</div>
    <div class="item"><b>Holding Horizon:</b> {horizon}</div>
    <div class="item"><b>Unit Price:</b> {unit_price_fmt}</div>
    <div class="item"><b>Shares to Buy:</b> {shares}</div>
    <div class="item"><b>Cash Leftover:</b> ${leftover:,.2f}</div>
    <div class="item"><b>📅 Next Earnings:</b> {next_earnings}</div>
    <div class="section-divider">🔬 Advanced Indicators:</div>
    <div class="item"><b>Market vs (3M):</b> <span style="color:{'#16a34a' if np.isfinite(rs_63d) and rs_63d > 0 else '#dc2626'}">{rs_fmt}</span></div>
    <div class="item"><b>Volume Surge:</b> {vol_surge_fmt}</div>
    <div class="item"><b>MA Alignment:</b> {ma_status}</div>
    <div class="item"><b>Quality Score:</b> {quality_fmt}</div>
    <div class="item"><b>Risk/Reward Ratio:</b> {rr_ratio_fmt}</div>
    <div class="item"><b>Momentum Consistency:</b> {mom_fmt}</div>
    <div class="item"><b>ATR/Price:</b> {atrp_fmt}</div>
    <div class="item"><b>Overextension:</b> {overx_fmt}</div>
    <div class="section-divider">💎 Fundamental Breakdown:</div>
    <div class="item"><b>Quality:</b> <span style="color:{qual_color};font-weight:600">{qual_fmt}</span></div>
    <div class="item"><b>Growth:</b> <span style="color:{growth_color};font-weight:600">{growth_fmt}</span></div>
    <div class="item"><b>Valuation:</b> <span style="color:{val_color};font-weight:600">{val_fmt}</span></div>
    <div class="item"><b>Leverage:</b> <span style="color:{lev_color};font-weight:600">{lev_fmt}</span></div>"""
            
            # Add provider attribution if available (Core cards)
            attribution = r.get("Fund_Attribution", "")
            if show_debug_attr:
                # Show raw provider map from _sources if available
                raw_sources = r.get("_sources", {})
                if raw_sources:
                    raw_html = html_escape.escape(str(raw_sources))
                    card_html += f"""
    <div class="item" style="grid-column:span 5;font-size:0.7em;color:#334155;background:#f1f5f9;border:1px dashed #cbd5e1;border-radius:6px;padding:4px;margin-top:4px"><b>RAW _sources:</b> {raw_html}</div>"""
            if attribution:
                card_html += f"""
    <div class="item" style="grid-column:span 5;font-size:0.75em;color:#6b7280;border-top:1px solid #e5e7eb;margin-top:4px;padding-top:4px"><b>📊 Data Sources:</b> {esc(attribution)}</div>"""
            
            card_html += """
  </div>
</div>
"""
            # Use reasonable height with scrolling enabled to prevent text cutoff
            st_html(card_html, height=700, scrolling=True)
    
    # Display Speculative recommendations
    if not spec_df.empty:
        st.markdown("### ⚡ Speculative Stocks — High Upside, High Risk")
        st.caption(f"⚠️ {len(spec_df)} stocks with a higher risk profile")
        st.warning("🔔 Warning: These stocks are classified as speculative due to partial data or elevated risk factors. Suitable for experienced investors only.")
        
        for _, r in spec_df.iterrows():
            mean = r.get("מחיר ממוצע", np.nan)
            std = r.get("סטיית תקן", np.nan)
            show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
            show_std = std if not np.isnan(std) else "N/A"
            sources = r.get("מקורות מחיר", "N/A")
            buy_amt = float(r.get("סכום קנייה ($)", 0.0))
            horizon = r.get("טווח החזקה", "N/A")
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
            
            # ML scoring info
            ml_prob = r.get("ML_Probability", np.nan)
            ml_confidence = r.get("ML_Confidence", "N/A")
            
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
            
            # ML confidence badge
            if np.isfinite(ml_prob):
                if ml_prob >= 0.50:
                    ml_badge_color = "#16a34a"  # green
                    ml_badge_text = "גבוה"
                elif ml_prob >= 0.30:
                    ml_badge_color = "#f59e0b"  # orange
                    ml_badge_text = "בינוני"
                else:
                    ml_badge_color = "#dc2626"  # red
                    ml_badge_text = "נמוך"
                ml_badge_html = f"""<span style='display:inline-block;padding:3px 8px;border-radius:4px;background:{ml_badge_color};color:white;font-weight:bold;font-size:0.85em;margin-left:8px'>ML: {ml_badge_text} ({ml_prob*100:.0f}%)</span>"""
            else:
                ml_badge_html = ""
            
            show_mean_fmt = f"{show_mean:.2f}" if np.isfinite(show_mean) else "N/A"
            unit_price_fmt = f"{unit_price:.2f}" if np.isfinite(unit_price) else "N/A"
            rr_fmt = f"{rr:.2f}R" if np.isfinite(rr) else "N/A"
            atrp_fmt = f"{atrp:.2f}" if np.isfinite(atrp) else "N/A"
            overx_fmt = f"{overx:.2f}" if np.isfinite(overx) else "N/A"
            near52_fmt = f"{near52:.1f}" if np.isfinite(near52) else "N/A"
            rs_fmt = f"{rs_63d*100:+.1f}%" if np.isfinite(rs_63d) else "N/A"
            vol_surge_fmt = f"{vol_surge:.2f}x" if np.isfinite(vol_surge) else "N/A"
            ma_status = "✅ Aligned" if ma_aligned else "⚠️ Not aligned"
            quality_fmt = f"{quality_score:.0f}/50"
            rr_ratio_fmt = f"{rr_ratio:.2f}" if np.isfinite(rr_ratio) else "N/A"
            mom_fmt = f"{mom_consistency*100:.0f}%"
            confidence_badge = f"{confidence_level.upper()}" if confidence_level else "LOW"
            
            qual_score_f = r.get("Quality_Score_F", np.nan)
            qual_label = r.get("Quality_Label", "N/A")
            growth_score_f = r.get("Growth_Score_F", np.nan)
            growth_label = r.get("Growth_Label", "N/A")
            val_score_f = r.get("Valuation_Score_F", np.nan)
            val_label = r.get("Valuation_Label", "N/A")
            lev_score_f = r.get("Leverage_Score_F", np.nan)
            lev_label = r.get("Leverage_Label", "N/A")
            
            qual_fmt = f"{qual_score_f:.0f} ({qual_label})" if np.isfinite(qual_score_f) else "N/A"
            growth_fmt = f"{growth_score_f:.0f} ({growth_label})" if np.isfinite(growth_score_f) else "N/A"
            val_fmt = f"{val_score_f:.0f} ({val_label})" if np.isfinite(val_score_f) else "N/A"
            lev_fmt = f"{lev_score_f:.0f} ({lev_label})" if np.isfinite(lev_score_f) else "N/A"
            
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
            next_earnings = r.get("NextEarnings", "Unknown")
            warnings_esc = esc(warnings) if warnings else ""
            
            # Inject CSS for iframe isolation
            card_html = get_card_css() + f"""
<div class="modern-card card-speculative">
    <h3 class="flex-wrap" style="margin:0 0 6px 0">
        <span class="modern-badge">{ticker}</span>
        <span class="modern-badge badge-warning">⚡ Speculative</span>
        <span class="modern-badge {quality_badge_class}">{quality_icon} Quality: {quality_pct}</span>
        <span class="modern-badge badge-danger" style="font-size:0.8em">Confidence: {confidence_badge}</span>
        {ml_badge_html}
    </h3>
    {f'<div class="warning-box"><b>⚠️ Warnings:</b> {warnings_esc}</div>' if warnings_esc else ''}
    <div class="modern-grid">
    <div class="item"><b>Average Price:</b> {show_mean_fmt}</div>
    <div class="item"><b>Std Dev:</b> {show_std}</div>
    <div class="item"><b>RSI:</b> {rsi_v if not np.isnan(rsi_v) else 'N/A'}</div>
    <div class="item"><b>Near 52w High:</b> {near52_fmt}%</div>
    <div class="item"><b>Score:</b> {int(round(score))}</div>
    <div class="item"><b>Sources:</b> {sources_esc.replace(' · ','&nbsp;•&nbsp;')}</div>
    <div class="item"><b># Price Sources:</b> {r.get('Price_Sources_Count', 0)}</div>
    <div class="item"><b># Fund Sources:</b> {r.get('Fundamental_Sources_Count', 0)}</div>
    <div class="item"><b>Price Reliability:</b> {price_rel_fmt}</div>
    <div class="item"><b>Fund Reliability:</b> {fund_rel_fmt}</div>
    <div class="item"><b>Reliability Score:</b> {rel_score_fmt}</div>
    <div class="item"><b>Recommended Buy ($):</b> ${buy_amt:,.0f}</div>
    <div class="item"><b>Holding Horizon:</b> {horizon}</div>
    <div class="item"><b>Unit Price:</b> {unit_price_fmt}</div>
    <div class="item"><b>Shares to Buy:</b> {shares}</div>
    <div class="item"><b>Cash Leftover:</b> ${leftover:,.2f}</div>
    <div class="item"><b>📅 Next Earnings:</b> {next_earnings}</div>
    <div class="section-divider">🔬 Advanced Indicators:</div>
    <div class="item"><b>Market vs (3M):</b> <span style="color:{'#16a34a' if np.isfinite(rs_63d) and rs_63d > 0 else '#dc2626'}">{rs_fmt}</span></div>
    <div class="item"><b>Volume Surge:</b> {vol_surge_fmt}</div>
    <div class="item"><b>MA Alignment:</b> {ma_status}</div>
    <div class="item"><b>Quality Score:</b> {quality_fmt}</div>
    <div class="item"><b>Risk/Reward Ratio:</b> {rr_ratio_fmt}</div>
    <div class="item"><b>Momentum Consistency:</b> {mom_fmt}</div>
    <div class="item"><b>ATR/Price:</b> {atrp_fmt}</div>
    <div class="item"><b>Overextension:</b> {overx_fmt}</div>
    <div class="section-divider">💎 Fundamental Breakdown:</div>
    <div class="item"><b>Quality:</b> <span style="color:{qual_color};font-weight:600">{qual_fmt}</span></div>
    <div class="item"><b>Growth:</b> <span style="color:{growth_color};font-weight:600">{growth_fmt}</span></div>
    <div class="item"><b>Valuation:</b> <span style="color:{val_color};font-weight:600">{val_fmt}</span></div>
    <div class="item"><b>Leverage:</b> <span style="color:{lev_color};font-weight:600">{lev_fmt}</span></div>"""
            
            # Add provider attribution if available (Speculative cards)
            attribution_spec = r.get("Fund_Attribution", "")
            if attribution_spec:
                card_html += f"""
    <div class="item" style="grid-column:span 5;font-size:0.75em;color:#6b7280;border-top:1px solid #e5e7eb;margin-top:4px;padding-top:4px"><b>📊 Data Sources:</b> {esc(attribution_spec)}</div>"""
            if show_debug_attr:
                raw_sources = r.get("_sources", {})
                if raw_sources:
                    raw_html = html_escape.escape(str(raw_sources))
                    card_html += f"""
    <div class="item" style="grid-column:span 5;font-size:0.7em;color:#334155;background:#f1f5f9;border:1px dashed #cbd5e1;border-radius:6px;padding:4px;margin-top:4px"><b>RAW _sources:</b> {raw_html}</div>"""
            
            card_html += """
  </div>
</div>
"""
            st_html(card_html, height=700, scrolling=True)

# Inject compact mode JS to hide advanced/fundamental sections
if st.session_state.get("compact_mode"):
    st.markdown("""
<script>
for(const el of document.querySelectorAll('.card')){el.classList.add('compact-mode');}
for(const el of document.querySelectorAll('.compact-mode .section-divider')){
  if(el.textContent.includes('🔬')||el.textContent.includes('💎')){
    let next=el.nextElementSibling;
    while(next && !next.classList.contains('section-divider')){
      let toHide=next;
      next=next.nextElementSibling;
      toHide.style.display='none';
    }
    el.style.display='none';
  }
}
</script>
""", unsafe_allow_html=True)
# ==================== Results table + CSV ====================
st.subheader("🎯 Filtered & Ranked Results")
view_df_source = rec_df if not rec_df.empty else results

# Augment with sources count if reliability columns present
if "Reliability_Score" in view_df_source.columns and "Source_List" in view_df_source.columns:
    view_df_source["Sources_Count"] = view_df_source["Source_List"].apply(lambda s: len(str(s).split(" · ")) if isinstance(s, str) and s else 0)

hebrew_cols = {
    "Ticker": "Ticker",
    "Price_Yahoo": "Price (Yahoo)",
    "Price_Mean": "Average Price",
    "Unit_Price": "Unit Price (calc)",
    "Price_STD": "Std Dev",
    "Source_List": "Price Sources",
    "Price_Sources_Count": "# Price Sources",
    "Fundamental_Sources_Count": "# Fund Sources",
    "Price_Reliability": "Price Reliability",
    "Fundamental_Reliability": "Fund Reliability",
    "Reliability_Score": "Reliability Score",
    "Sources_Count": "Sources Count",
    # "Price_IEX" removed
    "Price_Polygon": "Price Polygon",
    "Price_Tiingo": "Price Tiingo",
    "Price_Marketstack": "Price Marketstack",
    "Price_NasdaqDL": "Price NasdaqDL",
    "Price_EODHD": "Price EODHD",
    "Score": "Score",
    "Score_Tech": "Tech Score",
    "Fundamental_S": "Fundamental Score",
    "Quality_Score_F": "Fund Quality Score",
    "Quality_Label": "Quality Label",
    "Growth_Score_F": "Growth Score",
    "Growth_Label": "Growth Label",
    "Valuation_Score_F": "Valuation Score",
    "Valuation_Label": "Valuation Label",
    "Leverage_Score_F": "Leverage Score",
    "Leverage_Label": "Leverage Label",
    "Sector": "Sector",
    "RSI": "RSI",
    "Near52w": "Near 52w High (%)",
    "Volx20d": "Volume / 20d Avg",
    "OverextRatio": "Overextension vs MA_L",
    "ATR_Price": "ATR/Price",
    "RewardRisk": "Reward/Risk (≈R)",
    "Beta": "Beta",
    "טווח החזקה": "Holding Horizon",
    "סכום קנייה ($)": "Buy Amount ($)",
    "מניות לקנייה": "Shares to Buy",
    "עודף ($)": "Leftover ($)",
    "PE_f": "P/E",
    "PS_f": "P/S",
    "ROE_f": "ROE",
    "ROIC_f": "ROIC",
    "GM_f": "Margin",
    "DE_f": "Debt/Equity",
    "RevG_f": "Revenue YoY",
    "EPSG_f": "EPS YoY",
    "RS_63d": "Market vs (3M) (%)",
    "Volume_Surge": "Volume Surge (x)",
    "MA_Aligned": "MA Aligned",
    "Quality_Score": "Quality Score",
    "RR_Ratio": "Risk/Reward",
    "Momentum_Consistency": "Momentum Consistency (%)",
    "High_Confidence": "High Confidence",
    "Risk_Level": "Risk Level",
    "Data_Quality": "Data Quality",
    "Confidence_Level": "Confidence Level",
    "Classification_Warnings": "Warnings",
    "ML_Probability": "ML Probability",
    "ML_Confidence": "ML Confidence"
}
show_order = [
    "Ticker",
    "Sector",
    "Risk Level",
    "Data Quality",
    "ML Probability",
    "ML Confidence",
    "Confidence Level",
    "Reliability Score",
    "Fund Reliability",
    "Price Reliability",
    "Sources Count",
    "Score",
    "Quality Score",
    "Average Price",
    "Unit Price (calc)",
    "Buy Amount ($)",
    "Shares to Buy",
    "Leftover ($)",
    "Price Sources",
    "Price Marketstack",
    "Price NasdaqDL",
    "Price EODHD",
    "Tech Score",
    "Fundamental Score",
    "RSI",
    "Market vs (3M) (%)",
    "Volume Surge (x)",
    "MA Aligned",
    "Risk/Reward",
    "Momentum Consistency (%)",
    "Near 52w High (%)",
    "Volume / 20d Avg",
    "Overextension vs MA_L",
    "ATR/Price",
    "Reward/Risk (≈R)",
    "Beta",
    "Holding Horizon",
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
        "📥 Download Results (CSV)",
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
        "📊 Download Results (JSON)",
        data=json_data.encode("utf-8"),
        file_name=f"stock_scout_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

# Force button text color override with inline styles
st.markdown(
    """
    <style>
    button, button p, button div, button span {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="rtl-table">', unsafe_allow_html=True)
st.dataframe(
    csv_df[[c for c in show_order if c in csv_df.columns]],
    use_container_width=True,
    hide_index=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ==================== Quick chart ====================
st.subheader("🔍 Chart Ticker from Results")
choices = ["(Select)"] + view_df_source["Ticker"].astype(str).tolist()
choice = st.selectbox("Select ticker", choices, index=0)
if choice and choice != "(Select)" and choice in data_map:
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
            name="Price",
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
with st.expander("ℹ️ Methodology (Summary)"):
    st.markdown(
        """
- **Historical Data**: Yahoo Finance (`yfinance`). Price verification: Alpha Vantage, Finnhub, Polygon, Tiingo.
- **Technical Scoring**: Moving averages, momentum (1/3/6 months with Sigmoid), RSI range, Near-High bell curve, 
  Overextension vs MA_L, Pullback detection, ATR/Price ratio, Reward/Risk, MACD/ADX.
- **Fundamental Scoring**: Alpha Vantage OVERVIEW with Finnhub fallback (Growth/Quality/Valuation metrics + high leverage penalty).
- **Filters & Rules**: Minimum dollar-volume, ATR/Price and Overextension caps (hard disqualification), earnings blackout period, 
  beta filter, sector diversification caps, and budget allocation with maximum position size limits.
"""
    )
