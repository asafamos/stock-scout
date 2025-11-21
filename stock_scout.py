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
from core.data_sources_v2 import aggregate_fundamentals as agg_fund_v2, fetch_price_multi_source as fetch_price_multi_v2, aggregate_price as aggregate_price_v2
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
from core.v2_risk_engine import score_ticker_v2_enhanced
from indicators import rsi, atr, macd_line, adx, _sigmoid
from core.scoring_engine import evaluate_rr_unified

# Helper: build clean minimal card

def build_clean_card(row: pd.Series, speculative: bool = False) -> str:
    """
    Build professional minimal card with:
    - Header: Ticker, Badge, Overall Score only
    - Top 6 fields: Target, RR, Risk, Reliability, ML, Quality
    - Rest in <details> collapsible section
    - No emojis except ⚠️ for warnings
    - Tabular numbers, consistent formatting
    """
    esc = html_escape.escape
    ticker = esc(str(row.get('Ticker','N/A')))
    overall_rank = row.get('Overall_Rank','N/A')
    # Use pretty score for display (60-90 range), raw score for internal logic
    overall_score = row.get('overall_score_pretty', row.get('overall_score', row.get('conviction_v2_final', np.nan)))
    target_price = row.get('Target_Price', np.nan)
    entry_price = row.get('Entry_Price', np.nan)
    target_date = row.get('Target_Date','N/A')
    target_source = row.get('Target_Source','N/A')
    rr_ratio = row.get('rr', np.nan)
    rr_score = row.get('rr_score_v2', np.nan)
    rr_band = row.get('rr_band','')
    
    # Get display bands
    risk_meter = row.get('risk_meter_v2', np.nan)
    risk_band_label = row.get('risk_band', 'N/A')
    reliability_pct = row.get('reliability_pct', np.nan)
    reliability_band_label = row.get('reliability_band', 'N/A')
    ml_prob = row.get('ML_Probability', np.nan)
    # Derive confidence band with explicit Low/Medium/High thresholds; fallback message if missing
    def ml_conf_band(p: float) -> str:
        if not np.isfinite(p):
            return 'N/A'
        if p < 0.60:
            return 'Low'
        if p < 0.75:
            return 'Medium'
        return 'High'
    ml_conf_band_label = row.get('ml_conf_band', ml_conf_band(ml_prob))
    
    quality_level = row.get('Quality_Level', 'N/A')
    quality_score = row.get('Quality_Score_Numeric', np.nan)
    conv_base = row.get('conviction_v2_base', np.nan)
    
    # Component scores for <details>
    fund_score = row.get('Fundamental_S', np.nan)
    tech_score = row.get('Technical_S', np.nan)
    # Compressed data sources line (prices + fundamentals providers if available)
    price_sources = row.get('Price_Sources_Line', '')  # expected precomputed concise string
    fund_sources = row.get('Fund_Sources_Line', '')
    sources_line = ''
    if price_sources or fund_sources:
        sources_line = f"Data sources: Prices – {price_sources or 'N/A'}; Fundamentals – {fund_sources or 'N/A'}"

    def fmt_money(v):
        return f"${v:.2f}" if np.isfinite(v) else 'N/A'
    def fmt_pct(v):
        return f"{v:.1f}%" if np.isfinite(v) else 'N/A'
    def fmt_score(v):
        # Handle lists by taking first element or count
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return '0'
            v = len(v) if all(isinstance(x, str) for x in v) else v[0]
        # Handle non-numeric types
        try:
            return f"{float(v):.0f}" if np.isfinite(float(v)) else 'N/A'
        except (TypeError, ValueError):
            return str(v) if v is not None else 'N/A'

    entry_fmt = fmt_money(entry_price)
    target_fmt = fmt_money(target_price)
    if np.isfinite(entry_price) and np.isfinite(target_price) and entry_price>0:
        potential_gain_pct = ((target_price-entry_price)/entry_price)*100
        potential_fmt = f"+{potential_gain_pct:.1f}%"
    else:
        potential_fmt = 'N/A'
    
    target_badge = ''
    if target_source == 'AI':
        target_badge = '<span class="badge ai">AI</span>'
    elif target_source == 'Technical':
        target_badge = '<span class="badge tech">Tech</span>'

    rr_ratio_fmt = f"{rr_ratio:.2f}" if np.isfinite(rr_ratio) else 'N/A'
    overall_score_fmt = fmt_score(overall_score)
    quality_score_fmt = f"{quality_score:.2f}" if np.isfinite(quality_score) else 'N/A'
    
    # Get Fund and Price reliability separately for detailed display
    fund_reliability = row.get('Fundamental_Reliability_v2', row.get('Fundamental_Reliability', np.nan))
    price_reliability = row.get('Price_Reliability_v2', row.get('Price_Reliability', np.nan))
    fund_rel_fmt = fmt_score(fund_reliability) if np.isfinite(fund_reliability) else 'N/A'
    price_rel_fmt = fmt_score(price_reliability) if np.isfinite(price_reliability) else 'N/A'
    
    # Format display values with bands
    risk_fmt = f"{fmt_score(risk_meter)} ({risk_band_label})"
    reliability_fmt = f"{reliability_band_label} (F:{fund_rel_fmt}% / P:{price_rel_fmt}%)"
    ml_fmt = f"{ml_conf_band_label} (p={ml_prob:.2f})" if np.isfinite(ml_prob) else "N/A (no model data)"

    type_badge = 'SPEC' if speculative else 'CORE'
    
    # Warning indicator
    warning = ''
    if rr_ratio < 1.5 or (np.isfinite(risk_meter) and risk_meter > 70):
        warning = ' ⚠️'

    return f"""
<div class='clean-card { 'speculative' if speculative else 'core' }'>
  <div class='card-header'>
    <div class='ticker-line'><span class='ticker-badge'>{ticker}</span><span class='type-badge'>{type_badge}</span></div>
    <h2 class='overall-score'>{overall_score_fmt}<span class='score-label'>/100</span>{warning}</h2>
  </div>
  <div class='top-grid'>
    <div class='field'><span class='label'>Target</span><span class='value tabular'>{target_fmt} {target_badge} ({potential_fmt})</span></div>
    <div class='field'><span class='label'>R/R</span><span class='value tabular'>{rr_ratio_fmt} <span class='band'>{rr_band}</span></span></div>
    <div class='field'><span class='label'>Risk</span><span class='value tabular'>{risk_fmt}</span></div>
    <div class='field'><span class='label'>Reliability</span><span class='value tabular'>{reliability_fmt}</span></div>
    <div class='field'><span class='label'>ML</span><span class='value tabular'>{ml_fmt}</span></div>
    <div class='field'><span class='label'>Quality</span><span class='value tabular'>{quality_level} ({quality_score_fmt})</span></div>
  </div>
    <details class='more-info'>
    <summary>More Details</summary>
    <div class='detail-grid'>
      <div class='field'><span class='label'>Entry</span><span class='value'>{entry_fmt}</span></div>
      <div class='field'><span class='label'>Target Date</span><span class='value'>{target_date}</span></div>
      <div class='field'><span class='label'>Fundamental Score</span><span class='value'>{fmt_score(fund_score)}</span></div>
      <div class='field'><span class='label'>ML Probability</span><span class='value'>{fmt_pct(ml_prob * 100) if np.isfinite(ml_prob) else 'N/A'}</span></div>
      <div class='field'><span class='label'>Base Conviction</span><span class='value'>{fmt_score(conv_base)}</span></div>
    <div class='field'><span class='label'>Fund Sources</span><span class='value'>{fmt_score(row.get('fund_sources_used_v2', 0))}</span></div>
    <div class='field'><span class='label'>Price Sources</span><span class='value'>{fmt_score(row.get('price_sources_used_v2', 0))}</span></div>
      <div class='field'><span class='label'>Price Std Dev</span><span class='value'>{fmt_money(row.get('Price_STD_v2', np.nan))}</span></div>"""
    + (f"""
      <div class='field' style='grid-column:span 2'><span class='label'>Sources</span><span class='value'>{html_escape.escape(sources_line)}</span></div>""" if sources_line else "") + """
    </div>
  </details>
</div>
"""


# OpenAI for enhanced target price predictions
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

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
    # --- Performance / Fast Mode Flags ---
    "PERF_FAST_MODE": False,          # Set True to reduce external waits for interactive exploration
    "PERF_MULTI_SOURCE_TOP_N": 8,     # In fast mode: only compute multi-source reliability for top N by score
    "PERF_ALPHA_ENABLED": True,       # In fast mode we force this False to skip Alpha Vantage entirely
    "PERF_FUND_TIMEOUT": 15,          # Normal per-provider future timeout
    "PERF_FUND_TIMEOUT_FAST": 6,      # Fast mode per-provider future timeout
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
    """
    Assign ML confidence tier based on probability.
    
    Recalibrated thresholds for realistic diversity:
    - High: ≥0.75 (strong prediction)
    - Medium: 0.60-0.74 (moderate confidence)
    - Low: <0.60 (weak prediction)
    """
    if not isinstance(prob, (int, float)) or not np.isfinite(prob):
        return "N/A"
    if prob >= 0.75:
        return "High"
    if prob >= 0.60:
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
    """Throttle Alpha Vantage calls to respect 5 calls/minute (25 calls/day on free tier).
    
    Args:
        min_gap_seconds: Minimum seconds between calls (default 12s = 5 calls/min)
    """
    ts_key = "_alpha_last_call_ts"
    calls_key = "av_calls"
    
    # Track number of calls in session
    call_count = st.session_state.get(calls_key, 0)
    
    # Reset counter daily (conservative approach)
    last_reset_key = "_alpha_reset_date"
    today = datetime.utcnow().date().isoformat()
    if st.session_state.get(last_reset_key) != today:
        st.session_state[calls_key] = 0
        st.session_state[last_reset_key] = today
        call_count = 0
    
    # Check daily limit (25 on free tier, be conservative with 20)
    if call_count >= 20:
        logger.warning(f"Alpha Vantage daily limit reached ({call_count} calls), skipping")
        return
    
    # Enforce rate limit
    last = st.session_state.get(ts_key, 0.0)
    now = time.time()
    elapsed = now - last
    if elapsed < min_gap_seconds:
        sleep_time = min_gap_seconds - elapsed
        logger.debug(f"Alpha Vantage throttle: sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)
    
    # Update state
    st.session_state[ts_key] = time.time()
    st.session_state[calls_key] = call_count + 1


# --- Build Universe (restored) ---
def build_universe(limit: int) -> List[str]:
    """Fetch S&P 500 tickers (wikipedia) then fallback to common mega-cap list.
    Limit result length to `limit`.
    
    CRITICAL FIX: Deduplicates by company name to prevent GOOG/GOOGL duplicates.
    """
    try:
        # Wikipedia requires a User-Agent header
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            storage_options={'User-Agent': 'Mozilla/5.0'}
        )
        # S&P500 table is typically the second table (index 1)
        df_sp = tables[1]
        
        # CRITICAL FIX: Deduplicate by company name (keep first ticker, usually Class A)
        # Prevents duplicate recommendations for GOOG/GOOGL, BRK.A/BRK.B, etc.
        original_count = len(df_sp)
        df_sp = df_sp.drop_duplicates(subset='Security', keep='first')
        logger.info(f"✓ Deduplicated {original_count} → {len(df_sp)} unique companies (removed {original_count - len(df_sp)} multi-class tickers)")
        
        tickers = df_sp['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
        logger.info(f"✓ Loaded {len(tickers)} unique S&P500 companies from Wikipedia")
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


def calculate_rr(entry_price: float, target_price: float, atr_value: float, history_df: pd.DataFrame = None) -> float:
    """
    Stable Reward/Risk calculation used across the app.
    - reward = target_price - entry_price
    - risk   = max(atr_value * 2, entry_price * 0.01)
    - rr     = reward / risk
    Clamped to [0, 5]. Returns numeric (float). If ATR missing and history_df provided,
    estimate ATR as mean(high-low) over last 14 candles.
    """
    try:
        if not (isinstance(entry_price, (int, float)) and np.isfinite(entry_price)):
            return 0.0
        if not (isinstance(target_price, (int, float)) and np.isfinite(target_price)):
            return 0.0

        atr = atr_value if (isinstance(atr_value, (int, float)) and np.isfinite(atr_value)) else np.nan
        if (not np.isfinite(atr)) and history_df is not None:
            try:
                last = history_df.tail(14)
                if not last.empty and "High" in last.columns and "Low" in last.columns:
                    est_atr = (last["High"] - last["Low"]).abs().dropna().mean()
                    if np.isfinite(est_atr) and est_atr > 0:
                        atr = float(est_atr)
            except Exception:
                atr = np.nan

        # Fallback risk if ATR still missing
        risk = None
        if np.isfinite(atr):
            risk = max(atr * 2.0, entry_price * 0.01)
        else:
            risk = max(entry_price * 0.01, 0.01)

        reward = max(0.0, float(target_price) - float(entry_price))
        rr = reward / max(risk, 1e-9)
        rr = float(np.clip(rr, 0.0, 5.0))
        return rr
    except Exception:
        return 0.0


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
        "oper_margin": np.nan,
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
        # Accept common operating margin keys from providers (if present)
        oper_keys = ["oper_margin", "operatingMargin", "operating_margin", "operatingProfitMargin", "operatingProfitMarginTTM"]
        for k in ["roe","roic","gm","ps","pe","de","rev_g_yoy","eps_g_yoy","oper_margin"]:
            v_cur = merged.get(k, np.nan)
            # First try direct normalized key
            v_new = src.get(k, np.nan)
            # If looking for operating margin, check alternate provider keys as well
            if k == "oper_margin" and (not isinstance(v_new, (int, float)) or not np.isfinite(v_new)):
                for ok in oper_keys:
                    if ok in src and isinstance(src.get(ok), (int, float)) and np.isfinite(src.get(ok)):
                        v_new = src.get(ok)
                        break

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
        
        # Tiingo (comprehensive fundamentals)
        if CONFIG.get("ENABLE_TIINGO", True):  # Enable by default
            tk = _env("TIINGO_API_KEY")
            if tk:
                futures['tiingo'] = ex.submit(_tiingo_fundamentals_fetch, ticker)
        
        # Collect results with timeout
        results = {}
        fund_timeout = CONFIG.get("PERF_FUND_TIMEOUT_FAST") if CONFIG.get("PERF_FAST_MODE") else CONFIG.get("PERF_FUND_TIMEOUT")
        for source, fut in futures.items():
            try:
                results[source] = fut.result(timeout=fund_timeout)
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

    # Prefer Finnhub data before Alpha for per-field fallbacks
    if results.get('finnhub'):
        _merge(results['finnhub'], "from_finnhub", "Finnhub")
        logger.debug(f"Fundamentals merge: Finnhub ✓ {ticker}")

    if results.get('alpha'):
        _merge(results['alpha'], "from_alpha", "Alpha")
        logger.debug(f"Fundamentals merge: Alpha ✓ {ticker}")
    
    if results.get('eodhd'):
        _merge(results['eodhd'], "from_eodhd", "EODHD")
        logger.debug(f"Fundamentals merge: EODHD ✓ {ticker}")
    
    if results.get('tiingo'):
        _merge(results['tiingo'], "from_tiingo", "Tiingo")
        logger.debug(f"Fundamentals merge: Tiingo ✓ {ticker}")

    # Per-field explicit fallbacks (FMP -> Finnhub -> Alpha -> Tiingo)
    # Ensure that if a field wasn't provided by FMP we try Finnhub then Alpha then Tiingo
    fallback_order = [
        ('finnhub', 'Finnhub'),
        ('alpha', 'Alpha'),
        ('tiingo', 'Tiingo'),
    ]
    for k in ["roe","roic","gm","ps","pe","de","rev_g_yoy","eps_g_yoy"]:
        if not isinstance(merged.get(k), (int, float)) or not np.isfinite(merged.get(k)):
            for src_key, src_name in fallback_order:
                src = results.get(src_key, {}) if isinstance(results.get(src_key, {}), dict) else {}
                v_new = src.get(k, np.nan)
                if isinstance(v_new, (int, float)) and np.isfinite(v_new):
                    merged[k] = float(v_new)
                    merged["_sources"][k] = src_name
                    # mark source flag and record
                    merged[f"from_{src_key}"] = True
                    if f"from_{src_key}" not in merged["_sources_used"]:
                        merged["_sources_used"].append(f"from_{src_key}")
                    break

    # Try operating margin fallback explicitly
    if not np.isfinite(merged.get("oper_margin", np.nan)):
        for src_key, src_name in fallback_order:
            src = results.get(src_key, {}) if isinstance(results.get(src_key, {}), dict) else {}
            # check common oper margin keys in provider dicts
            for ok in ["oper_margin", "operatingMargin", "operating_margin", "operatingProfitMargin", "operatingProfitMarginTTM"]:
                v_new = src.get(ok, np.nan)
                if isinstance(v_new, (int, float)) and np.isfinite(v_new):
                    merged["oper_margin"] = float(v_new)
                    merged["_sources"]["oper_margin"] = src_name
                    merged[f"from_{src_key}"] = True
                    if f"from_{src_key}" not in merged["_sources_used"]:
                        merged["_sources_used"].append(f"from_{src_key}")
                    break
            if np.isfinite(merged.get("oper_margin", np.nan)):
                break

    # Compute coverage after fallbacks (map our internal names to requested external names)
    cov_fields = [merged.get(k) for k in ["pe","ps","rev_g_yoy","eps_g_yoy","gm","de","oper_margin","roe"]]
    valid_count = sum(isinstance(v, (int, float)) and np.isfinite(v) for v in cov_fields)
    merged["Fund_Coverage_Pct"] = float(valid_count) / float(len(cov_fields))

    # Log missing fields for debugging
    missing_fields = [k for k, v in zip(["pe","ps","rev_g_yoy","eps_g_yoy","gm","de","oper_margin","roe"], cov_fields) if not (isinstance(v, (int, float)) and np.isfinite(v))]
    if missing_fields:
        print(f"[FUND] Missing for {ticker}: {missing_fields}")

    # If at least one provider responded (any from_* flag) but coverage is zero, set a small floor
    provider_responded = any(merged.get(f, False) for f in ["from_fmp_full","from_fmp","from_alpha","from_finnhub","from_simfin","from_eodhd","from_tiingo"])
    if provider_responded and valid_count == 0:
        # Inject neutral defaults so the ticker is not dropped and to allow partial scoring
        neutral_defaults = {
            "pe": 20.0,
            "ps": 2.0,
            "rev_g_yoy": 0.0,
            "eps_g_yoy": 0.0,
            "gm": 0.25,
            "de": 1.0,
            "oper_margin": 0.05,
            "roe": 8.0,
        }
        merged.setdefault("_defaulted_fields", [])
        for f, v in neutral_defaults.items():
            if not (isinstance(merged.get(f), (int, float)) and np.isfinite(merged.get(f))):
                merged[f] = float(v)
                merged["_defaulted_fields"].append(f)
        # Set a small coverage floor to reflect presence of providers
        merged["Fund_Coverage_Pct"] = max(merged.get("Fund_Coverage_Pct", 0.0), 0.05)

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
    """Fetch Alpha Vantage OVERVIEW with improved error handling and comprehensive fields.
    
    Returns: dict with roe, roic, gm, ps, pe, de, rev_g_yoy, eps_g_yoy, sector
    """
    ak = _env("ALPHA_VANTAGE_API_KEY")
    if not ak:
        return {}
    try:
        alpha_throttle(12.0)  # 5 calls/minute = 12 seconds between calls
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ak}"
        r = http_get_retry(url, tries=2, timeout=10)
        if not r:
            return {}
        j = r.json()
        
        # Check for Alpha Vantage errors
        if isinstance(j, dict):
            if "Note" in j or "Information" in j:
                logger.warning(f"Alpha Vantage rate limit: {j.get('Note') or j.get('Information')}")
                return {}
            if not j.get("Symbol"):
                return {}
        else:
            return {}

        def fnum(k):
            try:
                v = j.get(k)
                if v in (None, "None", "-", ""):
                    return np.nan
                v = float(v)
                return v if np.isfinite(v) else np.nan
            except Exception:
                return np.nan

        # Calculate gross margin from raw numbers if available
        gp = fnum("GrossProfitTTM")
        tr = fnum("RevenueTTM")
        gm_calc = (gp / tr) if (np.isfinite(gp) and np.isfinite(tr) and tr > 0) else np.nan
        
        # Fallback to profit margin if gross margin not available
        pm = fnum("ProfitMargin")
        
        out = {
            "roe": fnum("ReturnOnEquityTTM"),
            "roic": np.nan,  # Alpha doesn't provide ROIC directly
            "gm": gm_calc if np.isfinite(gm_calc) else pm,
            "ps": fnum("PriceToSalesRatioTTM"),
            "pe": fnum("PERatio"),
            "de": fnum("DebtToEquity"),
            "rev_g_yoy": fnum("QuarterlyRevenueGrowthYOY"),
            "eps_g_yoy": fnum("QuarterlyEarningsGrowthYOY"),
            "sector": j.get("Sector") or "Unknown",
        }
        
        # Count valid fields
        valid_count = sum(1 for k, v in out.items() if k != "sector" and isinstance(v, (int, float)) and np.isfinite(v))
        if valid_count >= 3:  # At least 3 valid fields
            out["from_alpha"] = True
            out["_alpha_field_count"] = valid_count
            return out
        return {}
    except Exception as e:
        logger.debug(f"Alpha Vantage fetch failed for {ticker}: {e}")
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
    """Fetch fundamentals from Tiingo - comprehensive data from fundamentals endpoint.
    
    Returns: dict with pe, ps, pb, roe, gm, de, rev_g_yoy, eps_g_yoy if available.
    """
    tk = _env("TIINGO_API_KEY")
    if not tk:
        return {}
    try:
        # Use Tiingo fundamentals endpoint for comprehensive data
        fund_url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/daily?token={tk}"
        fund_r = http_get_retry(fund_url, tries=2, timeout=10)
        
        if not fund_r:
            return {}
        
        fund_j = fund_r.json()
        if not fund_j or not isinstance(fund_j, list) or len(fund_j) == 0:
            return {}
        
        latest = fund_j[0]  # Most recent data
        
        def fnum(key):
            try:
                v = latest.get(key)
                if v is None:
                    return np.nan
                return float(v) if np.isfinite(float(v)) else np.nan
            except:
                return np.nan
        
        out = {
            "pe": fnum("pe"),
            "ps": fnum("priceToSalesRatio"),
            "pb": fnum("pb"),
            "roe": fnum("roe"),
            "gm": fnum("grossMargin"),
            "de": fnum("debtToEquity"),
            "rev_g_yoy": fnum("revenueGrowth"),
            "eps_g_yoy": fnum("epsGrowth"),
        }
        
        valid_count = sum(1 for v in out.values() if isinstance(v, (int, float)) and np.isfinite(v))
        if valid_count >= 2:  # At least 2 valid fields
            out["from_tiingo"] = True
            out["_tiingo_field_count"] = valid_count
            return out
        return {}
    except Exception as e:
        logger.debug(f"Tiingo fundamentals fetch failed for {ticker}: {e}")
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

###############################
# Canonical Data Sources Table
###############################
st.markdown("### 🔌 Data Sources Overview")

# NOTE: Tiingo fundamentals presently lightly used; price fallback + occasional fundamental fields.
# TODO: integrate broader Tiingo fundamentals (ratios, growth) into scoring engine.
# NOTE: SimFin deprecated (row kept for transparency when key present).

DATA_SOURCES = {
    "Yahoo": {
        "uses_price": True,
        "uses_fundamentals": False,
        "uses_ml": False,
        "env_keys": [],
    },
    "FMP": {
        "uses_price": False,  # fundamentals only (primary)
        "uses_fundamentals": bool(fmp_ok),
        "uses_ml": False,
        "env_keys": ["FMP_API_KEY"],
    },
    "Alpha Vantage": {
        "uses_price": bool(alpha_ok),
        "uses_fundamentals": bool(alpha_ok),  # OVERVIEW endpoints
        "uses_ml": False,
        "env_keys": ["ALPHA_VANTAGE_API_KEY"],
    },
    "Finnhub": {
        "uses_price": bool(finn_ok),  # occasionally price verify
        "uses_fundamentals": bool(finn_ok),
        "uses_ml": False,
        "env_keys": ["FINNHUB_API_KEY"],
    },
    "Polygon": {
        "uses_price": bool(poly_ok),
        "uses_fundamentals": False,
        "uses_ml": False,
        "env_keys": ["POLYGON_API_KEY"],
    },
    "Tiingo": {
        "uses_price": bool(tiin_ok),
        "uses_fundamentals": bool(tiin_ok),  # limited usage currently
        "uses_ml": False,
        "env_keys": ["TIINGO_API_KEY"],
    },
    "SimFin": {
        "uses_price": False,
        "uses_fundamentals": bool(_env("SIMFIN_API_KEY")),
        "uses_ml": False,
        "env_keys": ["SIMFIN_API_KEY"],
    },
    "Marketstack": {
        "uses_price": False,  # disabled
        "uses_fundamentals": False,
        "uses_ml": False,
        "env_keys": ["MARKETSTACK_API_KEY"],
    },
    "NasdaqDL": {
        "uses_price": False,
        "uses_fundamentals": False,
        "uses_ml": False,
        "env_keys": ["NASDAQ_API_KEY", "NASDAQ_DL_API_KEY"],
    },
    "EODHD": {
        "uses_price": False,
        "uses_fundamentals": bool(_env("EODHD_API_KEY")),
        "uses_ml": False,
        "env_keys": ["EODHD_API_KEY"],
    },
    "OpenAI": {
        "uses_price": False,
        "uses_fundamentals": False,
        "uses_ml": bool(OPENAI_AVAILABLE and (os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY"))),
        "env_keys": ["OPENAI_API_KEY"],
    },
}

def _is_active(entry: dict) -> bool:
    for k in entry.get("env_keys", []):
        if _env(k) or os.getenv(k) or st.secrets.get(k):
            return True
    # Built-ins like Yahoo active by default
    if not entry.get("env_keys"):
        return True
    return False

dot_on = "<span style='color:#16a34a;font-weight:700'>●</span>"
dot_off = "<span style='color:#94a3b8'>●</span>"

table_rows = ["<tr><th style='text-align:left'>Provider</th><th>Price</th><th>Fundamentals</th><th>ML/AI</th><th>Active</th></tr>"]
for name, meta in DATA_SOURCES.items():
    price_dot = dot_on if meta["uses_price"] else dot_off
    fund_dot = dot_on if meta["uses_fundamentals"] else dot_off
    ml_dot = dot_on if meta["uses_ml"] else dot_off
    active_dot = dot_on if _is_active(meta) else dot_off
    table_rows.append(
        f"<tr><td style='text-align:left'>{name}</td><td>{price_dot}</td><td>{fund_dot}</td><td>{ml_dot}</td><td>{active_dot}</td></tr>"
    )

sources_css = """
<style>
.sources-overview table {width:100%; border-collapse:collapse; margin:4px 0 12px 0;}
.sources-overview th, .sources-overview td {padding:4px 6px; font-size:12px; text-align:center; border-bottom:1px solid #e5e7eb;}
.sources-overview th {background:#f8fafc; font-weight:600; font-size:11px; letter-spacing:0.5px;}
.sources-overview tr:last-child td {border-bottom:none;}
.sources-overview td:first-child, .sources-overview th:first-child {text-align:left;}
@media (max-width:600px){.sources-overview th,.sources-overview td{font-size:11px;padding:3px 4px;}}
</style>
"""
st.markdown(sources_css + "<div class='sources-overview'><table>" + "".join(table_rows) + "</table></div>", unsafe_allow_html=True)

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
with st.spinner("🔍 Building stock universe..."):
    universe = (
        build_universe(limit=CONFIG["UNIVERSE_LIMIT"])
        if CONFIG["SMART_SCAN"]
        else build_universe(limit=200)
    )
phase_times["build_universe"] = t_end(t0)

# 2) History
t0 = t_start()
with st.spinner(f"📊 Fetching historical data for {len(universe)} stocks..."):
    data_map = fetch_history_bulk(universe, CONFIG["LOOKBACK_DAYS"], CONFIG["MA_LONG"])
phase_times["fetch_history"] = t_end(t0)

# 3) Technical score + hard filters
t0 = t_start()
with st.spinner(f"📈 Computing technical indicators for {len(data_map)} stocks..."):
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

    # Compute Reward/Risk using stable helper (never N/A)
    try:
        entry_price_for_rr = price if np.isfinite(price) else np.nan
        target_price_for_rr = hi_52w if (np.isfinite(hi_52w) and hi_52w > 0) else (price * 1.10 if np.isfinite(price) else np.nan)
        reward_risk = calculate_rr(entry_price_for_rr, target_price_for_rr, atr14, history_df=df)
        rr_score = min(1.0, reward_risk / 4.0) if np.isfinite(reward_risk) else 0.0
    except Exception:
        reward_risk, rr_score = 0.0, 0.0

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
            # Always show numeric RewardRisk (force 0.0 when missing)
            "RewardRisk": round(float(reward_risk) if np.isfinite(reward_risk) else 0.0, 2),
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

with st.spinner("🔬 Running advanced filters (dynamic penalties)..."):
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

# Step 1: Compute all signals first (without rejection)
signals_store = []  # (idx, signals, enhanced_score)
for idx in results.index:
    tkr = results.at[idx, "Ticker"]
    if tkr not in data_map or benchmark_df.empty:
        signals_store.append((idx, {}, results.at[idx, "Score"]))
        continue
    df = data_map[tkr]
    base_score = results.at[idx, "Score"]
    enhanced_score, signals = compute_advanced_score(tkr, df, benchmark_df, base_score)
    signals_store.append((idx, signals, enhanced_score))

# Step 2: Calculate dynamic thresholds from all signals
rs_vals = []
mom_vals = []
rr_vals = []
for (_, sig, _) in signals_store:
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

# More permissive percentiles to allow more stocks through
rs_thresh_dyn = _q(rs_vals, 0.05, -0.30)    # 5th percentile (was 15th), allow more underperformers
mom_thresh_dyn = _q(mom_vals, 0.10, 0.15)   # 10th percentile (was 20th), allow weaker momentum
rr_thresh_dyn = _q(rr_vals, 0.10, 0.50)     # 10th percentile (was 25th), allow lower RR
logger.info(f"Dynamic thresholds -> RS:{rs_thresh_dyn:.3f} MOM:{mom_thresh_dyn:.3f} RR:{rr_thresh_dyn:.3f}")

# Step 3: NOW apply rejection with dynamic thresholds
dynamic_thresholds = {
    "rs_63d": rs_thresh_dyn,
    "momentum_consistency": mom_thresh_dyn,
    "risk_reward_ratio": rr_thresh_dyn
}

catastrophic_count = 0
kept = 0
for (idx, sig, enhanced_score) in signals_store:
    # Apply rejection with dynamic thresholds
    catastrophic, reason = should_reject_ticker(sig, dynamic=dynamic_thresholds)
    
    if catastrophic:
        catastrophic_count += 1
        results.loc[idx, "RejectionReason"] = reason
        # Skip penalty calculation for rejected stocks
        results.loc[idx, "RS_63d"] = sig.get("rs_63d", np.nan)
        results.loc[idx, "Volume_Surge"] = sig.get("volume_surge", np.nan)
        results.loc[idx, "MA_Aligned"] = sig.get("ma_aligned", False)
        results.loc[idx, "Quality_Score"] = sig.get("quality_score", 0.0)
        results.loc[idx, "RR_Ratio"] = sig.get("risk_reward_ratio", np.nan)
        results.loc[idx, "Momentum_Consistency"] = sig.get("momentum_consistency", 0.0)
        results.loc[idx, "High_Confidence"] = sig.get("high_confidence", False)
        results.loc[idx, "AdvPenalty"] = 0.0
        results.loc[idx, "AdvFlags"] = "REJECTED"
        results.loc[idx, "Score"] = 0.0
        continue
    
    # Not catastrophic - apply normal penalty logic
    kept += 1
    flags = []
    penalty = 0.0
    if sig:
        rs_val = sig.get("rs_63d", np.nan)
        mom_val = sig.get("momentum_consistency", 0.0)
        rr_val = sig.get("risk_reward_ratio", np.nan)
        if np.isfinite(rs_val) and rs_val < rs_thresh_dyn:
            penalty += 2.0  # reduced from 8.0
            flags.append("LowRS")
        if mom_val < mom_thresh_dyn:
            penalty += 2.0  # reduced from 6.0
            flags.append("WeakMomentum")
        if np.isfinite(rr_val) and rr_val < rr_thresh_dyn:
            penalty += 3.0  # reduced from 10.0
            flags.append("LowRR")
    
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

if catastrophic_count == len(signals_store):
    logger.warning("All stocks met catastrophic rejection; overriding to keep all for inspection.")
    kept = len(signals_store)

logger.info(f"Advanced filters dynamic: kept {kept}/{len(signals_store)} catastrophic={catastrophic_count}")

if catastrophic_count > 0 and catastrophic_count < len(signals_store):
        # Remove catastrophic stocks
        drop_indices = results[results["Score"] == 0.0].index
        results = results[~results.index.isin(drop_indices)].reset_index(drop=True)

    results = results.sort_values(["Score", "Ticker"], ascending=[False, True]).reset_index(drop=True)
phase_times["advanced_filters"] = t_end(t0)

if results.empty:
    st.warning("Advanced filters produced empty set even after penalties.")
    st.stop()

# 3d) Fetch Fundamentals for stocks that passed advanced_filters
if CONFIG["FUNDAMENTAL_ENABLED"] and fundamental_available:
    t0 = t_start()
    
    # Fetch for all stocks that passed advanced_filters (typically 15-25)
    take_k = len(results)
    
    with st.spinner(f"📊 Fetching fundamentals for {take_k} stocks..."):
        for idx in results.index:
        tkr = results.at[idx, "Ticker"]
        rank = list(results.index).index(idx) + 1  # 1-based rank
        # Smart Alpha: enable only for top 15 to respect 25/day rate limit
        use_alpha = (rank <= 15)
        # Fast mode disables Alpha entirely to avoid throttle sleeps
        if CONFIG.get("PERF_FAST_MODE"):
            use_alpha = False
        d = fetch_fundamentals_bundle(tkr, enable_alpha_smart=use_alpha)
        
        # DEBUG: Log FMP data quality for key tickers (transparency check)
        if tkr in ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'META', 'AMZN', 'JPM', 'V']:
            fmp_ok = d.get('from_fmp', False) or d.get('from_fmp_full', False)
            coverage = d.get('Fund_Coverage_Pct', 0.0)
            quality_f = d.get('Quality_Score_F', np.nan)
            roe = d.get('roe', np.nan)
            quality_str = f"{quality_f:.1f}" if np.isfinite(quality_f) else "NaN"
            logger.info(f"🔍 {tkr} FMP Debug: from_fmp={fmp_ok}, coverage={coverage:.0%}, "
                       f"roe={roe}, quality_score_f={quality_str}")
        
        # Store provider metadata
        results.loc[idx, "Fund_from_FMP"] = d.get("from_fmp", False) or d.get("from_fmp_full", False)
        results.loc[idx, "Fund_from_Alpha"] = d.get("from_alpha", False)
        results.loc[idx, "Fund_from_Finnhub"] = d.get("from_finnhub", False)
        results.loc[idx, "Fund_from_SimFin"] = d.get("from_simfin", False)
        results.loc[idx, "Fund_from_EODHD"] = d.get("from_eodhd", False)
        # Also store canonical per-provider boolean flags (used later for reliability calculation)
        for _flag in [
            'from_fmp_full', 'from_fmp', 'from_alpha', 'from_finnhub',
            'from_simfin', 'from_eodhd', 'from_tiingo'
        ]:
            results.loc[idx, _flag] = bool(d.get(_flag, False))
        
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

# --- Multi-Source Aggregation & Reliability Injection (v2) ---
# Apply after initial per-ticker fundamentals fetch; enrich with multi-source merged view
multi_sources_fields = [
    "pe", "ps", "pb", "roe", "margin", "rev_yoy", "eps_yoy", "debt_equity", "market_cap", "beta"
]

fund_sources_used_list = []
fund_disagreement_list = []
fund_coverage_pct_list = []
fund_reliability_list = []
price_sources_used_list = []
price_variance_score_list = []
price_reliability_list = []
price_mean_list = []
price_std_list = []

for idx in results.index:
    tkr = results.at[idx, "Ticker"]
    pos = list(results.index).index(idx)
    fast_mode = CONFIG.get("PERF_FAST_MODE")
    top_n_limit = CONFIG.get("PERF_MULTI_SOURCE_TOP_N", 0)
    # If fast mode and beyond top N tickers: skip heavy multi-source calls, append neutral defaults
    if fast_mode and top_n_limit > 0 and pos >= top_n_limit:
        fund_sources_used_list.append(0)
        fund_disagreement_list.append(1.0)
        fund_coverage_pct_list.append(0.0)
        fund_reliability_list.append(np.nan)
        price_sources_used_list.append(0)
        price_variance_score_list.append(1.0)
        price_reliability_list.append(np.nan)
        price_mean_list.append(np.nan)
        price_std_list.append(np.nan)
        continue
    try:
        agg_fund = agg_fund_v2(tkr)
    except Exception:
        agg_fund = {"sources_used": [], "coverage": {}, "disagreement_score": 1.0}
    # Fundamentals source metrics
    f_sources = agg_fund.get("sources_used", []) or []
    f_disagreement = float(agg_fund.get("disagreement_score", 1.0) or 1.0)
    coverage_dict = agg_fund.get("coverage", {}) or {}
    covered_fields = sum(1 for f in multi_sources_fields if coverage_dict.get(f))
    coverage_pct = covered_fields / float(len(multi_sources_fields)) if multi_sources_fields else 0.0
    # Fundamental reliability formula:
    #   base = coverage_pct * (1 - disagreement)
    #   source_factor = 0.5 + 0.5 * min(count/4, 1)
    #   reliability_raw = base * source_factor
    #   floor 0.15 if at least one source
    source_factor = 0.5 + 0.5 * min(len(f_sources) / 4.0, 1.0)
    fund_raw = coverage_pct * (1.0 - f_disagreement) * source_factor
    if len(f_sources) >= 1:
        fund_raw = max(fund_raw, 0.15 * source_factor)  # ensure partial reliability when any source contributes
    fund_reliability_pct = max(0.0, min(fund_raw * 100.0, 100.0))

    # Price multi-source
    try:
        prices_map = fetch_price_multi_v2(tkr)
    except Exception:
        prices_map = {}
    mean_price, std_price, price_count = aggregate_price_v2(prices_map)
    if mean_price and mean_price > 0 and np.isfinite(std_price):
        variance_ratio = min(std_price / mean_price, 1.0)
    else:
        variance_ratio = 1.0
    # Price reliability:
    #   stability = (1 - variance_ratio)
    #   source_factor_price = min(price_count/5, 1)
    #   reliability_raw = stability * (0.4 + 0.6*source_factor_price)
    #   floor 0.20 if any source
    source_factor_price = min(price_count / 5.0, 1.0)
    price_raw = (1.0 - variance_ratio) * (0.4 + 0.6 * source_factor_price)
    if price_count >= 1:
        price_raw = max(price_raw, 0.20 * (0.4 + 0.6 * source_factor_price))
    price_reliability_pct = max(0.0, min(price_raw * 100.0, 100.0))

    # Append lists
    fund_sources_used_list.append(len(f_sources))
    fund_disagreement_list.append(f_disagreement)
    fund_coverage_pct_list.append(coverage_pct)
    fund_reliability_list.append(fund_reliability_pct)
    price_sources_used_list.append(price_count)
    price_variance_score_list.append(variance_ratio)
    price_reliability_list.append(price_reliability_pct)
    price_mean_list.append(mean_price)
    price_std_list.append(std_price)

# Inject columns
results["fund_sources_used_v2"] = fund_sources_used_list
results["fund_disagreement_score_v2"] = fund_disagreement_list
results["fund_coverage_pct_v2"] = fund_coverage_pct_list
results["Fundamental_Reliability_v2"] = fund_reliability_list
results["price_sources_used_v2"] = price_sources_used_list
results["price_variance_score_v2"] = price_variance_score_list
results["Price_Reliability_v2"] = price_reliability_list
results["Price_Mean_v2"] = price_mean_list
results["Price_STD_v2"] = price_std_list

# Combined reliability (override or create reliability_v2)
if "reliability_v2" not in results.columns or results["reliability_v2"].isna().all():
    combined_rel = 0.6 * results["Fundamental_Reliability_v2"] + 0.4 * results["Price_Reliability_v2"]
    results["reliability_v2"] = combined_rel.clip(0, 100)
    
    # V2 RISK ENGINE: Apply FULL risk gates, reliability penalties, and position sizing
    st.info("🚨 Applying V2 risk gates and reliability enforcement...")
    
    budget = float(st.session_state.get("budget", CONFIG.get("BUDGET", 5000)))
    
    # Apply enhanced V2 scoring to each ticker
    v2_results = []
    for idx, row in results.iterrows():
        ticker = row.get("Ticker")
        v2_result = score_ticker_v2_enhanced(
            ticker,
            row,
            budget_total=float(st.session_state.get("total_budget", CONFIG.get("BUDGET_TOTAL", 5000))),
            min_position=float(CONFIG.get("MIN_POSITION", 50.0)),
            enable_ml=bool(CONFIG.get("ENABLE_ML", True))
        )
        v2_results.append(v2_result)
    
    # Add V2 columns to results DataFrame
    v2_df = pd.DataFrame(v2_results)
    
    # Merge V2 results back into main results
    for col in v2_df.columns:
        if col != "ticker":
            results[col] = v2_df[col].values
    
    # Use V2 final conviction as Score
    results["Score"] = results["conviction_v2_final"]
    
    # Add risk meter
    results["risk_meter_v2"] = 100 - results["conviction_v2_final"]
    
    logger.info(f"✓ Applied V2 risk gates to {len(results)} stocks")
    logger.info(f"  - Blocked: {len(results[results['risk_gate_status_v2'] == 'blocked'])} stocks")
    logger.info(f"  - Severely reduced: {len(results[results['risk_gate_status_v2'] == 'severely_reduced'])} stocks")
    logger.info(f"  - Reduced: {len(results[results['risk_gate_status_v2'] == 'reduced'])} stocks")
    logger.info(f"  - Full allocation: {len(results[results['risk_gate_status_v2'] == 'full'])} stocks")
    
    results = results.sort_values(
        ["Score", "Ticker"], ascending=[False, True]
    ).reset_index(drop=True)
    phase_times["fundamentals_alpha_finnhub"] = t_end(t0)
    logger.info(f"✓ Scored {take_k} stocks with V2-enhanced conviction")

    # Ensure canonical V2 column aliases exist for UI/CSV and enforce blocked zeros
    # Map rr -> reward_risk_v2, reliability -> reliability_score_v2
    if "rr_ratio_v2" in results.columns:
        results["reward_risk_v2"] = results["rr_ratio_v2"]
    else:
        results["reward_risk_v2"] = results.get("RR_Ratio", np.nan)

    if "reliability_v2" in results.columns:
        results["reliability_score_v2"] = results["reliability_v2"]
    else:
        results["reliability_score_v2"] = results.get("Reliability_Score", np.nan)

    # Ensure presence of other requested keys (fill with defaults if missing)
    for k in [
        "risk_gate_status_v2", "risk_gate_reason_v2", "conviction_v2_base", "conviction_v2_final",
        "buy_amount_v2", "shares_to_buy_v2", "fund_sources_used_v2", "price_sources_used_v2",
        "fund_disagreement_score_v2", "price_variance_score_v2"
    ]:
        if k not in results.columns:
            results[k] = np.nan

    # Unified RR evaluation (raw ratio + penalized score + band)
    results["rr"] = results.get("RewardRisk", results.get("RR_Ratio", np.nan))
    def _rr_eval(row_rr):
        try:
            score, ratio_adj, band = evaluate_rr_unified(float(row_rr))
            return score, band
        except Exception:
            return np.nan, "N/A"
    rr_evals = results["rr"].apply(_rr_eval)
    results["rr_score_v2"] = [e[0] for e in rr_evals]
    results["rr_band"] = [e[1] for e in rr_evals]

    # Provide canonical aliases expected by unified scoring engine
    # RR_Score (0-100), RR ratio (RR), Reliability_v2 (0-100)
    results["RR_Score"] = results["rr_score_v2"].copy()
    # Ensure raw ratio alias RR for penalty logic; fall back gracefully
    if "rr" in results.columns:
        results["RR"] = results["rr"].copy()
    else:
        results["RR"] = results.get("RewardRisk", results.get("RR_Ratio", np.nan))
    if "reliability_v2" in results.columns:
        results["Reliability_v2"] = results["reliability_v2"].copy()
    else:
        results["Reliability_v2"] = results.get("reliability_score_v2", np.nan)

    # === COMPUTE OVERALL SCORE WITH EXPLICIT FORMULA ===
    # Use the new compute_overall_score function from scoring_engine
    # Formula: 35% fund + 35% tech + 15% RR + 15% reliability ± ML (max ±10%)
    # Includes penalties for realistic spread
    
    def _compute_overall(row):
        try:
            from core.scoring_engine import compute_overall_score
            score, components = compute_overall_score(row)
            return pd.Series({
                'overall_score': score,
                'fund_component': components.get('fund_component', 0.0),
                'tech_component': components.get('tech_component', 0.0),
                'rr_component': components.get('rr_component', 0.0),
                'reliability_component': components.get('reliability_component', 0.0),
                'base_score': components.get('base_score', 0.0),
                'ml_delta': components.get('ml_delta', 0.0),
                'score_before_penalties': components.get('score_before_penalties', 0.0),
                'penalty_total': components.get('penalty_total', 0.0),
            })
        except Exception as e:
            logger.warning(f"compute_overall_score failed: {e}")
            # Fallback to old logic
            return pd.Series({
                'overall_score': row.get('conviction_v2_final', 50.0),
                'fund_component': 0.0,
                'tech_component': 0.0,
                'rr_component': 0.0,
                'reliability_component': 0.0,
                'base_score': 0.0,
                'ml_delta': 0.0,
                'score_before_penalties': 0.0,
                'penalty_total': 0.0,
            })
    
    overall_components = results.apply(_compute_overall, axis=1)
    results = pd.concat([results, overall_components], axis=1)

    # === PRESERVE RAW SCORE + COMPUTE PRETTY SCORE FOR DISPLAY ===
    # Keep overall_score as the raw model score for all internal logic
    results["overall_score_raw"] = results["overall_score"].copy()
    
    # Compute "pretty" score in 60-90 range for display only
    raw_scores = results["overall_score_raw"]
    s_min = float(raw_scores.min())
    s_max = float(raw_scores.max())
    
    if s_max == s_min:
        # All scores are identical, use neutral value
        results["overall_score_pretty"] = 75.0
    else:
        # Normalize to 0-1, then scale to 60-90
        normalized = (raw_scores - s_min) / (s_max - s_min)
        pretty = 60.0 + normalized * 30.0
        results["overall_score_pretty"] = pretty.clip(0, 100)
    
    logger.info(f"Score mapping: raw [{s_min:.1f}, {s_max:.1f}] → pretty [60, 90]")

    # === ADD DISPLAY BANDS FOR UI ===
    # Reliability band (High/Medium/Low)
    def reliability_band(x):
        if pd.isna(x) or not np.isfinite(x):
            return "N/A"
        if x >= 75:
            return "High"
        if x >= 40:
            return "Medium"
        return "Low"
    
    # Get reliability score (0-100 scale)
    rel_col = results.get("reliability_v2", results.get("Reliability_v2", results.get("Reliability Score v2", np.nan)))
    if isinstance(rel_col, pd.Series):
        results["reliability_pct"] = rel_col.clip(0, 100)
    else:
        results["reliability_pct"] = 50.0
    results["reliability_band"] = results["reliability_pct"].apply(reliability_band)
    
    # Risk band (Low/Medium/High/Very High)
    def risk_band(v):
        if pd.isna(v) or not np.isfinite(v):
            return "N/A"
        if v < 45:
            return "Low"
        if v < 60:
            return "Medium"
        if v < 75:
            return "High"
        return "Very High"
    
    risk_col = results.get("risk_meter_v2", results.get("RiskMeter", np.nan))
    if isinstance(risk_col, pd.Series):
        results["risk_band"] = risk_col.apply(risk_band)
    else:
        results["risk_band"] = "N/A"
    
    # ML Confidence band (Low/Medium/High)
    def ml_conf_band(p):
        if pd.isna(p) or not np.isfinite(p):
            return "N/A"
        if p < 0.60:
            return "Low"
        if p < 0.75:
            return "Medium"
        return "High"
    
    ml_prob_col = results.get("ML_Probability", np.nan)
    if isinstance(ml_prob_col, pd.Series):
        results["ml_conf_band"] = ml_prob_col.apply(ml_conf_band)
    else:
        results["ml_conf_band"] = "N/A"

    # Component breakdowns (fallback to legacy names if needed)
    results["fund_score"] = results.get("fundamental_score_v2", results.get("Fundamental Score", np.nan))
    results["tech_score"] = results.get("technical_score_v2", results.get("Tech Score", np.nan))

    results["rr_score"] = results.get("rr_score_v2", np.nan)
    results["reliability_score"] = results.get("reliability_score_v2", results.get("Reliability Score", np.nan))
    base_conv = results.get("conviction_v2_base", results.get("conviction_v2_final", np.nan))
    results["ml_delta"] = results["overall_score"] - base_conv

    # Enforce blocked rows have zero buy and shares
    if "risk_gate_status_v2" in results.columns:
        blocked_mask = results["risk_gate_status_v2"] == "blocked"
        if "buy_amount_v2" in results.columns:
            results.loc[blocked_mask, "buy_amount_v2"] = 0.0
        if "shares_to_buy_v2" in results.columns:
            results.loc[blocked_mask, "shares_to_buy_v2"] = 0


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

# Fallback: if no Core, promote top relaxed candidates from Speculative
# Updated Nov 2025: More lenient fallback aligned with balanced criteria
if len(core_stocks) == 0 and not spec_stocks.empty:
    st.warning("⚠️ No Core stocks classified. Applying adaptive Core fallback…")
    try:
        rr = spec_stocks.get("RewardRisk")
        if rr is None:
            rr = spec_stocks.get("RR_Ratio")
        # Fallback uses even more relaxed criteria (RSI 20-60, RR≥0.8)
        mask = (
            spec_stocks.get("RSI").between(20, 60, inclusive="both")
            & (rr.fillna(0) >= 0.8)
            & (spec_stocks.get("Momentum_Consistency", pd.Series([0]*len(spec_stocks))).fillna(0) >= 0.3)
            & (spec_stocks.get("ATR_Price", pd.Series([1]*len(spec_stocks))).fillna(1) <= 0.09)
            & (spec_stocks.get("OverextRatio", pd.Series([1]*len(spec_stocks))).fillna(1) <= 0.15)
            & (spec_stocks.get("Should_Display", pd.Series([True]*len(spec_stocks))).fillna(True))
        )
        fallback = spec_stocks[mask].copy().sort_values(["Score","Ticker"], ascending=[False, True]).head(5)
        if not fallback.empty:
            fallback["Risk_Level"] = "core"
            # Mark reason
            if "Classification_Warnings" in fallback.columns:
                fallback["Classification_Warnings"] = (
                    fallback["Classification_Warnings"].fillna("") + "; Adaptive Core fallback"
                ).str.strip("; ")
            else:
                fallback["Classification_Warnings"] = "Adaptive Core fallback"
            # Lower confidence if missing
            if "Confidence_Level" in fallback.columns:
                fallback.loc[fallback["Confidence_Level"].isna(), "Confidence_Level"] = "low"
            else:
                fallback["Confidence_Level"] = "low"
            # Update sets
            core_stocks = fallback
            spec_stocks = spec_stocks.drop(fallback.index)
            st.info(f"🔄 Promoted {len(core_stocks)} fallback Core candidates based on relaxed criteria")
        else:
            logger.info("Adaptive Core fallback found no eligible candidates")
    except Exception as e:
        logger.warning(f"Adaptive Core fallback error: {e}")

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
        def _ml_band(p: float) -> str:
            if not np.isfinite(p):
                return 'N/A'
            if p < 0.60:
                return 'Low'
            if p < 0.75:
                return 'Medium'
            return 'High'
        core_filtered['ml_conf_band'] = core_filtered['ML_Probability'].apply(_ml_band)
        core_filtered['ML_Confidence'] = core_filtered['ml_conf_band']
        core_filtered = core_filtered.sort_values('ML_Probability', ascending=False)
        logger.info(f"Core stocks scored: avg probability {core_filtered['ML_Probability'].mean():.3f}")
    
    # Score Speculative stocks
    if not spec_filtered.empty:
        spec_filtered['ML_Probability'] = spec_filtered.apply(score_with_xgboost, axis=1)
        spec_filtered['ml_conf_band'] = spec_filtered['ML_Probability'].apply(_ml_band)
        spec_filtered['ML_Confidence'] = spec_filtered['ml_conf_band']
        spec_filtered = spec_filtered.sort_values('ML_Probability', ascending=False)
        logger.info(f"Speculative stocks scored: avg probability {spec_filtered['ML_Probability'].mean():.3f}")
else:
    logger.info("ML scoring skipped - model not available")
    if not core_filtered.empty:
        core_filtered['ML_Probability'] = np.nan
        core_filtered['ml_conf_band'] = 'N/A'
        core_filtered['ML_Confidence'] = 'N/A'
    if not spec_filtered.empty:
        spec_filtered['ML_Probability'] = np.nan
        spec_filtered['ml_conf_band'] = 'N/A'
        spec_filtered['ML_Confidence'] = 'N/A'

# === CALCULATE QUALITY SCORE FOR ALL STOCKS ===
from core.scoring_engine import calculate_quality_score

def _calc_quality(row):
    try:
        score, level = calculate_quality_score(row)
        return pd.Series({'Quality_Score_Numeric': score, 'Quality_Level': level})
    except Exception as e:
        logger.warning(f"Quality calculation failed: {e}")
        return pd.Series({'Quality_Score_Numeric': 0.5, 'Quality_Level': "Medium"})

if not core_filtered.empty:
    quality_data = core_filtered.apply(_calc_quality, axis=1)
    core_filtered['Quality_Score_Numeric'] = quality_data['Quality_Score_Numeric']
    core_filtered['Quality_Level'] = quality_data['Quality_Level']
    logger.info(f"Core quality: {core_filtered['Quality_Level'].value_counts().to_dict()}")

if not spec_filtered.empty:
    quality_data = spec_filtered.apply(_calc_quality, axis=1)
    spec_filtered['Quality_Score_Numeric'] = quality_data['Quality_Score_Numeric']
    spec_filtered['Quality_Level'] = quality_data['Quality_Level']
    logger.info(f"Spec quality: {spec_filtered['Quality_Level'].value_counts().to_dict()}")

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
    st.write(
        f"- 🔴 Consider relaxing CONFIG: MIN_QUALITY_SCORE_CORE={CONFIG.get('MIN_QUALITY_SCORE_CORE', 25.0)}, "
        f"MAX_ATR_PRICE_CORE={CONFIG.get('MAX_ATR_PRICE_CORE', 0.06)}"
    )
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

# Compute standard deviation from historical price data (last 20-30 candles)
results["Historical_StdDev"] = np.nan
for i, row in results.iterrows():
    ticker = row.get("Ticker", "")
    if ticker in data_map:
        hist = data_map[ticker]
        if len(hist) >= 5:  # Minimum 5 candles
            # Use last 20-30 candles for std dev
            recent = hist["Close"].tail(min(30, len(hist)))
            if len(recent) >= 5:
                results.at[i, "Historical_StdDev"] = float(recent.std())


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
    with st.spinner(f"💵 Verifying prices across {len(subset_idx)} top stocks..."):
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
    # Price reliability metric (enhanced with better spread: 0.1-1.0)
    results["Price_Reliability"] = np.nan
    for i, row in results.iterrows():
        pmean = row.get("Price_Mean", np.nan)
        pstd = row.get("Price_STD", np.nan)
        providers = str(row.get("Source_List", "")).split(" · ") if isinstance(row.get("Source_List"), str) else []
        providers = [p for p in providers if p]
        count = len(providers)
        
        if np.isfinite(pmean) and pmean > 0 and np.isfinite(pstd):
            pct_std = pstd / pmean
            
            # Source count component (0-4 sources → 0.2-1.0)
            if count == 1:
                source_score = 0.2
            elif count == 2:
                source_score = 0.5
            elif count == 3:
                source_score = 0.75
            else:  # 4+
                source_score = 1.0
            
            # Variance component (lower variance = higher reliability)
            # 0% variance → 1.0, 5% variance → 0.5, 10%+ variance → 0.2
            if pct_std < 0.01:  # <1%
                variance_score = 1.0
            elif pct_std < 0.03:  # 1-3%
                variance_score = 0.85
            elif pct_std < 0.05:  # 3-5%
                variance_score = 0.65
            elif pct_std < 0.10:  # 5-10%
                variance_score = 0.4
            else:  # >10%
                variance_score = 0.2
            
            # Combined: 60% source count, 40% variance
            reliability = (source_score * 0.6) + (variance_score * 0.4)
            results.at[i, "Price_Reliability"] = round(np.clip(reliability, 0.1, 1.0), 4)
        else:
            # Fallback based on source count only
            if count == 1:
                results.at[i, "Price_Reliability"] = 0.15
            elif count == 2:
                results.at[i, "Price_Reliability"] = 0.35
            elif count == 3:
                results.at[i, "Price_Reliability"] = 0.55
            else:
                results.at[i, "Price_Reliability"] = 0.75

    # Price sources count column
    results["Price_Sources_Count"] = results["Source_List"].apply(lambda s: len(str(s).split(" · ")) if isinstance(s, str) and s else 0)

    # Fundamental reliability metric
    # Simplified Fundamental reliability metric based on number of distinct
    # fundamental providers that contributed any field. Mapping:
    #   0 -> 0.0
    #   1 -> 0.33
    #   2 -> 0.66
    #  >=3 -> 1.0
    def _map_sources_to_reliability(n: int) -> float:
        if n <= 0:
            return 0.0
        if n == 1:
            return 0.33
        if n == 2:
            return 0.66
        return 1.0

    if "Fund_Coverage_Pct" in results.columns:
        results["Fundamental_Reliability"] = 0.0
        fund_flags = [
            'from_fmp_full', 'from_fmp', 'from_simfin', 'from_eodhd',
            'from_alpha', 'from_finnhub', 'from_tiingo'
        ]
        for i, row in results.iterrows():
            # Count providers that truly contributed (flags are written earlier)
            provider_count = int(sum(bool(row.get(f)) for f in fund_flags))
            # If coverage indicates data but flags are zero, treat as 1 source
            cov = row.get("Fund_Coverage_Pct", np.nan)
            if provider_count == 0 and isinstance(cov, (int, float)) and np.isfinite(cov) and cov > 0:
                provider_count = 1
            final_rel = _map_sources_to_reliability(provider_count)
            results.at[i, "Fundamental_Reliability"] = round(final_rel, 4)
    else:
        results["Fundamental_Reliability"] = 0.0

    # Fundamental sources count column (flags) — exact count (0 allowed if truly none)
    fund_flags = ["from_fmp_full", "from_fmp", "from_simfin", "from_eodhd", "from_alpha", "from_finnhub", "from_tiingo"]
    def _fund_count(row: pd.Series) -> int:
        return int(sum(bool(row.get(f)) for f in fund_flags))
    results["Fundamental_Sources_Count"] = results.apply(_fund_count, axis=1)

    # Expose canonical fund reliability field using the simple mapping helper
    try:
        from core.scoring.fundamental import compute_fund_reliability
    except Exception:
        compute_fund_reliability = None

    if compute_fund_reliability is not None:
        results["fund_reliability"] = results["Fundamental_Sources_Count"].apply(lambda n: float(compute_fund_reliability(n)))
        # Keep legacy column name in sync
        results["Fundamental_Reliability"] = results["fund_reliability"].copy()
    else:
        # Fallback: copy existing value or 0.0
        results["fund_reliability"] = results.get("Fundamental_Reliability", 0.0)

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

# Show ALL stocks that passed filters (no limit)
TOPN = len(results)

# Apply ML confidence threshold
ml_threshold_value = float(st.session_state.get("ml_threshold", 0)) / 100.0
if ml_threshold_value > 0 and "ML_Prob" in results.columns:
    before_ml = len(results)
    results = results[results["ML_Prob"] >= ml_threshold_value].copy()
    after_ml = len(results)
    logger.info(f"ML confidence filter: {before_ml} → {after_ml} stocks (threshold={ml_threshold_value:.0%})")
    TOPN = len(results)  # Update TOPN after filtering

alloc_df = results.reset_index(drop=True).copy()

# V2 ALLOCATION: Use buy_amount_v2 from risk engine (already has gates + penalties applied)
# For backward compatibility, keep old AllocScore but use buy_amount_v2 as primary allocation
if "buy_amount_v2" in alloc_df.columns:
    logger.info("Using V2 position sizing with risk gates and reliability penalties")
    
    # Use buy_amount_v2 directly - it already includes:
    # - Risk gate penalties (blocked stocks = $0)
    # - Reliability penalties (low reliability = reduced allocation)
    # - Core vs speculative caps (3% vs 15% of budget)
    # - ML boost applied AFTER penalties
    alloc_df["Buy_Amount"] = alloc_df["buy_amount_v2"]
    alloc_df["AllocScore"] = alloc_df["Score"].clip(lower=0)  # Keep for display only
    
else:
    # Fallback to old allocation logic if V2 columns missing
    logger.warning("V2 columns missing - using legacy allocation logic")
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

# Skip legacy allocate_budget if using V2 (already sized)
if "buy_amount_v2" not in alloc_df.columns:
    results = allocate_budget(
        alloc_df,
        float(st.session_state.get("total_budget", CONFIG["BUDGET_TOTAL"])),
        float(st.session_state.get("min_position", CONFIG["MIN_POSITION"])),
        float(st.session_state.get("max_position_pct", CONFIG["MAX_POSITION_PCT"])),
        score_col="AllocScore",
    )
else:
    # V2: Use already-computed Buy_Amount
    results = alloc_df.copy()
    # Ensure Hebrew purchase amount column exists for downstream calculations
    if "סכום קנייה ($)" not in results.columns:
        if "Buy_Amount" in results.columns:
            results["סכום קנייה ($)"] = results["Buy_Amount"].round(2)
        elif "buy_amount_v2" in results.columns:
            results["סכום קנייה ($)"] = results["buy_amount_v2"].round(2)
        else:
            results["סכום קנייה ($)"] = 0.0
    
results["מניות לקנייה"] = np.floor(
    np.where(
        results["Unit_Price"] > 0, results["סכום קנייה ($)"] / results["Unit_Price"], 0
    )
).astype(int)
results["עודף ($)"] = np.round(
    results["סכום קנייה ($)"] - results["מניות לקנייה"] * results["Unit_Price"], 2
)

# === Global budget cap enforcement (scaling if needed) ===
total_budget_value = float(st.session_state.get('total_budget', CONFIG['BUDGET_TOTAL']))
results['position_value'] = results['Unit_Price'].fillna(0) * results['מניות לקנייה']
total_alloc = float(results['position_value'].sum())
if total_alloc > total_budget_value and total_alloc > 0:
    scale = total_budget_value / total_alloc
    scaled_shares = (results['מניות לקנייה'] * scale).apply(lambda x: max(int(round(x)), 0))
    results['מניות לקנייה'] = scaled_shares
    results['position_value'] = results['Unit_Price'].fillna(0) * results['מניות לקנייה']
    # Recompute leftover and purchase amount columns to reflect scaled allocation
    results['סכום קנייה ($)'] = results['position_value'].round(2)
    results['עודף ($)'] = 0.0  # leftover per row not tracked post-scale


# KPI
budget_used = float(
    results["מניות לקנייה"].to_numpy() @ results["Unit_Price"].fillna(0).to_numpy()
)
k0, k1, k2, k3 = st.columns(4)
k0.metric("Universe size after history filtering", len(data_map))
k1.metric("Results after filtering", len(results))
total_budget_value = float(st.session_state.get('total_budget', CONFIG['BUDGET_TOTAL']))
budget_used = min(budget_used, total_budget_value)  # safety clamp
k2.metric("Budget used (≈$)", f"{budget_used:,.0f}")
k3.metric("Remaining budget (≈$)", f"{max(0.0, total_budget_value - budget_used):,.0f}")

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

    # V2 SCORING ENGINE (NOW DEFAULT)
    st.markdown("---")
    st.subheader("🚀 Advanced Scoring")
    st.caption("Multi-source fundamentals (FMP→Finnhub→Tiingo→Alpha) • Unified conviction (35% fund, 35% tech, 15% RR, 15% reliability) • ML boost ±10%")
    
    # Always enabled, but allow configuration
    enable_multi_source = st.checkbox(
        "Fetch multi-source fundamentals",
        value=bool(st.session_state.get("enable_multi_source", True)),
        help="Fetch and aggregate fundamentals from all available sources for better reliability."
    )
    st.session_state["enable_multi_source"] = enable_multi_source
    
    enable_ml_boost = st.checkbox(
        "Enable ML confidence boost",
        value=bool(st.session_state.get("enable_ml_boost", True)),
        help="Allow ML model to adjust conviction by ±10% based on historical patterns."
    )
    st.session_state["enable_ml_boost"] = enable_ml_boost
    
    # OpenAI target price enhancement (ENABLED BY DEFAULT)
    if OPENAI_AVAILABLE and (os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")):
        enable_openai_targets = st.checkbox(
            "🤖 Enable AI-enhanced target prices & timing",
            value=bool(st.session_state.get("enable_openai_targets", True)),
            help="Use OpenAI GPT-4o-mini to predict target prices AND holding periods based on fundamentals + technicals. Without this, timing is calculated from RR + RSI + Momentum."
        )
        st.session_state["enable_openai_targets"] = enable_openai_targets
        
        # Show status
        if enable_openai_targets:
            st.success("✅ AI predictions ACTIVE - target dates will be AI-generated")
        else:
            st.info("ℹ️ AI predictions OFF - using technical calculation (RR + RSI + Momentum)")
    else:
        st.session_state["enable_openai_targets"] = False
        if not OPENAI_AVAILABLE:
            st.caption("ℹ️ Install `openai` package for AI target predictions")
        elif not (os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")):
            st.caption("ℹ️ Add OPENAI_API_KEY to enable AI predictions")

    st.markdown("---")
    st.subheader("💰 Allocation")
    
    # ML confidence threshold
    ml_threshold = st.slider(
        "ML confidence threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=int(st.session_state.get("ml_threshold", 0)), 
        step=5,
        help="Minimum ML probability to include stock (0=disabled). 🔥High: 70%+, 🟡Med: 50-70%, ⚠️Low: <50%"
    )
    st.session_state["ml_threshold"] = int(ml_threshold)
    
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
# Prefer V2 strict buy amounts for recommendations; fallback to legacy Hebrew buy column
if "buy_amount_v2" in results.columns:
    rec_df = results[results["buy_amount_v2"].fillna(0) > 0].copy()
elif "סכום קנייה ($)" in results.columns:
    rec_df = results[results["סכום קנייה ($)"].fillna(0) > 0].copy()
else:
    # Fallback: empty selection if no buy columns present
    rec_df = results.copy()

# Explicitly exclude tickers blocked by the strict V2 gate
if "risk_gate_status_v2" in rec_df.columns:
    rec_df = rec_df[rec_df["risk_gate_status_v2"] != "blocked"].copy()

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

# Add overall ranking (1 = best)
rec_df = rec_df.sort_values("Score", ascending=False).reset_index(drop=True)
rec_df["Overall_Rank"] = range(1, len(rec_df) + 1)

# Calculate target prices and dates WITH OPTIONAL OPENAI ENHANCEMENT
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)
def get_openai_target_prediction(ticker: str, current_price: float, fundamentals: dict, technicals: dict) -> Optional[Tuple[float, int]]:
    """
    Use OpenAI to predict realistic target price AND holding period based on fundamentals and technicals.
    Returns (target_price, days_to_target) tuple or None if API unavailable or request fails.
    """
    if not OPENAI_AVAILABLE:
        return None
    
    openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not openai_key:
        return None
    
    try:
        client = OpenAI(api_key=openai_key)
        
        # Build context from fundamentals
        fund_str = ", ".join([f"{k}: {v}" for k, v in fundamentals.items() if v is not None and not (isinstance(v, float) and np.isnan(v))])
        tech_str = ", ".join([f"{k}: {v:.2f}" for k, v in technicals.items() if isinstance(v, (int, float)) and np.isfinite(v)])
        
        prompt = f"""You are a financial analyst. Based on the following data for {ticker}:

Current Price: ${current_price:.2f}

Fundamentals: {fund_str}

Technical Indicators: {tech_str}

Provide TWO predictions as a JSON object:
1. Target Price: realistic price target considering growth trends, valuation, momentum, and risk/reward
2. Days to Target: estimated holding period in days to reach this target (typically 7-180 days based on momentum and catalysts)

Return ONLY a JSON object with this exact format:
{{"target_price": <number>, "days_to_target": <integer>}}

JSON:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        # Extract JSON from response
        answer = response.choices[0].message.content.strip()
        import json
        import re
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', answer)
        if json_match:
            data = json.loads(json_match.group(0))
            target = float(data.get("target_price", 0))
            days = int(data.get("days_to_target", 20))
            # Sanity checks
            if current_price * 0.5 <= target <= current_price * 3.0 and 7 <= days <= 365:
                return (target, days)
    except Exception as e:
        logger.warning(f"OpenAI target prediction failed for {ticker}: {e}")
    
    return None

def calculate_targets(row):
    """Calculate entry price, target price, and target date based on ATR, RR, and optional OpenAI prediction"""
    ticker = row.get("Ticker", "")
    current_price = row.get("Unit_Price", row.get("Price_Yahoo", np.nan))
    atr = row.get("ATR", np.nan)
    rr = row.get("RewardRisk", np.nan)
    rsi = row.get("RSI", np.nan)
    momentum = row.get("Momentum_63d", np.nan)
    sector = row.get("Sector", "")
    ml_prob = row.get("ml_probability", 0.5)
    
    if np.isfinite(current_price) and current_price > 0:
        # Entry price: current - 0.5*ATR (wait for slight pullback)
        if np.isfinite(atr):
            entry_price = current_price - (0.5 * atr)
        else:
            entry_price = current_price * 0.98  # 2% below if no ATR
        
        # Calculate volatility factor for date variability
        atr_pct = (atr / current_price) if (np.isfinite(atr) and current_price > 0) else 0.02
        volatility_factor = np.clip(atr_pct / 0.03, 0.5, 2.5)  # 0.5x to 2.5x multiplier
        
        # Sector-based offset (defensive sectors slower, growth sectors faster)
        sector_offsets = {
            "Utilities": 1.3, "Consumer Defensive": 1.2, "Real Estate": 1.15,
            "Financials": 1.1, "Healthcare": 1.0, "Industrials": 0.95,
            "Energy": 0.9, "Consumer Cyclical": 0.85, "Technology": 0.75,
            "Communication Services": 0.8
        }
        sector_mult = sector_offsets.get(sector, 1.0)
        
        # ML probability influence (higher confidence = shorter timeline)
        ml_mult = 1.0
        if isinstance(ml_prob, (int, float)) and np.isfinite(ml_prob):
            ml_mult = 1.2 - (ml_prob * 0.4)  # 0.5→1.0, 1.0→0.8 (high conf = faster)
        
        # Calculate fallback days from multiple factors (more dynamic)
        if np.isfinite(rr):
            # Base days from RR
            base_days = 20 + (rr * 10)
            
            # Adjust based on RSI: oversold (< 40) = faster, overbought (> 70) = slower
            if np.isfinite(rsi):
                if rsi < 40:
                    base_days *= 0.75  # Strong momentum, faster target
                elif rsi > 70:
                    base_days *= 1.3   # Overbought, slower target
            
            # Adjust based on momentum: strong momentum = faster
            if np.isfinite(momentum) and momentum > 0.05:
                base_days *= 0.9  # Strong uptrend, faster
            elif np.isfinite(momentum) and momentum < -0.05:
                base_days *= 1.2  # Weak trend, slower
            
            # Apply volatility, sector, and ML multipliers
            base_days *= volatility_factor * sector_mult * ml_mult
            
            # Add ticker-specific variance (hash-based to keep consistent per ticker)
            ticker_seed = sum(ord(c) for c in ticker) % 20
            base_days += ticker_seed
            
            days = int(min(180, max(14, base_days)))
        else:
            # Fallback: use volatility factor for diverse dates (30-180 days)
            base_days = 60 * volatility_factor * sector_mult * ml_mult
            ticker_seed = sum(ord(c) for c in ticker) % 30
            days = int(min(180, max(30, base_days + ticker_seed)))
        
        # Try OpenAI-enhanced target (returns both price and days)
        ai_result = None
        if st.session_state.get("enable_openai_targets", False):
            fundamentals = {
                "PE": row.get("PERatio", np.nan),
                "PB": row.get("PBRatio", np.nan),
                "ROE": row.get("ROE", np.nan),
                "Margin": row.get("ProfitMargin", np.nan),
                "RevenueGrowth": row.get("RevenueGrowthYoY", np.nan)
            }
            technicals = {
                "RSI": rsi,
                "Momentum_63d": momentum,
                "RewardRisk": rr,
                "ATR": atr
            }
            try:
                ai_result = get_openai_target_prediction(ticker, current_price, fundamentals, technicals)
            except Exception as e:
                logger.warning(f"OpenAI call failed for {ticker}: {e}")
                ai_result = None
        
        if ai_result is not None:
            # Use AI prediction for both target price AND timing
            target_price, days = ai_result
            target_source = "AI"
        elif np.isfinite(atr) and np.isfinite(rr):
            # Fallback to technical calculation: entry + (RR * ATR)
            target_price = entry_price + (rr * atr)
            target_source = "Technical"
            # days already calculated above from RR + RSI + momentum
        else:
            # Conservative default: 10% above entry
            target_price = entry_price * 1.10
            target_source = "Default"
            # days already set to 30
        
        # Target date: today + holding horizon (now from AI or calculated)
        target_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        return entry_price, target_price, target_date, target_source
    else:
        return current_price, np.nan, "N/A", "N/A"

with st.spinner(f"🎯 Calculating targets for {len(rec_df)} stocks..."):
    rec_df[["Entry_Price", "Target_Price", "Target_Date", "Target_Source"]] = rec_df.apply(
        lambda row: pd.Series(calculate_targets(row)), axis=1
    )


def calculate_rr(entry_price: float, target_price: float, atr_value: float, fallback_price: float = None) -> float:
    """
    Calculate Reward/Risk using the requested formula:
    RR = clamp( (target_price - entry_price) / max(atr_value*2, entry_price*0.01), 0, 5 )
    If ATR is missing, try to use provided ATR value or a conservative fallback (1% of price).
    Returns numeric RR (float).
    """
    try:
        if not (isinstance(entry_price, (int, float)) and np.isfinite(entry_price)):
            return np.nan
        if not (isinstance(target_price, (int, float)) and np.isfinite(target_price)):
            return np.nan

        atr = atr_value if isinstance(atr_value, (int, float)) and np.isfinite(atr_value) else np.nan
        # If atr not provided, try common fallback field name from row via fallback_price
        if not np.isfinite(atr) and isinstance(fallback_price, (int, float)) and np.isfinite(fallback_price):
            atr = fallback_price

        # Final ATR fallback: 1% of entry price
        if not np.isfinite(atr):
            atr = max(0.01 * float(entry_price), 1e-6)

        risk = max(atr * 2.0, float(entry_price) * 0.01)
        reward = float(target_price) - float(entry_price)
        rr = 0.0 if risk <= 0 else reward / risk
        rr = float(np.clip(rr, 0.0, 5.0))
        return rr
    except Exception:
        return np.nan


# Recalculate Reward/Risk after targets are known so RR uses the actual target price and ATR
if not rec_df.empty:
    # ATR value may be stored under 'ATR' or 'ATR14' or 'ATR_Price'
    def _compute_rr_row(r):
        entry = r.get("Entry_Price", r.get("Unit_Price", r.get("Price_Yahoo", np.nan)))
        target = r.get("Target_Price", np.nan)
        atr = r.get("ATR", r.get("ATR14", r.get("ATR14", np.nan)))
        # fallback ATR in price terms (ATR_Price)
        atr_price = r.get("ATR_Price", np.nan)
        rr_val = calculate_rr(entry, target, atr, fallback_price=atr_price)
        return rr_val

    rec_df["RewardRisk"] = rec_df.apply(lambda r: round(_compute_rr_row(r), 2) if np.isfinite(_compute_rr_row(r)) else np.nan, axis=1)
    # Also update RR_Ratio alias used in classification
    rec_df["RR_Ratio"] = rec_df["RewardRisk"]

    # Propagate recalculated RewardRisk back into the main `results` frame so
    # downstream fields (rr alias, rr_score_v2, conviction) use the updated values.
    try:
        rr_map = rec_df.set_index("Ticker")["RewardRisk"].to_dict()
        # Only update tickers present in the map; keep existing values otherwise
        results["RewardRisk"] = results["Ticker"].map(rr_map).fillna(results.get("RewardRisk", np.nan))
        results["RR_Ratio"] = results["RewardRisk"]
        # alias used elsewhere
        results["rr"] = results["RewardRisk"]
        # Recompute normalized rr_score_v2 (0-100) for any updated rr values
        def _norm_rr_local(x):
            try:
                xr = float(x)
                if not np.isfinite(xr):
                    return np.nan
                xr = min(max(xr, 0.0), 5.0)
                return float((xr / 5.0) * 100.0)
            except Exception:
                return np.nan
        results["rr_score_v2"] = results["rr"].apply(_norm_rr_local)
        # Keep canonical RR_Score alias in sync if present
        results["RR_Score"] = results["rr_score_v2"].copy()
        results["RR"] = results["rr"].copy()
    except Exception:
        pass
    except Exception:
        # if anything goes wrong, leave results unchanged
        pass

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
        
        # Sector diversification warning
        if "Sector" in core_df.columns:
            sector_counts = core_df["Sector"].value_counts()
            total_stocks = len(core_df)
            concentrated_sectors = []
            for sector, count in sector_counts.items():
                pct = (count / total_stocks) * 100
                if pct > 30:
                    concentrated_sectors.append(f"{sector} ({count}/{total_stocks}, {pct:.0f}%)")
            
            if concentrated_sectors:
                st.warning(f"⚠️ **Sector Concentration Alert:** {', '.join(concentrated_sectors)}. Consider diversifying across more sectors to reduce correlation risk.")
        
        st.markdown("""
<div style='direction:ltr;text-align:left;font-size:0.75em;margin:4px 0 10px 0'>
<b>Reliability legend:</b> <span style='color:#16a34a;font-weight:600'>High ≥ 0.75</span> · <span style='color:#f59e0b;font-weight:600'>Medium 0.40–0.74</span> · <span style='color:#dc2626;font-weight:600'>Low &lt; 0.40</span>
</div>
""", unsafe_allow_html=True)
        
        for _, r in core_df.iterrows():
            mean = r.get("מחיר ממוצע", np.nan)
            std = r.get("סטיית תקן", np.nan)
            hist_std = r.get("Historical_StdDev", np.nan)  # NEW: Use historical price std dev
            show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
            # Prefer Historical_StdDev if available, fallback to old std
            show_std = f"${hist_std:.2f}" if np.isfinite(hist_std) else (f"${std:.2f}" if np.isfinite(std) else "N/A")
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
            
            # ML confidence badge: 🔥High≥70%, 🟡Med 50-70%, ⚠️Low<50%
            if np.isfinite(ml_prob):
                if ml_prob >= 0.70:
                    ml_badge_color = "#16a34a"  # green
                    ml_badge_text = "🔥 גבוה"
                elif ml_prob >= 0.50:
                    ml_badge_color = "#f59e0b"  # orange
                    ml_badge_text = "🟡 בינוני"
                else:
                    ml_badge_color = "#dc2626"  # red
                    ml_badge_text = "⚠️ נמוך"
                ml_badge_html = f"""<span style='display:inline-block;padding:3px 8px;border-radius:4px;background:{ml_badge_color};color:white;font-weight:bold;font-size:0.85em;margin-left:8px'>ML: {ml_badge_text} ({ml_prob*100:.0f}%)</span>"""
                ml_status_esc = f"{ml_badge_text} ({ml_prob*100:.0f}%)"
            else:
                ml_badge_html = ""
                ml_status_esc = "N/A"

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

            # Detect missing fundamental data
            missing_fundamental_count = 0
            fundamental_fields = ['ROE_f', 'ROIC_f', 'DE_f', 'PE_f', 'GM_f']
            for field in fundamental_fields:
                val = r.get(field, np.nan)
                if not np.isfinite(val):
                    missing_fundamental_count += 1
            
            # Create partial data badge if applicable
            data_quality_badge = ""
            if missing_fundamental_count >= 4:
                data_quality_badge = "<span class='modern-badge badge-missing'>⚠️ Missing Data</span>"
            elif missing_fundamental_count >= 2:
                data_quality_badge = "<span class='modern-badge badge-partial'>📊 Partial Data</span>"

            esc = html_escape.escape
            ticker = esc(str(r["Ticker"]))
            sources_esc = esc(str(sources))
            
            # Next earnings date
            next_earnings = r.get("NextEarnings", "Unknown")
            
            # NEW: Ranking and target prices
            overall_rank = r.get("Overall_Rank", "N/A")
            entry_price = r.get("Entry_Price", np.nan)
            target_price = r.get("Target_Price", np.nan)
            target_date = r.get("Target_Date", "N/A")
            target_source = r.get("Target_Source", "N/A")
            
            entry_price_fmt = f"${entry_price:.2f}" if np.isfinite(entry_price) else "N/A"
            target_price_fmt = f"${target_price:.2f}" if np.isfinite(target_price) else "N/A"
            
            # Add badge for AI-enhanced targets
            target_badge = ""
            if target_source == "AI":
                target_badge = " <span style='background:#10b981;color:white;padding:2px 6px;border-radius:4px;font-size:0.75em;font-weight:bold'>🤖 AI</span>"
            elif target_source == "Technical":
                target_badge = " <span style='background:#6366f1;color:white;padding:2px 6px;border-radius:4px;font-size:0.75em'>📊 Tech</span>"
            elif target_source == "Default":
                target_badge = " <span style='background:#6b7280;color:white;padding:2px 6px;border-radius:4px;font-size:0.75em'>📐 Est</span>"
            
            # Calculate potential gain %
            if np.isfinite(entry_price) and np.isfinite(target_price) and entry_price > 0:
                potential_gain_pct = ((target_price - entry_price) / entry_price) * 100
                gain_fmt = f"+{potential_gain_pct:.1f}%"
                gain_color = "#16a34a"
            else:
                gain_fmt = "N/A"
                gain_color = "#6b7280"
                target_badge = ""  # No badge if no valid target
            
            # Inject CSS for iframe isolation
            price_rel_fmt = format_rel(r.get('Price_Reliability', np.nan))
            fund_rel_fmt = format_rel(r.get('Fundamental_Reliability', np.nan))
            rel_score_fmt = format_rel(r.get('Reliability_Score', np.nan))
            
            # V2 SCORES (NOW ALWAYS ENABLED AS DEFAULT)
            v2_html = ""
            conv_v2 = r.get("conviction_v2_final", np.nan)
            conv_v2_base = r.get("conviction_v2_base", np.nan)
            fund_v2 = r.get("fundamental_score_v2", np.nan)
            tech_v2 = r.get("technical_score_v2", np.nan)
            rr_v2 = r.get("rr_score_v2", np.nan)
            rel_v2 = r.get("reliability_score_v2", np.nan)
            risk_v2 = r.get("risk_meter_v2", np.nan)
            risk_label_v2 = r.get("risk_label_v2", "N/A")
            ml_boost_v2 = r.get("ml_boost", 0.0)
            # Strict gate status and V2 allocations
            gate_status = r.get("risk_gate_status_v2", None)
            gate_reason = r.get("risk_gate_reason_v2", "")
            buy_amount_v2 = float(r.get("buy_amount_v2", 0.0) or 0.0)
            shares_v2 = int(r.get("shares_to_buy_v2", 0) or 0)

            # Build strict-mode badge
            badge_html = ""
            if gate_status == "blocked":
                badge_html = "<span style='background:#dc2626;color:white;padding:4px 8px;border-radius:6px;font-weight:700;margin-left:8px'>❌ Blocked (Strict Risk Gate)</span>"
            elif gate_status == "reduced" or gate_status == "severely_reduced":
                badge_html = "<span style='background:#f59e0b;color:black;padding:4px 8px;border-radius:6px;font-weight:700;margin-left:8px'>⚠️ Reduced (Strict Risk Gate)</span>"
            elif gate_status == "full":
                badge_html = "<span style='background:#16a34a;color:white;padding:4px 8px;border-radius:6px;font-weight:700;margin-left:8px'>✅ Full Allocation Allowed (Strict Mode)</span>"

            # Format V2 scores for inline display
            conv_v2_fmt = f"{conv_v2:.0f}" if np.isfinite(conv_v2) else "N/A"
            fund_v2_fmt = f"{fund_v2:.0f}" if np.isfinite(fund_v2) else "N/A"
            tech_v2_fmt = f"{tech_v2:.0f}" if np.isfinite(tech_v2) else "N/A"
            rr_v2_fmt = f"{rr_v2:.0f}" if np.isfinite(rr_v2) else "N/A"
            rel_v2_fmt = f"{rel_v2:.0f}" if np.isfinite(rel_v2) else "N/A"
            risk_v2_fmt = f"{risk_v2:.0f}" if np.isfinite(risk_v2) else "N/A"
            
            # Conviction meter color
            if np.isfinite(conv_v2):
                if conv_v2 >= 75:
                    conv_color = "#16a34a"  # green
                elif conv_v2 >= 60:
                    conv_color = "#f59e0b"  # orange
                else:
                    conv_color = "#dc2626"  # red
            else:
                conv_color = "#6b7280"
            
            # Risk meter color (inverted: low risk = green)
            if np.isfinite(risk_v2):
                if risk_v2 < 35:
                    risk_color = "#16a34a"  # green
                elif risk_v2 < 65:
                    risk_color = "#f59e0b"  # orange
                else:
                    risk_color = "#dc2626"  # red
            else:
                risk_color = "#6b7280"
            
            card_html = get_card_css() + build_clean_card(r, speculative=False)
            
            # Add provider attribution if available (Core cards)
            attribution = r.get("Fund_Attribution", "")
            if show_debug_attr:
                # Show raw provider map from _sources if available
                raw_sources = r.get("_sources", {})
                if raw_sources:
                    raw_html = html_escape.escape(str(raw_sources))
                    card_html += f"""
    <div class="item" style="grid-column:span 5;font-size:0.7em;color:#334155;background:#f1f5f9;border:1px dashed #cbd5e1;border-radius:6px;padding:4px;margin-top:4px"><b>RAW _sources:</b> {raw_html}</div>"""
            # Remove top-level data sources line (moved to details section inside card)
            
            card_html += """
  </div>
</div>
"""
            # Use reasonable height with scrolling enabled to prevent text cutoff
            # Reduced iframe height to tighten vertical spacing between cards.
            # Previous fixed height (700) created large empty gaps below content.
            # 430 provides enough space for collapsed + expanded details without excess.
            # Render card directly (no inner scroll iframe) so expanding details pushes subsequent cards down.
            st.markdown(card_html, unsafe_allow_html=True)
    
    # Display Speculative recommendations
    if not spec_df.empty:
        st.markdown("### ⚡ Speculative Stocks — High Upside, High Risk")
        st.caption(f"⚠️ {len(spec_df)} stocks with a higher risk profile")
        
        # Sector diversification warning
        if "Sector" in spec_df.columns:
            sector_counts = spec_df["Sector"].value_counts()
            total_stocks = len(spec_df)
            concentrated_sectors = []
            for sector, count in sector_counts.items():
                pct = (count / total_stocks) * 100
                if pct > 30:
                    concentrated_sectors.append(f"{sector} ({count}/{total_stocks}, {pct:.0f}%)")
            
            if concentrated_sectors:
                st.warning(f"⚠️ **Sector Concentration Alert:** {', '.join(concentrated_sectors)}. Consider diversifying across more sectors to reduce correlation risk.")
        
        st.warning("🔔 Warning: These stocks are classified as speculative due to partial data or elevated risk factors. Suitable for experienced investors only.")
        
        for _, r in spec_df.iterrows():
            mean = r.get("מחיר ממוצע", np.nan)
            std = r.get("סטיית תקן", np.nan)
            hist_std = r.get("Historical_StdDev", np.nan)  # NEW: Use historical price std dev
            show_mean = mean if not np.isnan(mean) else r["Price_Yahoo"]
            # Prefer Historical_StdDev if available, fallback to old std
            show_std = f"${hist_std:.2f}" if np.isfinite(hist_std) else (f"${std:.2f}" if np.isfinite(std) else "N/A")
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
            
            # ML confidence badge: 🔥High≥70%, 🟡Med 50-70%, ⚠️Low<50%
            if np.isfinite(ml_prob):
                if ml_prob >= 0.70:
                    ml_badge_color = "#16a34a"  # green
                    ml_badge_text = "🔥 גבוה"
                elif ml_prob >= 0.50:
                    ml_badge_color = "#f59e0b"  # orange
                    ml_badge_text = "🟡 בינוני"
                else:
                    ml_badge_color = "#dc2626"  # red
                    ml_badge_text = "⚠️ נמוך"
                ml_badge_html = f"""<span style='display:inline-block;padding:3px 8px;border-radius:4px;background:{ml_badge_color};color:white;font-weight:bold;font-size:0.85em;margin-left:8px'>ML: {ml_badge_text} ({ml_prob*100:.0f}%)</span>"""
                ml_status_esc = f"{ml_badge_text} ({ml_prob*100:.0f}%)"
            else:
                ml_badge_html = ""
                ml_status_esc = "N/A"
            
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
            
            # Detect missing fundamental data
            missing_fundamental_count = 0
            fundamental_fields = ['ROE_f', 'ROIC_f', 'DE_f', 'PE_f', 'GM_f']
            for field in fundamental_fields:
                val = r.get(field, np.nan)
                if not np.isfinite(val):
                    missing_fundamental_count += 1
            
            # Create partial data badge if applicable
            data_quality_badge = ""
            if missing_fundamental_count >= 4:
                data_quality_badge = "<span class='modern-badge badge-missing'>⚠️ Missing Data</span>"
            elif missing_fundamental_count >= 2:
                data_quality_badge = "<span class='modern-badge badge-partial'>📊 Partial Data</span>"
            
            # Reliability scores formatting (same as Core section)
            def format_rel(val):
                if np.isfinite(val):
                    return f"{val:.1f}%"
                return "N/A"
            
            price_rel_fmt = format_rel(r.get('Price_Reliability', np.nan))
            fund_rel_fmt = format_rel(r.get('Fundamental_Reliability', np.nan))
            rel_score_fmt = format_rel(r.get('Reliability_Score', np.nan))
            
            esc = html_escape.escape
            ticker = esc(str(r["Ticker"]))
            sources_esc = esc(str(sources))
            next_earnings = r.get("NextEarnings", "Unknown")
            warnings_esc = esc(warnings) if warnings else ""
            
            # NEW: Ranking and target prices
            overall_rank = r.get("Overall_Rank", "N/A")
            entry_price = r.get("Entry_Price", np.nan)
            target_price = r.get("Target_Price", np.nan)
            target_date = r.get("Target_Date", "N/A")
            target_source = r.get("Target_Source", "N/A")
            
            entry_price_fmt = f"${entry_price:.2f}" if np.isfinite(entry_price) else "N/A"
            target_price_fmt = f"${target_price:.2f}" if np.isfinite(target_price) else "N/A"
            
            # Add badge for AI-enhanced targets
            target_badge_spec = ""
            if target_source == "AI":
                target_badge_spec = " <span style='background:#10b981;color:white;padding:2px 6px;border-radius:4px;font-size:0.75em;font-weight:bold'>🤖 AI</span>"
            elif target_source == "Technical":
                target_badge_spec = " <span style='background:#6366f1;color:white;padding:2px 6px;border-radius:4px;font-size:0.75em'>📊 Tech</span>"
            
            # Calculate potential gain %
            if np.isfinite(entry_price) and np.isfinite(target_price) and entry_price > 0:
                potential_gain_pct = ((target_price - entry_price) / entry_price) * 100
                gain_fmt = f"+{potential_gain_pct:.1f}%"
                gain_color = "#16a34a"
            else:
                gain_fmt = "N/A"
                gain_color = "#6b7280"
            
            # Inject CSS for iframe isolation
            
            # V2 SCORES (NOW ALWAYS ENABLED AS DEFAULT)
            conv_v2 = r.get("conviction_v2_final", np.nan)
            conv_v2_base = r.get("conviction_v2_base", np.nan)
            fund_v2 = r.get("fundamental_score_v2", np.nan)
            tech_v2 = r.get("technical_score_v2", np.nan)
            rr_v2 = r.get("rr_score_v2", np.nan)
            rel_v2 = r.get("reliability_score_v2", np.nan)
            risk_v2 = r.get("risk_meter_v2", np.nan)
            risk_label_v2 = r.get("risk_label_v2", "N/A")
            ml_boost_v2 = r.get("ml_boost", 0.0)
            # Strict V2 gate fields for speculative cards
            gate_status = r.get("risk_gate_status_v2", None)
            gate_reason = r.get("risk_gate_reason_v2", "")
            buy_amount_v2 = float(r.get("buy_amount_v2", 0.0) or 0.0)
            shares_v2 = int(r.get("shares_to_buy_v2", 0) or 0)
            
            # Format V2 scores for inline display
            conv_v2_fmt = f"{conv_v2:.0f}" if np.isfinite(conv_v2) else "N/A"
            fund_v2_fmt = f"{fund_v2:.0f}" if np.isfinite(fund_v2) else "N/A"
            tech_v2_fmt = f"{tech_v2:.0f}" if np.isfinite(tech_v2) else "N/A"
            rr_v2_fmt = f"{rr_v2:.0f}" if np.isfinite(rr_v2) else "N/A"
            rel_v2_fmt = f"{rel_v2:.0f}" if np.isfinite(rel_v2) else "N/A"
            risk_v2_fmt = f"{risk_v2:.0f}" if np.isfinite(risk_v2) else "N/A"
            
            if np.isfinite(conv_v2):
                if conv_v2 >= 75:
                    conv_color = "#16a34a"
                elif conv_v2 >= 60:
                    conv_color = "#f59e0b"
                else:
                    conv_color = "#dc2626"
            else:
                conv_color = "#6b7280"
            
            if np.isfinite(risk_v2):
                if risk_v2 < 35:
                    risk_color = "#16a34a"
                elif risk_v2 < 65:
                    risk_color = "#f59e0b"
                else:
                    risk_color = "#dc2626"
            else:
                risk_color = "#6b7280"
            
            # Build strict-mode badge for speculative
            badge_html_spec = ""
            if gate_status == "blocked":
                badge_html_spec = "<span style='background:#dc2626;color:white;padding:4px 8px;border-radius:6px;font-weight:700;margin-left:8px'>❌ Blocked (Strict Risk Gate)</span>"
            elif gate_status == "reduced" or gate_status == "severely_reduced":
                badge_html_spec = "<span style='background:#f59e0b;color:black;padding:4px 8px;border-radius:6px;font-weight:700;margin-left:8px'>⚠️ Reduced (Strict Risk Gate)</span>"
            elif gate_status == "full":
                badge_html_spec = "<span style='background:#16a34a;color:white;padding:4px 8px;border-radius:6px;font-weight:700;margin-left:8px'>✅ Full Allocation Allowed (Strict Mode)</span>"
            
            # Minimal speculative card
            overall_score_val = r.get("overall_score", conv_v2)
            rr_ratio_val = r.get("rr", np.nan)
            rr_band = r.get('rr_band', '')
            card_html = get_card_css() + build_clean_card(r, speculative=True)
            
            # Add provider attribution if available (Speculative cards)
            attribution_spec = r.get("Fund_Attribution", "")
            # Remove top-level data sources line for speculative cards as well
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
            # Match reduced height for speculative cards to remove large gaps.
            st.markdown(card_html, unsafe_allow_html=True)

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
    ,
    # V2 strict columns (export-friendly labels)
    "conviction_v2_base": "Conviction v2 Base",
    "conviction_v2_final": "Conviction v2 Final",
    "reliability_v2": "Reliability Score v2",
    "reliability_score_v2": "Reliability Score v2",
    "risk_gate_status_v2": "Risk Gate Status v2",
    "risk_gate_reason_v2": "Risk Gate Reason v2",
    "rr_ratio_v2": "Reward/Risk v2",
    "reward_risk_v2": "Reward/Risk v2",
    "buy_amount_v2": "Buy Amount v2",
    "shares_to_buy_v2": "Shares to Buy v2",
    "fund_sources_used_v2": "Fund Sources Used",
    "price_sources_used_v2": "Price Sources Used",
    "fund_disagreement_score_v2": "Fund Disagreement Score",
    "price_variance_score_v2": "Price Variance Score",
    # Newly exposed transparency columns
    "overall_score": "Overall Score",
    "fund_score": "Fund Score",
    "tech_score": "Tech Score",
    "rr_score": "RR Score",
    "reliability_score": "Reliability Score (Unified)",
    "ml_delta": "ML Delta",
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
    "reliability_v2",  # raw V2 reliability (0-100)
    "Fund Reliability",
    "Price Reliability",
    "Fundamental_Reliability_v2",  # New multi-source fund reliability
    "Price_Reliability_v2",  # New multi-source price reliability
    "fund_sources_used_v2",  # Number of fundamental sources
    "price_sources_used_v2",  # Number of price sources
    "fund_disagreement_score_v2",  # Disagreement between sources
    "price_variance_score_v2",  # Price variance ratio
    "fund_coverage_pct_v2",  # Field coverage percentage
    "Price_Mean_v2",  # Multi-source mean price
    "Price_STD_v2",  # Multi-source price std deviation
    "Sources Count",
    "risk_gate_status_v2",  # blocked / severely_reduced / reduced / full
    "risk_gate_penalty_v2",
    "reliability_score_v2",
    "Score",
    "overall_score",
    "overall_score_raw",  # True model score (internal logic)
    "overall_score_pretty",  # Display score (60-90 range)
    "reliability_pct",  # Reliability percentage (0-100)
    "reliability_band",  # High/Medium/Low
    "risk_band",  # Low/Medium/High/Very High
    "ml_conf_band",  # ML confidence band (Low/Medium/High)
    "fund_score",
    "tech_score",
    "rr_score",
    "reliability_score",
    "ml_delta",
    "conviction_v2_base",
    "conviction_v2_final",
    "ml_boost_v2",
    "reward_risk_v2",
    # Human-friendly V2 export labels (also include raw keys above for robustness)
    "Conviction v2 Base",
    "Conviction v2 Final",
    "Reliability Score v2",
    "Risk Gate Status v2",
    "Risk Gate Reason v2",
    "Reward/Risk v2",
    "Buy Amount v2",
    "Shares to Buy v2",
    "Fund Sources Used",
    "Price Sources Used",
    "Fund Disagreement Score",
    "Price Variance Score",
    "Quality Score",
    "Average Price",
    "Unit Price (calc)",
    "Buy Amount ($)",
    "buy_amount_v2",  # v2 raw dollar amount pre-hebrew mapping
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
