# -*- coding: utf-8 -*-
"""
Asaf Stock Scout — 2026 (Auto Mode, Zero-Input) — FMP-free
-----------------------------------------------------------
• Technical score: MA, Momentum(1/3/6m), RSI band, Near-High bell, Overextension vs MA_L,
  Pullback window, ATR/Price, Reward/Risk, MACD/ADX.
• Fundamentals (Alpha OVERVIEW -> Finnhub fallback): Growth (Rev/EPS YoY), Quality (ROE/Margin),
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
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import html as html_escape

# --- FORCE API KEYS (USER CONFIGURATION / SAFE INJECTION) ---
# Inject API keys from environment, .env, or Streamlit secrets BEFORE core imports.
# This ensures all core modules initialize with real provider credentials.
try:
    # Load .env without overriding existing env
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
except Exception:
    pass

def _clean_key(val: Optional[str]) -> Optional[str]:
    try:
        if val is None:
            return None
        return str(val).strip().strip('"').strip("'")
    except Exception:
        return val

def _inject_api_keys_early() -> None:
    keys = [
        "OPENAI_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "FINNHUB_API_KEY",
        "POLYGON_API_KEY",
        "TIINGO_API_KEY",
        "FMP_API_KEY",
        "MARKETSTACK_API_KEY",
        "NASDAQ_API_KEY",
        "NASDAQ_DL_API_KEY",
        "EODHD_API_KEY",
        "EODHD_TOKEN",
        "SIMFIN_API_KEY",
    ]
    for k in keys:
        val = _clean_key(os.getenv(k))
        # Prefer Streamlit secrets if available
        try:
            if not val and hasattr(st, "secrets") and k in st.secrets:
                val = _clean_key(st.secrets[k])
        except Exception:
            pass
        if val:
            os.environ[k] = str(val)
    # Universe limit: default to 2000 unless explicitly set
    if not os.getenv("UNIVERSE_LIMIT"):
        os.environ["UNIVERSE_LIMIT"] = "2000"

_inject_api_keys_early()
# ------------------------------------------------------------

# ============================================================================
# NEW UNIFIED API - Use wrapper modules
# ============================================================================
from core.data import (
    fetch_price_multi_source,
    aggregate_price,
    aggregate_fundamentals,
    fetch_fundamentals_batch,
    build_technical_indicators,
)
from core.scoring import (
    compute_tech_score_20d_v2,
    predict_20d_prob_from_row,
    apply_live_v3_adjustments,
    compute_final_scores_20d,
    apply_20d_sorting,
    score_ticker_v2_enhanced,
    calculate_reliability_v2,
    compute_fundamental_score_with_breakdown,
    evaluate_rr_unified,
)
from core.filters import (
    apply_technical_filters,
    compute_advanced_score,
    should_reject_ticker,
    fetch_benchmark_data,
)
from core.allocation import allocate_budget, _normalize_weights
from core.classifier import apply_classification, filter_core_recommendations

# ============================================================================
# LEGACY IMPORTS - For backward compatibility (deprecated, will be removed)
# ============================================================================
from core.unified_logic import (
    compute_technical_score,
    compute_final_score,
    compute_overall_score_20d,
)
from indicators import rsi, atr, macd_line, adx, _sigmoid

# ============================================================================
# UI & CONFIG
# ============================================================================
from card_styles import get_card_css
from ui_redesign import (
    render_simplified_sidebar,
    render_native_recommendation_row,
)
from hebrew_ui import (
    setup_hebrew_rtl,
    render_top_control_bar,
    render_hebrew_sidebar_expander,
    render_view_controls,
    render_recommendation_row_hebrew,
    render_core_section_hebrew,
    render_speculative_section_hebrew,
    render_kpi_cards_hebrew,
    force_ml_and_sorting,
)
from core.config import get_config
from core.market_regime import detect_market_regime, adjust_target_for_regime
from core.ui_helpers import (
    StatusManager,
    SourcesOverview,
    get_pipeline_stages,
    show_config_summary,
    create_debug_expander,
)
from core.pipeline_runner import run_scan_pipeline, fetch_top_us_tickers_by_market_cap, LAST_UNIVERSE_PROVIDER

# Helper: build clean minimal card

# Small safety helpers to avoid None-related TypeErrors when building HTML
def _safe_str(v, default: str = "N/A") -> str:
    """Return a safe string for value `v`. If v is None return `default`.
    If v is already a string return it unchanged."""
    if v is None:
        return default
    try:
        return str(v)
    except Exception:
        return default


def _num(v) -> float:
    """Coerce v to float if possible, otherwise return np.nan. Safe for None."""
    try:
        if v is None:
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _is_finite(v) -> bool:
    """Safe wrapper around np.isfinite that returns False for non-numeric/None."""
    try:
        return np.isfinite(v)
    except Exception:
        return False

# --- Robust .env loading (earlier + multi-path) ---------------------------------
def _force_load_env() -> None:
    """Attempt to load .env from common locations before any key access.
    Uses override=True so local development updates take effect immediately.
    Safe to call multiple times (idempotent)."""
    try:
        from dotenv import load_dotenv

        candidates = [Path(".env"), Path(__file__).parent / ".env", Path.cwd() / ".env"]
        for p in candidates:
            if p.exists():
                load_dotenv(p, override=True)
    except Exception:
        pass


_force_load_env()


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
    ticker = esc(_safe_str(row.get("Ticker", "N/A")))
    overall_rank = row.get("Overall_Rank", "N/A")
    # Use pretty score for display (60-90 range), raw score for internal logic
    # Show both pretty score and 20d score
    overall_score = row.get(
        "overall_score_pretty",
        row.get("overall_score", row.get("conviction_v2_final", np.nan)),
    )
    score_20d = row.get("overall_score_20d", None)
    target_price = _num(row.get("Target_Price", np.nan))
    # Use broader fallback chain for precomputed scans (no UI-only columns)
    entry_price = _num(
        row.get(
            "Entry_Price",
            row.get(
                "Price_Yahoo",
                row.get(
                    "Unit_Price",
                    row.get("Close", np.nan)
                )
            )
        )
    )
    target_date = _safe_str(row.get("Target_Date", "N/A"))
    target_source = _safe_str(row.get("Target_Source", "N/A"))
    # Prefer v2/alias RR_Ratio if present
    rr_ratio = _num(row.get("RR_Ratio", row.get("rr", np.nan)))
    rr_score = _num(row.get("rr_score_v2", np.nan))
    rr_band = _safe_str(row.get("rr_band", ""), "")

    # Get display bands
    risk_meter = _num(row.get("risk_meter_v2", np.nan))
    risk_band_label = _safe_str(row.get("risk_band", row.get("Risk_Level", "N/A")))
    reliability_pct = _num(row.get("reliability_pct", row.get("Reliability_v2", np.nan)))
    reliability_band_label = _safe_str(row.get("reliability_band", "N/A"))
    # Fallback: derive reliability band from percentage if missing
    if (not reliability_band_label) or reliability_band_label == "N/A":
        if np.isfinite(reliability_pct):
            if reliability_pct >= 75:
                reliability_band_label = "High"
            elif reliability_pct >= 40:
                reliability_band_label = "Medium"
            else:
                reliability_band_label = "Low"
    # Prefer canonical ML probability from 20d pipeline
    ml_prob = _num(row.get("ML_20d_Prob", row.get("ML_Probability", np.nan)))

    # Derive confidence band with explicit Low/Medium/High thresholds; fallback message if missing
    def ml_conf_band(p: float) -> str:
        if not np.isfinite(p):
            return "N/A"
        if p < 0.60:
            return "Low"
        if p < 0.75:
            return "Medium"
        return "High"

    ml_conf_band_label = _safe_str(row.get("ml_conf_band", ml_conf_band(ml_prob)))

    quality_level = _safe_str(row.get("Quality_Level", row.get("quality_level", "N/A")))
    quality_score = _num(row.get("Quality_Score_Numeric", np.nan))
    conv_base = _num(row.get("conviction_v2_base", np.nan))

    # Component scores for <details>
    fund_score = _num(row.get("Fundamental_S", np.nan))
    tech_score = _num(row.get("Technical_S", np.nan))
    # Compressed data sources line (prices + fundamentals providers if available)
    price_sources = _safe_str(row.get("Price_Sources_Line", row.get("price_sources_line", "")), "")
    fund_sources = _safe_str(row.get("Fund_Sources_Line", row.get("fund_sources_line", "")), "")
    sources_line = ""
    if price_sources or fund_sources:
        sources_line = f"Data sources: Prices - {price_sources or 'N/A'}; Fundamentals - {fund_sources or 'N/A'}"

    def fmt_money(v):
        return f"${v:.2f}" if _is_finite(v) else "N/A"

    def fmt_pct(v):
        """Format percentage values with 1 decimal place."""
        return f"{v:.1f}%" if _is_finite(v) else "N/A"

    def fmt_score(v):
        # Handle lists by taking first element or count
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return "0"
            v = len(v) if all(isinstance(x, str) for x in v) else v[0]
        # Handle non-numeric types
        try:
            fv = _num(v)
            return f"{float(fv):.0f}" if _is_finite(fv) else "N/A"
        except (TypeError, ValueError):
            return _safe_str(v, "N/A")

    entry_fmt = fmt_money(entry_price)
    target_fmt = fmt_money(target_price)
    if _is_finite(entry_price) and _is_finite(target_price) and entry_price > 0:
        potential_gain_pct = ((target_price - entry_price) / entry_price) * 100
        potential_fmt = f"+{potential_gain_pct:.1f}%"
    else:
        potential_fmt = "N/A"

    target_badge = ""
    if target_source == "AI":
        target_badge = '<span class="badge ai">AI</span>'
    elif target_source == "Technical":
        target_badge = '<span class="badge tech">Tech</span>'

    # Ratios -> 2 decimals, Scores -> int or (if fractional) 1 decimal, Percentages handled by fmt_pct
    rr_ratio_fmt = f"{rr_ratio:.2f}" if _is_finite(rr_ratio) else "N/A"
    overall_score_fmt = fmt_score(overall_score)  # already integer style
    score_20d_fmt = fmt_score(score_20d) if score_20d is not None else "N/A"
    if _is_finite(quality_score):
        quality_score_fmt = f"{quality_score:.1f}" if abs(quality_score - round(quality_score)) > 0.05 else f"{int(round(quality_score))}"
    else:
        quality_score_fmt = "N/A"

    # Get Fund and Price reliability separately for detailed display
    fund_reliability = _num(row.get("Fundamental_Reliability_v2", row.get("Fundamental_Reliability", np.nan)))
    price_reliability = _num(row.get("Price_Reliability_v2", row.get("Price_Reliability", np.nan)))
    # Reliability percentages -> 1 decimal place
    fund_rel_fmt = f"{fund_reliability:.1f}" if _is_finite(fund_reliability) else "N/A"
    price_rel_fmt = f"{price_reliability:.1f}" if _is_finite(price_reliability) else "N/A"

    # Format display values with bands
    risk_fmt = f"{fmt_score(risk_meter)} ({_safe_str(risk_band_label,'N/A')})"
    reliability_fmt = f"{_safe_str(reliability_band_label,'N/A')} (F:{fund_rel_fmt}% / P:{price_rel_fmt}%)"
    ml_fmt = f"{_safe_str(ml_conf_band_label,'N/A')} (p={ml_prob*100:.1f}%)" if _is_finite(ml_prob) else "N/A (no model data)"

    type_badge = "SPEC" if speculative else "CORE"
    # Fallback badge if this stock is shown only due to emergency/fallback logic
    if bool(row.get("Fallback_Display", False)):
        type_badge += " (FB)"

    # Warning indicator
    warning = ""
    if (_is_finite(rr_ratio) and rr_ratio < 1.5) or (_is_finite(risk_meter) and risk_meter > 70):
        warning = " ⚠️"
    # Badges: Coiled + Growth Boost
    rr_5_20 = _num(row.get("RangeRatio_5_20", row.get("range_ratio_5_20", np.nan)))
    tightness = _num(row.get("Tightness_Ratio", row.get("tightness_ratio", np.nan)))
    eps_g = _num(row.get("eps_g_yoy", row.get("EPS YoY", np.nan)))
    is_coiled = (np.isfinite(rr_5_20) and rr_5_20 < 0.7) or (np.isfinite(tightness) and tightness < 0.6)
    is_growth = np.isfinite(eps_g) and eps_g > 0.25

    coil_badge = "<span class='badge coil'>🎯 COILED</span>" if is_coiled else ""
    growth_badge = "<span class='badge growth'>🚀 GROWTH BOOST</span>" if is_growth else ""

    # Allocation note when buy amount is zero or missing
    buy_amt = _num(row.get("buy_amount_v2", row.get("סכום קנייה ($)", np.nan)))
    allocation_note = ""
    if not np.isfinite(buy_amt) or buy_amt <= 0:
        allocation_note = "<div class='allocation-note'>Insufficient confidence for allocation</div>"

    # Build explanation bullets (top-level quick rationale) only when warning
    bullets = []
    bullet_html = ""
    if warning:
        if _is_finite(fund_score):
            if fund_score >= 60:
                bullets.append(f"Fundamentals solid ({fmt_score(fund_score)})")
            elif fund_score < 50:
                bullets.append(f"Weak fundamentals ({fmt_score(fund_score)})")
        if _is_finite(tech_score) and tech_score >= 65:
            bullets.append("Technical momentum")
        if _is_finite(rr_ratio) and rr_ratio >= 1.5:
            bullets.append(f"RR {rr_ratio_fmt}x")
        if reliability_band_label:
            bullets.append(f"Reliability {reliability_band_label}")
        if ml_conf_band_label in ("High", "Medium"):
            bullets.append(f"ML {ml_conf_band_label}")
        if overall_rank not in (None, "N/A"):
            bullets.append(f"Rank #{overall_rank}")
        if potential_fmt not in ("N/A", None):
            bullets.append(f"Upside {potential_fmt}")
        if bullets:
            items = "".join(f"<li>{html_escape.escape(b)}</li>" for b in bullets[:6])
            bullet_html = f"<ul class='signal-bullets'>{items}</ul>"

    # Card HTML rendering block (always render)
    card_html = f"""
<div class='clean-card { 'speculative' if speculative else 'core' }'>
    <div class='card-header'>
        <div class='ticker-line'><span class='ticker-badge ltr'>{ticker}</span><span class='type-badge'>{type_badge}</span><span class='rank-badge ltr'>#{overall_rank}</span> {coil_badge} {growth_badge}</div>
        <h2 class='overall-score'>{overall_score_fmt}<span class='score-label ltr'>/100</span>{warning}</h2>
    </div>
    <div class='entry-target-line'>Entry <b class='ltr'>{entry_fmt}</b> -> Target <b class='ltr'>{target_fmt}</b> {target_badge} <span class='potential ltr'>{potential_fmt}</span></div>
    {bullet_html}
    {allocation_note}
    <div class='top-grid'>
        <div class='field'><span class='label'>R/R</span><span class='value tabular ltr'>{rr_ratio_fmt} <span class='band ltr'>{rr_band}</span></span></div>
        <div class='field'><span class='label'>Risk</span><span class='value tabular ltr'>{risk_fmt}</span></div>
        <div class='field'><span class='label'>Reliability</span><span class='value tabular ltr'>{reliability_fmt}</span></div>
        <div class='field'><span class='label'>ML</span><span class='value tabular ltr'>{ml_fmt}</span></div>
        <div class='field'><span class='label'>Quality</span><span class='value tabular ltr'>{quality_level} ({quality_score_fmt})</span></div>
        <div class='field'><span class='label'>Fundamental Score</span><span class='value tabular ltr'>{fmt_score(fund_score)}</span></div>
    </div>
    <details class='more-info'>
        <summary>More Details</summary>
        <div class='detail-grid'>
            <div class='field'><span class='label'>Target Date</span><span class='value ltr'>{target_date}</span></div>
            <div class='field'><span class='label'>ML Probability</span><span class='value ltr'>{fmt_pct(ml_prob * 100) if np.isfinite(ml_prob) else 'N/A'}</span></div>
            <div class='field'><span class='label'>Base Conviction</span><span class='value ltr'>{fmt_score(conv_base)}</span></div>
            <div class='field'><span class='label'>Fund Sources</span><span class='value ltr'>{fmt_score(row.get('fund_sources_used_v2', row.get('Fundamental_Sources_Count', 0)))}</span></div>
            <div class='field'><span class='label'>Price Sources</span><span class='value ltr'>{fmt_score(row.get('price_sources_used_v2', row.get('Price_Sources_Count', 0)))}</span></div>
            <div class='field'><span class='label'>Price Std Dev</span><span class='value ltr'>{fmt_money(row.get('Price_STD_v2', np.nan))}</span></div>
        </div>
    </details>
</div>
"""
    return card_html
# Deterministic ranking helper (score desc, ticker asc) prior to Core/Spec split
def apply_deterministic_ranking(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if 'Ticker' not in df.columns:
        return df
    score_col = None
    for c in ['overall_score_20d','Score','overall_score','conviction_v2_final','overall_score_pretty']:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        # Ensure robust fallback if none of the expected score columns exist
        df['Rank'] = range(1, len(df)+1)
        if 'Overall_Rank' not in df.columns:
            df['Overall_Rank'] = df['Rank']
        return df
    df = df.sort_values(by=[score_col,'Ticker'], ascending=[False, True]).copy()
    df['Rank'] = range(1, len(df)+1)
    if 'Overall_Rank' not in df.columns:
        df['Overall_Rank'] = df['Rank']
    return df


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
    **{
        k: getattr(_config_obj, k)
        for k in dir(_config_obj)
        if not k.startswith("_") and not callable(getattr(_config_obj, k))
    },
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
    # Canonical lowercase key used by pipeline_runner
    "fundamental_enabled": _config_obj.fundamental_enabled,
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
    "ENABLE_SIMFIN": False,  # SimFin API v2 deprecated, v3 requires paid subscription
    "ENABLE_MARKETSTACK": False,  # Monthly usage limit reached (free tier exhausted)
    "ENABLE_NASDAQ_DL": False,  # API access blocked (403 Forbidden)
    "ENABLE_EODHD": False,  # Requires paid subscription (402 Payment Required)
    # --- Performance / Fast Mode Flags ---
    "PERF_FAST_MODE": False,  # Set True to reduce external waits for interactive exploration
    "PERF_MULTI_SOURCE_TOP_N": 8,  # In fast mode: only compute multi-source reliability for top N by score
    "PERF_ALPHA_ENABLED": True,  # In fast mode we force this False to skip Alpha Vantage entirely
    "PERF_FUND_TIMEOUT": 15,  # Normal per-provider future timeout
    "PERF_FUND_TIMEOUT_FAST": 6,  # Fast mode per-provider future timeout
    # --- Debug / Developer Flags ---
    "DEBUG_MODE": os.getenv("STOCK_SCOUT_DEBUG", "false").lower() in ("true", "1", "yes"),
    # --- Remote autoscan artifacts (GitHub raw) ---
    "USE_REMOTE_AUTOSCAN": True,
    "REMOTE_AUTOSCAN_REPO": os.getenv("REMOTE_AUTOSCAN_REPO", "asafamos/stock-scout"),
    "REMOTE_AUTOSCAN_BRANCH": os.getenv("REMOTE_AUTOSCAN_BRANCH", "main"),
}

# -----------------------------------------------------------------
# Canonical fundamentals schema (numeric + string fields)
# These fields are used across all provider normalization helpers and
# the fusion layer. New providers should map into these names only.
# -----------------------------------------------------------------
FUND_SCHEMA_FIELDS = [
    "oper_margin",  # Operating margin
    "roe",          # Return on Equity
    "roic",         # Return on Invested Capital
    "gm",           # Gross Margin
    "ps",           # Price to Sales
    "pe",           # Price to Earnings
    "de",           # Debt to Equity
    "rev_g_yoy",    # Revenue growth YoY
    "eps_g_yoy",    # EPS growth YoY
]
FUND_STRING_FIELDS = ["sector", "industry"]

# Utility: create empty fundamentals dict for a ticker (all NaN / Unknown)
def _empty_fund_row() -> dict:
    out = {f: np.nan for f in FUND_SCHEMA_FIELDS}
    out["sector"] = "Unknown"
    out["industry"] = "Unknown"
    out["_sources"] = {}
    out["_sources_used"] = []
    out["Fund_Coverage_Pct"] = 0.0
    out["fundamentals_available"] = False
    return out

# Provider usage tracker promoted to module level (was nested in legacy fundamentals function)
def mark_provider_usage(provider: str, category: str):
    """Record usage of a provider for a given category (price/fundamentals/ml). Safe no-throw.
    
    DEPRECATED: Use SourcesOverview.mark_usage() instead. Kept for backward compatibility.
    """
    try:
        usage = st.session_state.setdefault("provider_usage", {})
        cats = usage.setdefault(provider, set())
        cats.add(category)
        usage[provider] = cats
    except Exception:
        pass


def render_data_sources_overview(provider_status: dict, provider_usage: dict, results: pd.DataFrame) -> None:
    """
    Render a dynamic, compact data sources table showing which providers were actually used in this run.
    Uses emoji indicators and Hebrew labels with RTL layout and avoids any raw HTML inside the dataframe.
    """
    import pandas as pd
    import streamlit as st

    # Canonical provider names; map to internal usage labels if needed
    synonyms = {
        "Alpha Vantage": "Alpha",
        "Nasdaq": "NasdaqDL",
        "Yahoo": "Yahoo",
    }

    table_rows = []
    for provider_name, status_info in provider_status.items():
        ok = bool(status_info.get("ok", False))
        status_icon = "🟢" if ok else "🔴"
        status_text = "פעיל" if ok else "תקלה / חסום"

        usage_key = provider_name if provider_name in provider_usage else synonyms.get(provider_name, provider_name)
        usage_info = provider_usage.get(usage_key, {})

        used_price = bool(usage_info.get("used_price"))
        used_fund = bool(usage_info.get("used_fundamentals"))
        used_ml = bool(usage_info.get("used_ml"))
        implemented = bool(usage_info.get("implemented", True))

        if not implemented:
            status_icon = "⚪"
            status_text = "לא רלוונטי בריצה זו"

        if used_price or used_fund or used_ml:
            used_icon = "🟢"
            used_text = "בשימוש"
        else:
            used_icon = "⚪"
            used_text = "לא בשימוש"

        details_parts = []
        if used_price:
            details_parts.append("מחיר")
        if used_fund:
            details_parts.append("פונדמנטלי")
        if used_ml:
            details_parts.append("ML")

        usage_detail = " | ".join(details_parts) if details_parts else "—"

        table_rows.append(
            {
                "ספק": provider_name,
                "סטטוס": f"{status_icon} {status_text}",
                "שימוש": f"{used_icon} {used_text}",
                "פרטים": usage_detail,
            }
        )

    if not table_rows:
        return

    df_sources = pd.DataFrame(table_rows)
    df_sources["ספק"] = df_sources["ספק"].astype(str)

    styled = (
        df_sources.style
        .set_properties(
            subset=["ספק"],
            **{
                "direction": "ltr",
                "text-align": "left",
                "font-size": "14px",
                "white-space": "nowrap",
            }
        )
        .set_properties(
            subset=["סטטוס", "שימוש", "פרטים"],
            **{
                "text-align": "center",
                "font-size": "14px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "center"), ("font-size", "15px")],
                }
            ]
        )
    )

    st.markdown("### 🔌 מקורות נתונים")
    st.dataframe(styled, width='stretch', hide_index=True)

    used_count = 0
    for provider_name in provider_status.keys():
        usage_key = provider_name if provider_name in provider_usage else synonyms.get(provider_name, provider_name)
        info = provider_usage.get(usage_key, {})
        if info.get("used_price") or info.get("used_fundamentals") or info.get("used_ml"):
            used_count += 1
    st.caption(f"סה\"כ ספקים פעילים: {used_count} / {len(provider_status)}")


# Load environment variables
warnings.simplefilter("ignore", FutureWarning)

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Original load (kept for backwards compatibility); `_force_load_env` above already tried explicit paths.
load_dotenv(find_dotenv(usecwd=True))

# Display ML 20d readiness using unified inference loader
try:
    from core.ml_20d_inference import ML_20D_AVAILABLE, FEATURE_COLS_20D, PREFERRED_SCORING_MODE_20D
    if ML_20D_AVAILABLE:
        try:
            st.info(f"✓ ML 20d ready (features: {len(FEATURE_COLS_20D)}; mode: {PREFERRED_SCORING_MODE_20D})")
        except Exception:
            pass
    else:
        st.info("ML 20d model not found; ML scoring will be neutral.")
except Exception as _e:
    st.warning(f"ML 20d loader issue: {_e}")


def assign_confidence_tier(prob: float) -> str:
    """
    Assign ML confidence tier based on probability.

    Recalibrated thresholds for realistic diversity:
    - High: >=0.75 (strong prediction)
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
    # Try Streamlit secrets first (Cloud deployment)
    try:
        if hasattr(st, "secrets"):
            # Try direct access (top-level key)
            try:
                val = st.secrets[key]
                if val:  # Ensure it's not empty
                    return str(val)
            except (KeyError, FileNotFoundError):
                pass

            # Try nested sections (api_keys, keys, secrets, tokens)
            for section in ("api_keys", "keys", "secrets", "tokens"):
                try:
                    val = st.secrets[section][key]
                    if val:
                        return str(val)
                except (KeyError, FileNotFoundError, AttributeError):
                    continue
    except Exception as e:
        # Log for debugging in cloud
        if hasattr(st, "warning"):
            st.warning(f"⚠️ Secret access error for {key}: {e}")

    # Fallback to environment variable (local .env)
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
                _log.debug(
                    f"HTTP {resp.status_code} -> retry attempt {attempt}/{tries}"
                )
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
        min_gap_seconds: Minimum seconds between calls (default '12s' = 5 calls per min)
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
        logger.warning(
            f"Alpha Vantage daily limit reached ({call_count} calls), skipping"
        )
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
    """Build a wide US universe (NASDAQ + NYSE + AMEX) with sane filters.

    Priority: cached file data/universe_full.csv (Symbol column). If missing, fetch
    nasdaqlisted + otherlisted symbol directories, drop ETFs/test/warrants, dedupe,
    and fall back to S&P500 if everything fails.
    """

    def _clean_symbols(df: pd.DataFrame, sym_col: str = "Symbol") -> List[str]:
        if sym_col not in df.columns:
            return []
        syms = (
            df[sym_col]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        # Basic sanity: drop empty and very long symbols
        syms = [s for s in syms if s and len(s) <= 6]
        return syms

    # 1) Cached file if exists
    cache_path = Path("data/universe_full.csv")
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            cached = _clean_symbols(df_cache, sym_col="Symbol") or _clean_symbols(df_cache, sym_col="symbol")
            uniq = list(pd.unique(cached))
            logger.info(f"✓ Loaded cached universe_full.csv with {len(uniq)} symbols")
            return uniq[:limit]
        except Exception as e:
            logger.warning(f"Cached universe_full.csv load failed: {e}")

    # 2) NASDAQ + OTHERLISTED (full US common stock list)
    try:
        nasdaq_url = "https://ftp.nasdaqtrader.com/dynamic/SYMBOLDirectory/nasdaqlisted.txt"
        other_url = "https://ftp.nasdaqtrader.com/dynamic/SYMBOLDirectory/otherlisted.txt"
        nasdaq_df = pd.read_csv(nasdaq_url, sep="|")
        other_df = pd.read_csv(other_url, sep="|")

        # Filter out ETFs, test issues, rights/warrants
        if "ETF" in nasdaq_df.columns:
            nasdaq_df = nasdaq_df[nasdaq_df["ETF"] != "Y"]
        if "Test Issue" in nasdaq_df.columns:
            nasdaq_df = nasdaq_df[nasdaq_df["Test Issue"] != "Y"]
        if "Exchange" in other_df.columns:
            other_df = other_df[other_df["Exchange"].isin(["A", "N", "Q"])]
        for col in ("Test Issue", "ETF"):
            if col in other_df.columns:
                other_df = other_df[other_df[col] != "Y"]

        syms = _clean_symbols(nasdaq_df, sym_col="Symbol") + _clean_symbols(other_df, sym_col="ACT Symbol")
        uniq = list(pd.unique(syms))
        logger.info(f"✓ Loaded {len(uniq)} symbols from NASDAQ/NYSE/AMEX lists")
        return uniq[:limit]
    except Exception as e:
        logger.warning(f"NASDAQ/OTHERLISTED fetch failed ({e}), falling back to S&P500")

    # 3) Fallback to S&P500 (previous behavior)
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            storage_options={"User-Agent": "Mozilla/5.0"},
        )
        df_sp = tables[1]
        df_sp = df_sp.drop_duplicates(subset="Security", keep="first")
        tickers = df_sp["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        logger.info(f"✓ Loaded {len(tickers)} S&P500 symbols (fallback)")
        return tickers[:limit]
    except Exception as e:
        logger.warning(f"Fallback S&P500 fetch failed ({e}), using tiny fallback list")
        fallback = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "WMT", "UNH", "AVGO"]
        return fallback[:limit]


def fetch_history_bulk(
    tickers: List[str], period_days: int, ma_long: int
) -> Dict[str, pd.DataFrame]:
    """Fetch bulk historical data with sufficient lookback for moving averages."""
    import yfinance as yf

    end = datetime.utcnow()
    start = end - timedelta(days=period_days + 50)

    data_map: Dict[str, pd.DataFrame] = {}
    min_rows = ma_long + 40

    for tkr in tickers:
        try:
            df = yf.download(tkr, start=start, end=end, progress=False)
            if df is not None and len(df) >= min_rows:
                data_map[tkr] = df
        except Exception as exc:  # best-effort fetch per ticker
            logger.warning(f"Historical fetch failed for {tkr}: {exc}")

    return data_map
# ==================== Earnings ====================
@st.cache_data(ttl=60 * 60)
def get_next_earnings_date(ticker: str) -> Optional[datetime]:
    """Get next earnings date from Finnhub -> yfinance fallback."""
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


# ==================== Fundamentals (Alpha -> Finnhub) ====================
def _to_01(x: float, low: float, high: float) -> float:
    if not isinstance(x, (int, float)) or not np.isfinite(x):
        return np.nan
    return np.clip((x - low) / (high - low), 0, 1)


def calculate_rr(
    entry_price: float,
    target_price: float,
    atr_value: float,
    history_df: pd.DataFrame = None,
) -> float:
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

        atr = (
            atr_value
            if (isinstance(atr_value, (int, float)) and np.isfinite(atr_value))
            else np.nan
        )
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

    Merge priority (updated): Tiingo -> Alpha (smart) -> FMP (full + legacy) -> Finnhub -> SimFin -> EODHD
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

    # Track provider usage for UI diagnostics (fundamentals/price/ml)
    def mark_provider_usage(provider: str, category: str):
        try:
            usage = st.session_state.setdefault("provider_usage", {})
            cats = usage.setdefault(provider, set())
            cats.add(category)
            usage[provider] = cats
        except Exception:
            pass

    def _merge(src: dict, flag: str, source_name: str):
        if not src:
            return
        # Accept common operating margin keys from providers (if present)
        oper_keys = [
            "oper_margin",
            "operatingMargin",
            "operating_margin",
            "operatingProfitMargin",
            "operatingProfitMarginTTM",
        ]
        for k in [
            "roe",
            "roic",
            "gm",
            "ps",
            "pe",
            "de",
            "rev_g_yoy",
            "eps_g_yoy",
            "oper_margin",
        ]:
            v_cur = merged.get(k, np.nan)
            # First try direct normalized key
            v_new = src.get(k, np.nan)
            # If looking for operating margin, check alternate provider keys as well
            if k == "oper_margin" and (
                not isinstance(v_new, (int, float)) or not np.isfinite(v_new)
            ):
                for ok in oper_keys:
                    if (
                        ok in src
                        and isinstance(src.get(ok), (int, float))
                        and np.isfinite(src.get(ok))
                    ):
                        v_new = src.get(ok)
                        break

            if (
                (not np.isfinite(v_cur))
                and isinstance(v_new, (int, float))
                and np.isfinite(v_new)
            ):
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
        # Mark fundamentals usage for provider (UI status table later)
        mark_provider_usage(source_name, "fundamentals")

    # Helper: sequential FMP fetch with internal delay + global minimal spacing
    def _fmp_sequential_wrap(t: str, key: str) -> dict:
        try:
            # Global spacing to reduce burst pressure
            min_gap = CONFIG.get("FMP_MIN_INTERVAL", 0.6)
            last_ts = st.session_state.get("_fmp_last_call_ts", 0.0)
            now = time.time()
            if now - last_ts < min_gap:
                time.sleep(min_gap - (now - last_ts))
            st.session_state["_fmp_last_call_ts"] = time.time()
        except Exception:
            pass
        out_full = _fmp_full_bundle_fetch(t, key)
        time.sleep(CONFIG.get("FMP_INTER_CALL_DELAY", 0.8))  # spacing between endpoints
        out_legacy = _fmp_metrics_fetch(t, key)
        return {"full": out_full, "legacy": out_legacy}

    # ========== PARALLEL FETCHING (adjusted) ==========
    # Keep overall worker cap modest; FMP endpoints now sequentially wrapped.
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {}

        # Tiingo first (primary provider)
        if CONFIG.get("ENABLE_TIINGO", True):
            tk = _env("TIINGO_API_KEY")
            if tk:
                futures["tiingo"] = ex.submit(_tiingo_fundamentals_fetch, ticker)

        # Alpha Vantage (smart/targeted)
        if (
            enable_alpha_smart
            and bool(st.session_state.get("_alpha_ok"))
            and bool(_env("ALPHA_VANTAGE_API_KEY"))
        ):
            futures["alpha"] = ex.submit(_alpha_overview_fetch, ticker)

        # FMP (wrapped sequentially to reduce pressure)
        fmp_key = _env("FMP_API_KEY")
        if fmp_key:
            futures["fmp_combo"] = ex.submit(_fmp_sequential_wrap, ticker, fmp_key)

        # Finnhub (supplemental)
        futures["finnhub"] = ex.submit(_finnhub_metrics_fetch, ticker)

        # SimFin (supplemental)
        if CONFIG.get("ENABLE_SIMFIN"):
            sim_key = _env("SIMFIN_API_KEY")
            if sim_key:
                futures["simfin"] = ex.submit(_simfin_fetch, ticker, sim_key)

        # EODHD (supplemental)
        if CONFIG.get("ENABLE_EODHD"):
            ek = _env("EODHD_API_KEY") or _env("EODHD_TOKEN")
            if ek:
                futures["eodhd"] = ex.submit(_eodhd_fetch_fundamentals, ticker, ek)

        # Collect results
        results = {}
        fund_timeout = (
            CONFIG.get("PERF_FUND_TIMEOUT_FAST")
            if CONFIG.get("PERF_FAST_MODE")
            else CONFIG.get("PERF_FUND_TIMEOUT")
        )
        for source, fut in futures.items():
            try:
                results[source] = fut.result(timeout=fund_timeout)
            except Exception as e:
                logger.warning(f"Parallel fetch failed for {source}/{ticker}: {e}")
                results[source] = {}

    # Unpack FMP combo
    if results.get("fmp_combo"):
        combo = results.get("fmp_combo", {})
        results["fmp_full"] = combo.get("full", {})
        results["fmp_legacy"] = combo.get("legacy", {})

    # ========== MERGE IN NEW PRIORITY ORDER ==========
    # Primary providers first: Tiingo -> Finnhub -> Alpha -> FMP, then supplemental providers.
    if results.get("tiingo"):
        _merge(results["tiingo"], "from_tiingo", "Tiingo")
        logger.debug(f"Fundamentals merge: Tiingo ✓ {ticker}")

    # Prefer Finnhub over Alpha for primary merging so that Finnhub-provided
    # fields take precedence when both respond (matches expected fallback
    # semantics used in tests and historical behavior).
    if results.get("finnhub"):
        _merge(results["finnhub"], "from_finnhub", "Finnhub")
        logger.debug(f"Fundamentals merge: Finnhub ✓ {ticker}")

    if results.get("alpha"):
        _merge(results["alpha"], "from_alpha", "Alpha")
        logger.debug(f"Fundamentals merge: Alpha ✓ {ticker}")

    if results.get("fmp_full"):
        _merge(results["fmp_full"], "from_fmp_full", "FMP")
        merged["from_fmp"] = True
        logger.debug(
            f"Fundamentals merge: FMP/full ✓ {ticker} fields={results['fmp_full'].get('_fmp_field_count')}"
        )
    if results.get("fmp_legacy"):
        _merge(results["fmp_legacy"], "from_fmp", "FMP")
        logger.debug(f"Fundamentals merge: FMP/legacy ✓ {ticker}")

    if results.get("finnhub"):
        _merge(results["finnhub"], "from_finnhub", "Finnhub")
        logger.debug(f"Fundamentals merge: Finnhub ✓ {ticker}")

    if results.get("simfin"):
        _merge(results["simfin"], "from_simfin", "SimFin")
        logger.debug(f"Fundamentals merge: SimFin ✓ {ticker}")

    if results.get("eodhd"):
        _merge(results["eodhd"], "from_eodhd", "EODHD")
        logger.debug(f"Fundamentals merge: EODHD ✓ {ticker}")

    # Per-field explicit fallbacks now (after Tiingo primary): Alpha -> FMP -> Finnhub -> SimFin -> EODHD
    fallback_order = [
        ("alpha", "Alpha"),
        ("fmp_full", "FMP"),
        ("fmp_legacy", "FMP"),
        ("finnhub", "Finnhub"),
        ("simfin", "SimFin"),
        ("eodhd", "EODHD"),
    ]
    for k in ["roe", "roic", "gm", "ps", "pe", "de", "rev_g_yoy", "eps_g_yoy"]:
        if not isinstance(merged.get(k), (int, float)) or not np.isfinite(
            merged.get(k)
        ):
            for src_key, src_name in fallback_order:
                src = (
                    results.get(src_key, {})
                    if isinstance(results.get(src_key, {}), dict)
                    else {}
                )
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
            src = (
                results.get(src_key, {})
                if isinstance(results.get(src_key, {}), dict)
                else {}
            )
            # check common oper margin keys in provider dicts
            for ok in [
                "oper_margin",
                "operatingMargin",
                "operating_margin",
                "operatingProfitMargin",
                "operatingProfitMarginTTM",
            ]:
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
    cov_fields = [
        merged.get(k)
        for k in [
            "pe",
            "ps",
            "rev_g_yoy",
            "eps_g_yoy",
            "gm",
            "de",
            "oper_margin",
            "roe",
        ]
    ]
    valid_count = sum(
        isinstance(v, (int, float)) and np.isfinite(v) for v in cov_fields
    )
    merged["Fund_Coverage_Pct"] = float(valid_count) / float(len(cov_fields))

    # Log missing fields for debugging
    missing_fields = [
        k
        for k, v in zip(
            ["pe", "ps", "rev_g_yoy", "eps_g_yoy", "gm", "de", "oper_margin", "roe"],
            cov_fields,
        )
        if not (isinstance(v, (int, float)) and np.isfinite(v))
    ]
    if missing_fields:
        print(f"[FUND] Missing for {ticker}: {missing_fields}")

    # If at least one provider responded (any from_* flag) but coverage is zero, set a small floor
    provider_responded = any(
        merged.get(f, False)
        for f in [
            "from_fmp_full",
            "from_fmp",
            "from_alpha",
            "from_finnhub",
            "from_simfin",
            "from_eodhd",
            "from_tiingo",
        ]
    )
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
            if not (
                isinstance(merged.get(f), (int, float)) and np.isfinite(merged.get(f))
            ):
                merged[f] = float(v)
                merged["_defaulted_fields"].append(f)
        # Set a small coverage floor to reflect presence of providers
        merged["Fund_Coverage_Pct"] = max(merged.get("Fund_Coverage_Pct", 0.0), 0.05)

    # Flag partial fundamentals coverage (for UI note) if providers responded but not full coverage
    try:
        if provider_responded and valid_count < len(cov_fields):
            st.session_state.setdefault("_fundamentals_partial", True)
    except Exception:
        pass

    # Ensure 'overall_score_20d' is always present for robust downstream compatibility
    if 'overall_score_20d' not in merged:
        print(f"[DIAG] 'overall_score_20d' missing in merged for {ticker}, injecting np.nan")
        merged['overall_score_20d'] = np.nan
    assert 'overall_score_20d' in merged, "[DIAG] 'overall_score_20d' still missing after injection!"
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
                v = (
                    highlights.get(k)
                    or valuation.get(k)
                    or ratios.get(k)
                    or growth.get(k)
                )
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
            "sector": (
                fj.get("General", {}).get("Sector", "Unknown")
                if isinstance(fj.get("General"), dict)
                else "Unknown"
            ),
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
                if (
                    profile_data
                    and isinstance(profile_data, list)
                    and len(profile_data) > 0
                ):
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


@st.cache_data(ttl=60 * 60)
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
            fut_map = {
                ex.submit(http_get_retry, url, 2, 8): name
                for name, url in endpoints.items()
            }
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
            "oper_margin": ffloat(key_metrics, "operatingProfitMargin"),
            "roe": ffloat(key_metrics, "roe"),
            "roic": ffloat(key_metrics, "roic"),
            "gm": ffloat(key_metrics, "grossProfitMargin"),
            "ps": ffloat(ratios, "priceToSalesRatioTTM"),
            "pe": ffloat(ratios, "priceEarningsRatioTTM"),
            "de": ffloat(ratios, "debtEquityRatioTTM"),
            "rev_g_yoy": ffloat(growth, "revenueGrowth"),
            "eps_g_yoy": ffloat(growth, "epsGrowth"),
            "sector": (
                profile.get("sector", "Unknown")
                if isinstance(profile, dict)
                else "Unknown"
            ),
        }

        # Alternate gross margin if missing
        if not np.isfinite(out.get("gm", np.nan)):
            alt_gm = ffloat(ratios, "grossProfitMarginTTM")
            if np.isfinite(alt_gm):
                out["gm"] = alt_gm

        valid_fields = sum(
            1
            for k, v in out.items()
            if k not in ("sector",) and isinstance(v, (int, float)) and np.isfinite(v)
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
                logger.warning(
                    f"Alpha Vantage rate limit: {j.get('Note') or j.get('Information')}"
                )
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
        gm_calc = (
            (gp / tr) if (np.isfinite(gp) and np.isfinite(tr) and tr > 0) else np.nan
        )

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
        valid_count = sum(
            1
            for k, v in out.items()
            if k != "sector" and isinstance(v, (int, float)) and np.isfinite(v)
        )
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
        gm = (
            (gp / rev) if (np.isfinite(gp) and np.isfinite(rev) and rev > 0) else np.nan
        )
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
        valid_fields = sum(
            1
            for k, v in out.items()
            if k != "sector" and isinstance(v, (int, float)) and np.isfinite(v)
        )
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
        fund_url = (
            f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/daily?token={tk}"
        )
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

        valid_count = sum(
            1 for v in out.values() if isinstance(v, (int, float)) and np.isfinite(v)
        )
        if valid_count >= 2:  # At least 2 valid fields
            out["from_tiingo"] = True
            out["_tiingo_field_count"] = valid_count
            return out
        return {}
    except Exception as e:
        logger.debug(f"Tiingo fundamentals fetch failed for {ticker}: {e}")
        return {}

# -----------------------------------------------------------------
# Canonical merge + batch fundamentals fetch
# -----------------------------------------------------------------
def merge_fundamentals(provider_map: dict) -> dict:
    """Merge provider fundamentals into canonical schema with priority.

    provider_map: { provider_label: dict }
    Priority (highest first): SimFin -> Tiingo -> FMP -> Alpha -> Finnhub -> EODHD -> Marketstack -> Nasdaq
    Fields only filled once; later providers cannot overwrite earlier non-NaN values.
    """
    merged = {f: np.nan for f in FUND_SCHEMA_FIELDS}
    merged["sector"] = "Unknown"
    merged["industry"] = "Unknown"
    merged["_sources"] = {}
    merged["_sources_used"] = []

    priority = [
        ("simfin", "SimFin"),
        ("tiingo", "Tiingo"),
        ("fmp", "FMP"),
        ("alpha", "Alpha"),
        ("finnhub", "Finnhub"),
        ("eodhd", "EODHD"),
        ("marketstack", "Marketstack"),
        ("nasdaq", "Nasdaq"),
    ]
    for key, label in priority:
        src = provider_map.get(key) or {}
        if not isinstance(src, dict) or not src:
            continue
        contributed = False
        for field in FUND_SCHEMA_FIELDS:
            val = src.get(field, np.nan)
            if (field not in merged) or (field in merged and np.isnan(merged[field])):
                if isinstance(val, (int, float)) and np.isfinite(val):
                    merged[field] = float(val)
                    merged["_sources"][field] = label
                    contributed = True
        for f_str in FUND_STRING_FIELDS:
            sval = src.get(f_str)
            if sval and merged[f_str] == "Unknown":
                merged[f_str] = str(sval)
                contributed = True
        if contributed:
            merged["_sources_used"].append(label)

    finite_count = sum(
        1 for f in FUND_SCHEMA_FIELDS if isinstance(merged.get(f), (int, float)) and np.isfinite(merged.get(f))
    )
    merged["Fund_Coverage_Pct"] = finite_count / float(len(FUND_SCHEMA_FIELDS)) if FUND_SCHEMA_FIELDS else 0.0
    merged["fundamentals_available"] = finite_count > 0
    return merged

def _fetch_single_fused(ticker: str, alpha_enabled: bool) -> dict:
    """Fetch fundamentals from all configured providers for a single ticker and merge.
    Fully defensive: never raises.
    """
    try:
        prov = {}
        # SimFin
        if CONFIG.get("ENABLE_SIMFIN") and _env("SIMFIN_API_KEY"):
            try:
                prov["simfin"] = _simfin_fetch(ticker, _env("SIMFIN_API_KEY")) or {}
            except Exception:
                prov["simfin"] = {}
        # Tiingo
        if CONFIG.get("ENABLE_TIINGO", True):
            try:
                prov["tiingo"] = _tiingo_fundamentals_fetch(ticker) or {}
            except Exception:
                prov["tiingo"] = {}
        # FMP (full + legacy sequential throttling)
        fmp_key = _env("FMP_API_KEY")
        if fmp_key:
            try:
                # Respect minimal interval
                min_gap = CONFIG.get("FMP_MIN_INTERVAL", 0.6)
                last_ts = st.session_state.get("_fmp_last_call_ts", 0.0)
                now_ts = time.time()
                gap = now_ts - last_ts
                if gap < min_gap:
                    time.sleep(min_gap - gap)
                st.session_state["_fmp_last_call_ts"] = time.time()
                full = _fmp_full_bundle_fetch(ticker, fmp_key) or {}
                time.sleep(CONFIG.get("FMP_INTER_CALL_DELAY", 0.8))
                legacy = _fmp_metrics_fetch(ticker, fmp_key) or {}
                combo = {**legacy, **full}
                prov["fmp"] = combo
            except Exception as e:
                logger.warning(f"FMP fundamentals failed for {ticker}: {e}")
                prov["fmp"] = {}
        # Alpha Vantage (smart)
        if alpha_enabled:
            try:
                prov["alpha"] = _alpha_overview_fetch(ticker) or {}
            except Exception:
                prov["alpha"] = {}
        # Finnhub
        try:
            prov["finnhub"] = _finnhub_metrics_fetch(ticker) or {}
        except Exception:
            prov["finnhub"] = {}
        # EODHD
        ek = _env("EODHD_API_KEY")
        if ek:
            try:
                prov["eodhd"] = _eodhd_fetch_fundamentals(ticker, ek) or {}
            except Exception:
                prov["eodhd"] = {}
        # Marketstack (stub until implemented)
        if CONFIG.get("ENABLE_MARKETSTACK") and _env("MARKETSTACK_API_KEY"):
            prov["marketstack"] = {}
        # Nasdaq (stub; used primarily for sector/industry fallback)
        if CONFIG.get("ENABLE_NASDAQ_DL") and (_env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY")):
            prov["nasdaq"] = {}
        merged = merge_fundamentals(prov)
        # Mark provider usage
        for src_label in merged.get("_sources_used", []):
            try:
                mark_provider_usage(src_label, "fundamentals")
            except Exception:
                pass
        merged["Ticker"] = ticker
        return merged
    except Exception as e:
        logger.warning(f"Fundamentals fused fetch catastrophic failure for {ticker}: {e}")
        m = _empty_fund_row()
        m["Ticker"] = ticker
        return m

def fetch_fundamentals_batch(tickers: list, alpha_top_n: int = 15) -> pd.DataFrame:
    """Batch fundamentals enrichment.
    Always returns one row per ticker. On total failure returns empty rows with coverage=0.
    """
    if not tickers:
        return pd.DataFrame(columns=["Ticker"] + FUND_SCHEMA_FIELDS + FUND_STRING_FIELDS)
    rows = []
    # Parallel per-ticker to avoid one slow provider blocking others
    max_workers = min(8, max(1, CONFIG.get("FUND_BATCH_MAX_WORKERS", 8)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {}
        for rank, tkr in enumerate(tickers, 1):
            alpha_enabled = rank <= alpha_top_n and not CONFIG.get("PERF_FAST_MODE")
            fut_map[ex.submit(_fetch_single_fused, tkr, alpha_enabled)] = tkr
        for fut in as_completed(fut_map):
            try:
                rows.append(fut.result())
            except Exception:
                tkr = fut_map[fut]
                m = _empty_fund_row()
                m["Ticker"] = tkr
                rows.append(m)
    df = pd.DataFrame(rows).set_index("Ticker")
    return df


@st.cache_data(ttl=60 * 60)
@st.cache_data(ttl=3600, show_spinner=False)
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_beta_vs_benchmark(ticker: str, bench: str = "SPY", days: int = 252) -> float:
    """Calculate beta with timeout protection and caching."""
    try:
        from concurrent.futures import (
            ThreadPoolExecutor,
            TimeoutError as FuturesTimeoutError,
        )

        def _download_both():
            end = datetime.utcnow()
            start = end - timedelta(days=days + 30)
            df_t = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                timeout=8,
            )
            df_b = yf.download(
                bench, start=start, end=end, auto_adjust=True, progress=False, timeout=8
            )
            return df_t, df_b

        # Use ThreadPoolExecutor for timeout (works on all platforms)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_download_both)
            try:
                df_t, df_b = future.result(timeout=10)
            except FuturesTimeoutError:
                return np.nan

        if df_t.empty or df_b.empty:
            return np.nan
        j = pd.concat(
            [
                df_t["Close"].pct_change(fill_method=None).dropna(),
                df_b["Close"].pct_change(fill_method=None).dropna(),
            ],
            axis=1,
        ).dropna()
        j.columns = ["rt", "rb"]
        if len(j) < 40:
            return np.nan
        slope = np.polyfit(j["rb"].to_numpy(), j["rt"].to_numpy(), 1)[0]
        return float(slope)
    except (Exception, TimeoutError):
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
st.set_page_config(
    page_title="סקאוט מניות — 2026", page_icon="📈", layout="wide"
)

# === HEBREW RTL STYLING WITH LTR ENGLISH TEXT ===
st.markdown("""
<style>
    /* Global RTL direction */
    body, .stApp, .main, .block-container {
        direction: rtl;
        text-align: right;
    }
    
    /* Streamlit overrides */
    .css-1l269bu { direction: rtl; }  /* Main content */
    .stSidebar { direction: rtl; }
    
    /* RTL text alignment */
    h1, h2, h3, h4, h5, h6 { text-align: right; }
    
    /* Force LTR for English text, tickers, numbers, provider names */
    span.ltr, .ltr, .stMetricDelta, [class*="st-emotion"] {
        direction: ltr !important;
        text-align: left !important;
        unicode-bidi: embed;
    }
    
    /* Button styling */
    .stButton > button { border-radius: 8px; }
    
    /* Card styling */
    .stContainer { direction: rtl; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    
    /* Metric styling */
    .stMetric { text-align: right; }
    
    /* Margin adjustments */
    h1, h2, h3 { margin-top: 1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 סקאוט מניות — 2026 אסף")
st.caption("🇮🇱 סקאן מניות אישי בעברית | כלי למחקר בלבד. לא ייעוץ השקעות.")

st.markdown("### 📊 Read-Only Dashboard")
st.session_state["enable_openai_targets"] = False

# Force ML always on (no visible toggle)
st.session_state["ENABLE_ML"] = True
st.session_state["USE_FINAL_SCORE_SORT"] = True

st.markdown("---")


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
marketstack_key = (
    _env("MARKETSTACK_API_KEY") if CONFIG.get("ENABLE_MARKETSTACK") else None
)
nasdaq_key = (
    (_env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY"))
    if CONFIG.get("ENABLE_NASDAQ_DL")
    else None
)
eodhd_key = (
    (_env("EODHD_API_KEY") or _env("EODHD_TOKEN"))
    if CONFIG.get("ENABLE_EODHD")
    else None
)

# Critical API keys presence check (non-blocking).
# Consider it CRITICAL only if neither key is present at all.
alpha_key_present = bool(_env("ALPHA_VANTAGE_API_KEY") or _env("ALPHAVANTAGE_API_KEY"))
finn_key_present = bool(_env("FINNHUB_API_KEY"))

if not (alpha_key_present or finn_key_present):
    st.error(
        f"""
**CRITICAL: Missing API Keys**

At least one of the following is required for fundamentals:
`ALPHA_VANTAGE_API_KEY` or `FINNHUB_API_KEY`

**Streamlit Cloud:** Add either Alpha Vantage or Finnhub key (or both) under Settings -> Secrets.
Do not include quote characters around the value.

**Local:** Put keys in `.env` and ensure `load_dotenv()` runs.

The pipeline will continue with technical-only scoring.
"""
    )
elif not (alpha_ok or finn_ok):
    st.warning(
        "⚠️ Providers detected but currently unavailable (rate limits or network). Falling back to technical-only scoring and any other price sources."
    )
elif not alpha_ok and finn_ok:
    st.warning(
        "⚠️ Alpha Vantage unavailable (rate limits or config) - falling back to Finnhub and other providers. Recommendations will still be generated."
    )

# Store provider status in session state for connectivity checks
st.session_state["_alpha_vantage_ok"] = alpha_ok
st.session_state["_finnhub_ok"] = finn_ok
st.session_state["_polygon_ok"] = poly_ok
st.session_state["_tiingo_ok"] = tiin_ok
st.session_state["_fmp_ok"] = fmp_ok

# Initialize centralized status manager
status_manager = StatusManager(get_pipeline_stages())

# Map pipeline detail messages to status stage advancements
_stage_triggers = [
    ("Fetching historical data", "Historical Data Fetch"),
    ("Computing technical indicators", "Technical Indicators"),
    ("Applying Beta filter", "Beta Filter"),
    ("Applying advanced filters", "Advanced Filters"),
    ("Fetching fundamentals", "Fundamentals Enrichment"),
    ("Classifying & Allocating", "Risk Classification"),
]
_completed_stages: Set[str] = set()


def status_with_progress(message: str) -> None:
    """Update detail text and advance progress when key milestones fire."""
    status_manager.update_detail(message)
    for trigger, stage_name in _stage_triggers:
        if trigger in message and stage_name not in _completed_stages:
            status_manager.advance(stage_name)
            _completed_stages.add(stage_name)

# timers
def t_start() -> float:
    return time.perf_counter()


def t_end(t0: float) -> float:
    return time.perf_counter() - t0


phase_times: Dict[str, float] = {}
if "av_calls" not in st.session_state:
    st.session_state.av_calls = 0

# ==================== DATA SOURCE MODE ====================
st.markdown("---")
st.markdown("### ⚡ מצב נתונים")

# One-shot Live Scan button:
# - By default, the app prefers precomputed scan (if it exists).
# - If the user clicks the button, we force a single live run and ignore the snapshot for this run only.
if st.button("🔄 הרץ לייב סריקה עכשיו", key="live_scan_button"):
    st.session_state["force_live_scan_once"] = True
    st.session_state["skip_pipeline"] = False  # ensure live path
    st.rerun()

force_live_scan_once = st.session_state.get("force_live_scan_once", False)

# Import scan I/O helpers
from core.scan_io import load_latest_scan, save_scan as save_scan_helper
import time
import io

def save_latest_scan_from_results(results_df: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
    """Helper to save scan results using scan_io.save_scan with proper paths.
    Saves even empty DataFrames to avoid missing snapshot state."""
    if results_df is None:
        logger.warning("Cannot save results: DataFrame is None")
        return
    
    # Ensure parquet-safe types (convert complex objects to strings)
    results_to_save = results_df.copy()
    for col in results_to_save.columns:
        if results_to_save[col].dtype == "object":
            results_to_save[col] = results_to_save[col].apply(
                lambda v: v
                if isinstance(v, (str, int, float, bool, np.bool_, np.integer, np.floating)) or v is None
                else str(v)
            )
    
    output_dir = Path(__file__).parent / "data" / "scans"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save live runs under a separate filename to avoid shadowing autoscan snapshots
    path_latest = output_dir / "latest_scan_live.parquet"
    
    # Prepare metadata
    meta = metadata.copy() if metadata else {}
    meta["total_tickers"] = len(results_df)
    meta["scan_type"] = "live_streamlit"
    # Add build commit for parity with autoscan artifacts
    try:
        import subprocess
        meta["build_commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        )
    except Exception:
        pass
    
    try:
        save_scan_helper(
            results_df=results_to_save,
            config=CONFIG,
            path_latest=path_latest,
            path_timestamped=None,  # Don't create timestamped backup for live scans
            metadata=meta
        )
        logger.info(f"✅ Saved live scan results: {len(results_df)} tickers to {path_latest}")
    except Exception as e:
        logger.error(f"❌ Failed to save scan: {e}")
        raise

# Attempt to load precomputed scan (internal, no user dropdown)
precomputed_df = None
precomputed_meta = None
use_precomputed = False


def _load_precomputed_scan_with_fallback(scan_dir: Path):
    """Load the freshest available snapshot.
    Considers:
      - Any latest_scan*.parquet (including latest_scan_live.parquet)
      - Timestamped backups: scan_*.parquet (newest)
    Chooses the newest by metadata timestamp or file mtime.
    """
    latest_candidates = sorted(scan_dir.glob("latest_scan*.parquet"))

    def _load(path: Path):
        df, meta = load_latest_scan(path)
        if df is None or meta is None:
            return None, None, None
        # Try to parse ISO timestamp, else use file mtime
        ts_meta = meta.get("timestamp")
        try:
            ts_parsed = datetime.fromisoformat((ts_meta or "").replace("Z", "+00:00")) if ts_meta else None
        except Exception:
            ts_parsed = None
        try:
            ts_file = datetime.fromtimestamp(path.stat().st_mtime)
        except Exception:
            ts_file = None
        ts_effective = max([t for t in [ts_parsed, ts_file] if t is not None], default=None)
        return df, meta, ts_effective

    best_df, best_meta, best_ts, best_path = None, None, None, None

    # Load any latest_scan*.parquet candidates if present
    for candidate_path in latest_candidates:
        if candidate_path.exists():
            df_c, meta_c, ts_c = _load(candidate_path)
            if df_c is not None and meta_c is not None:
                if best_ts is None or (ts_c and ts_c > best_ts):
                    best_df, best_meta, best_ts, best_path = df_c, meta_c, ts_c, candidate_path

    if best_df is not None and best_meta is not None:
        return best_df, best_meta, best_path

    # Fallback: pick newest timestamped backup
    candidates = sorted(scan_dir.glob("scan_*.parquet"), reverse=True)
    for candidate in candidates:
        df_cand, meta_cand = load_latest_scan(candidate)
        if df_cand is not None and meta_cand is not None:
            meta_cand.setdefault("timestamp", candidate.stem.replace("scan_", ""))
            meta_cand.setdefault("total_tickers", len(df_cand))
            return df_cand, meta_cand, candidate

    # If nothing found, default to conventional latest path for display
    return None, None, (scan_dir / "latest_scan.parquet")


scan_dir = Path(__file__).parent / "data" / "scans"
t0_precomputed = time.perf_counter()
try:
    status_manager.update_detail("Loading precomputed scan from disk...")
    precomputed_df, precomputed_meta, scan_path = _load_precomputed_scan_with_fallback(scan_dir)
    t1_precomputed = time.perf_counter()
    load_time = t1_precomputed - t0_precomputed
    logger.info(f"[PERF] Precomputed scan load time: {load_time:.3f}s (path={scan_path})")
except Exception as exc:
    logger.warning(f"Precomputed scan load failed: {exc}")
    scan_path = scan_dir / "latest_scan.parquet"
    t1_precomputed = time.perf_counter()
    load_time = t1_precomputed - t0_precomputed

# Optionally prefer remote autoscan artifacts from GitHub if newer than local/live
def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception:
        return None

if CONFIG.get("USE_REMOTE_AUTOSCAN", True):
    try:
        repo = CONFIG.get("REMOTE_AUTOSCAN_REPO", "asafamos/stock-scout")
        branch = CONFIG.get("REMOTE_AUTOSCAN_BRANCH", "main")
        base = f"https://raw.githubusercontent.com/{repo}/{branch}/data/scans"
        # Fetch remote metadata first
        r_meta = requests.get(f"{base}/latest_scan.json", timeout=8)
        if r_meta.ok:
            meta_remote = r_meta.json()
            ts_remote = _parse_iso(meta_remote.get("timestamp", ""))

            # Determine current local timestamp (auto/live whichever was chosen)
            ts_current = None
            if precomputed_meta and precomputed_meta.get("timestamp"):
                ts_current = _parse_iso(precomputed_meta.get("timestamp"))
            if not ts_current:
                try:
                    mtime_source = (scan_path.with_suffix('.json') if (scan_path and scan_path.with_suffix('.json').exists()) else scan_path)
                    ts_current = datetime.fromtimestamp(mtime_source.stat().st_mtime)
                except Exception:
                    ts_current = None

            # Only prefer remote if it's newer than current
            if ts_remote and (ts_current is None or ts_remote > ts_current) and not force_live_scan_once:
                r_pq = requests.get(f"{base}/latest_scan.parquet", timeout=20)
                if r_pq.ok:
                    try:
                        df_remote = pd.read_parquet(io.BytesIO(r_pq.content), engine="pyarrow")
                        meta_remote.setdefault("total_tickers", len(df_remote))
                        precomputed_df, precomputed_meta = df_remote, meta_remote
                        scan_path = Path("REMOTE:latest_scan.parquet")
                        logger.info("Using remote autoscan artifacts from GitHub (remote is newer)")
                    except Exception as e:
                        logger.warning(f"Failed to parse remote parquet: {e}")
    except Exception as e:
        logger.warning(f"Remote autoscan fetch skipped/failed: {e}")

timestamp_str = "unknown"
universe_size = 0
scan_age_hours = None
scan_too_old = False

if precomputed_meta is not None:
    timestamp_str = precomputed_meta.get("timestamp", "unknown")
    universe_size = precomputed_meta.get("total_tickers", precomputed_meta.get("universe_size", 0))
    
    # Check scan age (12 hour limit)
    try:
        # Parse metadata timestamp (may be naive or UTC)
        scan_time_meta = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        # Also consider file modification time (parquet/json), use the freshest
        try:
            meta_path = scan_path.with_suffix('.json')
            mtime_source = meta_path if meta_path.exists() else scan_path
            scan_time_file = datetime.fromtimestamp(mtime_source.stat().st_mtime)
        except Exception:
            scan_time_file = scan_time_meta

        # Prefer the most recent timestamp available
        scan_time = max(scan_time_meta, scan_time_file)
        scan_age_hours = (datetime.now() - scan_time).total_seconds() / 3600
        scan_too_old = scan_age_hours > 12
    except Exception:
        scan_too_old = True

# Helper: show snapshot provenance banner
def _render_snapshot_banner(meta: Dict[str, Any], path_obj: Path, age_hours: Optional[float]) -> None:
    src = "Remote autoscan (GitHub)" if str(path_obj).startswith("REMOTE:") else (
        "Local latest" if path_obj.name == "latest_scan.parquet" else "Local backup"
    )
    ts = meta.get("timestamp", "unknown")
    commit = meta.get("build_commit", "unknown")
    saved = meta.get("total_tickers")
    uni = meta.get("universe_size")
    age_txt = f"{age_hours:.1f} שעות" if isinstance(age_hours, (int, float)) else "unknown"
    if saved is not None and uni is not None:
        st.caption(f"🧾 מקור: {src} • קומיט: {commit} • זמן: {ts} • גיל: {age_txt} • מניות נשמרו: {saved} מתוך נסקרו: {uni}")
    else:
        size = saved or uni or "unknown"
        st.caption(f"🧾 מקור: {src} • קומיט: {commit} • זמן: {ts} • גיל: {age_txt} • מניות: {size}")

if force_live_scan_once:
    # User explicitly forced a live run: ignore any snapshot age/status
    st.info("🔄 סריקה חיה נכפית - מתעלם מסריקה אוטומטית.")
    st.caption(f"📊 סריקה אוטומטית מ-{timestamp_str} מתעלמת עבור הרצה זו.")
    use_precomputed = False
    st.session_state["skip_pipeline"] = False
elif precomputed_df is not None and precomputed_meta is not None and not scan_too_old:
    # Successfully loaded and NOT too old -> use precomputed snapshot
    status_manager.advance(
        f"Precomputed scan loaded: {universe_size} tickers (last updated: {timestamp_str})"
    )
    age_display = f"{scan_age_hours:.1f}" if isinstance(scan_age_hours, (int, float)) else "unknown"
    st.success(f"✅ נתונים עדכניים מסריקה אוטומטית ({age_display} שעות)")
    _render_snapshot_banner(precomputed_meta, scan_path, scan_age_hours)
    st.caption(f"📊 {universe_size} מניות נותחו | ⏰ סריקה אוטומטית פעמיים ביום (8:00 + 20:00 UTC)")

    st.session_state["skip_pipeline"] = True
    st.session_state["precomputed_results"] = precomputed_df
    try:
        st.session_state["universe_size"] = int(universe_size)
    except Exception:
        pass
    logger.info(f"[PERF] Precomputed scan: DataFrame shape {precomputed_df.shape}")
    use_precomputed = True
else:
    # Either no snapshot exists, or scan is too old
    if scan_too_old and precomputed_df is not None:
        age_display = f"{scan_age_hours:.1f}" if isinstance(scan_age_hours, (int, float)) else "unknown"
        st.warning(f"⚠️ הסריקה הקיימת ישנה מדי ({age_display} שעות) - מחכה לסריקה אוטומטית הבאה")
        _render_snapshot_banner(precomputed_meta, scan_path, scan_age_hours)
        st.info("💡 סריקה אוטומטית חדשה תתבצע תוך מספר שעות (פעמיים ביום: 8:00 + 20:00 UTC)")
        # Use old scan anyway but warn user
        st.session_state["skip_pipeline"] = True
        st.session_state["precomputed_results"] = precomputed_df
        try:
            st.session_state["universe_size"] = int(universe_size)
        except Exception:
            pass
        use_precomputed = True
    else:
        st.info("📊 אין סריקה זמינה - מחכה לסריקה אוטומטית הבאה.")
        st.caption("💡 סריקות אוטומטיות רצות פעמיים ביום דרך GitHub Actions.")
        use_precomputed = False
        st.session_state["skip_pipeline"] = False
    
# Reset the one-shot flag (always)
st.session_state["force_live_scan_once"] = False

# ==================== MAIN PIPELINE ====================
st.markdown("---")
st.markdown("### 🚀 Pipeline Execution")

# Check if we should skip pipeline (precomputed mode)
skip_pipeline = st.session_state.get("skip_pipeline", False) and use_precomputed

# Always show market regime
t0_regime = t_start()
if not skip_pipeline:
    status_manager.update_detail("Analyzing SPY/QQQ/VIX trends...")

@st.cache_data(ttl=3600)
def _cached_detect_market_regime(lookback_days: int = 60):
    return detect_market_regime(lookback_days=lookback_days)

market_regime_data = _cached_detect_market_regime(lookback_days=60)
regime = market_regime_data.get("regime", "neutral")
regime_confidence = market_regime_data.get("confidence", 50)
phase_times["market_regime"] = t_end(t0_regime)

# Show regime with color coding
regime_emoji = {"bullish": "📈", "neutral": "➡️", "bearish": "📉"}
regime_color = {"bullish": "#16a34a", "neutral": "#f59e0b", "bearish": "#dc2626"}

st.markdown(
    f"""<div style='background:{regime_color[regime]};color:white;padding:12px;border-radius:8px;margin:10px 0'>
    <strong>{regime_emoji[regime]} Market Regime: {regime.upper()}</strong> (confidence: {regime_confidence}%)<br>
    <small>{market_regime_data.get('details', '')}</small>
    </div>""",
    unsafe_allow_html=True
)

# Store in session state for use in target calculations
st.session_state['market_regime'] = market_regime_data

if skip_pipeline:
    # Use precomputed results from full pipeline
    t0_stage3 = time.perf_counter()
    results = st.session_state.get("precomputed_results")
    logger.info(f"[PRECOMPUTED] Loaded {len(results)} tickers from full pipeline scan")
    
    # Add 'Score' alias for overall_score_20d to maintain compatibility
    if "overall_score_20d" in results.columns and "Score" not in results.columns:
        results["Score"] = results["overall_score_20d"]
    elif "FinalScore_20d" in results.columns and "Score" not in results.columns:
        results["Score"] = results["FinalScore_20d"]
    
    # UI now shows all pipeline results; no adaptive percentile or top-N capping
    logger.info(f"[PRECOMPUTED] Final display: {len(results)} stocks (no adaptive/top-N filter applied)")
    
    # IMPORTANT: Update session state with filtered results so they persist
    st.session_state["precomputed_results"] = results.copy()
    
    try:
        status_manager.update_detail(f"Precomputed scan: {len(results)} top stocks")
        status_manager.set_progress(1.0)
    except Exception:
        pass
    
    # Show summary to user
    try:
        original_count = int(st.session_state.get("universe_size", 0))
        if not original_count:
            # Fall back to computed universe_size (from metadata) or current results length
            try:
                original_count = int(universe_size)
            except Exception:
                original_count = len(results)
    except Exception:
        original_count = len(results)
    st.info(f"⚡ **{len(results)} מניות מובילות** מתוך {original_count} שעברו סריקה מלאה")
    st.caption("✅ כולל: ML model, Technical scoring, Fundamental data, Risk assessment, Classification")
    
    t1_stage3 = time.perf_counter()
    logger.info(f"[PERF] Precomputed stage 3/10 (post-load to recommendations) time: {t1_stage3-t0_stage3:.3f}s")
    
else:
    # Live scan execution fallback
    with st.status("🚀 Running Live Scan...", expanded=True) as status:
        status.write("Initializing pipeline...")
        # 1. Fetch Universe
        universe = fetch_top_us_tickers_by_market_cap(limit=CONFIG["UNIVERSE_LIMIT"])
        status.write(f"Fetched universe: {len(universe)} tickers")
        
        # 2. Run Pipeline
        results, data_map = run_scan_pipeline(
            universe, 
            CONFIG, 
            status_callback=status.write
        )
        
        # 3. Save Results
        if not results.empty:
            try:
                # Ensure metadata includes timestamp and type
                meta = {
                    "universe_count": len(universe),
                    "timestamp": datetime.utcnow().isoformat(),
                    "scan_type": "live_streamlit"
                }
                save_latest_scan_from_results(results, metadata=meta)
                st.session_state["precomputed_results"] = results
                st.success(f"✅ Scan complete: {len(results)} results found")
            except Exception as e:
                logger.warning(f"Failed to save live scan: {e}")
        else:
            st.error("❌ Live scan returned 0 results. Check logs/filtering.")

# Debug logging if enabled
create_debug_expander({
    "regime": regime,
    "confidence": regime_confidence,
    "spy_trend": market_regime_data.get("spy_trend", 0),
    "qqq_trend": market_regime_data.get("qqq_trend", 0),
    "vix_level": market_regime_data.get("vix", 0),
}, title="📊 Market Regime Details")

# API keys and universe provider diagnostics
try:
    api_status = {
        "FMP_API_KEY": "OK" if bool(_env("FMP_API_KEY")) else "MISSING",
        "POLYGON_API_KEY": "OK" if bool(_env("POLYGON_API_KEY")) else "MISSING",
        "EODHD_API_KEY": "OK" if bool(_env("EODHD_API_KEY")) else "MISSING",
        "FINNHUB_API_KEY": "OK" if bool(_env("FINNHUB_API_KEY")) else "MISSING",
        "ALPHA_VANTAGE_API_KEY": "OK" if bool(_env("ALPHA_VANTAGE_API_KEY")) else "MISSING",
        "TIINGO_API_KEY": "OK" if bool(_env("TIINGO_API_KEY")) else "MISSING",
        "Universe_Provider": LAST_UNIVERSE_PROVIDER,
    }
    create_debug_expander(api_status, title="🔑 API Keys & Universe Provider")
except Exception:
    pass

# Initialize sources tracker
sources_overview = SourcesOverview()

## Read-only mode: live scan execution removed. Always use precomputed results loaded above.

# External price verification (Top-K)
t0 = t_start()
if CONFIG.get("EXTERNAL_PRICE_VERIFY", False):
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
    results["Historical_StdDev"] = np.nan  # Initialize for all, fill selectively below


def _fetch_external_for(
    tkr: str, py: float
) -> Tuple[str, Dict[str, Optional[float]], List[str]]:
    vals: Dict[str, Optional[float]] = {}
    srcs: List[str] = []

    # helper to mark price usage
    def _mark_price(provider: str):
        try:
            usage = st.session_state.setdefault("provider_usage", {})
            cats = usage.setdefault(provider, set())
            cats.add("price")
            usage[provider] = cats
        except Exception:
            pass

    if np.isfinite(py):
        vals["Yahoo"] = float(py)
        srcs.append("🟡Yahoo")
        _mark_price("Yahoo")
    if alpha_ok:
        p = get_alpha_price(tkr)
        if p is not None:
            vals.setdefault("Alpha", p)
            srcs.append("🟣Alpha")
            st.session_state.av_calls = st.session_state.get("av_calls", 0) + 1
            _mark_price("Alpha")
    if finn_ok:
        p = get_finnhub_price(tkr)
        if p is not None:
            vals.setdefault("Finnhub", p)
            srcs.append("🔵Finnhub")
            _mark_price("Finnhub")
    if poly_ok and _env("POLYGON_API_KEY"):
        p = get_polygon_price(tkr)
        if p is not None:
            vals.setdefault("Polygon", p)
            srcs.append("🟢Polygon")
            _mark_price("Polygon")
    if tiin_ok and _env("TIINGO_API_KEY"):
        p = get_tiingo_price(tkr)
        if p is not None:
            vals.setdefault("Tiingo", p)
            srcs.append("🟠Tiingo")
            _mark_price("Tiingo")
    if CONFIG.get("ENABLE_MARKETSTACK") and _env("MARKETSTACK_API_KEY"):
        p = get_marketstack_price(tkr)
        if p is not None:
            vals.setdefault("Marketstack", p)
            srcs.append("🧩Marketstack")
            _mark_price("Marketstack")
    if CONFIG.get("ENABLE_NASDAQ_DL") and (
        _env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY")
    ):
        p = get_nasdaq_price(tkr)
        if p is not None:
            vals.setdefault("NasdaqDL", p)
            srcs.append("🏛NasdaqDL")
            _mark_price("Nasdaq")
    if CONFIG.get("ENABLE_EODHD") and (_env("EODHD_API_KEY") or _env("EODHD_TOKEN")):
        p = get_eodhd_price(tkr)
        if p is not None:
            vals.setdefault("EODHD", p)
            srcs.append("📘EODHD")
            _mark_price("EODHD")
    # Return collected prices and source badges
    return tkr, vals, srcs
    # Stage tracking now handled by StatusManager


# External price verification - run if ANY provider is available
any_price_provider = (
    finn_ok
    or (poly_ok and _env("POLYGON_API_KEY"))
    or (CONFIG.get("ENABLE_MARKETSTACK") and _env("MARKETSTACK_API_KEY"))
    or (
        CONFIG.get("ENABLE_NASDAQ_DL")
        and (_env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY"))
    )
    or (CONFIG.get("ENABLE_EODHD") and (_env("EODHD_API_KEY") or _env("EODHD_TOKEN")))
)

if CONFIG["EXTERNAL_PRICE_VERIFY"] and any_price_provider and "Price_Yahoo" in results.columns:
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
                    " - ".join(srcs),
                ]
                # Compute historical std dev for this ticker (only for verified subset)
                ticker = results.loc[idx, "Ticker"]
                if not skip_pipeline and ticker in data_map:
                    hist = data_map[ticker]
                    if len(hist) >= 5:  # Minimum 5 candles
                        recent = hist["Close"].tail(min(30, len(hist)))
                        if len(recent) >= 5:
                            results.at[idx, "Historical_StdDev"] = float(recent.std())
    # Price reliability metric (enhanced with better spread: 0.1-1.0)
    results["Price_Reliability"] = np.nan
    for i, row in results.iterrows():
        pmean = row.get("Price_Mean", np.nan)
        pstd = row.get("Price_STD", np.nan)
        providers = (
            str(row.get("Source_List", "")).split(" - ")
            if isinstance(row.get("Source_List"), str)
            else []
        )
        providers = [p for p in providers if p]
        count = len(providers)

        if np.isfinite(pmean) and pmean > 0 and np.isfinite(pstd):
            pct_std = pstd / pmean

            # Source count component (0-4 sources -> 0.2-1.0)
            if count == 1:
                source_score = 0.2
            elif count == 2:
                source_score = 0.5
            elif count == 3:
                source_score = 0.75
            else:  # 4+
                source_score = 1.0

            # Variance component (lower variance = higher reliability)
            # 0% variance -> 1.0, 5% variance -> 0.5, 10%+ variance -> 0.2
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
            results.at[i, "Price_Reliability"] = round(
                np.clip(reliability, 0.1, 1.0), 4
            )
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
    results["Price_Sources_Count"] = results["Source_List"].apply(
        lambda s: len(str(s).split(" - ")) if isinstance(s, str) and s else 0
    )

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
            "from_fmp_full",
            "from_fmp",
            "from_simfin",
            "from_eodhd",
            "from_alpha",
            "from_finnhub",
            "from_tiingo",
        ]
        for i, row in results.iterrows():
            # Count providers that truly contributed (flags are written earlier)
            provider_count = int(sum(bool(row.get(f)) for f in fund_flags))
            # If coverage indicates data but flags are zero, treat as 1 source
            cov = row.get("Fund_Coverage_Pct", np.nan)
            if (
                provider_count == 0
                and isinstance(cov, (int, float))
                and np.isfinite(cov)
                and cov > 0
            ):
                provider_count = 1
            final_rel = _map_sources_to_reliability(provider_count)
            results.at[i, "Fundamental_Reliability"] = round(final_rel, 4)
    else:
        results["Fundamental_Reliability"] = 0.0

    # Fundamental sources count column (flags) — exact count (0 allowed if truly none)
    fund_flags = [
        "from_fmp_full",
        "from_fmp",
        "from_simfin",
        "from_eodhd",
        "from_alpha",
        "from_finnhub",
        "from_tiingo",
    ]

    def _fund_count(row: pd.Series) -> int:
        return int(sum(bool(row.get(f)) for f in fund_flags))

    results["Fundamental_Sources_Count"] = results.apply(_fund_count, axis=1)

    # Expose canonical fund reliability field using the simple mapping helper
    try:
        from core.scoring.fundamental import compute_fund_reliability
    except Exception:
        compute_fund_reliability = None

    if compute_fund_reliability is not None:
        results["fund_reliability"] = results["Fundamental_Sources_Count"].apply(
            lambda n: float(compute_fund_reliability(n))
        )
        # Keep legacy column name in sync
        results["Fundamental_Reliability"] = results["fund_reliability"].copy()
    else:
        # Fallback: copy existing value or 0.0
        results["fund_reliability"] = results.get("Fundamental_Reliability", 0.0)

    # Combined reliability score
    if (
        "Price_Reliability" in results.columns
        and "Fundamental_Reliability" in results.columns
    ):
        results["Reliability_Score"] = (
            0.4 * results["Price_Reliability"].fillna(0)
            + 0.6 * results["Fundamental_Reliability"].fillna(0)
        ).round(4)
    else:
        results["Reliability_Score"] = np.nan
phase_times["price_verification"] = t_end(t0)
status_manager.update_detail(f"Price verification: {len(results)} validated")
status_manager.advance("Price Verification")


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
    # Vectorized: rank within each sector, keep rank <= cap
    df["_rank"] = df.groupby("Sector", sort=False).cumcount() + 1
    result = df[df["_rank"] <= cap].drop("_rank", axis=1).reset_index(drop=True)
    return result


score_candidates = [
    "overall_score_20d",
    "Score",
    "overall_score",
    "conviction_v2_final",
    "overall_score_pretty",
]
score_col = None
for c in score_candidates:
    if c in results.columns:
        score_col = c
        break

# Choose a safe ticker column if available
ticker_candidates = ["Ticker", "symbol", "Symbol", "ticker"]
ticker_col = None
for t in ticker_candidates:
    if t in results.columns:
        ticker_col = t
        break

sort_by = []
ascending = []
if score_col is not None:
    sort_by.append(score_col)
    ascending.append(False)
if ticker_col is not None:
    sort_by.append(ticker_col)
    ascending.append(True)

if sort_by:
    sorted_results = results.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
else:
    # No valid sort keys; keep original order
    sorted_results = results.reset_index(drop=True)

results = apply_sector_cap(
    sorted_results,
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
        for provider in price_sources.split(" - "):
            if provider and provider not in badges:
                badges.append(provider)
    if not badges:
        badges.append("🟡Yahoo")
    return " - ".join(badges)


results["Price_Sources"] = results.apply(source_badges, axis=1)
results["Price_Mean_Rounded"] = results["Price_Mean"].round(2)
results["Price_STD_Rounded"] = results["Price_STD"].round(4)
results["Unit_Price"] = np.where(
    results["Price_Mean_Rounded"].notna(),
    results["Price_Mean_Rounded"],
    results.get("Price_Yahoo", results.get("Close", np.nan)),
)
results["Unit_Price"] = pd.to_numeric(results["Unit_Price"], errors="coerce")

# Show ALL stocks that passed filters (no limit)
TOPN = len(results)

# Apply ML confidence threshold (support multiple column aliases)
ml_threshold_value = float(st.session_state.get("ml_threshold", 0)) / 100.0
ml_col = None
for cand in ("ML_Prob", "ML_20d_Prob", "ML_Probability"):
    if cand in results.columns:
        ml_col = cand
        break
if ml_threshold_value > 0 and ml_col is not None:
    before_ml = len(results)
    results = results[pd.to_numeric(results[ml_col], errors="coerce") >= ml_threshold_value].copy()
    after_ml = len(results)
    logger.info(
        f"ML confidence filter: {before_ml} -> {after_ml} stocks (threshold={ml_threshold_value:.0%}, col={ml_col})"
    )
    TOPN = len(results)  # Update TOPN after filtering

# === ALLOCATE BUDGET (must happen before using 'סכום קנייה ($)') ===
total_budget = float(st.session_state.get("total_budget", CONFIG["BUDGET_TOTAL"]))
min_position = float(st.session_state.get("min_position", 50.0))
max_position_pct = float(st.session_state.get("max_position_pct", 15.0))
results = allocate_budget(
    results,
    total=total_budget,
    min_pos=min_position,
    max_pos_pct=max_position_pct,
    score_col="Score" if "Score" in results.columns else "conviction_v2_final",
    dynamic_sizing=True
)

alloc_df = results.reset_index(drop=True).copy()

# Defensive: ensure allocation column exists even if upstream function changed
if "סכום קנייה ($)" not in results.columns:
    if "buy_amount_v2" in results.columns:
        results["סכום קנייה ($)"] = pd.to_numeric(results["buy_amount_v2"], errors="coerce")
    elif "Buy Amount ($)" in results.columns:
        results["סכום קנייה ($)"] = pd.to_numeric(results["Buy Amount ($)"], errors="coerce")
    else:
        results["סכום קנייה ($)"] = np.nan

results["מניות לקנייה"] = np.floor(
    np.where(
        (results["Unit_Price"] > 0) & (results["סכום קנייה ($)"].fillna(0) > 0),
        results["סכום קנייה ($)"].fillna(0) / results["Unit_Price"],
        0,
    )
).astype(int)
results["עודף ($)"] = np.round(
    np.where(
        pd.notna(results["סכום קנייה ($)"]) & (results["מניות לקנייה"] > 0),
        results["סכום קנייה ($)"] - results["מניות לקנייה"] * results["Unit_Price"],
        0.0,
    ),
    2,
)

# === Global budget cap enforcement (scaling if needed) ===
total_budget_value = float(st.session_state.get("total_budget", CONFIG["BUDGET_TOTAL"]))
results["position_value"] = results["Unit_Price"].fillna(0) * results["מניות לקנייה"]
total_alloc = float(results["position_value"].sum())
if total_alloc > total_budget_value and total_alloc > 0:
    scale = total_budget_value / total_alloc
    scaled_shares = (results["מניות לקנייה"] * scale).apply(
        lambda x: max(int(round(x)), 0)
    )
    results["מניות לקנייה"] = scaled_shares
    results["position_value"] = (
        results["Unit_Price"].fillna(0) * results["מניות לקנייה"]
    )
    # Recompute leftover and purchase amount columns to reflect scaled allocation
    results["סכום קנייה ($)"] = results["position_value"].round(2)
    results["עודף ($)"] = 0.0  # leftover per row not tracked post-scale

    # KPI
    budget_used = float(
        results["מניות לקנייה"].to_numpy() @ results["Unit_Price"].fillna(0).to_numpy()
    )
    k0, k1, k2, k3 = st.columns(4)
    k0.metric("Universe size after history filtering", len(data_map))
    k1.metric("Results after filtering", len(results))
    total_budget_value = float(st.session_state.get("total_budget", CONFIG["BUDGET_TOTAL"]))
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
    
# Live-mode-only: offer saving this run as latest precomputed scan
# If a precomputed scan was loaded earlier, prefer it for rendering instead of re-running the pipeline.
# NOTE: Use the already-filtered results from session state (updated in skip_pipeline block above)
if st.session_state.get("precomputed_results") is not None and st.session_state.get("skip_pipeline", False):
    try:
        # Use the filtered results from session state (already limited to top 15 by skip_pipeline block)
        # The skip_pipeline block above already filtered and updated session state at line 2644
        results = st.session_state.get("precomputed_results").copy()
        
        # Rename columns to match live pipeline naming
        column_renames = {
            'Close': 'Price_Yahoo',
            'Technical_Score': 'Score_Tech',
            'Fundamental_Score': 'Fundamental_S',
            'Overall_Score': 'Score',
            'Fund_Sources_Count': 'fund_sources_used_v2',
        }
        results = results.rename(columns=column_renames)
        # Ensure both reliability aliases exist for UI fallbacks
        if 'Reliability_v2' not in results.columns and 'reliability_v2' in results.columns:
            results['Reliability_v2'] = results['reliability_v2']
        if 'reliability_pct' not in results.columns and 'Reliability_v2' in results.columns:
            results['reliability_pct'] = results['Reliability_v2']
        # Sector alias
        if 'Sector' not in results.columns and 'sector' in results.columns:
            results['Sector'] = results['sector']
    except Exception:
        # Fallback to session state if rename fails
        results = st.session_state.get("precomputed_results")
    data_map = {}
    phase_times = phase_times if 'phase_times' in locals() else {}
    logger.info(f"Rendering using precomputed scan with {len(results)} tickers (already filtered to top 15)")
    try:
        status_manager.update_detail("Precomputed scan loaded — using cached results")
        status_manager.set_progress(1.0)
    except Exception:
        pass
    st.info("⚡ Rendering using precomputed scan (no live pipeline run)")

# Note: Auto-save is now handled in the live pipeline section above
# No need for manual save button - results are saved automatically

# Mark pipeline completion at UI level
try:
    status_manager.advance("Recommendations & Allocation")
    status_manager.complete("✅ Pipeline complete")
except Exception:
    pass

# Close the skip_pipeline conditional block
# (The if not skip_pipeline block ends here)







st.subheader("🤖 Recommendations Now")
st.caption("These cards are buy recommendations only. This is not investment advice.")

# Sidebar filters
# Sidebar removed - all controls moved to top bar above

# Read session state values set by top control bar
universe_size = int(st.session_state.get("universe_size", 2000))
fast_mode = bool(st.session_state.get("fast_mode", False))
total_budget = float(st.session_state.get("total_budget", CONFIG["BUDGET_TOTAL"]))
min_position = float(st.session_state.get("min_position", max(50.0, round(total_budget * 0.10))))
max_position_pct = float(st.session_state.get("max_position_pct", CONFIG["MAX_POSITION_PCT"]))
alloc_style_idx = int(st.session_state.get("alloc_style_idx", 0))
enable_multi_source = bool(st.session_state.get("enable_multi_source", True))
enable_ml_boost = bool(st.session_state.get("enable_ml_boost", True))
ml_threshold = int(st.session_state.get("ml_threshold", 0))
enable_openai_targets = bool(st.session_state.get("enable_openai_targets", False))

# Legacy variables for backward compatibility
RELAXED_MODE = bool(st.session_state.get("RELAXED_MODE", False))
FAST_MODE = bool(st.session_state.get("FAST_MODE", False))
DEBUG_SKIP_PIPELINE = bool(st.session_state.get("DEBUG_SKIP_PIPELINE", False))
use_full_export = bool(st.session_state.get("use_full_export", False))

# Initialize view filters from session state or defaults
risk_filter = ["core", "speculative"]
quality_filter = ["high", "medium", "low"]
sector_filter = []

# Continue with main app flow
st.markdown("---")

# Initialize view parameters for read-only display
show_debug_attr = False
compact_mode = False

# Prepare recommendations view: sort by FinalScore_20d (primary) and show all
initial_rec_count = len(results)
rec_df = results.copy()
# Prefer FinalScore_20d for ranking; fallback if missing
score_col = "FinalScore_20d" if "FinalScore_20d" in rec_df.columns else (
    "Score" if "Score" in rec_df.columns else (
        "overall_score_20d" if "overall_score_20d" in rec_df.columns else (
            "overall_score" if "overall_score" in rec_df.columns else None
        )
    )
)
if score_col:
    rec_df = (
        rec_df.assign(_score_numeric=pd.to_numeric(rec_df[score_col], errors="coerce"))
        .sort_values(by=["_score_numeric"], ascending=[False])
        .drop(columns=["_score_numeric"])
        .copy()
    )
else:
    logger.warning("[DISPLAY] No score column found for ranking; preserving source order")
    # No TOPN cap — show all viable candidates

# Apply display threshold: show candidates with FinalScore_20d (or fallback score) >= 30, regardless of allocation
if not rec_df.empty:
    threshold_col = None
    for c in ["FinalScore_20d", "Score", "overall_score_20d", "overall_score"]:
        if c in rec_df.columns:
            threshold_col = c
            break
    if threshold_col is not None:
        rec_df = rec_df[pd.to_numeric(rec_df[threshold_col], errors="coerce") >= 30].copy()

if not rec_df.empty:
    # Apply risk filter (inclusive defaults)
    if risk_filter and "Risk_Level" in rec_df.columns:
        rec_df = rec_df[rec_df["Risk_Level"].isin(risk_filter)]

    # Apply quality filter
    if quality_filter and "Data_Quality" in rec_df.columns:
        rec_df = rec_df[rec_df["Data_Quality"].isin(quality_filter)]

    # Apply sector filter
    if sector_filter and "Sector" in rec_df.columns:
        rec_df = rec_df[rec_df["Sector"].isin(sector_filter)]

    # Drop any explicit REJECT decisions if present
    for _col in ["Decision", "Recommendation", "Status"]:
        if _col in rec_df.columns:
            rec_df = rec_df[rec_df[_col].astype(str).str.upper() != "REJECT"]

    # Score-only mode: show technically valid actions regardless of allocation
    if "Action" in rec_df.columns:
        rec_df = rec_df[rec_df["Action"].astype(str).str.upper().isin(["BUY", "ACCUMULATE", "HOLD"])]

    # Removed UI market cap filter: the UI now mirrors the pipeline output without local cap constraints

    # Removed Coiled/Growth hard filters — handled as scoring bonuses in backend

logger.info(f"[FILTER] Final recommendations after all filters: {len(rec_df)} stocks (started with {initial_rec_count})")

# Hunter display logic: if no CORE, show top SPEC by VCP
core_count = 0
if "Risk_Level" in rec_df.columns:
    try:
        core_count = int((rec_df["Risk_Level"].astype(str) == "core").sum())
    except Exception:
        core_count = 0

    if core_count == 0 and not rec_df.empty:
        spec_df = rec_df[rec_df["Risk_Level"].astype(str) == "speculative"].copy() if "Risk_Level" in rec_df.columns else rec_df.copy()
        sort_col = "Volatility_Contraction_Score" if "Volatility_Contraction_Score" in spec_df.columns else ("FinalScore_20d" if "FinalScore_20d" in spec_df.columns else None)
        if sort_col:
            spec_df = (
                spec_df.assign(_sort_val=pd.to_numeric(spec_df[sort_col], errors="coerce"))
                .sort_values(by=["_sort_val"], ascending=[False])
                .drop(columns=["_sort_val"])
            )
        if spec_df.empty:
            st.warning("🔍 No high-conviction setups found. Check back after the next automated scan.")
        else:
            st.info(f"🟠 Showing {len(spec_df)} recommendations (no CORE found)")
            rec_df = spec_df
else:
    if rec_df.empty:
        st.warning("🔍 No high-conviction setups found. Check back after the next automated scan.")
    else:
        st.info(f"📊 **{len(rec_df)} מניות** עברו את כל המסננים (מתוך {initial_rec_count} שנבדקו)")

# KPI: Totals and pass-through
try:
    total_candidates = int(st.session_state.get("pre_scan_universe_count", 0))
    # Count stocks with FinalScore_20d > 10 from the full results (not only top-15)
    sc_full = None
    for c in ["FinalScore_20d", "conviction_v2_final", "overall_score_20d", "overall_score", "Score"]:
        if c in results.columns:
            sc_full = c
            break
    passed_count = 0
    if sc_full is not None and not results.empty:
        passed_count = int(pd.to_numeric(results[sc_full], errors="coerce").gt(10).sum())
    k1, k2 = st.columns(2)
    with k1:
        st.metric("Total Candidates", f"{total_candidates}")
    with k2:
        st.metric("Stocks Passed", f"{passed_count}")
except Exception:
    pass

if initial_rec_count > 0 and len(rec_df) < initial_rec_count:
    removed = initial_rec_count - len(rec_df)
    st.caption(f"🔍 {removed} מניות סוננו על ידי: Risk management, Buy amount allocation, Quality filters")

# --- DEBUG: Show top 5 with canonical 20d ML columns ---
if not rec_df.empty and "FinalScore_20d" in rec_df.columns:
    debug_cols = ["Ticker", "TechScore_20d", "ML_20d_Prob", "FinalScore_20d", "FinalScore"]
    available_debug_cols = [c for c in debug_cols if c in rec_df.columns]
    if available_debug_cols:
        st.info("🔍 Top 5 recommendations (20d ML scoring):")
        top5_debug = rec_df.head(5)[available_debug_cols].copy()
        st.dataframe(top5_debug, width='stretch', hide_index=True)

rec_df = rec_df.copy()


# Responsive recommendation grid + card styles (full-width cards, auto-fit columns)
# (CSS and markdown blocks should be inside st.markdown or string, not as stray lines)

# Deterministic ranking pre Core/Spec split
if "Score" in rec_df.columns and "Ticker" in rec_df.columns:
    rec_df = apply_deterministic_ranking(rec_df)
    # Maintain legacy Overall_Rank for compatibility, guard if Rank missing
    if "Overall_Rank" not in rec_df.columns:
        if "Rank" in rec_df.columns:
            rec_df["Overall_Rank"] = rec_df["Rank"]
        else:
            # Fallback: sequential rank if helper failed to produce Rank
            rec_df["Overall_Rank"] = np.arange(1, len(rec_df) + 1)

# --- Fallback Logic: if no stocks have positive allocation, show top technical candidates ---
if rec_df.empty:
    st.warning(
        "No stocks passed allocation filters (all buy amounts zero or blocked). Showing top technical candidates (fallback mode)."
    )
    fallback_n = min(10, len(results))
    rec_df = results.head(fallback_n).copy()
    rec_df["Fallback_Display"] = True
else:
    rec_df["Fallback_Display"] = False

# Provider usage tracking (aggregate from source lines)
# Build accurate provider usage tracker using session markers and per-row flags
providers_meta = {
    "Yahoo": {"env": None, "implemented": True, "label": "Yahoo"},
    "Alpha": {"env": "ALPHA_VANTAGE_API_KEY", "implemented": True, "label": "Alpha"},
    "Finnhub": {"env": "FINNHUB_API_KEY", "implemented": True, "label": "Finnhub"},
    "Tiingo": {"env": "TIINGO_API_KEY", "implemented": True, "label": "Tiingo"},
    "Polygon": {"env": "POLYGON_API_KEY", "implemented": True, "label": "Polygon"},
    "FMP": {"env": "FMP_API_KEY", "implemented": True, "label": "FMP"},
    "OpenAI": {"env": "OPENAI_API_KEY", "implemented": True, "label": "OpenAI"},
    # Mark optional providers as implemented by default; connectivity/key presence will reflect availability
    "SimFin": {"env": "SIMFIN_API_KEY", "implemented": True, "label": "SimFin"},
    "Marketstack": {"env": "MARKETSTACK_API_KEY", "implemented": True, "label": "Marketstack"},
    "EODHD": {"env": "EODHD_API_KEY", "implemented": True, "label": "EODHD"},
    "NasdaqDL": {"env": "NASDAQ_API_KEY", "implemented": True, "label": "NasdaqDL"},
}

# session-level usage markers updated by data fetch helpers (see _fetch_external_for and mark_provider_usage)
session_usage = st.session_state.get("provider_usage", {}) or {}

provider_usage = {}
for p, meta in providers_meta.items():
    key_present = False
    if meta["env"]:
        try:
            key_present = bool(_env(meta["env"]))
        except Exception:
            key_present = False
    else:
        # Providers that don't need a key (Yahoo)
        key_present = True

    implemented = bool(meta.get("implemented", True))

    # used flags: check session markers and per-row flags in results
    used_price = False
    used_fund = False
    used_ml = False

    # Session usage (set in fetch helpers)
    if p in session_usage:
        cats = session_usage.get(p) or set()
        used_price = used_price or ("price" in cats)
        used_fund = used_fund or ("fundamentals" in cats)
        used_ml = used_ml or ("ml" in cats)

    # Per-row indicators (fundamentals)
    try:
        if p == "Alpha" and "Fund_from_Alpha" in results.columns and results["Fund_from_Alpha"].any():
            used_fund = True
        if p == "Tiingo" and "Fund_from_Tiingo" in results.columns and results["Fund_from_Tiingo"].any():
            used_fund = True
        if p == "Finnhub" and "Fund_from_Finnhub" in results.columns and results["Fund_from_Finnhub"].any():
            used_fund = True
        if p == "SimFin" and "Fund_from_SimFin" in results.columns and results["Fund_from_SimFin"].any():
            used_fund = True
        if p == "EODHD" and "Fund_from_EODHD" in results.columns and results["Fund_from_EODHD"].any():
            used_fund = True
        if p == "Alpha" and ("Price_Alpha" in results.columns and results["Price_Alpha"].notna().any()):
            used_price = True
    except Exception:
        pass

    # Price sources from Source_List column
    try:
        if "Source_List" in results.columns:
            # join all source_list strings and search for provider label appearance
            all_sources = " ".join([str(x) for x in results["Source_List"].fillna("") if x])
            if meta["label"].lower() in all_sources.lower():
                used_price = True
    except Exception:
        pass

    # OpenAI/ML usage heuristic
    try:
        if p == "OpenAI":
            if st.session_state.get("enable_openai_targets", False):
                # if any ML probabilities exist, consider ML used
                if "ML_Probability" in results.columns and results["ML_Probability"].notna().any():
                    used_ml = True
                # also consider session_usage
                if "OpenAI" in session_usage and "ml" in (session_usage.get("OpenAI") or set()):
                    used_ml = True
    except Exception:
        pass

    provider_usage[p] = {
        "key_present": bool(key_present),
        "used_price": bool(used_price),
        "used_fundamentals": bool(used_fund),
        "used_ml": bool(used_ml),
        "implemented": bool(implemented),
    }

# Count used providers (any usage)
used_count = sum(1 for v in provider_usage.values() if v.get("used_price") or v.get("used_fundamentals") or v.get("used_ml"))

# Render data sources overview (single dynamic table, no HTML)
# Build comprehensive provider status map for overview table
openai_key = _env("OPENAI_API_KEY")
simfin_ok = bool(simfin_key)
marketstack_ok = bool(marketstack_key)
nasdaq_ok = bool(nasdaq_key)
eodhd_ok = bool(eodhd_key)

# Build provider status map from canonical list in SourcesOverview
status_lookup = {
    "Yahoo": True,
    "Alpha Vantage": bool(alpha_ok),
    "Finnhub": bool(finn_ok),
    "Polygon": bool(poly_ok),
    "Tiingo": bool(tiin_ok),
    "FMP": bool(fmp_ok),
    "OpenAI": bool(openai_key),
    "SimFin": bool(simfin_ok),
    "Marketstack": bool(marketstack_ok),
    "EODHD": bool(eodhd_ok),
    "Nasdaq": bool(nasdaq_ok),
}
provider_status_map = {name: {"ok": bool(status_lookup.get(name, False))} for name in SourcesOverview.PROVIDERS.keys()}

render_data_sources_overview(
    provider_status=provider_status_map,
    provider_usage=provider_usage,
    results=results
)

# Calculate target prices and dates WITH OPTIONAL OPENAI ENHANCEMENT
from datetime import datetime, timedelta


@st.cache_data(ttl=3600)
def get_openai_target_prediction(
    ticker: str, current_price: float, fundamentals: dict, technicals: dict
) -> Optional[Tuple[float, int]]:
    """
    Use OpenAI to predict realistic target price AND holding period based on fundamentals and technicals.
    Returns (target_price, days_to_target) tuple or None if API unavailable or request fails.
    """
    if not OPENAI_AVAILABLE:
        return None

    openai_key = _env("OPENAI_API_KEY")
    if not openai_key:
        return None

    try:
        client = OpenAI(api_key=openai_key)

        # Build context from fundamentals
        fund_str = ", ".join(
            [
                f"{k}: {v}"
                for k, v in fundamentals.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
        )
        tech_str = ", ".join(
            [
                f"{k}: {v:.2f}"
                for k, v in technicals.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            ]
        )

        prompt = (
            f"You are a financial analyst. Based on the following data for {ticker}:\n"
            f"Current Price: USD {current_price:.2f}\n"
            f"Fundamentals: {fund_str}\n"
            f"Technical Indicators: {tech_str}\n"
            "Provide TWO predictions as a JSON object:\n"
            "1. Target Price: realistic price target considering growth trends, valuation, momentum, and risk/reward\n"
            "2. Days to Target: estimated holding period in days to reach this target (typically 7-180 days based on momentum and catalysts)\n"
            "Return ONLY a JSON object with this exact format:\n"
            '{"target_price": <number>, "days_to_target": <integer>}\n'
            "JSON:"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )

        # Extract JSON from response
        answer = response.choices[0].message.content.strip()
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r"\{[^}]+\}", answer)
        if json_match:
            data = json.loads(json_match.group(0))
            target = float(data.get("target_price", 0))
            days = int(data.get("days_to_target", 20))
            # Sanity checks
            if (
                current_price * 0.5 <= target <= current_price * 3.0
                and 7 <= days <= 365
            ):
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
    # Use canonical ML probability when available
    ml_prob = row.get("ML_20d_Prob", row.get("ml_probability", 0.5))

    if np.isfinite(current_price) and current_price > 0:
        # Entry price: current - 0.5*ATR (wait for slight pullback)
        if np.isfinite(atr):
            entry_price = current_price - (0.5 * atr)
        else:
            entry_price = current_price * 0.98  # 2% below if no ATR

        # Calculate volatility factor for date variability
        atr_pct = (
            (atr / current_price) if (np.isfinite(atr) and current_price > 0) else 0.02
        )
        volatility_factor = np.clip(atr_pct / 0.03, 0.5, 2.5)  # 0.5x to 2.5x multiplier

        # Sector-based offset (defensive sectors slower, growth sectors faster)
        sector_offsets = {
            "Utilities": 1.3,
            "Consumer Defensive": 1.2,
            "Real Estate": 1.15,
            "Financials": 1.1,
            "Healthcare": 1.0,
            "Industrials": 0.95,
            "Energy": 0.9,
            "Consumer Cyclical": 0.85,
            "Technology": 0.75,
            "Communication Services": 0.8,
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
                    base_days *= 1.3  # Overbought, slower target

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
        # Skip OpenAI calls in precomputed mode for faster loading
        skip_openai = st.session_state.get("skip_pipeline", False)
        if not skip_openai and st.session_state.get("enable_openai_targets", False):
            fundamentals = {
                "PE": row.get("PERatio", np.nan),
                "PB": row.get("PBRatio", np.nan),
                "ROE": row.get("ROE", np.nan),
                "Margin": row.get("ProfitMargin", np.nan),
                "RevenueGrowth": row.get("RevenueGrowthYoY", np.nan),
            }
            technicals = {
                "RSI": rsi,
                "Momentum_63d": momentum,
                "RewardRisk": rr,
                "ATR": atr,
            }
            try:
                ai_result = get_openai_target_prediction(
                    ticker, current_price, fundamentals, technicals
                )
            except Exception as e:
                logger.warning(f"OpenAI call failed for {ticker}: {e}")
                ai_result = None

        if ai_result is not None:
            # Use AI prediction for both target price AND timing
            target_price, days = ai_result
            target_source = "AI"
        elif np.isfinite(atr) and np.isfinite(rr):
            # Fallback to technical calculation: entry + (RR * ATR)
            base_target_pct = rr * (atr / current_price) if current_price > 0 else 0.10
            
            # REGIME-AWARE ADJUSTMENT
            reliability = row.get("Reliability_v2", row.get("reliability_pct", 50.0))
            risk_meter = row.get("risk_meter_v2", row.get("RiskMeter", 50.0))
            regime_data = st.session_state.get('market_regime', {"regime": "neutral", "confidence": 50})
            
            adjusted_target_pct, adjustment_explanation = adjust_target_for_regime(
                base_target_pct, 
                reliability, 
                risk_meter, 
                regime_data
            )
            
            target_price = entry_price * (1 + adjusted_target_pct)
            target_source = "AI"  # Mark as AI to show regime-aware calculation
            
            # Debug logging
            if CONFIG.get("DEBUG_MODE") and adjustment_explanation != "no adjustments":
                logger.debug(f"{ticker}: Target adjusted from {base_target_pct*100:.1f}% to {adjusted_target_pct*100:.1f}% ({adjustment_explanation})")
            
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


"""Respect pipeline targets: only compute if missing."""
# Determine if targets already exist and are populated
target_cols = ["Entry_Price", "Target_Price", "Stop_Loss"]
has_target_cols = all(c in rec_df.columns for c in target_cols)
has_any_targets = False
if has_target_cols:
    try:
        vals = pd.to_numeric(rec_df["Target_Price"], errors="coerce")
        has_any_targets = bool(np.isfinite(vals).any())
    except Exception:
        has_any_targets = False

if not has_any_targets:
    with st.spinner(f"🎯 Calculating targets for {len(rec_df)} stocks..."):
        if not rec_df.empty:
            entry_prices = []
            target_prices = []
            target_dates = []
            target_sources = []

            for idx, row in rec_df.iterrows():
                entry, target, date, source = calculate_targets(row)
                entry_prices.append(entry)
                target_prices.append(target)
                target_dates.append(date)
                target_sources.append(source)

            rec_df["Entry_Price"] = entry_prices
            rec_df["Target_Price"] = target_prices
            rec_df["Target_Date"] = target_dates
            rec_df["Target_Source"] = target_sources
    logger.info(f"Calculated targets for {len(rec_df)} stocks (UI fallback)")
else:
    logger.info(f"Using pipeline targets for {len(rec_df)} stocks")


def calculate_rr(
    entry_price: float,
    target_price: float,
    atr_value: float,
    fallback_price: float = None,
) -> float:
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

        atr = (
            atr_value
            if isinstance(atr_value, (int, float)) and np.isfinite(atr_value)
            else np.nan
        )
        # If atr not provided, try common fallback field name from row via fallback_price
        if (
            not np.isfinite(atr)
            and isinstance(fallback_price, (int, float))
            and np.isfinite(fallback_price)
        ):
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


# Recalculate Reward/Risk only if pipeline RR not present; otherwise respect pipeline RR
if not rec_df.empty:
    rr_present = False
    try:
        if "RR" in rec_df.columns:
            rr_present = pd.to_numeric(rec_df["RR"], errors="coerce").notna().any()
        elif "RR_Ratio" in rec_df.columns or "RewardRisk" in rec_df.columns:
            base_rr = rec_df.get("RR_Ratio", rec_df.get("RewardRisk", pd.Series([np.nan] * len(rec_df), index=rec_df.index)))
            rr_present = pd.to_numeric(base_rr, errors="coerce").notna().any()
    except Exception:
        rr_present = False

if not rec_df.empty and not rr_present:
    # ATR value may be stored under 'ATR' or 'ATR14' or 'ATR_Price'
    def _compute_rr_row(r):
        entry = r.get("Entry_Price", r.get("Unit_Price", r.get("Price_Yahoo", np.nan)))
        target = r.get("Target_Price", np.nan)
        atr = r.get("ATR", r.get("ATR14", r.get("ATR14", np.nan)))
        # fallback ATR in price terms (ATR_Price)
        atr_price = r.get("ATR_Price", np.nan)
        rr_val = calculate_rr(entry, target, atr, fallback_price=atr_price)
        return rr_val

    rec_df["RewardRisk"] = rec_df.apply(
        lambda r: (
            round(_compute_rr_row(r), 2) if np.isfinite(_compute_rr_row(r)) else np.nan
        ),
        axis=1,
    )
    # Also update RR_Ratio alias used in classification
    rec_df["RR_Ratio"] = rec_df["RewardRisk"]

    # Propagate recalculated RewardRisk back into the main `results` frame so
    # downstream fields (rr alias, rr_score_v2, conviction) use the updated values.
    try:
        rr_map = rec_df.set_index("Ticker")["RewardRisk"].to_dict()
        results["RewardRisk"] = (
            results["Ticker"].map(rr_map).fillna(results.get("RewardRisk", np.nan))
        )
        results["RR_Ratio"] = results["RewardRisk"]
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

        # Recompute rr_band after RR update using evaluate_rr_unified
        from core.scoring_engine import evaluate_rr_unified

        def _rr_eval_local(row_rr):
            try:
                score, ratio_adj, band = evaluate_rr_unified(float(row_rr))
                return band
            except Exception:
                return "N/A"

        results["rr_band"] = results["rr"].apply(_rr_eval_local)

        # Sync updated RR fields back to rec_df (recommendation dataframe)
        if not rec_df.empty:
            for tkr_idx in rec_df.index:
                tkr_val = rec_df.at[tkr_idx, "Ticker"]
                if tkr_val in rr_map:
                    rec_df.at[tkr_idx, "rr"] = rr_map[tkr_val]
                    rec_df.at[tkr_idx, "RewardRisk"] = rr_map[tkr_val]
                    rec_df.at[tkr_idx, "RR_Ratio"] = rr_map[tkr_val]
            # Recompute rr_band for rec_df
            rec_df["rr_band"] = rec_df["rr"].apply(_rr_eval_local)
    except Exception as e:
        logger.warning(f"RR sync failed: {e}")

# Add new export fields for 2025 improvements
if not rec_df.empty:
    # Market regime info
    regime_data = st.session_state.get('market_regime', {"regime": "neutral", "confidence": 50})
    rec_df["Market_Regime"] = regime_data.get("regime", "neutral")
    rec_df["Regime_Confidence"] = regime_data.get("confidence", 50)
    
    # Reliability band (High/Medium/Low based on reliability score)
    def _get_reliability_band(reliability_val):
        if pd.notna(reliability_val) and isinstance(reliability_val, (int, float)):
            if reliability_val >= 75:
                return "High"
            elif reliability_val >= 40:
                return "Medium"
            else:
                return "Low"
        return "Unknown"
    
    # Get reliability values, defaulting to 50 if columns don't exist
    reliability_series = rec_df.get("Reliability_v2", rec_df.get("reliability_pct", pd.Series([50] * len(rec_df), index=rec_df.index)))
    rec_df["Reliability_Band"] = reliability_series.apply(_get_reliability_band)
    
    # Reliability components summary
    def _get_reliability_components(row):
        fund_rel = row.get("Fundamental_Reliability_v2", 0)
        price_rel = row.get("Price_Reliability_v2", 0)
        fund_sources = row.get("fund_sources_used_v2", 0)
        price_sources = row.get("price_sources_used_v2", 0)
        return f"F:{fund_rel:.0f}%(n={fund_sources}),P:{price_rel:.0f}%(n={price_sources})"
    
    rec_df["Reliability_Components"] = rec_df.apply(_get_reliability_components, axis=1)
    
    # Risk band (based on risk_meter_v2)
    rec_df["Risk_Band"] = rec_df.get("risk_band", "Unknown")
    
    # Fundamental coverage percentage
    if "Fund_Coverage_Pct" in rec_df.columns:
        rec_df["Fund_Coverage_Pct"] = rec_df["Fund_Coverage_Pct"].fillna(0)
    else:
        rec_df["Fund_Coverage_Pct"] = 0
    
    # Volatility penalty (from reliability calculation)
    atr_price = rec_df.get("ATR_Price")
    if atr_price is not None:
        rec_df["Volatility_Penalty"] = atr_price.apply(
            lambda x: "High" if x > 0.08 else ("Moderate" if x > 0.04 else "Low") if pd.notna(x) else "Unknown"
        )
    else:
        rec_df["Volatility_Penalty"] = "Unknown"
    
    # Safety caps applied (if overall score was capped)
    rec_df["Safety_Caps_Applied"] = "No"  # Placeholder - would need to track during scoring
    
    # Debug logging for sample stocks
    if CONFIG.get("DEBUG_MODE") and len(rec_df) > 0:
        sample_tickers = rec_df["Ticker"].head(3).tolist()
        for ticker in sample_tickers:
            row = rec_df[rec_df["Ticker"] == ticker].iloc[0]
            def _fmt_num(val, fmt):
                try:
                    return format(float(val), fmt) if val is not None and str(val) not in ("N/A", "nan") else str(val)
                except Exception:
                    return str(val)
            logger.info(
                f"DEBUG: {ticker} Breakdown:\n"
                f"  Overall Score: {row.get('overall_score_pretty', row.get('Score', 'N/A'))}\n"
                f"  Fundamentals: {_fmt_num(row.get('Fundamental_S', 'N/A'), '.0f')} (coverage: {_fmt_num(row.get('Fund_Coverage_Pct', 0), '.1f')}%)\n"
                f"  Technical: {_fmt_num(row.get('Technical_S', 'N/A'), '.0f')}\n"
                f"  RR: {_fmt_num(row.get('rr_score_v2', 'N/A'), '.0f')} (ratio: {_fmt_num(row.get('rr', 'N/A'), '.2f')})\n"
                f"  Reliability: {_fmt_num(row.get('Reliability_v2', 'N/A'), '.0f')} ({row.get('Reliability_Band', 'N/A')})\n"
                f"  Risk: {_fmt_num(row.get('risk_meter_v2', 'N/A'), '.0f')} ({row.get('risk_band', 'N/A')})\n"
                f"  Classification: {row.get('Risk_Level', 'N/A')}\n"
                f"  Market Regime: {regime_data.get('regime', 'N/A')} ({regime_data.get('confidence', 0)}%)"
            )

# CSS now loaded from design_system.py - no need for separate CARD_CSS


def format_rel(val) -> str:
    if not isinstance(val, (int, float)) or not np.isfinite(val):
        return "Not available"
    tier = "High" if val >= 0.75 else ("Medium" if val >= 0.4 else "Low")
    color = (
        "#16a34a" if tier == "High" else ("#f59e0b" if tier == "Medium" else "#dc2626")
    )
    return f"<span style='color:{color};font-weight:600'>{val:.2f} ({tier})</span>"


if rec_df.empty:
    st.info("No stocks currently pass the threshold with a positive buy amount.")
else:
    # Split into Core and Speculative
    if "Risk_Level" in rec_df.columns:
        levels = rec_df["Risk_Level"].astype(str).str.lower()
        core_df = rec_df[levels == "core"].copy()
        spec_df = rec_df[levels == "speculative"].copy()
    else:
        core_df = rec_df.copy()
        spec_df = pd.DataFrame()

    # Summary info
    total_candidates = len(core_df) + len(spec_df)
    funded_count = 0
    try:
        if 'buy_amount_v2' in rec_df.columns:
            funded_count = int((rec_df['buy_amount_v2'].fillna(0) > 0).sum())
        elif 'סכום קנייה ($)' in rec_df.columns:
            funded_count = int((rec_df['סכום קנייה ($)'].fillna(0) > 0).sum())
    except Exception:
        funded_count = total_candidates
    
    if funded_count and funded_count != total_candidates:
        st.info(
            f"📊 Showing {funded_count} funded positions (out of {total_candidates} candidates) — {len(core_df)} Core, {len(spec_df)} Speculative"
        )
    else:
        st.info(
            f"📊 Showing {total_candidates} stocks after filters ({len(core_df)} Core, {len(spec_df)} Speculative)"
        )

    # Legend for ML badge thresholds
    st.caption("ML badge legend: 🟢 >60% · 🟡 40–60% · 🔴 <40%")

    # Determine score label based on schema
    score_label = "FinalScore_20d" if "FinalScore_20d" in rec_df.columns else "Score"

    # Small helpers for compact card rendering (headline vs details)
    def _to_float(val) -> float:
        try:
            # Coerce common non-numeric placeholders safely
            if val in (None, "", "N/A", "nan"):
                return np.nan
            return float(val)
        except Exception:
            return np.nan

    def _ml_badge(p: float) -> str:
        try:
            pv = _to_float(p)
            if not (isinstance(pv, float) and np.isfinite(pv)):
                return "⚪ N/A"
            if pv > 0.60:
                return f"🟢 {pv*100:.0f}%"
            if pv >= 0.40:
                return f"🟡 {pv*100:.0f}%"
            return f"🔴 {pv*100:.0f}%"
        except Exception:
            return "⚪ N/A"

    def _risk_class(row: pd.Series) -> str:
        rc = row.get("RiskClass")
        if isinstance(rc, str) and rc:
            return rc
        # Fallback to legacy Risk_Level
        rl = str(row.get("Risk_Level", "speculative")).lower()
        return "CORE" if rl == "core" else ("SPEC" if rl else "SPEC")

    def _headline_story(row: pd.Series) -> str:
        # Headline fields: FinalScore_20d, ML_20d_Prob, RiskClass, RR, ReliabilityScore
        fund = _to_float(row.get("FundamentalScore", row.get("Fundamental_S", np.nan)))
        mom = _to_float(row.get("MomentumScore", row.get("TechScore_20d", np.nan)))
        rr = _to_float(row.get("RR", row.get("RR_Ratio", row.get("RewardRisk", np.nan))))
        rel = _to_float(row.get("ReliabilityScore", row.get("Reliability_Score", row.get("Reliability_v2", np.nan))))

        parts = []
        # Quality / fundamentals
        if isinstance(fund, float) and np.isfinite(fund):
            if fund >= 70:
                parts.append("Quality business")
            elif fund >= 50:
                parts.append("Decent fundamentals")
            else:
                parts.append("Weak fundamentals")
        # Momentum
        if isinstance(mom, float) and np.isfinite(mom):
            if mom >= 70:
                parts.append("strong momentum")
            elif mom >= 50:
                parts.append("moderate momentum")
            else:
                parts.append("weak momentum")
        # RR
        if isinstance(rr, float) and np.isfinite(rr):
            if rr >= 2.5:
                parts.append("excellent RR")
            elif rr >= 1.5:
                parts.append("good RR")
            else:
                parts.append("poor RR")
        # Reliability
        if isinstance(rel, float) and np.isfinite(rel):
            # Normalize legacy 0-1
            rel_val = rel
            if rel_val <= 1.0:
                rel_val *= 100.0
            if rel_val >= 75:
                parts.append("high data reliability")
            elif rel_val >= 40:
                parts.append("medium reliability")
            else:
                parts.append("low reliability")
        return ", ".join(parts[:4])

    def _fmt_num(val, fmt, na="N/A"):
        try:
            v = _to_float(val)
            return format(v, fmt) if isinstance(v, float) and np.isfinite(v) else na
        except Exception:
            return na

    # UI displays exactly what the pipeline provided (no extra slicing)
    st.write(f"Showing {len(rec_df)} recommendations")

    # Core recommendations
    if not core_df.empty:
        st.markdown("### 🛡️ Core Stocks — Lower Relative Risk")
        st.caption(f"Showing {len(core_df)} recommendations")
        
        @st.cache_data(ttl=3600)
        def _fallback_sector_yf(ticker: str) -> str:
            try:
                import yfinance as yf
                info = yf.Ticker(ticker).info
                sec = info.get('sector') or info.get('industry')
                return sec or 'Unknown'
            except Exception:
                return 'Unknown'

        for _, r in core_df.iterrows():
            ticker = r.get("Ticker", "N/A")
            sector = r.get("Sector", r.get("sector", "Unknown"))
            company = r.get("shortName", r.get("Company", r.get("Name", "")))
            if sector in (None, "", "Unknown") and isinstance(ticker, str) and ticker and ticker != "N/A":
                sector = _fallback_sector_yf(ticker)

            final_score = r.get("FinalScore_20d", r.get("Score", np.nan))
            ml_prob = r.get("ML_20d_Prob", r.get("ML_Probability", np.nan))
            rr = r.get("RR", r.get("RR_Ratio", r.get("RewardRisk", np.nan)))
            rel = r.get("ReliabilityScore", r.get("Reliability_Score", r.get("Reliability_v2", np.nan)))
            risk_c = _risk_class(r)

            # Headline compact view
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
                with c1:
                    title = ticker if not company else f"{ticker} · {company}"
                    st.subheader(title)
                    st.caption(f"Sector: {sector}")
                with c2:
                    st.metric(score_label, _fmt_num(final_score, '.0f'))
                with c3:
                    st.metric("Risk", risk_c)
                with c4:
                    st.metric("ML", _ml_badge(ml_prob))
                with c5:
                    rr_val = _to_float(rr)
                    rr_fmt = f"{rr_val:.2f}x" if isinstance(rr_val, float) and np.isfinite(rr_val) else "N/A"
                    st.metric("R/R", rr_fmt)

                # Short storyline
                storyline = _headline_story(r)
                if storyline:
                    st.caption(storyline)

                # Details expander: deep-dive fields
                with st.expander("Details", expanded=False):
                    # Technical indicators
                    t1, t2, t3, t4 = st.columns(4)
                    with t1:
                        st.text("RSI")
                        st.code(_fmt_num(r.get('RSI', np.nan), '.1f'))
                    with t2:
                        atrv = r.get('ATR_Price', r.get('ATR_Pct', np.nan))
                        st.text("ATR/Price")
                        st.code(_fmt_num(atrv, '.3f'))
                    with t3:
                        st.text("Momentum (Tech)")
                        momv = r.get('MomentumScore', r.get('TechScore_20d', np.nan))
                        st.code(_fmt_num(momv, '.0f'))
                    with t4:
                        st.text("ML Prob")
                        st.code(_fmt_num(ml_prob, '.3f'))

                    # Fundamentals breakdown
                    f1, f2, f3, f4 = st.columns(4)
                    with f1:
                        st.text("Fundamental")
                        fsv = r.get('FundamentalScore', r.get('Fundamental_S', np.nan))
                        st.code(_fmt_num(fsv, '.0f'))
                    with f2:
                        st.text("Quality/Growth")
                        st.code(f"{r.get('Quality_Score_F', np.nan)} / {r.get('Growth_Score_F', np.nan)}")
                    with f3:
                        st.text("Valuation")
                        st.code(f"{r.get('Valuation_Score_F', np.nan)}")
                    with f4:
                        st.text("Leverage (D/E)")
                        st.code(f"{r.get('DE_f', r.get('debt_to_equity', np.nan))}")

                    # Reliability breakdown
                    rel1, rel2, rel3, rel4 = st.columns(4)
                    with rel1:
                        st.text("Reliability")
                        st.code(_fmt_num(rel, '.0f'))
                    with rel2:
                        st.text("Fund sources")
                        st.code(f"{r.get('Fundamental_Sources_Count', r.get('fund_sources_used_v2', np.nan))}")
                    with rel3:
                        st.text("Price sources")
                        st.code(f"{r.get('Price_Sources_Count', r.get('price_sources_used_v2', np.nan))}")
                    with rel4:
                        st.text("Price STD")
                        st.code(f"{r.get('Price_STD', r.get('price_std', np.nan))}")

    # Speculative recommendations
    if not spec_df.empty:
        st.markdown("### ⚡ Speculative Stocks — High Upside, High Risk")
        st.caption(f"Showing {len(spec_df)} recommendations")
        st.warning("🔔 Warning: These stocks are classified as speculative due to partial data or elevated risk factors. Suitable for experienced investors only.")
        
        @st.cache_data(ttl=3600)
        def _fallback_sector_yf_spec(ticker: str) -> str:
            try:
                import yfinance as yf
                info = yf.Ticker(ticker).info
                sec = info.get('sector') or info.get('industry')
                return sec or 'Unknown'
            except Exception:
                return 'Unknown'

        for _, r in spec_df.iterrows():
            ticker = r.get("Ticker", "N/A")
            sector = r.get("Sector", r.get("sector", "Unknown"))
            company = r.get("shortName", r.get("Company", r.get("Name", "")))
            if sector in (None, "", "Unknown") and isinstance(ticker, str) and ticker and ticker != "N/A":
                sector = _fallback_sector_yf_spec(ticker)

            final_score = r.get("FinalScore_20d", r.get("Score", np.nan))
            ml_prob = r.get("ML_20d_Prob", r.get("ML_Probability", np.nan))
            rr = r.get("RR", r.get("RR_Ratio", r.get("RewardRisk", np.nan)))
            rel = r.get("ReliabilityScore", r.get("Reliability_Score", r.get("Reliability_v2", np.nan)))
            risk_c = _risk_class(r)

            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
                with c1:
                    title = ticker if not company else f"{ticker} · {company}"
                    st.subheader(title)
                    st.caption(f"Sector: {sector}")
                with c2:
                    st.metric(score_label, _fmt_num(final_score, '.0f'))
                with c3:
                    st.metric("Risk", risk_c)
                with c4:
                    st.metric("ML", _ml_badge(ml_prob))
                with c5:
                    rr_val = _to_float(rr)
                    rr_fmt = f"{rr_val:.2f}x" if isinstance(rr_val, float) and np.isfinite(rr_val) else "N/A"
                    st.metric("R/R", rr_fmt)

                storyline = _headline_story(r)
                if storyline:
                    st.caption(storyline)

                with st.expander("Details", expanded=False):
                    t1, t2, t3, t4 = st.columns(4)
                    with t1:
                        st.text("RSI")
                        st.code(_fmt_num(r.get('RSI', np.nan), '.1f'))
                    with t2:
                        atrv = r.get('ATR_Price', r.get('ATR_Pct', np.nan))
                        st.text("ATR/Price")
                        st.code(_fmt_num(atrv, '.3f'))
                    with t3:
                        st.text("Momentum (Tech)")
                        momv = r.get('MomentumScore', r.get('TechScore_20d', np.nan))
                        st.code(_fmt_num(momv, '.0f'))
                    with t4:
                        st.text("ML Prob")
                        st.code(_fmt_num(ml_prob, '.3f'))

                    f1, f2, f3, f4 = st.columns(4)
                    with f1:
                        st.text("Fundamental")
                        fsv = r.get('FundamentalScore', r.get('Fundamental_S', np.nan))
                        st.code(_fmt_num(fsv, '.0f'))
                    with f2:
                        st.text("Quality/Growth")
                        st.code(f"{r.get('Quality_Score_F', np.nan)} / {r.get('Growth_Score_F', np.nan)}")
                    with f3:
                        st.text("Valuation")
                        st.code(f"{r.get('Valuation_Score_F', np.nan)}")
                    with f4:
                        st.text("Leverage (D/E)")
                        st.code(f"{r.get('DE_f', r.get('debt_to_equity', np.nan))}")

                    rel1, rel2, rel3, rel4 = st.columns(4)
                    with rel1:
                        st.text("Reliability")
                        st.code(_fmt_num(rel, '.0f'))
                    with rel2:
                        st.text("Fund sources")
                        st.code(f"{r.get('Fundamental_Sources_Count', r.get('fund_sources_used_v2', np.nan))}")
                    with rel3:
                        st.text("Price sources")
                        st.code(f"{r.get('Price_Sources_Count', r.get('price_sources_used_v2', np.nan))}")
                    with rel4:
                        st.text("Price STD")
                        st.code(f"{r.get('Price_STD', r.get('price_std', np.nan))}")

    # Export section (single, unified)
            mean = r.get("מחיר ממוצע", np.nan)
            std = r.get("סטיית תקן", np.nan)
            hist_std = r.get(
                "Historical_StdDev", np.nan
            )  # NEW: Use historical price std dev
            # Robust fallback if Price_Yahoo is missing: try Close, then Unit_Price
            base_price = r.get("Price_Yahoo", r.get("Close", r.get("Unit_Price", np.nan)))
            show_mean = mean if np.isfinite(mean) else base_price
            # Prefer Historical_StdDev if available, fallback to old std
            show_std = (
                f"${hist_std:.2f}"
                if np.isfinite(hist_std)
                else (f"${std:.2f}" if np.isfinite(std) else "N/A")
            )
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

            # ML confidence badge: High>=70%, Med 50-70%, Low<50%
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
                def _fmt_pct(val, fmt):
                    try:
                        return format(float(val), fmt) if val is not None and str(val) not in ("N/A", "nan") else str(val)
                    except Exception:
                        return str(val)
                ml_badge_html = f"""<span style='display:inline-block;padding:3px 8px;border-radius:4px;background:{ml_badge_color};color:white;font-weight:bold;font-size:0.85em;margin-left:8px'>ML: {ml_badge_text} ({_fmt_pct(ml_prob*100, '.0f')}%)</span>"""
                ml_status_esc = f"{ml_badge_text} ({_fmt_pct(ml_prob*100, '.0f')}%)"
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
            confidence_badge = (
                f"{confidence_level.upper()}" if confidence_level else "MEDIUM"
            )

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
            qual_fmt = (
                f"{qual_score_f:.0f} ({qual_label})"
                if np.isfinite(qual_score_f)
                else "N/A"
            )
            growth_fmt = (
                f"{growth_score_f:.0f} ({growth_label})"
                if np.isfinite(growth_score_f)
                else "N/A"
            )
            val_fmt = (
                f"{val_score_f:.0f} ({val_label})"
                if np.isfinite(val_score_f)
                else "N/A"
            )
            lev_fmt = (
                f"{lev_score_f:.0f} ({lev_label})"
                if np.isfinite(lev_score_f)
                else "N/A"
            )

            # Color coding for labels
            def label_color(label, good_vals):
                if label in good_vals:
                    return "#16a34a"  # green
                elif label in ["Medium", "Fair", "Moderate"]:
                    return "#f59e0b"  # orange
                else:
                    return "#dc2626"  # red

            qual_color = label_color(qual_label, ["High"])
            growth_color = label_color(growth_label, ["Fast", "Moderate"])
            val_color = label_color(val_label, ["Cheap", "Fair"])
            lev_color = label_color(lev_label, ["Low", "Medium"])

            # Detect missing fundamental data
            missing_fundamental_count = 0
            fundamental_fields = ["ROE_f", "ROIC_f", "DE_f", "PE_f", "GM_f"]
            for field in fundamental_fields:
                val = r.get(field, np.nan)
                if not np.isfinite(val):
                    missing_fundamental_count += 1

            # Create partial data badge if applicable
            data_quality_badge = ""
            if missing_fundamental_count >= 4:
                data_quality_badge = (
                    "<span class='modern-badge badge-missing'>⚠️ Missing Data</span>"
                )
            elif missing_fundamental_count >= 2:
                data_quality_badge = (
                    "<span class='modern-badge badge-partial'>📊 Partial Data</span>"
                )

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

            entry_price_fmt = (
                f"${entry_price:.2f}" if np.isfinite(entry_price) else "N/A"
            )
            target_price_fmt = (
                f"${target_price:.2f}" if np.isfinite(target_price) else "N/A"
            )

            # Add badge for AI-enhanced targets
            target_badge = ""
            if target_source == "AI":
                target_badge = " <span style='background:#10b981;color:white;padding:2px 6px;border-radius:4px;font-size:0.75em;font-weight:bold'>🤖 AI</span>"
            elif target_source == "Technical":
                target_badge = " <span style='background:#6366f1;color:white;padding:2px 6px;border-radius:4px;font-size:0.75em'>📊 Tech</span>"
            elif target_source == "Default":
                target_badge = " <span style='background:#6b7280;color:white;padding:2px 6px;border-radius:4px;font-size:0.75em'>📐 Est</span>"

            # Calculate potential gain %
            if (
                np.isfinite(entry_price)
                and np.isfinite(target_price)
                and entry_price > 0
            ):
                potential_gain_pct = ((target_price - entry_price) / entry_price) * 100
                gain_fmt = f"+{potential_gain_pct:.1f}%"
                gain_color = "#16a34a"
            else:
                gain_fmt = "N/A"
                gain_color = "#6b7280"
                target_badge = ""  # No badge if no valid target

            # Inject CSS for iframe isolation
            price_rel_fmt = format_rel(r.get("Price_Reliability", np.nan))
            fund_rel_fmt = format_rel(r.get("Fundamental_Reliability", np.nan))
            rel_score_fmt = format_rel(r.get("Reliability_Score", np.nan))

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

    # Export section (single, unified)
show_order = [
    "Ticker",
    "TechScore_20d",
    "ML_20d_Prob",
    "ML_Features_Fallback",
    "FinalScore_20d",
    "FinalScore",
    "overall_score_20d",  # 20-day conviction score
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
    "Blended_RS_Value",
    "Tightness_Ratio",
    "Growth_Boost",
    "MarketCap_B",
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
    # NEW FIELDS (2025 improvements)
    "Market_Regime",
    "Regime_Confidence",
    "Reliability_Band",
    "Reliability_Components",
    "Risk_Band",
    "Fund_Coverage_Pct",
    "Volatility_Penalty",
    "Safety_Caps_Applied",
    # ML live_v3 debug columns
    "ML_20d_Prob_live_v3",
    "ML_20d_Prob_raw",
    "TechScore_20d_v2",
    "ATR_Pct_percentile",
    "Price_As_Of_Date",
    "ReliabilityFactor",
]
# --- Hebrew column mapping for CSV export ---
hebrew_cols = {
    "Ticker": "סימול",
    "Sector": "סקטור",
    "Risk Level": "רמת סיכון",
    "Data Quality": "איכות נתונים",
    "ML Probability": "הסתברות ML",
    "ML Confidence": "רמת ביטחון ML",
    "ML_Features_Fallback": "מאפייני ML חלופיים",
    "Confidence Level": "רמת ביטחון",
    "Reliability Score": "ציון אמינות",
    "Score": "ציון כולל",
    "Buy Amount v2": "סכום קנייה ($)",
    "Shares to Buy v2": "מניות לקנייה",
    "Buy Amount ($)": "סכום קנייה ($)",
    "Shares to Buy": "מניות לקנייה",
    "Leftover ($)": "עודף ($)",
    "Entry_Price": "מחיר כניסה",
    "Target_Price": "מחיר יעד",
    "RR": "סיכון/סיכוי",
    "Reliability_Band": "רצועת אמינות",
    "Market_Regime": "מצב שוק",
    "Regime_Confidence": "ביטחון מצב שוק",
    "Fundamental_S": "ציון יסודות",
    "Technical_S": "ציון טכני",
    "overall_score_20d": "ציון 20 יום",
    "Blended_RS_Value": "RS משולב",
    "Tightness_Ratio": "יחס הידוק",
    "Growth_Boost": "בוסט צמיחה",
    "MarketCap_B": "שווי שוק (מיליארדים)",
    # Add more mappings as needed for full export
}

# Ensure aliases for UI columns
try:
    if "Blended_RS_Value" not in rec_df.columns:
        rec_df["Blended_RS_Value"] = rec_df.get("market_rs_blended", rec_df.get("RS_63d", np.nan))
    # Tightness alias
    if "Tightness_Ratio" not in rec_df.columns:
        rec_df["Tightness_Ratio"] = rec_df.get("tightness_ratio", rec_df.get("VCP_Tightness", np.nan))
    # Growth boost alias
    if "Growth_Boost" not in rec_df.columns:
        rec_df["Growth_Boost"] = rec_df.get("growth_boost", rec_df.get("growth_accel_bonus", np.nan))
    # MarketCap in Billions
    if "MarketCap_B" not in rec_df.columns:
        mc = pd.to_numeric(rec_df.get("MarketCap", rec_df.get("market_cap", np.nan)), errors="coerce")
        rec_df["MarketCap_B"] = (mc / 1e9).where(mc.notna(), np.nan)
except Exception:
    pass

csv_df = rec_df.rename(columns=hebrew_cols)


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

# Deduplicate show_order for display to avoid Arrow errors
show_order_unique = []
seen_show = set()
for c in show_order:
    if c not in seen_show:
        show_order_unique.append(c)
        seen_show.add(c)

lean_export_fields = [
    c for c in cols_for_export if c in {
        "Ticker","Score","Overall_Rank","Rank","Entry_Price","Target_Price","RR","Risk_Level","Reliability_Band","Market_Regime","Regime_Confidence","Fundamental_S","Technical_S"
    }
]
full_export_fields = cols_for_export  # preserve full order

# Sidebar checkbox stored earlier under dev options (create if missing)
use_full_export = bool(st.session_state.get("use_full_export", False))
export_fields = full_export_fields if use_full_export else lean_export_fields
csv_bytes = csv_df[export_fields].to_csv(index=False).encode("utf-8-sig")

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
    json_data = csv_df[export_fields].to_json(
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

st.dataframe(
    csv_df[[c for c in show_order_unique if c in csv_df.columns]],
    width='stretch',
    hide_index=True,
)

# ==================== Quick chart ====================
st.subheader("🔍 Chart Ticker from Results")
# Choose a safe ticker column for charting
_ticker_candidates = ["Ticker", "symbol", "Symbol", "ticker"]
_tcol = None
for _tc in _ticker_candidates:
    if _tc in rec_df.columns:
        _tcol = _tc
        break

if _tcol is None or rec_df.empty:
    st.caption("No ticker column available for charting.")
else:
    choices = ["(Select)"] + rec_df[_tcol].astype(str).tolist()
    choice = st.selectbox("Select ticker", choices, index=0)
    if choice and choice != "(Select)" and isinstance(data_map, dict) and choice in data_map:
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
        st.plotly_chart(fig, width='stretch')
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
        st.plotly_chart(fig2, width='stretch')

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
