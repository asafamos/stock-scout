"""
Stock Scout Application Configuration

This module provides the runtime configuration dictionary (CONFIG)
that combines values from core.config with UI-specific settings.

Usage:
    from app_config import CONFIG, FUND_SCHEMA_FIELDS
"""
from __future__ import annotations

import os
import numpy as np
from typing import Dict, Any, List

from core.config import get_config

# Get config object
_config_obj = get_config()

# Build CONFIG dictionary from dataclass
CONFIG: Dict[str, Any] = {
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
    # Disabled providers (legacy/deprecated/paid)
    "ENABLE_SIMFIN": False,
    "ENABLE_MARKETSTACK": False,
    "ENABLE_NASDAQ_DL": False,
    "ENABLE_EODHD": False,
    # Performance / Fast Mode Flags
    "PERF_FAST_MODE": False,
    "PERF_MULTI_SOURCE_TOP_N": 8,
    "PERF_ALPHA_ENABLED": True,
    "PERF_FUND_TIMEOUT": 15,
    "PERF_FUND_TIMEOUT_FAST": 6,
    # Debug / Developer Flags
    "DEBUG_MODE": os.getenv("STOCK_SCOUT_DEBUG", "false").lower() in ("true", "1", "yes"),
    # Remote autoscan artifacts
    "USE_REMOTE_AUTOSCAN": True,
    "REMOTE_AUTOSCAN_REPO": os.getenv("REMOTE_AUTOSCAN_REPO", "asafamos/stock-scout"),
    "REMOTE_AUTOSCAN_BRANCH": os.getenv("REMOTE_AUTOSCAN_BRANCH", "main"),
}

# Canonical fundamentals schema
FUND_SCHEMA_FIELDS: List[str] = [
    "oper_margin",
    "roe",
    "roic",
    "gm",
    "ps",
    "pe",
    "de",
    "rev_g_yoy",
    "eps_g_yoy",
]

FUND_STRING_FIELDS: List[str] = ["sector", "industry"]


def empty_fund_row() -> Dict[str, Any]:
    """Create empty fundamentals dict for a ticker (all NaN / Unknown)."""
    out = {f: np.nan for f in FUND_SCHEMA_FIELDS}
    out["sector"] = "Unknown"
    out["industry"] = "Unknown"
    out["_sources"] = {}
    out["_sources_used"] = []
    out["Fund_Coverage_Pct"] = 0.0
    out["fundamentals_available"] = False
    return out
