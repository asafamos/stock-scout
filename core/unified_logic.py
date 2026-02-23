"""Backward-compatibility shim for ``core.unified_logic``.

All logic has been relocated to focused modules:

    core/indicators.py                — compute_rsi, compute_atr, build_technical_indicators
    core/scoring/technical.py         — compute_technical_score, compute_tech_score_20d_v2
    core/scoring/final.py             — compute_final_score, compute_final_score_with_patterns
    core/scoring/big_winner.py        — compute_big_winner_signal_20d
    core/scoring/recommendation.py    — compute_recommendation_scores, RecommendationResult
    core/scoring/backtest_helpers.py  — compute_forward_returns

This file re-exports all public names so existing callers continue to work.
New code should import from the specific modules instead.
"""
from __future__ import annotations

# ── Re-exports: Indicators ───────────────────────────────────────
from core.indicators import (                          # noqa: F401
    build_technical_indicators,
    compute_atr,
    compute_momentum_consistency,
    compute_reward_risk,
    compute_rsi,
    compute_volume_surge,
)

# ── Re-exports: Technical Scoring ────────────────────────────────
from core.scoring.technical import (                   # noqa: F401
    compute_tech_score_20d_v2,
    compute_tech_score_20d_v2_components,
    compute_technical_score,
)

# ── Re-exports: Final Score ──────────────────────────────────────
from core.scoring.final import (                       # noqa: F401
    compute_final_score,
    compute_final_score_with_patterns,
)

# ── Re-exports: Big Winner ───────────────────────────────────────
from core.scoring.big_winner import (                  # noqa: F401
    compute_big_winner_signal_20d,
)

# ── Re-exports: Recommendation ───────────────────────────────────
from core.scoring.recommendation import (              # noqa: F401
    RecommendationResult,
    apply_technical_filters,
    compute_recommendation_scores,
    score_with_ml_model,
)

# ── Re-exports: Backtest Helpers ─────────────────────────────────
from core.scoring.backtest_helpers import (            # noqa: F401
    compute_forward_returns,
)

# ── Remaining: Market Context & Data Fetching ────────────────────
# These stay here temporarily until Phase 3 migration.

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from core.config import get_config

logger = logging.getLogger(__name__)


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data from Yahoo Finance.

    Returns DataFrame with [Open, High, Low, Close, Volume] or None.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def build_market_context_table(
    start_date: str,
    end_date: str,
    provider_status: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """Build market context table with SPY/VIX features and regime classification.

    Computes for each trading date:
    - SPY returns (20d, 60d) and drawdown
    - VIX percentile (responsive)
    - Market regime (TREND_UP, SIDEWAYS, CORRECTION, PANIC, DISTRIBUTION)
    - One-hot regime flags
    """
    from core.market_context import get_benchmark_series, get_market_breadth

    spy_df = get_benchmark_series("SPY", period="18mo")
    if spy_df is None or spy_df.empty:
        raise RuntimeError("Market context unavailable: SPY series missing")

    vix_df = get_benchmark_series("^VIX", period="18mo")
    if vix_df is None or vix_df.empty:
        raise RuntimeError("Market context unavailable: VIX series missing")

    # SPY features
    spy_df = spy_df.sort_values("date").reset_index(drop=True)
    spy_df["SPY_20d_ret"] = spy_df["close"].pct_change(20)
    spy_df["SPY_60d_ret"] = spy_df["close"].pct_change(60)
    spy_df["SPY_high_60d"] = spy_df["close"].rolling(60).max()
    spy_df["SPY_drawdown_60d"] = (spy_df["close"] / spy_df["SPY_high_60d"]) - 1.0
    spy_df["SPY_momentum_20d"] = spy_df["close"].pct_change(5) * 4

    # VIX features
    vix_df = vix_df.sort_values("date").reset_index(drop=True)
    vix_df["VIX_close"] = vix_df["close"]
    vix_df["VIX_pct"] = vix_df["VIX_close"].rolling(63, min_periods=20).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
    )
    vix_df["VIX_pct_20d"] = vix_df["VIX_close"].rolling(20, min_periods=10).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
    )
    vix_df["VIX_pct_responsive"] = vix_df[["VIX_pct", "VIX_pct_20d"]].max(axis=1)

    # Merge
    context_df = spy_df[["date", "SPY_20d_ret", "SPY_60d_ret", "SPY_drawdown_60d", "SPY_momentum_20d"]].copy()
    context_df = context_df.merge(
        vix_df[["date", "VIX_close", "VIX_pct", "VIX_pct_responsive"]],
        on="date",
        how="left",
    )
    context_df["VIX_close"] = context_df["VIX_close"].fillna(20.0)
    context_df["VIX_pct"] = context_df["VIX_pct"].fillna(0.5)
    context_df["VIX_pct_responsive"] = context_df["VIX_pct_responsive"].fillna(0.5)
    context_df["SPY_momentum_20d"] = context_df["SPY_momentum_20d"].fillna(0.0)

    # Regime classification
    def classify_regime(r):
        try:
            row_date = r.get("date") if isinstance(r, dict) else r.get("date", None)
            date_key = None
            try:
                if row_date is not None:
                    date_key = pd.to_datetime(row_date).strftime("%Y-%m-%d")
            except Exception:
                pass
            breadth = float(get_market_breadth(date_key))
            dd = r.get("SPY_drawdown_60d", 0)
            ret_60d = r.get("SPY_60d_ret", 0)
            vix_pct = r.get("VIX_pct_responsive", r.get("VIX_pct", 0.5))
            vix_close = r.get("VIX_close", 20.0)
            momentum = r.get("SPY_momentum_20d", 0.0)
            if pd.isna(dd) or pd.isna(ret_60d) or pd.isna(vix_pct) or not np.isfinite(breadth):
                return "SIDEWAYS"
            # ── Negative regimes (checked first) ──
            if dd < -0.15 or vix_pct > 0.85 or vix_close > 30:
                return "PANIC"
            if dd < -0.08 or vix_pct > 0.70 or vix_close > 25:
                return "CORRECTION"
            if abs(ret_60d) <= 0.02 and breadth < 0.40:
                return "DISTRIBUTION"
            # ── Strong uptrend ──
            if ret_60d > 0.08 and dd > -0.05 and breadth > 0.55 and vix_close < 22:
                return "TREND_UP"
            if momentum > 0.12 and breadth > 0.50 and vix_close < 23:
                return "TREND_UP"
            # ── Moderate uptrend (new: fills gap between SIDEWAYS and TREND_UP) ──
            if ret_60d > 0.03 and dd > -0.06 and vix_close < 24:
                return "MODERATE_UP"
            if momentum > 0.05 and ret_60d > 0.0 and vix_close < 25:
                return "MODERATE_UP"
            return "SIDEWAYS"
        except Exception:
            return "SIDEWAYS"

    context_df["Market_Regime"] = context_df.apply(classify_regime, axis=1)

    # One-hot flags
    context_df["Regime_TrendUp"] = (context_df["Market_Regime"] == "TREND_UP").astype(int)
    context_df["Regime_ModerateUp"] = (context_df["Market_Regime"] == "MODERATE_UP").astype(int)
    context_df["Regime_Sideways"] = (context_df["Market_Regime"] == "SIDEWAYS").astype(int)
    context_df["Regime_Correction"] = (context_df["Market_Regime"] == "CORRECTION").astype(int)
    context_df["Regime_Panic"] = (context_df["Market_Regime"] == "PANIC").astype(int)

    # Filter to requested range
    context_df = context_df[
        (context_df["date"] >= pd.to_datetime(start_date))
        & (context_df["date"] <= pd.to_datetime(end_date))
    ].copy()

    logger.info("Built market context: %d dates, regimes: %s", len(context_df), context_df["Market_Regime"].value_counts().to_dict())
    return context_df
