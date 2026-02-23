"""
Technical Scoring for Stock Scout.

Deterministic technical score computation based on weighted factor model.
All weights sourced from :mod:`core.scoring_config` (single source of truth).

Functions:
    compute_technical_score        — 10-factor weighted score (0-100)
    compute_tech_score_20d_v2      — Hybrid 4-component score (raw 0-1)
    compute_tech_score_20d_v2_components — Detailed component breakdown
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.scoring_config import ATR_RULES, REQUIRED_TECH_COLS, TECH_WEIGHTS

logger = logging.getLogger(__name__)


def compute_technical_score(
    row: pd.Series,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute a deterministic technical score (0-100) using shared weights.

    Combines 10+ technical factors (MA alignment, momentum, RSI, volume,
    overextension, pullback, R/R, MACD, ADX, VCP, Relative Strength)
    weighted according to ``scoring_config.TECH_WEIGHTS``.

    Args:
        row: Series with technical indicators from :func:`build_technical_indicators`.
        weights: Optional weight overrides (merged with TECH_WEIGHTS).

    Returns:
        Technical score in [0, 100].
    """
    missing = [c for c in REQUIRED_TECH_COLS if c not in row]
    if missing:
        logger.warning("compute_technical_score: missing columns %s", missing)

    # Merge & normalize weights
    raw_w = {**TECH_WEIGHTS, **(weights or {})}
    sanitized = {k: max(_coerce_float(v), 0.0) for k, v in raw_w.items()}
    total_w = sum(sanitized.values()) or 1.0
    nw = {k: v / total_w for k, v in sanitized.items()}

    def _get(col: str, default: float = np.nan) -> float:
        val = row.get(col, default)
        return float(val) if pd.notna(val) else float(default)

    # ── Individual factor scores (each 0-1) ──────────────────────
    ma_score = 1.0 if bool(row.get("MA_Aligned", False)) else 0.0
    mom_score = float(np.clip(_get("Momentum_Consistency", 0.0), 0.0, 1.0))

    rsi_val = _get("RSI", np.nan)
    meteor_mode = os.getenv("METEOR_MODE", "0") == "1"
    if pd.isna(rsi_val):
        rsi_score = 0.5
    elif meteor_mode:
        rsi_score = 1.0 if rsi_val >= 55 else (0.8 if rsi_val >= 45 else 0.4)
    else:
        rsi_score = 1.0 if 25 <= rsi_val <= 75 else max(0.0, 1.0 - (abs(rsi_val - 50) - 25) / 50.0)

    vol_surge = _get("VolSurge", 1.0)
    vol_score = float(np.clip(vol_surge / 2.0, 0.0, 1.0))

    overext = _get("Overext", 0.0)
    overext_score = float(np.clip(1.0 - (overext / 0.2), 0.0, 1.0))

    near_high = _get("Near52w", np.nan)
    pullback_score = 0.5 if pd.isna(near_high) else float(np.clip(1.0 - (near_high / 100.0), 0.0, 1.0))

    rr_raw = _get("RR", 1.0)
    atr_pct = _get("ATR_Pct", np.nan)

    # ATR scaling
    atr_factor = 1.0
    if pd.notna(atr_pct):
        if atr_pct >= ATR_RULES["extreme_high"]["min"]:
            atr_factor = ATR_RULES["extreme_high"].get("factor", 1.0)
        elif ATR_RULES["high"]["min"] <= atr_pct < ATR_RULES["high"]["max"]:
            atr_factor = ATR_RULES["high"].get("factor", 1.0)
        elif ATR_RULES["sweet_spot"]["min"] <= atr_pct <= ATR_RULES["sweet_spot"]["max"]:
            atr_factor = ATR_RULES["sweet_spot"].get("factor", 1.0)
        elif atr_pct < ATR_RULES["low"]["max"]:
            atr_factor = ATR_RULES["low"].get("factor", 1.0)

    rr_score = float(np.clip((max(rr_raw, 0.0) * atr_factor) / 3.0, 0.0, 1.5))

    macd_score = 1.0 if _get("MACD_Pos", 0.0) > 0 else 0.0
    adx_score = float(np.clip(_get("ADX14", 0.0) / 50.0, 0.0, 1.0))

    vcp_score = float(np.clip(row.get("Volatility_Contraction_Score", 0.0), 0.0, 1.0))

    # Relative Strength vs SPY
    rs_val = row.get("relative_strength_20d", np.nan)
    if pd.isna(rs_val):
        rs_val = row.get("Relative_Strength_vs_SPY", np.nan)
    rs_score = 0.5 if pd.isna(rs_val) else float(np.clip(rs_val / 0.50, 0.0, 1.0))

    # ── Weighted sum ─────────────────────────────────────────────
    base = (
        nw["ma"] * ma_score
        + nw["mom"] * mom_score
        + nw["rsi"] * rsi_score
        + nw["near_high_bell"] * pullback_score
        + nw["vol"] * vol_score
        + nw["overext"] * overext_score
        + nw["pullback"] * pullback_score
        + nw["risk_reward"] * rr_score
        + nw["macd"] * macd_score
        + nw["adx"] * adx_score
        + nw.get("relative_strength", 0.0) * rs_score
        + nw.get("vcp", 0.0) * vcp_score
    )

    # Volatility adjustment
    vol_adjust = 1.0
    if pd.notna(atr_pct):
        if atr_pct >= ATR_RULES["extreme_high"]["min"]:
            vol_adjust = 1.0 - ATR_RULES["extreme_high"].get("penalty", 0.0)
        elif ATR_RULES["sweet_spot"]["min"] <= atr_pct <= ATR_RULES["sweet_spot"].get("max", atr_pct):
            vol_adjust = ATR_RULES["sweet_spot"].get("factor", 1.0)
        elif atr_pct < ATR_RULES["low"]["max"]:
            vol_adjust = ATR_RULES["low"].get("factor", 1.0)

    return float(np.clip(base * vol_adjust * 100.0, 0.0, 100.0))


# ---------------------------------------------------------------------------
# V2 hybrid score
# ---------------------------------------------------------------------------

def compute_tech_score_20d_v2_components(row: pd.Series) -> dict:
    """Compute 4 component scores (each 0-1) for TechScore_20d_v2.

    Components and weights:
        - TrendScore (40%): price vs MA50, MA50 vs MA200, MA50 slope
        - MomentumScore (35%): 1m/3m/6m returns (de-emphasize parabolic)
        - VolatilityScore (15%): ATR_Pct sweet-spot
        - LocationScore (10%): penalize extreme RSI / near-highs

    Returns:
        Dict with individual scores plus ``TechScore_20d_v2_raw``.
    """
    # Trend (40%)
    trend_score = 0.5
    price_vs_ma50 = row.get("Overext", np.nan)
    ma50 = row.get("MA50", np.nan)
    ma200 = row.get("MA200", np.nan)
    ma50_slope = row.get("MA50_Slope", np.nan)
    if pd.notna(price_vs_ma50) and pd.notna(ma50) and pd.notna(ma200):
        if price_vs_ma50 > 0:
            trend_score = 0.6
            if ma50 > ma200:
                trend_score = 0.8
                if pd.notna(ma50_slope) and ma50_slope > 0:
                    trend_score = 1.0
        elif price_vs_ma50 < -0.05:
            trend_score = 0.3
            if ma50 < ma200:
                trend_score = 0.1

    # Momentum (35%)
    momentum_score = 0.5
    rets = [r for r in [row.get("Return_1m"), row.get("Return_3m"), row.get("Return_6m")] if pd.notna(r)]
    if rets:
        avg_ret = np.mean(rets)
        if 0.05 <= avg_ret <= 0.25:
            momentum_score = 1.0
        elif 0.0 <= avg_ret < 0.05:
            momentum_score = 0.7
        elif 0.25 < avg_ret <= 0.50:
            momentum_score = 0.6
        elif avg_ret > 0.50:
            momentum_score = 0.3
        elif -0.10 <= avg_ret < 0.0:
            momentum_score = 0.4
        else:
            momentum_score = 0.2

    # Volatility (15%)
    volatility_score = 0.5
    atr_pct = row.get("ATR_Pct", np.nan)
    if pd.notna(atr_pct):
        if 0.015 <= atr_pct <= 0.045:
            volatility_score = 1.0
        elif 0.01 <= atr_pct < 0.015:
            volatility_score = 0.6
        elif 0.045 < atr_pct <= 0.08:
            volatility_score = 0.7
        elif atr_pct < 0.01:
            volatility_score = 0.2
        else:
            volatility_score = 0.3

    # Location (10%)
    location_score = 0.5
    rsi = row.get("RSI", np.nan)
    if pd.notna(rsi):
        if rsi >= 75:
            location_score = 0.2
        elif rsi >= 65:
            location_score = 0.5
        elif 40 <= rsi < 65:
            location_score = 0.8
        elif 30 <= rsi < 40:
            location_score = 0.7
        elif rsi < 30:
            location_score = 0.5
    near_52w = row.get("Near52w", np.nan)
    if pd.notna(near_52w) and near_52w > 95:
        location_score *= 0.7

    raw_score = 0.40 * trend_score + 0.35 * momentum_score + 0.15 * volatility_score + 0.10 * location_score
    return {
        "TrendScore": trend_score,
        "MomentumScore": momentum_score,
        "VolatilityScore": volatility_score,
        "LocationScore": location_score,
        "TechScore_20d_v2_raw": raw_score,
    }


def compute_tech_score_20d_v2(row: pd.Series) -> float:
    """Hybrid technical score — raw value in [0, 1].

    Should be normalized to [0, 100] using percent-rank per date.
    """
    return compute_tech_score_20d_v2_components(row)["TechScore_20d_v2_raw"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _coerce_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0
