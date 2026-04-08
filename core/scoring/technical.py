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
        # Bell curve centered at 52 (mild bullish sweet spot), width 20.
        # RSI 52 → 1.0, RSI 70 → ~0.44, RSI 75 → ~0.28, RSI 30 → ~0.28
        rsi_score = float(np.exp(-0.5 * ((rsi_val - 52.0) / 20.0) ** 2))

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

def _smooth(x: float, center: float, width: float) -> float:
    """Smooth bell-curve score (0-1) centered at *center* with given *width*."""
    return float(np.exp(-0.5 * ((x - center) / max(width, 1e-9)) ** 2))


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear ramp from 0 at *lo* to 1 at *hi*, clamped to [0, 1]."""
    if hi == lo:
        return 1.0 if x >= lo else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def compute_tech_score_20d_v2_components(row: pd.Series) -> dict:
    """Compute 4 component scores (each 0-1) for TechScore_20d_v2.

    Components and weights:
        - TrendScore (40%): price vs MA50, MA50 vs MA200, MA50 slope
        - MomentumScore (35%): 5d/10d/20d returns (swing-trade aligned)
        - VolatilityScore (15%): ATR_Pct sweet-spot (3-6% for swing trades)
        - LocationScore (10%): RSI sweet spot + breakout proximity bonus

    Uses continuous smooth scoring (sigmoid/bell-curve) instead of discrete
    buckets to maximize differentiation across similar-quality stocks.

    Returns:
        Dict with individual scores plus ``TechScore_20d_v2_raw``.
    """
    # Trend (40%) — continuous blend of 3 sub-signals
    price_vs_ma50 = row.get("Overext", np.nan)
    ma50 = row.get("MA50", np.nan)
    ma200 = row.get("MA200", np.nan)
    ma50_slope = row.get("MA50_Slope", np.nan)

    # Sub-signal 1: price position relative to MA50 (ramp from -5% to +10%)
    if pd.notna(price_vs_ma50):
        price_sub = _ramp(price_vs_ma50, -0.05, 0.10)
    else:
        price_sub = 0.5

    # Sub-signal 2: MA alignment (MA50 vs MA200, continuous gap)
    if pd.notna(ma50) and pd.notna(ma200) and ma200 > 0:
        ma_gap = (ma50 - ma200) / ma200  # e.g. +0.05 = 5% above
        ma_sub = _ramp(ma_gap, -0.03, 0.08)
    else:
        ma_sub = 0.5

    # Sub-signal 3: MA50 slope (momentum of trend)
    if pd.notna(ma50_slope):
        slope_sub = _ramp(ma50_slope, -0.002, 0.005)
    else:
        slope_sub = 0.5

    trend_score = 0.40 * price_sub + 0.35 * ma_sub + 0.25 * slope_sub

    # Momentum (35%) — swing-trade aligned: 5d/10d/20d returns
    # Weight recent more heavily: 5d=40%, 10d=35%, 20d=25%
    r5 = row.get("Return_5d", np.nan)
    r10 = row.get("Return_10d", np.nan)
    r20 = row.get("Return_20d", np.nan)
    rets = [(r5, 0.40), (r10, 0.35), (r20, 0.25)]
    valid = [(float(r), w) for r, w in rets if pd.notna(r)]
    if not valid:
        # Fallback to legacy columns if short-term returns unavailable
        legacy = [r for r in [row.get("Return_1m"), row.get("Return_3m")] if pd.notna(r)]
        if legacy:
            valid = [(float(r), 1.0 / len(legacy)) for r in legacy]
    if valid:
        total_w = sum(w for _, w in valid)
        avg_ret = sum(r * w for r, w in valid) / total_w
        # Sweet spot: ~8% return in 20 days (annualized ~100%). Peak at 8%, width 12%.
        if avg_ret >= 0:
            momentum_score = _smooth(avg_ret, 0.08, 0.12)
            # Floor: even small positive returns get decent score
            momentum_score = max(momentum_score, _ramp(avg_ret, -0.02, 0.04) * 0.6)
        else:
            # Negative returns: linear decay from 0 to -10%
            momentum_score = max(0.1, _ramp(avg_ret, -0.10, 0.0) * 0.5)
    else:
        momentum_score = 0.5

    # Volatility (15%) — swing-trade sweet spot: 3-6% daily ATR
    atr_pct = row.get("ATR_Pct", np.nan)
    if pd.notna(atr_pct):
        # Peak at 4.5% ATR (enough movement to profit in 20 days), width 2.5%
        volatility_score = _smooth(atr_pct, 0.045, 0.025)
        # Floor: even low-vol stocks get a base score (don't eliminate them)
        volatility_score = max(volatility_score, 0.20)
    else:
        volatility_score = 0.5

    # Location (10%) — RSI sweet spot + near-high penalty
    rsi = row.get("RSI", np.nan)
    if pd.notna(rsi):
        # Best RSI range: 45-60 (mild bullish). Penalize extremes.
        location_score = _smooth(rsi, 52.0, 15.0)
        # Slight boost for oversold bounce potential (RSI 30-40)
        if 30 <= rsi <= 42:
            location_score = max(location_score, 0.65)
    else:
        location_score = 0.5

    near_52w = row.get("Near52w", np.nan)
    if pd.notna(near_52w) and near_52w > 85:
        # Breakout bonus ONLY if consolidation setup exists (VCP or tight range).
        # Stocks near highs WITHOUT a setup are overextended, not breakout candidates.
        _vcp = row.get("Volatility_Contraction_Score", 0)
        _tight = row.get("Tightness_Ratio", 1.0)
        _has_setup = (pd.notna(_vcp) and float(_vcp) > 0.3) or (pd.notna(_tight) and float(_tight) < 0.6)
        if _has_setup:
            breakout_bonus = _ramp(near_52w, 85.0, 98.0) * 0.15
            location_score = min(1.0, location_score + breakout_bonus)

    raw_score = 0.40 * trend_score + 0.35 * momentum_score + 0.15 * volatility_score + 0.10 * location_score
    return {
        "TrendScore": float(trend_score),
        "MomentumScore": float(momentum_score),
        "VolatilityScore": float(volatility_score),
        "LocationScore": float(location_score),
        "TechScore_20d_v2_raw": float(raw_score),
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
