
from __future__ import annotations
"""Centralized scoring configuration for Stock Scout.

All weights, thresholds, and scoring constants live here so that
both the pipeline and unified_logic share a single source of truth.
"""
import numpy as np
import pandas as pd

# Canonical score column and aliases
CANONICAL_SCORE_COLUMN = "FinalScore_20d"
SCORE_ALIASES = ["Score", "overall_score_20d", "overall_score", "overall_score_pretty"]

def get_canonical_score(row: pd.Series) -> float:
    for col in [CANONICAL_SCORE_COLUMN] + SCORE_ALIASES:
        if col in row and pd.notna(row.get(col)):
            return float(row[col])
    return np.nan

from dataclasses import dataclass, field
from typing import Dict

# Technical scoring weights (must sum to ~1.0; normalized internally)
# NOTE: VCP and RS weights increased (2026-02-03) to emphasize breakout patterns.
TECH_WEIGHTS: Dict[str, float] = {
    "ma": 0.12,
    "mom": 0.12,
    "rsi": 0.08,          # Decreased - RSI less predictive for breakouts
    "near_high_bell": 0.05,
    "vol": 0.05,
    "overext": 0.03,
    "pullback": 0.03,
    "risk_reward": 0.04,
    "macd": 0.03,
    "adx": 0.04,
    # VCP and RS emphasized for breakout detection (normalized downstream)
    "vcp": 0.25,           # Increased from 0.20 - VCP is key for breakouts
    "relative_strength": 0.20,  # Increased from 0.17
}

# Final score combination weights (3-component: tech + fund + ML)
# NOTE (2026-02-17): ML weight reduced from 0.35 to 0.15 because the production
# model (V3.1) has AUC=0.553 (barely above random). Technical and fundamental
# weights increased to compensate until a better model is trained.
FINAL_SCORE_WEIGHTS: Dict[str, float] = {
    "technical": 0.55,
    "fundamental": 0.30,
    "ml": 0.15,
}

# Pattern-enhanced score weights (5-component, used by compute_final_score_with_patterns)
# When Big Winner + Pattern data is available, ML share is redistributed.
# Weights are auto-normalized to sum to 1.0 at runtime.
PATTERN_SCORE_WEIGHTS: Dict[str, float] = {
    "technical": 0.45,
    "fundamental": 0.20,
    "ml": 0.15,
    "big_winner": 0.10,
    "pattern": 0.10,
}

# Conviction score weights (used by compute_final_score_20d)
# Row-level scoring after fundamentals merge: emphasizes reliability & risk/reward.
CONVICTION_WEIGHTS: Dict[str, float] = {
    "fundamental": 0.30,
    "momentum": 0.30,
    "risk_reward": 0.20,
    "reliability": 0.20,
}

# Base score weights (tech/fund only, before ML boost is applied)
# Used by UnifiedScorer and similar 2-component scoring
# Derived from FINAL_SCORE_WEIGHTS by removing ML and renormalizing:
# tech=0.55/(0.55+0.30)=0.647, fund=0.30/(0.55+0.30)=0.353
BASE_SCORE_WEIGHTS: Dict[str, float] = {
    "technical": 0.65,
    "fundamental": 0.35,
}

# ML gate thresholds and multipliers (single source of truth)
# NOTE (2026-02-17): Thresholds adjusted because V3.1 model outputs are
# clustered in a narrow 0.46-0.74 range with AUC=0.553. The old bonus_gt
# of 0.62 was giving a 15% bonus to almost every stock.
# New thresholds: penalty only for truly negative signals, modest bonus
# only for the most extreme positive signals. Effect is mild until a
# better model is trained.
ML_GATES: Dict[str, float] = {
    "penalty_lt": 0.30,     # was 0.15 — more stocks penalized when model says NO
    "bonus_gt": 0.75,       # was 0.62 — only exceptional ML confidence gets bonus
    "penalty_mult": 0.85,   # was 0.60 — softer penalty (ML not reliable enough for harsh cut)
    "bonus_mult": 1.08,     # was 1.15 — smaller bonus (ML doesn't warrant big boost)
}

# Market regime multipliers
REGIME_MULTIPLIERS: Dict[str, float] = {
    "TREND_UP": 1.10,
    "BULLISH": 1.10,    # alias used by detect_market_regime()
    "PANIC": 0.70,
    "CORRECTION": 0.70,
    "BEARISH": 0.70,    # alias used by detect_market_regime()
    "SIDEWAYS": 1.00,   # alias used by detect_market_regime()
    "NEUTRAL": 1.00,    # alias used by detect_market_regime()
}

# ATR-based volatility rules used to adjust technical score and RR
ATR_RULES = {
    "extreme_high": {"min": 0.06, "factor": 0.5, "penalty": 0.15},
    "high": {"min": 0.05, "max": 0.06, "factor": 0.7},
    "sweet_spot": {"min": 0.02, "max": 0.05, "factor": 1.1},
    "low": {"max": 0.02, "factor": 0.8},
}

# Advanced filter defaults (lenient so pipeline can prune later)
ADVANCED_FILTER_DEFAULTS = {
    "rs_63d_min": 0.0,  # percentile values; existing logic uses dynamic thresholds
    "momentum_consistency_min": 0.0,
    "rr_min": 0.0,
}

# Required technical columns for compute_technical_score
REQUIRED_TECH_COLS = [
    "MA_Aligned",
    "Momentum_Consistency",
    "RSI",
    "VolSurge",
    "Overext",
    "Near52w",
    "RR",
    "ATR_Pct",
    "MACD_Pos",
    "ADX14",
]

# Risk meter thresholds (used by calculate_risk_meter)
RISK_LEVEL_THRESHOLDS = {
    "rr_bands": [(0.8, 95), (1.2, 75), (1.8, 55), (2.5, 35), (3.5, 20), (float('inf'), 10)],
    "beta_bands": [(1.8, 85), (1.4, 65), (1.1, 50), (0.9, 40), (0.6, 25), (0, 15)],
    "atr_bands": [(0.08, 90), (0.05, 70), (0.03, 50), (0.02, 30), (0, 15)],
    "leverage_bands": [(3.0, 85), (2.0, 65), (1.5, 50), (0.8, 35), (0.3, 20), (0, 10)],
    "label_thresholds": [(75, "VERY HIGH"), (60, "HIGH"), (45, "MODERATE-HIGH"), (30, "MODERATE"), (20, "LOW"), (0, "VERY LOW")],
}

# Reliability score weights (used by calculate_reliability_score)
RELIABILITY_WEIGHTS = {
    "data_completeness": 0.40,
    "price_variance": 0.30,
    "fundamental_coverage": 0.20,
    "source_count": 0.10,
}

# Big winner signal thresholds for 20d horizon
BIG_WINNER_THRESHOLDS = {
    "atr_min": 0.03,
    "atr_bonus": 0.05,
    "score_mid_low": 40,
    "score_mid_high": 80,
    "score_weak_low": 30,
    "score_weak_high": 90,
    "rsi_max": 70,
    "rsi_bonus": 60,
    "rr_min": 1.2,
    "rr_max": 3.0,
    "signal_threshold": 60,  # Flag = 1 if score >= this
}

# --- Signal Engine (Allocation-Free) Thresholds ---
# Minimum final score (0-100 scale) to consider as a signal candidate
SIGNAL_MIN_SCORE: float = 55.0
# ML probability threshold to qualify as a candidate even if final score is lower
ML_PROB_THRESHOLD: float = 0.62
# Default top-K limit for ranked signal output
TOP_SIGNAL_K: int = 30
# Technical score threshold indicating strong momentum
TECH_STRONG_THRESHOLD: float = 65.0


@dataclass
class MultiSourceData:
    """Normalized container for multi-provider fundamentals & price reliability."""
    pe: float | None = None
    ps: float | None = None
    pb: float | None = None
    roe: float | None = None
    margin: float | None = None
    rev_yoy: float | None = None
    eps_yoy: float | None = None
    debt_to_equity: float | None = None
    price_mean: float | None = None
    price_std: float | None = None
    beta: float | None = None
    market_cap: float | None = None
    sources_used: list[str] = field(default_factory=list)
    disagreement_score: float | None = None
    price_sources: int | None = None

    @classmethod
    def from_dict(cls, data: Dict) -> "MultiSourceData":
        return cls(
            pe=data.get("pe"),
            ps=data.get("ps"),
            pb=data.get("pb"),
            roe=data.get("roe"),
            margin=data.get("margin"),
            rev_yoy=data.get("rev_yoy"),
            eps_yoy=data.get("eps_yoy"),
            debt_to_equity=data.get("debt_equity") or data.get("debt_to_equity"),
            price_mean=data.get("price_mean"),
            price_std=data.get("price_std"),
            beta=data.get("beta"),
            market_cap=data.get("market_cap"),
            sources_used=data.get("sources_used", []) or [],
            disagreement_score=data.get("disagreement_score"),
            price_sources=data.get("price_sources"),
        )
