"""Centralized scoring configuration for Stock Scout.

All weights, thresholds, and scoring constants live here so that
both the pipeline and unified_logic share a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Technical scoring weights (must sum to ~1.0; normalized internally)
TECH_WEIGHTS: Dict[str, float] = {
    "ma": 0.15,
    "mom": 0.15,
    "rsi": 0.12,
    "near_high_bell": 0.07,
    "vol": 0.06,
    "overext": 0.04,
    "pullback": 0.03,
    "risk_reward": 0.04,
    "macd": 0.03,
    "adx": 0.04,
    "vcp": 0.12,
    "relative_strength": 0.15,
}

# Final score combination weights
FINAL_SCORE_WEIGHTS: Dict[str, float] = {
    "technical": 0.55,
    "fundamental": 0.25,
    "ml": 0.20,
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
