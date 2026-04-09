
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
# NOTE (2026-02-28): Swing-trade aligned. Fundamentals reduced from 0.30 to 0.15
# because they don't move price in 20 days. ML weight increased to 0.25 because
# the ML circuit breaker in scoring/final.py scales it down for weak models.
FINAL_SCORE_WEIGHTS: Dict[str, float] = {
    "technical": 0.60,
    "fundamental": 0.15,
    "ml": 0.25,
}

# Pattern-enhanced score weights (5-component, used by compute_final_score_with_patterns)
# Swing-trade aligned: technical dominates, fundamentals are a minor quality check.
# Weights are auto-normalized to sum to 1.0 at runtime.
PATTERN_SCORE_WEIGHTS: Dict[str, float] = {
    "technical": 0.50,
    "fundamental": 0.10,
    "ml": 0.15,
    "big_winner": 0.10,
    "pattern": 0.15,
}

# Conviction score weights (used by compute_final_score_20d)
# Swing-trade aligned: momentum leads but fundamentals validate quality.
# Rationale (2026-03-07 v2): Deep analysis showed that even at 20% fundamental
# weight, the ROE difference between 38% and 4.5% contributed only ~1.5 pts to
# the final score — not enough to separate quality from junk.  Raised to 25%
# so a 7.5-pt Fundamental_S gap yields ~1.9 pts (still modest, but meaningful
# when combined with RSI timing adjustments and lower default for missing data).
# Momentum at 30% + pattern/BW bonuses (up to +10 pts) + Coil_Bonus still gives
# momentum-driven stocks a significant edge, but quality now acts as a real gate.
# NOTE (2026-03-08): Walk-forward optimization (train 2024-01→2025-06, test 2025-07→2026-03)
# confirmed these weights are near-optimal. Optimized: fund=0.24, mom=0.31, rr=0.26, rel=0.19
# Test Sharpe improvement was only +0.005 (below 0.1 adoption threshold). Keeping originals.
CONVICTION_WEIGHTS: Dict[str, float] = {
    "fundamental": 0.15,
    "momentum": 0.40,
    "risk_reward": 0.25,
    "reliability": 0.20,
}

# Base score weights (tech/fund only, before ML boost is applied)
# Used by UnifiedScorer and similar 2-component scoring
# Derived from FINAL_SCORE_WEIGHTS by removing ML and renormalizing:
# tech=0.55/(0.55+0.30)=0.647, fund=0.30/(0.55+0.30)=0.353
BASE_SCORE_WEIGHTS: Dict[str, float] = {
    "technical": 0.80,
    "fundamental": 0.20,
}

# ML gate thresholds and multipliers (single source of truth)
# NOTE (2026-02-17): Thresholds adjusted because V3.1 model outputs are
# clustered in a narrow 0.46-0.74 range with AUC=0.553. The old bonus_gt
# of 0.62 was giving a 15% bonus to almost every stock.
# New thresholds: penalty only for truly negative signals, modest bonus
# only for the most extreme positive signals. Effect is mild until a
# better model is trained.
ML_GATES: Dict[str, float] = {
    "penalty_lt": 0.45,     # was 0.30 — penalize anything below neutral-ish
    "bonus_gt": 0.75,       # only exceptional ML confidence gets bonus
    "penalty_mult": 0.88,   # multiplicative penalty (12% reduction)
    "bonus_mult": 1.08,     # smaller bonus (ML doesn't warrant big boost)
}

# Strong ML gates — activated when model AUC ≥ 0.58 (proven signal)
# Lower bonus threshold (model outputs are reliable) and bigger bonus multiplier
ML_GATES_STRONG: Dict[str, float] = {
    "penalty_lt": 0.45,     # was 0.25 — penalize below neutral
    "bonus_gt": 0.60,       # Lower bar — model has earned trust at AUC ≥ 0.58
    "penalty_mult": 0.82,   # Stronger penalty — model's NO signal is reliable (18% cut)
    "bonus_mult": 1.15,     # Bigger bonus — model's YES signal is reliable
}

# AUC-tiered ML boost: maps model quality to max ML score adjustment
# This determines how much the ML signal can move the final score
ML_BOOST_RANGE: Dict[str, Dict[str, float]] = {
    "weak":   {"min_auc": 0.0,  "max_auc": 0.55, "max_points": 3},    # AUC < 0.55: barely move score
    "modest": {"min_auc": 0.55, "max_auc": 0.58, "max_points": 8},    # AUC 0.55-0.58: modest influence
    "good":   {"min_auc": 0.58, "max_auc": 0.62, "max_points": 15},   # AUC 0.58-0.62: real signal
    "strong": {"min_auc": 0.62, "max_auc": 1.0,  "max_points": 25},   # AUC > 0.62: strong alpha
}

# Risk/Reward gate thresholds and multipliers
# Stocks with poor R/R should never be top recommendations.
# A stock with RR < 1 means expected loss > expected gain — that's a bad trade.
RR_GATES: Dict[str, float] = {
    "harsh_penalty_lt": 1.0,   # RR < 1.0: risk exceeds reward — harsh penalty
    "harsh_penalty_mult": 0.70,
    "mild_penalty_lt": 1.5,    # RR < 1.5: below professional minimum
    "mild_penalty_mult": 0.85,
    "mild_bonus_gt": 2.5,      # RR > 2.5: good asymmetric trade
    "mild_bonus_mult": 1.06,
    "strong_bonus_gt": 4.0,    # RR > 4.0: excellent risk/reward
    "strong_bonus_mult": 1.12,
}

# Entry timing thresholds (used by compute_final_score_20d)
# Penalize stocks near ATH without consolidation setup; reward pullback entries.
ENTRY_TIMING: Dict[str, float] = {
    "near_ath_penalty": 5.0,      # Within 3% of 52w high, no VCP setup (was 8.0 — too aggressive for momentum)
    "near_high_penalty": 2.0,     # Within 5% of 52w high, no VCP setup (was 4.0)
    "pullback_bonus": 3.0,        # 5-15% below 52w high (quality entry zone) (was 5.0)
    "pullback_max_return": 0.10,  # Pullback bonus only if return_20d < 10% (genuine pullback)
    "runup_threshold": 0.15,      # 15% return in 20d = already extended
    "runup_penalty": 5.0,         # Penalty for rapid run-up (restored from 3.0 — need to filter extended stocks like COKE +34%)
    "extreme_runup_threshold": 0.25,  # 25% return in 20d = severely overextended
    "extreme_runup_penalty": 10.0,    # Strong penalty for extreme run-up
    "vcp_ath_threshold": 0.4,     # VCP score to bypass ATH penalty
    "vcp_near_threshold": 0.3,    # VCP score to bypass near-high penalty
    # Momentum damping near ATH: reduce momentum contribution when stock is
    # near 52w high without a VCP setup.  Addresses inflated scores (e.g. RRC 97)
    # caused by momentum (40% weight) being naturally maxed at highs.
    "ath_momentum_damper": 0.70,       # Within 3% of ATH → mom *= 0.70
    "near_high_momentum_damper": 0.85, # Within 5% of ATH → mom *= 0.85
    # RR hard caps — prevent momentum from overriding poor risk/reward
    "rr_cap_harsh_lt": 1.5,       # RR < 1.5 → cap score at 90
    "rr_cap_harsh_max": 90.0,
    "rr_cap_mild_lt": 2.0,        # RR < 2.0 → cap score at 95
    "rr_cap_mild_max": 95.0,
}

# Bonus configuration for compute_final_score_20d pattern/VCP/BW bonuses
# Centralizes magic numbers from scoring_engine.py into a single source of truth.
# NOTE (2026-03-08): Ablation study (2024-01 to 2026-03, 247 trades) showed:
#   - Patterns HURT: no_patterns Sharpe=0.34 vs full_system=0.29 (+0.85% return)
#   - ML helps modestly: +0.10 Sharpe, +1.6% return
#   - Pattern bonuses shifted selection toward stocks that underperform
# Action: Reduced pattern/BW bonuses by ~40%, lowered total cap from 10→6.
# Weight optimization confirmed current CONVICTION_WEIGHTS are near-optimal
# (test Sharpe delta = +0.005, below 0.1 threshold for adoption).
BONUS_CONFIG: Dict[str, float] = {
    "coil_amplifier": 1.05,              # Multiplier for momentum when Coil_Bonus active
    "reliability_low_threshold": 40.0,   # Below this → heavily clamp ML boost
    "reliability_low_ml_mult": 0.25,     # ML boost multiplier for low reliability
    "reliability_med_threshold": 60.0,   # Below this → half ML boost
    "reliability_med_ml_mult": 0.50,     # ML boost multiplier for medium-low reliability
    "vcp_multiplier": 3.0,              # VCP score → bonus multiplier
    "vcp_bonus_max": 3.0,               # Max VCP-only bonus points
    "tightness_ratio_threshold": 0.6,   # Tightness_Ratio below this → extra bonus
    "tightness_bonus": 2.0,             # Bonus for tight consolidation
    "vcp_tightness_cap": 4.0,           # Local cap for VCP + tightness bonus (was 5.0)
    "pattern_multiplier": 0.0,          # Disabled — ablation study showed patterns HURT (Sharpe 0.34→0.29)
    "pattern_bonus_max": 0.0,           # Disabled — patterns shifted selection toward underperformers
    "big_winner_threshold": 50.0,       # BW signal above this → bonus
    "big_winner_multiplier": 0.0,       # Disabled — part of pattern noise per ablation study
    "total_bonus_cap": 4.0,             # Reduced from 6.0 — only VCP+tightness remain
    "hunter_floor": 45.0,               # Min score for VCP/coil setups with good RR
}

# RSI timing adjustment thresholds for compute_final_score_20d
RSI_ADJUSTMENTS: Dict[str, float] = {
    "overbought_hard_threshold": 75.0,  # Above this → strong penalty
    "overbought_hard_penalty": 8.0,     # Points deducted for strongly overbought
    "overbought_threshold": 70.0,       # Above this → moderate penalty
    "overbought_penalty": 5.0,          # Points deducted for overbought
    "sweet_spot_min": 45.0,             # Sweet spot lower bound
    "sweet_spot_max": 60.0,             # Sweet spot upper bound
    "sweet_spot_bonus": 2.0,            # Points added for sweet spot RSI
}

# Regime-aware RSI adjustments: overbought is more dangerous in distribution/bearish.
# Values are additive adjustments (negative = penalty, positive = bonus).
# Fallback to NEUTRAL when regime is missing or unknown.
RSI_REGIME_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "BULLISH":      {"overbought_75": -6.0, "overbought_70": -4.0, "sweet_spot": 2.0},
    "TREND_UP":     {"overbought_75": -6.0, "overbought_70": -4.0, "sweet_spot": 2.0},
    "MODERATE_UP":  {"overbought_75": -7.0, "overbought_70": -5.0, "sweet_spot": 2.0},
    "NEUTRAL":      {"overbought_75": -8.0, "overbought_70": -5.0, "sweet_spot": 2.0},
    "SIDEWAYS":     {"overbought_75": -8.0, "overbought_70": -5.0, "sweet_spot": 2.0},
    "DISTRIBUTION": {"overbought_75": -10.0, "overbought_70": -7.0, "sweet_spot": 1.0},
    "BEARISH":      {"overbought_75": -12.0, "overbought_70": -8.0, "sweet_spot": 0.0},
    "CORRECTION":   {"overbought_75": -12.0, "overbought_70": -8.0, "sweet_spot": 0.0},
    "PANIC":        {"overbought_75": -12.0, "overbought_70": -8.0, "sweet_spot": 0.0},
}

# R:R Score floor: prevent weak R:R stocks from being top-ranked.
# Works alongside ENTRY_TIMING rr_cap_* (which use raw ratio, not score).
RR_SCORE_FLOOR: Dict[str, float] = {
    "min_rr_score": 50.0,       # Minimum R:R Score (0-100 from evaluate_rr_unified)
    "floor_cap": 75.0,          # Max final score when R:R Score below floor
}

# Market regime multipliers
REGIME_MULTIPLIERS: Dict[str, float] = {
    "TREND_UP": 1.10,
    "MODERATE_UP": 1.05,  # new: moderate uptrend between SIDEWAYS and TREND_UP
    "BULLISH": 1.10,      # alias used by detect_market_regime()
    "PANIC": 0.70,
    "CORRECTION": 0.70,
    "DISTRIBUTION": 0.88, # Wyckoff distribution: mild penalty (ATR targets already conservative)
    "BEARISH": 0.70,      # alias used by detect_market_regime()
    "SIDEWAYS": 0.95,     # mild penalty: neutral market requires higher quality
    "NEUTRAL": 0.95,      # mild penalty: neutral market requires higher quality
}

# ATR-based volatility rules — swing-trade aligned.
# 3-6% ATR is the sweet spot: enough daily movement to profit in 20 days.
# Higher thresholds than value-investing (where 2-4% was "sweet spot").
ATR_RULES = {
    "extreme_high": {"min": 0.10, "factor": 0.6, "penalty": 0.10},
    "high": {"min": 0.06, "max": 0.10, "factor": 0.9},
    "sweet_spot": {"min": 0.025, "max": 0.06, "factor": 1.1},
    "low": {"max": 0.025, "factor": 0.7},
}

# Hard filters — stocks violating these are REJECTED (SafetyBlocked=True)
HARD_FILTERS: Dict[str, object] = {
    "min_rr": 1.5,                      # minimum Reward:Risk ratio
    "min_roe": 5.0,                     # block ROE below 5% (was 3.0 — too lenient, let through marginal names like ANDE 3.8%, BWLP 4.8%)
    "require_fundamental_data": True,   # block if BOTH ROE and MarketCap are missing
    "max_rsi": 70.0,                    # block overbought stocks (was 72.0 — tightened to filter extended momentum like BTSG RSI 71.9)
}

# Dynamic R:R adjustments — per-stock target/stop modifiers.
# Used by _compute_rr_for_row() to make R:R vary between stocks instead of
# being fixed at ATR_mult / stop_mult for all stocks in the same regime.
DYNAMIC_RR_CONFIG: Dict[str, float] = {
    # Relative strength adjustments to ATR target multiplier
    "rs_strong_threshold": 1.2,     # RS_63d above this → strong adj
    "rs_strong_adj": 0.3,
    "rs_above_avg_threshold": 1.0,  # RS_63d above this → mild adj
    "rs_above_avg_adj": 0.15,
    "rs_weak_threshold": 0.8,       # RS_63d below this → negative adj
    "rs_weak_adj": -0.2,
    # VCP adjustments
    "vcp_strong_threshold": 0.7,    # VCS above this → strong adj
    "vcp_strong_adj": 0.2,
    "vcp_moderate_threshold": 0.4,  # VCS above this → mild adj
    "vcp_moderate_adj": 0.1,
    # Momentum consistency
    "momentum_cons_threshold": 0.65,
    "momentum_cons_adj": 0.1,
    # ATR multiplier clamp
    "atr_mult_min": 1.5,
    "atr_mult_max": 5.0,
    # VCP stop tightening
    "vcp_stop_threshold": 0.5,      # VCS above this → tighter stop
    "vcp_max_stop_pct": 0.08,       # Max 8% stop loss for VCP setups
}

# Regime-aware ATR multipliers for target-price calculation
# IMPORTANT: target multiplier must be > stop multiplier (ATR_STOP_MULTIPLIER)
# otherwise R:R will always be ≈1.0 and stocks will fail the min_rr hard filter.
# With stop=1.5 ATR → bearish base RR=2.3/1.5=1.53 (just clears 1.5 hard filter),
# neutral base RR=2.5/1.5=1.67, bullish base RR=3.0/1.5=2.0.
# Old bearish base was 2.0 → pure-ATR RR=1.33 which FAILS the 1.5 hard filter,
# causing many valid stocks to be rejected in distribution/bearish regimes.
ATR_TARGET_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "bullish": {"base": 3.0, "breakout": 3.5},
    "neutral": {"base": 2.5, "breakout": 3.0},
    "bearish": {"base": 2.3, "breakout": 2.8},
}
# Stop-loss ATR multiplier (used by _compute_rr_for_row in pipeline/helpers.py)
ATR_STOP_MULTIPLIER: float = 1.5

# Maximum allowed upside (%) for target price relative to entry.
# Prevents unrealistic targets when resistance (e.g. 52w high) is far above entry.
# Stocks that crashed 50%+ from highs would otherwise show 100%+ upside in 12 days.
MAX_TARGET_UPSIDE_PCT: float = 0.30  # 30% max upside cap

# Entry offset: simulate limit order below Close. Entry = Close - ENTRY_OFFSET * ATR14.
# Set to 0.0 to revert to close-based entry (legacy behavior).
ENTRY_OFFSET: float = 0.3

# Support-based stop loss configuration for ALL stocks (not just VCP).
# Extends the VCP support-stop concept: chooses the HIGHEST support level
# that's at least min_risk_pct below entry, with ATR as the absolute floor.
# Set "enabled" to False to revert to legacy min(low_5, atr_stop) behavior.
SUPPORT_STOP_CONFIG: Dict[str, object] = {
    "enabled": True,
    "min_risk_pct": 0.02,             # Support must be >= 2% below entry to count
    "max_risk_pct": 0.10,             # Maximum risk (ATR floor fallback)
    "bollinger_periods": 20,
    "bollinger_std": 2.0,
    "low_10d_enabled": True,
    "low_20d_enabled": True,
    "bollinger_enabled": True,
}

# VIX-aware R:R minimum thresholds. Higher VIX → higher R:R required.
# Falls back to HARD_FILTERS["min_rr"] when VIX is unavailable.
VIX_RR_TIERS: list = [
    {"vix_max": 20.0, "min_rr": 1.5},   # low VIX
    {"vix_max": 25.0, "min_rr": 1.6},   # elevated
    {"vix_max": 30.0, "min_rr": 1.8},   # high
    {"vix_max": 999.0, "min_rr": 2.0},  # extreme
]


def get_vix_min_rr(vix_value) -> float:
    """Return the min R:R for the given VIX level, ascending tiers."""
    if vix_value is None or not isinstance(vix_value, (int, float)):
        return float(HARD_FILTERS.get("min_rr", 1.5))
    try:
        import math
        if not math.isfinite(vix_value):
            return float(HARD_FILTERS.get("min_rr", 1.5))
    except Exception:
        return float(HARD_FILTERS.get("min_rr", 1.5))
    for tier in VIX_RR_TIERS:
        if vix_value < tier.get("vix_max", 999.0):
            return float(tier.get("min_rr", 1.5))
    return float(HARD_FILTERS.get("min_rr", 1.5))


# Regime-based R:R floor: in adverse regimes, demand higher R:R regardless of VIX.
# Applied as max(vix_min_rr, regime_rr_floor) in the post-RR safety check.
# Rationale: distribution/correction environments have lower win rates, so
# each trade must have a more favorable risk/reward to compensate.
REGIME_RR_FLOOR: Dict[str, float] = {
    "DISTRIBUTION": 2.0,   # distribution → only high-quality R:R
    "CORRECTION": 2.5,     # correction → very selective
    "BEARISH": 2.5,        # bearish → very selective
    "PANIC": 99.0,         # effectively blocks all (matches REGIME_MIN_SCORE=100)
}


# ROE quality gate: penalty zone for marginal profitability (3-8%)
# Extended from 3-5% (max 5 pts) to 3-8% (max 8 pts) so that stocks with
# ROE 5-8% (e.g. ESLT 6.3%, HAFN 6.8%, NSA 7.5%) face a meaningful penalty
# that can actually affect top-K selection, not just the absolute worst names.
ROE_QUALITY_GATE: Dict[str, float] = {
    "min_roe": 5.0,               # Stocks with ROE below this are hard-blocked (aligned with HARD_FILTERS)
    "penalty_zone_max": 10.0,     # ROE between 5% and 10% gets score penalty (was 8% — extend to penalize marginal profitability)
    "penalty_points": 10.0,       # Max penalty points for ROE at the boundary (was 8.0)
}

# Analyst consensus cross-check: penalize when system target diverges
# significantly from Wall Street consensus, especially when analysts
# think a stock is already overvalued (negative upside).
ANALYST_TARGET_PENALTY: Dict[str, object] = {
    "enabled": True,
    "overestimate_threshold": 0.20,           # system target exceeds analyst by >20%
    "penalty_points": 8.0,                    # score penalty for overestimate (was 5.0)
    "extreme_overestimate_threshold": 0.40,   # system target exceeds analyst by >40%
    "extreme_penalty_points": 12.0,           # severe penalty for extreme divergence
    "negative_upside_penalty": 12.0,          # analyst PT < current price (was 8.0)
    "hard_block_negative_upside": True,       # BLOCK stock entirely when analyst consensus PT < current price (catches M&A inflated prices, overvalued stocks)
    "hard_block_threshold": -0.10,            # block when analyst upside < -10% (significant overvaluation per consensus)
    "min_analysts_for_block": 3,              # require at least 3 analysts to trust the consensus
}

# News sentiment penalty: penalize stocks with strongly negative recent news.
# Uses existing fetch_news_sentiment_finnhub() from core/sentiment_data.py.
# Applied as a post-score adjustment in the pipeline runner (like analyst penalty).
NEWS_SENTIMENT_PENALTY: Dict[str, object] = {
    "enabled": True,
    "negative_threshold": -0.15,        # sentiment_avg below this → negative news
    "negative_penalty": 5.0,            # score penalty for negative sentiment
    "strong_negative_threshold": -0.30,  # strongly negative news
    "strong_negative_penalty": 8.0,     # larger penalty
    "min_news_count": 0.03,             # normalized; need at least ~3 articles for signal
}

# Crisis-momentum penalty: stocks in geopolitically-sensitive sectors that
# rallied sharply (e.g. tankers during Hormuz crisis) risk reversal when
# the crisis de-escalates.  Penalize high-momentum crisis-sector stocks
# to temper score inflation from event-driven momentum.
CRISIS_MOMENTUM_PENALTY: Dict[str, object] = {
    "enabled": True,
    "sectors": ["Energy", "Materials"],   # sectors most sensitive to geopolitical events
    "high_momentum_threshold": 0.20,      # 20%+ return in 20d → crisis-driven rally
    "penalty_points": 4.0,               # base score penalty
    "rsi_amplifier_threshold": 65.0,     # if RSI also elevated → double penalty
}

# Volume confirmation in distribution/correction: penalize stocks where
# up-day volume is weak relative to down-day volume — a classic sign of
# distribution where rallies lack conviction and sell-offs have conviction.
DISTRIBUTION_VOLUME_PENALTY: Dict[str, object] = {
    "enabled": True,
    "regimes": ["distribution", "correction"],
    "up_down_volume_ratio_threshold": 0.85,   # up-day vol / down-day vol
    "penalty_points": 3.0,                     # score penalty
    "lookback_days": 20,                       # days of volume to analyze
}

# Reliability band thresholds (used by v2_risk_engine and classification)
RELIABILITY_BANDS: Dict[str, int] = {
    "high_min": 65,     # lowered from 75 to allow more differentiation
    "medium_min": 45,   # raised from 40
}

# Minimum reliability for CORE classification
CORE_MIN_RELIABILITY: float = 55.0

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
# Regime-aware minimum scores: in weaker regimes, demand higher quality.
# Prevents recommending stocks that merely "survived" a penalty multiplier.
REGIME_MIN_SCORE: Dict[str, float] = {
    "TREND_UP": 55.0,
    "BULLISH": 55.0,
    "MODERATE_UP": 60.0,
    "SIDEWAYS": 70.0,   # raised from 65 — in neutral markets demand higher quality
    "NEUTRAL": 70.0,    # raised from 65
    "DISTRIBUTION": 75.0,
    "CORRECTION": 80.0,
    "BEARISH": 80.0,
    "PANIC": 100.0,  # effectively blocks all recommendations
}
# Minimum FinalScore_20d for pattern-only signal bypass.
# Prevents low-score stocks from passing solely due to marginal pattern matches.
PATTERN_MIN_SCORE: float = 35.0
# ML probability threshold to qualify as a candidate even if final score is lower
ML_PROB_THRESHOLD: float = 0.70  # was 0.62 — raised because ML AUC is only 0.63, bypass should require strong conviction

# Regime-aware bypass policy: in strict regimes (effective_min_score >= this),
# disable ML and pattern bypasses entirely — stocks MUST meet the regime score.
# Without this, any stock with ML_prob >= 0.62 or Pattern_Score > 0 bypasses
# the quality gate, making REGIME_MIN_SCORE ineffective.
BYPASS_DISABLED_ABOVE_MIN_SCORE: float = 70.0
# Default top-K limit for ranked signal output
TOP_SIGNAL_K: int = 30
# Minimum results to always return (with LowConfidence flag if thresholds not met)
MIN_FALLBACK_K: int = 5
# VIX-aware max signals: higher VIX → fewer recommendations (force concentration).
# Applied as an additional cap on top of REGIME_MIN_SCORE filtering.
VIX_MAX_SIGNALS: list = [
    {"vix_max": 20.0, "max_signals": 30},  # calm market: show all
    {"vix_max": 25.0, "max_signals": 20},  # elevated VIX: limit
    {"vix_max": 30.0, "max_signals": 12},  # high VIX: concentrate on best
    {"vix_max": 999.0, "max_signals": 8},  # extreme VIX: only highest conviction
]
# Technical score threshold indicating strong momentum
TECH_STRONG_THRESHOLD: float = 65.0

# Time-stop: exit stagnant positions that haven't made enough progress toward target.
# After halfway_days, if the position has made less than min_progress_pct of the
# distance from entry to target, exit (reason="time_stop").
# Rationale: capital stuck in a non-moving stock should be redeployed.
# Set min_progress_pct=0.0 to disable without removing the feature.
TIME_STOP_CONFIG: Dict[str, float] = {
    "halfway_days": 15,           # evaluate at 75% of holding period (was 10 — too early)
    "min_progress_pct": 0.15,     # only exit truly stagnant positions (<15% of target; was 0.30)
    "buffer_days": 3,             # don't trigger within this many days of expiry
}

# Break-even and trailing stop: lock in gains as a position advances toward target.
# Both are applied as stop_price updates (never moving stop DOWN).
# Triggers set conservatively: only activate near target to avoid cutting momentum winners.
TRAILING_STOP_CONFIG: Dict[str, float] = {
    "breakeven_trigger_pct": 0.80,  # move stop to entry when 80% of target reached (was 0.50)
    "trail_trigger_pct": 0.90,      # start trailing stop when 90% of target reached (was 0.75)
    "trail_atr_mult": 2.0,          # wider trail = less noise-stopped (was 1.5)
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
