"""
Unified Scoring Module for Stock Scout.

Aggregates all scoring functions with lazy imports to avoid import-time failures
when optional dependencies (sklearn, joblib) aren't available.

Preferred entry point::

    from core.scoring import UnifiedScorer, score_ticker

    scorer = UnifiedScorer(config={"enable_ml": True})
    result = scorer.score(ticker_data, indicators, fundamentals)
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# UNIFIED SCORER (Preferred Entry Point)
# ============================================================================
from core.scoring.unified_scorer import (  # noqa: F401
    UnifiedScorer,
    ScoringResult,
    score_ticker,
    score_dataframe,
)

# ============================================================================
# TECHNICAL SCORING (from decomposed modules)
# ============================================================================
from core.indicators import build_technical_indicators  # noqa: F401
from core.scoring.technical import (  # noqa: F401
    compute_technical_score,
    compute_tech_score_20d_v2,
    compute_tech_score_20d_v2_components,
)

# ============================================================================
# FINAL SCORE COMPUTATION
# ============================================================================
from core.scoring.final import (  # noqa: F401
    compute_final_score,
    compute_final_score_with_patterns,
)

# ============================================================================
# BIG WINNER
# ============================================================================
from core.scoring.big_winner import compute_big_winner_signal_20d  # noqa: F401

# ============================================================================
# FUNDAMENTAL SCORING
# ============================================================================
from core.scoring.fundamental import (  # noqa: F401
    compute_fundamental_score_with_breakdown,
)

# ============================================================================
# LEGACY SCORING (backward compatibility) — lazy-loaded to avoid circular import
# with scoring_engine → scoring.utils → scoring.__init__
# ============================================================================


# ============================================================================
# LAZY IMPORTS (ML and heavy dependencies)
# ============================================================================
# These are loaded lazily because they depend on sklearn/joblib which may
# not be available, and trigger model deserialization at import time.

def __getattr__(name: str) -> Any:
    """Lazy-load ML-dependent symbols on first access."""
    _ml_inference_names = {
        "predict_20d_prob_from_row",
        "apply_live_v3_adjustments",
        "ML_20D_AVAILABLE",
    }
    _ml_features_names = {
        "compute_multi_period_returns",
        "compute_relative_strength_features",
    }
    _scoring_pipeline_names = {
        "compute_final_scores_20d",
        "apply_20d_sorting",
        "score_universe_20d",
    }
    _risk_names = {
        "calculate_reliability_v2",
        "score_ticker_v2_enhanced",
    }
    _scoring_engine_names = {
        "evaluate_rr_unified",
        "compute_overall_score",
    }

    try:
        if name in _ml_inference_names:
            mod = importlib.import_module("core.ml_20d_inference")
            return getattr(mod, name)
        if name in _ml_features_names:
            mod = importlib.import_module("core.ml_features_v3")
            return getattr(mod, name)
        if name in _scoring_pipeline_names:
            mod = importlib.import_module("core.scoring_pipeline_20d")
            return getattr(mod, name)
        if name in _risk_names:
            mod = importlib.import_module("core.v2_risk_engine")
            return getattr(mod, name)
        if name in _scoring_engine_names:
            mod = importlib.import_module("core.scoring_engine")
            return getattr(mod, name)
    except (ImportError, TypeError) as exc:
        logger.warning("Lazy import of %s failed: %s", name, exc)
        raise AttributeError(f"module 'core.scoring' has no attribute {name!r}") from exc

    raise AttributeError(f"module 'core.scoring' has no attribute {name!r}")


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Unified Scorer (Preferred)
    "UnifiedScorer",
    "ScoringResult",
    "score_ticker",
    "score_dataframe",
    # Technical
    "build_technical_indicators",
    "compute_technical_score",
    "compute_tech_score_20d_v2",
    "compute_tech_score_20d_v2_components",
    # Final Score
    "compute_final_score",
    "compute_final_score_with_patterns",
    # Big Winner
    "compute_big_winner_signal_20d",
    # Fundamental
    "compute_fundamental_score_with_breakdown",
    # Legacy
    "evaluate_rr_unified",
    "compute_overall_score",
    # Lazy-loaded
    "predict_20d_prob_from_row",
    "apply_live_v3_adjustments",
    "ML_20D_AVAILABLE",
    "compute_multi_period_returns",
    "compute_relative_strength_features",
    "compute_final_scores_20d",
    "apply_20d_sorting",
    "score_universe_20d",
    "calculate_reliability_v2",
    "score_ticker_v2_enhanced",
]
