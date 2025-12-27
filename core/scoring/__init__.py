"""
Unified Scoring Module

This module aggregates all scoring functions from across the codebase:
- Technical scoring (V2)
- ML scoring (V3)
- Fundamental scoring
- Risk scoring (V2)
- Final score computation
"""

# ============================================================================
# TECHNICAL SCORING 
# ============================================================================
from core.unified_logic import (
    build_technical_indicators,
    compute_technical_score,
)
# Note: apply_technical_filters is in core.filters, not unified_logic

# ============================================================================
# ML SCORING (from core.ml_20d_inference + core.ml_features_v3)
# ============================================================================
from core.ml_20d_inference import (
    predict_20d_prob_from_row,
    apply_live_v3_adjustments,
    ML_20D_AVAILABLE,
)
from core.ml_features_v3 import (
    compute_multi_period_returns,
    compute_relative_strength_features,
)

# ============================================================================
# FINAL SCORE COMPUTATION (from core.scoring_pipeline_20d)
# ============================================================================
from core.scoring_pipeline_20d import (
    compute_final_scores_20d,
    apply_20d_sorting,
    score_universe_20d,
)

# ============================================================================
# FUNDAMENTAL SCORING (from core.scoring.fundamental)
# ============================================================================
from core.scoring.fundamental import (
    compute_fundamental_score_with_breakdown,
)

# ============================================================================
# RISK SCORING (from core.v2_risk_engine)
# ============================================================================
from core.v2_risk_engine import (
    calculate_reliability_v2,
    score_ticker_v2_enhanced,
)

# ============================================================================
# LEGACY SCORING (from core.scoring_engine - for backward compatibility)
# ============================================================================
from core.scoring_engine import (
    evaluate_rr_unified,
    compute_overall_score,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Technical Scoring V2
    "build_technical_indicators",
    "compute_tech_score_20d_v2",
    "apply_technical_filters",
    
    # ML Scoring V3
    "predict_20d_prob_from_row",
    "apply_live_v3_adjustments",
    "ML_20D_AVAILABLE",
    "compute_multi_period_returns",
    "compute_relative_strength_features",
    
    # Final Scores 20d
    "compute_final_scores_20d",
    "apply_20d_sorting",
    "score_universe_20d",
    
    # Fundamental Scoring
    "compute_fundamental_score_with_breakdown",
    
    # Risk Scoring V2
    "calculate_reliability_v2",
    "score_ticker_v2_enhanced",
    
    # Legacy (for backward compatibility)
    "evaluate_rr_unified",
    "compute_overall_score",
]
