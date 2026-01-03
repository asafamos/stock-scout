"""
Unified Classification Module

This module provides stock classification (Core vs Speculative, Risk levels, Quality).
"""

# ============================================================================
# CLASSIFICATION (from core.classification)
# ============================================================================
from core.classification import (
    apply_classification,
    filter_core_recommendations,
    apply_safety_filters,
    assign_risk_class,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "apply_classification",
    "filter_core_recommendations",
    "apply_safety_filters",
    "assign_risk_class",
]
