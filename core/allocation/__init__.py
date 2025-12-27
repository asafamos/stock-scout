"""
Unified Allocation Module

This module aggregates budget allocation and position sizing functions.
"""

# ============================================================================
# BUDGET ALLOCATION (from core.portfolio)
# ============================================================================
from core.portfolio import (
    allocate_budget,
    _normalize_weights,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "allocate_budget",
    "_normalize_weights",
]
