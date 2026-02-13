"""
Unified Filters Module

This module aggregates all filtering functions:
- Technical filters (3 tiers: strict/normal/relaxed)
- Advanced filters (RS, momentum, risk/reward)
- Risk gates (from v2_risk_engine)
"""

# ============================================================================
# TECHNICAL FILTERS (from core.unified_logic)
# ============================================================================
from core.unified_logic import (
    apply_technical_filters,
)

# ============================================================================
# ADVANCED FILTERS (from advanced_filters.py in project root)
# ============================================================================
from advanced_filters import (
    compute_advanced_score,
    should_reject_ticker,
    fetch_benchmark_data,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Technical Filters
    "apply_technical_filters",
    
    # Advanced Filters
    "compute_advanced_score",
    "should_reject_ticker",
    "fetch_benchmark_data",
]
