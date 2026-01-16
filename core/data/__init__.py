"""
Unified Data Module

This module aggregates all data fetching and processing functions:
- Technical indicators
- Historical price data
- Fundamentals data
"""

# ============================================================================
# PRICE & HISTORY (from core.data_sources_v2)
# ============================================================================
from core.data_sources_v2 import (
    fetch_price_multi_source,
    aggregate_price,
    aggregate_fundamentals,
    fetch_fundamentals_batch,
    get_fundamentals_safe,
)

# ============================================================================
# TECHNICAL INDICATORS (from core.unified_logic)
# ============================================================================
from core.unified_logic import (
    build_technical_indicators,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Price & History
    "fetch_price_multi_source",
    "aggregate_price",
    "aggregate_fundamentals",
    "fetch_fundamentals_batch",
    "get_fundamentals_safe",
    
    # Technical Indicators
    "build_technical_indicators",
]
