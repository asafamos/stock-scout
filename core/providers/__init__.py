"""
Providers package — thin facades over core.data_sources_v2.

Each provider module re-exports the canonical implementation from
data_sources_v2 via lazy imports (no circular dependency).
The base module provides shared helper re-exports.
"""
from .base import _http_get_with_retry, WindowRateLimiter
from .fmp import get_fmp_fundamentals
from .finnhub import get_finnhub_fundamentals
from .tiingo import get_tiingo_fundamentals
from .alpha_vantage import get_alpha_vantage_fundamentals
from .polygon import get_polygon_fundamentals
from .eodhd import get_eodhd_fundamentals

# get_fundamentals_safe lives in core.data_sources_v2 (not here)
