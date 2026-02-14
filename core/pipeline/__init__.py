"""core.pipeline — Split pipeline sub-modules.

Re-exports the public API for convenient access via ``core.pipeline``.
Backward compatibility is handled by the shim at ``core/pipeline_runner.py``.
"""

from core.pipeline.fallback_tracking import (
    get_fallback_status,
    reset_fallback_state,
)
from core.pipeline.helpers import (
    check_earnings_blackout,
    _to_float,
)
from core.pipeline.config_norm import _normalize_config
from core.pipeline.market_data import (
    fetch_history_bulk,
    fetch_beta_vs_benchmark,
)
from core.pipeline.universe import (
    fetch_top_us_tickers_by_market_cap,
    preflight_check,
    LAST_UNIVERSE_PROVIDER,
)
from core.pipeline.news_sentiment import (
    fetch_latest_company_news,
    analyze_sentiment_openai,
)
from core.pipeline.runner import (
    run_scan_pipeline,
    run_scan,
    run_scan_smoke,
)

__all__ = [
    "run_scan_pipeline",
    "run_scan",
    "run_scan_smoke",
    "fetch_top_us_tickers_by_market_cap",
    "preflight_check",
    "LAST_UNIVERSE_PROVIDER",
    "fetch_history_bulk",
    "fetch_beta_vs_benchmark",
    "_normalize_config",
    "check_earnings_blackout",
    "fetch_latest_company_news",
    "analyze_sentiment_openai",
    "get_fallback_status",
    "reset_fallback_state",
    "_to_float",
]
