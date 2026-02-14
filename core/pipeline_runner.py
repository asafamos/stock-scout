"""Backward-compatibility shim — delegates to ``core.pipeline.*`` sub-modules.

All logic now lives in:
  core/pipeline/runner.py          – phase functions, run_scan_pipeline, run_scan
  core/pipeline/universe.py        – fetch_top_us_tickers_by_market_cap, preflight_check
  core/pipeline/market_data.py     – fetch_history_bulk, fetch_beta_vs_benchmark, market context
  core/pipeline/ticker_scoring.py  – _process_single_ticker, scoring helpers
  core/pipeline/helpers.py         – _to_float, check_earnings_blackout, RR helpers
  core/pipeline/config_norm.py     – _normalize_config
  core/pipeline/news_sentiment.py  – news + sentiment
  core/pipeline/fallback_tracking.py – legacy fallback tracking

This file only re-exports names so that ``from core.pipeline_runner import X``
continues to work without changes to callers.
"""

# ── Re-exports from core.pipeline sub-modules ────────────────────────

from core.pipeline.runner import (                        # noqa: F401
    run_scan_pipeline,
    run_scan,
    run_scan_smoke,
    main,
    _phase_init_context,
    _phase_fetch_and_tier1,
    _phase_score_and_filter,
    _phase_enrich_fundamentals,
    _phase_finalize,
    _apply_sector_mapping,
    _build_pipeline_meta,
    _PipelineContext,
)
from core.pipeline.universe import (                      # noqa: F401
    fetch_top_us_tickers_by_market_cap,
    preflight_check,
    _normalize_symbols,
    LAST_UNIVERSE_PROVIDER,
)
from core.pipeline.market_data import (                   # noqa: F401
    fetch_history_bulk,
    fetch_beta_vs_benchmark,
    _compute_global_market_context,
    _compute_sector_etf_returns,
    _get_sector_context_for_ticker,
    _initialize_ml_context,
    _step_fetch_and_prepare_base_data,
)
from core.pipeline.ticker_scoring import (                # noqa: F401
    _process_single_ticker,
    _step_compute_scores_with_unified_logic,
)
from core.pipeline.helpers import (                       # noqa: F401
    _canon_column_name,
    _compute_rr_for_row,
    _quantile_safe,
    _t2_pass_and_reasons,
    _to_float,
    check_earnings_blackout,
)
from core.pipeline.config_norm import _normalize_config   # noqa: F401
from core.pipeline.news_sentiment import (                # noqa: F401
    fetch_latest_company_news,
    analyze_sentiment_openai,
)
from core.pipeline.fallback_tracking import (             # noqa: F401
    _record_legacy_fallback,
    get_fallback_status,
)

if __name__ == "__main__":
    main()
