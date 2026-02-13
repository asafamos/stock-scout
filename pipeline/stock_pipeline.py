"""
Pipeline orchestration utilities for Stock Scout (extracted from stock_scout.py)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

def run_full_pipeline(config: dict, universe: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Orchestrate the full data pipeline: build universe, fetch history, compute indicators, score, filter, and allocate.
    Args:
        config: App configuration dict (CONFIG)
        universe: Optional pre-built universe DataFrame
    Returns:
        Dict with keys: 'universe', 'history', 'scored', 'filtered', 'portfolio', 'provider_status', 'provider_usage', etc.
    """
    from core.pipeline_runner import build_universe, fetch_history_bulk, compute_indicators, score_universe, apply_filters, allocate_budget
    from core.price_verify import get_provider_status, get_provider_usage
    # 1. Build universe
    if universe is None:
        universe = build_universe(config)
    # 2. Fetch history
    history = fetch_history_bulk(universe, config)
    # 3. Compute indicators
    indicators = compute_indicators(history, config)
    # 4. Score universe
    scored = score_universe(universe, indicators, config)
    # 5. Apply filters
    filtered = apply_filters(scored, config)
    # 6. Allocate budget
    portfolio = allocate_budget(filtered, config)
    # 7. Provider status/usage
    provider_status = get_provider_status()
    provider_usage = get_provider_usage()
    return {
        'universe': universe,
        'history': history,
        'scored': scored,
        'filtered': filtered,
        'portfolio': portfolio,
        'provider_status': provider_status,
        'provider_usage': provider_usage,
    }
