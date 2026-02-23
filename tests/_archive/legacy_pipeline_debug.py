#!/usr/bin/env python
"""Debug run_scan_pipeline by adding logging (legacy script)."""

import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from core.config import get_config
from core.pipeline_runner import run_scan_pipeline, fetch_history_bulk

CONFIG = get_config()
config_dict = {k: v for k, v in CONFIG.__dict__.items()} if hasattr(CONFIG, '__dict__') else CONFIG

universe = ["AAPL", "MSFT", "JPM"]

data_map = fetch_history_bulk(
    universe,
    config_dict.get("lookback_days", 200),
    config_dict.get("ma_long", 200),
)

results, returned_data_map = run_scan_pipeline(
    universe=universe,
    config=config_dict,
    status_callback=lambda msg: print(f"[Status] {msg}"),
    data_map=data_map,
)
print(f"Results shape: {results.shape}")
