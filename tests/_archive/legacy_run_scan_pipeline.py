#!/usr/bin/env python
"""Legacy script to run run_scan_pipeline directly."""
from core.config import get_config
from core.pipeline_runner import run_scan_pipeline

CONFIG = get_config()
config_dict = {k: v for k, v in CONFIG.__dict__.items()} if hasattr(CONFIG, '__dict__') else CONFIG

results, _ = run_scan_pipeline(
    universe=["AAPL", "MSFT", "JPM"],
    config=config_dict,
)
print(f"Results shape: {results.shape}")
