#!/usr/bin/env python
"""Test run_scan_pipeline directly"""

import traceback
from core.config import get_config

print("[1] Getting CONFIG...")
CONFIG = get_config()
print(f"    ✅ CONFIG obtained, type={type(CONFIG)}")

print("\n[2] Importing run_scan_pipeline...")
try:
    from core.pipeline_runner import run_scan_pipeline
    print(f"    ✅ run_scan_pipeline imported successfully")
except Exception as e:
    print(f"    ❌ Failed to import: {e}")
    traceback.print_exc()
    exit(1)

print("\n[3] Calling run_scan_pipeline...")
try:
    config_dict = {k: v for k, v in CONFIG.__dict__.items()} if hasattr(CONFIG, '__dict__') else CONFIG
    results, data_map = run_scan_pipeline(
        universe=["AAPL", "MSFT", "JPM"],
        config=config_dict,
        status_callback=None
    )
    print(f"    ✅ Pipeline executed")
    print(f"       Results shape: {results.shape}")
    print(f"       Tickers: {results['Ticker'].unique().tolist() if not results.empty else 'EMPTY'}")
except Exception as e:
    print(f"    ❌ Failed to call: {e}")
    traceback.print_exc()
