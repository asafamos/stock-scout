#!/usr/bin/env python
"""Debug run_scan_pipeline by adding logging"""

import traceback
import logging
import sys

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("[1] Getting CONFIG...")
from core.config import get_config
CONFIG = get_config()
config_dict = {k: v for k, v in CONFIG.__dict__.items()} if hasattr(CONFIG, '__dict__') else CONFIG
print(f"    ✅ CONFIG obtained")
print(f"    CONFIG_DICT keys: {list(config_dict.keys())}")
print(f"    MA_LONG={config_dict.get('ma_long')}, LOOKBACK_DAYS={config_dict.get('lookback_days')}")

print("\n[2] Importing pipeline functions...")
try:
    from core.pipeline_runner import run_scan_pipeline, fetch_history_bulk
    print(f"    ✅ Imports successful")
except Exception as e:
    print(f"    ❌ Failed: {e}")
    traceback.print_exc()
    exit(1)

print("\n[3] Fetching data manually first...")
universe = ["AAPL", "MSFT", "JPM"]
try:
    data_map = fetch_history_bulk(
        universe, 
        config_dict.get("lookback_days", 200),
        config_dict.get("ma_long", 200)
    )
    print(f"    ✅ Fetched {len(data_map)} tickers")
    for tkr, df in data_map.items():
        print(f"       {tkr}: {len(df)} rows")
except Exception as e:
    print(f"    ❌ Failed: {e}")
    traceback.print_exc()

print("\n[4] Calling run_scan_pipeline...")
try:
    results, returned_data_map = run_scan_pipeline(
        universe=universe,
        config=config_dict,
        status_callback=lambda msg: print(f"    [Status] {msg}"),
        data_map=data_map
    )
    print(f"\n    ✅ Pipeline executed")
    print(f"       Results shape: {results.shape}")
    if not results.empty:
        print(f"       Tickers: {results['Ticker'].unique().tolist()}")
        print(f"\n       Sample rows:")
        for _, r in results.iterrows():
            print(f"         {r['Ticker']}: Tech_V2={r.get('TechScore_20d_v2_raw', 0):.1f}")
    else:
        print(f"       ❌ EMPTY RESULTS")
except Exception as e:
    print(f"\n    ❌ Failed: {e}")
    traceback.print_exc()
