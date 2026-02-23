#!/usr/bin/env python
"""Test fetch_history_bulk from pipeline_runner"""

from core.pipeline_runner import fetch_history_bulk

tickers = ["AAPL", "MSFT", "JPM"]
period_days = 90
ma_long = 50

print("[1] Calling fetch_history_bulk from pipeline_runner...")
data_map = fetch_history_bulk(tickers, period_days, ma_long)

print(f"[2] Result:")
print(f"    data_map type: {type(data_map)}")
print(f"    data_map length: {len(data_map)}")
print(f"    data_map keys: {list(data_map.keys())}")

if data_map:
    for tkr, df in data_map.items():
        print(f"    {tkr}: {len(df)} rows")
else:
    print(f"    ❌ EMPTY!")
