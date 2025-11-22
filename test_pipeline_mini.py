#!/usr/bin/env python3
"""Minimal test: check where stock_scout hangs."""

import sys
sys.path.insert(0, '/workspaces/stock-scout-2')

print("1. Importing stock_scout...")
from stock_scout import CONFIG, build_universe, fetch_history_bulk
print("✅ Import successful")

print("\n2. Building universe...")
try:
    universe = build_universe(limit=5)
    print(f"✅ Universe: {universe}")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

print("\n3. Fetching history...")
try:
    data_map = fetch_history_bulk(universe, 90, 50)
    print(f"✅ History: {len(data_map)} stocks")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

print("\n✅ All basic steps work! Issue must be in Streamlit rendering or later stages.")
