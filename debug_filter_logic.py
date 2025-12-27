#!/usr/bin/env python
"""Debug apply_technical_filters behavior"""

import pandas as pd
import numpy as np
from core.unified_logic import build_technical_indicators, apply_technical_filters
import yfinance as yf

# Download data for one ticker
print("[1] Downloading AAPL data...")
df = yf.download("AAPL", start="2024-10-01", end="2025-01-10", progress=False)
print(f"    Downloaded {len(df)} rows")

# Build indicators
print("\n[2] Building technical indicators...")
tech_df = build_technical_indicators(df)
print(f"    Built indicators: {tech_df.shape}")
print(f"    Columns: {list(tech_df.columns)}")

# Get last row
row_indicators = tech_df.iloc[-1]
print(f"\n[3] Last row indicators:")
print(f"    Close: {row_indicators.get('Close')}")
print(f"    RSI: {row_indicators.get('RSI')}")
print(f"    ATR: {row_indicators.get('ATR')}")
print(f"    Overextension: {row_indicators.get('Overextension')}")
print(f"    Momentum_Consensus: {row_indicators.get('Momentum_Consensus')}")
print(f"    ATR_Pct: {row_indicators.get('ATR_Pct')}")

# Apply filters with debug
print(f"\n[4] Calling apply_technical_filters(strict=False)...")
try:
    result = apply_technical_filters(row_indicators, strict=False)
    print(f"    Result: {result}")
    if not result:
        print(f"    ❌ Filter REJECTED this row!")
    else:
        print(f"    ✅ Filter PASSED this row")
except Exception as e:
    print(f"    ❌ Exception: {e}")
    import traceback
    traceback.print_exc()

# Also test with strict=True
print(f"\n[5] Calling apply_technical_filters(strict=True)...")
try:
    result = apply_technical_filters(row_indicators, strict=True)
    print(f"    Result: {result}")
    if not result:
        print(f"    ❌ Filter REJECTED this row!")
    else:
        print(f"    ✅ Filter PASSED this row")
except Exception as e:
    print(f"    ❌ Exception: {e}")
    import traceback
    traceback.print_exc()
