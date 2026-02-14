#!/usr/bin/env python
"""Debug pipeline_runner in detail"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import sys

# Get config
from core.config import get_config
CONFIG = get_config()
CONFIG_DICT = {k: v for k, v in CONFIG.__dict__.items()} if hasattr(CONFIG, '__dict__') else CONFIG

print(f"[DEBUG] CONFIG type: {type(CONFIG)}")

# Import all needed functions
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters,
    compute_technical_score,
    compute_tech_score_20d_v2,
)
from core.filters import fetch_benchmark_data


from core.risk import calculate_rr
from core.pipeline.market_data import fetch_history_bulk

# Try to import ML inference, make it optional
try:
    from core.ml_20d_inference import predict_20d_prob_from_row
    ML_20D_AVAILABLE = True
except:
    ML_20D_AVAILABLE = False
    predict_20d_prob_from_row = lambda x: 0.5

print("[1] Fetching data for 3 tickers...")
tickers = ["AAPL", "MSFT", "JPM"]
data = fetch_history_bulk(tickers, days=252)
print(f"    Fetched {len(data)} tickers")
for t in tickers:
    if t in data:
        print(f"      {t}: {len(data[t])} rows")
    else:
        print(f"      {t}: NOT FETCHED")

print("[2] Processing tickers...")
rows = []

for tkr, df in data.items():
    print(f"\n    Processing {tkr}...")
    if df.empty:
        print(f"      → DataFrame is empty, skipping")
        continue
    
    print(f"      → DataFrame has {len(df)} rows")
    
    # Build indicators
    try:
        tech_df = build_technical_indicators(df)
        row_indicators = tech_df.iloc[-1]
        print(f"      → Built indicators: {tech_df.shape}")
    except Exception as e:
        print(f"      → ❌ Failed to build indicators: {e}")
        continue
    
    # Apply filter
    print(f"      → Applying technical filters (strict=False)...")
    try:
        filter_result = apply_technical_filters(row_indicators, strict=False)
        if not filter_result:
            print(f"      → ❌ REJECTED by technical filters!")
            continue
        print(f"      → ✅ PASSED technical filters")
    except Exception as e:
        print(f"      → ❌ Exception in filter: {e}")
        traceback.print_exc()
        continue
    
    # Compute scores
    try:
        tech_score = compute_technical_score(row_indicators, weights=CONFIG_DICT.get("WEIGHTS"))
        tech_v2 = compute_tech_score_20d_v2(row_indicators) * 100.0
        ml_prob = predict_20d_prob_from_row(row_indicators) if ML_20D_AVAILABLE else 0.5
        
        row = row_indicators.to_dict()
        row["Ticker"] = tkr
        row["Score_Tech"] = tech_score
        row["TechScore_20d_v2_raw"] = tech_v2
        row["ML_20d_Prob_raw"] = ml_prob
        
        # Initial RR
        price = row.get("Close")
        atr = row.get("ATR")
        if price and atr:
            target = price + (2 * atr)
            row["RewardRisk"] = calculate_rr(price, target, atr)
        
        rows.append(row)
        print(f"      → ✅ ADDED to results (tech_v2: {tech_v2:.1f})")
    except Exception as e:
        print(f"      → ❌ Failed to compute scores: {e}")
        traceback.print_exc()
        continue

print(f"\n[3] Summary:")
print(f"    Total rows collected: {len(rows)}")
if rows:
    results = pd.DataFrame(rows)
    print(f"    Results shape: {results.shape}")
    print(f"    Tickers: {results['Ticker'].unique().tolist()}")
    print(f"\n    Sample:")
    for _, r in results.iterrows():
        print(f"      {r['Ticker']}: Close=${r.get('Close', 0):.2f}, Tech_V2={r.get('TechScore_20d_v2_raw', 0):.1f}")
else:
    print(f"    ❌ No rows added!")
