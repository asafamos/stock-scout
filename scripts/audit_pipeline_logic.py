import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_config
from core.unified_logic import build_technical_indicators, compute_technical_score
from advanced_filters import compute_advanced_score, should_reject_ticker, fetch_benchmark_data
# Note: We are mocking the risk engine call to verify flow logic based on imports available

def audit_single_ticker(ticker="AAPL"):
    print(f"\n🕵️  AUDITING PIPELINE LOGIC FOR: {ticker}")
    print("=" * 60)
    
    # 1. Fetch Data
    import yfinance as yf
    print("1. Fetching History...")
    df = yf.download(ticker, period="1y", progress=False)
    if df.empty:
        print("❌ No data found.")
        return

    # 2. Technical Score
    print("\n2. Computing Technical Score...")
    indicators = build_technical_indicators(df)
    latest = indicators.iloc[-1]
    config = get_config()
    tech_score = compute_technical_score(latest, weights=config.weights)
    print(f"   ➡️  Base Technical Score: {tech_score:.2f}")
    
    current_score = tech_score / 100.0 # Normalize to 0-1 like in the app
    
    # 3. Advanced Filters (Penalties)
    print("\n3. Applying Advanced Filters...")
    bench = fetch_benchmark_data("SPY", 252)
    enhanced_score, signals = compute_advanced_score(ticker, df, bench, current_score)
    
    # Simulate the penalty logic from stock_scout.py
    penalty = 0.0
    if signals.get("rs_63d", 0) < -0.3: penalty += 2.0
    if signals.get("momentum_consistency", 0) < 0.15: penalty += 2.0
    
    score_after_filters = max(0.0, enhanced_score - penalty)
    print(f"   ➡️  Score BEFORE filters: {enhanced_score:.2f}")
    print(f"   ➡️  Penalty Applied: {penalty}")
    print(f"   ➡️  Score AFTER filters: {score_after_filters:.2f}")
    
    # 4. Risk Engine Simulation
    print("\n4. Risk Engine Logic Check...")
    # In stock_scout.py, the risk engine is called. 
    # If the risk engine calculates conviction based on raw data, it might ignore step 3.
    # We check if 'score_after_filters' is passed to it.
    
    print("   ⚠️  CRITICAL CHECK:")
    print("   In stock_scout.py, 'score_ticker_v2_enhanced' is called with the row.")
    print("   If that function re-calculates the score from scratch, the penalty from step 3 is LOST.")
    
    print("\n✅ Recommendation:")
    print("   Ensure 'score_ticker_v2_enhanced' accepts an explicit 'base_score' argument")
    print("   and pass 'results.loc[idx, \"Score\"]' (the penalized score) to it.")

if __name__ == "__main__":
    audit_single_ticker("AAPL")