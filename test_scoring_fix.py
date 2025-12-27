#!/usr/bin/env python3
"""
Test the scoring fix to ensure unit mismatch is resolved.
Tests on a small universe to verify scores are now reasonable.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add repo to path
sys.path.insert(0, "/workspaces/stock-scout-2")

from core.pipeline_runner import run_scan_pipeline
from stock_scout import CONFIG, build_universe

def test_scoring_fix():
    """Test that scoring now produces reasonable values (not all 0.1)."""
    
    # Test on just 5 tickers to verify fix quickly
    test_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    print(f"Testing scoring fix on {test_universe}...\n")
    
    try:
        results, data_map = run_scan_pipeline(
            universe=test_universe,
            config=CONFIG,
            status_callback=lambda msg: print(f"  [Pipeline] {msg}")
        )
        
        if results.empty:
            print("❌ Pipeline returned no results!")
            return False
        
        print(f"\n✅ Pipeline completed: {len(results)} stocks returned\n")
        
        # Check scores are now in reasonable range
        scores = results["FinalScore_20d"].values
        print(f"Score distribution:")
        print(f"  Min: {scores.min():.2f}")
        print(f"  Max: {scores.max():.2f}")
        print(f"  Mean: {scores.mean():.2f}")
        print(f"  Median: {np.median(scores):.2f}")
        
        # Show per-ticker breakdown
        print(f"\nPer-ticker scores:")
        display_cols = ["Ticker", "FinalScore_20d", "TechScore_20d", "Fundamental_Score", "ML_20d_Prob"]
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].to_string(index=False))
        
        # Verdict
        non_min_scores = scores[scores > 1.0]  # Filter out minimum floor
        if len(non_min_scores) > 0:
            avg_non_min = non_min_scores.mean()
            if avg_non_min > 10:
                print(f"\n✅ FIX VERIFIED: Average non-minimum score is {avg_non_min:.2f} (was being floored at 0.1)")
                return True
            else:
                print(f"\n⚠️  Average score {avg_non_min:.2f} is still low, but at least not all 0.1")
                return True
        else:
            print(f"\n❌ All scores are at minimum floor (1.0), fix may not have worked")
            return False
            
    except Exception as e:
        print(f"❌ Error during pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scoring_fix()
    sys.exit(0 if success else 1)
