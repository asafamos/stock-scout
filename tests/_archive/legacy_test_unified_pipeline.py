"""
Quick verification test for unified scoring pipeline.
Runs a small scan and validates that all unified columns are present and correct.
"""
import sys
import os

# Prevent Streamlit from auto-starting
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_PORT"] = "0"

import pandas as pd
import numpy as np
from core.pipeline_runner import run_scan
from core.serialization import scanresult_to_dataframe
from core.config import get_config

def test_unified_pipeline():
    """Test that the unified pipeline produces all expected columns."""
    
    # Small test universe
    test_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    
    # Use a simple dict config
    config = {
        "BUDGET_TOTAL": 5000,
        "MIN_POSITION": 500,
        "beta_filter_enabled": False,
        "EARNINGS_BLACKOUT_DAYS": 0,
        "fundamental_enabled": True,
        "lookback_days": 100,
        "ma_long": 200,
        "beta_benchmark": "SPY",
    }
    
    print(f"\n{'='*80}")
    print("UNIFIED PIPELINE VERIFICATION TEST")
    print(f"{'='*80}")
    print(f"Testing with universe: {test_tickers}")
    
    # Run pipeline
    sr = run_scan(
        universe=test_tickers,
        config=config,
        status_callback=lambda msg: print(f"  → {msg}")
    )
    results = scanresult_to_dataframe(sr)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    # Data map removed in contracts output; results only
    print(f"Results returned: {len(results)}")
    
    if results.empty:
        print("⚠️  WARNING: No results returned (possibly all filtered out)")
        return
    
    # Check for unified score columns
    expected_cols = [
        "Ticker",
        "TechScore_20d",
        "Fundamental_Score", 
        "ML_20d_Prob",
        "FinalScore_20d",
        "ConvictionScore",
        "Reliability_Score",
        "Risk_Label",
        "Risk_Meter",
        "Score",  # Legacy alias
    ]
    
    print(f"\n{'='*80}")
    print("COLUMN VERIFICATION")
    print(f"{'='*80}")
    
    missing_cols = [col for col in expected_cols if col not in results.columns]
    if missing_cols:
        print(f"❌ MISSING COLUMNS: {missing_cols}")
    else:
        print("✅ All expected unified columns present")
    
    present_cols = [col for col in expected_cols if col in results.columns]
    print(f"\nPresent unified columns ({len(present_cols)}/{len(expected_cols)}):")
    for col in present_cols:
        print(f"  ✓ {col}")
    
    # Check that Score and FinalScore_20d are aligned
    if "Score" in results.columns and "FinalScore_20d" in results.columns:
        score_match = (results["Score"] == results["FinalScore_20d"]).all()
        if score_match:
            print("\n✅ Score column correctly aliased to FinalScore_20d")
        else:
            print("\n⚠️  WARNING: Score and FinalScore_20d don't match")
            diff_rows = results[results["Score"] != results["FinalScore_20d"]]
            print(f"   Mismatched rows: {len(diff_rows)}")
    
    # Check for legacy columns that should NOT be present
    legacy_cols = ["conviction_v2_final", "conviction_v2_base"]
    legacy_found = [col for col in legacy_cols if col in results.columns]
    if legacy_found:
        print(f"\n⚠️  WARNING: Legacy columns still present: {legacy_found}")
    else:
        print("\n✅ No legacy conviction columns found")
    
    # Show sample scores
    print(f"\n{'='*80}")
    print("SAMPLE SCORES (Top 3 by FinalScore_20d)")
    print(f"{'='*80}")
    
    score_col = "FinalScore_20d" if "FinalScore_20d" in results.columns else "Score"
    top3 = results.nlargest(3, score_col)
    
    display_cols = [c for c in ["Ticker", "TechScore_20d", "Fundamental_Score", 
                                  "ML_20d_Prob", "FinalScore_20d", "ConvictionScore",
                                  "Reliability_Score", "Risk_Label", "Risk_Meter"]
                    if c in top3.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(top3[display_cols].to_string(index=False))
    
    # Check score ranges
    print(f"\n{'='*80}")
    print("SCORE RANGE VALIDATION")
    print(f"{'='*80}")
    
    score_cols_to_check = {
        "TechScore_20d": (0, 100),
        "Fundamental_Score": (0, 100),
        "ML_20d_Prob": (0, 1),
        "FinalScore_20d": (0, 100),
        "ConvictionScore": (0, 100),
        "Reliability_Score": (0, 100),
        "Risk_Meter": (0, 100),
    }
    
    all_valid = True
    for col, (min_val, max_val) in score_cols_to_check.items():
        if col in results.columns:
            col_min = results[col].min()
            col_max = results[col].max()
            
            # Allow NaN for optional fields
            valid_range = (pd.isna(col_min) or col_min >= min_val) and \
                         (pd.isna(col_max) or col_max <= max_val)
            
            status = "✅" if valid_range else "❌"
            print(f"{status} {col:20s}: [{col_min:6.2f}, {col_max:6.2f}] (expected [{min_val}, {max_val}])")
            
            if not valid_range:
                all_valid = False
        else:
            print(f"⚠️  {col:20s}: MISSING")
    
    if all_valid:
        print("\n✅ All score ranges valid")
    else:
        print("\n❌ Some scores out of expected range")
    
    # Check allocation columns
    print(f"\n{'='*80}")
    print("ALLOCATION VERIFICATION")
    print(f"{'='*80}")
    
    alloc_cols = ["buy_amount_v2", "NumShares", "Allocation_Pct"]
    present_alloc = [c for c in alloc_cols if c in results.columns]
    
    if present_alloc:
        print(f"✅ Allocation columns present: {present_alloc}")
        if "buy_amount_v2" in results.columns:
            total_allocated = results["buy_amount_v2"].sum()
            print(f"   Total allocated: ${total_allocated:,.2f} (budget: ${config['BUDGET_TOTAL']:,.2f})")
    else:
        print("⚠️  No allocation columns found")
    
    print(f"\n{'='*80}")
    print("STRICT RULE ENFORCEMENT TEST")
    print(f"{'='*80}")
    
    # CRITICAL TEST: Score must ALWAYS equal FinalScore_20d
    if "Score" in results.columns and "FinalScore_20d" in results.columns:
        score_mismatch = results[results["Score"] != results["FinalScore_20d"]]
        if len(score_mismatch) == 0:
            print("✅ PASS: Score == FinalScore_20d for all rows")
        else:
            print(f"❌ FAIL: {len(score_mismatch)} rows have Score != FinalScore_20d")
            print(f"\nMismatched rows:")
            print(score_mismatch[["Ticker", "Score", "FinalScore_20d"]].to_string())
            raise AssertionError("STRICT RULE VIOLATED: Score must always equal FinalScore_20d")
    else:
        print("⚠️  Cannot test: Score or FinalScore_20d column missing")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_unified_pipeline()
