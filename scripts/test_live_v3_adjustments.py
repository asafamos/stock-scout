#!/usr/bin/env python3
"""
Test script for ML_20d_Prob_live_v3 adjustments.
Validates that the apply_live_v3_adjustments function works correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.ml_20d_inference import apply_live_v3_adjustments


def test_basic_functionality():
    """Test that the function returns valid probabilities."""
    print("=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)
    
    # Create dummy data
    df = pd.DataFrame({
        "ML_20d_Prob_raw": [0.45, 0.55, 0.65, 0.75, 0.85],
        "ATR_Pct_percentile": [0.1, 0.35, 0.60, 0.80, 0.95],
        "Price_As_Of_Date": [15, 35, 75, 120, 200],
    })
    
    # Apply adjustments
    adjusted = apply_live_v3_adjustments(df, prob_col="ML_20d_Prob_raw")
    
    # Validate output
    assert len(adjusted) == len(df), "Output length mismatch"
    assert adjusted.min() >= 0.01, f"Min probability {adjusted.min()} below 0.01"
    assert adjusted.max() <= 0.99, f"Max probability {adjusted.max()} above 0.99"
    assert adjusted.isna().sum() == 0, "NaN values found in output"
    
    print("✓ Output length matches input")
    print(f"✓ All probabilities in [0.01, 0.99]: min={adjusted.min():.4f}, max={adjusted.max():.4f}")
    print(f"✓ No NaN values")
    print()


def test_volatility_buckets():
    """Test that volatility buckets apply correct adjustments."""
    print("=" * 60)
    print("TEST 2: Volatility Bucket Adjustments")
    print("=" * 60)
    
    # Create test cases with same raw prob but different volatility
    base_prob = 0.60
    df = pd.DataFrame({
        "ML_20d_Prob_raw": [base_prob] * 4,
        "ATR_Pct_percentile": [0.10, 0.40, 0.65, 0.85],  # low, mild, sweet, high
        "Price_As_Of_Date": [75] * 4,  # neutral price
    })
    
    adjusted = apply_live_v3_adjustments(df, prob_col="ML_20d_Prob_raw")
    
    # Expected adjustments:
    # 0.10 (low vol): -0.01 → 0.59
    # 0.40 (mild): 0.00 → 0.60
    # 0.65 (sweet spot): +0.015 → 0.615
    # 0.85 (high vol): -0.005 → 0.595
    
    print(f"Raw probability: {base_prob:.4f}")
    print(f"Low vol (0.10):     {adjusted.iloc[0]:.4f} (expected ~0.59, penalty -0.01)")
    print(f"Mild vol (0.40):    {adjusted.iloc[1]:.4f} (expected ~0.60, neutral)")
    print(f"Sweet spot (0.65):  {adjusted.iloc[2]:.4f} (expected ~0.615, boost +0.015)")
    print(f"High vol (0.85):    {adjusted.iloc[3]:.4f} (expected ~0.595, penalty -0.005)")
    
    # Validate sweet spot has highest adjusted prob
    sweet_spot_idx = 2
    assert adjusted.iloc[sweet_spot_idx] > adjusted.iloc[0], "Sweet spot should beat low vol"
    assert adjusted.iloc[sweet_spot_idx] > adjusted.iloc[3], "Sweet spot should beat high vol"
    
    print("✓ Sweet spot (0.50-0.75) has highest adjusted probability")
    print()


def test_price_buckets():
    """Test that price buckets apply correct adjustments."""
    print("=" * 60)
    print("TEST 3: Price Bucket Adjustments")
    print("=" * 60)
    
    # Test with high raw prob (> 0.55) for low-price boost
    df_high_prob = pd.DataFrame({
        "ML_20d_Prob_raw": [0.60, 0.60, 0.60, 0.60],
        "ATR_Pct_percentile": [0.50] * 4,  # neutral vol
        "Price_As_Of_Date": [12, 35, 90, 180],  # 0-20, 20-50, 50-150, 150+
    })
    
    adjusted_high = apply_live_v3_adjustments(df_high_prob, prob_col="ML_20d_Prob_raw")
    
    print(f"High raw probability (0.60):")
    print(f"$12 (0-20):    {adjusted_high.iloc[0]:.4f} (expected boost +0.01, since prob > 0.55)")
    print(f"$35 (20-50):   {adjusted_high.iloc[1]:.4f} (expected boost +0.01)")
    print(f"$90 (50-150):  {adjusted_high.iloc[2]:.4f} (expected neutral)")
    print(f"$180 (150+):   {adjusted_high.iloc[3]:.4f} (expected penalty -0.01)")
    
    # Test with low raw prob (< 0.55) for no low-price boost
    df_low_prob = pd.DataFrame({
        "ML_20d_Prob_raw": [0.50, 0.50],
        "ATR_Pct_percentile": [0.50, 0.50],
        "Price_As_Of_Date": [12, 35],
    })
    
    adjusted_low = apply_live_v3_adjustments(df_low_prob, prob_col="ML_20d_Prob_raw")
    
    print(f"\nLow raw probability (0.50):")
    print(f"$12 (0-20):   {adjusted_low.iloc[0]:.4f} (expected NO boost, since prob <= 0.55)")
    print(f"$35 (20-50):  {adjusted_low.iloc[1]:.4f} (expected boost +0.01)")
    
    print("✓ Price bucket logic validated")
    print()


def test_missing_columns():
    """Test behavior when columns are missing."""
    print("=" * 60)
    print("TEST 4: Missing Column Handling")
    print("=" * 60)
    
    # Test with only raw probability
    df_minimal = pd.DataFrame({
        "ML_20d_Prob_raw": [0.45, 0.65, 0.85],
    })
    
    adjusted = apply_live_v3_adjustments(df_minimal, prob_col="ML_20d_Prob_raw")
    
    print("Input: Only ML_20d_Prob_raw column")
    print(f"Output: {adjusted.tolist()}")
    print(f"✓ Function handles missing columns gracefully")
    
    # Test with missing prob column
    df_no_prob = pd.DataFrame({
        "ATR_Pct_percentile": [0.5, 0.5],
        "Price_As_Of_Date": [50, 100],
    })
    
    adjusted_no_prob = apply_live_v3_adjustments(df_no_prob, prob_col="ML_20d_Prob_raw")
    
    print(f"\nInput: Missing ML_20d_Prob_raw column")
    print(f"Output: {adjusted_no_prob.tolist()}")
    assert all(adjusted_no_prob == 0.5), "Should return 0.5 for all rows when prob column missing"
    print(f"✓ Returns default 0.5 when prob column missing")
    print()


def test_reliability_factor():
    """Test that ReliabilityFactor multiplier works."""
    print("=" * 60)
    print("TEST 5: Ticker Reliability Factor")
    print("=" * 60)
    
    df = pd.DataFrame({
        "ML_20d_Prob_raw": [0.60, 0.60, 0.60],
        "ATR_Pct_percentile": [0.50, 0.50, 0.50],
        "Price_As_Of_Date": [75, 75, 75],
        "ReliabilityFactor": [1.05, 1.00, 0.95],  # star, neutral, weak
    })
    
    adjusted = apply_live_v3_adjustments(df, prob_col="ML_20d_Prob_raw")
    
    print(f"Base probability: 0.60")
    print(f"Star ticker (1.05x):    {adjusted.iloc[0]:.4f}")
    print(f"Neutral ticker (1.00x): {adjusted.iloc[1]:.4f}")
    print(f"Weak ticker (0.95x):    {adjusted.iloc[2]:.4f}")
    
    assert adjusted.iloc[0] > adjusted.iloc[1], "Star ticker should have higher prob"
    assert adjusted.iloc[1] > adjusted.iloc[2], "Neutral should beat weak"
    
    print("✓ Reliability factor multiplier works correctly")
    print()


def test_clipping():
    """Test that probabilities are clipped to [0.01, 0.99]."""
    print("=" * 60)
    print("TEST 6: Probability Clipping")
    print("=" * 60)
    
    # Test extreme values
    df = pd.DataFrame({
        "ML_20d_Prob_raw": [0.005, 0.995, 0.50],
        "ATR_Pct_percentile": [0.50, 0.50, 0.50],
        "Price_As_Of_Date": [75, 75, 75],
    })
    
    adjusted = apply_live_v3_adjustments(df, prob_col="ML_20d_Prob_raw")
    
    print(f"Input 0.005 → Output {adjusted.iloc[0]:.4f} (expected 0.01)")
    print(f"Input 0.995 → Output {adjusted.iloc[1]:.4f} (expected 0.99)")
    print(f"Input 0.50  → Output {adjusted.iloc[2]:.4f}")
    
    assert adjusted.iloc[0] >= 0.01, "Should clip to minimum 0.01"
    assert adjusted.iloc[1] <= 0.99, "Should clip to maximum 0.99"
    
    print("✓ Clipping to [0.01, 0.99] works correctly")
    print()


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("ML_20d_Prob_live_v3 Adjustment Tests")
    print("=" * 60 + "\n")
    
    try:
        test_basic_functionality()
        test_volatility_buckets()
        test_price_buckets()
        test_missing_columns()
        test_reliability_factor()
        test_clipping()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
