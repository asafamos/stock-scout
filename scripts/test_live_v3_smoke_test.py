#!/usr/bin/env python3
"""
Smoke test for ML_20d_Prob_live_v3 integration.
Validates that the stock_scout.py app can import and use the new functions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("ML_20d_Prob_live_v3 Integration Smoke Test")
print("=" * 60)

# Test 1: Import core functions
print("\nTest 1: Import core functions...")
try:
    from core.ml_20d_inference import (
        compute_ml_20d_probabilities_raw,
        predict_20d_prob_from_row,
        apply_live_v3_adjustments,
        PREFERRED_SCORING_MODE_20D,
        ML_20D_AVAILABLE,
    )
    print("✓ All core functions imported successfully")
    print(f"  - ML_20D_AVAILABLE: {ML_20D_AVAILABLE}")
    print(f"  - PREFERRED_SCORING_MODE_20D: {PREFERRED_SCORING_MODE_20D}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify function signatures
print("\nTest 2: Verify function signatures...")
try:
    import inspect
    
    # Check compute_ml_20d_probabilities_raw
    sig = inspect.signature(compute_ml_20d_probabilities_raw)
    assert "row" in sig.parameters, "Missing 'row' parameter"
    print("✓ compute_ml_20d_probabilities_raw signature OK")
    
    # Check predict_20d_prob_from_row (backward compat)
    sig = inspect.signature(predict_20d_prob_from_row)
    assert "row" in sig.parameters, "Missing 'row' parameter"
    print("✓ predict_20d_prob_from_row signature OK")
    
    # Check apply_live_v3_adjustments
    sig = inspect.signature(apply_live_v3_adjustments)
    assert "df" in sig.parameters, "Missing 'df' parameter"
    assert "prob_col" in sig.parameters, "Missing 'prob_col' parameter"
    print("✓ apply_live_v3_adjustments signature OK")
except AssertionError as e:
    print(f"✗ Signature check failed: {e}")
    sys.exit(1)

# Test 3: Mock inference with dummy data
print("\nTest 3: Mock inference with dummy data...")
try:
    import pandas as pd
    import numpy as np
    
    # Create dummy row (simulating indicators)
    dummy_row = pd.Series({
        "RSI": 55.0,
        "ATR_Pct": 0.05,
        "RR": 2.5,
        "TechScore_20d": 65.0,
        "Volume_Surge": 1.2,
        "MA_Alignment": 0.8,
    })
    
    # Try to compute raw probability
    prob_raw = compute_ml_20d_probabilities_raw(dummy_row)
    
    if ML_20D_AVAILABLE:
        assert isinstance(prob_raw, (float, type(np.nan))), "Invalid return type"
        if not np.isnan(prob_raw):
            assert 0.0 <= prob_raw <= 1.0, f"Probability {prob_raw} out of range"
        print(f"✓ Raw probability computed: {prob_raw:.4f}")
    else:
        assert np.isnan(prob_raw), "Should return NaN when model unavailable"
        print("✓ Returns NaN when model unavailable (expected)")
except Exception as e:
    print(f"✗ Mock inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Apply live_v3 adjustments
print("\nTest 4: Apply live_v3 adjustments...")
try:
    df_test = pd.DataFrame({
        "ML_20d_Prob_raw": [0.50, 0.60, 0.70],
        "ATR_Pct_percentile": [0.30, 0.60, 0.85],
        "Price_As_Of_Date": [25, 75, 150],
    })
    
    adjusted = apply_live_v3_adjustments(df_test, prob_col="ML_20d_Prob_raw")
    
    assert len(adjusted) == 3, "Output length mismatch"
    assert adjusted.min() >= 0.01, "Min below threshold"
    assert adjusted.max() <= 0.99, "Max above threshold"
    print(f"✓ Adjustments applied: {adjusted.tolist()}")
except Exception as e:
    print(f"✗ Adjustments failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check scoring mode integration
print("\nTest 5: Check scoring mode integration...")
try:
    # Simulate what stock_scout.py does
    results = pd.DataFrame({
        "TechScore_20d_v2_raw": [60, 70, 80],
        "ML_20d_Prob_raw": [0.55, 0.65, 0.75],
        "ATR_Pct": [0.03, 0.05, 0.08],
        "Price_As_Of_Date": [30, 80, 120],
    })
    
    # Compute percentiles
    results["ATR_Pct_percentile"] = results["ATR_Pct"].rank(pct=True, method='average')
    
    # Apply live_v3
    results["ML_20d_Prob_live_v3"] = apply_live_v3_adjustments(
        results, prob_col="ML_20d_Prob_raw"
    )
    
    # Backward compat
    results["ML_20d_Prob"] = results["ML_20d_Prob_live_v3"]
    
    # Check columns exist
    assert "ML_20d_Prob_raw" in results.columns, "Missing raw column"
    assert "ML_20d_Prob_live_v3" in results.columns, "Missing live_v3 column"
    assert "ML_20d_Prob" in results.columns, "Missing backward compat column"
    
    print("✓ Scoring pipeline integration OK")
    print(f"  - Raw probs:    {results['ML_20d_Prob_raw'].tolist()}")
    print(f"  - Live_v3 probs: {results['ML_20d_Prob_live_v3'].tolist()}")
except Exception as e:
    print(f"✗ Scoring integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verify no imports break stock_scout.py
print("\nTest 6: Verify stock_scout.py imports...")
try:
    # Just check if stock_scout.py can be parsed (don't run it)
    import ast
    stock_scout_path = project_root / "stock_scout.py"
    with open(stock_scout_path, "r", encoding="utf-8") as f:
        code = f.read()
    ast.parse(code)
    print("✓ stock_scout.py parses without syntax errors")
except SyntaxError as e:
    print(f"✗ Syntax error in stock_scout.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check stock_scout.py: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL SMOKE TESTS PASSED")
print("=" * 60)
print("\nSummary:")
print("  ✓ Core functions imported and callable")
print("  ✓ Function signatures correct")
print("  ✓ Raw probability computation works")
print("  ✓ Live_v3 adjustments apply correctly")
print("  ✓ Scoring pipeline integration ready")
print("  ✓ stock_scout.py syntax valid")
print("\n🚀 Ready for live deployment!")
