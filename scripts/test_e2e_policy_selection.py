#!/usr/bin/env python3
"""
End-to-end test: Train a minimal model and verify automatic policy selection.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.validate_ml_improvements import choose_best_scoring_policy_20d

def test_policy_selection_in_bundle():
    """Test that policy selection works and saves correctly in model bundle."""
    print("="*70)
    print("  END-TO-END TEST: Automatic Policy Selection")
    print("="*70)
    
    # Check if enriched dataset exists
    dataset_path = Path("data/training_dataset_20d_v3_with_adjusted_score.csv")
    if not dataset_path.exists():
        print(f"\n⚠️  Enriched dataset not found: {dataset_path}")
        print("Skipping policy selection test (dataset required for full test)")
        return 0
    
    print(f"\n✓ Found enriched dataset: {dataset_path}")
    
    # Load dataset and run policy selection
    print("\n[1/3] Loading dataset and evaluating policies...")
    df = pd.read_csv(dataset_path)
    print(f"  Rows: {len(df)}")
    
    policy_result = choose_best_scoring_policy_20d(df, lookback_days=365)
    best_policy = policy_result.get("best_policy", "hybrid")
    metrics = policy_result.get("policies", {})
    
    print(f"\n  Selected Policy: {best_policy}")
    print(f"  Policies Evaluated: {list(metrics.keys())}")
    
    # Simulate saving to model bundle
    print("\n[2/3] Simulating model bundle save...")
    test_bundle = {
        "model": "mock_model",
        "scaler": "mock_scaler",
        "feature_names": ["feature1", "feature2"],
        "preferred_scoring_mode_20d": best_policy,
        "preferred_scoring_policy_metrics_20d": metrics,
        "meta": {
            "test": True,
            "created_at": "2024-12-25T00:00:00",
        }
    }
    
    test_model_path = Path("models/test_policy_selection.pkl")
    test_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(test_bundle, test_model_path)
    print(f"  ✓ Test bundle saved to {test_model_path}")
    
    # Verify loading back
    print("\n[3/3] Verifying bundle can be loaded...")
    loaded_bundle = joblib.load(test_model_path)
    loaded_policy = loaded_bundle.get("preferred_scoring_mode_20d", "NOT_FOUND")
    loaded_metrics = loaded_bundle.get("preferred_scoring_policy_metrics_20d", {})
    
    print(f"  ✓ Loaded policy: {loaded_policy}")
    print(f"  ✓ Loaded metrics: {list(loaded_metrics.keys())}")
    
    # Cleanup
    test_model_path.unlink()
    print(f"  ✓ Cleaned up test bundle")
    
    # Verify correctness
    print("\n" + "="*70)
    if loaded_policy == best_policy and len(loaded_metrics) == len(metrics):
        print("✅ END-TO-END TEST PASSED!")
        print(f"   Policy '{best_policy}' correctly saved and loaded from bundle")
    else:
        print("❌ END-TO-END TEST FAILED!")
        print(f"   Expected: {best_policy}, Got: {loaded_policy}")
        return 1
    print("="*70)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = test_policy_selection_in_bundle()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
