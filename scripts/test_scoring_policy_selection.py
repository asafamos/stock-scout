#!/usr/bin/env python3
"""
Test script to verify automatic scoring policy selection.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from experiments.validate_ml_improvements import choose_best_scoring_policy_20d

def main():
    print("="*70)
    print("  TESTING AUTOMATIC SCORING POLICY SELECTION")
    print("="*70)
    
    # Load the enriched dataset
    dataset_path = Path("data/training_dataset_20d_v3_with_adjusted_score.csv")
    
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("Please run the training pipeline first to generate the enriched dataset.")
        return 1
    
    print(f"\n✓ Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)[:10]}...")
    
    # Test the policy selection function
    print("\n" + "="*70)
    print("  EVALUATING SCORING POLICIES (365-day lookback)")
    print("="*70)
    
    result = choose_best_scoring_policy_20d(df, lookback_days=365)
    
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    
    best_policy = result.get("best_policy", "unknown")
    policies = result.get("policies", {})
    
    print(f"\n🏆 SELECTED POLICY: {best_policy.upper()}\n")
    
    if policies:
        print("Policy Performance Summary:")
        print("-" * 70)
        for policy_name, metrics in sorted(policies.items(), key=lambda x: x[1]["utility"], reverse=True):
            print(f"\n{policy_name.upper()}:")
            print(f"  Baseline Pos Rate: {metrics['baseline_pos_rate']:.4f}")
            print(f"  Top Decile Pos Rate: {metrics['top_pos_rate']:.4f} (+{(metrics['top_pos_rate'] - metrics['baseline_pos_rate']):.4f})")
            print(f"  Top Decile Avg Return: {metrics['top_avg_return']:.4f}")
            print(f"  Utility Score: {metrics['utility']:.4f}")
            print(f"  Evaluated Rows: {metrics['total_rows']} (Top Decile: {metrics['top_decile_count']})")
    else:
        print("\n⚠️  No policy metrics available (insufficient data or errors)")
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
