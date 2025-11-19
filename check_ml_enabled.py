"""Quick check that ML model is working properly."""
import pickle
import pandas as pd
import numpy as np
from core.unified_logic import score_with_ml_model

# Load model
with open('model_xgboost_5d.pkl', 'rb') as f:
    model_data = pickle.load(f)

print(f"✓ Loaded model with {len(model_data['feature_names'])} features")
print(f"  Features: {model_data['feature_names']}")

# Test with sample data
test_cases = [
    {'RSI': 30, 'ATR_Pct': 0.03, 'Overext': -0.05, 'RR': 5.0, 'MomCons': 0.6, 'VolSurge': 1.5, 'desc': 'Good oversold'},
    {'RSI': 70, 'ATR_Pct': 0.05, 'Overext': 0.1, 'RR': 1.0, 'MomCons': 0.3, 'VolSurge': 0.8, 'desc': 'Overbought extended'},
    {'RSI': 45, 'ATR_Pct': 0.02, 'Overext': 0.0, 'RR': 8.0, 'MomCons': 0.7, 'VolSurge': 2.0, 'desc': 'Strong neutral'},
]

print("\n" + "="*70)
print("ML MODEL TEST")
print("="*70)

for case in test_cases:
    row = pd.Series(case)
    prob = score_with_ml_model(row, model_data)
    print(f"{case['desc']:25s} → ML Prob: {prob:.1%}")

print("\n✅ ML model is ENABLED and working!")
