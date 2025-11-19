#!/usr/bin/env python3
"""Extract current recommendations from Stock Scout and validate them.

Runs the core logic to get today's recommendations, then validates each one.
"""

import sys
sys.path.insert(0, '/workspaces/stock-scout-2')

from datetime import datetime
import pandas as pd
from core.config import get_config
from core.unified_logic import fetch_stock_data, build_technical_indicators
from stock_scout import (
    build_universe,
    fetch_history_bulk,
    compute_advanced_score,
    classify_stock,
)
import pickle

print("🚀 Getting today's recommendations...")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

# Load config and model
CONFIG = get_config()
try:
    with open('model_xgboost_5d.pkl', 'rb') as f:
        model_data = pickle.load(f)
    print(f"✅ Loaded model: {model_data.get('model_type', 'unknown')}")
except:
    model_data = None
    print("⚠️  No ML model found")

# Build universe (limited to 50 for speed)
print("\n📊 Building stock universe...")
tickers = build_universe(limit=50)
print(f"✅ Universe: {len(tickers)} tickers")

# Fetch history
print("\n⬇️  Downloading data...")
df_dict = fetch_history_bulk(tickers, days=CONFIG['LOOKBACK_DAYS'])
print(f"✅ Downloaded: {len(df_dict)} stocks")

# Process each stock
results = []
for ticker, df in df_dict.items():
    if df is None or len(df) < 60:
        continue
    
    try:
        # Build indicators
        ind_df = build_technical_indicators(df)
        if ind_df.empty:
            continue
        
        # Get latest row
        latest = ind_df.iloc[-1]
        
        # Compute advanced score
        enhanced_score, signals = compute_advanced_score(
            latest, df, model_data, CONFIG
        )
        
        # Classify
        classification = classify_stock(signals, CONFIG)
        
        if classification in ['Core', 'Speculative']:
            results.append({
                'Ticker': ticker,
                'Classification': classification,
                'Price': latest['Close'],
                'RSI': signals.get('RSI'),
                'RR': signals.get('Risk_Reward'),
                'MomCons': signals.get('Momentum_Consistency'),
                'Score': enhanced_score,
                'ML_Prob': signals.get('ML_Probability'),
            })
    except Exception as e:
        continue

# Sort and display
df_results = pd.DataFrame(results)
if not df_results.empty:
    df_results = df_results.sort_values('Score', ascending=False)
    
    core = df_results[df_results['Classification'] == 'Core']
    spec = df_results[df_results['Classification'] == 'Speculative']
    
    print(f"\n{'='*70}")
    print(f"📋 TODAY'S RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"\n🎯 CORE ({len(core)} stocks)")
    if not core.empty:
        print(core[['Ticker', 'Price', 'RSI', 'RR', 'MomCons', 'Score']].to_string(index=False))
    
    print(f"\n⚡ SPECULATIVE ({len(spec)} stocks)")
    if not spec.empty:
        print(spec[['Ticker', 'Price', 'RSI', 'RR', 'MomCons', 'Score']].head(10).to_string(index=False))
    
    # Save to CSV
    df_results.to_csv('current_recommendations.csv', index=False)
    print(f"\n✅ Saved to: current_recommendations.csv")
    
    # Pick examples for validation
    if not core.empty:
        core_example = core.iloc[0]['Ticker']
        print(f"\n🔍 Will validate Core example: {core_example}")
    
    if not spec.empty:
        spec_example = spec.iloc[0]['Ticker']
        print(f"🔍 Will validate Speculative example: {spec_example}")
else:
    print("\n❌ No recommendations found!")
