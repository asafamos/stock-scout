#!/usr/bin/env python3
"""
Debug script: Check what pipeline_runner returns with actual config
"""

import sys
sys.path.insert(0, '/workspaces/stock-scout-2')

import logging
import pandas as pd
from core.config import get_config
from core.pipeline_runner import run_scan_pipeline

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def test_pipeline_debug():
    """Test pipeline with small universe"""
    print("\n" + "="*70)
    print("DEBUG: PIPELINE EXECUTION")
    print("="*70)
    
    config = get_config()
    config_dict = config.__dict__ if hasattr(config, '__dict__') else config
    
    print("\n📋 Config Settings:")
    for key in ['UNIVERSE_LIMIT', 'LOOKBACK_DAYS', 'BETA_FILTER_ENABLED', 'FUNDAMENTAL_ENABLED']:
        val = config_dict.get(key, 'N/A')
        print(f"   {key}: {val}")
    
    # Small universe for testing
    test_universe = ["JPM", "AAPL", "MSFT", "NVDA", "TSLA", "GE", "F", "BAC"]
    
    print(f"\n🔄 Running pipeline on: {test_universe}")
    print("-" * 70)
    
    try:
        results, data_map = run_scan_pipeline(
            universe=test_universe,
            config=config_dict,
            status_callback=lambda msg: print(f"   ℹ️  {msg}")
        )
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        if results is not None and not results.empty:
            print(f"\n✅ Pipeline returned: {len(results)} stocks")
            
            print(f"\nReturned tickers: {results['Ticker'].tolist()}")
            
            print(f"\nColumns available: {results.columns.tolist()[:15]}")
            
            print("\nScores:")
            score_cols = ['Score', 'conviction_v2_final', 'TechScore_20d', 'TechScore_20d_v2_raw']
            available_scores = [c for c in score_cols if c in results.columns]
            if available_scores:
                print(results[['Ticker'] + available_scores].to_string())
            
            print("\nBuy Amounts:")
            if 'buy_amount_v2' in results.columns:
                print(results[['Ticker', 'buy_amount_v2']].to_string())
            
        else:
            print(f"\n⚠️  Pipeline returned empty results!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline_debug()
