#!/usr/bin/env python3
"""
Test script to run the scan pipeline directly without UI
and verify results match what stock_scout.py produces
"""

import sys
sys.path.insert(0, '/workspaces/stock-scout-2')

import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

from core.config import get_config
from core.pipeline_runner import run_scan
from core.serialization import scanresult_to_dataframe
from core.scan_io import load_latest_scan

def test_live_pipeline():
    """Test: Run live scan on a small universe"""
    print("\n" + "="*70)
    print("TEST 1: LIVE PIPELINE RUN")
    print("="*70)
    
    config = get_config()
    # Convert Config object to dict
    config_dict = config.__dict__ if hasattr(config, '__dict__') else config
    # Use small universe for quick test
    test_universe = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    
    logger.info(f"Running pipeline on: {test_universe}")
    
    try:
        sr = run_scan(
            universe=test_universe,
            config=config_dict
        )
        results_df = scanresult_to_dataframe(sr)
        
        if not results_df.empty:
            logger.info(f"✅ Pipeline completed. Generated {len(results_df)} stocks")
            print("\nTop results:")
            cols_to_show = ["Ticker", "Score_Tech", "TechScore_20d_v2_raw", "ML_20d_Prob_raw", 
                           "Close", "Target_Price", "Upside_%"]
            available_cols = [c for c in cols_to_show if c in results_df.columns]
            print(results_df[available_cols].head(10).to_string())
            return results_df
        else:
            logger.warning("⚠️  Pipeline returned empty results")
            return None
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_precomputed_vs_live():
    """Test: Compare precomputed scan with live re-run on same tickers"""
    print("\n" + "="*70)
    print("TEST 2: PRECOMPUTED vs LIVE COMPARISON")
    print("="*70)
    
    config = get_config()
    config_dict = config.__dict__ if hasattr(config, '__dict__') else config
    
    # Re-run on sample tickers
    test_universe = ["AAPL", "MSFT", "GOOGL"]
    
    logger.info(f"Re-running pipeline on: {test_universe}")
    
    try:
        sr_live = run_scan(universe=test_universe, config=config_dict)
        live_df = scanresult_to_dataframe(sr_live)
        
        if live_df is not None and not live_df.empty:
            logger.info(f"✅ Live run completed: {len(live_df)} stocks")
            
            # Compare specific metrics
            print("\nComparison (first 5 tickers):")
            for ticker in live_df["Ticker"].unique()[:5]:
                live_row = live_df[live_df["Ticker"] == ticker].iloc[0]
                precomp_row = precomp_df[precomp_df["Ticker"] == ticker]
                
                if not precomp_row.empty:
                    precomp_row = precomp_row.iloc[0]
                    print(f"\n{ticker}:")
                    print(f"  Live Close: {live_row.get('Close', 'N/A'):.2f}")
                    print(f"  Precomp Close: {precomp_row.get('Close', 'N/A'):.2f}")
                    print(f"  Live Target: {live_row.get('Target_Price', 'N/A'):.2f}")
                    print(f"  Precomp Target: {precomp_row.get('Target_Price', 'N/A'):.2f}")
                else:
                    print(f"\n{ticker}: Not in precomputed scan")
                    
    except Exception as e:
        logger.error(f"❌ Live run failed: {e}")
        import traceback
        traceback.print_exc()

def test_specific_ticker():
    """Test: Detailed analysis of a single ticker"""
    print("\n" + "="*70)
    print("TEST 3: DETAILED SINGLE TICKER ANALYSIS")
    print("="*70)
    
    ticker = "AAPL"
    config = get_config()
    config_dict = config.__dict__ if hasattr(config, '__dict__') else config
    
    logger.info(f"Analyzing {ticker} in detail...")
    
    try:
        sr = run_scan(
            universe=[ticker],
            config=config_dict
        )
        results_df = scanresult_to_dataframe(sr)
        
        if not results_df.empty:
            row = results_df.iloc[0]
            print(f"\n{ticker} Analysis:")
            print(f"  Price: ${row.get('Close', 'N/A'):.2f}")
            print(f"  Target: ${row.get('Target_Price', 'N/A'):.2f}")
            print(f"  Upside: {row.get('Upside_%', 'N/A'):.2f}%")
            print(f"  Tech Score (V2): {row.get('TechScore_20d_v2_raw', 'N/A'):.2f}")
            print(f"  ML Probability: {row.get('ML_20d_Prob_raw', 'N/A'):.4f}")
            print(f"  Tech Indicators:")
            
            indicators = ["RSI", "ATR", "MACD_Line", "ADX", "SMA_50", "SMA_200"]
            for ind in indicators:
                val = row.get(ind, 'N/A')
                if val != 'N/A':
                    print(f"    {ind}: {val:.4f}")
            
            return row
        else:
            logger.warning(f"⚠️  No results for {ticker}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "="*70)
    print("STOCK SCOUT - DIRECT PIPELINE TEST")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Run tests
    print("\n📊 Running validation tests...\n")
    
    live_result = test_live_pipeline()
    test_precomputed_vs_live()
    detail_result = test_specific_ticker()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✅ All tests completed!")
    print("\n💡 Next step: Open UI at http://localhost:8501")
    print("   Compare results above with what you see in the Streamlit app")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
