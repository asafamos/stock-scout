#!/usr/bin/env python3
"""
Debug: Check why pipeline filters everything out
"""

import sys
sys.path.insert(0, '/workspaces/stock-scout-2')

import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters
)

def test_single_ticker_filter(ticker):
    """Test technical filters on one ticker"""
    end = datetime.utcnow()
    start = end - timedelta(days=250)
    
    # Download
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty or len(df) < 100:
        print(f"❌ {ticker}: No data")
        return False
    
    # Build indicators
    tech_df = build_technical_indicators(df)
    row = tech_df.iloc[-1]
    
    # Check if passes filter
    strict_pass = apply_technical_filters(row, strict=True)
    relaxed_pass = apply_technical_filters(row, strict=False)
    momentum_pass = apply_technical_filters(row, relaxed=True)
    
    print(f"\n{ticker}:")
    print(f"  Strict (Conservative):  {'✅ PASS' if strict_pass else '❌ FAIL'}")
    print(f"  Relaxed (Speculative):  {'✅ PASS' if relaxed_pass else '❌ FAIL'}")
    print(f"  Momentum-First (Aggressiv): {'✅ PASS' if momentum_pass else '❌ FAIL'}")
    
    # Show metrics
    print(f"  Metrics:")
    metrics = {
        'RSI': row.get('RSI'),
        'Overext': row.get('Overext'),
        'ATR_Pct': row.get('ATR_Pct'),
        'RR': row.get('RR'),
        'MomCons': row.get('MomCons'),
    }
    
    for key, val in metrics.items():
        if pd.notna(val):
            print(f"    {key}: {val:.4f}")
        else:
            print(f"    {key}: NaN")
    
    return strict_pass or relaxed_pass or momentum_pass

def main():
    print("\n" + "="*70)
    print("DEBUG: TECHNICAL FILTERS")
    print("="*70)
    
    tickers = ["JPM", "AAPL", "MSFT", "NVDA", "TSLA"]
    
    results = []
    for ticker in tickers:
        try:
            passes = test_single_ticker_filter(ticker)
            results.append((ticker, passes))
        except Exception as e:
            print(f"\n{ticker}: Error - {e}")
            results.append((ticker, False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passing = [t for t, p in results if p]
    print(f"\nTickers passing at least one filter: {passing}")
    print(f"Total: {len(passing)} out of {len(results)}")
    
    if len(passing) == 0:
        print("\n⚠️  No tickers pass any filter!")
        print("Suggestion: Check filter thresholds in apply_technical_filters()")

if __name__ == "__main__":
    main()
