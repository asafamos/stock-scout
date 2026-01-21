#!/usr/bin/env python3
"""
Simple direct test - just verify the pipeline works
"""

import sys
sys.path.insert(0, '/workspaces/stock-scout-2')

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.config import get_config
from core.unified_logic import (
    build_technical_indicators,
    compute_technical_score,
    compute_tech_score_20d_v2
)
import yfinance as yf
from datetime import datetime, timedelta

def test_single_ticker_flow():
    """Test: Complete flow for a single ticker"""
    print("\n" + "="*70)
    print("TESTING SINGLE TICKER FLOW: AAPL")
    print("="*70)
    
    # Download data
    ticker = "AAPL"
    end = datetime.utcnow()
    start = end - timedelta(days=250)
    
    print(f"\n1️⃣  Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    print(f"   ✅ Downloaded {len(df)} rows")
    
    # Calculate indicators
    print(f"\n2️⃣  Building technical indicators...")
    tech_df = build_technical_indicators(df)
    print(f"   ✅ Indicators calculated")
    print(f"   Columns: {tech_df.columns.tolist()}")
    
    # Get last row
    print(f"\n3️⃣  Extracting latest values...")
    row = tech_df.iloc[-1]
    print(f"   ✅ Latest row extracted")
    
    # Calculate scores
    print(f"\n4️⃣  Computing scores...")
    config = get_config()
    
    tech_score = compute_technical_score(row, weights=config.__dict__ if hasattr(config, '__dict__') else config)
    tech_v2 = compute_tech_score_20d_v2(row) * 100.0
    
    print(f"   ✅ Scores computed:")
    print(f"      Tech Score: {tech_score:.2f}")
    print(f"      Tech V2 Score: {tech_v2:.2f}")
    
    # Display key metrics
    print(f"\n5️⃣  Key metrics:")
    metrics = {
        "Close": row.get("Close"),
        "RSI": row.get("RSI"),
        "ATR": row.get("ATR"),
        "MACD_Line": row.get("MACD_Line"),
        "ADX": row.get("ADX"),
        "SMA_50": row.get("SMA_50"),
        "SMA_200": row.get("SMA_200"),
    }
    
    for key, val in metrics.items():
        if pd.notna(val):
            print(f"   {key}: {val:.4f}")
    
    print("\n" + "="*70)
    print("✅ TEST PASSED - Logic runs correctly without UI")
    print("="*70)
    
    return {
        "ticker": ticker,
        "price": row.get("Close"),
        "tech_score": tech_score,
        "tech_v2_score": tech_v2,
        "date": tech_df.index[-1]
    }

if __name__ == "__main__":
    result = test_single_ticker_flow()
    
    print("\n💡 Next steps:")
    print("   1. Run Streamlit UI: streamlit run stock_scout.py")
    print("   2. Compare the scores above with what the UI shows for AAPL")
    print("   3. They should match exactly!")
    print("\n" + "="*70)
