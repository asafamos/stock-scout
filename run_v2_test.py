#!/usr/bin/env python3
"""
V2 Test Runner - Verify Risk-Aware Behavior
============================================

Usage:
    python run_v2_test.py TICKER
    
Example:
    python run_v2_test.py AAPL
"""

import sys
import json
import pandas as pd
import yfinance as yf
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.v2_risk_engine import (
    calculate_reliability_v2,
    calculate_risk_gate_v2,
    calculate_position_size_v2,
    apply_v2_conviction_adjustments,
    score_ticker_v2_enhanced
)


def test_ticker_v2(ticker: str, budget: float = 5000.0):
    """Test V2 scoring for a single ticker."""
    
    print(f"\n{'='*60}")
    print(f"V2 RISK ENGINE TEST: {ticker}")
    print(f"{'='*60}\n")
    
    # Fetch basic data
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="3mo")
        
        if hist.empty:
            print(f"❌ No price data for {ticker}")
            return
        
        current_price = hist['Close'].iloc[-1]
        
    except Exception as e:
        print(f"❌ Failed to fetch data for {ticker}: {e}")
        return
    
    # Create mock row with various data quality scenarios
    scenarios = [
        {
            "name": "HIGH QUALITY (Good R/R, Multiple Sources)",
            "row": pd.Series({
                "Ticker": ticker,
                "Price_Yahoo": current_price,
                "Price_Mean": current_price,
                "Price_STD": current_price * 0.005,  # 0.5% variance
                "Price_Sources_Count": 4,
                "Fundamental_Sources_Count": 3,
                "Fund_from_FMP": True,
                "Fund_from_Finnhub": True,
                "Fund_from_SimFin": True,
                "PE_f": 25.0,
                "PS_f": 5.0,
                "ROE_f": 20.0,
                "ROIC_f": 15.0,
                "GM_f": 40.0,
                "ProfitMargin": 20.0,
                "DE_f": 0.5,
                "RevG_f": 15.0,
                "EPSG_f": 20.0,
                "RevenueGrowthYoY": 12.0,
                "EPSGrowthYoY": 18.0,
                "PBRatio": 3.0,
                "Fundamental_S": 75.0,
                "Quality_Score": 45.0,
                "Quality_Score_F": 45.0,
                "RewardRisk": 2.5,
                "RR_Ratio": 2.5,
                "Score": 75.0,
                "Score_Tech": 75.0,
                "Risk_Level": "core",
                "Unit_Price": current_price,
                "סכום קנייה ($)": 500.0,
                "ML_Probability": 0.75
            })
        },
        {
            "name": "LOW QUALITY (Poor R/R, No Fund Sources)",
            "row": pd.Series({
                "Ticker": ticker,
                "Price_Yahoo": current_price,
                "Price_Mean": current_price,
                "Price_STD": current_price * 0.05,  # 5% variance - high
                "Price_Sources_Count": 1,
                "Fundamental_Sources_Count": 0,
                "Fund_from_FMP": False,
                "Fund_from_Finnhub": False,
                "Fund_from_SimFin": False,
                "PE_f": None,
                "PS_f": None,
                "ROE_f": None,
                "RewardRisk": 0.23,
                "RR_Ratio": 0.23,
                "Score": 45.0,
                "Score_Tech": 45.0,
                "Quality_Score": 19.0,
                "Quality_Score_F": 19.0,
                "Risk_Level": "speculative",
                "Unit_Price": current_price,
                "סכום קנייה ($)": 500.0,
                "ML_Probability": 0.45
            })
        },
        {
            "name": "MEDIUM QUALITY (Moderate R/R, 1 Source)",
            "row": pd.Series({
                "Ticker": ticker,
                "Price_Yahoo": current_price,
                "Price_Mean": current_price,
                "Price_STD": current_price * 0.02,  # 2% variance
                "Price_Sources_Count": 2,
                "Fundamental_Sources_Count": 1,
                "Fund_from_FMP": True,
                "PE_f": 30.0,
                "PS_f": 3.0,
                "ROE_f": 12.0,
                "GM_f": 25.0,
                "Fundamental_S": 55.0,
                "RewardRisk": 1.6,
                "RR_Ratio": 1.6,
                "Score": 60.0,
                "Score_Tech": 60.0,
                "Quality_Score": 30.0,
                "Quality_Score_F": 30.0,
                "Risk_Level": "speculative",
                "Unit_Price": current_price,
                "סכום קנייה ($)": 500.0,
                "ML_Probability": 0.60
            })
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'─'*60}")
        print(f"📊 SCENARIO: {scenario['name']}")
        print(f"{'─'*60}")
        
        row = scenario['row']
        
        # Run V2 scoring
        result = score_ticker_v2_enhanced(
            ticker,
            row,
            budget_total=budget,
            min_position=50.0,
            enable_ml=True
        )
        
        # Display results
        print(f"\n🔍 INPUT DATA:")
        print(f"  • R/R Ratio: {row.get('RewardRisk', 0):.2f}")
        print(f"  • # Fund Sources: {row.get('Fundamental_Sources_Count', 0)}")
        print(f"  • # Price Sources: {row.get('Price_Sources_Count', 0)}")
        print(f"  • Quality Score: {row.get('Quality_Score', 0):.0f}/50")
        print(f"  • Risk Level: {row.get('Risk_Level', 'unknown')}")
        print(f"  • Base Allocation: ${row.get('סכום קנייה ($)', 0):.0f}")
        
        print(f"\n📈 V2 RELIABILITY:")
        print(f"  • Reliability v2: {result['reliability_v2']:.1f}/100")
        print(f"  • Fund Completeness: {result['fund_completeness_pct']:.1f}%")
        print(f"  • Price Variance Penalty: {result['price_variance_penalty']:.1f}")
        
        print(f"\n🚦 V2 RISK GATE:")
        print(f"  • Gate Status: {result['risk_gate_status_v2']}")
        print(f"  • Gate Penalty: {result['risk_gate_penalty_v2']:.2f}x")
        
        print(f"\n🎯 V2 CONVICTION:")
        print(f"  • Base Conviction: {result['conviction_v2_base']:.1f}/100")
        print(f"  • Final Conviction: {result['conviction_v2_final']:.1f}/100")
        print(f"  • ML Boost: {result['ml_boost_v2']:+.1f}")
        
        print(f"\n💰 V2 POSITION SIZING:")
        print(f"  • Buy Amount v2: ${result['buy_amount_v2']:.2f}")
        print(f"  • Shares v2: {result['shares_to_buy_v2']}")
        print(f"  • Unit Price: ${row.get('Unit_Price', 0):.2f}")
        
        # Verdict
        if result['risk_gate_status_v2'] == 'blocked':
            verdict = "❌ BLOCKED - Do not trade"
        elif result['buy_amount_v2'] < 100:
            verdict = "⚠️ SEVERELY REDUCED - Minimal position only"
        elif result['buy_amount_v2'] < 300:
            verdict = "⚡ REDUCED - Small position"
        else:
            verdict = "✅ APPROVED - Normal position"
        
        print(f"\n{verdict}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_v2_test.py TICKER")
        print("Example: python run_v2_test.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    test_ticker_v2(ticker)
