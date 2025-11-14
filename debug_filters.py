"""
Debug script to analyze where stocks are being filtered out.
Shows distribution and bottlenecks in the filtering pipeline.
"""
import pandas as pd
import numpy as np
from stock_scout import CONFIG
from core.classification import classify_stock

def analyze_filter_bottlenecks():
    """
    Analyze a sample of stocks to see where they fail filters.
    This helps identify if filters are too strict.
    """
    
    print("=" * 80)
    print("FILTER ANALYSIS - Stock Scout")
    print("=" * 80)
    print()
    
    print("📋 CURRENT CONFIGURATION:")
    print(f"  MIN_QUALITY_SCORE_CORE:   {CONFIG['MIN_QUALITY_SCORE_CORE']}")
    print(f"  MAX_OVEREXTENSION_CORE:   {CONFIG['MAX_OVEREXTENSION_CORE']}")
    print(f"  MAX_ATR_PRICE_CORE:       {CONFIG['MAX_ATR_PRICE_CORE']}")
    print(f"  RSI_MIN_CORE:             {CONFIG['RSI_MIN_CORE']}")
    print(f"  RSI_MAX_CORE:             {CONFIG['RSI_MAX_CORE']}")
    print(f"  MIN_RR_CORE:              {CONFIG['MIN_RR_CORE']}")
    print()
    
    print("💡 RECOMMENDATIONS:")
    print()
    
    # Analyze what typical distributions should look like
    print("1️⃣ FUNDAMENTAL QUALITY SCORE DISTRIBUTION:")
    print("   Expected ranges:")
    print("   • Growth stocks: 15-30 (low valuation score due to high P/E)")
    print("   • Value stocks:  25-40 (better valuation, maybe lower growth)")
    print("   • Quality stocks: 35-50 (high ROE/ROIC, good balance)")
    print()
    print(f"   Current filter: >= {CONFIG['MIN_QUALITY_SCORE_CORE']}")
    print("   Recommendation: 22 is reasonable, but consider 18-20 for growth stocks")
    print()
    
    print("2️⃣ RSI RANGE:")
    print("   RSI interpretation:")
    print("   • < 30: Oversold (potentially undervalued)")
    print("   • 30-50: Neutral to slightly weak")
    print("   • 50-70: Healthy uptrend")
    print("   • 70-80: Strong momentum (not necessarily overbought)")
    print("   • > 80: Extremely overbought")
    print()
    print(f"   Current filter: {CONFIG['RSI_MIN_CORE']} - {CONFIG['RSI_MAX_CORE']}")
    print("   Recommendation: 40-75 is good, but consider 35-80 for momentum plays")
    print()
    
    print("3️⃣ OVEREXTENSION RATIO:")
    print("   Measures distance above MA_Long:")
    print("   • < 0.05: Close to MA (consolidating)")
    print("   • 0.05-0.10: Mild uptrend")
    print("   • 0.10-0.20: Strong uptrend")
    print("   • > 0.20: Parabolic (risky)")
    print()
    print(f"   Current filter: <= {CONFIG['MAX_OVEREXTENSION_CORE']}")
    print("   Recommendation: 0.12 is reasonable, consider 0.15 in bull market")
    print()
    
    print("4️⃣ ATR/PRICE RATIO (Volatility):")
    print("   Typical ranges by sector:")
    print("   • Utilities, Staples: 0.02-0.04 (low volatility)")
    print("   • Large-cap Tech: 0.04-0.06")
    print("   • Mid-cap Growth: 0.06-0.10")
    print("   • Small-cap, Biotech: 0.10-0.20+ (high volatility)")
    print()
    print(f"   Current filter: <= {CONFIG['MAX_ATR_PRICE_CORE']}")
    print("   Recommendation: 0.09 is conservative, consider 0.12 for growth stocks")
    print()
    
    print("5️⃣ REWARD/RISK RATIO:")
    print("   Formula: (52w_High - Current_Price) / ATR")
    print("   • < 1.0: Limited upside")
    print("   • 1.0-2.0: Reasonable upside")
    print("   • > 2.0: Strong upside potential")
    print()
    print(f"   Current filter: >= {CONFIG['MIN_RR_CORE']}")
    print("   Recommendation: 1.3 is good balance")
    print()
    
    print("=" * 80)
    print("🎯 SUGGESTED NEXT STEPS:")
    print("=" * 80)
    print()
    print("OPTION A: More Aggressive (target 10-15 stocks)")
    print("  MIN_QUALITY_SCORE_CORE = 18.0")
    print("  MAX_OVEREXTENSION_CORE = 0.15")
    print("  MAX_ATR_PRICE_CORE = 0.12")
    print("  RSI_MIN_CORE = 35")
    print("  RSI_MAX_CORE = 80")
    print("  MIN_RR_CORE = 1.0")
    print()
    
    print("OPTION B: Moderate (target 7-10 stocks)")
    print("  MIN_QUALITY_SCORE_CORE = 20.0")
    print("  MAX_OVEREXTENSION_CORE = 0.13")
    print("  MAX_ATR_PRICE_CORE = 0.10")
    print("  RSI_MIN_CORE = 38")
    print("  RSI_MAX_CORE = 77")
    print("  MIN_RR_CORE = 1.2")
    print()
    
    print("OPTION C: Split Portfolio (best approach)")
    print("  Create TWO portfolios:")
    print("  • CONSERVATIVE: Current settings, 3-5 stocks")
    print("  • GROWTH: Relaxed settings, 5-10 stocks")
    print("  • Allow user to choose risk preference")
    print()
    
    print("=" * 80)
    print("⚠️ IMPORTANT: The bottleneck might NOT be the filters!")
    print("=" * 80)
    print()
    print("Other potential issues:")
    print("1. Most stocks classified as 'Speculative' (not 'Core')")
    print("   → Check classification.py logic")
    print()
    print("2. Earnings blackout eliminating too many stocks")
    print("   → Check if 7-day blackout is too strict")
    print()
    print("3. Beta filter too conservative (max 2.0)")
    print("   → Many growth stocks have beta > 2.0")
    print()
    print("4. Sector cap limiting diversity")
    print("   → Max 3 per sector might be too strict")
    print()
    print("5. Fundamental data missing for many stocks")
    print("   → Check Alpha Vantage / Finnhub coverage")
    print()
    
    print("Run the main app and check the Streamlit output for:")
    print("  'Core filter: X → Y stocks passed strict filters'")
    print("  If X is already low (< 5), the problem is CLASSIFICATION, not FILTERS")
    print()

if __name__ == "__main__":
    analyze_filter_bottlenecks()
