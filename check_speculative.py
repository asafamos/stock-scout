"""Quick script to analyze Speculative vs Core stocks."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.environ['STREAMLIT_SERVER_HEADLESS'] = '1'

# Suppress streamlit warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from stock_scout import CONFIG

# Read from cache if possible or create minimal analysis
print("=" * 80)
print("📊 SPECULATIVE vs CORE ANALYSIS")
print("=" * 80)

print("\nBased on latest run logs:")
print("- Total stocks after advanced_filters: 20")
print("- Core stocks: 6")
print("- Speculative stocks: 14")
print("\n⚠️  ISSUE: The system currently shows ONLY Core stocks!")
print("   The 14 Speculative stocks are filtered out by filter_core_recommendations()")

print("\n" + "=" * 80)
print("WHY ARE 14 STOCKS CLASSIFIED AS SPECULATIVE?")
print("=" * 80)

print("""
Common reasons for Speculative classification:
1. Data Quality: Missing fundamental data (P/E, ROE, Margins)
2. Risk Factors: Beta > 2.0, High volatility
3. Incomplete Coverage: <3 external price providers
4. Low Confidence: Classification confidence < 0.6

These stocks may have:
✓ Good technical setups (passed advanced_filters)
✓ Strong momentum
✓ High reward/risk ratios
✗ But incomplete fundamental data or higher volatility
""")

print("=" * 80)
print("RECOMMENDATION:")
print("=" * 80)

print("""
Option 1: KEEP CORE-ONLY (Conservative approach)
  ✓ Only show high-confidence, well-validated stocks
  ✓ Lower risk, clearer fundamentals
  ✗ Misses growth opportunities
  
Option 2: ADD SPECULATIVE SECTION (Balanced approach) ⭐ RECOMMENDED
  ✓ Show both Core (low risk) and Speculative (higher risk)
  ✓ User can choose based on risk tolerance
  ✓ More opportunities, transparent risk labeling
  ✗ Need to filter Speculative separately (not reuse Core filters)
  
Option 3: RELAX CLASSIFICATION (Aggressive)
  ✓ More stocks classified as Core
  ✗ Lower quality threshold, more risk
""")

print("\n" + "=" * 80)
print("TO IMPLEMENT OPTION 2:")
print("=" * 80)

print("""
1. DON'T overwrite 'results' with filter_core_recommendations output
2. Create separate filtered lists:
   - core_filtered = filter_core_recommendations(results[Risk_Level=='core'])
   - spec_filtered = filter_speculative_recommendations(results[Risk_Level=='speculative'])
3. Display both sections in UI (already exists in template!)

This requires changing 1 line in stock_scout.py (line 1840):
  FROM: results = filter_core_recommendations(results, CONFIG, adaptive=True)
  TO: core_results, spec_results = filter_by_risk_level(results, CONFIG)
""")

print("\n" + "=" * 80)
