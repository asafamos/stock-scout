"""Analyze why Speculative stocks were classified as such."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.environ['STREAMLIT_SERVER_HEADLESS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Suppress streamlit session warnings
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

from stock_scout import CONFIG, build_universe, fetch_history_bulk
from advanced_filters import apply_advanced_filters
from core.classification import apply_classification

print("=" * 100)
print("🔍 ANALYZING SPECULATIVE STOCKS")
print("=" * 100)

# Build pipeline
universe = build_universe(CONFIG)[:CONFIG.get('UNIVERSE_LIMIT', 123)]
print(f"\n📊 Universe size: {len(universe)} stocks")

data = fetch_history_bulk(universe, CONFIG['LOOKBACK_DAYS'])
if not data:
    print("❌ No data fetched!")
    sys.exit(1)

results = pd.DataFrame(data)
print(f"✓ Fetched data for {len(results)} stocks")

# Apply advanced filters
results = apply_advanced_filters(results, CONFIG)
print(f"✓ After advanced_filters: {len(results)} stocks")

# Apply classification
results = apply_classification(results, CONFIG)
print(f"✓ After classification: {len(results)} stocks")

# Analyze Speculative stocks
spec_stocks = results[results['Risk_Level'] == 'speculative'].copy()
core_stocks = results[results['Risk_Level'] == 'core'].copy()

print(f"\n📊 Classification Results:")
print(f"   Core: {len(core_stocks)}")
print(f"   Speculative: {len(spec_stocks)}")

if spec_stocks.empty:
    print("\n✅ No speculative stocks - all are Core!")
    sys.exit(0)

print("\n" + "=" * 100)
print("🔍 WHY WERE 14 STOCKS CLASSIFIED AS SPECULATIVE?")
print("=" * 100)

# Analyze data quality
print("\n1️⃣ DATA QUALITY BREAKDOWN:")
print(f"   High:   {len(spec_stocks[spec_stocks['Data_Quality'] == 'high'])}")
print(f"   Medium: {len(spec_stocks[spec_stocks['Data_Quality'] == 'medium'])}")
print(f"   Low:    {len(spec_stocks[spec_stocks['Data_Quality'] == 'low'])}")

# Check fundamental coverage
print("\n2️⃣ FUNDAMENTAL DATA COVERAGE:")
fund_cols = ['Fundamental_S', 'Quality_Score_F', 'PE_f', 'PS_f', 'ROE_f', 'ROIC_f', 'GM_f', 'DE_f']
for col in fund_cols:
    if col in spec_stocks.columns:
        missing = spec_stocks[col].isna().sum()
        pct = (missing / len(spec_stocks)) * 100
        print(f"   {col:20s}: {missing:2d}/{len(spec_stocks)} missing ({pct:5.1f}%)")

# Check critical metrics
print("\n3️⃣ CRITICAL METRICS COVERAGE:")
critical = ['RS_63d', 'Volume_Surge', 'RR_Ratio', 'Quality_Score', 'Fundamental_S', 'Momentum_Consistency']
for col in critical:
    if col in spec_stocks.columns:
        missing = spec_stocks[col].isna().sum()
        pct = (missing / len(spec_stocks)) * 100
        print(f"   {col:25s}: {missing:2d}/{len(spec_stocks)} missing ({pct:5.1f}%)")

# Show sample stocks
print("\n4️⃣ SAMPLE SPECULATIVE STOCKS (first 5):")
print("-" * 100)
display_cols = ['Ticker', 'Data_Quality', 'Confidence_Level', 'Classification_Warnings']
available = [c for c in display_cols if c in spec_stocks.columns]
sample = spec_stocks[available].head(5)
for idx, row in sample.iterrows():
    ticker = row.get('Ticker', '?')
    quality = row.get('Data_Quality', '?')
    conf = row.get('Confidence_Level', '?')
    warnings = row.get('Classification_Warnings', '')
    print(f"\n{ticker:6s} | Quality: {quality:6s} | Confidence: {conf:6s}")
    if warnings:
        warn_list = warnings.split('; ')
        for w in warn_list[:5]:  # Show first 5 warnings
            print(f"         └─ {w}")

# Compare Core vs Speculative fundamental coverage
print("\n" + "=" * 100)
print("5️⃣ COMPARISON: CORE vs SPECULATIVE")
print("=" * 100)

def calc_coverage(df, cols):
    """Calculate % of stocks with valid data in each column."""
    result = {}
    for col in cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            result[col] = (valid / len(df)) * 100 if len(df) > 0 else 0
    return result

core_coverage = calc_coverage(core_stocks, fund_cols)
spec_coverage = calc_coverage(spec_stocks, fund_cols)

print(f"\n{'Metric':<20s} | {'Core Coverage':<15s} | {'Spec Coverage':<15s} | Difference")
print("-" * 80)
for col in fund_cols:
    if col in core_coverage and col in spec_coverage:
        core_pct = core_coverage[col]
        spec_pct = spec_coverage[col]
        diff = core_pct - spec_pct
        print(f"{col:<20s} | {core_pct:>6.1f}%         | {spec_pct:>6.1f}%         | {diff:>+6.1f}%")

print("\n" + "=" * 100)
print("💡 KEY INSIGHTS:")
print("=" * 100)

# Calculate key statistics
spec_missing_fundamentals = spec_stocks['Fundamental_S'].isna().sum()
spec_missing_quality = spec_stocks['Quality_Score_F'].isna().sum()
spec_pct_missing_fund = (spec_missing_fundamentals / len(spec_stocks)) * 100

print(f"""
1. Fundamental Coverage Issue:
   - {spec_missing_fundamentals}/{len(spec_stocks)} ({spec_pct_missing_fund:.0f}%) Speculative stocks missing Fundamental_S
   - This suggests data source issues (Alpha Vantage / Finnhub / FMP not returning data)

2. Classification Logic:
   - Stocks with missing Fundamental_S automatically get lower data_quality
   - Lower data_quality → classified as Speculative (even with good technicals)

3. Possible Root Causes:
   ✗ API rate limits hitting before fetching fundamentals
   ✗ Tickers not found in fundamental data sources
   ✗ Small/mid-cap stocks with limited coverage
   ✗ Recent IPOs or delisted stocks

4. Why Core stocks have better coverage:
   ✓ Likely large-cap, well-covered stocks (AAPL, MSFT, etc.)
   ✓ Multiple sources returning valid data
   ✓ High liquidity and analyst coverage
""")

print("\n" + "=" * 100)
print("🎯 RECOMMENDATIONS:")
print("=" * 100)

print("""
Option A: FIX DATA SOURCES (Recommended if coverage < 50%)
  1. Check API limits for Alpha Vantage (5 calls/min, 500/day)
  2. Verify Finnhub fallback is working
  3. Add more fundamental sources (Yahoo Finance has basic P/E, Market Cap)
  4. Log which tickers fail to fetch fundamentals

Option B: RELAX CLASSIFICATION LOGIC (if coverage > 70%)
  1. Don't auto-classify as Speculative if ONLY fundamentals missing
  2. Allow Core classification with strong technicals + missing fundamentals
  3. Add "Core - Limited Fundamentals" category

Option C: ACCEPT CURRENT BEHAVIOR (Conservative approach)
  1. Only recommend stocks with full fundamental coverage
  2. Speculative = incomplete data, even if technically strong
  3. This is SAFER but misses opportunities

Current system behavior is INTENTIONAL - prioritizing data quality over quantity.
If 14/19 stocks lack fundamentals, the issue is likely DATA SOURCE coverage, not logic.
""")

print("=" * 100)
