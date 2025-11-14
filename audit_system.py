"""Comprehensive audit of the Stock Scout recommendation system."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.environ['STREAMLIT_SERVER_HEADLESS'] = '1'

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

import pandas as pd
import numpy as np

print("=" * 100)
print("🔍 STOCK SCOUT SYSTEM AUDIT")
print("=" * 100)

# ============================================================================
# 1. CHECK DATA SOURCES CONFIGURATION
# ============================================================================
print("\n📊 1. DATA SOURCES AVAILABILITY:")
print("-" * 100)

from stock_scout import CONFIG
import os

sources = {
    "Alpha Vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "Finnhub": os.getenv("FINNHUB_API_KEY"),
    "FMP": os.getenv("FMP_API_KEY"),
    "Polygon": os.getenv("POLYGON_API_KEY"),
    "Tiingo": os.getenv("TIINGO_API_KEY"),
    "SimFin": os.getenv("SIMFIN_API_KEY"),
    "EODHD": os.getenv("EODHD_API_KEY"),
    "Marketstack": os.getenv("MARKETSTACK_API_KEY"),
    "Nasdaq Data Link": os.getenv("NASDAQ_DATA_LINK_API_KEY"),
}

for name, key in sources.items():
    status = "✅ Configured" if key else "❌ Missing"
    print(f"   {name:20s}: {status}")

# ============================================================================
# 2. CHECK SCORING WEIGHTS
# ============================================================================
print("\n" + "=" * 100)
print("⚖️  2. SCORING WEIGHTS CONFIGURATION:")
print("-" * 100)

print("\n📈 Technical Weights (CONFIG['WEIGHTS']):")
weights = CONFIG.get('WEIGHTS', {})
total_weight = sum(weights.values())
print(f"   Total weight: {total_weight:.4f} (should be ~1.0)")

for indicator, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    pct = (weight / total_weight * 100) if total_weight > 0 else 0
    print(f"   {indicator:15s}: {weight:.4f} ({pct:5.1f}%)")

print(f"\n🎯 Fundamental Weight: {CONFIG.get('FUNDAMENTAL_WEIGHT', 0.15)} ({CONFIG.get('FUNDAMENTAL_WEIGHT', 0.15)*100:.0f}%)")
print(f"   Technical Weight: {1-CONFIG.get('FUNDAMENTAL_WEIGHT', 0.15)} ({(1-CONFIG.get('FUNDAMENTAL_WEIGHT', 0.15))*100:.0f}%)")

# ============================================================================
# 3. CHECK FILTER THRESHOLDS
# ============================================================================
print("\n" + "=" * 100)
print("🔬 3. FILTER THRESHOLDS:")
print("-" * 100)

print("\n📉 Advanced Filters (advanced_filters.py):")
print(f"   Risk/Reward minimum: 0.5")
print(f"   Market underperformance: -20%")
print(f"   Momentum consistency: 0.2")
print(f"   MA alignment: 0.1")

print("\n🛡️  Core Filters (CONFIG):")
filters = [
    ("MIN_QUALITY_SCORE_CORE", "Minimum fundamental quality"),
    ("MAX_OVEREXTENSION_CORE", "Maximum overextension vs MA"),
    ("MAX_ATR_PRICE_CORE", "Maximum volatility (ATR/Price)"),
    ("RSI_MIN_CORE", "Minimum RSI"),
    ("RSI_MAX_CORE", "Maximum RSI"),
    ("MIN_RR_CORE", "Minimum Reward/Risk ratio"),
]

for key, desc in filters:
    value = CONFIG.get(key, "N/A")
    print(f"   {desc:35s}: {value}")

# ============================================================================
# 4. RUN ACTUAL PIPELINE AND ANALYZE
# ============================================================================
print("\n" + "=" * 100)
print("🔄 4. RUNNING PIPELINE TO ANALYZE ACTUAL DATA:")
print("-" * 100)

from stock_scout import build_universe, fetch_history_bulk, CONFIG
from advanced_filters import compute_advanced_score, should_reject_ticker, fetch_benchmark_data
from core.classification import apply_classification

# Build small sample
universe = build_universe(CONFIG)[:30]  # Smaller for speed
print(f"\n✓ Universe: {len(universe)} stocks")

data = fetch_history_bulk(universe, CONFIG['LOOKBACK_DAYS'])
if not data:
    print("❌ No data fetched!")
    sys.exit(1)

results = pd.DataFrame(data)
print(f"✓ Historical data: {len(results)} stocks")

# Advanced filters
benchmark_df = fetch_benchmark_data(CONFIG["BETA_BENCHMARK"], CONFIG["LOOKBACK_DAYS"])
print(f"✓ Benchmark data loaded")

print(f"\nRunning advanced filters...")
from stock_scout import data_map
advanced_keep = []
rejection_reasons = {}

for idx in results.index:
    tkr = results.at[idx, "Ticker"]
    if tkr not in data_map or benchmark_df.empty:
        advanced_keep.append(True)
        continue
    
    df = data_map[tkr]
    base_score = results.at[idx, "Score_Tech"]
    enhanced_score, signals = compute_advanced_score(tkr, df, benchmark_df, base_score)
    should_reject, reason = should_reject_ticker(signals)
    
    if should_reject:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        advanced_keep.append(False)
    else:
        # Store signals
        results.loc[idx, "RS_63d"] = signals.get("rs_63d", np.nan)
        results.loc[idx, "Volume_Surge"] = signals.get("volume_surge", np.nan)
        results.loc[idx, "Quality_Score"] = signals.get("quality_score", 0.0)
        results.loc[idx, "RR_Ratio"] = signals.get("risk_reward_ratio", np.nan)
        results.loc[idx, "Momentum_Consistency"] = signals.get("momentum_consistency", 0.0)
        advanced_keep.append(True)

results = results[advanced_keep].reset_index(drop=True)
print(f"✓ After advanced filters: {len(results)}/{len(advanced_keep)} passed")
if rejection_reasons:
    print(f"   Rejection breakdown: {rejection_reasons}")

# Fetch fundamentals (simulating the new flow)
print(f"\n✓ Would fetch fundamentals for all {len(results)} stocks")

# Apply classification
results = apply_classification(results)
print(f"✓ Classification applied")

# ============================================================================
# 5. ANALYZE CLASSIFICATION RESULTS
# ============================================================================
print("\n" + "=" * 100)
print("📊 5. CLASSIFICATION ANALYSIS:")
print("-" * 100)

core_count = len(results[results["Risk_Level"] == "core"])
spec_count = len(results[results["Risk_Level"] == "speculative"])

print(f"\nOverall: {core_count} Core, {spec_count} Speculative (total: {len(results)})")

print("\n🛡️  Core Stocks:")
if core_count > 0:
    core_stocks = results[results["Risk_Level"] == "core"]
    print(f"   Data Quality: High={len(core_stocks[core_stocks['Data_Quality']=='high'])}, "
          f"Medium={len(core_stocks[core_stocks['Data_Quality']=='medium'])}, "
          f"Low={len(core_stocks[core_stocks['Data_Quality']=='low'])}")
    
    # Show sample
    sample_cols = ["Ticker", "Score", "Data_Quality", "Confidence_Level"]
    available = [c for c in sample_cols if c in core_stocks.columns]
    print(f"\n   Sample Core stocks (first 5):")
    for _, row in core_stocks[available].head(5).iterrows():
        print(f"      {row.get('Ticker', '?'):6s} | Score: {row.get('Score', 0):5.1f} | "
              f"Quality: {row.get('Data_Quality', '?'):6s} | Confidence: {row.get('Confidence_Level', '?')}")

print(f"\n⚡ Speculative Stocks: {spec_count}")
if spec_count > 0:
    spec_stocks = results[results["Risk_Level"] == "speculative"]
    print(f"   Data Quality: High={len(spec_stocks[spec_stocks['Data_Quality']=='high'])}, "
          f"Medium={len(spec_stocks[spec_stocks['Data_Quality']=='medium'])}, "
          f"Low={len(spec_stocks[spec_stocks['Data_Quality']=='low'])}")
else:
    print(f"   No speculative stocks in this sample")

# ============================================================================
# 6. SCORING ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("🎯 6. SCORING DISTRIBUTION:")
print("-" * 100)

if "Score_Tech" in results.columns:
    print(f"\nTechnical Scores:")
    print(f"   Mean:   {results['Score_Tech'].mean():.2f}")
    print(f"   Median: {results['Score_Tech'].median():.2f}")
    print(f"   Min:    {results['Score_Tech'].min():.2f}")
    print(f"   Max:    {results['Score_Tech'].max():.2f}")

if "Fundamental_S" in results.columns:
    fund_scores = results['Fundamental_S'].dropna()
    if len(fund_scores) > 0:
        print(f"\nFundamental Scores:")
        print(f"   Mean:   {fund_scores.mean():.2f}")
        print(f"   Median: {fund_scores.median():.2f}")
        print(f"   Min:    {fund_scores.min():.2f}")
        print(f"   Max:    {fund_scores.max():.2f}")
        print(f"   Coverage: {len(fund_scores)}/{len(results)} ({len(fund_scores)/len(results)*100:.0f}%)")

# ============================================================================
# 7. RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 100)
print("💡 7. SYSTEM AUDIT FINDINGS:")
print("=" * 100)

findings = []

# Check data sources
missing_sources = [name for name, key in sources.items() if not key]
if missing_sources:
    findings.append(f"⚠️  Missing API keys: {', '.join(missing_sources)}")
else:
    findings.append(f"✅ All data sources configured")

# Check weights
if abs(total_weight - 1.0) > 0.01:
    findings.append(f"⚠️  Technical weights sum to {total_weight:.4f}, not 1.0")
else:
    findings.append(f"✅ Technical weights properly normalized")

# Check balance
if weights.get('mom', 0) > 0.3:
    findings.append(f"⚠️  Momentum weight is {weights.get('mom', 0):.2f} (>30%) - may dominate scoring")

if weights.get('risk_reward', 0) < 0.05:
    findings.append(f"⚠️  Risk/Reward weight is {weights.get('risk_reward', 0):.2f} (<5%) - may be underweighted")

# Check classification
if spec_count == 0 and len(results) > 10:
    findings.append(f"ℹ️  All stocks classified as Core - classification may be too lenient")
    findings.append(f"   Consider: Lower data_quality threshold or add volatility/beta to classification")

if core_count < 5 and len(results) > 15:
    findings.append(f"⚠️  Only {core_count} Core stocks - filters may be too strict")

# Print findings
for finding in findings:
    print(f"\n{finding}")

print("\n" + "=" * 100)
print("✅ AUDIT COMPLETE")
print("=" * 100)
