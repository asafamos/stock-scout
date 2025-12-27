"""
Debug script: Run 1500-stock scan with detailed drop-off logging and score distribution analysis.
Identifies where stocks are filtered out and reveals score distribution across pipeline stages.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

print("=" * 100)
print("🔍 DEBUG: Stock Scout Full Scan (1500) - Drop-off & Distribution Analysis")
print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 100)

from core.pipeline_runner import run_scan_pipeline
from core.config import get_config
from stock_scout import build_universe

# Build wide universe
UNIVERSE_LIMIT = 1500
universe = build_universe(limit=UNIVERSE_LIMIT)
print(f"\n📊 STAGE 0: Universe built")
print(f"   Input: {len(universe)} tickers")

# Config
config_obj = get_config()
config = {
    "UNIVERSE_LIMIT": len(universe),
    "LOOKBACK_DAYS": 120,
    "SMART_SCAN": False,
    "EXTERNAL_PRICE_VERIFY": False,
    "FUNDAMENTAL_ENABLED": True,
    "BETA_FILTER_ENABLED": True,
    "BETA_MAX_ALLOWED": 2.0,
    "BETA_TOP_K": 100,
    "EARNINGS_BLACKOUT_DAYS": 7,
    "MA_SHORT": config_obj.ma_short,
    "MA_LONG": config_obj.ma_long,
    "WEIGHTS": config_obj.weights,
    "BUDGET_TOTAL": 5000.0,
    "MIN_POSITION": 500.0,
    "MAX_POSITION_PCT": 0.15,
}

print(f"\n⚙️  Config:")
print(f"   LOOKBACK_DAYS: {config['LOOKBACK_DAYS']}")
print(f"   BETA_FILTER: {config['BETA_FILTER_ENABLED']} (max {config['BETA_MAX_ALLOWED']})")
print(f"   FUNDAMENTALS: {config['FUNDAMENTAL_ENABLED']}")

def status_update(msg):
    print(f"   → {msg}")

# Run pipeline
import time
t_start = time.time()

try:
    results_df, data_map = run_scan_pipeline(
        universe=universe,
        config=config,
        status_callback=status_update,
        data_map=None
    )
    t_elapsed = time.time() - t_start
    
    if results_df is None or results_df.empty:
        print(f"\n❌ Pipeline returned no results!")
        sys.exit(1)
    
    print(f"\n✅ Pipeline completed in {t_elapsed:.1f}s")
    print(f"   Output: {len(results_df)} stocks")
    
except Exception as e:
    print(f"\n❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== ANALYSIS ==========

print(f"\n" + "=" * 100)
print("📈 SCORE DISTRIBUTION ANALYSIS")
print("=" * 100)

# Identify score column
score_candidates = ["conviction_v2_final", "Score", "FinalScore_20d", "overall_score_20d", "TechScore_20d"]
score_col = next((c for c in score_candidates if c in results_df.columns), None)

if not score_col:
    print("❌ No score column found!")
    sys.exit(1)

print(f"\n✅ Using score column: {score_col}")

# Score distribution
scores = pd.to_numeric(results_df[score_col], errors='coerce').dropna()
print(f"\n📊 Score Statistics ({len(scores)} valid scores):")
print(f"   Min:        {scores.min():.2f}")
print(f"   P10:        {scores.quantile(0.10):.2f}")
print(f"   P25:        {scores.quantile(0.25):.2f}")
print(f"   Median:     {scores.quantile(0.50):.2f}")
print(f"   P75:        {scores.quantile(0.75):.2f}")
print(f"   P90:        {scores.quantile(0.90):.2f}")
print(f"   Max:        {scores.max():.2f}")
print(f"   Mean:       {scores.mean():.2f}")
print(f"   Std Dev:    {scores.std():.2f}")

# Histogram buckets
buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
counts, _ = np.histogram(scores, bins=buckets)
print(f"\n📊 Score Histogram (bucket counts):")
for i, (low, high) in enumerate(zip(buckets[:-1], buckets[1:])):
    pct = (counts[i] / len(scores)) * 100 if len(scores) > 0 else 0
    bar = "█" * int(pct / 2)
    print(f"   {low:3d}-{high:3d}: {int(counts[i]):4d} ({pct:5.1f}%) {bar}")

# Top recommendations
print(f"\n🏆 Top 20 Recommendations:")
top20 = results_df.nlargest(20, score_col)[["Ticker", score_col, "Risk_Label", "Fundamental_S"]].copy()
top20.columns = ["Ticker", "Score", "Risk", "Fund"]
for idx, row in top20.iterrows():
    print(f"   {row['Ticker']:6s} | Score {row['Score']:6.2f} | Risk {row['Risk']:15s} | Fund {row['Fund']:6.1f}")

# Breakdown by risk classification
print(f"\n🎯 Breakdown by Risk Classification:")
if "Risk_Level" in results_df.columns:
    for risk_level in ["core", "speculative", "high-risk"]:
        count = (results_df["Risk_Level"] == risk_level).sum()
        if count > 0:
            subset_scores = pd.to_numeric(results_df[results_df["Risk_Level"] == risk_level][score_col], errors='coerce').dropna()
            avg_score = subset_scores.mean()
            print(f"   {risk_level:15s}: {count:4d} stocks (avg score {avg_score:6.2f})")

# Big Winner signal if available
if "BigWinnerScore_20d" in results_df.columns:
    bw_scores = pd.to_numeric(results_df["BigWinnerScore_20d"], errors='coerce').dropna()
    print(f"\n🚀 Big Winner Signal:")
    print(f"   Count with signal:  {(results_df['BigWinnerFlag_20d'] == 1).sum()}")
    print(f"   Avg Big Winner score: {bw_scores.mean():.2f}")
    print(f"   Big Winner-only top 10:")
    bw_filtered = results_df[results_df["BigWinnerFlag_20d"] == 1].nlargest(10, "BigWinnerScore_20d")
    for idx, row in bw_filtered.iterrows():
        print(f"      {row['Ticker']:6s} | BW Score {row.get('BigWinnerScore_20d', 0):6.2f}")

# Data quality check
print(f"\n✅ Data Quality Check:")
print(f"   Missing Fundamental_S: {results_df['Fundamental_S'].isna().sum()}")
print(f"   Missing Risk_Meter: {results_df.get('Risk_Meter', pd.Series()).isna().sum()}")
print(f"   Missing ML_Prob: {results_df.get('ML_20d_Prob', pd.Series()).isna().sum()}")

# Save debug output
debug_path = Path("reports/debug_distribution.csv")
debug_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(debug_path, index=False)
print(f"\n💾 Full results saved to {debug_path}")

print(f"\n" + "=" * 100)
print("✅ Debug analysis complete!")
print("=" * 100)
