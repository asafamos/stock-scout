"""
Automated Stock Scanner - Runs twice daily via GitHub Actions.
Uses FULL pipeline with all scoring logic, ML models, and filters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import os

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("🤖 Stock Scout Auto Scan - FULL PIPELINE")
print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# Import core modules
from core.pipeline_runner import run_scan_pipeline
from core.config import get_config
from stock_scout import build_universe
from core.scoring import (
    build_technical_indicators,
    compute_tech_score_20d_v2,
    compute_final_scores_20d,
    score_ticker_v2_enhanced,
)
from core.filters import (
    apply_technical_filters,
    compute_advanced_score,
    should_reject_ticker,
    fetch_benchmark_data,
)
from core.data import (
    fetch_price_multi_source,
    aggregate_fundamentals,
    fetch_fundamentals_batch,
)
from core.allocation import allocate_budget
from core.classifier import apply_classification

UNIVERSE_LIMIT = int(os.getenv("AUTO_SCAN_UNIVERSE_LIMIT", "1500"))
UNIVERSE = build_universe(limit=UNIVERSE_LIMIT)

print(f"🎯 Universe size: {len(UNIVERSE)} stocks (limit {UNIVERSE_LIMIT})")
print(f"⚙️  Loading configuration and initializing pipeline...")

# Load configuration
config_obj = get_config()
config = {
    "UNIVERSE_LIMIT": len(UNIVERSE),
    "LOOKBACK_DAYS": 200,
    "SMART_SCAN": False,
    "EXTERNAL_PRICE_VERIFY": False,
    "PERF_FAST_MODE": True,
    # Canonical lowercase toggles (pipeline normalizes legacy keys too)
    "fundamental_enabled": True,  # ✅ Enable fundamentals
    "beta_filter_enabled": True,
    "beta_max_allowed": 2.0,
    "beta_top_k": 60,
    "beta_benchmark": "SPY",
    # Other settings retained as-is (uppercase used in pipeline)
    "SECTOR_CAP_ENABLED": True,
    "SECTOR_CAP_MAX": 3,
    "EARNINGS_BLACKOUT_DAYS": 7,  # ✅ Enable earnings check
    "EARNINGS_CHECK_TOPK": 30,
    "MA_SHORT": config_obj.ma_short,
    "MA_LONG": config_obj.ma_long,
    "WEIGHTS": config_obj.weights,
    "BUDGET_TOTAL": 5000.0,
    "MIN_POSITION": 500.0,
    "MAX_POSITION_PCT": 0.15,
}

print(f"📥 Running full pipeline with:")
print(f"   - Technical indicators (20+ metrics)")
print(f"   - ML model (XGBoost 20d)")
print(f"   - Fundamental scoring (Alpha/Finnhub/FMP)")
print(f"   - Risk assessment (V2 engine)")
print(f"   - Classification (Core/Speculative)")

# Status callback for progress
def status_update(msg: str):
    print(f"   {msg}")

# Run the FULL pipeline
import time
t_start = time.time()

try:
    results_df, data_map = run_scan_pipeline(
        universe=UNIVERSE,
        config=config,
        status_callback=status_update,
        data_map=None
    )
    
    t_elapsed = time.time() - t_start
    
    if results_df is None or results_df.empty:
        print(f"\n❌ Pipeline returned no results!")
        sys.exit(1)
    
    print(f"\n✅ Pipeline completed in {t_elapsed:.1f}s")
    print(f"📊 Results: {len(results_df)} stocks passed all filters")
    
except Exception as e:
    print(f"\n❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Prepare for saving
print(f"\n💾 Preparing results for export...")

# FILTERING: Apply quality gates to select top stocks
print(f"\n🔍 Applying quality filters...")
original_count = len(results_df)

# DEBUG: Print available columns
print(f"   Available columns: {len(results_df.columns)}")
filter_cols = ['Score', 'conviction_v2_final', 'TechScore_20d', 'ML_20d_Prob']
for col in filter_cols:
    status = "✅" if col in results_df.columns else "❌"
    print(f"     {status} {col}")

# IMPORTANT: Risk Engine V2 blocks stocks when NO fundamentals are available
# This is overly aggressive. Instead, use technical scores when fundamentals missing.

# Strategy: Sort by best available score
# Priority: conviction_v2_final > Score > TechScore_20d (fallback)
score_col = 'conviction_v2_final' if 'conviction_v2_final' in results_df.columns else 'Score'
if score_col not in results_df.columns:
    score_col = 'TechScore_20d' if 'TechScore_20d' in results_df.columns else 'Score'

# Filter 1: Minimum score threshold (only if score is not too low)
if score_col in results_df.columns:
    before_score = len(results_df)
    score_values = pd.to_numeric(results_df[score_col], errors='coerce')
    
    # Determine appropriate minimum based on scale
    if (score_values.dropna() > 10).any():  # Appears to be 0-100 scale
        min_score = 20.0  # Reduced from 50 to accommodate tech-only filtering
    else:  # 0-10 scale or similar
        min_score = 3.0
    
    results_df = results_df[score_values >= min_score].copy()
    filtered_score = before_score - len(results_df)
    if filtered_score > 0:
        print(f"   ❌ Removed {filtered_score} below minimum score ({min_score})")

# Filter 2: Take only top N by score (regardless of data source)
top_n = 15
if len(results_df) > top_n:
    results_df = results_df.nlargest(top_n, score_col).copy()
    print(f"   ✂️ Kept top {top_n} by score")

print(f"\n   ✅ Final results: {len(results_df)} stocks (from {original_count})")

# Ensure required columns exist
required_cols = [
    'Ticker', 'Score', 'FinalScore_20d', 'TechScore_20d', 'ML_20d_Prob',
    'RSI', 'Close', 'Volume', 'Sector', 'Risk_Level', 'Data_Quality'
]

# Add aliases for compatibility
if 'Score' in results_df.columns and 'overall_score_20d' not in results_df.columns:
    results_df['overall_score_20d'] = results_df['Score']
elif 'FinalScore_20d' in results_df.columns and 'overall_score_20d' not in results_df.columns:
    results_df['overall_score_20d'] = results_df['FinalScore_20d']

# Add rank
if 'Overall_Rank' not in results_df.columns:
    results_df = results_df.sort_values('overall_score_20d', ascending=False)
    results_df['Overall_Rank'] = range(1, len(results_df) + 1)

# Select columns to save (comprehensive set from full pipeline)
save_cols = [
    # Core identification
    'Ticker', 'Sector',
    # Scoring (multiple versions for compatibility)
    'overall_score_20d', 'Score', 'FinalScore_20d', 'TechScore_20d', 'ML_20d_Prob',
    'conviction_v2_final', 'conviction_v2_base',
    # Technical indicators
    'RSI', 'ATR_Pct', 'Close', 'Volume', 'RewardRisk', 'RR_Ratio',
    'MA_Aligned', 'Momentum_Consistency', 'Volume_Surge',
    # Classification & Quality
    'Risk_Level', 'Data_Quality', 'Confidence_Level',
    # Fundamental scores
    'Fundamental_S', 'Quality_Score_F', 'Growth_Score_F', 'Valuation_Score_F',
    # Risk & Reliability
    'reliability_v2', 'risk_gate_status_v2',
    # UI cards expect these when available
    'risk_meter_v2', 'risk_band', 'reliability_pct', 'reliability_band',
    'Fundamental_Reliability_v2', 'Price_Reliability_v2',
    # Quality display
    'Quality_Level', 'Quality_Score_Numeric',
    # Position sizing
    'buy_amount_v2', 'shares_to_buy_v2',
    # Ranking
    'Overall_Rank',
    # Additional metrics
    'Beta', 'RS_63d', 'AdvPenalty',
    # Prices
    'Price_Yahoo'
]

# Only keep columns that exist
save_cols_actual = [c for c in save_cols if c in results_df.columns]
df_to_save = results_df[save_cols_actual].copy()

# Aliases for UI compatibility (cards expect Unit_Price and sometimes ML_Probability)
if "Close" in df_to_save.columns and "Unit_Price" not in df_to_save.columns:
    df_to_save["Unit_Price"] = df_to_save["Close"]
if "ML_20d_Prob" in df_to_save.columns and "ML_Probability" not in df_to_save.columns:
    df_to_save["ML_Probability"] = df_to_save["ML_20d_Prob"]

# Provide commonly-used card fields when missing
# Entry_Price: prefer Unit_Price → Close → Price_Yahoo
if "Entry_Price" not in df_to_save.columns:
    if "Unit_Price" in df_to_save.columns:
        df_to_save["Entry_Price"] = df_to_save["Unit_Price"]
    elif "Close" in df_to_save.columns:
        df_to_save["Entry_Price"] = df_to_save["Close"]
    elif "Price_Yahoo" in df_to_save.columns:
        df_to_save["Entry_Price"] = df_to_save["Price_Yahoo"]

# Reliability aliases: expose a unified percent and band if present in v2
if "reliability_pct" not in df_to_save.columns and "reliability_v2" in results_df.columns:
    df_to_save["reliability_pct"] = results_df["reliability_v2"]
if "reliability_band" not in df_to_save.columns and "reliability_band" in results_df.columns:
    df_to_save["reliability_band"] = results_df["reliability_band"]
if "risk_band" not in df_to_save.columns and "risk_band" in results_df.columns:
    df_to_save["risk_band"] = results_df["risk_band"]
if "risk_meter_v2" not in df_to_save.columns and "risk_meter_v2" in results_df.columns:
    df_to_save["risk_meter_v2"] = results_df["risk_meter_v2"]

# Default targets: provide conservative placeholders to avoid N/A in cards
from datetime import timedelta
if "Target_Price" not in df_to_save.columns:
    base_price_col = "Entry_Price" if "Entry_Price" in df_to_save.columns else ("Unit_Price" if "Unit_Price" in df_to_save.columns else None)
    if base_price_col:
        try:
            df_to_save["Target_Price"] = (pd.to_numeric(df_to_save[base_price_col], errors="coerce") * 1.10).round(4)
            df_to_save["Target_Source"] = "Default"
            df_to_save["Target_Date"] = (pd.Timestamp.now() + timedelta(days=30)).dt.strftime("%Y-%m-%d")
        except Exception:
            # If vectorized date formatting fails, fill with simple string
            df_to_save["Target_Price"] = pd.to_numeric(df_to_save[base_price_col], errors="coerce") * 1.10
            df_to_save["Target_Source"] = "Default"
            df_to_save["Target_Date"] = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

# Save results
output_dir = Path('data/scans')
output_dir.mkdir(parents=True, exist_ok=True)

df_to_save.to_parquet(output_dir / 'latest_scan.parquet', index=False)

metadata = {
    "timestamp": datetime.now().isoformat(),
    "scan_type": "automated_full_pipeline",
    "total_tickers": len(df_to_save),
    "universe_size": len(UNIVERSE),
    "pipeline_version": "v2_with_ml",
    "top_ticker": df_to_save.iloc[0]['Ticker'] if len(df_to_save) > 0 else None,
    "top_score": float(df_to_save.iloc[0]['overall_score_20d']) if len(df_to_save) > 0 else 0,
    "avg_score": float(df_to_save['overall_score_20d'].mean()),
    "scan_duration_seconds": t_elapsed,
    "columns_saved": save_cols_actual,
    # Include build info to help caches and age checks
    "build_commit": None,
}

try:
    import subprocess
    metadata["build_commit"] = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    )
except Exception:
    pass

with open(output_dir / 'latest_scan.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Results saved:")
print(f"   File: data/scans/latest_scan.parquet")
print(f"   Columns: {len(save_cols_actual)}")
print(f"   Top stock: {metadata['top_ticker']} (score: {metadata['top_score']:.1f})")
print(f"   Average score: {metadata['avg_score']:.1f}")

print("\n" + "=" * 80)
print("🏆 Top 10 stocks:")
print("=" * 80)
for i, row in df_to_save.head(10).iterrows():
    ticker = row['Ticker']
    score = row['overall_score_20d']
    ml_prob = row.get('ML_20d_Prob', 0) * 100
    rsi = row.get('RSI', 0)
    risk = row.get('Risk_Level', 'N/A')[:4]
    print(f"{row['Overall_Rank']:2d}. {ticker:6s} - Score: {score:6.1f} | ML: {ml_prob:4.0f}% | RSI: {rsi:4.0f} | {risk}")

print("=" * 80)
print(f"✅ Auto scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)
