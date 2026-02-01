"""Quick scan to generate latest_scan.parquet for development."""
import sys
import os
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Force environment settings for this scan
os.environ["UNIVERSE_LIMIT"] = "500"  # Moderate size for dev
os.environ["FUNDAMENTALS_TOP_N_CAP"] = "25"
os.environ["FUNDAMENTALS_SCORE_THRESHOLD"] = "55"
os.environ["METEOR_MODE"] = "0"

from datetime import datetime
import json
import pandas as pd

from core.pipeline_runner import run_scan_pipeline, fetch_top_us_tickers_by_market_cap
from core.config import get_config

print("=" * 60)
print("⚡ Quick Development Scan")
print(f"⏰ {datetime.now()}")
print("=" * 60)

# Get universe
universe = fetch_top_us_tickers_by_market_cap(limit=500)
print(f"📊 Universe: {len(universe)} stocks")

# Config
config = get_config()
cfg = {
    "fundamental_enabled": True,
    "beta_filter_enabled": config.beta_filter_enabled,
    "meteor_mode": False,
}

# Run pipeline
def status_cb(msg):
    print(f"   {msg}")

print("\n🚀 Running pipeline...")
try:
    result_obj = run_scan_pipeline(universe[:500], cfg, status_callback=status_cb)
    
    # Handle new wrapped format
    if isinstance(result_obj, dict) and "result" in result_obj:
        results_df = result_obj["result"].get("results_df", pd.DataFrame())
        meta = result_obj.get("meta", {})
    elif isinstance(result_obj, tuple):
        results_df, _ = result_obj
        meta = {}
    else:
        results_df = result_obj
        meta = {}
    
    if results_df is None or results_df.empty:
        print("❌ No results!")
        sys.exit(1)
    
    print(f"\n✅ Pipeline returned {len(results_df)} stocks")
    
    # Save to data/scans
    output_dir = ROOT / "data" / "scans"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "scan_type": "quick_dev",
        "universe_size": len(universe),
        "total_tickers": len(results_df),
        "top_ticker": str(results_df.iloc[0]["Ticker"]) if len(results_df) > 0 else None,
    }
    
    # Save
    results_df.to_parquet(output_dir / "latest_scan_live.parquet", index=False)
    with open(output_dir / "latest_scan_live.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Saved to {output_dir}")
    print(f"   Tickers: {len(results_df)}")
    
    # Show top 10
    print("\n🏆 Top 10:")
    score_col = next((c for c in ["FinalScore_20d", "Score", "TechScore_20d"] if c in results_df.columns), None)
    if score_col:
        top = results_df.nlargest(10, score_col)
        for i, (_, row) in enumerate(top.iterrows(), 1):
            print(f"  {i}. {row['Ticker']:6s} - Score: {row[score_col]:.1f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
