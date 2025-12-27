from __future__ import annotations
"""
Stock Scout Batch Scanner — Offline Deep Analysis.

Usage:
    python batch_scan.py --universe-size 100 --lookback-days 60 --output-dir data/scans
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from core.config import get_config
from core.scan_io import save_scan, get_scan_summary
from core.row_builder import build_rows_from_universe
from core.pipeline_runner import run_scan_pipeline
from core.scoring import (
    build_technical_indicators,
    compute_tech_score_20d_v2,
    compute_fundamental_score_with_breakdown,
    calculate_reliability_v2,
)
from core.filters import apply_technical_filters
from core.data import (
    aggregate_fundamentals,
    fetch_price_multi_source,
)
from core.classifier import apply_classification


def build_universe(limit: int = 100) -> List[str]:
    """Build a simple universe (S&P 500 subset)."""
    import yfinance as yf  # Local import to avoid hard dependency at module import

    logger.info(f"Building universe (limit={limit})...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500_df = tables[0]
        tickers = [t.replace(".", "-") for t in sp500_df["Symbol"].tolist()]
        return tickers[:limit]
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 tickers: {e}")
        fallback = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "V", "JNJ"]
        return fallback[:limit]


def fetch_history_bulk(tickers: List[str], lookback_days: int, ma_long: int) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for all tickers.
    
    Args:
        tickers: List of ticker symbols
        lookback_days: Days of history to fetch
        ma_long: Long MA period (determines required history)
    
    Returns:
        Dict mapping ticker -> DataFrame with OHLCV data
    """
    import yfinance as yf
    
    logger.info(f"Fetching historical data for {len(tickers)} tickers...")
    
    data_map = {}
    min_len = max(ma_long + 5, lookback_days)
    
    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.0f}%)")
        
        try:
            df = yf.download(ticker, period=f"{lookback_days + 60}d", progress=False)
            if df is not None and len(df) >= min_len:
                data_map[ticker] = df
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
    
    logger.info(f"Historical data fetched: {len(data_map)}/{len(tickers)} successful")
    return data_map


def run_batch_scan(
    universe_size: int = 100,
    lookback_days: int = 60,
    output_dir: Path = None,
    custom_universe: List[str] = None,
) -> pd.DataFrame:
    """
    Run full batch scan pipeline.
    
    Args:
        universe_size: Number of tickers to scan
        lookback_days: Days of historical data
        output_dir: Output directory for scan files
        custom_universe: Optional list of tickers to use instead of auto-building
    
    Returns:
        Results DataFrame
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "scans"
    
    logger.info("=" * 60)
    logger.info("Stock Scout Batch Scanner — Starting")
    logger.info("=" * 60)
    
    # Load config
    config_obj = get_config()
    config = {
        k: getattr(config_obj, k)
        for k in dir(config_obj)
        if not k.startswith("_") and not callable(getattr(config_obj, k))
    }
    config["UNIVERSE_LIMIT"] = universe_size
    config["LOOKBACK_DAYS"] = lookback_days
    
    logger.info(f"Config: universe_size={universe_size}, lookback_days={lookback_days}")
    
    # Use unified pipeline
    if custom_universe:
        logger.info(f"Using custom universe of {len(custom_universe)} tickers")
        universe = custom_universe[:universe_size]
    else:
        universe = build_universe(limit=universe_size)
    
    results, _ = run_scan_pipeline(universe, config, status_callback=logger.info)
    
    # Save results
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    path_latest = output_dir / "latest_scan.parquet"
    path_timestamped = output_dir / f"scan_{timestamp_str}.parquet"
    
    metadata = {
        "scan_type": "batch_deep",
        "pipeline_stages": 6,
    }
    
    save_scan(results, config, path_latest, path_timestamped, metadata)
    
    # Print summary
    summary = get_scan_summary(results)
    logger.info("=" * 60)
    logger.info("Scan Summary:")
    logger.info(f"  Total tickers scanned: {summary['total_tickers']}")
    logger.info(f"  Avg overall score: {summary.get('avg_overall_score', 0):.1f}")
    logger.info(f"  Avg reliability: {summary.get('avg_reliability', 0):.1f}")
    if "classification_counts" in summary:
        logger.info(f"  Classifications: {summary['classification_counts']}")
    logger.info("=" * 60)
    logger.info(f"✓ Batch scan complete. Results saved to {output_dir}")
    
    return results


def main():
    """Main entry point for batch scanner."""
    parser = argparse.ArgumentParser(
        description="Stock Scout Batch Scanner — Offline Deep Analysis"
    )
    parser.add_argument(
        "--universe-size",
        type=int,
        default=100,
        help="Number of tickers to scan (default: 100)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=60,
        help="Days of historical data (default: 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for scan files (default: data/scans)",
    )
    
    args = parser.parse_args()
    
    try:
        results = run_batch_scan(
            universe_size=args.universe_size,
            lookback_days=args.lookback_days,
            output_dir=args.output_dir,
        )
        
        logger.info(f"SUCCESS: Scanned {len(results)} tickers")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
