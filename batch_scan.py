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
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters,
    compute_technical_score,
)
from core.scoring.fundamental import compute_fundamental_score_with_breakdown
from core.v2_risk_engine import calculate_reliability_v2
from core.data_sources_v2 import aggregate_fundamentals, fetch_price_multi_source
from core.classification import apply_classification


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
) -> pd.DataFrame:
    """
    Run full batch scan pipeline.
    
    Args:
        universe_size: Number of tickers to scan
        lookback_days: Days of historical data
        output_dir: Output directory for scan files
    
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
    
    # Stage 1: Build universe
    t0 = time.perf_counter()
    universe = build_universe(limit=universe_size)
    logger.info(f"✓ Stage 1/6: Universe ({len(universe)} tickers) — {time.perf_counter()-t0:.1f}s")
    
    # Stage 2: Fetch historical data
    t0 = time.perf_counter()
    data_map = fetch_history_bulk(universe, lookback_days, config.get("MA_LONG", 50))
    logger.info(f"✓ Stage 2/6: Historical data ({len(data_map)} tickers) — {time.perf_counter()-t0:.1f}s")
    
    # Stage 3: Build technical indicators and score
    t0 = time.perf_counter()
    rows = []
    
    for idx, (ticker, df) in enumerate(data_map.items(), 1):
        if idx % 10 == 0:
            logger.info(f"  Processing indicators: {idx}/{len(data_map)} ({idx/len(data_map)*100:.0f}%)")
        
        try:
            # Build indicators
            tech_df = build_technical_indicators(df)
            row_indicators = tech_df.iloc[-1]
            
            # Apply filters
            if not apply_technical_filters(row_indicators, strict=True):
                continue
            
            # Extract to dict
            row = {
                "Ticker": ticker,
                "Close": float(row_indicators.get("Close", np.nan)),
                "Volume": float(row_indicators.get("Volume", np.nan)),
                "RSI": float(row_indicators.get("RSI", np.nan)),
                "ATR_Pct": float(row_indicators.get("ATR_Pct", np.nan)),
                "MACD": float(row_indicators.get("MACD", np.nan)),
                "ADX": float(row_indicators.get("ADX", np.nan)),
                "Beta": float(row_indicators.get("Beta", 1.0)),
            }
            
            # Compute technical score
            tech_score = compute_technical_score(row_indicators, config)
            row["Technical_Score"] = tech_score
            
            rows.append(row)
            
        except Exception:
            logger.exception(f"Failed to process {ticker}")
    
    results = pd.DataFrame(rows)
    logger.info(f"✓ Stage 3/6: Technical indicators ({len(results)} tickers) — {time.perf_counter()-t0:.1f}s")
    
    if results.empty:
        logger.error("No tickers passed technical filters. Aborting.")
        sys.exit(1)
    
    # Stage 4: Fetch fundamentals
    t0 = time.perf_counter()
    fund_scores = []
    fund_coverage = []
    fund_sources = []
    
    for idx, row in results.iterrows():
        ticker = row["Ticker"]
        
        if idx % 10 == 0:
            logger.info(f"  Fetching fundamentals: {idx+1}/{len(results)} ({(idx+1)/len(results)*100:.0f}%)")
        
        try:
            # Aggregate fundamentals from all sources
            agg_fund = aggregate_fundamentals(ticker)
            
            # Extract metrics
            fund_data = {
                "roe": agg_fund.get("roe", np.nan),
                "pe": agg_fund.get("pe", np.nan),
                "ps": agg_fund.get("ps", np.nan),
                "de": agg_fund.get("de", np.nan),
                "rev_g_yoy": agg_fund.get("rev_g_yoy", np.nan),
                "eps_g_yoy": agg_fund.get("eps_g_yoy", np.nan),
                "gm": agg_fund.get("gm", np.nan),
                "oper_margin": agg_fund.get("oper_margin", np.nan),
                "roic": agg_fund.get("roic", np.nan),
            }
            
            coverage_pct = agg_fund.get("coverage_pct", 0.0)
            sources_used = len(agg_fund.get("sources_used", []))
            
            # Compute fundamental score (FundamentalScore.total is the overall score)
            fund_score_obj = compute_fundamental_score_with_breakdown(fund_data, coverage_pct=coverage_pct)
            # Defensive access: use .total if present, else try 'score' or dict fallback
            try:
                fund_score = float(getattr(fund_score_obj, "total", float("nan")))
            except Exception:
                try:
                    fund_score = float(fund_score_obj.get("total", float("nan")))
                except Exception:
                    fund_score = float("nan")
            
            fund_scores.append(fund_score)
            fund_coverage.append(coverage_pct)
            fund_sources.append(sources_used)
            
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
            fund_scores.append(50.0)  # Neutral
            fund_coverage.append(0.0)
            fund_sources.append(0)
    
    results["Fundamental_Score"] = fund_scores
    results["Fund_Coverage_Pct"] = fund_coverage
    results["Fund_Sources_Count"] = fund_sources
    
    logger.info(f"✓ Stage 4/6: Fundamentals ({len(results)} tickers) — {time.perf_counter()-t0:.1f}s")
    
    # Stage 5: Calculate reliability
    t0 = time.perf_counter()
    reliability_scores = []
    
    for idx, row in results.iterrows():
        try:
            rel, _ = calculate_reliability_v2(row)
            reliability_scores.append(rel)
        except Exception as e:
            logger.warning(f"Reliability calc failed for {row['Ticker']}: {e}")
            reliability_scores.append(50.0)
    
    results["Reliability_v2"] = reliability_scores
    logger.info(f"✓ Stage 5/6: Reliability ({len(results)} tickers) — {time.perf_counter()-t0:.1f}s")
    
    # Stage 6: Compute overall score and classification
    t0 = time.perf_counter()
    
    # Weights from config
    w_tech = config.get("WEIGHTS", {}).get("technical", 0.35)
    w_fund = config.get("WEIGHTS", {}).get("fundamental", 0.35)
    w_rel = config.get("WEIGHTS", {}).get("reliability", 0.15)
    w_rr = config.get("WEIGHTS", {}).get("reward_risk", 0.15)
    
    # Overall score (simplified - no ML boost for batch)
    results["Overall_Score"] = (
        results["Technical_Score"] * w_tech +
        results["Fundamental_Score"] * w_fund +
        results["Reliability_v2"] * w_rel +
        50.0 * w_rr  # Neutral RR component
    )
    
    # Apply classification
    results = apply_classification(results)
    
    # Sort by overall score
    results = results.sort_values("Overall_Score", ascending=False).reset_index(drop=True)
    results["Overall_Rank"] = range(1, len(results) + 1)
    
    logger.info(f"✓ Stage 6/6: Overall scoring ({len(results)} tickers) — {time.perf_counter()-t0:.1f}s")
    
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
