#!/usr/bin/env python3
"""Run full-pipeline backtest with optional ablation study.

Usage::

    python scripts/run_full_backtest.py
    python scripts/run_full_backtest.py --start 2024-01-01 --end 2025-12-31
    python scripts/run_full_backtest.py --ablation
    python scripts/run_full_backtest.py --top-k 15 --holding-days 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_full_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-pipeline backtest")
    parser.add_argument("--start", type=str, default="2025-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-03-13", help="End date (YYYY-MM-DD)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K stocks per rebalance")
    parser.add_argument("--holding-days", type=int, default=20, help="Holding period in trading days")
    parser.add_argument("--rebalance", type=str, default="monthly", help="Rebalance frequency")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML scoring")
    parser.add_argument("--no-fundamentals", action="store_true", help="Disable fundamental scoring")
    parser.add_argument("--no-patterns", action="store_true", help="Disable pattern matching")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study (all variants)")
    parser.add_argument("--output", type=str, default="reports/backtest_latest.json", help="Output file")
    args = parser.parse_args()

    config = {
        "start_date": args.start,
        "end_date": args.end,
        "top_k": args.top_k,
        "holding_days": args.holding_days,
        "rebalance_freq": args.rebalance,
        "initial_capital": args.capital,
        "enable_ml": not args.no_ml,
        "enable_fundamentals": not args.no_fundamentals,
        "enable_patterns": not args.no_patterns,
    }

    def status(msg: str) -> None:
        logger.info("[STATUS] %s", msg)

    if args.ablation:
        from core.backtest.attribution import PerformanceAttribution

        attr = PerformanceAttribution(config)
        results = attr.run_ablation_study(status_callback=status)

        # Print comparison
        table = attr.ablation_comparison_table(results)
        logger.info("\n%s", table.to_string())

        # Save each variant
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        all_results = {name: r.to_dict() for name, r in results.items()}
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Saved ablation results to %s", args.output)

        # Full system summary
        if "full_system" in results:
            print(results["full_system"].summary())
    else:
        from core.backtest.engine import FullPipelineBacktest

        config["status_callback"] = status
        engine = FullPipelineBacktest(**config)
        result = engine.run()

        print(result.summary())

        # Save
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("Saved results to %s", args.output)

        # Score quintile analysis
        if not result.trade_log.empty:
            from core.backtest.attribution import PerformanceAttribution

            qa = PerformanceAttribution.score_quintile_analysis(result.trade_log)
            if not qa.empty:
                logger.info("\nScore Quintile Analysis:")
                logger.info("\n%s", qa.to_string(index=False))

            corr = PerformanceAttribution.compute_component_correlations(result.trade_log)
            if not corr.empty:
                logger.info("\nComponent Correlations:")
                logger.info("\n%s", corr.to_string())


if __name__ == "__main__":
    main()
