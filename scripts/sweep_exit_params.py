#!/usr/bin/env python3
"""Sweep trailing-stop and time-stop parameter combinations to find optimal exits.

Usage:
    python scripts/sweep_exit_params.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweep_exit_params")

BASE_CONFIG = {
    "start_date": "2025-06-01",
    "end_date": "2026-03-23",
    "top_k": 10,
    "holding_days": 20,
    "rebalance_freq": "monthly",
    "initial_capital": 100_000,
    "enable_ml": True,
    "enable_fundamentals": True,
    "enable_patterns": True,
}

VARIANTS = {
    "no_exits": {
        "trailing": {"breakeven_trigger_pct": 9.99, "trail_trigger_pct": 9.99, "trail_atr_mult": 2.0},
        "time":     {"halfway_days": 999, "min_progress_pct": 0.0, "buffer_days": 3},
    },
    "trailing_only_loose": {
        "trailing": {"breakeven_trigger_pct": 0.90, "trail_trigger_pct": 0.95, "trail_atr_mult": 2.5},
        "time":     {"halfway_days": 999, "min_progress_pct": 0.0, "buffer_days": 3},
    },
    "trailing_only_current": {
        "trailing": {"breakeven_trigger_pct": 0.80, "trail_trigger_pct": 0.90, "trail_atr_mult": 2.0},
        "time":     {"halfway_days": 999, "min_progress_pct": 0.0, "buffer_days": 3},
    },
    "time_only": {
        "trailing": {"breakeven_trigger_pct": 9.99, "trail_trigger_pct": 9.99, "trail_atr_mult": 2.0},
        "time":     {"halfway_days": 15, "min_progress_pct": 0.15, "buffer_days": 3},
    },
    "both_very_loose": {
        "trailing": {"breakeven_trigger_pct": 0.92, "trail_trigger_pct": 0.97, "trail_atr_mult": 2.5},
        "time":     {"halfway_days": 15, "min_progress_pct": 0.10, "buffer_days": 3},
    },
    "both_current_tuned": {
        "trailing": {"breakeven_trigger_pct": 0.80, "trail_trigger_pct": 0.90, "trail_atr_mult": 2.0},
        "time":     {"halfway_days": 15, "min_progress_pct": 0.15, "buffer_days": 3},
    },
}


def run_variant(name: str, trailing: dict, time: dict) -> dict:
    import core.scoring_config as sc

    # Patch configs in-place
    original_trailing = deepcopy(sc.TRAILING_STOP_CONFIG)
    original_time = deepcopy(sc.TIME_STOP_CONFIG)

    sc.TRAILING_STOP_CONFIG.update(trailing)
    sc.TIME_STOP_CONFIG.update(time)

    try:
        from core.backtest.engine import FullPipelineBacktest
        engine = FullPipelineBacktest(**BASE_CONFIG)
        result = engine.run()
        r = result.to_dict()
        return {
            "cagr": round(r["cagr"], 2),
            "sharpe": round(r["sharpe"], 2),
            "win_rate": round(r["win_rate_pct"], 1),
            "excess_return": round(r["excess_return"], 2),
            "max_dd": round(r["max_dd"], 2),
            "profit_factor": round(r["profit_factor"], 2),
            "n_trades": r["n_trades"],
        }
    finally:
        sc.TRAILING_STOP_CONFIG.update(original_trailing)
        sc.TIME_STOP_CONFIG.update(original_time)


def main() -> None:
    results = {}
    for name, params in VARIANTS.items():
        logger.info("Running variant: %s ...", name)
        try:
            r = run_variant(name, params["trailing"], params["time"])
            results[name] = r
            logger.info(
                "  CAGR=%s%% | Sharpe=%s | WinRate=%s%% | vs SPY=%s%% | MaxDD=%s%%",
                r["cagr"], r["sharpe"], r["win_rate"], r["excess_return"], r["max_dd"],
            )
        except Exception as e:
            logger.error("  FAILED: %s", e)
            results[name] = {"error": str(e)}

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Variant':<25} {'CAGR':>7} {'Sharpe':>7} {'WinRate':>8} {'vs SPY':>8} {'MaxDD':>7} {'Trades':>7}")
    print("=" * 80)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<25} ERROR: {r['error']}")
        else:
            print(
                f"{name:<25} {r['cagr']:>6.1f}% {r['sharpe']:>7.2f} {r['win_rate']:>7.1f}% "
                f"{r['excess_return']:>+7.2f}% {r['max_dd']:>6.2f}% {r['n_trades']:>7}"
            )

    os.makedirs("reports", exist_ok=True)
    with open("reports/exit_params_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to reports/exit_params_sweep.json")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    main()
