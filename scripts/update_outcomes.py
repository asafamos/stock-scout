#!/usr/bin/env python3
"""Daily outcome tracking update.

Fetches current prices for all pending/partial recommendations and updates
forward returns in DuckDB.  Designed to run as a CI job after market close.

Usage::

    python scripts/update_outcomes.py
    python scripts/update_outcomes.py --as-of 2026-02-20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("update_outcomes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update outcome tracking")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Date to use for price lookups (YYYY-MM-DD). Default: today.",
    )
    args = parser.parse_args()

    as_of_date = None
    if args.as_of:
        as_of_date = datetime.strptime(args.as_of, "%Y-%m-%d").date()

    try:
        from core.db.store import get_scan_store
        from core.db.outcome_tracker import OutcomeTracker
    except ImportError as e:
        logger.error("Failed to import DuckDB modules: %s", e)
        logger.error("Ensure duckdb is installed: pip install duckdb")
        sys.exit(1)

    store = get_scan_store()
    stats = store.get_stats()
    logger.info("DB stats: %s", json.dumps(stats, default=str))

    tracker = OutcomeTracker(store)
    summary = tracker.update_outcomes(as_of_date=as_of_date)

    logger.info("=" * 60)
    logger.info("OUTCOME UPDATE SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)

    # Also print performance if we have completed outcomes
    perf = tracker.get_performance_summary(days=90)
    if perf.get("n_completed", 0) > 0:
        logger.info("")
        logger.info("PERFORMANCE (last 90 days):")
        for k, v in perf.items():
            if isinstance(v, float):
                logger.info("  %s: %.2f", k, v)
            else:
                logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
