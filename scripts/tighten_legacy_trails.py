"""Tighten TRAIL orders on the 3 legacy positions (ARCB/AEO/IVZ).

These 3 positions were bought under the pre-fix regime_score_floor bug
and would all be REJECTED under current gates. They're blocking new
picks by consuming all 3 position slots. Tightening trail % via IB
order MODIFY (allowed sub-$2k, works outside RTH) will fire the stops
on the next small dip and free capital.

Runs against IB Gateway. Reports what it changed. Idempotent.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
from core.trading.config import CONFIG
from core.trading.ibkr_client import IBKRClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Target trail % per ticker.
# Chosen to leave ~1-3% headroom from current price so orders don't fire
# instantly on market open but WILL fire on any modest dip.
TARGETS = {
    "ARCB": 3.0,   # entry $147.21, current ~$145, was 5.5% → will fire on ~2% dip
    "AEO":  3.5,   # entry $16.68,  current ~$16.35, was 5.5%
    "IVZ":  3.5,   # entry $27.08,  current ~$27.23, was 9.0% (Phase A)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show changes without submitting")
    args = parser.parse_args()

    client = IBKRClient(CONFIG)
    if not client.connect():
        logger.error("Cannot connect to IB Gateway")
        return 1

    try:
        open_orders = client.get_open_orders()
        trail_orders = [o for o in open_orders if o.get("order_type") == "TRAIL"]
        logger.info("Found %d TRAIL orders on IB", len(trail_orders))
        for o in trail_orders:
            logger.info("  %s: order_id=%s trail=%.2f%%", o.get("ticker"), o.get("order_id"), o.get("trailing_percent") or 0)

        for ticker, target_pct in TARGETS.items():
            match = next((o for o in trail_orders if o.get("ticker") == ticker), None)
            if match is None:
                logger.warning("%s: no TRAIL order found — skipping", ticker)
                continue

            current_pct = match.get("trailing_percent")
            order_id = match.get("order_id")
            if current_pct is None or order_id is None:
                logger.warning("%s: missing trailing_percent or order_id — skipping", ticker)
                continue

            if float(current_pct) <= float(target_pct) + 0.01:
                logger.info("%s: already at target (%.2f%% <= %.2f%%)", ticker, current_pct, target_pct)
                continue

            logger.info("%s: MODIFY trail %.2f%% → %.2f%% (order #%s)",
                        ticker, current_pct, target_pct, order_id)
            if args.dry_run:
                logger.info("  [DRY RUN] skip actual submit")
                continue

            result = client.modify_trailing_pct(int(order_id), float(target_pct))
            logger.info("  result: status=%s error=%s", result.status, getattr(result, "error", None))

    finally:
        client.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
