"""Entry point: execute auto-trades from latest scan results.

Usage:
    # Dry run (default — logs only, no real orders):
    python -m scripts.run_auto_trade

    # Resubmit protective orders for existing positions:
    python -m scripts.run_auto_trade --resubmit

    # Paper trading (real orders on IBKR paper account):
    TRADE_DRY_RUN=0 TRADE_PAPER_MODE=1 python -m scripts.run_auto_trade

    # LIVE trading (real money — use with caution):
    TRADE_DRY_RUN=0 TRADE_PAPER_MODE=0 python -m scripts.run_auto_trade
"""

import logging
import os
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from core.trading.config import CONFIG
    from core.trading.order_manager import OrderManager

    resubmit_mode = "--resubmit" in sys.argv

    print("=" * 60)
    if resubmit_mode:
        print("  StockScout — Resubmit Protective Orders")
    else:
        print("  StockScout Auto-Trade")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)
    print()
    print(CONFIG.summary())
    print()

    # Safety confirmation for live mode
    if not CONFIG.dry_run and not CONFIG.paper_mode:
        if os.getenv("TRADE_AUTO_CONFIRM") == "1":
            logger.warning("LIVE TRADING — auto-confirmed via TRADE_AUTO_CONFIRM")
        else:
            print("WARNING: LIVE TRADING MODE")
            confirm = input("Type 'CONFIRM LIVE' to proceed: ")
            if confirm.strip() != "CONFIRM LIVE":
                print("Aborted.")
                sys.exit(0)

    manager = OrderManager()

    if resubmit_mode:
        print(manager.tracker.summary())
        print()
        results = manager.resubmit_protections()

        print()
        print("=" * 60)
        print("  Resubmit Results")
        print("=" * 60)
        if not results:
            print("  No positions to resubmit.")
        else:
            for r in results:
                ticker = r.get("ticker", "?")
                if r["status"] == "success":
                    print(f"  OK {ticker}: Trail {r['trail_pct']}% + Target ${r['target_price']:.2f} (GTC)")
                else:
                    print(f"  ERROR {ticker}")
    else:
        results = manager.execute_recommendations()

        print()
        print("=" * 60)
        print("  Results Summary")
        print("=" * 60)

        if not results:
            print("  No trades executed.")
        else:
            for r in results:
                status = r.get("status", "unknown")
                ticker = r.get("ticker", "?")
                if status == "success":
                    print(f"  BUY {ticker}: {r['quantity']} shares @ ${r['entry_price']:.2f} "
                          f"| Trail: {r['trailing_stop_pct']}% | Target: ${r['target_price']:.2f}")
                elif status == "skipped":
                    print(f"  SKIP {ticker}: {r['reason']}")
                elif status == "error":
                    print(f"  ERROR {ticker}: {r['error']}")

    print()
    print(manager.tracker.summary())


if __name__ == "__main__":
    main()
