"""Position monitor daemon — runs during market hours.

Checks every 5 minutes for:
- Filled exit orders (trailing stop / limit sell)
- Target date exits
- Position sync with IBKR
- Sends Telegram notifications

Usage:
    # Run once:
    python -m scripts.monitor_positions

    # Run as daemon (loops during market hours):
    python -m scripts.monitor_positions --daemon
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CHECK_INTERVAL = 300  # 5 minutes


def run_check():
    """Single monitoring cycle."""
    from core.trading.config import CONFIG
    from core.trading.ibkr_client import IBKRClient
    from core.trading.position_tracker import PositionTracker
    from core.trading import notifications as notify

    tracker = PositionTracker()
    client = IBKRClient()
    positions = tracker.get_open_positions()

    if not positions:
        logger.info("No open positions to monitor")
        return

    logger.info("Monitoring %d positions...", len(positions))

    if not client.connect():
        notify.notify_error("Monitor", "Failed to connect to IBKR")
        return

    try:
        # 1. Sync with IBKR — check what's still held
        ibkr_positions = {p.ticker: p for p in client.sync_positions()}
        ibkr_orders = client.get_open_orders()

        for pos in positions:
            ticker = pos["ticker"]

            # Check if position still exists in IBKR
            if ticker not in ibkr_positions and not CONFIG.dry_run:
                # Position was closed (stop or target hit)
                logger.info("Position %s no longer in IBKR — marking closed", ticker)

                # Try to determine exit price and reason from filled orders
                exit_price = 0.0
                reason = "closed_externally"

                # Check completed trades for fill info
                for trade in client._ib.trades():
                    if (trade.contract.symbol == ticker
                            and trade.order.action == "SELL"
                            and trade.orderStatus.status == "Filled"):
                        exit_price = trade.orderStatus.avgFillPrice or 0.0
                        reason = f"{trade.order.orderType}_filled"
                        break

                # Fallback: check OCA orders
                if exit_price == 0.0:
                    oca = pos.get("order_ids", {}).get("oca_group", "")
                    if oca:
                        for order in ibkr_orders:
                            if order.get("oca_group") == oca and order.get("filled", 0) > 0:
                                reason = f"{order.get('order_type', 'unknown')}_filled"
                                break

                tracker.remove_position(ticker, exit_price, reason)
                pnl = (exit_price - pos["entry_price"]) * pos["quantity"] if exit_price else 0
                notify.notify_sell(ticker, pos["quantity"], exit_price, reason, pnl)

        # 2. Check target date exits
        expired = tracker.check_target_date_exits()
        for ticker in expired:
            pos = tracker.get_position(ticker)
            if pos:
                logger.info("Target date reached for %s — needs manual review", ticker)
                notify.notify_sell(
                    ticker, pos["quantity"], 0.0,
                    f"target_date_reached ({pos.get('target_date')})",
                )

        # 3. Daily summary (at ~15:50 UTC / 11:50 ET, near close)
        now = datetime.utcnow()
        if now.hour == 19 and now.minute < 10:  # ~3:00-3:10 PM ET
            cash = client.get_cash_balance()
            net = client.get_net_liquidation()
            notify.notify_daily_summary(
                tracker.get_open_positions(), cash, net
            )

    except Exception as e:
        logger.error("Monitor error: %s", e)
        notify.notify_error("Monitor", str(e))
    finally:
        client.disconnect()


def daemon_loop():
    """Run monitoring loop during market hours."""
    from core.trading.ibkr_client import IBKRClient

    logger.info("Position monitor daemon started")
    client = IBKRClient()

    while True:
        if client.is_market_open():
            try:
                run_check()
            except Exception as e:
                logger.error("Monitor cycle failed: %s", e)
        else:
            logger.debug("Market closed — sleeping")

        time.sleep(CHECK_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="StockScout Position Monitor")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as daemon (loop during market hours)")
    args = parser.parse_args()

    if args.daemon:
        daemon_loop()
    else:
        run_check()


if __name__ == "__main__":
    main()
