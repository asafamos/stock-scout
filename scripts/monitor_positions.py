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
            # But skip if there's a pending BUY order (not filled yet)
            has_pending_buy = any(
                o.get("ticker") == ticker and o.get("action") == "BUY"
                and o.get("status") in ("PreSubmitted", "Submitted")
                for o in ibkr_orders
            )
            if ticker not in ibkr_positions and not CONFIG.dry_run and not has_pending_buy:
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

        # 2. Verify protective orders — every position MUST have live orders
        _verify_protections(tracker, client, ibkr_orders, notify)

        # 3. Check target date exits — sell at market if date passed
        expired = tracker.check_target_date_exits()
        for ticker in expired:
            pos = tracker.get_position(ticker)
            if not pos:
                continue
            if ticker in ibkr_positions:
                logger.info("Target date reached for %s — selling at market", ticker)
                # Cancel existing protective orders first
                oca = pos.get("order_ids", {}).get("oca_group", "")
                if oca:
                    for o in ibkr_orders:
                        if o.get("oca_group") == oca:
                            try:
                                for t in client._ib.openTrades():
                                    if t.order.orderId == o["order_id"]:
                                        client._ib.cancelOrder(t.order)
                                        break
                            except Exception:
                                pass
                    client._ib.sleep(1)

                result = client._sell_market(ticker, pos["quantity"])
                exit_price = result.filled_price if result.status == "Filled" else 0.0
                reason = "target_date_exit"
                if exit_price > 0:
                    tracker.remove_position(ticker, exit_price, reason)
                    pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    notify.notify_sell(ticker, pos["quantity"], exit_price, reason, pnl)
                else:
                    logger.warning("Target date sell for %s not filled: %s", ticker, result.status)
                    notify.notify_error("Monitor",
                        f"Target date sell for {ticker} failed: {result.status}")

        # 4. Daily summary (at ~4:00-4:10 PM ET = 20:00-20:10 UTC, after market close)
        now = datetime.utcnow()
        if now.hour == 20 and now.minute < 10:
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


def _verify_protections(tracker, client, ibkr_orders, notify):
    """Ensure every open position has live trailing stop + limit sell.

    If orders are missing (cancelled, expired, margin issue), resubmit them
    and alert via Telegram.
    """
    positions = tracker.get_open_positions()
    if not positions:
        return

    # Build set of active OCA groups from live orders
    active_ocas = set()
    for o in ibkr_orders:
        oca = o.get("oca_group", "")
        if oca and o.get("status") in ("Submitted", "PreSubmitted"):
            active_ocas.add(oca)

    for pos in positions:
        ticker = pos["ticker"]
        oca = pos.get("order_ids", {}).get("oca_group", "")

        # Check if this position's OCA group has live orders
        if oca and oca in active_ocas:
            # Count how many live orders in this OCA group
            live_count = sum(
                1 for o in ibkr_orders
                if o.get("oca_group") == oca
                and o.get("status") in ("Submitted", "PreSubmitted")
            )
            if live_count >= 2:
                logger.info("✓ %s protected (%d orders in OCA %s)", ticker, live_count, oca)
                continue
            else:
                logger.warning("⚠ %s has only %d protective order(s) — resubmitting", ticker, live_count)
        else:
            logger.warning("⚠ %s has NO protective orders — resubmitting", ticker)

        # Resubmit protective orders
        qty = pos["quantity"]
        trail_pct = pos.get("trailing_stop_pct", 5.0)
        target_price = pos.get("target_price", 0)

        logger.info("Resubmitting protections for %s: trail=%.1f%%, target=$%.2f",
                     ticker, trail_pct, target_price)

        # Cancel old OCA orders first to prevent double-sells
        old_oca = pos.get("order_ids", {}).get("oca_group", "")
        if old_oca:
            for o in ibkr_orders:
                if o.get("oca_group") == old_oca and o.get("status") in ("Submitted", "PreSubmitted"):
                    try:
                        for t in client._ib.openTrades():
                            if t.order.orderId == o["order_id"]:
                                logger.info("Cancelling old OCA order #%d for %s", o["order_id"], ticker)
                                client._ib.cancelOrder(t.order)
                                break
                    except Exception:
                        pass
            client._ib.sleep(1)

        result = client.resubmit_protective_orders(
            ticker=ticker,
            qty=int(qty),
            trail_pct=trail_pct,
            target_price=target_price,
        )

        trail_ok = result["trailing_stop"].status not in ("Error", "Cancelled", "Inactive")
        limit_ok = result["limit_sell"].status not in ("Error", "Cancelled", "Inactive")

        if trail_ok and limit_ok:
            # Update tracker with new order IDs
            new_ids = {
                "buy": pos.get("order_ids", {}).get("buy", 0),
                "trailing_stop": result["trailing_stop"].order_id,
                "limit_sell": result["limit_sell"].order_id,
                "oca_group": result.get("oca_group", ""),
            }
            all_pos = tracker.get_open_positions()
            for p in all_pos:
                if p["ticker"] == ticker:
                    p["order_ids"] = new_ids
                    break
            tracker._save_positions(all_pos)

            logger.info("✓ %s protections resubmitted successfully", ticker)
            notify.notify_buy(
                ticker, int(qty), pos["entry_price"],
                pos.get("stop_loss", 0), target_price,
                pos.get("score", 0),
                trail_pct=trail_pct, rr=0,
                target_date=pos.get("target_date", ""),
                prefix="🔄 AUTO-RESUBMIT",
            )
        else:
            logger.error("✗ Failed to resubmit protections for %s: trail=%s, limit=%s",
                         ticker, result["trailing_stop"].status, result["limit_sell"].status)
            notify.notify_error("Protection",
                f"CRITICAL: {ticker} has NO protective orders! "
                f"Trail: {result['trailing_stop'].status}, "
                f"Limit: {result['limit_sell'].status}"
            )


def daemon_loop():
    """Run monitoring loop during market hours."""
    from core.trading.ibkr_client import IBKRClient

    logger.info("Position monitor daemon started (checking every %ds)", CHECK_INTERVAL)
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

    # Log config for confirmation
    from core.trading.config import CONFIG
    logger.info("Config: port=%d, paper=%s, dry_run=%s",
                CONFIG.ibkr_port, CONFIG.paper_mode, CONFIG.dry_run)

    if args.daemon:
        daemon_loop()
    else:
        run_check()


if __name__ == "__main__":
    main()
