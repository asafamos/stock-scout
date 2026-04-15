#!/usr/bin/env python3
"""
Re-apply GTC trailing stop + limit sell bracket orders for existing positions.

The original DAY orders expired at market close. This script:
1. Connects to IB Gateway
2. Checks for any surviving open orders (unlikely)
3. Places new GTC trailing stop + limit sell (OCA) for each position
4. Updates position tracker with new order IDs

Usage:
    .venv/bin/python -m scripts.reapply_brackets
    # Add --dry-run to preview without placing orders
"""
import argparse
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

POSITIONS_FILE = Path("data/trades/open_positions.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    positions = json.loads(POSITIONS_FILE.read_text())
    if not positions:
        logger.info("No open positions — nothing to do.")
        return

    logger.info("Found %d open positions to re-bracket", len(positions))
    for p in positions:
        logger.info("  %s: %d shares @ $%.2f | trail %.1f%% | target $%.2f",
                     p["ticker"], p["quantity"], p["entry_price"],
                     p["trailing_stop_pct"], p["target_price"])

    if args.dry_run:
        logger.info("[DRY RUN] Would place GTC brackets for the above. Exiting.")
        return

    # Connect to IB
    from ib_insync import IB, Stock, Order, LimitOrder
    ib = IB()
    logger.info("Connecting to IB Gateway on 127.0.0.1:7496 ...")
    ib.connect("127.0.0.1", 7496, clientId=10)
    logger.info("Connected. Server version: %s", ib.client.serverVersion())

    # Check existing open orders
    ib.sleep(1)
    open_orders = ib.openTrades()
    if open_orders:
        logger.info("Found %d existing open orders:", len(open_orders))
        for t in open_orders:
            logger.info("  %s %s %s qty=%s status=%s oca=%s",
                        t.contract.symbol, t.order.action,
                        t.order.orderType, t.order.totalQuantity,
                        t.orderStatus.status, t.order.ocaGroup or "-")
    else:
        logger.info("No existing open orders (as expected — DAY orders expired).")

    # Place new GTC brackets for each position
    for p in positions:
        ticker = p["ticker"]
        qty = p["quantity"]
        trail_pct = p["trailing_stop_pct"]
        target_price = p["target_price"]
        oca_group = f"SS_{ticker}_{int(time.time())}"

        logger.info("Placing GTC brackets for %s ...", ticker)

        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)

        # Trailing stop (GTC + OCA)
        trail_order = Order()
        trail_order.action = "SELL"
        trail_order.totalQuantity = qty
        trail_order.orderType = "TRAIL"
        trail_order.trailingPercent = trail_pct
        trail_order.tif = "GTC"
        trail_order.ocaGroup = oca_group
        trail_order.ocaType = 1
        trail_order.transmit = True
        trail_trade = ib.placeOrder(contract, trail_order)
        logger.info("  Trailing stop placed: orderId=%s status=%s",
                     trail_trade.order.orderId, trail_trade.orderStatus.status)

        # Limit sell at target (GTC + OCA same group)
        limit_order = LimitOrder("SELL", qty, target_price)
        limit_order.tif = "GTC"
        limit_order.ocaGroup = oca_group
        limit_order.ocaType = 1
        limit_order.transmit = True
        limit_trade = ib.placeOrder(contract, limit_order)
        logger.info("  Limit sell placed: orderId=%s status=%s @ $%.2f",
                     limit_trade.order.orderId, limit_trade.orderStatus.status,
                     target_price)

        # Update position tracker
        p["order_ids"] = {
            "buy": p["order_ids"].get("buy", 0),
            "trailing_stop": trail_trade.order.orderId,
            "limit_sell": limit_trade.order.orderId,
            "oca_group": oca_group,
        }
        logger.info("  %s bracketed: OCA=%s", ticker, oca_group)

    ib.sleep(2)

    # Save updated positions
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2, default=str))
    logger.info("Positions file updated with new order IDs.")

    # Final verification
    open_orders = ib.openTrades()
    logger.info("Verification — %d open orders after re-bracketing:", len(open_orders))
    for t in open_orders:
        logger.info("  %s %s %s qty=%s tif=%s status=%s oca=%s",
                     t.contract.symbol, t.order.action,
                     t.order.orderType, t.order.totalQuantity,
                     t.order.tif, t.orderStatus.status,
                     t.order.ocaGroup or "-")

    ib.disconnect()
    logger.info("Done. All positions now have GTC protection.")


if __name__ == "__main__":
    main()
