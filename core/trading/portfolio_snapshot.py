"""Portfolio snapshot writer — pushes live portfolio state to Supabase
so the Streamlit UI can display it in real-time.

Writes one row (upsert) to `trading_snapshot` table with:
- positions: live positions with PnL
- orders: active protective orders
- cash, net_liquidation, total_pnl
- updated_at timestamp
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def build_snapshot(client, tracker) -> Dict[str, Any]:
    """Build snapshot dict from live IB data + tracker."""
    try:
        # Live portfolio from IB
        portfolio = []
        total_pnl = 0.0
        tracker_positions = {p["ticker"]: p for p in tracker.get_open_positions()}

        for p in client._ib.portfolio():
            if p.position == 0:
                continue
            ticker = p.contract.symbol
            tracker_pos = tracker_positions.get(ticker, {})
            portfolio.append({
                "ticker": ticker,
                "quantity": int(p.position),
                "avg_cost": round(p.averageCost, 2),
                "market_price": round(p.marketPrice, 2),
                "market_value": round(p.marketValue, 2),
                "unrealized_pnl": round(p.unrealizedPNL, 2),
                "pnl_pct": round((p.marketPrice / p.averageCost - 1) * 100, 2) if p.averageCost else 0,
                "entry_price": tracker_pos.get("entry_price"),
                "target_price": tracker_pos.get("target_price"),
                "stop_loss": tracker_pos.get("stop_loss"),
                "target_date": str(tracker_pos.get("target_date", "")),
                "peak_price": tracker_pos.get("peak_price"),
                "stop_floor": tracker_pos.get("stop_floor"),
                "trailing_stop_pct": tracker_pos.get("trailing_stop_pct"),
                "score": tracker_pos.get("score"),
            })
            total_pnl += p.unrealizedPNL

        # Active orders
        orders = []
        client._ib.reqAllOpenOrders()
        client._ib.sleep(1)
        for t in client._ib.trades():
            if t.orderStatus.status not in ("Submitted", "PreSubmitted"):
                continue
            o = t.order
            order_info = {
                "order_id": o.orderId,
                "ticker": t.contract.symbol,
                "action": o.action,
                "type": o.orderType,
                "qty": int(o.totalQuantity),
                "tif": o.tif,
                "oca_group": o.ocaGroup or "",
                "status": t.orderStatus.status,
            }
            if o.orderType == "TRAIL":
                order_info["trail_pct"] = o.trailingPercent
                order_info["stop_price"] = round(o.trailStopPrice, 2) if o.trailStopPrice else None
            elif o.orderType == "LMT":
                order_info["limit_price"] = round(o.lmtPrice, 2)
            elif o.orderType == "STP":
                order_info["stop_price"] = round(o.auxPrice, 2)
            orders.append(order_info)

        # Account summary
        cash = 0.0
        net_liquidation = 0.0
        for item in client._ib.accountSummary():
            if item.tag == "NetLiquidation" and item.currency == "USD":
                net_liquidation = float(item.value)
            elif item.tag == "TotalCashValue" and item.currency == "USD":
                cash = float(item.value)

        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "positions": portfolio,
            "orders": orders,
            "cash": round(cash, 2),
            "net_liquidation": round(net_liquidation, 2),
            "total_unrealized_pnl": round(total_pnl, 2),
            "position_count": len(portfolio),
            "vps_status": "connected",
        }
    except Exception as e:
        logger.error("Failed to build snapshot: %s", e)
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "vps_status": "error",
            "error": str(e),
        }


def write_snapshot(client, tracker) -> bool:
    """Write portfolio snapshot to local JSON file.

    File is pushed to GitHub via systemd timer (stockscout-snapshot-sync)
    every 15 min, and Streamlit Cloud auto-reads on redeploy.
    """
    snapshot = build_snapshot(client, tracker)

    try:
        from pathlib import Path
        snapshot_file = Path("data/trades/portfolio_snapshot.json")
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file.write_text(json.dumps(snapshot, indent=2, default=str))
        logger.info(
            "Portfolio snapshot written (%d positions, %d orders, PnL $%+.2f)",
            snapshot.get("position_count", 0),
            len(snapshot.get("orders", [])),
            snapshot.get("total_unrealized_pnl", 0),
        )
        return True
    except Exception as e:
        logger.error("Snapshot write failed: %s", e)
        return False
