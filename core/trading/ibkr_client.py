"""Thin wrapper around ib_insync for IBKR connectivity.

Supports DRY_RUN mode: all methods return realistic mock objects
when dry_run=True, so the rest of the trading engine can run
end-to-end without an actual IBKR connection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from core.trading.config import CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight data classes (avoid hard dependency on ib_insync at import time)
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    quantity: float
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class TradeResult:
    ticker: str
    action: str  # BUY / SELL
    order_type: str  # MKT / TRAIL / LMT
    quantity: int
    filled_price: float
    status: str  # Filled / Submitted / Error
    order_id: int = 0
    timestamp: str = ""
    error: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# IBKR Client
# ---------------------------------------------------------------------------

class IBKRClient:
    """Interactive Brokers API client with DRY_RUN support."""

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self._ib = None
        self._connected = False

    # ── Connection ─────────────────────────────────────────────

    def connect(self) -> bool:
        if self.cfg.dry_run:
            logger.info("[DRY RUN] Simulating IBKR connection to %s:%d",
                        self.cfg.ibkr_host, self.cfg.ibkr_port)
            self._connected = True
            return True

        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(
                self.cfg.ibkr_host,
                self.cfg.ibkr_port,
                clientId=self.cfg.ibkr_client_id,
                timeout=self.cfg.ibkr_timeout,
            )
            self._connected = True
            mode = "PAPER" if self.cfg.paper_mode else "LIVE"
            logger.info("Connected to IBKR [%s] at %s:%d",
                        mode, self.cfg.ibkr_host, self.cfg.ibkr_port)
            return True
        except Exception as e:
            logger.error("Failed to connect to IBKR: %s", e)
            self._connected = False
            return False

    def disconnect(self):
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("Disconnected from IBKR")

    @property
    def connected(self) -> bool:
        if self.cfg.dry_run:
            return self._connected
        return self._ib is not None and self._ib.isConnected()

    # ── Account Info ──────────────────────────────────────────

    def get_cash_balance(self) -> float:
        if self.cfg.dry_run:
            return 10_000.0  # Simulated balance
        try:
            summary = self._ib.accountSummary()
            for item in summary:
                if item.tag == "TotalCashValue" and item.currency == "USD":
                    return float(item.value)
        except Exception as e:
            logger.error("Failed to get cash balance: %s", e)
        return 0.0

    def get_net_liquidation(self) -> float:
        if self.cfg.dry_run:
            return 12_000.0
        try:
            summary = self._ib.accountSummary()
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "USD":
                    return float(item.value)
        except Exception as e:
            logger.error("Failed to get net liquidation: %s", e)
        return 0.0

    def get_positions(self) -> List[Position]:
        if self.cfg.dry_run:
            return []
        try:
            positions = []
            for p in self._ib.positions():
                positions.append(Position(
                    ticker=p.contract.symbol,
                    quantity=p.position,
                    avg_cost=p.avgCost,
                ))
            return positions
        except Exception as e:
            logger.error("Failed to get positions: %s", e)
            return []

    # ── Orders ────────────────────────────────────────────────

    def buy_market(self, ticker: str, qty: int) -> TradeResult:
        """Place a market buy order."""
        if self.cfg.dry_run:
            logger.info("[DRY RUN] BUY MKT %d x %s", qty, ticker)
            return TradeResult(
                ticker=ticker, action="BUY", order_type="MKT",
                quantity=qty, filled_price=0.0, status="DRY_RUN",
            )
        try:
            from ib_insync import Stock, MarketOrder
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order = MarketOrder("BUY", qty)
            trade = self._ib.placeOrder(contract, order)

            # Wait for fill (up to 30s)
            for _ in range(30):
                self._ib.sleep(1)
                if trade.orderStatus.status == "Filled":
                    break

            return TradeResult(
                ticker=ticker, action="BUY", order_type="MKT",
                quantity=qty,
                filled_price=trade.orderStatus.avgFillPrice or 0.0,
                status=trade.orderStatus.status,
                order_id=trade.order.orderId,
            )
        except Exception as e:
            logger.error("BUY MKT failed for %s: %s", ticker, e)
            return TradeResult(
                ticker=ticker, action="BUY", order_type="MKT",
                quantity=qty, filled_price=0.0, status="Error",
                error=str(e),
            )

    def set_trailing_stop(self, ticker: str, qty: int,
                          trail_pct: float) -> TradeResult:
        """Place a trailing stop sell order."""
        if self.cfg.dry_run:
            logger.info("[DRY RUN] TRAIL STOP SELL %d x %s @ %.1f%%",
                        qty, ticker, trail_pct)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="TRAIL",
                quantity=qty, filled_price=0.0, status="DRY_RUN",
            )
        try:
            from ib_insync import Stock, Order
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            order = Order()
            order.action = "SELL"
            order.totalQuantity = qty
            order.orderType = "TRAIL"
            order.trailingPercent = trail_pct

            trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(2)

            return TradeResult(
                ticker=ticker, action="SELL", order_type="TRAIL",
                quantity=qty, filled_price=0.0,
                status=trade.orderStatus.status,
                order_id=trade.order.orderId,
            )
        except Exception as e:
            logger.error("TRAIL STOP failed for %s: %s", ticker, e)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="TRAIL",
                quantity=qty, filled_price=0.0, status="Error",
                error=str(e),
            )

    def set_limit_sell(self, ticker: str, qty: int,
                       price: float) -> TradeResult:
        """Place a limit sell order at target price."""
        if self.cfg.dry_run:
            logger.info("[DRY RUN] LIMIT SELL %d x %s @ $%.2f",
                        qty, ticker, price)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="LMT",
                quantity=qty, filled_price=0.0, status="DRY_RUN",
            )
        try:
            from ib_insync import Stock, LimitOrder
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order = LimitOrder("SELL", qty, price)
            trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(2)

            return TradeResult(
                ticker=ticker, action="SELL", order_type="LMT",
                quantity=qty, filled_price=0.0,
                status=trade.orderStatus.status,
                order_id=trade.order.orderId,
            )
        except Exception as e:
            logger.error("LIMIT SELL failed for %s: %s", ticker, e)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="LMT",
                quantity=qty, filled_price=0.0, status="Error",
                error=str(e),
            )

    def cancel_all_orders(self) -> bool:
        """Emergency: cancel all open orders."""
        if self.cfg.dry_run:
            logger.info("[DRY RUN] CANCEL ALL ORDERS")
            return True
        try:
            self._ib.reqGlobalCancel()
            logger.warning("ALL ORDERS CANCELLED")
            return True
        except Exception as e:
            logger.error("Failed to cancel all orders: %s", e)
            return False

    def is_market_open(self) -> bool:
        """Check if US stock market is currently open (approximate)."""
        now = datetime.utcnow()
        # US market: Mon-Fri 13:30-20:00 UTC (9:30-16:00 ET)
        if now.weekday() >= 5:
            return False
        hour_min = now.hour * 100 + now.minute
        return 1330 <= hour_min <= 2000
