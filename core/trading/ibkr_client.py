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
    attempts: int = 1

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


def _retry_order(fn, *, max_attempts: int = 3, backoff: float = 2.0,
                 description: str = "order"):
    """Retry helper for order placement with exponential backoff.

    Retries on: Error status, Cancelled, network blip.
    Does NOT retry on: margin rejection (useless), invalid ticker.
    """
    import time as _time
    last_result = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            # Success — return immediately
            if result and hasattr(result, "status"):
                if result.status in ("Filled", "Submitted", "PreSubmitted", "PendingSubmit"):
                    result.attempts = attempt
                    return result
                # Don't retry on permanent errors
                err = (result.error or "").lower()
                if any(x in err for x in ["margin", "not accepted", "insufficient", "invalid", "permission"]):
                    logger.warning("%s: permanent error, no retry: %s", description, err[:100])
                    result.attempts = attempt
                    return result
            last_result = result
            if attempt < max_attempts:
                wait = backoff ** attempt
                logger.info("%s attempt %d failed, retrying in %.1fs", description, attempt, wait)
                _time.sleep(wait)
        except Exception as e:
            logger.error("%s attempt %d threw: %s", description, attempt, e)
            if attempt < max_attempts:
                _time.sleep(backoff ** attempt)
            last_result = None
    # All attempts failed
    if last_result and hasattr(last_result, "attempts"):
        last_result.attempts = max_attempts
    return last_result


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
            order.tif = "GTC"

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
            order = LimitOrder("SELL", qty, round(price, 2))
            order.tif = "GTC"
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

    def buy_market_retry(self, ticker: str, qty: int,
                         max_attempts: int = 3) -> TradeResult:
        """Market buy with retry logic (for transient failures)."""
        return _retry_order(
            lambda: self.buy_market(ticker, qty),
            max_attempts=max_attempts,
            description=f"BUY {qty} {ticker}",
        )

    @staticmethod
    def _next_trading_day_open_utc(from_dt=None) -> str:
        """Return next US market open in IBKR's goodAfterTime format.

        Used to prevent day-trade violations on cash accounts:
        protective orders placed today get goodAfterTime = next market open,
        so they can't fire same day as the buy.

        Returns: "YYYYMMDD HH:MM:SS US/Eastern" — skips weekends.
        Market open: 9:30 AM ET (13:30 UTC EDT / 14:30 UTC EST)
        """
        from datetime import datetime, timedelta, timezone
        now = from_dt or datetime.now(timezone.utc)
        # Move to next day
        d = now + timedelta(days=1)
        # Skip weekends (5=Sat, 6=Sun)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        # 9:30 AM ET = 13:30 UTC (EDT) — use 9:31 to be safe
        return d.strftime("%Y%m%d 09:31:00 US/Eastern")

    def buy_with_bracket(
        self,
        ticker: str,
        qty: int,
        trail_pct: float,
        target_price: float,
    ) -> dict:
        """Buy + trailing stop + limit sell as OCA bracket.

        When one exit order fills, the other is automatically cancelled.
        Returns dict with order IDs for all three legs.
        """
        if self.cfg.dry_run:
            logger.info(
                "[DRY RUN] BRACKET BUY %d x %s | Trail: %.1f%% | Target: $%.2f",
                qty, ticker, trail_pct, target_price,
            )
            return {
                "buy": TradeResult(ticker=ticker, action="BUY", order_type="MKT",
                                   quantity=qty, filled_price=0.0, status="DRY_RUN"),
                "trailing_stop": TradeResult(ticker=ticker, action="SELL",
                                              order_type="TRAIL", quantity=qty,
                                              filled_price=0.0, status="DRY_RUN"),
                "limit_sell": TradeResult(ticker=ticker, action="SELL",
                                          order_type="LMT", quantity=qty,
                                          filled_price=0.0, status="DRY_RUN"),
            }
        try:
            from ib_insync import Stock, Order
            import time as _time

            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Generate OCA group name (unique per trade)
            oca_group = f"SS_{ticker}_{int(_time.time())}"

            # Leg 1: Market buy (parent)
            buy_order = Order()
            buy_order.action = "BUY"
            buy_order.totalQuantity = qty
            buy_order.orderType = "MKT"
            buy_order.transmit = True
            buy_trade = self._ib.placeOrder(contract, buy_order)

            # Wait for fill
            for _ in range(30):
                self._ib.sleep(1)
                if buy_trade.orderStatus.status == "Filled":
                    break

            filled = buy_trade.orderStatus.avgFillPrice or 0.0

            # Leg 2: Trailing stop (OCA)
            trail_order = Order()
            # Prevent day-trade violation: protective orders can't fire same day
            _next_open = self._next_trading_day_open_utc()
            logger.info("Setting goodAfterTime=%s on protective orders (prevent day-trade)", _next_open)

            trail_order.action = "SELL"
            trail_order.totalQuantity = qty
            trail_order.orderType = "TRAIL"
            trail_order.trailingPercent = trail_pct
            trail_order.tif = "GTC"
            trail_order.goodAfterTime = _next_open  # Prevent same-day fill
            trail_order.ocaGroup = oca_group
            trail_order.ocaType = 1  # Cancel remaining on fill
            trail_order.transmit = True
            trail_trade = self._ib.placeOrder(contract, trail_order)

            # Leg 3: Limit sell at target (OCA — same group)
            if target_price > 0:
                limit_order = Order()
                limit_order.action = "SELL"
                limit_order.totalQuantity = qty
                limit_order.orderType = "LMT"
                limit_order.lmtPrice = round(target_price, 2)
                limit_order.tif = "GTC"
                limit_order.goodAfterTime = _next_open  # Prevent same-day fill
                limit_order.ocaGroup = oca_group
                limit_order.ocaType = 1
                limit_order.transmit = True
                limit_trade = self._ib.placeOrder(contract, limit_order)
            else:
                limit_trade = None

            self._ib.sleep(2)

            return {
                "buy": TradeResult(
                    ticker=ticker, action="BUY", order_type="MKT",
                    quantity=qty, filled_price=filled,
                    status=buy_trade.orderStatus.status,
                    order_id=buy_trade.order.orderId,
                ),
                "trailing_stop": TradeResult(
                    ticker=ticker, action="SELL", order_type="TRAIL",
                    quantity=qty, filled_price=0.0,
                    status=trail_trade.orderStatus.status,
                    order_id=trail_trade.order.orderId,
                ),
                "limit_sell": TradeResult(
                    ticker=ticker, action="SELL", order_type="LMT",
                    quantity=qty, filled_price=0.0,
                    status=limit_trade.orderStatus.status if limit_trade else "N/A",
                    order_id=limit_trade.order.orderId if limit_trade else 0,
                ),
                "oca_group": oca_group,
            }
        except Exception as e:
            logger.error("BRACKET order failed for %s: %s", ticker, e)
            return {
                "buy": TradeResult(ticker=ticker, action="BUY", order_type="MKT",
                                   quantity=qty, filled_price=0.0, status="Error",
                                   error=str(e)),
                "trailing_stop": TradeResult(ticker=ticker, action="SELL",
                                              order_type="TRAIL", quantity=qty,
                                              filled_price=0.0, status="Error"),
                "limit_sell": TradeResult(ticker=ticker, action="SELL",
                                          order_type="LMT", quantity=qty,
                                          filled_price=0.0, status="Error"),
            }

    def resubmit_protective_orders(
        self,
        ticker: str,
        qty: int,
        trail_pct: float,
        target_price: float,
        same_day_guard: bool = False,
    ) -> dict:
        """Re-submit trailing stop + limit sell as OCA for an existing position.

        If same_day_guard=True, adds goodAfterTime=next market open to prevent
        day-trade violations (for positions opened today).
        """
        if self.cfg.dry_run:
            logger.info(
                "[DRY RUN] RESUBMIT protection for %d x %s | Trail: %.1f%% | Target: $%.2f",
                qty, ticker, trail_pct, target_price,
            )
            return {
                "trailing_stop": TradeResult(ticker=ticker, action="SELL",
                                              order_type="TRAIL", quantity=qty,
                                              filled_price=0.0, status="DRY_RUN"),
                "limit_sell": TradeResult(ticker=ticker, action="SELL",
                                          order_type="LMT", quantity=qty,
                                          filled_price=0.0, status="DRY_RUN"),
            }
        try:
            from ib_insync import Stock, Order
            import time as _time

            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            oca_group = f"SS_{ticker}_{int(_time.time())}"

            # Day-trade guard: set goodAfterTime if position opened today
            _gat = self._next_trading_day_open_utc() if same_day_guard else ""

            # Trailing stop (GTC + OCA)
            trail_order = Order()
            trail_order.action = "SELL"
            trail_order.totalQuantity = qty
            trail_order.orderType = "TRAIL"
            trail_order.trailingPercent = trail_pct
            trail_order.tif = "GTC"
            if _gat:
                trail_order.goodAfterTime = _gat
            trail_order.ocaGroup = oca_group
            trail_order.ocaType = 1
            trail_order.transmit = True
            trail_trade = self._ib.placeOrder(contract, trail_order)

            # Limit sell at target (GTC + OCA)
            limit_trade = None
            if target_price > 0:
                limit_order = Order()
                limit_order.action = "SELL"
                limit_order.totalQuantity = qty
                limit_order.orderType = "LMT"
                limit_order.lmtPrice = round(target_price, 2)
                limit_order.tif = "GTC"
                if _gat:
                    limit_order.goodAfterTime = _gat
                limit_order.ocaGroup = oca_group
                limit_order.ocaType = 1
                limit_order.transmit = True
                limit_trade = self._ib.placeOrder(contract, limit_order)

            self._ib.sleep(2)

            return {
                "trailing_stop": TradeResult(
                    ticker=ticker, action="SELL", order_type="TRAIL",
                    quantity=qty, filled_price=0.0,
                    status=trail_trade.orderStatus.status,
                    order_id=trail_trade.order.orderId,
                ),
                "limit_sell": TradeResult(
                    ticker=ticker, action="SELL", order_type="LMT",
                    quantity=qty, filled_price=0.0,
                    status=limit_trade.orderStatus.status if limit_trade else "N/A",
                    order_id=limit_trade.order.orderId if limit_trade else 0,
                ),
                "oca_group": oca_group,
            }
        except Exception as e:
            logger.error("RESUBMIT protective orders failed for %s: %s", ticker, e)
            return {
                "trailing_stop": TradeResult(ticker=ticker, action="SELL",
                                              order_type="TRAIL", quantity=qty,
                                              filled_price=0.0, status="Error",
                                              error=str(e)),
                "limit_sell": TradeResult(ticker=ticker, action="SELL",
                                          order_type="LMT", quantity=qty,
                                          filled_price=0.0, status="Error"),
            }

    def resubmit_protective_orders_retry(
        self, ticker: str, qty: int, trail_pct: float,
        target_price: float, max_attempts: int = 3,
        same_day_guard: bool = False,
    ) -> dict:
        """Resubmit protective orders with retry on transient failures."""
        import time as _time
        last_result = None
        for attempt in range(1, max_attempts + 1):
            result = self.resubmit_protective_orders(
                ticker, qty, trail_pct, target_price, same_day_guard=same_day_guard
            )
            trail_ok = result["trailing_stop"].status not in ("Error", "Cancelled", "Inactive")
            limit_ok = result["limit_sell"].status not in ("Error", "Cancelled", "Inactive")
            if trail_ok and limit_ok:
                result["attempts"] = attempt
                return result
            # Check if it's a margin rejection (don't retry)
            err = (result["trailing_stop"].error or "").lower()
            if "margin" in err or "insufficient" in err:
                logger.warning("Margin rejection for %s — no retry", ticker)
                result["attempts"] = attempt
                return result
            last_result = result
            if attempt < max_attempts:
                wait = 2 ** attempt
                logger.info("Protective orders for %s failed, retry %d in %ds", ticker, attempt, wait)
                _time.sleep(wait)
        if last_result:
            last_result["attempts"] = max_attempts
        return last_result or {"trailing_stop": TradeResult(ticker, "SELL", "TRAIL", qty, 0, "Error"),
                               "limit_sell": TradeResult(ticker, "SELL", "LMT", qty, 0, "Error")}

    def place_hard_stop(self, ticker: str, qty: int, stop_price: float,
                        oca_group: str = "") -> TradeResult:
        """Place a hard STOP sell order at a specific price (for profit locking).

        Used by the ratcheting logic to replace trailing stops with fixed stops
        when a position has run up enough to lock in profit.
        """
        if self.cfg.dry_run:
            logger.info("[DRY RUN] HARD STOP SELL %d x %s @ $%.2f", qty, ticker, stop_price)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="STP",
                quantity=qty, filled_price=0.0, status="DRY_RUN",
            )
        try:
            from ib_insync import Stock, Order
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            order = Order()
            order.action = "SELL"
            order.totalQuantity = qty
            order.orderType = "STP"
            order.auxPrice = round(stop_price, 2)
            order.tif = "GTC"
            if oca_group:
                order.ocaGroup = oca_group
                order.ocaType = 1
            order.transmit = True

            trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(2)

            return TradeResult(
                ticker=ticker, action="SELL", order_type="STP",
                quantity=qty, filled_price=0.0,
                status=trade.orderStatus.status,
                order_id=trade.order.orderId,
            )
        except Exception as e:
            logger.error("HARD STOP failed for %s: %s", ticker, e)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="STP",
                quantity=qty, filled_price=0.0, status="Error",
                error=str(e),
            )

    def _sell_market(self, ticker: str, qty: int) -> TradeResult:
        """Place a market sell order (used for target-date exits)."""
        if self.cfg.dry_run:
            logger.info("[DRY RUN] SELL MKT %d x %s", qty, ticker)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="MKT",
                quantity=qty, filled_price=0.0, status="DRY_RUN",
            )
        try:
            from ib_insync import Stock, MarketOrder
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order = MarketOrder("SELL", qty)
            trade = self._ib.placeOrder(contract, order)

            for _ in range(30):
                self._ib.sleep(1)
                if trade.orderStatus.status == "Filled":
                    break

            return TradeResult(
                ticker=ticker, action="SELL", order_type="MKT",
                quantity=qty,
                filled_price=trade.orderStatus.avgFillPrice or 0.0,
                status=trade.orderStatus.status,
                order_id=trade.order.orderId,
            )
        except Exception as e:
            logger.error("SELL MKT failed for %s: %s", ticker, e)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="MKT",
                quantity=qty, filled_price=0.0, status="Error",
                error=str(e),
            )

    def get_open_orders(self) -> List[dict]:
        """Get all open/pending orders."""
        if self.cfg.dry_run:
            return []
        try:
            trades = self._ib.openTrades()
            return [
                {
                    "order_id": t.order.orderId,
                    "ticker": t.contract.symbol,
                    "action": t.order.action,
                    "order_type": t.order.orderType,
                    "quantity": t.order.totalQuantity,
                    "status": t.orderStatus.status,
                    "filled": t.orderStatus.filled,
                    "oca_group": t.order.ocaGroup or "",
                }
                for t in trades
            ]
        except Exception as e:
            logger.error("Failed to get open orders: %s", e)
            return []

    def sync_positions(self) -> List[Position]:
        """Get live positions from IBKR for reconciliation."""
        return self.get_positions()

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
