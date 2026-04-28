"""Thin wrapper around ib_insync for IBKR connectivity.

Supports DRY_RUN mode: all methods return realistic mock objects
when dry_run=True, so the rest of the trading engine can run
end-to-end without an actual IBKR connection.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from core.trading.config import CONFIG

logger = logging.getLogger(__name__)


def _make_oca_group(ticker: str) -> str:
    """Generate a guaranteed-unique OCA group name.

    Previously used int(time.time()) which collides when two orders are
    submitted within the same second (e.g. manual resubmit racing with
    the monitor's auto-resubmit). Collision links UNRELATED positions'
    protective orders: filling one would cancel the other.
    uuid4 hex is globally unique even under concurrent callers.
    """
    return f"SS_{ticker}_{uuid.uuid4().hex[:10]}"


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

        from ib_insync import IB
        self._ib = IB()

        # Try the configured clientId first, then fall back to random IDs on
        # collision. IB's "Error 326: clientId already in use" surfaces as
        # ib_insync's TimeoutError after the peer closes the connection;
        # we detect either path and retry with a fresh ID. Solves the
        # 2026-04-28 case: monitor + pipeline both used clientId=1, monitor
        # alerted "Failed to connect to IBKR" twice while the pipeline's
        # trade run was holding the ID.
        import random
        ids_to_try = [self.cfg.ibkr_client_id]
        # Up to 4 random fallbacks in the 100..999 range — far enough from
        # the canonical 1/2/3 clients to avoid colliding with anything else.
        for _ in range(4):
            ids_to_try.append(random.randint(100, 999))

        last_err = None
        mode = "PAPER" if self.cfg.paper_mode else "LIVE"
        for cid in ids_to_try:
            try:
                self._ib.connect(
                    self.cfg.ibkr_host,
                    self.cfg.ibkr_port,
                    clientId=cid,
                    timeout=self.cfg.ibkr_timeout,
                )
                self._connected = True
                if cid != self.cfg.ibkr_client_id:
                    logger.warning(
                        "Connected to IBKR [%s] at %s:%d with FALLBACK clientId %d "
                        "(configured %d was in use)",
                        mode, self.cfg.ibkr_host, self.cfg.ibkr_port,
                        cid, self.cfg.ibkr_client_id,
                    )
                else:
                    logger.info(
                        "Connected to IBKR [%s] at %s:%d (clientId %d)",
                        mode, self.cfg.ibkr_host, self.cfg.ibkr_port, cid,
                    )
                return True
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Recognize the collision case (clientId in use) — only then retry.
                # Other errors (network, auth) shouldn't burn through the pool.
                if "client id" in msg or "326" in msg or isinstance(e, TimeoutError):
                    logger.warning(
                        "IBKR connect attempt with clientId %d failed: %s — retrying with new ID",
                        cid, e,
                    )
                    try:
                        self._ib.disconnect()
                    except Exception:
                        pass
                    self._ib = IB()  # fresh IB instance for next attempt
                    continue
                # Non-collision error — give up immediately
                break

        logger.error("Failed to connect to IBKR after %d attempts: %s",
                     len(ids_to_try), last_err)
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

    def get_live_price(self, ticker: str, timeout: float = 6.0) -> Optional[float]:
        """Fetch a near-real-time price via IBKR delayed market data (15-min lag).

        Our IBKR account doesn't have a paid live market data subscription
        (free tier only), so we use ``reqMarketDataType(3)`` for delayed.
        15-min lag is still 4-6× fresher than the 60-90 min scan, which
        is the gap this method exists to close.

        Falls back to None on any failure (timeout, no data, dry run) so
        callers can degrade to the scan price safely.
        """
        if self.cfg.dry_run:
            return None
        try:
            from ib_insync import Stock
            contract = Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            # 3 = delayed (~15 min lag, free with all IB accounts).
            # We set this once per call; it's idempotent and cheap.
            try:
                self._ib.reqMarketDataType(3)
            except Exception:
                pass
            # Streaming (not snapshot) — snapshot mode requires live
            # subscription per IB error 10089. Streaming + delayed works.
            ticker_obj = self._ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
            # Delayed feed takes longer to populate than live snapshot.
            self._ib.sleep(timeout)
            try:
                # Try every field that might carry a price (live, delayed, close).
                # ib_insync exposes delayed values on the same fields when
                # marketDataType=3 is active.
                candidates = [
                    getattr(ticker_obj, "last", None),
                    getattr(ticker_obj, "delayedLast", None),
                    getattr(ticker_obj, "close", None),
                    getattr(ticker_obj, "delayedClose", None),
                ]
                for v in candidates:
                    if v is None:
                        continue
                    try:
                        f = float(v)
                        if f > 0 and f == f:  # not NaN
                            return f
                    except (TypeError, ValueError):
                        continue
                # Mid-quote fallback (live or delayed)
                bid = getattr(ticker_obj, "bid", None) or getattr(ticker_obj, "delayedBid", None)
                ask = getattr(ticker_obj, "ask", None) or getattr(ticker_obj, "delayedAsk", None)
                try:
                    bid_f = float(bid) if bid is not None else float("nan")
                    ask_f = float(ask) if ask is not None else float("nan")
                    if bid_f > 0 and ask_f > 0 and bid_f == bid_f and ask_f == ask_f:
                        return (bid_f + ask_f) / 2.0
                except (TypeError, ValueError):
                    pass
            finally:
                try:
                    self._ib.cancelMktData(contract)
                except Exception:
                    pass
            return None
        except Exception as e:
            logger.warning("get_live_price failed for %s: %s", ticker, e)
            return None

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
                          trail_pct: float,
                          oca_group: str = "") -> TradeResult:
        """Place a trailing stop sell order.

        If oca_group is provided, the order joins that OCA group so it
        gets cancelled when a sibling order (e.g. limit_sell at target)
        fills. Used by the ratchet migration path to keep the bracket
        intact when replacing a legacy STP with a fresh TRAIL.
        """
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
            if oca_group:
                order.ocaGroup = oca_group
                order.ocaType = 1

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
            oca_group = _make_oca_group(ticker)

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

            # Cancel any existing live protective orders for this ticker
            # BEFORE placing new ones. Prevents duplicate OCA groups that
            # could double-sell when stops trigger. Especially important
            # during reconnect scenarios where the monitor's stored OCA
            # may not match IB's active OCA.
            try:
                self._ib.reqAllOpenOrders()
                self._ib.sleep(1)
                cancelled = 0
                for tr in list(self._ib.openTrades()):
                    if (tr.contract.symbol == ticker
                            and tr.order.action == "SELL"
                            and tr.orderStatus.status in ("Submitted", "PreSubmitted")):
                        try:
                            self._ib.cancelOrder(tr.order)
                            cancelled += 1
                        except Exception as _ce:
                            logger.debug("Cancel skipped for %s order %d: %s",
                                         ticker, tr.order.orderId, _ce)
                if cancelled:
                    logger.info("Cancelled %d live %s protective orders before resubmit",
                                cancelled, ticker)
                    self._ib.sleep(2)  # let cancellations propagate
            except Exception as _pre_e:
                logger.warning("Pre-resubmit cancel pass failed for %s: %s",
                               ticker, _pre_e)

            oca_group = _make_oca_group(ticker)

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
            # Check for permanent rejections — don't retry if IB will reject again
            err = (result["trailing_stop"].error or "").lower()
            if "margin" in err or "insufficient" in err:
                logger.warning("Margin rejection for %s — no retry", ticker)
                result["attempts"] = attempt
                return result
            # Cash-account rule: "minimum of 2000" / "sell short" / "purchase on margin"
            # Retrying hammers IB and can cause earlier successful OCA groups
            # to be collateralized-cancelled. Fail fast.
            if any(s in err for s in ("minimum of 2000", "sell short", "purchase on margin")):
                logger.warning(
                    "Cash-account rule rejected %s — no retry (existing OCA may remain live)",
                    ticker,
                )
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

    def modify_trailing_pct(self, order_id: int,
                            new_trail_pct: float) -> TradeResult:
        """Modify an existing TRAIL order's trailingPercent in-place.

        IB recognizes a re-submission with the same orderId as a
        modification — the order keeps its identity (and its OCA
        membership) and just gets a new trail %. The server continues
        tracking the peak from there; we don't reset anything.

        Used by the ratcheting logic to TIGHTEN protection as a position
        runs up, without ever cancelling+replacing the order (which would
        lose the OCA link, momentarily expose the position, and create
        log clutter).
        """
        if self.cfg.dry_run:
            logger.info("[DRY RUN] MODIFY TRAIL #%d → %.1f%%",
                        order_id, new_trail_pct)
            return TradeResult(
                ticker="", action="SELL", order_type="TRAIL",
                quantity=0, filled_price=0.0, status="DRY_RUN",
                order_id=order_id,
            )
        try:
            # Find the existing trade by orderId
            target_trade = None
            for t in self._ib.openTrades():
                if t.order.orderId == order_id:
                    target_trade = t
                    break
            if target_trade is None:
                # Try reqAllOpenOrders for cross-client visibility
                self._ib.reqAllOpenOrders()
                self._ib.sleep(1)
                for t in self._ib.openTrades():
                    if t.order.orderId == order_id:
                        target_trade = t
                        break
            if target_trade is None:
                logger.warning(
                    "modify_trailing_pct: order #%d not found in openTrades",
                    order_id,
                )
                return TradeResult(
                    ticker="", action="SELL", order_type="TRAIL",
                    quantity=0, filled_price=0.0, status="Error",
                    order_id=order_id, error="order not found",
                )

            old_pct = float(getattr(target_trade.order, "trailingPercent", 0) or 0)
            target_trade.order.trailingPercent = float(new_trail_pct)
            # Re-submit with same orderId — IB treats this as a modification.
            self._ib.placeOrder(target_trade.contract, target_trade.order)
            self._ib.sleep(2)

            ticker = target_trade.contract.symbol
            qty = int(target_trade.order.totalQuantity)
            logger.info(
                "✓ MODIFIED TRAIL %s #%d: %.1f%% → %.1f%%",
                ticker, order_id, old_pct, new_trail_pct,
            )
            return TradeResult(
                ticker=ticker, action="SELL", order_type="TRAIL",
                quantity=qty, filled_price=0.0,
                status=target_trade.orderStatus.status,
                order_id=order_id,
            )
        except Exception as e:
            logger.error("modify_trailing_pct failed for #%d: %s", order_id, e)
            return TradeResult(
                ticker="", action="SELL", order_type="TRAIL",
                quantity=0, filled_price=0.0, status="Error",
                order_id=order_id, error=str(e),
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
        """Get all open/pending orders across ALL clients.

        IMPORTANT: openTrades() alone only returns orders from the current
        clientId. After a reconnect / monitor restart, protective orders
        submitted by a prior clientId would be invisible — causing the monitor
        to falsely detect "no protection" and trigger redundant resubmits
        (which then get rejected on cash accounts < $2000 by IB's risk check).

        reqAllOpenOrders() fetches orders from every client, and openTrades()
        then includes them.
        """
        if self.cfg.dry_run:
            return []
        try:
            try:
                self._ib.reqAllOpenOrders()
                self._ib.sleep(1)
            except Exception as e:
                logger.warning("reqAllOpenOrders failed (using current-client view only): %s", e)
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

    # US stock market holidays (static list — add new years as needed).
    # Keeps the monitor from running on NYSE holidays where portfolio()
    # returns stale data and no orders will fill.
    _US_MARKET_HOLIDAYS_2026 = {
        "2026-01-01",  # New Year's Day
        "2026-01-19",  # MLK Day
        "2026-02-16",  # Presidents' Day
        "2026-04-03",  # Good Friday
        "2026-05-25",  # Memorial Day
        "2026-06-19",  # Juneteenth
        "2026-07-03",  # Independence Day (observed)
        "2026-09-07",  # Labor Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Christmas
    }

    def is_market_open(self) -> bool:
        """Check if US regular trading hours are active.

        Regular session only: Mon-Fri, 9:30-16:00 ET, excluding NYSE holidays.
        Does NOT include pre-market (4:00-9:30 ET) or after-hours (16:00-20:00 ET)
        because our OCA trailing stops don't fill reliably outside RTH, and
        price data during extended hours is misleading for ratchet logic.
        """
        now = datetime.utcnow()
        if now.weekday() >= 5:
            return False
        if now.strftime("%Y-%m-%d") in self._US_MARKET_HOLIDAYS_2026:
            return False
        # 13:30-20:00 UTC = 9:30-16:00 ET during US Eastern Daylight Time.
        # Note: during standard time (Nov-Mar), this is 14:30-21:00 UTC; the
        # daemon's 5-min cadence tolerates this 1-hour approximation.
        hour_min = now.hour * 100 + now.minute
        return 1330 <= hour_min <= 2000
