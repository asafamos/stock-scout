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


def _ib_symbol(ticker: str) -> str:
    """Convert a Yahoo/scanner ticker to IB's expected symbol format.

    IB uses a SPACE for class-share separators on US equities ("BRK B",
    "BF B"), while Yahoo and our scanner use a DOT ("BRK.B", "BF.B").
    For affected tickers, qualifyContracts() throws and the trade
    silently fails. Run every Stock() construction through this helper.

    Delegates to `core.trading.policy.normalize_ticker_for_ib` so the
    list of affected symbols is maintained in one place.
    """
    try:
        from core.trading.policy import normalize_ticker_for_ib
        return normalize_ticker_for_ib(ticker)
    except Exception:
        return str(ticker).strip().upper() if ticker else ticker


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

        # All retries exhausted — clean up the dangling IB() instance so
        # we don't leak file descriptors on repeated reconnect attempts.
        # The daemon runs forever; without this, every Gateway 2FA expiry
        # would burn 5 fds and eventually break with EMFILE. (Audit
        # finding #7 — monitor section.)
        try:
            if self._ib is not None:
                self._ib.disconnect()
        except Exception:
            pass
        self._ib = None
        logger.error("Failed to connect to IBKR after %d attempts: %s",
                     len(ids_to_try), last_err)
        self._connected = False
        return False

    def disconnect(self):
        # Always try to disconnect the IB instance, regardless of whether
        # we think we're "connected". Otherwise reconnect attempts that
        # failed at handshake stage leave dangling sockets. (Audit
        # finding #7 — monitor section.)
        if self._ib is not None:
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

    def get_fills(self) -> List[dict]:
        """Return this session's executions as normalized dicts.

        The atom of the event-sourced ledger (core/trading/ledger.py). Each
        row is keyed by IB's globally-unique, stable ``execId`` so the ledger
        can upsert idempotently — an execution can never be counted twice
        (kills the 2026-05-01 double-CLOSE class) and is never estimated
        (kills the KNX/PAAS phantom class), because we only ever record what
        IB actually executed.

        ``realized_pnl`` comes straight from IB's CommissionReport and is
        NET OF COMMISSIONS on both legs — the broker's own number, which
        reconciles to NetLiquidation by construction. Only populated on the
        closing (SELL, for long-only) leg; the opening leg reports the IB
        "unset" sentinel which we normalize to None.

        Note: ``ib.fills()`` is SESSION-scoped (roughly today / since gateway
        connect). The durable ledger accumulates across sessions by appending
        these every monitor cycle; it never re-derives, so a stale session
        cannot corrupt history.
        """
        if self.cfg.dry_run:
            return []
        if self._ib is None:
            return []
        _UNSET = 1.7976931348623157e308  # IB's UNSET_DOUBLE sentinel
        out: List[dict] = []
        try:
            fills = self._ib.fills()
        except Exception as e:
            logger.warning("get_fills: ib.fills() failed: %s", e)
            return []
        for f in fills:
            try:
                ex = f.execution
                exec_id = getattr(ex, "execId", "") or ""
                if not exec_id:
                    continue
                side_raw = (getattr(ex, "side", "") or "").upper()
                side = "BUY" if side_raw in ("BOT", "BUY") else (
                    "SELL" if side_raw in ("SLD", "SELL") else side_raw
                )
                t = getattr(ex, "time", None)
                # Normalize to tz-aware ISO (UTC) so downstream date math is safe.
                t_iso = None
                if t is not None:
                    try:
                        from datetime import timezone as _tz
                        tt = t if getattr(t, "tzinfo", None) else t.replace(tzinfo=_tz.utc)
                        t_iso = tt.isoformat()
                    except Exception:
                        t_iso = str(t)
                cr = getattr(f, "commissionReport", None)
                commission = None
                realized = None
                if cr is not None:
                    try:
                        c = float(getattr(cr, "commission", None))
                        if c == c and abs(c) < _UNSET / 2:
                            commission = round(c, 4)
                    except (TypeError, ValueError):
                        pass
                    try:
                        rp = float(getattr(cr, "realizedPNL", None))
                        # Filter NaN and the UNSET sentinel (opening legs).
                        if rp == rp and abs(rp) < _UNSET / 2:
                            realized = round(rp, 2)
                    except (TypeError, ValueError):
                        pass
                out.append({
                    "exec_id": exec_id,
                    "ticker": getattr(f.contract, "symbol", "") or "",
                    "side": side,
                    "shares": float(getattr(ex, "shares", 0) or 0),
                    "price": float(getattr(ex, "price", 0) or 0),
                    "time": t_iso,
                    "commission": commission,
                    "realized_pnl": realized,
                })
            except Exception as _e:
                logger.debug("get_fills: skipping malformed fill: %s", _e)
                continue
        return out

    def get_account_realized_pnl(self) -> Optional[float]:
        """IB's account-level DAILY realized P&L (net), or None if unavailable.

        A cheap independent cross-check for the ledger's today-realized:
        both should agree. Uses accountSummary RealizedPnL tag.
        """
        if self.cfg.dry_run or self._ib is None:
            return None
        try:
            for item in self._ib.accountSummary():
                if item.tag == "RealizedPnL":
                    return float(item.value)
        except Exception as e:
            logger.debug("get_account_realized_pnl failed: %s", e)
        return None

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
            contract = Stock(_ib_symbol(ticker), "SMART", "USD")
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
                # Try every field that might carry a CURRENT price.
                # IMPORTANT: `close` and `delayedClose` are YESTERDAY'S close
                # under marketDataType=3. They were originally in this list
                # as fallbacks but caused real harm: in pre/post-market when
                # last/delayedLast are NaN, returning yesterday's close as
                # "live price" would size stops/targets to a stale anchor and
                # IB would trigger them on the open as soon as today's price
                # diverged. (See audit 2026-04-30 finding #4.)
                # If no live tick is available, return None and let the caller
                # fall back to scan_price + slippage guard rather than trade
                # against day-old data.
                candidates = [
                    getattr(ticker_obj, "last", None),
                    getattr(ticker_obj, "delayedLast", None),
                    getattr(ticker_obj, "marketPrice", None),
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
            contract = Stock(_ib_symbol(ticker), "SMART", "USD")
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
            contract = Stock(_ib_symbol(ticker), "SMART", "USD")
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
            contract = Stock(_ib_symbol(ticker), "SMART", "USD")
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
        limit_price: float = 0.0,
    ) -> dict:
        """Buy + trailing stop + limit sell as OCA bracket.

        When one exit order fills, the other is automatically cancelled.
        Returns dict with order IDs for all three legs.

        2026-05-29 — entry slippage fix. If `limit_price` > 0 and
        CONFIG.entry_use_limit is True, the parent buy is a MARKETABLE
        LIMIT at limit_price instead of a MARKET order. This caps the
        worst-case fill: on a normal spread the limit fills immediately
        (it's set a small buffer above the live ask), but if the price
        has run away the order simply doesn't fill within the wait window
        and we cancel + report unfilled — the caller then skips the
        position rather than chasing the stock several % higher (which
        was costing ~1.93% avg / $74 total across the first 11 live buys).
        Passing limit_price=0 preserves the legacy MARKET behavior.
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

            contract = Stock(_ib_symbol(ticker), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Generate OCA group name (unique per trade)
            oca_group = _make_oca_group(ticker)

            # Leg 1: parent buy — MARKETABLE LIMIT (default) or MARKET (legacy).
            use_limit = (
                bool(getattr(self.cfg, "entry_use_limit", True))
                and limit_price and limit_price > 0
            )
            buy_order = Order()
            buy_order.action = "BUY"
            buy_order.totalQuantity = qty
            if use_limit:
                buy_order.orderType = "LMT"
                buy_order.lmtPrice = round(float(limit_price), 2)
                buy_order.tif = "DAY"
                logger.info(
                    "BUY LMT %d x %s @ $%.2f (marketable limit — caps slippage)",
                    qty, ticker, buy_order.lmtPrice,
                )
            else:
                buy_order.orderType = "MKT"
                # 2026-05-19: explicit TIF=DAY silences "Error 10349: TIF was
                # set to DAY based on order preset" warning that polluted logs.
                # GTC doesn't apply to MKT (fills immediately) — DAY is correct.
                buy_order.tif = "DAY"
                logger.info("BUY MKT %d x %s (no limit_price — legacy path)", qty, ticker)
            buy_order.transmit = True
            buy_trade = self._ib.placeOrder(contract, buy_order)

            # Wait for fill. CRITICAL: if the buy doesn't fill we must NOT
            # proceed to place protective orders or report success —
            # otherwise the tracker records a phantom OPEN with zero qty,
            # then reconciliation later writes a fake CLOSE with estimated
            # P&L. (See KNX phantom 2026-04-28; audit finding #2.)
            #
            # For a LIMIT, the wait is intentional: a marketable limit fills
            # near-instantly on a normal spread; if it's still unfilled when
            # the window elapses, the price ran above our limit and we
            # WANT to bail (no-chase). MARKET keeps the legacy 30s.
            _fill_wait = (
                int(getattr(self.cfg, "entry_limit_fill_wait_sec", 20))
                if use_limit else 30
            )
            for _ in range(_fill_wait):
                self._ib.sleep(1)
                if buy_trade.orderStatus.status == "Filled":
                    break
                # Also exit early on hard rejects so we don't burn the window
                if buy_trade.orderStatus.status in ("Cancelled", "Inactive", "ApiCancelled"):
                    break

            # LIMIT no-chase: if a marketable limit didn't fully fill in the
            # window, cancel the remainder. A partial fill is kept (the
            # protective-order qty is truncated to filled_qty below); a zero
            # fill returns the standard unfilled path (caller skips position).
            if use_limit and buy_trade.orderStatus.status not in (
                "Filled", "Cancelled", "Inactive", "ApiCancelled"
            ):
                _filled_so_far = int(buy_trade.orderStatus.filled or 0)
                logger.warning(
                    "BUY LMT %s unfilled after %ds (price ran above $%.2f) — "
                    "cancelling remainder (filled %d/%d, no-chase)",
                    ticker, _fill_wait, buy_order.lmtPrice, _filled_so_far, qty,
                )
                try:
                    self._ib.cancelOrder(buy_order)
                    self._ib.sleep(1)
                except Exception:
                    pass

            filled = buy_trade.orderStatus.avgFillPrice or 0.0
            filled_qty = int(buy_trade.orderStatus.filled or 0)
            buy_status = buy_trade.orderStatus.status

            # Hard reject if buy didn't fill — return status=Error so the
            # caller skips add_position and skips placing protective orders
            # for shares we don't own.
            if buy_status != "Filled" or filled_qty <= 0 or filled <= 0:
                err = (
                    f"buy did not fill: status={buy_status}, "
                    f"filled_qty={filled_qty}, avg_price={filled}"
                )
                logger.error("BRACKET buy unfilled for %s: %s", ticker, err)

                # AUTO-BLOCK on IB-side restrictions (added 2026-05-05).
                # Some tickers fail at IB's gate regardless of our gates:
                # Error 201 with messages like "verify with the token we
                # emailed you" (foreign-listed stocks needing extra
                # verification — VFS/VinFast hit this 2026-05-05 14:03).
                # Without auto-block, the same ticker re-ranks #1 in the
                # next pipeline, the evaluator burns ~6s setting it up,
                # IB rejects, position-size budget is held momentarily,
                # logs spam. Auto-block for 90 days surfaces a clean
                # Telegram alert ONCE and skips the ticker thereafter.
                try:
                    err_codes = [
                        getattr(le, "errorCode", 0) or 0
                        for le in (buy_trade.log or [])
                    ]
                    err_msgs = " | ".join(
                        getattr(le, "message", "") or ""
                        for le in (buy_trade.log or [])
                        if getattr(le, "errorCode", 0)
                    )
                    # IB Error 201 = "Order rejected" (account-side reasons).
                    # Combined with phrases like "verify" or "Not allowed"
                    # this is a permanent restriction, not a transient one.
                    is_permanent_reject = (
                        201 in err_codes and (
                            "verify" in err_msgs.lower()
                            or "not allowed to open" in err_msgs.lower()
                            or "minimum of 2000" in err_msgs.lower()
                        )
                    )
                    if is_permanent_reject:
                        from core.control.command_bus import cmd_block
                        block_result = cmd_block(
                            ticker, days=90, source="auto-error-201"
                        )
                        logger.warning(
                            "AUTO-BLOCKED %s for 90d (IB Error 201): %s",
                            ticker, block_result,
                        )
                        # Telegram alert via the existing notifications path
                        try:
                            from core.trading import notifications as _nf
                            _nf.notify_error(
                                f"Auto-blocked {ticker}",
                                f"⛔ <b>{ticker} auto-blocked 90d</b>\n"
                                f"IB Error 201 (permanent reject — "
                                f"requires manual Client Portal action).\n"
                                f"<pre>{err_msgs[:200]}</pre>\n"
                                f"Unblock when ready: <code>/unblock {ticker}</code>"
                            )
                        except Exception as _ne:
                            logger.debug("auto-block notify failed: %s", _ne)
                except Exception as _be:
                    logger.warning("auto-block check failed: %s", _be)

                # Best-effort cancel the buy so it doesn't fill late
                try:
                    self._ib.cancelOrder(buy_trade.order)
                except Exception:
                    pass
                return {
                    "buy": TradeResult(
                        ticker=ticker, action="BUY", order_type="MKT",
                        quantity=qty, filled_price=0.0, status="Error",
                        error=err, order_id=buy_trade.order.orderId,
                    ),
                    "trailing_stop": TradeResult(
                        ticker=ticker, action="SELL", order_type="TRAIL",
                        quantity=qty, filled_price=0.0, status="Error",
                        error="skipped — buy did not fill",
                    ),
                    "limit_sell": TradeResult(
                        ticker=ticker, action="SELL", order_type="LMT",
                        quantity=qty, filled_price=0.0, status="Error",
                        error="skipped — buy did not fill",
                    ),
                }

            # Partial fill — log + truncate qty for protective orders.
            # The position tracker should also use filled_qty, not the
            # ordered qty (handled in order_manager.py).
            if filled_qty < qty:
                logger.warning(
                    "PARTIAL BUY %s: ordered %d, filled %d @ $%.2f — "
                    "using filled_qty for protective orders",
                    ticker, qty, filled_qty, filled,
                )
                qty = filled_qty

            # Account-tier-aware day-trade protection.
            #
            # Tier rules (see risk_manager.get_account_tier):
            #   TIER_SUB_2K (<$2k):     LMT goodAfterTime — IB strict on
            #                            small accounts, will reject same-day
            #                            sells.
            #   TIER_2K_TO_25K (cash):  LMT goodAfterTime — T+1 settlement
            #                            means same-day profit round-trip
            #                            uses unsettled funds.
            #   TIER_25K_PLUS (margin): NO goodAfterTime — PDT-eligible,
            #                            T+0 on margin, can intraday round-trip.
            #
            # In ALL tiers, TRAIL has NO goodAfterTime — stop-loss exits
            # don't violate day-trade rules and the position MUST be
            # protected from flash crashes / news events from minute zero.
            # (Audit 2026-04-30 finding #5.)
            account_tier = "sub_2k"  # conservative default if read fails
            try:
                net_liq = float(self.get_net_liquidation() or 0)
                if net_liq >= 25000:
                    account_tier = "margin_pdt"
                elif net_liq >= 2000:
                    account_tier = "cash"
            except Exception:
                pass
            apply_goodaftertime_to_lmt = account_tier in ("sub_2k", "cash")

            _next_open = self._next_trading_day_open_utc() if apply_goodaftertime_to_lmt else ""
            logger.info(
                "Bracket: tier=%s, TRAIL immediate, LMT goodAfterTime=%s",
                account_tier, _next_open or "none",
            )

            trail_order = Order()
            trail_order.action = "SELL"
            trail_order.totalQuantity = qty
            trail_order.orderType = "TRAIL"
            trail_order.trailingPercent = trail_pct
            trail_order.tif = "GTC"
            # NO goodAfterTime by default — stop-loss must be active from the
            # moment we hold the position (flash-crash / news protection).
            #
            # Optional: TRAIL activation delay for sub-2k tier ONLY. Reads
            # TRADE_TRAIL_DELAY_HOURS_SUB_2K (default 0 = immediate). When >0,
            # the TRAIL goes live N hours after placement — protects against
            # intraday-noise stopouts and Good-Faith-Violation risk on cash
            # accounts that round-trip same-day. TRADEOFF: loses flash-crash
            # protection during the delay window. Use only if the trail-floor
            # widening (#1) is insufficient.
            try:
                import os as _os
                _delay_hours = float(_os.getenv("TRADE_TRAIL_DELAY_HOURS_SUB_2K", "0") or 0)
            except Exception:
                _delay_hours = 0.0
            if _delay_hours > 0 and account_tier == "sub_2k":
                from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                _activate_at = _dt.now(_tz.utc) + _td(hours=_delay_hours)
                # IB goodAfterTime format: "YYYYMMDD HH:MM:SS UTC"
                trail_order.goodAfterTime = _activate_at.strftime("%Y%m%d %H:%M:%S UTC")
                logger.info(
                    "TRAIL goodAfterTime=%s (sub_2k tier, delay %.1fh) — "
                    "stop-loss inactive during cushion window",
                    trail_order.goodAfterTime, _delay_hours,
                )
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
                if apply_goodaftertime_to_lmt:
                    limit_order.goodAfterTime = _next_open
                limit_order.ocaGroup = oca_group
                limit_order.ocaType = 1
                limit_order.transmit = True
                limit_trade = self._ib.placeOrder(contract, limit_order)
            else:
                limit_trade = None

            # Audit H4 (2026-05-01): wait up to 5s for protective orders to
            # transition out of "PendingSubmit" so the caller sees their true
            # status. Previously a 2s sleep was sometimes insufficient on
            # slow connections — caller would see status=PendingSubmit and
            # assume success, then the order errored 30s later when nobody
            # was listening. Native IB bracket (transmit=False/True) is NOT
            # safe for our market-buy + potential-partial-fill flow because
            # IB sizes child legs from the parent's `totalQuantity` rather
            # than `filledQuantity` — that would over-sell on partials. The
            # existing place-buy → wait-fill → place-protective flow is the
            # correct pattern for this design; this commit just tightens the
            # status-confirmation window so we don't return false-success.
            for _wait_i in range(10):  # up to 5s in 0.5s increments
                self._ib.sleep(0.5)
                trail_status = trail_trade.orderStatus.status
                limit_status = limit_trade.orderStatus.status if limit_trade else "n/a"
                # Both legs settled into a known state (good or bad)?
                if trail_status not in ("PendingSubmit", "ApiPending", ""):
                    if limit_trade is None or limit_status not in ("PendingSubmit", "ApiPending", ""):
                        break

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

            contract = Stock(_ib_symbol(ticker), "SMART", "USD")
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
            contract = Stock(_ib_symbol(ticker), "SMART", "USD")
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
                            new_trail_pct: float,
                            allow_loosen: bool = False) -> TradeResult:
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
        # ── Runtime invariants (2026-05-15) ──
        # The caller (ratchet / break-even / earnings) should only EVER
        # call this with a tighter trail %. A bug that loosens trail
        # would silently un-protect the position. The 0.5-9.0 range
        # matches MIN/MAX_INITIAL_TRAIL_PCT in order_manager.py and the
        # config clamps; anything outside is a logic error.
        if not (0.5 <= new_trail_pct <= 9.0):
            raise ValueError(
                f"modify_trailing_pct: new_trail_pct {new_trail_pct:.2f}% out of [0.5, 9.0] "
                f"safety band (order #{order_id}). This is a logic error in the caller."
            )

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

            # SAFETY GUARD (2026-07-08, N4 from NEXT_SESSION.md):
            # ratchet must only tighten TRAIL, never loosen it. A wider
            # trail = looser stop = larger unprotected downside. Every
            # caller in monitor_positions.py has its own "already tighter,
            # skip" gate above modify_trailing_pct, but the assert here
            # is defense-in-depth in case a future caller forgets. Set
            # allow_loosen=True only for explicit rollback flows.
            if not allow_loosen and old_pct > 0 and new_trail_pct > old_pct + 0.01:
                logger.error(
                    "modify_trailing_pct #%d: refusing LOOSEN %.2f%% → %.2f%% "
                    "(new > old). Pass allow_loosen=True to override.",
                    order_id, old_pct, new_trail_pct,
                )
                return TradeResult(
                    ticker="", action="SELL", order_type="TRAIL",
                    quantity=0, filled_price=0.0, status="Rejected_Loosen",
                    order_id=order_id,
                    error=f"refuse loosen {old_pct:.2f}% → {new_trail_pct:.2f}%",
                )

            target_trade.order.trailingPercent = float(new_trail_pct)
            target_trade.order.transmit = True  # explicit — don't inherit
            # Re-submit with same orderId — IB treats this as a modification.
            # Capture the returned Trade — its log/orderStatus reflects the
            # ACTUAL IB response (success vs Error 103 / 161 / 202 / etc.).
            # The previous code threw away the return value, then queried
            # openTrades() — which returns the same in-memory object with
            # the LOCALLY-MUTATED trailingPercent (we just set it 2 lines
            # above), masking IB-side rejection. Real-world failure: ORCL
            # 2026-05-05 12:06 + 12:14 — Error 103 "Duplicate order id"
            # cancelled the modify, the cache showed trailingPercent=3.0,
            # the verify said "✓ MODIFIED ... verified live", and IB still
            # had 4.6%. By inspecting the returned Trade.log for any
            # errorCode > 0 in the seconds after placeOrder, we catch
            # rejections that the cached spec hides.
            modify_trade = self._ib.placeOrder(target_trade.contract,
                                               target_trade.order)
            self._ib.sleep(3)
            # Authoritative check #1: did the returned Trade collect any
            # errorCode in its log? Any non-zero errorCode means IB rejected
            # the modify (103 = duplicate id, 161 = no order to modify,
            # 201 = rejected, 202 = cancelled).
            try:
                rejection_code = 0
                rejection_msg = ""
                for log_entry in (modify_trade.log or []):
                    ec = getattr(log_entry, "errorCode", 0) or 0
                    if ec and ec > 0:
                        rejection_code = ec
                        rejection_msg = getattr(log_entry, "message", "") or "(no msg)"
                        break

                if rejection_code:
                    # FALLBACK on Error 103 "Duplicate order id" (added 2026-05-05).
                    # This happens when the order belongs to a clientId that
                    # the current connection doesn't own (e.g. ORCL #1172 was
                    # placed under clientId=1 but monitor connects as clientId=2).
                    # Modify will keep failing forever. Cancel + place a fresh
                    # order under the current clientId to take ownership.
                    # Brief unprotected window (~5s) — acceptable because the
                    # alternative is permanent ratchet failure on this position.
                    if rejection_code == 103:
                        logger.warning(
                            "MODIFY TRAIL %s #%d hit Error 103 — falling back to cancel+replace",
                            target_trade.contract.symbol, order_id,
                        )
                        try:
                            from ib_insync import Order as _Order, IB as _IB
                            _ticker = target_trade.contract.symbol
                            _qty = int(target_trade.order.totalQuantity)
                            _oca = target_trade.order.ocaGroup
                            _contract = target_trade.contract
                            # Restore original target_order spec before cancel
                            target_trade.order.trailingPercent = old_pct

                            # OWNERSHIP-AWARE CANCEL (added 2026-05-05):
                            # The 103 means our current clientId doesn't own
                            # the order. cancelOrder from a non-owning client
                            # is silently ignored — leaves the order ALIVE
                            # while we place the replacement → 2 active TRAILs
                            # (real-world failure: ORCL #1172 + #1700 both
                            # alive 2026-05-05 17:18). Fix: open a TEMPORARY
                            # connection as the OWNING clientId, cancel from
                            # there, then disconnect. Read the owning clientId
                            # off the target order itself.
                            # 2026-05-21 REWRITE: the old fallback was cancel-then-place-fresh.
                            # That fails fatally for sub-$2k cash accounts because IB
                            # rejects the replacement SELL with Error 201 ("MINIMUM
                            # 2000 USD REQUIRED") — IB sees a fresh SELL order
                            # without its bracket context and treats it as a
                            # potential short. Real-world impact: every ratchet
                            # and break-even attempt would silently fail; trail
                            # stayed at 4% forever; portfolio gave back ALL
                            # ILMN+LYB peak gains 5/15-5/20 because we couldn't
                            # tighten.
                            #
                            # New approach: MODIFY in-place via secondary
                            # connection (same orderId, just updated trailingPercent).
                            # IB treats this as a modify of the original bracket
                            # leg, which is allowed even for cash<$2k. We NEVER
                            # cancel the live protection until we have a working
                            # replacement.
                            # 2026-05-21 KEY FIX: read the OWNER clientId from
                            # orderStatus.clientId, NOT from order.clientId.
                            # The latter gets overwritten by ib_insync to the
                            # CURRENT client when we call placeOrder. The
                            # former preserves the original placing client.
                            # Real-world confirmed: ORCL #12 was placed by a
                            # scan_and_trade.sh random-fallback clientId 390;
                            # target_trade.order.clientId reported 2 (us),
                            # target_trade.orderStatus.clientId reported 390
                            # (truth). Manual modify via 390 succeeded
                            # instantly at 10:46 IL Wed 2026-05-21.
                            owner_cid_from_order = int(getattr(target_trade.order, "clientId", 0) or 0)
                            owner_cid_from_status = int(getattr(target_trade.orderStatus, "clientId", 0) or 0)
                            self_cid = int(getattr(self._ib.client, "clientId", 0) or 0)
                            # Prefer orderStatus.clientId because order.clientId
                            # gets mutated by our own placeOrder attempt above.
                            owner_cid = owner_cid_from_status or owner_cid_from_order
                            # Candidate clientIds to try in order of likelihood:
                            #   1. The status-reported owner (most accurate)
                            #   2. 1 (default auto-trade clientId)
                            #   3. The order-reported clientId (may be stale)
                            #   4. Any clientId seen in the order's log history
                            owner_candidates = []
                            seen = set()
                            for cid in (owner_cid_from_status, 1, owner_cid_from_order):
                                if cid and cid != self_cid and cid not in seen:
                                    owner_candidates.append(cid)
                                    seen.add(cid)
                            modify_via_secondary_ok = False

                            for cand_cid in owner_candidates:
                                if modify_via_secondary_ok:
                                    break
                                logger.info(
                                    "TRAIL #%d: trying secondary MODIFY via clientId=%d "
                                    "(self=%d, status_owner=%d, order_owner=%d)",
                                    order_id, cand_cid, self_cid,
                                    owner_cid_from_status, owner_cid_from_order,
                                )
                                try:
                                    _aux = _IB()
                                    _aux.connect("127.0.0.1", 7496,
                                                 clientId=cand_cid, timeout=8)
                                    _aux.reqAllOpenOrders()
                                    _aux.sleep(2)
                                    matched = False
                                    for _t in _aux.openTrades():
                                        if (_t.order.orderId == order_id
                                                and _t.contract.symbol == _ticker
                                                and _t.order.action == "SELL"):
                                            # KEY: same orderId + new trailingPercent = MODIFY,
                                            # not new order. IB accepts even cash<$2k.
                                            _t.order.trailingPercent = float(new_trail_pct)
                                            _aux.placeOrder(_t.contract, _t.order)
                                            _aux.sleep(3)
                                            mod_status = _t.orderStatus.status
                                            mod_errors = [
                                                (le.errorCode, le.message[:80])
                                                for le in (_t.log or [])
                                                if (le.errorCode or 0) > 0
                                                   and le.errorCode != 103
                                            ]
                                            logger.info(
                                                "Secondary MODIFY #%d via clientId=%d: "
                                                "status=%s errors=%s",
                                                order_id, cand_cid, mod_status, mod_errors,
                                            )
                                            matched = True
                                            if (mod_status in ("PreSubmitted",
                                                               "Submitted",
                                                               "PendingSubmit")
                                                    and not mod_errors):
                                                modify_via_secondary_ok = True
                                            break
                                    if not matched:
                                        logger.debug(
                                            "clientId=%d doesn't see order #%d — "
                                            "trying next candidate", cand_cid, order_id,
                                        )
                                    _aux.disconnect()
                                except Exception as _ce:
                                    logger.debug(
                                        "Secondary in-place modify via cid=%d failed: %s — "
                                        "trying next candidate", cand_cid, _ce,
                                    )

                            if modify_via_secondary_ok:
                                logger.info(
                                    "✓ MODIFY-VIA-SECONDARY SUCCESS %s #%d: "
                                    "trail %.2f%% → %.2f%% (cash<$2k safe path)",
                                    _ticker, order_id, old_pct, new_trail_pct,
                                )
                                return TradeResult(
                                    ticker=_ticker, action="SELL",
                                    order_type="TRAIL", quantity=_qty,
                                    filled_price=0.0,
                                    status="PreSubmitted",
                                    order_id=order_id,
                                )

                            # If secondary modify didn't work (no candidate
                            # clientId succeeded), we DO NOT cancel+replace —
                            # that path is poison for cash<$2k accounts. We
                            # leave the existing trail in place at the old
                            # percent and surface a loud error so the user
                            # knows the tightening attempt failed.
                            logger.error(
                                "TRAIL TIGHTEN FAILED %s #%d: "
                                "self_cid=%d candidates_tried=%s. "
                                "Leaving existing trail at %.2f%% (NOT cancelling — "
                                "cancel+replace fails for cash<$2k). "
                                "Position remains protected at the wider trail.",
                                _ticker, order_id, self_cid, owner_candidates, old_pct,
                            )
                            # Fall through to the original error return
                        except Exception as _fe:
                            logger.error(
                                "FALLBACK exception for %s: %s",
                                target_trade.contract.symbol, _fe,
                            )

                    # Original rejection-return path (non-103 errors or
                    # fallback also failed).
                    logger.error(
                        "MODIFY TRAIL %s #%d REJECTED by IB: errorCode=%d %s",
                        target_trade.contract.symbol, order_id,
                        rejection_code, rejection_msg,
                    )
                    return TradeResult(
                        ticker=target_trade.contract.symbol, action="SELL",
                        order_type="TRAIL", quantity=int(target_trade.order.totalQuantity),
                        filled_price=0.0, status="Error", order_id=order_id,
                        error=f"IB error {rejection_code}: {rejection_msg}",
                    )
            except Exception as _le:
                logger.debug("modify log inspection failed (non-fatal): %s", _le)
            # Authoritative check #2: orderStatus on the modify-trade.
            try:
                ms = modify_trade.orderStatus.status
                if ms in ("Cancelled", "ApiCancelled", "Inactive", "Rejected"):
                    logger.error(
                        "MODIFY TRAIL %s #%d REJECTED: modify-trade status=%s",
                        target_trade.contract.symbol, order_id, ms,
                    )
                    return TradeResult(
                        ticker=target_trade.contract.symbol, action="SELL",
                        order_type="TRAIL", quantity=int(target_trade.order.totalQuantity),
                        filled_price=0.0, status="Error", order_id=order_id,
                        error=f"modify-trade status={ms}",
                    )
            except Exception:
                pass

            ticker = target_trade.contract.symbol
            qty = int(target_trade.order.totalQuantity)

            # POST-VERIFY: re-fetch the live order from IB and confirm the
            # trailingPercent actually changed. IB acks the placeOrder
            # before validating the modification — a rejection (errorEvent
            # 161/202: "no such order modify" or "invalid order state")
            # may arrive after our sleep. Without verification, the tracker
            # would record success while IB still has the old %. (Audit
            # finding #8.)
            actual_pct = None
            actual_status = None
            actual_log_tail = ""
            try:
                # Re-pull fresh state via reqAllOpenOrders to get current values
                self._ib.reqAllOpenOrders()
                self._ib.sleep(1)
                for t in self._ib.openTrades():
                    if t.order.orderId == order_id:
                        actual_pct = float(getattr(t.order, "trailingPercent", -1) or -1)
                        # CRITICAL (added 2026-05-05): also capture orderStatus
                        # and the most-recent log entry. The previous version
                        # only checked trailingPercent — so a Cancelled trade
                        # whose in-memory order spec still showed the requested
                        # percent (because the placeOrder sent that spec)
                        # would falsely report "verified". Real-world failure:
                        # ORCL 2026-05-05 12:06:03 — Error 103 "Duplicate
                        # order id" cancelled the modify, but trailingPercent
                        # in openTrades reflected the SENT spec (3.0%), not
                        # IB's accepted spec (still 4.6%). Tracker recorded
                        # "modify success" → drift between tracker and IB.
                        try:
                            actual_status = t.orderStatus.status
                        except Exception:
                            actual_status = None
                        try:
                            if t.log:
                                last = t.log[-1]
                                actual_log_tail = (
                                    f"{last.status}: {last.message or '(no msg)'}"
                                )
                        except Exception:
                            pass
                        break
            except Exception as _verr:
                logger.warning("Post-verify of TRAIL #%d failed: %s", order_id, _verr)

            if actual_pct is None:
                # Order disappeared — likely cancelled. That's a hard error.
                logger.error(
                    "MODIFY TRAIL %s #%d: order not found after submit — treating as REJECTED",
                    ticker, order_id,
                )
                return TradeResult(
                    ticker=ticker, action="SELL", order_type="TRAIL",
                    quantity=qty, filled_price=0.0, status="Error",
                    order_id=order_id, error="order disappeared after modify",
                )

            # Status check — Cancelled / Inactive / Rejected means the modify
            # didn't take effect even if trailingPercent looks right in cache.
            if actual_status in ("Cancelled", "Inactive", "Rejected", "ApiCancelled"):
                logger.error(
                    "MODIFY TRAIL %s #%d REJECTED: orderStatus=%s, last log: %s",
                    ticker, order_id, actual_status, actual_log_tail,
                )
                return TradeResult(
                    ticker=ticker, action="SELL", order_type="TRAIL",
                    quantity=qty, filled_price=0.0, status="Error",
                    order_id=order_id,
                    error=f"modify rejected: status={actual_status} ({actual_log_tail})",
                )

            # Allow tiny float jitter — but the modification was rejected
            # if the live % is materially different from what we set.
            if abs(actual_pct - new_trail_pct) > 0.05:
                logger.error(
                    "MODIFY TRAIL %s #%d REJECTED: requested %.1f%% but IB still has %.1f%%",
                    ticker, order_id, new_trail_pct, actual_pct,
                )
                return TradeResult(
                    ticker=ticker, action="SELL", order_type="TRAIL",
                    quantity=qty, filled_price=0.0, status="Error",
                    order_id=order_id,
                    error=f"modify rejected: IB has {actual_pct:.2f}% not {new_trail_pct:.2f}%",
                )

            logger.info(
                "✓ MODIFIED TRAIL %s #%d: %.1f%% → %.1f%% (verified live)",
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
        """Sell `qty` shares — uses LIMIT+GTC to bypass sub-$2k Error 201.

        Sub-$2k accounts get IB Error 201 on MarketOrder SELL because IB's
        preset auto-converts TIF to DAY, then flags it as margin/short and
        rejects. LIMIT with tif=GTC at an aggressive price (0.5% below the
        live mark) bypasses that path — the order is a plain long-close,
        not a margin transaction, and fills near-immediately in liquid
        names. Falls back to MarketOrder only if the limit doesn't fill
        in ~10 seconds AND the account is above the $2k tier where market
        orders work reliably.
        """
        if self.cfg.dry_run:
            logger.info("[DRY RUN] SELL LMT %d x %s", qty, ticker)
            return TradeResult(
                ticker=ticker, action="SELL", order_type="LMT",
                quantity=qty, filled_price=0.0, status="DRY_RUN",
            )
        try:
            from ib_insync import Stock, LimitOrder, MarketOrder
            contract = Stock(_ib_symbol(ticker), "SMART", "USD")
            self._ib.qualifyContracts(contract)

            # Snapshot current mark for aggressive-limit pricing.
            ref_price = 0.0
            try:
                tickers = self._ib.reqTickers(contract)
                if tickers:
                    t0 = tickers[0]
                    for cand in (t0.marketPrice(), t0.last, t0.close):
                        try:
                            v = float(cand)
                        except (TypeError, ValueError):
                            continue
                        import math as _m
                        if _m.isfinite(v) and v > 0:
                            ref_price = v
                            break
            except Exception as _e:
                logger.warning("SELL %s: reqTickers failed, will retry with 0 ref: %s", ticker, _e)

            if ref_price > 0:
                # LIMIT 0.5% below live mark → near-instant fill in liquid names,
                # avoids the DAY-preset → Error 201 path.
                limit_price = round(ref_price * 0.995, 2)
                order = LimitOrder("SELL", qty, limit_price)
                order.tif = "GTC"
                logger.info("SELL LMT %d x %s @ %.2f (mark %.2f, GTC)",
                            qty, ticker, limit_price, ref_price)
                trade = self._ib.placeOrder(contract, order)

                # Wait up to 10s for the limit to fill.
                for _ in range(10):
                    self._ib.sleep(1)
                    if trade.orderStatus.status == "Filled":
                        break

                if trade.orderStatus.status == "Filled":
                    return TradeResult(
                        ticker=ticker, action="SELL", order_type="LMT",
                        quantity=qty,
                        filled_price=trade.orderStatus.avgFillPrice or limit_price,
                        status="Filled",
                        order_id=trade.order.orderId,
                    )

                # Not filled — cancel and consider fallback.
                logger.warning("SELL LMT %s not filled in 10s (status=%s); cancelling",
                               ticker, trade.orderStatus.status)
                try:
                    self._ib.cancelOrder(trade.order)
                    self._ib.sleep(2)
                except Exception:
                    pass

                # Fallback to MarketOrder ONLY if account is >= $2k tier.
                net_liq = 0.0
                try:
                    net_liq = float(self.get_net_liquidation() or 0)
                except Exception:
                    pass

                if net_liq < 2000:
                    logger.error(
                        "SELL %s: limit unfilled AND account sub-$2k — cannot fallback "
                        "to MKT (Error 201). Will retry next cycle with fresh mark.",
                        ticker,
                    )
                    return TradeResult(
                        ticker=ticker, action="SELL", order_type="LMT",
                        quantity=qty, filled_price=0.0,
                        status="Cancelled_LimitUnfilled_SubTier",
                        order_id=trade.order.orderId,
                    )

            # Either no ref price OR account >= $2k → use MarketOrder.
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
            logger.error("SELL failed for %s: %s", ticker, e)
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
                    # Price fields — needed by drift-check adoption to
                    # recover the original stop %, limit target, and stop
                    # price for an IB-only position. None when the order
                    # type doesn't carry that field (e.g. MKT has no lmt).
                    "trailing_percent": getattr(t.order, "trailingPercent", None),
                    "lmt_price": getattr(t.order, "lmtPrice", None),
                    "aux_price": getattr(t.order, "auxPrice", None),
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
