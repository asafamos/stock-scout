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
import signal
import time
from contextlib import contextmanager
from datetime import datetime, date


# ── Cycle timeout ────────────────────────────────────────────────────
# IB can hang indefinitely on portfolio()/fills()/openTrades() if the
# network blips or the gateway stalls. A per-cycle alarm ensures we
# abort and retry next cycle instead of freezing the daemon.
CYCLE_TIMEOUT_SECONDS = 120  # 2 min — generous but bounded


class MonitorTimeout(Exception):
    pass


@contextmanager
def _cycle_timeout(seconds: int):
    """SIGALRM-based hard timeout. Unix only; safe because the monitor
    daemon runs as main thread of its own process.
    """
    def _handler(signum, frame):
        raise MonitorTimeout(f"run_check exceeded {seconds}s budget")
    # Only install alarm if we're in the main thread (signals don't work otherwise)
    try:
        old = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # Not main thread — skip timeout protection
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CHECK_INTERVAL = 300  # 5 minutes

# Cooldown state — prevents Telegram spam and repeated resubmit attempts
# when IB rejects orders with the $2000-minimum cash-account rule.
# Keyed by (ticker, event_type). Value: epoch seconds of last alert.
_ALERT_COOLDOWN: dict[tuple, float] = {}
_ALERT_COOLDOWN_SECONDS = 3600  # 1 hour between the same alert
_RESUBMIT_COOLDOWN: dict[str, float] = {}  # per-ticker
_RESUBMIT_COOLDOWN_SECONDS = 1800  # 30 min between resubmit attempts

# Transient-miss buffer for protective-order alerts (added 2026-05-13).
# Counts consecutive monitor cycles where a position had NO protective
# orders. Only fires the CRITICAL Telegram alert after the count reaches
# _PROTECTION_MISS_THRESHOLD, suppressing single-cycle artifacts caused
# by deploys, restarts, or temporary IB API hiccups (which auto-resubmit
# normally fixes on the same cycle). The 2026-05-07 user-visible 21:48
# CRITICAL alerts on LYB+RSI were exactly this class — protections were
# fully restored within 5 minutes but the alarms were already sent.
_PROTECTION_MISS_COUNT: dict[str, int] = {}
_PROTECTION_MISS_THRESHOLD = 2  # 2 cycles = ~10 minutes
# IBKR error fragments that mean "no point retrying — the account rule blocks it"
_ACCOUNT_RESTRICTION_ERRORS = (
    "minimum of 2000",
    "purchase on margin",
    "sell short",
)


def _cooldown_ok(key, state, seconds):
    """Return True if enough time passed since last event for this key."""
    now = time.time()
    last = state.get(key, 0)
    if now - last < seconds:
        return False
    state[key] = now
    return True


def _is_account_restriction_error(msg: str) -> bool:
    low = (msg or "").lower()
    return any(frag in low for frag in _ACCOUNT_RESTRICTION_ERRORS)


# ── Opportunistic trade trigger (audit followup 2026-05-05) ──
# Module-level cooldown so we don't re-fire on every monitor cycle if
# multiple positions close in quick succession. Persists per-process.
_OPPORTUNISTIC_LAST_FIRED: float = 0.0


def _try_opportunistic_buy(client, tracker, notify, reason: str = "manual"):
    """Re-evaluate the latest scan against current state and buy if eligible.

    Triggered after a position closes intraday — the freed cash + slot
    might be the best opportunity of the day, and waiting for the next
    scheduled pipeline run (up to 4h away) wastes that window.

    Safety: uses the SAME order_manager.execute_recommendations path as
    the scheduled pipeline. All risk gates apply identically. The only
    difference is the trigger source (close event instead of timer).

    Skips when:
      - Feature disabled (cfg.opportunistic_buy_enabled)
      - Cooldown not elapsed (cfg.opportunistic_buy_cooldown_sec)
      - Market closed (no buys outside RTH)
      - Latest scan stale (handled by _load_scan_results staleness check)
      - max_open_positions or max_daily_buys already reached
        (handled by risk_manager.can_open_position)
    """
    import time as _time
    global _OPPORTUNISTIC_LAST_FIRED

    from core.trading.config import CONFIG
    from core.trading import notifications as _n

    if not getattr(CONFIG, "opportunistic_buy_enabled", True):
        logger.debug("opportunistic_buy disabled in cfg")
        return

    cooldown = float(getattr(CONFIG, "opportunistic_buy_cooldown_sec", 300))
    elapsed = _time.time() - _OPPORTUNISTIC_LAST_FIRED
    if elapsed < cooldown:
        logger.info(
            "opportunistic_buy cooldown: %.0fs elapsed, need %.0fs — skipping",
            elapsed, cooldown,
        )
        return

    # Market-hours guard. Outside RTH no protective orders fill reliably,
    # so we shouldn't open new positions either.
    if not client.is_market_open():
        logger.info("opportunistic_buy: market closed — skipping")
        return

    # Capacity guard: do we have room for another buy?
    open_count = tracker.open_count
    if open_count >= CONFIG.max_open_positions:
        logger.info(
            "opportunistic_buy: at max_open_positions (%d) — skipping",
            open_count,
        )
        return

    daily_count = tracker.daily_buy_count()
    if daily_count >= CONFIG.max_daily_buys:
        logger.info(
            "opportunistic_buy: daily buy limit reached (%d/%d) — skipping",
            daily_count, CONFIG.max_daily_buys,
        )
        return

    logger.info(
        "🎯 OPPORTUNISTIC TRADE TRIGGER (reason=%s, open=%d/%d, "
        "daily=%d/%d) — re-evaluating latest scan...",
        reason, open_count, CONFIG.max_open_positions,
        daily_count, CONFIG.max_daily_buys,
    )

    try:
        from core.trading.risk_manager import RiskManager
        from core.trading.order_manager import OrderManager
        # CRITICAL BUG FIX (2026-05-14): OrderManager.__init__ accepts only
        # `config`, but old code passed (client, risk, tracker, CONFIG) — 4
        # positional args. Result: every opportunistic_buy crashed silently
        # with "takes from 1 to 2 positional arguments but 5 were given".
        # This means after every position close, the system attempted to
        # find a replacement candidate but the call CRASHED — so we never
        # actually bought replacements organically.
        # Fix: pass only CONFIG. OrderManager builds its own client/risk/
        # tracker internally (see __init__ in order_manager.py).
        mgr = OrderManager(CONFIG)

        # IMPORTANT: order_manager.execute_recommendations() will:
        #   1. Re-load the latest scan (with our fresh staleness check)
        #   2. Re-filter candidates (live IB held positions are deduped)
        #   3. For each top candidate: re-fetch live price, re-rank,
        #      run can_open_position (all 12+ gates), execute
        #   4. Stop on max_daily_buys
        # No special "opportunistic" code path — same logic as scheduled run.
        results = mgr.execute_recommendations()

        bought = [r for r in results if r.get("status") == "success"]
        skipped = [r for r in results if r.get("status") == "skipped"]
        if bought:
            _OPPORTUNISTIC_LAST_FIRED = _time.time()
            logger.info(
                "🎯 OPPORTUNISTIC TRADE: bought %d position(s) %s",
                len(bought), [r.get("ticker") for r in bought],
            )
            try:
                tickers = ", ".join(r.get("ticker", "?") for r in bought)
                _n._send(
                    f"🎯 <b>OPPORTUNISTIC BUY</b>\n"
                    f"After {reason}, found {len(bought)} eligible: <code>{tickers}</code>"
                )
            except Exception:
                pass
        else:
            # Don't burn the cooldown if we found nothing — the next
            # close may be more relevant. But log so we can see what
            # blocked.
            logger.info(
                "opportunistic_buy: 0 buys (top skip: %s)",
                skipped[0].get("reason", "?") if skipped else "no candidates",
            )
    except Exception as e:
        logger.error("opportunistic_buy raised: %s", e, exc_info=True)


# ── Consecutive-miss counter for sync_positions() ─────────────────
# `client.sync_positions()` sometimes returns a PARTIAL result —
# 2 of 3 tracked positions, missing one of them, even though IB
# actually holds all three (verified on 2026-05-01 with ORCL: a
# /status call ran 2 minutes apart showed ORCL fully tracked AND
# a DRIFT alert claiming the tracker didn't know about it. The
# tracker was correct; sync_positions() had silently dropped ORCL
# from its return on one cycle, so reconcile_drop ran and removed
# it, then drift_check on the very next cycle re-detected it).
#
# Existing guard `if not ibkr_positions: skip` only catches
# zero-result sync. This counter requires N consecutive misses
# before we treat a position as genuinely closed. One transient
# missing cycle is now invisible to the user; two in a row trips
# the close path.
_MISSING_COUNT: dict = {}
_MISS_THRESHOLD = 2  # Need 2 consecutive missing cycles to close


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

        # 1a. Ledger ingest (deep fix for tracker↔IB drift). Idempotently
        # record IB's OWN executions, keyed by execId. This is the event
        # source: a SELL execution here IS the close — no position-diff
        # inference, no reconcile_drop, no fabricated P&L. Returns only the
        # NEW executions so we can react to fresh fills in the close path.
        new_execs: list = []
        if getattr(CONFIG, "ledger_enabled", False):
            try:
                from core.trading import ledger
                new_execs = ledger.ingest(client)
            except Exception as _le:
                logger.warning("ledger ingest failed (non-fatal): %s", _le)

        # Reset miss counter for tickers that ARE in this sync
        # (so a transient miss followed by a hit clears the count).
        for ticker_seen in ibkr_positions:
            _MISSING_COUNT.pop(ticker_seen, None)

        for pos in positions:
            ticker = pos["ticker"]

            # Check if position still exists in IBKR
            # Safety: skip if there's a pending BUY, OR if we have very few
            # positions from IBKR (could mean incomplete sync after reconnect)
            has_pending_buy = any(
                o.get("ticker") == ticker and o.get("action") == "BUY"
                and o.get("status") in ("PreSubmitted", "Submitted")
                for o in ibkr_orders
            )
            # Don't close if IBKR returned 0 positions (likely a sync issue)
            if not ibkr_positions:
                logger.warning("IBKR returned 0 positions — skipping close check (possible sync issue)")
                break
            if ticker not in ibkr_positions and not CONFIG.dry_run and not has_pending_buy:
                # Double-check: verify we have protective orders for OTHER positions
                # If we don't see ANY positions, it's likely a connection issue, not a real close
                other_positions_exist = any(t != ticker for t in ibkr_positions)
                if not other_positions_exist and len(positions) > 1:
                    logger.warning("Only %s missing but no other IB positions seen — skipping (possible sync issue)", ticker)
                    continue

                # Consecutive-miss guard: require N cycles of missing
                # before treating as closed. Defends against partial
                # sync_positions() results where IB returned some
                # tracked positions but not all (the ORCL adopt/drop
                # loop on 2026-05-01).
                _MISSING_COUNT[ticker] = _MISSING_COUNT.get(ticker, 0) + 1
                if _MISSING_COUNT[ticker] < _MISS_THRESHOLD:
                    logger.warning(
                        "%s missing from IB sync (%d/%d cycles) — "
                        "waiting for confirmation before close",
                        ticker, _MISSING_COUNT[ticker], _MISS_THRESHOLD,
                    )
                    continue
                # Confirmed missing for N cycles — proceed with close
                logger.info(
                    "%s confirmed missing for %d cycles — marking closed",
                    ticker, _MISSING_COUNT[ticker],
                )

                # Try to determine exit price + PRECISE reason from filled orders.
                # IB's fills can be sparse right after a close — try twice with
                # a short gap to give the API time to propagate the execution.
                #
                # Reason precision matters for post-mortem: "trail_fired" tells
                # us the trail was too tight (or the move reversed); "target_hit"
                # tells us the thesis worked. Old generic "stop_or_target_filled"
                # blurred these together so we couldn't tune the trail floor
                # against actual data. (Audit 2026-05-05 — 6 closes, all logged
                # as "closed_externally" because fills() ran before trades().)
                exit_price = 0.0
                reason = "closed_externally"
                target_price_for_reason = float(pos.get("target_price", 0) or 0)
                entry_price_for_reason = float(pos.get("entry_price", 0) or 0)

                def _classify_reason(price: float) -> str:
                    """Map an exit price to trail_fired / target_hit / partial."""
                    if price <= 0:
                        return "fill_detected"
                    # Target hit: within 1% of LMT target (IB may fill a few
                    # cents through). Robust to rounding.
                    if (target_price_for_reason > 0
                            and price >= target_price_for_reason * 0.99):
                        return "target_hit"
                    # Below entry → trail fired (or stop)
                    if (entry_price_for_reason > 0
                            and price < entry_price_for_reason):
                        return "trail_fired"
                    # Above entry but well below target → trail fired in profit
                    if entry_price_for_reason > 0:
                        return "trail_fired_in_profit"
                    return "fill_detected"

                # PRIMARY: check trades() — gives us orderType directly so we
                # know whether it was the TRAIL or the LMT that filled.
                def _try_trades():
                    try:
                        for trade in client._ib.trades():
                            if (trade.contract.symbol == ticker
                                    and trade.order.action == "SELL"
                                    and trade.orderStatus.status == "Filled"):
                                fp = float(trade.orderStatus.avgFillPrice or 0)
                                if fp > 0:
                                    ot = trade.order.orderType or ""
                                    if "TRAIL" in ot.upper():
                                        return fp, "trail_fired"
                                    if ot.upper() in ("LMT", "LIMIT"):
                                        return fp, "target_hit"
                                    return fp, _classify_reason(fp)
                    except Exception as _e:
                        logger.debug("Trades check failed: %s", _e)
                    return 0.0, ""

                # SECONDARY: fills() gives price but not orderType — classify
                # by price-vs-target/entry distance.
                def _try_fills():
                    try:
                        for f in client._ib.fills():
                            if (f.contract.symbol == ticker
                                    and f.execution.side in ("SLD", "SELL")):
                                fp = float(f.execution.price or 0)
                                if fp > 0:
                                    return fp, _classify_reason(fp)
                    except Exception as _e:
                        logger.debug("Fills check failed: %s", _e)
                    return 0.0, ""

                exit_price, reason_detected = _try_trades()
                if exit_price == 0.0:
                    exit_price, reason_detected = _try_fills()
                if exit_price == 0.0:
                    # Wait briefly and try once more — fills may still be landing
                    try:
                        client._ib.sleep(2)
                    except Exception:
                        pass
                    exit_price, reason_detected = _try_trades()
                    if exit_price == 0.0:
                        exit_price, reason_detected = _try_fills()
                if reason_detected:
                    reason = reason_detected

                # Fallback: check OCA orders for order_type
                if exit_price == 0.0:
                    oca = pos.get("order_ids", {}).get("oca_group", "")
                    if oca:
                        for order in ibkr_orders:
                            if order.get("oca_group") == oca and order.get("filled", 0) > 0:
                                ot = (order.get("order_type") or "").upper()
                                if "TRAIL" in ot:
                                    reason = "trail_fired"
                                elif ot in ("LMT", "LIMIT"):
                                    reason = "target_hit"
                                else:
                                    reason = f"{order.get('order_type', 'unknown')}_filled"
                                break

                # Last resort: use executions() which has recent fills
                if exit_price == 0.0:
                    try:
                        for e in client._ib.executions():
                            if (e.contract.symbol == ticker
                                    and e.execution.side in ("SLD", "SELL")):
                                exit_price = float(e.execution.price or 0)
                                if exit_price > 0:
                                    reason = _classify_reason(exit_price)
                                    break
                    except Exception:
                        pass

                # ── LEDGER MODE (deep fix) ───────────────────────────
                # IB is the source of truth. The SELL execution (if real)
                # is already in the ledger as broker-truth P&L, recorded by
                # the ingest at the top of this cycle. We do NOT infer a
                # price or fabricate P&L: we drop the stale metadata row and
                # react to any fresh SELL execution. This removes the entire
                # inference/reconcile_drop machinery that was the drift
                # engine (KNX 2026-04-28, PAAS 2026-06-03 — losses that
                # vanished with pnl=None).
                if getattr(CONFIG, "ledger_enabled", False):
                    matched = [
                        e for e in new_execs
                        if e.get("ticker") == ticker and e.get("side") == "SELL"
                    ]
                    tracker.drop_metadata(ticker)
                    if matched:
                        rp = sum(float(e.get("realized_pnl") or 0)
                                 for e in matched
                                 if e.get("realized_pnl") is not None)
                        xp = float(matched[-1].get("price") or exit_price or 0.0)
                        notify.notify_sell(ticker, pos["quantity"], xp, reason, rp)
                        _try_opportunistic_buy(client, tracker, notify,
                                               reason=f"after_close_{ticker}")
                    else:
                        # Gone from IB but no SELL execution surfaced this
                        # cycle — the fill was ingested in a prior cycle
                        # (P&L already in the ledger) or it's a manual move.
                        # No fabricated number; the account-truth
                        # reconciliation (/pnl) is the backstop.
                        logger.info(
                            "%s gone from IB, no new SELL exec this cycle — "
                            "metadata dropped (P&L, if any, is in ledger)",
                            ticker,
                        )
                        _try_opportunistic_buy(client, tracker, notify,
                                               reason=f"after_close_{ticker}")

                # ── LEGACY MODE (ledger disabled) ────────────────────
                # If we found a real exit price → record a proper CLOSE
                # with accurate P&L. If we DIDN'T → use reconcile_drop
                # which removes the position from the tracker WITHOUT
                # writing a CLOSE row with a fake P&L.
                elif exit_price > 0:
                    tracker.remove_position(ticker, exit_price, reason)
                    pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    notify.notify_sell(ticker, pos["quantity"], exit_price, reason, pnl)
                    # OPPORTUNISTIC TRADE TRIGGER (audit followup 2026-05-05).
                    # A position just closed → cash freed → check if there's
                    # something good in the latest scan. Fires only during
                    # market hours, with cooldown, and uses the same
                    # risk_manager + policy gates as the scheduled pipeline.
                    _try_opportunistic_buy(client, tracker, notify,
                                           reason=f"after_close_{ticker}")
                else:
                    # Genuinely couldn't find an exit price. Drop it from
                    # the tracker silently for accounting; alert the user
                    # so they can verify in IB if it was a real close.
                    tracker.reconcile_drop(ticker, reason=f"{reason}_no_fill_data")
                    try:
                        notify.notify_error(
                            "Tracker drift — position dropped",
                            f"⚠️ {ticker} not found in IB and no fill data "
                            f"available. Tracker entry dropped (no P&L "
                            f"recorded). Verify in IB whether this was a "
                            f"real close, an unfilled order, or a sync glitch."
                        )
                    except Exception:
                        pass

        # 1b. Drift check — surface tracker↔IB anomalies before they bite.
        _drift_check(tracker, client, ibkr_orders, notify)

        # 2. Verify protective orders — every position MUST have live orders
        _verify_protections(tracker, client, ibkr_orders, notify)

        # 2a2. TARGET HIT auto-sell — replaces the LMT order for sub_2k
        # cash accounts where IB rejects OCA LMT (Error 201: margin
        # insufficient because OCA double-counts margin). Without this,
        # a position that touches its target_price gets NO automatic
        # exit — TRAIL alone gives back 4% before firing. Live impact
        # 2026-05-15 to 5-20: ILMN + LYB peaked above target_price area
        # but trail captured the descent, leaving meaningful profit on
        # the table. This pass catches target-touch even WITHOUT the
        # LMT order being placed on IB.
        _target_hit_pass(tracker, client, ibkr_orders, notify)

        # 2b. Ratchet stops up as positions run up (lock in profits)
        if CONFIG.ratchet_enabled:
            _ratchet_stops(tracker, client, ibkr_orders, notify)

        # 2b2. Partial profit-taking (sell half when intermediate target hit)
        if CONFIG.partial_profit_enabled:
            _take_partial_profit(tracker, client, notify)

        # 2b3. Earnings exit — close positions BEFORE earnings to avoid the
        # binary-risk gap. The entry gate (risk_manager.check_earnings_window)
        # blocks NEW buys near earnings, but doesn't help positions opened
        # 2 weeks ago that are now within the danger zone. This pass
        # tightens the trail to 2% on positions with earnings <2 trading
        # days away — letting profitable ones lock current gains and
        # forcing losing ones to exit before the binary event.
        if CONFIG.earnings_gate_enabled:
            _earnings_exit_pass(tracker, client, ibkr_orders, notify)

        # 2c. Push portfolio snapshot to Supabase (for Streamlit UI)
        try:
            from core.trading.portfolio_snapshot import write_snapshot
            write_snapshot(client, tracker)
        except Exception as e:
            logger.debug("Snapshot push failed: %s", e)

        # 3. Check target date exits — but only execute in the last 30 min
        # before close (19:30-20:00 UTC = 15:30-16:00 ET). Earlier in the
        # day, let the TRAIL and LIMIT orders work — they might catch a
        # late-day rally we'd miss with an early-morning forced exit.
        # The MOST common reason a position has target_date == today is
        # earnings the next day (earnings-aware target_date capping
        # added 2026-05-05 sets target_date = earnings_date - 1). For
        # those, we MUST exit before close — but we want maximum time
        # for the position to hit target_price first.
        expired = tracker.check_target_date_exits()
        now_utc = datetime.utcnow()
        # Close window: 19:30-20:00 UTC (last 30 min of regular session)
        in_close_window = (
            now_utc.hour == 19 and now_utc.minute >= 30
        ) or now_utc.hour >= 20
        if expired and not in_close_window:
            logger.info(
                "Target-date exit pending for %s but NOT in close window yet "
                "(now %02d:%02d UTC, window 19:30-20:00) — waiting for trail/target to work",
                expired, now_utc.hour, now_utc.minute,
            )
            expired = []  # skip this cycle, will retry in 5 min

        for ticker in expired:
            pos = tracker.get_position(ticker)
            if not pos:
                continue
            if ticker in ibkr_positions:
                logger.info(
                    "Target date EXITING %s at MKT (in last-30-min close window)",
                    ticker,
                )
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


def _earnings_exit_pass(tracker, client, ibkr_orders, notify):
    """Graduated trail tightening for positions approaching earnings.

    The entry gate refuses new buys near earnings, but doesn't help
    positions already held when their earnings date approaches. This
    function progressively tightens the TRAIL as earnings nears:

      T-4..-5 days: trail ≤ 3.0%   (warn band — defensive)
      T-2..-3 days: trail ≤ 2.5%   (mid band)
      T-0..-1 days: trail ≤ 2.0%   (final band — near-exit posture)

    Configurable via EARNINGS_PROXIMITY_* env vars. The "only tighten"
    rule still applies: ratchet-tightened positions (e.g., 2.0% at +28%
    peak) are preserved if already tighter than the earnings band.

    Why not auto-close: a sudden close could trigger day-trade rules
    on a same-day-opened position, and forcing market-sell on a
    profitable position throws away expected value when the user might
    legitimately want to hold through earnings. Tightening the trail
    is a less-invasive intermediate step — the user gets a warning AND
    the protective order locks current gains/limits future losses.
    """
    from core.trading.risk_manager import RiskManager
    from core.trading.config import CONFIG  # 2026-05-17: missing import caused
        # NameError on every cycle since the graduated-earnings change shipped
        # on 2026-05-15 19:00 UTC. Monitor stayed alive (systemd) but crashed
        # on this function, blocking everything downstream — snapshot writes,
        # target_date exits, etc. Stop_loss / take_profit protected by OCA
        # orders on IB servers (independent of monitor), so no positions
        # were at risk, but the in-process tightening features were dead.
    from datetime import date as _date

    positions = tracker.get_open_positions()
    if not positions:
        return

    # Read graduated config (with safe defaults for legacy systems)
    proximity_enabled = bool(getattr(CONFIG, "earnings_proximity_enabled", True))
    if not proximity_enabled:
        return
    warn_days = int(getattr(CONFIG, "earnings_proximity_warn_days", 5))
    warn_trail = float(getattr(CONFIG, "earnings_proximity_warn_trail_pct", 3.0))
    mid_days = int(getattr(CONFIG, "earnings_proximity_mid_days", 3))
    mid_trail = float(getattr(CONFIG, "earnings_proximity_mid_trail_pct", 2.5))
    final_days = int(getattr(CONFIG, "earnings_proximity_final_days", 1))
    final_trail = float(getattr(CONFIG, "earnings_proximity_final_trail_pct", 2.0))

    # Reuse RiskManager's earnings cache (cheap)
    rm = RiskManager(client, tracker)
    today = _date.today()

    orders_by_ticker = {}
    for o in ibkr_orders:
        t = o.get("ticker", "")
        if t and o.get("status") in ("Submitted", "PreSubmitted"):
            orders_by_ticker.setdefault(t, []).append(o)

    for pos in positions:
        ticker = pos["ticker"]
        try:
            cached = RiskManager._EARNINGS_CACHE.get(ticker)
            if cached is None:
                cached = rm._fetch_earnings_date(ticker)
                RiskManager._EARNINGS_CACHE[ticker] = cached
            if cached == "none":
                continue
            ed = _date.fromisoformat(cached)
            days_until = (ed - today).days
        except Exception:
            continue

        # Determine target trail % based on graduated bands (tightest first)
        if days_until < 0 or days_until > warn_days:
            continue
        if days_until <= final_days:
            target_pct = final_trail
            band = "FINAL"
        elif days_until <= mid_days:
            target_pct = mid_trail
            band = "MID"
        else:
            target_pct = warn_trail
            band = "WARN"

        # Find the active TRAIL order for this position
        trail_order = None
        for o in orders_by_ticker.get(ticker, []):
            if o.get("order_type") == "TRAIL":
                trail_order = o
                break
        if not trail_order:
            continue

        current_pct = pos.get("trailing_stop_pct", 0) or 0
        if current_pct > 0 and current_pct <= target_pct + 0.05:
            continue  # already at or below target — preserve ratchet's tighter trail

        # Track whether we've already alerted at this band (avoid spam)
        already_at_band = pos.get("earnings_band") == band
        if already_at_band:
            continue

        result = client.modify_trailing_pct(trail_order["order_id"], target_pct)
        if result.status in ("Submitted", "PreSubmitted", "PendingSubmit", "DRY_RUN"):
            pos["trailing_stop_pct"] = target_pct
            pos["earnings_tightened"] = True
            pos["earnings_band"] = band
            try:
                # Emoji escalates with band severity
                emoji = {"WARN": "📅", "MID": "⚠️", "FINAL": "🚨"}.get(band, "⚠️")
                notify._send(
                    f"{emoji} <b>EARNINGS-{band} {ticker}</b>\n"
                    f"  Earnings in {days_until}d ({ed.isoformat()})\n"
                    f"  Trail: {current_pct:.1f}% → <b>{target_pct:.1f}%</b>\n"
                    f"  {'Consider closing manually if you do not want to hold through earnings.' if band == 'FINAL' else 'Defensive tightening — binary risk window approaching.'}"
                )
            except Exception:
                pass
        else:
            logger.warning(
                "Earnings tighten %s (band=%s) FAILED: %s",
                ticker, band, getattr(result, "error", "")
            )


def _try_adopt_ib_only(sym, qty, ibkr_orders, ib_pos_obj, tracker, notify):
    """Adopt an IB-only position into the tracker if it has a complete
    SS_* OCA group (TRAIL + LMT) active. Returns True iff adopted.

    Safety filters:
      - Long only (qty > 0; caller already filtered).
      - OCA group must start with 'SS_' (our system's signature, see
        ibkr_client._oca_group_name) — prevents adopting random manual
        shorts/swing trades the user placed outside the system.
      - Both TRAIL and LMT must be active in the same OCA group.
      - avg_cost must be available from IB position object.

    On success, the next cycle's exit logic (target_date, ratchet, etc.)
    will treat the adopted position normally; the trade_log entry is
    RECONCILE_ADOPT (not OPEN) so analytics filter it out of P&L stats.
    """
    # Group live SELL orders for this ticker by OCA
    sell_orders = [
        o for o in ibkr_orders
        if o.get("ticker") == sym
        and o.get("action") == "SELL"
        and o.get("status") in ("Submitted", "PreSubmitted")
    ]
    by_oca: dict = {}
    for o in sell_orders:
        g = o.get("oca_group", "")
        if g and g.startswith("SS_"):
            by_oca.setdefault(g, []).append(o)

    chosen_oca = None
    trail_order = None
    limit_order = None
    for g, orders in by_oca.items():
        t_ord = next((o for o in orders if o.get("order_type") == "TRAIL"), None)
        l_ord = next((o for o in orders if o.get("order_type") == "LMT"), None)
        if t_ord and l_ord:
            chosen_oca = g
            trail_order = t_ord
            limit_order = l_ord
            break

    if not chosen_oca:
        logger.info(
            "adopt %s: no complete SS_* OCA group with TRAIL+LMT — leaving as drift",
            sym,
        )
        return False

    if ib_pos_obj is None:
        logger.warning("adopt %s: no IB position object — cannot derive entry price", sym)
        return False

    raw_avg_cost = getattr(ib_pos_obj, "averageCost", None)
    if raw_avg_cost is None:
        raw_avg_cost = getattr(ib_pos_obj, "avgCost", None)
    try:
        entry_price = float(raw_avg_cost or 0)
    except (TypeError, ValueError):
        entry_price = 0.0
    if entry_price <= 0:
        logger.warning("adopt %s: avgCost is %r — cannot adopt safely", sym, raw_avg_cost)
        return False

    try:
        trail_pct = float(trail_order.get("trailing_percent") or 0)
    except (TypeError, ValueError):
        trail_pct = 0.0
    try:
        target_price = float(limit_order.get("lmt_price") or 0)
    except (TypeError, ValueError):
        target_price = 0.0

    if trail_pct <= 0 or target_price <= 0:
        logger.warning(
            "adopt %s: missing trail_pct (%.2f) or target (%.2f) — skipping",
            sym, trail_pct, target_price,
        )
        return False

    # Conservative initial stop = entry × (1 - trail_pct/100). Real
    # trailing stop on IB will move with price; this just gives the
    # tracker a non-zero baseline so monitor's exit math doesn't divide
    # by zero on `(price - stop) / price`.
    stop_loss = round(entry_price * (1 - trail_pct / 100.0), 2)

    order_ids = {
        "trailing_stop": trail_order.get("order_id", 0),
        "limit_sell": limit_order.get("order_id", 0),
    }

    try:
        tracker.reconcile_adopt(
            ticker=sym,
            quantity=int(qty),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            trailing_stop_pct=trail_pct,
            oca_group=chosen_oca,
            order_ids=order_ids,
            reason="ib_only_drift",
        )
    except Exception as e:
        logger.error("adopt %s: tracker.reconcile_adopt failed: %s", sym, e)
        return False

    try:
        notify._send(
            f"✅ <b>RECONCILE: {sym} adopted</b>\n"
            f"IB-only position pulled into tracker.\n"
            f"Qty: {int(qty)} @ ${entry_price:.2f}\n"
            f"Trail: {trail_pct:.1f}% | Target: ${target_price:.2f}\n"
            f"OCA: <code>{chosen_oca}</code>\n"
            f"Monitor's exit logic will now manage this position."
        )
    except Exception:
        pass

    return True


def _drift_check(tracker, client, ibkr_orders, notify):
    """Surface tracker↔IB drift before it becomes a silent bug.

    Three checks, single consolidated alert per cycle (cooldown 30 min):

    1. Untracked IB position — IB holds something the tracker doesn't know
       about. Means a manual buy or a buy that the tracker missed
       persisting. Without this alert, the position would have NO
       protective orders managed by us.

    2. Protective-order count mismatch — every tracked position should
       have exactly 2 active sell orders (TRAIL + LMT, same OCA). If
       count != 2, something cancelled or duplicated. (We check ≥1 of
       each type; OCA cleanup handles the rest.)

    3. Tracker entry with no fill record — an OPEN that's missing from
       IB AND from completed-orders. This was the KNX phantom on
       2026-04-28: the buy was placed but rejected by IB rules, yet the
       tracker recorded an OPEN. Caught by reconciliation later, but the
       fake P&L estimate misleads the daily summary.
    """
    issues = []

    # ── Pre-check: skip drift entirely if the IB connection isn't
    # actually live. Two scenarios that previously caused noisy
    # "DRIFT CHECK SKIPPED" Telegram spam every cycle:
    #
    #   1. dry_run=True. `connect()` returns True without setting
    #      `_ib` (the dry-run path skips the real IB() instantiation).
    #      Any `_ib.portfolio()` then raises 'NoneType' has no
    #      attribute 'portfolio'. There is no real portfolio to compare
    #      against in dry-run anyway — the alert is meaningless.
    #
    #   2. Connection dropped between cycles. `_ib` exists but
    #      `_ib.isConnected()` is False (Gateway restarted, 2FA
    #      expired, network blip). Calling `.portfolio()` on a
    #      disconnected IB() returns empty/raises depending on
    #      ib_insync version — neither produces useful drift output.
    #
    # Better behavior: log at DEBUG and return silently. The next
    # successful cycle will catch any genuine drift; one missed cycle
    # is invisible to the user and far better than a constant alarm.
    if getattr(client, "cfg", None) and getattr(client.cfg, "dry_run", False):
        logger.debug("drift_check skipped: client is in DRY_RUN mode")
        return
    if client._ib is None or not client._ib.isConnected():
        logger.warning(
            "drift_check skipped: IB connection not live "
            "(_ib=%s, connected=%s)",
            "None" if client._ib is None else "set",
            False if client._ib is None else client._ib.isConnected(),
        )
        return

    # IB positions
    try:
        ib_pos_by_ticker = {p.contract.symbol: float(p.position)
                            for p in client._ib.portfolio()
                            if p.position != 0}
    except Exception as _e:
        # Surface portfolio-fetch failures (with cooldown) so silent
        # degradation of the drift check doesn't go unnoticed. Previously
        # this returned silently, making drift detection a no-op for
        # entire cycles. (Audit finding #6 — monitor.)
        if _cooldown_ok(("drift_check", "portfolio_fail"),
                        _ALERT_COOLDOWN, 1800):
            try:
                notify._send(
                    f"⚠️ <b>DRIFT CHECK SKIPPED</b>\n"
                    f"Portfolio fetch failed: {_e}\n"
                    f"Drift detection is currently a no-op until next cycle."
                )
            except Exception:
                pass
        return

    tracked = {p["ticker"]: p for p in tracker.get_open_positions()}

    # Check 1: untracked IB positions.
    # Race-window suppression: if a buy filled in IB but the tracker write
    # hasn't landed yet, drift_check would alert spuriously and the user
    # would learn to ignore it. Suppress if the position appears < 5 min
    # old (recent BUY fill in executions). (Audit finding #6 — monitor.)
    recent_buy_tickers = set()
    try:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        cutoff = _dt.now(_tz.utc) - _td(minutes=5)
        for ex in client._ib.executions():
            try:
                if ex.execution.side in ("BOT", "BUY") and ex.time >= cutoff:
                    recent_buy_tickers.add(ex.contract.symbol)
            except Exception:
                continue
    except Exception:
        pass

    # Build a per-ticker IB-position lookup so we can pull avg_cost during
    # adoption (entry_price is required to compute realistic stop_loss).
    ib_position_objs = {}
    try:
        for p in client._ib.portfolio():
            if p.position != 0:
                ib_position_objs[p.contract.symbol] = p
    except Exception:
        pass

    for sym, qty in ib_pos_by_ticker.items():
        if sym not in tracked:
            if sym in recent_buy_tickers:
                logger.debug(
                    "drift_check: skipping %s (BUY filled <5 min ago, "
                    "tracker write may still be propagating)", sym,
                )
                continue
            # Try to auto-adopt before warning. Symmetric to reconcile_drop:
            # if the position has a complete SS_* OCA group (TRAIL + LMT)
            # active, it's almost certainly one of ours that lost its
            # tracker row (manual buy via IB, crashed tracker write, or
            # restored backup). Adopting is safer than letting it fire
            # DRIFT every cooldown forever and stay invisible to monitor's
            # exit logic.
            if qty > 0:
                adopted = _try_adopt_ib_only(
                    sym, qty, ibkr_orders, ib_position_objs.get(sym),
                    tracker, notify,
                )
                if adopted:
                    continue
            issues.append(f"IB holds {sym} ({int(qty)} shares) but tracker doesn't")

    # Check 2: protective-order count per tracked position
    orders_by_ticker = {}
    for o in ibkr_orders:
        t = o.get("ticker", "")
        if t and o.get("status") in ("Submitted", "PreSubmitted"):
            orders_by_ticker.setdefault(t, []).append(o)

    # 2026-05-22: detect cash<$2k tier so we don't flag missing LMT as drift.
    # IB rejects LMT-in-OCA with Error 201 when AvailableFunds is too low;
    # _target_hit_pass in the monitor is the software replacement. Flagging
    # the missing LMT every 30min as DRIFT is noise that obscures real drift.
    try:
        _net_liq = float(client.get_net_liquidation() or 0)
    except Exception:
        _net_liq = 0.0
    _cash_under_2k = (0 < _net_liq < 2000)

    # Grace window for freshly-opened positions. The ibkr_orders snapshot
    # used here is fetched earlier in the cycle — BEFORE an opportunistic
    # buy places the new position's OCA bracket. So a position bought this
    # cycle (e.g. PLTR right after SOLS closed on 2026-05-29) appears to
    # have "NO active stop" even though its TRAIL+LMT landed seconds later.
    # Skip drift checks for positions younger than this; it self-resolves
    # next cycle once the orders are in the snapshot. Env: TRADE_DRIFT_FRESH_GRACE_SEC.
    from datetime import datetime as _dt_drift
    from core.trading.config import CONFIG as _CFG_drift
    _fresh_grace = float(getattr(_CFG_drift, "drift_fresh_grace_sec", 300))

    for sym, pos in tracked.items():
        # Skip if position not in IB yet (just opened, may be pending)
        if sym not in ib_pos_by_ticker:
            continue
        # Skip freshly-opened positions — their protective orders may not be
        # in this cycle's order snapshot yet (race with opportunistic buy).
        try:
            _o = str(pos.get("opened_at", "") or "")[:19]
            _age_s = (_dt_drift.utcnow() - _dt_drift.fromisoformat(_o)).total_seconds() if _o else 1e9
        except Exception:
            _age_s = 1e9
        if _age_s < _fresh_grace:
            logger.debug(
                "drift_check: skipping %s — opened %.0fs ago (<%.0fs grace, "
                "protective orders may still be landing)", sym, _age_s, _fresh_grace,
            )
            continue
        orders = orders_by_ticker.get(sym, [])
        order_types = {o.get("order_type") for o in orders}
        has_stop = bool(order_types & {"TRAIL", "STP"})
        has_target = "LMT" in order_types
        if not has_stop:
            issues.append(f"{sym}: NO active stop order (TRAIL/STP)")
        if not has_target and not _cash_under_2k:
            # Above $2k: missing LMT is real drift
            issues.append(f"{sym}: NO active target order (LMT)")
        # cash<$2k: missing LMT is EXPECTED (IB Error 201) — software
        # _target_hit_pass handles target exits; no drift to flag.
        # Detect zombie duplicates: more than one stop OR target in active set
        stop_count = sum(1 for o in orders if o.get("order_type") in ("TRAIL", "STP"))
        target_count = sum(1 for o in orders if o.get("order_type") == "LMT")
        if stop_count > 1:
            issues.append(f"{sym}: {stop_count} stop orders active (expected 1)")
        if target_count > 1:
            issues.append(f"{sym}: {target_count} target orders active (expected 1)")

    if issues and _cooldown_ok(("drift_check", "all"),
                                _ALERT_COOLDOWN, 1800):
        msg = "⚠️ <b>DRIFT DETECTED</b>\n" + "\n".join(f"  • {i}" for i in issues)
        notify._send(msg)
        for i in issues:
            logger.warning("DRIFT: %s", i)


def _verify_protections(tracker, client, ibkr_orders, notify):
    """Ensure every open position has live trailing stop + limit sell.

    If orders are missing (cancelled, expired, margin issue), resubmit them
    and alert via Telegram.

    2026-05-22 cash<$2k handling: IB rejects LMT in OCA brackets with
    Error 201 for accounts below $2000. The TRAIL still gets placed
    successfully. Without this gating, the missing LMT would trigger
    repeated CRITICAL alerts every cycle even though the position IS
    protected by TRAIL (and the monitor's _target_hit_pass provides
    software target-exit). Now: if account is cash<$2k, treat TRAIL-only
    as fully protected.
    """
    positions = tracker.get_open_positions()
    if not positions:
        return

    # Detect cash<$2k tier — affects "fully protected" definition.
    try:
        _net_liq = float(client.get_net_liquidation() or 0)
    except Exception:
        _net_liq = 0.0
    cash_under_2k = (0 < _net_liq < 2000)
    if cash_under_2k:
        logger.debug(
            "_verify_protections: cash<$2k tier (net=$%.0f) — TRAIL-only is acceptable; "
            "LMT replaced by software _target_hit_pass",
            _net_liq,
        )

    # Build set of active OCA groups from live orders
    active_ocas = set()
    for o in ibkr_orders:
        oca = o.get("oca_group", "")
        if oca and o.get("status") in ("Submitted", "PreSubmitted"):
            active_ocas.add(oca)

    # Build set of tickers that actually exist in IBKR (filled, not pending)
    ibkr_held = set()
    try:
        for p in client.get_positions():
            if p.quantity > 0:
                ibkr_held.add(p.ticker)
    except Exception:
        pass

    for pos in positions:
        ticker = pos["ticker"]
        oca = pos.get("order_ids", {}).get("oca_group", "")

        # Skip if position not yet in IBKR (buy not filled yet)
        if ticker not in ibkr_held:
            logger.info("⏳ %s not yet in IBKR (buy pending) — skipping protection check", ticker)
            continue

        # Check if this position's OCA group has live orders
        if oca and oca in active_ocas:
            # Count + classify live orders in this OCA group
            oca_orders = [
                o for o in ibkr_orders
                if o.get("oca_group") == oca
                and o.get("status") in ("Submitted", "PreSubmitted")
            ]
            live_count = len(oca_orders)
            has_trail = any(o.get("order_type") in ("TRAIL", "STP") for o in oca_orders)
            has_limit = any(o.get("order_type") == "LMT" for o in oca_orders)

            # Healthy definition:
            #   - account >= $2k: both stop-side AND limit-side present (live_count >= 2)
            #   - account <  $2k: TRAIL is sufficient (LMT is impossible for this tier;
            #     target-side is handled by monitor's _target_hit_pass)
            healthy = (
                (live_count >= 2)
                or (cash_under_2k and has_trail)
            )
            if healthy:
                if cash_under_2k and not has_limit:
                    logger.info(
                        "✓ %s protected (TRAIL active in OCA %s; LMT not required at cash<$2k tier)",
                        ticker, oca,
                    )
                else:
                    logger.info("✓ %s protected (%d orders in OCA %s)", ticker, live_count, oca)
                # Clear miss counter — protections confirmed healthy.
                _PROTECTION_MISS_COUNT.pop(ticker, None)
                continue
            else:
                logger.warning(
                    "⚠ %s OCA %s has %d order(s) — stop=%s limit=%s — resubmitting",
                    ticker, oca, live_count, has_trail, has_limit,
                )
        else:
            # FALLBACK: tracker's stored OCA may be stale (e.g. after a manual
            # resubmit or reconnect). Look for ANY live OCA group on this
            # ticker that has both a TRAIL and a LMT SELL order — adopt it.
            live_for_ticker = [
                o for o in ibkr_orders
                if o.get("ticker") == ticker
                and o.get("action") == "SELL"
                and o.get("status") in ("Submitted", "PreSubmitted")
            ]
            # Group by OCA
            by_oca: dict = {}
            for o in live_for_ticker:
                g = o.get("oca_group", "")
                if g:
                    by_oca.setdefault(g, []).append(o)
            # Find a complete group:
            #   - cash >= $2k: need TRAIL/STP AND LMT
            #   - cash <  $2k: TRAIL/STP alone counts (LMT impossible at this tier)
            adopted_oca = None
            for g, orders in by_oca.items():
                types = {o.get("order_type") for o in orders}
                has_stop = bool(types & {"TRAIL", "STP"})
                has_limit = "LMT" in types
                if has_stop and (has_limit or cash_under_2k):
                    adopted_oca = g
                    break
            if adopted_oca:
                # Adopt the live OCA into tracker and skip resubmit
                logger.info(
                    "✓ %s protected — adopting live OCA %s (tracker had stale %r)",
                    ticker, adopted_oca, oca,
                )
                all_pos = tracker.get_open_positions()
                for p in all_pos:
                    if p["ticker"] == ticker:
                        oids = p.get("order_ids", {}) or {}
                        oids["oca_group"] = adopted_oca
                        p["order_ids"] = oids
                        break
                tracker._save_positions(all_pos)
                continue
            logger.warning("⚠ %s has NO protective orders — resubmitting", ticker)

        # 2026-05-26: in cash<$2k tier, the resubmit path below is POISON.
        # It does:
        #   1. cancelOrder(t.order) on each order in the (presumed-dead)
        #      OCA group — these often fail with Error 10147 "not found"
        #      because the order isn't actually dead, it's mid-race.
        #   2. resubmit_protective_orders_retry() places fresh TRAIL+LMT —
        #      both rejected with Error 201 ("MINIMUM 2000 USD REQUIRED"
        #      or "Equity insufficient") because IB treats fresh SELLs in
        #      this tier as potential shorts requiring margin.
        # Result: brief no-protection window + Telegram spam + zero recovery.
        # Real-world failure observed 2026-05-22 13:30-15:54 UTC across
        # ORCL and TGTX. Both positions ended up still protected by the
        # original TRAIL (which recovered on its own from the race), but
        # the monitor wasted ~5 attempts and confused itself.
        #
        # New behavior in cash<$2k: bail out gracefully. The position is
        # either (a) still protected by an original TRAIL that's just in
        # a momentary weird state, OR (b) genuinely unprotected — in
        # which case the user needs to intervene (send /resubmit TICKER
        # manually after market settles, or transfer funds to escape
        # the tier). Either way, doing nothing is strictly better than
        # cancel+replace here.
        if cash_under_2k:
            _PROTECTION_MISS_COUNT[ticker] = _PROTECTION_MISS_COUNT.get(ticker, 0) + 1
            miss = _PROTECTION_MISS_COUNT[ticker]
            logger.warning(
                "⚠ %s appears unprotected (miss #%d) but cash<$2k tier "
                "blocks cancel+replace (Error 201). Skipping resubmit. "
                "Will retry next cycle.",
                ticker, miss,
            )
            # Persistent miss = real problem. Page the user after 3 cycles
            # (~15 min during pre-market poll, or 15 min during market).
            if miss == 3:
                try:
                    notify.notify_error(
                        "Monitor",
                        f"⚠ {ticker} unprotected for {miss} cycles. "
                        f"Cash<$2k tier blocks auto-resubmit (IB Error 201). "
                        f"Send <code>/resubmit {ticker}</code> manually, or "
                        f"transfer funds to escape the tier.",
                    )
                except Exception:
                    pass
            continue

        # Resubmit cooldown: avoid hammering IB with retries that will fail
        # for the same reason (account restriction). Prevents Telegram spam.
        if not _cooldown_ok(ticker, _RESUBMIT_COOLDOWN, _RESUBMIT_COOLDOWN_SECONDS):
            logger.info(
                "Resubmit cooldown active for %s — skipping this cycle",
                ticker,
            )
            continue

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

        # Day-trade guard: skip same-day sell if position opened today
        _same_day = False
        try:
            from datetime import date as _d_sd
            _opened_sd = str(pos.get("opened_at", ""))[:10]
            if _opened_sd == _d_sd.today().isoformat():
                _same_day = True
                logger.info("Resubmit %s with same_day_guard (opened today)", ticker)
        except Exception:
            pass

        result = client.resubmit_protective_orders_retry(
            ticker=ticker,
            qty=int(qty),
            trail_pct=trail_pct,
            target_price=target_price,
            max_attempts=3,
            same_day_guard=_same_day,
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
            # Use the dedicated resubmit notification — it shows the ACTUAL
            # projected TRAIL stop based on peak_price, not the stale
            # scan-derived stop_loss (which was misleading for positions
            # that have already run up).
            notify.notify_resubmit(
                ticker=ticker,
                qty=int(qty),
                entry=pos["entry_price"],
                trail_pct=trail_pct,
                target=target_price,
                peak_price=pos.get("peak_price", 0),
                score=pos.get("score", 0),
                target_date=pos.get("target_date", ""),
            )
        else:
            err_trail = getattr(result["trailing_stop"], "error", "") or ""
            err_limit = getattr(result["limit_sell"], "error", "") or ""
            account_block = _is_account_restriction_error(err_trail + " " + err_limit)

            logger.error(
                "✗ Failed to resubmit protections for %s: trail=%s, limit=%s (account_rule=%s)",
                ticker,
                result["trailing_stop"].status,
                result["limit_sell"].status,
                account_block,
            )

            # Only alert once per hour for the same ticker — prevents spam
            # Transient-miss buffer: only alert if the same ticker hit
            # this code path on TWO consecutive cycles. First miss is
            # logged but silent — auto-resubmit usually fixes it within
            # the same cycle. Only persistent failure (account rule,
            # IBKR backend hiccup) deserves a Telegram CRITICAL.
            _PROTECTION_MISS_COUNT[ticker] = _PROTECTION_MISS_COUNT.get(ticker, 0) + 1
            miss_n = _PROTECTION_MISS_COUNT[ticker]
            if miss_n < _PROTECTION_MISS_THRESHOLD:
                logger.warning(
                    "%s: protective orders missing (miss %d/%d) — "
                    "deferring alert, will retry next cycle",
                    ticker, miss_n, _PROTECTION_MISS_THRESHOLD,
                )
            elif _cooldown_ok(("protection", ticker), _ALERT_COOLDOWN, _ALERT_COOLDOWN_SECONDS):
                extra = ""
                if account_block:
                    extra = (
                        "\n\n⚠️ IBKR rejected the order: cash account < $2000 "
                        "blocks new protective orders. Existing orders (if any) "
                        "may still be active — check the Portfolio Status."
                    )
                notify.notify_error("Protection",
                    f"CRITICAL: {ticker} has NO protective orders ({miss_n} cycles)! "
                    f"Trail: {result['trailing_stop'].status}, "
                    f"Limit: {result['limit_sell'].status}{extra}"
                )


def _take_partial_profit(tracker, client, notify):
    """Sell half of a position when it crosses partial_profit_trigger_pct.

    Prevents giving back gains on stocks that spike then pull back.
    Flag stored in position.partial_taken to avoid double-execution.
    Also dispatches the profit-LADDER (tiered fractional sells) for
    eligible tiers — same place because both are partial sells with
    the same day-trade and account-tier safety constraints.
    """
    from core.trading.config import CONFIG

    positions = tracker.get_open_positions()
    if not positions:
        return

    # ACCOUNT-TIER-AWARE PRE-MARKET GUARD (added 2026-05-05).
    # Sub-$2k cash accounts get rejected by IB Error 201 on partial-sell
    # MarketOrders submitted pre-market or after-hours: "MINIMUM OF $2000
    # REQUIRED IN ORDER TO PURCHASE ON MARGIN, SELL SHORT, TRADE CURRENCY
    # OR FUTURE." Selling a long shouldn't need margin, but IB's pre-market
    # routing flags it that way for sub-2k accounts.
    #
    # Above-$2k accounts (cash or margin_pdt) DON'T have this restriction
    # and CAN partial-sell pre-market normally — so this guard auto-detects
    # tier and only blocks the sub-2k case. The system gracefully upgrades
    # itself when the account crosses $2k without code changes.
    #
    # Ratchet (TRAIL modify) is unaffected at any tier — IB allows order
    # MODIFICATION outside RTH; only NEW MarketOrders trip Error 201.
    if not client.is_market_open():
        try:
            net_liq = float(client.get_net_liquidation() or 0)
        except Exception:
            net_liq = 0.0  # fail-closed: treat as sub-2k
        if net_liq < 2000:
            logger.debug(
                "Partial / ladder skipped: market closed AND account "
                "sub-$2k (net=$%.0f). Will retry on next cycle after open.",
                net_liq,
            )
            return
        # >=$2k account: pre-market partials allowed, fall through.
        logger.debug(
            "Pre-market partial allowed (account $%.0f >= $2k tier)",
            net_liq,
        )

    try:
        portfolio = {p.contract.symbol: p for p in client._ib.portfolio()
                     if p.position != 0}
    except Exception:
        return

    changed = False
    for pos in positions:
        if pos.get("partial_taken"):
            continue  # Already sold partial
        ticker = pos["ticker"]

        # DAY-TRADE GUARD (hardened): if we can't prove the position was
        # opened BEFORE today, refuse to sell. Failing closed means:
        #  - corrupt/missing opened_at → skip (conservative)
        #  - position opened today     → skip
        #  - only when we're SURE it's a prior-day position do we proceed
        # This prevents accidental day-trade violations from exception paths.
        today_iso = date.today().isoformat()
        opened_at = str(pos.get("opened_at", "") or "")[:10]
        # Strict ISO parse — see ratchet copy below for full rationale.
        # Fail-CLOSED on parse failure (treat as same-day → skip).
        is_prior_day = False
        try:
            _od = date.fromisoformat(opened_at)
            is_prior_day = _od < date.today()
        except (ValueError, TypeError):
            is_prior_day = False
        if not is_prior_day:
            logger.debug(
                "Partial profit skip %s: opened_at=%r, today=%s (need prior-day)",
                ticker, opened_at, today_iso,
            )
            continue

        port_item = portfolio.get(ticker)
        if not port_item:
            continue

        # NaN guard — IB returns nan for marketPrice on halted stocks /
        # outside RTH for some symbols. Without explicit math.isfinite,
        # nan slips through every comparison (`nan > 0` is False, `nan < x`
        # is False, etc.) and we'd record nan into trade_log, breaking
        # the JSON downstream. (Audit finding #8.)
        import math as _m
        try:
            current_price = float(port_item.marketPrice)
        except (TypeError, ValueError):
            continue
        if not _m.isfinite(current_price) or current_price <= 0:
            continue
        entry = pos["entry_price"]
        if entry <= 0:
            continue

        pnl_pct = (current_price - entry) / entry * 100
        # Use peak_gain (not current PnL) for ladder triggers — locks
        # gains based on best-seen price, not on a pullback.
        peak_price = max(pos.get("peak_price", entry), current_price)
        peak_gain_pct = (peak_price - entry) / entry * 100

        # ── LADDER MODE (preferred): tier-based fractional sells ─────
        if CONFIG.profit_ladder_enabled:
            # 2026-05-29: skip the ladder on small positions. With original
            # qty < ladder_min_qty, "25%" either rounds to 0 (no-op) or is
            # forced to 1 share = >33% of the lot — far more aggressive than
            # intended, and it directly undercuts the validated wide-trail
            # change: a 2-share winner would dump 50% at +10% right as the
            # trend starts (ORCL ran to +16% after the +10% mark). For small
            # lots the TRAIL manages the whole position; the ladder resumes
            # once capital grows and positions are large enough to fraction
            # cleanly (qty ≥ 4 → 25% = a clean 1 share). Env: TRADE_LADDER_MIN_QTY.
            _ladder_min_qty = int(getattr(CONFIG, "ladder_min_qty", 4))
            _orig_q = int(pos.get("original_quantity", pos.get("quantity", 0)) or 0)
            if _orig_q < _ladder_min_qty:
                logger.debug(
                    "Ladder skipped for %s: qty %d < min %d — trail manages whole lot",
                    ticker, _orig_q, _ladder_min_qty,
                )
                continue
            # Determine which ladder tier the position currently qualifies for.
            tiers = [
                (3, CONFIG.profit_ladder_tier3_gain, CONFIG.profit_ladder_tier3_fraction),
                (2, CONFIG.profit_ladder_tier2_gain, CONFIG.profit_ladder_tier2_fraction),
                (1, CONFIG.profit_ladder_tier1_gain, CONFIG.profit_ladder_tier1_fraction),
            ]
            target_tier = None
            for tier_num, threshold, frac in tiers:
                if peak_gain_pct >= threshold:
                    target_tier = (tier_num, threshold, frac)
                    break
            if target_tier is None:
                continue

            tier_num, threshold, _frac = target_tier
            ladder_state = pos.get("ladder_tiers_fired", [])  # list of int tier numbers
            already_fired = sorted(ladder_state)

            # Fire ALL tiers up to target_tier that haven't fired yet,
            # in ASCENDING order (a stock that opened, surged 30% in a single
            # cycle should sell tier 1 + tier 2 + tier 3 in sequence —
            # otherwise it skips the lower tiers and undershoots cumulative).
            tiers_to_fire = []
            for tn, thr, fr in sorted(tiers, key=lambda x: x[0]):
                if tn <= tier_num and tn not in already_fired and peak_gain_pct >= thr:
                    tiers_to_fire.append((tn, thr, fr))

            if not tiers_to_fire:
                continue

            qty = int(pos["quantity"])
            original_qty = pos.get("original_quantity", qty + sum(
                int(pos.get(f"ladder_qty_t{t}", 0)) for t in already_fired
            ))
            # Ensure we record original_quantity once
            if "original_quantity" not in pos:
                pos["original_quantity"] = original_qty
                changed = True

            for tn, thr, fr in tiers_to_fire:
                # Fraction is of the ORIGINAL position, not remaining.
                # 25% × 4 tiers = 100% (last 25% reserved for TRAIL/target).
                sell_qty = max(1, int(round(original_qty * fr)))
                if sell_qty >= qty:
                    sell_qty = qty - 1  # leave at least 1 share for trail
                if sell_qty <= 0:
                    continue

                logger.info(
                    "LADDER T%d %s: peak +%.1f%% (>=%.0f%%) — selling %d (frac=%.2f of orig %d)",
                    tn, ticker, peak_gain_pct, thr, sell_qty, fr, original_qty,
                )
                result = client._sell_market(ticker, sell_qty)
                if result.status in ("Filled", "Submitted", "PreSubmitted"):
                    pnl_abs = (current_price - entry) * sell_qty
                    qty -= sell_qty
                    pos["quantity"] = qty
                    pos.setdefault("ladder_tiers_fired", []).append(tn)
                    pos[f"ladder_qty_t{tn}"] = sell_qty
                    pos[f"ladder_price_t{tn}"] = current_price
                    pos["partial_taken"] = True  # backwards-compat flag
                    changed = True
                    try:
                        tracker._log_trade("PARTIAL", ticker, sell_qty, current_price, {
                            "entry_price": entry,
                            "pnl": round(pnl_abs, 2),
                            "reason": f"ladder_t{tn}_peak_{thr:.0f}pct",
                            "remaining_qty": qty,
                            "peak_gain_pct": round(peak_gain_pct, 2),
                        })
                    except Exception as _le:
                        logger.error("Ladder log failed: %s", _le)
                    try:
                        notify._send(
                            f"📈 <b>LADDER T{tn} {ticker}</b>\n"
                            f"  Peak: +{peak_gain_pct:.1f}% (≥{thr:.0f}% threshold)\n"
                            f"  Sold {sell_qty} @ ${current_price:.2f} "
                            f"(P&L ${pnl_abs:+.2f})\n"
                            f"  Remaining: {qty} shares riding TRAIL"
                        )
                    except Exception:
                        pass
                else:
                    logger.error(
                        "LADDER T%d %s SELL failed: status=%s",
                        tn, ticker, result.status,
                    )
                    break  # stop firing this position this cycle
            continue  # move to next position

        # ── LEGACY single-trigger partial (kept for backwards-compat) ──
        if pnl_pct < CONFIG.partial_profit_trigger_pct:
            continue

        # Calculate shares to sell
        qty = int(pos["quantity"])
        sell_qty = max(1, int(qty * CONFIG.partial_profit_fraction))
        if sell_qty >= qty:
            continue  # Don't sell whole position — that's the limit_sell's job

        logger.info("PARTIAL PROFIT: %s +%.1f%% — selling %d of %d",
                     ticker, pnl_pct, sell_qty, qty)

        # Execute market sell
        result = client._sell_market(ticker, sell_qty)
        if result.status in ("Filled", "Submitted", "PreSubmitted"):
            pnl_abs = (current_price - entry) * sell_qty
            # Update tracker: reduce qty, mark partial_taken
            pos["quantity"] = qty - sell_qty
            pos["partial_taken"] = True
            pos["partial_sold_qty"] = sell_qty
            pos["partial_sold_price"] = current_price
            changed = True

            # Log to trade log
            try:
                tracker._log_trade("PARTIAL", ticker, sell_qty, current_price, {
                    "entry_price": entry,
                    "pnl": round(pnl_abs, 2),
                    "reason": f"partial_profit_{CONFIG.partial_profit_trigger_pct}%_trigger",
                    "remaining_qty": qty - sell_qty,
                })
            except Exception:
                pass

            notify._send(
                f"💰 <b>PARTIAL PROFIT {ticker}</b>\n"
                f"  Sold {sell_qty}/{qty} @ ${current_price:.2f} (+{pnl_pct:.1f}%)\n"
                f"  Locked ${pnl_abs:+.2f}\n"
                f"  Remaining: {qty - sell_qty} shares riding target"
            )
        else:
            err_msg = getattr(result, "error", "") or ""
            account_block = _is_account_restriction_error(err_msg)
            logger.warning(
                "Partial profit sell failed for %s: %s (account_rule=%s)",
                ticker, result.status, account_block,
            )
            # Position stays intact (the limit-sell at target still handles
            # full profit). But notify the user — they should know we
            # WOULD have taken a partial profit if the account weren't
            # restricted. One alert per ticker per hour (cooldown keeps
            # this from spamming during the whole cash-account window).
            if _cooldown_ok(("partial_blocked", ticker), _ALERT_COOLDOWN, _ALERT_COOLDOWN_SECONDS):
                extra = (
                    "\n\n⚠️ IBKR blocked: cash account < $2000 — partial sell rejected. "
                    "Position is unchanged; full exit will happen on trail stop "
                    "or limit-sell at target."
                ) if account_block else ""
                try:
                    notify.notify_error(
                        "Partial profit blocked",
                        f"{ticker} at +{pnl_pct:.1f}%: could not take partial "
                        f"profit ({result.status}).{extra}"
                    )
                except Exception:
                    pass

    if changed:
        tracker._save_positions(positions)


def _target_hit_pass(tracker, client, ibkr_orders, notify):
    """Detect positions whose current price reached the target_price and
    market-sell them. Software replacement for the OCA LMT order that
    sub-$2k IB cash accounts cannot reliably place (Error 201 — margin
    requirement of OCA group exceeds account equity).

    Behavior:
      For each open position:
        1. Read current market price from IB portfolio
        2. If current_price >= target_price × 0.999 (1bp slop for fills):
           a. Cancel the existing TRAIL (and any sibling OCA orders)
           b. Place MKT SELL for the full quantity
           c. Wait up to 10s for fill
           d. Record the close in tracker + notify Telegram
      Otherwise: leave alone (TRAIL keeps protecting downside).

    Idempotency: if a previous cycle already initiated the sell but the
    fill hasn't propagated yet, IB will return the existing trade. The
    `target_hit_initiated_at` tag on the position guards against double-
    submit within the same trading minute.

    This runs BEFORE _ratchet_stops so a target-hit position exits
    cleanly without the ratchet tightening it mid-flight.
    """
    from core.trading.config import CONFIG
    import math as _m
    from datetime import datetime as _dt, timezone as _tz

    positions = tracker.get_open_positions()
    if not positions:
        return

    try:
        portfolio = {p.contract.symbol: p for p in client._ib.portfolio()
                     if p.position != 0}
    except Exception:
        logger.warning("Target-hit pass: couldn't read portfolio")
        return

    # Map of ticker -> active TRAIL/LMT orders (for cancel before market sell)
    orders_by_ticker = {}
    for o in ibkr_orders:
        t = o.get("ticker", "")
        if t and o.get("status") in ("Submitted", "PreSubmitted"):
            orders_by_ticker.setdefault(t, []).append(o)

    changed = False
    for pos in positions:
        ticker = pos["ticker"]
        entry = float(pos.get("entry_price", 0) or 0)
        target = float(pos.get("target_price", 0) or 0)
        qty = int(pos.get("quantity", 0) or 0)
        if entry <= 0 or target <= 0 or qty <= 0:
            continue

        port_item = portfolio.get(ticker)
        if not port_item:
            continue
        current_price = float(port_item.marketPrice)
        if not _m.isfinite(current_price) or current_price <= 0:
            continue

        # Slop: 0.1% under target counts as "hit" (covers IB rounding +
        # the gap between marketPrice and actual fill price). Picked to
        # match the existing _classify_reason logic in run_check (uses 1%).
        hit_threshold = target * 0.999
        if current_price < hit_threshold:
            continue

        # Idempotency — don't re-fire if already sold this cycle
        last_initiated = pos.get("target_hit_initiated_at")
        if last_initiated:
            try:
                _dt_last = _dt.fromisoformat(last_initiated.replace("Z", "+00:00"))
                now = _dt.now(_tz.utc)
                if (now - _dt_last).total_seconds() < 120:
                    logger.info("Target-hit %s already initiated %.0fs ago — skipping",
                                ticker, (now - _dt_last).total_seconds())
                    continue
            except Exception:
                pass

        gain_pct = (current_price - entry) / entry * 100
        logger.info(
            "🎯 TARGET HIT %s: current $%.2f >= target $%.2f (gain +%.2f%%) — selling",
            ticker, current_price, target, gain_pct,
        )

        # Mark intent immediately so we don't re-fire on parallel cycles.
        pos["target_hit_initiated_at"] = _dt.utcnow().isoformat() + "Z"
        changed = True
        try:
            tracker._save_positions(positions)
        except Exception:
            pass

        # Cancel sibling OCA orders first so they don't fire on the
        # downward tick after our market sell.
        oca = pos.get("order_ids", {}).get("oca_group", "")
        cancelled = 0
        for o in orders_by_ticker.get(ticker, []):
            if oca and o.get("oca_group") != oca:
                continue
            if o.get("order_type") in ("TRAIL", "LMT", "STP"):
                try:
                    for t in client._ib.openTrades():
                        if t.order.orderId == o.get("order_id"):
                            client._ib.cancelOrder(t.order)
                            cancelled += 1
                            break
                except Exception as _ce:
                    logger.warning("Cancel %s #%s failed: %s",
                                   ticker, o.get("order_id"), _ce)
        if cancelled:
            client._ib.sleep(2)

        # Market sell
        try:
            result = client._sell_market(ticker, qty)
        except Exception as _se:
            logger.error("Target-hit market sell FAILED for %s: %s", ticker, _se)
            notify.notify_error("Monitor", f"Target-hit sell FAILED {ticker}: {_se}")
            continue

        exit_price = getattr(result, "filled_price", 0.0) or 0.0
        if exit_price <= 0:
            logger.warning(
                "Target-hit %s: sell status=%s, no fill price yet — tracker "
                "will reconcile on next cycle",
                ticker, getattr(result, "status", "?"),
            )
            continue

        pnl = (exit_price - entry) * qty
        try:
            tracker.remove_position(ticker, exit_price, "target_hit")
        except Exception as _re:
            logger.warning("remove_position failed for %s: %s", ticker, _re)

        try:
            notify.notify_sell(ticker, qty, exit_price, "target_hit", pnl)
        except Exception:
            pass

        # Explicit alert distinguishing target-hit from trail
        try:
            notify._send(
                f"🎯 <b>TARGET HIT {ticker}</b>\n"
                f"  Entry: ${entry:.2f} → Exit: ${exit_price:.2f}\n"
                f"  Target was: ${target:.2f}\n"
                f"  Gain: +{gain_pct:.2f}% (${pnl:+.2f})\n"
                f"  Monitor-side fill (LMT was missing due to IB margin cap)"
            )
        except Exception:
            pass


def _ratchet_stops(tracker, client, ibkr_orders, notify):
    """Dynamic trail-tightening — make protection more aggressive as the
    position runs up, while keeping IB's native peak-tracking continuous.

    Old behavior (pre 2026-04-28): replaced the TRAIL with a static STP
    at a profit floor when peak_gain crossed a threshold. Problem: between
    tiers the floor was frozen, so a stock peaking at +17.9% only locked
    +3% (the tier-1 floor). The static STP couldn't track new peaks above
    the threshold.

    New behavior: MODIFY the existing TRAIL's trailingPercent in-place on
    IB's server. IB continues tracking the peak — the only thing that
    changes is HOW CLOSE the stop trails. Every cent the stock rises
    above a tier raises the stop too.

    Tiers:
      Peak +10% → trail tightens to 4% (was 5–8% scan-derived)
      Peak +18% → trail tightens to 3%
      Peak +28% → trail tightens to 2%

    Only TIGHTENS — never loosens. A position that started with TRAIL 3%
    keeps TRAIL 3% even if the tier table says 4%.
    """
    from core.trading.config import CONFIG

    positions = tracker.get_open_positions()
    if not positions:
        return

    # Get live portfolio (current prices)
    try:
        portfolio = {p.contract.symbol: p for p in client._ib.portfolio()
                     if p.position != 0}
    except Exception:
        logger.warning("Ratchet: couldn't get portfolio")
        return

    # Build active ticker orders map
    orders_by_ticker = {}
    for o in ibkr_orders:
        t = o.get("ticker", "")
        if t and o.get("status") in ("Submitted", "PreSubmitted"):
            orders_by_ticker.setdefault(t, []).append(o)

    # Ordered HIGHEST-gain first so the highest-applicable tier wins
    # (a stock at +25% peak should get tier 2's tighter trail, not
    # tier 0's wider one). Tier 0 added 2026-05-05 — early lock at +5%.
    tiers = [
        (CONFIG.ratchet_tier3_gain, CONFIG.ratchet_tier3_trail_pct),
        (CONFIG.ratchet_tier2_gain, CONFIG.ratchet_tier2_trail_pct),
        (CONFIG.ratchet_tier1_gain, CONFIG.ratchet_tier1_trail_pct),
        (CONFIG.ratchet_tier0_gain, CONFIG.ratchet_tier0_trail_pct),
    ]

    changed = False
    for pos in positions:
        ticker = pos["ticker"]
        entry = pos["entry_price"]
        oca = pos.get("order_ids", {}).get("oca_group", "")

        # DAY-TRADE GUARD: tightening trail can immediately trigger if the
        # new % is closer than the current pullback. Fail closed for
        # same-day positions (cash account = T+1 day-trade violation).
        today_iso = date.today().isoformat()
        opened_at = str(pos.get("opened_at", "") or "")[:10]
        # Strict ISO parse — lexical compare alone breaks on malformed
        # dates like "2026-4-30" or " 2026-04-30" (lex < "2026-04-30" lex
        # incorrectly says is_prior_day=True). Fail-CLOSED on parse failure
        # so we treat ambiguous as same-day and skip the action that
        # could trigger a day-trade violation. (Audit finding #5.)
        is_prior_day = False
        try:
            _od = date.fromisoformat(opened_at)
            is_prior_day = _od < date.today()
        except (ValueError, TypeError):
            is_prior_day = False  # treat malformed as same-day (skip)
        if not is_prior_day:
            logger.debug(
                "Ratchet skip %s: opened_at=%r, today=%s (need prior-day)",
                ticker, opened_at, today_iso,
            )
            continue

        port_item = portfolio.get(ticker)
        if not port_item:
            continue

        current_price = float(port_item.marketPrice)
        if current_price <= 0:
            continue

        # Update peak tracking.
        # 2026-05-19 BUGFIX: the comparison `peak_price != pos.get("peak_price")`
        # used to fail-closed: when the field was MISSING (e.g. brand-new
        # position), pos.get returned `entry` as default, and if
        # current_price <= entry, peak_price was max(entry,current) = entry,
        # so "entry != entry" = False → field NEVER persisted. Positions
        # whose first monitor cycle saw current<entry would have peak_price
        # absent forever, breaking break-even + ratchet permanently for
        # them. Now we check field existence explicitly.
        peak_price = max(pos.get("peak_price", entry), current_price)
        if "peak_price" not in pos or peak_price != pos["peak_price"]:
            pos["peak_price"] = peak_price
            changed = True

        # Calculate peak gain %
        peak_gain_pct = (peak_price - entry) / entry * 100

        # Find applicable tier (highest first)
        # T0 hold-days gate (added 2026-05-07 after backtest revealed T0
        # killed winners by triggering on day-1 FOMO spikes). T0 is the
        # LOWEST tier (lowest gain threshold) — stocks that move +5-8% on
        # day 1 are typically momentum chasers that fade. Letting them
        # mature for 2+ days before tightening filters that noise.
        # T1/T2/T3 still fire at any age; their thresholds (10/18/28%)
        # are high enough that "already a real run" is implicit.
        try:
            from datetime import datetime as _dt
            opened_at_str = str(pos.get("opened_at", "") or "")[:19]
            opened_dt = _dt.fromisoformat(opened_at_str) if opened_at_str else None
            hold_days = (_dt.utcnow() - opened_dt).days if opened_dt else 0
        except Exception:
            hold_days = 0
        t0_min_days = int(getattr(CONFIG, "min_hold_days_for_t0", 2))
        t0_gain_threshold = float(getattr(CONFIG, "ratchet_tier0_gain", 8.0))

        target_trail_pct = None
        _trail_source = None  # for logging / notification differentiation
        for threshold, trail_pct in tiers:
            if peak_gain_pct >= threshold:
                # T0-specific time gate: skip if too young.
                # (The tiers list has T0 as the lowest threshold — we detect
                # by matching the configured T0 gain, not by position in
                # the list, so a future config rewrite stays safe.)
                if abs(threshold - t0_gain_threshold) < 0.01 and hold_days < t0_min_days:
                    logger.debug(
                        "Ratchet T0 skip %s: peak +%.1f%% qualifies but "
                        "hold_days=%d < min %d (giving early move time to "
                        "prove itself)",
                        ticker, peak_gain_pct, hold_days, t0_min_days,
                    )
                    continue
                target_trail_pct = trail_pct
                _trail_source = "ratchet"
                break

        # ── BREAK-EVEN PROTECTION (added 2026-05-15) ──
        # Closes the gap between entry and the lowest ratchet tier (T0 = +8%)
        # where the base trail (~4%) could still exit at a loss.
        #
        # Dynamic: compute the exact trail % needed so the stop floors at
        # entry × break_even_floor_mult (default 1.002 = +0.2% above entry,
        # covers IBKR commissions). As peak rises, this becomes a wider
        # trail; once it widens past the current trail OR past a ratchet
        # tier's tighter trail, the existing "only-tighten" guard naturally
        # ignores it. So it only fires in the narrow window where the base
        # trail's stop is below break-even but the position is in profit.
        #
        # Forensic justification: in the last 7 trades, RSI peaked +1.17%
        # then closed at -2.88% (4% trail × +1.17% peak = stop at -2.83%).
        # Break-even @ +2% threshold would have moved the stop to entry +0.2%
        # at the moment peak crossed +2%, locking in scratch instead of loss.
        be_cfg_enabled = bool(getattr(CONFIG, "break_even_enabled", True))
        if be_cfg_enabled:
            be_threshold = float(getattr(CONFIG, "break_even_threshold_pct", 2.0))
            be_floor_mult = float(getattr(CONFIG, "break_even_floor_mult", 1.002))
            be_min_trail = float(getattr(CONFIG, "break_even_min_trail_pct", 0.5))
            if peak_gain_pct >= be_threshold:
                target_floor_be = entry * be_floor_mult
                if peak_price > target_floor_be:
                    be_trail_pct = (peak_price - target_floor_be) / peak_price * 100
                    # Only use if it's a TIGHTER candidate than any ratchet
                    # tier already chosen, AND not so tight it would fire on
                    # noise. The downstream "only-tighten vs current_trail"
                    # guard then makes the final apply/skip decision.
                    if be_trail_pct >= be_min_trail and (
                        target_trail_pct is None or be_trail_pct < target_trail_pct
                    ):
                        target_trail_pct = be_trail_pct
                        _trail_source = "break_even"

        if target_trail_pct is None:
            continue  # Not profitable enough yet

        # Find the active TRAIL or STP order for this position
        # (after the old code's runs, some positions may still be on STP —
        # we'll handle them by cancelling and replacing with TRAIL).
        active_protective = None
        for o in orders_by_ticker.get(ticker, []):
            if o.get("oca_group") and o.get("oca_group") != oca:
                continue
            if o.get("order_type") in ("TRAIL", "STP"):
                active_protective = o
                break

        if not active_protective:
            logger.debug("Ratchet %s: no active TRAIL/STP found, skipping", ticker)
            continue

        order_type = active_protective.get("order_type")
        order_id = active_protective.get("order_id")

        # ── INVARIANT: Ratchet only TIGHTENS, never loosens. ──
        # `config.py` documents this guarantee; THIS LINE enforces it.
        # If you remove the guard below, a stock that started with a 3%
        # trail and then ran to +12% (tier 1 = 4%) would have its trail
        # WIDENED to 4%, giving up 1pp of locked-in gain. Always verify
        # this condition before any code that calls modify_trailing_pct.
        # (Audit N4 documentation 2026-05-01.)
        current_trail = float(pos.get("trailing_stop_pct", 0) or 0)
        if order_type == "TRAIL" and current_trail > 0 and target_trail_pct >= current_trail:
            # target_trail_pct >= current_trail means the candidate would
            # NOT be tighter — skip. (>= covers the equal case so we don't
            # pay an IB modify roundtrip for no behavioral change.)
            continue

        # Safety: skip ONLY when the projected stop would be ABOVE
        # current price (and would fire immediately). The old check used
        # `>= current * 0.995` unconditionally, which blocked tightening
        # when a stock pulled back from peak — exactly when you most want
        # the protection ratchet (peak +28%, current +25% → tier 3 trail
        # 2% projected at peak * 0.98 = current * 1.005 would skip with
        # the old check, leaving the loose trail in place). Now: skip
        # only if the trail would be ABOVE current price; below current
        # is fine because IB's trailing-stop logic is "don't fire until
        # market drops to stop" and the starting trail amount means
        # current price is above stop already. (Audit finding #4 — monitor.)
        projected_stop = peak_price * (1 - target_trail_pct / 100)
        if projected_stop >= current_price:
            logger.debug(
                "Ratchet skip %s: projected stop $%.2f at-or-above current $%.2f "
                "(would fire immediately)",
                ticker, projected_stop, current_price,
            )
            continue

        # ── Runtime invariant (2026-05-15) ──
        # The "only-tighten" guard above should have rejected any widening
        # candidate. Belt-and-suspenders assert: if we got here with a
        # trail >= current, that's a logic bug somewhere.
        if current_trail > 0 and target_trail_pct >= current_trail:
            logger.error(
                "INVARIANT: ratchet tried to LOOSEN %s trail %.2f%% → %.2f%% "
                "(source=%s). REFUSING. The only-tighten gate above should "
                "have caught this — there is a bug.",
                ticker, current_trail, target_trail_pct, _trail_source,
            )
            continue

        # ── Path A: Order is already a TRAIL → modify the % in place
        if order_type == "TRAIL":
            result = client.modify_trailing_pct(order_id, target_trail_pct)
            if result.status in ("Submitted", "PreSubmitted", "PendingSubmit", "DRY_RUN"):
                pos["trailing_stop_pct"] = target_trail_pct
                pos["stop_floor"] = projected_stop
                # If the modify went through the cancel+replace fallback
                # (Error 103 path), result.order_id is the NEW order. Update
                # tracker so the next ratchet cycle finds the right order.
                # Without this, tracker points to a cancelled stale ID and
                # while monitor's order-by-ticker map (built from IB) still
                # works for finding the current order, anything reading
                # order_ids directly (e.g. /resubmit, manual cancel) breaks.
                new_oid = getattr(result, "order_id", 0) or 0
                if new_oid and new_oid != order_id:
                    logger.info(
                        "Ratchet fallback created new TRAIL #%d for %s "
                        "(was #%d) — updating tracker order_ids",
                        new_oid, ticker, order_id,
                    )
                    pos.setdefault("order_ids", {})["trailing_stop"] = new_oid
                changed = True
                # Telegram notification — distinguish break-even from ratchet
                lock_pct = (projected_stop - entry) / entry * 100
                lock_amt = (projected_stop - entry) * int(pos["quantity"])
                if _trail_source == "break_even":
                    notify._send(
                        f"🛡 <b>BREAK-EVEN {ticker}</b>\n"
                        f"  Peak: +{peak_gain_pct:.1f}% (${peak_price:.2f})\n"
                        f"  Trail: {current_trail:.1f}% → <b>{target_trail_pct:.2f}%</b>\n"
                        f"  Projected stop: ${projected_stop:.2f} "
                        f"(+{lock_pct:.2f}%, ${lock_amt:+.2f}) — scratch lock"
                    )
                else:
                    notify._send(
                        f"🔒 <b>TIGHTEN {ticker}</b>\n"
                        f"  Peak: +{peak_gain_pct:.1f}% (${peak_price:.2f})\n"
                        f"  Trail: {current_trail:.1f}% → <b>{target_trail_pct:.1f}%</b>\n"
                        f"  Projected stop: ${projected_stop:.2f} "
                        f"(+{lock_pct:.1f}%, ${lock_amt:+.2f})"
                    )
            else:
                logger.error("Ratchet modify FAILED for %s: %s",
                             ticker, getattr(result, "error", ""))
            continue

        # ── Path B: Order is a STATIC STP (legacy from old ratchet code).
        # Cancel ALL TRAIL/STP orders in this OCA group (the old code stacked
        # multiple STPs — e.g. CF had STP $114.67 AND breakeven STP $111.33;
        # we want a single clean TRAIL replacing all of them). Then place a
        # new TRAIL with the target %. This is the ONE-TIME migration path.
        qty = int(pos["quantity"])
        cancelled_ids = []
        try:
            for o in orders_by_ticker.get(ticker, []):
                if o.get("oca_group") != oca:
                    continue
                if o.get("order_type") not in ("TRAIL", "STP"):
                    continue
                oid = o.get("order_id")
                for t in client._ib.openTrades():
                    if t.order.orderId == oid:
                        client._ib.cancelOrder(t.order)
                        cancelled_ids.append(oid)
                        logger.info(
                            "Ratchet migrate: cancelled legacy %s #%d for %s",
                            o.get("order_type"), oid, ticker,
                        )
                        break
            if cancelled_ids:
                client._ib.sleep(2)
        except Exception as e:
            logger.error("Ratchet legacy cancel failed for %s: %s", ticker, e)
            continue

        # Place new TRAIL — use the existing OCA so it cancels with limit_sell
        result = client.set_trailing_stop(ticker, qty, target_trail_pct, oca_group=oca)
        if result.status in ("Submitted", "PreSubmitted", "PendingSubmit", "DRY_RUN"):
            pos["order_ids"]["trailing_stop"] = result.order_id
            pos["order_ids"]["stop_type"] = "TRAIL"
            pos["trailing_stop_pct"] = target_trail_pct
            pos["stop_floor"] = projected_stop
            changed = True
            lock_pct = (projected_stop - entry) / entry * 100
            lock_amt = (projected_stop - entry) * qty
            notify._send(
                f"🔄 <b>MIGRATE→TRAIL {ticker}</b>\n"
                f"  Replaced legacy static STP with TRAIL {target_trail_pct:.1f}%\n"
                f"  Peak: +{peak_gain_pct:.1f}% (${peak_price:.2f})\n"
                f"  Projected stop: ${projected_stop:.2f} "
                f"(+{lock_pct:.1f}%, ${lock_amt:+.2f})"
            )
        else:
            err_msg = getattr(result, "error", "") or ""
            account_block = _is_account_restriction_error(err_msg)
            logger.error(
                "Ratchet migrate FAILED for %s: status=%s (account_rule=%s)",
                ticker, result.status, account_block,
            )
            # Only alert once per hour — avoid spam when cash-account rule blocks.
            if _cooldown_ok(("ratchet", ticker), _ALERT_COOLDOWN, _ALERT_COOLDOWN_SECONDS):
                extra = ""
                if account_block:
                    extra = " (IBKR blocked: cash account < $2000)"
                notify.notify_error(
                    "Ratchet",
                    f"{ticker}: failed to migrate STP→TRAIL "
                    f"@ {target_trail_pct:.1f}%: {result.status}{extra}"
                )

    if changed:
        tracker._save_positions(positions)


def daemon_loop():
    """Run monitoring loop. Adaptive sleep so we don't lag at market open.

    Pre-market we poll `is_market_open()` every 60s — when the market
    transitions open, the FIRST cycle runs within ~60s of the open bell
    instead of "current sleep finishes + market check". With CHECK_INTERVAL=300
    that previously meant up to 5 min of lag at open, during which the
    ratchet didn't fire and trail tightening lagged behind first-30-min
    moves. The 60s pre-market poll is cheap (single is_market_open() call,
    no IB queries) and lets us hit the bell within the first minute.
    """
    from core.trading.ibkr_client import IBKRClient

    logger.info(
        "Position monitor daemon started "
        "(market-hours interval %ds, pre-market poll 60s)",
        CHECK_INTERVAL,
    )
    client = IBKRClient()
    PRE_MARKET_POLL_SEC = 60

    # ── Startup reconcile (added 2026-05-05) ──
    # Runs ONCE before the main loop, regardless of market state, so
    # orphan IB positions from a prior crash/SIGTERM mid-trade are
    # detected immediately instead of waiting for market open.
    # The kill-mid-execution scenario: SIGTERM hits between
    # broker.buy_with_bracket() (which placed the order on IB) and
    # tracker.add_position() (which records it locally) — IB is now
    # holding a position the tracker doesn't know about. The existing
    # _try_adopt_ib_only logic handles this on the FIRST cycle that
    # runs run_check(), but daemon_loop's market_open gate previously
    # delayed that recovery up to ~16 hours overnight. Now we always
    # reconcile at startup. Tolerates IB connection failure (e.g. the
    # IB session needed 2FA) — just logs and continues to the loop
    # which will retry.
    logger.info("Startup reconcile — checking for orphan IB positions...")
    try:
        with _cycle_timeout(CYCLE_TIMEOUT_SECONDS):
            run_check()
        logger.info("Startup reconcile complete")
    except MonitorTimeout:
        logger.warning("Startup reconcile timed out — continuing to loop")
    except Exception as e:
        logger.warning("Startup reconcile failed: %s — continuing to loop", e)

    # 2026-05-26: liveness signals during off-hours.
    #
    # Background: the previous loop only called run_check() (which writes
    # the snapshot file) when is_market_open(). During the Memorial Day
    # weekend the monitor was in the else branch for 62+ hours straight —
    # silently polling every 60s but never logging (DEBUG level) and
    # never updating the snapshot file. From the outside it was
    # indistinguishable from a stuck process, which triggered repeated
    # false-positive "Portfolio snapshot STALE" alerts on Sunday.
    #
    # Fix:
    #   a) Touch the snapshot file every off-hours cycle so its mtime
    #      always reflects "monitor is alive and polling". The content
    #      stays the last-known state from the most recent run_check().
    #   b) Emit an INFO heartbeat log every N off-hours polls so
    #      journalctl can confirm liveness without having to enable DEBUG.
    from pathlib import Path
    _SNAPSHOT_PATH = Path("data/trades/portfolio_snapshot.json")
    OFFHOURS_HEARTBEAT_EVERY = 5   # log every N polls = ~5 min
    _offhours_poll = 0

    while True:
        if client.is_market_open():
            try:
                with _cycle_timeout(CYCLE_TIMEOUT_SECONDS):
                    run_check()
            except MonitorTimeout as e:
                logger.error(
                    "Monitor cycle TIMED OUT (%s) — skipping to next cycle",
                    e,
                )
                # Best-effort cleanup: ib_insync connections may be stuck
                try:
                    client.disconnect()
                except Exception:
                    pass
            except Exception as e:
                logger.error("Monitor cycle failed: %s", e)
            _offhours_poll = 0  # reset so first off-hours poll always logs
            time.sleep(CHECK_INTERVAL)
        else:
            # Closed — short poll so we catch the open bell within ~60s.
            _offhours_poll += 1
            try:
                _SNAPSHOT_PATH.touch(exist_ok=True)
            except Exception as _touch_exc:
                # Don't fail the loop if FS is misbehaving; just log once.
                if _offhours_poll == 1:
                    logger.warning(
                        "Could not touch snapshot for liveness: %s",
                        _touch_exc,
                    )
            if _offhours_poll % OFFHOURS_HEARTBEAT_EVERY == 1:
                logger.info(
                    "Heartbeat: market closed, idle (poll #%d, snapshot mtime refreshed)",
                    _offhours_poll,
                )
            time.sleep(PRE_MARKET_POLL_SEC)


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
