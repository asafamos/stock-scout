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
                # Position was closed (stop or target hit)
                logger.info("Position %s no longer in IBKR — marking closed", ticker)

                # Try to determine exit price and reason from filled orders.
                # IB's fills can be sparse right after a close — try twice with
                # a short gap to give the API time to propagate the execution.
                exit_price = 0.0
                reason = "closed_externally"

                def _try_fills():
                    try:
                        for f in client._ib.fills():
                            if (f.contract.symbol == ticker
                                    and f.execution.side in ("SLD", "SELL")):
                                fp = float(f.execution.price or 0)
                                if fp > 0:
                                    return fp
                    except Exception as _e:
                        logger.debug("Fills check failed: %s", _e)
                    return 0.0

                exit_price = _try_fills()
                if exit_price == 0.0:
                    # Wait briefly and try once more — fills may still be landing
                    try:
                        client._ib.sleep(2)
                    except Exception:
                        pass
                    exit_price = _try_fills()
                if exit_price > 0:
                    reason = "stop_or_target_filled"

                # Secondary: check completed trades for fill info
                if exit_price == 0.0:
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

                # Last resort: use executions() which has recent fills
                if exit_price == 0.0:
                    try:
                        for e in client._ib.executions():
                            if (e.contract.symbol == ticker
                                    and e.execution.side in ("SLD", "SELL")):
                                exit_price = float(e.execution.price or 0)
                                if exit_price > 0:
                                    reason = "fill_detected"
                                    break
                    except Exception:
                        pass

                # FINAL fallback: estimate from stop_loss or peak_price.
                # Flag this clearly so the user knows to verify in IB directly.
                estimated = False
                if exit_price == 0.0:
                    exit_price = (pos.get("stop_loss") or pos.get("peak_price")
                                  or pos["entry_price"])
                    reason = f"{reason}_estimated"
                    estimated = True

                tracker.remove_position(ticker, exit_price, reason)
                pnl = (exit_price - pos["entry_price"]) * pos["quantity"] if exit_price else 0
                notify.notify_sell(ticker, pos["quantity"], exit_price, reason, pnl)
                if estimated:
                    # Emphasized follow-up so the user checks IB for true fill price
                    try:
                        notify.notify_error(
                            "Exit price estimated",
                            f"⚠️ {ticker} exit price could not be retrieved from IB — "
                            f"using estimate ${exit_price:.2f}. "
                            f"Please verify the actual fill in IB and correct "
                            f"the trade_log if needed."
                        )
                    except Exception:
                        pass

        # 2. Verify protective orders — every position MUST have live orders
        _verify_protections(tracker, client, ibkr_orders, notify)

        # 2b. Ratchet stops up as positions run up (lock in profits)
        if CONFIG.ratchet_enabled:
            _ratchet_stops(tracker, client, ibkr_orders, notify)

        # 2b2. Partial profit-taking (sell half when intermediate target hit)
        if CONFIG.partial_profit_enabled:
            _take_partial_profit(tracker, client, notify)

        # 2c. Push portfolio snapshot to Supabase (for Streamlit UI)
        try:
            from core.trading.portfolio_snapshot import write_snapshot
            write_snapshot(client, tracker)
        except Exception as e:
            logger.debug("Snapshot push failed: %s", e)

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
            # Find a complete group (TRAIL + LMT or STP + LMT)
            adopted_oca = None
            for g, orders in by_oca.items():
                types = {o.get("order_type") for o in orders}
                has_stop = bool(types & {"TRAIL", "STP"})
                has_limit = "LMT" in types
                if has_stop and has_limit:
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
            notify.notify_buy(
                ticker, int(qty), pos["entry_price"],
                pos.get("stop_loss", 0), target_price,
                pos.get("score", 0),
                trail_pct=trail_pct, rr=0,
                target_date=pos.get("target_date", ""),
                prefix="🔄 AUTO-RESUBMIT",
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
            # when IB's cash-account rule is the persistent cause.
            if _cooldown_ok(("protection", ticker), _ALERT_COOLDOWN, _ALERT_COOLDOWN_SECONDS):
                extra = ""
                if account_block:
                    extra = (
                        "\n\n⚠️ IBKR rejected the order: cash account < $2000 "
                        "blocks new protective orders. Existing orders (if any) "
                        "may still be active — check the Portfolio Status."
                    )
                notify.notify_error("Protection",
                    f"CRITICAL: {ticker} has NO protective orders! "
                    f"Trail: {result['trailing_stop'].status}, "
                    f"Limit: {result['limit_sell'].status}{extra}"
                )


def _take_partial_profit(tracker, client, notify):
    """Sell half of a position when it crosses partial_profit_trigger_pct.

    Prevents giving back gains on stocks that spike then pull back.
    Flag stored in position.partial_taken to avoid double-execution.
    """
    from core.trading.config import CONFIG

    positions = tracker.get_open_positions()
    if not positions:
        return

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
        is_prior_day = bool(opened_at) and opened_at != today_iso and opened_at < today_iso
        if not is_prior_day:
            logger.debug(
                "Partial profit skip %s: opened_at=%r, today=%s (need prior-day)",
                ticker, opened_at, today_iso,
            )
            continue

        port_item = portfolio.get(ticker)
        if not port_item:
            continue

        current_price = float(port_item.marketPrice)
        entry = pos["entry_price"]
        if entry <= 0 or current_price <= 0:
            continue

        pnl_pct = (current_price - entry) / entry * 100
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

    tiers = [
        (CONFIG.ratchet_tier3_gain, CONFIG.ratchet_tier3_trail_pct),
        (CONFIG.ratchet_tier2_gain, CONFIG.ratchet_tier2_trail_pct),
        (CONFIG.ratchet_tier1_gain, CONFIG.ratchet_tier1_trail_pct),
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
        is_prior_day = bool(opened_at) and opened_at < today_iso
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

        # Update peak tracking
        peak_price = max(pos.get("peak_price", entry), current_price)
        if peak_price != pos.get("peak_price"):
            pos["peak_price"] = peak_price
            changed = True

        # Calculate peak gain %
        peak_gain_pct = (peak_price - entry) / entry * 100

        # Find applicable tier (highest first)
        target_trail_pct = None
        for threshold, trail_pct in tiers:
            if peak_gain_pct >= threshold:
                target_trail_pct = trail_pct
                break

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

        # Only TIGHTEN — never loosen
        current_trail = float(pos.get("trailing_stop_pct", 0) or 0)
        if order_type == "TRAIL" and current_trail > 0 and target_trail_pct >= current_trail:
            continue  # Already tighter or equal

        # Safety: with the new trail %, the projected stop should still
        # be below current price by a reasonable margin (else immediate fill).
        projected_stop = peak_price * (1 - target_trail_pct / 100)
        if projected_stop >= current_price * 0.995:
            logger.debug(
                "Ratchet skip %s: projected stop $%.2f too close to current $%.2f",
                ticker, projected_stop, current_price,
            )
            continue

        # ── Path A: Order is already a TRAIL → modify the % in place
        if order_type == "TRAIL":
            result = client.modify_trailing_pct(order_id, target_trail_pct)
            if result.status in ("Submitted", "PreSubmitted", "PendingSubmit", "DRY_RUN"):
                pos["trailing_stop_pct"] = target_trail_pct
                pos["stop_floor"] = projected_stop
                changed = True
                # Telegram notification
                lock_pct = (projected_stop - entry) / entry * 100
                lock_amt = (projected_stop - entry) * int(pos["quantity"])
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
        # Cancel it and place a new TRAIL with the target %. This is the
        # ONE-TIME migration path; once everyone is on TRAIL we never hit it.
        qty = int(pos["quantity"])
        try:
            for t in client._ib.openTrades():
                if t.order.orderId == order_id:
                    client._ib.cancelOrder(t.order)
                    logger.info(
                        "Ratchet migrate: cancelled legacy STP #%d for %s",
                        order_id, ticker,
                    )
                    break
            client._ib.sleep(2)
        except Exception as e:
            logger.error("Ratchet legacy cancel failed for %s: %s", ticker, e)
            continue

        # Place new TRAIL — use the existing OCA so it cancels with limit_sell
        result = client.set_trailing_stop(ticker, qty, target_trail_pct)
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
    """Run monitoring loop during market hours."""
    from core.trading.ibkr_client import IBKRClient

    logger.info("Position monitor daemon started (checking every %ds)", CHECK_INTERVAL)
    client = IBKRClient()

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
