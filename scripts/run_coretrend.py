"""CoreTrend LIVE executor (one idempotent run per trading day, 09:45 ET via systemd). Default OFF.

Enable with TRADE_CORETREND=1 (VPS .env.trading). Real orders only with --live or systemd's TRADE_LIVE_CONFIRMED=1,
otherwise a DRY plan is printed. Kill switch: data/state/coretrend_disabled.json.
Rule/evidence: core/trading/coretrend.py. Planning: core/trading/coretrend_exec.py.
Target = the rule evaluated on QQQ month-end closes of COMPLETE months only (decision at month-end close, executed
from the next session's open, like the research). Safety gates: enabled flag, kill file, market open, trade lock,
no open BUY for the symbol, real-time price sanity (+-3% vs last close), cash reserve, whole shares, wide protective
TRAIL on every buy, Telegram on every action/error. Never uses margin (budget is bounded by cash).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("coretrend")
KILL = ROOT / "data" / "state" / "coretrend_disabled.json"


def current_signal(today: date):
    from core.trading import coretrend as ct
    from scripts.coretrend_paper import fetch_daily
    daily = [(d, c) for d, c in fetch_daily(ct.RISK_ASSET) if (d.year, d.month) < (today.year, today.month)]
    return ct.signal(ct.month_end_closes(daily)), daily[-1][0] if daily else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--netliq", type=float, default=0.0, help="DRY only: pretend NetLiq")
    ap.add_argument("--cash", type=float, default=0.0, help="DRY only: pretend cash")
    a = ap.parse_args(argv)
    if os.getenv("TRADE_CORETREND", "0").strip() not in ("1", "true", "True"):
        print("CoreTrend disabled (TRADE_CORETREND != 1)"); return 0
    if KILL.exists():
        print("CoreTrend kill-switch file present:", KILL.read_text()[:200]); return 0
    live = (a.live or os.getenv("TRADE_LIVE_CONFIRMED") == "1") and not a.dry_run
    os.environ["TRADE_DRY_RUN"] = "0" if live else "1"
    from core.trading.config import CONFIG
    from core.trading.market_hours import is_regular_session
    from core.trading import coretrend_exec as ex
    from core.trading import notifications as notify
    today = datetime.now(timezone.utc).date()
    if live and not is_regular_session():
        print("market closed — no live run"); return 0
    sig, asof = current_signal(today)
    if not sig:
        notify.notify_error("CoreTrend", "not enough QQQ history to compute the signal"); return 1
    logger.info("signal as of %s: %s (QQQ %.2f vs SMA10m %.2f, %+.1f%%) -> %s", asof, sig["state"], sig["close"], sig["sma"], sig["gap_pct"], ex.want_instrument(sig["state"]))
    from core.trading.ibkr_client import IBKRClient
    from core.trading.order_manager import _acquire_trade_lock, _release_trade_lock
    client = IBKRClient(CONFIG)
    lock = None if not live else _acquire_trade_lock()
    if live and lock is None:
        print("another trade run holds the lock — skipping"); return 0
    try:
        if not client.connect():
            notify.notify_error("CoreTrend", "cannot connect to IBKR"); return 1
        if live:
            holdings = {p.contract.symbol: float(p.position) for p in client._ib.positions() if p.position != 0}   # UNFILTERED (ignore-list hides ETFs)
            cash = float(client.get_cash_balance() or 0); netliq = float(client.get_net_liquidation() or 0)
            open_buys = set(client.get_open_buy_symbols())
        else:
            holdings = {}; netliq = a.netliq or float(client.get_net_liquidation() or 0); cash = a.cash or netliq; open_buys = set()
        prices = {}
        from core.trading.live_quote import get_realtime_price
        for sym in (ex.RISK_INSTRUMENT, ex.SAFE_INSTRUMENT):
            try:
                rt = get_realtime_price(sym); prices[sym] = rt[0] if rt else float(client.get_live_price(sym) or 0)
            except Exception as e:
                logger.warning("price for %s failed: %s", sym, e); prices[sym] = 0.0
        if not live:      # DRY planning outside market hours: fall back to the last daily close (never done for LIVE orders)
            from scripts.coretrend_paper import fetch_daily
            for sym in list(prices):
                if prices[sym] <= 0:
                    try:
                        prices[sym] = fetch_daily(sym, start=date.fromordinal(today.toordinal() - 10))[-1][1]
                    except Exception as e:
                        logger.warning("last-close fallback for %s failed: %s", sym, e)
        actions = ex.plan(sig["state"], holdings, prices, cash, netliq)
        logger.info("holdings=%s cash=%.2f netliq=%.2f prices=%s -> plan=%s", holdings, cash, netliq, prices, actions)
        if not actions:
            want = ex.want_instrument(sig["state"])
            if live and prices.get(want, 0) <= 0 and not holdings.get(want):
                # a silent "nothing to do" because the price feed failed would look exactly like a healthy no-op
                notify.notify_error("CoreTrend", f"no price for {want} at run time — NOTHING was done (will retry next run)")
                return 1
            print("CoreTrend: nothing to do"); return 0
        for act in actions:
            sym, qty = act["symbol"], int(act["qty"])
            if act["action"] == "BUY":
                if sym in open_buys:
                    logger.info("open BUY already working for %s — skip", sym); continue
                px = prices[sym]; limit = round(px * 1.003, 2)
                if not live:
                    print(f"[DRY] BUY {qty} x {sym} @ limit {limit} + wide TRAIL {ex.PROTECT_TRAIL_PCT}%  ({act['why']})"); continue
                res = client.buy_with_bracket(sym, qty, ex.PROTECT_TRAIL_PCT, round(px * 3, 2), limit_price=limit)
                st = res["buy"].status if isinstance(res, dict) and "buy" in res else "?"
                sent = notify._send(f"🧭 <b>CoreTrend BUY {sym}</b> x{qty} @~${px:.2f} — {act['why']} (status {st}); wide {ex.PROTECT_TRAIL_PCT:.0f}% protective trail")
                logger.info("telegram BUY alert sent=%s", sent)
                if st in ("Error", "?"):
                    notify.notify_error("CoreTrend", f"BUY {sym} x{qty} failed/unfilled ({st}) — will retry next run")
            else:
                if not live:
                    print(f"[DRY] SELL_ALL {qty} x {sym} via TRAIL-modify  ({act['why']})"); continue
                res = client.force_exit_via_trail(sym, aggressive=True)
                sent = notify._send(f"🧭 <b>CoreTrend SELL {sym}</b> x{qty} — {act['why']} (status {res.status}); the {ex.RISK_INSTRUMENT if sym == ex.SAFE_INSTRUMENT else ex.SAFE_INSTRUMENT} buy follows on the next run after settlement")
                logger.info("telegram SELL alert sent=%s", sent)
                if res.status in ("Error",):
                    notify.notify_error("CoreTrend", f"exit of {sym} failed ({getattr(res, 'error', '')}) — CHECK MANUALLY")
        return 0
    finally:
        try: client.disconnect()
        except Exception: pass
        if lock is not None: _release_trade_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())
