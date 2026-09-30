"""Weekly scorecard: account vs SPY (the hurdle the bot must beat, net of costs).

Every Friday after the close (stockscout-weekly-vs-spy.timer) sends one Telegram message:
  * this week: account return vs SPY return and the excess in percentage points
  * since inception: same comparison, from TRADE_STARTING_CAPITAL at TRADE_INCEPTION_DATE
  * v2 sleeve (closed trades, cumulative $, health) and the performance-guard level
No IB connection: NetLiq comes from data/state/system_state.json (written by the state broadcaster).
Deposits/withdrawals distort the comparison — tell the bot with TRADE_EXTERNAL_FLOWS (net USD added since
inception, default 0) and with --flow USD on the day you move money.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE = ROOT / "data" / "state"
NAV_PATH = STATE / "weekly_nav.jsonl"


def pct(a: float, b: float) -> float:
    return (a / b - 1.0) * 100.0 if b else float("nan")


def compare(nav_now: float, nav_then: float, flows: float, spy_now: float, spy_then: float) -> Dict[str, float]:
    """Account return net of external flows vs SPY over the same span."""
    acct = pct(nav_now - flows, nav_then)
    spy = pct(spy_now, spy_then)
    return {"account_pct": acct, "spy_pct": spy, "excess_pp": acct - spy}


def read_nav() -> List[Dict]:
    if not NAV_PATH.exists():
        return []
    out = []
    for line in NAV_PATH.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def current_netliq() -> Optional[float]:
    try:
        return float(json.loads((STATE / "system_state.json").read_text())["net_liquidation"])
    except Exception:
        return None


def spy_close_on(day: date) -> Optional[float]:
    """SPY close on or before `day` (FMP stable EOD endpoint)."""
    key = os.getenv("FMP_API_KEY")
    if not key:
        return None
    frm = (day - timedelta(days=10)).isoformat()
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol=SPY&from={frm}&to={day.isoformat()}&apikey={key}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            rows = json.loads(r.read())
        rows = sorted(rows, key=lambda x: x["date"])
        return float(rows[-1]["price"]) if rows else None
    except Exception:
        return None


def build_message(today: date, nav_now: float, prev: Optional[Dict], flows_week: float) -> str:
    start_cap = float(os.getenv("TRADE_STARTING_CAPITAL", "977.5"))
    inception = date.fromisoformat(os.getenv("TRADE_INCEPTION_DATE", "2026-04-12"))
    flows_total = float(os.getenv("TRADE_EXTERNAL_FLOWS", "0"))
    spy_now = spy_close_on(today)
    lines = [f"📊 <b>Weekly scorecard</b> — {today.isoformat()}", f"  NetLiq ${nav_now:,.2f}"]
    if spy_now:
        spy0 = spy_close_on(inception)
        if spy0:
            c = compare(nav_now, start_cap, flows_total, spy_now, spy0)
            lines.append(f"  Since {inception}: account {c['account_pct']:+.1f}% vs SPY {c['spy_pct']:+.1f}%  → excess {c['excess_pp']:+.1f}pp")
        if prev:
            sp = spy_close_on(date.fromisoformat(prev["date"]))
            if sp:
                c = compare(nav_now, float(prev["netliq"]), flows_week, spy_now, sp)
                lines.append(f"  Week: account {c['account_pct']:+.1f}% vs SPY {c['spy_pct']:+.1f}%  → excess {c['excess_pp']:+.1f}pp")
    else:
        lines.append("  (SPY price unavailable — comparison skipped)")
    try:
        from core.trading import ledger, v2_selector as v2
        from core.trading.config import CONFIG
        h = v2.sleeve_health(ledger.closed_round_trips(CONFIG))
        state = "STOPPED: " + (v2.disabled_reason() or "") if v2.disabled_reason() else ("ON" if v2.enabled() else "off")
        lines.append(f"  v2 sleeve {state}: {h['n']} closed, cum ${h['cum_usd']:+.0f}, mean {h['mean_pct']:+.2f}%/trade")
    except Exception as e:
        lines.append(f"  (sleeve stats unavailable: {e!r})")
    try:
        from core.trading import performance_guard as pg
        from core.trading.config import CONFIG
        res = pg.assess(pg.returns_from_ledger(CONFIG))
        lines.append(f"  perf guard: {res['level']} (n={res['n']})")
    except Exception:
        pass
    lines.append("  Hurdle: beat SPY net of costs. Shadow verdict ≈ mid-Jan 2027.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", type=float, default=0.0, help="net USD deposited this week (negative = withdrawal)")
    ap.add_argument("--no-send", action="store_true")
    a = ap.parse_args(argv)
    nav = current_netliq()
    if nav is None:
        print("no NetLiq in system_state.json")
        return 1
    today = datetime.now(timezone.utc).date()
    hist = read_nav()
    prev = hist[-1] if hist else None
    msg = build_message(today, nav, prev, a.flow)
    print(msg)
    STATE.mkdir(parents=True, exist_ok=True)
    with open(NAV_PATH, "a") as f:
        f.write(json.dumps({"date": today.isoformat(), "netliq": nav, "flow": a.flow}) + "\n")
    if not a.no_send:
        from core.trading import notifications as notify
        notify._send(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
