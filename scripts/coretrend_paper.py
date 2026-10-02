"""CoreTrend PAPER tracker (no orders). Computes the current CoreTrend decision and a paper NAV vs SPY since inception.

usage:
    python -m scripts.coretrend_paper                  # print status
    python -m scripts.coretrend_paper --alert-if-last-day   # Telegram ONLY on the last trading day of a month
Data: FMP daily closes (QQQ, IEF, SPY). Paper NAV is recomputed from scratch each run (pure function of prices),
so it can never drift. Starts at TRADE_STARTING NAV 821 on INCEPTION and holds what the rule said at the previous
month-end; switching costs 0.05% per switch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.trading import coretrend as ct  # noqa: E402

INCEPTION = date(2026, 10, 2)
START_NAV = 821.0
SWITCH_COST = 0.0005
STATE = ROOT / "data" / "state" / "coretrend_paper.json"


def fetch_daily(symbol: str, start: date = date(2004, 1, 1)) -> List[Tuple[date, float]]:
    key = os.getenv("FMP_API_KEY")
    if not key:
        raise RuntimeError("FMP_API_KEY missing")
    url = (f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol={symbol}"
           f"&from={start.isoformat()}&to={date.today().isoformat()}&apikey={key}")
    with urllib.request.urlopen(url, timeout=40) as r:
        rows = json.loads(r.read())
    out = sorted((date.fromisoformat(x["date"]), float(x["price"])) for x in rows if x.get("price"))
    return out


def decision_dates(daily: List[Tuple[date, float]]) -> List[date]:
    """Last trading day of each month present in `daily`."""
    last: Dict[tuple, date] = {}
    for d, _ in daily:
        last[(d.year, d.month)] = d
    return sorted(last.values())


def build_track(qqq, ief, spy, inception: date = INCEPTION, start_nav: float = START_NAV) -> dict:
    """Pure function: paper NAV of CoreTrend vs SPY since `inception`, plus the live decision."""
    q = dict(qqq); i = dict(ief); s = dict(spy)
    q_days = [d for d, _ in qqq]
    mends = decision_dates(qqq)
    # decision at each month-end (only months that are COMPLETE as of the last data day)
    last_day = q_days[-1]
    decisions = []
    for me in mends:
        if me == last_day and not ct.is_last_trading_day_of_month(me):
            continue                                   # current month still in progress
        closes = ct.month_end_closes([(d, c) for d, c in qqq if d <= me])
        sg = ct.signal(closes)
        if sg:
            decisions.append((me, sg))
    if not decisions:
        return {"error": "not enough history"}
    # holding schedule: decision at month-end me applies from the next trading day
    nav = start_nav; hold = None; started = False; spy0 = None; curve = []
    sched = {me: sg["target"] for me, sg in decisions}
    prev_day = None
    for d in q_days:
        if d < inception:
            if d in sched: hold_candidate = sched[d]
            prev_day = d; continue
        if not started:
            # asset held at inception = decision of the last month-end BEFORE inception
            prior = [me for me, _ in decisions if me < inception]
            hold = sched[prior[-1]] if prior else ct.SAFE_ASSET
            started = True; spy0 = s.get(prev_day) or s.get(d); base_prev = prev_day
        # earn today's close-to-close return on the asset held since yesterday
        px = q if hold == ct.RISK_ASSET else i
        if prev_day in px and d in px:
            nav *= px[d] / px[prev_day]
        # month-end decision made at today's close -> switch for tomorrow
        if d in sched and sched[d] != hold:
            nav *= (1 - SWITCH_COST); hold = sched[d]
        curve.append((d, nav, hold))
        prev_day = d
    spy_ret = (s[curve[-1][0]] / spy0 - 1) * 100 if spy0 and curve and curve[-1][0] in s else None
    cur = decisions[-1][1]
    return {"decision_date": decisions[-1][0].isoformat(), "state": cur["state"], "target": cur["target"],
            "qqq_close": cur["close"], "sma10m": cur["sma"], "gap_pct": cur["gap_pct"],
            "paper_nav": curve[-1][1] if curve else start_nav, "paper_ret_pct": (curve[-1][1] / start_nav - 1) * 100 if curve else 0.0,
            "spy_ret_pct": spy_ret, "holding_now": curve[-1][2] if curve else hold, "days": len(curve)}


def message(t: dict, last_day_alert: bool) -> str:
    head = "📆 <b>CoreTrend month-end decision</b>" if last_day_alert else "🧭 <b>CoreTrend (paper)</b>"
    lines = [head,
             f"  Signal {t['state']} → hold <b>{t['target']}</b>   (QQQ ${t['qqq_close']:.2f} vs 10m SMA ${t['sma10m']:.2f}, {t['gap_pct']:+.1f}%)",
             f"  Paper NAV since {INCEPTION}: ${t['paper_nav']:,.2f} ({t['paper_ret_pct']:+.1f}%)"
             + (f" vs SPY {t['spy_ret_pct']:+.1f}%" if t.get("spy_ret_pct") is not None else ""),
             "  Paper only — no orders are placed. Evidence is moderate (20 strategies tried, QQQ = mega-cap growth bet)."]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alert-if-last-day", action="store_true")
    a = ap.parse_args(argv)
    today = datetime.now(timezone.utc).date()
    if a.alert_if_last_day and not ct.is_last_trading_day_of_month(today):
        print("not the last trading day of the month — nothing to do"); return 0
    t = build_track(fetch_daily("QQQ"), fetch_daily("IEF"), fetch_daily("SPY"))
    if "error" in t:
        print(t["error"]); return 1
    msg = message(t, a.alert_if_last_day); print(msg)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **t}))
    if a.alert_if_last_day:
        from core.trading import notifications as notify
        notify._send(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
