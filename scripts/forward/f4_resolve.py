"""F4 resolver — COUNTS ONLY until the pre-registered stopping rule is met (prereg docs/research_prereg_F4_earnings_text.md: >= 1,000 resolved events across >= 30 entry dates).
No return statistic is computed or printed here before that. The evaluator is written and unit-tested before the rule can be met (tracker: due 2026-12-01).
  python scripts/forward/f4_resolve.py
"""
import json
from datetime import date, timedelta
from pathlib import Path

from scripts.forward.f4_collect import HOLIDAYS, data_dir

MIN_EVENTS, MIN_DATES, HORIZON = 1000, 30, 20


def trading_days_after(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in HOLIDAYS:
            n -= 1
    return d


def main(today: date = None) -> dict:
    today = today or date.today()
    scored = resolved = 0
    dates = set()
    for p in sorted((data_dir() / "batches").glob("*_scores.json")):
        d = json.load(open(p))
        ed = date.fromisoformat(d["entry_date"])
        scored += d["n"]
        if d["n"] and trading_days_after(ed, HORIZON) <= today:
            resolved += d["n"]
            dates.add(ed)
    met = resolved >= MIN_EVENTS and len(dates) >= MIN_DATES
    out = {"events_scored": scored, "events_resolved_by_calendar": resolved, "resolved_entry_dates": len(dates), "stopping_rule_met": met,
           "note": "counts only; no return statistic is computed before the stopping rule is met"}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
