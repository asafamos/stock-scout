"""Resolve shadow-logged scans into forward returns (see scripts/shadow_log.py).

Pre-registered measurement (docs/shadow_selector_prereg.md):
  entry  = OPEN of the first session AFTER the scan date        (no intraday-scan-price ambiguity)
  exit   = CLOSE of the 20th session counting the entry session (20 TRADING days, not calendar)
  bench  = SPY over the identical window
Only records with all 20 sessions available are resolved; others stay pending (never truncated).
Bars come from yfinance on today's share basis, so splits inside the window cancel out.
Gross returns only — costs are applied in the report, explicitly.

Usage: python -m scripts.shadow_resolve [--limit-tickers N] [--min-age-days 29]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("shadow_resolve")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "outcomes"
PICKS_PATH = OUT_DIR / "shadow_picks.jsonl"
OUTCOMES_PATH = OUT_DIR / "shadow_outcomes.jsonl"
EXIT_OUTCOMES_PATH = OUT_DIR / "shadow_exit_outcomes.jsonl"
HORIZON = 20


def _read(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def window_return(hist, scan_date: date, horizon: int = HORIZON) -> Optional[Dict]:
    """Entry-open -> horizon-th session close after `scan_date`, or None if the window is incomplete."""
    import pandas as pd
    if hist is None or len(hist) == 0:
        return None
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    bars = hist[idx.normalize() > pd.Timestamp(scan_date)].head(horizon)
    if len(bars) < horizon:
        return None
    entry = float(bars["Open"].iloc[0])
    exit_ = float(bars["Close"].iloc[-1])
    if not (entry > 0 and exit_ > 0):
        return None
    return {
        "entry_open": entry, "exit_close": exit_,
        "ret_pct": (exit_ / entry - 1) * 100,
        "max_ret_pct": (float(bars["High"].max()) / entry - 1) * 100,
        "min_ret_pct": (float(bars["Low"].min()) / entry - 1) * 100,
        "bars": len(bars),
    }


def _download(tickers: List[str], start: date, end: date) -> Dict[str, "object"]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        try:
            data = yf.download(chunk, start=start.isoformat(), end=end.isoformat(), interval="1d",
                               auto_adjust=False, group_by="ticker", progress=False, threads=False)
        except Exception as e:
            logger.warning("download failed for %s..: %s", chunk[:3], e)
            continue
        for t in chunk:
            try:
                sub = data[t] if len(chunk) > 1 else data
                sub = sub.dropna(subset=["Open", "Close"])
                if len(sub):
                    out[t] = sub
            except Exception:
                continue
    return out


def resolve(min_age_days: int = 29, limit_tickers: int = 400, picks_path: Path = PICKS_PATH,
            outcomes_path: Path = OUTCOMES_PATH, downloader=_download) -> int:
    picks = _read(picks_path)
    done = {(o["scan_date"], o["ticker"]) for o in _read(outcomes_path)}
    today = date.today()
    pending = [p for p in picks
               if (p["scan_date"], p["ticker"]) not in done
               and (today - date.fromisoformat(p["scan_date"])).days >= min_age_days]
    if not pending:
        logger.info("shadow_resolve: nothing matured")
        return 0
    tickers = sorted({p["ticker"] for p in pending})[:limit_tickers] + ["SPY"]
    tickers = list(dict.fromkeys(tickers))
    lo = min(date.fromisoformat(p["scan_date"]) for p in pending)
    hi = max(date.fromisoformat(p["scan_date"]) for p in pending)
    bars = downloader(tickers, lo, hi + timedelta(days=45))
    spy = bars.get("SPY")
    now = datetime.now(timezone.utc).isoformat()
    new = []
    for p in pending:
        h = bars.get(p["ticker"])
        sd = date.fromisoformat(p["scan_date"])
        r = window_return(h, sd)
        b = window_return(spy, sd)
        if r is None or b is None:
            continue
        new.append({"scan_date": p["scan_date"], "ticker": p["ticker"], **r,
                    "spy_ret_pct": b["ret_pct"], "resolved_at": now, "horizon": HORIZON})
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(outcomes_path, "a") as f:
        for o in new:
            f.write(json.dumps(o) + "\n")
    logger.info("shadow_resolve: %d pending, %d resolved", len(pending), len(new))
    return len(new)


def resolve_exits(min_age_days: int = 29, limit_tickers: int = 800, picks_path: Path = PICKS_PATH,
                  out_path: Path = EXIT_OUTCOMES_PATH, downloader=_download) -> int:
    """Simulate every exit policy in core/trading/exit_sim.POLICIES on every logged scan row.

    Entry = open of the first session after the scan date (same as the 20-session measurement). A
    (scan_date, ticker, policy) triple is written once the policy has FINISHED (stopped out, or its
    time limit reached); until then it stays pending. Gross returns; cost is applied in the report.
    """
    import numpy as np
    import pandas as pd
    from core.trading.exit_sim import POLICIES, simulate

    picks = _read(picks_path)
    done = {(o["scan_date"], o["ticker"], o["policy"]) for o in _read(out_path)}
    today = date.today()
    pend = []
    for p in picks:
        if (today - date.fromisoformat(p["scan_date"])).days < min_age_days:
            continue
        for pol in POLICIES:
            if (p["scan_date"], p["ticker"], pol) not in done:
                pend.append((p, pol))
    if not pend:
        logger.info("shadow_resolve_exits: nothing matured")
        return 0
    pend.sort(key=lambda x: (x[0]["scan_date"], x[0]["ticker"]))
    tickers = list(dict.fromkeys(p["ticker"] for p, _ in pend))[:limit_tickers]
    tset = set(tickers)
    pend = [(p, pol) for p, pol in pend if p["ticker"] in tset]
    lo = min(date.fromisoformat(p["scan_date"]) for p, _ in pend) - timedelta(days=45)
    hi = max(date.fromisoformat(p["scan_date"]) for p, _ in pend) + timedelta(days=120)
    bars = downloader(tickers, lo, min(hi, today + timedelta(days=1)))
    arr = {}
    now = datetime.now(timezone.utc).isoformat()
    new = []
    for p, pol in pend:
        h = bars.get(p["ticker"])
        if h is None or len(h) == 0:
            continue
        if p["ticker"] not in arr:
            idx = h.index.tz_localize(None) if getattr(h.index, "tz", None) is not None else h.index
            arr[p["ticker"]] = (idx.normalize(), *(h[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close")))
        idx, o, hi_, lo_, c = arr[p["ticker"]]
        e = int(idx.searchsorted(pd.Timestamp(p["scan_date"]), side="right"))
        r = simulate(o, hi_, lo_, c, e, POLICIES[pol])
        if r and r.get("finished"):
            new.append({"scan_date": p["scan_date"], "ticker": p["ticker"], "policy": pol,
                        "ret_pct": r["ret_pct"], "days": r["days"], "reason": r["reason"], "resolved_at": now})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for o in new:
            f.write(json.dumps(o) + "\n")
    logger.info("shadow_resolve_exits: %d pending triples, %d finished", len(pend), len(new))
    return len(new)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-age-days", type=int, default=29)
    ap.add_argument("--limit-tickers", type=int, default=400)
    a = ap.parse_args(argv)
    resolve(a.min_age_days, a.limit_tickers)
    resolve_exits(a.min_age_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
