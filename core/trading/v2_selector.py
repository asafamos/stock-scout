"""v2 sleeve selector — rule S3_v1 (2026-09-29, owner-approved; default OFF via TRADE_V2_SLEEVE).

Why: the only thing that beat SPY in the 2019-2026 offline tests was a volatility + small-size tilt
(top-3/week with the canary exit: +3.7% excess vs SPY, t 2.1 — survivorship-inflated, swings from
-4.7% in 2021 to +12.7% in 2025). Fundamentals / momentum / Score / ML added nothing measurable.
S3_v1 = that tilt restricted to names a $300 order can actually trade.

Rule (frozen; a change = a new rule id): universe = scan rows with Close >= 5, ATR_Pct > 0,
market_cap > 0, ADDV (avg volume x close) >= $5M, sector not blocked. Score = pct_rank(ATR_Pct) +
pct_rank(-market_cap) inside that universe. Ties by ticker asc. The shadow logger records the same
rule as `s3_rank` (scripts/shadow_log.py) so what trades is exactly what is measured forward.

The sleeve holds at most V2_MAX_POSITIONS concurrent positions, sized by the atr_wide risk cap, and
switches itself OFF (needs a manual re-enable) if its own closed trades go bad — see sleeve_health().
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

MIN_PRICE = 5.0
MIN_ADDV_USD = 5_000_000.0
V2_MAX_POSITIONS = 1
MAX_GAP_PCT = 6.0          # skip an entry that opens more than this far from the signal close
# 2026-09-30 (owner-approved loosening): the original $60 / 10-close stop would have fired in ~25% of
# historical start dates (median after 5 trades) and cut off the rare large winners that carry this
# profile (best 6 trades = 63% of backtest profit). $100 is ~12% of NetLiq; the mean test needs 15 closes.
KILL_MIN_CLOSES = 15
KILL_MEAN_PCT = -2.0       # mean net % per closed trade after KILL_MIN_CLOSES
KILL_CUM_LOSS_USD = 100.0  # cumulative realized loss that stops the sleeve at any n >= 3

STATE_DIR = Path(os.getenv("TRADE_STATE_DIR", "data/state"))


def enabled() -> bool:
    return os.getenv("TRADE_V2_SLEEVE", "0").strip() in ("1", "true", "True")


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def pct_rank(vals: List[float]) -> List[float]:
    """Percentile ranks in (0, 1]; equal values share the AVERAGE rank (so ties do not depend on order)."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    rk = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = ((i + 1) + (j + 1)) / 2.0 / n
        for k in range(i, j + 1):
            rk[order[k]] = avg
        i = j + 1
    return rk


def select_s3(rows: Iterable[Dict], blocked_sectors: set, top_n: int = 3) -> Dict[str, int]:
    """rows: dicts with ticker, close, atr_pct, market_cap, vol_avg, sector. Returns {ticker: rank}."""
    uni = []
    for r in rows:
        close, atr, mc, va = _f(r.get("close")), _f(r.get("atr_pct")), _f(r.get("market_cap")), _f(r.get("vol_avg"))
        if not close or close < MIN_PRICE or not atr or atr <= 0 or not mc or mc <= 0 or not va:
            continue
        if va * close < MIN_ADDV_USD:
            continue
        if (r.get("sector") or "") in blocked_sectors:
            continue
        uni.append((r["ticker"], atr, mc))
    if not uni:
        return {}
    a = pct_rank([u[1] for u in uni])
    m = pct_rank([-u[2] for u in uni])
    order = sorted(range(len(uni)), key=lambda i: (-(a[i] + m[i]), uni[i][0]))
    return {uni[i][0]: k + 1 for k, i in enumerate(order[:top_n])}


def rows_from_scan(df) -> List[Dict]:
    """Normalise a scan DataFrame into the dict shape select_s3 expects."""
    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None
    c = {k: col(*v) for k, v in {
        "ticker": ("Ticker", "ticker"), "close": ("Close", "close"), "atr_pct": ("ATR_Pct", "atr_pct"),
        "market_cap": ("market_cap", "Market_Cap"), "vol_avg": ("vol_avg", "Vol_Avg"),
        "sector": ("Sector", "sector")}.items()}
    if not c["ticker"]:
        return []
    out = []
    for _, r in df.iterrows():
        out.append({k: (r[v] if v else None) for k, v in c.items()})
    return out


# ── sleeve bookkeeping + self-kill ────────────────────────────────────────────
def _entries_path() -> Path:
    return STATE_DIR / "v2_sleeve_entries.jsonl"


def _disabled_path() -> Path:
    return STATE_DIR / "v2_sleeve_disabled.json"


def record_entry(ticker: str, qty: int, price: float, ts: Optional[str] = None) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_entries_path(), "a") as f:
        f.write(json.dumps({"ticker": ticker, "qty": qty, "price": price,
                            "ts": ts or datetime.now(timezone.utc).isoformat()}) + "\n")


def disabled_reason() -> Optional[str]:
    p = _disabled_path()
    if p.exists():
        try:
            return json.loads(p.read_text()).get("reason", "disabled")
        except Exception:
            return "disabled"
    return None


def sleeve_trips(round_trips: List[dict]) -> List[dict]:
    """Closed ledger round trips that belong to the sleeve (ticker bought by the sleeve, exited after)."""
    p = _entries_path()
    if not p.exists():
        return []
    entries = []
    for line in p.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    used, out = set(), []
    for t in sorted(round_trips, key=lambda x: str(x.get("exit_time") or "")):
        for i, e in enumerate(entries):
            if i in used or e["ticker"] != t.get("ticker"):
                continue
            if str(t.get("exit_time") or "") >= e["ts"]:
                used.add(i)
                out.append(t)
                break
    return out


def sleeve_health(round_trips: List[dict]) -> Dict:
    """Decide whether the sleeve must stop. Returns {ok, n, mean_pct, cum_usd, reason}."""
    trips = sleeve_trips(round_trips)
    n = len(trips)
    cum = sum(float(t.get("realized_pnl") or 0.0) for t in trips)
    pcts = []
    for t in trips:
        cost = float(t.get("entry_price") or 0) * float(t.get("shares") or 0)
        if cost > 0:
            pcts.append(float(t["realized_pnl"]) / cost * 100.0)
    mean = sum(pcts) / len(pcts) if pcts else 0.0
    if n >= 3 and cum <= -KILL_CUM_LOSS_USD:
        return {"ok": False, "n": n, "mean_pct": mean, "cum_usd": cum,
                "reason": f"cumulative sleeve loss ${cum:.0f} <= -${KILL_CUM_LOSS_USD:.0f}"}
    if n >= KILL_MIN_CLOSES and mean < KILL_MEAN_PCT:
        return {"ok": False, "n": n, "mean_pct": mean, "cum_usd": cum,
                "reason": f"mean {mean:+.2f}%/trade over {n} closes < {KILL_MEAN_PCT}%"}
    return {"ok": True, "n": n, "mean_pct": mean, "cum_usd": cum, "reason": ""}


def disable(reason: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _disabled_path().write_text(json.dumps({"reason": reason, "at": datetime.now(timezone.utc).isoformat()}))
