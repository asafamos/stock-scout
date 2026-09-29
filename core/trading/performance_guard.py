"""Performance guard — replaces the old rolling-window "throttle" (2026-09-29, owner-approved redesign).

What was wrong with the throttle (measured on 402 real closed positions, Mar-May 2026, sd 7.9%/trade):
  * 10-trade window => standard error 2.5% vs a -1.5% halt line: a strategy with a TRUE +1.06%/trade
    edge trips the halt in ~15% of windows purely by chance (34% trip the "warn" line);
  * decision-time replay (window = closes known at each entry): prev-10 expectancy has ~0 predictive
    power (corr -0.04) — trades entered in HALT state actually averaged +2.9%; replaying the rule
    cost 34% of total return while trimming max drawdown only 17%;
  * a halt had no way out: no trades => the window never changes => permanent lock (100% of
    shuffled orderings) — the same trap as the drawdown breaker;
  * it read trade_log CLOSE rows, which the ledger mode does not write for trail/limit closes.

Design now:
  * evidence, not streaks: window of the last 30 closed trades, needs >= 20; per-trade net % returns;
  * "degraded" only when the one-sided 95% upper bound of the mean is below zero
    (mean + 1.645*SE < 0) — i.e. we are confident the edge is negative, not just unlucky;
  * NEVER halts. Modes (env TRADE_PERF_GUARD_MODE): "alert" (default: Telegram on level change, no
    effect on orders), "size" (degraded => half size; recovers as the window rolls, and trades keep
    happening so the window does update), "off";
  * drawdown control stays with the 2-stage drawdown breaker (risk_manager.check_drawdown_breaker).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

WINDOW = int(os.getenv("TRADE_PERF_GUARD_WINDOW", "30"))
MIN_N = int(os.getenv("TRADE_PERF_GUARD_MIN_N", "20"))
Z95 = 1.645
STATE_PATH = Path(os.getenv("TRADE_PERF_GUARD_STATE", "data/state/perf_guard.json"))


def mode() -> str:
    m = os.getenv("TRADE_PERF_GUARD_MODE", "alert").strip().lower()
    return m if m in ("alert", "size", "off") else "alert"


def returns_from_ledger(cfg=None) -> List[float]:
    """Net % return per closed round trip, chronological (broker realized P&L / cost basis)."""
    from core.trading import ledger
    trips = sorted(
        (t for t in ledger.closed_round_trips(cfg) if t.get("realized_pnl") is not None),
        key=lambda t: str(t.get("exit_time") or ""),
    )
    out = []
    for t in trips:
        cost = float(t.get("entry_price") or 0) * float(t.get("shares") or 0)
        if cost > 0:
            out.append(float(t["realized_pnl"]) / cost * 100.0)
    return out


def returns_from_trade_log(log: List[dict]) -> List[float]:
    out = []
    for t in log:
        if t.get("action") == "CLOSE" and t.get("pnl") is not None:
            cost = float(t.get("entry_price") or 0) * float(t.get("quantity") or 0)
            out.append(float(t["pnl"]) / cost * 100.0 if cost > 0 else float(t["pnl"]) / 300.0 * 100.0)
    return out


def assess(returns: List[float], window: int = WINDOW, min_n: int = MIN_N) -> Dict:
    r = [x for x in returns[-window:] if x is not None and math.isfinite(x)]
    n = len(r)
    if n < min_n:
        return {"level": "insufficient", "n": n, "mean": None, "se": None, "upper95": None}
    mean = sum(r) / n
    var = sum((x - mean) ** 2 for x in r) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    upper = mean + Z95 * se
    if upper < 0:
        level = "degraded"     # confident the per-trade edge is negative
    elif mean < 0:
        level = "watch"        # negative but within noise — informational only
    else:
        level = "ok"
    return {"level": level, "n": n, "mean": mean, "se": se, "upper95": upper}


def size_multiplier(level: str, guard_mode: Optional[str] = None) -> float:
    guard_mode = guard_mode or mode()
    return 0.5 if (guard_mode == "size" and level == "degraded") else 1.0


def maybe_alert(res: Dict, state_path: Path = STATE_PATH, notifier=None) -> bool:
    """Telegram once per level CHANGE (never per cycle). Returns True if an alert was sent."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            prev = json.loads(state_path.read_text()).get("level")
        except Exception:
            prev = None
        cur = res["level"]
        if cur == prev:
            return False
        state_path.write_text(json.dumps({"level": cur, "since": time.time(), "n": res.get("n"),
                                          "mean": res.get("mean")}))
        if cur in ("degraded", "watch") or (prev in ("degraded", "watch") and cur == "ok"):
            if notifier is None:
                from core.trading import notifications as notifier
            emoji = {"degraded": "🔻", "watch": "👀", "ok": "✅"}[cur]
            notifier.notify_error(
                "Performance guard",
                f"{emoji} {cur.upper()}: last {res['n']} closes avg {res['mean']:+.2f}%/trade "
                f"(SE {res['se']:.2f}, 95% upper bound {res['upper95']:+.2f}%). "
                f"Mode={mode()} — "
                + ("orders unaffected (alert mode)." if mode() != "size" else "sizing halved while degraded.")
            )
            return True
    except Exception as e:
        logger.warning("performance guard alert failed: %s", e)
    return False
