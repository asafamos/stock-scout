"""Daily-bar exit simulator shared by the shadow resolver and tests.

Model (same as the 2026-09-29 exit study): enter at the OPEN of bar `e`; from the second bar on a
trailing stop = highest HIGH so far * (1 - pct/100) is checked against each bar's LOW (gap-down fills
at the open); otherwise exit at the CLOSE of the last allowed session. The peak is updated AFTER the
stop check, so a bar cannot both raise the peak and stop out on its own high. Returns are net of
`cost_pct`. `finished=False` means the data ended before the policy did (resolve later).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from core.trading import exit_profile as xp

POLICIES: Dict[str, dict] = {
    # what the bot does today
    "LEGACY": {"kind": "legacy", "max": 20},
    "HOLD20": {"kind": "hold", "max": 20},
    # the canary (must mirror core/trading/exit_profile.py)
    "CANARY": {"kind": "atrpct", "K": xp.ATR_MULT, "lo": xp.TRAIL_MIN, "hi": xp.TRAIL_MAX, "max": xp.MAX_HOLD_SESSIONS},
    # neighbours, so the report can show the canary is not a knife-edge choice
    "ATR4_60": {"kind": "atrpct", "K": 4.0, "lo": 8.0, "hi": 20.0, "max": 60},
    "ATR5_60": {"kind": "atrpct", "K": 5.0, "lo": 8.0, "hi": 25.0, "max": 60},
    "TRAIL15_60": {"kind": "pct", "p": 15.0, "max": 60},
    "HOLD60": {"kind": "hold", "max": 60},
}


def atr_pct_at(high, low, close, e: int, n: int = 14) -> float:
    """ATR(n) at bar `e` (uses bars before e) as % of the entry-bar open is done by the caller."""
    lo = max(1, e - n)
    trs = [max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1])) for j in range(lo, e)]
    return sum(trs) / len(trs) if trs else float("nan")


def _pct_for(policy: dict, k: int, atr_pct_entry: float) -> Optional[float]:
    kind = policy["kind"]
    if kind == "hold":
        return None
    if kind == "legacy":
        return 9.0 if k < 7 else 5.5
    if kind == "pct":
        return policy["p"]
    if kind == "atrpct":
        if not math.isfinite(atr_pct_entry) or atr_pct_entry <= 0:
            return None
        return min(policy["hi"], max(policy["lo"], policy["K"] * atr_pct_entry))
    raise ValueError(kind)


def simulate(o, h, l, c, e: int, policy: dict, cost_pct: float = 0.0) -> Optional[Dict]:
    """Simulate `policy` for a trade entered at o[e]. Arrays are plain sequences of floats."""
    n = len(o)
    if e < 0 or e >= n or not (o[e] > 0):
        return None
    entry = o[e]
    atr_e = atr_pct_at(h, l, c, e) / entry * 100.0 if policy["kind"] == "atrpct" and e >= 2 else float("nan")
    peak = entry
    for k in range(policy["max"]):
        i = e + k
        if i >= n:
            return {"finished": False}
        pct = _pct_for(policy, k, atr_e)
        if pct is not None and k > 0:
            stop = peak * (1 - pct / 100.0)
            if l[i] <= stop:
                px = o[i] if o[i] < stop else stop
                return {"finished": True, "ret_pct": (px / entry - 1) * 100 - cost_pct, "days": k + 1, "reason": "stop"}
        if k == policy["max"] - 1:
            return {"finished": True, "ret_pct": (c[i] / entry - 1) * 100 - cost_pct, "days": k + 1, "reason": "time"}
        peak = max(peak, h[i])
    return None
