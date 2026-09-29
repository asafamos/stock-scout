"""Exit profiles — the ATR-wide canary (2026-09-29, owner-approved, DRY-tested before live).

Why: on 1,780 random entries 2019-2026 (net of 0.5% cost, paired vs the legacy exit, 356 date
clusters) the legacy exit (trail 9% for 7 sessions then 5.5%, <=20 sessions) was the WORST tested
policy in every calendar year and every ATR-scaled alternative beat it (paired CI lower bound > 0):
  legacy            mean +0.44%/trade  WR 45%  avg 12.9d
  hold 20d          mean +1.56%        WR 56%
  ATR-4x trail 8-20% <=30 sessions  mean +1.89%  WR 50%  avg 23.9d   (vs legacy +1.45pp, CI lo +0.95)
A tight trail on a stock whose normal daily range is 3-5% fires on noise. Caveats: survivorship-biased
universe, bull-tilted sample, ratchet tiers/targets not modelled (they only cut winners further).
The same policy lives in core/trading/exit_sim.py as policy "CANARY" so the shadow logger keeps
measuring it forward.

Profile "atr_wide" (env TRADE_EXIT_PROFILE, default "legacy"):
  * initial trail % = clip(ATR_MULT x ATR%, TRAIL_MIN, TRAIL_MAX)   (ATR% from the scan; NB ATR_Pct is a
    FRACTION in the scan — the legacy code multiplied it as if it were percent, so its ATR term was inert)
  * time exit = MAX_HOLD_CAL_DAYS calendar days (30 sessions ~ 42 days), earnings-aware cap still applies
  * no time-tighten and no profit ratchet on these positions (they re-introduce the whipsaw)
  * wide stop => risk-based size: shares*price*trail% <= RISK_PCT_NETLIQ of NetLiq
"""
from __future__ import annotations

import math
import os
from typing import Optional

PROFILE_LEGACY = "legacy"
PROFILE_ATR_WIDE = "atr_wide"

ATR_MULT = 4.0
TRAIL_MIN = 8.0
TRAIL_MAX = 20.0
MAX_HOLD_SESSIONS = 30
MAX_HOLD_CAL_DAYS = 42
RISK_PCT_NETLIQ = 4.0     # max loss if the initial trail is hit, as % of NetLiq


def profile() -> str:
    p = os.getenv("TRADE_EXIT_PROFILE", PROFILE_LEGACY).strip().lower()
    return p if p in (PROFILE_LEGACY, PROFILE_ATR_WIDE) else PROFILE_LEGACY


def is_wide(pos: Optional[dict]) -> bool:
    """True for a tracked position that was opened under the atr_wide profile."""
    return bool(pos) and pos.get("exit_profile") == PROFILE_ATR_WIDE


def atr_percent(atr_pct: float) -> float:
    """Scan ATR_Pct is a fraction (0.034); tolerate percent input (3.4)."""
    if not atr_pct or not math.isfinite(atr_pct) or atr_pct <= 0:
        return 0.0
    return atr_pct * 100.0 if atr_pct < 1.0 else atr_pct


def wide_trail_pct(atr_pct: float) -> float:
    """Initial trailing % for the atr_wide profile. Unknown ATR -> the conservative maximum band mid."""
    a = atr_percent(atr_pct)
    if a <= 0:
        return round((TRAIL_MIN + TRAIL_MAX) / 2, 1)
    return round(min(TRAIL_MAX, max(TRAIL_MIN, ATR_MULT * a)), 1)


def risk_capped_qty(price: float, trail_pct: float, netliq: float, qty: int) -> int:
    """Shrink `qty` so a full initial-trail loss stays within RISK_PCT_NETLIQ of NetLiq."""
    if qty <= 0 or price <= 0 or trail_pct <= 0 or netliq <= 0:
        return max(0, qty)
    max_loss = netliq * RISK_PCT_NETLIQ / 100.0
    cap = int(max_loss // (price * trail_pct / 100.0))
    return max(0, min(qty, cap))
