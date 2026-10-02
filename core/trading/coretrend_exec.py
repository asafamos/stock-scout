"""CoreTrend execution planning (pure, unit-tested). The runner (scripts/run_coretrend.py) does the I/O.

Holds ONE instrument at a time: the growth ETF while the trend rule says RISK_ON, the bond ETF otherwise.
Instruments (whole shares only; sub-$2k account): RISK = QQQM (same Nasdaq-100 index as QQQ at ~40% of the share
price, so it fits an ~$800 account in 2+ shares), SAFE = IEF (the backtest's bond leg).

One phase per run: if the WRONG instrument is held, only SELL it (sub-$2k accounts reject standalone sells, so the
sale is done by tightening the position's protective TRAIL — IBKRClient.force_exit_via_trail) and BUY the right one on
a LATER run, after the proceeds have settled (cash-account good-faith rules). A run that finds the right instrument
already held does nothing (idempotent, safe to run every day).
"""
from __future__ import annotations

import math
import os
from typing import Dict, List

RISK_SIGNAL_ASSET = "QQQ"        # the series the rule is computed on
RISK_INSTRUMENT = os.getenv("TRADE_CORETREND_RISK_INSTRUMENT", "QQQM")
SAFE_INSTRUMENT = os.getenv("TRADE_CORETREND_SAFE_INSTRUMENT", "IEF")
ALLOC_PCT = float(os.getenv("TRADE_CORETREND_ALLOC_PCT", "100"))     # % of NetLiq the strategy may hold
RESERVE_USD = float(os.getenv("TRADE_CORETREND_RESERVE_USD", "25"))  # cash always left untouched
PROTECT_TRAIL_PCT = 30.0          # wide emergency trail placed at buy (also the handle used to exit)


def want_instrument(state: str) -> str:
    return RISK_INSTRUMENT if state == "RISK_ON" else SAFE_INSTRUMENT


def plan(state: str, holdings: Dict[str, float], prices: Dict[str, float], cash: float, netliq: float,
         alloc_pct: float = None, reserve_usd: float = None, risk_inst: str = None, safe_inst: str = None) -> List[dict]:
    alloc_pct = ALLOC_PCT if alloc_pct is None else alloc_pct
    reserve_usd = RESERVE_USD if reserve_usd is None else reserve_usd
    risk_inst = risk_inst or RISK_INSTRUMENT
    safe_inst = safe_inst or SAFE_INSTRUMENT
    want = risk_inst if state == "RISK_ON" else safe_inst
    other = safe_inst if want == risk_inst else risk_inst
    q_other = int(holdings.get(other, 0) or 0)
    if q_other > 0:                                   # phase 1: get out of the wrong instrument, buy later
        return [{"action": "SELL_ALL", "symbol": other, "qty": q_other, "why": f"signal {state}: leave {other}"}]
    px = float(prices.get(want) or 0)
    if px <= 0:
        return []
    held_val = int(holdings.get(want, 0) or 0) * px
    budget = min(alloc_pct / 100.0 * netliq - held_val, cash - reserve_usd)
    qty = int(math.floor(budget / px)) if budget > 0 else 0
    if qty < 1:
        return []
    return [{"action": "BUY", "symbol": want, "qty": qty, "price": px, "why": f"signal {state}: hold {want}"}]
