"""CoreTrend — a low-turnover trend rule on ONE growth ETF with a bond fallback (2026-10-02).

Rule (decided on the LAST trading day of each month, from month-end closes):
    RISK_ON  : month-end close of RISK_ASSET  >  its 10-month simple moving average of month-end closes  -> hold RISK_ASSET
    RISK_OFF : otherwise                                                                                  -> hold SAFE_ASSET
Evidence (scripts/research/beat_spy_search.py, ETF data, 0.05%/side): QQQ 10m-trend -> IEF beat SPY in BOTH 2005-15
(8.1% vs 7.0%) and 2016-26 (15.2% vs 14.8%, maxDD -22.6% vs -23.9%), ~2-3 trades/year. Caveats: 20 strategies were
tried, QQQ is a bet on mega-cap growth, the edge over SPY is small and may be noise. This module only DECIDES; the
runner (scripts/coretrend_paper.py) records a PAPER track and alerts. It never places orders.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Sequence

RISK_ASSET = "QQQ"
SAFE_ASSET = "IEF"
SMA_MONTHS = 10
RISK_ON, RISK_OFF = "RISK_ON", "RISK_OFF"


def signal(month_end_closes: Sequence[float], sma_months: int = SMA_MONTHS) -> Optional[dict]:
    """month_end_closes: chronological month-end closes of the risk asset (the latest one LAST)."""
    xs = [float(x) for x in month_end_closes if x is not None]
    if len(xs) < sma_months:
        return None
    sma = sum(xs[-sma_months:]) / sma_months
    last = xs[-1]
    state = RISK_ON if last > sma else RISK_OFF
    return {"state": state, "close": last, "sma": sma, "gap_pct": (last / sma - 1.0) * 100.0,
            "target": RISK_ASSET if state == RISK_ON else SAFE_ASSET}


def month_end_closes(daily: List[tuple]) -> List[float]:
    """daily: [(date, close), ...] ascending. Returns the last close of each calendar month (incl. the current,
    possibly incomplete, month — callers decide whether to use it, see is_last_trading_day_of_month)."""
    out, cur_key, cur_close = [], None, None
    for d, c in daily:
        key = (d.year, d.month)
        if cur_key is not None and key != cur_key:
            out.append(cur_close)
        cur_key, cur_close = key, c
    if cur_key is not None:
        out.append(cur_close)
    return out


def is_last_trading_day_of_month(d: date) -> bool:
    from core.trading.market_hours import is_trading_day
    if not is_trading_day(d):
        return False
    nxt = date.fromordinal(d.toordinal() + 1)
    while not is_trading_day(nxt):
        nxt = date.fromordinal(nxt.toordinal() + 1)
    return nxt.month != d.month
