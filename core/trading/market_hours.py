"""NYSE regular-session clock (single source for the trading code).

2026-09-29 audit: ``IBKRClient.is_market_open`` hard-coded 13:30-20:00 UTC (EDT).
From the DST change (2026-11-01) the real session is 14:30-21:00 UTC, so the
monitor would have idled through the whole last trading hour (20:00-21:00 UTC)
and BUYs (risk_manager gates on market-open) would have been refused then.
The session is now evaluated in America/New_York, so DST is handled by the tz
database.

Holidays / early closes are static and must be extended each year. When the
current year is not covered the module logs a loud warning (once) rather than
silently treating a holiday as a trading day.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
OPEN_ET = time(9, 30)
CLOSE_ET = time(16, 0)
EARLY_CLOSE_ET = time(13, 0)

# NYSE full-day closures (observed dates).
HOLIDAYS = {
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}

# 13:00 ET closes.
EARLY_CLOSES = {
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26),
}

_COVERED_YEARS = {d.year for d in HOLIDAYS}
_warned_years: set = set()


def _warn_if_uncovered(year: int) -> None:
    if year not in _COVERED_YEARS and year not in _warned_years:
        _warned_years.add(year)
        logger.error(
            "market_hours: no NYSE holiday table for %d — holidays are NOT excluded. "
            "Extend core/trading/market_hours.py", year,
        )


def is_trading_day(d: date) -> bool:
    _warn_if_uncovered(d.year)
    return d.weekday() < 5 and d not in HOLIDAYS


def is_regular_session(now_utc: Optional[datetime] = None) -> bool:
    """True during the NYSE regular session (9:30-16:00 ET, 13:00 on early-close days)."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    et = now_utc.astimezone(NY)
    if not is_trading_day(et.date()):
        return False
    close = EARLY_CLOSE_ET if et.date() in EARLY_CLOSES else CLOSE_ET
    return OPEN_ET <= et.time() <= close


def minutes_to_close(now_utc: Optional[datetime] = None) -> Optional[float]:
    """Minutes until today's regular-session close, or None outside the session."""
    if not is_regular_session(now_utc):
        return None
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    et = now_utc.astimezone(NY)
    close = EARLY_CLOSE_ET if et.date() in EARLY_CLOSES else CLOSE_ET
    return (close.hour * 60 + close.minute) - (et.hour * 60 + et.minute) - et.second / 60.0


def in_close_window(now_utc: Optional[datetime] = None, minutes: int = 30) -> bool:
    """True in the last `minutes` of the session (DST-aware; early-close aware)."""
    m = minutes_to_close(now_utc)
    return m is not None and m <= minutes


def minutes_after_close(now_utc: Optional[datetime] = None) -> Optional[float]:
    """Minutes since today's regular-session close (trading days only), else None/negative-safe.

    Returns None on non-trading days or before the close.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    et = now_utc.astimezone(NY)
    if not is_trading_day(et.date()):
        return None
    close = EARLY_CLOSE_ET if et.date() in EARLY_CLOSES else CLOSE_ET
    m = (et.hour * 60 + et.minute) - (close.hour * 60 + close.minute)
    return float(m) if m > 0 else None


def last_completed_session(today: Optional[date] = None) -> date:
    """The most recent trading day strictly BEFORE `today` (ET date when omitted)."""
    if today is None:
        today = datetime.now(timezone.utc).astimezone(NY).date()
    d = today
    for _ in range(10):
        d = date.fromordinal(d.toordinal() - 1)
        if is_trading_day(d):
            return d
    return d
