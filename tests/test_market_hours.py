"""NYSE session clock must follow DST (2026-09-29 audit: hard-coded 13:30-20:00 UTC)."""
from datetime import datetime, timezone

from core.trading.market_hours import is_regular_session as rs


def u(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def test_summer_edt_window():
    assert not rs(u(2026, 10, 14, 13, 29))
    assert rs(u(2026, 10, 14, 13, 30))
    assert rs(u(2026, 10, 14, 20, 0))
    assert not rs(u(2026, 10, 14, 20, 1))


def test_winter_est_window_last_hour_is_open():
    # After 2026-11-01 the session is 14:30-21:00 UTC
    assert not rs(u(2026, 11, 3, 14, 29))
    assert rs(u(2026, 11, 3, 14, 30))
    assert rs(u(2026, 11, 3, 20, 30))   # old code: closed
    assert rs(u(2026, 11, 3, 21, 0))
    assert not rs(u(2026, 11, 3, 21, 1))
    assert not rs(u(2026, 11, 3, 13, 45))  # old code: open pre-market


def test_dst_boundary_days():
    assert rs(u(2026, 10, 30, 20, 30)) is False   # EDT Friday: closed after 20:00 UTC
    assert rs(u(2026, 11, 2, 20, 30)) is True     # EST Monday


def test_holidays_and_weekends_and_early_close():
    assert not rs(u(2026, 11, 26, 16))   # Thanksgiving
    assert not rs(u(2026, 6, 19, 16))    # Juneteenth
    assert not rs(u(2026, 10, 17, 16))   # Saturday
    assert rs(u(2026, 11, 27, 17, 30))   # early close day, 12:30 ET
    assert not rs(u(2026, 11, 27, 18, 30))  # 13:30 ET, after 13:00 close
    assert not rs(u(2027, 1, 18, 16))    # MLK 2027


def test_scripts_calendar_shares_table():
    from datetime import date
    from scripts.market_calendar import is_market_open
    assert is_market_open(date(2026, 6, 19)) is False


def test_close_window_follows_dst_and_early_close():
    from core.trading.market_hours import in_close_window as w
    assert w(u(2026, 10, 14, 19, 45)) and not w(u(2026, 10, 14, 19, 15))     # EDT
    assert w(u(2026, 11, 3, 20, 45)) and not w(u(2026, 11, 3, 19, 45))       # EST
    assert w(u(2026, 11, 27, 17, 45)) and not w(u(2026, 11, 27, 19, 45))     # early close 13:00 ET
    assert not w(u(2026, 11, 3, 22, 0))                                      # after close


def test_minutes_after_close_dst_aware():
    from core.trading.market_hours import minutes_after_close as m
    assert m(u(2026, 10, 14, 20, 5)) == 5      # EDT close 20:00 UTC
    assert m(u(2026, 11, 3, 21, 5)) == 5       # EST close 21:00 UTC
    assert m(u(2026, 11, 3, 20, 5)) is None    # still open
    assert m(u(2026, 10, 17, 20, 5)) is None   # Saturday


def test_last_completed_session_skips_weekends_and_holidays():
    from datetime import date
    from core.trading.market_hours import last_completed_session as l
    assert l(date(2026, 9, 30)) == date(2026, 9, 29)    # Wed -> Tue
    assert l(date(2026, 9, 28)) == date(2026, 9, 25)    # Mon -> Fri
    assert l(date(2026, 9, 8)) == date(2026, 9, 4)      # Tue after Labor Day (Mon 9/7) -> Fri
    assert l(date(2026, 11, 27)) == date(2026, 11, 25)  # day after Thanksgiving -> Wed
