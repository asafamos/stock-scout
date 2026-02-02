"""US Market Calendar Helper.

Provides utilities to check if the US stock market is open on a given date,
accounting for weekends and federal holidays observed by NYSE.

Usage:
    from scripts.market_calendar import is_market_open, get_market_status
    
    if is_market_open():
        run_scan()
    
    print(get_market_status())  # "OPEN", "CLOSED", "PRE-MARKET", "AFTER-HOURS"
"""
from datetime import date, datetime, time
from typing import Set

# US Market Holidays 2026 (update annually)
# Source: NYSE Holiday Schedule
US_MARKET_HOLIDAYS_2026: Set[date] = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # Martin Luther King Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),   # Independence Day (observed - July 4 is Saturday)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving Day
    date(2026, 12, 25), # Christmas Day
}

# Early close days (1:00 PM ET close instead of 4:00 PM)
US_MARKET_EARLY_CLOSE_2026: Set[date] = {
    date(2026, 11, 27), # Day after Thanksgiving
    date(2026, 12, 24), # Christmas Eve
}

# NYSE Regular Trading Hours (Eastern Time)
NYSE_OPEN_ET = time(9, 30)   # 9:30 AM ET
NYSE_CLOSE_ET = time(16, 0)  # 4:00 PM ET
NYSE_EARLY_CLOSE_ET = time(13, 0)  # 1:00 PM ET


def is_market_open(check_date: date = None) -> bool:
    """
    Check if US stock market is open on given date.
    
    Args:
        check_date: Date to check. Defaults to today.
        
    Returns:
        True if market is open, False if closed (weekend or holiday).
    """
    if check_date is None:
        check_date = date.today()

    # Weekend check (Monday=0, Sunday=6)
    if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Holiday check
    if check_date in US_MARKET_HOLIDAYS_2026:
        return False

    return True


def is_early_close(check_date: date = None) -> bool:
    """
    Check if market closes early on given date.
    
    Args:
        check_date: Date to check. Defaults to today.
        
    Returns:
        True if market closes at 1:00 PM ET instead of 4:00 PM.
    """
    if check_date is None:
        check_date = date.today()
    
    return check_date in US_MARKET_EARLY_CLOSE_2026


def get_market_status() -> str:
    """
    Get current market status string based on UTC time.
    
    Note: This uses UTC time and converts to approximate ET.
    For production, consider using a proper timezone library.
    
    Returns:
        One of: "CLOSED (holiday/weekend)", "PRE-MARKET", "OPEN", "AFTER-HOURS"
    """
    now = datetime.utcnow()
    today = now.date()

    if not is_market_open(today):
        return "CLOSED (holiday/weekend)"

    # Convert UTC to approximate ET (UTC-5 in winter, UTC-4 in summer)
    # This is a simplification - for production use pytz or zoneinfo
    utc_hour = now.hour
    utc_minute = now.minute
    
    # Approximate ET offset (simplified - assumes EST/UTC-5)
    # In reality, DST changes this to UTC-4 from March-November
    et_hour = utc_hour - 5
    if et_hour < 0:
        et_hour += 24
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open_minutes = 9 * 60 + 30  # 9:30 AM = 570 minutes
    market_close_minutes = 16 * 60     # 4:00 PM = 960 minutes
    
    if is_early_close(today):
        market_close_minutes = 13 * 60  # 1:00 PM = 780 minutes
    
    current_minutes = et_hour * 60 + utc_minute
    
    if current_minutes < market_open_minutes:
        return "PRE-MARKET"
    elif current_minutes < market_close_minutes:
        return "OPEN"
    else:
        return "AFTER-HOURS"


def get_next_market_day(from_date: date = None) -> date:
    """
    Get the next day the market is open.
    
    Args:
        from_date: Starting date. Defaults to today.
        
    Returns:
        Next date when market is open.
    """
    if from_date is None:
        from_date = date.today()
    
    from datetime import timedelta
    check_date = from_date + timedelta(days=1)
    
    # Look ahead up to 10 days (handles long weekends + holidays)
    for _ in range(10):
        if is_market_open(check_date):
            return check_date
        check_date += timedelta(days=1)
    
    return check_date  # Fallback


def get_holidays_in_range(start_date: date, end_date: date) -> list:
    """
    Get list of market holidays in a date range.
    
    Args:
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)
        
    Returns:
        List of holiday dates in range.
    """
    return sorted([
        h for h in US_MARKET_HOLIDAYS_2026
        if start_date <= h <= end_date
    ])


if __name__ == "__main__":
    print(f"Market status: {get_market_status()}")
    print(f"Is open today: {is_market_open()}")
    print(f"Is early close today: {is_early_close()}")
    print(f"Next market day: {get_next_market_day()}")
    print(f"\n2026 Holidays:")
    for h in sorted(US_MARKET_HOLIDAYS_2026):
        print(f"  {h.strftime('%Y-%m-%d %A')}")
