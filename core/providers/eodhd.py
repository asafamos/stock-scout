"""EODHD provider — re-exports canonical implementation from data_sources_v2."""

def get_eodhd_fundamentals(ticker, provider_status=None):
    """Fetch fundamentals from EODHD. Delegates to data_sources_v2."""
    from core.data_sources_v2 import fetch_fundamentals_eodhd
    return fetch_fundamentals_eodhd(ticker, provider_status)
