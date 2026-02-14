"""Tiingo provider — re-exports canonical implementation from data_sources_v2."""

def get_tiingo_fundamentals(ticker, provider_status=None):
    """Fetch fundamentals from Tiingo. Delegates to data_sources_v2."""
    from core.data_sources_v2 import fetch_fundamentals_tiingo
    return fetch_fundamentals_tiingo(ticker, provider_status)
