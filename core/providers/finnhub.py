"""Finnhub provider — re-exports canonical implementation from data_sources_v2."""

def get_finnhub_fundamentals(ticker, provider_status=None):
    """Fetch fundamentals from Finnhub. Delegates to data_sources_v2."""
    from core.data_sources_v2 import fetch_fundamentals_finnhub
    return fetch_fundamentals_finnhub(ticker, provider_status)
