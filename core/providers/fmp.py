"""FMP provider — re-exports canonical implementation from data_sources_v2."""

def get_fmp_fundamentals(ticker, provider_status=None):
    """Fetch fundamentals from FMP. Delegates to data_sources_v2."""
    from core.data_sources_v2 import fetch_fundamentals_fmp
    return fetch_fundamentals_fmp(ticker, provider_status)
