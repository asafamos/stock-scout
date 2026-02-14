"""Alpha Vantage provider — re-exports canonical implementation from data_sources_v2."""

def get_alpha_vantage_fundamentals(ticker, provider_status=None):
    """Fetch fundamentals from Alpha Vantage. Delegates to data_sources_v2."""
    from core.data_sources_v2 import fetch_fundamentals_alpha
    return fetch_fundamentals_alpha(ticker, provider_status)
