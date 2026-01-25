import pandas as pd
from core.data_sources_v2 import fetch_fundamentals_batch
from core.fundamental import DataMode

def test_fundamentals_batch_no_nameerror_empty_when_no_providers():
    provider_status = {
        'FMP': {'ok': False, 'can_fund': False, 'status': 'no_key'},
        'FINNHUB': {'ok': False, 'can_fund': False, 'status': 'no_key'},
        'TIINGO': {'ok': False, 'can_fund': False, 'status': 'no_key'},
        'ALPHAVANTAGE': {'ok': False, 'can_fund': False, 'status': 'no_key'},
    }
    df = fetch_fundamentals_batch(['AAPL'], provider_status=provider_status, mode=DataMode.LIVE)
    # Should not raise NameError; result can be empty or populated via store
    assert isinstance(df, pd.DataFrame)
    assert 'AAPL' in list(df.index)
