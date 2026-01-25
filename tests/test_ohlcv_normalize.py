import pandas as pd
import numpy as np

from core.data_sources_v2 import normalize_ohlcv


def test_normalize_ohlcv_basic_and_trailing_nan_row():
    # Input with Title-case columns and a trailing all-NaN row
    df = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=4, freq='D'),
        'Open': [10.0, 11.0, 12.0, np.nan],
        'High': [10.5, 11.5, 12.5, np.nan],
        'Low': [9.5, 10.5, 11.5, np.nan],
        'Close': [10.2, 11.2, 12.2, np.nan],
        'Volume': [100000, 110000, 120000, np.nan],
    })
    out = normalize_ohlcv(df)
    # Columns normalized
    assert 'close' in out.columns
    assert 'volume' in out.columns
    # Trailing NaN row dropped; last close should be 12.2
    last_close = pd.to_numeric(out['close'], errors='coerce').dropna().iloc[-1]
    assert last_close == 12.2
    # Volume remains numeric and NaN is preserved (not coerced to 0)
    assert out['volume'].isna().sum() == 0  # last row removed, so no NaNs remain


def test_normalize_ohlcv_adj_close_used_when_close_missing():
    df = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=3, freq='D'),
        'Adj Close': [5.0, 6.0, 7.0],
        'Volume': [50000, 60000, 70000],
    })
    out = normalize_ohlcv(df)
    assert 'close' in out.columns
    assert float(out['close'].iloc[-1]) == 7.0
