import pytest
import pandas as pd
from datetime import datetime

from ml.feature_pipeline import FeaturePipeline
from core.interfaces import TickerFeatures

pytestmark = pytest.mark.skip(reason="ml/feature_pipeline.py stub not yet implemented")


def make_df(start_date: str, end_date: str, ticker: str = "AAPL", sector: str = "Technology") -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    n = len(dates)
    data = {
        "Ticker": [ticker] * n,
        "Date": dates,
        "Sector": [sector] * n,
        "PE": [20 + i % 5 for i in range(n)],
        "RSI": [50 + (i % 10) for i in range(n)],
        "ATR_Pct": [0.02 + (i % 5) * 0.005 for i in range(n)],
        "MarketCap": [2_000_000_000_000] * n,
        "Volume": [50_000_000 + i * 1000 for i in range(n)],
    }
    return pd.DataFrame(data)


def test_fit_transform_cycle():
    fp = FeaturePipeline()
    df_train = make_df("2022-01-01", "2022-12-01")
    df_test = make_df("2023-01-01", "2023-12-01")

    fp.fit(df_train)
    out = fp.transform(df_test, as_of_date=datetime(2023, 12, 31))

    assert isinstance(out, list)
    assert len(out) == len(df_test)
    assert all(isinstance(o, TickerFeatures) for o in out)
    # Ensure model_features contain normalized floats
    mf = out[0].model_features
    assert set(["feat_rsi", "feat_atr_pct", "feat_fund_pe_sector_rel"]).issubset(set(mf.keys()))
    for v in mf.values():
        assert isinstance(v, float)


def test_leakage_guard_raises():
    fp = FeaturePipeline()
    df_train = make_df("2022-01-01", "2022-12-01")
    df_future = make_df("2024-01-01", "2024-12-01")

    fp.fit(df_train)
    with pytest.raises(ValueError):
        fp.transform(df_future, as_of_date=datetime(2023, 12, 31))


def test_transform_before_fit_raises():
    fp = FeaturePipeline()
    df_test = make_df("2023-01-01", "2023-12-01")
    with pytest.raises(RuntimeError):
        fp.transform(df_test)
