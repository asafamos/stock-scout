import numpy as np
import pandas as pd

def _make_history(n=300):
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.linspace(50, 100, n) + np.sin(np.linspace(0, 10, n))
    high = close * 1.01
    low = close * 0.99
    vol = np.linspace(1e6, 1.5e6, n)
    df = pd.DataFrame({
        "Date": dates,
        "Open": close * 1.0,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    }).set_index("Date")
    return df


def test_pipeline_uses_history_based_ml(monkeypatch):
    # Monkeypatch scoring module to avoid touching real ML bundle
    import core.scoring_pipeline_20d as sp20d
    monkeypatch.setattr(sp20d, "ML_20D_AVAILABLE", True, raising=False)
    monkeypatch.setattr(sp20d, "compute_ml_20d_probabilities_from_history", lambda df_hist: 0.73, raising=False)

    # Build a tiny price_data dict for score_universe_20d
    price_data = {"TEST": _make_history(300)}
    from core.scoring_pipeline_20d import score_universe_20d
    from datetime import datetime

    df = score_universe_20d(
        as_of_date=datetime(2025, 12, 31),
        horizon_days=20,
        universe=["TEST"],
        price_data=price_data,
        include_ml=True,
        logger=None,
        benchmark_df=None,
    )

    assert not df.empty
    assert "ML_20d_Prob" in df.columns
    val = float(df.iloc[0]["ML_20d_Prob"])  # after live_v3 calibration
    assert 0.0 <= val <= 1.0