import os
from pathlib import Path
import joblib
import sklearn
import pandas as pd
import numpy as np

from core.ml_features_v3 import build_ml_20d_features


def test_bundle_has_versions_and_features(tmp_path: Path):
    from sklearn.linear_model import LogisticRegression
    import numpy as _np
    # Train a tiny model so it pickles cleanly
    X_demo = _np.array([[0.0, 0.0, 0.0], [1.0, -0.1, 0.05], [0.5, -0.05, 0.02], [2.0, -0.2, 0.1]])
    y_demo = _np.array([0, 1, 0, 1])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_demo, y_demo)

    bundle = {
        "model": model,
        "feature_names": ["Volume_Surge_Ratio", "Dist_52w_High", "ADR_Pct"],
        "sklearn_version": sklearn.__version__,
        "created_at_utc": "2026-01-22T00:00:00Z",
        "git_commit": "deadbeef",
        "preferred_scoring_mode_20d": "hybrid",
        "feature_conventions": {"ADR_Pct": "fraction"},
    }
    out = tmp_path / "bundle.joblib"
    joblib.dump(bundle, out)

    loaded = joblib.load(out)
    assert isinstance(loaded, dict)
    assert "model" in loaded
    assert isinstance(loaded.get("feature_names"), list)
    assert set(["Volume_Surge_Ratio", "Dist_52w_High", "ADR_Pct"]).issubset(set(loaded["feature_names"]))
    assert isinstance(loaded.get("sklearn_version"), str) and len(loaded["sklearn_version"]) > 0
    assert loaded.get("feature_conventions", {}).get("ADR_Pct") == "fraction"


def test_feature_builder_outputs_expected_columns():
    # Build tiny synthetic OHLCV history (>= 260 rows for 52w computations)
    n = 260
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(50, 100, n) + np.sin(np.linspace(0, 10, n))
    high = close * (1 + 0.01)
    low = close * (1 - 0.01)
    volume = np.linspace(1e6, 2e6, n)
    df = pd.DataFrame({
        "Date": dates,
        "Open": close * 0.999,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }).set_index("Date")

    X, names = build_ml_20d_features(history_df=df)
    assert isinstance(names, list) and names == ["Volume_Surge_Ratio", "Dist_52w_High", "ADR_Pct"]
    assert all(col in X.columns for col in names)
    # All numeric and finite or NaN-safe
    for col in names:
        assert col in X
        assert X[col].dtype.kind in ("f", "i")