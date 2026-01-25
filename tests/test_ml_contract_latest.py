import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from core.ml_features_v3 import build_ml_20d_features
import core.ml_20d_inference as inf


def test_latest_metadata_version_matches_runtime():
    meta_path = Path("ml/bundles/latest/metadata.json")
    assert meta_path.exists(), "latest metadata.json not found"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bundle_ver = str(meta.get("sklearn_version") or "").strip()
    rt_ver = str(getattr(sklearn, "__version__", "")).strip()
    assert bundle_ver == rt_ver


def test_required_features_presence_or_degraded_flag():
    # Load required features from inference
    ok, model, req_features, _mode = inf._load_bundle_impl()
    assert ok and model is not None
    # Build meteor features from synthetic history
    n = 260
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(50, 100, n)
    high = close * 1.01
    low = close * 0.99
    vol = np.linspace(1e6, 2e6, n)
    hist = pd.DataFrame({
        "Date": dates,
        "Open": close * 0.999,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })
    X_meteor, names = build_ml_20d_features(hist)
    # Case 1: Use meteor-only features (expect degraded if bundle requires more)
    row_meteor = {c: float(X_meteor[c].iloc[0]) for c in names if c in X_meteor.columns}
    _ = inf.compute_ml_20d_probabilities_raw(pd.Series(row_meteor))
    meta1 = inf.get_ml_health_meta()
    missing = [c for c in req_features if c not in row_meteor]
    if missing:
        assert meta1.get("ml_degraded") is True
        assert set(missing).issubset(set(meta1.get("ml_missing_features", [])))
    else:
        assert meta1.get("ml_degraded") is False
