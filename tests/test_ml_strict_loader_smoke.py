import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def test_strict_loader_loads_valid_bundle(monkeypatch):
    # Build a tiny valid bundle in a temp file
    X = np.array([
        [1.0, -0.1, 0.02],
        [1.5, -0.05, 0.04],
        [0.8, -0.2, 0.01],
    ])
    y = np.array([0, 1, 0])
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    import sklearn
    bundle = {
        "model": model,
        "feature_names": ["Volume_Surge_Ratio", "Dist_52w_High", "ADR_Pct"],
        "sklearn_version": sklearn.__version__,
        "created_at_utc": "2025-01-01T00:00:00Z",
        "git_commit": "deadbeef",
        "preferred_scoring_mode_20d": "hybrid",
        "feature_conventions": {"ADR_Pct": "fraction"},
    }

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ml_20d_bundle.joblib"
        joblib.dump(bundle, p)

        # Monkeypatch config to point to our temp bundle
        import core.config as cfg_mod
        cfg = cfg_mod.get_config()
        old_path = cfg.ml_20d_bundle_path
        cfg.ml_20d_bundle_path = str(p)
        try:
            # Force reload of loader to pick up path
            import importlib
            ml = importlib.import_module("core.ml_20d_inference")
            importlib.reload(ml)
            ok, model_loaded, features, mode = ml._load_bundle_impl()
            assert ok is True
            assert set(["Volume_Surge_Ratio", "Dist_52w_High", "ADR_Pct"]) <= set(features)
            assert mode == "hybrid"
        finally:
            cfg.ml_20d_bundle_path = old_path
