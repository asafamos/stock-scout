import os
import tempfile
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


def test_strict_loader_loads_valid_bundle(monkeypatch):
    """Test that the ML loader can load a valid bundle from a custom path."""
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
    feature_names = ["Volume_Surge_Ratio", "Dist_52w_High", "ADR_Pct"]

    with tempfile.TemporaryDirectory() as td:
        bundle_dir = Path(td)
        model_path = bundle_dir / "model.joblib"
        meta_path = bundle_dir / "metadata.json"
        
        # Save model directly (new format)
        joblib.dump(model, model_path)
        
        # Save metadata
        meta = {
            "sklearn_version": sklearn.__version__,
            "feature_list": feature_names,
            "created_at_utc": "2025-01-01T00:00:00Z",
            "preferred_scoring_mode_20d": "hybrid",
        }
        meta_path.write_text(json.dumps(meta))

        # Monkeypatch environment variable to point to our temp bundle
        monkeypatch.setenv("ML_BUNDLE_DIR", str(bundle_dir))
        
        # Force reload of loader to pick up new path
        import importlib
        ml = importlib.import_module("core.ml_20d_inference")
        importlib.reload(ml)
        
        ok, model_loaded, features, mode = ml._load_bundle_impl()
        assert ok is True, "Expected bundle to load successfully"
        assert model_loaded is not None, "Expected model to be loaded"
        assert set(feature_names) <= set(features), f"Expected features {feature_names} to be in {features}"
        assert mode == "hybrid"
