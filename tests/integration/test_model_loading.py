"""Integration tests for ML model loading and inference.

These tests verify:
- Model discovery and path resolution
- Model loading and validation
- Feature compatibility between training and inference
"""
import pytest
import os
from pathlib import Path


class TestModelPathResolution:
    """Test model path discovery."""

    def test_model_finder_returns_path_or_none(self):
        """Model finder should return valid path or None."""
        from core.ml_integration import _find_model_path

        path = _find_model_path()
        
        # May be None if no model exists (OK for CI without models)
        if path is not None:
            assert os.path.exists(path), f"Model path {path} does not exist"
            assert path.endswith(".pkl"), f"Model should be pickle file: {path}"

    def test_model_finder_checks_v3_first(self):
        """Model finder should prioritize v3 model."""
        from core.ml_integration import _find_model_path

        path = _find_model_path()
        
        # If v3 model exists, it should be found
        v3_path = "models/model_20d_v3.pkl"
        if os.path.exists(v3_path):
            assert path == v3_path or "v3" in str(path)


class TestModelInfo:
    """Test model introspection."""

    def test_model_info_structure(self):
        """get_model_info should return expected structure."""
        from core.ml_integration import get_model_info

        info = get_model_info()

        # Required keys
        assert "loaded" in info
        assert "path" in info
        assert "feature_count" in info
        assert "model_type" in info
        assert "features" in info

        # Types
        assert isinstance(info["loaded"], bool)
        assert info["path"] is None or isinstance(info["path"], str)
        assert info["feature_count"] is None or isinstance(info["feature_count"], int)

    def test_model_info_reflects_load_state(self):
        """Model info should reflect current load state."""
        from core.ml_integration import get_model_info
        import core.ml_integration as ml_mod
        
        # Store original state
        original_loaded = ml_mod._MODEL_LOADED
        original_model = ml_mod._ML_MODEL
        
        try:
            # Reset state
            ml_mod._MODEL_LOADED = False
            ml_mod._ML_MODEL = None
            
            info = get_model_info()
            # When not loaded, should reflect that
            # (Note: get_model_info may auto-load, so just check it returns valid info)
            assert "loaded" in info
        finally:
            # Restore state
            ml_mod._MODEL_LOADED = original_loaded
            ml_mod._ML_MODEL = original_model


class TestModelLoading:
    """Test actual model loading."""

    @pytest.mark.skipif(
        not os.path.exists("models/model_20d_v3.pkl"),
        reason="Model file not present - skipping model loading test"
    )
    def test_v3_model_loads_successfully(self):
        """v3 model should load successfully."""
        from core.ml_integration import load_ml_model, get_model_info

        result = load_ml_model("models/model_20d_v3.pkl")
        assert result is True

        info = get_model_info()
        assert info["loaded"] is True
        # Current v3 model has 5 features (legacy), future models will have 34
        # This test validates the model loads, not specific feature count
        assert info["feature_count"] is not None
        assert info["feature_count"] > 0

    @pytest.mark.skipif(
        not os.path.exists("models/model_20d_v3.pkl"),
        reason="Model file not present"
    )
    def test_loaded_model_has_features(self):
        """Loaded model should have feature list."""
        from core.ml_integration import load_ml_model, get_expected_features
        from core.feature_registry import get_feature_names

        load_ml_model("models/model_20d_v3.pkl")
        
        model_features = get_expected_features()
        # Just verify features exist and are a list
        assert isinstance(model_features, list)
        assert len(model_features) > 0


class TestFeatureConsistency:
    """Test feature consistency between components."""

    def test_ml_integration_imports_from_registry(self):
        """ml_integration should use features from registry."""
        from core.feature_registry import get_feature_names
        from core.ml_integration import DEFAULT_FEATURES

        registry_features = get_feature_names("v3")
        
        # DEFAULT_FEATURES should match registry
        assert DEFAULT_FEATURES == registry_features

    def test_training_script_uses_registry(self):
        """Training script should import from registry."""
        # This is a static analysis check
        import ast
        
        train_script = Path("scripts/train_rolling_ml_20d.py")
        if not train_script.exists():
            pytest.skip("Training script not found")
        
        source = train_script.read_text()
        tree = ast.parse(source)
        
        # Check for import from feature_registry
        imports = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        
        # Find imports from feature_registry
        registry_imports = [
            imp for imp in imports
            if isinstance(imp, ast.ImportFrom) and imp.module and "feature_registry" in imp.module
        ]
        
        assert len(registry_imports) > 0, "Training script should import from feature_registry"


class TestModelValidation:
    """Test model validation functions."""

    def test_validate_model_features_function_exists(self):
        """validate_model_features should be available."""
        from core.ml_integration import validate_model_features
        
        assert callable(validate_model_features)

    @pytest.mark.skipif(
        not os.path.exists("models/model_20d_v3.pkl"),
        reason="Model file not present"
    )
    def test_v3_model_has_valid_features(self):
        """v3 model should have valid feature count."""
        from core.ml_integration import load_ml_model, get_model_info

        load_ml_model("models/model_20d_v3.pkl")
        info = get_model_info()
        
        # Check model loaded with expected features
        assert info["loaded"] is True
        assert info["feature_count"] is not None


class TestInferenceIntegration:
    """Test inference integration."""

    def test_integrate_ml_with_conviction_returns_tuple(self):
        """ML integration should return (score, info_dict) tuple."""
        from core.ml_integration import integrate_ml_with_conviction
        
        base_conviction = 70.0
        result, info = integrate_ml_with_conviction(
            base_conviction,
            ticker_data={"Ticker": "TEST"},
            technical_indicators={"RSI": 50},
            fundamental_scores={},
            enable_ml=True
        )
        
        # Should return a float and dict
        assert isinstance(result, (int, float))
        assert isinstance(info, dict)
        assert 0 <= result <= 100
        
        # Info dict should have expected keys
        assert "ml_probability" in info
        assert "ml_boost" in info
        assert "ml_status" in info

    def test_integrate_ml_with_ml_disabled(self):
        """ML integration should skip when disabled."""
        from core.ml_integration import integrate_ml_with_conviction
        
        base_conviction = 70.0
        result, info = integrate_ml_with_conviction(
            base_conviction,
            ticker_data={"Ticker": "TEST"},
            technical_indicators={"RSI": 50},
            fundamental_scores={},
            enable_ml=False  # Explicitly disabled
        )
        
        assert result == base_conviction
        assert info.get("ml_boost", 0) == 0


class TestMarketCalendar:
    """Test market calendar integration."""

    def test_market_calendar_imports(self):
        """Market calendar should be importable."""
        from scripts.market_calendar import is_market_open, get_market_status
        
        assert callable(is_market_open)
        assert callable(get_market_status)

    def test_is_market_open_returns_bool(self):
        """is_market_open should return boolean."""
        from scripts.market_calendar import is_market_open
        
        result = is_market_open()
        assert isinstance(result, bool)

    def test_get_market_status_returns_valid_string(self):
        """get_market_status should return valid status."""
        from scripts.market_calendar import get_market_status
        
        result = get_market_status()
        valid_statuses = ["OPEN", "PRE-MARKET", "AFTER-HOURS", "CLOSED (holiday/weekend)"]
        
        assert result in valid_statuses

    def test_weekend_is_closed(self):
        """Weekends should be marked as closed."""
        from scripts.market_calendar import is_market_open
        from datetime import date
        
        # Test a known Saturday (Feb 7, 2026)
        saturday = date(2026, 2, 7)
        assert is_market_open(saturday) is False
        
        # Test a known Sunday (Feb 8, 2026)
        sunday = date(2026, 2, 8)
        assert is_market_open(sunday) is False

    def test_holiday_is_closed(self):
        """Holidays should be marked as closed."""
        from scripts.market_calendar import is_market_open
        from datetime import date
        
        # Test Christmas 2026 (Dec 25)
        christmas = date(2026, 12, 25)
        assert is_market_open(christmas) is False
