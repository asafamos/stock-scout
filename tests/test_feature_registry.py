"""
Tests for feature registry consistency.

Ensures that the feature registry is the single source of truth and that
training and inference use the same feature definitions.
"""
import pytest
import pandas as pd
import numpy as np
from core.feature_registry import (
    get_feature_names,
    get_feature_specs,
    get_feature_defaults,
    get_feature_ranges,
    get_features_by_category,
    validate_features,
    clip_features_to_range,
    fill_missing_with_defaults,
    FEATURE_COUNT_V3,
    SUPPORTED_VERSIONS,
    FeatureSpec,
)


class TestFeatureCount:
    """Test feature count consistency."""
    
    def test_feature_count_v3(self):
        """Verify V3 has exactly 34 features."""
        assert len(get_feature_names("v3")) == 34
        assert FEATURE_COUNT_V3 == 34
    
    def test_feature_count_constant_matches_list(self):
        """Constant must match actual list length."""
        assert FEATURE_COUNT_V3 == len(get_feature_names("v3"))
    
    def test_unknown_version_raises(self):
        """Unknown version should raise ValueError."""
        with pytest.raises(ValueError):
            get_feature_names("v999")


class TestFeatureSpecifications:
    """Test feature specification completeness."""
    
    def test_all_features_have_specs(self):
        """All features must have complete specifications."""
        specs = get_feature_specs("v3")
        for spec in specs:
            assert spec.name, "Feature must have name"
            assert spec.description, "Feature must have description"
            assert isinstance(spec.default_value, (int, float)), \
                f"{spec.name} default must be numeric"
            assert len(spec.valid_range) == 2, \
                f"{spec.name} must have (min, max) range"
            assert spec.category, "Feature must have category"
    
    def test_valid_ranges_ordered(self):
        """Min must be <= max for all ranges."""
        specs = get_feature_specs("v3")
        for spec in specs:
            lo, hi = spec.valid_range
            assert lo <= hi, \
                f"{spec.name} has invalid range: [{lo}, {hi}]"
    
    def test_defaults_within_range(self):
        """Default values must be within valid range."""
        specs = get_feature_specs("v3")
        for spec in specs:
            lo, hi = spec.valid_range
            assert lo <= spec.default_value <= hi, \
                f"{spec.name} default {spec.default_value} outside range [{lo}, {hi}]"
    
    def test_unique_feature_names(self):
        """All feature names must be unique."""
        names = get_feature_names("v3")
        assert len(names) == len(set(names)), "Duplicate feature names found"
    
    def test_specs_match_names(self):
        """Specs list must produce same names as get_feature_names."""
        specs = get_feature_specs("v3")
        names = get_feature_names("v3")
        assert [s.name for s in specs] == names


class TestFeatureDefaults:
    """Test default value retrieval."""
    
    def test_defaults_complete(self):
        """Defaults dict must have all features."""
        defaults = get_feature_defaults("v3")
        names = get_feature_names("v3")
        assert set(defaults.keys()) == set(names)
    
    def test_defaults_numeric(self):
        """All defaults must be numeric."""
        defaults = get_feature_defaults("v3")
        for name, value in defaults.items():
            assert isinstance(value, (int, float)), \
                f"{name} default is not numeric: {type(value)}"


class TestFeatureRanges:
    """Test range retrieval."""
    
    def test_ranges_complete(self):
        """Ranges dict must have all features."""
        ranges = get_feature_ranges("v3")
        names = get_feature_names("v3")
        assert set(ranges.keys()) == set(names)
    
    def test_ranges_are_tuples(self):
        """All ranges must be (min, max) tuples."""
        ranges = get_feature_ranges("v3")
        for name, rng in ranges.items():
            assert isinstance(rng, tuple), f"{name} range is not tuple"
            assert len(rng) == 2, f"{name} range must have 2 elements"


class TestFeatureCategories:
    """Test feature category groupings."""
    
    def test_categories_cover_all_features(self):
        """All features must be in some category."""
        by_category = get_features_by_category("v3")
        all_from_categories = []
        for features in by_category.values():
            all_from_categories.extend(features)
        
        names = get_feature_names("v3")
        assert sorted(all_from_categories) == sorted(names)
    
    def test_known_categories_present(self):
        """Expected categories should be present."""
        by_category = get_features_by_category("v3")
        expected = ["technical", "volatility", "volume", "momentum", 
                    "market", "sector", "price_action"]
        for cat in expected:
            assert cat in by_category, f"Missing category: {cat}"


class TestValidateFeatures:
    """Test feature validation function."""
    
    def test_valid_dataframe(self):
        """Valid DataFrame should pass validation."""
        features = get_feature_names("v3")
        df_valid = pd.DataFrame([{f: 0.0 for f in features}])
        is_valid, missing = validate_features(df_valid, "v3")
        assert is_valid
        assert missing == []
    
    def test_missing_features_detected(self):
        """Missing features should be detected."""
        df_missing = pd.DataFrame([{"RSI": 50.0}])
        is_valid, missing = validate_features(df_missing, "v3")
        assert not is_valid
        assert len(missing) == 33  # All except RSI
    
    def test_empty_dataframe(self):
        """Empty DataFrame should fail validation."""
        df_empty = pd.DataFrame()
        is_valid, missing = validate_features(df_empty, "v3")
        assert not is_valid
        assert len(missing) == 34
    
    def test_extra_columns_ok(self):
        """Extra columns should not cause failure."""
        features = get_feature_names("v3")
        df = pd.DataFrame([{f: 0.0 for f in features}])
        df["extra_column"] = 999
        is_valid, missing = validate_features(df, "v3")
        assert is_valid
        assert missing == []


class TestClipFeatures:
    """Test feature clipping function."""
    
    def test_clip_extreme_values(self):
        """Values outside range should be clipped."""
        features = get_feature_names("v3")
        df = pd.DataFrame([{f: 999.0 for f in features}])
        
        clipped = clip_features_to_range(df, "v3", inplace=False)
        
        ranges = get_feature_ranges("v3")
        for feat in features:
            lo, hi = ranges[feat]
            assert clipped[feat].iloc[0] <= hi, \
                f"{feat} not clipped to max {hi}"
    
    def test_clip_negative_extremes(self):
        """Negative extremes should be clipped."""
        features = get_feature_names("v3")
        df = pd.DataFrame([{f: -999.0 for f in features}])
        
        clipped = clip_features_to_range(df, "v3", inplace=False)
        
        ranges = get_feature_ranges("v3")
        for feat in features:
            lo, hi = ranges[feat]
            assert clipped[feat].iloc[0] >= lo, \
                f"{feat} not clipped to min {lo}"
    
    def test_clip_inplace(self):
        """Inplace clipping should modify original."""
        features = get_feature_names("v3")
        df = pd.DataFrame([{f: 999.0 for f in features}])
        original_id = id(df)
        
        result = clip_features_to_range(df, "v3", inplace=True)
        
        assert id(result) == original_id


class TestFillMissingDefaults:
    """Test filling missing features with defaults."""
    
    def test_fill_empty_dict(self):
        """Empty dict should be filled with all defaults."""
        result = fill_missing_with_defaults({}, "v3")
        defaults = get_feature_defaults("v3")
        assert result == defaults
    
    def test_fill_partial_dict(self):
        """Partial dict should preserve existing values."""
        partial = {"RSI": 75.0, "ATR_Pct": 0.05}
        result = fill_missing_with_defaults(partial, "v3")
        
        assert result["RSI"] == 75.0
        assert result["ATR_Pct"] == 0.05
        assert len(result) == 34
    
    def test_fill_does_not_modify_original(self):
        """Original dict should not be modified."""
        partial = {"RSI": 75.0}
        original = partial.copy()
        fill_missing_with_defaults(partial, "v3")
        assert partial == original


class TestTrainingInferenceConsistency:
    """Test that training and inference use same features."""
    
    def test_inference_features_match_registry(self):
        """ml_integration.py must use features from registry."""
        # Import at test time to avoid circular imports
        from core.ml_integration import DEFAULT_FEATURES
        registry_features = get_feature_names("v3")
        
        assert DEFAULT_FEATURES == registry_features, \
            "ml_integration.py DEFAULT_FEATURES must match feature_registry!"
    
    def test_training_imports_registry(self):
        """Training script must import from registry."""
        # Import at test time
        from scripts.train_rolling_ml_20d import FEATURE_NAMES_V3
        registry_features = get_feature_names("v3")
        
        assert FEATURE_NAMES_V3 == registry_features, \
            "train_rolling_ml_20d.py must use features from registry!"


class TestFeatureOrdering:
    """Test that feature order is consistent."""
    
    def test_order_is_deterministic(self):
        """Multiple calls should return same order."""
        order1 = get_feature_names("v3")
        order2 = get_feature_names("v3")
        assert order1 == order2
    
    def test_first_feature_is_rsi(self):
        """First feature should be RSI (by convention)."""
        features = get_feature_names("v3")
        assert features[0] == "RSI"
    
    def test_last_feature_is_support_strength(self):
        """Last feature should be Support_Strength."""
        features = get_feature_names("v3")
        assert features[-1] == "Support_Strength"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
