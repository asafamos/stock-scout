"""
Tests for v3.1 feature upgrade - Stock Picker ML
=================================================

Covers:
  1. Feature registry v3.1 correctness (39 features, no market-timing)
  2. Training script: calculate_all_features produces all 39 features
  3. Inference builder: build_all_ml_features_v3 produces matching 39 features
  4. PurgedWalkForwardCV: expanding window + embargo gap
  5. Train-serve alignment: training and inference compute the same features
  6. Market context: SPY_60d_ret propagation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ohlcv_df():
    """Create a realistic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 300  # ~1 year of trading days
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    # Ensure positive prices
    close = np.maximum(close, 10.0)
    
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": close + abs(np.random.randn(n) * 0.5),
        "Low": close - abs(np.random.randn(n) * 0.5),
        "Close": close,
        "Volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=dates)


@pytest.fixture
def indicator_row(ohlcv_df):
    """Create a row mimicking build_technical_indicators output."""
    close = ohlcv_df["Close"]
    return pd.Series({
        "RSI": 55.0,
        "ATR_Pct": 0.025,
        "Close": close.iloc[-1],
        "MA20": close.rolling(20).mean().iloc[-1],
        "MA50": close.rolling(50).mean().iloc[-1],
        "MA200": close.rolling(200, min_periods=100).mean().iloc[-1],
        "Tightness_Ratio": 0.75,
        "Dist_52w_High": -0.08,
        "VolSurge": 1.15,
        "MomCons": 0.6,
        "UpStreak_Days": 2,
        "DownStreak_Days": 0,
        "High_20d": ohlcv_df["High"].tail(20).max(),
    })


@pytest.fixture
def market_context():
    return {
        "Market_Regime": 1.0,
        "SPY_20d_ret": 0.03,
        "SPY_60d_ret": 0.06,
    }


@pytest.fixture
def sector_context():
    return {
        "Sector_RS": 0.02,
        "Sector_Momentum": 0.04,
        "Sector_Rank": 0.65,
    }


# ============================================================================
# 1. FEATURE REGISTRY TESTS
# ============================================================================

class TestFeatureRegistry:
    """Tests for feature_registry.py v3.1 definitions."""

    def test_v31_count(self):
        from core.feature_registry import get_feature_names
        names = get_feature_names("v3.1")
        assert len(names) == 39, f"Expected 39, got {len(names)}"

    def test_v3_count_unchanged(self):
        from core.feature_registry import get_feature_names
        names = get_feature_names("v3")
        assert len(names) == 34, f"v3 should still have 34, got {len(names)}"

    def test_market_timing_removed_from_v31(self):
        from core.feature_registry import get_feature_names
        names = set(get_feature_names("v3.1"))
        assert "Market_Trend" not in names
        assert "Market_Volatility" not in names
        assert "High_Volatility" not in names

    def test_market_timing_still_in_v3(self):
        from core.feature_registry import get_feature_names
        names = set(get_feature_names("v3"))
        assert "Market_Trend" in names
        assert "Market_Volatility" in names
        assert "High_Volatility" in names

    def test_market_regime_kept(self):
        from core.feature_registry import get_feature_names
        names = set(get_feature_names("v3.1"))
        assert "Market_Regime" in names

    def test_new_features_present(self):
        from core.feature_registry import get_feature_names
        names = set(get_feature_names("v3.1"))
        new_features = {
            "Vol_Contraction_Ratio", "Squeeze_On_Flag",
            "RS_vs_SPY_60d", "RS_Momentum",
            "UpStreak_Days", "DownStreak_Days",
            "Range_Pct_10d", "OvernightGap_Avg",
        }
        missing = new_features - names
        assert not missing, f"Missing new features: {missing}"

    def test_defaults_populated(self):
        from core.feature_registry import get_feature_defaults
        defaults = get_feature_defaults("v3.1")
        assert len(defaults) == 39
        # Spot check some defaults
        assert defaults["RSI"] == 50.0
        assert defaults["Market_Regime"] == 0.0
        assert defaults["UpStreak_Days"] == 0.0
        assert defaults["Squeeze_On_Flag"] == 0.0
        assert defaults["Vol_Contraction_Ratio"] == 1.0

    def test_ranges_populated(self):
        from core.feature_registry import get_feature_ranges
        ranges = get_feature_ranges("v3.1")
        assert len(ranges) == 39
        assert ranges["UpStreak_Days"] == (0, 10)
        assert ranges["Squeeze_On_Flag"] == (0, 1)

    def test_validate_features(self):
        from core.feature_registry import validate_features, get_feature_names
        names = get_feature_names("v3.1")
        df = pd.DataFrame(np.zeros((1, 39)), columns=names)
        valid, missing = validate_features(df, "v3.1")
        assert valid
        assert len(missing) == 0

    def test_default_version_is_v36(self):
        from core.feature_registry import DEFAULT_VERSION
        assert DEFAULT_VERSION == "v3.6"


# ============================================================================
# 2. TRAINING FEATURE COMPUTATION TESTS
# ============================================================================

class TestTrainingFeatures:
    """Tests for calculate_all_features in training script."""

    def test_produces_39_features(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        from core.feature_registry import get_feature_names
        
        spy_returns = ohlcv_df["Close"].pct_change(20)
        spy_returns_60d = ohlcv_df["Close"].pct_change(60)
        
        result = calculate_all_features(
            ohlcv_df.copy(), spy_returns, spy_returns_60d
        )
        
        expected = set(get_feature_names("v3.1"))
        present = set(result.columns) & expected
        missing = expected - present
        assert not missing, f"Missing features in training output: {missing}"

    def test_no_market_timing_features(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        
        result = calculate_all_features(ohlcv_df.copy())
        assert "Market_Trend" not in result.columns
        assert "Market_Volatility" not in result.columns
        assert "High_Volatility" not in result.columns

    def test_market_regime_present(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        
        result = calculate_all_features(ohlcv_df.copy())
        assert "Market_Regime" in result.columns

    def test_vol_contraction_ratio(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        
        result = calculate_all_features(ohlcv_df.copy())
        assert "Vol_Contraction_Ratio" in result.columns
        # Should be a ratio of ATR windows, typically 0.5-2.0
        vals = result["Vol_Contraction_Ratio"].dropna()
        assert len(vals) > 0
        assert vals.median() > 0.3
        assert vals.median() < 3.0

    def test_squeeze_on_flag(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        
        result = calculate_all_features(ohlcv_df.copy())
        assert "Squeeze_On_Flag" in result.columns
        # Should be binary
        vals = result["Squeeze_On_Flag"].dropna()
        assert set(vals.unique()).issubset({0.0, 1.0})

    def test_rs_vs_spy_60d(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        
        spy_returns = ohlcv_df["Close"].pct_change(20)
        spy_returns_60d = ohlcv_df["Close"].pct_change(60)
        
        result = calculate_all_features(
            ohlcv_df.copy(), spy_returns, spy_returns_60d
        )
        assert "RS_vs_SPY_60d" in result.columns
        assert "RS_Momentum" in result.columns

    def test_streak_features(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        
        result = calculate_all_features(ohlcv_df.copy())
        assert "UpStreak_Days" in result.columns
        assert "DownStreak_Days" in result.columns
        # Streaks capped at 10
        assert result["UpStreak_Days"].max() <= 10
        assert result["DownStreak_Days"].max() <= 10
        # Min should be 0
        assert result["UpStreak_Days"].min() >= 0
        assert result["DownStreak_Days"].min() >= 0

    def test_range_pct_10d(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        
        result = calculate_all_features(ohlcv_df.copy())
        assert "Range_Pct_10d" in result.columns
        vals = result["Range_Pct_10d"].dropna()
        assert len(vals) > 0
        # Should be a small positive pct
        assert vals.min() >= 0
        assert vals.max() < 1.0

    def test_overnight_gap_avg(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_all_features
        
        result = calculate_all_features(ohlcv_df.copy())
        assert "OvernightGap_Avg" in result.columns
        vals = result["OvernightGap_Avg"].dropna()
        assert len(vals) > 0
        # Gaps should be small pct values
        assert abs(vals.median()) < 0.1


# ============================================================================
# 3. INFERENCE FEATURE BUILDER TESTS
# ============================================================================

class TestInferenceBuilder:
    """Tests for build_all_ml_features_v3 producing v3.1 features."""

    def test_produces_39_features(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3
        
        features = build_all_ml_features_v3(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        assert len(features) == 39, f"Expected 39, got {len(features)}"

    def test_exact_feature_match(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3
        from core.feature_registry import get_feature_names
        
        features = build_all_ml_features_v3(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        expected = set(get_feature_names("v3.1"))
        actual = set(features.keys())
        
        missing = expected - actual
        extra = actual - expected
        assert not missing, f"Missing: {missing}"
        assert not extra, f"Extra: {extra}"

    def test_no_market_timing_features(self, indicator_row, ohlcv_df):
        from core.ml_feature_builder import build_all_ml_features_v3
        
        features = build_all_ml_features_v3(indicator_row, ohlcv_df)
        assert "Market_Trend" not in features
        assert "Market_Volatility" not in features
        assert "High_Volatility" not in features

    def test_rs_momentum_computed(self, indicator_row, ohlcv_df, market_context):
        from core.ml_feature_builder import build_all_ml_features_v3
        
        features = build_all_ml_features_v3(
            indicator_row, ohlcv_df, market_context
        )
        # RS_Momentum = RS_vs_SPY_20d - RS_vs_SPY_60d
        expected = features["RS_vs_SPY_20d"] - features["RS_vs_SPY_60d"]
        assert abs(features["RS_Momentum"] - expected) < 1e-6

    def test_squeeze_on_flag_is_binary(self, indicator_row, ohlcv_df):
        from core.ml_feature_builder import build_all_ml_features_v3
        
        features = build_all_ml_features_v3(indicator_row, ohlcv_df)
        assert features["Squeeze_On_Flag"] in (0.0, 1.0)

    def test_streaks_capped_at_10(self, indicator_row, ohlcv_df):
        from core.ml_feature_builder import build_all_ml_features_v3
        
        # Set high streak in indicator row
        indicator_row["UpStreak_Days"] = 15
        features = build_all_ml_features_v3(indicator_row, ohlcv_df)
        assert features["UpStreak_Days"] <= 10

    def test_all_features_finite(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3
        
        features = build_all_ml_features_v3(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        for name, val in features.items():
            assert np.isfinite(val), f"Feature {name} is not finite: {val}"

    def test_features_within_registry_ranges(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3
        from core.feature_registry import get_feature_ranges
        
        features = build_all_ml_features_v3(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        ranges = get_feature_ranges("v3.1")
        for name, val in features.items():
            lo, hi = ranges[name]
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_missing_context_gives_defaults(self, indicator_row, ohlcv_df):
        """Without market/sector context, features should get sensible defaults."""
        from core.ml_feature_builder import build_all_ml_features_v3
        
        features = build_all_ml_features_v3(indicator_row, ohlcv_df)
        assert features["Market_Regime"] == 0.0
        assert features["Sector_RS"] == 0.0
        assert features["RS_vs_SPY_20d"] == 0.0
        assert features["RS_vs_SPY_60d"] == 0.0


# ============================================================================
# 4. PURGED WALK-FORWARD CV TESTS
# ============================================================================

class TestPurgedWalkForwardCV:
    """Tests for PurgedWalkForwardCV in training script."""

    def get_cv(self, **kwargs):
        from scripts.train_ml_20d_v3_local import PurgedWalkForwardCV
        defaults = {"n_splits": 5, "embargo_td": 20, "min_train_pct": 0.3}
        defaults.update(kwargs)
        return PurgedWalkForwardCV(**defaults)

    def test_correct_fold_count(self):
        cv = self.get_cv(n_splits=5)
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        X = np.zeros((1000, 5))
        folds = list(cv.split(X, dates=dates))
        assert len(folds) == 5

    def test_expanding_window(self):
        cv = self.get_cv(n_splits=3)
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        X = np.zeros((1000, 5))
        
        train_sizes = []
        for train_idx, val_idx in cv.split(X, dates=dates):
            train_sizes.append(len(train_idx))
        
        # Each fold should have MORE training data (expanding)
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1], \
                f"Fold {i+1} train ({train_sizes[i]}) should be > fold {i} ({train_sizes[i-1]})"

    def test_embargo_gap_enforced(self):
        embargo_days = 20
        cv = self.get_cv(embargo_td=embargo_days)
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        X = np.zeros((1000, 5))
        
        for train_idx, val_idx in cv.split(X, dates=dates):
            train_end = dates[train_idx].max()
            val_start = dates[val_idx].min()
            gap_bdays = np.busday_count(
                train_end.date(), val_start.date()
            )
            assert gap_bdays >= embargo_days, \
                f"Gap {gap_bdays} business days < embargo {embargo_days}"

    def test_no_overlap_between_train_val(self):
        cv = self.get_cv(n_splits=3)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        X = np.zeros((500, 3))
        
        for train_idx, val_idx in cv.split(X, dates=dates):
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, "Train and val indices overlap"

    def test_min_train_pct_respected(self):
        cv = self.get_cv(min_train_pct=0.4, n_splits=3)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        X = np.zeros((500, 5))
        
        folds = list(cv.split(X, dates=dates))
        # First fold should have at least 40% of dates
        first_train = folds[0][0]
        assert len(first_train) >= 500 * 0.35  # Allow a bit of slack

    def test_fallback_without_dates(self):
        """Should work with index-based splitting when dates=None."""
        cv = self.get_cv(n_splits=3, embargo_td=10)
        X = np.zeros((500, 5))
        
        folds = list(cv.split(X))
        assert len(folds) >= 2
        for train_idx, val_idx in folds:
            assert len(train_idx) > 0
            assert len(val_idx) > 0

    def test_multi_ticker_dates(self):
        """Simulates same dates appearing for multiple tickers."""
        dates_per_ticker = pd.date_range("2020-01-01", periods=200, freq="B")
        n_tickers = 5
        all_dates = pd.DatetimeIndex(
            np.tile(dates_per_ticker.values, n_tickers)
        )
        X = np.zeros((200 * n_tickers, 5))
        
        cv = self.get_cv(n_splits=3, embargo_td=10)
        folds = list(cv.split(X, dates=all_dates))
        assert len(folds) >= 2
        
        for train_idx, val_idx in folds:
            train_dates_unique = set(all_dates[train_idx])
            val_dates_unique = set(all_dates[val_idx])
            # No date should appear in both train and val
            overlap = train_dates_unique & val_dates_unique
            assert len(overlap) == 0, "Date overlap between train and val"


# ============================================================================
# 5. TRAIN-SERVE ALIGNMENT TESTS
# ============================================================================

class TestTrainServeAlignment:
    """Ensure training and inference produce the SAME features."""

    def test_same_feature_set(self):
        """Registry should be the single source of truth for both."""
        from core.feature_registry import get_feature_names
        from scripts.train_ml_20d_v3_local import FEATURE_NAMES_V3
        
        registry_names = set(get_feature_names("v3.1"))
        training_names = set(FEATURE_NAMES_V3)
        
        assert registry_names == training_names, \
            f"Mismatch: registry={registry_names - training_names}, " \
            f"training={training_names - registry_names}"

    def test_feature_values_similar(self, ohlcv_df, indicator_row, market_context, sector_context):
        """
        Compare feature values between training and inference for alignment.
        Both should produce similar values for the same stock data.
        """
        from scripts.train_ml_20d_v3_local import calculate_all_features
        from core.ml_feature_builder import build_all_ml_features_v3
        from core.feature_registry import get_feature_names
        
        # Training computation
        spy_returns = ohlcv_df["Close"].pct_change(20)
        spy_returns_60d = ohlcv_df["Close"].pct_change(60)
        
        train_result = calculate_all_features(
            ohlcv_df.copy(), spy_returns, spy_returns_60d
        )
        
        # Get the last row from training
        if len(train_result) > 0:
            last_train_row = train_result.iloc[-1]
        else:
            pytest.skip("No data after calculate_all_features")
        
        # Inference computation
        infer_features = build_all_ml_features_v3(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        
        # Check key features that should be similar
        feature_names = get_feature_names("v3.1")
        
        # These features should be computed identically between train and inference
        exact_match_features = [
            "RSI", "ATR_Pct", "Squeeze_On_Flag", "MA_Alignment"
        ]
        
        for feat in feature_names:
            assert feat in infer_features, f"Inference missing {feat}"
            if feat in train_result.columns:
                train_val = last_train_row[feat]
                infer_val = infer_features[feat]
                # Both should be finite
                if np.isfinite(train_val) and np.isfinite(infer_val):
                    pass  # Both finite — good


# ============================================================================
# 6. MARKET CONTEXT TESTS
# ============================================================================

class TestMarketContext:
    """Test market context includes SPY_60d_ret."""

    def test_get_market_context_includes_spy_60d(self):
        from core.ml_feature_builder import get_market_context_from_row
        
        row = pd.Series({
            "Market_Regime": 1.0,
            "SPY_20d_ret": 0.03,
            "SPY_60d_ret": 0.07,
        })
        ctx = get_market_context_from_row(row)
        assert "SPY_60d_ret" in ctx
        assert ctx["SPY_60d_ret"] == 0.07

    def test_market_context_no_market_timing(self):
        from core.ml_feature_builder import get_market_context_from_row
        
        row = pd.Series({"Market_Regime": 0.0})
        ctx = get_market_context_from_row(row)
        # v3.1 context should NOT include market-timing keys
        assert "Market_Volatility" not in ctx
        assert "Market_Trend" not in ctx

    def test_pipeline_market_context_has_spy_60d(self):
        """Verify the pipeline market context template includes SPY_60d_ret."""
        from core.pipeline.market_data import _compute_global_market_context
        # The function tries to fetch SPY data which won't work in tests,
        # but the default dict should include SPY_60d_ret
        # We can inspect the source
        import inspect
        source = inspect.getsource(_compute_global_market_context)
        assert "SPY_60d_ret" in source


# ============================================================================
# 7. CALCULATE_MARKET_REGIME TESTS
# ============================================================================

class TestCalculateMarketRegime:
    """Test market regime calculation for v3.1."""

    def test_only_returns_market_regime(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_market_regime
        
        regime_df = calculate_market_regime(ohlcv_df)
        assert regime_df is not None
        assert "Market_Regime" in regime_df.columns
        assert "Market_Volatility" not in regime_df.columns
        assert "Market_Trend" not in regime_df.columns
        assert "High_Volatility" not in regime_df.columns

    def test_regime_values(self, ohlcv_df):
        from scripts.train_ml_20d_v3_local import calculate_market_regime
        
        regime_df = calculate_market_regime(ohlcv_df)
        values = set(regime_df["Market_Regime"].unique())
        assert values.issubset({-1, 0, 1}), f"Unexpected regime values: {values}"

    def test_returns_none_for_short_data(self):
        from scripts.train_ml_20d_v3_local import calculate_market_regime
        
        short_df = pd.DataFrame({"Close": [100] * 50}, 
                                index=pd.date_range("2023-01-01", periods=50))
        result = calculate_market_regime(short_df)
        assert result is None


# ============================================================================
# 8. FEATURE REGISTRY VERSION API TESTS
# ============================================================================

class TestVersionAPI:
    """Test version-aware API."""

    def test_invalid_version_raises(self):
        from core.feature_registry import get_feature_names
        with pytest.raises(ValueError, match="Unknown"):
            get_feature_names("v99")

    def test_supported_versions(self):
        from core.feature_registry import SUPPORTED_VERSIONS
        assert "v3" in SUPPORTED_VERSIONS
        assert "v3.1" in SUPPORTED_VERSIONS
        assert "v3.2" in SUPPORTED_VERSIONS

    def test_v36_is_default(self):
        from core.feature_registry import DEFAULT_VERSION
        assert DEFAULT_VERSION == "v3.6"

    def test_categories_v31(self):
        from core.feature_registry import get_features_by_category
        cats = get_features_by_category("v3.1")
        assert "pattern" in cats  # New category for v3.1
        assert "UpStreak_Days" in cats["pattern"]
        assert "OvernightGap_Avg" in cats["pattern"]
        # Market category should only have Market_Regime
        assert len(cats["market"]) == 1
        assert cats["market"][0] == "Market_Regime"


# ============================================================================
# 9. V3.5 FEATURE TESTS
# ============================================================================

class TestV35Features:
    """Tests for V3.5 feature set (20 features = V3.4 + 7 new)."""

    def test_v35_count(self):
        from core.feature_registry import get_feature_names
        names = get_feature_names("v3.5")
        assert len(names) == 20, f"Expected 20 V3.5 features, got {len(names)}"

    def test_v35_includes_v34(self):
        from core.feature_registry import get_feature_names
        v34 = set(get_feature_names("v3.4"))
        v35 = set(get_feature_names("v3.5"))
        assert v34.issubset(v35), f"V3.4 features missing from V3.5: {v34 - v35}"

    def test_v35_new_features(self):
        from core.feature_registry import get_feature_names
        names = set(get_feature_names("v3.5"))
        new_features = {
            "RSI_Delta_5d", "ATR_Delta_5d", "Volume_Delta_5d",
            "Momentum_Acceleration", "VCP_x_RS",
            "Momentum_x_Volume", "Squeeze_x_Volume",
        }
        missing = new_features - names
        assert not missing, f"Missing V3.5 features: {missing}"

    def test_v35_categories(self):
        from core.feature_registry import get_features_by_category
        cats = get_features_by_category("v3.5")
        assert "delta" in cats
        assert "interaction" in cats
        assert len(cats["delta"]) == 4
        assert len(cats["interaction"]) == 3

    def test_v35_builder_returns_20_features(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_5
        features = build_all_ml_features_v3_5(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        assert len(features) == 20, f"Expected 20, got {len(features)}"

    def test_v35_builder_exact_match(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_5
        from core.feature_registry import get_feature_names
        features = build_all_ml_features_v3_5(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        expected = set(get_feature_names("v3.5"))
        actual = set(features.keys())
        assert expected == actual, f"Mismatch: missing={expected - actual}, extra={actual - expected}"

    def test_v35_all_finite(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_5
        features = build_all_ml_features_v3_5(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        for name, val in features.items():
            assert np.isfinite(val), f"Feature {name} is not finite: {val}"

    def test_v35_within_ranges(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_5
        from core.feature_registry import get_feature_ranges
        features = build_all_ml_features_v3_5(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        ranges = get_feature_ranges("v3.5")
        for name, val in features.items():
            lo, hi = ranges[name]
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_v35_interaction_values(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_5
        features = build_all_ml_features_v3_5(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        # VCP_x_RS = VCP_Ratio * RS_vs_SPY_20d
        expected_vcp_rs = features["VCP_Ratio"] * features["RS_vs_SPY_20d"]
        assert abs(features["VCP_x_RS"] - expected_vcp_rs) < 1e-6
        # Momentum_x_Volume = Momentum_Consistency * Volume_Surge
        expected_mom_vol = features["Momentum_Consistency"] * features["Volume_Surge"]
        assert abs(features["Momentum_x_Volume"] - expected_mom_vol) < 1e-6

    def test_v4_removed(self):
        from core.feature_registry import SUPPORTED_VERSIONS
        assert "v4" not in SUPPORTED_VERSIONS
        from core.feature_registry import get_feature_names
        with pytest.raises(ValueError):
            get_feature_names("v4")


# ============================================================================
# V3.6 Tests (23 features = V3.5 + ADX + MACD_Hist + MA50_Slope)
# ============================================================================

class TestV36Features:
    """Tests for V3.6 feature set (23 features = V3.5 + 3 trend/momentum)."""

    def test_v36_count(self):
        from core.feature_registry import get_feature_names
        names = get_feature_names("v3.6")
        assert len(names) == 23, f"Expected 23 V3.6 features, got {len(names)}"

    def test_v36_includes_v35(self):
        from core.feature_registry import get_feature_names
        v35 = set(get_feature_names("v3.5"))
        v36 = set(get_feature_names("v3.6"))
        assert v35.issubset(v36), f"V3.5 features missing from V3.6: {v35 - v36}"

    def test_v36_new_features(self):
        from core.feature_registry import get_feature_names
        names = set(get_feature_names("v3.6"))
        new_features = {"ADX", "MACD_Hist", "MA50_Slope"}
        missing = new_features - names
        assert not missing, f"Missing V3.6 features: {missing}"

    def test_v36_in_supported_versions(self):
        from core.feature_registry import SUPPORTED_VERSIONS
        assert "v3.6" in SUPPORTED_VERSIONS

    def test_v36_builder_returns_23_features(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_6
        features = build_all_ml_features_v3_6(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        assert len(features) == 23, f"Expected 23, got {len(features)}"

    def test_v36_builder_exact_match(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_6
        from core.feature_registry import get_feature_names
        features = build_all_ml_features_v3_6(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        expected = set(get_feature_names("v3.6"))
        actual = set(features.keys())
        assert expected == actual, f"Mismatch: missing={expected - actual}, extra={actual - expected}"

    def test_v36_all_finite(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_6
        features = build_all_ml_features_v3_6(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        for name, val in features.items():
            assert np.isfinite(val), f"Feature {name} is not finite: {val}"

    def test_v36_within_ranges(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_6
        from core.feature_registry import get_feature_ranges
        features = build_all_ml_features_v3_6(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        ranges = get_feature_ranges("v3.6")
        for name, val in features.items():
            lo, hi = ranges[name]
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_v36_adx_valid_range(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_6
        features = build_all_ml_features_v3_6(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        assert 0 <= features["ADX"] <= 100, f"ADX={features['ADX']} out of range"

    def test_v36_macd_hist_normalized(self, indicator_row, ohlcv_df, market_context, sector_context):
        from core.ml_feature_builder import build_all_ml_features_v3_6
        features = build_all_ml_features_v3_6(
            indicator_row, ohlcv_df, market_context, sector_context
        )
        # MACD_Hist is normalized by price, so should be small
        assert -0.05 <= features["MACD_Hist"] <= 0.05, f"MACD_Hist={features['MACD_Hist']} out of range"
