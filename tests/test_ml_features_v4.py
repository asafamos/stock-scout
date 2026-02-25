"""Tests for V4 ML features — 72-feature builder, registry, and rank targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestFeatureRegistryV4:
    def test_v4_feature_count(self):
        from core.feature_registry import get_feature_names
        names = get_feature_names("v4")
        assert len(names) == 72

    def test_v4_includes_v31(self):
        from core.feature_registry import get_feature_names
        v31 = set(get_feature_names("v3.1"))
        v4 = set(get_feature_names("v4"))
        assert v31.issubset(v4), f"Missing from v4: {v31 - v4}"

    def test_v4_defaults(self):
        from core.feature_registry import get_feature_defaults
        defaults = get_feature_defaults("v4")
        assert len(defaults) == 72
        # Check some new fundamental features have sensible defaults
        assert defaults["Fund_Quality_Score"] == 50.0
        assert defaults["MarketCap_Log"] == 10.0
        assert defaults["RSI_Rank"] == 0.5

    def test_v4_ranges(self):
        from core.feature_registry import get_feature_ranges
        ranges = get_feature_ranges("v4")
        assert len(ranges) == 72
        # Check fundamental score range
        assert ranges["Fund_Quality_Score"] == (0, 100)
        assert ranges["Debt_Risk"] == (0, 1)

    def test_v4_categories(self):
        from core.feature_registry import get_features_by_category
        cats = get_features_by_category("v4")
        assert "fundamental" in cats
        assert "rank" in cats
        assert "delta" in cats
        assert "interaction" in cats
        assert len(cats["fundamental"]) == 15
        assert len(cats["rank"]) == 8
        assert len(cats["delta"]) == 6
        assert len(cats["interaction"]) == 4

    def test_validate_features_v4(self):
        from core.feature_registry import get_feature_names, validate_features
        names = get_feature_names("v4")
        df = pd.DataFrame({name: [0.0] for name in names})
        valid, missing = validate_features(df, "v4")
        assert valid
        assert missing == []


class TestV4FeatureBuilder:
    @pytest.fixture
    def sample_data(self):
        """Create minimal OHLCV data and row for V4 feature building."""
        np.random.seed(42)
        n = 252
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10.0)  # ensure positive
        df = pd.DataFrame({
            "Open": close * (1 + np.random.randn(n) * 0.005),
            "High": close * (1 + abs(np.random.randn(n) * 0.01)),
            "Low": close * (1 - abs(np.random.randn(n) * 0.01)),
            "Close": close,
            "Volume": np.random.randint(100_000, 10_000_000, n).astype(float),
        }, index=dates)

        # Build indicators
        from core.indicators import build_technical_indicators
        df_ind = build_technical_indicators(df)
        row = df_ind.iloc[-1].copy()
        row["Ticker"] = "TEST"
        return row, df_ind

    def test_v4_builds_72_features(self, sample_data):
        from core.ml_feature_builder import build_all_ml_features_v4
        row, df_hist = sample_data
        features = build_all_ml_features_v4(row, df_hist)
        assert len(features) == 72

    def test_v4_all_finite(self, sample_data):
        from core.ml_feature_builder import build_all_ml_features_v4
        row, df_hist = sample_data
        features = build_all_ml_features_v4(row, df_hist)
        for name, val in features.items():
            assert np.isfinite(val), f"Feature {name} is not finite: {val}"

    def test_v4_with_fundamental_data(self, sample_data):
        from core.ml_feature_builder import build_all_ml_features_v4
        row, df_hist = sample_data
        fund_data = {
            "quality_score": 75.0,
            "growth_score": 60.0,
            "valuation_score": 40.0,
            "rev_g_yoy": 0.15,
            "eps_g_yoy": 0.25,
            "debt_equity": 0.8,
            "market_cap": 5e10,
            "coverage_pct": 0.75,
        }
        features = build_all_ml_features_v4(
            row, df_hist, fundamental_data=fund_data
        )
        assert features["Fund_Quality_Score"] == 75.0
        assert features["Fund_Growth_Score"] == 60.0
        assert features["Revenue_Growth_Bucket"] == 2.0  # moderate (0.15)
        assert features["EPS_Growth_Bucket"] == 3.0  # strong (0.25)
        assert features["Fund_Coverage"] == 0.75

    def test_v4_with_universe_stats(self, sample_data):
        from core.ml_feature_builder import build_all_ml_features_v4
        row, df_hist = sample_data
        stats = {
            "rsi_pctl": 0.75,
            "atr_pctl": 0.30,
            "momentum_pctl": 0.85,
        }
        features = build_all_ml_features_v4(
            row, df_hist, universe_stats=stats
        )
        assert features["RSI_Rank"] == 0.75
        assert features["ATR_Rank"] == 0.30
        assert features["Momentum_Rank"] == 0.85
        # Unset ranks default to 0.5
        assert features["Volume_Rank"] == 0.5

    def test_v4_interaction_features(self, sample_data):
        from core.ml_feature_builder import build_all_ml_features_v4
        row, df_hist = sample_data
        features = build_all_ml_features_v4(row, df_hist)
        # VCP_x_RS = VCP_Ratio * RS_vs_SPY_20d
        expected = features["VCP_Ratio"] * features["RS_vs_SPY_20d"]
        assert abs(features["VCP_x_RS"] - expected) < 0.01

    def test_v4_backward_compat_with_v31(self, sample_data):
        """V4 should produce same v3.1 features as the v3.1 builder."""
        from core.ml_feature_builder import build_all_ml_features_v3, build_all_ml_features_v4
        row, df_hist = sample_data
        v31 = build_all_ml_features_v3(row, df_hist)
        v4 = build_all_ml_features_v4(row, df_hist)
        # All v3.1 features should exist in v4 with same values
        for name in v31:
            assert name in v4, f"V3.1 feature {name} missing from V4"
            assert abs(v31[name] - v4[name]) < 0.001, (
                f"Feature {name}: v3.1={v31[name]}, v4={v4[name]}"
            )


class TestRankBasedTargets:
    def test_ranked_labels(self):
        from core.ml_targets import make_label_20d_ranked
        np.random.seed(42)
        n = 200
        returns = pd.Series(np.random.randn(n) * 0.05)
        dates = pd.Series(["2024-01-01"] * 100 + ["2024-02-01"] * 100)
        labels = make_label_20d_ranked(returns, dates)
        # Should have labels 0, 1, and NaN
        assert labels.dropna().isin([0, 1]).all()
        # Top 20% should be ~20 per date, bottom 40% should be ~40 per date
        label_1_count = (labels == 1).sum()
        label_0_count = (labels == 0).sum()
        assert label_1_count > 0
        assert label_0_count > 0
        # More 0s than 1s (40% vs 20%)
        assert label_0_count > label_1_count

    def test_ranked_labels_few_stocks(self):
        """With < 20 stocks per date, all should be NaN."""
        from core.ml_targets import make_label_20d_ranked
        returns = pd.Series([0.01, 0.02, -0.01] * 5)
        dates = pd.Series(["2024-01-01"] * 15)
        labels = make_label_20d_ranked(returns, dates)
        assert labels.isna().all()

    def test_ranked_labels_class_balance(self):
        """In different market regimes, class balance should be consistent."""
        from core.ml_targets import make_label_20d_ranked
        np.random.seed(42)
        # Bull market: all returns positive
        bull_rets = pd.Series(np.random.uniform(0.0, 0.20, 50))
        bull_dates = pd.Series(["2024-01-01"] * 50)
        bull_labels = make_label_20d_ranked(bull_rets, bull_dates)

        # Bear market: all returns negative
        bear_rets = pd.Series(np.random.uniform(-0.20, 0.0, 50))
        bear_dates = pd.Series(["2024-02-01"] * 50)
        bear_labels = make_label_20d_ranked(bear_rets, bear_dates)

        # Both should have similar class balance (~20% winner, ~40% loser)
        for labels, name in [(bull_labels, "bull"), (bear_labels, "bear")]:
            n1 = (labels == 1).sum()
            n0 = (labels == 0).sum()
            assert n1 > 0, f"{name}: no winners"
            assert n0 > 0, f"{name}: no losers"
            # ~20% should be winners
            assert abs(n1 / 50 - 0.20) < 0.10, f"{name}: winner rate off"
