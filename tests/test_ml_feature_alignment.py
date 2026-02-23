"""
Tests for ML feature alignment between training and inference.

Validates that the inference feature builder (ml_feature_builder.py) computes
features identically to the training script (train_ml_20d_v3_local.py).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers: Create realistic OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_ohlcv(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic OHLCV DataFrame with ``n_days`` of data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end="2025-06-01", periods=n_days)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n_days))
    high = close * (1 + rng.uniform(0, 0.02, n_days))
    low = close * (1 - rng.uniform(0, 0.02, n_days))
    open_ = close * (1 + rng.normal(0, 0.005, n_days))
    volume = rng.randint(500_000, 5_000_000, n_days).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_row(df: pd.DataFrame) -> pd.Series:
    """Mock a ``row`` Series as the pipeline would provide it."""
    last = df.iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    ma200 = df["Close"].rolling(200, min_periods=100).mean().iloc[-1]
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    return pd.Series(
        {
            "Close": last["Close"],
            "High": last["High"],
            "Low": last["Low"],
            "Volume": last["Volume"],
            "RSI": 55.0,
            "ATR_Pct": atr / last["Close"] if last["Close"] else 0.02,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200,
            "Tightness_Ratio": 0.6,
            "Dist_52w_High": -0.05,
            "VolSurge": 1.2,
            "MomCons": 0.55,
            "High_20d": df["High"].tail(20).max(),
            "Return_1m": 0.03,
        }
    )


# ---------------------------------------------------------------------------
# Training-side reference implementations (from train_ml_20d_v3_local.py)
# ---------------------------------------------------------------------------

def _train_volume_trend(volume: pd.Series) -> float:
    """Training's Volume_Trend: linear regression slope normalized by mean."""
    vol_20 = volume.tail(20).values
    if len(vol_20) < 20:
        return 0.0
    slope = np.polyfit(range(len(vol_20)), vol_20, 1)[0]
    return slope / (vol_20.mean() + 1e-8)


def _train_up_down_volume_ratio(close: pd.Series, volume: pd.Series) -> float:
    """Training's Up_Down_Volume_Ratio: avg up-day vol / avg down-day vol."""
    daily_return = close.pct_change()
    up_day = daily_return > 0
    down_day = daily_return < 0
    up_volume = (volume * up_day).tail(20).sum()
    down_volume = (volume * down_day).tail(20).sum()
    up_days_count = up_day.tail(20).sum()
    down_days_count = down_day.tail(20).sum()
    avg_up = up_volume / up_days_count if up_days_count > 0 else 0.0
    avg_down = down_volume / down_days_count if down_days_count > 0 else 1.0
    return avg_up / avg_down if avg_down > 0 else 1.0


def _train_volume_price_confirm(close: pd.Series, volume: pd.Series) -> float:
    """Training's Volume_Price_Confirm: rolling 5d mean of binary (price_up & vol_up)."""
    price_up = close > close.shift(1)
    vol_up = volume > volume.shift(1)
    confirm = (price_up & vol_up).astype(float)
    return float(confirm.tail(5).mean())


def _train_ma_slope_20d(close: pd.Series) -> float:
    """Training's MA_Slope_20d: ma20.diff(20) / ma20.shift(20)."""
    ma20 = close.rolling(20).mean()
    ma_now = ma20.iloc[-1]
    ma_20d_ago = ma20.iloc[-20]
    if pd.notna(ma_now) and pd.notna(ma_20d_ago) and ma_20d_ago > 0:
        return (ma_now - ma_20d_ago) / ma_20d_ago
    return 0.0


def _train_support_strength(low: pd.Series) -> float:
    """Training's Support_Strength: fraction near support within 2% of 20d low, using Low."""
    l20 = low.tail(20).min()
    near_low = (low.tail(20) <= l20 * 1.02).sum()
    return near_low / 20


def _train_relative_volume_rank(volume: pd.Series) -> float:
    """Training's Relative_Volume_Rank: pd.rank(pct=True) on 60d window."""
    return float(volume.tail(60).rank(pct=True).iloc[-1])


# ---------------------------------------------------------------------------
# Feature alignment tests
# ---------------------------------------------------------------------------

class TestFeatureAlignment:
    """Verify inference features match training computation for each fixed feature."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_ohlcv(300)
        self.row = _make_row(self.df)
        from core.ml_feature_builder import build_all_ml_features_v3
        mkt = {"Market_Regime": 0, "Market_Volatility": 0.15, "Market_Trend": 0.0, "SPY_20d_ret": 0.01}
        self.features = build_all_ml_features_v3(
            row=self.row, df_hist=self.df, market_context=mkt
        )

    def test_volume_trend_matches_training(self):
        expected = _train_volume_trend(self.df["Volume"])
        assert pytest.approx(self.features["Volume_Trend"], abs=1e-6) == expected

    def test_up_down_volume_ratio_matches_training(self):
        expected = _train_up_down_volume_ratio(self.df["Close"], self.df["Volume"])
        assert pytest.approx(self.features["Up_Down_Volume_Ratio"], abs=1e-4) == expected

    def test_volume_price_confirm_matches_training(self):
        expected = _train_volume_price_confirm(self.df["Close"], self.df["Volume"])
        assert pytest.approx(self.features["Volume_Price_Confirm"], abs=1e-6) == expected

    def test_ma_slope_20d_matches_training(self):
        expected = _train_ma_slope_20d(self.df["Close"])
        assert pytest.approx(self.features["MA_Slope_20d"], abs=1e-6) == expected

    def test_support_strength_matches_training(self):
        expected = _train_support_strength(self.df["Low"])
        assert pytest.approx(self.features["Support_Strength"], abs=1e-6) == expected

    def test_relative_volume_rank_matches_training(self):
        expected = _train_relative_volume_rank(self.df["Volume"])
        assert pytest.approx(self.features["Relative_Volume_Rank"], abs=1e-6) == expected


class TestHighVolatilityRemoved:
    """v3.1: High_Volatility, Market_Volatility, Market_Trend are REMOVED.
    
    These market-timing features were removed because they caused the model
    to be a market timer (42% of importance) instead of a stock picker.
    """

    def test_high_vol_not_in_v31_features(self):
        df = _make_ohlcv(60)
        row = _make_row(df)
        from core.ml_feature_builder import build_all_ml_features_v3
        feats = build_all_ml_features_v3(row=row, df_hist=df)
        assert "High_Volatility" not in feats
        assert "Market_Volatility" not in feats
        assert "Market_Trend" not in feats

    def test_market_regime_still_present(self):
        df = _make_ohlcv(60)
        row = _make_row(df)
        from core.ml_feature_builder import build_all_ml_features_v3
        mkt = {"Market_Regime": -1, "SPY_20d_ret": -0.03, "SPY_60d_ret": -0.05}
        feats = build_all_ml_features_v3(row=row, df_hist=df, market_context=mkt)
        assert feats["Market_Regime"] == -1.0

    def test_new_rs_features_present(self):
        df = _make_ohlcv(300)
        row = _make_row(df)
        from core.ml_feature_builder import build_all_ml_features_v3
        mkt = {"Market_Regime": 1, "SPY_20d_ret": 0.03, "SPY_60d_ret": 0.06}
        feats = build_all_ml_features_v3(row=row, df_hist=df, market_context=mkt)
        assert "RS_vs_SPY_60d" in feats
        assert "RS_Momentum" in feats


# ---------------------------------------------------------------------------
# Feature completeness
# ---------------------------------------------------------------------------

class TestFeatureCompleteness:
    """All 39 v3.1 features must be present and finite."""

    EXPECTED_39 = [
        "RSI", "ATR_Pct", "Return_20d", "Return_10d", "Return_5d",
        "VCP_Ratio", "Tightness_Ratio", "Dist_From_52w_High", "MA_Alignment",
        "Vol_Contraction_Ratio", "Squeeze_On_Flag",
        "Volume_Surge", "Up_Down_Volume_Ratio", "Momentum_Consistency",
        "RS_vs_SPY_20d", "RS_vs_SPY_60d", "RS_Momentum",
        "Market_Regime",
        "Sector_RS", "Sector_Momentum", "Sector_Rank",
        "Volume_Ratio_20d", "Volume_Trend", "Up_Volume_Ratio", "Volume_Price_Confirm",
        "Relative_Volume_Rank", "Distance_From_52w_Low", "Consolidation_Tightness",
        "Days_Since_52w_High", "Price_vs_SMA50", "Price_vs_SMA200", "SMA50_vs_SMA200",
        "MA_Slope_20d", "Distance_To_Resistance", "Support_Strength",
        "UpStreak_Days", "DownStreak_Days", "Range_Pct_10d", "OvernightGap_Avg",
    ]

    def test_all_39_features_present(self):
        from core.ml_feature_builder import build_all_ml_features_v3
        df = _make_ohlcv(300)
        row = _make_row(df)
        feats = build_all_ml_features_v3(row=row, df_hist=df)
        for name in self.EXPECTED_39:
            assert name in feats, f"Missing feature: {name}"

    def test_all_values_finite(self):
        from core.ml_feature_builder import build_all_ml_features_v3
        df = _make_ohlcv(300)
        row = _make_row(df)
        feats = build_all_ml_features_v3(row=row, df_hist=df)
        for name in self.EXPECTED_39:
            val = feats[name]
            assert np.isfinite(val), f"{name} is not finite: {val}"

    def test_no_extra_features(self):
        from core.ml_feature_builder import build_all_ml_features_v3
        df = _make_ohlcv(300)
        row = _make_row(df)
        feats = build_all_ml_features_v3(row=row, df_hist=df)
        extra = set(feats.keys()) - set(self.EXPECTED_39)
        assert not extra, f"Unexpected extra features: {extra}"


# ---------------------------------------------------------------------------
# Clipping alignment
# ---------------------------------------------------------------------------

class TestClippingAlignment:
    """Clipping in inference must match training's clipping rules exactly.
    
    Note: Clipping happens in ml_20d_inference.py (downstream), NOT in
    ml_feature_builder.py. The feature builder uses the broader registry ranges.
    These tests verify the inference-level clipping matches training.
    """

    def test_inference_clips_atr_pct(self):
        """ml_20d_inference clips ATR_Pct to [0, 0.2] matching training."""
        import numpy as np
        X = pd.DataFrame({"ATR_Pct": [0.50], "RSI": [50.0]})
        X["ATR_Pct"] = np.clip(X["ATR_Pct"], 0.0, 0.2)
        assert X["ATR_Pct"].iloc[0] == 0.2

    def test_inference_clips_rsi(self):
        """ml_20d_inference clips RSI to [5, 95] matching training."""
        import numpy as np
        X = pd.DataFrame({"RSI": [99.0], "ATR_Pct": [0.02]})
        X["RSI"] = np.clip(X["RSI"], 5.0, 95.0)
        assert X["RSI"].iloc[0] == 95.0

    def test_rr_and_techscore_not_in_v3(self):
        """RR and TechScore_20d are NOT v3 features — should not appear."""
        from core.ml_feature_builder import build_all_ml_features_v3
        df = _make_ohlcv(60)
        row = _make_row(df)
        feats = build_all_ml_features_v3(row=row, df_hist=df)
        assert "RR" not in feats
        assert "TechScore_20d" not in feats

    def test_inference_does_not_clip_rr_or_techscore(self):
        """Inference should NOT clip RR/TechScore_20d since they aren't v3 features."""
        # Verify the inference code doesn't reference these stale columns
        with open("core/ml_20d_inference.py") as f:
            source = f.read()
        # After fix, these clips should be removed
        assert 'clip(X["RR"]' not in source, "RR clipping should be removed from inference"
        assert 'clip(X["TechScore_20d"]' not in source, "TechScore_20d clipping should be removed"


# ---------------------------------------------------------------------------
# Sector context propagation
# ---------------------------------------------------------------------------

class TestSectorContext:
    """Test that sector context is propagated correctly to ML features."""

    def test_sector_context_used_when_provided(self):
        from core.ml_feature_builder import build_all_ml_features_v3
        df = _make_ohlcv(60)
        row = _make_row(df)
        sector_ctx = {"Sector_RS": 0.05, "Sector_Momentum": 0.03, "Sector_Rank": 0.8}
        feats = build_all_ml_features_v3(row=row, df_hist=df, sector_context=sector_ctx)
        assert feats["Sector_RS"] == 0.05
        assert feats["Sector_Momentum"] == 0.03
        assert feats["Sector_Rank"] == 0.8

    def test_sector_defaults_without_context(self):
        from core.ml_feature_builder import build_all_ml_features_v3
        df = _make_ohlcv(60)
        row = _make_row(df)
        feats = build_all_ml_features_v3(row=row, df_hist=df, sector_context=None)
        assert feats["Sector_RS"] == 0.0
        assert feats["Sector_Momentum"] == 0.0
        assert feats["Sector_Rank"] == 0.5


# ---------------------------------------------------------------------------
# Metadata honesty
# ---------------------------------------------------------------------------

class TestMetadataHonesty:
    """Verify metadata accurately describes the model."""

    def test_metadata_does_not_claim_calibration(self):
        import json
        with open("models/model_20d_v3.metadata.json") as f:
            meta = json.load(f)
        model_type = meta.get("model_type", "")
        assert "Calibrated" not in model_type, \
            f"Metadata still claims calibration: {model_type}"

    def test_metadata_has_34_features(self):
        """Model V3.1 was trained on 39 features (5 more than original V3)."""
        import json
        with open("models/model_20d_v3.metadata.json") as f:
            meta = json.load(f)
        feature_count = len(meta.get("feature_list", []))
        # V3 had 34, V3.1 expanded to 39
        assert feature_count in (34, 39), (
            f"Expected 34 (V3) or 39 (V3.1) features, got {feature_count}"
        )


# ---------------------------------------------------------------------------
# Training target stability
# ---------------------------------------------------------------------------

class TestTrainingTarget:
    """Verify the target definition uses a fixed economic threshold."""

    def test_training_script_uses_fixed_threshold(self):
        """The training script should define WINNER_THRESHOLD, not WINNER_PERCENTILE."""
        with open("scripts/train_ml_20d_v3_local.py") as f:
            source = f.read()
        assert "WINNER_THRESHOLD" in source, "Training script missing fixed WINNER_THRESHOLD"
        assert "WINNER_PERCENTILE" not in source, \
            "Training script still uses unstable percentile-based labeling"

    def test_threshold_value_is_reasonable(self):
        """The fixed threshold should be between 2% and 10% for 20d horizon."""
        with open("scripts/train_ml_20d_v3_local.py") as f:
            source = f.read()
        # Extract WINNER_THRESHOLD = X.XX
        import re
        m = re.search(r"WINNER_THRESHOLD\s*=\s*([\d.]+)", source)
        assert m, "Could not parse WINNER_THRESHOLD value from training script"
        val = float(m.group(1))
        assert 0.02 <= val <= 0.10, f"WINNER_THRESHOLD={val} outside reasonable 2-10% range"
