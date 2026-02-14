"""Unit tests for indicator edge cases and uncovered functions.

Covers:
- adx (fully uncovered)
- rsi / atr / macd_line edge cases (downtrend, NaN, short data)
- infer_horizon (stock_scout.py)
- apply_sector_cap (stock_scout.py)
- compute_fund_reliability (core/scoring/fundamental.py)
"""
import numpy as np
import pandas as pd
import pytest

from indicators import rsi, atr, macd_line, adx
from core.scoring.fundamental import compute_fund_reliability


# ══════════  Helpers  ══════════════════════════════════════════════


def _uptrend(n: int = 50) -> pd.Series:
    return pd.Series(range(100, 100 + n), dtype=float)


def _downtrend(n: int = 50) -> pd.Series:
    return pd.Series(range(100 + n, 100, -1), dtype=float)


def _flat(n: int = 50, val: float = 100.0) -> pd.Series:
    return pd.Series([val] * n, dtype=float)


def _ohlc(n: int = 50, *, base: float = 100.0, spread: float = 2.0) -> pd.DataFrame:
    """Generate a simple OHLC DataFrame with consistent spread."""
    close = pd.Series(np.linspace(base, base + n, n), dtype=float)
    return pd.DataFrame({
        "High": close + spread,
        "Low": close - spread,
        "Close": close,
    })


# ══════════  RSI edge cases  ══════════════════════════════════════


class TestRsiEdgeCases:
    def test_downtrend_low_rsi(self):
        vals = rsi(_downtrend(50))
        last = vals.dropna().iloc[-1]
        assert last < 30, f"Downtrend RSI should be low, got {last}"

    def test_constant_series(self):
        vals = rsi(_flat(50))
        last = vals.dropna().iloc[-1]
        # No gain, no loss → RSI ≈ 50 (with epsilon denominator → ~0)
        assert 0 <= last <= 100

    def test_short_series_all_nan(self):
        vals = rsi(pd.Series([100.0, 101.0]))
        # With default period=14 and only 2 datapoints, all should be NaN
        assert vals.dropna().empty

    def test_nan_in_series(self):
        s = pd.Series([100.0] * 20 + [np.nan] + [105.0] * 10)
        vals = rsi(s)
        # Should not raise
        assert len(vals) == len(s)


# ══════════  ATR edge cases  ═════════════════════════════════════


class TestAtrEdgeCases:
    def test_gap_up(self):
        """Previous close outside H/L range → TR should capture gap."""
        df = pd.DataFrame({
            "High": [10.0, 15.0],
            "Low": [9.0, 14.0],
            "Close": [9.5, 14.5],
        })
        tr_vals = atr(df, period=1)
        # Gap from 9.5 to 14-15: TR[1] = max(15-14, |15-9.5|, |14-9.5|) = 5.5
        assert tr_vals.iloc[1] == pytest.approx(5.5)

    def test_constant_spread(self):
        df = _ohlc(30, spread=1.0)
        vals = atr(df, period=14)
        last = vals.dropna().iloc[-1]
        # Spread is constant → ATR ≈ 2.0 (H-L)
        assert 1.5 <= last <= 2.5

    def test_insufficient_data(self):
        df = _ohlc(5, spread=1.0)
        vals = atr(df, period=14)
        assert vals.dropna().empty


# ══════════  MACD edge cases  ════════════════════════════════════


class TestMacdEdgeCases:
    def test_trending_up(self):
        macd_l, sig, hist = macd_line(_uptrend(60))
        # In a strong uptrend, MACD should be positive
        assert macd_l.iloc[-1] > 0

    def test_trending_down(self):
        macd_l, sig, hist = macd_line(_downtrend(60))
        assert macd_l.iloc[-1] < 0

    def test_custom_periods(self):
        macd_l, sig, hist = macd_line(_uptrend(60), fast=5, slow=10, signal=3)
        assert len(macd_l) == 60

    def test_histogram_is_macd_minus_signal(self):
        macd_l, sig, hist = macd_line(_uptrend(60))
        diff = (macd_l - sig).iloc[-1]
        assert hist.iloc[-1] == pytest.approx(diff)


# ══════════  ADX (fully uncovered)  ══════════════════════════════


class TestAdx:
    def test_returns_dataframe_columns(self):
        df = _ohlc(50)
        result = adx(df)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"ADX", "PLUS_DI", "MINUS_DI"}

    def test_strong_trend_high_adx(self):
        """Strong monotone uptrend → high ADX."""
        n = 100
        close = pd.Series(np.linspace(50, 150, n), dtype=float)
        df = pd.DataFrame({
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
        })
        result = adx(df, period=14)
        last_adx = result["ADX"].dropna().iloc[-1]
        assert last_adx > 30, f"Strong trend should give ADX > 30, got {last_adx}"

    def test_flat_market_low_adx(self):
        n = 100
        close = pd.Series([100.0] * n)
        df = pd.DataFrame({
            "High": close + 0.01,
            "Low": close - 0.01,
            "Close": close,
        })
        result = adx(df, period=14)
        last_adx = result["ADX"].dropna().iloc[-1]
        assert last_adx < 25, f"Flat market should give ADX < 25, got {last_adx}"

    def test_plus_di_exceeds_minus_di_in_uptrend(self):
        n = 100
        close = pd.Series(np.linspace(50, 150, n), dtype=float)
        df = pd.DataFrame({
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
        })
        result = adx(df, period=14)
        last = result.dropna().iloc[-1]
        assert last["PLUS_DI"] > last["MINUS_DI"]

    def test_values_non_negative(self):
        df = _ohlc(60)
        result = adx(df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all().all()

    def test_short_data_all_nan(self):
        df = _ohlc(5)
        result = adx(df, period=14)
        assert result["ADX"].dropna().empty


# ══════════  infer_horizon  ══════════════════════════════════════


class TestInferHorizon:
    @staticmethod
    def _make_row(**kwargs):
        defaults = {"RSI": 50.0, "Near52w": 50.0, "Score": 50.0}
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_short_horizon(self):
        from stock_scout import infer_horizon
        row = self._make_row(RSI=65.0, Near52w=95.0, Score=70.0)
        assert "Short" in infer_horizon(row)

    def test_medium_horizon(self):
        from stock_scout import infer_horizon
        row = self._make_row(RSI=30.0, Score=70.0)
        assert "Medium (3-6" in infer_horizon(row)

    def test_medium_long_horizon(self):
        from stock_scout import infer_horizon
        row = self._make_row(RSI=50.0, Score=75.0)
        assert "Medium-Long" in infer_horizon(row)

    def test_long_horizon_default(self):
        from stock_scout import infer_horizon
        row = self._make_row(RSI=50.0, Score=40.0)
        assert "Long" in infer_horizon(row)

    def test_nan_rsi_falls_through(self):
        from stock_scout import infer_horizon
        row = self._make_row(RSI=np.nan, Near52w=95.0, Score=50.0)
        assert "Long" in infer_horizon(row)


# ══════════  apply_sector_cap  ═══════════════════════════════════


class TestApplySectorCap:
    def test_caps_sectors(self):
        from stock_scout import apply_sector_cap, CONFIG
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C", "D", "E", "F"],
            "Sector": ["Tech", "Tech", "Tech", "Fin", "Fin", "Health"],
            "Score": [90, 80, 70, 60, 50, 40],
        })
        old_val = CONFIG.get("SECTOR_CAP_ENABLED")
        try:
            CONFIG["SECTOR_CAP_ENABLED"] = True
            result = apply_sector_cap(df, cap=2)
            assert len(result[result["Sector"] == "Tech"]) <= 2
            assert len(result[result["Sector"] == "Fin"]) <= 2
            assert len(result[result["Sector"] == "Health"]) <= 1  # only 1 exists
        finally:
            if old_val is not None:
                CONFIG["SECTOR_CAP_ENABLED"] = old_val

    def test_disabled_returns_all(self):
        from stock_scout import apply_sector_cap, CONFIG
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Sector": ["Tech", "Tech", "Tech"],
        })
        old_val = CONFIG.get("SECTOR_CAP_ENABLED")
        try:
            CONFIG["SECTOR_CAP_ENABLED"] = False
            result = apply_sector_cap(df, cap=1)
            assert len(result) == 3
        finally:
            if old_val is not None:
                CONFIG["SECTOR_CAP_ENABLED"] = old_val


# ══════════  compute_fund_reliability  ═══════════════════════════


class TestComputeFundReliability:
    def test_zero_sources(self):
        assert compute_fund_reliability(0) == 0.0

    def test_one_source(self):
        assert compute_fund_reliability(1) == pytest.approx(0.33)

    def test_two_sources(self):
        assert compute_fund_reliability(2) == pytest.approx(0.66)

    def test_three_sources(self):
        assert compute_fund_reliability(3) == 1.0

    def test_five_sources(self):
        assert compute_fund_reliability(5) == 1.0

    def test_negative(self):
        assert compute_fund_reliability(-1) == 0.0

    def test_non_numeric(self):
        assert compute_fund_reliability("abc") == 0.0

    def test_none(self):
        assert compute_fund_reliability(None) == 0.0
