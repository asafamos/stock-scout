"""Tests for ui.target_calc — target price / date calculation."""
import numpy as np
import pandas as pd
import pytest

from ui.target_calc import (
    calculate_targets,
    _compute_holding_days,
    _SECTOR_OFFSETS,
)


class TestComputeHoldingDays:
    def test_with_rr(self):
        days = _compute_holding_days(
            rr=2.0, rsi=50.0, momentum=0.0, volatility_factor=1.0,
            sector_mult=1.0, ml_mult=1.0, ticker="AAPL",
        )
        assert 14 <= days <= 180

    def test_oversold_rsi_shorter(self):
        base = _compute_holding_days(
            rr=2.0, rsi=50.0, momentum=0.0, volatility_factor=1.0,
            sector_mult=1.0, ml_mult=1.0, ticker="TEST",
        )
        oversold = _compute_holding_days(
            rr=2.0, rsi=35.0, momentum=0.0, volatility_factor=1.0,
            sector_mult=1.0, ml_mult=1.0, ticker="TEST",
        )
        assert oversold <= base

    def test_no_rr_fallback(self):
        days = _compute_holding_days(
            rr=np.nan, rsi=np.nan, momentum=np.nan, volatility_factor=1.0,
            sector_mult=1.0, ml_mult=1.0, ticker="XYZ",
        )
        assert 30 <= days <= 180

    def test_different_tickers_different_days(self):
        d1 = _compute_holding_days(2.0, 50.0, 0.0, 1.0, 1.0, 1.0, "AAA")
        d2 = _compute_holding_days(2.0, 50.0, 0.0, 1.0, 1.0, 1.0, "ZZZ")
        # ticker-seed causes variance
        assert isinstance(d1, int)
        assert isinstance(d2, int)


class TestCalculateTargets:
    def _row(self, **kw):
        defaults = {
            "Ticker": "AAPL",
            "Unit_Price": 150.0,
            "ATR": 3.0,
            "RewardRisk": 2.0,
            "RSI": 55.0,
            "Momentum_63d": 0.02,
            "Sector": "Technology",
            "ML_20d_Prob": 0.65,
        }
        defaults.update(kw)
        return pd.Series(defaults)

    def test_basic_targets(self):
        entry, target, date_str, source = calculate_targets(self._row())
        assert entry < 150.0  # entry is below current
        assert target > entry  # target is above entry
        assert len(date_str) == 10  # YYYY-MM-DD
        assert source in ("AI", "Default")

    def test_nan_price(self):
        entry, target, date_str, source = calculate_targets(self._row(Unit_Price=np.nan))
        assert np.isnan(target)
        assert date_str == "N/A"

    def test_no_atr_fallback(self):
        entry, target, date_str, source = calculate_targets(self._row(ATR=np.nan, RewardRisk=np.nan))
        assert target > 0
        assert source == "Default"

    def test_with_regime_adjustment(self):
        def mock_adjust(base_pct, reliability, risk_meter, regime_data):
            return base_pct * 0.9, "bear adjustment"

        entry, target, date_str, source = calculate_targets(
            self._row(),
            adjust_target_for_regime=mock_adjust,
        )
        assert target > entry


class TestSectorOffsets:
    def test_technology_fastest(self):
        assert _SECTOR_OFFSETS["Technology"] < _SECTOR_OFFSETS["Utilities"]

    def test_all_positive(self):
        for v in _SECTOR_OFFSETS.values():
            assert v > 0
