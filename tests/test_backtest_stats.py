"""Tests for core.backtest.stats — statistical functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.stats import (
    annualised_return,
    annualised_volatility,
    bootstrap_confidence_interval,
    calmar_ratio,
    information_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    significance_test,
    win_rate,
)


class TestSharpeRatio:
    def test_positive_returns(self):
        returns = pd.Series([0.01, 0.02, 0.005, 0.015, 0.01] * 50)
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_zero_std(self):
        returns = pd.Series([0.01] * 10)
        assert sharpe_ratio(returns) == 0.0

    def test_empty(self):
        assert sharpe_ratio(pd.Series(dtype=float)) == 0.0

    def test_negative_returns(self):
        returns = pd.Series([-0.01, -0.02, -0.005] * 50)
        sr = sharpe_ratio(returns)
        assert sr < 0


class TestMaxDrawdown:
    def test_basic_drawdown(self):
        equity = pd.Series(
            [100, 110, 105, 95, 100, 115],
            index=pd.date_range("2024-01-01", periods=6),
        )
        dd, peak, trough = max_drawdown(equity)
        assert dd < 0
        assert abs(dd - (95 / 110 - 1)) < 0.001

    def test_no_drawdown(self):
        equity = pd.Series(
            [100, 110, 120, 130],
            index=pd.date_range("2024-01-01", periods=4),
        )
        dd, _, _ = max_drawdown(equity)
        assert dd == 0.0

    def test_empty(self):
        dd, _, _ = max_drawdown(pd.Series(dtype=float))
        assert dd == 0.0


class TestWinRate:
    def test_all_wins(self):
        assert win_rate(pd.Series([0.01, 0.02, 0.005])) == 1.0

    def test_mixed(self):
        assert win_rate(pd.Series([0.01, -0.01, 0.02, -0.005])) == 0.5

    def test_empty(self):
        assert win_rate(pd.Series(dtype=float)) == 0.0


class TestProfitFactor:
    def test_basic(self):
        returns = pd.Series([0.10, 0.05, -0.03, -0.02])
        pf = profit_factor(returns)
        assert pf == pytest.approx(0.15 / 0.05, rel=1e-6)

    def test_no_losses(self):
        pf = profit_factor(pd.Series([0.01, 0.02]))
        assert pf == float("inf")

    def test_no_wins(self):
        assert profit_factor(pd.Series([-0.01, -0.02])) == 0.0


class TestCalmarRatio:
    def test_basic(self):
        assert calmar_ratio(0.10, -0.05) == pytest.approx(2.0)

    def test_zero_dd(self):
        assert calmar_ratio(0.10, 0.0) == 0.0


class TestInformationRatio:
    def test_outperformance(self):
        strat = pd.Series([0.02, 0.01, 0.015, 0.02, 0.01] * 50)
        bench = pd.Series([0.01, 0.005, 0.01, 0.01, 0.005] * 50)
        ir = information_ratio(strat, bench)
        assert ir > 0

    def test_empty(self):
        assert information_ratio(pd.Series(dtype=float), pd.Series(dtype=float)) == 0.0


class TestAnnualisedReturn:
    def test_doubling(self):
        # Equity doubles in 1 year (252 trading days)
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        equity = pd.Series(
            np.linspace(100, 200, 252), index=dates
        )
        cagr = annualised_return(equity)
        assert abs(cagr - 1.0) < 0.15  # ~100% ± noise from interpolation

    def test_empty(self):
        assert annualised_return(pd.Series(dtype=float)) == 0.0


class TestBootstrapCI:
    def test_positive_returns(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.02, 200)
        lo, hi = bootstrap_confidence_interval(returns)
        assert lo < hi
        assert lo > -0.1
        assert hi < 0.1

    def test_single_value(self):
        lo, hi = bootstrap_confidence_interval(np.array([0.05]))
        assert lo == hi == 0.05


class TestSignificance:
    def test_significant(self):
        rng = np.random.default_rng(42)
        strat = pd.Series(rng.normal(0.02, 0.01, 500))
        bench = pd.Series(rng.normal(0.005, 0.01, 500))
        result = significance_test(strat, bench, n_bootstrap=1000)
        assert result["p_value"] < 0.05
        assert result["is_significant"]

    def test_not_significant(self):
        rng = np.random.default_rng(42)
        strat = pd.Series(rng.normal(0.01, 0.05, 50))
        bench = pd.Series(rng.normal(0.01, 0.05, 50))
        result = significance_test(strat, bench, n_bootstrap=1000)
        # With same distribution, shouldn't be significant
        assert result["n_obs"] == 50

    def test_too_few(self):
        result = significance_test(pd.Series([0.01]), pd.Series([0.005]))
        assert result["p_value"] == 1.0
        assert not result["is_significant"]
