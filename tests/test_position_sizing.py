"""Unit tests for core/position_sizing.py pure functions.

Covers: compute_smart_position_size, categorize_risk, check_portfolio_concentration.
"""
import pytest
from core.position_sizing import (
    compute_smart_position_size,
    categorize_risk,
    check_portfolio_concentration,
)


# ── compute_smart_position_size ─────────────────────────────────────


class TestComputeSmartPositionSize:
    def test_low_risk_high_ml(self):
        result = compute_smart_position_size(2.0, 0.8, 100_000)
        assert result["position_pct"] > 0
        assert result["dollar_amount"] > 0
        assert result["stop_loss_pct"] > 0
        assert isinstance(result["warnings"], list)

    def test_high_risk_low_ml(self):
        result = compute_smart_position_size(9.0, 0.3, 100_000)
        assert result["position_pct"] < 0.05
        assert len(result["warnings"]) >= 2  # both high risk + low ML

    def test_moderate_risk(self):
        result = compute_smart_position_size(5.0, 0.6, 100_000)
        assert 0 < result["position_pct"] <= 0.10

    def test_max_position_respected(self):
        result = compute_smart_position_size(1.0, 0.9, 100_000, max_position_pct=0.05)
        assert result["position_pct"] <= 0.05

    def test_dollar_matches_pct(self):
        portfolio = 200_000
        result = compute_smart_position_size(3.0, 0.6, portfolio)
        expected = portfolio * result["position_pct"]
        assert result["dollar_amount"] == pytest.approx(expected)

    def test_contains_risk_category(self):
        result = compute_smart_position_size(3.0, 0.6, 50_000)
        assert "risk_category" in result

    def test_stop_loss_tighter_at_high_risk(self):
        low_risk = compute_smart_position_size(2.0, 0.6, 100_000)
        high_risk = compute_smart_position_size(9.0, 0.6, 100_000)
        assert high_risk["stop_loss_pct"] < low_risk["stop_loss_pct"]

    def test_take_profit_present(self):
        result = compute_smart_position_size(5.0, 0.5, 100_000)
        assert result["take_profit_pct"] == 0.15


# ── categorize_risk ─────────────────────────────────────────────────


class TestCategorizeRisk:
    def test_low_risk(self):
        cat = categorize_risk(3.0, 0.7)
        assert "LOW" in cat

    def test_medium_risk(self):
        cat = categorize_risk(5.0, 0.5)
        assert "MEDIUM" in cat

    def test_high_risk(self):
        cat = categorize_risk(7.0, 0.4)
        assert "HIGH" in cat

    def test_very_high_risk(self):
        cat = categorize_risk(9.0, 0.2)
        assert "VERY HIGH" in cat


# ── check_portfolio_concentration ───────────────────────────────────


class TestCheckPortfolioConcentration:
    def test_no_warnings_small_position(self):
        result = check_portfolio_concentration(
            {"AAPL": 10_000, "GOOG": 10_000}, "MSFT", 1_000
        )
        assert result["warnings"] == []
        assert result["concentration_pct"] < 0.15

    def test_warning_on_large_position(self):
        result = check_portfolio_concentration(
            {"AAPL": 10_000}, "GOOG", 10_000
        )
        # 10k / 20k = 50% > 15%
        assert len(result["warnings"]) >= 1
        assert result["concentration_pct"] > 0.15

    def test_empty_portfolio(self):
        result = check_portfolio_concentration({}, "AAPL", 10_000)
        # 10k / 10k = 100%
        assert result["concentration_pct"] == pytest.approx(1.0)

    def test_recommended_max(self):
        result = check_portfolio_concentration(
            {"AAPL": 85_000}, "GOOG", 15_000
        )
        assert result["recommended_max"] == pytest.approx(100_000 * 0.15)
