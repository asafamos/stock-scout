"""Tests for ui.card_helpers — extracted card rendering utilities."""
import numpy as np
import pandas as pd
import pytest

from ui.card_helpers import (
    to_float,
    normalize_prob,
    ml_badge,
    get_ml_prob_from_row,
    risk_class,
    headline_story,
    fmt_num,
    get_reliability_band,
    get_reliability_components,
)


# ── to_float ─────────────────────────────────────────────────────────

class TestToFloat:
    def test_valid_int(self):
        assert to_float(42) == 42.0

    def test_valid_float(self):
        assert to_float(3.14) == 3.14

    def test_string_number(self):
        assert to_float("99.9") == 99.9

    def test_none(self):
        assert np.isnan(to_float(None))

    def test_na_string(self):
        assert np.isnan(to_float("N/A"))

    def test_empty_string(self):
        assert np.isnan(to_float(""))

    def test_nan_string(self):
        assert np.isnan(to_float("nan"))

    def test_unparseable(self):
        assert np.isnan(to_float("abc"))


# ── normalize_prob ───────────────────────────────────────────────────

class TestNormalizeProb:
    def test_zero_one_range(self):
        assert normalize_prob(0.65) == pytest.approx(0.65)

    def test_percentage(self):
        assert normalize_prob(75) == pytest.approx(0.75)

    def test_over_100(self):
        assert normalize_prob(150) is None

    def test_clamps_negative(self):
        assert normalize_prob(-0.1) == 0.0

    def test_none(self):
        assert normalize_prob(None) is None

    def test_nan(self):
        assert normalize_prob(float("nan")) is None


# ── ml_badge ─────────────────────────────────────────────────────────

class TestMlBadge:
    def test_green(self):
        assert "🟢" in ml_badge(0.75)

    def test_yellow(self):
        assert "🟡" in ml_badge(0.50)

    def test_red(self):
        assert "🔴" in ml_badge(0.30)

    def test_none(self):
        assert ml_badge(None) == "—"


# ── get_ml_prob_from_row ─────────────────────────────────────────────

class TestGetMlProbFromRow:
    def test_primary_key(self):
        row = {"ML_20d_Prob_live_v3": 0.82}
        assert get_ml_prob_from_row(row) == pytest.approx(0.82)

    def test_fallback_key(self):
        row = {"ML_Probability": 0.55}
        assert get_ml_prob_from_row(row) == pytest.approx(0.55)

    def test_no_keys(self):
        assert np.isnan(get_ml_prob_from_row({}))


# ── risk_class ───────────────────────────────────────────────────────

class TestRiskClass:
    def test_risk_class_from_riskclass(self):
        row = pd.Series({"RiskClass": "CORE"})
        assert risk_class(row) == "CORE"

    def test_risk_class_from_level(self):
        row = pd.Series({"Risk_Level": "core"})
        assert risk_class(row) == "CORE"

    def test_risk_class_default(self):
        row = pd.Series({})
        assert risk_class(row) == "SPEC"


# ── headline_story ───────────────────────────────────────────────────

class TestHeadlineStory:
    def test_quality_strong(self):
        row = pd.Series({"FundamentalScore": 80, "MomentumScore": 75, "RR": 3.0, "Reliability_v2": 90})
        story = headline_story(row)
        assert "Quality" in story
        assert "strong momentum" in story
        assert "excellent RR" in story

    def test_empty_row(self):
        row = pd.Series({})
        assert headline_story(row) == ""


# ── fmt_num ──────────────────────────────────────────────────────────

class TestFmtNum:
    def test_format_float(self):
        assert fmt_num(3.1415, ".2f") == "3.14"

    def test_na(self):
        assert fmt_num(None, ".2f") == "N/A"

    def test_custom_na(self):
        assert fmt_num("N/A", ".0f", na="—") == "—"


# ── get_reliability_band ─────────────────────────────────────────────

class TestGetReliabilityBand:
    def test_high(self):
        assert get_reliability_band(80) == "High"

    def test_medium(self):
        assert get_reliability_band(50) == "Medium"

    def test_low(self):
        assert get_reliability_band(20) == "Low"

    def test_unknown(self):
        assert get_reliability_band(None) == "Unknown"


# ── get_reliability_components ───────────────────────────────────────

class TestGetReliabilityComponents:
    def test_basic(self):
        row = pd.Series({
            "Fundamental_Reliability_v2": 80,
            "Price_Reliability_v2": 60,
            "fund_sources_used_v2": 3,
            "price_sources_used_v2": 2,
        })
        result = get_reliability_components(row)
        assert "F:80%" in result
        assert "P:60%" in result
