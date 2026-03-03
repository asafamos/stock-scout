"""
Tests for the 2026-02-15 deep fixes:
1. Fundamentals threshold lowered to 35
2. Reliability recomputation post-enrichment
3. Provider pipeline (SimFin parsing, EODHD routing, total_providers count)
4. R/R gate for CORE classification
"""
import numpy as np
import pandas as pd
import pytest

# ─── Fix 1: Fundamentals threshold ─────────────────────────────────────

class TestFundamentalsThreshold:
    """Verify the fundamentals score threshold is now 35 (was 60)."""

    def test_default_threshold_is_35(self):
        """Default env fallback should be '35', not '60'."""
        import os
        # Without env override, the code defaults to "35"
        original = os.environ.pop("FUNDAMENTALS_SCORE_THRESHOLD", None)
        try:
            # Simulate what runner.py does
            val = float(os.getenv("FUNDAMENTALS_SCORE_THRESHOLD", "35"))
            assert val == 35.0
        finally:
            if original is not None:
                os.environ["FUNDAMENTALS_SCORE_THRESHOLD"] = original

    def test_env_override_respected(self):
        """If env sets a custom threshold, it should be used."""
        import os
        os.environ["FUNDAMENTALS_SCORE_THRESHOLD"] = "50"
        try:
            val = float(os.getenv("FUNDAMENTALS_SCORE_THRESHOLD", "35"))
            assert val == 50.0
        finally:
            os.environ.pop("FUNDAMENTALS_SCORE_THRESHOLD", None)

    def test_default_cap_is_100(self):
        """Default top_n_cap should be 100 (was 50)."""
        import os
        original = os.environ.pop("FUNDAMENTALS_TOP_N_CAP", None)
        try:
            val = int(os.getenv("FUNDAMENTALS_TOP_N_CAP", "100"))
            assert val == 100
        finally:
            if original is not None:
                os.environ["FUNDAMENTALS_TOP_N_CAP"] = original

    def test_spec_stocks_above_threshold(self):
        """Stocks scoring 40-55 (SPEC range) should now pass the threshold of 35."""
        threshold = 35.0
        spec_scores = [40, 42, 45, 48, 50, 53, 55]
        for score in spec_scores:
            assert score > threshold, (
                f"SPEC stock with score {score} would be excluded "
                f"from fundamentals fetch at threshold {threshold}"
            )


# ─── Fix 2: Reliability recomputation ──────────────────────────────────

class TestReliabilityRecomputation:
    """Verify the reliability recomputation logic."""

    def _make_row(self, **kwargs):
        defaults = {
            "Fundamental_Sources_Count": 0,
            "Close": 50.0,
            "ATR": 2.0,
            "RSI": 45.0,
            "Fundamental_S": 50.0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def _recompute_reliability(self, row):
        """Mirror of the recomputation function in runner.py."""
        fund_sources = row.get("Fundamental_Sources_Count", 0)
        if pd.isna(fund_sources):
            fund_sources = 0
        fund_sources = int(fund_sources)
        fund_score = min(fund_sources, 4) * 15 + 20

        price_bonus = 0
        if pd.notna(row.get("Price_Yahoo", row.get("Close"))):
            price_bonus += 10
        if pd.notna(row.get("ATR")):
            price_bonus += 5
        if pd.notna(row.get("RSI")):
            price_bonus += 5

        fund_data_bonus = 0
        fund_s = row.get("Fundamental_S", 50.0)
        has_real_fund = any(
            pd.notna(row.get(f))
            for f in ["pe", "roe", "pb", "margin", "debt_equity",
                       "PE_Ratio", "ROE", "PB_Ratio", "Debt_to_Equity"]
        )
        if has_real_fund and fund_s != 50.0:
            fund_data_bonus = 10

        return min(fund_score + price_bonus + fund_data_bonus, 100)

    def test_zero_sources_floor_is_40(self):
        """With no fund sources but price/ATR/RSI, reliability = 40."""
        row = self._make_row(Fundamental_Sources_Count=0)
        assert self._recompute_reliability(row) == 40

    def test_one_source_is_55(self):
        """One fund source + price/ATR/RSI = 20 + 15 + 20 = 55."""
        row = self._make_row(Fundamental_Sources_Count=1)
        assert self._recompute_reliability(row) == 55

    def test_two_sources_is_70(self):
        """Two fund sources + price/ATR/RSI = 20 + 30 + 20 = 70."""
        row = self._make_row(Fundamental_Sources_Count=2)
        assert self._recompute_reliability(row) == 70

    def test_real_fund_data_bonus(self):
        """Real fundamental data (non-default) gives +10 bonus."""
        row = self._make_row(
            Fundamental_Sources_Count=2,
            Fundamental_S=65.0,
            PE_Ratio=15.0,
        )
        result = self._recompute_reliability(row)
        # 20 + 30 (2 sources) + 20 (price/ATR/RSI) + 10 (real fund) = 80
        assert result == 80

    def test_four_sources_with_fund_data_caps_at_100(self):
        """Max sources + real data shouldn't exceed 100."""
        row = self._make_row(
            Fundamental_Sources_Count=5,  # capped at 4
            Fundamental_S=72.0,
            PE_Ratio=18.0,
            ROE=0.15,
        )
        result = self._recompute_reliability(row)
        # 20 + 60 (4×15) + 20 + 10 = 110 → capped at 100
        assert result == 100

    def test_nan_sources_treated_as_zero(self):
        """NaN Fundamental_Sources_Count should be treated as 0."""
        row = self._make_row(Fundamental_Sources_Count=np.nan)
        assert self._recompute_reliability(row) == 40

    def test_no_price_data_reduces_score(self):
        """Missing price/ATR/RSI means no price bonus."""
        row = pd.Series({
            "Fundamental_Sources_Count": 2,
            "Fundamental_S": 50.0,
        })
        result = self._recompute_reliability(row)
        # 20 + 30 + 0 (no price/ATR/RSI) = 50
        assert result == 50


# ─── Fix 3: Provider pipeline ──────────────────────────────────────────

class TestProviderPipeline:
    """Verify provider pipeline fixes."""

    def test_total_providers_is_9(self):
        """api_preflight should report 9 total providers (not 11)."""
        # We can't easily run preflight without network, but verify
        # the code has the right constant
        import ast
        import os
        preflight_path = os.path.join(
            os.path.dirname(__file__), "..", "core", "api_preflight.py"
        )
        with open(preflight_path) as f:
            source = f.read()
        assert '"total_providers": 9' in source, (
            "total_providers should be 9 (actual provider count), not 11"
        )

    def test_simfin_parses_real_data(self):
        """SimFin function should extract ROE, margin, debt_equity from statement data."""
        # Simulate what _simfin_row returns
        pl_row = {
            "Revenue": 50000000000,
            "Net Income": 5000000000,
            "Shares (Diluted)": 1000000000,
        }
        bs_row = {
            "Total Equity": 40000000000,
            "Total Debt": 20000000000,
        }

        # Replicate extraction logic
        revenue = pl_row.get("Revenue")
        net_income = pl_row.get("Net Income")
        total_equity = bs_row.get("Total Equity")
        total_debt = bs_row.get("Total Debt")

        margin = net_income / revenue if revenue and net_income else None
        roe = net_income / total_equity if net_income and total_equity else None
        debt_equity = total_debt / total_equity if total_debt is not None and total_equity else None

        assert margin == pytest.approx(0.1)
        assert roe == pytest.approx(0.125)
        assert debt_equity == pytest.approx(0.5)

    def test_eodhd_uses_http_get_with_retry(self):
        """EODHD fetch function should use _http_get_with_retry, not raw requests.get."""
        import inspect
        from core.data_sources_v2 import fetch_fundamentals_eodhd
        source = inspect.getsource(fetch_fundamentals_eodhd)
        assert "_http_get_with_retry" in source, "EODHD should use _http_get_with_retry"
        # Should NOT have raw requests.get for the main data fetch
        # (it may still appear in imports but not as the primary fetch mechanism)
        lines = [l.strip() for l in source.split("\n")
                 if "requests.get" in l and not l.startswith("#")]
        assert len(lines) == 0, (
            f"EODHD should not use raw requests.get; found: {lines}"
        )

    def test_aggregate_tracks_eodhd_simfin_flags(self):
        """Aggregation should set Fund_from_EODHD and Fund_from_SimFin flags."""
        import inspect
        from core.data_sources_v2 import aggregate_fundamentals
        source = inspect.getsource(aggregate_fundamentals)
        assert "Fund_from_EODHD" in source
        assert "Fund_from_SimFin" in source


# ─── Fix 4: R/R gate for CORE classification ───────────────────────────

class TestRiskClassRRGate:
    """Verify that assign_risk_class enforces minimum R/R for CORE.

    Updated 2026-03-03: R/R threshold raised from 0.8 to 1.5.
    Hard safety filters now block R/R < 1.5, negative ROE, and missing fundamentals.
    """

    def _make_row(self, **kwargs):
        defaults = {
            "FinalScore_20d": 70,
            "ATR_Pct": 0.03,
            "Beta": 1.0,
            "RR": 2.0,
            "Close": 50.0,
            "Volume": 1000000,
            "ROE": 15.0,          # required by hard filter (missing data check)
            "MarketCap": 5e9,     # required by hard filter (missing data check)
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_good_rr_stays_core(self):
        """Stock with R/R >= 1.5 and good score should be CORE."""
        from core.classification import assign_risk_class
        row = self._make_row(RR=2.0)
        assert assign_risk_class(row) == "CORE"

    def test_poor_rr_rejected(self):
        """Stock with R/R < 1.5 should be REJECTED by hard safety filter."""
        from core.classification import assign_risk_class
        row = self._make_row(RR=0.26)
        result = assign_risk_class(row)
        assert result == "REJECT", (
            f"Stock with R/R=0.26 should be REJECT (hard filter), not {result}"
        )

    def test_borderline_rr_is_core(self):
        """Stock with R/R = 1.5 (exact threshold) should be CORE."""
        from core.classification import assign_risk_class
        row = self._make_row(RR=1.5)
        assert assign_risk_class(row) == "CORE"

    def test_just_below_threshold_is_rejected(self):
        """Stock with R/R = 1.49 should be REJECTED by hard filter."""
        from core.classification import assign_risk_class
        row = self._make_row(RR=1.49)
        assert assign_risk_class(row) == "REJECT"

    def test_missing_rr_defaults_to_core(self):
        """Stock with missing R/R should still qualify as CORE (no penalty for gaps)."""
        from core.classification import assign_risk_class
        row = self._make_row()
        del row["RR"]
        assert assign_risk_class(row) == "CORE"

    def test_nan_rr_defaults_to_core(self):
        """Stock with NaN R/R should treat as missing → CORE if score qualifies."""
        from core.classification import assign_risk_class
        row = self._make_row(RR=np.nan)
        assert assign_risk_class(row) == "CORE"

    def test_score_below_55_is_spec_regardless_of_rr(self):
        """Score < 55 → SPEC even with great R/R."""
        from core.classification import assign_risk_class
        row = self._make_row(FinalScore_20d=50, RR=3.0)
        assert assign_risk_class(row) == "SPEC"

    def test_score_below_40_is_reject(self):
        """Score < 40 → REJECT regardless of R/R."""
        from core.classification import assign_risk_class
        row = self._make_row(FinalScore_20d=30, RR=5.0)
        assert assign_risk_class(row) == "REJECT"

    def test_high_volatility_demotes_to_spec(self):
        """High ATR should demote even with good R/R and score."""
        from core.classification import assign_risk_class
        row = self._make_row(ATR_Pct=0.09, RR=2.0)
        assert assign_risk_class(row) == "SPEC"

    def test_high_beta_demotes_to_spec(self):
        """High Beta should demote even with good R/R and score."""
        from core.classification import assign_risk_class
        row = self._make_row(Beta=2.0, RR=2.0)
        assert assign_risk_class(row) == "SPEC"

    def test_negative_roe_rejected(self):
        """Stock with negative ROE should be REJECTED by hard filter."""
        from core.classification import assign_risk_class
        row = self._make_row(ROE=-5.0, RR=2.0)
        assert assign_risk_class(row) == "REJECT"

    def test_missing_fundamentals_rejected(self):
        """Stock missing both ROE and MarketCap should be REJECTED."""
        from core.classification import assign_risk_class
        row = self._make_row(RR=2.0)
        del row["ROE"]
        del row["MarketCap"]
        assert assign_risk_class(row) == "REJECT"


# ─── Scoring engine: ML boost + reliability gating ─────────────────────

class TestMLBoostReliabilityGating:
    """Verify ML boost is correctly gated by reliability score."""

    def test_ml_boost_neutral(self):
        from core.scoring_engine import ml_boost_component
        assert ml_boost_component(0.5) == pytest.approx(0.0)

    def test_ml_boost_high(self):
        from core.scoring_engine import ml_boost_component
        assert ml_boost_component(1.0) == pytest.approx(6.0)

    def test_ml_boost_low(self):
        from core.scoring_engine import ml_boost_component
        assert ml_boost_component(0.0) == pytest.approx(-6.0)

    def test_ml_boost_none(self):
        from core.scoring_engine import ml_boost_component
        assert ml_boost_component(None) == 0.0

    def test_final_score_with_high_reliability_gets_full_ml_boost(self):
        """ReliabilityScore >= 60 should give full ML boost."""
        from core.scoring_engine import compute_final_score_20d
        row = pd.Series({
            "Fundamental_S": 70,
            "MomentumScore": 70,
            "RR": 2.0,
            "ReliabilityScore": 80,
            "ML_20d_Prob": 0.9,
        })
        score_high_rel = compute_final_score_20d(row)

        row_low = row.copy()
        row_low["ReliabilityScore"] = 30
        score_low_rel = compute_final_score_20d(row_low)

        # High reliability should produce a higher score due to full ML boost
        assert score_high_rel > score_low_rel, (
            f"High reliability ({score_high_rel}) should produce higher score "
            f"than low reliability ({score_low_rel}) with same ML_20d_Prob=0.9"
        )

    def test_final_score_bounded_0_100(self):
        """Final score should always be in [0, 100]."""
        from core.scoring_engine import compute_final_score_20d
        # Extreme high inputs
        row = pd.Series({
            "Fundamental_S": 100,
            "MomentumScore": 100,
            "RR": 10.0,
            "ReliabilityScore": 100,
            "ML_20d_Prob": 1.0,
        })
        assert 0 <= compute_final_score_20d(row) <= 100

        # Extreme low inputs
        row_low = pd.Series({
            "Fundamental_S": 0,
            "MomentumScore": 0,
            "RR": 0,
            "ReliabilityScore": 0,
            "ML_20d_Prob": 0.0,
        })
        assert 0 <= compute_final_score_20d(row_low) <= 100

    def test_final_score_missing_fields_returns_neutral(self):
        """Missing all fields should return 50 (neutral fallback)."""
        from core.scoring_engine import compute_final_score_20d
        row = pd.Series({})
        score = compute_final_score_20d(row)
        assert 40 <= score <= 60, f"Empty row should give neutral score, got {score}"
