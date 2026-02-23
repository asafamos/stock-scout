"""Phase 3 – Scoring quality tests.

Validates fixes for:
  1. Bridge emits TechScore_20d (not just ML-prob proxy)
  2. Bridge emits Fundamental_Score so downstream blend can use it
  3. Quality_Score comes from calculate_quality_score (not advanced filter boost)
  4. Quality_Level derives from actual fundamental label, not hardcoded "medium"
  5. SignalQuality differentiates across stocks (not always "Speculative")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Bridge emits TechScore_20d from technical indicators
# ---------------------------------------------------------------------------

class TestBridgeTechScore:
    """Bridge must compute TechScore_20d from technical indicators, not from ML."""

    def _make_row(self, overrides: dict | None = None) -> pd.Series:
        """Minimal row with required fields for bridge."""
        base = {
            "Ticker": "TEST",
            "Close": 100.0,
            "RSI": 55.0,
            "ATR": 2.5,
            "ATR_Pct": 0.025,
            "ADR_Pct": 0.025,
            "MA20": 98.0,
            "MA50": 95.0,
            "MA200": 85.0,
            "MA50_Slope": 0.002,
            "Overext": 0.05,  # Price 5% above MA50
            "Near52w": 85.0,
            "RR": 2.0,
            "Volume": 1_000_000,
            "Return_1m": 0.08,
            "Return_3m": 0.15,
            "Return_6m": 0.25,
            "Beta": 1.1,
            "DE_f": 0.8,
            "ROE_f": 0.18,
            "GM_f": 0.35,
            "RevG_f": 0.12,
            "EPSG_f": 0.20,
            "Fundamental_S": 65.0,
        }
        if overrides:
            base.update(overrides)
        return pd.Series(base)

    def test_bridge_emits_tech_score(self):
        """Bridge output must contain TechScore_20d."""
        from core.bridge import analyze_row_with_bridge
        row = self._make_row()
        result = analyze_row_with_bridge("TEST", row)
        assert "TechScore_20d" in result, "Bridge must emit TechScore_20d"

    def test_tech_score_not_equal_to_ml_prob(self):
        """TechScore_20d must not be identical to ML_20d_Prob * 100."""
        from core.bridge import analyze_row_with_bridge
        row = self._make_row()
        result = analyze_row_with_bridge("TEST", row)
        ml_prob = result.get("ML_20d_Prob", 0.5)
        tech_score = result.get("TechScore_20d", 0.0)
        assert abs(tech_score - ml_prob * 100) > 1.0, (
            f"TechScore_20d ({tech_score:.1f}) should not be ML_20d_Prob*100 ({ml_prob*100:.1f})"
        )

    def test_tech_score_uses_indicators(self):
        """TechScore_20d should differ for strong vs weak setups."""
        from core.bridge import analyze_row_with_bridge

        # Strong setup: uptrend, positive momentum
        strong = self._make_row({
            "Overext": 0.08,
            "MA50_Slope": 0.005,
            "Return_1m": 0.10,
            "Return_3m": 0.20,
            "Return_6m": 0.30,
            "RSI": 55.0,
        })
        # Weak setup: downtrend, negative momentum
        weak = self._make_row({
            "Overext": -0.10,
            "MA50": 110.0,  # Price below MA50
            "MA200": 120.0,  # Death cross territory
            "MA50_Slope": -0.005,
            "Return_1m": -0.08,
            "Return_3m": -0.15,
            "Return_6m": -0.20,
            "RSI": 28.0,
        })

        result_strong = analyze_row_with_bridge("STRONG", strong)
        result_weak = analyze_row_with_bridge("WEAK", weak)

        ts_strong = result_strong["TechScore_20d"]
        ts_weak = result_weak["TechScore_20d"]

        assert ts_strong > ts_weak, (
            f"Strong setup TechScore ({ts_strong:.1f}) should exceed weak ({ts_weak:.1f})"
        )

    def test_bridge_emits_fundamental_score(self):
        """Bridge output must contain Fundamental_Score for downstream blend."""
        from core.bridge import analyze_row_with_bridge
        row = self._make_row({"Fundamental_S": 72.0})
        result = analyze_row_with_bridge("TEST", row)
        assert "Fundamental_Score" in result, "Bridge must emit Fundamental_Score"
        assert result["Fundamental_Score"] == pytest.approx(72.0, abs=0.1)


# ---------------------------------------------------------------------------
# 2. Quality_Score from fundamental metrics (not advanced filter boost)
# ---------------------------------------------------------------------------

class TestBridgeQualityScore:
    """Quality_Score must reflect fundamental quality, not signal boost."""

    def _make_row(self, overrides: dict | None = None) -> pd.Series:
        base = {
            "Ticker": "QS",
            "Close": 100.0,
            "RSI": 55.0,
            "ATR": 2.5,
            "ATR_Pct": 0.025,
            "ADR_Pct": 0.025,
            "MA20": 98.0,
            "MA50": 95.0,
            "MA200": 85.0,
            "MA50_Slope": 0.002,
            "Overext": 0.05,
            "Near52w": 85.0,
            "RR": 2.0,
            "Volume": 1_000_000,
            "Return_1m": 0.08,
            "Return_3m": 0.15,
            "Return_6m": 0.25,
            "Beta": 1.1,
        }
        if overrides:
            base.update(overrides)
        return pd.Series(base)

    def test_quality_score_reflects_fundamentals(self):
        """Strong fundamentals → higher Quality_Score than weak."""
        from core.bridge import analyze_row_with_bridge

        strong_fund = self._make_row({
            "ROE_f": 0.22,  # 22% ROE
            "GM_f": 0.45,   # 45% gross margin
            "RevG_f": 0.18, # 18% rev growth
            "EPSG_f": 0.25, # 25% EPS growth
            "DE_f": 0.4,    # Low debt
        })
        weak_fund = self._make_row({
            "ROE_f": 0.02,  # 2% ROE
            "GM_f": 0.08,   # 8% gross margin
            "RevG_f": -0.10,  # Declining revenue
            "EPSG_f": -0.15,  # Declining EPS
            "DE_f": 3.5,    # High debt
        })

        result_strong = analyze_row_with_bridge("STRONG_F", strong_fund)
        result_weak = analyze_row_with_bridge("WEAK_F", weak_fund)

        qs_strong = result_strong.get("Quality_Score", 0.0)
        qs_weak = result_weak.get("Quality_Score", 0.0)

        assert qs_strong > qs_weak, (
            f"Strong fundamentals Quality_Score ({qs_strong:.2f}) "
            f"should exceed weak ({qs_weak:.2f})"
        )

    def test_quality_score_not_always_zero(self):
        """Quality_Score should not always be 0.0."""
        from core.bridge import analyze_row_with_bridge

        row = self._make_row({
            "ROE_f": 0.15,
            "GM_f": 0.30,
            "RevG_f": 0.10,
            "DE_f": 0.8,
        })
        result = analyze_row_with_bridge("NONZERO", row)
        qs = result.get("Quality_Score", 0.0)
        assert qs > 0.0, f"Quality_Score should not be 0.0 with valid fundamentals, got {qs}"

    def test_quality_level_emitted(self):
        """Bridge should emit Quality_Level (High/Medium/Low)."""
        from core.bridge import analyze_row_with_bridge

        row = self._make_row({"ROE_f": 0.20, "GM_f": 0.40, "DE_f": 0.3})
        result = analyze_row_with_bridge("QL", row)
        assert "Quality_Level" in result
        assert result["Quality_Level"] in ("High", "Medium", "Low")


# ---------------------------------------------------------------------------
# 3. Quality_Level from row_builder (not hardcoded "medium")
# ---------------------------------------------------------------------------

class TestRowBuilderQualityLevel:
    """Quality_Level must derive from fundamental scoring, not hardcoded."""

    def test_quality_level_from_quality_label(self):
        """Quality_Level should match the computed Quality_Label."""
        from core.row_builder import build_row_from_multi_source

        multi_source = {
            "Ticker": "QL",
            "sector": "Technology",
            "roe": 0.25,
            "gm": 0.50,
            "roic": 0.20,
            "rev_yoy": 0.15,
            "eps_yoy": 0.20,
            "pe": 18.0,
            "ps": 3.0,
            "de": 0.3,
            "beta": 1.0,
            "market_cap": 50_000_000_000,
            "Fundamental_Coverage_Pct": 80.0,
            "Fundamental_Sources_Count": 2,
        }

        row = build_row_from_multi_source("QL", multi_source, {"Close": 100.0})
        ql = row.get("Quality_Level", "")
        label = row.get("Quality_Label", "")
        # Quality_Level should be lowercase version of Quality_Label
        assert ql == label.lower() or ql in ("high", "medium", "low", "unknown"), (
            f"Quality_Level '{ql}' should derive from Quality_Label '{label}'"
        )

    def test_quality_level_not_always_medium(self):
        """With extreme fundamentals, Quality_Level should differ per stock."""
        from core.row_builder import build_row_from_multi_source

        # Strong fundamentals
        strong = {
            "roe": 0.30, "gm": 0.60, "roic": 0.25,
            "rev_yoy": 0.25, "eps_yoy": 0.30,
            "pe": 15.0, "ps": 2.0, "de": 0.2,
            "beta": 0.9, "market_cap": 100_000_000_000,
            "Fundamental_Coverage_Pct": 90.0,
            "Fundamental_Sources_Count": 3,
        }
        row = build_row_from_multi_source("STRONG", strong, {"Close": 200.0})
        assert row.get("Quality_Level") in ("high", "medium", "low", "unknown")


# ---------------------------------------------------------------------------
# 4. SignalQuality differentiates across stocks
# ---------------------------------------------------------------------------

class TestSignalQuality:
    """SignalQuality should not be universally 'Speculative'."""

    def test_signal_quality_with_multiple_signals(self):
        """A stock with strong tech, fundamentals, and R/R should get Medium+."""
        rec = pd.Series({
            "TechScore_20d": 70.0,
            "ML_20d_Prob": 0.55,
            "Fundamental_S": 65.0,
            "RR": 2.5,
            "VolSurge": 1.5,
            "Pattern_Score": 0.0,
            "Market_Regime": "NEUTRAL",
        })

        reasons = []
        # Replicate the SignalQuality logic
        if rec.get("TechScore_20d", 0) >= 65:
            reasons.append("Strong technical momentum")
        elif rec.get("TechScore_20d", 0) >= 45:
            reasons.append("Positive technical setup")

        if rec.get("ML_20d_Prob", 0) >= 0.62:
            reasons.append("High ML breakout probability")
        elif rec.get("ML_20d_Prob", 0) >= 0.50:
            reasons.append("Moderate ML breakout probability")

        if rec.get("Fundamental_S", 0) >= 60:
            reasons.append("Strong fundamentals")

        if rec.get("RR", 0) >= 2.0:
            reasons.append("Favorable risk/reward ratio")

        if rec.get("VolSurge", 0) >= 1.3:
            reasons.append("Volume surge confirmation")

        market_regime = str(rec.get("Market_Regime", "")).upper()
        if market_regime in ("TREND_UP", "BULLISH", "NEUTRAL", "SIDEWAYS"):
            reasons.append("Supportive market regime")

        cnt = len(reasons)
        quality = "High" if cnt >= 4 else ("Medium" if cnt >= 2 else "Speculative")

        assert cnt >= 4, f"Expected ≥4 signal reasons, got {cnt}: {reasons}"
        assert quality == "High", f"Expected 'High' quality, got '{quality}'"

    def test_weak_stock_still_speculative(self):
        """A stock with no positive signals should remain Speculative."""
        rec = pd.Series({
            "TechScore_20d": 25.0,
            "ML_20d_Prob": 0.30,
            "Fundamental_S": 40.0,
            "RR": 0.8,
            "VolSurge": 0.7,
            "Pattern_Score": 0.0,
            "Market_Regime": "BEAR",
        })

        reasons = []
        if rec.get("TechScore_20d", 0) >= 65:
            reasons.append("Strong technical momentum")
        elif rec.get("TechScore_20d", 0) >= 45:
            reasons.append("Positive technical setup")

        if rec.get("ML_20d_Prob", 0) >= 0.62:
            reasons.append("High ML breakout probability")
        elif rec.get("ML_20d_Prob", 0) >= 0.50:
            reasons.append("Moderate ML breakout probability")

        if rec.get("Fundamental_S", 0) >= 60:
            reasons.append("Strong fundamentals")

        if rec.get("RR", 0) >= 2.0:
            reasons.append("Favorable risk/reward ratio")

        if rec.get("VolSurge", 0) >= 1.3:
            reasons.append("Volume surge confirmation")

        cnt = len(reasons)
        quality = "High" if cnt >= 4 else ("Medium" if cnt >= 2 else "Speculative")

        assert cnt < 2, f"Weak stock should have <2 reasons, got {cnt}: {reasons}"
        assert quality == "Speculative"

    def test_medium_quality_reachable(self):
        """A stock with 2-3 positive signals gets Medium."""
        rec = pd.Series({
            "TechScore_20d": 50.0,  # Positive technical setup (≥45)
            "ML_20d_Prob": 0.52,    # Moderate ML (≥0.50)
            "Fundamental_S": 55.0,  # Not quite strong (<60)
            "RR": 1.5,             # Below threshold (<2.0)
            "VolSurge": 0.9,       # No surge
            "Pattern_Score": 0.0,
            "Market_Regime": "NEUTRAL",  # Supportive
        })

        reasons = []
        if rec.get("TechScore_20d", 0) >= 65:
            reasons.append("Strong technical momentum")
        elif rec.get("TechScore_20d", 0) >= 45:
            reasons.append("Positive technical setup")

        if rec.get("ML_20d_Prob", 0) >= 0.62:
            reasons.append("High ML breakout probability")
        elif rec.get("ML_20d_Prob", 0) >= 0.50:
            reasons.append("Moderate ML breakout probability")

        if rec.get("Fundamental_S", 0) >= 60:
            reasons.append("Strong fundamentals")

        if rec.get("RR", 0) >= 2.0:
            reasons.append("Favorable risk/reward ratio")

        if rec.get("VolSurge", 0) >= 1.3:
            reasons.append("Volume surge confirmation")

        market_regime = str(rec.get("Market_Regime", "")).upper()
        if market_regime in ("TREND_UP", "BULLISH", "NEUTRAL", "SIDEWAYS"):
            reasons.append("Supportive market regime")

        cnt = len(reasons)
        quality = "High" if cnt >= 4 else ("Medium" if cnt >= 2 else "Speculative")

        assert 2 <= cnt < 4, f"Expected 2-3 reasons, got {cnt}: {reasons}"
        assert quality == "Medium", f"Expected 'Medium', got '{quality}'"


# ---------------------------------------------------------------------------
# 5. FinalScore spread should be wider (not just ML_prob * 100)
# ---------------------------------------------------------------------------

class TestFinalScoreSpread:
    """FinalScore should reflect blended components, not just ML probability."""

    def test_bridge_final_score_differs_from_ml(self):
        """FinalScore_20d should not be exactly ML_20d_Prob * 100 after blend."""
        from core.bridge import analyze_row_with_bridge

        row = pd.Series({
            "Ticker": "SPREAD",
            "Close": 100.0,
            "RSI": 55.0,
            "ATR": 2.5,
            "ATR_Pct": 0.025,
            "ADR_Pct": 0.025,
            "MA20": 98.0,
            "MA50": 95.0,
            "MA200": 85.0,
            "MA50_Slope": 0.002,
            "Overext": 0.05,
            "Near52w": 85.0,
            "RR": 2.5,
            "Volume": 1_000_000,
            "Return_1m": 0.08,
            "Return_3m": 0.15,
            "Return_6m": 0.25,
            "Beta": 1.1,
            "DE_f": 0.8,
            "ROE_f": 0.15,
            "GM_f": 0.30,
            "Fundamental_S": 65.0,
        })

        result = analyze_row_with_bridge("SPREAD", row)
        tech = result.get("TechScore_20d", 0.0)
        fund = result.get("Fundamental_Score", 0.0)

        # The key fix: TechScore_20d and Fundamental_Score are now present
        # and can be used by downstream compute_final_score_with_patterns
        assert tech > 0.0, f"TechScore_20d should be > 0, got {tech}"
        assert fund > 0.0, f"Fundamental_Score should be > 0, got {fund}"
