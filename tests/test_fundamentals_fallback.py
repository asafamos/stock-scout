"""Tests for fundamentals aggregation with multi-source fallback.

Validates that aggregate_fundamentals in core.data_sources_v2:
1. Uses the preferred source first, then fills missing fields from others
2. Returns neutral defaults when all providers respond with empty data
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from core.data_sources_v2 import aggregate_fundamentals


def test_per_field_fallback_order(monkeypatch):
    """When a preferred source is missing a field, other sources fill it in."""
    # Mock each provider's fetch function to return partial data
    fmp_data = {"pe": 15.0, "ps": 2.0, "market_cap": 1e9, "beta": 1.1}
    finnhub_data = {"roe": 0.18, "margin": 0.25, "debt_equity": 0.5}
    tiingo_data = {"rev_yoy": 0.20, "eps_yoy": 0.15}
    alpha_data = {"pb": 3.0, "peg": 1.5}

    # Patch individual fetch functions to return controlled data
    with patch("core.data_sources_v2.fetch_fundamentals_fmp", return_value=fmp_data) as mock_fmp, \
         patch("core.data_sources_v2.fetch_fundamentals_finnhub", return_value=finnhub_data) as mock_finn, \
         patch("core.data_sources_v2.fetch_fundamentals_tiingo", return_value=tiingo_data) as mock_tiingo, \
         patch("core.data_sources_v2.fetch_fundamentals_alpha", return_value=alpha_data) as mock_alpha, \
         patch("core.data_sources_v2.fetch_fundamentals_eodhd", return_value=None), \
         patch("core.data_sources_v2.fetch_fundamentals_simfin", return_value=None), \
         patch("core.data_sources_v2.get_provider_guard") as mock_guard, \
         patch("core.data_sources_v2.load_fundamentals_as_of", return_value=None):

        # Guard allows all providers
        guard = MagicMock()
        guard.allow.return_value = (True, "ok", "allow")
        mock_guard.return_value = guard

        merged = aggregate_fundamentals("TST")

    # FMP fields present
    assert merged["pe"] == pytest.approx(15.0)
    assert merged["ps"] == pytest.approx(2.0)

    # Finnhub fields filled in
    assert merged["roe"] == pytest.approx(0.18)
    assert merged["margin"] == pytest.approx(0.25)

    # Tiingo fields filled in
    assert merged["rev_yoy"] == pytest.approx(0.20)
    assert merged["eps_yoy"] == pytest.approx(0.15)

    # Alpha fields filled in
    assert merged["pb"] == pytest.approx(3.0)

    # sources_used should list all contributing providers
    assert "fmp" in merged["sources_used"]
    assert "finnhub" in merged["sources_used"]

    # coverage should map fields to their source(s)
    assert "pe" in merged["coverage"]
    assert len(merged["coverage"]["pe"]) >= 1


def test_neutral_defaults_when_providers_respond_but_no_fields():
    """When all providers return empty or None, we get a neutral structure."""
    with patch("core.data_sources_v2.fetch_fundamentals_fmp", return_value=None), \
         patch("core.data_sources_v2.fetch_fundamentals_finnhub", return_value=None), \
         patch("core.data_sources_v2.fetch_fundamentals_tiingo", return_value=None), \
         patch("core.data_sources_v2.fetch_fundamentals_alpha", return_value=None), \
         patch("core.data_sources_v2.fetch_fundamentals_eodhd", return_value=None), \
         patch("core.data_sources_v2.fetch_fundamentals_simfin", return_value=None), \
         patch("core.data_sources_v2.get_provider_guard") as mock_guard, \
         patch("core.data_sources_v2.load_fundamentals_as_of", return_value=None):

        guard = MagicMock()
        guard.allow.return_value = (True, "ok", "allow")
        mock_guard.return_value = guard

        merged = aggregate_fundamentals("NOP")

    # Should get neutral structure, not crash
    assert merged["ticker"] == "NOP"
    assert merged["sources_used"] == []
    assert merged["Fundamental_Coverage_Pct"] == 0.0
    assert merged["Fundamental_Sources_Count"] == 0
    # Neutral fundamental score
    assert merged.get("Fundamental_S", 50.0) == pytest.approx(50.0)
    # Disagreement should be maximum (1.0) with no data
    assert merged["disagreement_score"] == pytest.approx(1.0)


def test_median_used_when_multiple_sources_provide_same_field():
    """When multiple sources provide the same field, median is used."""
    fmp_data = {"pe": 10.0, "market_cap": 1e9}
    finnhub_data = {"pe": 14.0, "roe": 0.15}
    tiingo_data = {"pe": 12.0, "rev_yoy": 0.10}

    with patch("core.data_sources_v2.fetch_fundamentals_fmp", return_value=fmp_data), \
         patch("core.data_sources_v2.fetch_fundamentals_finnhub", return_value=finnhub_data), \
         patch("core.data_sources_v2.fetch_fundamentals_tiingo", return_value=tiingo_data), \
         patch("core.data_sources_v2.fetch_fundamentals_alpha", return_value=None), \
         patch("core.data_sources_v2.fetch_fundamentals_eodhd", return_value=None), \
         patch("core.data_sources_v2.fetch_fundamentals_simfin", return_value=None), \
         patch("core.data_sources_v2.get_provider_guard") as mock_guard, \
         patch("core.data_sources_v2.load_fundamentals_as_of", return_value=None):

        guard = MagicMock()
        guard.allow.return_value = (True, "ok", "allow")
        mock_guard.return_value = guard

        merged = aggregate_fundamentals("MED")

    # PE: median of [10, 14, 12] = 12
    assert merged["pe"] == pytest.approx(12.0)
    # All three contributed to PE
    assert len(merged["coverage"]["pe"]) == 3

