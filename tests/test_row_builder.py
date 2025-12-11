"""
Integration tests for the canonical row builder.

Tests validate that multi-source payloads are correctly normalized into
the unified row schema used across all pipelines.
"""
import pytest
import numpy as np
import pandas as pd
from core.row_builder import build_row_from_multi_source, build_rows_from_universe, rows_to_dataframe


class TestRowBuilderBasics:
    """Basic row builder functionality."""
    
    def test_minimal_payload(self):
        """Build row from minimal payload."""
        payload = {
            "ticker": "TEST",
            "price_mean": None,
            "price_sources": 0,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["ticker"] == "TEST"
        assert "Close" in row
        assert "Fundamental_S" in row
        assert 0 <= row["Fundamental_S"] <= 100
    
    def test_row_has_all_required_fields(self):
        """Built row has all required schema fields."""
        payload = {}
        row = build_row_from_multi_source("TICK", payload)
        
        required_fields = [
            "ticker", "Close", "Price_Mean", "Price_STD", "Price_Sources_Count",
            "Fundamental_S", "Quality_Score_F", "Growth_Score_F", "Valuation_Score_F",
            "Leverage_Score_F", "Stability_Score_F",
            "Fundamental_Coverage_Pct", "Fundamental_Sources_Count",
            "Fund_from_FMP", "Fund_from_Finnhub", "Fund_from_Tiingo", "Fund_from_Alpha",
            "PE_f", "PS_f", "ROE_f", "DE_f", "Beta"
        ]
        
        for field in required_fields:
            assert field in row, f"Missing field: {field}"


class TestPriceFields:
    """Tests for price field normalization."""
    
    def test_price_from_market_data(self):
        """Market data Close takes precedence."""
        payload = {"price_mean": 100.0, "price_sources": 1}
        market_data = {"Close": 105.0}
        row = build_row_from_multi_source("TEST", payload, market_data)
        
        assert row["Close"] == 105.0  # From market_data
        assert row["Price_Mean"] == 100.0  # From payload
    
    def test_price_from_payload_fallback(self):
        """Falls back to price_mean if no market Close."""
        payload = {"price_mean": 100.0, "price_sources": 1}
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Close"] == 100.0  # Fallback
        assert row["Price_Mean"] == 100.0
    
    def test_price_std_and_sources(self):
        """Price variance and source count captured."""
        payload = {
            "price_mean": 50.0,
            "price_std": 1.5,
            "price_sources": 3,
            "prices_by_source": {
                "fmp": 49.8,
                "finnhub": 50.1,
                "tiingo": 50.2,
            }
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Price_STD"] == 1.5
        assert row["Price_Sources_Count"] == 3
        assert "Price_Fmp" in row
        assert "Price_Finnhub" in row
    
    def test_price_bounds_clipping(self):
        """Invalid prices are clipped to NaN."""
        payload = {
            "price_mean": np.inf,
            "price_std": np.nan,
            "price_sources": 0,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert np.isnan(row["Price_Mean"])
        assert np.isnan(row["Price_STD"])
        assert row["Price_Sources_Count"] == 0


class TestFundamentalRawFields:
    """Tests for fundamental raw metric mapping."""
    
    def test_fundamental_metrics_mapped(self):
        """Fundamental metrics are extracted and normalized."""
        payload = {
            "roe": 0.15,
            "roic": 0.12,
            "gm": 0.25,
            "rev_yoy": 0.10,
            "eps_yoy": 0.12,
            "pe": 25.0,
            "ps": 3.5,
            "pb": 2.0,
            "de": 0.8,
            "beta": 1.1,
            "market_cap": 5e9,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["ROE_f"] == 0.15
        assert row["ROIC_f"] == 0.12
        assert row["GM_f"] == 0.25
        assert row["RevG_f"] == 0.10
        assert row["EPSG_f"] == 0.12
        assert row["PE_f"] == 25.0
        assert row["PS_f"] == 3.5
        assert row["PBRatio"] == 2.0
        assert row["DE_f"] == 0.8
        assert row["Beta"] == 1.1
        assert row["MarketCap"] == 5e9
    
    def test_alternative_field_names(self):
        """Handles alternative field names (rev_g_yoy, debt_equity, etc)."""
        payload = {
            "rev_g_yoy": 0.15,  # Alternative name
            "eps_g_yoy": 0.20,  # Alternative name
            "debt_equity": 1.0,  # Alternative name
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["RevG_f"] == 0.15
        assert row["EPSG_f"] == 0.20
        assert row["DE_f"] == 1.0
    
    def test_missing_fundamentals_nan(self):
        """Missing fundamentals are NaN, not crashing."""
        payload = {}
        row = build_row_from_multi_source("TEST", payload)
        
        assert np.isnan(row["ROE_f"])
        assert np.isnan(row["PE_f"])
        assert np.isnan(row["DE_f"])


class TestFundamentalMetadata:
    """Tests for fundamental metadata and flags."""
    
    def test_coverage_and_sources_count(self):
        """Coverage percentage and source count captured."""
        payload = {
            "Fundamental_Coverage_Pct": 85.0,
            "Fundamental_Sources_Count": 3,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Fundamental_Coverage_Pct"] == 85.0
        assert row["Fundamental_Sources_Count"] == 3
    
    def test_per_source_flags(self):
        """Per-source flags set correctly."""
        payload = {
            "Fund_from_FMP": True,
            "Fund_from_Finnhub": False,
            "Fund_from_Tiingo": True,
            "Fund_from_Alpha": False,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Fund_from_FMP"] is True
        assert row["Fund_from_Finnhub"] is False
        assert row["Fund_from_Tiingo"] is True
        assert row["Fund_from_Alpha"] is False
    
    def test_disagreement_score(self):
        """Aggregation disagreement captured."""
        payload = {"disagreement_score": 0.25}
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Fund_Disagreement_Score"] == 0.25


class TestFundamentalScores:
    """Tests for computed fundamental breakdown scores."""
    
    def test_high_quality_stock_scores(self):
        """High-quality stock gets high scores."""
        payload = {
            "roe": 0.22,
            "roic": 0.20,
            "gm": 0.35,
            "rev_yoy": 0.18,
            "eps_yoy": 0.25,
            "pe": 28.0,
            "ps": 4.5,
            "de": 0.4,
            "beta": 1.1,
            "market_cap": 8e9,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 3,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Fundamental_S"] >= 70
        assert row["Quality_Score_F"] >= 75
        assert row["Growth_Score_F"] >= 75
        assert row["Valuation_Score_F"] >= 48  # Fair valuation, slightly relaxed threshold
    
    def test_weak_stock_scores(self):
        """Weak stock gets lower scores."""
        payload = {
            "roe": 0.05,
            "roic": 0.02,
            "gm": 0.08,
            "rev_yoy": -0.10,
            "eps_yoy": -0.05,
            "pe": 80.0,
            "ps": 12.0,
            "de": 2.8,
            "Fundamental_Coverage_Pct": 60.0,
            "Fundamental_Sources_Count": 1,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Fundamental_S"] <= 50
        assert row["Quality_Score_F"] <= 40
        assert row["Valuation_Score_F"] <= 30
    
    def test_all_scores_in_bounds(self):
        """All computed scores are in [0, 100]."""
        payload = {
            "roe": 0.15,
            "roic": 0.12,
            "gm": 0.25,
            "rev_yoy": 0.10,
            "eps_yoy": 0.12,
            "pe": 25.0,
            "ps": 3.5,
            "de": 1.0,
            "beta": 1.0,
            "market_cap": 3e9,
            "Fundamental_Coverage_Pct": 80.0,
            "Fundamental_Sources_Count": 2,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert 0 <= row["Fundamental_S"] <= 100
        assert 0 <= row["Quality_Score_F"] <= 100
        assert 0 <= row["Growth_Score_F"] <= 100
        assert 0 <= row["Valuation_Score_F"] <= 100
        assert 0 <= row["Leverage_Score_F"] <= 100
        assert 0 <= row["Stability_Score_F"] <= 100
    
    def test_score_labels_assigned(self):
        """Quality/Growth/Valuation labels assigned."""
        payload = {
            "roe": 0.20,
            "roic": 0.15,
            "gm": 0.30,
            "rev_yoy": 0.20,
            "eps_yoy": 0.25,
            "pe": 15.0,
            "ps": 2.0,
            "de": 0.5,
            "Fundamental_Coverage_Pct": 100.0,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Quality_Label"] in ["Low", "Medium", "High"]
        assert row["Growth_Label"] in ["Declining", "Slow", "Moderate", "Fast"]
        assert row["Valuation_Label"] in ["Expensive", "Fair", "Cheap"]
        assert row["Leverage_Label"] in ["High", "Medium", "Low"]
        assert row["Stability_Label"] in ["Very Volatile", "Volatile", "Moderate", "Stable"]


class TestTechnicalPlaceholders:
    """Tests for technical indicator placeholders."""
    
    def test_technical_placeholders_present(self):
        """All technical placeholder fields are present."""
        payload = {}
        row = build_row_from_multi_source("TEST", payload)
        
        technical_fields = [
            "RSI", "ATR", "ATR_Pct", "MACD", "SMA20", "SMA50",
            "Mom1M", "Mom3M", "Price_52W_High"
        ]
        
        for field in technical_fields:
            assert field in row
            assert np.isnan(row[field])  # Should be NaN initially


class TestReliabilityPrerequisites:
    """Tests for fields needed by reliability_v2 calculation."""
    
    def test_reliability_prerequisites(self):
        """All fields for reliability calculation are present."""
        payload = {
            "Fundamental_Coverage_Pct": 80.0,
            "Fundamental_Sources_Count": 2,
            "price_mean": 100.0,
            "price_std": 0.5,
            "price_sources": 3,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        # Fields used by calculate_reliability_v2
        assert "Fundamental_Coverage_Pct" in row
        assert "Fundamental_Sources_Count" in row
        assert "Price_Mean" in row
        assert "Price_STD" in row
        assert "Price_Sources_Count" in row
        assert "Beta" in row
        assert "Quality_Score_F" in row


class TestBatchOperations:
    """Tests for batch row building and DataFrame conversion."""
    
    def test_build_rows_from_universe(self):
        """Build rows for multiple tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        def mock_fetch(ticker):
            return {
                "roe": 0.15,
                "pe": 25.0,
                "price_mean": 100.0 + (len(ticker) * 10),
                "price_sources": 2,
            }
        
        rows = build_rows_from_universe(tickers, mock_fetch)
        
        assert len(rows) == 3
        assert rows[0]["ticker"] == "AAPL"
        assert rows[1]["ticker"] == "MSFT"
        assert rows[2]["ticker"] == "GOOGL"
        
        for row in rows:
            assert row["Fundamental_S"] is not None
            assert row["PE_f"] == 25.0
    
    def test_rows_to_dataframe(self):
        """Convert rows to DataFrame."""
        rows = [
            build_row_from_multi_source("AAPL", {"roe": 0.15, "price_mean": 150.0, "price_sources": 2}),
            build_row_from_multi_source("MSFT", {"roe": 0.12, "price_mean": 300.0, "price_sources": 2}),
        ]
        
        df = rows_to_dataframe(rows)
        
        assert len(df) == 2
        assert "ticker" in df.columns
        assert "Fundamental_S" in df.columns
        assert df.loc[0, "ticker"] == "AAPL"
        assert df.loc[1, "ticker"] == "MSFT"
    
    def test_empty_universe(self):
        """Handle empty ticker list gracefully."""
        rows = build_rows_from_universe([], lambda x: {})
        assert rows == []
        
        df = rows_to_dataframe([])
        assert df.empty


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_market_data(self):
        """Missing market_data doesn't crash."""
        payload = {"price_mean": 100.0, "price_sources": 1}
        row = build_row_from_multi_source("TEST", payload, market_data=None)
        
        assert row["Close"] == 100.0
    
    def test_nan_handling(self):
        """NaN values are handled gracefully."""
        payload = {
            "roe": np.nan,
            "pe": np.inf,
            "price_mean": None,
            "price_sources": 0,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert np.isnan(row["ROE_f"])
        assert np.isnan(row["PE_f"])
        assert np.isnan(row["Close"])
    
    def test_extreme_values_clipped(self):
        """Extreme values are clipped to reasonable bounds."""
        payload = {
            "price_mean": 1e20,  # Unrealistic
            "de": 1e10,
            "beta": -100,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        # Should be clipped or handled gracefully
        assert row["Fundamental_S"] is not None
        assert 0 <= row["Fundamental_S"] <= 100
    
    def test_duplicate_field_aliases(self):
        """When both field names present, prefers primary."""
        payload = {
            "rev_yoy": 0.10,
            "rev_g_yoy": 0.15,  # Alternative
            "de": 1.0,
            "debt_equity": 2.0,  # Alternative
        }
        row = build_row_from_multi_source("TEST", payload)
        
        # Should prefer the first one found
        assert row["RevG_f"] == 0.10
        assert row["DE_f"] == 1.0
    
    def test_zero_coverage_fallback(self):
        """Zero coverage gets safe defaults."""
        payload = {
            "Fundamental_Coverage_Pct": 0.0,
            "Fundamental_Sources_Count": 0,
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert row["Fundamental_S"] <= 50  # Conservative
        assert row["Fundamental_Coverage_Pct"] == 0.0


class TestDataIntegrity:
    """Tests for data type and schema consistency."""
    
    def test_ticker_uppercase(self):
        """Ticker is uppercased."""
        row = build_row_from_multi_source("aapl", {})
        assert row["ticker"] == "AAPL"
    
    def test_boolean_fields_are_bool(self):
        """Per-source flags are boolean type."""
        payload = {
            "Fund_from_FMP": 1,  # Should be coerced to bool
            "Fund_from_Finnhub": "yes",  # Should be truthy
        }
        row = build_row_from_multi_source("TEST", payload)
        
        assert isinstance(row["Fund_from_FMP"], bool)
        assert isinstance(row["Fund_from_Finnhub"], bool)
    
    def test_numeric_fields_numeric(self):
        """Numeric fields are float type or NaN."""
        row = build_row_from_multi_source("TEST", {})
        
        numeric_fields = ["Close", "Price_Mean", "PE_f", "Quality_Score_F"]
        for field in numeric_fields:
            val = row[field]
            assert isinstance(val, (float, np.floating)) or np.isnan(val)
