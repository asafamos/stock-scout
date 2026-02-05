"""Tests for core.price_verify module."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestPriceProviderBadges:
    """Tests for price provider badge mappings."""
    
    def test_all_badges_defined(self):
        """Ensure all expected providers have badges."""
        from core.price_verify import PRICE_PROVIDER_BADGES
        
        expected_providers = ["yahoo", "alpha", "finnhub", "polygon", "tiingo", 
                            "marketstack", "nasdaq", "eodhd", "fmp"]
        
        for provider in expected_providers:
            assert provider in PRICE_PROVIDER_BADGES
            assert PRICE_PROVIDER_BADGES[provider]  # Not empty
    
    def test_badges_contain_emoji(self):
        """All badges should contain emoji indicator."""
        from core.price_verify import PRICE_PROVIDER_BADGES
        
        for provider, badge in PRICE_PROVIDER_BADGES.items():
            # Badge should have emoji prefix (non-ASCII start)
            assert ord(badge[0]) > 127 or badge[0] in "🟡🟣🔵🟢🟠🧩🏛📘💰"


class TestComputePriceStats:
    """Tests for price statistics computation."""
    
    def test_single_price(self):
        """Single price should return that price with 0 std."""
        from core.price_verify import compute_price_stats
        
        mean, std, count = compute_price_stats({"Provider1": 100.0})
        
        assert mean == 100.0
        assert std == 0.0 or np.isnan(std)  # Single value has no std
        assert count == 1
    
    def test_multiple_prices(self):
        """Multiple prices should compute correct stats."""
        from core.price_verify import compute_price_stats
        
        prices = {"A": 100.0, "B": 102.0, "C": 98.0}
        mean, std, count = compute_price_stats(prices)
        
        assert mean == pytest.approx(100.0, rel=0.01)
        assert std > 0  # Should have some variance
        assert count == 3
    
    def test_with_none_values(self):
        """None values should be filtered out."""
        from core.price_verify import compute_price_stats
        
        prices = {"A": 100.0, "B": None, "C": 102.0}
        mean, std, count = compute_price_stats(prices)
        
        assert count == 2
        assert mean == pytest.approx(101.0, rel=0.01)
    
    def test_with_nan_values(self):
        """NaN values should be filtered out."""
        from core.price_verify import compute_price_stats
        
        prices = {"A": 100.0, "B": np.nan, "C": 102.0}
        mean, std, count = compute_price_stats(prices)
        
        assert count == 2
    
    def test_empty_prices(self):
        """Empty prices dict should return NaN."""
        from core.price_verify import compute_price_stats
        
        mean, std, count = compute_price_stats({})
        
        assert np.isnan(mean)
        assert np.isnan(std)
        assert count == 0
    
    def test_all_none_prices(self):
        """All None prices should return NaN."""
        from core.price_verify import compute_price_stats
        
        prices = {"A": None, "B": None}
        mean, std, count = compute_price_stats(prices)
        
        assert np.isnan(mean)
        assert count == 0


class TestFormatSourceBadges:
    """Tests for source badge formatting."""
    
    def test_multiple_badges(self):
        """Multiple badges should be joined."""
        from core.price_verify import format_source_badges
        
        badges = ["🟡Yahoo", "🔵Finnhub", "🟢Polygon"]
        result = format_source_badges(badges)
        
        assert "🟡Yahoo" in result
        assert "🔵Finnhub" in result
        assert "🟢Polygon" in result
    
    def test_empty_badges(self):
        """Empty list should return Yahoo default."""
        from core.price_verify import format_source_badges
        
        result = format_source_badges([])
        
        assert result == "🟡Yahoo"
    
    def test_custom_separator(self):
        """Custom separator should be used."""
        from core.price_verify import format_source_badges
        
        badges = ["A", "B"]
        result = format_source_badges(badges, separator=" | ")
        
        assert result == "A | B"


class TestFetchPricesForTicker:
    """Tests for main price fetching function."""
    
    @patch("core.price_verify.fetch_price_multi_source")
    def test_includes_yahoo_price(self, mock_fetch):
        """Yahoo price should be included when provided."""
        from core.price_verify import fetch_prices_for_ticker
        
        mock_fetch.return_value = {}
        
        vals, srcs = fetch_prices_for_ticker("AAPL", yahoo_price=150.0, track_usage=False)
        
        assert vals["Yahoo"] == 150.0
        assert "🟡Yahoo" in srcs
    
    @patch("core.price_verify.fetch_price_multi_source")
    def test_maps_v2_providers(self, mock_fetch):
        """Provider names should be mapped correctly."""
        from core.price_verify import fetch_prices_for_ticker
        
        mock_fetch.return_value = {
            "fmp": 150.5,
            "finnhub": 150.3,
            "polygon": 150.2,
        }
        
        vals, srcs = fetch_prices_for_ticker("AAPL", track_usage=False)
        
        assert "FMP" in vals
        assert "Finnhub" in vals
        assert "Polygon" in vals
    
    @patch("core.price_verify.fetch_price_multi_source")
    def test_filters_none_prices(self, mock_fetch):
        """None prices should not be included."""
        from core.price_verify import fetch_prices_for_ticker
        
        mock_fetch.return_value = {
            "fmp": 150.5,
            "finnhub": None,
            "polygon": np.nan,
        }
        
        vals, srcs = fetch_prices_for_ticker("AAPL", track_usage=False)
        
        assert "FMP" in vals
        assert "Finnhub" not in vals
        assert "Polygon" not in vals


class TestFetchExternalPricesForVerification:
    """Tests for drop-in replacement function."""
    
    @patch("core.price_verify.fetch_price_multi_source")
    def test_returns_tuple_format(self, mock_fetch):
        """Should return (ticker, vals, srcs) tuple."""
        from core.price_verify import fetch_external_prices_for_verification
        
        mock_fetch.return_value = {"fmp": 150.0}
        
        result = fetch_external_prices_for_verification("AAPL", 149.5)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        ticker, vals, srcs = result
        assert ticker == "AAPL"
        assert isinstance(vals, dict)
        assert isinstance(srcs, list)


class TestPricesDictToRowValues:
    """Tests for DataFrame row conversion."""
    
    def test_correct_column_order(self):
        """Values should match PRICE_COLUMNS order."""
        from core.price_verify import prices_dict_to_row_values, PRICE_COLUMNS
        
        vals = {"Alpha": 100.0, "Finnhub": 101.0}
        srcs = ["🟣Alpha", "🔵Finnhub"]
        
        row = prices_dict_to_row_values(vals, srcs)
        
        assert len(row) == len(PRICE_COLUMNS)
        assert row[0] == 100.0  # Price_Alpha
        assert row[1] == 101.0  # Price_Finnhub
    
    def test_missing_providers_are_nan(self):
        """Missing providers should be NaN."""
        from core.price_verify import prices_dict_to_row_values
        
        vals = {"Alpha": 100.0}
        srcs = ["🟣Alpha"]
        
        row = prices_dict_to_row_values(vals, srcs)
        
        assert row[0] == 100.0  # Alpha present
        assert np.isnan(row[1])  # Finnhub missing
        assert np.isnan(row[2])  # Polygon missing
