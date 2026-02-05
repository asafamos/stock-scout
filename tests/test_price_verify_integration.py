"""
Integration tests to verify that price_verify module returns compatible results.

These tests verify that:
1. The new price_verify module returns same format as stock_scout functions
2. All provider badges match expected format
3. Stats computation is correct
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import from new module
from core.price_verify import (
    fetch_prices_for_ticker,
    fetch_external_prices_for_verification,
    compute_price_stats,
    format_source_badges,
    prices_dict_to_row_values,
    PRICE_PROVIDER_BADGES,
    PRICE_COLUMNS,
)


class TestFetchExternalForCompatibility:
    """Test that fetch_external_prices_for_verification matches expected interface."""
    
    def test_returns_tuple_of_three(self):
        """Verify return type matches stock_scout._fetch_external_for."""
        with patch("core.price_verify.fetch_price_multi_source") as mock:
            mock.return_value = {"fmp": 150.0, "finnhub": 150.5}
            
            result = fetch_external_prices_for_verification(
                ticker="AAPL",
                yahoo_price=150.25,
                provider_status=None,
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 3
            
            ticker, vals, srcs = result
            assert ticker == "AAPL"
            assert isinstance(vals, dict)
            assert isinstance(srcs, list)
    
    def test_includes_yahoo_price(self):
        """Verify Yahoo price is included when provided."""
        with patch("core.price_verify.fetch_price_multi_source") as mock:
            mock.return_value = {}
            
            _, vals, srcs = fetch_external_prices_for_verification(
                ticker="MSFT",
                yahoo_price=300.0,
            )
            
            assert "Yahoo" in vals
            assert vals["Yahoo"] == 300.0
            assert "🟡Yahoo" in srcs
    
    def test_excludes_nan_yahoo(self):
        """Verify NaN Yahoo price is excluded."""
        with patch("core.price_verify.fetch_price_multi_source") as mock:
            mock.return_value = {"fmp": 150.0}
            
            _, vals, srcs = fetch_external_prices_for_verification(
                ticker="TEST",
                yahoo_price=np.nan,
            )
            
            assert "Yahoo" not in vals
            assert "🟡Yahoo" not in srcs


class TestPriceStats:
    """Test price statistics computation."""
    
    def test_empty_prices(self):
        """Test with no prices."""
        mean, std, count = compute_price_stats({})
        assert np.isnan(mean)
        assert np.isnan(std)
        assert count == 0
    
    def test_single_price(self):
        """Test with single price."""
        mean, std, count = compute_price_stats({"Yahoo": 100.0})
        assert mean == 100.0
        assert np.isnan(std)  # STD undefined for single value
        assert count == 1
    
    def test_multiple_prices(self):
        """Test with multiple prices."""
        prices = {"Yahoo": 100.0, "Finnhub": 101.0, "Polygon": 99.0}
        mean, std, count = compute_price_stats(prices)
        
        assert abs(mean - 100.0) < 0.01
        assert std > 0
        assert count == 3
    
    def test_ignores_none_values(self):
        """Test that None values are ignored."""
        prices = {"Yahoo": 100.0, "Finnhub": None, "Polygon": 102.0}
        mean, std, count = compute_price_stats(prices)
        
        assert abs(mean - 101.0) < 0.01
        assert count == 2


class TestSourceBadges:
    """Test source badge formatting."""
    
    def test_empty_sources(self):
        """Test with no sources returns default."""
        result = format_source_badges([])
        assert result == "🟡Yahoo"
    
    def test_single_source(self):
        """Test with single source."""
        result = format_source_badges(["🟡Yahoo"])
        assert result == "🟡Yahoo"
    
    def test_multiple_sources(self):
        """Test with multiple sources."""
        sources = ["🟡Yahoo", "🔵Finnhub", "🟢Polygon"]
        result = format_source_badges(sources)
        assert "🟡Yahoo" in result
        assert "🔵Finnhub" in result
        assert "🟢Polygon" in result
    
    def test_custom_separator(self):
        """Test with custom separator."""
        sources = ["🟡Yahoo", "🔵Finnhub"]
        result = format_source_badges(sources, separator="|")
        assert result == "🟡Yahoo|🔵Finnhub"


class TestPriceColumnsCompatibility:
    """Test that price columns match expected format."""
    
    def test_price_columns_exist(self):
        """Verify all expected columns are defined."""
        expected = [
            "Price_Alpha", "Price_Finnhub", "Price_Polygon",
            "Price_Tiingo", "Price_Marketstack", "Price_NasdaqDL",
            "Price_EODHD", "Price_FMP", "Price_Mean", "Price_STD",
            "Source_List"
        ]
        for col in expected:
            assert col in PRICE_COLUMNS, f"Missing column: {col}"
    
    def test_prices_dict_to_row_values_length(self):
        """Verify row values match columns length."""
        vals = {"Yahoo": 100.0, "Finnhub": 101.0}
        srcs = ["🟡Yahoo", "🔵Finnhub"]
        
        row = prices_dict_to_row_values(vals, srcs)
        assert len(row) == len(PRICE_COLUMNS)


class TestProviderBadges:
    """Test provider badge mappings."""
    
    def test_all_expected_badges_exist(self):
        """Verify all expected provider badges are defined."""
        expected_providers = [
            "yahoo", "alpha", "finnhub", "polygon",
            "tiingo", "marketstack", "nasdaq", "eodhd", "fmp"
        ]
        for provider in expected_providers:
            assert provider in PRICE_PROVIDER_BADGES, f"Missing badge for: {provider}"
    
    def test_badges_have_emoji(self):
        """Verify all badges start with emoji."""
        for provider, badge in PRICE_PROVIDER_BADGES.items():
            # Most badges start with emoji (unicode ord > 127)
            assert ord(badge[0]) > 127, f"Badge for {provider} missing emoji: {badge}"


class TestProviderTracking:
    """Test provider usage tracking."""
    
    def test_tracking_does_not_fail_outside_streamlit(self):
        """Verify tracking gracefully handles non-Streamlit context."""
        # This should not raise even without Streamlit
        with patch("core.price_verify.fetch_price_multi_source") as mock:
            mock.return_value = {"fmp": 150.0}
            
            # Should not raise
            result = fetch_external_prices_for_verification(
                ticker="TEST",
                yahoo_price=150.0,
            )
            assert result is not None
