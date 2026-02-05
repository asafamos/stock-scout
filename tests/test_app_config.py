"""Tests for app_config module."""
import pytest


class TestConfigDict:
    """Tests for CONFIG dictionary."""
    
    def test_config_is_dict(self):
        """CONFIG should be a dictionary."""
        from app_config import CONFIG
        
        assert isinstance(CONFIG, dict)
    
    def test_config_has_required_keys(self):
        """CONFIG should have essential keys."""
        from app_config import CONFIG
        
        required_keys = [
            "UNIVERSE_LIMIT",
            "LOOKBACK_DAYS",
            "MA_SHORT",
            "MA_LONG",
            "RSI_BOUNDS",
            "WEIGHTS",
            "BUDGET_TOTAL",
            "MIN_PRICE",
            "FUNDAMENTAL_ENABLED",
        ]
        
        for key in required_keys:
            assert key in CONFIG, f"Missing required key: {key}"
    
    def test_config_values_are_valid_types(self):
        """CONFIG values should be valid types."""
        from app_config import CONFIG
        
        # Numeric values
        assert isinstance(CONFIG["UNIVERSE_LIMIT"], int)
        assert isinstance(CONFIG["LOOKBACK_DAYS"], int)
        assert isinstance(CONFIG["BUDGET_TOTAL"], (int, float))
        assert isinstance(CONFIG["MIN_PRICE"], (int, float))
        
        # Boolean values
        assert isinstance(CONFIG["FUNDAMENTAL_ENABLED"], bool)
        
        # Tuple values
        assert isinstance(CONFIG["RSI_BOUNDS"], tuple)
        assert len(CONFIG["RSI_BOUNDS"]) == 2


class TestFundSchema:
    """Tests for fundamentals schema fields."""
    
    def test_fund_schema_fields_defined(self):
        """FUND_SCHEMA_FIELDS should have expected fields."""
        from app_config import FUND_SCHEMA_FIELDS
        
        expected = ["roe", "roic", "gm", "ps", "pe", "de", "rev_g_yoy", "eps_g_yoy"]
        
        for field in expected:
            assert field in FUND_SCHEMA_FIELDS
    
    def test_fund_string_fields_defined(self):
        """FUND_STRING_FIELDS should have sector and industry."""
        from app_config import FUND_STRING_FIELDS
        
        assert "sector" in FUND_STRING_FIELDS
        assert "industry" in FUND_STRING_FIELDS


class TestEmptyFundRow:
    """Tests for empty_fund_row function."""
    
    def test_returns_dict(self):
        """empty_fund_row should return a dictionary."""
        from app_config import empty_fund_row
        
        result = empty_fund_row()
        
        assert isinstance(result, dict)
    
    def test_has_all_schema_fields(self):
        """Result should have all schema fields."""
        from app_config import empty_fund_row, FUND_SCHEMA_FIELDS
        import numpy as np
        
        result = empty_fund_row()
        
        for field in FUND_SCHEMA_FIELDS:
            assert field in result
            assert np.isnan(result[field])
    
    def test_has_string_fields_with_defaults(self):
        """String fields should have default values."""
        from app_config import empty_fund_row
        
        result = empty_fund_row()
        
        assert result["sector"] == "Unknown"
        assert result["industry"] == "Unknown"
    
    def test_has_metadata_fields(self):
        """Result should have metadata fields."""
        from app_config import empty_fund_row
        
        result = empty_fund_row()
        
        assert "_sources" in result
        assert "_sources_used" in result
        assert "Fund_Coverage_Pct" in result
        assert "fundamentals_available" in result
        
        assert result["Fund_Coverage_Pct"] == 0.0
        assert result["fundamentals_available"] is False
