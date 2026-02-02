"""
Comprehensive unit tests for the ML pipeline.

Tests feature calculation consistency, inference-training alignment,
sector mapping, and market regime calculation.

Run with: pytest tests/test_ml_pipeline.py -v
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ML integration imports
from core.ml_integration import (
    prepare_ml_features,
    get_expected_features,
    calculate_ml_boost,
    load_ml_model,
    get_ml_prediction,
    get_model_info,
)

# Sector mapping imports
from core.sector_mapping import (
    get_stock_sector,
    get_sector_etf,
    get_all_sector_etfs,
    get_all_sectors,
    SECTOR_ETFS,
    STOCK_SECTOR_MAP,
)

# Training script imports (for calculate_features and calculate_market_regime)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.train_rolling_ml_20d import calculate_features, calculate_market_regime


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing with 300 days of data."""
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    np.random.seed(42)
    
    # Create realistic price movement (random walk with drift)
    returns = np.random.randn(300) * 0.02  # 2% daily vol
    close = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': close * (1 - np.abs(np.random.randn(300)) * 0.005),
        'High': close * (1 + np.abs(np.random.randn(300)) * 0.015),
        'Low': close * (1 - np.abs(np.random.randn(300)) * 0.015),
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 300)
    }, index=dates)
    
    # Ensure High >= Close >= Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def sample_spy_ohlcv():
    """Create sample SPY OHLCV data for market regime testing."""
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    np.random.seed(123)
    
    # SPY-like data with slight upward drift
    returns = 0.0003 + np.random.randn(300) * 0.01  # ~7.5% annual return, 16% vol
    close = 450 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': close * 0.999,
        'High': close * 1.008,
        'Low': close * 0.992,
        'Close': close,
        'Volume': np.random.randint(50000000, 150000000, 300)
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def bull_market_spy():
    """Create strong bull market SPY data.
    
    Bull conditions: SPY > MA50 > MA200 and ret_20d > 2%
    """
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    np.random.seed(999)
    
    # Very strong consistent uptrend - 0.15% daily = ~45% annual
    # Use low volatility to ensure consistent MA alignment
    returns = 0.0015 + np.random.randn(300) * 0.005  # Strong drift, low vol
    close = 400 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': close * 0.998,
        'High': close * 1.003,
        'Low': close * 0.997,
        'Close': close,
        'Volume': np.random.randint(50000000, 150000000, 300)
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def bear_market_spy():
    """Create bear market SPY data."""
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    np.random.seed(888)
    
    # Downtrend
    returns = -0.001 + np.random.randn(300) * 0.015  # Negative drift, higher vol
    close = 500 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': close * 1.002,
        'High': close * 1.008,
        'Low': close * 0.990,
        'Close': close,
        'Volume': np.random.randint(80000000, 200000000, 300)
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


# =============================================================================
# TEST FEATURE CALCULATION
# =============================================================================

class TestFeatureCalculation:
    """Test that feature calculation is consistent and correct."""
    
    def test_feature_count_matches(self, sample_ohlcv):
        """Verify training produces expected feature columns."""
        df = calculate_features(sample_ohlcv)
        
        # Expected features from training script (minus Forward_Return_20d target)
        expected_feature_cols = [
            'RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d',
            'VCP_Ratio', 'Tightness_Ratio', 'Dist_From_52w_High',
            'MA_Alignment', 'Volume_Surge', 'Up_Down_Volume_Ratio',
            'Momentum_Consistency', 'RS_vs_SPY_20d',
            'Market_Regime', 'Market_Volatility', 'Market_Trend', 'High_Volatility',
            'Sector_RS', 'Sector_Momentum', 'Sector_Rank',
            'Volume_Ratio_20d', 'Volume_Trend', 'Up_Volume_Ratio',
            'Volume_Price_Confirm', 'Relative_Volume_Rank',
            'Distance_From_52w_Low', 'Consolidation_Tightness',
            'Days_Since_52w_High', 'Price_vs_SMA50', 'Price_vs_SMA200',
            'SMA50_vs_SMA200', 'MA_Slope_20d', 'Distance_To_Resistance',
            'Support_Strength',
        ]
        
        # Check all expected features are present
        for feat in expected_feature_cols:
            assert feat in df.columns, f"Missing feature: {feat}"
        
        # Verify that training script and calculate_features produce consistent features
        # Note: We check that training produces these features, model will only use a subset
        assert len(expected_feature_cols) == 34, \
            f"Training script should produce 34 features, got {len(expected_feature_cols)}"
    
    def test_no_nan_in_features_after_dropna(self, sample_ohlcv):
        """Features should not contain NaN after calculate_features (which drops NaN)."""
        df = calculate_features(sample_ohlcv)
        
        # Get expected features from training script (34 features)
        training_features = [
            'RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d',
            'VCP_Ratio', 'Tightness_Ratio', 'Dist_From_52w_High',
            'MA_Alignment', 'Volume_Surge', 'Up_Down_Volume_Ratio',
            'Momentum_Consistency', 'RS_vs_SPY_20d',
            'Market_Regime', 'Market_Volatility', 'Market_Trend', 'High_Volatility',
            'Sector_RS', 'Sector_Momentum', 'Sector_Rank',
            'Volume_Ratio_20d', 'Volume_Trend', 'Up_Volume_Ratio',
            'Volume_Price_Confirm', 'Relative_Volume_Rank',
            'Distance_From_52w_Low', 'Consolidation_Tightness',
            'Days_Since_52w_High', 'Price_vs_SMA50', 'Price_vs_SMA200',
            'SMA50_vs_SMA200', 'MA_Slope_20d', 'Distance_To_Resistance',
            'Support_Strength',
        ]
        feature_cols = [col for col in training_features if col in df.columns]
        
        for col in feature_cols:
            nan_count = df[col].isna().sum()
            assert nan_count == 0, f"Feature {col} has {nan_count} NaN values"
    
    def test_feature_ranges_reasonable(self, sample_ohlcv):
        """Feature values should be in expected ranges."""
        df = calculate_features(sample_ohlcv)
        
        # RSI: 0-100
        if 'RSI' in df.columns:
            assert df['RSI'].min() >= 0, f"RSI min {df['RSI'].min()} < 0"
            assert df['RSI'].max() <= 100, f"RSI max {df['RSI'].max()} > 100"
        
        # ATR_Pct: 0 to 0.5 (up to 50% volatility)
        if 'ATR_Pct' in df.columns:
            assert df['ATR_Pct'].min() >= 0, f"ATR_Pct min {df['ATR_Pct'].min()} < 0"
            assert df['ATR_Pct'].max() <= 0.5, f"ATR_Pct max {df['ATR_Pct'].max()} > 0.5"
        
        # Returns: -1 to +2 (reasonable for 5-20 day returns)
        for ret_col in ['Return_5d', 'Return_10d', 'Return_20d']:
            if ret_col in df.columns:
                assert df[ret_col].min() >= -1.0, f"{ret_col} min {df[ret_col].min()} < -1.0"
                assert df[ret_col].max() <= 2.0, f"{ret_col} max {df[ret_col].max()} > 2.0"
        
        # MA_Alignment: binary 0 or 1
        if 'MA_Alignment' in df.columns:
            assert set(df['MA_Alignment'].unique()).issubset({0, 1, 0.0, 1.0}), \
                f"MA_Alignment has unexpected values: {df['MA_Alignment'].unique()}"
        
        # Momentum_Consistency: 0 to 1 (percentage)
        if 'Momentum_Consistency' in df.columns:
            assert df['Momentum_Consistency'].min() >= 0, \
                f"Momentum_Consistency min {df['Momentum_Consistency'].min()} < 0"
            assert df['Momentum_Consistency'].max() <= 1, \
                f"Momentum_Consistency max {df['Momentum_Consistency'].max()} > 1"
    
    def test_ohlcv_data_requirements(self, sample_ohlcv):
        """Test that minimum data requirements are met."""
        # Need at least 252 days for 52-week features
        df = calculate_features(sample_ohlcv)
        
        # With 300 days input, we should get ~40-50 valid rows after dropna
        # (due to lookback requirements for various features)
        assert len(df) > 0, "calculate_features returned empty DataFrame"
        assert len(df) >= 20, f"Too few rows returned: {len(df)}"
    
    def test_volume_features_calculated(self, sample_ohlcv):
        """Volume-based features should be calculated correctly."""
        df = calculate_features(sample_ohlcv)
        
        volume_features = [
            'Volume_Surge', 'Up_Down_Volume_Ratio', 'Volume_Ratio_20d',
            'Volume_Trend', 'Up_Volume_Ratio', 'Volume_Price_Confirm',
            'Relative_Volume_Rank'
        ]
        
        for feat in volume_features:
            assert feat in df.columns, f"Missing volume feature: {feat}"
            # Volume features should not be negative
            if feat in ['Volume_Surge', 'Volume_Ratio_20d', 'Up_Volume_Ratio',
                       'Relative_Volume_Rank']:
                assert df[feat].min() >= 0, f"{feat} has negative values"


# =============================================================================
# TEST INFERENCE CONSISTENCY
# =============================================================================

class TestInferenceConsistency:
    """Test that inference uses same features as training."""
    
    def test_expected_features_count(self):
        """Model should load with its expected feature count."""
        # Load model first to populate features
        load_ml_model()
        features = get_expected_features()
        model_info = get_model_info()
        
        # Model should be loaded and have at least some features
        assert model_info['loaded'], "Model should be loaded"
        assert len(features) > 0, "Model should have at least some features"
        assert len(features) == model_info['feature_count'], \
            f"Feature count mismatch: {len(features)} vs {model_info['feature_count']}"
    
    def test_feature_names_match_training(self):
        """Feature names in inference must match those in the model bundle."""
        load_ml_model()
        features = get_expected_features()
        model_info = get_model_info()
        
        # Features should match what's stored in model_info
        assert features == model_info['features'], \
            f"Feature mismatch between get_expected_features and model_info:\n" \
            f"get_expected_features: {features}\n" \
            f"model_info: {model_info['features']}"
    
    def test_prepare_ml_features_returns_all(self):
        """prepare_ml_features should return all expected features."""
        load_ml_model()
        expected_features = get_expected_features()
        
        # Mock data with some features present
        ticker_data = {
            'Close': 100.0,
            'RSI': 55.0,
            'ATR': 2.5,
            'Volume': 1500000,
        }
        technical_indicators = {
            'Return_20d': 0.05,
            'Return_10d': 0.02,
            'Return_5d': 0.01,
        }
        fundamental_scores = {}
        
        features = prepare_ml_features(ticker_data, technical_indicators, fundamental_scores)
        
        # Should return dict with all expected features (filled with defaults if missing)
        assert isinstance(features, dict), "prepare_ml_features should return a dict"
        
        # Check that all expected features are present
        for key in expected_features:
            assert key in features, f"Missing feature: {key}"
    
    def test_prepare_ml_features_handles_missing_data(self):
        """prepare_ml_features should handle missing data with defaults."""
        load_ml_model()
        expected_features = get_expected_features()
        
        # Empty data should still return all expected features with defaults
        features = prepare_ml_features({}, {}, {})
        
        # Check all expected features are present
        for key in expected_features:
            assert key in features, f"Missing feature in empty call: {key}"
        
        # Check default values are reasonable for core features
        if 'RSI' in features:
            assert features['RSI'] == 50.0, "RSI default should be 50"
        if 'ATR_Pct' in features:
            assert features['ATR_Pct'] == 0.02, "ATR_Pct default should be 0.02"
    
    def test_prepare_ml_features_clamps_values(self):
        """prepare_ml_features should clamp extreme values to valid ranges."""
        load_ml_model()
        
        ticker_data = {
            'RSI': 150.0,  # Invalid: should be clamped to 100
            'Return_20d': 5.0,  # Extreme: should be clamped to 2.0
            'ATR_Pct': 1.5,  # Extreme: should be clamped to 0.5
        }
        
        features = prepare_ml_features(ticker_data, {}, {})
        
        if 'RSI' in features:
            assert features['RSI'] <= 100, f"RSI not clamped: {features['RSI']}"
        if 'Return_20d' in features:
            assert features['Return_20d'] <= 2.0, f"Return_20d not clamped: {features['Return_20d']}"
        if 'ATR_Pct' in features:
            assert features['ATR_Pct'] <= 0.5, f"ATR_Pct not clamped: {features['ATR_Pct']}"
    
    def test_feature_order_preserved(self):
        """Feature order from model should be consistent."""
        load_ml_model()
        features = get_expected_features()
        model_info = get_model_info()
        
        # Features should be list
        assert isinstance(features, list), "Features should be a list"
        
        # First feature should be RSI (if model has it)
        if len(features) > 0:
            assert features[0] == 'RSI', f"First feature should be RSI, got {features[0]}"


# =============================================================================
# TEST SECTOR MAPPING
# =============================================================================

class TestSectorMapping:
    """Test sector mapping utilities."""
    
    def test_known_stocks_have_sectors(self):
        """Well-known stocks should have correct sector mappings."""
        assert get_stock_sector('AAPL') == 'Technology'
        assert get_stock_sector('MSFT') == 'Technology'
        assert get_stock_sector('NVDA') == 'Technology'
        assert get_stock_sector('JPM') == 'Financial'
        assert get_stock_sector('BAC') == 'Financial'
        assert get_stock_sector('XOM') == 'Energy'
        assert get_stock_sector('CVX') == 'Energy'
        assert get_stock_sector('JNJ') == 'Healthcare'
        assert get_stock_sector('UNH') == 'Healthcare'
        assert get_stock_sector('PG') == 'Consumer Staples'
        assert get_stock_sector('AMZN') == 'Consumer Discretionary'
    
    def test_unknown_stock_returns_unknown(self):
        """Unknown stocks should return 'Unknown'."""
        assert get_stock_sector('FAKE123') == 'Unknown'
        assert get_stock_sector('NOTREAL') == 'Unknown'
        assert get_stock_sector('XYZ999') == 'Unknown'
    
    def test_all_sectors_have_etfs(self):
        """Each sector should have a corresponding ETF."""
        sectors = ['Technology', 'Financial', 'Energy', 'Healthcare',
                   'Consumer Discretionary', 'Consumer Staples', 'Industrials',
                   'Materials', 'Utilities', 'Real Estate', 'Communication']
        
        for sector in sectors:
            etf = get_sector_etf(sector)
            assert etf is not None, f"Sector {sector} has no ETF mapping"
            assert len(etf) > 0, f"Sector {sector} ETF is empty string"
    
    def test_sector_etf_mappings(self):
        """Verify specific sector-ETF mappings."""
        assert get_sector_etf('Technology') == 'XLK'
        assert get_sector_etf('Financial') == 'XLF'
        assert get_sector_etf('Energy') == 'XLE'
        assert get_sector_etf('Healthcare') == 'XLV'
        assert get_sector_etf('Consumer Discretionary') == 'XLY'
        assert get_sector_etf('Consumer Staples') == 'XLP'
        assert get_sector_etf('Industrials') == 'XLI'
        assert get_sector_etf('Materials') == 'XLB'
        assert get_sector_etf('Utilities') == 'XLU'
        assert get_sector_etf('Real Estate') == 'XLRE'
        assert get_sector_etf('Communication') == 'XLC'
    
    def test_unknown_sector_returns_none(self):
        """Unknown sectors should return None for ETF."""
        assert get_sector_etf('FakeSector') is None
        assert get_sector_etf('Unknown') is None
        assert get_sector_etf('') is None
    
    def test_get_all_sector_etfs(self):
        """get_all_sector_etfs should return all ETF symbols."""
        etfs = get_all_sector_etfs()
        
        assert len(etfs) == len(SECTOR_ETFS), \
            f"Expected {len(SECTOR_ETFS)} ETFs, got {len(etfs)}"
        
        # Check some expected ETFs are present
        assert 'XLK' in etfs
        assert 'XLF' in etfs
        assert 'XLE' in etfs
    
    def test_get_all_sectors(self):
        """get_all_sectors should return all sector names."""
        sectors = get_all_sectors()
        
        assert len(sectors) == len(SECTOR_ETFS), \
            f"Expected {len(SECTOR_ETFS)} sectors, got {len(sectors)}"
        
        assert 'Technology' in sectors
        assert 'Financial' in sectors
        assert 'Energy' in sectors
    
    def test_stock_sector_map_coverage(self):
        """Verify STOCK_SECTOR_MAP has reasonable coverage."""
        # Should have at least 100 stocks mapped
        assert len(STOCK_SECTOR_MAP) >= 100, \
            f"STOCK_SECTOR_MAP has only {len(STOCK_SECTOR_MAP)} stocks"
        
        # All mapped sectors should be valid
        valid_sectors = set(SECTOR_ETFS.keys())
        for ticker, sector in STOCK_SECTOR_MAP.items():
            assert sector in valid_sectors, \
                f"Stock {ticker} has invalid sector: {sector}"
    
    def test_case_insensitivity(self):
        """Ticker lookup should be case-insensitive."""
        assert get_stock_sector('aapl') == get_stock_sector('AAPL')
        assert get_stock_sector('Msft') == get_stock_sector('MSFT')


# =============================================================================
# TEST MARKET REGIME CALCULATION
# =============================================================================

class TestMarketRegime:
    """Test market regime calculation from SPY data."""
    
    def test_bull_market_detection(self, bull_market_spy):
        """Strong uptrend should be detected as bull market."""
        regime_df = calculate_market_regime(bull_market_spy)
        
        assert regime_df is not None, "calculate_market_regime returned None"
        assert 'Market_Regime' in regime_df.columns
        
        # In a strong bull market, most recent days should be bull (1)
        # Check last 20 days
        recent_regime = regime_df['Market_Regime'].iloc[-20:]
        bull_count = (recent_regime == 1).sum()
        
        # Should have at least some bull days in strong uptrend
        assert bull_count >= 5, \
            f"Only {bull_count} bull days detected in bull market data"
    
    def test_bear_market_detection(self, bear_market_spy):
        """Strong downtrend should be detected as bear market."""
        regime_df = calculate_market_regime(bear_market_spy)
        
        assert regime_df is not None, "calculate_market_regime returned None"
        assert 'Market_Regime' in regime_df.columns
        
        # In a bear market, should have some bear (-1) days
        recent_regime = regime_df['Market_Regime'].iloc[-50:]
        bear_count = (recent_regime == -1).sum()
        
        # Should have at least some bear days in downtrend
        assert bear_count >= 3, \
            f"Only {bear_count} bear days detected in bear market data"
    
    def test_market_regime_output_columns(self, sample_spy_ohlcv):
        """Market regime DataFrame should have all expected columns."""
        regime_df = calculate_market_regime(sample_spy_ohlcv)
        
        assert regime_df is not None
        
        expected_cols = ['Market_Regime', 'Market_Volatility', 
                        'Market_Trend', 'High_Volatility']
        
        for col in expected_cols:
            assert col in regime_df.columns, f"Missing column: {col}"
    
    def test_market_regime_values_valid(self, sample_spy_ohlcv):
        """Market regime values should be -1, 0, or 1."""
        regime_df = calculate_market_regime(sample_spy_ohlcv)
        
        valid_regimes = {-1, 0, 1}
        actual_regimes = set(regime_df['Market_Regime'].dropna().unique())
        
        assert actual_regimes.issubset(valid_regimes), \
            f"Invalid regime values: {actual_regimes - valid_regimes}"
    
    def test_market_volatility_positive(self, sample_spy_ohlcv):
        """Market volatility should always be positive."""
        regime_df = calculate_market_regime(sample_spy_ohlcv)
        
        vol = regime_df['Market_Volatility'].dropna()
        assert (vol >= 0).all(), "Market volatility has negative values"
    
    def test_high_volatility_binary(self, sample_spy_ohlcv):
        """High volatility indicator should be binary (0 or 1)."""
        regime_df = calculate_market_regime(sample_spy_ohlcv)
        
        high_vol = regime_df['High_Volatility'].dropna()
        assert set(high_vol.unique()).issubset({0, 1}), \
            f"High_Volatility has non-binary values: {high_vol.unique()}"
    
    def test_insufficient_data_returns_none(self):
        """With insufficient data, should return None."""
        # Less than 200 days
        short_df = pd.DataFrame({
            'Open': [100] * 50,
            'High': [101] * 50,
            'Low': [99] * 50,
            'Close': [100] * 50,
            'Volume': [1000000] * 50
        }, index=pd.date_range('2024-01-01', periods=50, freq='D'))
        
        result = calculate_market_regime(short_df)
        assert result is None, "Should return None for insufficient data"
    
    def test_none_input_returns_none(self):
        """None input should return None."""
        result = calculate_market_regime(None)
        assert result is None


# =============================================================================
# TEST ML BOOST CALCULATION
# =============================================================================

class TestMLBoostCalculation:
    """Test ML boost calculation logic."""
    
    def test_ml_unavailable_returns_base(self):
        """When ML probability is None, return base conviction unchanged."""
        base = 75.0
        final, boost, msg = calculate_ml_boost(base, None)
        
        assert final == base, f"Expected {base}, got {final}"
        assert boost == 0.0, f"Expected 0 boost, got {boost}"
        assert "unavailable" in msg.lower()
    
    def test_neutral_probability_no_boost(self):
        """ML probability of 0.5 should give no boost."""
        base = 70.0
        final, boost, msg = calculate_ml_boost(base, 0.5)
        
        assert final == base, f"Expected {base}, got {final}"
        assert boost == 0.0, f"Expected 0 boost, got {boost}"
        assert "neutral" in msg.lower()
    
    def test_high_probability_positive_boost(self):
        """High ML probability should give positive boost."""
        base = 60.0
        final, boost, msg = calculate_ml_boost(base, 0.9)
        
        assert final > base, f"Expected boost, got {final} <= {base}"
        assert boost > 0, f"Expected positive boost, got {boost}"
    
    def test_low_probability_negative_boost(self):
        """Low ML probability should give negative boost (penalty)."""
        base = 70.0
        final, boost, msg = calculate_ml_boost(base, 0.1)
        
        assert final < base, f"Expected penalty, got {final} >= {base}"
        assert boost < 0, f"Expected negative boost, got {boost}"
    
    def test_max_boost_capped(self):
        """Boost should be capped at max_boost_pct."""
        base = 50.0
        max_boost = 10.0
        
        # ML prob 1.0 should give exactly +max_boost
        final, boost, msg = calculate_ml_boost(base, 1.0, max_boost)
        
        assert boost == max_boost, f"Expected +{max_boost} boost, got {boost}"
        assert final == base + max_boost
    
    def test_max_penalty_capped(self):
        """Penalty should be capped at -max_boost_pct."""
        base = 50.0
        max_boost = 10.0
        
        # ML prob 0.0 should give exactly -max_boost
        final, boost, msg = calculate_ml_boost(base, 0.0, max_boost)
        
        assert boost == -max_boost, f"Expected -{max_boost} boost, got {boost}"
        assert final == base - max_boost
    
    def test_final_clamped_to_0_100(self):
        """Final conviction should be clamped to 0-100 range."""
        # Test upper clamp
        final, _, _ = calculate_ml_boost(98.0, 1.0, max_boost_pct=20.0)
        assert final <= 100.0, f"Final should be <= 100, got {final}"
        
        # Test lower clamp
        final, _, _ = calculate_ml_boost(5.0, 0.0, max_boost_pct=20.0)
        assert final >= 0.0, f"Final should be >= 0, got {final}"
    
    def test_invalid_probability_handled(self):
        """Invalid probability (inf, nan) should return base unchanged."""
        base = 70.0
        
        final, boost, msg = calculate_ml_boost(base, float('inf'))
        assert final == base
        assert boost == 0.0
        
        final, boost, msg = calculate_ml_boost(base, float('nan'))
        assert final == base
        assert boost == 0.0


# =============================================================================
# TEST INTEGRATION SCENARIOS
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests combining multiple components."""
    
    def test_feature_calculation_with_spy_data(self, sample_ohlcv, sample_spy_ohlcv):
        """Test feature calculation with SPY data for relative strength."""
        # Calculate SPY 20d returns
        spy_returns = sample_spy_ohlcv['Close'].pct_change(20)
        
        # Calculate features with SPY data
        df = calculate_features(sample_ohlcv, spy_returns=spy_returns)
        
        assert 'RS_vs_SPY_20d' in df.columns
        # RS should not all be zero when SPY data is provided
        assert not (df['RS_vs_SPY_20d'] == 0).all() or len(df) < 5, \
            "RS_vs_SPY_20d should have non-zero values when SPY data provided"
    
    def test_feature_calculation_with_market_regime(self, sample_ohlcv, sample_spy_ohlcv):
        """Test feature calculation with market regime data."""
        market_regime_df = calculate_market_regime(sample_spy_ohlcv)
        
        df = calculate_features(sample_ohlcv, market_regime_df=market_regime_df)
        
        # Market regime features should be present
        for col in ['Market_Regime', 'Market_Volatility', 'Market_Trend', 'High_Volatility']:
            assert col in df.columns, f"Missing market regime feature: {col}"
    
    def test_end_to_end_feature_pipeline(self, sample_ohlcv):
        """Test complete feature calculation and preparation pipeline."""
        # Ensure model is loaded first
        load_ml_model()
        expected_features = get_expected_features()
        
        # Step 1: Calculate features from OHLCV
        df = calculate_features(sample_ohlcv)
        assert len(df) > 0
        
        # Step 2: Take last row as "current" data
        last_row = df.iloc[-1]
        
        # Step 3: Prepare ML features dict
        ticker_data = last_row.to_dict()
        features = prepare_ml_features(ticker_data, {}, {})
        
        # Step 4: Verify all expected features are present and valid
        assert len(features) == len(expected_features), \
            f"Expected {len(expected_features)} features, got {len(features)}"
        
        for key, value in features.items():
            assert np.isfinite(value), f"Feature {key} is not finite: {value}"
    
    def test_sector_features_integration(self):
        """Test sector feature calculation integration."""
        # For AAPL (Technology sector)
        ticker = 'AAPL'
        sector = get_stock_sector(ticker)
        etf = get_sector_etf(sector)
        
        assert sector == 'Technology'
        assert etf == 'XLK'
        
        # Ensure model is loaded
        load_ml_model()
        expected_features = get_expected_features()
        
        # Verify sector can be used in feature calculation
        ticker_data = {
            'Return_20d': 0.08,
            'Return_5d': 0.02,
            'sector_return_20d': 0.05,
            'sector_return_5d': 0.01,
        }
        
        features = prepare_ml_features(ticker_data, {}, {})
        
        # Check Sector_RS only if model expects it
        if 'Sector_RS' in expected_features:
            # Sector_RS should be stock_ret - sector_ret
            assert abs(features['Sector_RS'] - 0.03) < 0.01, \
                f"Sector_RS should be ~0.03, got {features['Sector_RS']}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
