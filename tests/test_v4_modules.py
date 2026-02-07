"""
Tests for V4 ML modules.

Tests:
- Feature registry
- Feature builder
- Monitoring
- Backtester
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


# ============================================================================
# Feature Registry V4 Tests
# ============================================================================
class TestFeatureRegistryV4:
    """Tests for core.feature_registry_v4"""
    
    def test_feature_count(self):
        """V4 should have 54 features."""
        from core.feature_registry_v4 import FEATURE_COUNT_V4, get_feature_names_v4
        
        assert FEATURE_COUNT_V4 == 54
        assert len(get_feature_names_v4()) == 54
    
    def test_feature_defaults_complete(self):
        """Every feature should have a default value."""
        from core.feature_registry_v4 import get_feature_names_v4, get_feature_defaults_v4
        
        names = get_feature_names_v4()
        defaults = get_feature_defaults_v4()
        
        for name in names:
            assert name in defaults, f"Feature {name} missing default"
    
    def test_feature_defaults_reasonable(self):
        """Default values should be reasonable (not inf/nan)."""
        from core.feature_registry_v4 import get_feature_defaults_v4
        
        defaults = get_feature_defaults_v4()
        
        for name, value in defaults.items():
            assert not np.isnan(value), f"Feature {name} has NaN default"
            assert not np.isinf(value), f"Feature {name} has inf default"
    
    def test_feature_specs_have_descriptions(self):
        """All features should have descriptions."""
        from core.feature_registry_v4 import FEATURE_SPECS_V4
        
        for spec in FEATURE_SPECS_V4:
            assert spec.description, f"Feature {spec.name} missing description"
            assert len(spec.description) > 5, f"Feature {spec.name} description too short"
    
    def test_feature_categories(self):
        """Features should be organized into categories."""
        from core.feature_registry_v4 import FEATURE_SPECS_V4
        
        categories = set(spec.category for spec in FEATURE_SPECS_V4)
        
        # Should have key categories
        expected_categories = {"technical", "volatility", "volume", "market", "sentiment"}
        for cat in expected_categories:
            assert cat in categories, f"Missing category: {cat}"


# ============================================================================
# Feature Builder V4 Tests
# ============================================================================
class TestFeatureBuilderV4:
    """Tests for core.ml_feature_builder_v4"""
    
    @pytest.fixture
    def sample_price_df(self):
        """Create sample price data."""
        dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
        np.random.seed(42)
        
        close = 100 * (1 + np.random.randn(300).cumsum() * 0.01)
        
        return pd.DataFrame({
            'Open': close * (1 + np.random.randn(300) * 0.005),
            'High': close * (1 + np.abs(np.random.randn(300)) * 0.01),
            'Low': close * (1 - np.abs(np.random.randn(300)) * 0.01),
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
    
    def test_build_features_returns_dict(self, sample_price_df):
        """build_features_v4 should return a dict."""
        from core.ml_feature_builder_v4 import build_features_v4
        
        features = build_features_v4(
            price_df=sample_price_df,
            ticker="TEST"
        )
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_build_features_has_required_keys(self, sample_price_df):
        """Features dict should have all required V4 features."""
        from core.ml_feature_builder_v4 import build_features_v4
        from core.feature_registry_v4 import get_feature_names_v4
        
        features = build_features_v4(
            price_df=sample_price_df,
            ticker="TEST"
        )
        
        required = get_feature_names_v4()
        for key in required:
            assert key in features, f"Missing feature: {key}"
    
    def test_build_features_no_nan(self, sample_price_df):
        """Features should not have NaN values."""
        from core.ml_feature_builder_v4 import build_features_v4
        
        features = build_features_v4(
            price_df=sample_price_df,
            ticker="TEST"
        )
        
        for key, value in features.items():
            if isinstance(value, float):
                assert not np.isnan(value), f"Feature {key} is NaN"
    
    def test_build_features_no_inf(self, sample_price_df):
        """Features should not have infinite values."""
        from core.ml_feature_builder_v4 import build_features_v4
        
        features = build_features_v4(
            price_df=sample_price_df,
            ticker="TEST"
        )
        
        for key, value in features.items():
            if isinstance(value, float):
                assert not np.isinf(value), f"Feature {key} is infinite"
    
    def test_build_features_rsi_range(self, sample_price_df):
        """RSI should be between 0 and 100."""
        from core.ml_feature_builder_v4 import build_features_v4
        
        features = build_features_v4(
            price_df=sample_price_df,
            ticker="TEST"
        )
        
        rsi = features.get("RSI", 50)
        assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"


# ============================================================================
# Monitoring Tests
# ============================================================================
class TestMonitoring:
    """Tests for core.monitoring"""
    
    def test_monitor_singleton(self):
        """Monitor should be a singleton."""
        from core.monitoring import Monitor
        
        m1 = Monitor.get_instance()
        m2 = Monitor.get_instance()
        
        assert m1 is m2
    
    def test_record_api_call(self):
        """Should record API calls without error."""
        from core.monitoring import Monitor
        
        monitor = Monitor()
        monitor.record_api_call(
            api_name="test_api",
            success=True,
            latency_ms=100.0
        )
        
        assert "test_api" in monitor.api_health
        assert monitor.api_health["test_api"].is_healthy
    
    def test_record_api_failure(self):
        """Should track API failures."""
        from core.monitoring import Monitor, AlertConfig
        
        config = AlertConfig(api_failure_threshold=3)
        monitor = Monitor(config=config)
        
        # Record failures
        for i in range(3):
            monitor.record_api_call(
                api_name="failing_api",
                success=False,
                latency_ms=5000.0,
                error="Connection timeout"
            )
        
        assert monitor.api_health["failing_api"].consecutive_failures >= 3
        assert not monitor.api_health["failing_api"].is_healthy
    
    def test_model_drift_detection(self):
        """Should detect model drift."""
        from core.monitoring import Monitor
        
        monitor = Monitor()
        
        # Set baseline
        baseline = np.random.normal(0.5, 0.2, 100)
        monitor.update_baseline(baseline)
        
        # Check stable predictions (no drift)
        stable = np.random.normal(0.5, 0.2, 100)
        has_drift = monitor.check_model_drift(stable)
        assert not has_drift
        
        # Check drifted predictions
        drifted = np.random.normal(0.8, 0.2, 100)  # Mean shifted significantly
        has_drift = monitor.check_model_drift(drifted)
        assert has_drift
    
    def test_health_report(self):
        """Should generate health report."""
        from core.monitoring import Monitor
        
        monitor = Monitor()
        report = monitor.get_health_report()
        
        assert "timestamp" in report
        assert "apis" in report
        assert "model_drift" in report
        assert "overall_healthy" in report


# ============================================================================
# Backtester Tests
# ============================================================================
class TestBacktester:
    """Tests for core.backtester"""
    
    def test_backtest_config_defaults(self):
        """BacktestConfig should have reasonable defaults."""
        from core.backtester import BacktestConfig
        
        config = BacktestConfig()
        
        assert config.max_positions == 20
        assert 0 < config.position_size < 1
        assert 0 < config.stop_loss_pct < 1
    
    def test_metrics_to_dict(self):
        """BacktestMetrics should convert to dict."""
        from core.backtester import BacktestMetrics
        
        metrics = BacktestMetrics(
            total_return=0.15,
            annualized_return=0.20,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=0.10,
            win_rate=0.55,
            profit_factor=1.8,
            avg_trade_return=0.02,
            avg_hold_days=15,
            num_trades=50,
            total_costs=0.01
        )
        
        d = metrics.to_dict()
        
        assert d["total_return"] == 0.15
        assert d["sharpe_ratio"] == 1.5
        assert d["win_rate"] == 0.55
    
    def test_compute_metrics_empty(self):
        """Should handle empty trades gracefully."""
        from core.backtester import Backtester
        
        bt = Backtester()
        metrics = bt._compute_metrics(pd.DataFrame())
        
        assert metrics.num_trades == 0
        assert metrics.total_return == 0


# ============================================================================
# Integration Tests
# ============================================================================
class TestV4Integration:
    """Integration tests for V4 ML system."""
    
    def test_feature_builder_matches_registry(self):
        """Feature builder should produce all registry features."""
        from core.feature_registry_v4 import get_feature_names_v4
        from core.ml_feature_builder_v4 import get_feature_defaults_v4
        
        registry_features = set(get_feature_names_v4())
        builder_defaults = set(get_feature_defaults_v4().keys())
        
        # All registry features should have builder defaults
        missing = registry_features - builder_defaults
        assert not missing, f"Builder missing features: {missing}"
    
    def test_alert_decorator(self):
        """Alert decorator should work without errors."""
        from core.monitoring import alert
        
        @alert(on_error=True, on_slow=1.0)
        def test_function():
            return 42
        
        result = test_function()
        assert result == 42
    
    def test_api_call_tracker(self):
        """APICallTracker context manager should work."""
        from core.monitoring import APICallTracker
        
        with APICallTracker("test_api") as tracker:
            tracker.set_rate_limit(100)
        
        # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
