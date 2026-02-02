# Changelog

All notable changes to Stock Scout are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v3.0.0] - 2026-02-02

### Added
- **Feature Registry** (`core/feature_registry.py`) - Single source of truth for ML features
  - 34 feature definitions with specs, defaults, and valid ranges
  - Functions: `get_feature_names()`, `get_feature_defaults()`, `validate_features()`, `clip_features_to_range()`
  - 29 unit tests for feature registry
  
- **API Key Manager** (`core/api_keys.py`) - Centralized, secure API key management
  - Functions: `get_api_key()`, `validate_keys()`, `log_api_key_status()`
  - Supports: Polygon, Finnhub, Alpha Vantage, Tiingo, FMP, OpenAI, Marketstack, EODHD
  - Keys masked in logs for security
  
- **UnifiedScorer** (`core/scoring/unified_scorer.py`) - Consolidated scoring logic
  - `ScoringResult` dataclass with complete breakdown
  - `UnifiedScorer` class with configurable weights
  - Convenience functions: `score_ticker()`, `score_dataframe()`
  - 21 unit tests for unified scorer
  
- **Market Calendar** (`scripts/market_calendar.py`) - US market holiday awareness
  - Functions: `is_market_open()`, `get_market_status()`, `get_next_market_day()`
  - 2026 NYSE holiday schedule
  - Early close day handling
  
- **Model Validation** - Automatic feature count verification on model load
  - `_find_model_path()` with intelligent fallback order
  - `validate_model_features()` for pre-load checks
  - `get_model_info()` and `get_expected_features()` introspection

### Changed
- **ML Model Architecture** - Now uses 34 features (up from 5)
  - Categories: Technical, Volatility, Volume Basic, Market Regime, Sector Relative, Volume Advanced, Price Action
  - Ensemble: HistGradientBoosting (45%) + RandomForest (35%) + LogisticRegression (20%)
  
- **Training Pipeline** - Improved validation
  - TimeSeriesSplit with 5 folds (was single train/test split)
  - Isotonic Regression calibration (was CalibratedClassifierCV hack)
  - Lazy API key loading in training script
  
- **GitHub Actions** - Aligned with NYSE trading hours
  - 4 scans daily: Pre-market (8:30 AM ET), Early session (10:00 AM ET), Late session (2:00 PM ET), End of day (3:30 PM ET)
  - US market holiday detection to skip unnecessary runs
  - Python 3.11 (was 3.10)
  
- **Model Path** - Now prioritizes `models/model_20d_v3.pkl`
  - Fallback order: v3 → v2 → legacy XGBoost → sklearn bundle
  - Dynamic feature extraction from model bundle

### Fixed
- **Model Path Mismatch** - Training saved to different path than inference loaded from
- **Exposed API Key** - Hardcoded Polygon key in training script
- **Inconsistent Scoring** - Different entry points produced different scores
- **Feature Mismatch** - Training and inference used different feature lists

### Security
- Removed all hardcoded API keys from:
  - `scripts/train_rolling_ml_20d.py`
  - `benchmark_apis.py`
  - `docs/archive/STREAMLIT_CLOUD_SETUP.md`
  - `COMPREHENSIVE_SYSTEM_REVIEW_2026.md`
- Added `.env.example` template with all supported keys
- Updated `.gitignore` for sensitive files

### Tests
- Added `tests/test_unified_scorer.py` (21 tests)
- Added `tests/test_feature_registry.py` (29 tests)
- Updated `tests/test_ml_pipeline.py` for dynamic features (40 tests)
- **Total: 90+ tests passing**

---

## [v2.1.0] - 2025-11-13

### Added
- Fundamental scoring with detailed breakdown (`core/scoring/fundamental.py`)
- Color-coded UI labels (Quality, Growth, Valuation, Leverage)
- Type hints for improved code quality
- Risk management module (`core/risk.py`)

### Changed
- Refactored from monolithic `stock_scout.py` to modular architecture
- Centralized configuration in `core/config.py`
- Structured logging via `core/logging_config.py`

---

## [v2.0.0] - 2025-10-XX

### Added
- Initial modular architecture (`core/` package)
- Multi-provider data sources with fallback
- Type-safe data models (`core/models.py`)
- XGBoost ML model integration

---

## [v1.x.x] - Previous versions

See git history for older changes.
