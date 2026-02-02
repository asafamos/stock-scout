# 🔧 פקודות לתיקון Stock Scout - Claude Agent

העבר את הפקודות האלה לקופיילוט ב-VS Code בסדר הזה.
כל פקודה היא בלוק נפרד - העתק והדבק כמו שהיא.

---

## 🚨 פקודה 1: הסרת API Keys חשופים (אבטחה - ראשון!)

```
Scan and fix all exposed API keys in the codebase. This is a security issue.

STEP 1 - SCAN:
Search all .py files for patterns that look like hardcoded API keys:
- Strings after "API_KEY" that contain actual values (not os.environ)
- Any string that looks like an API key (alphanumeric, 20+ chars)
- Focus on: scripts/train_rolling_ml_20d.py, core/data_sources.py, any file with "polygon", "finnhub", "alpha"

STEP 2 - CREATE API KEY MANAGER:
Create core/api_keys.py:

```python
"""
Centralized API Key Management.
All API keys must be loaded from environment variables.
NO hardcoded keys anywhere in the codebase.
"""
import os
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class APIKeyStatus:
    name: str
    available: bool
    masked_value: str  # Show only last 4 chars

class APIKeyManager:
    """Centralized API key management with validation."""

    KNOWN_KEYS = [
        "POLYGON_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "FINNHUB_API_KEY",
        "TIINGO_API_KEY",
        "FMP_API_KEY",
        "MARKETSTACK_API_KEY",
        "NASDAQ_API_KEY",
        "EODHD_API_KEY",
        "SIMFIN_API_KEY",
        "OPENAI_API_KEY",
    ]

    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}

    def get_key(self, name: str, required: bool = False) -> Optional[str]:
        """Get API key from environment. Raises if required but missing."""
        if name not in self._cache:
            self._cache[name] = os.environ.get(name)

        value = self._cache[name]

        if required and not value:
            raise EnvironmentError(
                f"Required API key '{name}' is not set. "
                f"Please set it in your environment or .env file."
            )

        return value

    def validate_required_keys(self, required: List[str]) -> Dict[str, bool]:
        """Check which required keys are available."""
        return {key: bool(self.get_key(key)) for key in required}

    def get_status(self) -> List[APIKeyStatus]:
        """Get status of all known API keys (for logging)."""
        statuses = []
        for key in self.KNOWN_KEYS:
            value = self.get_key(key)
            masked = f"***{value[-4:]}" if value and len(value) > 4 else "NOT SET"
            statuses.append(APIKeyStatus(key, bool(value), masked))
        return statuses

    def log_status(self):
        """Log API key availability (masked values)."""
        logger.info("API Key Status:")
        for status in self.get_status():
            icon = "✓" if status.available else "✗"
            logger.info(f"  {icon} {status.name}: {status.masked_value}")


# Global instance
_manager = APIKeyManager()

def get_api_key(name: str, required: bool = False) -> Optional[str]:
    """Get API key from environment."""
    return _manager.get_key(name, required)

def validate_keys(required: List[str]) -> Dict[str, bool]:
    """Validate required API keys."""
    return _manager.validate_required_keys(required)

def log_api_key_status():
    """Log status of all API keys."""
    _manager.log_status()
```

STEP 3 - FIX TRAIN SCRIPT:
In scripts/train_rolling_ml_20d.py:
- Remove any hardcoded API key values
- Change to:
```python
from core.api_keys import get_api_key

POLYGON_KEY = get_api_key("POLYGON_API_KEY", required=True)
```

STEP 4 - UPDATE .env.example:
Create or update .env.example with all required keys (NO VALUES):
```
# Stock Scout API Keys
# Copy this file to .env and fill in your keys

# Required for ML training
POLYGON_API_KEY=

# Data providers (at least one required)
ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
TIINGO_API_KEY=
FMP_API_KEY=
MARKETSTACK_API_KEY=
NASDAQ_API_KEY=
EODHD_API_KEY=
SIMFIN_API_KEY=

# Optional - for AI features
OPENAI_API_KEY=
```

STEP 5 - UPDATE .gitignore:
Add these lines if not present:
```
.env
.env.*
!.env.example
secrets.*
credentials.*
*_key.txt
*_token.txt
```

After changes, run:
git diff --stat
```

---

## 🚨 פקודה 2: תיקון נתיב המודל (קריטי!)

```
Fix the critical ML model path mismatch in core/ml_integration.py.

THE PROBLEM:
- Training saves model to: models/model_20d_v3.pkl
- Inference loads from: model_xgboost_5d.pkl (WRONG!)
- This means the new 34-feature model is never used

STEP 1 - UPDATE MODEL PATH:
In core/ml_integration.py, replace the _MODEL_PATH definition:

OLD:
```python
_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "model_xgboost_5d.pkl")
```

NEW:
```python
# Model path priority (first found is used):
# 1. models/model_20d_v3.pkl (latest trained model)
# 2. ml/bundles/latest/model.joblib (production bundle)
# 3. model_xgboost_5d.pkl (legacy fallback)

def _find_model_path() -> Optional[str]:
    """Find the best available model file."""
    project_root = Path(__file__).resolve().parents[1]

    candidates = [
        project_root / "models" / "model_20d_v3.pkl",
        project_root / "ml" / "bundles" / "latest" / "model.joblib",
        project_root / "model_xgboost_5d.pkl",  # Legacy
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return None

_MODEL_PATH = _find_model_path()
```

STEP 2 - ADD MODEL VALIDATION:
Add this function after load_ml_model():

```python
def validate_model_features(model, expected_features: List[str]) -> bool:
    """Validate that loaded model matches expected features."""
    try:
        # Create dummy input with expected features
        dummy_input = pd.DataFrame([{f: 0.0 for f in expected_features}])

        # Try to predict - will fail if features don't match
        model.predict_proba(dummy_input)
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False
```

STEP 3 - UPDATE load_ml_model():
Modify the function to validate after loading:

```python
def load_ml_model(model_path: Optional[str] = None) -> bool:
    global _ML_MODEL, _MODEL_LOADED, _MODEL_PATH

    if model_path:
        _MODEL_PATH = model_path
    elif _MODEL_PATH is None:
        _MODEL_PATH = _find_model_path()

    if _MODEL_LOADED:
        return _ML_MODEL is not None

    if _MODEL_PATH is None:
        logger.warning("No ML model file found in any expected location")
        _MODEL_LOADED = True
        return False

    try:
        import joblib

        # Try joblib first (newer format), fall back to pickle
        try:
            _ML_MODEL = joblib.load(_MODEL_PATH)
        except:
            with open(_MODEL_PATH, "rb") as f:
                _ML_MODEL = pickle.load(f)

        # Handle bundled models (dict with 'model' key)
        if isinstance(_ML_MODEL, dict) and 'model' in _ML_MODEL:
            logger.info(f"Loaded model bundle with keys: {list(_ML_MODEL.keys())}")
            _ML_MODEL = _ML_MODEL['model']

        # Validate the model
        if not validate_model_features(_ML_MODEL, EXPECTED_FEATURES):
            logger.error("Model features don't match expected features!")
            logger.error(f"Expected {len(EXPECTED_FEATURES)} features")
            _ML_MODEL = None
            _MODEL_LOADED = True
            return False

        logger.info(f"✓ ML model loaded from {_MODEL_PATH}")
        logger.info(f"  Model type: {type(_ML_MODEL).__name__}")
        logger.info(f"  Features: {len(EXPECTED_FEATURES)}")
        _MODEL_LOADED = True
        return True

    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        _ML_MODEL = None
        _MODEL_LOADED = True
        return False
```

STEP 4 - ADD get_model_info() FUNCTION:

```python
def get_model_info() -> Dict[str, any]:
    """Get information about the currently loaded model."""
    if not _MODEL_LOADED:
        load_ml_model()

    return {
        "loaded": _ML_MODEL is not None,
        "path": _MODEL_PATH,
        "feature_count": len(EXPECTED_FEATURES) if _ML_MODEL else 0,
        "model_type": type(_ML_MODEL).__name__ if _ML_MODEL else None,
        "features": EXPECTED_FEATURES if _ML_MODEL else [],
    }
```

STEP 5 - ADD IMPORT:
Make sure joblib is imported at the top:
```python
import joblib
```

After changes, verify with:
python -c "from core.ml_integration import load_ml_model, get_model_info; load_ml_model(); print(get_model_info())"
```

---

## 🚨 פקודה 3: יצירת Feature Registry

```
Create a single source of truth for ML features to prevent train/inference mismatch.

STEP 1 - CREATE FEATURE REGISTRY:
Create new file core/feature_registry.py:

```python
"""
Feature Registry - Single Source of Truth for ML Features.

This module defines ALL features used in ML models. Both training and inference
MUST import from here to ensure consistency.

DO NOT hardcode feature lists anywhere else in the codebase!
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    description: str
    default_value: float
    valid_range: Tuple[float, float]  # (min, max)
    category: str  # e.g., "technical", "volume", "market_regime"


# =============================================================================
# FEATURE DEFINITIONS V3 (34 features)
# =============================================================================
FEATURE_SPECS_V3: List[FeatureSpec] = [
    # --- Technical Base (5) ---
    FeatureSpec("RSI", "Relative Strength Index (0-100)", 50.0, (0, 100), "technical"),
    FeatureSpec("ATR_Pct", "Average True Range as % of price", 0.02, (0.001, 0.5), "technical"),
    FeatureSpec("Return_20d", "20-day price return", 0.0, (-1.0, 2.0), "technical"),
    FeatureSpec("Return_10d", "10-day price return", 0.0, (-0.5, 1.0), "technical"),
    FeatureSpec("Return_5d", "5-day price return", 0.0, (-0.3, 0.5), "technical"),

    # --- Volatility Patterns (4) ---
    FeatureSpec("VCP_Ratio", "Volatility Contraction: ATR(10)/ATR(30)", 1.0, (0.1, 5.0), "volatility"),
    FeatureSpec("Tightness_Ratio", "Range contraction: range_5d/range_20d", 1.0, (0.05, 2.0), "volatility"),
    FeatureSpec("Dist_From_52w_High", "(Close/52w_High)-1", -0.1, (-0.8, 0.1), "volatility"),
    FeatureSpec("MA_Alignment", "1 if Close>MA20>MA50>MA200", 0.0, (0, 1), "volatility"),

    # --- Volume Basic (3) ---
    FeatureSpec("Volume_Surge", "vol_5d_avg / vol_20d_avg", 1.0, (0.1, 10.0), "volume"),
    FeatureSpec("Up_Down_Volume_Ratio", "up-day vol / down-day vol", 1.0, (0.1, 10.0), "volume"),
    FeatureSpec("Momentum_Consistency", "% positive days (last 20)", 0.5, (0, 1), "momentum"),

    # --- Relative Strength (1) ---
    FeatureSpec("RS_vs_SPY_20d", "stock_ret_20d - spy_ret_20d", 0.0, (-1.0, 1.0), "momentum"),

    # --- Market Regime (4) ---
    FeatureSpec("Market_Regime", "Bull(1), Sideways(0), Bear(-1)", 0.0, (-1, 1), "market"),
    FeatureSpec("Market_Volatility", "SPY 20d volatility (annualized)", 0.15, (0.05, 0.8), "market"),
    FeatureSpec("Market_Trend", "SPY 50d return", 0.0, (-0.5, 0.5), "market"),
    FeatureSpec("High_Volatility", "1 if vol > 75th percentile", 0.0, (0, 1), "market"),

    # --- Sector Relative (3) ---
    FeatureSpec("Sector_RS", "stock_ret_20d - sector_ret_20d", 0.0, (-1.0, 1.0), "sector"),
    FeatureSpec("Sector_Momentum", "sector_etf_ret_20d", 0.0, (-0.5, 0.5), "sector"),
    FeatureSpec("Sector_Rank", "1 if stock beats sector in 5d", 0.5, (0, 1), "sector"),

    # --- Volume Advanced (5) ---
    FeatureSpec("Volume_Ratio_20d", "current vol / 20d avg", 1.0, (0.1, 20.0), "volume"),
    FeatureSpec("Volume_Trend", "volume slope (accumulation)", 0.0, (-1.0, 1.0), "volume"),
    FeatureSpec("Up_Volume_Ratio", "up-day vol / total vol", 0.5, (0, 1), "volume"),
    FeatureSpec("Volume_Price_Confirm", "price+vol up confirmation", 0.5, (0, 1), "volume"),
    FeatureSpec("Relative_Volume_Rank", "vol percentile vs 60d", 0.5, (0, 1), "volume"),

    # --- Price Action (9) ---
    FeatureSpec("Distance_From_52w_Low", "(close-52w_low)/52w_low", 0.5, (-0.5, 5.0), "price_action"),
    FeatureSpec("Consolidation_Tightness", "(20d_high-low)/avg", 0.1, (0.01, 0.5), "price_action"),
    FeatureSpec("Days_Since_52w_High", "normalized 0-1", 0.5, (0, 1), "price_action"),
    FeatureSpec("Price_vs_SMA50", "(close-sma50)/sma50", 0.0, (-0.5, 0.5), "price_action"),
    FeatureSpec("Price_vs_SMA200", "(close-sma200)/sma200", 0.0, (-0.5, 0.5), "price_action"),
    FeatureSpec("SMA50_vs_SMA200", "(sma50-sma200)/sma200", 0.0, (-0.3, 0.3), "price_action"),
    FeatureSpec("MA_Slope_20d", "slope of 20d MA", 0.0, (-0.3, 0.3), "price_action"),
    FeatureSpec("Distance_To_Resistance", "(20d_high-close)/close", 0.05, (0, 0.5), "price_action"),
    FeatureSpec("Support_Strength", "fraction days near support", 0.2, (0, 1), "price_action"),
]

# =============================================================================
# PUBLIC API
# =============================================================================

def get_feature_names(version: str = "v3") -> List[str]:
    """Get ordered list of feature names."""
    if version == "v3":
        return [f.name for f in FEATURE_SPECS_V3]
    raise ValueError(f"Unknown feature version: {version}")


def get_feature_specs(version: str = "v3") -> List[FeatureSpec]:
    """Get full feature specifications."""
    if version == "v3":
        return FEATURE_SPECS_V3
    raise ValueError(f"Unknown feature version: {version}")


def get_feature_defaults(version: str = "v3") -> Dict[str, float]:
    """Get dict of feature name -> default value."""
    return {f.name: f.default_value for f in get_feature_specs(version)}


def get_feature_ranges(version: str = "v3") -> Dict[str, Tuple[float, float]]:
    """Get dict of feature name -> (min, max) valid range."""
    return {f.name: f.valid_range for f in get_feature_specs(version)}


def validate_features(df, version: str = "v3") -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has all required features.

    Returns:
        (is_valid, missing_features)
    """
    required = set(get_feature_names(version))
    present = set(df.columns)
    missing = required - present
    return len(missing) == 0, list(missing)


def clip_features_to_range(df, version: str = "v3"):
    """Clip feature values to valid ranges (in-place)."""
    ranges = get_feature_ranges(version)
    for feat, (lo, hi) in ranges.items():
        if feat in df.columns:
            df[feat] = np.clip(df[feat], lo, hi)
    return df


# Feature count for assertions
FEATURE_COUNT_V3 = len(FEATURE_SPECS_V3)
assert FEATURE_COUNT_V3 == 34, f"Expected 34 features, got {FEATURE_COUNT_V3}"
```

STEP 2 - UPDATE ml_integration.py:
Replace the hardcoded EXPECTED_FEATURES with import:

```python
# At the top of the file, add:
from core.feature_registry import get_feature_names, get_feature_defaults, validate_features

# Replace the EXPECTED_FEATURES list with:
EXPECTED_FEATURES: List[str] = get_feature_names("v3")
```

STEP 3 - UPDATE train_rolling_ml_20d.py:
In scripts/train_rolling_ml_20d.py, import and use the registry:

```python
# At the top:
from core.feature_registry import get_feature_names, FEATURE_COUNT_V3

# In train_and_save_bundle(), replace the features list with:
features = get_feature_names("v3")
assert len(features) == FEATURE_COUNT_V3, "Feature count mismatch!"
```

STEP 4 - CREATE TEST:
Create tests/test_feature_registry.py:

```python
"""Tests for feature registry consistency."""
import pytest
from core.feature_registry import (
    get_feature_names, get_feature_specs, get_feature_defaults,
    validate_features, FEATURE_COUNT_V3
)
import pandas as pd


def test_feature_count():
    """Verify expected feature count."""
    assert len(get_feature_names("v3")) == 34
    assert FEATURE_COUNT_V3 == 34


def test_all_features_have_specs():
    """All features must have complete specifications."""
    specs = get_feature_specs("v3")
    for spec in specs:
        assert spec.name, "Feature must have name"
        assert spec.description, "Feature must have description"
        assert spec.valid_range[0] <= spec.valid_range[1], "Invalid range"


def test_defaults_within_range():
    """Default values must be within valid range."""
    specs = get_feature_specs("v3")
    for spec in specs:
        lo, hi = spec.valid_range
        assert lo <= spec.default_value <= hi, \
            f"{spec.name} default {spec.default_value} outside range [{lo}, {hi}]"


def test_validate_features():
    """Test feature validation."""
    features = get_feature_names("v3")

    # Valid DataFrame
    df_valid = pd.DataFrame([{f: 0.0 for f in features}])
    is_valid, missing = validate_features(df_valid, "v3")
    assert is_valid
    assert missing == []

    # Missing features
    df_missing = pd.DataFrame([{"RSI": 50.0}])
    is_valid, missing = validate_features(df_missing, "v3")
    assert not is_valid
    assert len(missing) == 33


def test_feature_names_match_training():
    """Verify inference features match training features."""
    # This ensures train and inference are in sync
    from core.ml_integration import EXPECTED_FEATURES
    registry_features = get_feature_names("v3")

    assert EXPECTED_FEATURES == registry_features, \
        "ml_integration.py EXPECTED_FEATURES must match feature_registry!"
```

After changes, run:
pytest tests/test_feature_registry.py -v
```

---

## 🔧 פקודה 4: איחוד Scoring Logic

```
Consolidate all scoring logic into UnifiedScorer to eliminate inconsistencies.

STEP 1 - AUDIT EXISTING SCORING:
First, list all files with scoring functions:
- core/scoring_engine.py
- core/unified_logic.py
- core/ml_integration.py
- core/scoring/__init__.py

Identify which functions compute scores and their signatures.

STEP 2 - CREATE UNIFIED SCORER:
Create core/scoring/unified_scorer.py:

```python
"""
UnifiedScorer - Single Source of Truth for Stock Scoring.

All scoring in the application should go through this class.
This ensures consistent scores regardless of entry point.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Complete scoring result with breakdown."""
    # Final scores
    final_conviction: float  # 0-100, the main score

    # Component scores
    technical_score: float  # 0-100
    fundamental_score: float  # 0-100

    # ML adjustment
    ml_probability: Optional[float]  # 0-1 or None
    ml_boost: float  # typically -10 to +10
    ml_status: str  # "enabled", "disabled", "error"

    # Breakdown for transparency
    breakdown: Dict[str, Any] = field(default_factory=dict)

    # Quality indicators
    reliability_pct: float = 0.0
    data_quality: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame/JSON."""
        return {
            "final_conviction": self.final_conviction,
            "technical_score": self.technical_score,
            "fundamental_score": self.fundamental_score,
            "ml_probability": self.ml_probability,
            "ml_boost": self.ml_boost,
            "ml_status": self.ml_status,
            "reliability_pct": self.reliability_pct,
            "data_quality": self.data_quality,
            **self.breakdown,
        }


class UnifiedScorer:
    """
    Single entry point for all scoring operations.

    Usage:
        scorer = UnifiedScorer(config)
        result = scorer.score(ticker_data, indicators, fundamentals)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ml_enabled = self.config.get("enable_ml", True)
        self.ml_max_boost = self.config.get("ml_max_boost_pct", 10.0)

        # Weight configuration
        self.technical_weight = self.config.get("technical_weight", 0.60)
        self.fundamental_weight = self.config.get("fundamental_weight", 0.40)

    def score(
        self,
        ticker_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        fundamental_data: Dict[str, Any],
    ) -> ScoringResult:
        """
        Compute complete scoring for a ticker.

        Args:
            ticker_data: Raw ticker data (price, volume, etc.)
            technical_indicators: Computed technical indicators
            fundamental_data: Fundamental metrics (PE, ROE, etc.)

        Returns:
            ScoringResult with all components
        """
        # 1. Compute technical score
        tech_score = self._compute_technical_score(technical_indicators)

        # 2. Compute fundamental score
        fund_score = self._compute_fundamental_score(fundamental_data)

        # 3. Compute base conviction (weighted average)
        base_conviction = (
            tech_score * self.technical_weight +
            fund_score * self.fundamental_weight
        )

        # 4. Apply ML boost (if enabled)
        ml_prob = None
        ml_boost = 0.0
        ml_status = "disabled"

        if self.ml_enabled:
            ml_prob, ml_boost, ml_status = self._apply_ml_boost(
                base_conviction, ticker_data, technical_indicators, fundamental_data
            )

        # 5. Compute final conviction
        final_conviction = np.clip(base_conviction + ml_boost, 0, 100)

        # 6. Compute reliability
        reliability = self._compute_reliability(ticker_data)

        return ScoringResult(
            final_conviction=float(final_conviction),
            technical_score=float(tech_score),
            fundamental_score=float(fund_score),
            ml_probability=ml_prob,
            ml_boost=float(ml_boost),
            ml_status=ml_status,
            reliability_pct=float(reliability),
            data_quality=self._assess_data_quality(ticker_data),
            breakdown={
                "base_conviction": float(base_conviction),
                "technical_weight": self.technical_weight,
                "fundamental_weight": self.fundamental_weight,
            }
        )

    def _compute_technical_score(self, indicators: Dict) -> float:
        """Compute technical score from indicators."""
        from core.scoring import compute_tech_score_20d_v2

        try:
            return compute_tech_score_20d_v2(indicators)
        except Exception as e:
            logger.warning(f"Technical scoring failed: {e}")
            return 50.0  # Neutral default

    def _compute_fundamental_score(self, fundamentals: Dict) -> float:
        """Compute fundamental score."""
        from core.scoring import compute_fundamental_score_with_breakdown

        try:
            result = compute_fundamental_score_with_breakdown(fundamentals)
            return result.get("total_score", 50.0)
        except Exception as e:
            logger.warning(f"Fundamental scoring failed: {e}")
            return 50.0  # Neutral default

    def _apply_ml_boost(
        self, base_conviction: float, ticker_data: Dict,
        indicators: Dict, fundamentals: Dict
    ) -> tuple:
        """Apply ML prediction boost."""
        from core.ml_integration import integrate_ml_with_conviction

        try:
            final, ml_info = integrate_ml_with_conviction(
                base_conviction, ticker_data, indicators, fundamentals,
                enable_ml=True
            )
            return (
                ml_info.get("ml_probability"),
                ml_info.get("ml_boost", 0.0),
                ml_info.get("ml_status", "unknown")
            )
        except Exception as e:
            logger.warning(f"ML boost failed: {e}")
            return None, 0.0, f"error: {e}"

    def _compute_reliability(self, ticker_data: Dict) -> float:
        """Compute data reliability percentage."""
        from core.scoring import calculate_reliability_v2

        try:
            return calculate_reliability_v2(ticker_data)
        except:
            return 50.0

    def _assess_data_quality(self, ticker_data: Dict) -> str:
        """Assess overall data quality."""
        sources = ticker_data.get("data_sources", [])
        if len(sources) >= 3:
            return "high"
        elif len(sources) >= 1:
            return "medium"
        return "low"


# Convenience function for simple usage
def score_ticker(
    ticker_data: Dict,
    technical_indicators: Dict,
    fundamental_data: Dict,
    config: Optional[Dict] = None
) -> ScoringResult:
    """Convenience function to score a single ticker."""
    scorer = UnifiedScorer(config)
    return scorer.score(ticker_data, technical_indicators, fundamental_data)
```

STEP 3 - ADD DEPRECATION TO OLD FUNCTIONS:
In core/unified_logic.py, add deprecation warnings:

```python
import warnings

def compute_overall_score_20d(*args, **kwargs):
    warnings.warn(
        "compute_overall_score_20d is deprecated. Use UnifiedScorer instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Keep working for backward compatibility
    from core.scoring.unified_scorer import score_ticker
    # ... adapt args to new interface
```

STEP 4 - UPDATE stock_scout.py:
Add import and use UnifiedScorer where appropriate:

```python
from core.scoring.unified_scorer import UnifiedScorer, score_ticker
```

After changes:
python -c "from core.scoring.unified_scorer import UnifiedScorer; print('UnifiedScorer imported OK')"
```

---

## 🔧 פקודה 5: תזמון GitHub Actions

```
Fix GitHub Actions to run during NYSE trading hours only.

STEP 1 - UPDATE WORKFLOW:
Edit .github/workflows/auto_scan.yml:

```yaml
name: Auto Stock Scan

on:
  schedule:
    # Pre-market scan (1 hour before NYSE open)
    - cron: '30 13 * * 1-5'  # 13:30 UTC = 8:30 AM ET

    # Early session scan (30 min after open)
    - cron: '0 15 * * 1-5'   # 15:00 UTC = 10:00 AM ET

    # Late session scan (1 hour before close)
    - cron: '0 19 * * 1-5'   # 19:00 UTC = 2:00 PM ET

    # End of day scan (30 min after close)
    - cron: '30 20 * * 1-5'  # 20:30 UTC = 3:30 PM ET

  workflow_dispatch:  # Allow manual trigger

env:
  POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}
  FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
  # Add other API keys as needed

jobs:
  scan:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Check if market is open
        id: market_check
        run: |
          # Simple holiday check (expand this list as needed)
          TODAY=$(date +%Y-%m-%d)
          HOLIDAYS="2026-01-01 2026-01-20 2026-02-17 2026-04-03 2026-05-25 2026-07-03 2026-09-07 2026-11-26 2026-12-25"

          if echo "$HOLIDAYS" | grep -q "$TODAY"; then
            echo "market_closed=true" >> $GITHUB_OUTPUT
            echo "Today is a US market holiday - skipping scan"
          else
            echo "market_closed=false" >> $GITHUB_OUTPUT
          fi

      - name: Setup Python
        if: steps.market_check.outputs.market_closed != 'true'
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        if: steps.market_check.outputs.market_closed != 'true'
        run: pip install -r requirements.txt

      - name: Run scan
        if: steps.market_check.outputs.market_closed != 'true'
        run: |
          python scripts/run_full_scan.py
        continue-on-error: true

      - name: Send alert on failure
        if: failure() && env.ALERT_WEBHOOK_URL != ''
        env:
          ALERT_WEBHOOK_URL: ${{ secrets.ALERT_WEBHOOK_URL }}
        run: |
          curl -X POST "$ALERT_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d '{
              "workflow": "auto_scan",
              "status": "failed",
              "run_url": "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
            }'

      - name: Commit results
        if: steps.market_check.outputs.market_closed != 'true'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add reports/ data/
          git diff --staged --quiet || git commit -m "Auto scan results $(date +%Y-%m-%d)"
          git push
```

STEP 2 - CREATE MARKET CALENDAR HELPER:
Create scripts/market_calendar.py:

```python
"""US Market Calendar Helper."""
from datetime import date, datetime
from typing import Set

# US Market Holidays 2026 (update annually)
US_MARKET_HOLIDAYS_2026: Set[date] = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
}

def is_market_open(check_date: date = None) -> bool:
    """Check if US stock market is open on given date."""
    if check_date is None:
        check_date = date.today()

    # Weekend check
    if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Holiday check
    if check_date in US_MARKET_HOLIDAYS_2026:
        return False

    return True

def get_market_status() -> str:
    """Get current market status string."""
    now = datetime.utcnow()
    today = now.date()

    if not is_market_open(today):
        return "CLOSED (holiday/weekend)"

    hour = now.hour
    if hour < 14:  # Before 14:00 UTC (9:00 ET)
        return "PRE-MARKET"
    elif hour < 21:  # 14:00-21:00 UTC (9:00-4:00 ET)
        return "OPEN"
    else:
        return "AFTER-HOURS"

if __name__ == "__main__":
    print(f"Market status: {get_market_status()}")
    print(f"Is open today: {is_market_open()}")
```

After changes, verify:
cat .github/workflows/auto_scan.yml
```

---

## 🔧 פקודה 6: עדכון דוקומנטציה

```
Update all documentation to reflect the current state of the codebase.

STEP 1 - UPDATE COMPREHENSIVE_SYSTEM_REVIEW_2026.md:
Add a "RESOLVED ISSUES" section at the top:

```markdown
## ✅ בעיות שתוקנו

### תאריך: [TODAY'S DATE]

| בעיה | סטטוס | פתרון |
|------|-------|-------|
| מודל ML עם 5 features בלבד | ✅ תוקן | המודל v3 משתמש ב-34 features |
| חסר Cross-Validation | ✅ תוקן | TimeSeriesSplit עם 5 folds |
| API Key חשוף | ✅ תוקן | נוסף APIKeyManager |
| Calibration Hack | ✅ תוקן | Isotonic Regression בזמן training |
| נתיב מודל שגוי | ✅ תוקן | _find_model_path() עם fallbacks |
| Feature mismatch | ✅ תוקן | Feature Registry כ-Single Source of Truth |
```

STEP 2 - UPDATE README.md ML Section:

Replace the ML Model Performance section with:

```markdown
## 📊 ML Model Performance

### Model Architecture (v3)
- **Ensemble**: HistGradientBoosting (45%) + RandomForest (35%) + LogisticRegression (20%)
- **Features**: 34 engineered features across 7 categories
- **Calibration**: Isotonic regression for reliable probabilities
- **Validation**: Time-Series Cross-Validation (5 folds)

### Feature Categories (34 total)
| Category | Count | Examples |
|----------|-------|----------|
| Technical | 5 | RSI, ATR_Pct, Returns |
| Volatility | 4 | VCP_Ratio, Tightness, MA_Alignment |
| Volume Basic | 3 | Volume_Surge, Up_Down_Ratio |
| Market Regime | 4 | Market_Regime, Volatility, Trend |
| Sector Relative | 3 | Sector_RS, Sector_Momentum |
| Volume Advanced | 5 | Volume_Trend, Accumulation signals |
| Price Action | 9 | 52w positioning, Support/Resistance |

### Validation Metrics
- **Out-of-Sample AUC**: Check models/model_20d_v3.pkl.metadata.json
- **Precision@20**: Top 20 predictions accuracy
- **Lift**: Improvement over random baseline
```

STEP 3 - CREATE CHANGELOG.md:

```markdown
# Changelog

All notable changes to Stock Scout are documented here.

## [v3.0.0] - 2026-02-XX

### Added
- Feature Registry (core/feature_registry.py) - Single source of truth for ML features
- API Key Manager (core/api_keys.py) - Centralized, secure API key management
- UnifiedScorer (core/scoring/unified_scorer.py) - Consolidated scoring logic
- Model validation on load with feature count verification
- Market calendar for holiday-aware scheduling

### Changed
- ML model now uses 34 features (up from 5)
- Training uses TimeSeriesSplit with 5 folds
- Model output calibrated with Isotonic Regression
- GitHub Actions timing aligned with NYSE hours
- Model path now prioritizes models/model_20d_v3.pkl

### Fixed
- Model path mismatch between training and inference
- Exposed API key in train script
- Inconsistent scoring across different entry points

### Security
- Removed all hardcoded API keys
- Added .env.example template
- Updated .gitignore for sensitive files

## [v2.x.x] - Previous versions
See git history for older changes.
```

STEP 4 - UPDATE ARCHITECTURE.md:
Add new sections for the new modules:

```markdown
## New Modules (v3)

### core/feature_registry.py
Single source of truth for ML features. Both training and inference import from here.

### core/api_keys.py
Centralized API key management with validation and logging.

### core/scoring/unified_scorer.py
Consolidated scoring entry point. All other scoring functions should delegate here.
```

After changes:
git diff --stat *.md
```

---

## 🔧 פקודה 7: טסטים לאינטגרציה

```
Add comprehensive integration tests to verify everything works together.

STEP 1 - CREATE tests/integration/ DIRECTORY:
mkdir -p tests/integration

STEP 2 - CREATE tests/integration/test_full_pipeline.py:

```python
"""Integration tests for the full scoring pipeline."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_ticker_data():
    """Realistic ticker data for AAPL."""
    return {
        "Ticker": "AAPL",
        "Close": 185.50,
        "Open": 184.00,
        "High": 186.20,
        "Low": 183.50,
        "Volume": 52_000_000,
        "High_52w": 199.62,
        "Low_52w": 164.08,
        "MA20": 182.30,
        "MA50": 178.45,
        "MA200": 175.20,
        "data_sources": ["yahoo", "polygon"],
    }


@pytest.fixture
def sample_technical_indicators():
    """Computed technical indicators."""
    return {
        "RSI": 58.5,
        "ATR": 3.25,
        "ATR_Pct": 0.0175,
        "Return_20d": 0.045,
        "Return_10d": 0.022,
        "Return_5d": 0.008,
        "VCP_Ratio": 0.85,
        "Tightness_Ratio": 0.72,
        "MA_Alignment": 1,
        "Volume_Surge": 1.15,
        "Momentum_Consistency": 0.65,
    }


@pytest.fixture
def sample_fundamental_data():
    """Fundamental metrics."""
    return {
        "PE_Ratio": 28.5,
        "PB_Ratio": 45.2,
        "ROE": 0.147,
        "Revenue_Growth": 0.08,
        "Profit_Margin": 0.25,
        "Debt_to_Equity": 1.52,
    }


class TestUnifiedScorer:
    """Test UnifiedScorer integration."""

    def test_scorer_produces_valid_result(
        self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data
    ):
        """UnifiedScorer should produce a valid ScoringResult."""
        from core.scoring.unified_scorer import UnifiedScorer, ScoringResult

        scorer = UnifiedScorer({"enable_ml": False})  # Disable ML for faster test
        result = scorer.score(
            sample_ticker_data,
            sample_technical_indicators,
            sample_fundamental_data
        )

        assert isinstance(result, ScoringResult)
        assert 0 <= result.final_conviction <= 100
        assert 0 <= result.technical_score <= 100
        assert 0 <= result.fundamental_score <= 100

    def test_scorer_breakdown_contains_expected_keys(
        self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data
    ):
        """Scoring breakdown should have required keys."""
        from core.scoring.unified_scorer import UnifiedScorer

        scorer = UnifiedScorer({"enable_ml": False})
        result = scorer.score(
            sample_ticker_data,
            sample_technical_indicators,
            sample_fundamental_data
        )

        assert "base_conviction" in result.breakdown
        assert "technical_weight" in result.breakdown
        assert "fundamental_weight" in result.breakdown


class TestFeatureRegistry:
    """Test feature registry integration."""

    def test_feature_count_matches_model_expectation(self):
        """Feature registry should have exactly 34 features."""
        from core.feature_registry import get_feature_names, FEATURE_COUNT_V3

        features = get_feature_names("v3")
        assert len(features) == 34
        assert len(features) == FEATURE_COUNT_V3

    def test_ml_integration_uses_registry_features(self):
        """ml_integration.py should use features from registry."""
        from core.feature_registry import get_feature_names
        from core.ml_integration import EXPECTED_FEATURES

        registry_features = get_feature_names("v3")
        assert EXPECTED_FEATURES == registry_features


class TestAPIKeyManager:
    """Test API key management."""

    def test_missing_optional_key_returns_none(self):
        """Optional keys should return None if not set."""
        from core.api_keys import get_api_key

        # Use a key that's unlikely to be set
        result = get_api_key("DEFINITELY_NOT_SET_KEY_12345", required=False)
        assert result is None

    def test_missing_required_key_raises(self):
        """Required keys should raise if not set."""
        from core.api_keys import get_api_key

        with pytest.raises(EnvironmentError):
            get_api_key("DEFINITELY_NOT_SET_KEY_12345", required=True)
```

STEP 3 - CREATE tests/integration/test_model_loading.py:

```python
"""Test ML model loading and inference."""
import pytest
import os
from pathlib import Path


class TestModelLoading:
    """Test that models load correctly."""

    def test_model_path_resolution(self):
        """Model finder should locate a model file."""
        from core.ml_integration import _find_model_path

        path = _find_model_path()
        # May be None if no model exists, which is OK for CI
        if path is not None:
            assert os.path.exists(path), f"Model path {path} does not exist"

    def test_model_info_structure(self):
        """get_model_info should return expected structure."""
        from core.ml_integration import get_model_info

        info = get_model_info()

        assert "loaded" in info
        assert "path" in info
        assert "feature_count" in info
        assert "model_type" in info
        assert "features" in info

    @pytest.mark.skipif(
        not os.path.exists("models/model_20d_v3.pkl"),
        reason="Model file not present"
    )
    def test_v3_model_loads_successfully(self):
        """v3 model should load and validate."""
        from core.ml_integration import load_ml_model, get_model_info

        result = load_ml_model("models/model_20d_v3.pkl")
        assert result is True

        info = get_model_info()
        assert info["loaded"] is True
        assert info["feature_count"] == 34
```

STEP 4 - CREATE tests/conftest.py fixtures if not exists:

```python
"""Shared pytest fixtures."""
import pytest
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

After changes, run:
pytest tests/integration/ -v --tb=short
pytest tests/test_feature_registry.py -v
```

---

## ✅ פקודה סיום: בדיקה כוללת

```
Run final verification to ensure all fixes are working:

1. Run all tests:
   pytest tests/ -v --tb=short -x

2. Verify model loading:
   python -c "
from core.ml_integration import load_ml_model, get_model_info
load_ml_model()
info = get_model_info()
print(f'Model loaded: {info[\"loaded\"]}')
print(f'Model path: {info[\"path\"]}')
print(f'Feature count: {info[\"feature_count\"]}')
"

3. Verify feature registry:
   python -c "
from core.feature_registry import get_feature_names, FEATURE_COUNT_V3
from core.ml_integration import EXPECTED_FEATURES
print(f'Registry features: {len(get_feature_names(\"v3\"))}')
print(f'ML Integration features: {len(EXPECTED_FEATURES)}')
print(f'Match: {get_feature_names(\"v3\") == EXPECTED_FEATURES}')
"

4. Verify API keys module:
   python -c "
from core.api_keys import log_api_key_status
log_api_key_status()
"

5. Verify UnifiedScorer:
   python -c "
from core.scoring.unified_scorer import UnifiedScorer
scorer = UnifiedScorer({'enable_ml': False})
print('UnifiedScorer initialized OK')
"

6. Show git status:
   git status
   git diff --stat

If all checks pass, commit the changes:
git add -A
git commit -m 'fix: resolve all critical issues

- Add APIKeyManager for secure key handling
- Fix model path to use models/model_20d_v3.pkl
- Create Feature Registry as single source of truth
- Add UnifiedScorer for consistent scoring
- Update GitHub Actions timing for NYSE hours
- Add integration tests
- Update documentation

Closes #XX (if applicable)'
```

---

## 📋 סיכום

| פקודה | מה היא עושה | זמן משוער |
|-------|-------------|-----------|
| 1 | הסרת API Keys חשופים | 10 דקות |
| 2 | תיקון נתיב המודל | 15 דקות |
| 3 | יצירת Feature Registry | 20 דקות |
| 4 | איחוד Scoring Logic | 25 דקות |
| 5 | תזמון GitHub Actions | 10 דקות |
| 6 | עדכון דוקומנטציה | 15 דקות |
| 7 | טסטים לאינטגרציה | 20 דקות |
| סיום | בדיקה כוללת | 5 דקות |

**סה"כ: ~2 שעות**

בהצלחה! 🚀
