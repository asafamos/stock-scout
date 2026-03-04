"""ML 20-day inference module.

Loads the trained model bundle at import time and provides:
- ``predict_20d_prob_from_row(row)`` — single-row raw probability
- ``apply_live_v3_adjustments(df)`` — batch probability adjustments
- ``calibrate_ml_20d_prob(prob_raw, ...)`` — single-row calibration
- ``get_ml_health_meta()`` — model health / diagnostics
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# Import EnsembleClassifier so pickle can find it when unpickling the model
from core.ensemble import EnsembleClassifier  # noqa: F401 — needed for unpickling

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (populated by _load_bundle_cached at import time)
# ---------------------------------------------------------------------------
ML_20D_AVAILABLE: bool = False
BUNDLE_MODEL: Any = None
FEATURE_COLS_20D: list[str] = []
PREFERRED_SCORING_MODE_20D: str = "hybrid"

# Health / diagnostics
ML_VERSION_WARNING: bool = False
ML_VERSION_WARNING_REASON: Optional[str] = None
ML_MISSING_FEATURES: List[str] = []
BUNDLE_HAS_MISSING_METEOR_FEATURES: bool = False
BUNDLE_AUC: Optional[float] = None  # OOS AUC from metadata (for circuit breaker)

# Feature aliases: maps expected_name → fallback_name(s) in row data
_FEATURE_ALIASES: Dict[str, List[str]] = {
    "ADR_Pct": ["ATR_Pct"],
    "Return_20d": ["Return_1m"],
}


# ============================================================================
# Model loading helpers
# ============================================================================

def _capture_version_warnings(wlist: list) -> Optional[str]:
    """Inspect captured warnings for sklearn version mismatch.

    Returns the warning message string if found, else None.
    """
    for w in wlist or []:
        name = getattr(w.category, "__name__", "")
        msg = str(getattr(w, "message", ""))
        if "InconsistentVersionWarning" in name or "InconsistentVersionWarning" in msg:
            return msg
    return None


def _patch_sklearn_compat(model: Any) -> None:
    """Patch sklearn version compatibility issues on a loaded model.

    When a model is trained with a newer sklearn (e.g., 1.8.0) and loaded
    with an older one (e.g., 1.6.1), some attributes may be missing.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return

    def _patch_one(est: Any) -> None:
        if isinstance(est, LogisticRegression) and not hasattr(est, "multi_class"):
            est.multi_class = "auto"

    def _patch_recursive(obj: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        _patch_one(obj)
        # EnsembleClassifier stores sub-models in .models
        for sub in getattr(obj, "models", []):
            _patch_recursive(sub, depth + 1)
        # CalibratedClassifierCV stores calibrated classifiers
        for cc in getattr(obj, "calibrated_classifiers_", []):
            _patch_recursive(cc, depth + 1)
            # _CalibratedClassifier wraps the estimator
            inner = getattr(cc, "estimator", None)
            if inner is not None:
                _patch_recursive(inner, depth + 1)
        # Sklearn meta-estimators (Pipeline, VotingClassifier, etc.)
        for est in getattr(obj, "estimators_", []):
            _patch_recursive(est[1] if isinstance(est, (list, tuple)) else est, depth + 1)

    _patch_recursive(model)


def _extract_model_feature_names(model: Any) -> list[str]:
    """Extract feature names from a fitted model/ensemble.

    Checks the model itself first, then sub-models (for EnsembleClassifier).
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    for sub in getattr(model, "models", []):
        if hasattr(sub, "feature_names_in_"):
            return list(sub.feature_names_in_)
    for est in getattr(model, "estimators_", []):
        item = est[1] if isinstance(est, (list, tuple)) else est
        if hasattr(item, "feature_names_in_"):
            return list(item.feature_names_in_)
    return []


def _load_joblib_safe(path: Path) -> tuple[Any, Optional[str]]:
    """Load a joblib/pickle file, capturing version warnings.

    Returns (loaded_object, version_warning_msg_or_None).
    """
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        obj = joblib.load(path)
    return obj, _capture_version_warnings(wlist)


def _finalize_model(
    model: Any,
    feature_names: list[str],
    meta_auc: Optional[float] = None,
) -> tuple[bool, Any, list[str], str]:
    """Common post-load steps: patch, extract features, validate."""
    if model is None or not hasattr(model, "predict_proba"):
        logger.error("Model missing or lacks predict_proba method")
        return False, None, [], "hybrid"

    _patch_sklearn_compat(model)

    # Prefer the model's own feature list over metadata (metadata can be stale)
    model_features = _extract_model_feature_names(model)
    if model_features:
        if feature_names and set(model_features) != set(feature_names):
            logger.warning(
                "Metadata lists %d features but model expects %d — using model's own list.",
                len(feature_names), len(model_features),
            )
        feature_names = model_features

    if not feature_names:
        logger.warning("No feature names found — treating as degraded")
        global BUNDLE_HAS_MISSING_METEOR_FEATURES
        BUNDLE_HAS_MISSING_METEOR_FEATURES = True

    # Version safety: validate feature count matches a known registry version
    _KNOWN_FEATURE_COUNTS = {34: "v3", 39: "v3.1", 16: "v3.3", 13: "v3.4", 20: "v3.5"}
    n_feat = len(feature_names)
    detected_version = _KNOWN_FEATURE_COUNTS.get(n_feat)
    if detected_version:
        logger.info("ML model feature count %d matches registry %s", n_feat, detected_version)
    elif n_feat > 0:
        logger.warning(
            "ML model has %d features — does not match any known registry version "
            "(v3=34, v3.1=39, v3.3=16, v3.4=13, v3.5=20). Model may be stale or from an intermediate build.",
            n_feat,
        )

    global BUNDLE_AUC
    BUNDLE_AUC = meta_auc

    logger.info(
        "✓ ML model loaded: %d features, AUC=%s",
        len(feature_names),
        f"{meta_auc:.4f}" if meta_auc else "unknown",
    )
    return True, model, feature_names, "hybrid"


# ============================================================================
# Bundle loading
# ============================================================================

def _load_bundle_impl() -> tuple[bool, Any, list[str], str]:
    """Load model bundle from disk.

    Lookup order:
    1. ``ml/bundles/latest/model.joblib``  (preferred, with metadata.json)
    2. ``models/model_20d_v3.pkl``         (legacy fallback)

    The directory can be overridden via the ``ML_BUNDLE_DIR`` env-var.
    """
    global ML_VERSION_WARNING, ML_VERSION_WARNING_REASON

    try:
        root = Path(__file__).resolve().parent.parent  # project root
        bundle_dir = Path(os.getenv(
            "ML_BUNDLE_DIR",
            str(root / "ml" / "bundles" / "latest"),
        ))
        model_path = bundle_dir / "model.joblib"
        meta_path = bundle_dir / "metadata.json"

        # ----- New bundle (preferred) -----
        if model_path.exists():
            model, warn_msg = _load_joblib_safe(model_path)
            if warn_msg:
                ML_VERSION_WARNING = True
                ML_VERSION_WARNING_REASON = warn_msg
                logger.warning("ML bundle version warning: %s", warn_msg)

            # Read metadata
            feature_names: list[str] = []
            meta_auc: Optional[float] = None
            try:
                meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                fl = meta_obj.get("feature_list") or []
                if isinstance(fl, list):
                    feature_names = list(fl)
                # AUC for circuit breaker
                metrics = meta_obj.get("metrics") or {}
                raw_auc = metrics.get("oos_auc") or metrics.get("cv_auc_mean")
                if raw_auc is not None:
                    meta_auc = float(raw_auc)
                # Version check
                try:
                    import sklearn
                    meta_ver = str(meta_obj.get("sklearn_version") or "").strip()
                    rt_ver = str(getattr(sklearn, "__version__", "")).strip()
                    if meta_ver and rt_ver and meta_ver != rt_ver:
                        ML_VERSION_WARNING = True
                        ML_VERSION_WARNING_REASON = (
                            f"Bundle sklearn_version={meta_ver} but runtime={rt_ver}"
                        )
                        logger.warning(ML_VERSION_WARNING_REASON)
                except Exception:
                    pass
            except Exception as exc:
                logger.warning("Failed to read bundle metadata: %s", exc)

            return _finalize_model(model, feature_names, meta_auc)

        # ----- Legacy fallback -----
        legacy_path = root / "models" / "model_20d_v3.pkl"
        if legacy_path.exists():
            logger.warning("Using legacy model bundle (no metadata)")
            bundle, warn_msg = _load_joblib_safe(legacy_path)
            if warn_msg:
                ML_VERSION_WARNING = True
                ML_VERSION_WARNING_REASON = warn_msg
                logger.warning("ML bundle version warning: %s", warn_msg)

            if isinstance(bundle, dict):
                model = bundle.get("model")
                feature_names = list(bundle.get("feature_names", []))
            else:
                model = bundle
                feature_names = []
            return _finalize_model(model, feature_names)

        logger.error("No ML model found at %s or legacy path", model_path)
        return False, None, [], "hybrid"

    except Exception as exc:
        logger.error("Failed to load ML bundle: %s", exc, exc_info=True)
        return False, None, [], "hybrid"


@lru_cache(maxsize=1)
def _load_bundle_cached():
    return _load_bundle_impl()


# Initialise module globals at import time
_success, BUNDLE_MODEL, FEATURE_COLS_20D, PREFERRED_SCORING_MODE_20D = (
    _load_bundle_cached()
)
PREFERRED_SCORING_MODE_20D = "hybrid"  # always override to hybrid
ML_20D_AVAILABLE = _success


# ============================================================================
# Circuit breaker: when model AUC is weak, reduce its influence
# ============================================================================

def get_ml_weight_multiplier() -> float:
    """Return a weight multiplier (0.0–1.0) based on model quality.

    AUC ≤ 0.52  → 0.0  (disabled — indistinguishable from random)
    AUC ≤ 0.55  → 0.25 (heavily reduced — marginal signal at best)
    AUC ≤ 0.58  → 0.5  (halved — some signal but noisy)
    AUC > 0.58  → 1.0  (full weight — meaningful signal)

    After retraining with v3.3 (16 features, rank-based labels), AUC should improve.
    These gates remain conservative until the retrained model proves itself.
    """
    if BUNDLE_AUC is None:
        return 0.5  # unknown AUC — don't trust or distrust; halve ML weight
    if BUNDLE_AUC <= 0.52:
        return 0.0
    if BUNDLE_AUC <= 0.55:
        return 0.25
    if BUNDLE_AUC <= 0.58:
        return 0.5
    return 1.0


# ============================================================================
# Prediction
# ============================================================================

def compute_ml_20d_probabilities_raw(row: pd.Series) -> float:
    """Compute RAW ML 20-day probability.

    Returns:
        float in [0, 1] if model available, ``np.nan`` otherwise.
    """
    if not ML_20D_AVAILABLE or BUNDLE_MODEL is None or not FEATURE_COLS_20D:
        return np.nan

    try:
        # Feature defaults from registry (graceful if unavailable)
        try:
            from core.feature_registry import get_feature_defaults, clip_features_to_range
            _n = len(FEATURE_COLS_20D)
            _version = "v3.1" if _n >= 39 else ("v3.5" if _n >= 20 else ("v3.3" if _n >= 16 else ("v3.4" if _n >= 13 else "v3")))
            defaults = get_feature_defaults(_version)
        except Exception:
            defaults = {}
            clip_features_to_range = None  # type: ignore[assignment]

        # Build row dict (shallow copy to avoid mutating caller)
        row_data: dict = dict(row) if isinstance(row, pd.Series) else dict(row)

        # Apply known feature aliases
        for expected, fallbacks in _FEATURE_ALIASES.items():
            if expected in FEATURE_COLS_20D and expected not in row_data:
                for fb in fallbacks:
                    if fb in row_data:
                        row_data[expected] = row_data[fb]
                        break

        # Build feature vector in exact training order
        feature_dict: dict = {}
        missing: list[str] = []

        for col in FEATURE_COLS_20D:
            val = row_data.get(col, np.nan)
            if not isinstance(val, (int, float)) or (isinstance(val, float) and np.isnan(val)):
                missing.append(col)
                val = defaults.get(col, 0.0)
            feature_dict[col] = val

        # NOTE: Feature muting removed — model is now trained on v3.3 (16 pruned
        # features) instead of muting harmful features at inference time.
        # The model itself only expects the 16 features that actually help.

        # Track missing features for health reporting
        if missing:
            global ML_MISSING_FEATURES, BUNDLE_HAS_MISSING_METEOR_FEATURES
            current = set(ML_MISSING_FEATURES or [])
            current.update(missing)
            ML_MISSING_FEATURES = sorted(current)
            BUNDLE_HAS_MISSING_METEOR_FEATURES = True
            if len(missing) > 10:
                logger.debug("ML 20d: filled %d missing features with defaults", len(missing))

        # Build DataFrame
        X = pd.DataFrame([feature_dict])[FEATURE_COLS_20D]
        X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Clip features using registry ranges (if available)
        if clip_features_to_range is not None:
            try:
                X = clip_features_to_range(X, _version)
            except Exception:
                # Fallback: minimal hardcoded clipping
                if "ATR_Pct" in X.columns:
                    X["ATR_Pct"] = np.clip(X["ATR_Pct"], 0.0, 0.2)
                if "RSI" in X.columns:
                    X["RSI"] = np.clip(X["RSI"], 5.0, 95.0)
        else:
            if "ATR_Pct" in X.columns:
                X["ATR_Pct"] = np.clip(X["ATR_Pct"], 0.0, 0.2)
            if "RSI" in X.columns:
                X["RSI"] = np.clip(X["RSI"], 5.0, 95.0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            proba = BUNDLE_MODEL.predict_proba(X)
        return float(np.clip(proba[0, 1], 0.0, 1.0))

    except Exception as exc:
        logger.warning("ML 20d prediction failed: %s", exc, exc_info=True)
        return np.nan


def predict_20d_prob_from_row(row: pd.Series) -> float:
    """Backward-compatible wrapper — delegates to ``compute_ml_20d_probabilities_raw``."""
    return compute_ml_20d_probabilities_raw(row)


# ============================================================================
# Probability adjustments (live_v3)
# ============================================================================

def apply_live_v3_adjustments(
    df: pd.DataFrame,
    prob_col: str = "ML_20d_Prob_raw",
    enable_adjustments: bool = False,
) -> pd.Series:
    """Apply live_v3 adjustments to raw ML probabilities.

    WARNING: These adjustments are not empirically validated.
    Set *enable_adjustments=True* to activate.
    """
    if prob_col not in df.columns:
        logger.warning("Column %s not found — returning 0.5 for all rows", prob_col)
        return pd.Series(0.5, index=df.index)

    adjusted = df[prob_col].copy()
    if not enable_adjustments:
        return pd.Series(np.clip(adjusted, 0.01, 0.99), index=df.index)

    # Volatility bucket adjustments
    if "ATR_Pct_percentile" in df.columns:
        vol = df["ATR_Pct_percentile"].fillna(0.5)
        adjusted = np.where((vol < 0.25), adjusted - 0.01, adjusted)
        adjusted = np.where((vol >= 0.50) & (vol < 0.75), adjusted + 0.015, adjusted)
        adjusted = np.where((vol >= 0.75), adjusted - 0.005, adjusted)

    # Price bucket adjustments
    if "Price_As_Of_Date" in df.columns:
        price = df["Price_As_Of_Date"].fillna(50.0)
        adjusted = np.where(
            (price > 0) & (price < 20) & (df[prob_col] > 0.55),
            adjusted + 0.01, adjusted,
        )
        adjusted = np.where((price >= 20) & (price < 50), adjusted + 0.01, adjusted)
        adjusted = np.where((price >= 150), adjusted - 0.01, adjusted)

    # Reliability multiplier
    if "ReliabilityFactor" in df.columns:
        adjusted = adjusted * df["ReliabilityFactor"].fillna(1.0)

    return pd.Series(np.clip(adjusted, 0.01, 0.99), index=df.index)


def calibrate_ml_20d_prob(
    prob_raw: float,
    *,
    atr_pct_percentile: float | None = None,
    price_as_of: float | None = None,
    reliability_factor: float | None = None,
    market_regime: str | None = None,
    rsi: float | None = None,
    enable_adjustments: bool = False,
) -> float:
    """Calibrate a single raw ML 20d probability.

    WARNING: Adjustments are not empirically validated.
    Set *enable_adjustments=True* to activate.
    """
    try:
        if prob_raw is None or not np.isfinite(prob_raw):
            return np.nan
        adj = float(prob_raw)
        if not enable_adjustments:
            return float(np.clip(adj, 0.01, 0.99))

        # Volatility
        if atr_pct_percentile is not None and np.isfinite(atr_pct_percentile):
            v = float(atr_pct_percentile)
            if v < 0.25:
                adj -= 0.01
            elif 0.50 <= v < 0.75:
                boost = 0.035 if (isinstance(market_regime, str) and market_regime.upper() == "BULLISH") else 0.015
                adj += boost
            elif v >= 0.75:
                adj -= 0.005

        # Price
        if price_as_of is not None and np.isfinite(price_as_of):
            p = float(price_as_of)
            if 0 < p < 20 and adj > 0.55:
                adj += 0.01
            elif 20 <= p < 50:
                adj += 0.01
            elif p >= 150:
                adj -= 0.01

        # Reliability
        if reliability_factor is not None and np.isfinite(reliability_factor):
            adj *= float(reliability_factor)
        else:
            adj = 0.95 * adj + 0.05 * 0.5

        # Regime
        regime = (market_regime or "").upper()
        rsi_val = float(rsi) if (rsi is not None and np.isfinite(rsi)) else 50.0
        if regime in {"BEARISH", "PANIC"} and rsi_val > 65.0:
            adj -= 0.06
        if regime == "CORRECTION" and reliability_factor is not None:
            if np.isfinite(reliability_factor) and float(reliability_factor) > 1.1:
                adj += 0.02

        return float(np.clip(adj, 0.01, 0.99))
    except Exception:
        logger.warning("calibrate_ml_20d_prob failed", exc_info=True)
        return np.nan


# ============================================================================
# Ranking column selection
# ============================================================================

def choose_rank_col_20d(df: pd.DataFrame) -> str:
    """Select the ranking column for 20d scoring based on bundle policy."""
    cols = pd.Index(df.columns)
    mode = PREFERRED_SCORING_MODE_20D or "hybrid"

    def first_available(candidates: list[str]) -> str:
        for c in candidates:
            if c in cols:
                return c
        return cols[0] if len(cols) else ""

    if mode == "ml_only":
        return first_available(["ML_20d_Prob", "ML_20d_Prob_live_v3", "ML_20d_Prob_raw", "FinalScore_20d", "FinalScore"])

    overlay = cols[cols.str.contains("OverlayV2_20d|AdjustedScore_20d", regex=True)]
    if mode == "hybrid_overlay" and len(overlay):
        return overlay[0]

    if mode in {"hybrid", "hybrid_overlay"}:
        return first_available(["FinalScore_20d", "HybridFinalScore_20d", "FinalScore"])

    return first_available(["FinalScore_20d", "FinalScore", "HybridFinalScore_20d", "TechScore_20d_v2", "TechScore_20d"])


# ============================================================================
# Health / diagnostics
# ============================================================================

def get_ml_health_meta() -> Dict[str, Any]:
    """Return ML health metadata for pipeline diagnostics."""
    try:
        degraded = (
            (not ML_20D_AVAILABLE)
            or BUNDLE_HAS_MISSING_METEOR_FEATURES
            or bool(ML_VERSION_WARNING)
        )
        _n = len(FEATURE_COLS_20D or [])
        _detected = {34: "v3", 39: "v3.1", 16: "v3.3", 13: "v3.4", 20: "v3.5"}.get(_n, f"unknown({_n})")
        return {
            "ml_bundle_version_warning": bool(ML_VERSION_WARNING),
            "ml_bundle_warning_reason": ML_VERSION_WARNING_REASON,
            "ml_degraded": bool(degraded),
            "ml_missing_features": list(ML_MISSING_FEATURES or []),
            "ml_required_features_count": _n,
            "ml_detected_version": _detected,
            "ml_auc": BUNDLE_AUC,
            "ml_weight_multiplier": get_ml_weight_multiplier(),
        }
    except Exception:
        return {
            "ml_bundle_version_warning": False,
            "ml_bundle_warning_reason": None,
            "ml_degraded": not ML_20D_AVAILABLE,
            "ml_missing_features": [],
            "ml_required_features_count": 0,
            "ml_auc": None,
            "ml_weight_multiplier": 1.0,
        }
