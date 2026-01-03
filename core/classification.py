"""
Classification: safety filters and risk class assignment only.

This module is responsible for two things:
- Hard safety filters (earnings blackout, broken/missing data, extremely low reliability).
- Assigning a simple risk class (CORE / SPEC / REJECT) based primarily on
  FinalScore_20d and a couple of risk-related metrics (volatility/beta).

All numeric scoring, technical combinations, and rule-based score blending is
centralized in core/scoring_engine.py. Do NOT add RSI/RR/fundamentals combos here.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from core.logging_config import get_logger

logger = get_logger("classification")


# -----------------------------
# Safety Filters (Hard Blocks)
# -----------------------------
def apply_safety_filters(row: pd.Series, earnings_window_days: int = 7) -> Dict[str, object]:
    """
    Evaluate hard safety filters that can block a stock from consideration.

    Returns a dict with:
    - blocked: bool
    - reasons: List[str]

    Rules kept intentionally simple and data-availability tolerant.
    """
    reasons: List[str] = []

    # 1) Earnings blackout (± earnings_window_days) if earnings info is available
    # Supported fields: 'NextEarningsDate' (datetime), 'EarningsDate', 'days_to_earnings'
    try:
        from datetime import datetime
        now = datetime.utcnow()
        # days_to_earnings has priority if present
        dte = row.get("days_to_earnings", row.get("DaysToEarnings", None))
        if dte is not None and np.isfinite(dte):
            if abs(float(dte)) <= float(earnings_window_days):
                reasons.append(f"earnings_blackout(±{earnings_window_days}d)")
        else:
            for key in ("NextEarningsDate", "EarningsDate", "Earnings_Date"):
                dt = row.get(key, None)
                if isinstance(dt, (pd.Timestamp,)) or getattr(dt, "isoformat", None):
                    dt_val = pd.to_datetime(dt)
                    delta_days = abs((dt_val - now).days)
                    if delta_days <= earnings_window_days:
                        reasons.append(f"earnings_blackout(±{earnings_window_days}d)")
                        break
    except Exception:
        # Be conservative: if parsing fails, do not block solely on earnings
        pass

    # 2) Broken/missing price or volume data
    close = row.get("Close", row.get("Unit_Price", row.get("Price_Yahoo", np.nan)))
    vol = row.get("Volume", row.get("AvgVolume", np.nan))
    if not (isinstance(close, (int, float)) and np.isfinite(close) and close > 0):
        reasons.append("bad_price_data")
    if vol is not None and not (isinstance(vol, (int, float)) and np.isfinite(vol) and vol > 0):
        reasons.append("bad_volume_data")

    # 3) Extremely low data reliability (any available reliability proxy)
    # Prefer v2 reliability (0-100), fallback to legacy 0-1
    rel_candidates = [
        row.get("reliability_v2"),
        row.get("Reliability_v2"),
        row.get("Price_Reliability_v2"),
        row.get("Reliability_Score"),  # legacy 0-1
    ]
    reliability_value = None
    for val in rel_candidates:
        if val is not None and isinstance(val, (int, float)) and np.isfinite(val):
            reliability_value = float(val)
            break
    if reliability_value is not None:
        # If it's likely 0-1 scale, convert to 0-100 for a single threshold
        if reliability_value <= 1.0:
            reliability_value = reliability_value * 100.0
        if reliability_value < 10.0:
            reasons.append("low_reliability")

    blocked = len(reasons) > 0
    return {"blocked": blocked, "reasons": reasons}


# -----------------------------
# Risk Class Assignment
# -----------------------------
def assign_risk_class(row: pd.Series) -> str:
    """
    Assign a simple risk class using FinalScore_20d and a few risk metrics.

    Returns one of: 'CORE', 'SPEC', 'REJECT'.
    Policy:
    - If safety filters block → 'REJECT'.
    - Else if FinalScore_20d is high and volatility/beta are moderate → 'CORE'.
    - Else if FinalScore_20d is decent → 'SPEC'.
    - Else → 'REJECT'.
    """
    safety = apply_safety_filters(row)
    if safety.get("blocked", False):
        return "REJECT"

    # Primary driver: final score (single source of truth from scoring_engine)
    score = row.get("FinalScore_20d", row.get("Score", None))
    try:
        score_val = float(score) if score is not None and np.isfinite(score) else np.nan
    except Exception:
        score_val = np.nan

    # Risk-related features (tolerant to missing)
    vol = row.get("ATR_Price", row.get("ATR_Pct", None))
    beta = row.get("Beta", None)
    vol_ok = (vol is None) or (isinstance(vol, (int, float)) and np.isfinite(vol) and vol <= 0.06)
    beta_ok = (beta is None) or (isinstance(beta, (int, float)) and np.isfinite(beta) and beta <= 1.6)

    # Clear thresholds; adjust as needed based on backtests
    if np.isfinite(score_val) and score_val >= 70 and vol_ok and beta_ok:
        return "CORE"
    if np.isfinite(score_val) and score_val >= 50:
        return "SPEC"
    return "REJECT"


# ---------------------------------
# Batch application / Compatibility
# ---------------------------------
def _derive_quality_band(row: pd.Series) -> str:
    """Coarse data quality band from reliability proxies (High/Medium/Low)."""
    rel = row.get("reliability_v2", row.get("Reliability_v2", row.get("Reliability_Score", None)))
    try:
        if rel is None or not np.isfinite(rel):
            return "medium"
        rel_val = float(rel)
        if rel_val <= 1.0:
            rel_val *= 100.0
        if rel_val >= 75:
            return "high"
        if rel_val >= 40:
            return "medium"
        return "low"
    except Exception:
        return "medium"


def _derive_confidence(row: pd.Series, blocked: bool) -> str:
    score = row.get("FinalScore_20d", row.get("Score", None))
    try:
        s = float(score) if score is not None and np.isfinite(score) else np.nan
    except Exception:
        s = np.nan
    if blocked:
        return "none"
    if np.isfinite(s) and s >= 80:
        return "high"
    if np.isfinite(s) and s >= 60:
        return "medium"
    return "low"


def apply_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply safety filters and risk class across a DataFrame.

    Adds new fields:
    - RiskClass: 'CORE' | 'SPEC' | 'REJECT'
    - SafetyBlocked: bool
    - SafetyReasons: string (semicolon-separated)

    For backward compatibility, also populates legacy fields expected elsewhere:
    - Risk_Level: 'core' or 'speculative' (REJECT maps to 'speculative')
    - Data_Quality: 'high'|'medium'|'low' (derived from reliability)
    - Confidence_Level: 'high'|'medium'|'low'|'none' (from score & safety)
    - Should_Display: bool (False if blocked or REJECT)
    - Classification_Warnings: SafetyReasons string
    """
    if df is None or df.empty:
        return df

    risk_classes: List[str] = []
    safety_flags: List[bool] = []
    safety_reasons_out: List[str] = []
    data_quality_out: List[str] = []
    confidence_out: List[str] = []
    risk_level_legacy: List[str] = []
    should_display_out: List[bool] = []

    for _, row in df.iterrows():
        safety = apply_safety_filters(row)
        blocked = bool(safety.get("blocked", False))
        reasons_list = safety.get("reasons", []) or []
        reasons_str = "; ".join([str(r) for r in reasons_list])

        rc = assign_risk_class(row) if not blocked else "REJECT"

        # Legacy compatibility mapping
        legacy_level = "core" if rc == "CORE" else "speculative"
        should_display = (rc != "REJECT") and (not blocked)

        risk_classes.append(rc)
        safety_flags.append(blocked)
        safety_reasons_out.append(reasons_str)
        data_quality_out.append(_derive_quality_band(row))
        confidence_out.append(_derive_confidence(row, blocked))
        risk_level_legacy.append(legacy_level)
        should_display_out.append(should_display)

    df = df.copy()
    df["RiskClass"] = risk_classes
    df["SafetyBlocked"] = safety_flags
    df["SafetyReasons"] = safety_reasons_out

    # Legacy columns (kept for downstream compatibility during migration)
    df["Risk_Level"] = risk_level_legacy
    df["Data_Quality"] = data_quality_out
    df["Confidence_Level"] = confidence_out
    df["Should_Display"] = should_display_out
    df["Classification_Warnings"] = df["SafetyReasons"]

    logger.info(
        f"Classified {len(df)} stocks → CORE={sum(df['RiskClass']=='CORE')}, "
        f"SPEC={sum(df['RiskClass']=='SPEC')}, REJECT={sum(df['RiskClass']=='REJECT')}"
    )

    return df


def filter_core_recommendations(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    adaptive: bool = True
) -> pd.DataFrame:
    """
    Deprecated heavy filtering replaced with a thin wrapper.

    Returns only rows with RiskClass == 'CORE' and not SafetyBlocked.
    The previous RSI/RR/ATR/overextension rule soup has been removed — those
    belong in scoring or safety gates upstream.
    """
    if df is None or df.empty:
        return df
    cols_needed = {"RiskClass", "SafetyBlocked"}
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        logger.debug(f"filter_core_recommendations: missing {missing}, applying classification first")
        df = apply_classification(df)
    return df[(df["RiskClass"] == "CORE") & (~df["SafetyBlocked"])].reset_index(drop=True)
