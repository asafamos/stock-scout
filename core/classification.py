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

    # 4) Hard R:R minimum filter
    try:
        from core.scoring_config import HARD_FILTERS
        _min_rr = float(HARD_FILTERS.get("min_rr", 0.0))
        if _min_rr > 0:
            _rr_val = row.get("RR", row.get("RR_Ratio", row.get("RewardRisk", None)))
            if _rr_val is not None and isinstance(_rr_val, (int, float)) and np.isfinite(_rr_val):
                if float(_rr_val) < _min_rr:
                    reasons.append(f"rr_below_minimum({float(_rr_val):.2f}<{_min_rr})")
    except Exception:
        pass

    # 5) Negative ROE filter
    try:
        from core.scoring_config import HARD_FILTERS as _hf
        _min_roe = _hf.get("min_roe", None)
        if _min_roe is not None:
            _roe_val = row.get("roe", row.get("ROE", row.get("ROE_f", None)))
            if _roe_val is not None and isinstance(_roe_val, (int, float)) and np.isfinite(_roe_val):
                _roe_pct = float(_roe_val) * 100.0 if abs(float(_roe_val)) < 0.5 else float(_roe_val)
                if _roe_pct < float(_min_roe):
                    reasons.append(f"low_roe({_roe_pct:.1f}%<{float(_min_roe):.0f}%)")
    except Exception:
        pass

    # 6) Missing fundamental data filter (both ROE and MarketCap absent)
    try:
        from core.scoring_config import HARD_FILTERS as _hf2
        if _hf2.get("require_fundamental_data", False):
            _roe_v = row.get("roe", row.get("ROE", row.get("ROE_f", None)))
            _mcap_v = row.get("MarketCap", row.get("market_cap", row.get("Market_Cap", None)))
            _roe_missing = (_roe_v is None) or (isinstance(_roe_v, float) and not np.isfinite(_roe_v))
            _mcap_missing = (_mcap_v is None) or (isinstance(_mcap_v, float) and not np.isfinite(_mcap_v))
            if _roe_missing and _mcap_missing:
                reasons.append("missing_fundamental_data(ROE+MarketCap)")
    except Exception:
        pass

    blocked = len(reasons) > 0
    return {"blocked": blocked, "reasons": reasons}


# -----------------------------
# Risk Class Assignment
# -----------------------------
def assign_risk_class(row: pd.Series) -> str:
    """
    Assign a simple risk class using FinalScore_20d and risk metrics.

    Returns one of: 'CORE', 'SPEC', 'REJECT'.
    Policy:
    - If safety filters block → 'REJECT'.
    - Else if FinalScore_20d >= 55, vol/beta moderate, and R/R >= 0.8 → 'CORE'.
    - Else if FinalScore_20d >= 40 → 'SPEC'.
    - Else → 'REJECT'.

    NOTE (2026-02-15): Added minimum R/R ratio check for CORE classification.
    Stocks with very poor risk/reward (R/R < 0.8) are demoted to SPEC even
    if their score and volatility qualify, since the downside exceeds the
    upside significantly. The threshold of 0.8 is lenient enough to avoid
    rejecting stocks with moderate setups but strict enough to filter out
    clearly unfavorable risk profiles (e.g., R/R = 0.26).
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

    # Risk/Reward gate for CORE (2026-02-15)
    # Missing R/R is tolerated (rr_ok defaults True) to avoid penalizing data gaps.
    # Only stocks with a *known* poor R/R are demoted.
    rr = row.get("RR", row.get("RR_Ratio", row.get("RewardRisk", None)))
    rr_ok = True  # default: pass if R/R data unavailable
    try:
        if rr is not None and isinstance(rr, (int, float)) and np.isfinite(rr):
            rr_ok = float(rr) >= 1.5  # raised from 0.8 to align with HARD_FILTERS min_rr
    except (TypeError, ValueError):
        pass

    # Reliability gate for CORE (stocks without sufficient data quality stay SPEC)
    rel_ok = True  # default: pass if unavailable
    try:
        from core.scoring_config import CORE_MIN_RELIABILITY
        _rel = row.get("reliability_v2", row.get("Reliability_v2", row.get("Reliability_Score", None)))
        if _rel is not None and isinstance(_rel, (int, float)) and np.isfinite(_rel):
            _rel_f = float(_rel)
            if _rel_f <= 1.0:
                _rel_f *= 100.0
            rel_ok = _rel_f >= CORE_MIN_RELIABILITY
    except Exception:
        pass

    # CORE: High quality stocks with strong scores, moderate risk, acceptable R/R, good reliability
    # SPEC: Speculative but promising stocks
    # REJECT: Below threshold or blocked by safety filters
    if np.isfinite(score_val) and score_val >= 55 and vol_ok and beta_ok and rr_ok and rel_ok:
        return "CORE"
    if np.isfinite(score_val) and score_val >= 40:
        return "SPEC"
    return "REJECT"


# ---------------------------------
# Batch application / Compatibility
# ---------------------------------
def _derive_quality_band(row: pd.Series) -> str:
    """Coarse data quality band from reliability proxies (High/Medium/Low)."""
    try:
        from core.scoring_config import RELIABILITY_BANDS
        _high_min = RELIABILITY_BANDS.get("high_min", 65)
        _medium_min = RELIABILITY_BANDS.get("medium_min", 45)
    except Exception:
        _high_min, _medium_min = 65, 45

    rel = row.get("reliability_v2", row.get("Reliability_v2", row.get("Reliability_Score", None)))
    try:
        if rel is None or not np.isfinite(rel):
            return "medium"
        rel_val = float(rel)
        if rel_val <= 1.0:
            rel_val *= 100.0
        if rel_val >= _high_min:
            return "high"
        if rel_val >= _medium_min:
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

        # Hunter rule: if data is incomplete but the setup is a technical gem,
        # keep it visible as SPEC instead of REJECT.
        try:
            data_integrity = str(row.get("Data_Integrity", "OK")).upper()
            coil_bonus_active = bool(row.get("Coil_Bonus", 0)) or str(row.get("Coil_Bonus", "0")) in ("1", "True")
            vcp_score = row.get("Volatility_Contraction_Score", 0.0)
            vcp_good = isinstance(vcp_score, (int, float)) and np.isfinite(vcp_score) and float(vcp_score) > 0.6
            if (rc == "REJECT") and (not blocked) and (data_integrity == "DATA_INCOMPLETE") and (coil_bonus_active or vcp_good):
                rc = "SPEC"
        except Exception:
            pass

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
