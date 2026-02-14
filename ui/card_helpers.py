"""Card-rendering helpers — pure-logic functions extracted from stock_scout.py."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_float(val) -> float:
    """Coerce *val* to ``float``, returning ``NaN`` on failure."""
    try:
        if val in (None, "", "N/A", "nan"):
            return np.nan
        return float(val)
    except Exception:
        return np.nan


def normalize_prob(p) -> Optional[float]:
    """Normalize a probability value to 0-1 range, or return *None*."""
    try:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return None
        pv = float(p)
        if not np.isfinite(pv):
            return None
        if pv > 1 and pv <= 100:
            pv = pv / 100.0
        elif pv > 100:
            return None
        return max(0.0, min(1.0, pv))
    except Exception:
        return None


def ml_badge(p) -> str:
    """Return a colored badge string for ML probability *p*."""
    norm = normalize_prob(p)
    if norm is None:
        return "—"
    if norm > 0.60:
        return f"🟢 {norm * 100:.0f}%"
    if norm >= 0.40:
        return f"🟡 {norm * 100:.0f}%"
    return f"🔴 {norm * 100:.0f}%"


def get_ml_prob_from_row(r) -> float:
    """Extract the best available ML probability from a row."""
    for k in ("ML_20d_Prob_live_v3", "ML_20d_Prob", "ML_20d_Prob_raw", "ML_Probability"):
        v = r.get(k, np.nan)
        if v is not None and pd.notna(v):
            try:
                return float(v)
            except Exception:
                continue
    return np.nan


def risk_class(row: pd.Series) -> str:
    """Return risk class string for *row*."""
    rc = row.get("RiskClass")
    if isinstance(rc, str) and rc:
        return rc
    rl = str(row.get("Risk_Level", "speculative")).lower()
    return "CORE" if rl == "core" else "SPEC"


def headline_story(row: pd.Series) -> str:
    """Build a one-line narrative from key scores."""
    fund = to_float(row.get("FundamentalScore", row.get("Fundamental_S", np.nan)))
    mom = to_float(row.get("MomentumScore", row.get("TechScore_20d", np.nan)))
    rr = to_float(row.get("RR", row.get("RR_Ratio", row.get("RewardRisk", np.nan))))
    rel = to_float(row.get("ReliabilityScore", row.get("Reliability_Score", row.get("Reliability_v2", np.nan))))

    parts: list[str] = []
    if isinstance(fund, float) and np.isfinite(fund):
        if fund >= 70:
            parts.append("Quality business")
        elif fund >= 50:
            parts.append("Decent fundamentals")
        else:
            parts.append("Weak fundamentals")
    if isinstance(mom, float) and np.isfinite(mom):
        if mom >= 70:
            parts.append("strong momentum")
        elif mom >= 50:
            parts.append("moderate momentum")
        else:
            parts.append("weak momentum")
    if isinstance(rr, float) and np.isfinite(rr):
        if rr >= 2.5:
            parts.append("excellent RR")
        elif rr >= 1.5:
            parts.append("good RR")
        else:
            parts.append("poor RR")
    if isinstance(rel, float) and np.isfinite(rel):
        rel_val = rel * 100.0 if rel <= 1.0 else rel
        if rel_val >= 75:
            parts.append("high data reliability")
        elif rel_val >= 40:
            parts.append("medium reliability")
        else:
            parts.append("low reliability")
    return ", ".join(parts[:4])


def fmt_num(val, fmt: str, na: str = "N/A") -> str:
    """Format *val* with *fmt*, falling back to *na*."""
    try:
        v = to_float(val)
        return format(v, fmt) if isinstance(v, float) and np.isfinite(v) else na
    except Exception:
        return na


def get_reliability_band(reliability_val) -> str:
    """Map a reliability score to High/Medium/Low."""
    if pd.notna(reliability_val) and isinstance(reliability_val, (int, float)):
        if reliability_val >= 75:
            return "High"
        elif reliability_val >= 40:
            return "Medium"
        else:
            return "Low"
    return "Unknown"


def get_reliability_components(row: pd.Series) -> str:
    """Summarize reliability components as a compact string."""
    fund_rel = row.get("Fundamental_Reliability_v2", 0)
    price_rel = row.get("Price_Reliability_v2", 0)
    fund_sources = row.get("fund_sources_used_v2", 0)
    price_sources = row.get("price_sources_used_v2", 0)
    return f"F:{fund_rel:.0f}%(n={fund_sources}),P:{price_rel:.0f}%(n={price_sources})"
