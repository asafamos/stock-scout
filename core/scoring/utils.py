"""
Scoring utility functions shared across all scoring modules.

Functions:
    normalize_score  — Map arbitrary range to [0, 100]
    safe_divide      — Division with zero/NaN protection
    evaluate_rr_unified — Risk/Reward ratio → (score, band)
    ml_boost_component  — ML probability → bounded ±10 tilt
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def normalize_score(
    value: float,
    min_val: float = 0.0,
    max_val: float = 100.0,
    default: float = 50.0,
) -> float:
    """Normalize *value* to [0, 100] range.

    ``normalized = ((value - min_val) / (max_val - min_val)) × 100``

    Returns *default* for NaN / inf / equal range.
    """
    if not np.isfinite(value):
        return default
    if max_val == min_val:
        return default
    normalized = ((value - min_val) / (max_val - min_val)) * 100.0
    return float(np.clip(normalized, 0.0, 100.0))


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safe division — returns *default* on zero/NaN/inf."""
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return default
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def evaluate_rr_unified(
    rr_ratio: Optional[float],
) -> Tuple[float, float, str]:
    """Unified Risk/Reward evaluation — single source of truth.

    Scoring bands (tiered):
        < 1.0  → Very Poor  (max 20 pts)
        1.0–1.5 → Poor       (20–40 pts)
        1.5–2.0 → Fair       (40–70 pts)
        2.0–3.0 → Good       (70–90 pts)
        ≥ 3.0   → Excellent  (90–100 pts)

    Returns:
        ``(rr_score, rr_ratio_clipped, rr_band)``
    """
    if rr_ratio is None or not np.isfinite(rr_ratio) or rr_ratio < 0:
        return 0.0, 0.0, "N/A"

    ratio = float(np.clip(rr_ratio, 0, 15.0))

    if ratio < 1.0:
        score = normalize_score(ratio, 0, 1.0, 0) * 0.2
        band = "Very Poor"
    elif ratio < 1.5:
        score = normalize_score(ratio, 1.0, 1.5, 0) * 0.2 + 20
        band = "Poor"
    elif ratio < 2.0:
        score = normalize_score(ratio, 1.5, 2.0, 0) * 0.3 + 40
        band = "Fair"
    elif ratio < 3.0:
        score = normalize_score(ratio, 2.0, 3.0, 0) * 0.2 + 70
        band = "Good"
    else:
        score = normalize_score(np.clip(ratio, 3.0, 5.0), 3.0, 5.0, 0) * 0.1 + 90
        band = "Excellent"

    return float(np.clip(score, 0, 100)), ratio, band


def ml_boost_component(prob: float) -> float:
    """Return a bounded adjustment (±6) based on ML probability.

    Reduced from ±10 to ±6 because the current model (AUC ~0.55)
    doesn't warrant large score swings. The AUC gate in
    compute_final_score_20d further scales this down for weak models.

    Neutral (0.5) → 0; High (1.0) → +6; Low (0.0) → −6.
    """
    try:
        if prob is None or not np.isfinite(prob):
            return 0.0
        p = float(np.clip(prob, 0.0, 1.0))
        delta = (p - 0.5) * 2.0 * 6.0
        return float(np.clip(delta, -6.0, 6.0))
    except Exception:
        return 0.0
