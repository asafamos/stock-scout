"""
Reliability Score Computation for Stock Scout.

Measures data confidence: how many sources confirm the data, how
consistent are the prices, and how complete is the fundamental coverage.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.scoring.utils import normalize_score


def calculate_reliability_score(
    price_sources: int = 0,
    fund_sources: int = 0,
    price_std: Optional[float] = None,
    price_mean: Optional[float] = None,
    fundamental_confidence: float = 0.0,
    data_completeness: float = 0.0,
) -> float:
    """Calculate unified reliability score (0-100).

    Components and weights:
        - Data completeness (40%)
        - Cross-source price variance (30%) — lower variance = higher score
        - Fundamental coverage (20%)
        - Number of data sources (10%) — capped at 6
    """
    components: list[float] = []
    weights: list[float] = []

    # Data completeness (40%)
    if 0 <= data_completeness <= 100:
        components.append(data_completeness)
        weights.append(0.40)

    # Price variance — lower is better (30%)
    if (
        price_std is not None
        and price_mean is not None
        and np.isfinite(price_std)
        and np.isfinite(price_mean)
        and price_mean > 0
    ):
        cv = (price_std / price_mean) * 100  # Coefficient of variation
        variance_score = 100 - normalize_score(np.clip(cv, 0, 5), 0, 5, 50)
        components.append(variance_score)
        weights.append(0.30)

    # Fundamental coverage (20%)
    if 0 <= fundamental_confidence <= 100:
        components.append(fundamental_confidence)
        weights.append(0.20)

    # Source count — more = better (10%)
    total_sources = price_sources + fund_sources
    source_score = normalize_score(min(total_sources, 6), 0, 6, 50)
    components.append(source_score)
    weights.append(0.10)

    if not components:
        return 50.0

    total_weight = sum(weights)
    if total_weight == 0:
        return 50.0

    nw = [w / total_weight for w in weights]
    final = sum(c * w for c, w in zip(components, nw))
    return float(np.clip(final, 0, 100))
