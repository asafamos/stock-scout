"""
Portfolio allocation utilities.

This module contains helper functions for portfolio construction
and weight normalization.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a weights dictionary so that values sum to 1.0.
    
    Args:
        weights: Dictionary mapping keys to numeric weights
        
    Returns:
        Dictionary with same keys but normalized values summing to 1.0
        
    Example:
        >>> _normalize_weights({"a": 2, "b": 3, "c": 5})
        {"a": 0.2, "b": 0.3, "c": 0.5}
    """
    total = sum(weights.values())
    if total <= 0 or not np.isfinite(total):
        # if invalid total, return equal weights
        n = len(weights)
        return {k: 1.0 / n if n > 0 else 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}
