"""ML target/label generation for Stock Scout training pipeline.

Provides both absolute-threshold and rank-based labeling strategies.
Rank-based labeling is preferred for production: it makes the model a
STOCK PICKER (relative to peers) instead of a MARKET TIMER (absolute returns).
"""
import numpy as np
import pandas as pd

from .ml_target_config import (
    HORIZON_DAYS,
    UP_THRESHOLD,
    DOWN_THRESHOLD,
    TARGET_MODE,
    RANK_TOP_PCT,
    RANK_BOTTOM_PCT,
)


def compute_forward_return(close: pd.Series, horizon: int = HORIZON_DAYS) -> pd.Series:
    return close.shift(-horizon) / close - 1


# ---------------------------------------------------------------------------
# Absolute-threshold labeling (legacy)
# ---------------------------------------------------------------------------

def make_label_20d(forward_ret: pd.Series) -> pd.Series:
    """Label using absolute return thresholds (UP_THRESHOLD / DOWN_THRESHOLD)."""
    label = np.where(forward_ret >= UP_THRESHOLD, 1,
                     np.where(forward_ret <= DOWN_THRESHOLD, 0, np.nan))
    return pd.Series(label, index=forward_ret.index)


def make_label_20d_strict(forward_ret: pd.Series) -> pd.Series:
    """Label: 1 if return >= 0.15, 0 if <= 0.02, else np.nan"""
    return pd.Series(
        np.where(forward_ret >= 0.15, 1, np.where(forward_ret <= 0.02, 0, np.nan)),
        index=forward_ret.index,
    )


def make_label_20d_soft(forward_ret: pd.Series) -> pd.Series:
    """Label: 1 if return >= 0.08, 0 if <= 0.00, else np.nan"""
    return pd.Series(
        np.where(forward_ret >= 0.08, 1, np.where(forward_ret <= 0.00, 0, np.nan)),
        index=forward_ret.index,
    )


# ---------------------------------------------------------------------------
# Rank-based labeling (recommended for production)
# ---------------------------------------------------------------------------

def make_label_20d_ranked(
    forward_returns: pd.Series,
    dates: pd.Series,
    top_pct: float = 0.20,
    bottom_pct: float = 0.40,
) -> pd.Series:
    """Cross-sectional rank-based labeling.

    For each date:
      - Rank all stocks by forward_return_20d
      - Top 20% = label 1 (winner)
      - Bottom 40% = label 0 (loser)
      - Middle 40% = NaN (ambiguous, skip)

    This is better than absolute thresholds because:
    1. Class balance is consistent across market regimes
    2. In bear markets, "winners" are stocks that fall least
    3. Makes the model a STOCK PICKER, not a market timer
    """
    labels = pd.Series(np.nan, index=forward_returns.index)

    for dt in dates.unique():
        mask = dates == dt
        rets = forward_returns[mask].dropna()
        if len(rets) < 20:
            continue
        top_thresh = rets.quantile(1 - top_pct)
        bottom_thresh = rets.quantile(bottom_pct)
        labels.loc[mask & (forward_returns >= top_thresh)] = 1
        labels.loc[mask & (forward_returns <= bottom_thresh)] = 0

    return labels
