from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_smart_targets(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Compute risk-adjusted future targets for ML training.

    Adds the following columns to the returned DataFrame:
    - target_log_return: ln(Close_{t+h} / Close_t)
    - future_volatility: std of log returns over the forward window (t+1..t+h)
    - forward_sharpe: target_log_return / (future_volatility + 1e-6)
    - target_class: 2 (Gold), 1 (Silver), 0 (Noise)

    Notes:
    - Uses shift(-horizon) to align future values to the current index.
    - Leaves NaNs at the tail where future data is unavailable.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    out = df.copy()

    # Future log return over the horizon
    out["target_log_return"] = np.log(out["Close"].shift(-horizon) / out["Close"])  # ln(C_{t+h}/C_t)

    # Realized future volatility during the forward window (true forward window)
    # r_t = ln(C_t / C_{t-1})
    r = np.log(out["Close"]).diff()
    # Build forward window matrix: columns are r.shift(-1), ..., r.shift(-H)
    fw = pd.concat([r.shift(-k) for k in range(1, horizon + 1)], axis=1)
    # Std across the forward horizon at each t (population std, ddof=0)
    out["future_volatility"] = fw.std(axis=1, ddof=0)
    # Enforce NaN for the last H rows (incomplete forward windows)
    if horizon > 0:
        out.loc[out.index[-horizon:], "future_volatility"] = np.nan

    # Risk-adjusted target via forward Sharpe
    eps = 1e-6
    # Forward Sharpe defined using raw forward return over H divided by forward volatility
    # Keep small epsilon to avoid division by zero in perfectly constant-growth windows
    out["forward_sharpe"] = (out["Close"].shift(-horizon) / out["Close"] - 1.0) / (out["future_volatility"] + eps)

    # Raw forward return (not log), for Silver class rule
    out["raw_forward_return"] = out["Close"].shift(-horizon) / out["Close"] - 1.0

    # Classification labels
    # Class 2 (Gold): Sharpe > 1.5
    # Class 1 (Silver): Raw Return > 3% AND not Class 2
    # Class 0: Everything else
    cond_gold = out["forward_sharpe"] > 1.5
    cond_silver = (out["raw_forward_return"] > 0.03) & (~cond_gold)

    out["target_class"] = np.where(cond_gold, 2, np.where(cond_silver, 1, 0))

    return out
