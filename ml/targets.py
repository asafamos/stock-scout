from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Optional


def compute_smart_targets(
    df: pd.DataFrame,
    horizon: int = 20,
    sharpe_gold_threshold: float = 1.5,
    return_silver_threshold: float = 0.03,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Compute forward-looking targets for ML training.
    
    This function calculates various target metrics for each row based on
    the price movement over the next `horizon` periods.
    
    Target Classes:
        - 2 (Gold): High forward Sharpe ratio (>1.5) - quality moves with low volatility
        - 1 (Silver): Positive return (>3%) but higher volatility
        - 0 (Noise): Flat or negative returns
    
    Output columns:
        - target_log_return: ln(Close[t+horizon] / Close[t])
        - raw_forward_return: (Close[t+horizon] - Close[t]) / Close[t]
        - future_volatility: std of log returns in forward window
        - forward_sharpe: target_log_return / future_volatility (annualized)
        - target_class: 0, 1, or 2 based on above thresholds
        - target: alias for target_class (for compatibility)
    
    Args:
        df: DataFrame with 'Close' column (and optionally Date index)
        horizon: Number of periods to look forward (default 20)
        sharpe_gold_threshold: Sharpe threshold for class 2 (default 1.5)
        return_silver_threshold: Return threshold for class 1 (default 0.03)
    
    Returns:
        DataFrame with original columns plus target columns
    """
    out = df.copy()
    n = len(out)
    
    if "Close" not in out.columns:
        raise ValueError("DataFrame must have 'Close' column")
    
    close = out["Close"].values
    
    # Initialize output columns with NaN
    out["target_log_return"] = np.nan
    out["raw_forward_return"] = np.nan
    out["future_volatility"] = np.nan
    out["forward_sharpe"] = np.nan
    out["target_class"] = np.nan
    
    # Calculate log returns for volatility calculation
    log_returns = np.log(close[1:] / close[:-1])
    
    # For each valid starting point, compute forward targets
    for i in range(n - horizon):
        # Forward prices
        c0 = close[i]
        ch = close[i + horizon]
        
        # Log return over horizon
        target_log_return = np.log(ch / c0)
        out.loc[out.index[i], "target_log_return"] = target_log_return
        
        # Raw percentage return
        raw_return = (ch - c0) / c0
        out.loc[out.index[i], "raw_forward_return"] = raw_return
        
        # Future volatility: std of log returns in forward window [i+1, i+horizon]
        # This is the volatility of returns WITHIN the forward window
        if i + horizon <= len(log_returns):
            window_returns = log_returns[i:i + horizon]
            # Population std (ddof=0) for consistent results
            future_vol = float(np.std(window_returns, ddof=0))
        else:
            future_vol = np.nan
        out.loc[out.index[i], "future_volatility"] = future_vol
        
        # Forward Sharpe ratio (not annualized for simplicity)
        # Sharpe = return / volatility
        if future_vol > 1e-10:
            forward_sharpe = target_log_return / future_vol
        else:
            # Zero volatility means infinite Sharpe if positive return
            forward_sharpe = np.inf if target_log_return > 0 else 0.0
        out.loc[out.index[i], "forward_sharpe"] = forward_sharpe
        
        # Classification
        if forward_sharpe >= sharpe_gold_threshold:
            target_class = 2  # Gold: high quality move
        elif raw_return >= return_silver_threshold:
            target_class = 1  # Silver: good return but volatile
        else:
            target_class = 0  # Noise: flat or down
        out.loc[out.index[i], "target_class"] = target_class
    
    # Alias for compatibility
    out["target"] = out["target_class"]
    
    return out
