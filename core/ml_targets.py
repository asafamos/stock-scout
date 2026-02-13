def make_label_20d_strict(df, ret_col="Forward_Return_20d"):
    """Label: 1 if return >= 0.15, 0 if <= 0.02, else np.nan"""
    up = 0.15
    down = 0.02
    return np.where(df[ret_col] >= up, 1, np.where(df[ret_col] <= down, 0, np.nan))

def make_label_20d_soft(df, ret_col="Forward_Return_20d"):
    """Label: 1 if return >= 0.08, 0 if <= 0.00, else np.nan"""
    up = 0.08
    down = 0.00
    return np.where(df[ret_col] >= up, 1, np.where(df[ret_col] <= down, 0, np.nan))
import numpy as np
import pandas as pd
from .ml_target_config import HORIZON_DAYS, UP_THRESHOLD, DOWN_THRESHOLD

def compute_forward_return(close: pd.Series, horizon: int = HORIZON_DAYS) -> pd.Series:
    return close.shift(-horizon) / close - 1

def make_label_20d(forward_ret: pd.Series) -> pd.Series:
    label = np.where(forward_ret >= UP_THRESHOLD, 1,
                     np.where(forward_ret <= DOWN_THRESHOLD, 0, np.nan))
    return pd.Series(label, index=forward_ret.index)
