"""
Backtest helper functions for Stock Scout.

Provides forward-return computation used by backtesting and time-validation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_forward_returns(
    df: pd.DataFrame,
    date: pd.Timestamp,
    horizons: Optional[List[int]] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute forward (future) returns from *date* — for backtesting only.

    Args:
        df: DataFrame with ``Close`` column (DatetimeIndex).
        date: Starting date.
        horizons: Forward periods in trading days (default [5, 10, 20]).
        benchmark_df: Optional benchmark DataFrame for excess return calculation.

    Returns:
        Dict with ``R_5d``, ``R_10d``, ``R_20d`` (and Excess variants if benchmark provided).
    """
    if horizons is None:
        horizons = [5, 10, 20]

    results: Dict[str, float] = {}

    if date not in df.index:
        return {f"R_{h}d": np.nan for h in horizons}

    idx = df.index.get_loc(date)
    close_val = df.loc[date, "Close"]
    price_start = float(close_val.iloc[0]) if isinstance(close_val, pd.Series) else float(close_val)

    for h in horizons:
        end_idx = idx + h
        if end_idx >= len(df):
            results[f"R_{h}d"] = np.nan
            results[f"Excess_{h}d"] = np.nan
            continue

        end_date = df.index[end_idx]
        close_end = df.loc[end_date, "Close"]
        price_end = float(close_end.iloc[0]) if isinstance(close_end, pd.Series) else float(close_end)

        ret = (price_end / price_start - 1) * 100
        results[f"R_{h}d"] = float(ret)

        if benchmark_df is not None and date in benchmark_df.index:
            bench_idx = benchmark_df.index.get_loc(date)
            if bench_idx + h < len(benchmark_df):
                bench_end_date = benchmark_df.index[bench_idx + h]
                bs = benchmark_df.loc[date, "Close"]
                be = benchmark_df.loc[bench_end_date, "Close"]
                bs = float(bs.iloc[0]) if isinstance(bs, pd.Series) else float(bs)
                be = float(be.iloc[0]) if isinstance(be, pd.Series) else float(be)
                results[f"Excess_{h}d"] = float(ret - ((be / bs - 1) * 100))
            else:
                results[f"Excess_{h}d"] = np.nan
        else:
            results[f"Excess_{h}d"] = np.nan

    return results
