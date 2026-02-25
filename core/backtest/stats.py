"""Statistical significance testing for backtest results.

Provides Sharpe, drawdown, bootstrap CIs, and significance tests
to determine whether strategy performance is real or noise.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Core metrics
# ------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: float = 252,
) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Per-period (e.g. daily or per-trade) returns as fractions.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Trading days (252) or trades-per-year.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if std < 1e-10:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def max_drawdown(equity_curve: pd.Series) -> Tuple[float, Optional[date], Optional[date]]:
    """Maximum drawdown with peak/trough dates.

    Args:
        equity_curve: Indexed by date, values = portfolio equity.

    Returns:
        (max_dd_pct, peak_date, trough_date)  — dd is negative.
    """
    if equity_curve.empty:
        return 0.0, None, None
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    trough_idx = drawdown.idxmin()
    peak_idx = equity_curve.loc[:trough_idx].idxmax()
    dd_val = float(drawdown.min())
    p_date = peak_idx.date() if hasattr(peak_idx, "date") else peak_idx
    t_date = trough_idx.date() if hasattr(trough_idx, "date") else trough_idx
    return dd_val, p_date, t_date


def calmar_ratio(
    annualised_return: float,
    max_dd: float,
) -> float:
    """Annualised return / abs(max drawdown)."""
    if max_dd == 0:
        return 0.0
    return annualised_return / abs(max_dd)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float = 252,
) -> float:
    """Risk-adjusted excess return (annualised)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - benchmark_returns
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def win_rate(returns: pd.Series) -> float:
    """Fraction of positive returns."""
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Sum of wins / abs(sum of losses)."""
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def annualised_return(equity_curve: pd.Series) -> float:
    """CAGR from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    start, end = equity_curve.iloc[0], equity_curve.iloc[-1]
    if start <= 0:
        return 0.0
    idx = equity_curve.index
    if hasattr(idx[0], "date"):
        days = (idx[-1] - idx[0]).days
    else:
        days = len(equity_curve)
    if days <= 0:
        return 0.0
    years = days / 365.25
    return float((end / start) ** (1.0 / years) - 1)


def annualised_volatility(returns: pd.Series, periods_per_year: float = 252) -> float:
    """Annualised standard deviation of returns."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


# ------------------------------------------------------------------
# Bootstrap confidence intervals
# ------------------------------------------------------------------

def bootstrap_confidence_interval(
    returns: np.ndarray,
    n_samples: int = 10_000,
    ci: float = 0.95,
    statistic: str = "mean",
) -> Tuple[float, float]:
    """Bootstrap CI for a statistic of returns.

    Args:
        returns: 1-D array of returns.
        n_samples: Bootstrap iterations.
        ci: Confidence level (e.g. 0.95).
        statistic: "mean" or "median".

    Returns:
        (lower, upper) bounds.
    """
    if len(returns) < 2:
        m = float(np.mean(returns)) if len(returns) else 0.0
        return m, m
    rng = np.random.default_rng(42)
    fn = np.mean if statistic == "mean" else np.median
    boot = np.array(
        [fn(rng.choice(returns, size=len(returns), replace=True)) for _ in range(n_samples)]
    )
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100))


# ------------------------------------------------------------------
# Significance testing
# ------------------------------------------------------------------

def significance_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    method: str = "bootstrap",
    n_bootstrap: int = 10_000,
) -> Dict:
    """Test if strategy significantly outperforms benchmark.

    Returns dict with p_value, t_stat, ci, is_significant.
    """
    excess = (strategy_returns - benchmark_returns).dropna()
    if len(excess) < 5:
        return {
            "p_value": 1.0,
            "t_stat": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "is_significant": False,
            "n_obs": len(excess),
        }

    from scipy import stats as sp_stats

    t_stat, p_value = sp_stats.ttest_1samp(excess, 0)
    ci_lo, ci_hi = bootstrap_confidence_interval(excess.values, n_bootstrap)

    return {
        "p_value": float(p_value),
        "t_stat": float(t_stat),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "is_significant": float(p_value) < 0.05 and float(t_stat) > 0,
        "n_obs": len(excess),
    }
