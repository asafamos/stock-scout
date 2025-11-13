from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any

def backtest(
    close_prices: pd.DataFrame,  # index=dates, columns=tickers
    meta: pd.DataFrame,          # index=ticker -> sector, market_cap
    signals_fn: Callable[[pd.Timestamp], pd.DataFrame],
    rebalance: str = "M",
    slippage_bps: float = 0.001,
    commission_per_trade: float = 0.0,
    initial_cash: float = 100_000.0,
) -> Dict[str, Any]:
    """
    Simple rebalancing backtest skeleton. signals_fn(date) should return a DataFrame
    indexed by ticker with a 'dollar_target' column (or share targets).
    """
    dates = pd.to_datetime(close_prices.index)
    if rebalance.upper().startswith("M"):
        rebalance_dates = pd.to_datetime(dates.to_series().resample("M").last().index)
    else:
        rebalance_dates = dates

    cash = initial_cash
    holdings = pd.Series(0, index=close_prices.columns, dtype=float)
    equity_curve = []

    for dt in rebalance_dates:
        if dt not in close_prices.index:
            continue
        prices = close_prices.loc[dt].reindex(holdings.index)
        signals = signals_fn(dt)
        if signals is None or signals.empty:
            equity = cash + (holdings * prices).sum()
            equity_curve.append((dt, equity))
            continue

        targets = signals.get("dollar_target")
        if targets is None:
            equity = cash + (holdings * prices).sum()
            equity_curve.append((dt, equity))
            continue

        # compute desired shares (floor)
        desired_shares = (targets / prices).fillna(0).apply(np.floor).astype(int)

        # compute trades and costs (simple slippage model)
        trades = desired_shares - holdings.astype(int)
        trade_cost = (trades.abs() * prices * (1 + slippage_bps)).sum() + (trades.astype(int) != 0).sum() * commission_per_trade

        # settle trades: assume enough cash, else scale down proportionally (simple)
        total_cost = (desired_shares * prices).sum()
        if total_cost > cash + (holdings * prices).sum():
            factor = (cash + (holdings * prices).sum()) / total_cost if total_cost > 0 else 0.0
            desired_shares = (desired_shares * factor).apply(np.floor).astype(int)

        # update holdings and cash
        trades = desired_shares - holdings.astype(int)
        cash -= (trades * prices * (1 + slippage_bps)).sum()
        cash -= (trades.astype(int) != 0).sum() * commission_per_trade
        holdings = desired_shares.astype(float)

        equity = cash + (holdings * prices).sum()
        equity_curve.append((dt, equity))

    eq = pd.Series({d: v for d, v in equity_curve})
    returns = eq.pct_change().dropna()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252.0 / len(returns)) - 1 if len(returns) > 0 else 0.0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else float("nan")
    max_dd = (eq / eq.cummax() - 1).min()
    turnover = np.nan  # can be computed from trades history if recorded
    return {"equity": eq, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "turnover": turnover}