"""BacktestResult — immutable container for backtest outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.backtest.stats import (
    annualised_return,
    annualised_volatility,
    bootstrap_confidence_interval,
    calmar_ratio,
    information_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    significance_test,
    win_rate,
)


@dataclass
class BacktestResult:
    """Complete output of a full-pipeline backtest run."""

    # ---- Period ----
    start_date: date
    end_date: date
    n_periods: int = 0
    n_trades: int = 0

    # ---- Returns ----
    total_return: float = 0.0
    cagr: float = 0.0
    avg_trade_return: float = 0.0
    median_trade_return: float = 0.0

    # ---- Risk ----
    sharpe: float = 0.0
    max_dd: float = 0.0
    max_dd_peak: Optional[date] = None
    max_dd_trough: Optional[date] = None
    calmar: float = 0.0
    volatility: float = 0.0

    # ---- Hit rates ----
    win_rate_pct: float = 0.0
    profit_factor_val: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # ---- Vs benchmark ----
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    information_ratio_val: float = 0.0

    # ---- Statistical ----
    p_value: float = 1.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    # ---- Data ----
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # ---- Config used ----
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trades(
        cls,
        trade_log: pd.DataFrame,
        equity_curve: pd.DataFrame,
        benchmark_equity: pd.DataFrame,
        config: Dict[str, Any],
    ) -> "BacktestResult":
        """Build BacktestResult from raw trade log and equity curve.

        Args:
            trade_log: DataFrame with columns [ticker, entry_date, exit_date,
                       entry_price, exit_price, return_pct, holding_days,
                       final_score, tech_score, fundamental_score, ml_prob,
                       market_regime, sector].
            equity_curve: DataFrame indexed by date with column 'equity'.
            benchmark_equity: DataFrame indexed by date with column 'equity'.
            config: Dict of backtest parameters.
        """
        trade_returns = trade_log["return_pct"] / 100 if not trade_log.empty else pd.Series(dtype=float)
        eq = equity_curve["equity"] if "equity" in equity_curve.columns else equity_curve.iloc[:, 0]
        bm = benchmark_equity["equity"] if "equity" in benchmark_equity.columns else benchmark_equity.iloc[:, 0]

        # Daily returns for Sharpe etc.
        daily_ret = eq.pct_change().dropna()
        bm_daily_ret = bm.pct_change().dropna()
        # Align
        common_idx = daily_ret.index.intersection(bm_daily_ret.index)
        daily_ret = daily_ret.loc[common_idx]
        bm_daily_ret = bm_daily_ret.loc[common_idx]

        total_ret = float((eq.iloc[-1] / eq.iloc[0] - 1) * 100) if len(eq) > 1 else 0.0
        bm_ret = float((bm.iloc[-1] / bm.iloc[0] - 1) * 100) if len(bm) > 1 else 0.0

        dd_val, dd_peak, dd_trough = max_drawdown(eq)
        cagr_val = annualised_return(eq)
        vol = annualised_volatility(daily_ret)
        sr = sharpe_ratio(daily_ret)
        ir = information_ratio(daily_ret, bm_daily_ret)
        cal = calmar_ratio(cagr_val, dd_val)

        wr = win_rate(trade_returns)
        pf = profit_factor(trade_returns)
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        sig = significance_test(daily_ret, bm_daily_ret)

        # Monthly returns
        monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        dates = eq.index
        start = dates[0].date() if hasattr(dates[0], "date") else dates[0]
        end = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]

        return cls(
            start_date=start,
            end_date=end,
            n_periods=len(dates),
            n_trades=len(trade_log),
            total_return=total_ret,
            cagr=cagr_val * 100,
            avg_trade_return=float(trade_returns.mean() * 100) if len(trade_returns) else 0.0,
            median_trade_return=float(trade_returns.median() * 100) if len(trade_returns) else 0.0,
            sharpe=sr,
            max_dd=dd_val * 100,
            max_dd_peak=dd_peak,
            max_dd_trough=dd_trough,
            calmar=cal,
            volatility=vol * 100,
            win_rate_pct=wr * 100,
            profit_factor_val=pf,
            avg_win=float(wins.mean() * 100) if len(wins) else 0.0,
            avg_loss=float(losses.mean() * 100) if len(losses) else 0.0,
            benchmark_return=bm_ret,
            excess_return=total_ret - bm_ret,
            information_ratio_val=ir,
            p_value=sig["p_value"],
            ci_lower=sig["ci_lower"],
            ci_upper=sig["ci_upper"],
            equity_curve=equity_curve,
            trade_log=trade_log,
            monthly_returns=monthly,
            config=config,
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Period:          {self.start_date} → {self.end_date}  ({self.n_periods} days)",
            f"Trades:          {self.n_trades}",
            "",
            "── Returns ──",
            f"Total Return:    {self.total_return:+.1f}%",
            f"CAGR:            {self.cagr:+.1f}%",
            f"Avg Trade:       {self.avg_trade_return:+.2f}%",
            f"Median Trade:    {self.median_trade_return:+.2f}%",
            "",
            "── Risk ──",
            f"Sharpe Ratio:    {self.sharpe:.2f}",
            f"Max Drawdown:    {self.max_dd:.1f}%",
            f"Calmar Ratio:    {self.calmar:.2f}",
            f"Volatility:      {self.volatility:.1f}%",
            "",
            "── Hit Rates ──",
            f"Win Rate:        {self.win_rate_pct:.1f}%",
            f"Profit Factor:   {self.profit_factor_val:.2f}",
            f"Avg Win:         {self.avg_win:+.2f}%",
            f"Avg Loss:        {self.avg_loss:+.2f}%",
            "",
            "── vs Benchmark (SPY) ──",
            f"Benchmark:       {self.benchmark_return:+.1f}%",
            f"Excess Return:   {self.excess_return:+.1f}%",
            f"Info Ratio:      {self.information_ratio_val:.2f}",
            "",
            "── Statistical ──",
            f"p-value:         {self.p_value:.4f}",
            f"95% CI:          [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Significant:     {'YES' if self.p_value < 0.05 else 'NO'}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable dict (no DataFrames)."""
        return {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "n_periods": self.n_periods,
            "n_trades": self.n_trades,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "avg_trade_return": self.avg_trade_return,
            "median_trade_return": self.median_trade_return,
            "sharpe": self.sharpe,
            "max_dd": self.max_dd,
            "calmar": self.calmar,
            "volatility": self.volatility,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor_val,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "information_ratio": self.information_ratio_val,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "config": self.config,
        }
