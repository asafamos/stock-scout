"""Performance attribution — ablation study for scoring components.

Runs the backtest multiple times, each time neutralising one scoring
component, to determine which parts of the system contribute real alpha.

Variants tested:
  1. Full system (baseline)
  2. No ML (ml_prob = 0.5 for all)
  3. No fundamentals (fund_score = 50.0 for all)
  4. No patterns (pattern_score = 0)
  5. Random selection (same K, random picks)

Also provides:
  - Component correlation matrix (score vs actual return)
  - Regime breakdown (performance by market regime)
  - Sector breakdown (performance by sector)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.backtest.engine import FullPipelineBacktest
from core.backtest.result import BacktestResult

logger = logging.getLogger("stock_scout.backtest.attribution")


class PerformanceAttribution:
    """Determine which scoring components contribute alpha."""

    def __init__(self, base_config: Dict[str, Any]):
        """Args: base_config — kwargs for FullPipelineBacktest."""
        self._config = base_config

    def run_ablation_study(
        self,
        status_callback=None,
    ) -> Dict[str, BacktestResult]:
        """Run backtest with different configurations.

        Returns: dict mapping variant name → BacktestResult.
        """
        cb = status_callback or (lambda _: None)
        variants: Dict[str, Dict[str, Any]] = {
            "full_system": {},
            "no_ml": {"enable_ml": False},
            "no_fundamentals": {"enable_fundamentals": False},
            "no_patterns": {"enable_patterns": False},
            "no_ml_no_patterns": {"enable_ml": False, "enable_patterns": False},
        }

        results = {}
        for name, overrides in variants.items():
            cb(f"Running variant: {name}")
            logger.info("Ablation: running variant '%s'", name)

            config = {**self._config, **overrides}
            engine = FullPipelineBacktest(**config)
            try:
                result = engine.run()
                results[name] = result
                logger.info(
                    "  %s: return=%.1f%%, sharpe=%.2f, trades=%d",
                    name, result.total_return, result.sharpe, result.n_trades,
                )
            except Exception as e:
                logger.error("Ablation '%s' failed: %s", name, e)

        return results

    @staticmethod
    def compute_component_correlations(trade_log: pd.DataFrame) -> pd.DataFrame:
        """Correlation matrix: each score component vs actual return.

        Args:
            trade_log: From BacktestResult.trade_log with columns
                [return_pct, final_score, tech_score, fundamental_score, ml_prob].
        """
        score_cols = [
            "final_score", "tech_score", "fundamental_score", "ml_prob",
        ]
        available = [c for c in score_cols if c in trade_log.columns]
        if not available or "return_pct" not in trade_log.columns:
            return pd.DataFrame()

        cols = available + ["return_pct"]
        return trade_log[cols].corr()

    @staticmethod
    def regime_breakdown(trade_log: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Performance metrics split by market regime."""
        if trade_log.empty or "market_regime" not in trade_log.columns:
            return {}

        breakdown = {}
        for regime, grp in trade_log.groupby("market_regime"):
            if not regime or regime == "nan":
                continue
            rets = grp["return_pct"]
            breakdown[str(regime)] = {
                "n_trades": len(grp),
                "avg_return": float(rets.mean()),
                "median_return": float(rets.median()),
                "win_rate": float((rets > 0).mean()) * 100,
                "avg_final_score": float(grp["final_score"].mean()),
            }
        return breakdown

    @staticmethod
    def sector_breakdown(trade_log: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Performance metrics split by sector."""
        if trade_log.empty or "sector" not in trade_log.columns:
            return {}

        breakdown = {}
        for sector, grp in trade_log.groupby("sector"):
            if not sector or sector == "nan":
                continue
            rets = grp["return_pct"]
            breakdown[str(sector)] = {
                "n_trades": len(grp),
                "avg_return": float(rets.mean()),
                "win_rate": float((rets > 0).mean()) * 100,
            }
        return breakdown

    @staticmethod
    def score_quintile_analysis(trade_log: pd.DataFrame) -> pd.DataFrame:
        """Break trades into score quintiles and show performance per quintile.

        Helps determine if higher scores → better returns (monotonic relationship).
        """
        if trade_log.empty or "final_score" not in trade_log.columns:
            return pd.DataFrame()

        tl = trade_log.dropna(subset=["final_score", "return_pct"])
        if len(tl) < 10:
            return pd.DataFrame()

        tl = tl.copy()
        tl["quintile"] = pd.qcut(
            tl["final_score"], q=5, labels=False, duplicates="drop"
        )

        rows = []
        for q, grp in tl.groupby("quintile"):
            rets = grp["return_pct"]
            rows.append({
                "quintile": int(q),
                "n_trades": len(grp),
                "avg_score": float(grp["final_score"].mean()),
                "avg_return": float(rets.mean()),
                "median_return": float(rets.median()),
                "win_rate": float((rets > 0).mean()) * 100,
                "avg_win": float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0,
                "avg_loss": float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def ablation_comparison_table(results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Create a comparison table across ablation variants."""
        rows = []
        for name, r in results.items():
            rows.append({
                "variant": name,
                "total_return": r.total_return,
                "cagr": r.cagr,
                "sharpe": r.sharpe,
                "max_dd": r.max_dd,
                "win_rate": r.win_rate_pct,
                "n_trades": r.n_trades,
                "excess_return": r.excess_return,
                "p_value": r.p_value,
            })
        return pd.DataFrame(rows).set_index("variant")
