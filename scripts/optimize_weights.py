#!/usr/bin/env python3
"""Data-driven CONVICTION_WEIGHTS optimization via walk-forward backtest.

Approach:
1. Run ONE full pipeline pass to score all stocks at each rebalance date.
2. Re-weight component scores with trial weights and re-select top-K.
3. Simulate trades using actual price data.
4. Optimize Sharpe ratio on training period, validate on test period.

Usage::

    python scripts/optimize_weights.py
    python scripts/optimize_weights.py --train-end 2025-06-30 --test-end 2026-03-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("optimize_weights")


# ---------------------------------------------------------------------------
# Score recomputation with custom weights
# ---------------------------------------------------------------------------

def _recompute_final_score(row: pd.Series, weights: Dict[str, float]) -> float:
    """Recompute FinalScore_20d with custom weights.

    Mirrors compute_final_score_20d from core/scoring_engine.py but uses
    the provided weights dict instead of CONVICTION_WEIGHTS.  All post-
    adjustments (ML, patterns, timing, regime) are applied identically.
    """
    from core.scoring_engine import (
        _safe_score,
        evaluate_rr_unified,
        ml_boost_component,
    )
    from core.scoring_config import (
        BONUS_CONFIG,
        REGIME_MULTIPLIERS,
    )

    try:
        fund = _safe_score(row, "Fundamental_S", "FundamentalScore", "Fundamental_Score", "fund_score")
        mom = _safe_score(row, "TechScore_20d_raw", "MomentumScore", "TechScore_20d", "tech_score")
        rel = _safe_score(row, "Reliability_Score", "ReliabilityScore", "Reliability_v2", "reliability_pct")

        # Coil amplifier
        try:
            coil_active = bool(row.get("Coil_Bonus", 0)) or str(row.get("Coil_Bonus", "0")) in ("1", "True")
            if coil_active:
                mom = float(mom) * BONUS_CONFIG.get("coil_amplifier", 1.05)
        except Exception:
            pass

        rr_ratio = row.get("RR", None)
        rr_score, _, _ = evaluate_rr_unified(rr_ratio) if rr_ratio is not None else (50.0, 0.0, "N/A")

        base = (
            weights["fundamental"] * fund
            + weights["momentum"] * mom
            + weights["risk_reward"] * rr_score
            + weights["reliability"] * rel
        )

        # ML adjustment (same as scoring_engine.py)
        ml_prob = row.get("ML_20d_Prob", None)
        delta = ml_boost_component(ml_prob)
        try:
            from core.ml_20d_inference import get_ml_weight_multiplier
            ml_mult = get_ml_weight_multiplier()
            if ml_mult < 0.5:
                delta = 0.0
            else:
                delta *= ml_mult
        except Exception:
            pass
        reliability = float(row.get("ReliabilityScore", row.get("Reliability_Score", 50.0)))
        if reliability < BONUS_CONFIG.get("reliability_low_threshold", 30):
            delta *= BONUS_CONFIG.get("reliability_low_ml_mult", 0.3)
        elif reliability < BONUS_CONFIG.get("reliability_med_threshold", 50):
            delta *= BONUS_CONFIG.get("reliability_med_ml_mult", 0.6)

        # VCP/pattern/BW bonuses (same as scoring_engine.py)
        bonus = 0.0
        try:
            vcp = float(row.get("Volatility_Contraction_Score", 0.0))
            vcp = vcp if np.isfinite(vcp) else 0.0
            tight_ratio = row.get("Tightness_Ratio", np.nan)
            if vcp > 0:
                bonus += min(BONUS_CONFIG.get("vcp_bonus_max", 3.0), BONUS_CONFIG.get("vcp_multiplier", 3.0) * vcp)
            if np.isfinite(tight_ratio) and tight_ratio < BONUS_CONFIG.get("tightness_ratio_threshold", 0.03):
                bonus += BONUS_CONFIG.get("tightness_bonus", 2.0)
            bonus = float(np.clip(bonus, 0.0, BONUS_CONFIG.get("vcp_tightness_cap", 4.0)))
        except Exception:
            pass
        try:
            patt_score = float(row.get("Pattern_Score", 0.0)) if pd.notna(row.get("Pattern_Score")) else 0.0
            if patt_score > 0:
                bonus += min(BONUS_CONFIG.get("pattern_bonus_max", 5.0), patt_score * BONUS_CONFIG.get("pattern_multiplier", 0.3))
        except Exception:
            pass
        try:
            bw_signal = float(row.get("Big_Winner_Signal", 0.0)) if pd.notna(row.get("Big_Winner_Signal")) else 0.0
            if bw_signal > 0:
                bonus += min(BONUS_CONFIG.get("big_winner_bonus_max", 4.0), bw_signal * BONUS_CONFIG.get("big_winner_multiplier", 0.5))
        except Exception:
            pass
        bonus = float(np.clip(bonus, 0.0, BONUS_CONFIG.get("total_bonus_cap", 10.0)))

        score = base + delta + bonus

        # Hunter floor
        try:
            hunter_min = BONUS_CONFIG.get("hunter_floor", 45.0)
            rr_raw = float(row.get("RR", 0)) if pd.notna(row.get("RR")) else 0.0
            coil_active = bool(row.get("Coil_Bonus", 0))
            vcp_raw = float(row.get("Volatility_Contraction_Score", 0.0))
            if (coil_active or vcp_raw > 0.3) and rr_raw >= 1.0:
                score = max(score, hunter_min)
        except Exception:
            pass

        # Entry timing
        try:
            dist_52w = float(row.get("Dist_52w_High", np.nan))
            if np.isfinite(dist_52w):
                if dist_52w < 0.05:
                    score += 3.0
                elif dist_52w < 0.10:
                    score += 1.5
                pullback = float(row.get("Pullback_From_High_20d", np.nan))
                if np.isfinite(pullback) and 0.03 <= pullback <= 0.12:
                    score += 2.0
        except Exception:
            pass

        # RSI timing (simplified)
        try:
            rsi = float(row.get("RSI_14", np.nan))
            if np.isfinite(rsi):
                regime = str(row.get("Market_Regime", "neutral")).lower()
                if regime in ("bullish", "trend_up", "moderate_up"):
                    if 40 <= rsi <= 60:
                        score += 2.0
                    elif rsi > 75:
                        score -= 3.0
                else:
                    if 30 <= rsi <= 50:
                        score += 3.0
                    elif rsi > 70:
                        score -= 5.0
        except Exception:
            pass

        # RR hard caps
        try:
            rr_raw = float(row.get("RR", np.nan))
            if np.isfinite(rr_raw):
                if rr_raw < 1.0:
                    score = min(score, 55.0)
                elif rr_raw < 1.5:
                    score = min(score, 70.0)
            if rr_score < 50:
                score = min(score, 75.0)
        except Exception:
            pass

        # Market regime multiplier
        try:
            regime = str(row.get("Market_Regime", "")).upper()
            mult = REGIME_MULTIPLIERS.get(regime, 1.0)
            score *= mult
        except Exception:
            pass

        return float(np.clip(score, 0.0, 100.0))
    except Exception:
        return 50.0


# ---------------------------------------------------------------------------
# Simulate trades with recomputed scores
# ---------------------------------------------------------------------------

def _simulate_with_weights(
    scored_universes: Dict[date, pd.DataFrame],
    price_cache: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    top_k: int = 10,
    holding_days: int = 20,
) -> pd.DataFrame:
    """Simulate trades by re-scoring with custom weights and selecting top-K.

    Returns a trade log DataFrame with return_pct per trade.
    """
    trades: List[Dict[str, Any]] = []

    for rebal_date, universe_df in sorted(scored_universes.items()):
        # Recompute scores with new weights
        new_scores = universe_df.apply(
            lambda row: _recompute_final_score(row, weights), axis=1
        )
        universe_df = universe_df.copy()
        universe_df["_opt_score"] = new_scores

        # Select top-K by new score
        top = universe_df.nlargest(top_k, "_opt_score")

        for _, row in top.iterrows():
            ticker = str(row.get("Ticker", ""))
            if not ticker or ticker not in price_cache:
                continue

            hist = price_cache[ticker]
            # Find entry price on rebal_date
            entry_mask = hist.index.date >= rebal_date
            entry_df = hist[entry_mask]
            if entry_df.empty:
                continue
            entry_price = float(entry_df["Close"].iloc[0])
            entry_date = entry_df.index[0].date() if hasattr(entry_df.index[0], "date") else entry_df.index[0]

            # Find exit
            stop = float(row.get("Stop_Loss", entry_price * 0.92))
            target = float(row.get("Target_Price", entry_price * 1.15))
            hd = int(row.get("Holding_Days", holding_days))

            # Walk forward through prices
            exit_price = entry_price
            exit_date = entry_date
            days_held = 0

            for idx_dt, px_row in entry_df.iloc[1:].iterrows():
                days_held += 1
                px_date = idx_dt.date() if hasattr(idx_dt, "date") else idx_dt
                close = float(px_row["Close"])
                low = float(px_row.get("Low", close))
                high = float(px_row.get("High", close))

                # Stop hit
                if low <= stop:
                    exit_price = stop
                    exit_date = px_date
                    break
                # Target hit
                if high >= target:
                    exit_price = target
                    exit_date = px_date
                    break
                # Holding period expired
                if days_held >= hd:
                    exit_price = close
                    exit_date = px_date
                    break

                exit_price = close
                exit_date = px_date

            ret = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
            trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": ret,
                "holding_days": days_held,
                "final_score": float(row.get("_opt_score", 0)),
                "market_regime": str(row.get("Market_Regime", "")),
            })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def _compute_sharpe(trade_log: pd.DataFrame) -> float:
    """Compute Sharpe ratio from trade returns."""
    if trade_log.empty or len(trade_log) < 5:
        return -10.0
    rets = trade_log["return_pct"].values
    mean_r = np.mean(rets)
    std_r = np.std(rets, ddof=1)
    if std_r < 1e-8:
        return 0.0
    # Annualize: ~12 rebalances/year with monthly
    return float(mean_r / std_r * np.sqrt(12))


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def _weights_from_vector(x: np.ndarray) -> Dict[str, float]:
    """Convert 3-element vector to 4 normalized weights.

    We optimize 3 free params; the 4th = 1 - sum(others).
    """
    # Softmax-like normalization to ensure [0, 1] and sum=1
    x_clipped = np.clip(x, 0.05, 0.60)
    w4 = max(0.05, 1.0 - x_clipped.sum())
    if w4 > 0.60:
        # Re-scale
        total = x_clipped.sum() + w4
        x_clipped = x_clipped / total
        w4 = w4 / total
    names = ["fundamental", "momentum", "risk_reward", "reliability"]
    vals = list(x_clipped) + [w4]
    # Normalize to exactly sum to 1
    total = sum(vals)
    return {names[i]: vals[i] / total for i in range(4)}


def _objective(
    x: np.ndarray,
    scored_universes: Dict[date, pd.DataFrame],
    price_cache: Dict[str, pd.DataFrame],
    top_k: int,
    holding_days: int,
) -> float:
    """Negative Sharpe (minimize)."""
    weights = _weights_from_vector(x)
    tl = _simulate_with_weights(
        scored_universes, price_cache, weights, top_k, holding_days
    )
    sharpe = _compute_sharpe(tl)
    return -sharpe


def run_optimization(
    train_start: str = "2024-01-01",
    train_end: str = "2025-06-30",
    test_start: str = "2025-07-01",
    test_end: str = "2026-03-01",
    top_k: int = 10,
    holding_days: int = 20,
    max_iter: int = 60,
) -> Dict[str, Any]:
    """Run walk-forward weight optimization.

    1. Run full pipeline on training period to get scored universes.
    2. Optimize weights via Nelder-Mead on training Sharpe.
    3. Validate on test period.
    """
    from scipy.optimize import minimize
    from core.backtest.engine import FullPipelineBacktest
    from core.scoring_config import CONVICTION_WEIGHTS

    original_weights = dict(CONVICTION_WEIGHTS)
    logger.info("Original weights: %s", original_weights)

    # --- Phase 1: Run full pipeline on FULL period to get scored universes ---
    logger.info("Phase 1: Running full pipeline %s → %s ...", train_start, test_end)
    engine = FullPipelineBacktest(
        start_date=train_start,
        end_date=test_end,
        top_k=top_k,
        holding_days=holding_days,
        initial_capital=100_000,
        status_callback=lambda msg: logger.info("[ENGINE] %s", msg),
    )
    engine.run()

    if not engine._scored_universes:
        logger.error("No scored universes — cannot optimize")
        return {"error": "No scored universes"}

    price_cache = engine._price_cache
    all_universes = engine._scored_universes
    logger.info("Got %d rebalance dates with scored universes", len(all_universes))

    # Split into train/test
    train_end_date = date.fromisoformat(train_end)
    test_start_date = date.fromisoformat(test_start)

    train_universes = {d: df for d, df in all_universes.items() if d <= train_end_date}
    test_universes = {d: df for d, df in all_universes.items() if d >= test_start_date}
    logger.info("Train dates: %d, Test dates: %d", len(train_universes), len(test_universes))

    if len(train_universes) < 3:
        logger.error("Too few training dates (%d)", len(train_universes))
        return {"error": "Too few training dates"}

    # --- Phase 2: Baseline Sharpe with original weights ---
    train_tl_orig = _simulate_with_weights(
        train_universes, price_cache, original_weights, top_k, holding_days
    )
    train_sharpe_orig = _compute_sharpe(train_tl_orig)
    logger.info(
        "Baseline train: Sharpe=%.3f, trades=%d, winrate=%.1f%%",
        train_sharpe_orig,
        len(train_tl_orig),
        (train_tl_orig["return_pct"] > 0).mean() * 100 if len(train_tl_orig) > 0 else 0,
    )

    # --- Phase 3: Optimize ---
    logger.info("Phase 3: Optimizing weights (max_iter=%d)...", max_iter)
    x0 = np.array([
        original_weights["fundamental"],
        original_weights["momentum"],
        original_weights["risk_reward"],
    ])

    eval_count = [0]
    best_sharpe = [train_sharpe_orig]

    def callback(xk):
        eval_count[0] += 1
        w = _weights_from_vector(xk)
        tl = _simulate_with_weights(train_universes, price_cache, w, top_k, holding_days)
        sh = _compute_sharpe(tl)
        if sh > best_sharpe[0]:
            best_sharpe[0] = sh
            logger.info(
                "Iter %d: NEW BEST Sharpe=%.3f weights=%s",
                eval_count[0], sh,
                {k: round(v, 3) for k, v in w.items()},
            )

    result = minimize(
        _objective,
        x0,
        args=(train_universes, price_cache, top_k, holding_days),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 0.01, "fatol": 0.01},
        callback=callback,
    )

    optimized_weights = _weights_from_vector(result.x)
    logger.info("Optimized weights: %s", {k: round(v, 3) for k, v in optimized_weights.items()})

    # --- Phase 4: Evaluate on both periods ---
    # Training period
    train_tl_opt = _simulate_with_weights(
        train_universes, price_cache, optimized_weights, top_k, holding_days
    )
    train_sharpe_opt = _compute_sharpe(train_tl_opt)

    # Test period
    test_tl_orig = _simulate_with_weights(
        test_universes, price_cache, original_weights, top_k, holding_days
    )
    test_sharpe_orig = _compute_sharpe(test_tl_orig)

    test_tl_opt = _simulate_with_weights(
        test_universes, price_cache, optimized_weights, top_k, holding_days
    )
    test_sharpe_opt = _compute_sharpe(test_tl_opt)

    # --- Results ---
    results = {
        "original_weights": original_weights,
        "optimized_weights": {k: round(v, 4) for k, v in optimized_weights.items()},
        "train_period": f"{train_start} → {train_end}",
        "test_period": f"{test_start} → {test_end}",
        "train": {
            "original_sharpe": round(train_sharpe_orig, 3),
            "optimized_sharpe": round(train_sharpe_opt, 3),
            "improvement": round(train_sharpe_opt - train_sharpe_orig, 3),
            "original_trades": len(train_tl_orig),
            "optimized_trades": len(train_tl_opt),
            "original_winrate": round((train_tl_orig["return_pct"] > 0).mean() * 100, 1) if len(train_tl_orig) > 0 else 0,
            "optimized_winrate": round((train_tl_opt["return_pct"] > 0).mean() * 100, 1) if len(train_tl_opt) > 0 else 0,
            "original_avg_return": round(train_tl_orig["return_pct"].mean() * 100, 2) if len(train_tl_orig) > 0 else 0,
            "optimized_avg_return": round(train_tl_opt["return_pct"].mean() * 100, 2) if len(train_tl_opt) > 0 else 0,
        },
        "test": {
            "original_sharpe": round(test_sharpe_orig, 3),
            "optimized_sharpe": round(test_sharpe_opt, 3),
            "improvement": round(test_sharpe_opt - test_sharpe_orig, 3),
            "original_trades": len(test_tl_orig),
            "optimized_trades": len(test_tl_opt),
            "original_winrate": round((test_tl_orig["return_pct"] > 0).mean() * 100, 1) if len(test_tl_orig) > 0 else 0,
            "optimized_winrate": round((test_tl_opt["return_pct"] > 0).mean() * 100, 1) if len(test_tl_opt) > 0 else 0,
            "original_avg_return": round(test_tl_orig["return_pct"].mean() * 100, 2) if len(test_tl_orig) > 0 else 0,
            "optimized_avg_return": round(test_tl_opt["return_pct"].mean() * 100, 2) if len(test_tl_opt) > 0 else 0,
        },
        "adopt_recommendation": "YES" if (test_sharpe_opt - test_sharpe_orig) > 0.1 else "NO",
        "optimizer_converged": bool(result.success),
        "optimizer_iterations": int(result.nit),
    }

    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info("Original weights: %s", original_weights)
    logger.info("Optimized weights: %s", results["optimized_weights"])
    logger.info("")
    logger.info("TRAIN (%s):", results["train_period"])
    logger.info("  Original:  Sharpe=%.3f, WinRate=%.1f%%, AvgRet=%.2f%%",
                results["train"]["original_sharpe"],
                results["train"]["original_winrate"],
                results["train"]["original_avg_return"])
    logger.info("  Optimized: Sharpe=%.3f, WinRate=%.1f%%, AvgRet=%.2f%%",
                results["train"]["optimized_sharpe"],
                results["train"]["optimized_winrate"],
                results["train"]["optimized_avg_return"])
    logger.info("")
    logger.info("TEST (%s) — out-of-sample validation:", results["test_period"])
    logger.info("  Original:  Sharpe=%.3f, WinRate=%.1f%%, AvgRet=%.2f%%",
                results["test"]["original_sharpe"],
                results["test"]["original_winrate"],
                results["test"]["original_avg_return"])
    logger.info("  Optimized: Sharpe=%.3f, WinRate=%.1f%%, AvgRet=%.2f%%",
                results["test"]["optimized_sharpe"],
                results["test"]["optimized_winrate"],
                results["test"]["optimized_avg_return"])
    logger.info("")
    logger.info("Adopt recommendation: %s (threshold: test improvement > 0.1 Sharpe)",
                results["adopt_recommendation"])
    logger.info("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize CONVICTION_WEIGHTS")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--test-start", default="2025-07-01")
    parser.add_argument("--test-end", default="2026-03-01")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--holding-days", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--output", default="reports/weight_optimization.json")
    args = parser.parse_args()

    results = run_optimization(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        top_k=args.top_k,
        holding_days=args.holding_days,
        max_iter=args.max_iter,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
