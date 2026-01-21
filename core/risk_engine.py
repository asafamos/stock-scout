from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

from core.interfaces import (
    Action,
    TickerFeatures,
    ModelOutput,
    TradeDecision,
)


class RiskEngine:
    """Rules-based risk checks to override ML decisions when safety criteria fail."""

    def __init__(self) -> None:
        pass

    def evaluate(self, features: TickerFeatures, model_output: ModelOutput) -> TradeDecision:
        """Apply risk rules and produce a TradeDecision.

        Rules:
        1. Hard Rejection: Low Liquidity if volume_avg (fallback to volume) < 200,000
        2. Hard Rejection: Earnings Risk if days_to_earnings < 3
        3. Penalty: High Volatility if atr_pct_raw > 0.05 -> conviction -20, add penalty tag
        4. Sizing: conviction (0-100) derived from model_output.prediction_prob minus penalties
        """
        rm: Dict[str, Any] = features.risk_metadata or {}

        # Extract liquidity and earnings metadata with sensible fallbacks
        volume_avg = rm.get("volume_avg")
        if volume_avg is None:
            volume_avg = rm.get("volume")  # fallback to current volume
        try:
            vol_val = float(volume_avg) if volume_avg is not None else 0.0
        except Exception:
            vol_val = 0.0

        days_to_earnings = rm.get("days_to_earnings")
        try:
            dte_val = float(days_to_earnings) if days_to_earnings is not None else 999.0
        except Exception:
            dte_val = 999.0

        atr_pct_raw = rm.get("atr_pct_raw")
        try:
            atr_val = float(atr_pct_raw) if atr_pct_raw is not None else 0.0
        except Exception:
            atr_val = 0.0

        risk_penalties: List[str] = []
        active_filters: List[str] = []

        # Hard rejections
        if vol_val < 200_000:
            return TradeDecision(
                ticker=features.ticker,
                action=Action.REJECT,
                quantity=0,
                limit_price=None,
                stop_loss_price=0.0,
                target_price=0.0,
                conviction=0.0,
                estimated_commission=0.0,
                primary_reason="Low Liquidity",
                active_filters=["liquidity_check"],
                risk_penalties=[],
                explain_id=None,
            )

        if dte_val < 3:
            return TradeDecision(
                ticker=features.ticker,
                action=Action.REJECT,
                quantity=0,
                limit_price=None,
                stop_loss_price=0.0,
                target_price=0.0,
                conviction=0.0,
                estimated_commission=0.0,
                primary_reason="Earnings Risk",
                active_filters=["earnings_check"],
                risk_penalties=[],
                explain_id=None,
            )

        # Volatility penalty
        conviction_penalty = 0.0
        if atr_val > 0.05:
            conviction_penalty += 20.0
            risk_penalties.append("High Volatility")
            active_filters.append("volatility_penalty")

        # Base conviction from model prediction probability
        base_conviction = float(model_output.prediction_prob) * 100.0
        conviction = max(0.0, min(100.0, base_conviction - conviction_penalty))

        # Simple sizing: quantity scaled by conviction
        quantity = max(1, int(conviction // 10))

        decision = TradeDecision(
            ticker=features.ticker,
            action=Action.BUY,  # default BUY if not rejected; sizing via conviction
            quantity=quantity,
            limit_price=None,
            stop_loss_price=0.0,
            target_price=0.0,
            conviction=conviction,
            estimated_commission=0.0,
            primary_reason="Model-driven",
            active_filters=active_filters,
            risk_penalties=risk_penalties,
            explain_id=None,
        )
        return decision
