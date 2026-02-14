"""Target price / date calculation — extracted from stock_scout.py."""
from __future__ import annotations

import json
import logging
import re
import datetime
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-enhanced target prediction
# ---------------------------------------------------------------------------

def get_openai_target_prediction(
    ticker: str,
    current_price: float,
    fundamentals: dict,
    technicals: dict,
    *,
    openai_key: Optional[str] = None,
) -> Optional[Tuple[float, int]]:
    """Return ``(target_price, days_to_target)`` via OpenAI, or *None*."""
    if not openai_key:
        return None

    try:
        from openai import OpenAI  # keep import lazy

        client = OpenAI(api_key=openai_key)

        fund_str = ", ".join(
            f"{k}: {v}"
            for k, v in fundamentals.items()
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        )
        tech_str = ", ".join(
            f"{k}: {v:.2f}"
            for k, v in technicals.items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        )

        prompt = (
            f"You are a financial analyst. Based on the following data for {ticker}:\n"
            f"Current Price: USD {current_price:.2f}\n"
            f"Fundamentals: {fund_str}\n"
            f"Technical Indicators: {tech_str}\n"
            "Provide TWO predictions as a JSON object:\n"
            "1. Target Price: realistic price target considering growth trends, valuation, momentum, and risk/reward\n"
            "2. Days to Target: estimated holding period in days to reach this target (typically 7-180 days based on momentum and catalysts)\n"
            "Return ONLY a JSON object with this exact format:\n"
            '{"target_price": <number>, "days_to_target": <integer>}\n'
            "JSON:"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )

        answer = response.choices[0].message.content.strip()
        json_match = re.search(r"\{[^}]+\}", answer)
        if json_match:
            data = json.loads(json_match.group(0))
            target = float(data.get("target_price", 0))
            days = int(data.get("days_to_target", 20))
            if current_price * 0.5 <= target <= current_price * 3.0 and 7 <= days <= 365:
                return (target, days)
    except Exception as e:
        logger.warning(f"OpenAI target prediction failed for {ticker}: {e}")

    return None


# ---------------------------------------------------------------------------
# Pure-logic target calculation (no Streamlit dependency)
# ---------------------------------------------------------------------------

_SECTOR_OFFSETS = {
    "Utilities": 1.3,
    "Consumer Defensive": 1.2,
    "Real Estate": 1.15,
    "Financials": 1.1,
    "Healthcare": 1.0,
    "Industrials": 0.95,
    "Energy": 0.9,
    "Consumer Cyclical": 0.85,
    "Technology": 0.75,
    "Communication Services": 0.8,
}


def _compute_holding_days(
    rr: float,
    rsi: float,
    momentum: float,
    volatility_factor: float,
    sector_mult: float,
    ml_mult: float,
    ticker: str,
) -> int:
    """Return estimated holding-period days (14-180)."""
    if np.isfinite(rr):
        base_days = 20 + (rr * 10)
        if np.isfinite(rsi):
            base_days *= 0.75 if rsi < 40 else (1.3 if rsi > 70 else 1.0)
        if np.isfinite(momentum) and momentum > 0.05:
            base_days *= 0.9
        elif np.isfinite(momentum) and momentum < -0.05:
            base_days *= 1.2
        base_days *= volatility_factor * sector_mult * ml_mult
        ticker_seed = sum(ord(c) for c in ticker) % 20
        base_days += ticker_seed
        return int(min(180, max(14, base_days)))
    else:
        base_days = 60 * volatility_factor * sector_mult * ml_mult
        ticker_seed = sum(ord(c) for c in ticker) % 30
        return int(min(180, max(30, base_days + ticker_seed)))


def calculate_targets(
    row: pd.Series,
    *,
    regime_data: Optional[dict] = None,
    openai_key: Optional[str] = None,
    enable_openai: bool = False,
    skip_openai: bool = False,
    adjust_target_for_regime=None,
    config: Optional[dict] = None,
) -> Tuple[float, float, str, str]:
    """Return ``(entry_price, target_price, target_date_str, source)``."""
    ticker = row.get("Ticker", "")
    current_price = row.get("Unit_Price", row.get("Price_Yahoo", np.nan))
    atr_val = row.get("ATR", np.nan)
    rr = row.get("RewardRisk", np.nan)
    rsi_val = row.get("RSI", np.nan)
    momentum = row.get("Momentum_63d", np.nan)
    sector = row.get("Sector", "")
    ml_prob = row.get("ML_20d_Prob", row.get("ml_probability", 0.5))
    if regime_data is None:
        regime_data = {"regime": "neutral", "confidence": 50}
    if config is None:
        config = {}

    if not (np.isfinite(current_price) and current_price > 0):
        return current_price, np.nan, "N/A", "N/A"

    # -- entry price --------------------------------------------------------
    entry_price = (current_price - 0.5 * atr_val) if np.isfinite(atr_val) else current_price * 0.98

    # -- multipliers --------------------------------------------------------
    atr_pct = (atr_val / current_price) if (np.isfinite(atr_val) and current_price > 0) else 0.02
    volatility_factor = np.clip(atr_pct / 0.03, 0.5, 2.5)
    sector_mult = _SECTOR_OFFSETS.get(sector, 1.0)
    ml_mult = 1.0
    if isinstance(ml_prob, (int, float)) and np.isfinite(ml_prob):
        ml_mult = 1.2 - (ml_prob * 0.4)

    # -- holding days -------------------------------------------------------
    days = _compute_holding_days(rr, rsi_val, momentum, volatility_factor, sector_mult, ml_mult, ticker)

    # -- AI target ----------------------------------------------------------
    ai_result = None
    if not skip_openai and enable_openai:
        fundamentals = {
            "PE": row.get("PERatio", np.nan),
            "PB": row.get("PBRatio", np.nan),
            "ROE": row.get("ROE", np.nan),
            "Margin": row.get("ProfitMargin", np.nan),
            "RevenueGrowth": row.get("RevenueGrowthYoY", np.nan),
        }
        technicals = {"RSI": rsi_val, "Momentum_63d": momentum, "RewardRisk": rr, "ATR": atr_val}
        try:
            ai_result = get_openai_target_prediction(ticker, current_price, fundamentals, technicals, openai_key=openai_key)
        except Exception as e:
            logger.warning(f"OpenAI call failed for {ticker}: {e}")

    if ai_result is not None:
        target_price, days = ai_result
        target_source = "AI"
    elif np.isfinite(atr_val) and np.isfinite(rr):
        base_target_pct = rr * (atr_val / current_price) if current_price > 0 else 0.10
        if adjust_target_for_regime is not None:
            reliability = row.get("Reliability_v2", row.get("reliability_pct", 50.0))
            risk_meter = row.get("risk_meter_v2", row.get("RiskMeter", 50.0))
            adjusted_target_pct, _ = adjust_target_for_regime(base_target_pct, reliability, risk_meter, regime_data)
        else:
            adjusted_target_pct = base_target_pct
        target_price = entry_price * (1 + adjusted_target_pct)
        target_source = "AI"
    else:
        target_price = entry_price * 1.10
        target_source = "Default"

    target_date = (datetime.datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    return entry_price, target_price, target_date, target_source
