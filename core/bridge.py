from __future__ import annotations

import os
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ml.inference import InferenceEngine
from core.risk_engine import RiskEngine
from core.scoring_engine import (
    calculate_reliability_score,
    calculate_risk_meter,
    calculate_conviction_score,
    calculate_quality_score,
)
from core.unified_logic import compute_tech_score_20d_v2


_logger = logging.getLogger(__name__)


class StockScoutBridge:
    """Bridge legacy UI dictionary outputs to the new ML/Risk engines.

    Loads the inference and risk engines once, and translates their outputs
    into the flat dictionary expected by the Streamlit app.
    """

    _inference_engine: InferenceEngine | None = None
    _risk_engine: RiskEngine | None = None

    def __init__(self, models_dir: str = os.path.join("models", "v2")) -> None:
        if StockScoutBridge._inference_engine is None:
            StockScoutBridge._inference_engine = InferenceEngine(models_dir=models_dir)
        if StockScoutBridge._risk_engine is None:
            StockScoutBridge._risk_engine = RiskEngine()
        self.infer = StockScoutBridge._inference_engine
        self.risk = StockScoutBridge._risk_engine

    def analyze_ticker(self, ticker: str, raw_data_row: pd.Series) -> Dict[str, Any]:
        """Analyze a single ticker row and return legacy-style dictionary.

        - Converts the input Series to a dict for the inference engine
        - Runs inference to produce ModelOutput
        - Evaluates risk to produce TradeDecision
        - Computes proper Risk_Label (from risk_meter), ConvictionScore (multi-source),
          and ReliabilityScore (data quality) instead of simple ML-prob aliases.
        - Returns a merged dictionary with overrides to match the legacy UI expected keys
        """
        base: Dict[str, Any] = dict(raw_data_row) if isinstance(raw_data_row, pd.Series) else dict(raw_data_row)
        base["Ticker"] = base.get("Ticker", ticker)
        try:
            mo = self.infer.predict_single(base)
            tf = self.infer.make_features(base)
            decision = self.risk.evaluate(tf, mo)

            # --- Compute proper Risk_Label via risk_meter (not from Action enum) ---
            rr_ratio = base.get("RR", base.get("RR_Ratio", base.get("RewardRisk")))
            try:
                rr_val = float(rr_ratio) if rr_ratio is not None else None
                if rr_val is not None and not np.isfinite(rr_val):
                    rr_val = None
            except (TypeError, ValueError):
                rr_val = None

            beta = base.get("Beta")
            try:
                beta_val = float(beta) if beta is not None else None
                if beta_val is not None and not np.isfinite(beta_val):
                    beta_val = None
            except (TypeError, ValueError):
                beta_val = None

            atr_pct = base.get("ATR_Pct")
            try:
                atr_val = float(atr_pct) if atr_pct is not None else None
                if atr_val is not None and not np.isfinite(atr_val):
                    atr_val = None
            except (TypeError, ValueError):
                atr_val = None

            leverage = base.get("DE_f", base.get("debt_to_equity"))
            try:
                lev_val = float(leverage) if leverage is not None else None
                if lev_val is not None and not np.isfinite(lev_val):
                    lev_val = None
            except (TypeError, ValueError):
                lev_val = None

            risk_meter, risk_label = calculate_risk_meter(
                rr_ratio=rr_val,
                beta=beta_val,
                atr_pct=atr_val,
                leverage=lev_val,
            )

            # --- Compute proper ReliabilityScore from actual data quality ---
            price_sources = 0
            fund_sources = 0
            for src_key in ("Fund_from_FMP", "Fund_from_Finnhub", "Fund_from_Alpha",
                            "Fund_from_Tiingo", "Fund_from_EODHD", "Fund_from_SimFin"):
                try:
                    if base.get(src_key):
                        fund_sources += 1
                except Exception:
                    pass
            try:
                price_sources = int(base.get("Price_Sources_Count", 0) or 0)
            except (TypeError, ValueError):
                price_sources = 0

            # Data completeness from core tech fields
            tech_fields = ["RSI", "ATR", "MA20", "MA50", "Overext", "Near52w", "RR",
                           "Volume", "Close", "ADR_Pct", "ATR_Pct"]
            valid_count = sum(1 for f in tech_fields if f in base and base.get(f) is not None
                             and (not isinstance(base.get(f), float) or np.isfinite(base[f])))
            data_completeness = (valid_count / len(tech_fields)) * 100.0

            price_std = base.get("Price_STD")
            price_mean = base.get("Price_Mean")
            try:
                p_std = float(price_std) if price_std is not None else None
            except (TypeError, ValueError):
                p_std = None
            try:
                p_mean = float(price_mean) if price_mean is not None else None
            except (TypeError, ValueError):
                p_mean = None

            reliability_score = calculate_reliability_score(
                price_sources=price_sources,
                fund_sources=fund_sources,
                price_std=p_std,
                price_mean=p_mean,
                fundamental_confidence=data_completeness,
                data_completeness=data_completeness,
            )

            # --- Compute proper TechScore_20d from technical indicators ---
            # Use the v2 technical scoring function (returns 0-1, scale to 0-100)
            try:
                base_series = pd.Series(base)
                tech_raw = compute_tech_score_20d_v2(base_series)
                tech_score_20d = float(np.clip(tech_raw * 100.0, 0.0, 100.0))
            except Exception:
                tech_score_20d = 50.0  # Neutral fallback

            # --- Compute proper ConvictionScore using multi-factor blend ---
            tech_score = tech_score_20d
            fund_score = float(base.get("Fundamental_S", base.get("FundamentalScore", 50.0)) or 50.0)
            try:
                if not np.isfinite(fund_score):
                    fund_score = 50.0
            except (TypeError, ValueError):
                fund_score = 50.0

            # RR score: convert ratio to 0-100
            rr_score = 50.0
            if rr_val is not None:
                if rr_val >= 3.0:
                    rr_score = 90.0
                elif rr_val >= 2.0:
                    rr_score = 75.0
                elif rr_val >= 1.5:
                    rr_score = 60.0
                elif rr_val >= 1.0:
                    rr_score = 45.0
                else:
                    rr_score = 25.0

            conviction_score, _ = calculate_conviction_score(
                fundamental_score=fund_score,
                fundamental_confidence=data_completeness,
                momentum_score=tech_score,
                momentum_confidence=min(data_completeness, 80.0),
                rr_score=rr_score,
                rr_confidence=80.0 if rr_val is not None else 30.0,
                reliability_score=reliability_score,
                ml_probability=mo.prediction_prob,
            )

            # --- Compute Quality_Score from fundamental metrics ---
            try:
                qual_score, qual_level = calculate_quality_score(base_series)
            except Exception:
                qual_score, qual_level = 0.5, "Medium"

            overrides: Dict[str, Any] = {
                "TechScore_20d": tech_score_20d,
                "Fundamental_Score": fund_score,
                "FinalScore_20d": decision.conviction,
                "Score": decision.conviction,
                "ML_20d_Prob": mo.prediction_prob,
                "Quality_Score": qual_score,
                "Quality_Level": qual_level,
                "Risk_Label": risk_label,
                "Risk_Meter": risk_meter,
                "ConvictionScore": conviction_score,
                "ReliabilityScore": reliability_score,
                "Reliability_Score": reliability_score,
                "Evaluation_Mode": "V2_ML_Risk_Engine",
                "Primary_Reason": decision.primary_reason,
                "Active_Penalties": ",".join(decision.risk_penalties) if decision.risk_penalties else "",
                "Action": decision.action.value,
            }

            out = {**base, **overrides}
            return out
        except Exception as e:
            _logger.exception("Bridge analyze_ticker failed for %s: %s", ticker, e)
            # Fallback: return the original row unchanged
            return base


def analyze_row_with_bridge(ticker: str, row: pd.Series) -> Dict[str, Any]:
    bridge = StockScoutBridge()
    return bridge.analyze_ticker(ticker, row)
