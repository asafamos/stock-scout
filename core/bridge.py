from __future__ import annotations

import os
import logging
from typing import Any, Dict

import pandas as pd

from ml.inference import InferenceEngine
from core.risk_engine import RiskEngine


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
        - Returns a merged dictionary with overrides to match the legacy UI expected keys
        """
        base: Dict[str, Any] = dict(raw_data_row) if isinstance(raw_data_row, pd.Series) else dict(raw_data_row)
        base["Ticker"] = base.get("Ticker", ticker)
        try:
            mo = self.infer.predict_single(base)
            tf = self.infer.make_features(base)
            decision = self.risk.evaluate(tf, mo)

            overrides: Dict[str, Any] = {
                "FinalScore_20d": decision.conviction,
                "Score": decision.conviction,
                "ML_20d_Prob": mo.prediction_prob,
                "Risk_Label": decision.action.value,
                "Risk_Meter": (1.0 - decision.conviction / 100.0) * 100.0,
                "ConvictionScore": decision.conviction,
                "Evaluation_Mode": "V2_ML_Risk_Engine",
                "Primary_Reason": decision.primary_reason,
                "Active_Penalties": ",".join(decision.risk_penalties) if decision.risk_penalties else "",
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
