from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ModelOutput:
    prediction_prob: float = 0.5


@dataclass
class FeaturesForRisk:
    """
    Minimal features container expected by RiskEngine.
    Provides `.risk_metadata` and `.ticker` attributes.
    """
    ticker: str
    risk_metadata: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


class InferenceEngine:
    """
    Minimal, import-safe inference engine used by bridge/tests.

    - __init__(models_dir: str = "models/v2") does not crash if models missing
    - make_features(row) returns a FeaturesForRisk object
    - predict_single(row) returns ModelOutput with prediction_prob
    """

    def __init__(self, models_dir: str = "models/v2") -> None:
        self.models_dir = models_dir

    def make_features(self, row: Dict[str, Any]) -> FeaturesForRisk:
        data = dict(row or {})
        ticker = data.get("Ticker") or data.get("ticker") or "UNKNOWN"
        rm: Dict[str, Any] = {}
        try:
            vol = data.get("Volume") or data.get("volume")
            rm["volume"] = vol
            rm["atr_pct_raw"] = data.get("ATR_Pct")
        except Exception:
            pass
        return FeaturesForRisk(ticker=ticker, risk_metadata=rm, data=data)

    def predict_single(self, row: Dict[str, Any]) -> ModelOutput:
        try:
            from core.ml_20d_inference import (
                ML_20D_AVAILABLE,
                compute_ml_20d_probabilities_raw,
                calibrate_ml_20d_prob,
            )
            if ML_20D_AVAILABLE:
                s = pd.Series(dict(row or {}))
                prob_raw = compute_ml_20d_probabilities_raw(s)
                rsi = s.get("RSI")
                try:
                    prob = calibrate_ml_20d_prob(
                        prob_raw,
                        atr_pct_percentile=None,
                        price_as_of=None,
                        reliability_factor=1.0,
                        market_regime=None,
                        rsi=float(rsi) if pd.notna(rsi) else None,
                    )
                except Exception:
                    prob = prob_raw
                try:
                    p = float(prob)
                    if not (0.0 <= p <= 1.0):
                        p = 0.5
                except Exception:
                    p = 0.5
                return ModelOutput(prediction_prob=p)
        except Exception:
            pass
        return ModelOutput(prediction_prob=0.5)
