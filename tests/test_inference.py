from datetime import datetime

import os
import pytest

from core.interfaces import Action
from core.risk_engine import RiskEngine
from ml.inference import InferenceEngine


@pytest.mark.skipif(not os.path.exists("models/v2/model_xgb.json") or not os.path.exists("models/v2/feature_pipeline.joblib"), reason="Artifacts missing; run training first")
def test_full_inference_and_risk_flow():
    engine = InferenceEngine()
    risk = RiskEngine()

    raw = {
        "Ticker": "AAPL",
        "Date": datetime.utcnow(),
        "Sector": "Technology",
        "PE": 25.0,
        "RSI": 55.0,
        "ATR_Pct": 0.02,
        "MarketCap": 2_500_000_000_000,
        "Volume": 50_000_000,
        "AvgVolume": 50_000_000,
        "DaysToEarnings": 10,
    }

    mo = engine.predict_single(raw)
    tf = engine.make_features(raw)
    decision = risk.evaluate(tf, mo)

    assert decision.action in {Action.BUY, Action.REJECT}

    # Rejection case: low liquidity
    raw_low = dict(raw)
    raw_low["AvgVolume"] = 100
    raw_low["Volume"] = 100
    raw_low["DaysToEarnings"] = 2

    mo2 = engine.predict_single(raw_low)
    tf2 = engine.make_features(raw_low)
    decision2 = risk.evaluate(tf2, mo2)

    assert decision2.action == Action.REJECT
