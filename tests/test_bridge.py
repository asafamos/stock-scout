import os
import sys
from datetime import datetime

import pandas as pd
import pytest

from core.bridge import StockScoutBridge


skip_reason = "Artifacts missing; run training first"
need_artifacts = not (os.path.exists("models/v2/model_xgb.json") and os.path.exists("models/v2/feature_pipeline.joblib"))


@pytest.mark.skipif(need_artifacts, reason=skip_reason)
def test_bridge_outputs_legacy_dict():
    bridge = StockScoutBridge()
    row = pd.Series({
        "Ticker": "AAPL",
        "Date": datetime.utcnow(),
        "Close": 150.0,
        "Volume": 1_000_000,
        "RSI": 50.0,
        # Optional fields may be missing; inference fills defaults
    })

    out = bridge.analyze_ticker("AAPL", row)
    assert "FinalScore_20d" in out
    assert out.get("Evaluation_Mode") == "V2_ML_Risk_Engine"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
