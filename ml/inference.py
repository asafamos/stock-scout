from __future__ import annotations

import os
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from core.interfaces import TickerFeatures, ModelOutput


class InferenceEngine:
    """Loads trained artifacts and performs single-record inference."""

    def __init__(self, models_dir: str = os.path.join("models", "v2")) -> None:
        pipeline_path = os.path.join(models_dir, "feature_pipeline.joblib")
        model_path = os.path.join(models_dir, "model_xgb.json")
        self.pipeline = joblib.load(pipeline_path)
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)
        self.model_version = "v2_xgb"

    def _raw_to_df(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        # Minimal columns expected by FeaturePipeline
        df = pd.DataFrame([
            {
                "Ticker": raw_data.get("Ticker", "UNKNOWN"),
                "Date": pd.to_datetime(raw_data.get("Date", datetime.utcnow())),
                "Sector": raw_data.get("Sector", "Technology"),
                "PE": float(raw_data.get("PE", 25.0)),
                "RSI": float(raw_data.get("RSI", 50.0)),
                "ATR_Pct": float(raw_data.get("ATR_Pct", 0.02)),
                "MarketCap": float(raw_data.get("MarketCap", np.nan)),
                "Volume": float(raw_data.get("Volume", 1_000_000.0)),
            }
        ])
        return df

    def make_features(self, raw_data: Dict[str, Any]) -> TickerFeatures:
        df = self._raw_to_df(raw_data)
        as_of = pd.to_datetime(df.iloc[0]["Date"]).to_pydatetime()
        features_list: List[TickerFeatures] = self.pipeline.transform(df, as_of_date=as_of)
        tf = features_list[0]

        # Augment risk metadata with optional keys from raw input
        extra_rm: Dict[str, Any] = {}
        if "DaysToEarnings" in raw_data:
            extra_rm["days_to_earnings"] = raw_data["DaysToEarnings"]
        if "AvgVolume" in raw_data:
            extra_rm["volume_avg"] = raw_data["AvgVolume"]

        if extra_rm:
            new_rm = dict(tf.risk_metadata)
            new_rm.update(extra_rm)
            tf = replace(tf, risk_metadata=new_rm)
        return tf

    def predict_single(self, raw_data: Dict[str, Any]) -> ModelOutput:
        tf = self.make_features(raw_data)
        # Build feature vector in fixed order
        X = np.array([[
            tf.model_features["feat_rsi"],
            tf.model_features["feat_atr_pct"],
            tf.model_features["feat_fund_pe_sector_rel"],
        ]], dtype=float)

        dmat = xgb.DMatrix(X)
        t0 = time.perf_counter()
        probs = self.booster.predict(dmat)
        dt = time.perf_counter() - t0

        # Multi-class probs: select Class 2 (Gold)
        if probs.ndim == 2 and probs.shape[1] >= 3:
            gold_prob = float(probs[0, 2])
        else:
            gold_prob = float(probs[0]) if probs.ndim > 0 else 0.0

        # Simple heuristic for confidence (placeholder)
        confidence = 1.0

        mo = ModelOutput(
            prediction_prob=gold_prob,
            expected_return=0.0,
            confidence_score=confidence,
            calibration_factor=1.0,
            model_version=self.model_version,
            generation_time=dt,
        )
        return mo
