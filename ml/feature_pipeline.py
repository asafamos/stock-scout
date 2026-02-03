from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from core.interfaces import TickerFeatures, DataQuality


class FeaturePipeline:
    """
    Feature pipeline for ML training and inference.
    
    - fit(): Learn normalization parameters from training data
    - transform(): Transform data into TickerFeatures objects with normalized features
    
    Features are normalized to have zero mean and unit variance based on training data.
    Includes leakage guard to prevent using future data.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._fitted = False
        self._feature_stats: Dict[str, Dict[str, float]] = {}  # {feature: {mean, std}}
        self._sector_pe_stats: Dict[str, Dict[str, float]] = {}  # {sector: {mean, std}}
        self._fit_end_date: Optional[datetime] = None

    def fit(self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> "FeaturePipeline":
        """
        Fit the pipeline on training data.
        
        Learns:
        - Mean and std for RSI, ATR_Pct normalization
        - Sector-specific PE statistics for relative PE calculation
        """
        # Store the latest date in training data for leakage guard
        if "Date" in df.columns:
            self._fit_end_date = pd.to_datetime(df["Date"]).max().to_pydatetime()
        
        # Learn normalization stats for key features
        for col in ["RSI", "ATR_Pct"]:
            if col in df.columns:
                values = df[col].dropna()
                self._feature_stats[col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()) if len(values) > 1 else 1.0
                }
        
        # Learn sector-specific PE stats
        if "Sector" in df.columns and "PE" in df.columns:
            for sector, group in df.groupby("Sector"):
                pe_values = group["PE"].dropna()
                if len(pe_values) > 0:
                    self._sector_pe_stats[str(sector)] = {
                        "mean": float(pe_values.mean()),
                        "std": float(pe_values.std()) if len(pe_values) > 1 else 1.0
                    }
        
        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        as_of_date: Optional[datetime] = None,
        **kwargs: Any
    ) -> List[TickerFeatures]:
        """
        Transform data into TickerFeatures objects.
        
        Args:
            df: DataFrame with columns Ticker, Date, Sector, PE, RSI, ATR_Pct, etc.
            as_of_date: The "present" date for leakage guard. Data after this date raises ValueError.
        
        Returns:
            List of TickerFeatures objects with normalized model features
        
        Raises:
            RuntimeError: If transform called before fit
            ValueError: If data contains dates after as_of_date (leakage)
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        # Leakage guard: check for future data
        if as_of_date is not None and "Date" in df.columns:
            max_data_date = pd.to_datetime(df["Date"]).max().to_pydatetime()
            if max_data_date > as_of_date:
                raise ValueError(
                    f"Data leakage detected: data contains dates up to {max_data_date} "
                    f"but as_of_date is {as_of_date}"
                )
        
        results: List[TickerFeatures] = []
        
        for idx, row in df.iterrows():
            ticker = str(row.get("Ticker", "UNKNOWN"))
            data_date = pd.to_datetime(row.get("Date", datetime.now())).to_pydatetime()
            sector = str(row.get("Sector", "Unknown"))
            
            # Build normalized features
            model_features: Dict[str, float] = {}
            
            # Normalize RSI
            if "RSI" in row and pd.notna(row["RSI"]):
                stats = self._feature_stats.get("RSI", {"mean": 50.0, "std": 20.0})
                std = stats["std"] if stats["std"] > 0 else 1.0
                model_features["feat_rsi"] = float((row["RSI"] - stats["mean"]) / std)
            else:
                model_features["feat_rsi"] = 0.0
            
            # Normalize ATR_Pct
            if "ATR_Pct" in row and pd.notna(row["ATR_Pct"]):
                stats = self._feature_stats.get("ATR_Pct", {"mean": 0.02, "std": 0.01})
                std = stats["std"] if stats["std"] > 0 else 1.0
                model_features["feat_atr_pct"] = float((row["ATR_Pct"] - stats["mean"]) / std)
            else:
                model_features["feat_atr_pct"] = 0.0
            
            # Sector-relative PE
            if "PE" in row and pd.notna(row["PE"]):
                sector_stats = self._sector_pe_stats.get(sector, {"mean": 20.0, "std": 10.0})
                std = sector_stats["std"] if sector_stats["std"] > 0 else 1.0
                model_features["feat_fund_pe_sector_rel"] = float(
                    (row["PE"] - sector_stats["mean"]) / std
                )
            else:
                model_features["feat_fund_pe_sector_rel"] = 0.0
            
            # Create TickerFeatures object
            tf = TickerFeatures(
                ticker=ticker,
                as_of_date=as_of_date or data_date,
                data_timestamp=data_date,
                source_map={"price": "historical", "fundamentals": "provided"},
                quality=DataQuality.MEDIUM,
                point_in_time_ok=True,
                model_features=model_features,
                risk_metadata={}
            )
            results.append(tf)
        
        return results
