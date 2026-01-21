from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, QuantileTransformer

from core.interfaces import TickerFeatures, DataQuality


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Stateful feature pipeline enforcing point-in-time correctness.

    - Fits scalers and sector statistics on training data only.
    - Transforms future data using stored state to avoid leakage.
    - Produces a list of TickerFeatures adhering to the contracts.
    """

    def __init__(self) -> None:
        # Scalers for different feature groups
        self.scalers: Dict[str, Any] = {
            "technicals": RobustScaler(),
            "volatility": QuantileTransformer(output_distribution="normal", n_quantiles=50),
        }
        # Sector statistics learned during fit (e.g., mean P/E per sector)
        self.sector_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

        # Columns used by the pipeline (expected in input DataFrame)
        self.col_ticker = "Ticker"
        self.col_date = "Date"
        self.col_sector = "Sector"
        self.col_pe = "PE"
        self.col_rsi = "RSI"
        self.col_atr_pct = "ATR_Pct"
        self.col_mktcap = "MarketCap"
        self.col_volume = "Volume"

    # ---- sklearn interface ----
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeaturePipeline":
        """Learn distributions and sector statistics from training data only.

        - Fits RobustScaler on RSI.
        - Fits QuantileTransformer on ATR_Pct as volatility proxy.
        - Computes and stores mean P/E per sector.
        """
        self._validate_required_columns(X, required=[
            self.col_ticker,
            self.col_date,
            self.col_sector,
            self.col_pe,
            self.col_rsi,
            self.col_atr_pct,
        ])

        # Fit scalers on training data (no re-fitting during transform)
        rsi_values = X[[self.col_rsi]].astype(float)
        atr_values = X[[self.col_atr_pct]].astype(float)

        self.scalers["technicals"].fit(rsi_values)
        self.scalers["volatility"].fit(atr_values)

        # Compute sector mean P/E
        sector_group = X[[self.col_sector, self.col_pe]].dropna()
        sector_group[self.col_pe] = sector_group[self.col_pe].astype(float)
        pe_means = sector_group.groupby(self.col_sector)[self.col_pe].mean()
        self.sector_stats = {sector: {"pe_mean": float(mean)} for sector, mean in pe_means.items()}

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, as_of_date: Optional[datetime] = None) -> List[TickerFeatures]:
        """Transform data using stored state and produce TickerFeatures.

        - Guards against leakage when as_of_date is provided.
        - Applies fitted scalers (no re-fitting).
        - Computes sector-relative P/E using stored sector means.
        - Separates model_features (normalized floats) from risk_metadata (raw fields).
        """
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform().")

        # Work on a positional index to avoid misalignment with numpy arrays
        X_proc = X.reset_index(drop=True)

        self._validate_required_columns(X_proc, required=[
            self.col_ticker,
            self.col_date,
            self.col_sector,
            self.col_pe,
            self.col_rsi,
            self.col_atr_pct,
            self.col_mktcap,
            self.col_volume,
        ])

        # Leakage guard: ensure no dates beyond as_of_date
        if as_of_date is not None:
            # Ensure Date column is datetime
            dates = pd.to_datetime(X_proc[self.col_date])
            if (dates > as_of_date).any():
                raise ValueError("Leakage detected: X contains rows beyond as_of_date.")

        # Prepare arrays for transformation
        rsi_vals = X_proc[[self.col_rsi]].astype(float)
        atr_vals = X_proc[[self.col_atr_pct]].astype(float)

        rsi_norm = self.scalers["technicals"].transform(rsi_vals)
        atr_norm = self.scalers["volatility"].transform(atr_vals)

        # Build TickerFeatures objects
        results: List[TickerFeatures] = []
        now_ts = datetime.utcnow()

        # Iterate row-wise to compose contracts
        for i, row in X_proc.iterrows():
            ticker = str(row[self.col_ticker])
            date_val = pd.to_datetime(row[self.col_date]).to_pydatetime()
            sector = str(row[self.col_sector]) if pd.notna(row[self.col_sector]) else "UNKNOWN"
            pe = float(row[self.col_pe]) if pd.notna(row[self.col_pe]) else np.nan

            # Sector-relative P/E using stored mean from fit
            sector_mean = self.sector_stats.get(sector, {}).get("pe_mean", np.nan)
            if sector_mean and sector_mean != 0 and not np.isnan(pe):
                pe_rel = (pe - sector_mean) / sector_mean
            else:
                pe_rel = 0.0

            # Normalized technical features (floats)
            feat_rsi = float(rsi_norm[i, 0])
            feat_atr = float(atr_norm[i, 0])

            model_features: Dict[str, float] = {
                "feat_rsi": feat_rsi,
                "feat_atr_pct": feat_atr,
                "feat_fund_pe_sector_rel": float(pe_rel),
            }

            # Raw risk metadata for rules/checks
            risk_metadata: Dict[str, Any] = {
                "sector": sector,
                "market_cap": float(row[self.col_mktcap]) if pd.notna(row[self.col_mktcap]) else np.nan,
                "volume": float(row[self.col_volume]) if pd.notna(row[self.col_volume]) else np.nan,
                "pe": pe,
                "rsi_raw": float(row[self.col_rsi]) if pd.notna(row[self.col_rsi]) else np.nan,
                "atr_pct_raw": float(row[self.col_atr_pct]) if pd.notna(row[self.col_atr_pct]) else np.nan,
            }

            tf = TickerFeatures(
                ticker=ticker,
                as_of_date=date_val,
                data_timestamp=now_ts,
                source_map={"features": "FeaturePipeline"},
                quality=DataQuality.HIGH,
                point_in_time_ok=True if (as_of_date is None or date_val <= as_of_date) else False,
                model_features=model_features,
                risk_metadata=risk_metadata,
            )
            results.append(tf)

        return results

    # ---- helpers ----
    def _validate_required_columns(self, X: pd.DataFrame, required: List[str]) -> None:
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
