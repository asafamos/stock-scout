"""
Risk classification and data quality evaluation for stock recommendations.
"""
from __future__ import annotations
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from core.logging_config import get_logger

logger = get_logger("classification")


@dataclass
class StockClassification:
    """Classification result for a stock."""
    risk_level: str  # "core" or "speculative"
    data_quality: str  # "high", "medium", "low"
    confidence_level: str  # "high", "medium", "low", "none"
    should_display: bool  # Whether to show in recommendations
    warnings: list[str]  # List of warning messages
    

def evaluate_data_quality(row: pd.Series) -> Tuple[str, int, list[str]]:
    """
    Evaluate data quality based on missing/invalid metrics.
    
    Returns:
        Tuple of (quality_level, valid_count, warnings)
        quality_level: "high", "medium", "low"
        valid_count: number of valid critical metrics
        warnings: list of issues found
    """
    warnings = []
    critical_metrics = {
        "RS_63d": row.get("RS_63d"),
        "Volume_Surge": row.get("Volume_Surge"),
        "RR_Ratio": row.get("RR_Ratio"),
        "Quality_Score": row.get("Quality_Score"),
        "Fundamental_S": row.get("Fundamental_S"),
        "Momentum_Consistency": row.get("Momentum_Consistency"),
    }
    
    valid_count = 0
    for metric_name, value in critical_metrics.items():
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            valid_count += 1
        else:
            warnings.append(f"Missing: {metric_name}")
    
    # Check technical metrics
    rsi = row.get("RSI")
    if rsi is None or (isinstance(rsi, float) and np.isnan(rsi)):
        warnings.append("Missing: RSI")
    
    atr_price = row.get("ATR_Price")
    if atr_price is None or (isinstance(atr_price, float) and np.isnan(atr_price)):
        warnings.append("Missing: ATR/Price")
    
    # Determine quality level
    total_critical = len(critical_metrics)
    if valid_count >= total_critical * 0.85:  # 85%+ metrics valid
        quality = "high"
    elif valid_count >= total_critical * 0.60:  # 60-85% metrics valid
        quality = "medium"
    else:
        quality = "low"
    
    return quality, valid_count, warnings


def evaluate_risk_factors(row: pd.Series) -> list[str]:
    """
    Evaluate risk factors for a stock.
    
    Returns:
        List of risk warnings
    """
    risk_warnings = []
    
    # Check RSI (overbought)
    rsi = row.get("RSI")
    if rsi is not None and isinstance(rsi, (int, float)) and rsi > 75:
        risk_warnings.append("Overbought (RSI > 75)")
    
    # Check volatility
    atr_price = row.get("ATR_Price")
    if atr_price is not None and isinstance(atr_price, (int, float)) and atr_price > 0.06:
        risk_warnings.append("High volatility (ATR/Price > 6%)")
    
    # Check overextension
    overext = row.get("OverextRatio")
    if overext is not None and isinstance(overext, (int, float)) and overext > 0.25:
        risk_warnings.append("Overextended vs MA")
    
    # Check beta
    beta = row.get("Beta")
    if beta is not None and isinstance(beta, (int, float)) and beta > 1.5:
        risk_warnings.append(f"High beta ({beta:.2f})")
    
    # Check debt
    de = row.get("DE_f")
    if de is not None and isinstance(de, (int, float)) and de > 2.0:
        risk_warnings.append(f"High debt (D/E={de:.2f})")
    
    # Check fundamental quality
    qual_label = row.get("Quality_Label")
    if qual_label == "Low":
        risk_warnings.append("Low fundamental quality")
    
    return risk_warnings


def classify_stock(row: pd.Series) -> StockClassification:
    """
    Classify a stock as core or speculative based on data quality and risk.
    
    Args:
        row: DataFrame row with all stock metrics
        
    Returns:
        StockClassification with risk level, data quality, and confidence
    """
    # Evaluate data quality
    data_quality, valid_count, data_warnings = evaluate_data_quality(row)
    
    # Evaluate risk factors
    risk_warnings = evaluate_risk_factors(row)
    
    # Combine all warnings
    all_warnings = data_warnings + risk_warnings
    
    # Initial classification based on data quality
    if data_quality == "low":
        risk_level = "speculative"
        confidence_level = "none"
        should_display = False  # Don't show low-quality stocks
    elif data_quality == "medium" and len(risk_warnings) >= 3:
        risk_level = "speculative"
        confidence_level = "low"
        should_display = True
    elif len(risk_warnings) >= 4:
        risk_level = "speculative"
        confidence_level = "low"
        should_display = True
    else:
        # Could be core
        risk_level = "core"
        
        # Determine confidence level for core stocks
        if data_quality == "high" and len(risk_warnings) == 0:
            confidence_level = "high"
        elif data_quality == "high" and len(risk_warnings) <= 1:
            confidence_level = "medium"
        elif data_quality == "medium" and len(risk_warnings) <= 1:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        should_display = True
    
    # Override high confidence if original high_confidence flag is False
    original_high_conf = row.get("High_Confidence", False)
    if not original_high_conf and confidence_level == "high":
        confidence_level = "medium"
    
    # Check MA alignment
    ma_aligned = row.get("MA_Aligned", False)
    if not ma_aligned and confidence_level == "high":
        confidence_level = "medium"
        all_warnings.append("MAs not aligned")
    
    return StockClassification(
        risk_level=risk_level,
        data_quality=data_quality,
        confidence_level=confidence_level,
        should_display=should_display,
        warnings=all_warnings
    )


def apply_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply classification to entire DataFrame.
    
    Adds columns:
    - Risk_Level: "core" or "speculative"
    - Data_Quality: "high", "medium", "low"
    - Confidence_Level: "high", "medium", "low", "none"
    - Should_Display: boolean
    - Classification_Warnings: list of warning strings
    
    Args:
        df: DataFrame with stock data
        
    Returns:
        DataFrame with classification columns added
    """
    classifications = []
    
    for idx, row in df.iterrows():
        classification = classify_stock(row)
        classifications.append({
            "Risk_Level": classification.risk_level,
            "Data_Quality": classification.data_quality,
            "Confidence_Level": classification.confidence_level,
            "Should_Display": classification.should_display,
            "Classification_Warnings": "; ".join(classification.warnings) if classification.warnings else ""
        })
    
    # Add classification columns
    for col in ["Risk_Level", "Data_Quality", "Confidence_Level", "Should_Display", "Classification_Warnings"]:
        df[col] = [c[col] for c in classifications]
    
    logger.info(f"Classified {len(df)} stocks: "
                f"Core={sum(df['Risk_Level']=='core')}, "
                f"Speculative={sum(df['Risk_Level']=='speculative')}, "
                f"Hidden={sum(~df['Should_Display'])}")
    
    return df
