"""
Risk classification and data quality evaluation for stock recommendations.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
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

    # Fundamentals coverage (do not auto-force low; allow medium/high based on other metrics for display purposes)
    fundamental_total = row.get("Fundamental_S")
    fundamental_quality = row.get("Quality_Score_F")
    coverage_pct = row.get("Fund_Coverage_Pct")
    fundamentals_missing = (
        fundamental_total is None or (isinstance(fundamental_total, float) and np.isnan(fundamental_total)) or
        fundamental_quality is None or (isinstance(fundamental_quality, float) and np.isnan(fundamental_quality))
    )
    if fundamentals_missing:
        warnings.append("Missing: Fundamental_S/Quality_Score_F")
    if isinstance(coverage_pct, (int, float)) and np.isfinite(coverage_pct):
        if coverage_pct < 0.25:
            warnings.append("Low fundamental coverage (<25%)")
    
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
        quality = "high" if not fundamentals_missing else "medium"  # downgrade one level if fundamentals missing
    elif valid_count >= total_critical * 0.60:  # 60-85% metrics valid
        quality = "medium" if not fundamentals_missing else "low"
    else:
        quality = "low"
    # Second downgrade if coverage extremely low
    if isinstance(coverage_pct, (int,float)) and np.isfinite(coverage_pct) and coverage_pct < 0.25 and quality == "medium":
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
    ticker = row.get("Ticker", "UNKNOWN")
    
    # Evaluate data quality
    data_quality, valid_count, data_warnings = evaluate_data_quality(row)
    
    # Evaluate risk factors
    risk_warnings = evaluate_risk_factors(row)
    
    # Combine all warnings
    all_warnings = data_warnings + risk_warnings
    
    # Debug logging for first few stocks
    import random
    if random.random() < 0.05:  # Log ~5% of stocks
        logger.debug(f"🔍 {ticker}: DataQuality={data_quality}, ValidFields={valid_count}/6, "
                    f"RiskWarnings={len(risk_warnings)}, Warnings={all_warnings[:2]}")
    
    fundamentals_missing = (
        row.get("Fundamental_S") is None or (isinstance(row.get("Fundamental_S"), float) and np.isnan(row.get("Fundamental_S"))) or
        row.get("Quality_Score_F") is None or (isinstance(row.get("Quality_Score_F"), float) and np.isnan(row.get("Quality_Score_F")))
    )
    if fundamentals_missing:
        all_warnings.append("Fundamentals missing")

    # Initial classification - stricter Core criteria to populate Speculative category
    if data_quality == "low":
        risk_level = "speculative"
        confidence_level = "none"
        tech_fields = [row.get("RS_63d"), row.get("Volume_Surge"), row.get("RR_Ratio"), row.get("Momentum_Consistency"), row.get("RSI"), row.get("ATR_Price")]
        tech_valid = sum(
            v is not None and not (isinstance(v, float) and np.isnan(v))
            for v in tech_fields
        )
        should_display = tech_valid >= 4
    elif data_quality == "medium" and len(risk_warnings) >= 2:  # More aggressive - medium quality with 2+ warnings
        risk_level = "speculative"
        confidence_level = "medium" if len(risk_warnings) <= 3 else "low"
        should_display = True
    elif len(risk_warnings) >= 3:  # More aggressive - 3+ warnings = speculative
        risk_level = "speculative"
        confidence_level = "medium" if len(risk_warnings) <= 4 else "low"
        should_display = True
    elif data_quality == "medium":  # Medium quality even without many warnings → Speculative
        risk_level = "speculative"
        confidence_level = "medium"
        should_display = True
    else:
        # Core only for high quality with minimal warnings
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


def filter_core_recommendations(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    adaptive: bool = True
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only high-quality "Core" stocks that pass conservative thresholds.
    
    Uses CONFIG constants for filtering:
    - MIN_QUALITY_SCORE_CORE: minimum fundamental quality score (default 27/50)
    - MAX_OVEREXTENSION_CORE: maximum overextension ratio (default 0.10)
    - MAX_ATR_PRICE_CORE: maximum ATR/Price ratio (default 0.08)
    - RSI_MIN_CORE, RSI_MAX_CORE: RSI bounds (default 45-70)
    - MIN_RR_CORE: minimum reward/risk ratio (default 1.5)
    
    All existing safety checks (earnings blackout, beta, liquidity, etc.) 
    are preserved upstream in the pipeline.
    
    Args:
        df: DataFrame with classified stocks
        config: Optional config dict with threshold constants
        
    Returns:
        Filtered DataFrame with only Core stocks that pass all filters
    """
    if df.empty:
        return df
    
    # Default config if not provided - more selective for Core to allow Speculative category
    if config is None:
        config = {
            "MIN_QUALITY_SCORE_CORE": 30.0,  # Raised to be more selective (best stocks only)
            "MAX_OVEREXTENSION_CORE": 0.08,  # Tightened back
            "MAX_ATR_PRICE_CORE": 0.07,      # Tightened - Core should be stable
            "RSI_MIN_CORE": 45,              # Back to conservative
            "RSI_MAX_CORE": 68,              # Avoid overbought for Core
            "MIN_RR_CORE": 1.5,              # Core needs good risk/reward
        }
    
    initial_count = len(df)
    filtered = df.copy()
    
    # Debug: Show initial distribution
    if "Risk_Level" in filtered.columns:
        core_count = (filtered["Risk_Level"] == "core").sum()
        spec_count = (filtered["Risk_Level"] == "speculative").sum()
        logger.info(f"📊 Initial distribution: {core_count} Core, {spec_count} Speculative (total: {initial_count})")
    
    # Filter 1: Only Core stocks
    if "Risk_Level" in filtered.columns:
        filtered = filtered[filtered["Risk_Level"] == "core"]
        logger.info(f"After Core filter: {len(filtered)}/{initial_count}")
        
        # Debug: If no Core stocks, show why
        if len(filtered) == 0:
            logger.warning("⚠️ NO CORE STOCKS FOUND - All stocks classified as Speculative!")
            logger.info("Checking first 3 stocks for classification reasons...")
            for idx, row in df.head(3).iterrows():
                logger.info(f"  {row.get('Ticker', '?')}: Quality={row.get('Quality_Score', 'N/A')}, "
                            f"Data_Quality={row.get('Data_Quality', 'N/A')}, "
                            f"Warnings={row.get('Classification_Warnings', '')[:80]}")
    
    # Filter 2: Minimum quality score (fundamental)
    if "Quality_Score_F" in filtered.columns:
        min_qual = config.get("MIN_QUALITY_SCORE_CORE", 27.0)
        filtered = filtered[
            (filtered["Quality_Score_F"].isna()) | 
            (filtered["Quality_Score_F"] >= min_qual)
        ]
        logger.info(f"After fundamental quality score >= {min_qual}: {len(filtered)}/{initial_count}")
    
    # Filter 3: Maximum overextension
    if "OverextRatio" in filtered.columns:
        max_overext = config.get("MAX_OVEREXTENSION_CORE", 0.10)
        filtered = filtered[
            (filtered["OverextRatio"].isna()) | 
            (filtered["OverextRatio"] <= max_overext)
        ]
        logger.info(f"After overextension <= {max_overext}: {len(filtered)}/{initial_count}")
    
    # Filter 4: Maximum ATR/Price (volatility)
    if "ATR_Price" in filtered.columns:
        max_atr = config.get("MAX_ATR_PRICE_CORE", 0.08)
        filtered = filtered[
            (filtered["ATR_Price"].isna()) | 
            (filtered["ATR_Price"] <= max_atr)
        ]
        logger.info(f"After ATR/Price <= {max_atr}: {len(filtered)}/{initial_count}")
    
    # Filter 5: RSI bounds
    if "RSI" in filtered.columns:
        rsi_min = config.get("RSI_MIN_CORE", 45)
        rsi_max = config.get("RSI_MAX_CORE", 70)
        filtered = filtered[
            (filtered["RSI"].isna()) | 
            ((filtered["RSI"] >= rsi_min) & (filtered["RSI"] <= rsi_max))
        ]
        logger.info(f"After RSI in [{rsi_min}, {rsi_max}]: {len(filtered)}/{initial_count}")
    
    # Filter 6: Minimum reward/risk ratio
    if "RewardRisk" in filtered.columns:
        min_rr = config.get("MIN_RR_CORE", 1.5)
        filtered = filtered[
            (filtered["RewardRisk"].isna()) | 
            (filtered["RewardRisk"] >= min_rr)
        ]
        logger.info(f"After RewardRisk >= {min_rr}: {len(filtered)}/{initial_count}")
    
    # Adaptive relaxation if enabled and still empty after filters
    if adaptive and len(filtered) == 0:
        relax_cfg = {
            "MIN_QUALITY_SCORE_CORE": max(config.get("MIN_QUALITY_SCORE_CORE", 27.0) - 5, 15),
            "MAX_OVEREXTENSION_CORE": config.get("MAX_OVEREXTENSION_CORE", 0.10) + 0.05,
            "MAX_ATR_PRICE_CORE": config.get("MAX_ATR_PRICE_CORE", 0.08) + 0.04,
            "RSI_MIN_CORE": max(config.get("RSI_MIN_CORE", 45) - 5, 30),
            "RSI_MAX_CORE": min(config.get("RSI_MAX_CORE", 70) + 5, 80),
            "MIN_RR_CORE": max(config.get("MIN_RR_CORE", 1.5) - 0.3, 1.0),
        }
        logger.warning(f"🔄 Adaptive relaxation engaged: {relax_cfg}")
        pool = df[df.get("Should_Display", True)]  # showable stocks
        # Apply relaxed technical filters on speculative / medium-quality candidates
        relaxed = pool.copy()
        if "Quality_Score_F" in relaxed.columns:
            relaxed = relaxed[(relaxed["Quality_Score_F"].isna()) | (relaxed["Quality_Score_F"] >= relax_cfg["MIN_QUALITY_SCORE_CORE"])]
        if "OverextRatio" in relaxed.columns:
            relaxed = relaxed[(relaxed["OverextRatio"].isna()) | (relaxed["OverextRatio"] <= relax_cfg["MAX_OVEREXTENSION_CORE"])]
        if "ATR_Price" in relaxed.columns:
            relaxed = relaxed[(relaxed["ATR_Price"].isna()) | (relaxed["ATR_Price"] <= relax_cfg["MAX_ATR_PRICE_CORE"])]
        if "RSI" in relaxed.columns:
            relaxed = relaxed[(relaxed["RSI"].isna()) | ((relaxed["RSI"] >= relax_cfg["RSI_MIN_CORE"]) & (relaxed["RSI"] <= relax_cfg["RSI_MAX_CORE"]))]
        if "RewardRisk" in relaxed.columns:
            relaxed = relaxed[(relaxed["RewardRisk"].isna()) | (relaxed["RewardRisk"] >= relax_cfg["MIN_RR_CORE"])]
        # Limit size
        relaxed = relaxed.head( min( max(5, len(relaxed)), 15 ) )
        relaxed["Adaptive_Relaxed"] = True
        filtered = relaxed
        logger.info(f"Adaptive relaxation produced {len(filtered)} provisional candidates")

    logger.info(f"Final Core recommendations (including adaptive if any): {len(filtered)}/{initial_count}")
    return filtered.reset_index(drop=True)
