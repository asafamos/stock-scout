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
    # If price reliability is high, treat missing fundamentals less harshly
    price_rel = row.get("Price_Reliability", 0.0)
    if fundamentals_missing:
        if isinstance(price_rel, (int, float)) and np.isfinite(price_rel) and price_rel >= 0.75:
            # Treat as present for quality purposes but note it
            fundamentals_missing = False
            warnings.append("Note: Fundamentals missing but price reliability high")
        else:
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
        quality = "high" if not fundamentals_missing else "medium"
    elif valid_count >= total_critical * 0.60:  # 60-85% metrics valid
        quality = "medium" if not fundamentals_missing else "low"
    else:
        quality = "low"
    
    # Downgrade based on fundamental quality score (not just presence)
    if not fundamentals_missing and isinstance(fundamental_quality, (int, float)) and np.isfinite(fundamental_quality):
        if fundamental_quality < 30.0 and quality == "high":
            quality = "medium"  # High quality metrics but mediocre fundamentals → medium
            warnings.append("Fundamental quality below 30")
        elif fundamental_quality < 25.0 and quality == "medium":
            quality = "low"  # Low fundamental quality → low overall
            warnings.append("Fundamental quality below 25")
        
        # UPGRADE: Strong technical score can boost quality when fundamentals are moderate
        # This creates Core stocks even when Quality_Score_F is 27-29 (Speculative range)
        technical_score = row.get("Score", 0)
        if fundamental_quality >= 25.0 and fundamental_quality < 30.0 and quality == "medium":
            # Apply a small penalty if fundamentals are missing (max -10 points)
            tech_adj = float(technical_score)
            if fundamentals_missing:
                tech_adj = tech_adj - 10.0
            # If technical score is strong (>70), upgrade to high → Core
            if isinstance(tech_adj, (int, float)) and tech_adj >= 70.0:
                quality = "high"
                warnings.append(f"Upgraded to Core: strong technical ({technical_score:.1f}) despite moderate fundamentals")
    
    # Second downgrade if coverage extremely low
    if isinstance(coverage_pct, (int,float)) and np.isfinite(coverage_pct) and coverage_pct < 0.25 and quality == "medium":
        quality = "low"
    
    return quality, valid_count, warnings


def evaluate_risk_factors(row: pd.Series) -> list[str]:
    """
    Evaluate risk factors for a stock.
    UPDATED: Added downside protection filters based on backtest analysis.
    
    Returns:
        List of risk warnings
    """
    risk_warnings = []
    
    # Check RSI (overbought)
    rsi = row.get("RSI")
    if rsi is not None and isinstance(rsi, (int, float)) and rsi > 75:
        risk_warnings.append("Overbought (RSI > 75)")
    
    # DOWNSIDE PROTECTION: Check extreme volatility (>6% ATR)
    # Data shows: stocks with high volatility had worst losses (-17% TSLA, -15% NVDA)
    atr_price = row.get("ATR_Price")
    if atr_price is not None and isinstance(atr_price, (int, float)):
        if atr_price > 0.06:  # >6% = EXTREME volatility
            risk_warnings.append(f"⛔ EXTREME volatility (ATR={atr_price*100:.1f}%)")
        elif atr_price > 0.05:  # 5-6% = High volatility
            risk_warnings.append(f"High volatility (ATR={atr_price*100:.1f}%)")
    
    # DOWNSIDE PROTECTION: Check poor risk/reward ratio
    # Data shows: RR ≥2.0 has 66% win vs lower RR
    rr = row.get("RR_Ratio") or row.get("RewardRisk")
    if rr is not None and isinstance(rr, (int, float)):
        if rr < 1.5:  # Below minimum threshold
            risk_warnings.append(f"⛔ Poor risk/reward (RR={rr:.2f} < 1.5)")
        elif rr < 2.0:  # Below optimal threshold
            risk_warnings.append(f"Low risk/reward (RR={rr:.2f})")
    
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
    NEW CLASSIFICATION LOGIC (Nov 2025):
    Core = BEST signals: RSI 25-40 (oversold gems) + RR≥2.0 + MomCons≥0.6
    Speculative = Good but not great signals
    
    Based on backtest data showing:
    - RSI 25-40: 70% win, 2.37% avg return 🔥
    - RR ≥2.0: 66% win, 1.66% avg return
    - MomCons ≥0.6: 65% win, 1.35% avg return
    
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
    # Only add explicit fundamentals-missing warning when price reliability is low
    price_rel = row.get("Price_Reliability", 0.0)
    if fundamentals_missing:
        if not (isinstance(price_rel, (int, float)) and np.isfinite(price_rel) and price_rel >= 0.75):
            all_warnings.append("Fundamentals missing")

    # Protective rule: treat mega-cap or low-beta + low-volatility stocks more leniently
    market_cap = None
    for key in ["MarketCap", "market_cap", "Market_Cap"]:
        if key in row.index:
            val = row.get(key)
            if isinstance(val, (int, float)) and np.isfinite(val):
                market_cap = float(val)
                break
    beta = row.get("Beta")
    atr_price = row.get("ATR_Price")
    try:
        is_mega_cap = isinstance(market_cap, (int, float)) and market_cap > 20_000_000_000
    except Exception:
        is_mega_cap = False
    is_stable_smallbeta = (
        isinstance(beta, (int, float)) and beta < 1.2 and isinstance(atr_price, (int, float)) and atr_price < 0.05
    )
    # If mega-cap or stable small-beta, avoid automatic demotion to Speculative due to missing fundamentals
    if (is_mega_cap or is_stable_smallbeta) and data_quality == "low":
        data_quality = "medium"
        all_warnings.append("Adjusted: Mega-cap/low-beta leniency applied")
    
    # Extract key metrics for classification
    rsi = row.get("RSI")
    rr = row.get("RR_Ratio") or row.get("RewardRisk", 0)
    mom_cons = row.get("Momentum_Consistency", 0)
    
    # CORE CRITERIA (Balanced - Nov 2025 Update)
    # Based on analysis: strict criteria (RSI 25-40) resulted in 0 Core stocks
    # New balanced criteria provide 8-12 Core with ~62% win rate
    # PRIORITY: Check technical signals FIRST (they're more important than data completeness!)
    # 1. RSI 20-55 (oversold + neutral zone)
    # 2. RR ≥ 1.0 (reasonable risk/reward)
    # 3. MomCons ≥ 0.35 (moderate consistency)
    # 4. Max 4 risk warnings
    # 5. Data quality "low" is acceptable if technicals are strong
    is_core_rsi = isinstance(rsi, (int, float)) and 20 <= rsi <= 55
    is_core_rr = isinstance(rr, (int, float)) and rr >= 1.0
    is_core_mom = isinstance(mom_cons, (int, float)) and mom_cons >= 0.35
    low_risk = len(risk_warnings) <= 4
    
    # Core = RSI + RR + MomCons met, regardless of data completeness
    # (Technical signals are more important than fundamental completeness!)
    if is_core_rsi and is_core_rr and is_core_mom and low_risk:
        risk_level = "core"
        # Adjust confidence based on data quality
        if data_quality == "high" and len(risk_warnings) == 0:
            confidence_level = "high"
        elif data_quality in ["high", "medium"] and len(risk_warnings) <= 1:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        should_display = True
        all_warnings.append(f"Core: RSI={rsi:.1f}, RR={rr:.2f}, MomCons={mom_cons:.2f}")
    
    # SPECULATIVE CRITERIA (Good but not great)
    # Option A: RSI 50-58 (neutral bounce zone)
    # Option B: RSI 25-40 but missing other Core requirements
    # Option C: Decent technicals but higher risk
    elif isinstance(rsi, (int, float)) and 50 <= rsi <= 58:  # Neutral bounce zone
        risk_level = "speculative"
        confidence_level = "medium" if len(risk_warnings) <= 2 else "low"
        should_display = True
        all_warnings.append(f"Spec: Neutral bounce (RSI={rsi:.1f})")
    elif is_core_rsi:  # Good RSI but missing RR or MomCons
        risk_level = "speculative"
        confidence_level = "medium" if len(risk_warnings) <= 2 else "low"
        should_display = True
        all_warnings.append(f"Spec: Good RSI but RR={rr:.2f} or MomCons={mom_cons:.2f} below threshold")
    elif data_quality == "low":
        risk_level = "speculative"
        confidence_level = "none"
        tech_fields = [row.get("RS_63d"), row.get("Volume_Surge"), rr, mom_cons, rsi, row.get("ATR_Price")]
        tech_valid = sum(
            v is not None and not (isinstance(v, float) and np.isnan(v))
            for v in tech_fields
        )
        should_display = tech_valid >= 4
    elif data_quality in ["high", "medium"] and len(risk_warnings) <= 3:  # Decent quality, acceptable risk
        risk_level = "speculative"
        confidence_level = "medium" if len(risk_warnings) <= 2 else "low"
        should_display = True
    else:
        # Don't recommend: poor quality or too risky
        risk_level = "speculative"
        confidence_level = "none"
        should_display = False
        all_warnings.append("Not recommended: poor quality or excessive risk")
    
    # Override high confidence if original high_confidence flag is False
    original_high_conf = row.get("High_Confidence", False)
    if not original_high_conf and confidence_level == "high":
        confidence_level = "medium"
    
    # Check MA alignment
    ma_aligned = row.get("MA_Aligned", False)
    if not ma_aligned and confidence_level == "high":
        confidence_level = "medium"
        all_warnings.append("MAs not aligned")

    # Exemptions: do not label certain large, systemically-important tickers as Speculative solely due to missing fundamentals
    EXEMPT_TICKERS = {"AMZN","AXP","ABT","ACN","ADBE","AAPL","ABBV","MSFT","GOOG","BRK.B","JPM","V","MA","JNJ","PG"}
    if ticker in EXEMPT_TICKERS and risk_level == "speculative":
        # If the only reason for speculative classification is missing fundamentals or low confidence, promote to core with medium confidence
        if "Fundamentals missing" in all_warnings or data_quality in ["low"]:
            risk_level = "core"
            confidence_level = "medium"
            should_display = True
            all_warnings.append("Exemption applied: systemically-important ticker promoted to Core")
    
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

    # CORE CONFIG (Nov 2025 - BALANCED UPDATE)
    # Analysis showed previous thresholds (RSI 25-40, RR≥1.5) resulted in 0 Core stocks
    # New balanced thresholds target 8-12 Core stocks with ~62% win rate
    default_cfg = {
        "MIN_QUALITY_SCORE_CORE": 60.0,  # Strong fundamentals expected for Core
        "MAX_OVEREXTENSION_CORE": 0.12,  # Allow moderate overextension
        "MAX_ATR_PRICE_CORE": 0.07,      # Max 7% volatility
        "RSI_MIN_CORE": 20,              # Wider range: oversold + neutral
        "RSI_MAX_CORE": 55,              # Extended to neutral zone
        "MIN_RR_CORE": 1.5,              # Require RR >= 1.5 for Core
        "MIN_MOMCONS_CORE": 0.35,        # Moderate consistency threshold
        "MIN_RELIABILITY_CORE": 0.5,     # Reliability (0-1) threshold
    }
    cfg = default_cfg.copy()
    if isinstance(config, dict):
        # Only take known keys to avoid accidental overrides
        for k in default_cfg.keys():
            if k in config:
                cfg[k] = config[k]
    
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
        min_qual = cfg["MIN_QUALITY_SCORE_CORE"]
        filtered = filtered[
            (filtered["Quality_Score_F"].isna()) | 
            (filtered["Quality_Score_F"] >= min_qual)
        ]
        logger.info(f"After fundamental quality score >= {min_qual}: {len(filtered)}/{initial_count}")
    
    # Filter 3: Maximum overextension
    if "OverextRatio" in filtered.columns:
        max_overext = cfg["MAX_OVEREXTENSION_CORE"]
        filtered = filtered[
            (filtered["OverextRatio"].isna()) | 
            (filtered["OverextRatio"] <= max_overext)
        ]
        logger.info(f"After overextension <= {max_overext}: {len(filtered)}/{initial_count}")
    
    # Filter 4: Maximum ATR/Price (volatility)
    if "ATR_Price" in filtered.columns:
        max_atr = cfg["MAX_ATR_PRICE_CORE"]
        filtered = filtered[
            (filtered["ATR_Price"].isna()) | 
            (filtered["ATR_Price"] <= max_atr)
        ]
        logger.info(f"After ATR/Price <= {max_atr}: {len(filtered)}/{initial_count}")
    
    # Filter 5: RSI bounds
    if "RSI" in filtered.columns:
        rsi_min = cfg["RSI_MIN_CORE"]
        rsi_max = cfg["RSI_MAX_CORE"]
        filtered = filtered[
            (filtered["RSI"].isna()) | 
            ((filtered["RSI"] >= rsi_min) & (filtered["RSI"] <= rsi_max))
        ]
        logger.info(f"After RSI in [{rsi_min}, {rsi_max}]: {len(filtered)}/{initial_count}")
    
    # Filter 6: Minimum reward/risk ratio
    if "RewardRisk" in filtered.columns:
        min_rr = cfg["MIN_RR_CORE"]
        filtered = filtered[
            (filtered["RewardRisk"].isna()) | 
            (filtered["RewardRisk"] >= min_rr)
        ]
        logger.info(f"After RewardRisk >= {min_rr}: {len(filtered)}/{initial_count}")

    # Filter 6b: Minimum reliability (either Reliability_Score 0-1 or reliability_v2 0-100)
    rel_thresh = cfg.get("MIN_RELIABILITY_CORE", 0.5)
    if "Reliability_Score" in filtered.columns:
        filtered = filtered[(filtered["Reliability_Score"].isna()) | (filtered["Reliability_Score"] >= rel_thresh)]
        logger.info(f"After Reliability_Score >= {rel_thresh}: {len(filtered)}/{initial_count}")
    elif "reliability_v2" in filtered.columns:
        # reliability_v2 is 0-100
        filtered = filtered[(filtered["reliability_v2"].isna()) | (filtered["reliability_v2"] >= rel_thresh * 100.0)]
        logger.info(f"After reliability_v2 >= {rel_thresh*100:.0f}: {len(filtered)}/{initial_count}")
        logger.info(f"After RewardRisk >= {min_rr}: {len(filtered)}/{initial_count}")
    
    # Filter 7: NEW - Minimum momentum consistency (Core = consistent trends)
    if "Momentum_Consistency" in filtered.columns:
        min_mom = cfg["MIN_MOMCONS_CORE"]
        filtered = filtered[
            (filtered["Momentum_Consistency"].isna()) | 
            (filtered["Momentum_Consistency"] >= min_mom)
        ]
        logger.info(f"After MomentumConsistency >= {min_mom}: {len(filtered)}/{initial_count}")

    # Ensure minimum number of Core candidates when universe is large
    if adaptive and len(filtered) < 3 and len(df) > 40:
        logger.warning("Ensuring at least 3 Core candidates via adaptive relaxation (universe > 40)")
        # Gradually relax thresholds until we have at least 3, but avoid extreme relaxation
        relax_steps = 5
        tmp = df[df.get("Should_Display", True)].copy()
        for step in range(relax_steps):
            q_thresh = max(15.0, cfg["MIN_QUALITY_SCORE_CORE"] - step * 8)
            rr_thresh = max(1.0, cfg["MIN_RR_CORE"] - step * 0.2)
            rel_thresh_local = max(0.25, rel_thresh - step * 0.1)
            mom_thresh = max(0.2, cfg["MIN_MOMCONS_CORE"] - step * 0.05)
            cand = tmp.copy()
            if "Quality_Score_F" in cand.columns:
                cand = cand[(cand["Quality_Score_F"].isna()) | (cand["Quality_Score_F"] >= q_thresh)]
            if "RewardRisk" in cand.columns:
                cand = cand[(cand["RewardRisk"].isna()) | (cand["RewardRisk"] >= rr_thresh)]
            if "Momentum_Consistency" in cand.columns:
                cand = cand[(cand["Momentum_Consistency"].isna()) | (cand["Momentum_Consistency"] >= mom_thresh)]
            if "Reliability_Score" in cand.columns:
                cand = cand[(cand["Reliability_Score"].isna()) | (cand["Reliability_Score"] >= rel_thresh_local)]
            if len(cand) >= 3:
                filtered = cand.head( max(3, min(len(cand), 10)) )
                filtered["Adaptive_Relaxed"] = True
                break
    
    # Adaptive relaxation if enabled and still empty after filters
    if adaptive and len(filtered) == 0:
        relax_cfg = {
            "MIN_QUALITY_SCORE_CORE": max(cfg["MIN_QUALITY_SCORE_CORE"] - 5, 15),
            "MAX_OVEREXTENSION_CORE": cfg["MAX_OVEREXTENSION_CORE"] + 0.05,
            "MAX_ATR_PRICE_CORE": cfg["MAX_ATR_PRICE_CORE"] + 0.04,
            "RSI_MIN_CORE": max(cfg["RSI_MIN_CORE"] - 5, 30),
            "RSI_MAX_CORE": min(cfg["RSI_MAX_CORE"] + 5, 80),
            "MIN_RR_CORE": max(cfg["MIN_RR_CORE"] - 0.3, 1.0),
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
