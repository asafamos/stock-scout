"""
V2 Risk Engine - Risk-Aware and Reliability-Aware Position Sizing
==================================================================

This module implements strict risk gates and reliability scoring
to prevent allocating capital to low-quality opportunities.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_reliability_v2(
    row: pd.Series,
    fundamental_data: Optional[Dict] = None,
    price_sources: Optional[Dict] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate MEANINGFUL reliability score (0-100) based on:
    1. Fundamental completeness (% of fields filled)
    2. Number of fundamental sources (0-4+)
    3. Price source reliability (count + variance)
    
    Returns:
        (reliability_v2: 0-100, details_dict)
    """
    details = {
        "fundamental_completeness_pct": 0.0,
        "fund_sources_count": 0,
        "price_sources_count": 0,
        "price_variance_penalty": 0.0,
        "fund_disagreement": 0.0
    }
    
    # 1) FUNDAMENTAL COMPLETENESS (0-100 scale)
    # Check all important fundamental fields
    fundamental_fields = [
        "PE_f", "PS_f", "PBRatio", "ROE_f", "ROIC_f", "GM_f", "ProfitMargin",
        "DE_f", "RevG_f", "EPSG_f", "RevenueGrowthYoY", "EPSGrowthYoY",
        "Fundamental_S", "Quality_Score_F", "Growth_Score_F", "Valuation_Score_F"
    ]
    
    filled_count = 0
    for field in fundamental_fields:
        val = row.get(field)
        if pd.notna(val) and val not in [0, "", "N/A"]:
            filled_count += 1
    
    completeness_pct = (filled_count / len(fundamental_fields)) * 100
    details["fundamental_completeness_pct"] = completeness_pct
    
    # 2) NUMBER OF FUNDAMENTAL SOURCES
    fund_sources = 0
    for src in ["Fund_from_FMP", "Fund_from_Alpha", "Fund_from_Finnhub", 
                "Fund_from_SimFin", "Fund_from_EODHD"]:
        if row.get(src, False):
            fund_sources += 1
    
    # Also check the count column if it exists
    if "Fundamental_Sources_Count" in row.index:
        fund_sources = max(fund_sources, int(row.get("Fundamental_Sources_Count", 0)))
    
    details["fund_sources_count"] = fund_sources
    
    # 3) PRICE SOURCE RELIABILITY
    price_sources_count = int(row.get("Price_Sources_Count", 1))
    details["price_sources_count"] = price_sources_count
    
    # Price variance penalty
    price_std = row.get("Price_STD", 0.0)
    price_mean = row.get("Price_Mean", row.get("Price_Yahoo", 0.0))
    
    if price_mean > 0 and pd.notna(price_std):
        cv = (price_std / price_mean) * 100  # Coefficient of variation
        # CV > 2% is concerning, > 5% is very bad
        variance_penalty = min(cv / 5.0, 1.0) * 30  # Up to 30 point penalty
        details["price_variance_penalty"] = variance_penalty
    else:
        variance_penalty = 0.0
    
    # 4) CALCULATE RELIABILITY V2 (0-100)
    # Weights:
    # - 40% completeness
    # - 30% fund sources
    # - 20% price sources
    # - 10% variance penalty
    
    # Completeness score (already 0-100)
    completeness_score = completeness_pct
    
    # Fund sources score (0 sources = 0, 1 = 40, 2 = 70, 3 = 85, 4+ = 100)
    if fund_sources == 0:
        fund_sources_score = 0.0
    elif fund_sources == 1:
        fund_sources_score = 40.0
    elif fund_sources == 2:
        fund_sources_score = 70.0
    elif fund_sources == 3:
        fund_sources_score = 85.0
    else:  # 4+
        fund_sources_score = 100.0
    
    # Price sources score (1 = 50, 2 = 70, 3+ = 100)
    if price_sources_count <= 1:
        price_sources_score = 50.0
    elif price_sources_count == 2:
        price_sources_score = 70.0
    else:  # 3+
        price_sources_score = 100.0
    
    # Weighted combination
    reliability_v2 = (
        0.40 * completeness_score +
        0.30 * fund_sources_score +
        0.20 * price_sources_score +
        0.10 * (100 - variance_penalty)
    )
    
    reliability_v2 = float(np.clip(reliability_v2, 0, 100))
    
    return reliability_v2, details


def calculate_risk_gate_v2(
    rr_ratio: float,
    reliability_v2: float,
    fund_sources_count: int = 0,
    quality_score: float = 50.0
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Risk/Reward gate that BLOCKS or REDUCES position sizing.
    
    Args:
        rr_ratio: Risk/Reward ratio (target: >= 2.0)
        reliability_v2: Reliability score 0-100
        quality_score: Fundamental quality score 0-100
    
    Returns:
        (gate_status, penalty_factor, details)
        
    Gate statuses:
        - "blocked": RR < 1.0 OR reliability < 15 → penalty = 0.0
        - "severely_reduced": 1.0 <= RR < 1.5 OR reliability < 30 → penalty = 0.3
        - "reduced": 1.5 <= RR < 2.0 OR reliability < 50 → penalty = 0.6
        - "full": RR >= 2.0 AND reliability >= 50 → penalty = 1.0
    """
    details = {
        "rr_ratio": rr_ratio,
        "reliability_v2": reliability_v2,
        "quality_score": quality_score,
        "rr_gate": "unknown",
        "reliability_gate": "unknown",
        "quality_gate": "unknown"
    }
    
    # Check R/R ratio (strict)
    if rr_ratio < 1.0:
        rr_gate = "blocked"
        rr_penalty = 0.0
    elif rr_ratio < 1.5:
        rr_gate = "reduced"
        rr_penalty = 0.5
    elif rr_ratio < 2.0:
        rr_gate = "reduced"
        rr_penalty = 0.7
    else:
        rr_gate = "full"
        rr_penalty = 1.0
    
    details["rr_gate"] = rr_gate
    
    # Check reliability (strict thresholds)
    if reliability_v2 < 10:
        reliability_gate = "blocked"
        reliability_penalty = 0.0
    elif reliability_v2 < 30:
        reliability_gate = "reduced"
        reliability_penalty = 0.5
    elif reliability_v2 < 50:
        reliability_gate = "reduced"
        reliability_penalty = 0.7
    else:
        reliability_gate = "full"
        reliability_penalty = 1.0
    
    details["reliability_gate"] = reliability_gate
    
    # Check quality
    if quality_score < 20:
        quality_gate = "poor"
        quality_penalty = 0.5
    elif quality_score < 40:
        quality_gate = "fair"
        quality_penalty = 0.75
    else:
        quality_gate = "good"
        quality_penalty = 1.0
    
    details["quality_gate"] = quality_gate
    
    # Special strict rule: if no fund sources -> BLOCK
    if fund_sources_count <= 0:
        details["reason"] = "no_fund_sources"
        details["combined_penalty"] = 0.0
        return "blocked", 0.0, details

    # Combined conviction factor: more restrictive of rr vs reliability, then scaled by quality
    combined_penalty = min(rr_penalty, reliability_penalty) * quality_penalty
    details["combined_penalty"] = combined_penalty

    # Map numeric combined_penalty to status (strict mapping)
    if combined_penalty == 0.0:
        gate_status = "blocked"
    elif combined_penalty < 0.4:
        gate_status = "reduced"
    elif combined_penalty < 0.85:
        gate_status = "reduced"
    else:
        gate_status = "full"

    # Reason string for CSV/UI
    reason = f"rr_gate={rr_gate};rel_gate={reliability_gate};quality={quality_gate}"
    details["reason"] = reason

    return gate_status, float(combined_penalty), details


def calculate_position_size_v2(
    base_allocation: float,
    risk_gate_status: str,
    risk_gate_penalty: float,
    risk_level: str,
    reliability_v2: float,
    budget_total: float,
    min_position: float = 50.0,
    max_core_pct: float = 15.0,
    max_speculative_pct: float = 3.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate V2 position size with strict risk and reliability adjustments.
    
    Args:
        base_allocation: Initial allocation from v1 logic
        risk_gate_status: "blocked", "severely_reduced", "reduced", "full"
        risk_gate_penalty: 0.0 to 1.0
        risk_level: "core" or "speculative"
        reliability_v2: 0-100
        budget_total: Total budget
        min_position: Minimum position size
        max_core_pct: Max % of budget for core stocks
        max_speculative_pct: Max % of budget for speculative stocks
    
    Returns:
        (buy_amount_v2, details)
    """
    details = {
        "base_allocation": base_allocation,
        "risk_gate_penalty": risk_gate_penalty,
        "reliability_factor": 0.0,
        "risk_level_cap": 0.0,
        "final_amount": 0.0
    }
    
    # If blocked, return minimal or zero
    if risk_gate_status == "blocked":
        details["final_amount"] = 0.0
        return 0.0, details
    
    # Apply risk gate penalty
    amount = base_allocation * risk_gate_penalty
    
    # Apply reliability factor (scale by reliability_v2 / 100)
    reliability_factor = reliability_v2 / 100.0
    amount = amount * (0.5 + 0.5 * reliability_factor)  # At least 50% scaling
    details["reliability_factor"] = reliability_factor
    
    # Apply risk level caps
    if risk_level == "speculative":
        # Speculative stocks: max 3% of budget (or custom)
        max_speculative = budget_total * (max_speculative_pct / 100.0)
        amount = min(amount, max_speculative)
        details["risk_level_cap"] = max_speculative
    else:
        # Core stocks: max 15% of budget (or custom)
        max_core = budget_total * (max_core_pct / 100.0)
        amount = min(amount, max_core)
        details["risk_level_cap"] = max_core
    
    # Enforce minimum
    if amount > 0:
        amount = max(amount, min_position)
    
    details["final_amount"] = amount
    
    return float(amount), details


def apply_v2_conviction_adjustments(
    base_conviction: float,
    reliability_v2: float,
    risk_gate_penalty: float,
    ml_probability: Optional[float] = None,
    enable_ml_boost: bool = True
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Apply reliability and risk gate adjustments to conviction score.
    ML boost is added AFTER risk/reliability penalties.
    
    Args:
        base_conviction: Base conviction 0-100
        reliability_v2: Reliability score 0-100
        risk_gate_penalty: 0.0 to 1.0
        ml_probability: ML prediction probability (0-1)
        enable_ml_boost: Whether to apply ML boost
    
    Returns:
        (conviction_v2_final, ml_boost, details)
    """
    details = {
        "base_conviction": base_conviction,
        "reliability_adjusted": 0.0,
        "risk_adjusted": 0.0,
        "ml_boost": 0.0,
        "final": 0.0
    }
    
    # Step 1: Apply reliability penalty
    # Low reliability reduces conviction
    reliability_factor = 0.5 + 0.5 * (reliability_v2 / 100.0)  # 0.5 to 1.0
    conviction_after_reliability = base_conviction * reliability_factor
    details["reliability_adjusted"] = conviction_after_reliability
    
    # Step 2: Apply risk gate penalty
    conviction_after_risk = conviction_after_reliability * risk_gate_penalty
    details["risk_adjusted"] = conviction_after_risk
    
    # Step 3: Apply ML boost (bounded ±10%)
    ml_boost = 0.0
    if enable_ml_boost and ml_probability is not None and np.isfinite(ml_probability):
        # ML boost: -10% to +10% based on probability
        # probability 0.5 = no boost, 0.0 = -10%, 1.0 = +10%
        ml_boost = (ml_probability - 0.5) * 20.0  # -10 to +10
        ml_boost = np.clip(ml_boost, -10, 10)
        details["ml_boost"] = ml_boost
    
    # Apply ML boost to risk-adjusted conviction
    conviction_v2_final = conviction_after_risk + ml_boost
    conviction_v2_final = float(np.clip(conviction_v2_final, 0, 100))
    details["final"] = conviction_v2_final
    
    return conviction_v2_final, ml_boost, details


def score_ticker_v2_enhanced(
    ticker: str,
    row: pd.Series,
    budget_total: float = 5000.0,
    min_position: float = 50.0,
    enable_ml: bool = True
) -> Dict[str, Any]:
    """
    Enhanced V2 scoring with full risk and reliability gates.
    
    Returns comprehensive dict with all V2 metrics.
    """
    from core.ml_integration import get_ml_prediction, load_ml_model
    
    result = {
        "ticker": ticker,
        # V2 Reliability
        "reliability_v2": 0.0,
        "fund_sources_used_v2": [],
        "price_sources_used_v2": [],
        "fund_sources_count_v2": 0,
        "price_sources_count_v2": 0,
        "fund_completeness_pct": 0.0,
        "price_variance_penalty": 0.0,
        "fund_disagreement_score_v2": 0.0,
        "price_variance_score_v2": 0.0,
        
        # V2 Risk Gate
        "rr_ratio_v2": 0.0,
        "risk_gate_status_v2": "unknown",
        "risk_gate_penalty_v2": 0.0,
        
        # V2 Conviction
        "conviction_v2_base": 50.0,
        "conviction_v2_final": 50.0,
        "ml_probability_v2": None,
        "ml_boost_v2": 0.0,
        
        # V2 Position Sizing
        "buy_amount_v2": 0.0,
        "shares_to_buy_v2": 0,
        
        # Metadata
        "risk_level": row.get("Risk_Level", "unknown"),
        "quality_score": row.get("Quality_Score", 0.0)
    }
    
    try:
        # 1) Calculate Reliability V2
        reliability_v2, rel_details = calculate_reliability_v2(row)
        result["reliability_v2"] = reliability_v2
        result["fund_sources_count_v2"] = rel_details["fund_sources_count"]
        result["price_sources_count_v2"] = rel_details["price_sources_count"]
        result["fund_completeness_pct"] = rel_details["fundamental_completeness_pct"]
        result["price_variance_penalty"] = rel_details["price_variance_penalty"]
        
        # 2) Calculate Risk Gate V2
        rr_ratio = row.get("RewardRisk", row.get("RR_Ratio", 0.0))
        quality_score = row.get("Quality_Score", row.get("Quality_Score_F", 50.0))
        
        # Pass the detected number of fund sources to the gate for strict blocking
        fund_src_count = int(rel_details.get("fund_sources_count", 0))
        gate_status, gate_penalty, gate_details = calculate_risk_gate_v2(
            rr_ratio, reliability_v2, fund_src_count, quality_score
        )
        result["rr_ratio_v2"] = rr_ratio
        result["risk_gate_status_v2"] = gate_status
        result["risk_gate_penalty_v2"] = gate_penalty
        result["risk_gate_reason_v2"] = gate_details.get("reason", "")
        
        # 3) Calculate Base Conviction (use existing score as base)
        base_conviction = row.get("Score", row.get("Score_Tech", 50.0))
        result["conviction_v2_base"] = base_conviction
        
        # 4) Get ML Prediction
        ml_prob = None
        if enable_ml:
            try:
                model = load_ml_model()
                if model is not None:
                    # Prepare features (simplified - would need full feature prep)
                    ml_prob = row.get("ML_Probability", None)
                    if ml_prob is None and "ML_Confidence" in row.index:
                        ml_prob = row.get("ML_Confidence", 0.5)
            except Exception as e:
                logger.debug(f"ML prediction failed for {ticker}: {e}")
        
        result["ml_probability_v2"] = ml_prob
        
        # 5) Apply V2 Conviction Adjustments
        conviction_final, ml_boost, conv_details = apply_v2_conviction_adjustments(
            base_conviction, reliability_v2, gate_penalty, ml_prob, enable_ml
        )
        result["conviction_v2_final"] = conviction_final
        result["ml_boost_v2"] = ml_boost
        
        # 6) Calculate V2 Position Size
        base_allocation = row.get("סכום קנייה ($)", min_position)
        risk_level = row.get("Risk_Level", "speculative")
        
        buy_amount_v2, size_details = calculate_position_size_v2(
            base_allocation,
            gate_status,
            gate_penalty,
            risk_level,
            reliability_v2,
            budget_total,
            min_position
        )
        result["buy_amount_v2"] = buy_amount_v2
        
        # Calculate shares
        unit_price = row.get("Unit_Price", row.get("Price_Yahoo", 0.0))
        if unit_price > 0:
            shares_v2 = int(buy_amount_v2 / unit_price)
        else:
            shares_v2 = 0
        result["shares_to_buy_v2"] = shares_v2

        # Populate sources lists (JSON-friendly)
        fund_sources_used = []
        for src in ["Fund_from_FMP", "Fund_from_Alpha", "Fund_from_Finnhub", "Fund_from_SimFin", "Fund_from_EODHD"]:
            if bool(row.get(src, False)):
                fund_sources_used.append(src)
        result["fund_sources_used_v2"] = fund_sources_used

        price_sources_used = []
        for pcol in ["Price_Marketstack", "Price_NasdaqDL", "Price_EODHD", "Price_Polygon", "Price_Tiingo", "Price_Yahoo"]:
            val = row.get(pcol, None)
            if pd.notna(val) and val not in [0, "", None]:
                price_sources_used.append(pcol)
        result["price_sources_used_v2"] = price_sources_used

        # Fund disagreement score: complement of completeness (0-100)
        fund_disagreement = float(np.clip(100.0 - rel_details.get("fundamental_completeness_pct", 100.0), 0, 100))
        result["fund_disagreement_score_v2"] = fund_disagreement

        # Price variance score: invert the variance penalty (0-100)
        pv_pen = rel_details.get("price_variance_penalty", 0.0)
        price_variance_score = float(np.clip(100.0 - pv_pen, 0, 100))
        result["price_variance_score_v2"] = price_variance_score
        
    except Exception as e:
        logger.error(f"V2 enhanced scoring failed for {ticker}: {e}")
    
    return result
