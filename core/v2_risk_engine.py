"""
V2 Risk Engine - Risk-Aware and Reliability-Aware Position Sizing
==================================================================

Narrow responsibility:
- Translate existing score + risk info into a position risk factor.
- Avoid re-implementing numeric scoring or complex rule-based combos.

Canonical inputs expected from the pipeline/classification:
- FinalScore_20d (0-100)
- RiskClass ('CORE'|'SPEC'|'REJECT') and SafetyBlocked (bool)
- RR (risk/reward ratio, canonicalized)
- ReliabilityScore (0-100)
- Optional nuance: ATR/Price (volatility), Beta, RiskMeter.

Legacy helpers (calculate_reliability_v2, calculate_risk_gate_v2, etc.)
are retained for backward compatibility but new flows should prefer
compute_position_risk_factor(row) for sizing.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import logging

from core.scoring_config import CONVICTION_WEIGHTS

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# New, canonical position risk factor
# -------------------------------------------------------------
def _get_canonical_rr(row: pd.Series) -> float:
    rr = row.get("RR", None)
    if rr is None:
        rr = row.get("RR_Ratio", row.get("RewardRisk", None))
    try:
        return float(rr) if rr is not None and np.isfinite(rr) else np.nan
    except Exception:
        return np.nan


def _get_canonical_reliability(row: pd.Series) -> float:
    rel = row.get("ReliabilityScore", None)
    if rel is None:
        rel = row.get("reliability_v2", row.get("Reliability_v2", row.get("Reliability_Score", None)))
    try:
        val = float(rel) if rel is not None and np.isfinite(rel) else np.nan
        # Legacy 0-1 scaling
        if np.isfinite(val) and val <= 1.0:
            val *= 100.0
        return val
    except Exception:
        return np.nan


def compute_position_risk_factor(row: pd.Series) -> float:
    """
    Returns a factor in [0.0, 1.0] to scale nominal position size.

    Policy:
    - Hard blocks: if RiskClass == 'REJECT' or SafetyBlocked → 0.0
    - Base factor by RiskClass:
        CORE → 1.00
        SPEC → 0.65
      (fallback to legacy Risk_Level if RiskClass missing)
    - Reliability adjustment (ReliabilityScore 0-100):
        <10 → 0.0 (force block)
        <30 → x0.5
        <50 → x0.7
        else → x1.0
    - RR adjustment (RR):
        <1.2 → x0.80
        1.2–1.5 → x0.90
        1.5–2.5 → x1.00
        ≥2.5 → x1.05
    - Volatility/Beta nuance (mild, monotonic):
        ATR/Price > 8% → x0.85; >5% → x0.93
        Beta > 1.8 → x0.90

    The result is clamped to [0, 1].
    """
    # Coil/VCP override detection
    coil_vcp_override = False
    try:
        coil_flag = bool(row.get("Coil_Bonus", False))
    except Exception:
        coil_flag = False
    try:
        vcp_score = float(row.get("Volatility_Contraction_Score", 0.0) or 0.0)
    except Exception:
        vcp_score = 0.0
    coil_vcp_override = coil_flag or (vcp_score >= 0.6)

    # Safety gates
    risk_class = row.get("RiskClass", None)
    if risk_class is None:
        # Map legacy
        rl = str(row.get("Risk_Level", "speculative")).lower()
        risk_class = "CORE" if rl == "core" else "SPEC"
    safety_blocked = bool(row.get("SafetyBlocked", False))
    if (str(risk_class).upper() == "REJECT" or safety_blocked) and not coil_vcp_override:
        return 0.0

    # Base factor by class
    base = 1.0 if str(risk_class).upper() == "CORE" else 0.65

    # Reliability adjustment
    rel = _get_canonical_reliability(row)
    if np.isfinite(rel):
        if rel < 10.0:
            # Do not hard-block coil/VCP; apply heavy reduction instead
            if coil_vcp_override:
                base *= 0.50
            else:
                return 0.0
        elif rel < 30.0:
            base *= 0.50
        elif rel < 50.0:
            base *= 0.70
        else:
            base *= 1.00

    # RR adjustment
    rr = _get_canonical_rr(row)
    if np.isfinite(rr):
        if rr < 1.2:
            base *= 0.80
        elif rr < 1.5:
            base *= 0.90
        elif rr < 2.5:
            base *= 1.00
        else:
            base *= 1.05

    # Volatility/Beta nuance
    vol = row.get("ATR_Price", row.get("ATR_Pct", None))
    if isinstance(vol, (int, float)) and np.isfinite(vol):
        if vol > 0.08:
            base *= 0.85
        elif vol > 0.05:
            base *= 0.93

    beta = row.get("Beta", None)
    if isinstance(beta, (int, float)) and np.isfinite(beta) and beta > 1.8:
        base *= 0.90

    # Clamp
    base = float(np.clip(base, 0.0, 1.0))
    return base


def calculate_reliability_v2(
    row: pd.Series,
    fundamental_data: Optional[Dict] = None,
    price_sources: Optional[Dict] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Reliability v2 — multi-source trust score in [0, 100].

    Inputs are taken from the unified row and should include:
    - Fundamental_Coverage_Pct: float [0,100] — percent of key fields present
    - Fundamental_Sources_Count: int — number of fundamental providers (FMP/Finnhub/Tiingo/Alpha)
    - Price_Sources_Count: int — number of distinct price providers used
    - Price_STD: float — standard deviation across price providers
    - Price_Mean: float — mean of prices across providers
    - ATR_Price or ATR_Pct: float — volatility proxy (ATR/Price)
    - Quality_Score or Quality_Score_F: float [0,100] — optional trading quality proxy

    Semantics:
    - 0 = very low trust (missing fundamentals, single/zero price source, high disagreement)
    - 100 = very high trust (multiple agreeing price sources, high fundamental coverage, decent liquidity/volatility)

    Formula overview (weighted components):
    - Fundamentals reliability (40%): coverage x source_factor, penalized when sources=0
    - Price reliability (35%): agreement factor from CV (std/mean) and source count
    - Volatility/Liquidity (15%): lower ATR/Price increases reliability, unknown incurs penalty
    - Quality bonus (10%): optional, modest lift for higher quality

    Strong penalties:
    - Fundamental_Sources_Count == 0 or coverage very low → heavy reduction
    - Price_Sources_Count in {0,1} → capped contribution and added penalty if high variance
    - High CV (>5%) → large penalty; CV computed as std/mean

    Returns:
        (reliability_v2: float in [0,100], details_dict: component breakdown)
    """
    details = {
        "fundamental_completeness_pct": 0.0,
        "fund_sources_count": 0,
        "price_sources_count": 0,
        "price_variance_penalty": 0.0,
        "volatility_penalty": 0.0,
        "fund_component": 0.0,
        "price_component": 0.0,
        "vol_component": 0.0,
        "quality_component": 0.0,
    }
    
    # 1) FUNDAMENTAL COMPLETENESS (0-100 scale)
    # Check all important fundamental fields
    # Prefer explicit coverage metric if available
    cov_pct_explicit = row.get("Fundamental_Coverage_Pct", None)
    if cov_pct_explicit is not None and np.isfinite(cov_pct_explicit):
        completeness_pct = float(np.clip(cov_pct_explicit, 0.0, 100.0))
    else:
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
    for src in ["Fund_from_FMP", "Fund_from_Alpha", "Fund_from_Finnhub", "Fund_from_Tiingo"]:
        if bool(row.get(src, False)):
            fund_sources += 1
    # Use explicit count if provided
    if "Fundamental_Sources_Count" in row.index or row.get("Fundamental_Sources_Count") is not None:
        try:
            fund_sources = max(fund_sources, int(row.get("Fundamental_Sources_Count", 0) or 0))
        except Exception:
            pass
    
    details["fund_sources_count"] = fund_sources
    
    # 3) PRICE SOURCE RELIABILITY
    price_sources_count = int(row.get("Price_Sources_Count", row.get("price_sources", 1) or 1))
    details["price_sources_count"] = price_sources_count
    
    # Price variance penalty
    price_std = row.get("Price_STD", row.get("price_std", 0.0))
    price_mean = row.get("Price_Mean", row.get("price_mean", row.get("Price_Yahoo", 0.0)))
    cv = None
    if price_mean and np.isfinite(price_mean) and price_mean > 0 and price_std is not None and np.isfinite(price_std):
        cv = float(price_std) / float(price_mean)
        # Penalty scales steeply after 2% and saturates at 10%
        variance_penalty = float(np.clip((cv - 0.02) / 0.08, 0.0, 1.0)) * 35.0
    else:
        variance_penalty = 20.0 if price_sources_count <= 1 else 10.0  # Unknown variance penalty
    details["price_variance_penalty"] = variance_penalty
    
    # 4) VOLATILITY PENALTY (inverse ATR/Price)
    # Lower volatility = higher reliability
    atr_price = row.get("ATR_Price", row.get("ATR_Pct", np.nan))
    if pd.notna(atr_price) and atr_price > 0:
        # Ideal band roughly 2-4%; harsher outside
        if atr_price > 0.08:
            volatility_penalty = 45.0
        elif atr_price > 0.05:
            volatility_penalty = 30.0
        elif atr_price > 0.03:
            volatility_penalty = 15.0
        elif atr_price > 0.015:
            volatility_penalty = 5.0
        else:
            volatility_penalty = 10.0  # Too quiet can also reduce trust
        details["volatility_penalty"] = volatility_penalty
    else:
        volatility_penalty = 15.0  # Unknown volatility = small penalty
        details["volatility_penalty"] = volatility_penalty
    
    # 5) CALCULATE RELIABILITY V2 (0-100)
    # Weights:
    # - 30% completeness
    # - 25% fund sources
    # - 20% price sources
    # - 15% volatility (inverse)
    # - 10% price variance
    
    # Fundamentals component: coverage scaled by sources
    # source_factor: 0 sources → 0; 1 → 0.6; 2 → 0.8; 3+ → 1.0
    if fund_sources <= 0:
        source_factor_f = 0.0
    elif fund_sources == 1:
        source_factor_f = 0.6
    elif fund_sources == 2:
        source_factor_f = 0.8
    else:
        source_factor_f = 1.0
    fund_component = float(np.clip(completeness_pct, 0, 100)) * source_factor_f
    details["fund_component"] = fund_component

    # Price component: agreement (1 - penalty) scaled by sources
    if price_sources_count <= 0:
        source_factor_p = 0.0
    elif price_sources_count == 1:
        source_factor_p = 0.6
    elif price_sources_count == 2:
        source_factor_p = 0.8
    else:
        source_factor_p = 1.0
    price_agreement_score = float(np.clip(100.0 - variance_penalty, 0, 100))
    price_component = price_agreement_score * source_factor_p
    details["price_component"] = price_component

    # Volatility component: invert penalty
    vol_component = float(np.clip(100.0 - volatility_penalty, 0, 100))
    details["vol_component"] = vol_component

    # Quality component: optional small boost
    quality = row.get("Quality_Score", row.get("Quality_Score_F", None))
    if quality is not None and np.isfinite(quality):
        quality_component = float(np.clip(quality, 0, 100))
    else:
        quality_component = 50.0
    details["quality_component"] = quality_component

    # Weighted combination
    reliability_v2 = (
        0.40 * fund_component +
        0.35 * price_component +
        0.15 * vol_component +
        0.10 * quality_component
    ) / 100.0 * 100.0
    
    reliability_v2 = float(np.clip(reliability_v2, 0, 100))
    
    # Map to High/Medium/Low bands
    if reliability_v2 >= 75:
        reliability_band = "High"
    elif reliability_v2 >= 40:
        reliability_band = "Medium"
    else:
        reliability_band = "Low"
    
    details["reliability_band"] = reliability_band
    
    return reliability_v2, details


def calculate_risk_gate_v2(
    rr_ratio: float,
    reliability_v2: float,
    fund_sources_count: int = 0,
    quality_score: float = 50.0,
    coil_vcp_override: bool = False
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
        if coil_vcp_override:
            # Bypass hard block for coil/VCP; treat as reduced
            rr_gate = "reduced"
            rr_penalty = 0.6
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
        if coil_vcp_override:
            reliability_gate = "reduced"
            reliability_penalty = 0.6
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
        if coil_vcp_override:
            # Allow reduced sizing instead of full block
            details["combined_penalty"] = 0.6
            return "reduced", 0.6, details
        else:
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
    
    # If blocked by gate or safety, return zero
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
    rr_ratio: Optional[float] = None,
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
    
    # Step 3: Incorporate RR as a separate component (15% weight)
    # RR component normalized: rr=2.0 → 100, rr=1.0 → 50, capped at 5.0
    rr_component = None
    if rr_ratio is not None and isinstance(rr_ratio, (int, float)) and not np.isnan(rr_ratio):
        rr_component = float(np.clip(rr_ratio / 2.0, 0.0, 1.0)) * 100.0

    # Step 4: Apply ML boost (bounded ±10%)
    ml_boost = 0.0
    if enable_ml_boost and ml_probability is not None and np.isfinite(ml_probability):
        # ML boost: -10% to +10% based on probability
        # probability 0.5 = no boost, 0.0 = -10%, 1.0 = +10%
        ml_boost = (ml_probability - 0.5) * 20.0  # -10 to +10
        ml_boost = np.clip(ml_boost, -10, 10)
        details["ml_boost"] = ml_boost
    
    # Mix in RR component if available (weights: 85% existing conviction_after_risk, 15% RR)
    if rr_component is not None:
        conviction_mixed = 0.85 * conviction_after_risk + 0.15 * rr_component
    else:
        conviction_mixed = conviction_after_risk

    # Apply ML boost to the mixed conviction
    conviction_v2_final = conviction_mixed + ml_boost
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
    # Unified ML handled via core.ml_20d_inference; legacy ml_integration not used
    
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
        # Coil/VCP override flag
        coil_flag = bool(row.get("Coil_Bonus", False))
        try:
            vcp_score = float(row.get("Volatility_Contraction_Score", 0.0) or 0.0)
        except Exception:
            vcp_score = 0.0
        coil_override = coil_flag or (vcp_score >= 0.6)

        gate_status, gate_penalty, gate_details = calculate_risk_gate_v2(
            rr_ratio, reliability_v2, fund_src_count, quality_score, coil_vcp_override=coil_override
        )
        result["rr_ratio_v2"] = rr_ratio
        result["risk_gate_status_v2"] = gate_status
        result["risk_gate_penalty_v2"] = gate_penalty
        result["risk_gate_reason_v2"] = gate_details.get("reason", "")
        
        # 3) Calculate Base Conviction (use existing score as base)
        # Compose base conviction from fund/tech/RR/reliability with weights:
        # 35% fund, 35% tech, 15% RR, 15% reliability (0-100 scale)
        # Normalize available fund score (Quality_Score_F is typically 0-50)
        raw_fund = row.get("Quality_Score_F", None)
        if isinstance(raw_fund, (int, float)) and np.isfinite(raw_fund):
            fund_score = float(raw_fund) / 50.0 * 100.0
        else:
            # Try alternate field (Fundamental_S) assumed already 0-100
            fund_score = float(row.get("Fundamental_S", 50.0)) if np.isfinite(row.get("Fundamental_S", np.nan)) else 50.0

        tech_score = float(row.get("Score_Tech", row.get("Score", 50.0))) if np.isfinite(row.get("Score_Tech", np.nan)) or np.isfinite(row.get("Score", np.nan)) else 50.0

        rr_ratio = row.get("RewardRisk", row.get("RR_Ratio", 0.0))
        if not (isinstance(rr_ratio, (int, float)) and np.isfinite(rr_ratio)):
            rr_ratio = 0.0
        rr_norm = (min(max(float(rr_ratio), 0.0), 5.0) / 5.0) * 100.0

        # Use centralized weights from scoring_config.py
        w_fund = CONVICTION_WEIGHTS.get("fundamental", 0.30)
        w_tech = CONVICTION_WEIGHTS.get("momentum", 0.30)  # tech score maps to momentum weight
        w_rr = CONVICTION_WEIGHTS.get("risk_reward", 0.20)
        w_rel = CONVICTION_WEIGHTS.get("reliability", 0.20)
        base_conviction = (
            w_fund * fund_score + w_tech * tech_score + w_rr * rr_norm + w_rel * float(reliability_v2)
        )
        result["conviction_v2_base"] = float(np.clip(base_conviction, 0.0, 100.0))
        
        # 4) Get ML Prediction (unified 20d inference)
        ml_prob = None
        if enable_ml:
            try:
                from core.ml_20d_inference import compute_ml_20d_probabilities_raw
                ml_prob = compute_ml_20d_probabilities_raw(row)
                if not (isinstance(ml_prob, (int, float)) and np.isfinite(ml_prob)):
                    ml_prob = row.get("ML_20d_Prob", None)
            except Exception as e:
                logger.debug(f"ML 20d prediction failed for {ticker}: {e}")
        
        result["ml_probability_v2"] = ml_prob
        
        # 5) Apply V2 Conviction Adjustments
        rr_ratio = rr_ratio = float(row.get("RewardRisk", row.get("RR_Ratio", 0.0)) or 0.0)
        conviction_final, ml_boost, conv_details = apply_v2_conviction_adjustments(
            base_conviction, reliability_v2, gate_penalty, rr_ratio, ml_prob, enable_ml
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
