"""
Market Regime Detection for Stock Scout
=========================================

Detects bullish/neutral/bearish market conditions based on:
- SPY and QQQ trend analysis
- VIX levels (if available)
- Recent volatility
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def detect_market_regime(
    lookback_days: int = 60,
    spy_data: Optional[pd.DataFrame] = None,
    qqq_data: Optional[pd.DataFrame] = None
) -> Dict[str, any]:
    """
    Detect current market regime based on SPY/QQQ trends and volatility.
    
    Args:
        lookback_days: Number of days to look back for trend analysis
        spy_data: Pre-fetched SPY data (optional)
        qqq_data: Pre-fetched QQQ data (optional)
    
    Returns:
        Dict with:
            - regime: "bullish", "neutral", or "bearish"
            - confidence: 0-100 score
            - spy_trend: SPY trend score (-1 to 1)
            - qqq_trend: QQQ trend score (-1 to 1)
            - vix_level: VIX category ("low", "normal", "elevated", "high")
            - details: Human-readable description
    """
    result = {
        "regime": "neutral",
        "confidence": 50,
        "spy_trend": 0.0,
        "qqq_trend": 0.0,
        "vix_level": "normal",
        "details": "Market regime detection in progress"
    }
    
    try:
        # Fetch SPY data if not provided
        if spy_data is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 20)
            spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)
        
        # Fetch QQQ data if not provided
        if qqq_data is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 20)
            qqq_data = yf.download("QQQ", start=start_date, end=end_date, progress=False)
        
        # Check if we have valid data
        if spy_data is None or len(spy_data) < 20:
            logger.warning("Insufficient SPY data for regime detection")
            return result
        
        if qqq_data is None or len(qqq_data) < 20:
            logger.warning("Insufficient QQQ data for regime detection")
            # Use only SPY
            qqq_data = None
        
        # Calculate SPY trend
        spy_close = spy_data['Close'] if 'Close' in spy_data.columns else spy_data['Adj Close']
        spy_ma20 = spy_close.rolling(20).mean()
        spy_ma50 = spy_close.rolling(50).mean() if len(spy_close) >= 50 else spy_ma20
        
        current_spy = spy_close.iloc[-1]
        spy_vs_ma20 = (current_spy / spy_ma20.iloc[-1] - 1) if pd.notna(spy_ma20.iloc[-1]) else 0
        spy_vs_ma50 = (current_spy / spy_ma50.iloc[-1] - 1) if pd.notna(spy_ma50.iloc[-1]) else 0
        spy_momentum = (current_spy / spy_close.iloc[-20] - 1) if len(spy_close) >= 20 else 0
        
        # SPY trend score: -1 (bearish) to +1 (bullish)
        spy_trend = np.clip((spy_vs_ma20 * 2 + spy_vs_ma50 + spy_momentum) / 4, -1, 1)
        result["spy_trend"] = float(spy_trend)
        
        # Calculate QQQ trend if available
        if qqq_data is not None:
            qqq_close = qqq_data['Close'] if 'Close' in qqq_data.columns else qqq_data['Adj Close']
            qqq_ma20 = qqq_close.rolling(20).mean()
            qqq_ma50 = qqq_close.rolling(50).mean() if len(qqq_close) >= 50 else qqq_ma20
            
            current_qqq = qqq_close.iloc[-1]
            qqq_vs_ma20 = (current_qqq / qqq_ma20.iloc[-1] - 1) if pd.notna(qqq_ma20.iloc[-1]) else 0
            qqq_vs_ma50 = (current_qqq / qqq_ma50.iloc[-1] - 1) if pd.notna(qqq_ma50.iloc[-1]) else 0
            qqq_momentum = (current_qqq / qqq_close.iloc[-20] - 1) if len(qqq_close) >= 20 else 0
            
            qqq_trend = np.clip((qqq_vs_ma20 * 2 + qqq_vs_ma50 + qqq_momentum) / 4, -1, 1)
            result["qqq_trend"] = float(qqq_trend)
        else:
            qqq_trend = spy_trend
            result["qqq_trend"] = float(spy_trend)
        
        # Calculate VIX level if available
        try:
            vix_data = yf.download("^VIX", period="5d", progress=False)
            if vix_data is not None and len(vix_data) > 0:
                vix_close = vix_data['Close'] if 'Close' in vix_data.columns else vix_data['Adj Close']
                current_vix = vix_close.iloc[-1]
                
                if current_vix < 15:
                    vix_level = "low"
                    vix_score = 0.3  # Complacent = slightly negative
                elif current_vix < 20:
                    vix_level = "normal"
                    vix_score = 0.0  # Neutral
                elif current_vix < 30:
                    vix_level = "elevated"
                    vix_score = -0.3  # Caution
                else:
                    vix_level = "high"
                    vix_score = -0.6  # Fear
                
                result["vix_level"] = vix_level
            else:
                vix_score = 0.0
        except Exception:
            vix_score = 0.0
        
        # Combine signals to determine regime
        # Weight: 40% SPY, 40% QQQ, 20% VIX
        combined_score = 0.4 * spy_trend + 0.4 * qqq_trend + 0.2 * vix_score
        
        # Determine regime
        if combined_score > 0.20:
            regime = "bullish"
            confidence = min(100, int((combined_score + 0.5) * 100))
            details = f"Bullish: SPY {spy_trend:.2f}, QQQ {qqq_trend:.2f}, VIX {result['vix_level']}"
        elif combined_score < -0.20:
            regime = "bearish"
            confidence = min(100, int((0.5 - combined_score) * 100))
            details = f"Bearish: SPY {spy_trend:.2f}, QQQ {qqq_trend:.2f}, VIX {result['vix_level']}"
        else:
            regime = "neutral"
            confidence = int((0.5 - abs(combined_score)) * 100)
            details = f"Neutral: SPY {spy_trend:.2f}, QQQ {qqq_trend:.2f}, VIX {result['vix_level']}"
        
        result["regime"] = regime
        result["confidence"] = confidence
        result["details"] = details
        
        logger.info(f"Market regime detected: {regime} (confidence: {confidence}%)")
        
    except Exception as e:
        logger.error(f"Error detecting market regime: {e}")
        result["details"] = f"Error: {str(e)}"
    
    return result


def adjust_target_for_regime(
    base_target_pct: float,
    reliability: float,
    risk_meter: float,
    regime_data: Dict
) -> Tuple[float, str]:
    """
    Adjust target price percentage based on market regime, reliability, and risk.
    
    Args:
        base_target_pct: Base target gain percentage (e.g., 0.10 for 10%)
        reliability: Reliability score 0-100
        risk_meter: Risk score 0-100
        regime_data: Output from detect_market_regime()
    
    Returns:
        (adjusted_target_pct, explanation)
    """
    regime = regime_data.get("regime", "neutral")
    confidence = regime_data.get("confidence", 50)
    
    # Start with base target
    adjusted_pct = base_target_pct
    adjustments = []
    
    # 1) Regime adjustment
    if regime == "bullish" and confidence > 60:
        regime_boost = 0.03  # +3% in strong bullish
        adjusted_pct += regime_boost
        adjustments.append(f"+{regime_boost*100:.1f}% (bullish)")
    elif regime == "bearish" and confidence > 60:
        regime_reduction = -0.02  # -2% in strong bearish
        adjusted_pct += regime_reduction
        adjustments.append(f"{regime_reduction*100:.1f}% (bearish)")
    
    # 2) Reliability adjustment
    if reliability < 40:
        # Low reliability: cap upside
        reliability_cap = -0.03
        adjusted_pct = min(adjusted_pct + reliability_cap, 0.08)  # Max 8% for low reliability
        adjustments.append(f"capped at 8% (low reliability)")
    elif reliability >= 75:
        # High reliability: allow higher targets in bullish regime
        if regime == "bullish":
            reliability_boost = 0.02
            adjusted_pct += reliability_boost
            adjustments.append(f"+{reliability_boost*100:.1f}% (high reliability)")
    
    # 3) Risk adjustment
    if risk_meter > 70:
        # High risk: reduce target
        risk_reduction = -0.02
        adjusted_pct += risk_reduction
        adjustments.append(f"{risk_reduction*100:.1f}% (high risk)")
    
    # 4) Safety bounds
    adjusted_pct = float(np.clip(adjusted_pct, 0.03, 0.25))  # Min 3%, max 25%
    
    explanation = ", ".join(adjustments) if adjustments else "no adjustments"
    
    return adjusted_pct, explanation
