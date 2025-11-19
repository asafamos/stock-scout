"""
Enhanced position sizing and risk warnings.
"""

import pandas as pd
import numpy as np
from typing import Dict


def compute_smart_position_size(
    risk_score: float,
    ml_prob: float,
    portfolio_value: float,
    max_position_pct: float = 0.10,
) -> dict:
    """
    Smart position sizing based on risk AND ML confidence.
    
    Rules:
    - High ML prob + Low risk = bigger position
    - Low ML prob OR High risk = smaller position
    - Very high risk (>8) = warning, tiny position
    
    Args:
        risk_score: 1-10 scale (1=safest, 10=riskiest)
        ml_prob: 0-1 ML model probability
        portfolio_value: Total portfolio value
        max_position_pct: Max % of portfolio for single position
    
    Returns:
        dict with position_pct, dollar_amount, stop_loss_pct, warnings
    """
    warnings = []
    
    # Base position from risk
    if risk_score <= 3:
        base_pct = max_position_pct          # 10%
        stop_loss = 0.07                      # 7%
    elif risk_score <= 5:
        base_pct = max_position_pct * 0.7    # 7%
        stop_loss = 0.06                      # 6%
    elif risk_score <= 7:
        base_pct = max_position_pct * 0.4    # 4%
        stop_loss = 0.05                      # 5%
        warnings.append("⚠️ High risk - use tight stop loss")
    else:
        base_pct = max_position_pct * 0.2    # 2%
        stop_loss = 0.04                      # 4%
        warnings.append("🚨 Very high risk - consider skipping")
    
    # Adjust by ML confidence
    if ml_prob >= 0.7:
        confidence_multiplier = 1.2           # Increase 20%
    elif ml_prob >= 0.5:
        confidence_multiplier = 1.0           # Keep as is
    else:
        confidence_multiplier = 0.7           # Reduce 30%
        warnings.append("⚠️ Low ML confidence - reduce position")
    
    final_pct = min(base_pct * confidence_multiplier, max_position_pct)
    
    # Calculate share count (approximate, needs current price)
    dollar_amount = portfolio_value * final_pct
    
    return {
        'position_pct': final_pct,
        'dollar_amount': dollar_amount,
        'stop_loss_pct': stop_loss,
        'stop_loss_dollar': dollar_amount * stop_loss,
        'take_profit_pct': 0.15,              # Standard 15% target
        'warnings': warnings,
        'risk_category': categorize_risk(risk_score, ml_prob),
    }


def categorize_risk(risk_score: float, ml_prob: float) -> str:
    """
    Categorize overall risk level.
    """
    if risk_score <= 4 and ml_prob >= 0.6:
        return "🟢 LOW RISK"
    elif risk_score <= 6 and ml_prob >= 0.5:
        return "🟡 MEDIUM RISK"
    elif risk_score <= 8:
        return "🟠 HIGH RISK"
    else:
        return "🔴 VERY HIGH RISK"


def check_portfolio_concentration(
    current_positions: Dict[str, float],
    new_ticker: str,
    new_dollar: float,
) -> Dict:
    """
    Check if adding new position creates concentration risk.
    
    Args:
        current_positions: {ticker: dollar_amount}
        new_ticker: Ticker to add
        new_dollar: Dollar amount to invest
    
    Returns:
        dict with warnings and concentration metrics
    """
    warnings = []
    
    total_portfolio = sum(current_positions.values()) + new_dollar
    new_pct = new_dollar / total_portfolio if total_portfolio > 0 else 0
    
    if new_pct > 0.15:
        warnings.append(f"⚠️ Position would be {new_pct:.1%} of portfolio (>15%)")
    
    # Check correlation risk (simplified: same sector = correlated)
    # TODO: integrate with sector data
    
    return {
        'concentration_pct': new_pct,
        'warnings': warnings,
        'recommended_max': total_portfolio * 0.15,
    }


def generate_risk_report(
    ticker: str,
    risk_score: float,
    ml_prob: float,
    rsi: float,
    atr_pct: float,
    position_info: dict,
) -> str:
    """
    Generate human-readable risk report.
    """
    report = f"""
    
📊 RISK REPORT: {ticker}
{'='*50}
Risk Score: {risk_score:.1f}/10
ML Confidence: {ml_prob:.1%}
Category: {position_info['risk_category']}

POSITION SIZING:
• Recommended: ${position_info['dollar_amount']:,.0f} ({position_info['position_pct']:.1%} of portfolio)
• Stop Loss: -{position_info['stop_loss_pct']:.1%} (${position_info['stop_loss_dollar']:,.0f})
• Take Profit: +{position_info['take_profit_pct']:.1%}

TECHNICAL SETUP:
• RSI: {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
• Volatility (ATR): {atr_pct:.2%} ({'Low' if atr_pct < 0.02 else 'High' if atr_pct > 0.04 else 'Normal'})

WARNINGS:
"""
    
    if position_info['warnings']:
        for warning in position_info['warnings']:
            report += f"  {warning}\n"
    else:
        report += "  ✅ No major warnings\n"
    
    report += "\n" + "="*50
    
    return report
