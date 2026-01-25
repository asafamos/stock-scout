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
    """Robust market regime detection with safe fallbacks.

    Always returns simple primitives (no Series). On any failure returns neutral baseline.
    """
    fallback = {
        "regime": "neutral",
        "confidence": 50,
        "spy_trend": 0.0,
        "qqq_trend": 0.0,
        "vix_level": "unknown",
        "details": "Benchmark unavailable; neutral fallback",
        "benchmark_status": "UNAVAILABLE",
    }
    try:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days + 25)
        if spy_data is None:
            spy_data = yf.download("SPY", start=start_dt, end=end_dt, progress=False)
        if qqq_data is None:
            qqq_data = yf.download("QQQ", start=start_dt, end=end_dt, progress=False)
        if spy_data is None or len(spy_data.index) < 40:
            return fallback
        # Ensure we have a Series of close prices (avoid ambiguous DataFrame/Series boolean ops)
        if isinstance(spy_data, pd.Series):
            spy_df = spy_data.to_frame()
        else:
            spy_df = spy_data
        if "Close" in spy_df.columns:
            spy_close = spy_df["Close"].astype(float)
        else:
            spy_close = spy_df.iloc[:, 0].astype(float)
        # Force a 1-D Series (avoid DataFrame or multi-column edgecases)
        spy_close = pd.Series(spy_close.values.ravel(), index=spy_close.index)
        ma20 = spy_close.rolling(20).mean(); ma50 = spy_close.rolling(50).mean()
        # Strict validity: require non-NaN MA values and sufficient history
        if pd.isna(ma20.iloc[-1]) or pd.isna(ma50.iloc[-1]) or len(spy_close) < 50:
            return fallback
        last = float(spy_close.iloc[-1])
        vs20 = float(last / ma20.iloc[-1] - 1)
        vs50 = float(last / ma50.iloc[-1] - 1)
        momentum = float(last / spy_close.iloc[-20] - 1) if len(spy_close) >= 21 else 0.0
        spy_trend = float(np.clip((vs20 * 2 + vs50 + momentum) / 4.0, -1, 1))
        # Defensive: if SPY appears flat (no variance) and trend computes to 0.0, treat as unavailable
        try:
            if float(np.nanstd(spy_close.tail(50))) == 0.0 and abs(spy_trend) < 1e-12:
                return fallback
        except Exception:
            pass
        if qqq_data is not None and len(qqq_data.index) >= 40:
            if isinstance(qqq_data, pd.Series):
                qqq_df = qqq_data.to_frame()
            else:
                qqq_df = qqq_data
            if "Close" in qqq_df.columns:
                qqq_close = qqq_df["Close"].astype(float)
            else:
                qqq_close = qqq_df.iloc[:, 0].astype(float)
            qqq_close = pd.Series(qqq_close.values.ravel(), index=qqq_close.index)
            qma20 = qqq_close.rolling(20).mean(); qma50 = qqq_close.rolling(50).mean()
            if pd.isna(qma20.iloc[-1]) or pd.isna(qma50.iloc[-1]) or len(qqq_close) < 50:
                # Keep SPY trend, but mark benchmark unavailable in details later
                qqq_trend = spy_trend
            else:
                qlast = float(qqq_close.iloc[-1])
                qvs20 = float(qlast / qma20.iloc[-1] - 1)
                qvs50 = float(qlast / qma50.iloc[-1] - 1)
                qmom = float(qlast / qqq_close.iloc[-20] - 1) if len(qqq_close) >= 21 else 0.0
                qqq_trend = float(np.clip((qvs20 * 2 + qvs50 + qmom) / 4.0, -1, 1))
                try:
                    if float(np.nanstd(qqq_close.tail(50))) == 0.0 and abs(qqq_trend) < 1e-12:
                        # Keep SPY trend; mark unavailability later
                        qqq_trend = spy_trend
                except Exception:
                    pass
        else:
            qqq_trend = spy_trend
        try:
            vix_df = yf.download("^VIX", period="7d", progress=False)
            if vix_df is not None and len(vix_df.index) > 0:
                if isinstance(vix_df, pd.Series):
                    vix_df = vix_df.to_frame()
                if "Close" in vix_df.columns:
                    vix_close = vix_df["Close"].astype(float)
                else:
                    vix_close = vix_df.iloc[:, 0].astype(float)
                vix_close = pd.Series(vix_close.values.ravel(), index=vix_close.index)
                vix_val = float(vix_close.iloc[-1])
            else:
                vix_val = 20.0
        except Exception:
            vix_val = 20.0
        if vix_val < 15: vix_level, vix_score = "low", 0.25
        elif vix_val < 20: vix_level, vix_score = "normal", 0.0
        elif vix_val < 30: vix_level, vix_score = "elevated", -0.25
        else: vix_level, vix_score = "high", -0.50
        composite = 0.4*spy_trend + 0.4*qqq_trend + 0.2*vix_score
        if composite > 0.20:
            regime = "bullish"; confidence = int(min(100,(composite+0.5)*100))
        elif composite < -0.20:
            regime = "bearish"; confidence = int(min(100,(0.5-composite)*100))
        else:
            regime = "neutral"; confidence = int((0.5-abs(composite))*100)
        details = f"SPY {spy_trend:.2f} | QQQ {qqq_trend:.2f} | VIX {vix_level} ({vix_val:.1f})"
        out = {
            "regime": regime,
            "confidence": confidence,
            "spy_trend": spy_trend,
            "qqq_trend": qqq_trend,
            "vix_level": vix_level,
            "details": details,
            "benchmark_status": "OK",
        }
        # Reflect unavailability if earlier checks triggered
        try:
            if not np.isfinite(spy_trend) or spy_trend == 0.0 and not (vs20 or vs50 or momentum):
                out["benchmark_status"] = "UNAVAILABLE"
                out["details"] = "Benchmark unavailable; neutral fallback"
        except Exception:
            pass
        return out
    except Exception as exc:
        logger.exception("Regime detection failed, neutral fallback used")
        return fallback


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
