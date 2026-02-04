"""
ML Feature Builder - Builds all 34 features required by ML model v3.

This module provides the critical bridge between raw indicators and the ML model.
It ensures all features are computed with EXACT names matching the model metadata.

Usage:
    from core.ml_feature_builder import build_all_ml_features_v3

    features = build_all_ml_features_v3(
        row=indicators_row,
        df_hist=price_history,
        market_context=market_data,  # Optional
        sector_context=sector_data,  # Optional
    )

    # features is a dict with all 34 feature names ready for ML inference
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


def build_all_ml_features_v3(
    row: pd.Series,
    df_hist: pd.DataFrame,
    market_context: Optional[Dict[str, float]] = None,
    sector_context: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build all 34 ML features with exact names matching model_20d_v3.metadata.json.

    Args:
        row: Series with basic technical indicators from build_technical_indicators()
        df_hist: Historical OHLCV DataFrame for computing additional features
        market_context: Optional dict with Market_Regime, Market_Volatility, Market_Trend, SPY_20d_ret
        sector_context: Optional dict with Sector_RS, Sector_Momentum, Sector_Rank for this stock's sector

    Returns:
        Dict with all 34 features named exactly as expected by the model:
        RSI, ATR_Pct, Return_20d, Return_10d, Return_5d, VCP_Ratio, Tightness_Ratio,
        Dist_From_52w_High, MA_Alignment, Volume_Surge, Up_Down_Volume_Ratio,
        Momentum_Consistency, RS_vs_SPY_20d, Market_Regime, Market_Volatility,
        Market_Trend, High_Volatility, Sector_RS, Sector_Momentum, Sector_Rank,
        Volume_Ratio_20d, Volume_Trend, Up_Volume_Ratio, Volume_Price_Confirm,
        Relative_Volume_Rank, Distance_From_52w_Low, Consolidation_Tightness,
        Days_Since_52w_High, Price_vs_SMA50, Price_vs_SMA200, SMA50_vs_SMA200,
        MA_Slope_20d, Distance_To_Resistance, Support_Strength
    """
    features = {}

    # Helper to safely get float values
    def safe_get(key: str, default: float = np.nan) -> float:
        val = row.get(key, default)
        if pd.isna(val):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # Prepare price data
    close = df_hist['Close'] if 'Close' in df_hist.columns else pd.Series(dtype=float)
    high = df_hist['High'] if 'High' in df_hist.columns else close
    low = df_hist['Low'] if 'Low' in df_hist.columns else close
    volume = df_hist['Volume'] if 'Volume' in df_hist.columns else pd.Series(1.0, index=df_hist.index)

    # ========================================================================
    # GROUP 1: TECHNICAL BASE (5 features)
    # ========================================================================

    # RSI - direct mapping
    features['RSI'] = safe_get('RSI', 50.0)

    # ATR_Pct - direct mapping
    features['ATR_Pct'] = safe_get('ATR_Pct', 0.02)

    # Returns - compute from history if not in row
    if len(close) >= 20:
        features['Return_20d'] = float((close.iloc[-1] / close.iloc[-20] - 1.0)) if close.iloc[-20] != 0 else 0.0
    else:
        features['Return_20d'] = safe_get('Return_1m', 0.0)  # Fallback

    if len(close) >= 10:
        features['Return_10d'] = float((close.iloc[-1] / close.iloc[-10] - 1.0)) if close.iloc[-10] != 0 else 0.0
    else:
        features['Return_10d'] = safe_get('Return_10d', 0.0)

    if len(close) >= 5:
        features['Return_5d'] = float((close.iloc[-1] / close.iloc[-5] - 1.0)) if close.iloc[-5] != 0 else 0.0
    else:
        features['Return_5d'] = safe_get('Return_5d', 0.0)

    # ========================================================================
    # GROUP 2: VOLATILITY PATTERNS (4 features)
    # ========================================================================

    # VCP_Ratio - ATR(10) / ATR(30)
    try:
        if len(df_hist) >= 30:
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_10 = tr.rolling(10).mean().iloc[-1]
            atr_30 = tr.rolling(30).mean().iloc[-1]
            features['VCP_Ratio'] = float(atr_10 / atr_30) if atr_30 > 0 else 1.0
        else:
            features['VCP_Ratio'] = 1.0
    except Exception:
        features['VCP_Ratio'] = 1.0

    # Tightness_Ratio - direct mapping
    features['Tightness_Ratio'] = safe_get('Tightness_Ratio', 1.0)

    # Dist_From_52w_High - MAPPED from Dist_52w_High
    features['Dist_From_52w_High'] = safe_get('Dist_52w_High', -0.1)

    # MA_Alignment - Close > MA20 > MA50 > MA200
    try:
        ma20 = safe_get('MA20', np.nan)
        ma50 = safe_get('MA50', np.nan)
        ma200 = safe_get('MA200', np.nan)
        current_close = safe_get('Close', np.nan)

        if all(pd.notna([current_close, ma20, ma50, ma200])):
            aligned = (current_close > ma20) and (ma20 > ma50) and (ma50 > ma200)
            features['MA_Alignment'] = 1.0 if aligned else 0.0
        else:
            features['MA_Alignment'] = 0.0
    except Exception:
        features['MA_Alignment'] = 0.0

    # ========================================================================
    # GROUP 3: VOLUME BASIC (3 features)
    # ========================================================================

    # Volume_Surge - MAPPED from VolSurge
    features['Volume_Surge'] = safe_get('VolSurge', 1.0)
    if features['Volume_Surge'] == 1.0:
        features['Volume_Surge'] = safe_get('Volume_Surge_Ratio', 1.0)

    # Up_Down_Volume_Ratio - compute from history
    try:
        if len(df_hist) >= 20:
            price_change = close.diff()
            up_vol = volume[price_change > 0].tail(20).sum()
            down_vol = volume[price_change < 0].tail(20).sum()
            features['Up_Down_Volume_Ratio'] = float(up_vol / down_vol) if down_vol > 0 else 1.0
        else:
            features['Up_Down_Volume_Ratio'] = 1.0
    except Exception:
        features['Up_Down_Volume_Ratio'] = 1.0

    # Momentum_Consistency - MAPPED from MomCons
    features['Momentum_Consistency'] = safe_get('MomCons', 0.5)

    # ========================================================================
    # GROUP 4: RELATIVE STRENGTH (1 feature)
    # ========================================================================

    # RS_vs_SPY_20d - requires market context
    if market_context and 'SPY_20d_ret' in market_context:
        spy_ret = market_context['SPY_20d_ret']
        stock_ret = features.get('Return_20d', 0.0)
        features['RS_vs_SPY_20d'] = stock_ret - spy_ret if pd.notna(spy_ret) else 0.0
    else:
        features['RS_vs_SPY_20d'] = 0.0  # Neutral default

    # ========================================================================
    # GROUP 5: MARKET REGIME (4 features)
    # ========================================================================

    if market_context:
        features['Market_Regime'] = market_context.get('Market_Regime', 0.0)
        features['Market_Volatility'] = market_context.get('Market_Volatility', 0.15)
        features['Market_Trend'] = market_context.get('Market_Trend', 0.0)
    else:
        features['Market_Regime'] = 0.0  # Neutral
        features['Market_Volatility'] = 0.15  # Average
        features['Market_Trend'] = 0.0  # Flat

    # High_Volatility - based on ATR percentile
    atr_pct = features['ATR_Pct']
    features['High_Volatility'] = 1.0 if atr_pct > 0.04 else 0.0  # Above 4% is high vol

    # ========================================================================
    # GROUP 6: SECTOR RELATIVE (3 features)
    # ========================================================================

    if sector_context:
        features['Sector_RS'] = sector_context.get('Sector_RS', 0.0)
        features['Sector_Momentum'] = sector_context.get('Sector_Momentum', 0.0)
        features['Sector_Rank'] = sector_context.get('Sector_Rank', 0.5)
    else:
        features['Sector_RS'] = 0.0
        features['Sector_Momentum'] = 0.0
        features['Sector_Rank'] = 0.5

    # ========================================================================
    # GROUP 7: VOLUME ADVANCED (5 features)
    # ========================================================================

    # Volume_Ratio_20d - current vol / 20d avg
    try:
        if len(volume) >= 20:
            vol_avg_20 = volume.tail(20).mean()
            features['Volume_Ratio_20d'] = float(volume.iloc[-1] / vol_avg_20) if vol_avg_20 > 0 else 1.0
        else:
            features['Volume_Ratio_20d'] = safe_get('Volume_Surge_Ratio', 1.0)
    except Exception:
        features['Volume_Ratio_20d'] = 1.0

    # Volume_Trend - OBV slope normalized
    try:
        if len(df_hist) >= 20:
            price_change = close.diff()
            obv = (np.sign(price_change) * volume).cumsum()
            obv_slope = (obv.iloc[-1] - obv.iloc[-20]) / (obv.iloc[-20] if obv.iloc[-20] != 0 else 1)
            features['Volume_Trend'] = float(np.clip(obv_slope, -1.0, 1.0))
        else:
            features['Volume_Trend'] = 0.0
    except Exception:
        features['Volume_Trend'] = 0.0

    # Up_Volume_Ratio - up-day volume / total volume
    try:
        if len(df_hist) >= 20:
            price_change = close.diff()
            up_vol = volume[price_change > 0].tail(20).sum()
            total_vol = volume.tail(20).sum()
            features['Up_Volume_Ratio'] = float(up_vol / total_vol) if total_vol > 0 else 0.5
        else:
            features['Up_Volume_Ratio'] = 0.5
    except Exception:
        features['Up_Volume_Ratio'] = 0.5

    # Volume_Price_Confirm - price up AND volume up
    try:
        if len(df_hist) >= 5:
            price_up = close.iloc[-1] > close.iloc[-5]
            vol_up = volume.iloc[-1] > volume.tail(20).mean()
            features['Volume_Price_Confirm'] = 1.0 if (price_up and vol_up) else 0.0
        else:
            features['Volume_Price_Confirm'] = 0.5
    except Exception:
        features['Volume_Price_Confirm'] = 0.5

    # Relative_Volume_Rank - volume percentile vs 60d
    try:
        if len(volume) >= 60:
            vol_pctile = (volume.tail(60) <= volume.iloc[-1]).sum() / 60
            features['Relative_Volume_Rank'] = float(vol_pctile)
        else:
            features['Relative_Volume_Rank'] = 0.5
    except Exception:
        features['Relative_Volume_Rank'] = 0.5

    # ========================================================================
    # GROUP 8: PRICE ACTION (9 features)
    # ========================================================================

    # Distance_From_52w_Low
    try:
        if len(low) >= 252:
            low_52w = low.tail(252).min()
        else:
            low_52w = low.min()
        current = close.iloc[-1] if len(close) > 0 else np.nan
        features['Distance_From_52w_Low'] = float((current - low_52w) / low_52w) if low_52w > 0 else 0.5
    except Exception:
        features['Distance_From_52w_Low'] = 0.5

    # Consolidation_Tightness - (20d_high - 20d_low) / avg_price
    try:
        if len(df_hist) >= 20:
            high_20 = high.tail(20).max()
            low_20 = low.tail(20).min()
            avg_price = close.tail(20).mean()
            features['Consolidation_Tightness'] = float((high_20 - low_20) / avg_price) if avg_price > 0 else 0.1
        else:
            features['Consolidation_Tightness'] = 0.1
    except Exception:
        features['Consolidation_Tightness'] = 0.1

    # Days_Since_52w_High - normalized 0-1
    try:
        if len(high) >= 252:
            high_252 = high.tail(252)
        else:
            high_252 = high
        high_52w = high_252.max()
        high_idx = high_252.idxmax()
        days_since = len(high_252) - high_252.index.get_loc(high_idx) - 1
        features['Days_Since_52w_High'] = float(min(days_since / 252, 1.0))
    except Exception:
        features['Days_Since_52w_High'] = 0.5

    # Price_vs_SMA50
    try:
        ma50 = safe_get('MA50', np.nan)
        current = safe_get('Close', np.nan)
        if pd.notna(ma50) and pd.notna(current) and ma50 > 0:
            features['Price_vs_SMA50'] = float((current - ma50) / ma50)
        else:
            features['Price_vs_SMA50'] = 0.0
    except Exception:
        features['Price_vs_SMA50'] = 0.0

    # Price_vs_SMA200
    try:
        ma200 = safe_get('MA200', np.nan)
        current = safe_get('Close', np.nan)
        if pd.notna(ma200) and pd.notna(current) and ma200 > 0:
            features['Price_vs_SMA200'] = float((current - ma200) / ma200)
        else:
            features['Price_vs_SMA200'] = 0.0
    except Exception:
        features['Price_vs_SMA200'] = 0.0

    # SMA50_vs_SMA200
    try:
        ma50 = safe_get('MA50', np.nan)
        ma200 = safe_get('MA200', np.nan)
        if pd.notna(ma50) and pd.notna(ma200) and ma200 > 0:
            features['SMA50_vs_SMA200'] = float((ma50 - ma200) / ma200)
        else:
            features['SMA50_vs_SMA200'] = 0.0
    except Exception:
        features['SMA50_vs_SMA200'] = 0.0

    # MA_Slope_20d - slope of 20d MA
    try:
        if len(close) >= 25:
            ma20_series = close.rolling(20).mean()
            ma_now = ma20_series.iloc[-1]
            ma_5d_ago = ma20_series.iloc[-5]
            if pd.notna(ma_now) and pd.notna(ma_5d_ago) and ma_5d_ago > 0:
                features['MA_Slope_20d'] = float((ma_now - ma_5d_ago) / ma_5d_ago)
            else:
                features['MA_Slope_20d'] = 0.0
        else:
            features['MA_Slope_20d'] = safe_get('MA50_Slope', 0.0)
    except Exception:
        features['MA_Slope_20d'] = 0.0

    # Distance_To_Resistance - (20d_high - close) / close
    try:
        high_20 = safe_get('High_20d', np.nan)
        current = safe_get('Close', np.nan)
        if pd.notna(high_20) and pd.notna(current) and current > 0:
            features['Distance_To_Resistance'] = float((high_20 - current) / current)
        else:
            features['Distance_To_Resistance'] = 0.05
    except Exception:
        features['Distance_To_Resistance'] = 0.05

    # Support_Strength - fraction of days near support (within 5% of 20d low)
    try:
        if len(df_hist) >= 20:
            low_20 = low.tail(20).min()
            support_threshold = low_20 * 1.05
            days_near_support = (close.tail(20) <= support_threshold).sum()
            features['Support_Strength'] = float(days_near_support / 20)
        else:
            features['Support_Strength'] = 0.2
    except Exception:
        features['Support_Strength'] = 0.2

    # ========================================================================
    # VALIDATION: Ensure all 34 features exist with valid values
    # ========================================================================

    expected_features = [
        'RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d',
        'VCP_Ratio', 'Tightness_Ratio', 'Dist_From_52w_High', 'MA_Alignment',
        'Volume_Surge', 'Up_Down_Volume_Ratio', 'Momentum_Consistency',
        'RS_vs_SPY_20d', 'Market_Regime', 'Market_Volatility', 'Market_Trend',
        'High_Volatility', 'Sector_RS', 'Sector_Momentum', 'Sector_Rank',
        'Volume_Ratio_20d', 'Volume_Trend', 'Up_Volume_Ratio', 'Volume_Price_Confirm',
        'Relative_Volume_Rank', 'Distance_From_52w_Low', 'Consolidation_Tightness',
        'Days_Since_52w_High', 'Price_vs_SMA50', 'Price_vs_SMA200', 'SMA50_vs_SMA200',
        'MA_Slope_20d', 'Distance_To_Resistance', 'Support_Strength'
    ]

    # Import defaults from registry
    try:
        from core.feature_registry import get_feature_defaults
        defaults = get_feature_defaults("v3")
    except Exception:
        defaults = {}

    # Fill any missing features with defaults
    for feat in expected_features:
        if feat not in features or pd.isna(features[feat]) or not np.isfinite(features[feat]):
            features[feat] = defaults.get(feat, 0.0)

    # Clip to valid ranges
    try:
        from core.feature_registry import get_feature_ranges
        ranges = get_feature_ranges("v3")
        for feat, (lo, hi) in ranges.items():
            if feat in features:
                features[feat] = float(np.clip(features[feat], lo, hi))
    except Exception:
        pass

    return features


def get_market_context_from_row(row: pd.Series) -> Dict[str, float]:
    """
    Extract market context from a row that might have SPY data.

    Args:
        row: Series that may contain SPY_20d_ret, Market_Regime, etc.

    Returns:
        Dict with market context features
    """
    context = {
        'Market_Regime': 0.0,
        'Market_Volatility': 0.15,
        'Market_Trend': 0.0,
        'SPY_20d_ret': 0.0,
    }

    # Try to extract from row
    for key in context.keys():
        if key in row.index:
            val = row[key]
            if pd.notna(val):
                context[key] = float(val)

    return context
