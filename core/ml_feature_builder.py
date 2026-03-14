"""
ML Feature Builder - Builds all 39 features required by ML model v3.1.

This module provides the critical bridge between raw indicators and the ML model.
It ensures all features are computed with EXACT names matching the model metadata.

v3.1 changes:
  - Removed: Market_Trend, Market_Volatility, High_Volatility (market-timing)
  - Added: Vol_Contraction_Ratio, Squeeze_On_Flag, RS_vs_SPY_60d, RS_Momentum,
           UpStreak_Days, DownStreak_Days, Range_Pct_10d, OvernightGap_Avg

Usage:
    from core.ml_feature_builder import build_all_ml_features_v3

    features = build_all_ml_features_v3(
        row=indicators_row,
        df_hist=price_history,
        market_context=market_data,  # Optional
        sector_context=sector_data,  # Optional
    )

    # features is a dict with all 39 feature names ready for ML inference
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

from core.feature_registry import get_feature_names

logger = logging.getLogger(__name__)


def build_all_ml_features_v3(
    row: pd.Series,
    df_hist: pd.DataFrame,
    market_context: Optional[Dict[str, float]] = None,
    sector_context: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build all 39 ML features with exact names matching model v3.1.

    Args:
        row: Series with basic technical indicators from build_technical_indicators()
        df_hist: Historical OHLCV DataFrame for computing additional features
        market_context: Optional dict with Market_Regime, SPY_20d_ret, SPY_60d_ret
        sector_context: Optional dict with Sector_RS, Sector_Momentum, Sector_Rank

    Returns:
        Dict with all 39 features named exactly as expected by the model.
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

    # Up_Down_Volume_Ratio - avg up-day volume / avg down-day volume (match training)
    try:
        if len(df_hist) >= 20:
            daily_return = close.pct_change()
            up_day = daily_return > 0
            down_day = daily_return < 0
            up_volume_sum = (volume * up_day).tail(20).sum()
            down_volume_sum = (volume * down_day).tail(20).sum()
            up_days_count = up_day.tail(20).sum()
            down_days_count = down_day.tail(20).sum()
            avg_up_vol = up_volume_sum / up_days_count if up_days_count > 0 else 0.0
            avg_down_vol = down_volume_sum / down_days_count if down_days_count > 0 else 1.0
            features['Up_Down_Volume_Ratio'] = float(avg_up_vol / avg_down_vol) if avg_down_vol > 0 else 1.0
        else:
            features['Up_Down_Volume_Ratio'] = 1.0
    except Exception:
        features['Up_Down_Volume_Ratio'] = 1.0

    # Momentum_Consistency - MAPPED from MomCons
    features['Momentum_Consistency'] = safe_get('MomCons', 0.5)

    # ========================================================================
    # GROUP 4: RELATIVE STRENGTH (3 features — expanded in v3.1)
    # ========================================================================

    # RS_vs_SPY_20d - requires market context
    if market_context and 'SPY_20d_ret' in market_context:
        spy_ret = market_context['SPY_20d_ret']
        stock_ret = features.get('Return_20d', 0.0)
        features['RS_vs_SPY_20d'] = stock_ret - spy_ret if pd.notna(spy_ret) else 0.0
    else:
        features['RS_vs_SPY_20d'] = 0.0  # Neutral default

    # RS_vs_SPY_60d - longer-horizon relative strength
    try:
        if len(close) >= 60:
            stock_ret_60d = float(close.iloc[-1] / close.iloc[-60] - 1.0) if close.iloc[-60] != 0 else 0.0
        else:
            stock_ret_60d = 0.0
        if market_context and 'SPY_60d_ret' in market_context:
            spy_ret_60d = market_context['SPY_60d_ret']
            features['RS_vs_SPY_60d'] = stock_ret_60d - spy_ret_60d if pd.notna(spy_ret_60d) else 0.0
        else:
            features['RS_vs_SPY_60d'] = 0.0
    except Exception:
        features['RS_vs_SPY_60d'] = 0.0

    # RS_Momentum - acceleration of relative strength (20d RS - 60d RS)
    features['RS_Momentum'] = features['RS_vs_SPY_20d'] - features['RS_vs_SPY_60d']

    # ========================================================================
    # GROUP 5: MARKET REGIME (1 feature — v3.1 stripped market-timing)
    # ========================================================================

    if market_context:
        features['Market_Regime'] = market_context.get('Market_Regime', 0.0)
    else:
        features['Market_Regime'] = 0.0  # Neutral

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

    # Volume_Trend - linear regression slope of volume normalized by mean (match training)
    try:
        if len(df_hist) >= 20:
            vol_window = volume.tail(20).values
            if len(vol_window) >= 20:
                slope = np.polyfit(range(len(vol_window)), vol_window, 1)[0]
                vol_mean = vol_window.mean()
                features['Volume_Trend'] = float(slope / (vol_mean + 1e-8))
            else:
                features['Volume_Trend'] = 0.0
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

    # Volume_Price_Confirm - rolling 5d mean of (price_up & vol_up) binary (match training)
    try:
        if len(df_hist) >= 5:
            price_up = close > close.shift(1)
            vol_up = volume > volume.shift(1)
            confirm = (price_up & vol_up).astype(float)
            features['Volume_Price_Confirm'] = float(confirm.tail(5).mean())
        else:
            features['Volume_Price_Confirm'] = 0.5
    except Exception:
        features['Volume_Price_Confirm'] = 0.5

    # Relative_Volume_Rank - pd.rank(pct=True) on rolling 60d window (match training)
    try:
        if len(volume) >= 60:
            vol_window = volume.tail(60)
            features['Relative_Volume_Rank'] = float(vol_window.rank(pct=True).iloc[-1])
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

    # MA_Slope_20d - 20d delta of MA20 (match training: ma20.diff(20) / ma20.shift(20))
    try:
        if len(close) >= 40:  # Need 20 for MA + 20 for diff
            ma20_series = close.rolling(20).mean()
            ma_now = ma20_series.iloc[-1]
            ma_20d_ago = ma20_series.iloc[-20]
            if pd.notna(ma_now) and pd.notna(ma_20d_ago) and ma_20d_ago > 0:
                features['MA_Slope_20d'] = float((ma_now - ma_20d_ago) / ma_20d_ago)
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

    # Support_Strength - fraction of days near support (within 2% of 20d low, match training)
    try:
        if len(df_hist) >= 20:
            low_20 = low.tail(20).min()
            support_threshold = low_20 * 1.02  # Training uses 1.02, not 1.05
            days_near_support = (low.tail(20) <= support_threshold).sum()  # Training uses Low, not Close
            features['Support_Strength'] = float(days_near_support / 20)
        else:
            features['Support_Strength'] = 0.2
    except Exception:
        features['Support_Strength'] = 0.2

    # ========================================================================
    # GROUP 9: VOLATILITY ADDITIONS (v3.1 — 2 features)
    # ========================================================================

    # Vol_Contraction_Ratio: ATR(20) / ATR(50) — tighter = breakout setup
    try:
        if len(df_hist) >= 50:
            tr_series = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_20 = tr_series.rolling(20).mean().iloc[-1]
            atr_50 = tr_series.rolling(50).mean().iloc[-1]
            features['Vol_Contraction_Ratio'] = float(atr_20 / atr_50) if atr_50 > 0 else 1.0
        else:
            features['Vol_Contraction_Ratio'] = 1.0
    except Exception:
        features['Vol_Contraction_Ratio'] = 1.0

    # Squeeze_On_Flag: Bollinger Bands inside Keltner Channels (pre-breakout)
    try:
        if len(close) >= 20:
            ma20_val = close.rolling(20).mean().iloc[-1]
            bb_std = close.rolling(20).std().iloc[-1]
            bb_upper = ma20_val + 2 * bb_std
            bb_lower = ma20_val - 2 * bb_std
            # Need ATR_20 for Keltner
            tr_series = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            kc_atr = tr_series.rolling(20).mean().iloc[-1]
            kc_upper = ma20_val + 1.5 * kc_atr
            kc_lower = ma20_val - 1.5 * kc_atr
            features['Squeeze_On_Flag'] = 1.0 if (bb_upper < kc_upper and bb_lower > kc_lower) else 0.0
        else:
            features['Squeeze_On_Flag'] = 0.0
    except Exception:
        features['Squeeze_On_Flag'] = 0.0

    # ========================================================================
    # GROUP 10: STREAK & PATTERN FEATURES (v3.1 — 4 features)
    # ========================================================================

    # UpStreak_Days: consecutive up-close days (capped at 10)
    try:
        # Check if already computed by build_technical_indicators
        up_streak = safe_get('UpStreak_Days', np.nan)
        if pd.notna(up_streak):
            features['UpStreak_Days'] = min(float(up_streak), 10.0)
        elif len(close) >= 2:
            up_close = (close > close.shift(1)).astype(int)
            streak_break = (up_close == 0).cumsum()
            streaks = up_close.groupby(streak_break).cumsum()
            features['UpStreak_Days'] = min(float(streaks.iloc[-1]), 10.0)
        else:
            features['UpStreak_Days'] = 0.0
    except Exception:
        features['UpStreak_Days'] = 0.0

    # DownStreak_Days: consecutive down-close days (capped at 10)
    try:
        down_streak = safe_get('DownStreak_Days', np.nan)
        if pd.notna(down_streak):
            features['DownStreak_Days'] = min(float(down_streak), 10.0)
        elif len(close) >= 2:
            down_close = (close < close.shift(1)).astype(int)
            streak_break = (down_close == 0).cumsum()
            streaks = down_close.groupby(streak_break).cumsum()
            features['DownStreak_Days'] = min(float(streaks.iloc[-1]), 10.0)
        else:
            features['DownStreak_Days'] = 0.0
    except Exception:
        features['DownStreak_Days'] = 0.0

    # Range_Pct_10d: average intraday range as pct of close (10d)
    try:
        if len(df_hist) >= 10:
            intraday_range = (high - low) / close.replace(0, np.nan)
            features['Range_Pct_10d'] = float(intraday_range.tail(10).mean())
        else:
            features['Range_Pct_10d'] = 0.02
    except Exception:
        features['Range_Pct_10d'] = 0.02

    # OvernightGap_Avg: average overnight gap pct over 5 days
    try:
        open_col = df_hist['Open'] if 'Open' in df_hist.columns else close
        if len(df_hist) >= 6:
            overnight_gap = (open_col - close.shift(1)) / close.shift(1).replace(0, np.nan)
            features['OvernightGap_Avg'] = float(overnight_gap.tail(5).mean())
        else:
            features['OvernightGap_Avg'] = 0.0
    except Exception:
        features['OvernightGap_Avg'] = 0.0

    # ========================================================================
    # VALIDATION: Ensure all 39 features exist with valid values
    # ========================================================================

    expected_features = get_feature_names("v3.1")

    # Import defaults from registry
    try:
        from core.feature_registry import get_feature_defaults
        defaults = get_feature_defaults("v3.1")
    except Exception:
        defaults = {}

    # Fill any missing features with defaults
    for feat in expected_features:
        if feat not in features or pd.isna(features[feat]) or not np.isfinite(features[feat]):
            features[feat] = defaults.get(feat, 0.0)

    # Clip to valid ranges
    try:
        from core.feature_registry import get_feature_ranges
        ranges = get_feature_ranges("v3.1")
        for feat, (lo, hi) in ranges.items():
            if feat in features:
                features[feat] = float(np.clip(features[feat], lo, hi))
    except Exception:
        pass

    return features


def build_all_ml_features_v3_5(
    row: pd.Series,
    df_hist: pd.DataFrame,
    market_context: Optional[Dict[str, float]] = None,
    sector_context: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Build 20-feature vector for V3.5 model.

    V3.5 = V3.4 (13 features) + 7 cherry-picked from V4:
      Delta (4): RSI_Delta_5d, ATR_Delta_5d, Volume_Delta_5d, Momentum_Acceleration
      Interaction (3): VCP_x_RS, Momentum_x_Volume, Squeeze_x_Volume

    All features are OHLCV-derived (no external APIs needed).

    Args:
        row: Series with technical indicators from build_technical_indicators()
        df_hist: Historical OHLCV DataFrame
        market_context: Optional market regime/SPY data
        sector_context: Optional sector relative data

    Returns:
        Dict with all 20 V3.5 features.
    """
    # Start with all V3.1 features (39 features — superset of V3.4's 13)
    all_features = build_all_ml_features_v3(row, df_hist, market_context, sector_context)

    close = df_hist['Close'] if 'Close' in df_hist.columns else pd.Series(dtype=float)

    # ── 4 Delta Features ────────────────────────────────────────────

    # RSI_Delta_5d: RSI acceleration
    try:
        if len(close) >= 20:
            # Compute RSI series from close prices
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs_val = gain / loss.replace(0, np.nan)
            rsi_series = 100.0 - (100.0 / (1.0 + rs_val))
            rsi_now = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0
            rsi_5d = float(rsi_series.iloc[-5]) if len(rsi_series) >= 5 and pd.notna(rsi_series.iloc[-5]) else rsi_now
            all_features["RSI_Delta_5d"] = rsi_now - rsi_5d
        else:
            all_features["RSI_Delta_5d"] = 0.0
    except Exception:
        all_features["RSI_Delta_5d"] = 0.0

    # ATR_Delta_5d: volatility acceleration
    try:
        atr_now = all_features.get("ATR_Pct", 0.02)
        if len(df_hist) >= 20:
            high = df_hist['High'] if 'High' in df_hist.columns else close
            low = df_hist['Low'] if 'Low' in df_hist.columns else close
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(14).mean()
            atr_pct_series = atr_series / close.replace(0, np.nan)
            atr_5d = float(atr_pct_series.iloc[-5]) if len(atr_pct_series) >= 5 and pd.notna(atr_pct_series.iloc[-5]) else atr_now
            all_features["ATR_Delta_5d"] = float(atr_now - atr_5d)
        else:
            all_features["ATR_Delta_5d"] = 0.0
    except Exception:
        all_features["ATR_Delta_5d"] = 0.0

    # Volume_Delta_5d: volume surge acceleration
    try:
        vol_surge_now = all_features.get("Volume_Surge", 1.0)
        if len(df_hist) >= 25:
            vol = df_hist["Volume"] if "Volume" in df_hist.columns else pd.Series(dtype=float)
            vol_avg_20 = vol.rolling(20).mean()
            vol_surge_5d = float(vol.iloc[-5] / vol_avg_20.iloc[-5]) if vol_avg_20.iloc[-5] > 0 else 1.0
            all_features["Volume_Delta_5d"] = float(vol_surge_now - vol_surge_5d)
        else:
            all_features["Volume_Delta_5d"] = 0.0
    except Exception:
        all_features["Volume_Delta_5d"] = 0.0

    # Momentum_Acceleration: short-term momentum vs medium-term pace
    ret_5d = all_features.get("Return_5d", 0.0)
    ret_20d = all_features.get("Return_20d", 0.0)
    all_features["Momentum_Acceleration"] = ret_5d - (ret_20d - ret_5d) / 3.0

    # ── 3 Interaction Features ──────────────────────────────────────

    vcp = all_features.get("VCP_Ratio", 1.0)
    rs = all_features.get("RS_vs_SPY_20d", 0.0)
    mom_cons = all_features.get("Momentum_Consistency", 0.5)
    vol_surge = all_features.get("Volume_Surge", 1.0)
    squeeze = all_features.get("Squeeze_On_Flag", 0.0)

    all_features["VCP_x_RS"] = vcp * rs
    all_features["Momentum_x_Volume"] = mom_cons * vol_surge
    all_features["Squeeze_x_Volume"] = squeeze * vol_surge

    # ── Filter to exactly the 20 V3.5 features ─────────────────────
    v3_5_names = get_feature_names("v3.5")
    try:
        from core.feature_registry import get_feature_defaults, get_feature_ranges
        defaults = get_feature_defaults("v3.5")
        ranges = get_feature_ranges("v3.5")
    except Exception:
        defaults, ranges = {}, {}

    result: Dict[str, float] = {}
    for feat in v3_5_names:
        val = all_features.get(feat)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            val = defaults.get(feat, 0.0)
        result[feat] = float(val)

    # Clip to valid ranges
    for feat, (lo, hi) in ranges.items():
        if feat in result:
            result[feat] = float(np.clip(result[feat], lo, hi))

    return result


def build_all_ml_features_v3_6(
    row: pd.Series,
    df_hist: pd.DataFrame,
    market_context: Optional[Dict[str, float]] = None,
    sector_context: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Build 23-feature vector for V3.6 model.

    V3.6 = V3.5 (20 features) + 3 trend/momentum indicators:
      - ADX: Average Directional Index (trend strength, 0-100)
      - MACD_Hist: MACD histogram / price (momentum divergence)
      - MA50_Slope: 50-day MA pct change over 10d (trend direction)

    All features are OHLCV-derived (no external APIs needed).

    Args:
        row: Series with technical indicators from build_technical_indicators()
        df_hist: Historical OHLCV DataFrame
        market_context: Optional market regime/SPY data
        sector_context: Optional sector relative data

    Returns:
        Dict with all 23 V3.6 features.
    """
    # Start with all V3.5 features (20 features)
    all_features = build_all_ml_features_v3_5(row, df_hist, market_context, sector_context)

    close = df_hist['Close'] if 'Close' in df_hist.columns else pd.Series(dtype=float)
    high = df_hist['High'] if 'High' in df_hist.columns else close
    low = df_hist['Low'] if 'Low' in df_hist.columns else close

    # ── ADX: Average Directional Index ────────────────────────────
    try:
        if len(df_hist) >= 30:
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            # Zero out when the other direction is larger
            plus_dm = plus_dm.where(plus_dm >= minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm >= plus_dm, 0)

            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean()

            plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
            adx_series = dx.rolling(14).mean()

            adx_val = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 25.0
            all_features["ADX"] = adx_val
        else:
            all_features["ADX"] = 25.0
    except Exception:
        all_features["ADX"] = 25.0

    # ── MACD Histogram (normalized by price) ──────────────────────
    try:
        if len(close) >= 35:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = (macd_line - signal_line) / close.replace(0, np.nan)

            val = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0.0
            all_features["MACD_Hist"] = val
        else:
            all_features["MACD_Hist"] = 0.0
    except Exception:
        all_features["MACD_Hist"] = 0.0

    # ── MA50_Slope: 50-day MA trend direction ─────────────────────
    try:
        # Try getting from row first (already computed in indicators.py)
        ma50_slope = None
        for key in ["MA50_Slope", "ma50_slope"]:
            val = row.get(key) if hasattr(row, 'get') else getattr(row, key, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ma50_slope = float(val)
                break

        if ma50_slope is None and len(close) >= 60:
            ma50 = close.rolling(50).mean()
            slope = ma50.pct_change(10)
            ma50_slope = float(slope.iloc[-1]) if pd.notna(slope.iloc[-1]) else 0.0

        all_features["MA50_Slope"] = ma50_slope if ma50_slope is not None else 0.0
    except Exception:
        all_features["MA50_Slope"] = 0.0

    # ── Filter to exactly the 23 V3.6 features ──────────────────
    v3_6_names = get_feature_names("v3.6")
    try:
        from core.feature_registry import get_feature_defaults, get_feature_ranges
        defaults = get_feature_defaults("v3.6")
        ranges = get_feature_ranges("v3.6")
    except Exception:
        defaults, ranges = {}, {}

    result: Dict[str, float] = {}
    for feat in v3_6_names:
        val = all_features.get(feat)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            val = defaults.get(feat, 0.0)
        result[feat] = float(val)

    # Clip to valid ranges
    for feat, (lo, hi) in ranges.items():
        if feat in result:
            result[feat] = float(np.clip(result[feat], lo, hi))

    return result


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
        'SPY_20d_ret': 0.0,
        'SPY_60d_ret': 0.0,
    }

    # Try to extract from row
    for key in context.keys():
        if key in row.index:
            val = row[key]
            if pd.notna(val):
                context[key] = float(val)

    return context


# =============================================================================
# V4.0 Feature Builder — 30 features (19 technical + 7 fundamental + 4 alt)
# =============================================================================

# Population statistics for z-scoring at inference time.
# Updated by training pipeline when a new model is saved to metadata.json.
# These are reasonable defaults for the US large-cap universe.
_FUNDAMENTAL_STATS = {
    "ROE_mean": 15.0,
    "ROE_std": 25.0,
    "DE_mean": 1.2,
    "DE_std": 1.5,
    "PE_median": 22.0,
}


def load_fundamental_stats(stats_dict: Optional[Dict[str, float]] = None) -> None:
    """Update population statistics from model metadata (called at load time)."""
    global _FUNDAMENTAL_STATS
    if stats_dict:
        _FUNDAMENTAL_STATS.update(stats_dict)


