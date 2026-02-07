"""
ML Feature Builder V4 - Enhanced Feature Engineering

Computes all 52 V4 features for ML training and inference.
Includes:
- Technical indicators (price action, volatility, momentum)
- Sentiment features (news, social)
- Institutional signals (ownership, insider trades)
- Earnings context
- Cross-asset context (VIX)

Usage:
    from core.ml_feature_builder_v4 import build_features_v4
    features = build_features_v4(price_df, ticker, context)
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple, List
from datetime import datetime, timedelta

from core.feature_registry_v4 import (
    get_feature_names_v4, 
    get_feature_defaults_v4,
    FEATURE_COUNT_V4
)

logger = logging.getLogger(__name__)


# =============================================================================
# TECHNICAL INDICATOR HELPERS
# =============================================================================

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI for the latest bar."""
    if len(prices) < period + 1:
        return 50.0
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR as percentage of price."""
    if len(df) < period + 1:
        return 0.02
    
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    latest_close = close.iloc[-1]
    
    if latest_close > 0:
        return float(atr.iloc[-1] / latest_close)
    return 0.02


def compute_returns(df: pd.DataFrame) -> Dict[str, float]:
    """Compute multi-period returns."""
    close = df['close'] if 'close' in df.columns else df['Close']
    
    returns = {}
    for days, name in [(5, "Return_5d"), (10, "Return_10d"), (20, "Return_20d")]:
        if len(close) >= days + 1:
            ret = (close.iloc[-1] / close.iloc[-(days+1)]) - 1
            returns[name] = float(ret)
        else:
            returns[name] = 0.0
    
    return returns


def compute_volatility_features(df: pd.DataFrame) -> Dict[str, float]:
    """Compute volatility-related features."""
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    features = {}
    
    # VCP Ratio: ATR(10) / ATR(30)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr10 = tr.rolling(10).mean().iloc[-1] if len(tr) >= 10 else np.nan
    atr30 = tr.rolling(30).mean().iloc[-1] if len(tr) >= 30 else np.nan
    
    features["VCP_Ratio"] = float(atr10 / atr30) if atr30 and atr30 > 0 else 1.0
    
    # Vol Contraction: ATR(20) / ATR(50)
    atr20 = tr.rolling(20).mean().iloc[-1] if len(tr) >= 20 else np.nan
    atr50 = tr.rolling(50).mean().iloc[-1] if len(tr) >= 50 else np.nan
    features["Vol_Contraction_Ratio"] = float(atr20 / atr50) if atr50 and atr50 > 0 else 1.0
    
    # Tightness Ratio: range_5d / range_20d
    range_5d = high.tail(5).max() - low.tail(5).min()
    range_20d = high.tail(20).max() - low.tail(20).min()
    features["Tightness_Ratio"] = float(range_5d / range_20d) if range_20d > 0 else 1.0
    
    # Distance from 52w high
    if len(close) >= 252:
        high_52w = high.tail(252).max()
        features["Dist_From_52w_High"] = float((close.iloc[-1] / high_52w) - 1)
    else:
        features["Dist_From_52w_High"] = 0.0
    
    # Consolidation tightness
    if len(df) >= 20:
        h20 = high.tail(20).max()
        l20 = low.tail(20).min()
        avg20 = close.tail(20).mean()
        features["Consolidation_Tightness"] = float((h20 - l20) / avg20) if avg20 > 0 else 0.1
    else:
        features["Consolidation_Tightness"] = 0.1
    
    # Squeeze detection (Bollinger Bands inside Keltner Channel)
    if len(df) >= 20:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        
        atr = tr.rolling(20).mean()
        kc_upper = sma20 + 1.5 * atr
        kc_lower = sma20 - 1.5 * atr
        
        # Squeeze is on when BB is inside KC
        squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
        features["Squeeze_On_Flag"] = 1.0 if squeeze else 0.0
    else:
        features["Squeeze_On_Flag"] = 0.0
    
    return features


def compute_volume_features(df: pd.DataFrame) -> Dict[str, float]:
    """Compute volume-related features."""
    close = df['close'] if 'close' in df.columns else df['Close']
    volume = df['volume'] if 'volume' in df.columns else df['Volume']
    
    features = {}
    
    # Volume Surge
    vol_5d = volume.tail(5).mean()
    vol_20d = volume.tail(20).mean()
    features["Volume_Surge"] = float(vol_5d / vol_20d) if vol_20d > 0 else 1.0
    
    # Volume Ratio 20d
    features["Volume_Ratio_20d"] = float(volume.iloc[-1] / vol_20d) if vol_20d > 0 else 1.0
    
    # Up/Down Volume Ratio
    daily_change = close.diff()
    up_vol = volume.where(daily_change > 0, 0).tail(20).sum()
    down_vol = volume.where(daily_change < 0, 0).tail(20).sum()
    features["Up_Down_Volume_Ratio"] = float(up_vol / down_vol) if down_vol > 0 else 1.0
    
    # Up Volume Ratio
    total_vol = up_vol + down_vol
    features["Up_Volume_Ratio"] = float(up_vol / total_vol) if total_vol > 0 else 0.5
    
    # Volume Trend (OBV-based)
    obv = (np.sign(daily_change) * volume).cumsum()
    if len(obv) >= 20:
        obv_change = (obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) if obv.iloc[-20] != 0 else 0
        features["Volume_Trend"] = float(np.clip(obv_change, -1, 1))
    else:
        features["Volume_Trend"] = 0.0
    
    # Momentum Consistency (% positive days)
    features["Momentum_Consistency"] = float((daily_change.tail(20) > 0).mean())
    
    # Volume-Price Confirmation
    price_up = daily_change.tail(5).sum() > 0
    vol_up = volume.tail(5).mean() > volume.tail(20).mean()
    features["Volume_Price_Confirm"] = 1.0 if price_up and vol_up else 0.0
    
    # Relative Volume Rank (vs 60d)
    if len(volume) >= 60:
        vol_percentile = (volume.tail(60) <= volume.iloc[-1]).mean()
        features["Relative_Volume_Rank"] = float(vol_percentile)
    else:
        features["Relative_Volume_Rank"] = 0.5
    
    return features


def compute_price_action_features(df: pd.DataFrame) -> Dict[str, float]:
    """Compute price action features."""
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    features = {}
    
    # Distance from 52w low
    if len(close) >= 252:
        low_52w = low.tail(252).min()
        features["Distance_From_52w_Low"] = float((close.iloc[-1] - low_52w) / low_52w) if low_52w > 0 else 0.5
    else:
        features["Distance_From_52w_Low"] = 0.5
    
    # Days since 52w high (normalized)
    if len(close) >= 252:
        idx_52w_high = high.tail(252).idxmax()
        if isinstance(idx_52w_high, (pd.Timestamp, datetime)):
            days_since = (df.index[-1] - idx_52w_high).days if isinstance(df.index[-1], (pd.Timestamp, datetime)) else 126
        else:
            days_since = 126
        features["Days_Since_52w_High"] = float(min(1.0, days_since / 252))
    else:
        features["Days_Since_52w_High"] = 0.5
    
    # MA features
    sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.iloc[-1]
    
    features["Price_vs_SMA50"] = float((close.iloc[-1] - sma50) / sma50) if sma50 > 0 else 0.0
    features["Price_vs_SMA200"] = float((close.iloc[-1] - sma200) / sma200) if sma200 > 0 else 0.0
    features["SMA50_vs_SMA200"] = float((sma50 - sma200) / sma200) if sma200 > 0 else 0.0
    
    # MA Alignment
    if len(close) >= 200:
        aligned = (close.iloc[-1] > sma20) and (sma20 > sma50) and (sma50 > sma200)
        features["MA_Alignment"] = 1.0 if aligned else 0.0
    else:
        features["MA_Alignment"] = 0.5
    
    # MA Slope
    if len(close) >= 25:
        sma20_series = close.rolling(20).mean()
        slope = (sma20_series.iloc[-1] - sma20_series.iloc[-5]) / sma20_series.iloc[-5] if sma20_series.iloc[-5] > 0 else 0
        features["MA_Slope_20d"] = float(np.clip(slope, -0.3, 0.3))
    else:
        features["MA_Slope_20d"] = 0.0
    
    # Distance to resistance (20d high)
    h20 = high.tail(20).max()
    features["Distance_To_Resistance"] = float((h20 - close.iloc[-1]) / close.iloc[-1]) if close.iloc[-1] > 0 else 0.05
    
    # Support strength
    l20 = low.tail(20).min()
    if len(close) >= 20:
        near_support = (close.tail(20) <= l20 * 1.02).mean()
        features["Support_Strength"] = float(near_support)
    else:
        features["Support_Strength"] = 0.2
    
    return features


def compute_streak_features(df: pd.DataFrame) -> Dict[str, float]:
    """Compute streak and pattern features."""
    close = df['close'] if 'close' in df.columns else df['Close']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    
    features = {}
    
    # Up/Down streaks
    daily_change = close.diff()
    
    up_streak = 0
    down_streak = 0
    for change in daily_change.tail(10).values[::-1]:
        if change > 0:
            up_streak += 1
            down_streak = 0
        elif change < 0:
            down_streak += 1
            up_streak = 0
        else:
            break
    
    features["UpStreak_Days"] = float(min(up_streak, 10))
    features["DownStreak_Days"] = float(min(down_streak, 10))
    
    # Overnight gap average
    if 'open' in df.columns or 'Open' in df.columns:
        open_col = df['open'] if 'open' in df.columns else df['Open']
        gaps = (open_col - close.shift()) / close.shift()
        features["OvernightGap_Avg"] = float(gaps.tail(5).mean()) if len(gaps) >= 5 else 0.0
    else:
        features["OvernightGap_Avg"] = 0.0
    
    # Intraday range
    ranges = (high - low) / close
    features["Range_Pct_10d"] = float(ranges.tail(10).mean()) if len(ranges) >= 10 else 0.02
    
    return features


# =============================================================================
# MARKET CONTEXT FEATURES
# =============================================================================

def compute_market_context_features(
    spy_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """Compute market regime and context features."""
    features = {
        "Market_Regime": 0.0,
        "Market_Volatility": 0.15,
        "Market_Trend": 0.0,
        "High_Volatility": 0.0,
        "VIX_Level": 0.15
    }
    
    if spy_df is not None and len(spy_df) >= 50:
        close = spy_df['close'] if 'close' in spy_df.columns else spy_df['Close']
        
        # Market trend (50d return)
        ret_50d = (close.iloc[-1] / close.iloc[-50]) - 1 if len(close) >= 50 else 0
        features["Market_Trend"] = float(np.clip(ret_50d, -0.5, 0.5))
        
        # Market regime
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50
        
        if close.iloc[-1] > sma50 and sma50 > sma200:
            features["Market_Regime"] = 1.0  # Bullish
        elif close.iloc[-1] < sma50 and sma50 < sma200:
            features["Market_Regime"] = -1.0  # Bearish
        else:
            features["Market_Regime"] = 0.0  # Sideways
        
        # Market volatility (annualized 20d)
        returns = close.pct_change()
        vol_20d = returns.tail(20).std() * np.sqrt(252)
        features["Market_Volatility"] = float(np.clip(vol_20d, 0.05, 0.8))
        
        # High volatility flag
        if len(returns) >= 60:
            vol_60d = returns.tail(60).std() * np.sqrt(252)
            features["High_Volatility"] = 1.0 if vol_20d > vol_60d * 1.2 else 0.0
    
    if vix_df is not None and len(vix_df) > 0:
        close = vix_df['close'] if 'close' in vix_df.columns else vix_df['Close']
        features["VIX_Level"] = float(close.iloc[-1] / 100)  # Normalize VIX
    
    return features


def compute_relative_strength_features(
    stock_df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """Compute relative strength vs benchmark and sector."""
    close = stock_df['close'] if 'close' in stock_df.columns else stock_df['Close']
    
    features = {
        "RS_vs_SPY_20d": 0.0,
        "RS_vs_SPY_60d": 0.0,
        "RS_Momentum": 0.0,
        "Sector_RS": 0.0,
        "Sector_Momentum": 0.0,
        "Sector_Rank": 0.5
    }
    
    # Returns
    stock_ret_20d = (close.iloc[-1] / close.iloc[-21]) - 1 if len(close) >= 21 else 0
    stock_ret_60d = (close.iloc[-1] / close.iloc[-61]) - 1 if len(close) >= 61 else 0
    stock_ret_5d = (close.iloc[-1] / close.iloc[-6]) - 1 if len(close) >= 6 else 0
    
    if spy_df is not None and len(spy_df) >= 61:
        spy_close = spy_df['close'] if 'close' in spy_df.columns else spy_df['Close']
        spy_ret_20d = (spy_close.iloc[-1] / spy_close.iloc[-21]) - 1 if len(spy_close) >= 21 else 0
        spy_ret_60d = (spy_close.iloc[-1] / spy_close.iloc[-61]) - 1 if len(spy_close) >= 61 else 0
        
        features["RS_vs_SPY_20d"] = float(stock_ret_20d - spy_ret_20d)
        features["RS_vs_SPY_60d"] = float(stock_ret_60d - spy_ret_60d)
        features["RS_Momentum"] = features["RS_vs_SPY_20d"] - features["RS_vs_SPY_60d"]
    
    if sector_df is not None and len(sector_df) >= 21:
        sector_close = sector_df['close'] if 'close' in sector_df.columns else sector_df['Close']
        sector_ret_20d = (sector_close.iloc[-1] / sector_close.iloc[-21]) - 1 if len(sector_close) >= 21 else 0
        sector_ret_5d = (sector_close.iloc[-1] / sector_close.iloc[-6]) - 1 if len(sector_close) >= 6 else 0
        
        features["Sector_RS"] = float(stock_ret_20d - sector_ret_20d)
        features["Sector_Momentum"] = float(sector_ret_20d)
        features["Sector_Rank"] = 1.0 if stock_ret_5d > sector_ret_5d else 0.0
    
    return features


# =============================================================================
# MAIN FEATURE BUILDER
# =============================================================================

def build_features_v4(
    price_df: pd.DataFrame,
    ticker: str,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    sentiment_data: Optional[Dict] = None,
    earnings_data: Optional[Dict] = None,
    institutional_data: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Build all V4 features for a single stock.
    
    Args:
        price_df: OHLCV DataFrame with at least 200 rows
        ticker: Stock ticker symbol
        spy_df: SPY price DataFrame for market context
        sector_df: Sector ETF price DataFrame
        vix_df: VIX price DataFrame
        sentiment_data: Pre-fetched sentiment data (optional)
        earnings_data: Pre-fetched earnings data (optional)
        institutional_data: Pre-fetched institutional data (optional)
    
    Returns:
        Dict with all 52 V4 features
    """
    # Start with defaults
    features = get_feature_defaults_v4()
    
    try:
        # Normalize column names
        df = price_df.copy()
        df.columns = df.columns.str.lower()
        
        if len(df) < 50:
            logger.warning(f"{ticker}: Insufficient price history ({len(df)} rows)")
            return features
        
        # Technical base
        features["RSI"] = compute_rsi(df['close'])
        features["ATR_Pct"] = compute_atr_pct(df)
        
        # Returns
        returns = compute_returns(df)
        features.update(returns)
        
        # Volatility
        vol_features = compute_volatility_features(df)
        features.update(vol_features)
        
        # Volume
        vol_features = compute_volume_features(df)
        features.update(vol_features)
        
        # Price action
        pa_features = compute_price_action_features(df)
        features.update(pa_features)
        
        # Streaks
        streak_features = compute_streak_features(df)
        features.update(streak_features)
        
        # Market context
        market_features = compute_market_context_features(spy_df, vix_df)
        features.update(market_features)
        
        # Relative strength
        rs_features = compute_relative_strength_features(df, spy_df, sector_df)
        features.update(rs_features)
        
        # Sentiment features (from pre-fetched data or defaults)
        if sentiment_data:
            features["News_Sentiment_7d"] = sentiment_data.get("sentiment_avg", 0.0)
            features["News_Volume_7d"] = sentiment_data.get("news_count", 0.5)
            features["Sentiment_Momentum"] = sentiment_data.get("sentiment_momentum", 0.0)
        features["Social_Buzz_Score"] = 0.5  # Placeholder
        
        # Earnings features
        if earnings_data:
            features["Days_To_Earnings"] = earnings_data.get("days_to_earnings", 45)
            features["In_Earnings_Window"] = earnings_data.get("in_earnings_window", 0)
            features["Post_Earnings_Drift"] = earnings_data.get("last_earnings_surprise", 0.0)
        
        # Institutional features
        if institutional_data:
            features["Institutional_Change_QoQ"] = institutional_data.get("institutional_change_qoq", 0.0)
            features["Insider_Net_30d"] = institutional_data.get("insider_net_30d", 0.0)
            features["Analyst_Rating_Change"] = institutional_data.get("analyst_rating_change", 0.0)
            features["Price_Target_Upside"] = institutional_data.get("price_target_upside", 0.1)
        
        # Validate ranges
        ranges = {
            "RSI": (0, 100),
            "ATR_Pct": (0.001, 0.5),
            "VCP_Ratio": (0.1, 5.0),
            "Tightness_Ratio": (0.05, 2.0),
        }
        
        for name, (min_val, max_val) in ranges.items():
            if name in features:
                features[name] = float(np.clip(features[name], min_val, max_val))
        
    except Exception as e:
        logger.warning(f"Error building features for {ticker}: {e}")
    
    return features


def build_features_batch_v4(
    tickers: List[str],
    price_data: Dict[str, pd.DataFrame],
    spy_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    sector_mapping: Optional[Dict[str, pd.DataFrame]] = None,
    include_sentiment: bool = False
) -> pd.DataFrame:
    """
    Build V4 features for a batch of tickers.
    
    Args:
        tickers: List of ticker symbols
        price_data: Dict mapping ticker -> price DataFrame
        spy_df: SPY price DataFrame
        vix_df: VIX price DataFrame
        sector_mapping: Dict mapping ticker -> sector ETF DataFrame
        include_sentiment: Whether to fetch sentiment data (slower)
    
    Returns:
        DataFrame with Ticker column + all 52 V4 features
    """
    # Optionally fetch sentiment data in batch
    sentiment_cache = {}
    if include_sentiment:
        try:
            from core.sentiment_data import fetch_alternative_data_batch
            alt_df = fetch_alternative_data_batch(tickers)
            for _, row in alt_df.iterrows():
                t = row.get("Ticker")
                if t:
                    sentiment_cache[t] = row.to_dict()
        except Exception as e:
            logger.warning(f"Sentiment fetch failed: {e}")
    
    rows = []
    for ticker in tickers:
        price_df = price_data.get(ticker)
        if price_df is None or len(price_df) < 50:
            continue
        
        sector_df = sector_mapping.get(ticker) if sector_mapping else None
        sentiment_data = sentiment_cache.get(ticker)
        
        features = build_features_v4(
            price_df=price_df,
            ticker=ticker,
            spy_df=spy_df,
            sector_df=sector_df,
            vix_df=vix_df,
            sentiment_data=sentiment_data
        )
        
        features["Ticker"] = ticker
        rows.append(features)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    cols = ["Ticker"] + get_feature_names_v4()
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]
    
    logger.info(f"Built {len(df)} feature rows with {len(df.columns)} columns")
    return df
