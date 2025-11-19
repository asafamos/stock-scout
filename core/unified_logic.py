"""
Unified logic for Stock Scout - shared between live app and backtest.
Ensures exact same calculations for consistency and validation.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from core.config import get_config
from core.scoring.fundamental import compute_fundamental_score_with_breakdown
from core.classification import apply_classification


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential moving average."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    range1 = high - low
    range2 = np.abs(high - prev_close)
    range3 = np.abs(low - prev_close)
    
    true_range = np.maximum(range1, np.maximum(range2, range3))
    atr = true_range.rolling(window=period).mean()
    return atr


def compute_momentum_consistency(close: pd.Series, lookback: int = 14) -> pd.Series:
    """Fraction of up days in rolling window."""
    up_days = (close.diff() > 0).astype(int)
    consistency = up_days.rolling(window=lookback).mean()
    return consistency


def compute_volume_surge(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Current volume / 20-day average."""
    avg_volume = volume.rolling(window=lookback).mean()
    surge = volume / avg_volume
    return surge


def compute_reward_risk(close: pd.Series, lookback: int = 20) -> pd.Series:
    """(20d_high - close) / (close - 20d_low), capped at 10."""
    high_20d = close.rolling(window=lookback).max()
    low_20d = close.rolling(window=lookback).min()
    
    reward = high_20d - close
    risk = close - low_20d
    
    rr = reward / risk.replace(0, np.nan)
    rr = rr.clip(upper=10)
    return rr


def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all technical indicators from OHLCV data.
    Returns DataFrame with indicators as columns.
    """
    result = pd.DataFrame(index=df.index)
    
    # Extract series (handle multi-level columns from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    else:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
    
    # Moving averages
    result['MA20'] = close.rolling(window=20).mean()
    result['MA50'] = close.rolling(window=50).mean()
    result['MA200'] = close.rolling(window=200).mean()
    
    # Technical indicators
    result['RSI'] = compute_rsi(close, period=14)
    result['ATR'] = compute_atr(df, period=14)
    result['ATR_Pct'] = result['ATR'] / close
    
    # Price-based features
    result['Overext'] = (close / result['MA50']) - 1
    result['Near52w'] = (close / close.rolling(window=252).max()) * 100
    
    # Momentum and volume
    result['MomCons'] = compute_momentum_consistency(close, lookback=14)
    result['VolSurge'] = compute_volume_surge(volume, lookback=20)
    result['RR'] = compute_reward_risk(close, lookback=20)
    
    # Derived features for ML
    result['RR_MomCons'] = result['RR'] * result['MomCons']
    result['RSI_Neutral'] = (result['RSI'] - 50).abs()
    result['RSI_Squared'] = result['RSI'] ** 2
    result['Risk_Score'] = result['Overext'].abs() + result['ATR_Pct']
    result['Vol_Mom'] = result['VolSurge'] * result['MomCons']
    result['Overext_Mom_Div'] = result['Overext'] * result['MomCons']
    result['RR_Risk_Adj'] = result['RR'] / (1 + result['Overext'].abs())
    
    # ATR regime (simplified binning)
    result['ATR_Regime'] = pd.cut(result['ATR_Pct'], bins=[0, 0.02, 0.04, 1.0], labels=[1, 2, 3]).astype(float)
    
    # Timing indicators
    result['Vol_Breakout'] = volume > (volume.rolling(20).mean() * 1.5)
    result['Price_Breakout'] = close > close.rolling(20).quantile(0.90)
    result['Mom_Acceleration'] = (close.pct_change(5) > close.shift(5).pct_change(5))
    
    # Copy price/volume for reference
    result['Close'] = close
    result['Volume'] = volume
    result['High'] = high
    result['Low'] = low
    
    return result


def apply_technical_filters(row: pd.Series, strict: bool = True, relaxed: bool = False) -> bool:
    """
    Apply technical filters to a stock.
    
    Args:
        row: Series with technical indicators
        strict: If True, use Core filters; if False, use Speculative filters
        
    Returns:
        True if stock passes filters, False otherwise
    """
    if relaxed:
        # Ultra-relaxed (Momentum-first) mode
        rsi_min, rsi_max = 15, 90
        max_overext = 0.40
        max_atr_pct = 0.28
        min_rr = -1.5
        min_mom_cons = 0.10
    elif strict:
        # Core filters (loosened)
        rsi_min, rsi_max = 25, 75
        max_overext = 0.20
        max_atr_pct = 0.12
        min_rr = -0.5
        min_mom_cons = 0.40
    else:
        # Speculative filters (relaxed tier)
        rsi_min, rsi_max = 20, 85
        max_overext = 0.30
        max_atr_pct = 0.22
        min_rr = -1.0
        min_mom_cons = 0.20
    
    # RSI check
    rsi = row.get('RSI', np.nan)
    if pd.notna(rsi) and (rsi < rsi_min or rsi > rsi_max):
        return False
    
    # Overextension check
    overext = row.get('Overext', np.nan)
    if pd.notna(overext) and abs(overext) > max_overext:
        return False
    
    # ATR/Price check
    atr_pct = row.get('ATR_Pct', np.nan)
    if pd.notna(atr_pct) and atr_pct > max_atr_pct:
        return False
    
    # Reward/Risk check
    rr = row.get('RR', np.nan)
    if pd.notna(rr) and rr < min_rr:
        return False
    
    # Momentum consistency check
    mom_cons = row.get('MomCons', np.nan)
    if pd.notna(mom_cons) and mom_cons < min_mom_cons:
        return False
    
    return True


def score_with_ml_model(row: pd.Series, model_data: Optional[Dict] = None) -> float:
    """
    Score stock with ML model.
    
    Uses retrained XGBoost model (14 features, AUC 0.555) for probability scoring.
    Model predicts probability of positive 20-day return.
    
    Args:
        row: Series with technical indicators
        model_data: Dict with 'model' and 'feature_names' keys
        
    Returns:
        Probability between 0 and 1 (0.5 if model unavailable)
    """
    if model_data is None or model_data.get('model') is None:
        return 0.5  # Neutral if no model
    
    try:
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Extract base features from row
        base_features = ['RSI', 'ATR_Pct', 'Overext', 'RR', 'MomCons', 'VolSurge']
        features = {}
        for fname in base_features:
            if fname in row.index:
                features[fname] = row[fname]
            else:
                # Provide sensible defaults
                features[fname] = 0.5 if fname == 'MomCons' else 1.0
        
        # Engineer derived features (must match train_recommender.py)
        X = pd.DataFrame([features])
        
        # RR_MomCons
        if 'RR' in X.columns and 'MomCons' in X.columns:
            X['RR_MomCons'] = X['RR'] * X['MomCons']
        
        # RSI_Neutral, RSI_Squared
        if 'RSI' in X.columns:
            X['RSI_Neutral'] = (X['RSI'] - 50).abs()
            X['RSI_Squared'] = X['RSI'] ** 2
        
        # Risk_Score
        if 'Overext' in X.columns and 'ATR_Pct' in X.columns:
            X['Risk_Score'] = X['Overext'].abs() + X['ATR_Pct']
        
        # Vol_Mom
        if 'VolSurge' in X.columns and 'MomCons' in X.columns:
            X['Vol_Mom'] = X['VolSurge'] * X['MomCons']
        
        # Overext_Mom_Div
        if 'Overext' in X.columns and 'MomCons' in X.columns:
            X['Overext_Mom_Div'] = X['Overext'] * X['MomCons']
        
        # RR_Risk_Adj
        if 'RR' in X.columns and 'Overext' in X.columns:
            X['RR_Risk_Adj'] = X['RR'] / (1 + X['Overext'].abs())
        
        # ATR_Regime
        if 'ATR_Pct' in X.columns:
            # Simple regime: low (<0.02), med (0.02-0.04), high (>0.04)
            atr_val = X['ATR_Pct'].iloc[0]
            if atr_val < 0.02:
                X['ATR_Regime'] = 1.0
            elif atr_val < 0.04:
                X['ATR_Regime'] = 2.0
            else:
                X['ATR_Regime'] = 3.0
        
        # Ensure all model features exist
        for fname in feature_names:
            if fname not in X.columns:
                X[fname] = 0.5  # fallback
        
        # Select features in correct order
        X_model = X[feature_names].fillna(0.5)
        
        # Predict probability
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(X_model.values)[0][1])
        else:
            # XGBoost API
            prob = float(model.predict_proba(X_model)[0][1])
        
        # Clip to valid range
        return max(0.0, min(1.0, prob))
        
    except Exception as e:
        # Fallback to neutral on any error
        return 0.5


def compute_forward_returns(
    df: pd.DataFrame, 
    date: pd.Timestamp, 
    horizons: List[int] = [5, 10, 20],
    benchmark_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Compute forward returns from a specific date.
    
    Args:
        df: DataFrame with Close prices (indexed by date)
        date: Starting date
        horizons: List of forward periods in days
        benchmark_df: Optional benchmark DataFrame for relative returns
        
    Returns:
        Dict with keys like 'R_5d', 'Excess_5d', etc. (all float values)
    """
    results = {}
    
    if date not in df.index:
        return {f'R_{h}d': np.nan for h in horizons}
    
    idx = df.index.get_loc(date)
    
    # Extract close price (handle potential Series)
    close_val = df.loc[date, 'Close']
    if isinstance(close_val, pd.Series):
        price_start = float(close_val.iloc[0])
    else:
        price_start = float(close_val)
    
    for h in horizons:
        end_idx = idx + h
        if end_idx >= len(df):
            results[f'R_{h}d'] = np.nan
            results[f'Excess_{h}d'] = np.nan
            continue
        
        end_date = df.index[end_idx]
        
        # Extract end price (handle potential Series)
        close_end_val = df.loc[end_date, 'Close']
        if isinstance(close_end_val, pd.Series):
            price_end = float(close_end_val.iloc[0])
        else:
            price_end = float(close_end_val)
        
        # Stock return
        ret = (price_end / price_start - 1) * 100
        results[f'R_{h}d'] = float(ret)
        
        # Benchmark return
        if benchmark_df is not None and date in benchmark_df.index:
            bench_idx = benchmark_df.index.get_loc(date)
            bench_end_idx = bench_idx + h
            
            if bench_end_idx < len(benchmark_df):
                bench_end_date = benchmark_df.index[bench_end_idx]
                
                bench_start_val = benchmark_df.loc[date, 'Close']
                bench_end_val = benchmark_df.loc[bench_end_date, 'Close']
                
                # Handle potential Series
                if isinstance(bench_start_val, pd.Series):
                    bench_start = float(bench_start_val.iloc[0])
                else:
                    bench_start = float(bench_start_val)
                    
                if isinstance(bench_end_val, pd.Series):
                    bench_end = float(bench_end_val.iloc[0])
                else:
                    bench_end = float(bench_end_val)
                
                bench_ret = (bench_end / bench_start - 1) * 100
                results[f'Excess_{h}d'] = float(ret - bench_ret)
            else:
                results[f'Excess_{h}d'] = np.nan
        else:
            results[f'Excess_{h}d'] = np.nan
    
    return results


def compute_technical_score(row: pd.Series, weights: Optional[Dict[str, float]] = None) -> float:
    """Compute a deterministic technical score (0-100) from indicators in `row`.
    
    NEW (Nov 2025): Volatility-adjusted scoring
    - Penalize extreme volatility (ATR > 6%)
    - Reward stable movers (ATR 2-4%)
    - Adjust RR by volatility (risk-adjusted RR)

    This function centralizes the simple scoring logic so both app and backtest
    use the same formula. `weights` is a mapping similar to CONFIG['WEIGHTS']
    but values will be normalized internally.
    """
    if weights is None:
        weights = {
            'ma': 0.2,
            'mom': 0.25,
            'rsi': 0.12,
            'near_high_bell': 0.10,
            'vol': 0.08,
            'overext': 0.06,
            'pullback': 0.05,
            'risk_reward': 0.06,
            'macd': 0.04,
            'adx': 0.04,
        }

    # Normalize weights to sum 1
    total_w = sum(weights.values())
    norm_w = {k: (v / total_w) for k, v in weights.items()}

    # Extract components from row (use safe defaults)
    ma_ok = float(row.get('MA_Aligned', 0.0)) if pd.notna(row.get('MA_Aligned', np.nan)) else 0.0
    mom = float(row.get('Momentum_Consistency', 0.0)) if pd.notna(row.get('Momentum_Consistency', np.nan)) else 0.0
    
    # RSI score: closer to mid-band (50) is neutral; reward 50-40/60 range
    rsi = row.get('RSI', np.nan)
    if pd.isna(rsi):
        rsi_score = 0.5
    else:
        # Map RSI 0-100 to 0-1 with preference for 25-75
        if 25 <= rsi <= 75:
            rsi_score = 1.0
        else:
            rsi_score = max(0.0, 1.0 - (abs(rsi - 50) - 25) / 50.0)

    vol = float(row.get('VolSurge', 1.0)) if pd.notna(row.get('VolSurge', np.nan)) else 1.0
    vol_score = min(2.0, vol) / 2.0

    overext = float(row.get('Overext', 0.0)) if pd.notna(row.get('Overext', np.nan)) else 0.0
    overext_score = max(0.0, 1.0 - (overext / 0.2))

    pullback = float(row.get('Near52w', np.nan)) / 100.0 if pd.notna(row.get('Near52w', np.nan)) else 0.5

    # VOLATILITY-ADJUSTED RISK/REWARD
    # Penalize high volatility, reward stable movers
    rr = float(row.get('RR', np.nan)) if pd.notna(row.get('RR', np.nan)) else 1.0
    atr_pct = float(row.get('ATR_Pct', np.nan)) if pd.notna(row.get('ATR_Pct', np.nan)) else 0.03
    
    # Volatility penalty/bonus
    if atr_pct > 0.06:  # >6% = extreme volatility
        vol_adjustment = 0.5  # Heavy penalty
    elif atr_pct > 0.05:  # 5-6% = high volatility
        vol_adjustment = 0.7
    elif atr_pct < 0.02:  # <2% = too stable (low opportunity)
        vol_adjustment = 0.8
    elif 0.02 <= atr_pct <= 0.04:  # Sweet spot: 2-4%
        vol_adjustment = 1.2  # Bonus!
    else:  # 4-5% = acceptable
        vol_adjustment = 1.0
    
    # Risk-adjusted RR = RR * volatility adjustment
    adjusted_rr = rr * vol_adjustment
    rr_score = max(0.0, min(1.0, (adjusted_rr + 1.0) / 5.0))

    macd = 1.0 if row.get('MACD_Pos', False) else 0.0
    adx = float(row.get('ADX14', 0.0)) if pd.notna(row.get('ADX14', np.nan)) else 0.0
    adx_score = np.clip((adx - 15.0) / 20.0, 0.0, 1.0)

    tech = (
        norm_w['ma'] * ma_ok
        + norm_w['mom'] * mom
        + norm_w['rsi'] * rsi_score
        + norm_w['near_high_bell'] * pullback
        + norm_w['vol'] * vol_score
        + norm_w['overext'] * overext_score
        + norm_w['pullback'] * pullback
        + norm_w['risk_reward'] * rr_score  # Now volatility-adjusted!
        + norm_w['macd'] * macd
        + norm_w['adx'] * adx_score
    )

    # Additional volatility penalty for extreme cases
    if atr_pct > 0.06:
        tech *= 0.85  # 15% penalty to overall score
    
    # scale to 0-100
    return float(np.clip(tech * 100.0, 0.0, 100.0))


def compute_final_score(tech_score: float, fundamental_score: Optional[float], ml_prob: float) -> float:
    """Unified final score = 0.55*technical + 0.25*fundamental + 0.20*ml_prob (ml_prob in 0-1)

    Returns value on 0-100 scale.
    """
    fund = 0.0 if fundamental_score is None or pd.isna(fundamental_score) else float(fundamental_score)
    final = 0.55 * tech_score + 0.25 * fund + 0.20 * (ml_prob * 100.0)
    return float(np.clip(final, 0.0, 100.0))


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch stock OHLCV data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty:
            return None
        return df
    except Exception:
        return None


# ============================================================================
# V2 SCORING ENGINE INTEGRATION (NON-BREAKING)
# ============================================================================

def score_ticker_v2(
    ticker: str,
    row: pd.Series,
    historical_df: Optional[pd.DataFrame] = None,
    enable_ml: bool = True,
    use_multi_source: bool = True
) -> Dict:
    """
    V2 scoring pipeline using multi-source data and unified scoring engine.
    
    This is a NON-BREAKING addition. V1 scoring remains unchanged.
    
    Args:
        ticker: Stock symbol
        row: Row from technical indicators DataFrame (contains RSI, ATR, etc.)
        historical_df: Full OHLCV DataFrame (optional, for additional calculations)
        enable_ml: Whether to use ML boost
        use_multi_source: Whether to fetch multi-source fundamentals
    
    Returns:
        Dict with all v2 scores and metadata:
        {
            "ticker": str,
            "fundamental_score_v2": float (0-100),
            "fundamental_confidence_v2": float (0-100),
            "technical_score_v2": float (0-100),
            "technical_confidence_v2": float (0-100),
            "rr_score_v2": float (0-100),
            "rr_confidence_v2": float (0-100),
            "reliability_score_v2": float (0-100),
            "conviction_v2_base": float (0-100),  # Before ML
            "conviction_v2_final": float (0-100),  # After ML
            "ml_probability": Optional[float],
            "ml_boost": float,
            "ml_status": str,
            "risk_meter_v2": float (0-100),
            "risk_label_v2": str,
            "warnings_v2": List[str],
            "sources_used": List[str],
            "disagreement_score": float,
            "breakdown": Dict  # Detailed component scores
        }
    """
    from core import scoring_engine
    from core import data_sources_v2
    from core import ml_integration
    
    result = {
        "ticker": ticker,
        "fundamental_score_v2": 50.0,
        "fundamental_confidence_v2": 0.0,
        "technical_score_v2": 50.0,
        "technical_confidence_v2": 0.0,
        "rr_score_v2": 50.0,
        "rr_confidence_v2": 0.0,
        "reliability_score_v2": 50.0,
        "conviction_v2_base": 50.0,
        "conviction_v2_final": 50.0,
        "ml_probability": None,
        "ml_boost": 0.0,
        "ml_status": "Not computed",
        "risk_meter_v2": 50.0,
        "risk_label_v2": "MODERATE",
        "warnings_v2": [],
        "sources_used": [],
        "disagreement_score": 0.0,
        "breakdown": {}
    }
    
    try:
        # ===== STEP 1: Fetch Multi-Source Fundamentals =====
        if use_multi_source:
            try:
                multi_source_data = data_sources_v2.fetch_multi_source_data(ticker)
                result["sources_used"] = multi_source_data.get("sources_used", [])
                result["disagreement_score"] = multi_source_data.get("disagreement_score", 0.0)
            except Exception as e:
                import logging
                logging.warning(f"Multi-source fetch failed for {ticker}: {e}")
                multi_source_data = {}
        else:
            multi_source_data = {}
        
        # ===== STEP 2: Calculate Fundamental Score =====
        fund_score, fund_confidence = scoring_engine.calculate_fundamental_score(
            pe=multi_source_data.get("pe"),
            ps=multi_source_data.get("ps"),
            pb=multi_source_data.get("pb"),
            roe=multi_source_data.get("roe"),
            margin=multi_source_data.get("margin"),
            rev_yoy=multi_source_data.get("rev_yoy"),
            eps_yoy=multi_source_data.get("eps_yoy"),
            debt_equity=multi_source_data.get("debt_equity")
        )
        
        result["fundamental_score_v2"] = fund_score
        result["fundamental_confidence_v2"] = fund_confidence
        
        # ===== STEP 3: Calculate Technical/Momentum Score =====
        # Extract technical indicators from row
        rsi = row.get("RSI", np.nan)
        atr = row.get("ATR", np.nan)
        close = row.get("Close", np.nan) if historical_df is None else historical_df['Close'].iloc[-1]
        atr_pct = (atr / close) if (np.isfinite(atr) and np.isfinite(close) and close > 0) else np.nan
        
        ma_aligned = row.get("MA_Aligned", False)
        overextension = row.get("OverextRatio", np.nan)
        volume_surge = row.get("Volume_Surge", np.nan)
        near_high = row.get("Near52w", np.nan)
        
        # Momentum periods (if available)
        mom_1m = row.get("Mom_1M", np.nan)
        mom_3m = row.get("Mom_3M", np.nan)
        mom_6m = row.get("Mom_6M", np.nan)
        
        tech_score, tech_confidence = scoring_engine.calculate_momentum_score(
            rsi=rsi,
            atr_pct=atr_pct,
            ma_aligned=ma_aligned,
            mom_1m=mom_1m,
            mom_3m=mom_3m,
            mom_6m=mom_6m,
            near_high=near_high,
            overextension=overextension,
            volume_surge=volume_surge
        )
        
        result["technical_score_v2"] = tech_score
        result["technical_confidence_v2"] = tech_confidence
        
        # ===== STEP 4: Calculate Risk/Reward Score =====
        rr_ratio = row.get("RewardRisk", np.nan)
        support = row.get("Support_20d", np.nan)
        resistance = row.get("Resistance_20d", np.nan)
        current_price = close
        
        rr_score, rr_confidence = scoring_engine.calculate_rr_score(
            rr_ratio=rr_ratio,
            atr=atr,
            support=support,
            resistance=resistance,
            current_price=current_price
        )
        
        result["rr_score_v2"] = rr_score
        result["rr_confidence_v2"] = rr_confidence
        
        # ===== STEP 5: Calculate Reliability Score =====
        price_sources = multi_source_data.get("price_sources", 1)
        fund_sources = len(result["sources_used"])
        price_std = multi_source_data.get("price_std", np.nan)
        price_mean = multi_source_data.get("price_mean", np.nan)
        
        # Data completeness: how many technical indicators are valid?
        tech_fields = ["RSI", "ATR", "MA_Aligned", "OverextRatio", "Volume_Surge", "Near52w", "RewardRisk"]
        valid_count = sum(1 for f in tech_fields if np.isfinite(row.get(f, np.nan)))
        data_completeness = (valid_count / len(tech_fields)) * 100.0
        
        reliability_score = scoring_engine.calculate_reliability_score(
            price_sources=price_sources,
            fund_sources=fund_sources,
            price_std=price_std,
            price_mean=price_mean,
            fundamental_confidence=fund_confidence,
            data_completeness=data_completeness
        )
        
        result["reliability_score_v2"] = reliability_score
        
        # ===== STEP 6: Calculate Base Conviction =====
        base_conviction, breakdown = scoring_engine.calculate_conviction_score(
            fundamental_score=fund_score,
            fundamental_confidence=fund_confidence,
            momentum_score=tech_score,
            momentum_confidence=tech_confidence,
            rr_score=rr_score,
            rr_confidence=rr_confidence,
            reliability_score=reliability_score,
            ml_probability=None  # Add ML separately
        )
        
        result["conviction_v2_base"] = base_conviction
        result["breakdown"] = breakdown
        
        # ===== STEP 7: Apply ML Boost (if enabled) =====
        if enable_ml:
            # Prepare data for ML
            ticker_data = {
                "near_high": near_high,
                "market_trend": row.get("Market_Trend", 0.0),
                "market_volatility": row.get("Market_Volatility", 0.02),
                "spy_rsi": row.get("SPY_RSI", 50.0),
                "relative_strength_20d": row.get("Relative_Strength_20d", 0.0),
                "dist_from_52w_high": row.get("Dist_From_52w_High", 0.1),
                "mom_acceleration": row.get("Mom_Acceleration", 0.0)
            }
            
            technical_indicators = {
                "rsi": rsi,
                "atr_pct": atr_pct,
                "overextension": overextension,
                "rr_ratio": rr_ratio,
                "momentum_consistency": row.get("Momentum_Consistency", 0.5),
                "volume_surge": volume_surge
            }
            
            fundamental_scores_dict = {
                "fund_score": fund_score,
                "fund_confidence": fund_confidence
            }
            
            final_conviction, ml_info = ml_integration.integrate_ml_with_conviction(
                base_conviction=base_conviction,
                ticker_data=ticker_data,
                technical_indicators=technical_indicators,
                fundamental_scores=fundamental_scores_dict,
                enable_ml=True
            )
            
            result["conviction_v2_final"] = final_conviction
            result["ml_probability"] = ml_info.get("ml_probability")
            result["ml_boost"] = ml_info.get("ml_boost", 0.0)
            result["ml_status"] = ml_info.get("ml_status", "Unknown")
        else:
            result["conviction_v2_final"] = base_conviction
            result["ml_status"] = "ML disabled"
        
        # ===== STEP 8: Calculate Risk Meter =====
        beta = multi_source_data.get("beta", np.nan)
        leverage = multi_source_data.get("debt_equity", np.nan)
        
        risk_meter, risk_label = scoring_engine.calculate_risk_meter(
            rr_ratio=rr_ratio,
            beta=beta,
            atr_pct=atr_pct,
            leverage=leverage
        )
        
        result["risk_meter_v2"] = risk_meter
        result["risk_label_v2"] = risk_label
        
        # ===== STEP 9: Generate Warnings =====
        warnings = scoring_engine.generate_warnings(
            rr_ratio=rr_ratio,
            fundamental_confidence=fund_confidence,
            beta=beta,
            atr_pct=atr_pct,
            reliability_score=reliability_score
        )
        
        result["warnings_v2"] = warnings
        
    except Exception as e:
        import logging
        logging.error(f"V2 scoring failed for {ticker}: {e}")
        result["warnings_v2"].append(f"⚠️ V2 scoring error: {str(e)[:100]}")
    
    return result


def batch_score_v2(
    tickers: List[str],
    indicators_df: pd.DataFrame,
    enable_ml: bool = True,
    use_multi_source: bool = True
) -> pd.DataFrame:
    """
    Batch v2 scoring for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        indicators_df: DataFrame with technical indicators (index = tickers)
        enable_ml: Whether to use ML
        use_multi_source: Whether to fetch multi-source data
    
    Returns:
        DataFrame with all v2 scores (one row per ticker)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    results = []
    
    for ticker in tickers:
        if ticker not in indicators_df.index:
            logger.warning(f"Ticker {ticker} not in indicators DataFrame")
            continue
        
        row = indicators_df.loc[ticker]
        
        try:
            v2_result = score_ticker_v2(
                ticker=ticker,
                row=row,
                enable_ml=enable_ml,
                use_multi_source=use_multi_source
            )
            results.append(v2_result)
        except Exception as e:
            logger.error(f"Failed to score {ticker}: {e}")
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df.set_index("ticker", inplace=True)
    
    return df
