import sys, os
# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

# Sector mapping for sector-relative features
from core.sector_mapping import get_stock_sector, get_sector_etf, get_all_sector_etfs
from core.api_keys import get_api_key
# Feature registry - Single Source of Truth for ML features
from core.feature_registry import get_feature_names, FEATURE_COUNT_V3_1
# Unified labelling logic
from core.ml_targets import make_label_20d
from core.ml_target_config import UP_THRESHOLD, DOWN_THRESHOLD
# Shared EnsembleClassifier for pickle compatibility
from core.ensemble import EnsembleClassifier

# --- CONFIG ---
# API key is lazily loaded when needed (not at import time) to allow tests to import this module
def _get_polygon_key() -> str:
    """Get Polygon API key - raises if not set."""
    return get_api_key("POLYGON_API_KEY", required=True)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")

# Get canonical feature list from registry (v3.1: 39 stock-picking features, no market-timers)
FEATURE_NAMES_V3 = get_feature_names("v3.1")
print(f"📋 Feature Registry v3.1: {len(FEATURE_NAMES_V3)} features")


def precision_at_k(y_true, y_pred_proba, k=20):
    """
    Calculate precision for top K predictions.
    
    This is the key metric for trading signals - we only trade the top K,
    so we need to know what fraction of those are actually winners.
    
    Args:
        y_true: Actual labels (0/1)
        y_pred_proba: Predicted probabilities for positive class
        k: Number of top predictions to evaluate
    
    Returns:
        Precision: fraction of top K predictions that are actually positive
    """
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return 0.0
    
    # Get indices of top K predictions (highest probability)
    top_k_idx = np.argsort(y_pred_proba)[-k:]
    
    # Calculate precision: how many of top K are actually winners?
    if hasattr(y_true, 'iloc'):
        precision = y_true.iloc[top_k_idx].mean()
    else:
        precision = y_true[top_k_idx].mean()
    
    return float(precision)


# --- FEATURE IMPORTANCE ANALYSIS ---
def analyze_feature_importance(model, feature_names, X_test, y_test, report_lines=None):
    """Analyze and display feature importance from multiple perspectives.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        X_test: Test features for permutation importance
        y_test: Test labels
        report_lines: Optional list to append report text to
    
    Returns:
        DataFrame with permutation importance results
    """
    from sklearn.inspection import permutation_importance
    
    if report_lines is None:
        report_lines = []
    
    header = "\n" + "="*60 + "\n📊 FEATURE IMPORTANCE ANALYSIS\n" + "="*60
    print(header)
    report_lines.append(header)
    
    # 1. Built-in importance (from HistGradientBoosting or ensemble average)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'builtin_importance': model.feature_importances_
        }).sort_values('builtin_importance', ascending=False)
        
        section = "\n🔹 Built-in Feature Importance (Top 15):"
        print(section)
        report_lines.append(section)
        
        for _, row in importance_df.head(15).iterrows():
            bar = "█" * int(row['builtin_importance'] * 50)
            line = f"   {row['feature']:28s} {row['builtin_importance']:.4f} {bar}"
            print(line)
            report_lines.append(line)
    
    # 2. Permutation importance (more reliable for model-agnostic analysis)
    print("\n   Computing permutation importance (this may take a moment)...")
    perm_importance = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1,
        scoring='roc_auc'
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    section = "\n🔹 Permutation Importance (AUC-based, Top 15):"
    print(section)
    report_lines.append(section)
    
    for _, row in perm_df.head(15).iterrows():
        bar = "█" * int(max(0, row['importance_mean']) * 200)  # Scale for visibility
        line = f"   {row['feature']:28s} {row['importance_mean']:+.4f} ± {row['importance_std']:.4f} {bar}"
        print(line)
        report_lines.append(line)
    
    # 3. Identify useless features (negative or near-zero importance)
    useless = perm_df[perm_df['importance_mean'] < 0.001]['feature'].tolist()
    harmful = perm_df[perm_df['importance_mean'] < 0]['feature'].tolist()
    
    if harmful:
        warning = f"\n⚠️  Harmful features (negative importance, consider removing): {harmful}"
        print(warning)
        report_lines.append(warning)
    
    if useless and not harmful:
        warning = f"\n⚠️  Low-value features (importance < 0.001): {[f for f in useless if f not in harmful]}"
        print(warning)
        report_lines.append(warning)
    
    # 4. Feature importance tiers
    section = "\n🔹 Feature Importance Tiers:"
    print(section)
    report_lines.append(section)
    
    tier1 = perm_df[perm_df['importance_mean'] >= 0.01]['feature'].tolist()
    tier2 = perm_df[(perm_df['importance_mean'] >= 0.005) & (perm_df['importance_mean'] < 0.01)]['feature'].tolist()
    tier3 = perm_df[(perm_df['importance_mean'] >= 0.001) & (perm_df['importance_mean'] < 0.005)]['feature'].tolist()
    tier4 = perm_df[perm_df['importance_mean'] < 0.001]['feature'].tolist()
    
    line = f"   🥇 High impact (≥0.01):     {len(tier1)} features"
    print(line)
    report_lines.append(line)
    line = f"   🥈 Medium impact (0.005-0.01): {len(tier2)} features"
    print(line)
    report_lines.append(line)
    line = f"   🥉 Low impact (0.001-0.005):   {len(tier3)} features"
    print(line)
    report_lines.append(line)
    line = f"   ❌ Negligible (<0.001):       {len(tier4)} features"
    print(line)
    report_lines.append(line)
    
    return perm_df, report_lines


def select_top_features(importance_df, threshold=0.001, min_features=15):
    """Select features above importance threshold.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance_mean' columns
        threshold: Minimum importance to keep feature
        min_features: Minimum number of features to keep regardless of threshold
    
    Returns:
        Tuple of (selected_features, removed_features)
    """
    sorted_df = importance_df.sort_values('importance_mean', ascending=False)
    
    # Always keep at least min_features
    above_threshold = sorted_df[sorted_df['importance_mean'] >= threshold]['feature'].tolist()
    
    if len(above_threshold) < min_features:
        selected = sorted_df.head(min_features)['feature'].tolist()
    else:
        selected = above_threshold
    
    removed = [f for f in sorted_df['feature'].tolist() if f not in selected]
    
    print(f"\n✅ Feature Selection: Kept {len(selected)}, Removed {len(removed)}")
    if removed:
        print(f"   Removed features: {removed}")
    
    return selected, removed


def remove_correlated_features(df, features, threshold=0.95, report_lines=None):
    """Remove highly correlated features to reduce redundancy.
    
    Args:
        df: DataFrame containing the features
        features: List of feature names to analyze
        threshold: Correlation threshold above which to remove features
        report_lines: Optional list to append report text to
    
    Returns:
        List of features with highly correlated ones removed
    """
    if report_lines is None:
        report_lines = []
    
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs of highly correlated features
    to_drop = set()
    correlated_pairs = []
    
    for column in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, column] > threshold:
                correlated_pairs.append((idx, column, upper.loc[idx, column]))
                # Drop the one that appears later in the list (arbitrary choice)
                to_drop.add(column)
    
    if correlated_pairs:
        section = f"\n🔄 Highly Correlated Features (r > {threshold}):"
        print(section)
        report_lines.append(section)
        
        for f1, f2, corr in correlated_pairs:
            line = f"   {f1} <-> {f2}: r={corr:.3f}"
            print(line)
            report_lines.append(line)
        
        line = f"   Removing: {list(to_drop)}"
        print(line)
        report_lines.append(line)
    else:
        line = f"\n✅ No highly correlated features found (threshold={threshold})"
        print(line)
        report_lines.append(line)
    
    return [f for f in features if f not in to_drop], report_lines


def save_feature_importance_report(report_lines, importance_df, output_path):
    """Save feature importance analysis to a text report file.
    
    Args:
        report_lines: List of report text lines
        importance_df: DataFrame with importance results
        output_path: Path to save the report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Feature Importance Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Write report lines
        for line in report_lines:
            f.write(line + "\n")
        
        # Write full importance table
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("FULL PERMUTATION IMPORTANCE TABLE\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"{'Feature':<30} {'Importance':>12} {'Std':>10}\n")
        f.write("-" * 55 + "\n")
        
        for _, row in importance_df.sort_values('importance_mean', ascending=False).iterrows():
            f.write(f"{row['feature']:<30} {row['importance_mean']:>+12.6f} {row['importance_std']:>10.6f}\n")
    
    print(f"\n📄 Feature importance report saved to: {output_path}")

def _check_api_key():
    """Verify POLYGON_API_KEY is set before proceeding."""
    try:
        _get_polygon_key()
    except EnvironmentError:
        print("❌ ERROR: POLYGON_API_KEY environment variable is required but not set.")
        print("   Please set it before running this script:")
        print("   export POLYGON_API_KEY=your_api_key_here")
        sys.exit(1)

# --- DATA INGESTION (POLYGON) ---
def fetch_polygon_history(ticker, start_str, end_str):
    """Fetch adjusted daily bars from Polygon.io (Fast & Reliable)."""
    # Using 'aggs' endpoint which allows fetching a range in one go
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": _get_polygon_key()
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "results" not in data or not data["results"]:
            return None
        
        # Parse
        df = pd.DataFrame(data["results"])
        # Map Polygon cols: v=Volume, o=Open, c=Close, h=High, l=Low, t=UnixMS
        df = df.rename(columns={"v": "Volume", "o": "Open", "c": "Close", "h": "High", "l": "Low", "t": "Date"})
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        return None

def get_universe_tickers(limit=2000):
    """Try to get full universe, fallback to S&P 500."""
    try:
        from core.data import get_universe
        return get_universe(limit=limit)
    except:
        # Fallback to standard list if core fails
        try:
            return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        except:
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "NFLX", "INTC"] # Minimal fallback

# --- FEATURE ENGINEERING (v3.1 — 39 features, stock-picking focused) ---
def calculate_features(df, spy_returns: pd.Series = None, spy_returns_60d: pd.Series = None,
                       market_regime_df: pd.DataFrame = None,
                       sector_etf_returns: pd.DataFrame = None, ticker: str = None):
    """Generate all 39 ML features from feature registry v3.1.

    v3.1 vs v3 changes:
      REMOVED (market-timing, distorted predictions):
        - Market_Trend, Market_Volatility, High_Volatility
      ADDED (stock-specific, all from OHLCV):
        - Vol_Contraction_Ratio, Squeeze_On_Flag
        - RS_vs_SPY_60d, RS_Momentum
        - UpStreak_Days, DownStreak_Days
        - Range_Pct_10d, OvernightGap_Avg

    Args:
        df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
        spy_returns: SPY 20-day returns Series (for RS_vs_SPY_20d)
        spy_returns_60d: SPY 60-day returns Series (for RS_vs_SPY_60d)
        market_regime_df: DataFrame with Market_Regime column only
        sector_etf_returns: DataFrame, columns=ETF symbols, values=20d returns
        ticker: Stock ticker for sector lookup
    """
    df = df.copy()

    # === BASIC TECHNICAL ===
    # RSI (14)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # True Range for ATR
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    df['ATR_Pct'] = tr.rolling(14).mean() / df['Close']

    # Returns
    df['Return_20d'] = df['Close'].pct_change(20)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_60d'] = df['Close'].pct_change(60)

    # === VOLATILITY PATTERNS ===
    atr_10 = tr.rolling(10).mean()
    atr_20 = tr.rolling(20).mean()
    atr_30 = tr.rolling(30).mean()
    atr_50 = tr.rolling(50).mean()

    df['VCP_Ratio'] = atr_10 / atr_30.replace(0, np.nan)

    range_5d = df['High'].rolling(5).max() - df['Low'].rolling(5).min()
    range_20d = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    df['Tightness_Ratio'] = range_5d / range_20d.replace(0, np.nan)

    high_52w = df['High'].rolling(252, min_periods=100).max()
    df['Dist_From_52w_High'] = (df['Close'] / high_52w.replace(0, np.nan)) - 1

    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()
    ma200 = df['Close'].rolling(200, min_periods=100).mean()
    df['MA_Alignment'] = ((df['Close'] > ma20) & (ma20 > ma50) & (ma50 > ma200)).astype(float)

    # v3.1 VOLATILITY ADDITIONS
    # Vol_Contraction_Ratio: ATR(20)/ATR(50) — tighter consolidation = breakout setup
    df['Vol_Contraction_Ratio'] = atr_20 / atr_50.replace(0, np.nan)

    # Squeeze_On_Flag: Bollinger Bands inside Keltner Channels (pre-breakout compression)
    bb_std = df['Close'].rolling(20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    kc_upper = ma20 + 1.5 * atr_20
    kc_lower = ma20 - 1.5 * atr_20
    df['Squeeze_On_Flag'] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)

    # === VOLUME ===
    vol_5d = df['Volume'].rolling(5).mean()
    vol_20d = df['Volume'].rolling(20).mean()
    df['Volume_Surge'] = vol_5d / vol_20d.replace(0, np.nan)

    daily_return = df['Close'].pct_change()
    up_day = daily_return > 0
    down_day = daily_return < 0
    up_volume = (df['Volume'] * up_day).rolling(20).sum()
    down_volume = (df['Volume'] * down_day).rolling(20).sum()
    up_days_count = up_day.rolling(20).sum()
    down_days_count = down_day.rolling(20).sum()
    avg_up_vol = up_volume / up_days_count.replace(0, np.nan)
    avg_down_vol = down_volume / down_days_count.replace(0, np.nan)
    df['Up_Down_Volume_Ratio'] = avg_up_vol / avg_down_vol.replace(0, np.nan)

    # === MOMENTUM ===
    positive_return = (daily_return > 0).astype(int)
    df['Momentum_Consistency'] = positive_return.rolling(20).mean()

    # === RELATIVE STRENGTH (v3.1: 3 features including 60d and momentum) ===
    if spy_returns is not None:
        spy_aligned = spy_returns.reindex(df.index, method='ffill')
        df['RS_vs_SPY_20d'] = df['Return_20d'] - spy_aligned.fillna(0)
    else:
        df['RS_vs_SPY_20d'] = 0.0

    if spy_returns_60d is not None:
        spy_60_aligned = spy_returns_60d.reindex(df.index, method='ffill')
        df['RS_vs_SPY_60d'] = df['Return_60d'] - spy_60_aligned.fillna(0)
    else:
        df['RS_vs_SPY_60d'] = 0.0

    # RS_Momentum: short-term RS minus long-term RS (acceleration)
    df['RS_Momentum'] = df['RS_vs_SPY_20d'] - df['RS_vs_SPY_60d']

    # === MARKET REGIME (v3.1: only Market_Regime, no market-timing) ===
    if market_regime_df is not None and 'Market_Regime' in market_regime_df.columns:
        df['Market_Regime'] = market_regime_df['Market_Regime'].reindex(df.index, method='ffill').fillna(0)
    else:
        df['Market_Regime'] = 0

    # === SECTOR RELATIVE ===
    if sector_etf_returns is not None and ticker is not None:
        sector = get_stock_sector(ticker)
        sector_etf = get_sector_etf(sector)
        if sector_etf is not None and sector_etf in sector_etf_returns.columns:
            sector_ret = sector_etf_returns[sector_etf].reindex(df.index, method='ffill').fillna(0)
            df['Sector_RS'] = df['Return_20d'] - sector_ret
            df['Sector_Momentum'] = sector_ret
            sector_ret_5d = sector_etf_returns[sector_etf].pct_change(5).reindex(df.index, method='ffill').fillna(0)
            df['Sector_Rank'] = (df['Return_5d'] > sector_ret_5d).astype(float)
        else:
            df['Sector_RS'] = 0.0
            df['Sector_Momentum'] = 0.0
            df['Sector_Rank'] = 0.5
    else:
        df['Sector_RS'] = 0.0
        df['Sector_Momentum'] = 0.0
        df['Sector_Rank'] = 0.5

    # === VOLUME ADVANCED ===
    df['Volume_Ratio_20d'] = df['Volume'] / vol_20d.replace(0, np.nan)

    def volume_slope(x):
        if len(x) < 20:
            return 0.0
        try:
            slope = np.polyfit(range(len(x)), x, 1)[0]
            return slope / (x.mean() + 1e-8)
        except Exception:
            return 0.0
    df['Volume_Trend'] = df['Volume'].rolling(20).apply(volume_slope, raw=True)

    total_vol_20d = df['Volume'].rolling(20).sum()
    df['Up_Volume_Ratio'] = up_volume / total_vol_20d.replace(0, np.nan)

    price_up = df['Close'] > df['Close'].shift(1)
    vol_up = df['Volume'] > df['Volume'].shift(1)
    df['Volume_Price_Confirm'] = (price_up & vol_up).astype(float).rolling(5).mean()

    df['Relative_Volume_Rank'] = df['Volume'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 60 else 0.5, raw=False
    )

    # === PRICE ACTION ===
    low_52w = df['Low'].rolling(252, min_periods=100).min()
    df['Distance_From_52w_Low'] = (df['Close'] - low_52w) / low_52w.replace(0, np.nan)

    h20 = df['High'].rolling(20).max()
    l20 = df['Low'].rolling(20).min()
    avg20 = df['Close'].rolling(20).mean()
    df['Consolidation_Tightness'] = (h20 - l20) / avg20.replace(0, np.nan)

    def days_since_high(s):
        if len(s) < 252:
            return 126
        try:
            return len(s) - np.argmax(s.values) - 1
        except Exception:
            return 126
    df['Days_Since_52w_High'] = df['High'].rolling(252, min_periods=100).apply(days_since_high, raw=False)
    df['Days_Since_52w_High'] = df['Days_Since_52w_High'] / 252.0

    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200, min_periods=100).mean()
    df['Price_vs_SMA50'] = (df['Close'] - sma50) / sma50.replace(0, np.nan)
    df['Price_vs_SMA200'] = (df['Close'] - sma200) / sma200.replace(0, np.nan)
    df['SMA50_vs_SMA200'] = (sma50 - sma200) / sma200.replace(0, np.nan)
    df['MA_Slope_20d'] = ma20.diff(20) / ma20.shift(20).replace(0, np.nan)
    df['Distance_To_Resistance'] = (h20 - df['Close']) / df['Close'].replace(0, np.nan)

    near_low = (df['Low'] <= l20 * 1.02).astype(float)
    df['Support_Strength'] = near_low.rolling(20).mean()

    # === v3.1 STREAK & PATTERN FEATURES ===
    # UpStreak_Days: consecutive up-close days (capped at 10)
    up_close = (df['Close'] > df['Close'].shift(1)).astype(int)
    streak_break_up = (up_close == 0).cumsum()
    df['UpStreak_Days'] = up_close.groupby(streak_break_up).cumsum().clip(upper=10)

    # DownStreak_Days: consecutive down-close days (capped at 10)
    down_close = (df['Close'] < df['Close'].shift(1)).astype(int)
    streak_break_down = (down_close == 0).cumsum()
    df['DownStreak_Days'] = down_close.groupby(streak_break_down).cumsum().clip(upper=10)

    # Range_Pct_10d: average intraday range as % of close (energy measure)
    intraday_range_pct = (df['High'] - df['Low']) / df['Close'].replace(0, np.nan)
    df['Range_Pct_10d'] = intraday_range_pct.rolling(10).mean()

    # OvernightGap_Avg: average overnight gap % last 5 days (institutional interest)
    overnight_gap = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1).replace(0, np.nan)
    df['OvernightGap_Avg'] = overnight_gap.rolling(5).mean()

    # === TARGET ===
    df['Forward_Return_20d'] = df['Close'].shift(-20) / df['Close'] - 1.0

    # Clean up: replace inf/nan in all feature columns
    for col in FEATURE_NAMES_V3:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Clip to match inference clipping (ml_20d_inference.py)
    if 'ATR_Pct' in df.columns:
        df['ATR_Pct'] = df['ATR_Pct'].clip(0.0, 0.2)
    if 'RSI' in df.columns:
        df['RSI'] = df['RSI'].clip(5.0, 95.0)

    return df.dropna(subset=['Forward_Return_20d'] + [f for f in FEATURE_NAMES_V3 if f in df.columns])


def calculate_market_regime(spy_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market regime from SPY data.

    v3.1: Only returns Market_Regime (1=Bull, 0=Sideways, -1=Bear).
    Removed: Market_Volatility, Market_Trend, High_Volatility — these caused
    the model to become a market timer rather than a stock picker.
    """
    if spy_df is None or len(spy_df) < 200:
        return None

    regime_df = pd.DataFrame(index=spy_df.index)
    spy_close = spy_df['Close']
    ma50 = spy_close.rolling(50).mean()
    ma200 = spy_close.rolling(200, min_periods=100).mean()
    ret_20d = spy_close.pct_change(20)

    regime_df['Market_Regime'] = 0
    bull = (spy_close > ma50) & (ma50 > ma200) & (ret_20d > 0.02)
    bear = (spy_close < ma50) & (ma50 < ma200) & (ret_20d < -0.02)
    regime_df.loc[bull, 'Market_Regime'] = 1
    regime_df.loc[bear, 'Market_Regime'] = -1

    return regime_df


# =============================================================================
# PURGED WALK-FORWARD CROSS-VALIDATION
# =============================================================================

class PurgedWalkForwardCV:
    """Walk-forward CV with purge embargo to prevent label leakage.

    Critical for 20-day forward return targets: a row at date T uses price
    data up to T+20. Without the 20-day embargo, rows near the train/val
    boundary contaminate validation via overlapping future prices.

    Expanding training window (walk-forward) simulates real retraining cycles.
    """

    def __init__(self, n_splits: int = 5, embargo_td: int = 20, min_train_pct: float = 0.3):
        self.n_splits = n_splits
        self.embargo_td = embargo_td
        self.min_train_pct = min_train_pct

    def split(self, X, y=None, groups=None, dates=None):
        n = len(X)
        if dates is not None and hasattr(dates, 'unique'):
            unique_dates = np.sort(dates.unique())
            n_dates = len(unique_dates)
            min_train_dates = int(n_dates * self.min_train_pct)
            remaining_dates = n_dates - min_train_dates
            val_size = remaining_dates // self.n_splits
            if val_size < 5:
                raise ValueError(f"Not enough dates for {self.n_splits} folds")
            for fold in range(self.n_splits):
                train_end_date = unique_dates[min_train_dates + fold * val_size - 1]
                val_start_date = unique_dates[min(min_train_dates + fold * val_size + self.embargo_td,
                                                   n_dates - 1)]
                val_end_idx = min(min_train_dates + (fold + 1) * val_size, n_dates)
                val_end_date = unique_dates[val_end_idx - 1]
                train_mask = dates <= train_end_date
                val_mask = (dates >= val_start_date) & (dates <= val_end_date)
                train_idx = np.where(train_mask)[0]
                val_idx = np.where(val_mask)[0]
                if len(train_idx) > 0 and len(val_idx) > 0:
                    yield train_idx, val_idx
        else:
            # Fallback: index-based embargo
            min_train = int(n * self.min_train_pct)
            val_size = (n - min_train) // self.n_splits
            for fold in range(self.n_splits):
                train_end = min_train + fold * val_size
                val_start = train_end + self.embargo_td
                val_end = min(train_end + val_size + self.embargo_td, n)
                if val_start < val_end and train_end > 0:
                    yield np.arange(train_end), np.arange(val_start, val_end)


# --- MAIN PIPELINE ---
def fetch_sector_etf_data(start_str: str, end_str: str) -> pd.DataFrame:
    """Fetch all sector ETF data and calculate 20d returns.
    
    Returns DataFrame with columns = ETF symbols, rows = dates,
    values = 20-day returns.
    """
    sector_etfs = get_all_sector_etfs()  # ['XLK', 'XLF', 'XLE', ...]
    sector_data = {}
    
    print(f"📥 Fetching {len(sector_etfs)} sector ETFs...")
    for etf in sector_etfs:
        df = fetch_polygon_history(etf, start_str, end_str)
        if df is not None and len(df) > 30:
            # Calculate 20-day returns for sector ETF
            sector_data[etf] = df['Close'].pct_change(20)
    
    if not sector_data:
        print("   ⚠️  No sector ETF data available")
        return None
    
    sector_df = pd.DataFrame(sector_data)
    print(f"   ✅ Sector ETF data loaded: {len(sector_data)} sectors, {len(sector_df)} days")
    return sector_df


def train_and_save_bundle():
    # Verify API key is available before proceeding
    _check_api_key()
    
    print("🚀 Starting Polygon-Powered Training Pipeline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Universe
    tickers = get_universe_tickers(2000)
    print(f"📋 Universe size: {len(tickers)} tickers")
    
    # 2. Download Data (Parallel)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years history for robust training
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Fetch SPY data once for relative strength calculation (20d + 60d)
    print("📥 Fetching SPY benchmark data...")
    spy_df = fetch_polygon_history("SPY", start_str, end_str)
    spy_returns = None
    spy_returns_60d = None
    market_regime_df = None

    if spy_df is not None and len(spy_df) > 60:
        spy_returns = spy_df['Close'].pct_change(20)
        spy_returns.name = 'SPY_Return_20d'
        spy_returns_60d = spy_df['Close'].pct_change(60)
        spy_returns_60d.name = 'SPY_Return_60d'
        print(f"   ✅ SPY data loaded ({len(spy_df)} days, {spy_returns.notna().sum()} valid 20d returns)")

        market_regime_df = calculate_market_regime(spy_df)
        if market_regime_df is not None:
            bull_days = (market_regime_df['Market_Regime'] == 1).sum()
            bear_days = (market_regime_df['Market_Regime'] == -1).sum()
            sideways_days = (market_regime_df['Market_Regime'] == 0).sum()
            print(f"   📊 Market regime: Bull={bull_days}, Sideways={sideways_days}, Bear={bear_days}")
    else:
        print("   ⚠️  SPY data unavailable — RS features default to 0")

    # Fetch sector ETF data for sector-relative features
    sector_etf_returns = fetch_sector_etf_data(start_str, end_str)

    all_data = []
    print("📥 Downloading data from Polygon (Threads=15)...")

    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {executor.submit(fetch_polygon_history, t, start_str, end_str): t for t in tickers}
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            if completed % 100 == 0: print(f"   ... processed {completed}/{len(tickers)}")

            t = future_to_ticker[future]
            df = future.result()
            if df is not None and len(df) > 60:
                df = calculate_features(df, spy_returns, spy_returns_60d, market_regime_df, sector_etf_returns, t)
                df['Ticker'] = t
                all_data.append(df)

    if not all_data:
        raise RuntimeError("No data downloaded! Check API Key or Network.")
    
    full_df = pd.concat(all_data)
    print(f"📊 Total Training Rows: {len(full_df)}")

    # 3. Labeling (Unified logic)
    full_df['Label'] = make_label_20d(full_df['Forward_Return_20d'])
    full_df = full_df.dropna(subset=["Label"]).copy()
    full_df["Label"] = full_df["Label"].astype(int)
    # Print class distribution
    pos_count = full_df['Label'].sum()
    neg_count = len(full_df) - pos_count
    print(f"📊 Class distribution: {pos_count} winners ({pos_count/len(full_df)*100:.1f}%), "
          f"{neg_count} losers ({neg_count/len(full_df)*100:.1f}%)")

    # Print performance by market regime
    print("\n📊 Performance by Market Regime:")
    for regime_val, regime_name in [(1, 'Bull'), (0, 'Sideways'), (-1, 'Bear')]:
        mask = full_df['Market_Regime'] == regime_val
        if mask.sum() > 100:
            regime_winners = full_df.loc[mask, 'Label'].mean()
            print(f"   {regime_name:10s}: {mask.sum():6d} samples, {regime_winners:.1%} winners")

    # 4. Purged Walk-Forward Cross-Validation (prevents label leakage)
    # Use feature registry v3.1 as SINGLE SOURCE OF TRUTH
    features = get_feature_names("v3.1")  # 39 features from feature_registry.py

    # Verify all features are present in training data
    available_features = [f for f in features if f in full_df.columns]
    missing_features = [f for f in features if f not in full_df.columns]
    if missing_features:
        print(f"⚠️  Missing features in data: {missing_features}")
        features = available_features
        print(f"    Proceeding with {len(features)} available features.")

    # Sort by date for proper time-series split
    full_df = full_df.sort_index()
    X = full_df[features]
    y = full_df['Label']
    dates = full_df.index

    print("📊 Running Purged Walk-Forward CV (5 folds, 20-day embargo)...")
    pwf_cv = PurgedWalkForwardCV(n_splits=5, embargo_td=20, min_train_pct=0.3)
    oos_aucs = []
    oos_logloss = []
    oos_p20 = []  # Precision@20
    oos_p50 = []  # Precision@50

    for fold, (train_idx, val_idx) in enumerate(pwf_cv.split(X, y, dates=dates), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Skip folds with insufficient positive samples
        if y_train.sum() < 10 or y_val.sum() < 5:
            print(f"   Fold {fold}: ⚠️  Skipped (insufficient positive samples)")
            continue
        
        fold_model = HistGradientBoostingClassifier(
            max_iter=300,              # Maximum iterations (will stop early)
            learning_rate=0.05,
            max_depth=4,               # Slightly deeper for complex patterns
            min_samples_leaf=20,       # Prevent overfitting on small groups
            l2_regularization=0.1,     # L2 regularization
            class_weight='balanced',   # Built-in class balancing
            early_stopping=True,       # Enable early stopping
            validation_fraction=0.15,  # Use 15% for early stopping validation
            n_iter_no_change=10,       # Stop if no improvement for 10 iterations
            random_state=42,
        )
        fold_model.fit(X_train, y_train)
        
        # Out-of-sample predictions
        y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        fold_auc = roc_auc_score(y_val, y_pred_proba)
        fold_ll = log_loss(y_val, y_pred_proba)
        fold_p20 = precision_at_k(y_val, y_pred_proba, k=20)
        fold_p50 = precision_at_k(y_val, y_pred_proba, k=50)
        
        oos_aucs.append(fold_auc)
        oos_logloss.append(fold_ll)
        oos_p20.append(fold_p20)
        oos_p50.append(fold_p50)
        
        # Calculate baseline (random) precision for comparison
        baseline = y_val.mean()  # ~20% if using threshold-based labeling
        lift_20 = fold_p20 / baseline if baseline > 0 else 0
        
        print(f"   Fold {fold}: AUC={fold_auc:.4f}, P@20={fold_p20:.1%} ({lift_20:.1f}x lift), P@50={fold_p50:.1%}")
    
    if not oos_aucs:
        raise RuntimeError("No valid CV folds! Check data quality.")
    
    mean_oos_auc = np.mean(oos_aucs)
    std_oos_auc = np.std(oos_aucs)
    mean_oos_ll = np.mean(oos_logloss)
    mean_p20 = np.mean(oos_p20) if oos_p20 else 0.0
    mean_p50 = np.mean(oos_p50) if oos_p50 else 0.0
    
    # Calculate baseline and lift
    baseline_precision = 0.20  # Since we label top 20% as winners
    lift_20 = mean_p20 / baseline_precision if baseline_precision > 0 else 0
    lift_50 = mean_p50 / baseline_precision if baseline_precision > 0 else 0
    
    print(f"\n📈 Cross-Validation Results:")
    print(f"   Out-of-Sample AUC: {mean_oos_auc:.4f} ± {std_oos_auc:.4f}")
    print(f"   Out-of-Sample LogLoss: {mean_oos_ll:.4f}")
    print(f"   Precision@20: {mean_p20:.1%} (lift: {lift_20:.2f}x vs random)")
    print(f"   Precision@50: {mean_p50:.1%} (lift: {lift_50:.2f}x vs random)")
    print(f"   ")
    print(f"   🎯 Target: P@20 > 40% (2x lift) = meaningful signal")
    print(f"   🏆 Excellent: P@20 > 60% (3x lift) = strong alpha")
    
    # 4.5. Feature Importance Analysis (using last fold's model)
    report_lines = []
    report_lines.append(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Features: {len(features)}")
    report_lines.append(f"Total Training Samples: {len(full_df)}")
    report_lines.append(f"CV Folds: {len(oos_aucs)}")
    report_lines.append(f"Mean OOS AUC: {mean_oos_auc:.4f} ± {std_oos_auc:.4f}")
    report_lines.append(f"Mean P@20: {mean_p20:.1%}")
    
    # Train a quick model on last 80% for importance analysis
    print("\n🔍 Analyzing feature importance...")
    split_for_analysis = int(len(X) * 0.8)
    X_train_analysis = X.iloc[:split_for_analysis]
    X_test_analysis = X.iloc[split_for_analysis:]
    y_train_analysis = y.iloc[:split_for_analysis]
    y_test_analysis = y.iloc[split_for_analysis:]
    
    analysis_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        class_weight='balanced',
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    analysis_model.fit(X_train_analysis, y_train_analysis)
    
    # Run feature importance analysis
    importance_df, report_lines = analyze_feature_importance(
        analysis_model, features, X_test_analysis, y_test_analysis, report_lines
    )
    
    # Check for correlated features
    _, report_lines = remove_correlated_features(full_df, features, threshold=0.95, report_lines=report_lines)
    
    # Recommend feature selection
    selected_features, removed_features = select_top_features(importance_df, threshold=0.001, min_features=20)
    report_lines.append(f"\nRecommended Features ({len(selected_features)}): {selected_features}")
    report_lines.append(f"Candidates for Removal ({len(removed_features)}): {removed_features}")
    
    # Save report
    report_path = MODELS_DIR / "feature_importance_report.txt"
    save_feature_importance_report(report_lines, importance_df, report_path)
    
    # 5. Train final model with probability calibration
    # Split: 80% for training, 20% for calibration (preserving time order)
    print("\n🧠 Training final model with probability calibration...")
    
    # Use time-based split (not random) to preserve temporal ordering
    split_idx = int(len(X) * 0.8)
    X_train_final, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_final, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Training samples: {len(X_train_final)}, Calibration samples: {len(X_calib)}")
    
    # Train base model 1: HistGradientBoostingClassifier (non-linear patterns)
    print("   Training Model 1/3: HistGradientBoostingClassifier...")
    model1 = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=0.1,
        class_weight='balanced',
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    model1.fit(X_train_final, y_train_final)
    print(f"      HistGB stopped at iteration {model1.n_iter_}")
    
    # Train base model 2: RandomForest (feature interactions, robust to outliers)
    print("   Training Model 2/3: RandomForestClassifier...")
    model2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model2.fit(X_train_final, y_train_final)
    print(f"      RandomForest trained with {model2.n_estimators} trees")
    
    # Train base model 3: LogisticRegression (linear baseline, needs scaling)
    print("   Training Model 3/3: LogisticRegression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    model3 = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=0.1,  # Regularization
        random_state=42,
    )
    model3.fit(X_train_scaled, y_train_final)
    model3._needs_scaling = True  # Mark for ensemble
    print("      LogisticRegression converged")
    
    # Create ensemble with weights favoring tree models
    ensemble_weights = [0.45, 0.35, 0.20]  # HistGB, RF, LR
    base_model = EnsembleClassifier(
        models=[model1, model2, model3],
        weights=ensemble_weights,
        scaler=scaler,
    )
    print(f"   Ensemble created with weights: HistGB={ensemble_weights[0]}, RF={ensemble_weights[1]}, LR={ensemble_weights[2]}")
    
    # Calibrate using isotonic regression on held-out 20%
    # Note: CalibratedClassifierCV with cv='prefit' uses provided data for calibration only
    calibrated_model = CalibratedClassifierCV(
        base_model, method='isotonic', cv='prefit'
    )
    calibrated_model.fit(X_calib, y_calib)
    
    # Use calibrated model as final model
    model = calibrated_model
    
    # Evaluate calibration quality with reliability diagram data
    print("\n📊 Calibration Quality Check:")
    y_pred_calib = calibrated_model.predict_proba(X_calib)[:, 1]
    y_pred_uncalib = base_model.predict_proba(X_calib)[:, 1]
    
    # Compute reliability metrics (binned calibration)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    print("   Predicted Prob  |  Actual Freq  |  Count  |  Calibration")
    print("   " + "-" * 55)
    
    calibration_errors = []
    for i in range(n_bins):
        mask = (y_pred_calib >= bin_edges[i]) & (y_pred_calib < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_pred = y_pred_calib[mask].mean()
            bin_actual = y_calib.iloc[mask.values].mean()
            count = mask.sum()
            error = abs(bin_pred - bin_actual)
            calibration_errors.append(error * count)  # Weighted by count
            status = "✓" if error < 0.1 else "⚠️" if error < 0.2 else "✗"
            print(f"   {bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}      |    {bin_actual:.3f}     |  {count:5d}  |  {status} (err={error:.3f})")
    
    mean_calibration_error = sum(calibration_errors) / len(X_calib) if calibration_errors else 0
    print(f"\n   Mean Calibration Error (MCE): {mean_calibration_error:.4f}")
    
    # Compare AUCs (calibration shouldn't hurt AUC much)
    auc_uncalib = roc_auc_score(y_calib, y_pred_uncalib)
    auc_calib = roc_auc_score(y_calib, y_pred_calib)
    print(f"   Calibration set AUC (uncalibrated): {auc_uncalib:.4f}")
    print(f"   Calibration set AUC (calibrated):   {auc_calib:.4f}")
    
    # Precision@K on calibration set (key trading metric)
    p20_calib = precision_at_k(y_calib, y_pred_calib, k=20)
    p50_calib = precision_at_k(y_calib, y_pred_calib, k=50)
    print(f"   Calibration set P@20: {p20_calib:.1%}")
    print(f"   Calibration set P@50: {p50_calib:.1%}")
    
    # In-sample AUC on full data (for reference only)
    insample_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"   Full data AUC (reference): {insample_auc:.4f}")
    
    # Feature Importance (averaged across ensemble)
    print("\n📊 Feature Importance (averaged across ensemble):")
    try:
        # HistGB importance
        imp1 = model1.feature_importances_
        # RF importance  
        imp2 = model2.feature_importances_
        # LR coefficients (absolute value as importance proxy)
        imp3 = np.abs(model3.coef_[0])
        imp3 = imp3 / imp3.sum()  # Normalize to sum to 1
        
        # Weighted average matching ensemble weights
        avg_importance = imp1 * ensemble_weights[0] + imp2 * ensemble_weights[1] + imp3 * ensemble_weights[2]
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': avg_importance,
            'histgb': imp1,
            'rf': imp2,
            'lr': imp3,
        }).sort_values('importance', ascending=False)
        
        for _, row in importance_df.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"   {row['feature']:22s} {row['importance']:.4f} {bar}")
    except Exception as e:
        print(f"   (Feature importance calculation failed: {e})")
        importance_df = None
    
    # 6. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    bundle = {
        "model": model,  # Calibrated model
        "feature_names": features,
        "metrics": {
            "oos_auc_mean": mean_oos_auc,
            "oos_auc_std": std_oos_auc,
            "oos_logloss_mean": mean_oos_ll,
            "precision_at_20_mean": mean_p20,
            "precision_at_50_mean": mean_p50,
            "lift_at_20": lift_20,
            "lift_at_50": lift_50,
            "baseline_precision": baseline_precision,
            "insample_auc": insample_auc,
            "cv_folds": len(oos_aucs),
            "cv_aucs": oos_aucs,
            "cv_p20": oos_p20,
            "cv_p50": oos_p50,
            "calibration_method": "isotonic",
            "calibration_samples": len(X_calib),
            "mean_calibration_error": mean_calibration_error,
            "calibration_auc": auc_calib,
            "calibration_p20": p20_calib,
            "calibration_p50": p50_calib,
        },
        "trained_at": timestamp
    }
    
    path = MODELS_DIR / "model_20d_v3.pkl"
    joblib.dump(bundle, path)
    # Write metadata.json next to model and to canonical latest path
    meta = {
        "sklearn_version": __import__("sklearn").__version__,
        "feature_list": features,
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "label_spec": {
            "horizon_days": 20,
            "threshold_type": "fixed",
            "up_threshold": UP_THRESHOLD,
            "down_threshold": DOWN_THRESHOLD,
            "label_name": f"Forward_Return_20d>={UP_THRESHOLD}",
            "class_weighting": "balanced",
        },
        "model_type": "CalibratedEnsemble(HistGB+RF+LR)",
        "model_name": "ml_20d_v3.1_ensemble_calibrated",
        "feature_version": "v3.1",
        "cv_method": "PurgedWalkForwardCV(embargo=20d, expanding_window)",
        "ensemble": {
            "models": ["HistGradientBoostingClassifier", "RandomForestClassifier", "LogisticRegression"],
            "weights": ensemble_weights,
        },
        "early_stopping": {
            "enabled": True,
            "histgb_stopped_at_iter": model1.n_iter_,
            "max_iter": 300,
            "n_iter_no_change": 15,
        },
        "regularization": {
            "histgb_l2": 0.1,
            "histgb_max_depth": 4,
            "rf_max_depth": 6,
            "lr_C": 0.1,
            "min_samples_leaf": 20,
        },
        "calibration": {
            "method": "isotonic",
            "samples": len(X_calib),
            "mean_calibration_error": mean_calibration_error,
        },
        "evaluation": {
            "precision_at_20": mean_p20,
            "precision_at_50": mean_p50,
            "lift_at_20": lift_20,
            "lift_at_50": lift_50,
            "baseline_precision": baseline_precision,
            "target_p20": 0.40,
            "excellent_p20": 0.60,
        },
    }
    try:
        with open(path.with_suffix(".metadata.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(meta, f)
    except Exception:
        pass

    # Also publish to canonical latest path for inference determinism
    latest_dir = Path("ml/bundles/latest")
    latest_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, latest_dir / "model.joblib")
    try:
        with open(latest_dir / "metadata.json", "w", encoding="utf-8") as f:
            import json
            json.dump(meta, f)
    except Exception:
        pass

    print(f"✅ Model saved to: {path}")
    print(f"🗂️  Latest bundle written to: {latest_dir}")
    print(f"🏆 Out-of-Sample AUC: {bundle['metrics']['oos_auc_mean']:.4f} ± {bundle['metrics']['oos_auc_std']:.4f}")
    
    return path, bundle

if __name__ == "__main__":
    train_and_save_bundle()