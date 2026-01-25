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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

# --- CONFIG ---
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "2kjHdOxfjiw3d9Q9slH871KSQQPgf4nh")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")

# --- DATA INGESTION (POLYGON) ---
def fetch_polygon_history(ticker, start_str, end_str):
    """Fetch adjusted daily bars from Polygon.io (Fast & Reliable)."""
    # Using 'aggs' endpoint which allows fetching a range in one go
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_KEY
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

# --- FEATURE ENGINEERING ---
def calculate_features(df):
    """Generate 20d ML features."""
    df = df.copy()
    # Returns
    df['Return_20d'] = df['Close'].pct_change(20)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    # Volatility
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Target (Next 20d Return)
    df['Forward_Return_20d'] = df['Close'].shift(-20) / df['Close'] - 1.0
    
    return df.dropna()

# --- MAIN PIPELINE ---
def train_and_save_bundle():
    print("🚀 Starting Polygon-Powered Training Pipeline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Universe
    tickers = get_universe_tickers(2000)
    print(f"📋 Universe size: {len(tickers)} tickers")
    
    # 2. Download Data (Parallel)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2) # 2 Years history
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
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
            if df is not None and len(df) > 50:
                df = calculate_features(df)
                df['Ticker'] = t
                all_data.append(df)

    if not all_data:
        raise RuntimeError("No data downloaded! Check API Key or Network.")
    
    full_df = pd.concat(all_data)
    print(f"📊 Total Training Rows: {len(full_df)}")
    
    # 3. Labeling (Dynamic Threshold)
    # Try 20% first, if not enough samples, drop to 10%
    threshold = 0.20
    full_df['Label'] = (full_df['Forward_Return_20d'] > threshold).astype(int)
    
    if full_df['Label'].sum() < 50:
        print("⚠️  Not enough winners at 20%. Lowering threshold to 10%...")
        threshold = 0.10
        full_df['Label'] = (full_df['Forward_Return_20d'] > threshold).astype(int)
        
    print(f"🎯 Positive Samples (Winners > {threshold*100}%): {full_df['Label'].sum()}")

    # 4. Train Model
    features = ['RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d']
    X = full_df[features]
    y = full_df['Label']
    
    print("🧠 Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # 5. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    bundle = {
        "model": model,
        "feature_names": features,
        "metrics": {"auc": roc_auc_score(y, model.predict_proba(X)[:, 1])},
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
            "threshold_return": threshold,
            "label_name": "Forward_Return_20d>threshold",
        },
        "model_type": "GradientBoostingClassifier",
        "model_name": "ml_20d_v3_gb",
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
    print(f"🏆 Final AUC: {bundle['metrics']['auc']:.4f}")
    
    return path, bundle

if __name__ == "__main__":
    train_and_save_bundle()