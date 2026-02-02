import requests
import time
import os
import pandas as pd
from datetime import datetime, timedelta

# --- API Keys - Load from environment variables (NEVER hardcode!) ---
# ⚠️ SECURITY: All API keys must be set in environment or .env file
KEYS = {
    "FMP": os.environ.get("FMP_API_KEY", ""),
    "POLYGON": os.environ.get("POLYGON_API_KEY", ""),
    "FINNHUB": os.environ.get("FINNHUB_API_KEY", ""),
    "TIINGO": os.environ.get("TIINGO_API_KEY", ""),
    "ALPHA_VANTAGE": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    "EODHD": os.environ.get("EODHD_API_KEY", ""),
    "MARKETSTACK": os.environ.get("MARKETSTACK_API_KEY", ""),
}

# Validate that at least one key is set
if not any(KEYS.values()):
    print("⚠️ WARNING: No API keys found in environment!")
    print("   Set keys in .env file or environment variables.")
    print("   Example: export POLYGON_API_KEY=your_key_here")

# הגדרות זמן
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d') # קיצרנו ל-100 יום כדי להקל על Alpha

def benchmark_provider(name, url):
    print(f"⏳ Testing {name}...", end="\r")
    start = time.time()
    try:
        # User-Agent חשוב לפעמים כדי לא להיחסם
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=15)
        elapsed = (time.time() - start) * 1000 # ms
        
        if r.status_code != 200:
            err_preview = r.text[:100].replace('\n', ' ')
            return {"Provider": name, "Status": f"ERR {r.status_code}", "Time (ms)": round(elapsed), "Rows": 0, "Note": err_preview}
        
        data = r.json()
        count = 0
        
        # לוגיקה ספציפית לכל ספק
        if name == "FMP": 
            # FMP v3 historical structure
            hist = data.get('historical', [])
            count = len(hist)
        elif name == "POLYGON": 
            count = data.get('resultsCount', 0)
        elif name == "FINNHUB": 
            # Finnhub returns 'c' (close prices) list
            count = len(data.get('c', []))
        elif name == "TIINGO": 
            count = len(data) if isinstance(data, list) else 0
        elif name == "ALPHA_VANTAGE": 
            # Alpha Vantage standard series
            ts = data.get('Time Series (Daily)', {})
            count = len(ts)
        elif name == "EODHD": 
            count = len(data) if isinstance(data, list) else 0
        elif name == "MARKETSTACK": 
            count = len(data.get('data', []))

        return {
            "Provider": name, 
            "Status": "✅ OK", 
            "Time (ms)": round(elapsed), 
            "Rows": count, 
            "Note": "Good Data" if count > 0 else "Empty Response"
        }
    except Exception as e:
        return {"Provider": name, "Status": "FAIL", "Time (ms)": 0, "Rows": 0, "Note": str(e)[:50]}

# --- כתובות מתוקנות (Endpoints מעודכנים) ---
tests = [
    # FMP: כתובת רגילה ללא פרמטרים מסובכים, אמור לעבוד לכל מנוי
    ("FMP", f"https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?apikey={KEYS['FMP']}"),
    
    # Polygon: נשאר אותו דבר, הוא עבד מצוין
    ("POLYGON", f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={KEYS['POLYGON']}"),
    
    # Alpha: Compact mode (100 days) is usually free/safer
    ("ALPHA_VANTAGE", f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=compact&apikey={KEYS['ALPHA_VANTAGE']}"),
    
    # Finnhub: נשאר, אך המפתח נוקה
    ("FINNHUB", f"https://finnhub.io/api/v1/stock/candle?symbol=AAPL&resolution=D&from={int(time.time()-8640000)}&to={int(time.time())}&token={KEYS['FINNHUB']}"),
    
    # Tiingo
    ("TIINGO", f"https://api.tiingo.com/tiingo/daily/AAPL/prices?startDate={start_date}&token={KEYS['TIINGO']}"),
]

results = []
print("\n🏎️  STARTING BENCHMARK V2 (Fixed Endpoints)...\n")

for name, url in tests:
    results.append(benchmark_provider(name, url))

# הצגת טבלה
df = pd.DataFrame(results).sort_values(by=["Rows", "Time (ms)"], ascending=[False, True])
print("\n" + "="*80)
print("🏁  FINAL RESULTS (Sorted by Data Quality & Speed)")
print("="*80)
print(df.to_string(index=False))
print("="*80 + "\n")