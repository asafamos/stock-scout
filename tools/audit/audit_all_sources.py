import requests
import time
import sys
import os
import yfinance as yf

# Add root to path
sys.path.append(os.getcwd())

try:
    from core.config import get_api_keys
except ImportError:
    print("❌ Error: Run from project root.")
    sys.exit(1)

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

print(f"{YELLOW}🔍 STARTING FULL 10-POINT DATA SOURCE AUDIT...{RESET}")

try:
    keys = get_api_keys()
except Exception:
    print("❌ Critical: Keys missing.")
    sys.exit(1)

TICKER = "AAPL"
sources = [
    # --- FMP (Stable API - migrated from deprecated v3) ---
    {
        'name': '1. FMP (Profile)',
        'url': f'https://financialmodelingprep.com/stable/profile?symbol={TICKER}&apikey={keys.fmp}',
        'key': getattr(keys, 'fmp', None)
    },
    {
        'name': '2. FMP (Metrics)',
        'url': f'https://financialmodelingprep.com/stable/key-metrics-ttm?symbol={TICKER}&apikey={keys.fmp}',
        'key': getattr(keys, 'fmp', None)
    },
    # --- FINNHUB ---
    {
        'name': '3. Finnhub (Prof)',
        'url': f'https://finnhub.io/api/v1/stock/profile2?symbol={TICKER}&token={getattr(keys, "finnhub", "")}',
        'key': getattr(keys, 'finnhub', None)
    },
    {
        'name': '4. Finnhub (Quote)',
        'url': f'https://finnhub.io/api/v1/quote?symbol={TICKER}&token={getattr(keys, "finnhub", "")}',
        'key': getattr(keys, 'finnhub', None)
    },
    # --- TIINGO ---
    {
        'name': '5. Tiingo (Fund)',
        'url': f'https://api.tiingo.com/tiingo/fundamentals/{TICKER}/statements?token={getattr(keys, "tiingo", "")}',
        'key': getattr(keys, 'tiingo', None)
    },
    {
        'name': '6. Tiingo (Price)',
        'url': f'https://api.tiingo.com/tiingo/daily/{TICKER}/prices?token={getattr(keys, "tiingo", "")}',
        'key': getattr(keys, 'tiingo', None)
    },
    # --- ALPHA VANTAGE ---
    {
        'name': '7. Alpha Vantage',
        'url': f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={TICKER}&apikey={getattr(keys, "alpha_vantage", "")}',
        'key': getattr(keys, 'alpha_vantage', None)
    },
    # --- POLYGON (Safe Check) ---
    {
        'name': '8. Polygon.io',
        'url': f'https://api.polygon.io/v3/reference/tickers/{TICKER}?apiKey={getattr(keys, "polygon", "")}',
        'key': getattr(keys, 'polygon', None)
    },
    # --- EODHD (Safe Check) ---
    {
        'name': '9. EODHD',
        'url': f'https://eodhd.com/api/fundamentals/{TICKER}.US?api_token={getattr(keys, "eodhd", "")}&fmt=json',
        'key': getattr(keys, 'eodhd', None)
    }
]

print("-" * 90)
print(f"{'SOURCE':<20} | {'STATUS':<12} | {'TIME':<8} | {'RESPONSE CHECK'}")
print("-" * 90)

# Check APIs
for s in sources:
    name = s['name']
    if not s['key']:
        print(f"{name:<20} | {YELLOW}SKIPPED{RESET}      | -        | No API Key")
        continue
    
    start = time.time()
    try:
        r = requests.get(s['url'], timeout=5)
        lat = f"{time.time() - start:.2f}s"
        
        if r.status_code == 200:
            try:
                data = r.json()
                # Error detection logic
                if isinstance(data, dict) and 'Error Message' in data:
                    print(f"{name:<20} | {RED}FAIL{RESET}        | {lat:<8} | ❌ API Error")
                elif isinstance(data, dict) and 'Note' in data:
                    print(f"{name:<20} | {RED}LIMIT{RESET}       | {lat:<8} | ❌ API Limit")
                elif not data:
                    print(f"{name:<20} | {RED}EMPTY{RESET}       | {lat:<8} | ⚠️ Returned [] or {{}}")
                else:
                    # Valid check
                    is_valid = (isinstance(data, list) and len(data) > 0) or (isinstance(data, dict) and len(data) > 0)
                    if is_valid:
                        print(f"{name:<20} | {GREEN}ONLINE{RESET}      | {lat:<8} | ✅ Data OK")
                    else:
                        print(f"{name:<20} | {RED}EMPTY DATA{RESET}  | {lat:<8} | Structure empty")
            except Exception:
                print(f"{name:<20} | {RED}BAD JSON{RESET}    | {lat:<8} | Parse Error")
        elif r.status_code == 429:
             print(f"{name:<20} | {RED}RATE LMT{RESET}    | {lat:<8} | ❌ 429 Hit")
        elif r.status_code == 403 or r.status_code == 401:
             print(f"{name:<20} | {RED}AUTH ERR{RESET}    | {lat:<8} | ❌ Invalid Key")
        else:
             print(f"{name:<20} | {RED}ERR {r.status_code}{RESET}      | {lat:<8} | Request Failed")

    except Exception:
        print(f"{name:<20} | {RED}TIMEOUT{RESET}     | > 5s     | ❌ Unreachable")

# Check Yahoo Finance (Library)
print("-" * 90)
start = time.time()
try:
    ticker = yf.Ticker(TICKER)
    info = ticker.info
    lat = f"{time.time() - start:.2f}s"
    if info and 'marketCap' in info:
        print(f"{'10. Yahoo Finance':<20} | {GREEN}ONLINE{RESET}      | {lat:<8} | ✅ yfinance Library OK")
    else:
        print(f"{'10. Yahoo Finance':<20} | {RED}FAIL{RESET}        | {lat:<8} | ❌ No data in .info")
except Exception as e:
    print(f"{'10. Yahoo Finance':<20} | {RED}ERROR{RESET}       | -        | ❌ Exception: {e}")

print("-" * 90)
