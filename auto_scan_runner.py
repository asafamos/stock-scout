"""
Automated Stock Scanner - Runs twice daily via GitHub Actions.
Downloads maximum stocks data and saves to latest_scan.parquet.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json

print("=" * 80)
print("🤖 Stock Scout Auto Scan")
print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# Comprehensive list of top 500 US stocks
UNIVERSE = [
    # Mega caps (50)
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "V",
    "JPM", "WMT", "UNH", "XOM", "JNJ", "MA", "LLY", "PG", "ORCL", "HD",
    "CVX", "COST", "ABBV", "MRK", "KO", "PEP", "BAC", "ADBE", "CRM", "TMO",
    "CSCO", "MCD", "ACN", "ABT", "NFLX", "AMD", "DHR", "NKE", "WFC", "LIN",
    "DIS", "PM", "QCOM", "TXN", "INTC", "IBM", "UNP", "INTU", "VZ", "NEE",
    
    # Large caps (100)
    "AMGN", "HON", "BA", "T", "RTX", "LOW", "SPGI", "SBUX", "ELV", "CAT",
    "PFE", "BMY", "AXP", "GILD", "BKNG", "DE", "MDLZ", "GS", "BLK", "ISRG",
    "ADP", "SYK", "ADI", "VRTX", "CI", "MMC", "TJX", "PLD", "REGN", "AMT",
    "C", "CB", "SO", "FI", "LRCX", "MO", "ZTS", "DUK", "SCHW", "MS",
    "BSX", "BDX", "EOG", "PGR", "EQIX", "CME", "SLB", "ETN", "ITW", "APD",
    "CL", "NOW", "CVS", "USB", "AMAT", "MU", "HCA", "PANW", "MCO", "SHW",
    "NOC", "PYPL", "TGT", "BX", "KLAC", "ICE", "WM", "ECL", "AON", "CSX",
    "GE", "NSC", "PNC", "FCX", "MAR", "AIG", "CCI", "PSA", "MCK", "EMR",
    "FDX", "PH", "ROP", "MSI", "CARR", "AFL", "AJG", "PCAR", "NEM", "GM",
    "TRV", "COF", "AZO", "F", "ROST", "AEP", "ADSK", "KMB", "WELL", "D",
    "NXPI", "SRE", "PSX", "PAYX", "FTNT", "ORLY", "DHI", "SLB", "KHC", "DLR",
    "RSG", "YUM", "ODFL", "CTAS", "MNST", "CMG", "LHX", "TEL", "HSY", "SPG",
    "CTVA", "KDP", "STZ", "IT", "EXC", "XEL", "EW", "IDXX", "MCHP", "HLT",
    "IQV", "O", "GIS", "VRSK", "CMI", "DD", "GWW", "OTIS", "PPG", "CHTR",
    
    # Mid caps (150)
    "CPRT", "VMC", "ED", "KR", "TT", "RMD", "HES", "ROK", "APH", "ON",
    "FAST", "DXCM", "AVB", "BK", "TROW", "ANSS", "MTD", "KEYS", "CDW", "FTV",
    "WEC", "HIG", "EXR", "BIIB", "DOV", "GEHC", "MLM", "GLW", "DFS", "ES",
    "MPWR", "EQR", "WBD", "ALB", "WTW", "VICI", "CHD", "IFF", "EFX", "ADM",
    "LYB", "A", "LEN", "TRGP", "HUBB", "MTB", "FE", "IR", "SBAC", "ARE",
    "ZBH", "INVH", "TDY", "BR", "AWK", "RF", "WST", "STT", "WAT", "TSCO",
    "WY", "HBAN", "VTR", "DTE", "PPL", "CLX", "ESS", "TYL", "APTV", "CINF",
    "CBRE", "FITB", "CAH", "TFX", "FANG", "PKG", "FDS", "STE", "CFG", "TTWO",
    "NVR", "BALL", "AEE", "EBAY", "EXPD", "LH", "IEX", "TER", "CMS", "MAA",
    "ALGN", "HOLX", "NTRS", "SYY", "BLDR", "DRI", "ENPH", "CNP", "OMC", "SWK",
    "GDDY", "WDC", "PHM", "MKC", "NTAP", "EVRG", "ULTA", "GRMN", "STLD", "ATO",
    "ZBRA", "IP", "J", "CTLT", "LVS", "WAB", "UAL", "DLTR", "K", "POOL",
    "AKAM", "CBOE", "EIX", "MOH", "JBHT", "HST", "FMC", "AMP", "CTRA", "TDG",
    "NDAQ", "PODD", "BBY", "ETR", "LDOS", "EMN", "L", "JKHY", "LUV", "CE",
    "VTRS", "TSN", "WYNN", "KIM", "SWKS", "DOC", "GEN", "LNT", "REG", "GNRC",
    
    # Smaller caps & growth (200)
    "DECK", "TPR", "PAYC", "RJF", "EXPE", "MOS", "NI", "TRMB", "BXP", "PFG",
    "TECH", "HII", "AIZ", "CRL", "MKTX", "CHRW", "CPT", "COO", "RHI", "INCY",
    "FFIV", "SNA", "BF-B", "HSIC", "WHR", "TXT", "ALLE", "PKI", "AES", "SJM",
    "IPG", "UDR", "NRG", "BEN", "GL", "FRT", "MGM", "DAY", "CCL", "BMRN",
    "IRM", "NDSN", "UHS", "HWM", "BIO", "NCLH", "PNR", "CPB", "LKQ", "TAP",
    "AAL", "RL", "LW", "BWA", "VFC", "PARA", "FOXA", "FOX", "DISH", "NWS",
    "MHK", "ALK", "FNF", "PNW", "JNPR", "HAS", "SEE", "NVT", "AIR", "OGN",
    "XRAY", "AOS", "BKR", "HAL", "DVN", "OXY", "MRO", "FANG", "APA", "HES",
    "COP", "PSX", "VLO", "MPC", "TSO", "WMB", "KMI", "OKE", "EPD", "ET",
    "TRGP", "LNG", "CHRD", "CTRA", "OVV", "PR", "CNQ", "IMO", "SU", "TRP",
    "ENB", "PAA", "MPLX", "WES", "DCP", "AM", "USAC", "AROC", "SM", "MGY",
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "ES",
    "PEG", "WEC", "EIX", "DTE", "PPL", "FE", "AEE", "CMS", "CNP", "ETR",
    "EVRG", "NI", "LNT", "ATO", "NWE", "SWX", "OGE", "PNW", "AVA", "NJR",
    "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "WELL", "AVB", "EQR", "DLR",
    "O", "VTR", "INVH", "MAA", "ESS", "ARE", "KIM", "UDR", "HST", "REG",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "ATVI",
    "GOOGL", "META", "CRM", "INTU", "ADBE", "NOW", "WDAY", "PANW", "FTNT", "ZS",
    "UBER", "LYFT", "ABNB", "DASH", "SQ", "COIN", "PLTR", "CRWD", "SNOW", "MDB",
    "NET", "DDOG", "OKTA", "DOCU", "ZM", "TWLO", "ROKU", "SPOT", "PINS", "SNAP"
]

print(f"🎯 Universe size: {len(UNIVERSE)} stocks")
print(f"📥 Downloading historical data (365 days lookback)...")

# Download data
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
try:
    data = yf.download(UNIVERSE, start=start_date, group_by='ticker', 
                      progress=False, threads=True)
    print(f"✅ Data download complete")
except Exception as e:
    print(f"❌ Download failed: {e}")
    sys.exit(1)

# Process stocks
print(f"⚙️  Processing indicators and scores...")
results = []
success_count = 0
fail_count = 0

for ticker in UNIVERSE:
    try:
        if ticker not in data.columns.get_level_values(0):
            fail_count += 1
            continue
            
        df = data[ticker]
        if len(df) < 30:
            fail_count += 1
            continue
        
        close = df['Close'].dropna()
        if len(close) < 30:
            fail_count += 1
            continue
        
        # Calculate metrics
        ret_20d = close.pct_change(20).iloc[-1]
        ret_60d = close.pct_change(60).iloc[-1] if len(close) >= 60 else None
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        
        # Volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized
        
        # Score calculation
        score = 0
        if not pd.isna(rsi_val):
            if 30 < rsi_val < 70:
                score += 5
            elif rsi_val < 30:
                score += 8  # Oversold bonus
        
        if not pd.isna(ret_20d) and ret_20d > 0:
            score += ret_20d * 50
        
        if not pd.isna(ret_60d) and ret_60d > 0:
            score += ret_60d * 30
        
        # Penalty for high volatility
        if volatility > 0.5:
            score *= 0.8
        
        results.append({
            'Ticker': ticker,
            'overall_score_20d': score,
            'RSI': rsi_val,
            'Return_20d': ret_20d,
            'Return_60d': ret_60d,
            'Volatility': volatility,
            'Close': close.iloc[-1],
            'Volume': df['Volume'].iloc[-1],
            'Sector': 'Unknown',  # Will be enriched later
            'Risk_Level': 'core' if score > 10 else 'speculative',
            'Data_Quality': 'high',
            'ML_20d_Prob': 0.5,
        })
        success_count += 1
        
        if success_count % 50 == 0:
            print(f"   Processed: {success_count}/{len(UNIVERSE)} ({success_count/len(UNIVERSE)*100:.0f}%)")
            
    except Exception as e:
        fail_count += 1

# Create DataFrame and sort
df = pd.DataFrame(results)
df = df.sort_values('overall_score_20d', ascending=False)
df['Overall_Rank'] = range(1, len(df) + 1)

print(f"\n✅ Processing complete:")
print(f"   Success: {success_count} stocks")
print(f"   Failed: {fail_count} stocks")
print(f"   Total in dataset: {len(df)}")

# Save results
output_dir = Path('data/scans')
output_dir.mkdir(parents=True, exist_ok=True)

df.to_parquet(output_dir / 'latest_scan.parquet', index=False)

metadata = {
    "timestamp": datetime.now().isoformat(),
    "scan_type": "automated_full",
    "total_tickers": len(df),
    "universe_size": len(UNIVERSE),
    "success_count": success_count,
    "fail_count": fail_count,
    "top_ticker": df.iloc[0]['Ticker'] if len(df) > 0 else None,
    "top_score": float(df.iloc[0]['overall_score_20d']) if len(df) > 0 else 0,
    "avg_score": float(df['overall_score_20d'].mean()),
    "scan_duration_seconds": 0,  # Will be calculated by GitHub Actions
}

with open(output_dir / 'latest_scan.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Results saved:")
print(f"   File: data/scans/latest_scan.parquet")
print(f"   Top stock: {metadata['top_ticker']} (score: {metadata['top_score']:.1f})")
print(f"   Average score: {metadata['avg_score']:.1f}")

print("\n" + "=" * 80)
print("🏆 Top 10 stocks:")
print("=" * 80)
for i, row in df.head(10).iterrows():
    print(f"{row['Overall_Rank']:2d}. {row['Ticker']:6s} - Score: {row['overall_score_20d']:6.1f} | RSI: {row['RSI']:4.0f}")

print("=" * 80)
print(f"✅ Auto scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)
