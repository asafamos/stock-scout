"""Quick 500-stock scan with hardcoded universe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Large universe of 500 major US stocks
UNIVERSE_500 = [
    # Tech (50)
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ADBE",
    "CRM", "ORCL", "CSCO", "INTC", "AMD", "QCOM", "TXN", "IBM", "NOW", "INTU",
    "PANW", "AMAT", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MCHP", "FTNT", "ANSS",
    "APH", "TEL", "ON", "MPWR", "KEYS", "TYL", "VRSN", "JNPR", "FFIV", "AKAM",
    "NTAP", "ZBRA", "ENPH", "SEDG", "FSLR", "TER", "WOLF", "SWKS", "QRVO", "MRVL",
    
    # Finance (50)
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
    "PNC", "COF", "TFC", "BK", "CME", "SPGI", "ICE", "MCO", "AON", "MMC",
    "AJG", "BRO", "CB", "PGR", "TRV", "ALL", "AIG", "MET", "PRU", "AFL",
    "HIG", "WRB", "CINF", "L", "GL", "AIZ", "ERIE", "RNR", "RE", "PFG",
    "FNF", "FAF", "MTG", "ALLY", "DFS", "SYF", "PYPL", "MA", "V", "FIS",
    
    # Healthcare (50)
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
    "AMGN", "CVS", "CI", "ELV", "HUM", "GILD", "REGN", "VRTX", "ISRG", "ZTS",
    "BSX", "SYK", "MDT", "EW", "BAX", "BDX", "A", "RMD", "IDXX", "IQV",
    "MTD", "DGX", "WST", "WAT", "DXCM", "HOLX", "ZBH", "ALGN", "TFX", "STE",
    "PODD", "TECH", "RVTY", "LH", "COO", "SOLV", "CRL", "HSIC", "MOH", "CAH",
    
    # Consumer (50)
    "WMT", "HD", "PG", "KO", "PEP", "COST", "MCD", "NKE", "TGT", "SBUX",
    "LOW", "TJX", "DG", "DLTR", "ROST", "CMG", "YUM", "DPZ", "QSR", "BKNG",
    "MAR", "HLT", "DRI", "EAT", "TXRH", "CAKE", "BLMN", "DIN", "BJRI", "CHUY",
    "EL", "CL", "CLX", "CHD", "KMB", "GIS", "K", "CPB", "CAG", "SJM",
    "HSY", "MDLZ", "MNST", "KDP", "STZ", "TAP", "BF.B", "SAM", "BREW", "FIZZ",
    
    # Industrial (50)
    "CAT", "BA", "HON", "UNP", "RTX", "LMT", "GE", "MMM", "DE", "EMR",
    "ITW", "CSX", "NSC", "FDX", "UPS", "WM", "RSG", "ETN", "PH", "ROK",
    "AME", "IR", "FAST", "PCAR", "ODFL", "CHRW", "XPO", "JBHT", "KNX", "EXPD",
    "URI", "PWR", "CMI", "HUBB", "AOS", "DOV", "BLDR", "VMC", "MLM", "NUE",
    "STLD", "RS", "POOL", "WCC", "GWW", "WSO", "MSM", "DCI", "SSD", "FLS",
    
    # Energy (50)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
    "KMI", "WMB", "PXD", "HES", "DVN", "FANG", "MRO", "APA", "EQT", "OKE",
    "TRGP", "LNG", "CHRD", "CTRA", "OVV", "PR", "CNQ", "IMO", "SU", "TRP",
    "ENB", "EPD", "ET", "PAA", "MPLX", "WES", "DCP", "HESM", "MMP", "ENLC",
    "AM", "USAC", "AROC", "SM", "MGY", "CPE", "CLR", "MTDR", "RRC", "VNOM",
    
    # Materials (40)
    "LIN", "APD", "SHW", "ECL", "DD", "DOW", "FCX", "NEM", "VMC", "MLM",
    "EMN", "CE", "FMC", "ALB", "SQM", "IFF", "PPG", "RPM", "SEE", "BALL",
    "PKG", "AMCR", "AVY", "CCK", "SON", "CF", "MOS", "NTR", "IP", "WRK",
    "KWR", "OLN", "SLGN", "HUN", "NEU", "HWKN", "MERC", "OMG", "SHOO", "CBT",
    
    # Utilities (35)
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "ES",
    "PEG", "WEC", "EIX", "DTE", "PPL", "FE", "AEE", "CMS", "CNP", "ETR",
    "EVRG", "NI", "LNT", "ATO", "NWE", "SWX", "OGE", "PNW", "AVA", "NJR",
    "SR", "BKH", "MDU", "CPK", "UTL",
    
    # Real Estate (35)
    "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "WELL", "AVB", "EQR", "DLR",
    "O", "VTR", "INVH", "MAA", "ESS", "ARE", "KIM", "UDR", "HST", "REG",
    "BXP", "VNO", "SLG", "HPP", "DEI", "HIW", "PDM", "CUZ", "KRG", "SKT",
    "ROIC", "WPC", "NNN", "ADC", "STOR",
    
    # Communication (35)
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "ATVI",
    "WBD", "PARA", "FOXA", "FOX", "LYV", "OMC", "IPG", "NWSA", "NWS", "DISH",
    "LUMN", "SIRI", "MSG", "MSGS", "CABO", "CNK", "IMAX", "RGS", "AMC", "GLMG",
    "GTLS", "LPSN", "MGNI", "WRB", "NYT",
    
    # Consumer Discretionary (30)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "BKNG", "LOW", "TJX", "SBUX", "TGT",
    "F", "GM", "APTV", "LVS", "WYNN", "MGM", "CZR", "PENN", "GRMN", "HAS",
    "POOL", "WHR", "TPR", "RL", "PVH", "UA", "UAA", "SKX", "CROX", "COLM",
    
    # Technology Services (30)
    "GOOGL", "META", "CRM", "INTU", "ADBE", "NOW", "WDAY", "PANW", "FTNT", "ZS",
    "DDOG", "NET", "OKTA", "CRWD", "S", "SNOW", "MDB", "TEAM", "ZM", "DOCU",
    "TWLO", "PCTY", "VEEV", "HUBS", "RNG", "BILL", "ESTC", "GTLB", "NCNO", "FROG",
]

print(f"📊 Universe Size: {len(UNIVERSE_500)} stocks")
print(f"🔄 Starting batch scan...\n")

# Run batch scan with the hardcoded universe
from batch_scan import run_batch_scan
from pathlib import Path

result = run_batch_scan(
    universe_size=len(UNIVERSE_500),
    lookback_days=180,
    output_dir=Path("data/scans"),
    custom_universe=UNIVERSE_500  # Pass our custom universe
)

print("\n✅ Scan completed!")
print(f"📄 Check data/scans/ for results")
