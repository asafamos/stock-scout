"""
Sector mapping utilities for Stock Scout ML features.

Maps stocks to their GICS sectors and provides sector ETF data for
calculating relative strength vs sector.
"""
from typing import Optional, Dict

# Sector ETFs (SPDR Select Sector ETFs)
SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Financial": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication": "XLC",
}

# Reverse mapping: ETF symbol to sector name
ETF_TO_SECTOR: Dict[str, str] = {v: k for k, v in SECTOR_ETFS.items()}

# Stock to sector mapping (Top ~150 S&P 500 stocks by market cap)
# This can be expanded or loaded from external source
STOCK_SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", 
    "AVGO": "Technology", "CSCO": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "ACN": "Technology",
    "AMD": "Technology", "INTC": "Technology", "IBM": "Technology",
    "TXN": "Technology", "QCOM": "Technology", "AMAT": "Technology",
    "MU": "Technology", "LRCX": "Technology", "ADI": "Technology",
    "KLAC": "Technology", "SNPS": "Technology", "CDNS": "Technology",
    "MRVL": "Technology", "FTNT": "Technology", "PANW": "Technology",
    "NOW": "Technology", "INTU": "Technology", "PLTR": "Technology",
    
    # Communication Services
    "GOOGL": "Communication", "GOOG": "Communication", "META": "Communication",
    "NFLX": "Communication", "DIS": "Communication", "CMCSA": "Communication",
    "VZ": "Communication", "T": "Communication", "TMUS": "Communication",
    "CHTR": "Communication", "EA": "Communication", "TTWO": "Communication",
    "WBD": "Communication", "OMC": "Communication", "IPG": "Communication",
    
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "MAR": "Consumer Discretionary", "ORLY": "Consumer Discretionary",
    "AZO": "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "DHI": "Consumer Discretionary", "LEN": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "F": "Consumer Discretionary",
    "APTV": "Consumer Discretionary", "GRMN": "Consumer Discretionary",
    
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "COST": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    "KMB": "Consumer Staples", "GIS": "Consumer Staples", "K": "Consumer Staples",
    "HSY": "Consumer Staples", "STZ": "Consumer Staples", "KHC": "Consumer Staples",
    "SYY": "Consumer Staples", "KR": "Consumer Staples", "WBA": "Consumer Staples",
    
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "PFE": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "AMGN": "Healthcare", "BMY": "Healthcare", "GILD": "Healthcare",
    "VRTX": "Healthcare", "ISRG": "Healthcare", "REGN": "Healthcare",
    "MDT": "Healthcare", "SYK": "Healthcare", "BSX": "Healthcare",
    "ZTS": "Healthcare", "ELV": "Healthcare", "CI": "Healthcare",
    "HUM": "Healthcare", "MCK": "Healthcare", "CVS": "Healthcare",
    "MRNA": "Healthcare", "BIIB": "Healthcare", "ILMN": "Healthcare",
    
    # Financials
    "BRK.B": "Financial", "JPM": "Financial", "V": "Financial",
    "MA": "Financial", "BAC": "Financial", "WFC": "Financial",
    "GS": "Financial", "MS": "Financial", "SPGI": "Financial",
    "BLK": "Financial", "C": "Financial", "AXP": "Financial",
    "SCHW": "Financial", "CB": "Financial", "PGR": "Financial",
    "MMC": "Financial", "AON": "Financial", "CME": "Financial",
    "ICE": "Financial", "USB": "Financial", "PNC": "Financial",
    "TFC": "Financial", "AIG": "Financial", "MET": "Financial",
    "PRU": "Financial", "AFL": "Financial", "TRV": "Financial",
    "PYPL": "Financial", "COF": "Financial", "BK": "Financial",
    
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "UNP": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "BA": "Industrials", "LMT": "Industrials", "GE": "Industrials",
    "MMM": "Industrials", "ADP": "Industrials", "ETN": "Industrials",
    "ITW": "Industrials", "EMR": "Industrials", "FDX": "Industrials",
    "NSC": "Industrials", "CSX": "Industrials", "WM": "Industrials",
    "JCI": "Industrials", "GD": "Industrials", "NOC": "Industrials",
    "TT": "Industrials", "CARR": "Industrials", "OTIS": "Industrials",
    "PH": "Industrials", "ROK": "Industrials", "FAST": "Industrials",
    "CTAS": "Industrials", "PAYX": "Industrials", "VRSK": "Industrials",
    
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    "PXD": "Energy", "WMB": "Energy", "KMI": "Energy",
    "HAL": "Energy", "DVN": "Energy", "FANG": "Energy",
    "HES": "Energy", "BKR": "Energy", "OKE": "Energy",
    
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "ECL": "Materials", "FCX": "Materials", "NEM": "Materials",
    "NUE": "Materials", "DOW": "Materials", "DD": "Materials",
    "PPG": "Materials", "VMC": "Materials", "MLM": "Materials",
    "CTVA": "Materials", "ALB": "Materials", "CF": "Materials",
    "MOS": "Materials", "IFF": "Materials", "LYB": "Materials",
    
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "SRE": "Utilities",
    "EXC": "Utilities", "XEL": "Utilities", "PEG": "Utilities",
    "ED": "Utilities", "WEC": "Utilities", "ES": "Utilities",
    "AWK": "Utilities", "DTE": "Utilities", "AEE": "Utilities",
    "CMS": "Utilities", "FE": "Utilities", "ETR": "Utilities",
    
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate",
    "CCI": "Real Estate", "PSA": "Real Estate", "O": "Real Estate",
    "WELL": "Real Estate", "DLR": "Real Estate", "SPG": "Real Estate",
    "AVB": "Real Estate", "EQR": "Real Estate", "VICI": "Real Estate",
    "VTR": "Real Estate", "SBAC": "Real Estate", "WY": "Real Estate",
    "ARE": "Real Estate", "MAA": "Real Estate", "UDR": "Real Estate",
    "ESS": "Real Estate", "INVH": "Real Estate", "EXR": "Real Estate",
}


def get_stock_sector(ticker: str) -> str:
    """
    Get the sector for a given stock ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
    
    Returns:
        Sector name (e.g., "Technology") or "Unknown" if not mapped
    """
    # Normalize ticker (remove any class suffixes like .B)
    clean_ticker = ticker.upper().replace(".", "").replace("-", "")
    
    # Try exact match first
    if ticker.upper() in STOCK_SECTOR_MAP:
        return STOCK_SECTOR_MAP[ticker.upper()]
    
    # Try cleaned ticker
    if clean_ticker in STOCK_SECTOR_MAP:
        return STOCK_SECTOR_MAP[clean_ticker]
    
    # Try with common suffixes removed
    base_ticker = ticker.split(".")[0].split("-")[0].upper()
    if base_ticker in STOCK_SECTOR_MAP:
        return STOCK_SECTOR_MAP[base_ticker]
    
    return "Unknown"


def get_sector_etf(sector: str) -> Optional[str]:
    """
    Get the ETF symbol for a sector.
    
    Args:
        sector: Sector name (e.g., "Technology")
    
    Returns:
        ETF symbol (e.g., "XLK") or None if unknown sector
    """
    return SECTOR_ETFS.get(sector)


def get_all_sector_etfs() -> list:
    """Return list of all sector ETF symbols."""
    return list(SECTOR_ETFS.values())


def get_all_sectors() -> list:
    """Return list of all sector names."""
    return list(SECTOR_ETFS.keys())
