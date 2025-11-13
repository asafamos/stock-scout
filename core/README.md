# Stock Scout - Core Module

## 📁 Structure

```
core/
├── __init__.py              # Package initialization
├── config.py                # Configuration and API keys management
├── logging_config.py        # Centralized logging setup
├── models.py                # Data models and dataclasses
├── data_sources.py          # API clients with fallback logic
├── risk.py                  # Risk management and filters
└── scoring/                 # Scoring modules
    └── __init__.py
```

## 🎯 Design Principles

1. **Separation of Concerns**: Business logic separated from UI
2. **Error Resilience**: Graceful fallback between data providers
3. **Type Safety**: Comprehensive type hints throughout
4. **Logging**: Structured logging with sensitive data masking
5. **Configuration**: Centralized, dataclass-based configuration
6. **Risk Management**: Built-in risk constraints and position sizing

## 📦 Modules

### config.py
- `Config`: Main configuration dataclass
- `APIKeys`: API keys management with environment variable loading
- `get_config()`: Get global config instance
- `get_api_keys()`: Get global API keys instance

### logging_config.py
- `setup_logging()`: Initialize logging system
- `get_logger()`: Get logger instance
- `mask_sensitive()`: Mask API keys and sensitive data

### models.py
Data classes for type-safe data handling:
- `StockData`: Historical price data
- `TechnicalScore`: Technical analysis breakdown
- `FundamentalBreakdown`: Detailed fundamental metrics with labels
- `FundamentalScore`: Fundamental score with breakdown
- `AdvancedSignals`: Advanced filtering signals
- `StockRecommendation`: Complete recommendation with all data
- `Portfolio`: Portfolio allocation result

### data_sources.py
API clients with automatic fallback:
- `HTTPClient`: Retry logic with exponential backoff
- `AlphaVantageClient`: Alpha Vantage API (with rate limiting)
- `FinnhubClient`: Finnhub API
- `PolygonClient`: Polygon.io API
- `TiingoClient`: Tiingo API
- `YFinanceClient`: Yahoo Finance (yfinance)
- `DataSourceManager`: Orchestrates all sources with fallback

### risk.py
Risk management and filtering:
- `RiskConstraints`: Risk parameters dataclass
- `RiskManager`: Main risk management class
  - Liquidity filters (price, volume, dollar volume)
  - Market cap filters
  - Beta filters
  - Sector concentration limits
  - Position sizing with constraints
  - Earnings blackout period filtering

## 🔧 Usage Examples

### Basic Configuration

```python
from core.config import get_config, get_api_keys

config = get_config()
api_keys = get_api_keys()

print(f"Budget: ${config.budget_total}")
print(f"Has Alpha Vantage: {api_keys.has_alpha_vantage()}")
```

### Logging

```python
from core.logging_config import setup_logging, get_logger, mask_sensitive

setup_logging(level=logging.INFO)
logger = get_logger("my_module")

api_key = "ABCDEFGH1234"
logger.info(f"Using API key: {mask_sensitive(api_key)}")
# Output: [2025-11-13 10:30:00] [INFO] [my_module] Using API key: ***1234
```

### Data Sources

```python
from core.data_sources import DataSourceManager

manager = DataSourceManager()

# Check connectivity
status = manager.check_all_connectivity()
for source, (ok, msg) in status.items():
    print(f"{source}: {'✓' if ok else '✗'} {msg}")

# Get price with fallback
prices = manager.get_price_with_fallback("AAPL")
print(f"Alpha: ${prices['alpha']}, Finnhub: ${prices['finnhub']}")

# Get fundamentals (Alpha → Finnhub fallback)
fund = manager.get_fundamentals_with_fallback("AAPL")
print(f"ROE: {fund.get('roe')}, P/E: {fund.get('pe')}")
```

### Risk Management

```python
from core.risk import RiskManager
import pandas as pd

risk = RiskManager()

# Apply liquidity filters
df = pd.DataFrame({
    "Ticker": ["AAPL", "XYZ"],
    "Price": [150.0, 2.5],
    "Avg_Volume": [50_000_000, 100_000],
    "Final_Score": [85, 90]
})

df_filtered = risk.apply_liquidity_filters(df)
# XYZ filtered out due to price < $3

# Apply sector concentration
df["Sector"] = ["Technology", "Technology"]
df_limited = risk.apply_sector_concentration(df)
# Keeps max 3 per sector
```

### Models

```python
from core.models import StockRecommendation, TechnicalScore, FundamentalScore
from datetime import datetime

rec = StockRecommendation(
    ticker="AAPL",
    final_score=85.5,
    technical_score=TechnicalScore(total=80.0, ma_score=15.0),
    current_price=150.0,
    sector="Technology",
    allocation_amount=750.0,
    shares=5
)

# Convert to dictionary for JSON/display
data = rec.to_dict()
```

## 🔐 Security

- All API keys loaded from environment variables or Streamlit secrets
- No hardcoded credentials
- Sensitive data masked in logs
- HTTP requests use timeout and retry limits

## 🎛️ Configuration

Set environment variables or Streamlit secrets:

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
TIINGO_API_KEY=your_key_here
```

Or in `.streamlit/secrets.toml`:

```toml
ALPHA_VANTAGE_API_KEY = "your_key_here"
FINNHUB_API_KEY = "your_key_here"
POLYGON_API_KEY = "your_key_here"
TIINGO_API_KEY = "your_key_here"
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=core tests/
```

## 📝 Next Steps

1. Migrate remaining scoring logic to `core/scoring/`
2. Update `stock_scout.py` to use core modules
3. Add unit tests for all modules
4. Expand backtest framework
