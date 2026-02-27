# 📈 Stock Scout

**AI-powered stock recommendation system** combining technical analysis, fundamental scoring, and **machine learning** to identify high-probability trading opportunities in US equities.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red.svg)](https://streamlit.io/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.1-yellow.svg)](https://duckdb.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What Makes This Different?

### 1. **ML-Powered Confidence Scoring** 🤖
- **Ensemble model (v3.1)**: HistGradientBoosting + RandomForest + LogisticRegression
- **39 engineered features** across 7 categories (technical, volatility, volume, market regime, sector, price action)
- **Confidence tiers**: High (🟢 >50%), Medium (🟡 30-50%), Low (🔴 <30%)
- **Calibrated probabilities** via Isotonic Regression
- **Time-Series Cross-Validation** (5 folds) for robust evaluation

### 2. **Comprehensive Multi-Source Data** 📊
- **10 data providers**: Yahoo Finance, Alpha Vantage, Finnhub, Polygon, Tiingo, FMP, SimFin, Marketstack, Nasdaq Data Link, EODHD
- **Automatic fallback**: If one provider fails, seamlessly switches to next
- **Price verification**: Cross-validates prices across multiple sources
- **Reliability scoring**: Tracks data completeness and source agreement

### 3. **Advanced Technical & Fundamental Analysis** 📈

**Technical Indicators:**
- Moving averages (20/50/200), RSI, ATR, MACD, ADX
- Momentum consistency, volume surge detection
- Near 52-week high positioning
- Overextension vs MA_50 (pullback detection)
- Reward/risk ratio calculation

**Fundamental Scoring (Transparent Breakdown):**
- **Quality** (High/Medium/Low): ROE, ROIC, Gross Margin
- **Growth** (Fast/Moderate/Slow): Revenue YoY, EPS YoY
- **Valuation** (Cheap/Fair/Expensive): P/E, P/S ratios
- **Leverage** (Low/Medium/High): Debt-to-Equity
- Color-coded labels in UI for instant assessment

**Risk Management:**
- Earnings blackout window (7 days default)
- Beta filtering vs SPY/QQQ
- Sector diversification caps
- ATR-based position sizing
- Min/max position constraints

### 4. **DuckDB-Powered Outcome Tracking** 🦆
- **Persistent storage** for scan results and outcome tracking
- **Walk-forward backtesting** with full pipeline simulation
- **Portfolio attribution** analysis for performance decomposition

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API keys for data providers (optional but recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/asafamos/stock-scout.git
cd stock-scout

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API keys (optional):

```env
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
TIINGO_API_KEY=your_key_here
FMP_API_KEY=your_key_here
# ... (see .env.example for full list)
```

### Run

```bash
streamlit run stock_scout.py
```

Open browser at: http://localhost:8501

---

## 📊 ML Model Performance

### Model Architecture (v3.1)
- **Ensemble**: HistGradientBoosting (45%) + RandomForest (35%) + LogisticRegression (20%)
- **Features**: 39 engineered features across 7 categories
- **Calibration**: Isotonic regression for reliable probabilities
- **Validation**: Time-Series Cross-Validation (5 folds)
- **Scoring weights**: Technical 55% + Fundamental 25% + ML 20%

### Feature Categories (39 total)
| Category | Count | Examples |
|----------|-------|----------|
| Technical | 5 | RSI, ATR_Pct, Returns (5d/10d/20d) |
| Volatility | 4 | VCP_Ratio, Tightness, MA_Alignment |
| Volume Basic | 3 | Volume_Surge, Up_Down_Ratio |
| Market Regime | 4 | Market_Regime, Volatility, Trend |
| Sector Relative | 3 | Sector_RS, Sector_Momentum |
| Volume Advanced | 5 | Volume_Trend, Accumulation signals |
| Price Action | 9 | 52w positioning, Support/Resistance |
| Additional (v3.1) | 6 | Extended momentum & breadth features |

### Experimental: ML V4
- **54 features** with expanded feature engineering
- Training script: `scripts/train_ml_v4.py`
- Model: `models/ml_20d_v4.pkl`

### Key Files
- `core/feature_registry.py` - Single source of truth for feature definitions
- `core/ml_integration.py` - Model loading with validation
- `scripts/train_rolling_ml_20d.py` - Primary training script
- `models/model_20d_v3.pkl` - Current production model (v3.1)

---

## 🏗️ Architecture

```
stock-scout/
├── stock_scout.py              # Main Streamlit app (entry point)
├── core/                       # Business logic
│   ├── config.py               # Configuration management
│   ├── unified_logic.py        # Single entry point for scoring
│   ├── scoring_engine.py       # Scoring pipeline
│   ├── data_sources_v2.py      # Multi-provider data fetching (canonical)
│   ├── data/                   # Unified data API
│   ├── db/                     # DuckDB integration
│   │   ├── store.py            # Persistent scan storage
│   │   ├── outcome_tracker.py  # Outcome tracking
│   │   └── schema.py           # Database schema
│   ├── backtest/               # Walk-forward backtest engine
│   │   ├── engine.py           # Backtest orchestrator
│   │   ├── portfolio_sim.py    # Portfolio simulation
│   │   ├── attribution.py      # Performance attribution
│   │   └── stats.py            # Statistical analysis
│   ├── classification.py       # Risk classification (Core/Speculative)
│   ├── portfolio.py            # Position sizing & allocation
│   ├── scoring/                # Scoring modules (fundamental, recommendation)
│   ├── feature_registry.py     # ML feature definitions
│   ├── ml_integration.py       # ML model loading & inference
│   ├── filters/                # Filtering logic
│   ├── providers/              # Data providers
│   └── risk/                   # Risk management
├── ui/                         # UI layer
│   ├── design_system.py        # Modern design system
│   ├── stock_ui.py             # Main UI components
│   ├── components/             # Card components
│   └── styles/                 # CSS & RTL support
├── models/                     # Trained ML models
│   ├── model_20d_v3.pkl        # Production model (v3.1, 39 features)
│   └── ml_20d_v4.pkl           # Experimental V4 (54 features)
├── scripts/                    # Training, backtesting & scanning scripts
│   ├── backtest_recommendations.py
│   ├── train_recommender.py
│   ├── train_ml_v4.py
│   └── run_2000_meteor_scan.py
├── pipeline/                   # Data pipeline modules
├── tools/                      # Debug, audit, benchmark scripts
├── tests/                      # Unit & integration tests (75+ files)
└── data/                       # Scan history & ticker lists
```

---

## 🔄 CI/CD & Automation

### Automated Daily Scans
GitHub Actions runs **4 scans daily** aligned with NYSE trading hours:
- **8:30 AM ET** - Pre-market scan
- **10:00 AM ET** - Early session scan
- **3:00 PM ET** - Late session scan
- **4:30 PM ET** - End of day scan

Includes US market holiday detection to skip non-trading days.

### Additional Workflows
- **`ci.yml`** - Unit & integration tests on every push
- **`weekly-training.yml`** - Automated ML model retraining
- **`ml-a2-contract.yml`** - ML model validation contract
- **`weekly_backtest.yml`** - Historical backtest validation
- **`track_outcomes.yml`** - Outcome tracking for past recommendations

---

## 📖 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design overview
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Recent refactoring details
- **[docs/archive/](docs/archive/)** - Historical documentation
  - [SUMMARY_HE.md](docs/archive/SUMMARY_HE.md) - Hebrew summary of ML improvements
  - [MODEL_COMPARISON.md](docs/archive/MODEL_COMPARISON.md) - XGBoost vs Logistic performance
  - [PRODUCTION_INTEGRATION.md](docs/archive/PRODUCTION_INTEGRATION.md) - Deployment guide
  - [DATA_SOURCES_REPORT.md](docs/archive/DATA_SOURCES_REPORT.md) - Provider capabilities

---

## 🔧 Advanced Usage

### Meteor Mode (Aggressive Discovery)
- Purpose: Emphasizes tight VCP contractions, strong RS, and pocket pivots for large-universe momentum scouting.
- Env vars:
    - `METEOR_MODE=1`: Enable momentum-friendly RSI mapping and Meteor filters.
    - `MIN_MCAP`/`MAX_MCAP`: Universe market-cap bounds (e.g., `300000000` → `15000000000`).
    - `EARNINGS_THRESHOLD`: Optional blackout window in days (default `7`).
- Core signals: `VCP_Ratio` (ATR10/ATR30), `Dist_From_52w_High`, `RS_21d/RS_63d`, `Pocket_Pivot_Ratio`, `Volume_Surge_Ratio`.
- RS ranking: Early blended pass across full universe: `0.7*RS_63d + 0.3*RS_21d`, filter top 20% before deep scans.
- Runner example:

```bash
python scripts/run_2000_meteor_scan.py
```

Outputs CSV at `reports/meteor_results_YYYYMMDD.csv` including `Meteor_Confidence_Score`.

### Backtesting on Custom Date Range

```bash
python scripts/backtest_recommendations.py \
    --use-finnhub \
    --limit 500 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --horizons 5,10,20
```

### Retraining the ML Model

```bash
# Generate new signals
python scripts/backtest_recommendations.py --use-finnhub --limit 500

# Train model (v3.1)
python scripts/train_recommender.py \
    --signals backtest_signals_latest.csv \
    --horizon 5 \
    --model xgboost \
    --cv

# Train V4 model (experimental)
python scripts/train_ml_v4.py
```

### Running Tests

```bash
pytest tests/ -v
```

---

## 🎨 UI Features

- **Modern design system** with dark mode support
- **Hebrew RTL support** for local market
- **Card-based layout** with confidence badges
- **Interactive charts** (Plotly)
- **CSV export** with ML scores
- **Real-time filtering** by risk level, sector, score range
- **Data quality indicators** for each stock

---

## ⚠️ Disclaimer

**This is not investment advice.** Stock Scout is a research and educational tool. Always:
- Conduct your own due diligence
- Understand the risks involved in trading
- Never invest more than you can afford to lose
- Consult with a licensed financial advisor

Past performance does not guarantee future results.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Author**: Asaf
**Repository**: [github.com/asafamos/stock-scout](https://github.com/asafamos/stock-scout)

---

**Built with ❤️ using Python, Streamlit, scikit-learn, DuckDB, and lots of data**
