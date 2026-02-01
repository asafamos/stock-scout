# Stock Scout - ML Improvement Prompts Guide

## 📋 Summary
This document contains all the prompts used to upgrade the Stock Scout ML system from a basic 5-feature model to an advanced ensemble system with 30+ features.

---

## ✅ Phase 1: Critical Fixes (Prompts 1-5)

### Prompt 1: API Key Security
Remove hardcoded Polygon API key and require environment variable.

### Prompt 2: Cross-Validation (TimeSeriesSplit)
Replace random split with TimeSeriesSplit to prevent data leakage.

### Prompt 3: Expanded Feature Set (13 Features)
Expand from 5 to 13 features including VCP, momentum, and volatility metrics.

### Prompt 4: Remove Calibration Hack
Replace post-hoc calibration hack with proper CalibratedClassifierCV.

### Prompt 5: GitHub Actions Timing
Align automation with NYSE trading hours.

---

## ✅ Phase 2: Advanced ML (Prompts 6-8)

### Prompt 6: Model Ensembling
Create EnsembleClassifier combining:
- HistGradientBoostingClassifier (gradient boosting)
- RandomForestClassifier (bagging)
- LogisticRegression (linear baseline)

### Prompt 7: Precision@K Optimization
Add trading-specific metrics:
- precision_at_k() function
- P@20 and P@50 reporting
- Lift calculation vs random baseline

### Prompt 8: Market Regime Detection
Add market context features:
- calculate_market_regime() function
- Bull/Bear/Sideways classification
- SPY-based regime features (4 new features)

---

## ✅ Phase 3: Feature Engineering (Prompts 9-12)

### Prompt 9: Sector Features
```
Add sector-based features to the ML training pipeline. This requires:

1. Create a new file `core/sector_mapping.py` with:
   - SECTOR_ETFS dictionary mapping sector names to their ETF symbols:
     {"Technology": "XLK", "Financial": "XLF", "Energy": "XLE", "Healthcare": "XLV",
      "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Industrials": "XLI",
      "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE", "Communication": "XLC"}
   - STOCK_SECTOR_MAP dictionary (can start with top 100 S&P 500 stocks mapped to sectors)
   - Function get_stock_sector(ticker) that returns sector name or "Unknown"

2. In train_rolling_ml_20d.py, add function fetch_sector_etf_data(start_date, end_date):
   - Fetch daily close prices for all sector ETFs using yfinance
   - Calculate 20-day returns for each sector ETF
   - Return DataFrame with date index and columns like "XLK_Return_20d", "XLF_Return_20d", etc.

3. Add function calculate_sector_features(stock_df, ticker, sector_data):
   - Get the stock's sector using get_stock_sector(ticker)
   - Get the corresponding sector ETF return
   - Calculate:
     * Sector_RS = stock's 20d return - sector ETF's 20d return
     * Sector_Momentum = sector ETF's 20d return
     * Sector_Rank = rank of sector among all sectors (1-11, 1=best)
   - Return dict with these 3 features

4. Update FEATURES list to include: 'Sector_RS', 'Sector_Momentum', 'Sector_Rank'

5. Update EXPECTED_FEATURES in core/ml_integration.py to match

6. Update prepare_ml_features() in ml_integration.py to calculate sector features at inference time
```

### Prompt 10: Volume Profile Features
```
Add volume-based features to detect institutional accumulation patterns.

In train_rolling_ml_20d.py:

1. Add these volume features to the calculate features section:
   - Volume_Ratio_20d = current volume / 20-day average volume
   - Volume_Trend = linear regression slope of volume over last 20 days
   - Up_Volume_Ratio = sum of volume on up days / total volume over 20 days
   - Volume_Price_Confirm = 1 if (price up AND volume up) else 0 over last 5 days avg
   - Relative_Volume_Rank = percentile rank of today's volume vs last 60 days

2. Update FEATURES list to add these 5 new features

3. Handle edge cases: replace inf with 0, NaN with median

4. Update EXPECTED_FEATURES in core/ml_integration.py to match

5. Update prepare_ml_features() in ml_integration.py with same calculations
```

### Prompt 11: Price Action Patterns
```
Add price action pattern features to detect breakout setups and technical patterns.

In train_rolling_ml_20d.py, add these features:

1. Breakout/Consolidation features:
   - Distance_From_52w_High = (current price - 52w high) / 52w high
   - Distance_From_52w_Low = (current price - 52w low) / 52w low
   - Consolidation_Tightness = (20d high - 20d low) / 20d avg price
   - Days_Since_52w_High = trading days since last 52-week high

2. Moving Average features:
   - Price_vs_SMA50 = (close - SMA50) / SMA50
   - Price_vs_SMA200 = (close - SMA200) / SMA200
   - SMA50_vs_SMA200 = (SMA50 - SMA200) / SMA200
   - MA_Slope_20d = slope of 20-day SMA

3. Update FEATURES list with these 8 new features

4. Update EXPECTED_FEATURES in core/ml_integration.py

5. Update prepare_ml_features() in ml_integration.py with same calculations
```

### Prompt 12: Feature Importance & Selection
```
Add feature importance analysis and automatic feature selection to optimize the model.

In train_rolling_ml_20d.py:

1. After training, add feature importance analysis:
   - analyze_feature_importance() function using permutation importance
   - Display top 10 features with visual bars
   - Identify low-value features

2. Add automatic feature selection:
   - select_top_features() with importance threshold
   - remove_correlated_features() with 0.95 correlation threshold

3. Save importance report to 'models/feature_importance_report.txt'
```

---

## 🔮 Phase 4: Future Improvements (Prompts 13+)

### Prompt 13: Hyperparameter Tuning
- Add Optuna for hyperparameter optimization
- Define search space for each model in ensemble
- Use P@20 as optimization objective

### Prompt 14: Walk-Forward Validation
- Implement expanding window validation
- Monthly retraining simulation
- Track performance degradation over time

### Prompt 15: Model Persistence & Versioning
- Save models with metadata (features, params, metrics)
- Version control for models
- A/B testing framework

### Prompt 16: Real-time Monitoring
- Add prediction confidence tracking
- Alert on distribution shift
- Performance dashboard

---

## 📊 Feature Evolution

| Phase | Features Count | Key Additions |
|-------|---------------|---------------|
| Original | 5 | Basic price/volume |
| Prompt 3 | 13 | VCP, momentum, volatility |
| Prompt 8 | 17 | Market regime (4 features) |
| Prompt 9 | 20 | Sector features (3 features) |
| Prompt 10 | 25 | Volume profile (5 features) |
| Prompt 11 | 33 | Price action (8 features) |

---

## 🐛 Bug Fixes Applied

1. **Line 200**: Extra closing parenthesis - Fixed
2. **Line 329**: Extra closing parenthesis - Fixed in Prompt 8.1
3. **Feature mismatch**: Training vs inference feature count - Synced

---

## 📁 Files Modified

- `scripts/train_rolling_ml_20d.py` - Main training script
- `core/ml_integration.py` - Inference module
- `core/sector_mapping.py` - New file for sector data
- `.github/workflows/auto_scan.yml` - GitHub Actions
- `core/config.py` - API key handling

---

*Generated: February 2026*
*System: Stock Scout ML v2.0*
