# ML 20d Model v3 - Quick Reference

## Overview
Major upgrade to the 20d forward return prediction model with enriched contextual features.

**Model Type**: GradientBoostingClassifier (200 estimators, depth=5)  
**Training Label**: `Label_20d` = 1 if `Forward_Return_20d >= 0.15` (15%+), else 0  
**Feature Count**: 26 (up from 6 in v1)  
**Target Performance**: Top decile hit rate >40% (vs 11.8% baseline)

---

## Feature Categories

### 1. Original Technical Indicators (6 features)
- `TechScore_20d`: Legacy composite technical score
- `RSI`: Relative Strength Index (14-period)
- `ATR_Pct`: Average True Range as % of price
- `RR`: Reward/Risk ratio
- `MomCons`: Momentum consistency
- `VolSurge`: Volume surge indicator

### 2. Multi-Period Returns (5 features)
- `Return_5d`, `Return_10d`, `Return_20d`: Short-term momentum
- `Return_60d`, `Return_120d`: Medium-term trend strength
- **Purpose**: Capture momentum across multiple timeframes

### 3. Sequential Pattern Features (5 features)
- `UpStreak_Days`: Consecutive up days (0-20+)
- `DownStreak_Days`: Consecutive down days (0-20+)
- `PullbackFromHigh_20d`: Distance from 20-day high (0-1)
- `DistanceFromLow_20d`: Distance from 20-day low (0-1)
- `Range_Pct`: Current price position in 20d range (0-1)
- **Purpose**: Identify consolidations, breakouts, and pullback entries

### 4. Relative Strength vs Market (2 features)
- `RS_SPY_20d`: 20-day return vs SPY (outperformance)
- `RS_SPY_60d`: 60-day return vs SPY (sustained leadership)
- **Purpose**: Prefer relative strength leaders in all market regimes

### 5. Volatility Context (4 features)
- `ATR_Pct_percentile`: Volatility rank within universe (0-1)
- `Vol_Class`: Volatility classification (0=Low, 1=Medium, 2=High, 3=Extreme)
- `Vol_SweetSpot_Flag`: 1 if ATR in 50th-75th percentile (elevated but controlled)
- `Vol_Extreme_Flag`: 1 if ATR >90th percentile (warning flag)
- **Purpose**: Context-aware volatility filtering

### 6. Big Winner Signal (2 features)
- `BigWinnerScore_20d`: Composite breakout score
- `BigWinnerFlag_20d`: Binary flag for high-conviction setups
- **Purpose**: Tactical breakout identification

### 7. Advanced Scoring (2 features)
- `TechScore_20d_v2`: Enhanced technical score (4-component formula)
- `HybridFinalScore_20d`: 0.20 Tech + 0.80 ML_rank (per-date normalized)

---

## Training Pipeline

### Step 1: Generate Dataset
```bash
python experiments/offline_recommendation_audit.py \
    --mode dataset \
    --start 2023-01-01 \
    --end 2025-01-15 \
    --universe-limit 100 \
    --output data/training_dataset_20d_v3.csv \
    --drop-neutral
```

**Output**: CSV with ~10K-50K rows (depending on universe size and date range)

### Step 2: Train Model
```bash
python experiments/train_ml_20d.py \
    --input data/training_dataset_20d_v3.csv \
    --output-model models/model_20d_v3.pkl \
    --min-return 0.15
```

**Output**: 
- `models/model_20d_v3.pkl` (model bundle with scaler)
- `models/model_20d_v3_feature_importance.csv` (feature rankings)

### Step 3: Validate
```bash
python experiments/validate_ml_improvements.py \
    --input data/training_dataset_20d_v3.csv \
    --model-path models/model_20d_v3.pkl \
    > reports/ml_20d_v3_validation_report.txt
```

**Output**: Comprehensive validation report with:
- Dataset statistics
- Model performance (ROC-AUC, PR-AUC)
- Decile analysis (top 10% hit rate)
- Feature importance breakdown

### Quick Pipeline (All-in-One)
```bash
./scripts/train_ml_20d_v3.sh
```

Set environment variables for customization:
```bash
export START_DATE=2023-06-01
export END_DATE=2025-01-31
export UNIVERSE_LIMIT=150
./scripts/train_ml_20d_v3.sh
```

---

## Model Performance Targets

### Phase 1 (Hybrid Scoring) - ACHIEVED ✅
- **Top Decile Hit Rate**: 40.4% (vs 11.8% baseline)
- **Improvement**: +28.6pp over random selection
- **Test Period**: 2023 Q1 (2,088 rows)

### Phase 2 (ML v3 with Enriched Features) - TARGET 🎯
- **Top Decile Hit Rate**: >45% (stretch: 50%)
- **ROC-AUC**: >0.80 (current: 0.777)
- **Feature Diversity**: 26 features with balanced category contributions
- **Robustness**: Stable performance across market regimes

---

## Integration with stock_scout.py

### Option 1: Environment Variable (Recommended)
```bash
export ML_MODEL_PATH=/workspaces/stock-scout-2/models/model_20d_v3.pkl
streamlit run stock_scout.py
```

### Option 2: Update config.py
Edit `core/config.py`:
```python
MODEL_20D_PATH = "models/model_20d_v3.pkl"
```

### Scoring Modes (UI Selector)
1. **Hybrid (Default)**: 0.20 TechScore_20d_v2 + 0.80 ML_rank_20d
2. **ML-only**: 100% ML_20d_Prob (pure ML predictions)
3. **Legacy**: 0.50 TechScore_20d + 0.50 ML_20d_Prob (backward compatible)

---

## Feature Engineering Functions

All implemented in `core/ml_features_v3.py`:

1. `compute_multi_period_returns(df)` → Return_5d/10d/20d/60d/120d
2. `compute_relative_strength_features(row, spy_context)` → RS_SPY_20d/60d
3. `compute_volatility_context_features(atr_pct, universe_series)` → ATR_Pct_percentile, Vol_Class, flags
4. `compute_sequential_pattern_features(df)` → Streaks, pullbacks, range position
5. `compute_earnings_proximity_features(ticker, date, calendar)` → DaysToNextEarnings (future enhancement)

**Market Context** (SPY/VIX regime classification):
- `build_market_context_table(start, end)` in `core/unified_logic.py`
- Provides: SPY_Return_20d, VIX_Level, Market_Regime (TREND_UP, SIDEWAYS, CORRECTION, PANIC)

---

## Validation Checklist

Before deploying v3 model:

- [ ] Dataset generated with all 26 features
- [ ] No missing values in critical features (RS_SPY, Vol_Class)
- [ ] Model training completes without errors
- [ ] Feature importance saved and reviewed
- [ ] Top decile hit rate >40% on test set
- [ ] ROC-AUC >0.75 on holdout period
- [ ] Feature categories show balanced contributions
- [ ] Hybrid score computed correctly in stock_scout.py
- [ ] UI scoring mode selector works for all 3 modes
- [ ] Backward compatibility maintained (v2 model still works)

---

## Troubleshooting

### Issue: Missing features in dataset
**Solution**: Check `offline_recommendation_audit.py` row_dict construction. Ensure all v3 features are added with safe defaults (np.nan or 0).

### Issue: SPY data fetch fails
**Solution**: Fallback to local defaults for RS_SPY features. Set RS_SPY_20d/60d = 0 if market data unavailable.

### Issue: Model training fails with NaN
**Solution**: 
1. Check for inf/NaN in Return_120d (clip to [-0.6, 3.0])
2. Verify volatility percentiles computed per-date
3. Use `df.dropna(subset=feature_cols)` before training

### Issue: Feature importance shows only old features
**Solution**: Retrain model from scratch. Delete old model_20d_v2.pkl and regenerate dataset with `--drop-neutral` flag.

---

## Next Steps (Future Enhancements)

1. **Market Regime Features**: Add SPY_Regime flags (TREND_UP, CORRECTION, etc.) to features
2. **Sector Rotation**: Add RS_Sector features (requires sector mapping)
3. **Earnings Proximity**: Add DaysToNextEarnings and In_Earnings_Window_Flag
4. **Ensemble Models**: Combine GradientBoosting + LightGBM + CatBoost
5. **Feature Selection**: Use SHAP values to prune redundant features
6. **Hyperparameter Tuning**: Grid search on n_estimators, max_depth, learning_rate
7. **Walk-Forward Validation**: Quarterly retraining with expanding window

---

## File Locations

**Core Modules**:
- `core/unified_logic.py` - Technical indicators, TechScore_20d_v2, market context
- `core/ml_features_v3.py` - Feature engineering functions (NEW)
- `core/ml_20d_inference.py` - Model loading and prediction
- `core/data_sources_v2.py` - Multi-source data fetching (SPY/VIX)

**Experiment Scripts**:
- `experiments/offline_recommendation_audit.py` - Dataset generation
- `experiments/train_ml_20d.py` - Model training with v3 features
- `experiments/validate_ml_improvements.py` - Comprehensive validation
- `scripts/train_ml_20d_v3.sh` - End-to-end pipeline script

**Models**:
- `models/model_20d_v2.pkl` - Phase 1 baseline (6 features, LogisticRegression)
- `models/model_20d_v3.pkl` - Phase 2 target (26 features, GradientBoosting)
- `models/model_20d_v3_feature_importance.csv` - Feature rankings

**Data**:
- `data/training_dataset_20d_v2.csv` - Phase 1 dataset
- `data/training_dataset_20d_v3.csv` - Phase 2 dataset with enriched features

---

## Quick Commands

```bash
# Generate small test dataset (fast)
python experiments/offline_recommendation_audit.py --mode dataset --start 2024-01-01 --end 2024-03-31 --universe-limit 30 --output data/test_20d_v3.csv --drop-neutral

# Train on test dataset
python experiments/train_ml_20d.py --input data/test_20d_v3.csv --output-model models/test_20d_v3.pkl

# Validate
python experiments/validate_ml_improvements.py --input data/test_20d_v3.csv --model-path models/test_20d_v3.pkl

# Full production pipeline
export UNIVERSE_LIMIT=200
./scripts/train_ml_20d_v3.sh
```

---

**Last Updated**: 2025-01-15  
**Status**: Phase 2 implementation complete, ready for testing  
**Next Milestone**: Full dataset training and production deployment
