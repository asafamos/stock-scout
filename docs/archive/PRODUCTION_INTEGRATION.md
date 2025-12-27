# Production Integration: XGBoost Recommendation Scoring

## Quick Start

### 1. Use Trained Model in stock_scout.py

Add after line ~1930 (after Core/Speculative classification):

```python
# Load XGBoost scoring model
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'model_xgboost_5d.pkl'
XGBOOST_MODEL = None

if MODEL_PATH.exists():
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        XGBOOST_MODEL = model_data['model']
        XGBOOST_FEATURES = model_data['feature_names']

def score_with_xgboost(row: pd.Series) -> float:
    """Score stock with XGBoost model. Returns probability 0-1."""
    if XGBOOST_MODEL is None:
        return 0.5  # Fallback if model not loaded
    
    # Base features from stock_scout indicators
    features = {
        'RSI': row.get('RSI', 50),
        'ATR_Pct': row.get('ATR', 0) / max(row.get('Close', 1), 1),
        'Overext': (row.get('Close', 0) / row.get('MA50', 1) - 1) if row.get('MA50', 0) > 0 else 0,
        'RR': row.get('Reward_Risk', 1),
        'MomCons': row.get('MomConsistency', 0.5),
        'VolSurge': row.get('Volume', 1) / max(row.get('AvgVol', 1), 1)
    }
    
    # Engineer features (must match train_recommender.py)
    features['RR_MomCons'] = features['RR'] * features['MomCons']
    features['RSI_Neutral'] = abs(features['RSI'] - 50)
    features['Risk_Score'] = abs(features['Overext']) + features['ATR_Pct']
    features['Vol_Mom'] = features['VolSurge'] * features['MomCons']
    
    # Build feature vector in model's expected order
    X = pd.DataFrame([features])[XGBOOST_FEATURES]
    X = X.fillna(X.median())
    
    return float(XGBOOST_MODEL.predict_proba(X.values)[0][1])

# Apply scoring to final recommendations
recommendations['ML_Probability'] = recommendations.apply(score_with_xgboost, axis=1)

# Add confidence tier
def assign_tier(prob: float) -> str:
    if prob >= 0.50:
        return "🟢 High"
    elif prob >= 0.30:
        return "🟡 Medium"
    else:
        return "🔴 Low"

recommendations['Confidence'] = recommendations['ML_Probability'].apply(assign_tier)

# Sort by probability (highest first)
recommendations = recommendations.sort_values('ML_Probability', ascending=False)
```

### 2. Update UI to Display Confidence

In the recommendation card display (around line 2500):

```python
# Add to card HTML
confidence_badge = f"""
<div style="display: inline-block; padding: 4px 8px; border-radius: 4px; 
     background: {'#4CAF50' if tier.startswith('🟢') else '#FFC107' if tier.startswith('🟡') else '#FF5722'};
     color: white; font-weight: bold; font-size: 0.9em;">
    {tier} ({prob*100:.1f}%)
</div>
"""
```

### 3. Add Model Performance Dashboard (Optional)

Create new Streamlit page `pages/3_Model_Performance.py`:

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("🤖 XGBoost Model Performance")

# Load comparison data
comparison = pd.read_csv('MODEL_COMPARISON.md', sep='|', skiprows=7, nrows=2)
st.dataframe(comparison, use_container_width=True)

# SHAP plots
col1, col2 = st.columns(2)
with col1:
    st.subheader("Feature Importance")
    st.image('shap_importance.png')
with col2:
    st.subheader("Feature Impact Distribution")
    st.image('shap_summary.png')

# Time-test validation
st.subheader("Historical Validation")
time_test = pd.read_csv('time_test_20251115_112540.csv')
st.dataframe(time_test, use_container_width=True)

# Calibration curve
calib = pd.read_csv('calibration_curve.csv')
fig = px.line(calib, x='prob_pred', y='prob_true', 
              title='Probability Calibration',
              labels={'prob_pred': 'Predicted Probability', 'prob_true': 'Observed Frequency'})
fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration', 
                line=dict(dash='dash', color='gray'))
st.plotly_chart(fig, use_container_width=True)
```

## Maintenance

### Retraining Schedule
Run monthly backtest and retrain:

```bash
# Backtest last 12 months
python backtest_recommendations.py \
    --use-finnhub --limit 500 \
    --start $(date -d '12 months ago' +%Y-%m-%d) \
    --end $(date +%Y-%m-%d) \
    --horizons 5,10,20

# Train new model
python train_recommender.py \
    --signals backtest_signals_$(date +%Y%m%d).csv \
    --horizon 5 \
    --model xgboost \
    --cv \
    --out model_xgboost_5d_$(date +%Y%m%d).pkl

# Validate on recent moves
python time_test_validation.py \
    --model model_xgboost_5d_$(date +%Y%m%d).pkl \
    --cases recent_movers.csv

# If AUC improved, replace production model
mv model_xgboost_5d_$(date +%Y%m%d).pkl model_xgboost_5d.pkl
git add model_xgboost_5d.pkl
git commit -m "chore: update production model (AUC: $NEW_AUC)"
git push
```

### Monitoring
- Track weekly hit rate: % of High confidence picks with positive 5d return
- Alert if hit rate drops below 45% for 2 consecutive weeks
- Compare model predictions vs actual outcomes in `performance_log.csv`

### A/B Testing
Run parallel scoring with old/new models:

```python
# Score with both
recommendations['Prob_v1'] = recommendations.apply(score_v1, axis=1)
recommendations['Prob_v2'] = recommendations.apply(score_v2, axis=1)

# Track which performs better over next 30 days
# Promote winner to production
```

## Configuration

Add to `core/config.py`:

```python
ML_CONFIG = {
    'enabled': True,
    'model_path': 'model_xgboost_5d.pkl',
    'confidence_thresholds': {
        'high': 0.50,
        'medium': 0.30
    },
    'min_samples_for_high': 3,  # At least 3 high-confidence picks
    'fallback_on_error': True,  # Use technical filters if model fails
    'shap_explain_top_n': 5  # Generate SHAP for top 5 picks
}
```

## Testing

```bash
# Unit test feature engineering
pytest tests/test_ml_integration.py -v

# Integration test on historical data
python test_ml_integration.py --backtest

# Smoke test on live data
python stock_scout.py --test-ml
```

## Troubleshooting

### Model not loading
- Check `model_xgboost_5d.pkl` exists in project root
- Verify pickle protocol compatibility (use `protocol=4` for Python 3.11)

### Probability always 0.5
- Verify feature names match training (check `model_data['feature_names']`)
- Ensure engineered features computed correctly (RR_MomCons, etc.)

### Low confidence for all stocks
- Check if input features are normalized correctly
- Compare feature distributions: `df[XGBOOST_FEATURES].describe()` vs training data
- May need to retrain on more recent data

## Performance Benchmarks

| Metric | Baseline (Technical Only) | With XGBoost | Improvement |
|--------|--------------------------|--------------|-------------|
| 5d Hit Rate | 47.3% | **TBD** | +X% |
| Excess Return | +0.94% | **TBD** | +X% |
| False Positive Rate | 52.7% | **TBD** | -X% |
| Avg Confidence | N/A | **TBD** | N/A |

*Fill in after 1 month of production data*
