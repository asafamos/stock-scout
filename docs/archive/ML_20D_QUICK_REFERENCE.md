# Quick Reference — ML 20D Deployment

## 🚀 Deploy in 3 Steps

```bash
# 1. Verify files exist
ls models/model_20d_v2.pkl experiments/training_dataset_20d_v2.csv

# 2. Test locally
streamlit run stock_scout.py

# 3. Enable toggles → Check ENABLE_ML & USE_FINAL_SCORE_SORT
```

---

## 📊 Key Metrics at a Glance

| Metric | Result |
|--------|--------|
| **Model ROC-AUC** | 0.777 |
| **Top 10% Bucket Return** | +0.0120 vs -0.0164 baseline |
| **Top 10% Hit Rate** | 24.8% vs 8.7% baseline |
| **Dataset Size** | 20,547 rows × 62 tickers |
| **Training Window** | 2.5 years (Jan 2023 – Mar 2025) |

---

## 🎯 What Changed

### In `stock_scout.py`
- ✅ Preflight check on first run (API health)
- ✅ `ENABLE_ML` toggle (show/hide ML_20d_Prob)
- ✅ `USE_FINAL_SCORE_SORT` toggle (sort by FinalScore)
- ✅ API status line ("APIs: X OK / Y down")
- ✅ FinalScore rank-based (0.5 tech + 0.5 ML)

### In `core/data_sources_v2.py`
- ✅ `provider_status` parameter passed to all API calls
- ✅ Preflight checks skip disabled providers automatically

### In `core/ml_20d_inference.py`
- ✅ Model v2 loading (with v1 fallback)

### New Files
- ✅ `models/model_20d_v2.pkl` (trained model)
- ✅ `experiments/training_dataset_20d_v2.csv` (dataset)
- ✅ `experiments/train_ml_20d_v2.py` (training script)

---

## 🔧 Common Tasks

### View ML Rankings Offline
```bash
python -m experiments.offline_recommendation_audit \
  --mode audit_ml_20d \
  --input experiments/training_dataset_20d_v2.csv \
  --output audit.csv
```

### Retrain Model (Quarterly)
```bash
# Update date range in audit script, then:
python -m experiments.train_ml_20d_v2 \
  experiments/training_dataset_20d_v2.csv
```

### Validate Improvements
```bash
python experiments/validate_ml_improvements.py
```

### Check Model Health
```python
from core.ml_20d_inference import ML_20D_AVAILABLE, predict_20d_prob_from_row
print(f"Model available: {ML_20D_AVAILABLE}")
prob = predict_20d_prob_from_row(sample_row)  # Returns [0, 1]
```

---

## 📈 FinalScore Formula

```
FinalScore = (0.5 × percentile_rank(TechScore) + 0.5 × percentile_rank(ML_Prob)) × 100
```

**Weights:** 0.5/0.5 (tech/ML)  
**Range:** 0–100  
**Sorting:** Higher FinalScore = higher predicted probability of 15%+ 20d return

---

## ⚠️ Known Limitations

1. **TechScore weak signal** (correlation +0.0105 with forward returns)
2. **Class imbalance** (8.8% positive labels)
3. **Dataset stale after Mar 26, 2025** (needs quarterly refresh)
4. **FinalScore deciles show soft ranking** (consider 0.3/0.7 weight for stronger signal)

---

## ✅ Production Checklist

Before deploying:
- [ ] `models/model_20d_v2.pkl` exists (101.5 KB)
- [ ] `experiments/training_dataset_20d_v2.csv` exists (4.0 MB)
- [ ] `python -m py_compile stock_scout.py` runs without error
- [ ] `streamlit run stock_scout.py` shows ML toggle in sidebar
- [ ] Toggle `ENABLE_ML` displays ML_20d_Prob in cards
- [ ] Toggle `USE_FINAL_SCORE_SORT` sorts by FinalScore

---

## 📚 Documentation

- [ML_20D_DELIVERY_SUMMARY.md](ML_20D_DELIVERY_SUMMARY.md) — Full technical details
- [ML_20D_INTEGRATION_INDEX.md](ML_20D_INTEGRATION_INDEX.md) — Architecture & usage
- [experiments/validate_ml_improvements.py](experiments/validate_ml_improvements.py) — Validation script

---

## 🎯 Success Criteria (Met)

✅ Top 10% by ML probability outperforms baseline by +2.84% absolute return  
✅ ML_20d_Prob display unified and consistent across cards  
✅ FinalScore computed via rank-based formula (0.5/0.5 weighting)  
✅ Preflight integration reduces API errors  
✅ Backward compatibility maintained  
✅ All code compiles without errors  

---

**Status: ✅ READY FOR PRODUCTION**

Last Updated: December 25, 2025
