# 🎯 API החדש המאוחד - Stock Scout V2 Architecture

## סטטוס: ✅ **LIVE**

עכשיו יש לך API אחיד וברור לכל הלוגיקה של Stock Scout.

---

## 📦 המודולים החדשים

### 1. **core.scoring** — כל ה-Scoring
```python
from core.scoring import (
    # Technical Scoring V2
    build_technical_indicators,      # אינדיקטורים טכניים
    compute_tech_score_20d_v2,       # ציון טכני (V2)
    
    # ML Scoring V3
    predict_20d_prob_from_row,       # ML prediction
    apply_live_v3_adjustments,       # V3 adjustments
    
    # Final Scores
    compute_final_scores_20d,        # מיזוג טכני + ML
    apply_20d_sorting,               # sorting policies
    
    # Fundamental
    compute_fundamental_score_with_breakdown,
    
    # Risk V2
    score_ticker_v2_enhanced,        # risk gates + allocation
    calculate_reliability_v2,        # reliability scoring
    
    # Legacy (backward compatibility)
    evaluate_rr_unified,
    compute_overall_score,
)
```

### 2. **core.filters** — כל ה-Filters
```python
from core.filters import (
    apply_technical_filters,    # 3 tiers: strict/normal/relaxed
    compute_advanced_score,     # RS, momentum, RR filters
    should_reject_ticker,       # dynamic thresholds
    fetch_benchmark_data,       # SPY/QQQ for comparisons
)
```

### 3. **core.data** — כל הנתונים
```python
from core.data import (
    build_technical_indicators,        # technical indicators
    fetch_price_multi_source,          # fetch prices from multiple sources
    aggregate_price,                   # combine prices
    aggregate_fundamentals,            # combine fundamentals
    fetch_fundamentals_batch,          # fetch from APIs
)
```

### 4. **core.allocation** — Budget & Sizing
```python
from core.allocation import (
    allocate_budget,        # split budget to positions
    _normalize_weights,     # normalize position weights
)
```

### 5. **core.classification** — Classification
```python
from core.classification import (
    apply_classification,         # Core vs Speculative
    filter_core_recommendations,  # filter by classification
)
```

---

## 🔄 Flow: איך הכל עובד ביחד

```python
# 1. Fetch Data
prices = fetch_price_multi_source(tickers)
df_indicators = build_technical_indicators(prices)

# 2. Technical Filter
if not apply_technical_filters(df_indicators.iloc[-1]):
    continue  # Skip this ticker

# 3. Technical Score V2
tech_score = compute_tech_score_20d_v2(df_indicators.iloc[-1])

# 4. ML Prediction V3
ml_prob = predict_20d_prob_from_row(df_indicators.iloc[-1])
ml_prob_v3 = apply_live_v3_adjustments(results)

# 5. Final Scores
results = compute_final_scores_20d(results, include_ml=True)
# → TechScore_20d, ML_20d_Prob, FinalScore_20d

# 6. Advanced Filters
signals = compute_advanced_score(ticker, df, benchmark)
catastrophic, reason = should_reject_ticker(signals)

# 7. Fundamentals
fund_score = compute_fundamental_score_with_breakdown(row)

# 8. Risk Engine V2
res = score_ticker_v2_enhanced(ticker, row, budget=5000, min_pos=500)
# → conviction_v2_final, reliability_v2, buy_amount_v2

# 9. Classification
result = apply_classification(result)
# → Risk_Level, Data_Quality, Classification

# 10. Allocation
result = allocate_budget(result, budget=5000, min_position=500)
```

---

## 🗂️ שינויים שנעשו

### Before (Scattered):
```python
from core.unified_logic import build_technical_indicators
from core.ml_20d_inference import predict_20d_prob_from_row
from core.scoring_pipeline_20d import compute_final_scores_20d
from core.v2_risk_engine import score_ticker_v2_enhanced
from core.scoring_engine import evaluate_rr_unified
from core.data_sources_v2 import fetch_fundamentals_batch
from core.portfolio import allocate_budget
from advanced_filters import compute_advanced_score
from core.classification import apply_classification
```

### After (Organized):
```python
from core.scoring import (
    build_technical_indicators,
    predict_20d_prob_from_row,
    compute_final_scores_20d,
    score_ticker_v2_enhanced,
    evaluate_rr_unified,
    compute_fundamental_score_with_breakdown,
)
from core.filters import compute_advanced_score
from core.data import fetch_fundamentals_batch
from core.allocation import allocate_budget
from core.classification import apply_classification
```

---

## 📍 איפה מצאת כל דבר

| What | Module | Function |
|------|--------|----------|
| Technical Indicators | `core.scoring` | `build_technical_indicators()` |
| Technical Score V2 | `core.scoring` | `compute_tech_score_20d_v2()` |
| ML Prediction V3 | `core.scoring` | `predict_20d_prob_from_row()` |
| ML V3 Adjustments | `core.scoring` | `apply_live_v3_adjustments()` |
| Final Score 20d | `core.scoring` | `compute_final_scores_20d()` |
| Fundamental Score | `core.scoring` | `compute_fundamental_score_with_breakdown()` |
| Risk Gates V2 | `core.scoring` | `score_ticker_v2_enhanced()` |
| Reliability V2 | `core.scoring` | `calculate_reliability_v2()` |
| Technical Filters | `core.filters` | `apply_technical_filters()` |
| Advanced Filters | `core.filters` | `compute_advanced_score()` |
| Dynamic Thresholds | `core.filters` | `should_reject_ticker()` |
| Budget Allocation | `core.allocation` | `allocate_budget()` |
| Classification | `core.classification` | `apply_classification()` |
| Price Fetching | `core.data` | `fetch_price_multi_source()` |
| Fundamentals Fetching | `core.data` | `fetch_fundamentals_batch()` |

---

## 🚀 הצעד הבא

### Option A: Migration מלא
עדכן את כל הקבצים להשתמש בwrappers החדשים:
- `auto_scan_runner.py`
- `batch_scan.py`
- `core/pipeline_runner.py`
- וכו'

### Option B: Cleanup
1. הסר קבצים ישנים/מיותרים
2. בדוק שהכל עובד
3. תיעדך

### Option C: שמור את זה ככה
- יש לך API אחיד
- הקוד הישן עדיין עובד
- אפשר לעדכן הדרגתי

---

## ✅ תוצאה

✨ **אתה עכשיו יש מבנה נקי וברור שמקל על:**
- הבנה מהי הלוגיקה
- מציאת באגים
- הוספת features
- תחזוקה בעתיד

🎯 **ה-API הוא:**
- **ברור**: כל דבר במקום הנכון
- **מלא**: כל הפונקציות נגישות
- **בטוח**: הקוד הישן עדיין עובד
- **חדיש**: קל להוסיף דברים חדשים
