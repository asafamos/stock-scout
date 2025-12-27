# 📋 Migration Status: Unified API for Stock Scout

## ✅ Completed (סיים)

### Phase 1: Wrapper Layer
- ✅ Created core/scoring/__init__.py - exports all scoring functions
- ✅ Created core/filters/__init__.py - exports all filters
- ✅ Created core/data/__init__.py - exports data functions
- ✅ Created core/allocation/__init__.py - exports allocation
- ✅ Created core/classification/__init__.py - exports classification
- ✅ Updated stock_scout.py to use new unified imports
- ✅ Created API_UNIFIED.md documentation

### Status: **API is LIVE and backward compatible**

---

## 🚀 What's New

The new unified API makes it clear where each piece of logic lives:

```python
# OLD (scattered):
from core.unified_logic import build_technical_indicators
from core.ml_20d_inference import predict_20d_prob_from_row
from core.scoring_pipeline_20d import compute_final_scores_20d
from core.v2_risk_engine import score_ticker_v2_enhanced

# NEW (organized):
from core.scoring import (
    build_technical_indicators,
    predict_20d_prob_from_row,
    compute_final_scores_20d,
    score_ticker_v2_enhanced,
)
```

---

## 📁 File Structure

```
core/
├── scoring/           ← ALL scoring (technical, ML, fundamental, risk)
│   └── __init__.py    ← wrapper layer
├── filters/           ← ALL filters (technical, advanced, risk)
│   └── __init__.py    ← wrapper layer
├── data/              ← ALL data (indicators, prices, fundamentals)
│   └── __init__.py    ← wrapper layer
├── allocation/        ← Budget allocation & sizing
│   └── __init__.py    ← wrapper layer
├── classification/    ← Core vs Speculative classification
│   └── __init__.py    ← wrapper layer
└── [old files still here for backward compatibility]
```

---

## 🔗 Import Patterns

### Pattern 1: Scoring
```python
from core.scoring import (
    # Technical
    build_technical_indicators,
    compute_tech_score_20d_v2,
    # ML
    predict_20d_prob_from_row,
    apply_live_v3_adjustments,
    # Final
    compute_final_scores_20d,
    # Fundamental
    compute_fundamental_score_with_breakdown,
    # Risk V2
    score_ticker_v2_enhanced,
    calculate_reliability_v2,
)
```

### Pattern 2: Filters
```python
from core.filters import (
    apply_technical_filters,
    compute_advanced_score,
    should_reject_ticker,
    fetch_benchmark_data,
)
```

### Pattern 3: Data
```python
from core.data import (
    build_technical_indicators,
    fetch_price_multi_source,
    aggregate_fundamentals,
    fetch_fundamentals_batch,
)
```

### Pattern 4: Allocation
```python
from core.allocation import (
    allocate_budget,
    _normalize_weights,
)
```

### Pattern 5: Classification
```python
from core.classification import (
    apply_classification,
    filter_core_recommendations,
)
```

---

## 🎯 What This Means

### For You:
1. **Easier to find things** - all scoring in core/scoring, all filters in core/filters, etc.
2. **Clearer dependencies** - you can see what imports what
3. **Better for testing** - import only what you need
4. **Backward compatible** - old imports still work
5. **Ready for future** - easy to add new things

### For New Developers:
1. **No hunting** - look in docs/API_UNIFIED.md first
2. **Clear flow** - the wrapper shows the scoring pipeline
3. **Examples** - each module has docstrings
4. **One source of truth** - each function exported from one place

---

## 📊 Score Versions Used

| Component | Version | Location |
|-----------|---------|----------|
| ML Features | **V3** | core/ml_features_v3.py |
| ML Inference | V3 (live_v3) | core/ml_20d_inference.py |
| Technical Scoring | **V2** | core/unified_logic.py |
| Risk Engine | **V2** | core/v2_risk_engine.py |
| Data Sources | **V2** | core/data_sources_v2.py |

---

## ⏭️ Next Steps (Optional)

### Option A: Full Migration (No rush)
Update other files to use new imports:
- auto_scan_runner.py
- batch_scan.py
- experiment scripts
- etc.

### Option B: Gradual
Leave old imports alone, new files use new API.

### Option C: Cleanup (Later)
- Archive old imports
- Remove deprecated functions
- Consolidate duplicates

---

## 🧪 Testing

The new API is **production ready**:
- ✅ No breaking changes
- ✅ Backward compatible
- ✅ All imports work
- ✅ All functions accessible

---

## 📚 Documentation

Read these files for full understanding:
1. **docs/API_UNIFIED.md** - API overview and patterns
2. **docs/SCORING_FLOW_MAP.md** - Detailed scoring flow
3. **docs/REFACTORING_PLAN.md** - Original refactoring plan
4. **core/scoring/__init__.py** - Source code with comments
5. **core/filters/__init__.py** - Source code with comments
