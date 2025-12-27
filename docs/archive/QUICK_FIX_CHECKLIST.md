# 🚀 Quick Fix Guide - Stock Scout
**Priority Actions to Fix Right Now (Est. Time: 2 hours)**

---

## CRITICAL FIX #1: Streamlit Deprecation ⏰ 30 MIN

**Issue:** `use_container_width` will break on Jan 1, 2026

### Step 1: Update stock_scout.py (6 instances)

```bash
# Before running: check current instances
grep -n "use_container_width" stock_scout.py
```

**Replace in stock_scout.py:**

**Line 543:**
```python
# OLD:
st.dataframe(styled, use_container_width=True, hide_index=True)

# NEW:
st.dataframe(styled, width='stretch', hide_index=True)
```

**Line 2339:**
```python
# OLD:
run_scan = st.button("🚀 הרץ סריקה", use_container_width=True, type="primary")

# NEW:
run_scan = st.button("🚀 הרץ סריקה", width='stretch', type="primary")
```

**Lines 3262, 4468, 4507, 4519:** Same pattern — replace `use_container_width=True` with `width='stretch'`

---

## CRITICAL FIX #2: Type Hints for Better IDE Support ⏰ 1 HR

**Problem:** Scoring functions lack type hints → less IDE autocomplete

### Step 1: Update return types in core/unified_logic.py

Add return type hints to these functions:

```python
# Line ~535
def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """..."""

# Line ~672  
def apply_technical_filters(row: pd.Series, strict: bool = True, relaxed: bool = False) -> bool:
    """..."""

# Line ~774
def score_with_ml_model(row: pd.Series, model_data: Optional[Dict] = None) -> float:
    """..."""

# Line ~839
def compute_technical_score(row: pd.Series) -> float:
    """..."""

# Line ~869
def compute_final_score(row: pd.Series) -> float:
    """..."""
```

### Step 2: Update core/scoring_engine.py

```python
# Add full type hints to main scoring functions
def evaluate_rr_unified(rr_ratio: Optional[float]) -> Tuple[float, float, str]:
    """..."""

def compute_overall_score(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    """..."""
```

---

## OPTIONAL CLEANUP: Documentation Archive ⏰ 1 HR

**Problem:** 50+ .md files causing clutter

### Step 1: Create archive directory
```bash
mkdir -p docs/archive
```

### Step 2: Move old documentation
```bash
# Keep only essential docs
git mv DEPLOYMENT.md docs/archive/
git mv UI_CLEANUP_*.md docs/archive/
git mv ML_20D_*.md docs/archive/
git mv IMPLEMENTATION_*.md docs/archive/
git mv INTEGRATION_*.md docs/archive/

# Keep these:
# - README.md
# - ARCHITECTURE.md (if exists)
# - requirements.txt (already tracked)
```

### Step 3: Update .gitignore to ignore archive
```bash
echo "docs/archive/" >> .gitignore
```

---

## VALIDATION TESTS ⏰ 10 MIN

After applying fixes, run:

```bash
# Syntax check
python3 -m py_compile stock_scout.py core/*.py

# Full test suite
python3 -m pytest tests/ -v

# Check for remaining deprecations
grep -r "use_container_width" . --include="*.py" | grep -v "\.venv"
```

**Expected result:** 
- ✅ 0 syntax errors
- ✅ 182/182 tests pass
- ✅ 0 use_container_width instances in user code

---

## GIT COMMIT CHECKLIST

```bash
# 1. Check status
git status

# 2. Stage changes
git add stock_scout.py core/*.py

# 3. Commit with message
git commit -m "fix: replace deprecated use_container_width with width='stretch' for Streamlit 1.40+

- Replace 6 instances of use_container_width=True with width='stretch'
- Fix pct_change() FutureWarning with fill_method=None
- Add type hints to scoring functions
- Remove duplicate imports in unified_logic.py
- Clean up orphaned ml_info variable

Fixes:
- Prevents Streamlit breakage on Jan 1, 2026
- Suppresses pandas 2.1+ warnings
- Improves IDE autocomplete support"

# 4. Push
git push origin main
```

---

## POST-DEPLOYMENT CHECKLIST

- [ ] All tests pass (182/182)
- [ ] No use_container_width instances in production code
- [ ] No FutureWarnings on startup
- [ ] Streamlit app runs without errors
- [ ] Hebrew UI renders correctly
- [ ] CSV export works
- [ ] All 10 data providers properly configured

---

## OPTIONAL ENHANCEMENTS (For Next Sprint)

### 1. Add Integration Tests (2 hours)

Create `tests/test_integration.py`:

```python
import pytest
import pandas as pd
from core.unified_logic import build_technical_indicators, apply_technical_filters
from core.data_sources_v2 import aggregate_fundamentals

def test_full_pipeline_smoke():
    """Test complete pipeline with sample data."""
    # Create synthetic OHLCV data
    dates = pd.date_range('2024-01-01', periods=100)
    df = pd.DataFrame({
        'High': [100 + i*0.5 for i in range(100)],
        'Low': [98 + i*0.5 for i in range(100)],
        'Close': [99 + i*0.5 for i in range(100)],
        'Volume': [1000000]*100,
    }, index=dates)
    
    # Test indicator computation
    indicators = build_technical_indicators(df)
    assert len(indicators) == 100
    assert 'RSI' in indicators.columns
    assert 'ATR_Pct' in indicators.columns
    
    # Test filtering
    sample_row = indicators.iloc[-1]
    passes_filter = apply_technical_filters(sample_row, strict=True)
    assert isinstance(passes_filter, bool)
```

### 2. Add ML Feature Validation (1 hour)

```python
# core/ml_validation.py
def validate_ml_features(row: pd.Series) -> bool:
    """Ensure row has all required ML features."""
    required = [
        'RSI', 'ATR_Pct', 'RR', 'MomCons', 'VolSurge',
        'Return_1m', 'Return_3m', 'Return_6m',
        'Overext', 'MA50_Slope'
    ]
    return all(col in row.index for col in required)
```

### 3. Add Performance Monitoring (2 hours)

```python
# Track recommendation quality
def log_recommendation_outcome(ticker, predicted_target, actual_price_5d):
    """Log prediction accuracy for model calibration."""
    accuracy = abs(actual_price_5d - predicted_target) / predicted_target
    # Save to CSV for analysis
```

---

## COMMON ISSUES & FIXES

### Issue: Tests fail with "missing xgboost model"
**Solution:** Download trained model or train locally:
```bash
python3 train_recommender.py
```

### Issue: Streamlit app shows "missing API key"
**Solution:** Create `.env` file with required keys:
```
FINNHUB_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
```

### Issue: Data provider timeout
**Solution:** Increase timeout in `core/data_sources_v2.py`:
```python
timeout = 20  # Increase from 10 to 20 seconds
```

---

## USEFUL COMMANDS

```bash
# Run app locally
streamlit run stock_scout.py

# Run tests with coverage
pytest tests/ --cov=core --cov-report=html

# Check for security issues
python3 -m bandit stock_scout.py core/*.py

# Format code
black stock_scout.py core/*.py

# Type-check
python3 -m mypy core/ --ignore-missing-imports

# Profile memory
python3 -m memory_profiler stock_scout.py

# Check code complexity
python3 -m radon cc stock_scout.py -a
```

---

**Total Estimated Time:** 2-3 hours for fixes + optional enhancements  
**Risk Level:** LOW (all changes backward-compatible)  
**Ready to Deploy:** ✅ YES

Good luck! 🚀
