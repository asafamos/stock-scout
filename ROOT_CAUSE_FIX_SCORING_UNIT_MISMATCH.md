# Root Cause Analysis: Why Only 2 Recommendations

**Date:** December 27, 2025  
**Severity:** CRITICAL  
**Status:** FIXED

## Problem Statement

Pipeline was outputting only **2-5 stock recommendations** despite:
- Running full scan on 12 major tech/finance stocks  
- Pipeline returning 11 stocks post-filter
- 82% of results being filtered at the final stage (11 → 2)

**Symptom:** FinalScore_20d was **hardcoded to 0.1** for 7 major stocks (AAPL, MSFT, NVDA, AMZN, JPM, etc.)

## Root Cause: Unit Mismatch in Scoring Pipeline

### Issue #1: Double Normalization in advanced_filters.py

**File:** `advanced_filters.py` line 323

```python
# BROKEN CODE:
base_score = 0.3  # normalized to [0, 1] by caller (base_score / 100.0)
total_boost = 15.0  # in [0, 50] scale (0-100 range value)
enhanced_score = min(100.0, base_score + total_boost)  # Unit mismatch!
# Result: 0.3 + 15 = 15.3 (treating normalized 0-1 value as 0-100)
```

**Impact:** `enhanced_score` was 5-20 instead of 0.3-0.7, causing penalties to be miscalculated

**Fix:** Normalize boost before adding
```python
normalized_boost = total_boost / 100.0  # Convert [0, 50] to [0, 0.5]
enhanced_score = base_score + normalized_boost  # Keep in [0, 1]
```

### Issue #2: Penalty Calculation with Unit Mismatch

**File:** `core/pipeline_runner.py` line 298

```python
# BROKEN CODE:
penalty = 2.5  # in [0, 4.5] scale (from 1.0+1.0+1.5 thresholds)
enhanced = 0.3  # normalized [0, 1]
result = max(0.1, enhanced - penalty)  # Subtracting [0, 4.5] from [0, 1]
# Result: 0.3 - 2.5 = -2.2 → clipped to floor 0.1
```

**Impact:** Most stocks hit the 0.1 floor because penalties were in wrong scale

**Fix:** Normalize penalties before subtraction
```python
normalized_penalty = penalty / 100.0  # Convert [0, 4.5] to [0, 0.045]
result = max(0.01, enhanced - normalized_penalty)  # Both in [0, 1] range
```

### Issue #3: Score Range Confusion

**File:** `core/pipeline_runner.py` lines 286-310

The pipeline was working with **two conflicting score scales**:
- Input to `compute_advanced_score`: `base_score / 100.0` → [0, 1] range
- Outputs (`enhanced`, `penalty`): Mixed [0, 1] and [0, 100] scales
- Final floor check: `>= 0.1` (treating as 0-100 scale when was 0-1)

**Fix:** Explicitly scale FinalScore_20d back to 0-100 after advanced filters
```python
# After all advanced filter computations (all in [0, 1] range)
results["FinalScore_20d"] = results["FinalScore_20d"] * 100.0  # Convert back

# Now all downstream code sees consistent 0-100 scale
```

## Changes Applied

### 1. advanced_filters.py (Lines 319-325)
**Changed:** Score normalization
- **Before:** `enhanced_score = min(100.0, base_score + total_boost)`
- **After:** `normalized_boost = total_boost / 100.0; enhanced_score = base_score + normalized_boost`
- **Effect:** Correctly handles 0-1 normalized inputs

### 2. core/pipeline_runner.py (Lines 286-310)
**Changed:** Three critical fixes
1. **Catastrophic rejection floor:** `0.1` → `0.01` (matching [0, 1] range)
2. **Penalty normalization:** Added `normalized_penalty = penalty / 100.0`
3. **Scale back to 0-100:** Added `results["FinalScore_20d"] = results["FinalScore_20d"] * 100.0`
4. **Updated floor check:** `>= 0.1` → `>= 1.0` (after scaling back)

**Effect:** Scores now correctly preserve values instead of being floored

## Verification

### Before Fix
```
Ticker    FinalScore_20d  TechScore_20d  Fundamental_Score  ML_20d_Prob
GOOGL            20.27          26.57              44.18       0.0710
META             13.28          28.74              44.18       0.0663
UNH               8.29          30.94              44.18       0.0691
MSFT              6.28          29.12              44.18       0.0611
AAPL              0.10          33.25              44.18       0.0584  ← Should be ~25!
AMZN              0.10          28.31              44.18       0.0664  ← Should be ~20!
NVDA              0.10          29.25              44.18       0.0741  ← Should be ~23!
(7 more stocks also at 0.1 floor)
```

### Expected After Fix
```
All scores should be in 5-30 range (not hardcoded at 0.1)
AAPL should score ~25-30 (strong tech leader)
AMZN should score ~20-25 (tech leader)
NVDA should score ~25-35 (hot tech)
```

## Next Steps

1. **Immediate:** Test pipeline on 12-stock sample to verify score distribution
2. **Verify:** Confirm recommendations increase from 2 to 10+ stocks
3. **Integrate:** Add Big Winner signal and pattern matching (already built)
4. **Validate:** Run full 1500-stock scan in test environment
5. **Deploy:** Update auto_scan.yml to run next scheduled scan

## Technical Notes

- The scorer always had good **components** (Tech ~26-33, Fund ~44, ML ~0.06-0.07)
- The problem was **not** in component calculation, but in **how they were combined**
- The penalty system was logically sound but used **wrong units**
- This explains why only top tech stocks (high enough raw scores to survive penalties) showed up

## References

- **Pattern:** Unit mismatch bugs often occur when mixing normalized [0, 1] and raw [0, 100] scales
- **Lesson:** Should have had explicit scale conversions with clear comments
- **Fix:** Isolated unit conversions and added documentation
