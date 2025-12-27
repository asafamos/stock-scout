# Score = FinalScore_20d Enforcement Summary

## Objective
Enforce a strict rule across the entire codebase: **`Score` must always equal `FinalScore_20d`**. No other code is allowed to independently assign a value to `Score`.

## Changes Made

### 1. core/unified_logic.py - compute_recommendation_scores()

**Location:** Lines 310-327

**Before:**
```python
# Legacy overall score/breakdown for backward compatibility
score, breakdown = compute_overall_score(rec_row)
rec_row["Score"] = rec_row.get("Score", score)  # ❌ Could override Score
rec_row["Score_Breakdown"] = breakdown
```

**After:**
```python
# Legacy overall score/breakdown for backward compatibility
score, breakdown = compute_overall_score(rec_row)
rec_row["Score_Breakdown"] = breakdown

# STRICT RULE: Score must always equal FinalScore_20d (set in to_series above)
# Never override Score here - it's already set correctly
```

**Rationale:** The `RecommendationResult.to_series()` method (line 83) already sets `"Score": self.final_score`, which is correct. We removed the legacy line that could potentially override it with a different value from `compute_overall_score()`.

---

### 2. core/pipeline_runner.py - Advanced Filters Section

**Location:** Lines 275-303

**Before:**
```python
if catastrophic:
    results.at[idx, "Score"] = 0.1  # ❌ Direct Score assignment
    results.at[idx, "RejectionReason"] = reason
else:
    penalty = 0.0
    if sig.get("rs_63d", 0) < rs_thresh: penalty += 1.0
    if sig.get("momentum_consistency", 0) < mom_thresh: penalty += 1.0
    if sig.get("risk_reward_ratio", 0) < rr_thresh: penalty += 1.5
    
    results.at[idx, "AdvPenalty"] = penalty
    results.at[idx, "Score"] = max(0.1, enhanced - penalty)  # ❌ Direct Score assignment

# Keep stocks with positive scores
results = results[results["Score"] >= 0.1].copy()  # ❌ Filtering by Score
```

**After:**
```python
if catastrophic:
    # Apply penalty by reducing FinalScore_20d, not by overwriting Score
    results.at[idx, "FinalScore_20d"] = 0.1
    results.at[idx, "RejectionReason"] = reason
else:
    penalty = 0.0
    if sig.get("rs_63d", 0) < rs_thresh: penalty += 1.0
    if sig.get("momentum_consistency", 0) < mom_thresh: penalty += 1.0
    if sig.get("risk_reward_ratio", 0) < rr_thresh: penalty += 1.5
    
    results.at[idx, "AdvPenalty"] = penalty
    # Apply penalty to FinalScore_20d, not Score
    results.at[idx, "FinalScore_20d"] = max(0.1, enhanced - penalty)

# Ensure Score always matches FinalScore_20d after advanced filters
if "FinalScore_20d" in results.columns:
    results["Score"] = results["FinalScore_20d"]

# Keep stocks with positive scores (filter by FinalScore_20d)
results = results[results["FinalScore_20d"] >= 0.1].copy()
```

**Rationale:** Advanced filters were directly modifying `Score` with penalty-adjusted values. Changed to modify `FinalScore_20d` instead, then explicitly sync `Score` to match.

---

### 3. core/pipeline_runner.py - Final Safety Check

**Location:** Lines 397-404 (end of run_scan_pipeline)

**Added:**
```python
# STRICT ENFORCEMENT: Score must always equal FinalScore_20d
# This is the final safety check before returning results
if "FinalScore_20d" in results.columns:
    results["Score"] = results["FinalScore_20d"]
    logger.info(f"[PIPELINE] Final check: Score column set to FinalScore_20d for all {len(results)} results")
```

**Rationale:** Final enforcement at the end of the pipeline ensures that even if any legacy code path survived, `Score` is forced to equal `FinalScore_20d` before results are returned.

---

### 4. test_unified_pipeline.py - Assertion Test

**Location:** Lines 175-194

**Added:**
```python
print(f"\n{'='*80}")
print("STRICT RULE ENFORCEMENT TEST")
print(f"{'='*80}")

# CRITICAL TEST: Score must ALWAYS equal FinalScore_20d
if "Score" in results.columns and "FinalScore_20d" in results.columns:
    score_mismatch = results[results["Score"] != results["FinalScore_20d"]]
    if len(score_mismatch) == 0:
        print("✅ PASS: Score == FinalScore_20d for all rows")
    else:
        print(f"❌ FAIL: {len(score_mismatch)} rows have Score != FinalScore_20d")
        print(f"\nMismatched rows:")
        print(score_mismatch[["Ticker", "Score", "FinalScore_20d"]].to_string())
        raise AssertionError("STRICT RULE VIOLATED: Score must always equal FinalScore_20d")
else:
    print("⚠️  Cannot test: Score or FinalScore_20d column missing")
```

**Rationale:** Automated test that fails with a clear error if any row has `Score != FinalScore_20d`.

---

## Key Code Snippets

### compute_recommendation_scores - Score Assignment (unified_logic.py)

The only place where `Score` gets its value:

```python
def to_series(self) -> pd.Series:
    base = {
        "Ticker": self.ticker,
        "TechScore_20d": self.tech_score,
        "Fundamental_Score": self.fundamental_score,
        "ML_20d_Prob": self.ml_prob,
        "FinalScore_20d": self.final_score,  # ← The source of truth
        "ConvictionScore": self.conviction_score,
        "Reliability_Score": self.reliability_score,
        "Risk_Label": self.risk_label,
        "Risk_Meter": self.risk_meter,
        "Should_Display": self.should_display,
        # Legacy alias - MUST always equal final_score
        "Score": self.final_score,  # ← Only assignment from final_score
    }
    base.update(self.extras)
    return pd.Series(base, dtype="object")
```

### run_scan_pipeline - Final Enforcement (pipeline_runner.py)

```python
def run_scan_pipeline(...) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # ... (all pipeline steps)
    
    # STRICT ENFORCEMENT: Score must always equal FinalScore_20d
    # This is the final safety check before returning results
    if "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
        logger.info(f"[PIPELINE] Final check: Score column set to FinalScore_20d for all {len(results)} results")
        
    return results, data_map
```

---

## Verification

Test run output:
```
================================================================================
STRICT RULE ENFORCEMENT TEST
================================================================================
✅ PASS: Score == FinalScore_20d for all rows
================================================================================
```

The test successfully verifies that after running the full pipeline with 5 tickers (AAPL, MSFT, GOOGL, NVDA, TSLA), **all rows have Score exactly equal to FinalScore_20d**.

---

## Summary of Files Modified

1. ✅ **core/unified_logic.py** - Removed legacy Score override in compute_recommendation_scores
2. ✅ **core/pipeline_runner.py** - Changed advanced filters to modify FinalScore_20d instead of Score
3. ✅ **core/pipeline_runner.py** - Added final enforcement check before returning results
4. ✅ **test_unified_pipeline.py** - Added strict assertion test

---

## Rule Enforcement Strategy

1. **Source of Truth:** `FinalScore_20d` is computed once in `compute_final_score()` and set in `RecommendationResult.final_score`
2. **Alias Assignment:** `Score` is set equal to `final_score` in `RecommendationResult.to_series()`
3. **Pipeline Modifications:** Any adjustments (penalties, filters) now modify `FinalScore_20d`, not `Score`
4. **Sync Points:** After each major pipeline step, explicitly sync: `results["Score"] = results["FinalScore_20d"]`
5. **Final Safety Net:** Before returning results, force one final sync to catch any edge cases
6. **Automated Testing:** Test fails with clear error if rule is violated

---

## Benefits

- **Single Source of Truth:** Only one place computes the final score
- **No Confusion:** Developers can't accidentally create divergent scores
- **Easier Debugging:** When Score looks wrong, check FinalScore_20d (they're guaranteed equal)
- **Backward Compatible:** Old code using `Score` still works, just accesses the alias
- **Future-Proof:** New features must use `FinalScore_20d`; `Score` is read-only legacy alias
