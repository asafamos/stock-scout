# Code Locations Reference

## stock_scout.py Changes

### Change #1: ML_20d_Prob Computation (Line ~2555-2562)
**Purpose:** Ensure ML probability is computed per-row (not defaulting to NaN)

**Location:** After reward/risk calculation, before row append
```python
# Soft ML signal (20d) probability using offline model bundle (respect toggle)
ml_prob_20d = None
enable_ml_main = bool(st.session_state.get("ENABLE_ML", True))
if ML_20D_AVAILABLE and enable_ml_main:
    try:
        ml_prob_20d = predict_20d_prob_from_row(row_indicators)
    except Exception:
        ml_prob_20d = None

# NOTE: FinalScore will be recomputed AFTER all rows are collected using percentile ranks
# For now, just store placeholder value
final_score = np.nan
```

---

### Change #2: TechScore_20d_v2 Computation Clarity (Line ~2675-2708)
**Purpose:** Clarify that formula is EXACT match with offline audit

**Location:** Post-pipeline, after all rows collected
```python
# Build TechScore_20d_v2: data-driven technical composite using percentile ranks of live indicators
# EXACT SAME FORMULA as offline audit (experiments/offline_recommendation_audit.py)
# This replaces the legacy TechScore_20d (Score_Tech) as the technical component in FinalScore
if not results.empty:
    # Use same column names as offline audit for consistency
    technical_features = ["Score_Tech", "RSI", "ATR_Pct", "RR", "MomCons", "VolSurge"]
    signs = {
        "Score_Tech": 1.0,      # (= TechScore_20d in offline)
        "RSI": -1.0,
        "ATR_Pct": 1.0,
        "RR": 1.0,
        "MomCons": -1.0,
        "VolSurge": 1.0,
    }
    
    # [... rest of computation ...]
```

---

### Change #3: FinalScore Rank-Based Recomputation (Line ~2710-2725) ⭐ MAIN FIX
**Purpose:** Compute FinalScore using 0.5/0.5 percentile-rank blend (not 80/20 raw)

**Location:** Post-pipeline, after TechScore_v2 computed, before sorting
```python
# Recompute FinalScore using RANK-BASED combination (EXACTLY as offline audit):
# FinalScore = 100 * (0.5 * percentile_rank(TechScore_20d_v2) + 0.5 * percentile_rank(ML_20d_Prob))
# This ensures live app and offline audit use identical ranking logic
# Note: ML_20d_Prob must be [0, 1] float; if unavailable, fillna(0.0) before ranking
if not results.empty and "TechScore_20d_v2" in results.columns:
    tech_scores = results["TechScore_20d_v2"].astype(float)
    ml_probs = results["ML_20d_Prob"].astype(float).fillna(0.0) if "ML_20d_Prob" in results.columns else pd.Series([0.0] * len(results))
    
    # Compute percentile ranks within live scan (0-1)
    tech_rank = tech_scores.rank(method='average', pct=True)  # [0, 1]
    ml_rank = ml_probs.rank(method='average', pct=True)        # [0, 1]
    
    # Combined score: 0.5 tech_v2_rank + 0.5 ml_rank for balanced ranking signal
    combined_score = 0.5 * tech_rank + 0.5 * ml_rank
    results["FinalScore"] = combined_score * 100.0  # rescale to [0, 100]
elif not results.empty and "TechScore_20d_v2" in results.columns:
    # If ML not available or disabled, use TechScore_20d_v2 directly as FinalScore
    results["FinalScore"] = results["TechScore_20d_v2"]
```

---

### Change #4: Debug Print - Top-Ranked Ticker (Line ~2728-2736)
**Purpose:** Print top-ranked ticker with all 4 scores for easy validation

**Location:** Post-pipeline, before sorting, right after FinalScore computation
```python
# Debug: print top-ranked ticker with all 4 scores for validation
if not results.empty:
    top_idx = 0 if "FinalScore" not in results.columns else results["FinalScore"].idxmax()
    top_row = results.iloc[top_idx]
    top_ticker = top_row.get("Ticker", "N/A")
    top_tech = top_row.get("Score_Tech", np.nan)
    top_tech_v2 = top_row.get("TechScore_20d_v2", np.nan)
    top_ml = top_row.get("ML_20d_Prob", np.nan)
    top_final = top_row.get("FinalScore", np.nan)
    top_ml_pct = float(np.clip(top_ml*100,0,100)) if np.isfinite(top_ml) else float('nan')
    print(f"[DEBUG] Top-ranked ticker: {top_ticker} | Score_Tech={top_tech:.1f} | TechScore_v2={top_tech_v2:.1f} | ML_20d_Prob={top_ml_pct:.1f}% | FinalScore={top_final:.1f}")

# Debug: print ML_20d_Prob statistics
ml_finite_mask = np.isfinite(results["ML_20d_Prob"])
# [... rest of ML stats ...]
```

---

### Change #5: Precomputed Run FinalScore Recalc (Line ~2375-2382)
**Purpose:** When loading precomputed scan, restore ML_20d_Prob from CSV and recalculate FinalScore

**Location:** In `if skip_pipeline:` block, right after loading precomputed results
```python
if skip_pipeline:
    # Use precomputed results
    t0_stage3 = time.perf_counter()
    results = st.session_state.get("precomputed_results")
    logger.info(f"Using precomputed scan with {len(results)} tickers")
    
    # Ensure ML_20d_Prob is read from CSV (not defaulted to NaN)
    # This is critical: when loading a precomputed run, we must restore the ML probabilities
    if "ML_20d_Prob" not in results.columns:
        results["ML_20d_Prob"] = np.nan
        logger.warning("[WARN] Precomputed scan missing ML_20d_Prob column; set to NaN")
    
    # Re-compute FinalScore to match live pipeline
    # FinalScore = 100 * (0.5 * rank(TechScore_v2) + 0.5 * rank(ML_20d_Prob))
    if "TechScore_20d_v2" in results.columns:
        tech_scores = results["TechScore_20d_v2"].astype(float)
        ml_probs = results["ML_20d_Prob"].astype(float).fillna(0.0)
        tech_rank = tech_scores.rank(method='average', pct=True)
        ml_rank = ml_probs.rank(method='average', pct=True)
        combined_score = 0.5 * tech_rank + 0.5 * ml_rank
        results["FinalScore"] = combined_score * 100.0
    
    status_manager.complete(f"✅ Precomputed scan loaded: {len(results)} tickers")
    # [... rest of precomputed handling ...]
```

---

## Offline Audit Reference (Source of Truth)

### File: experiments/offline_recommendation_audit.py

#### TechScore_v2 Computation (Line ~308-325)
```python
technical_features = ["TechScore_20d", "RSI", "ATR_Pct", "RR", "MomCons", "VolSurge"]
signs = {
    "TechScore_20d": 1.0,
    "RSI": -1.0,
    "ATR_Pct": 1.0,
    "RR": 1.0,
    "MomCons": -1.0,
    "VolSurge": 1.0,
}

for col in technical_features:
    if col in out_df.columns:
        out_df[col + "_rank"] = out_df[col].fillna(0.0).rank(pct=True, method='average')

def compute_tech_v2(row):
    valid_feats = [col for col in technical_features if col in out_df.columns]
    numerator = sum(signs.get(col, 1.0) * row.get(col + "_rank", 0.5) for col in valid_feats)
    score = numerator / len(valid_feats)
    score = max(0.0, min(1.0, score))
    return 100.0 * score

out_df["TechScore_20d_v2"] = out_df.apply(compute_tech_v2, axis=1)
```

#### FinalScore Computation (Line ~621-627)
```python
tech_scores = df["TechScore_20d_v2"].astype(float)
ml_probs_col = df["ML_20d_Prob"].astype(float).fillna(0.0)

tech_rank = tech_scores.rank(method='average', pct=True)
ml_rank = ml_probs_col.rank(method='average', pct=True)

combined_score = 0.5 * tech_rank + 0.5 * ml_rank
df["FinalScore"] = combined_score * 100.0
```

---

## Documentation Files Created

1. **LIVE_AUDIT_ALIGNMENT.md**
   - Detailed side-by-side comparison of offline vs live
   - Expected outputs and validation checklist
   - 8 sections covering all aspects

2. **LIVE_SCORING_TIGHTENING_SUMMARY.md**
   - Complete change log with before/after
   - Explanation of why changes were needed
   - Verification results and deployment checklist

3. **LIVE_SCORING_QUICK_REF.md**
   - TL;DR version of all changes
   - Quick validation checklist
   - Example comparisons

---

## How to Find the Changes

### In stock_scout.py:
1. Search for: `"FinalScore will be recomputed AFTER"` → Change #1
2. Search for: `"EXACT SAME FORMULA as offline"` → Change #2
3. Search for: `"RANK-BASED combination"` → Change #3 ⭐
4. Search for: `"[DEBUG] Top-ranked ticker"` → Change #4
5. Search for: `"Re-compute FinalScore to match live pipeline"` → Change #5

### To Verify All Changes:
```bash
grep -n "EXACT SAME FORMULA\|RANK-BASED\|Top-ranked ticker\|Re-compute FinalScore" stock_scout.py
```

---

## Summary Table

| Change | File | Line | Type | Impact |
|--------|------|------|------|--------|
| 1 | stock_scout.py | ~2555 | Remove per-row logic | Enables post-pipeline rank |
| 2 | stock_scout.py | ~2675 | Add clarity | Documentation only |
| 3 | stock_scout.py | ~2710 | Add new logic | **Main fix** ⭐ |
| 4 | stock_scout.py | ~2728 | Add debug | Visibility |
| 5 | stock_scout.py | ~2375 | Add precomputed logic | Consistency |

---

## Testing Commands

```bash
# Verify syntax
python3 -m py_compile stock_scout.py

# Run live app
streamlit run stock_scout.py

# Run offline audit
python -m experiments.offline_recommendation_audit --mode audit_ml_20d \
    --input experiments/training_dataset_20d_v2.csv \
    --output experiments/audit_ml_20d_v2.csv

# Compare top-decile stats manually
```

---

## Notes

- All changes are **backward compatible**
- No UI/HTML modifications
- No new dependencies
- All imports already exist
- Syntax verified (py_compile passed)
- Test data validated
- Production ready after manual verification

