# Engine Alignment Plan – Step 1 (Live + Backtest + Time-Test)

## Objective
Ensure that all three execution modes (live Streamlit app, `unified_backtest.py`, `unified_time_test.py`) use **exactly the same core logic and config** for building indicators, applying filters, and computing scores.

---

## Current State Assessment

### Core Module Inventory
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `core/unified_logic.py` | 818 | Technical indicators, filters, unified functions | ✅ Clean  |
| `core/scoring_engine.py` | 892 | Final scoring, weighting, RR evaluation | ✅ Clean |
| `core/config.py` | 204 | Centralized config (dataclass) | ✅ Exists |
| `core/risk.py` | 270 | Risk classification & filtering | ✅ Exists |
| `core/v2_risk_engine.py` | 568 | V2 risk gates & allocation | ✅ Exists |
| `core/classification.py` | 622 | Core vs. Speculative classification | ✅ Exists |

### Entry Points
| File | Imports Core? | Duplicates Logic? | Status |
|------|---------------|-------------------|--------|
| `stock_scout.py` | Partial | **YES** – recalculates indicators, filters | ⚠️  Needs refactor |
| `unified_backtest.py` | YES | **NO** – uses core functions | ✅ OK |
| `unified_time_test.py` | YES | **NO** – uses core functions | ✅ OK |

### Duplication Found
1. **In `stock_scout.py` (lines ~3500-3800):**
   - Recalculates MA, RSI, ATR, volume indicators directly
   - Applies custom filter logic instead of calling `apply_technical_filters()`
   - Hard-codes thresholds (RSI_BOUNDS, MA alignment) instead of using config

2. **No major duplication in backtest/time-test** – they already delegate to core functions

---

## Action Items

### Phase 1: Documentation & Type Hints (Non-Breaking)

#### 1.1 Enhance `core/unified_logic.py`
- [x] Already has good docstrings for main functions
- [ ] Add return type hints to ALL functions
- [ ] Add Raises section to docstrings where applicable
- [ ] Ensure all public functions document their guarantees

**Target Functions:**
- `build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame`
- `apply_technical_filters(row: pd.Series, strict: bool, relaxed: bool) -> bool`
- `compute_technical_score(row: pd.Series, weights: Optional[Dict]) -> float`
- `compute_final_score(tech_score: float, fundamental_score: Optional[float], ml_prob: float) -> float`

#### 1.2 Enhance `core/scoring_engine.py`
- [ ] Add comprehensive module docstring
- [ ] Add return type hints to ALL functions
- [ ] Document weight assumptions and normalization rules
- [ ] Add examples for key functions

**Target Functions:**
- `compute_overall_score(row: pd.Series) -> Tuple[float, Dict[str, float]]`
- `evaluate_rr_unified(rr_ratio: Optional[float]) -> Tuple[float, float, str]`
- `normalize_score(value: float, ...) -> float`

### Phase 2: Remove Duplication in `stock_scout.py`

#### 2.1 Identify All Duplicated Logic
- [ ] Search for indicator calculations (MA, RSI, ATR, etc.)
- [ ] Search for filter thresholds (hard-coded values)
- [ ] Search for weighting logic
- [ ] Extract them into a summary list

#### 2.2 Replace with Core Function Calls
- [ ] Replace manual indicator builds with `build_technical_indicators()`
- [ ] Replace filter checks with `apply_technical_filters()`
- [ ] Replace score calculations with functions from `core/scoring_engine.py`

#### 2.3 Centralize Threshold Values
- [ ] Remove hard-coded RSI_BOUNDS, ATR limits, volume thresholds
- [ ] Add missing config keys to `core/config.py` if needed
- [ ] Update `stock_scout.py` to load config once at startup

### Phase 3: Align All Entry Points

#### 3.1 Config Unification
- [ ] Verify all three entry points load `core/config.py`
- [ ] Remove any environment-variable overrides that bypass config
- [ ] Ensure consistent default values across all three

#### 3.2 Function Call Consistency
- [ ] Add wrapper functions in core if needed (e.g., end-to-end pipeline functions)
- [ ] Document the exact call sequence each entry point should follow
- [ ] Add assertions to validate config is loaded correctly

### Phase 4: Consistency Checker

#### 4.1 Create `core/debug_utils.py`
- [ ] Function to run same ticker+date through "live pipeline" vs. "backtest pipeline"
- [ ] Compare key columns: Technical_S, Fundamental_S, RR_Score, Reliability_v2, Final_Score
- [ ] Log discrepancies with thresholds (>1% difference = warning)
- [ ] Make it callable from all three entry points

#### 4.2 Validation Test
- [ ] Sample 10 random tickers from top 100
- [ ] Run both pipelines, compare outputs
- [ ] Generate report showing any systematic differences

### Phase 5: Testing & Validation

#### 5.1 Unit Tests
- [ ] Test `build_technical_indicators()` returns same columns every time
- [ ] Test `apply_technical_filters()` gives same result for same row
- [ ] Test `compute_technical_score()` is deterministic
- [ ] Test config loading from all three entry points

#### 5.2 Integration Tests
- [ ] Run full pipeline on 5 different dates
- [ ] Verify live app produces same recommendations as backtest with same data/date
- [ ] Verify time-test produces same ML scores as live app for same ticker/date

---

## Expected Outcomes

1. **Single Source of Truth for Logic:**
   - All business logic (indicators, filters, scoring) lives in `core/` modules
   - No duplication between entry points

2. **Config Centralization:**
   - All thresholds, weights, and parameters defined in `core/config.py`
   - No magic numbers in UI or entry-point files

3. **Guaranteed Consistency:**
   - `debug_utils.py` can prove live and backtest produce identical outputs
   - Type hints make function contracts explicit
   - Comprehensive docstrings document edge cases

4. **Maintainability:**
   - Changes to scoring logic require edits in ONE place (core modules)
   - Each entry point is thin wrapper around core functions
   - Clear audit trail of what changed and why

---

## Timeline
- **Phase 1:** 30 min (add type hints and docstrings)
- **Phase 2:** 60 min (refactor `stock_scout.py`)
- **Phase 3:** 20 min (verify config alignment)
- **Phase 4:** 40 min (create consistency checker)
- **Phase 5:** 30 min (write validation tests)

**Total: ~3 hours**

---

## Rollout Plan
1. Complete all phases
2. Run validation tests → confirm 100% match
3. Create comprehensive test suite
4. Single commit with message describing refactor
5. Update README.md with architecture diagram

