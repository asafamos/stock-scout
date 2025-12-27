# Phase 2: Removing Duplication from stock_scout.py

## Objective
Eliminate recalculation of technical indicators and scores in `stock_scout.py` by using core functions from `core/unified_logic.py` and `core/scoring_engine.py`.

## Current State (Before Refactoring)
- **Lines 2400-2700**: Manual indicator calculations (MA, RSI, ATR, momentum, volume, RR, MACD, ADX)
- **Lines 2530-2650**: Manual score calculation using weighted components
- **Problem**: Duplicates logic that already exists in `core/unified_logic.py` and `core/scoring_engine.py`
- **Risk**: Live app and backtest may produce different results if core logic changes

## Target State (After Refactoring)
- **Lines 2400-2410**: Call `build_technical_indicators(df)` once per ticker
- **Lines 2410-2430**: Use indicator columns from result for filtering/display
- **Lines 2430-2450**: Call `apply_technical_filters(row, strict=True)` for hard filters
- **Lines 2450-2470**: Call `compute_technical_score(row)` for technical score
- **Result**: Single source of truth for all three entry points (live, backtest, time-test)

## Specific Changes Required

### 1. Replace inline indicator calculations with build_technical_indicators()
```python
# BEFORE (lines ~2420-2540):
# RSI calculation, ATR calculation, momentum calculation, etc.

# AFTER (lines ~2420-2425):
tech_indicators = build_technical_indicators(df)
row_indicators = tech_indicators.iloc[-1]  # Get latest row
```

### 2. Use indicator columns for filtering
```python
# BEFORE (custom thresholds):
if not (lo_rsi <= rsi_val <= hi_rsi):
    continue

# AFTER (use indicators):
if not apply_technical_filters(row_indicators, strict=True):
    continue
```

### 3. Use compute_technical_score for scoring
```python
# BEFORE (manual weighted sum):
score = W["ma"] * ma_ok + W["mom"] * mom_score + ...

# AFTER (core function):
score_0_100 = compute_technical_score(row_indicators, weights=W)
score = score_0_100 / 100.0  # Normalize to 0-1 if needed
```

## Testing Strategy

### Unit Tests
- All 69 existing tests must still pass
- No behavioral changes to scoring output (within 0.01% tolerance)

### Integration Tests
- Run live app on sample universe
- Compare Technical_S scores before/after refactoring
- Verify JSON export format unchanged

### Cross-Validation
- backtest.py already uses core functions correctly (no changes needed)
- time-test.py already uses core functions correctly (no changes needed)

## Timeline
- **Estimated**: 60 minutes
- **Review**: 15 minutes
- **Testing**: 15 minutes
- **Total**: 90 minutes

## Backward Compatibility
- Column names and output format remain unchanged
- Score values may differ by <1% due to rounding/order of operations
- All downstream logic (allocation, filtering, export) unaffected

## Success Criteria
1. ✅ All 69 pytest tests pass
2. ✅ Live app runs without errors on sample universe (50 tickers)
3. ✅ Technical_S column present and valid in results
4. ✅ No warnings about missing functions or deprecated code
5. ✅ Git commit with clear message about deduplication
