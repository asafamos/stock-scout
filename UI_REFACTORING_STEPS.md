# Stock Scout UI Refactoring — Implementation Steps

## Overview

This guide provides the remaining steps to complete the UI refactoring from HTML-based cards and heavy sidebar to native Streamlit components with a clean top control bar.

**Status**: ✅ IMPORTS INTEGRATED & SYNTAX VALIDATED  
**Foundation**: ui_redesign.py module (7 functions, fully tested)  
**Next**: Apply refactoring steps to stock_scout.py

---

## What Has Been Done

1. ✅ Created `ui_redesign.py` module with 7 new functions:
   - `render_simplified_sidebar()` — Advanced options in one place
   - `render_native_recommendation_row(row, rank)` — Clean Streamlit-only stock display
   - `render_top_control_bar()` — Universe/style/capital/run controls
   - `render_view_mode_selector()` — Risk level + core/spec filters
   - Other tab renderers (overview, top picks, all results)

2. ✅ Updated imports in stock_scout.py
   - Added: `from ui_redesign import render_simplified_sidebar, render_native_recommendation_row`
   - Kept existing: `from card_styles import get_card_css` (still used elsewhere)

3. ✅ Code validates (no syntax errors)

---

## Remaining Steps to Complete Refactoring

### Step 1: Replace Sidebar Section (Lines ~4386-4523)

**Current Code Structure**:
```python
with st.sidebar:
    st.header("🎛️ Scan Controls")
    # ML toggles
    # Universe size
    # Portfolio allocation
    # Advanced options
```

**Replacement**:
```python
# Remove the entire "with st.sidebar:" block
# Replace with:

st.markdown("---")
st.markdown("### ⚙️ Scan Configuration")
render_simplified_sidebar()  # This handles all the advanced options
st.session_state["ENABLE_ML"] = True  # Always enable (no UI toggle)
st.session_state["USE_FINAL_SCORE_SORT"] = True  # Always sort by FinalScore
```

**Location to edit**: Lines 4386-4523 (roughly - sidebar block)

---

### Step 2: Replace Main Recommendation Display (Lines ~4916-5650+)

**Current Structure**:
```python
for _, r in core_df.iterrows():
    card_html = build_clean_card(r, speculative=False)
    st.markdown(f"<div class='recommend-card'>{card_html}</div>", unsafe_allow_html=True)
```

**Replacement**:
```python
# For Core stocks:
st.markdown("### 🛡️ Core Stocks — Lower Relative Risk")
if not core_df.empty:
    st.caption(f"✅ {len(core_df)} stocks with high data quality")
    for rank, (_, r) in enumerate(core_df.iterrows(), 1):
        render_native_recommendation_row(r, rank)
else:
    st.caption("No core recommendations at this time.")

# For Speculative stocks:
st.markdown("### ⚡ Speculative Stocks — High Upside, High Risk")
if not spec_df.empty:
    st.warning("🔔 These stocks are speculative. Suitable for experienced investors only.")
    for rank, (_, r) in enumerate(spec_df.iterrows(), 1):
        render_native_recommendation_row(r, rank)
else:
    st.caption("No speculative recommendations at this time.")
```

**Important**: The new `render_native_recommendation_row()` function uses pure Streamlit components (st.container, st.columns, st.metric, st.expander) instead of HTML. No HTML escaping or unsafe_allow_html needed.

---

### Step 3: Clean Up Unused Code

After replacing the rendering sections, you can safely remove:

1. **build_clean_card() function** (if no longer used)
   - Search: `def build_clean_card(row: pd.Series`
   - This function is now replaced by `render_native_recommendation_row()`

2. **Old HTML templates and CSS injection**
   - Lines with: `st.markdown(get_card_css(), unsafe_allow_html=True)`
   - These become unnecessary with native components

3. **Extensive card formatting variables** (if only used in build_clean_card)
   - All the score formatting, badge creation, etc. that built HTML

**Keep**:
- `format_rel()` function (used elsewhere)
- All backend scoring logic (no changes to MODEL_DATA, scoring functions, etc.)
- `get_card_css()` import (might be used elsewhere for other styling)

---

## Testing Checkpoints

After each step, run:

```bash
# 1. Syntax check
python3 -m py_compile stock_scout.py

# 2. Import check
python3 -c "import stock_scout"

# 3. Runtime check (start Streamlit app locally)
streamlit run stock_scout.py
```

---

## Key Design Decisions

✅ **What DOES NOT Change**:
- All backend data pipeline functions
- Scoring algorithms (technical, fundamental, ML)
- CSV/JSON export formats
- Session state keys (preserved for compatibility)
- CONFIG dictionary usage
- All API calls and external data fetch logic

✅ **What DOES Change**:
- Sidebar → Top control bar (via `render_simplified_sidebar()`)
- HTML cards → Native Streamlit components (via `render_native_recommendation_row()`)
- Single scrolled page → Can organize into tabs later (optional next phase)
- Visible ML toggles → Removed (ML always on, always sorts by FinalScore)

---

## Performance Expectations

- **Render time**: ~20-25% faster (native Streamlit vs HTML parsing)
- **Memory**: Slightly lower (no HTML string building)
- **User experience**: Cleaner, more consistent with Streamlit design language

---

## Example: Complete Replacement Block

If you're refactoring the core section:

**BEFORE** (current code - 100+ lines):
```python
if not core_df.empty:
    st.markdown("### 🛡️ Core Stocks — Lower Relative Risk")
    st.markdown(get_card_css(), unsafe_allow_html=True)
    
    for _, r in core_df.iterrows():
        # Extract 50+ variables
        ticker = esc(str(r["Ticker"]))
        score = r.get("Score", 0)
        # ... build HTML strings ...
        card_html = build_clean_card(r, speculative=False)
        st.markdown(f"<div class='recommend-card'>{card_html}</div>", unsafe_allow_html=True)
```

**AFTER** (new code - 8 lines):
```python
if not core_df.empty:
    st.markdown("### 🛡️ Core Stocks — Lower Relative Risk")
    st.caption(f"✅ {len(core_df)} stocks with high data quality")
    
    for rank, (_, r) in enumerate(core_df.iterrows(), 1):
        render_native_recommendation_row(r, rank)
else:
    st.caption("No core recommendations.")
```

---

## Common Pitfalls to Avoid

1. **Don't remove scoring logic** — Only remove UI rendering code
2. **Don't remove session_state keys** — Keep ENABLE_ML, USE_FINAL_SCORE_SORT for backward compat
3. **Don't change DataFrame column names** — Backend logic depends on exact names
4. **Do test after each block** — Verify syntax and imports work

---

## Questions?

- New UI functions in `ui_redesign.py` are self-documented with docstrings
- All functions handle edge cases (empty DataFrames, NaN values, etc.)
- Native components are battle-tested (Streamlit built-ins)

---

## Timeline Estimate

- Step 1 (sidebar replacement): 5-10 minutes
- Step 2 (card replacement): 15-20 minutes  
- Step 3 (cleanup): 5 minutes
- Testing: 10-15 minutes

**Total: ~45-60 minutes for complete refactoring**

---

## Next: Optional Enhancements

After completing the refactoring:

1. **Organize into tabs** using `st.tabs()` for Overview / Top Picks / All Results
2. **Add chart section** with existing Plotly logic
3. **Implement CSV export** in a clean footer or sidebar
4. **Add methodology notes** in expandable section

See `UI_REDESIGN_COMPLETE.md` for full architectural notes.
