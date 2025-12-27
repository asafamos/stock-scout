## Stock Scout UI Refactoring - Complete Diff Guide

### CRITICAL CHANGES NEEDED:

#### 1. **REPLACE: Lines 4302-~5900 (Sidebar + Card Rendering)**

**REMOVE:**
```python
with st.sidebar:
    st.header("🎛️ Scan Controls")
    # ... entire sidebar content ...
```

**REPLACE WITH:**
```python
# ==================== HEBREW RTL MAIN PAGE LAYOUT ====================
st.markdown("---")

# Top control bar (replaces sidebar)
universe_size, style, capital = render_top_control_bar()
st.session_state["universe_size"] = universe_size
st.session_state["total_budget"] = capital

# Force ML always on
st.session_state["ENABLE_ML"] = True
st.session_state["USE_FINAL_SCORE_SORT"] = True

# View controls
view_mode, risk_levels, min_score, max_rsi = render_view_controls()

# Advanced options in expander
adv_options = render_hebrew_sidebar_expander(CONFIG)
st.session_state.update(adv_options)

st.markdown("---")

# Sort and filter
results = force_ml_and_sorting(results)

if "Score" in results.columns:
    results = results[results["Score"] >= min_score]

if "RSI" in results.columns:
    results = results[(results["RSI"].isna()) | (results["RSI"] <= max_rsi)]

if "Risk_Level" in results.columns and risk_levels:
    results = results[results["Risk_Level"].isin(risk_levels)]

# Split into core and speculative
core_df = results[results.get("Risk_Level", "speculative") == "core"].copy() if "Risk_Level" in results.columns else pd.DataFrame()
spec_df = results[results.get("Risk_Level", "core") == "speculative"].copy() if "Risk_Level" in results.columns else pd.DataFrame()

# Render sections
render_kpi_cards_hebrew(results, len(core_df), len(spec_df))

st.markdown("---")

# Show top picks or all
if view_mode == "הבחירות המובילות":
    render_core_section_hebrew(core_df.head(10))
    if not spec_df.empty:
        st.markdown("---")
        render_speculative_section_hebrew(spec_df.head(5))
else:
    render_core_section_hebrew(core_df)
    if not spec_df.empty:
        st.markdown("---")
        render_speculative_section_hebrew(spec_df)
```

---

### 2. **UPDATE: RTL/Hebrew Styling**
Already implemented in `hebrew_ui.py` via `setup_hebrew_rtl()` function.

---

### 3. **REMOVE: Old functions**
- Delete `build_clean_card()` function (lines ~125-300)
- Delete all old HTML card helpers
- Delete card CSS references

---

### 4. **KEY PRESERVATIONS:**
✅ **Backend Logic** - ALL unchanged:
- Technical indicators (RSI, ATR, MACD, ADX, etc.)
- Fundamentals fetching (all providers)
- ML scoring
- Risk gates
- Allocation logic
- CSV/JSON export format
- All data pipelines

✅ **Session State Keys** - ALL preserved:
- `ENABLE_ML`
- `USE_FINAL_SCORE_SORT`
- All CONFIG settings
- All provider flags

---

###5. **TESTING CHECKLIST:**
```bash
# 1. Verify syntax
python3 -m py_compile stock_scout.py

# 2. Start app
streamlit run stock_scout.py

# 3. Visual checks:
☐ No sidebar visible
☐ Top control bar shows with Hebrew labels
☐ RTL text direction (right-aligned)
☐ Recommendations show without HTML cards
☐ Native Streamlit containers with borders
☐ All scores/numbers still calculate correctly
☐ ML always on (no toggle)
☐ Results sorted by FinalScore
☐ CSV export still works
☐ All backend logic unchanged
```

---

### 6. **COLUMN NAME MAPPINGS:**
- Ensure FinalScore or overall_score_20d exists for sorting
- Risk_Level column must have "core" or "speculative"  
- ML_20d_Prob for ML display
- Price_Yahoo / Unit_Price for pricing
- Fundamental_S for fundamental score

---

### 7. **RTL + English text handling:**
All English tickers/numbers wrapped with `<span dir='ltr'>TICKER</span>` in markdown for proper display.

---

## Implementation Notes:

1. **No breaking changes** - All backend scoring, risk rules, CSV export remain identical
2. **Session state** - ML flags forced to True programmatically (no UI toggle)
3. **Sidebar removal** - Moves controls to top bar + expanders on main page
4. **Hebrew throughout** - UI labels are all Hebrew (עברית)
5. **RTL styling** - CSS `direction: rtl;` applied globally
6. **Native components** - st.container(border=True), st.columns(), st.metric(), st.expander()
