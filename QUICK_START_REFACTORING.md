# UI Refactoring Quick Start — Copy-Paste Reference

## TL;DR - What to Do

You have everything ready. Now you need to make these 2 key replacements in `stock_scout.py`:

### Replacement #1: Sidebar (Lines ~4386-4523)

**FIND THIS**:
```python
# Sidebar filters
with st.sidebar:
    st.header("🎛️ Scan Controls")
    
    # API status summary
    _ps = st.session_state.get("provider_status", {})
    # ... 100+ lines of sidebar code ...
    st.caption("📌 Disclaimer: For research only. Not investment advice.")
```

**REPLACE WITH**:
```python
# ==================== TOP CONTROL BAR ====================
st.markdown("---")
st.markdown("### ⚙️ Scan Configuration")

# Use ui_redesign simplified sidebar function
render_simplified_sidebar()

st.markdown("---")

# Set ML defaults (always enabled, no UI toggle)
st.session_state["ENABLE_ML"] = True
st.session_state["USE_FINAL_SCORE_SORT"] = True
```

---

### Replacement #2: Recommendation Cards (Lines ~4916-5650+)

**FIND THIS**:
```python
if rec_df.empty:
    st.info("No stocks currently pass the threshold with a positive buy amount.")
else:
    # Split into Core and Speculative (case-insensitive matching to avoid accidental drops)
    if "Risk_Level" in rec_df.columns:
        levels = rec_df["Risk_Level"].astype(str).str.lower()
        core_df = rec_df[levels == "core"].copy()
        spec_df = rec_df[levels == "speculative"].copy()
        # Inject card CSS once for all recommendation cards
        st.markdown(get_card_css(), unsafe_allow_html=True)
    else:
        # Fallback if Risk_Level column doesn't exist: treat all as core
        core_df = rec_df.copy()
        spec_df = pd.DataFrame()
    
    # ... 600+ lines of card rendering code ...
```

**REPLACE WITH**:
```python
if rec_df.empty:
    st.info("No stocks currently pass the threshold with a positive buy amount.")
else:
    # Split into Core and Speculative
    if "Risk_Level" in rec_df.columns:
        levels = rec_df["Risk_Level"].astype(str).str.lower()
        core_df = rec_df[levels == "core"].copy()
        spec_df = rec_df[levels == "speculative"].copy()
    else:
        core_df = rec_df.copy()
        spec_df = pd.DataFrame()

    # Summary
    total_candidates = len(core_df) + len(spec_df)
    st.info(f"📊 Showing {total_candidates} stocks ({len(core_df)} Core, {len(spec_df)} Speculative)")

    # Core Recommendations
    st.markdown("### 🛡️ Core Stocks — Lower Relative Risk")
    if not core_df.empty:
        st.caption(f"✅ {len(core_df)} stocks with high data quality")
        for rank, (_, r) in enumerate(core_df.iterrows(), 1):
            render_native_recommendation_row(r, rank)
    else:
        st.caption("No core recommendations at this time.")

    st.markdown("---")

    # Speculative Recommendations
    st.markdown("### ⚡ Speculative Stocks — High Upside, High Risk")
    if not spec_df.empty:
        st.caption(f"⚠️ {len(spec_df)} stocks with higher risk profile")
        st.warning("🔔 Warning: These are speculative. Suitable for experienced investors only.")
        for rank, (_, r) in enumerate(spec_df.iterrows(), 1):
            render_native_recommendation_row(r, rank)
    else:
        st.caption("No speculative recommendations at this time.")
```

---

## Key Points

1. **Delete everything from the old sidebar block** (including `with st.sidebar:`)
2. **Replace with one line**: `render_simplified_sidebar()`
3. **Delete all the card rendering loops** (the ones with `build_clean_card()`)
4. **Replace with**: `render_native_recommendation_row()` in a simple for loop
5. **Test**: `streamlit run stock_scout.py`

---

## Imports Already Done ✅

The necessary imports are already in stock_scout.py:
```python
from ui_redesign import (
    render_simplified_sidebar,
    render_native_recommendation_row,
)
```

No need to add anything to the top of the file.

---

## After Making Changes

```bash
# Test it compiles
python3 -m py_compile stock_scout.py

# Run the app
streamlit run stock_scout.py

# What you should see:
# ✅ NO sidebar
# ✅ Controls at top of page
# ✅ Clean recommendation rows (no raw HTML)
# ✅ Same scores as before (backend unchanged)
```

---

## If Something Goes Wrong

```bash
# Revert everything
git checkout stock_scout.py

# Start over
```

---

## Line Numbers (Approximate)

- **Sidebar block starts**: ~line 4386
- **Sidebar block ends**: ~line 4523 (rough range)
- **Card rendering starts**: ~line 4916
- **Card rendering ends**: ~line 5650+ (rough range)

Use Ctrl+F to search for:
- `with st.sidebar:` (sidebar start)
- `for _, r in core_df.iterrows():` (card loop start)
- `build_clean_card(r, speculative=False)` (HTML rendering)

---

## That's It!

Two replacements, test, done. The new UI is much cleaner and everything still works the same way behind the scenes.

Good luck! 🚀
