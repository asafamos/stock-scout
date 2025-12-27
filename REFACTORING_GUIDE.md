# REFACTORING GUIDE: UI/UX Redesign Implementation
# This file contains the specific changes to make stock_scout.py modern

"""
KEY CHANGES TO stock_scout.py:

1. AT TOP OF FILE - Add new import:
   from ui_redesign import (
       render_top_control_bar,
       render_view_mode_selector,
       render_native_recommendation_row,
       render_overview_tab,
       render_top_picks_tab,
       render_all_results_tab,
       render_simplified_sidebar,
   )

2. REMOVE IMPORTS:
   - Remove: from card_styles import get_card_css
   - This is no longer needed with native Streamlit

3. DELETE THE build_clean_card() FUNCTION (lines ~123-350)
   - It's entirely replaced by render_native_recommendation_row()
   - Backend logic unaffected

4. REMOVE st.set_page_config CSS styling:
   Delete or simplify the st.markdown('<style>...') block that adds custom CSS
   Keep the page_config call, just remove heavy CSS

5. MAJOR: REPLACE SIDEBAR-HEAVY LAYOUT:
   
   OLD APPROACH (around line 4386):
   ```python
   with st.sidebar:
       st.header("🎛️ Scan Controls")
       [many sidebar controls]
   
   # Filters and display options
   risk_filter = st.multiselect(...)
   quality_filter = st.multiselect(...)
   ```
   
   NEW APPROACH:
   ```python
   # TOP CONTROL BAR (right after title)
   st.markdown("---")
   universe, style, capital = render_top_control_bar()
   
   # Simplified sidebar (only advanced options)
   render_simplified_sidebar()
   
   # View mode selector
   st.markdown("---")
   view_mode, risk_levels = render_view_mode_selector()
   ```

6. REMOVE ALL HTML CARD RENDERING CODE:
   
   Delete sections like:
   ```python
   # Line ~5554 and ~5850: "Display Core recommendations"
   for _, r in core_df.iterrows():
       card_html = build_clean_card(r, speculative=False)
       st.markdown(f"<div class=\"recommend-card\">{card_html}</div>", unsafe_allow_html=True)
   
   # Line ~5900+: "Display Speculative recommendations"
   for _, r in spec_df.iterrows():
       card_html = build_clean_card(r, speculative=True)
       st.markdown(f"<div class=\"recommend-card\">{card_html}</div>", unsafe_allow_html=True)
   ```

7. REPLACE WITH TABBED LAYOUT:
   
   NEW CODE (replacing old HTML card display):
   ```python
   st.markdown("---")
   st.subheader("📈 Recommendations")
   
   # Separate core and spec
   core_df = rec_df[rec_df["Risk_Level"] == "core"] if "Risk_Level" in rec_df.columns else rec_df
   spec_df = rec_df[rec_df["Risk_Level"] == "speculative"] if "Risk_Level" in rec_df.columns else pd.DataFrame()
   core_count = len(core_df)
   spec_count = len(spec_df)
   
   # Create tabs
   tab1, tab2, tab3 = st.tabs(["📊 Overview", "⭐ Top Picks", "📋 All Results"])
   
   with tab1:
       render_overview_tab(market_regime_data, rec_df, core_count, spec_count)
   
   with tab2:
       render_top_picks_tab(rec_df, risk_levels)
   
   with tab3:
       render_all_results_tab(rec_df, risk_levels)
   ```

8. DELETE STYLE INJECTIONS:
   Remove all lines that inject CSS:
   - st.markdown(get_card_css(), unsafe_allow_html=True)
   - These were at ~line 5225 and ~5899

9. REMOVE DEBUG/EXPERIMENTAL UI:
   Delete these visible controls:
   - "Enable ML model" big toggle (keep in Advanced expander only)
   - "Debug", "Skip pipeline", "Check secrets", "Clear cache" buttons
   - Any visible "[DEBUG]" or "DEVELOPMENT" markers

10. KEEP THESE BACKEND SYSTEMS INTACT:
    - Data loading (fetch_history_bulk, build_universe)
    - Scoring (compute_technical_score, score_with_ml_model)
    - ML models (ML_20D_AVAILABLE, predict_20d_prob_from_row)
    - Allocation (allocate_budget)
    - CSV export (unchanged)
    - All fundamentals pipeline
    - All risk engines


TESTING CHECKLIST AFTER REFACTORING:

1. Syntax: python3 -m py_compile stock_scout.py ✓
2. Import check: ui_redesign module imports cleanly ✓
3. Visual test (local):
   - streamlit run stock_scout.py
   - Check top control bar renders
   - Check sidebar is simplified
   - Check tabs display Overview, Top Picks, All Results
   - Check recommendation rows show (not HTML)
4. Functional test:
   - Select universe → should update session state
   - Select investment style → allocation should change
   - Enter capital → updates budget
   - Run Scan button → re-executes
5. Data integrity:
   - Scores identical to before refactoring
   - CSV export has same columns/data
   - Rankings unchanged
   - ML probabilities same
6. Backwards compatibility:
   - Session state keys preserved
   - No API changes to data functions
   - CONFIG dictionary untouched


MIGRATION PATH (if deploying):

1. Test refactored code locally first
2. Create new git branch: git checkout -b ui-redesign
3. Replace stock_scout.py with refactored version
4. Add ui_redesign.py module
5. Test end-to-end on Streamlit Cloud
6. Compare outputs with previous version (scores, rankings)
7. Merge to main once verified


PERFORMANCE EXPECTATIONS:

- Faster rendering (no HTML/CSS injection)
- Reduced page size (no inline <style> blocks)
- Cleaner DOM (native Streamlit components)
- Same computation time (logic unchanged)
- Slightly faster initial load (no CSS parsing)


BREAKDOWN OF WHAT STAYS VS CHANGES:

CHANGES:
✗ UI layout and structure
✗ Card rendering system
✗ Sidebar organization
✗ CSS styling approach
✗ Debug UI elements

STAYS:
✓ All data loading functions
✓ All scoring algorithms
✓ ML model integration
✓ CSV export schema
✓ Configuration system
✓ Session state variables
✓ API integrations
✓ Risk engines
✓ Fundamentals pipeline
✓ Allocation logic
✓ Numerical results (scores, ranks)


ESTIMATED EFFORT:

- Implementation: 2-3 hours
- Testing: 1 hour
- Validation: 30 minutes
- Total: ~4 hours

The refactoring is surgical: it only touches the presentation layer.
All backend logic and data processing remain 100% identical.
"""
