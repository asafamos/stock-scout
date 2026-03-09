"""
Hebrew RTL UI Module - Complete replacement for sidebar-based layout

This module provides the complete modern Hebrew RTL UI for stock_scout.py:
1. Top control bar (universe, style, capital, run button)
2. Removes all sidebar controls and moves to main page
3. Replaces HTML cards with native Streamlit components
4. Hebrew labeling throughout
5. RTL styling applied globally
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from datetime import datetime


def setup_hebrew_rtl():
    """Apply global RTL and Hebrew styling to the page."""
    st.markdown("""
    <style>
        /* RTL only for main content, not global */
        .main, .stApp, .stContainer, .stMarkdown, .stDataFrame, .stTable, .stMetric, .stWrite, .stText, .stCaption {
            direction: rtl;
            text-align: right;
        }
        /* LTR for code, input, numbers */
        input, textarea, code, pre, .ltr, .stTextInput, .stNumberInput, .stSlider, .stSelectbox, .stRadio, .stCheckbox {
            direction: ltr;
            unicode-bidi: embed;
            text-align: left;
        }
        .stSidebar { display: none; }  /* Hide sidebar completely */
        /* Container styling */
        .stContainer { border-radius: 8px; }
        .stButton > button { border-radius: 8px; font-weight: 500; }
        h1, h2, h3 { margin-top: 1.5rem; margin-bottom: 0.5rem; }
        /* Data source badges */
        .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.875rem; margin: 0 0.25rem; }
        .badge-core { background: #d1d5db; color: #111827; }
        .badge-spec { background: #f97316; color: white; }
    </style>
    """, unsafe_allow_html=True)


def render_top_control_bar() -> Tuple[int, str, float]:
    """
    Render the top control bar with:
    - Universe selector
    - Investment style
    - Capital input
    - Run button
    
    Returns: (universe_size, style, capital)
    """
    st.markdown("### ⚙️ הגדרות סריקה מהירות")
    
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1, 1, 0.8])
    
    with col1:
        universe = st.selectbox(
            "יקום מניות",
            options={
                "S&P 500 (ברירת מחדל)": 500,
                "רשימה שלי": 20,
                "אנרגיה + הגנה": 100,
            }.keys(),
            index=0,
        )
        universe_map = {
            "S&P 500 (ברירת מחדל)": 500,
            "רשימה שלי": 20,
            "אנרגיה + הגנה": 100,
        }
        universe_size = universe_map[universe]
    
    with col2:
        style = st.selectbox(
            "סגנון השקעה",
            options=["שמרני", "מאוזן", "אגרסיבי"],
            index=1,
        )
        style_map = {"שמרני": "conservative", "מאוזן": "balanced", "אגרסיבי": "aggressive"}
    
    with col3:
        capital = st.number_input(
            "הון ($)",
            min_value=100.0,
            max_value=1000000.0,
            value=float(st.session_state.get("total_budget", 5000.0)),
            step=500.0,
        )
    
    with col4:
        st.markdown("")  # Spacer
    
    with col5:
        if st.button("▶️ הרץ", width='stretch', type="primary"):
            st.session_state["run_scan"] = True
            st.rerun()
    
    return universe_size, style_map.get(style, "balanced"), float(capital)


def render_hebrew_sidebar_expander(config: Dict) -> Dict:
    """
    Render advanced options as a Hebrew sidebar expander on the main page.
    
    Returns: updated config dict with user selections
    """
    with st.expander("🎛️ אפשרויות מתקדמות", expanded=False):
        st.markdown("**ML והדירוג**")
        col1, col2 = st.columns(2)
        
        with col1:
            enable_ml = st.checkbox(
                "הפעל ML 20d",
                value=True,
                help="מודל ML חוזה תשואות חיוביות ל-20 ימים",
            )
        
        with col2:
            use_final_score = st.checkbox(
                "דירוג לפי FinalScore",
                value=True,
                help="שילוב של טכני + ML",
            )
        
        st.markdown("---")
        st.markdown("**נתונים ובחירות**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            multi_source = st.checkbox(
                "ממקורות מרובים",
                value=True,
                help="צבירת יסודות מכמה ספקים",
            )
        
        with col4:
            show_attribution = st.checkbox(
                "הראה ייחוס",
                value=False,
                help="הצג מהם באו הנתונים",
            )
        
        st.markdown("---")
        st.markdown("**מצב מהיר vs עמוק**")
        
        fast_mode = st.checkbox(
            "מצב מהיר (דלג על ספקים איטיים)",
            value=False,
            help="צמצום שיחות API למצב מהיר",
        )
        
        return {
            "enable_ml": enable_ml,
            "use_final_score": use_final_score,
            "multi_source": multi_source,
            "show_attribution": show_attribution,
            "fast_mode": fast_mode,
        }


def render_view_controls() -> Tuple[str, List[str], float, float]:
    """
    Render view and filter controls on the main page.
    
    Returns: (view_mode, risk_levels_to_show, min_score, max_rsi)
    """
    st.markdown("### 👁️ תצוגה ותראייה")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        view_mode = st.radio(
            "מצב תצוגה",
            options=["הבחירות המובילות", "כל התוצאות"],
            index=0,
        )
    
    with col2:
        risk_levels = st.multiselect(
            "סוגי סיכון להצגה",
            options=["ליבה", "ספקולטיבי"],
            default=["ליבה", "ספקולטיבי"],
        )
        risk_levels = [
            "core" if x == "ליבה" else "speculative" for x in risk_levels
        ]
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        min_score = st.slider(
            "ניקוד מינימום",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
        )
    
    with col4:
        max_rsi = st.slider(
            "RSI מקסימום",
            min_value=0,
            max_value=100,
            value=80,
            step=5,
        )
    
    return view_mode, risk_levels, float(min_score), float(max_rsi)


def render_recommendation_row_hebrew(row: pd.Series, rank: int) -> None:
    """
    Render a single recommendation using pure Streamlit (no HTML).
    """
    from core.scoring_config import get_canonical_score
    ticker = row.get("Ticker", "N/A")
    score = get_canonical_score(row)
    risk_level = row.get("Risk_Level", "core").lower()
    from ui.card_helpers import get_entry_price
    price_entry = get_entry_price(row)
    price_target = row.get("Target_Price", np.nan)
    ml_prob = row.get("ML_20d_Prob", row.get("ML_Probability", np.nan))
    
    # Compute upside %
    upside = np.nan
    if np.isfinite(price_entry) and np.isfinite(price_target) and price_entry > 0:
        upside = ((price_target - price_entry) / price_entry) * 100
    
    # Container for card
    with st.container(border=True):
        # Header
        h_col1, h_col2, h_col3 = st.columns([0.5, 2, 1])
        
        with h_col1:
            st.markdown(f"### #{rank}")
        
        with h_col2:
            badge = "🛡️ ליבה" if risk_level == "core" else "⚡ ספקולטיבי"
            st.markdown(f"## <span dir='ltr'>{ticker}</span> {badge}", unsafe_allow_html=True)
        
        with h_col3:
            score_val = f"{float(score):.0f}" if np.isfinite(score) else "—"
            st.metric("ציון", f"{score_val}/100")
        
        st.divider()
        
        # Pricing
        p_col1, p_col2, p_col3 = st.columns(3)
        
        with p_col1:
            entry_str = f"${price_entry:.2f}" if np.isfinite(price_entry) else "—"
            st.metric("כניסה", entry_str)
        
        with p_col2:
            target_str = f"${price_target:.2f}" if np.isfinite(price_target) else "—"
            st.metric("יעד", target_str)
        
        with p_col3:
            upside_str = f"+{upside:.1f}%" if np.isfinite(upside) else "—"
            st.metric("עלייה", upside_str)
        
        st.divider()
        
        # ML Win Probability
        if np.isfinite(ml_prob) and ml_prob > 0:
            ml_pct = float(ml_prob * 100)
            ml_col1, ml_col2 = st.columns([0.7, 2])
            
            with ml_col1:
                st.metric("ML ניצחון", f"{ml_pct:.0f}%")
            
            with ml_col2:
                st.markdown(f"**מודל ML 20d**: {ml_pct:.0f}% הסתברות לתשואה חיובית ב-20 ימים")
        
        st.divider()
        
        # Quick reasons
        st.markdown("**הנימוקים העיקריים:**")
        
        reasons = []
        
        # Extract reasons
        fund_score = row.get("Fundamental_S", row.get("fundamental_score", np.nan))
        if np.isfinite(fund_score) and fund_score >= 60:
            reasons.append(f"יסודות חזקים ({fund_score:.0f})")
        
        rsi = row.get("RSI", np.nan)
        if np.isfinite(rsi) and rsi >= 65:
            reasons.append(f"מומנטום עולה (RSI: {rsi:.0f})")
        
        rr = row.get("RR_Ratio", row.get("reward_risk_v2", np.nan))
        if np.isfinite(rr) and rr >= 1.5:
            reasons.append(f"די לטוב/סיכון: {rr:.2f}x")
        
        quality = row.get("Quality_Level", row.get("Data_Quality", ""))
        if quality:
            q_str = "גבוהה" if "high" in str(quality).lower() else "בינונית" if "medium" in str(quality).lower() else ""
            if q_str:
                reasons.append(f"איכות נתונים: {q_str}")
        
        if not reasons:
            reasons = ["טכני חזק", "נתונים זמינים"]
        
        for reason in reasons[:4]:
            st.markdown(f"• {reason}")
        
        st.divider()
        
        # Details expander
        with st.expander("📊 מדדים מלאים"):
            det_col1, det_col2 = st.columns(2)
            
            with det_col1:
                st.markdown("**טכני**")
                if np.isfinite(rsi):
                    st.caption(f"RSI: {rsi:.1f}")
                
                near_high = row.get("Near52w", np.nan)
                if np.isfinite(near_high):
                    st.caption(f"קרוב לשיא 52 שבועות: {near_high:.1f}%")
                
                atr = row.get("ATR_Price", np.nan)
                if np.isfinite(atr):
                    st.caption(f"ATR/Price: {atr:.2%}")
            
            with det_col2:
                st.markdown("**יסודות**")
                
                pe = row.get("PE_f", row.get("pe", np.nan))
                if np.isfinite(pe):
                    st.caption(f"P/E: {pe:.1f}")
                
                roe = row.get("ROE_f", row.get("roe", np.nan))
                if np.isfinite(roe):
                    st.caption(f"ROE: {roe:.1%}")
                
                de = row.get("DE_f", row.get("debt_equity", np.nan))
                if np.isfinite(de):
                    st.caption(f"D/E: {de:.2f}")
            
            # Data sources
            sources = row.get("Price_Sources", row.get("Source_List", "Yahoo"))
            st.caption(f"📡 מקורות נתונים: {sources}")
            
            # Allocation (prefer v2 fields; show 'לא זמין' when missing/non-positive)
            buy_amt = row.get("buy_amount_v2", row.get("סכום קנייה ($)", None))
            shares = row.get("shares_to_buy_v2", row.get("מניות לקנייה", 0))
            try:
                buy_amt_val = float(buy_amt) if buy_amt is not None else float('nan')
            except Exception:
                buy_amt_val = float('nan')
            if np.isfinite(buy_amt_val) and buy_amt_val > 0:
                st.caption(f"💰 הקצאה: ${buy_amt_val:.0f} ({int(shares) if shares else 0} מניות)")
            else:
                st.caption("💰 הקצאה: לא זמין")


def render_core_section_hebrew(core_df: pd.DataFrame) -> None:
    """Render Core recommendations section in Hebrew."""
    if core_df.empty:
        st.info("❌ אין מניות ליבה זמינות כעת")
        return
    
    st.markdown("## 🛡️ מניות ליבה (סיכון נמוך יחסית)")
    st.caption(f"מניות עם יסודות חזקים וטכני יציב | {len(core_df)} מניות")
    
    for rank, (_, row) in enumerate(core_df.iterrows(), 1):
        render_recommendation_row_hebrew(row, rank=rank)
        st.markdown("")  # Small spacer


def render_speculative_section_hebrew(spec_df: pd.DataFrame) -> None:
    """Render Speculative recommendations section in Hebrew."""
    if spec_df.empty:
        st.info("❌ אין מניות ספקולטיביות זמינות כעת")
        return
    
    st.markdown("## ⚡ מניות ספקולטיביות (פוטנציאל גבוה, סיכון גבוה)")
    st.caption(f"מניות עם תנופה טכנית חזקה | {len(spec_df)} מניות")
    
    for rank, (_, row) in enumerate(spec_df.iterrows(), 1):
        render_recommendation_row_hebrew(row, rank=rank)
        st.markdown("")  # Small spacer


def render_kpi_cards_hebrew(results: pd.DataFrame, core_count: int, spec_count: int) -> None:
    """Render KPI cards in Hebrew."""
    st.markdown("### 📊 סיכום הסריקה")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("סה\"כ מניות", len(results))
    
    with col2:
        st.metric("מניות ליבה", core_count)
    
    with col3:
        st.metric("ספקולטיביות", spec_count)
    
    with col4:
        if not results.empty and "Score" in results.columns:
            avg_score = float(results["Score"].mean())
            st.metric("ציון ממוצע", f"{avg_score:.1f}")
        else:
            st.metric("ציון ממוצע", "—")


def force_ml_and_sorting(results: pd.DataFrame) -> pd.DataFrame:
    """Force ML enabled and FinalScore sorting."""
    st.session_state["ENABLE_ML"] = True
    st.session_state["USE_FINAL_SCORE_SORT"] = True
    
    # If FinalScore exists, use it; otherwise use Score
    if "FinalScore" in results.columns:
        results = results.sort_values("FinalScore", ascending=False)
    elif "overall_score_20d" in results.columns:
        results = results.sort_values("overall_score_20d", ascending=False)
    elif "Score" in results.columns:
        results = results.sort_values("Score", ascending=False)
    
    return results.reset_index(drop=True)
