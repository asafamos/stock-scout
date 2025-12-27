"""
UI Redesign Module - Modern Layout Helpers

This module contains Streamlit UI components for the 2025 redesign:
- Top control bar (universe, investment style, capital, run button)
- Native Streamlit recommendation rows (replacing HTML cards)
- Tab-based layout (Overview, Top Picks, All Results)
- Simplified sidebar with only essential controls
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime


def render_top_control_bar() -> Tuple[str, str, float]:
    """
    Render the top control bar with:
    - Universe selector
    - Investment style selector
    - Capital input
    - Run Scan button
    
    Returns: (selected_universe, selected_style, capital_amount)
    """
    st.markdown("### ⚙️ Quick Setup")
    
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])
    
    with col1:
        universe = st.selectbox(
            "Universe",
            options=[
                "S&P 500 (default)",
                "My watchlist",
                "Energy + Defense focus",
            ],
            index=0,
            help="Select the stock universe to scan"
        )
        # Map to internal universe size
        universe_size_map = {
            "S&P 500 (default)": 500,
            "My watchlist": 20,
            "Energy + Defense focus": 100,
        }
        st.session_state["universe_size"] = universe_size_map.get(universe, 500)
    
    with col2:
        investment_style = st.selectbox(
            "Style",
            options=["Conservative", "Balanced", "Aggressive"],
            index=1,
            help="Risk tolerance and allocation strategy"
        )
        # Map to allocation style
        style_idx_map = {
            "Conservative": 1,
            "Balanced": 0,
            "Aggressive": 2,
        }
        st.session_state["alloc_style_idx"] = style_idx_map.get(investment_style, 0)
    
    with col3:
        capital = st.number_input(
            "Capital ($)",
            min_value=100.0,
            max_value=1000000.0,
            value=float(st.session_state.get("total_budget", 5000.0)),
            step=500.0,
        )
        st.session_state["total_budget"] = float(capital)
    
    with col4:
        st.markdown("")  # Spacer for alignment
        if st.button("▶️ Run Scan", width='stretch', type="primary"):
            st.session_state["run_scan_clicked"] = True
            st.rerun()
    
    return universe, investment_style, capital


def render_view_mode_selector() -> Tuple[str, list]:
    """
    Render view mode selector with:
    - Top picks view / All results view (radio)
    - Show Core / Show Speculative (checkboxes)
    
    Returns: (view_mode, visible_risk_levels)
    """
    st.markdown("### 👁️ View & Filter")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        view_mode = st.radio(
            "View mode",
            options=["Top picks", "All results"],
            index=0,
            help="Top picks shows only the highest-scoring recommendations"
        )
    
    with col2:
        risk_types = st.multiselect(
            "Risk levels to show",
            options=["Core", "Speculative"],
            default=["Core", "Speculative"],
            help="Toggle which risk categories to display"
        )
        # Convert display names to internal names
        risk_levels = [
            "core" if r == "Core" else "speculative"
            for r in risk_types
        ]
    
    return view_mode, risk_levels


def render_native_recommendation_row(row: pd.Series, rank: int) -> None:
    """
    Render a single recommendation as a native Streamlit row (not HTML).
    
    Layout:
    - Header: Ticker, Badge (CORE/SPEC), Score
    - Pricing: Entry, Target, Upside
    - ML: Win probability
    - Reasons: 3-4 short bullet points
    - Expander: Full metrics
    """
    ticker = row.get("Ticker", "N/A")
    score = row.get("Score", np.nan)
    score_fmt = f"{float(score):.0f}" if np.isfinite(score) else "N/A"
    
    # Extract key values
    entry_price = row.get("Price_Yahoo", row.get("Unit_Price", np.nan))
    target_price = row.get("Target_Price", np.nan)
    ml_prob = row.get("ML_20d_Prob", row.get("ML_Probability", np.nan))
    risk_level = row.get("Risk_Level", "N/A").upper()
    upside_pct = row.get("Upside_Pct", np.nan)
    
    # Compute upside if not provided
    if np.isnan(upside_pct) if isinstance(upside_pct, float) else (upside_pct is None):
        if np.isfinite(entry_price) and np.isfinite(target_price) and entry_price > 0:
            upside_pct = ((target_price - entry_price) / entry_price) * 100
        else:
            upside_pct = np.nan
    
    upside_fmt = f"+{upside_pct:.1f}%" if np.isfinite(upside_pct) else "N/A"
    
    # Container for card-like appearance
    with st.container(border=True):
        # Header row: Ticker, Badge, Score
        header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
        
        with header_col1:
            st.markdown(f"## {ticker}")
        
        with header_col2:
            badge_color = "🛡️ CORE" if risk_level == "CORE" else "⚡ SPEC"
            st.caption(f"{badge_color} | Rank #{rank}")
        
        with header_col3:
            st.metric("Score", f"{score_fmt}/100")
        
        st.markdown("---")
        
        # Pricing & Upside row
        price_col1, price_col2, price_col3 = st.columns(3)
        
        with price_col1:
            entry_fmt = f"${entry_price:.2f}" if np.isfinite(entry_price) else "—"
            st.metric("Entry", entry_fmt)
        
        with price_col2:
            target_fmt = f"${target_price:.2f}" if np.isfinite(target_price) else "—"
            st.metric("Target", target_fmt)
        
        with price_col3:
            st.metric("Upside", upside_fmt, delta=None if upside_fmt == "N/A" else upside_fmt)
        
        st.markdown("---")
        
        # ML Probability (if available)
        if np.isfinite(ml_prob):
            ml_pct = float(ml_prob * 100)
            ml_col1, ml_col2 = st.columns([1, 2])
            with ml_col1:
                st.metric("ML Win Prob", f"{ml_pct:.1f}%")
            with ml_col2:
                st.markdown(
                    f"**ML 20D Model**: {ml_pct:.0f}% probability of positive return over 20 days"
                )
        
        st.markdown("---")
        
        # Quick reasons (bullet points)
        st.markdown("**Key Reasons:**")
        reasons = []
        
        # Extract key reason texts
        fund_score = row.get("Fundamental_S", np.nan)
        if np.isfinite(fund_score) and fund_score >= 60:
            reasons.append(f"Strong fundamentals ({fund_score:.0f})")
        
        rsi = row.get("RSI", np.nan)
        if np.isfinite(rsi) and rsi >= 65:
            reasons.append(f"Momentum building (RSI: {rsi:.0f})")
        
        quality = row.get("Quality_Level", "N/A")
        if quality != "N/A":
            reasons.append(f"Data quality: {quality}")
        
        rr = row.get("RR_Ratio", row.get("reward_risk_v2", np.nan))
        if np.isfinite(rr):
            reasons.append(f"Reward/Risk: {rr:.2f}x")
        
        if not reasons:
            reasons = ["Technical momentum", "Quality data available"]
        
        for reason in reasons[:4]:
            st.markdown(f"• {reason}")
        
        st.markdown("---")
        
        # Expandable details
        with st.expander("📊 Full Metrics"):
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.markdown("**Technical**")
                if np.isfinite(rsi):
                    st.caption(f"RSI: {rsi:.1f}")
                near_high = row.get("Near52w", np.nan)
                if np.isfinite(near_high):
                    st.caption(f"Near 52W High: {near_high:.1f}%")
                atr_pct = row.get("ATR_Price", np.nan)
                if np.isfinite(atr_pct):
                    st.caption(f"ATR/Price: {atr_pct:.2%}")
            
            with details_col2:
                st.markdown("**Fundamentals**")
                pe = row.get("PE_f", np.nan)
                if np.isfinite(pe):
                    st.caption(f"P/E: {pe:.1f}")
                roe = row.get("ROE_f", np.nan)
                if np.isfinite(roe):
                    st.caption(f"ROE: {roe:.1%}")
                debt_eq = row.get("DE_f", np.nan)
                if np.isfinite(debt_eq):
                    st.caption(f"D/E: {debt_eq:.2f}")
            
            # Data sources
            sources = row.get("Price_Sources", "Yahoo")
            st.caption(f"📡 Data: {sources}")
            
            # Allocation
            buy_amount = row.get("סכום קנייה ($)", row.get("buy_amount_v2", 0))
            shares = row.get("מניות לקנייה", 0)
            st.caption(f"💰 Allocation: {buy_amount:.0f}$ ({int(shares)} shares)")


def render_overview_tab(market_regime: dict, results: pd.DataFrame, core_count: int, spec_count: int) -> None:
    """
    Render the Overview tab with:
    - Market regime summary
    - Universe statistics
    - Filter results summary
    """
    st.markdown("### 📈 Market Overview")
    
    regime = market_regime.get("regime", "neutral").upper()
    confidence = market_regime.get("confidence", 50)
    regime_emoji = {"BULLISH": "📈", "NEUTRAL": "➡️", "BEARISH": "📉"}
    regime_color = {"BULLISH": "#10b981", "NEUTRAL": "#f59e0b", "BEARISH": "#dc2626"}
    
    # Market regime card
    st.markdown(
        f"""
        <div style='background:{regime_color.get(regime, "#6b7280")};color:white;padding:12px;border-radius:8px;margin:10px 0'>
        <strong>{regime_emoji.get(regime, "❓")} Market Regime: {regime}</strong><br>
        <small>Confidence: {confidence}% | {market_regime.get('details', '')}</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Universe Size", len(results) if not results.empty else 0)
    
    with col2:
        st.metric("Core Recommendations", core_count)
    
    with col3:
        st.metric("Speculative", spec_count)
    
    with col4:
        if not results.empty and "Score" in results.columns:
            median_score = float(results["Score"].median())
            st.metric("Median Score", f"{median_score:.0f}")
        else:
            st.metric("Median Score", "—")
    
    st.markdown("---")
    
    # Detailed summary
    st.markdown("### Summary")
    if not results.empty:
        st.write(
            f"""
            **Pipeline Results:**
            - Total recommendations: {len(results)}
            - Core (strict criteria): {core_count}
            - Speculative (relaxed criteria): {spec_count}
            - Average score: {float(results['Score'].mean()):.1f}/100
            
            **Market Context:**
            - Regime: {regime} (confidence {confidence}%)
            - {market_regime.get('details', 'No additional details available')}
            """
        )
    else:
        st.warning("No recommendations available. Try adjusting filters or checking market conditions.")


def render_top_picks_tab(results: pd.DataFrame, risk_levels: list) -> None:
    """
    Render Top Picks tab with best N recommendations.
    """
    st.markdown("### ⭐ Top Picks (Best 10)")
    
    if results.empty:
        st.info("No recommendations to display. Run scan with different parameters.")
        return
    
    # Filter by risk level
    filtered = results.copy()
    if risk_levels and "Risk_Level" in filtered.columns:
        filtered = filtered[filtered["Risk_Level"].isin(risk_levels)]
    
    if filtered.empty:
        st.warning("No stocks match the selected risk levels.")
        return
    
    # Show top 10 by score
    top_n = min(10, len(filtered))
    top_picks = filtered.head(top_n)
    
    for idx, (_, row) in enumerate(top_picks.iterrows(), 1):
        render_native_recommendation_row(row, rank=idx)
        st.markdown("")  # Small spacer


def render_all_results_tab(results: pd.DataFrame, risk_levels: list) -> None:
    """
    Render All Results tab with full table.
    """
    st.markdown("### 📊 Full Results")
    
    if results.empty:
        st.info("No recommendations to display.")
        return
    
    # Filter by risk level
    filtered = results.copy()
    if risk_levels and "Risk_Level" in filtered.columns:
        filtered = filtered[filtered["Risk_Level"].isin(risk_levels)]
    
    if filtered.empty:
        st.warning("No stocks match the selected risk levels.")
        return
    
    st.write(f"Showing {len(filtered)} stocks")
    
    # Select key columns for display
    display_columns = []
    for col in ["Ticker", "Price_Yahoo", "Score", "FinalScore", "ML_20d_Prob", 
                "TechScore_20d_v2", "Risk_Level", "Data_Quality"]:
        if col in filtered.columns:
            display_columns.append(col)
    
    if not display_columns:
        display_columns = list(filtered.columns[:8])  # Fallback to first 8 columns
    
    # Display table with formatting
    display_df = filtered[display_columns].copy()
    
    # Format numeric columns
    for col in display_df.columns:
        if col.endswith("_Prob") or col.endswith("_20d_v2") or col == "Score" or col == "FinalScore":
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1f}" if np.isfinite(x) else "—"
            )
        elif col.startswith("Price"):
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:.2f}" if np.isfinite(x) else "—"
            )
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.0f"),
            "FinalScore": st.column_config.NumberColumn(format="%.0f"),
        }
    )


def render_simplified_sidebar() -> None:
    """
    Render simplified sidebar with only essential controls.
    """
    with st.sidebar:
        st.title("📋 Settings")
        
        st.subheader("Advanced Options")
        
        with st.expander("🎛️ ML & Ranking", expanded=False):
            enable_ml = st.checkbox(
                "Enable ML 20d model",
                value=bool(st.session_state.get("ENABLE_ML", True)),
                help="ML model predicts 20-day positive returns"
            )
            st.session_state["ENABLE_ML"] = bool(enable_ml)
            
            sort_by_final = st.checkbox(
                "Sort by ML-aware score",
                value=bool(st.session_state.get("USE_FINAL_SCORE_SORT", True)),
                help="Use FinalScore combining technical + ML"
            )
            st.session_state["USE_FINAL_SCORE_SORT"] = bool(sort_by_final)
        
        with st.expander("🔬 Data & Quality", expanded=False):
            multi_source = st.checkbox(
                "Multi-source fundamentals",
                value=bool(st.session_state.get("enable_multi_source", True)),
                help="Fetch fundamentals from multiple providers"
            )
            st.session_state["enable_multi_source"] = bool(multi_source)
            
            show_sources = st.checkbox(
                "Show data attribution",
                value=bool(st.session_state.get("show_debug_attr", False)),
                help="Display which providers supplied each value"
            )
            st.session_state["show_debug_attr"] = bool(show_sources)
        
        with st.expander("⚙️ Developer", expanded=False):
            fast_mode = st.checkbox(
                "Fast mode (skip slow providers)",
                value=bool(st.session_state.get("FAST_MODE", False)),
                help="Reduce external API calls for faster scanning"
            )
            st.session_state["FAST_MODE"] = bool(fast_mode)
            
            ml_gating = st.checkbox(
                "ML Top-5% gating",
                value=bool(st.session_state.get("use_ml_gating", False)),
                help="Only show stocks in ML top 5 percentile"
            )
            st.session_state["use_ml_gating"] = bool(ml_gating)
        
        st.markdown("---")
        st.caption("📌 **Disclaimer**: For research only. Not investment advice.")
