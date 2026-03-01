import streamlit as st
from typing import Tuple


def render_top_control_bar(locale: str = "he") -> Tuple[int, str, float]:
    """
    Render the top control bar with universe selector, investment style,
    capital input, and run button.  Modern 2026 compact layout.
    Returns: (universe_size, style, capital)
    """
    if locale == "he":
        col1, col2, col3, col4 = st.columns([1.3, 1.0, 1.0, 0.7])
        with col1:
            universe = st.selectbox(
                "יקום מניות",
                options={
                    "S&P 500 (ברירת מחדל)": 500,
                    "רשימה שלי": 20,
                    "אנרגיה + הגנה": 100,
                },
                index=0,
            )
        with col2:
            style = st.selectbox("סגנון השקעה", ["Core", "Speculative"], index=0)
        with col3:
            capital = st.number_input(
                "הון להשקעה ($)", min_value=1000, max_value=1_000_000, value=10_000
            )
        with col4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button("🔍 הרץ סריקה", type="primary", use_container_width=True)
        return universe, style, capital
    else:
        col1, col2, col3, col4 = st.columns([1.3, 1.0, 1.0, 0.7])
        with col1:
            universe = st.selectbox(
                "Universe",
                options={
                    "S&P 500 (Default)": 500,
                    "My List": 20,
                    "Energy + Defense": 100,
                },
                index=0,
            )
        with col2:
            style = st.selectbox("Investment Style", ["Core", "Speculative"], index=0)
        with col3:
            capital = st.number_input(
                "Capital ($)", min_value=1000, max_value=1_000_000, value=10_000
            )
        with col4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button("🔍 Run Scan", type="primary", use_container_width=True)
        return universe, style, capital
