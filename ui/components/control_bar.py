import streamlit as st
from typing import Tuple

def render_top_control_bar(locale: str = "he") -> Tuple[int, str, float]:
    """
    Render the top control bar with universe selector, investment style, capital input, and run button.
    Returns: (universe_size, style, capital)
    """
    if locale == "he":
        st.markdown("### ⚙️ הגדרות סריקה מהירות")
        col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1, 1, 0.8])
        with col1:
            universe = st.selectbox(
                "יקום מניות",
                options={
                    "S&P 500 (ברירת מחדל)": 500,
                    "רשימה שלי": 20,
                    "אנרגיה + הגנה": 100,
                },
                index=0
            )
        with col2:
            style = st.selectbox("סגנון השקעה", ["Core", "Speculative"], index=0)
        with col3:
            capital = st.number_input("הון להשקעה ($)", min_value=1000, max_value=1000000, value=10000)
        with col4:
            st.button("הרץ סריקה")
        return universe, style, capital
    else:
        st.markdown("### Quick Scan Settings")
        col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1, 1, 0.8])
        with col1:
            universe = st.selectbox(
                "Universe",
                options={
                    "S&P 500 (Default)": 500,
                    "My List": 20,
                    "Energy + Defense": 100,
                },
                index=0
            )
        with col2:
            style = st.selectbox("Investment Style", ["Core", "Speculative"], index=0)
        with col3:
            capital = st.number_input("Capital ($)", min_value=1000, max_value=1000000, value=10000)
        with col4:
            st.button("Run Scan")
        return universe, style, capital
