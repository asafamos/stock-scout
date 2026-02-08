import streamlit as st

def apply_rtl():
    """Apply global RTL and Hebrew styling to the page."""
    st.markdown("""
    <style>
        /* Global RTL */
        html, body, * { direction: rtl; }
        .main { direction: rtl; }
        .stSidebar { display: none; }
        h1, h2, h3, h4, h5, h6, p, span, div, .stMetric, .stWrite { text-align: right; }
        .ltr { direction: ltr; }
        .stContainer { border-radius: 8px; }
        .stButton > button { border-radius: 8px; font-weight: 500; }
        h1, h2, h3 { margin-top: 1.5rem; margin-bottom: 0.5rem; }
    </style>
    """)
