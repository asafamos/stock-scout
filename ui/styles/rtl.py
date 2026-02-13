import streamlit as st

def apply_rtl():
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
        /* Hide sidebar if needed */
        .stSidebar { display: none; }
        /* Container styling */
        .stContainer { border-radius: 8px; }
        .stButton > button { border-radius: 8px; font-weight: 500; }
        h1, h2, h3 { margin-top: 1.5rem; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)
