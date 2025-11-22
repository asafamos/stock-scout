#!/usr/bin/env python3
"""Test beta calculation with new timeout implementation."""

import streamlit as st
import numpy as np
from stock_scout import fetch_beta_vs_benchmark

st.write("# 🧪 Beta Calculation Test")

st.write("Testing new ThreadPoolExecutor timeout (cloud-safe)...")

with st.spinner("Fetching beta for AAPL vs SPY..."):
    beta = fetch_beta_vs_benchmark("AAPL", "SPY", 252)

if np.isnan(beta):
    st.warning(f"⚠️ Beta calculation returned NaN (timeout or data issue)")
else:
    st.success(f"✅ Beta calculated: **{beta:.3f}**")

st.write("---")
st.write("Testing with invalid ticker (should timeout quickly)...")

with st.spinner("Fetching beta for INVALID vs SPY..."):
    beta_bad = fetch_beta_vs_benchmark("INVALID_TICKER_XYZ", "SPY", 252)

if np.isnan(beta_bad):
    st.success("✅ Invalid ticker handled correctly (NaN)")
else:
    st.error(f"❌ Unexpected beta value: {beta_bad}")

st.write("---")
st.info("✅ Test complete! If both tests pass, beta calculation is cloud-safe.")
