#!/usr/bin/env python3
"""Quick diagnostic: Show how to configure Streamlit secrets properly."""

import streamlit as st
import os

st.write("# 🔍 Secrets Debug")

# Check dotenv first
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv(usecwd=True)
st.write(f"**Dotenv path found:** `{dotenv_path if dotenv_path else 'NOT FOUND'}`")
if dotenv_path:
    load_dotenv(dotenv_path)

st.write("---")
st.write("## Environment Variables (os.getenv)")
for key in ["OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY", 
            "POLYGON_API_KEY", "TIINGO_API_KEY", "FMP_API_KEY"]:
    val = os.getenv(key)
    st.write(f"- **{key}**: {'✅ SET (len=' + str(len(val)) + ')' if val else '❌ MISSING'}")

st.write("---")
st.write("## Streamlit Secrets")

if not hasattr(st, 'secrets'):
    st.error("❌ `st.secrets` not available!")
else:
    st.success("✅ `st.secrets` exists")
    
    # Try to show structure
    try:
        st.write("### Top-level keys:")
        # Convert to dict for inspection
        secrets_dict = dict(st.secrets)
        for k in secrets_dict.keys():
            val = secrets_dict[k]
            if isinstance(val, dict):
                st.write(f"- **[{k}]** (section with {len(val)} keys)")
                for subkey in val.keys():
                    subval = val[subkey]
                    st.write(f"  - `{subkey}`: len={len(str(subval))}")
            else:
                st.write(f"- **{k}**: len={len(str(val))}")
    except Exception as e:
        st.warning(f"Cannot list secrets: {e}")
        st.write("### Direct key check:")
        for key in ["OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY"]:
            try:
                val = st.secrets[key]
                st.write(f"- `st.secrets['{key}']`: ✅ len={len(str(val))}")
            except Exception:
                st.write(f"- `st.secrets['{key}']`: ❌ NOT FOUND")

st.write("---")
st.write("## _env() helper test")

from stock_scout import _env

for key in ["OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY"]:
    val = _env(key)
    st.write(f"- `_env('{key}')`: {'✅ len=' + str(len(val)) if val else '❌ None'}")

st.write("---")
st.write("## ✅ Expected Secrets Format")
st.code("""
# In Streamlit Cloud settings:
# Either flat (top-level):

OPENAI_API_KEY = "sk-..."
ALPHA_VANTAGE_API_KEY = "ABC123..."
FINNHUB_API_KEY = "xyz..."
POLYGON_API_KEY = "..."
TIINGO_API_KEY = "..."
FMP_API_KEY = "..."

# OR nested under [api_keys]:

[api_keys]
OPENAI_API_KEY = "sk-..."
ALPHA_VANTAGE_API_KEY = "ABC123..."
FINNHUB_API_KEY = "xyz..."
POLYGON_API_KEY = "..."
TIINGO_API_KEY = "..."
FMP_API_KEY = "..."
""", language="toml")
