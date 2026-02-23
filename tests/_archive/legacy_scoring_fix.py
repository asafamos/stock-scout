#!/usr/bin/env python3
"""Legacy scoring fix manual runner (not a pytest test)."""
from core.pipeline_runner import run_scan
from core.serialization import scanresult_to_dataframe
from core.config import get_config

cfg = get_config()
sr = run_scan(["AAPL","MSFT","GOOGL","AMZN","NVDA"], cfg.__dict__ if hasattr(cfg,'__dict__') else cfg)
results = scanresult_to_dataframe(sr)
print(f"Rows: {len(results)}")
