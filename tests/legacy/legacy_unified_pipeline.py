"""Legacy unified pipeline verification script."""
from core.pipeline_runner import run_scan
from core.serialization import scanresult_to_dataframe

sr = run_scan(["AAPL","MSFT","GOOGL","NVDA","TSLA"], {"beta_filter_enabled": False})
results = scanresult_to_dataframe(sr)
print(f"Rows: {len(results)}")
