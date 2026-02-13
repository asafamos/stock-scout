"""
CSV and export utilities for Stock Scout (extracted from stock_scout.py)
"""
import pandas as pd
from typing import Optional, Dict

def export_to_csv(df: pd.DataFrame, filename: str, metadata: Optional[Dict] = None) -> None:
    """
    Export DataFrame to CSV, optionally with metadata as a header.
    """
    with open(filename, "w", encoding="utf-8") as f:
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}: {v}\n")
        df.to_csv(f, index=False)
