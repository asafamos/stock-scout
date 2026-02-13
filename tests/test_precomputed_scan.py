import pandas as pd
import numpy as np
from core.scan_io import load_precomputed_scan_with_fallback

def test_precomputed_scan_missing():
    # Simulate missing file
    df, meta, path = load_precomputed_scan_with_fallback("/tmp/nonexistent_scan_dir_xyz")
    assert df is None
    assert meta is None

# Add more edge cases as needed
