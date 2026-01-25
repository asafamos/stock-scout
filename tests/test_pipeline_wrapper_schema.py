import pandas as pd

from core.pipeline_runner import run_scan_smoke


def test_wrapper_schema_is_named_dict():
    out = run_scan_smoke()
    assert isinstance(out, dict)
    assert set(["result", "meta"]).issubset(set(out.keys()))
    payload = out.get("result")
    assert isinstance(payload, dict)
    assert set(["results_df", "data_map", "diagnostics"]).issubset(set(payload.keys()))
    df = payload.get("results_df")
    dm = payload.get("data_map")
    diags = payload.get("diagnostics")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(dm, dict) or dm is None
    assert isinstance(diags, dict)
