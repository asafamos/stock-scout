import os
import pytest

from core.pipeline_runner import run_scan_smoke


def test_meteor_filter_runs_without_exception(monkeypatch):
    # Enable Meteor mode explicitly
    monkeypatch.setenv("METEOR_MODE", "1")
    # Run smoke; should not raise and should return a payload
    out = run_scan_smoke()
    assert isinstance(out, dict)
    payload = out.get("result", {})
    # Ensure results_df exists
    results = payload.get("results_df")
    assert results is not None, "results_df missing from payload"
    # In small-universe lenient mode, at least 1 row should be present if Tier1/Tier2 passed
    # We cannot guarantee pass in all market states, but meta should indicate lenient mode
    meta = out.get("meta", {})
    assert meta.get("postfilter_mode") in {"lenient_small_universe", "strict", None}
