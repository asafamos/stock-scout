import pandas as pd
import types

from core.data_sources_v2 import fetch_fundamentals_batch


def test_provider_selection_respects_guard(monkeypatch):
    # Stub prioritized fetch to two providers
    monkeypatch.setattr(
        "core.data_sources_v2.get_prioritized_fetch_funcs",
        lambda ps=None: [("fmp", object()), ("finnhub", object())]
    )
    # Collect chosen providers via stub aggregate
    chosen = {}
    def stub_agg(tkr, prefer_source, *args, **kwargs):
        chosen[tkr] = prefer_source
        return {"ticker": tkr}
    monkeypatch.setattr("core.data_sources_v2.aggregate_fundamentals", stub_agg)

    # Fake guard: block FMP, allow FINNHUB
    class FakeGuard:
        def allow(self, provider, capability):
            up = str(provider).upper()
            if up == "FMP":
                return (False, "cooldown", "BLOCK_COOLDOWN")
            return (True, "", "ALLOW")
    monkeypatch.setattr("core.data_sources_v2.get_provider_guard", lambda: FakeGuard())

    df = fetch_fundamentals_batch(["AAPL", "MSFT"], provider_status={})
    assert isinstance(df, pd.DataFrame)
    # Ensure prefer_source per ticker is FINNHUB (not FMP)
    assert chosen.get("AAPL") == "finnhub"
    assert chosen.get("MSFT") == "finnhub"
