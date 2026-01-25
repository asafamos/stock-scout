from core.telemetry import Telemetry


def test_telemetry_export_contains_expected_keys():
    t = Telemetry()
    t.set_value('universe_provider', 'FMP')
    t.mark_used('price', 'POLYGON')
    t.mark_used('fundamentals', 'FINNHUB')
    t.mark_index('SPY', 'Polygon')
    t.record_fallback('index', 'FMP', 'Polygon', '403 forbidden')
    out = t.export()
    assert isinstance(out, dict)
    assert 'universe_provider' in out
    assert 'price' in out and isinstance(out['price'], dict)
    assert 'fundamentals' in out and isinstance(out['fundamentals'], dict)
    assert 'index' in out and isinstance(out['index'], dict)
    assert 'fallback_events' in out and isinstance(out['fallback_events'], list)
    assert out['universe_provider'] == 'FMP'
    assert out['price'].get('POLYGON') is True
    assert out['fundamentals'].get('FINNHUB') is True
    assert out['index'].get('SPY') == 'Polygon'
    assert len(out['fallback_events']) == 1
