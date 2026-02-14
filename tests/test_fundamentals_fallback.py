import pytest
import stock_scout


@pytest.mark.skip(reason="stock_scout._fmp_full_bundle_fetch was removed during modular refactor; "
                          "fundamentals now handled by core.fundamental / core.data_sources_v2")
def test_per_field_fallback_order(monkeypatch):
    # Ensure environment keys appear present so fetch tasks are scheduled
    monkeypatch.setattr(stock_scout, '_env', lambda k: 'DUMMY')

    # Mock provider fetches
    def mock_fmp_full(ticker, api_key):
        return {'ps': 2.0, 'sector': 'Technology'}  # no 'pe'

    def mock_fmp_metrics(ticker, api_key):
        return {}

    def mock_finnhub(ticker):
        return {'pe': 8.0, 'gm': 0.30}

    def mock_alpha(ticker):
        return {'pe': 9.0, 'eps_g_yoy': 0.11}

    def mock_tiingo(ticker):
        return {'rev_g_yoy': 0.20}

    monkeypatch.setattr(stock_scout, '_fmp_full_bundle_fetch', mock_fmp_full)
    monkeypatch.setattr(stock_scout, '_fmp_metrics_fetch', mock_fmp_metrics)
    monkeypatch.setattr(stock_scout, '_finnhub_metrics_fetch', mock_finnhub)
    monkeypatch.setattr(stock_scout, '_alpha_overview_fetch', mock_alpha)
    monkeypatch.setattr(stock_scout, '_tiingo_fundamentals_fetch', mock_tiingo)
    monkeypatch.setattr(stock_scout, '_simfin_fetch', lambda t, k: {})
    monkeypatch.setattr(stock_scout, '_eodhd_fetch_fundamentals', lambda t, k: {})
    # Ensure Alpha is considered OK in session state so enable_alpha_smart path runs
    try:
        stock_scout.st.session_state['_alpha_ok'] = True
    except Exception:
        # If session_state not available as mapping, set attribute fallback
        setattr(stock_scout.st, 'session_state', {'_alpha_ok': True})

    merged = stock_scout.fetch_fundamentals_bundle('TST', enable_alpha_smart=True)

    # FMP provided PS
    assert merged['ps'] == 2.0
    assert merged['_sources'].get('ps') == 'FMP'

    # PE should come from Finnhub (fallback order: Finnhub -> Alpha -> Tiingo)
    assert merged['pe'] == 8.0
    assert merged['_sources'].get('pe') == 'Finnhub'

    # GM should be from Finnhub
    assert merged['gm'] == pytest.approx(0.30)
    assert merged['_sources'].get('gm') == 'Finnhub'

    # EPS growth should be from Alpha
    assert merged['eps_g_yoy'] == pytest.approx(0.11)
    assert merged['_sources'].get('eps_g_yoy') == 'Alpha'

    # Rev growth from Tiingo
    assert merged['rev_g_yoy'] == pytest.approx(0.20)
    assert merged['_sources'].get('rev_g_yoy') == 'Tiingo'


@pytest.mark.skip(reason="stock_scout._fmp_full_bundle_fetch was removed during modular refactor; "
                          "fundamentals now handled by core.fundamental / core.data_sources_v2")
def test_neutral_defaults_when_providers_respond_but_no_fields(monkeypatch):
    # _env present so tasks scheduled
    monkeypatch.setattr(stock_scout, '_env', lambda k: 'DUMMY')

    # Mock providers: fmp_full responds but with no numeric fields
    monkeypatch.setattr(stock_scout, '_fmp_full_bundle_fetch', lambda t, k: {'ok': True})
    monkeypatch.setattr(stock_scout, '_fmp_metrics_fetch', lambda t, k: {})
    monkeypatch.setattr(stock_scout, '_finnhub_metrics_fetch', lambda t: {})
    monkeypatch.setattr(stock_scout, '_alpha_overview_fetch', lambda t: {})
    monkeypatch.setattr(stock_scout, '_tiingo_fundamentals_fetch', lambda t: {})
    monkeypatch.setattr(stock_scout, '_simfin_fetch', lambda t, k: {})
    monkeypatch.setattr(stock_scout, '_eodhd_fetch_fundamentals', lambda t, k: {})

    merged = stock_scout.fetch_fundamentals_bundle('NOP', enable_alpha_smart=False)

    # Provider flags should show at least FMP full responded
    assert merged.get('from_fmp_full', False) is True
    assert 'from_fmp_full' in merged.get('_sources_used', [])

    # Since no valid numeric fields provided, defaults should be injected
    assert '_defaulted_fields' in merged
    defaults = merged['_defaulted_fields']
    # Expect key defaults for core fields
    for k in ['pe','ps','rev_g_yoy','eps_g_yoy','gm','de','oper_margin','roe']:
        assert k in defaults
        assert isinstance(merged[k], float)

    # Fund_Coverage_Pct should be floored to at least 0.05
    assert merged['Fund_Coverage_Pct'] >= 0.05
