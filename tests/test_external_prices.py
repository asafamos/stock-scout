import numpy as np
from core.providers.base import _http_get_with_retry


def test_external_price_provider_timeout():
    """_http_get_with_retry returns None on timeout / connection error
    (it never raises – errors are swallowed and logged)."""
    result = _http_get_with_retry("http://example.com/timeout", timeout=0.001)
    assert result is None, "Should return None on timeout"
