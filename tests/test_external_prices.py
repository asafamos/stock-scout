import numpy as np
from core.providers.base import _http_get_with_retry

def test_external_price_provider_timeout():
    # Simulate timeout (should handle gracefully)
    try:
        _http_get_with_retry("http://example.com/timeout", timeout=0.001)
    except Exception:
        assert True
    else:
        assert False, "Should raise exception on timeout"
