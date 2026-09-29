import warnings
from urllib3.exceptions import NotOpenSSLWarning
import sys, os

# Ensure project root is on path for module imports
PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

# Silence urllib3 NotOpenSSLWarning in CI/dev environments where LibreSSL is used
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)



# ── Never let a test message the owner ──────────────────────────────────────
# 2026-09-29: production code paths under test (e.g. the post-BUY-exception cleanup) call the REAL
# notifications._send, which posts to Telegram using the local secrets. A regression test for an
# "UNTRACKED POSITION PBF" scenario therefore sent five real "IB holds 8 sh of PBF — verify stops
# now" alerts to the owner's phone. Every test now runs with (a) _send stubbed and (b) any direct
# HTTP call to api.telegram.org turned into a loud test failure.
import pytest


@pytest.fixture(autouse=True)
def _no_real_telegram(monkeypatch):
    try:
        import core.trading.notifications as _n
        monkeypatch.setattr(_n, "_send", lambda *a, **k: True, raising=False)
    except Exception:
        pass

    def _blocked(url, *a, **k):
        raise AssertionError(f"test attempted a REAL Telegram call: {url}")

    try:
        import requests
        _orig_post, _orig_request = requests.post, requests.Session.request

        def _post(url, *a, **k):
            if "api.telegram.org" in str(url):
                _blocked(url)
            return _orig_post(url, *a, **k)

        def _request(self, method, url, *a, **k):
            if "api.telegram.org" in str(url):
                _blocked(url)
            return _orig_request(self, method, url, *a, **k)

        monkeypatch.setattr(requests, "post", _post)
        monkeypatch.setattr(requests.Session, "request", _request)
    except Exception:
        pass
    try:
        import urllib.request as _u
        _orig_open = _u.urlopen

        def _urlopen(url, *a, **k):
            target = getattr(url, "full_url", url)
            if "api.telegram.org" in str(target):
                _blocked(target)
            return _orig_open(url, *a, **k)

        monkeypatch.setattr(_u, "urlopen", _urlopen)
    except Exception:
        pass
    yield
