"""FMP usage meter (2026-11-15).

Why: the FMP plan limit was exhausted on 2026-11-13 and we could not tell which of our ~10 FMP consumers
(scans, F1 collector, F2 research, VPS quotes, ...) caused it. This module counts every HTTP call to
financialmodelingprep.com made by a process — calls, HTTP status classes and response bytes, per endpoint
(query string and API key are NEVER recorded) — and writes one summary per process.

How: `install()` wraps `requests.Session.send` and `urllib.request.urlopen` (the two clients used in the repo).
It is idempotent, cheap (a host check per call) and never raises into the caller.

Output (best effort, never fatal)
- JSON line appended to data/state/fmp_usage.jsonl (override: FMP_METER_FILE) every ~10 min of activity and at exit.
- On GitHub Actions: a table appended to $GITHUB_STEP_SUMMARY and a `FMP-METER {...}` line on stdout.
- Report: `python -m scripts.fmp_usage_report [--days 7]`.
Disable with FMP_METER=0.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict
from urllib.parse import urlsplit

_HOST = "financialmodelingprep.com"
FLUSH_EVERY_S = 600
FLUSH_EVERY_CALLS = 500

_lock = threading.RLock()
_installed = False
_t0 = time.time()
_last_flush = time.time()
_calls_since_flush = 0
_by_ep: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "ok": 0, "e429": 0, "e_other": 0, "bytes": 0})


def _enabled() -> bool:
    return os.getenv("FMP_METER", "1").strip().lower() not in ("0", "false", "no", "off")


def _label() -> str:
    arg0 = sys.argv[0] if sys.argv and sys.argv[0] else "python"
    base = os.path.basename(arg0)
    return base if base not in ("-c", "") else "python-c"


def endpoint_of(url: str) -> str:
    """Path only, symbol/ids stripped: /stable/profile, /api/v3/historical-price-full ..."""
    try:
        parts = [p for p in urlsplit(url).path.split("/") if p]
    except ValueError:
        return "?"
    if not parts:
        return "/"
    keep = parts[:3] if parts[0] == "api" else parts[:2]
    return "/" + "/".join(keep)


def record(url: str, status: int | None, nbytes: int = 0) -> None:
    """Count one FMP call (status None = network error)."""
    global _calls_since_flush
    try:
        ep = endpoint_of(url)
        with _lock:
            d = _by_ep[ep]
            d["calls"] += 1
            d["bytes"] += int(nbytes or 0)
            if status is not None and 200 <= status < 300:
                d["ok"] += 1
            elif status == 429:
                d["e429"] += 1
            else:
                d["e_other"] += 1
            _calls_since_flush += 1
            if _calls_since_flush >= FLUSH_EVERY_CALLS or time.time() - _last_flush >= FLUSH_EVERY_S:
                flush()
    except Exception:  # metering must never break a caller
        pass


def snapshot() -> Dict[str, Any]:
    with _lock:
        eps = {k: dict(v) for k, v in _by_ep.items()}
    tot = {"calls": 0, "ok": 0, "e429": 0, "e_other": 0, "bytes": 0}
    for v in eps.values():
        for k in tot:
            tot[k] += v[k]
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label": _label(), "pid": os.getpid(),
            "window_s": int(time.time() - _t0), **tot, "by_endpoint": eps}


def flush() -> None:
    """Write the counters accumulated since the last flush, then reset them."""
    global _last_flush, _calls_since_flush
    with _lock:
        if not _by_ep:
            return
        snap = snapshot()
        _by_ep.clear()
        _last_flush = time.time()
        _calls_since_flush = 0
        _t_reset(snap)
    try:
        path = os.getenv("FMP_METER_FILE", "data/state/fmp_usage.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(snap) + "\n")
    except OSError:
        pass
    try:
        print("FMP-METER " + json.dumps({k: snap[k] for k in ("label", "calls", "ok", "e429", "e_other", "bytes")}),
              file=sys.stderr)
        summ = os.getenv("GITHUB_STEP_SUMMARY")
        if summ:
            with open(summ, "a") as f:
                f.write(f"\n**FMP usage — {snap['label']}**: {snap['calls']} calls, {snap['bytes'] / 1e6:.1f} MB, "
                        f"429s: {snap['e429']}\n\n| endpoint | calls | ok | 429 | other err | MB |\n|---|---|---|---|---|---|\n")
                for ep, v in sorted(snap["by_endpoint"].items(), key=lambda kv: -kv[1]["bytes"]):
                    f.write(f"| {ep} | {v['calls']} | {v['ok']} | {v['e429']} | {v['e_other']} | {v['bytes'] / 1e6:.2f} |\n")
    except OSError:
        pass


def _t_reset(_snap: Dict[str, Any]) -> None:
    global _t0
    _t0 = time.time()


# --------------------------------------------------------------------------- wrappers
def _wrap_requests() -> None:
    import requests

    orig = requests.sessions.Session.send

    def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        url = getattr(request, "url", "") or ""
        if _HOST not in url:
            return orig(self, request, **kwargs)
        try:
            resp = orig(self, request, **kwargs)
        except Exception:
            record(url, None)
            raise
        try:
            n = len(resp.content) if not kwargs.get("stream") else int(resp.headers.get("Content-Length") or 0)
        except Exception:
            n = 0
        record(url, resp.status_code, n)
        return resp

    requests.sessions.Session.send = send  # type: ignore[assignment]


class _CountingResponse:
    """Proxy for a urllib response that counts the bytes read."""

    def __init__(self, resp: Any, url: str) -> None:
        self._r = resp
        self._url = url
        self._n = 0
        self._done = False

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            record(self._url, getattr(self._r, "status", 200), self._n)

    def read(self, *a: Any, **k: Any):
        data = self._r.read(*a, **k)
        self._n += len(data or b"")
        if not a and not k:  # full read -> finished
            self._finish()
        return data

    def close(self) -> None:
        self._finish()
        self._r.close()

    def __enter__(self):
        self._r.__enter__()
        return self

    def __exit__(self, *exc: Any):
        self._finish()
        return self._r.__exit__(*exc)

    def __iter__(self):
        return iter(self._r)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._r, name)


def _wrap_urllib() -> None:
    import urllib.error
    import urllib.request

    orig = urllib.request.urlopen

    def urlopen(url, *a, **k):  # type: ignore[no-untyped-def]
        u = url if isinstance(url, str) else getattr(url, "full_url", "") or ""
        if _HOST not in u:
            return orig(url, *a, **k)
        try:
            resp = orig(url, *a, **k)
        except urllib.error.HTTPError as e:
            record(u, e.code, 0)
            raise
        except Exception:
            record(u, None)
            raise
        return _CountingResponse(resp, u)

    urllib.request.urlopen = urlopen  # type: ignore[assignment]


def install() -> None:
    """Idempotently start metering this process's FMP calls."""
    global _installed
    if _installed or not _enabled():
        return
    with _lock:
        if _installed:
            return
        try:
            _wrap_requests()
        except Exception:
            pass
        try:
            _wrap_urllib()
        except Exception:
            pass
        atexit.register(flush)
        _installed = True
