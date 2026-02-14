"""Tests for core.pipeline sub-modules (config_norm, helpers, fallback_tracking)."""
import numpy as np
import pandas as pd
import pytest

from core.pipeline.config_norm import _normalize_config
from core.pipeline.helpers import (
    _to_float,
    _canon_column_name,
    _quantile_safe,
    check_earnings_blackout,
)
from core.pipeline.fallback_tracking import (
    _record_legacy_fallback,
    get_fallback_status,
    reset_fallback_state,
)


# ── _normalize_config ────────────────────────────────────────────────

class TestNormalizeConfig:
    def test_dict_passthrough(self):
        cfg = {"LOOKBACK_DAYS": 90, "UNIVERSE_LIMIT": 50}
        result = _normalize_config(cfg)
        assert isinstance(result, dict)
        assert result.get("LOOKBACK_DAYS") == 90


# ── _to_float ────────────────────────────────────────────────────────

class TestToFloat:
    def test_number(self):
        assert _to_float(3.14) == pytest.approx(3.14)

    def test_string(self):
        assert _to_float("42") == pytest.approx(42.0)

    def test_nan(self):
        assert _to_float(None) is None

    def test_bad_string(self):
        assert _to_float("abc") is None


# ── _canon_column_name ───────────────────────────────────────────────

class TestCanonColumnName:
    def test_lowercase(self):
        assert _canon_column_name("ATR") == "atr"

    def test_preserves_spaces(self):
        # _canon_column_name lowercases but does not strip
        result = _canon_column_name("  Close  ")
        assert result == "  close  "


# ── _quantile_safe ───────────────────────────────────────────────────

class TestQuantileSafe:
    def test_basic(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = _quantile_safe(s, 0.5, default=0.0)
        assert result == pytest.approx(3.0)

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        result = _quantile_safe(s, 0.5, default=0.0)
        assert result == 0.0


# ── check_earnings_blackout ──────────────────────────────────────────

class TestEarningsBlackout:
    def test_no_date(self):
        # Non-existent ticker shouldn't crash
        result = check_earnings_blackout("FAKE_TICKER_12345", days=5)
        assert isinstance(result, bool)


# ── fallback_tracking ────────────────────────────────────────────────

class TestFallbackTracking:
    def test_record_and_read(self):
        reset_fallback_state()
        _record_legacy_fallback("test reason")
        status = get_fallback_status()
        assert isinstance(status, dict)
        assert status.get("fallback_used") is True

    def test_reset(self):
        reset_fallback_state()
        status = get_fallback_status()
        assert isinstance(status, dict)
        assert status.get("fallback_used") is False
