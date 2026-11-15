"""
Core business logic for Stock Scout.
Separates business logic from Streamlit UI.
"""

__version__ = "2.0.0"

try:  # FMP usage meter (2026-11-15): counts calls/bytes per endpoint; never fatal. Disable with FMP_METER=0.
    from core import fmp_meter as _fmp_meter

    _fmp_meter.install()
except Exception:  # pragma: no cover
    pass
