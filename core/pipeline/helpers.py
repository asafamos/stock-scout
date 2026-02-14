"""Pure helper / utility functions for the scan pipeline."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _canon_column_name(c) -> str:
    """Canonicalize a DataFrame column name to lowercase string.

    Handles MultiIndex tuples like ``('AAPL', 'Close')`` by extracting the
    second element.
    """
    try:
        if isinstance(c, tuple) and len(c) >= 2:
            return str(c[1]).lower()
        return str(c).lower()
    except (TypeError, AttributeError):
        return str(c)


def _quantile_safe(vals, q: float, default: float) -> float:
    """Return ``np.quantile(vals, q)`` or *default* when *vals* is empty."""
    try:
        if hasattr(vals, "__len__") and len(vals) == 0:
            return default
        return float(np.quantile(vals, q))
    except Exception:
        return default


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        val = float(x)
        if np.isfinite(val):
            return val
        return None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Earnings blackout check
# ---------------------------------------------------------------------------

def check_earnings_blackout(ticker: str, days: int) -> bool:
    """Return True if *ticker* has earnings within the next *days* days."""
    try:
        info = yf.Ticker(ticker).calendar
        if info is not None and 'Earnings Date' in info:
            earnings_dates = info['Earnings Date']
            if earnings_dates is not None and len(earnings_dates) > 0:
                next_date = pd.to_datetime(earnings_dates[0])
                days_until = (next_date - datetime.now()).days
                return 0 <= days_until <= days
    except Exception as e:
        logger.debug(f"Earnings check failed for {ticker}: {e}")
    return False


# ---------------------------------------------------------------------------
# Tier-2 pass/fail helper
# ---------------------------------------------------------------------------

def _t2_pass_and_reasons(
    row: pd.Series,
    diagnostics: Dict[str, Dict[str, Any]],
) -> Tuple[bool, str]:
    """Determine Tier-2 pass/fail and collect rejection reason strings.

    Returns ``(passed, reasons_text)`` where *passed* is ``False`` only when
    an ``ADVANCED_REJECT`` rule appears in the ticker's diagnostics.
    """
    tkr = str(row.get("Ticker"))
    rec = diagnostics.get(tkr, {}) if isinstance(diagnostics, dict) else {}
    t2 = rec.get("tier2_reasons") or []
    has_adv_reject = any(
        (r.get("rule") == "ADVANCED_REJECT") for r in t2 if isinstance(r, dict)
    )
    reasons_rules = [
        str(r.get("rule")) for r in t2 if isinstance(r, dict) and r.get("rule")
    ]
    reasons_text = row.get("RejectionReason")
    joined = (
        ";".join(reasons_rules)
        if reasons_rules
        else (str(reasons_text) if reasons_text else "")
    )
    return (not has_adv_reject, joined)


# ---------------------------------------------------------------------------
# Dynamic RR computation (Entry / Target / Stop)
# ---------------------------------------------------------------------------

def _compute_rr_for_row(
    row: pd.Series,
    data_map: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Compute Entry / Target / Stop / RR from ATR, Bollinger & resistance.

    *data_map* supplies the historical OHLCV DataFrame for the ticker.
    """
    _nan_rr = {
        "Entry_Price": np.nan,
        "Target_Price": np.nan,
        "Stop_Loss": np.nan,
        "RewardRisk": np.nan,
        "RR_Ratio": np.nan,
        "RR": np.nan,
        "Target_Source": "N/A",
    }
    tkr = str(row.get("Ticker"))
    hist = data_map.get(tkr)
    if hist is None or len(hist) < 5:
        return _nan_rr
    try:
        hdf = hist.copy()
        if "Close" not in hdf.columns or "High" not in hdf.columns or "Low" not in hdf.columns:
            return _nan_rr
        entry = float(hdf["Close"].iloc[-1])
        close_shift = hdf["Close"].shift(1)
        tr = pd.concat([
            (hdf["High"] - hdf["Low"]),
            (hdf["High"] - close_shift).abs(),
            (hdf["Low"] - close_shift).abs()
        ], axis=1).max(axis=1)
        atr14 = (
            float(tr.rolling(14, min_periods=5).mean().iloc[-1])
            if len(tr) >= 5
            else float((hdf["High"] - hdf["Low"]).tail(5).mean())
        )
        atr14 = max(atr14, 1e-6)
        low_5 = float(hdf["Low"].tail(5).min())
        stop_price = float(min(low_5, entry - 2.0 * atr14))
        ma20 = float(hdf["Close"].rolling(20, min_periods=5).mean().iloc[-1])
        std20 = float(hdf["Close"].rolling(20, min_periods=5).std(ddof=0).iloc[-1])
        bb_upper = (
            ma20 + 2.0 * std20
            if np.isfinite(ma20) and np.isfinite(std20)
            else float(hdf["High"].tail(20).max())
        )
        res_60 = float(hdf["High"].tail(60).max())
        target = float(max(res_60, bb_upper))
        risk = float(entry - stop_price)
        reward = float(target - entry)
        rr = np.nan
        if risk > 0 and reward > 0:
            rr = float(np.clip(reward / risk, 0.0, 15.0))
        return {
            "Entry_Price": entry,
            "Target_Price": target,
            "Stop_Loss": stop_price,
            "RewardRisk": rr,
            "RR_Ratio": rr,
            "RR": rr,
            "Target_Source": "Resistance/Bollinger",
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _nan_rr
