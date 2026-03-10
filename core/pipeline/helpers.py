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
    market_regime: str = "neutral",
) -> Dict[str, Any]:
    """Compute Entry / Target / Stop / RR using ATR-projected targets.

    Uses a forward-looking ATR-based target that doesn't penalize stocks
    near their highs (which are exactly the breakout candidates we want).

    Target methodology:
    - ATR projection: entry + K * ATR14 (K=2.5 base, 3.0 for breakouts)
    - Resistance level: max(60d high, Bollinger upper)
    - Final target: the HIGHER of ATR projection and resistance
    This ensures RR reflects forward potential, not just past price range.

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
        "Stop_Source": "N/A",
    }
    tkr = str(row.get("Ticker"))
    hist = data_map.get(tkr)
    if hist is None or len(hist) < 5:
        return _nan_rr
    try:
        hdf = hist.copy()
        if "Close" not in hdf.columns or "High" not in hdf.columns or "Low" not in hdf.columns:
            return _nan_rr
        _close = float(hdf["Close"].iloc[-1])
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

        # Limit-entry offset: simulate a limit order below close for better R:R.
        try:
            from core.scoring_config import ENTRY_OFFSET
            _entry_offset = float(ENTRY_OFFSET)
        except Exception:
            _entry_offset = 0.0
        entry = _close - _entry_offset * atr14
        entry = max(entry, _close * 0.95)  # Safety: never more than 5% below close

        # Stop loss: combines ATR-based and support-based levels.
        # VCP setups get tighter stops (defined risk from consolidation).
        try:
            from core.scoring_config import ATR_STOP_MULTIPLIER, DYNAMIC_RR_CONFIG, SUPPORT_STOP_CONFIG
            _stop_mult = float(ATR_STOP_MULTIPLIER)
        except Exception:
            _stop_mult = 1.5
            DYNAMIC_RR_CONFIG = {}
            SUPPORT_STOP_CONFIG = {"enabled": False}
        low_5 = float(hdf["Low"].tail(5).min())
        low_20 = float(hdf["Low"].tail(20).min()) if len(hdf) >= 20 else low_5
        atr_stop = entry - _stop_mult * atr14
        # VCP setups → tighter stop using support levels
        _vcp_val = row.get("Volatility_Contraction_Score", 0.0) if row is not None else 0.0
        _vcp_val = float(_vcp_val) if isinstance(_vcp_val, (int, float)) and np.isfinite(float(_vcp_val)) else 0.0
        _vcp_stop_thresh = DYNAMIC_RR_CONFIG.get("vcp_stop_threshold", 0.5)
        _vcp_max_stop = DYNAMIC_RR_CONFIG.get("vcp_max_stop_pct", 0.08)
        _stop_type = "legacy_atr"
        if _vcp_val > _vcp_stop_thresh:
            # Support-based stop: tighter (higher price), capped at max %
            support_stop = max(low_20, atr_stop)
            stop_price = float(max(support_stop, entry * (1.0 - _vcp_max_stop)))
            _stop_type = "vcp_support"
        else:
            # Support-based stop for ALL stocks (not just VCP)
            _sup_cfg = SUPPORT_STOP_CONFIG if 'SUPPORT_STOP_CONFIG' in dir() else {}
            if _sup_cfg.get("enabled", False):
                _min_risk = float(_sup_cfg.get("min_risk_pct", 0.02))
                _max_risk = float(_sup_cfg.get("max_risk_pct", 0.10))
                _min_stop_price = entry * (1.0 - _min_risk)
                _atr_floor = max(atr_stop, entry * (1.0 - _max_risk))

                support_candidates = []
                if _sup_cfg.get("low_20d_enabled", True) and len(hdf) >= 20:
                    _s_low20 = float(hdf["Low"].tail(20).min())
                    if _s_low20 < _min_stop_price:
                        support_candidates.append(_s_low20)
                if _sup_cfg.get("low_10d_enabled", True) and len(hdf) >= 10:
                    _s_low10 = float(hdf["Low"].tail(10).min())
                    if _s_low10 < _min_stop_price:
                        support_candidates.append(_s_low10)
                if _sup_cfg.get("bollinger_enabled", True):
                    _bb_p = int(_sup_cfg.get("bollinger_periods", 20))
                    _bb_s = float(_sup_cfg.get("bollinger_std", 2.0))
                    if len(hdf) >= _bb_p:
                        _bb_ma = float(hdf["Close"].rolling(_bb_p, min_periods=5).mean().iloc[-1])
                        _bb_sd = float(hdf["Close"].rolling(_bb_p, min_periods=5).std(ddof=0).iloc[-1])
                        if np.isfinite(_bb_ma) and np.isfinite(_bb_sd):
                            _bb_lower = _bb_ma - _bb_s * _bb_sd
                            if _bb_lower < _min_stop_price:
                                support_candidates.append(_bb_lower)

                if support_candidates:
                    stop_price = float(max(max(support_candidates), _atr_floor))
                    _stop_type = "support"
                else:
                    stop_price = float(min(low_5, atr_stop))
                    _stop_type = "legacy_atr"
            else:
                stop_price = float(min(low_5, atr_stop))
        # Ensure stop is below entry
        stop_price = min(stop_price, entry * 0.99)

        # Compute distance from 52w high early (used by both resistance and ATR sections)
        high_52w = float(hdf["High"].max()) if len(hdf) >= 20 else float(hdf["High"].tail(60).max())
        dist_from_high = (high_52w - entry) / high_52w if high_52w > 0 else 1.0

        # Resistance-based target (backward-looking)
        ma20 = float(hdf["Close"].rolling(20, min_periods=5).mean().iloc[-1])
        std20 = float(hdf["Close"].rolling(20, min_periods=5).std(ddof=0).iloc[-1])
        bb_upper = (
            ma20 + 2.0 * std20
            if np.isfinite(ma20) and np.isfinite(std20)
            else float(hdf["High"].tail(20).max())
        )
        res_60 = float(hdf["High"].tail(60).max())
        # 52w high is a strong resistance/target for stocks far below it
        res_52w = high_52w
        if dist_from_high > 0.10:  # >10% below 52w high → use it as target
            resistance_target = float(max(res_60, bb_upper, res_52w * 0.98))
        else:
            resistance_target = float(max(res_60, bb_upper))

        # ATR-projected target (forward-looking, regime-aware)
        # Stocks near 52w high get higher multiplier (breakout potential)
        # Regime-aware multipliers: conservative in neutral/bearish, full in bullish
        try:
            from core.scoring_config import ATR_TARGET_MULTIPLIERS
            _num_regime_map = {1.0: "bullish", 0.0: "neutral", -1.0: "bearish"}
            # Wyckoff phases → simple ATR regime
            _wyckoff_map = {
                "trend_up": "bullish", "moderate_up": "bullish",
                "sideways": "neutral",
                "distribution": "bearish", "correction": "bearish", "panic": "bearish",
            }
            if isinstance(market_regime, (int, float)):
                _regime_key = _num_regime_map.get(float(market_regime), "neutral")
            elif isinstance(market_regime, str):
                _lower = market_regime.lower()
                _regime_key = _wyckoff_map.get(_lower, _lower)
            else:
                _regime_key = "neutral"
            _mults = ATR_TARGET_MULTIPLIERS.get(_regime_key, ATR_TARGET_MULTIPLIERS.get("neutral", {"base": 2.0, "breakout": 2.5}))
            base_mult = _mults["breakout"] if dist_from_high < 0.05 else _mults["base"]
        except Exception:
            base_mult = 3.0 if dist_from_high < 0.05 else 2.5  # fallback to original

        # ── Dynamic per-stock adjustments ──────────────────────
        # Make R:R vary between stocks by adjusting ATR multiplier
        # based on relative strength, VCP, and momentum consistency.
        _dyn = DYNAMIC_RR_CONFIG if DYNAMIC_RR_CONFIG else {}
        momentum_adj = 0.0
        vcp_adj = 0.0

        # Relative strength adjustment
        _rs63 = row.get("RS_63d", np.nan) if row is not None else np.nan
        if isinstance(_rs63, (int, float)) and np.isfinite(float(_rs63)):
            _rs63 = float(_rs63)
            if _rs63 > _dyn.get("rs_strong_threshold", 1.2):
                momentum_adj = _dyn.get("rs_strong_adj", 0.3)
            elif _rs63 > _dyn.get("rs_above_avg_threshold", 1.0):
                momentum_adj = _dyn.get("rs_above_avg_adj", 0.15)
            elif _rs63 < _dyn.get("rs_weak_threshold", 0.8):
                momentum_adj = _dyn.get("rs_weak_adj", -0.2)

        # VCP adjustment: tight patterns → higher breakout potential
        if _vcp_val > _dyn.get("vcp_strong_threshold", 0.7):
            vcp_adj = _dyn.get("vcp_strong_adj", 0.2)
        elif _vcp_val > _dyn.get("vcp_moderate_threshold", 0.4):
            vcp_adj = _dyn.get("vcp_moderate_adj", 0.1)

        # Momentum consistency bonus
        _mom_cons = row.get("Momentum_Consistency", np.nan) if row is not None else np.nan
        if isinstance(_mom_cons, (int, float)) and np.isfinite(float(_mom_cons)):
            if float(_mom_cons) > _dyn.get("momentum_cons_threshold", 0.65):
                momentum_adj += _dyn.get("momentum_cons_adj", 0.1)

        atr_mult = float(np.clip(
            base_mult + momentum_adj + vcp_adj,
            _dyn.get("atr_mult_min", 1.5),
            _dyn.get("atr_mult_max", 5.0),
        ))
        atr_target = entry + atr_mult * atr14

        # Final target: higher of ATR projection and resistance.
        # In distribution regime, cap resistance at the ATR-projected target.
        # Rationale: the 60d/52w high in distribution is the distribution ceiling,
        # not a realistic breakout target. Using it inflates targets for stocks
        # that are topping out.
        _effective_resistance = resistance_target
        if isinstance(market_regime, str) and market_regime.lower() in ("distribution", "correction"):
            # If resistance is above ATR target, the stock is near its peak.
            # In distribution, lean on the more conservative ATR projection.
            if resistance_target > atr_target:
                _effective_resistance = atr_target
        target = float(max(atr_target, _effective_resistance))
        target_source = "ATR_Projection" if atr_target >= _effective_resistance else "Resistance/Bollinger"

        risk = float(entry - stop_price)
        reward = float(target - entry)
        rr = np.nan
        if risk > 0 and reward > 0:
            rr = float(np.clip(reward / risk, 0.0, 15.0))

        # Volume confirmation: in distribution/correction, compute up-day vs
        # down-day volume ratio.  Weak rally volume signals lack of conviction.
        _vol_ud_ratio = np.nan
        try:
            if (
                isinstance(market_regime, str)
                and market_regime.lower() in ("distribution", "correction")
                and "Volume" in hdf.columns
                and len(hdf) >= 10
            ):
                _close_chg = hdf["Close"].diff()
                _up_mask = _close_chg > 0
                _dn_mask = _close_chg < 0
                _up_vol = hdf.loc[_up_mask, "Volume"].tail(20).mean()
                _dn_vol = hdf.loc[_dn_mask, "Volume"].tail(20).mean()
                if _dn_vol and _dn_vol > 0:
                    _vol_ud_ratio = float(_up_vol / _dn_vol)
        except Exception:
            pass

        return {
            "Entry_Price": entry,
            "Target_Price": target,
            "Stop_Loss": stop_price,
            "RewardRisk": rr,
            "RR_Ratio": rr,
            "RR": rr,
            "Target_Source": target_source,
            "Stop_Source": _stop_type,
            "Volume_UpDown_Ratio": _vol_ud_ratio,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _nan_rr
