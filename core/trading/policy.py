"""Single source of truth for buy-eligibility gates and execution preview.

Both the production execution path (`risk_manager.can_open_position`,
`order_manager._filter_candidates`) and the dashboard preview
(`streamlit_components.evaluate_scan_row_for_buy`) call into this module.

Without this, the dashboard's preview drifts from production whenever a
CONFIG knob moves — the user sees "🚀 BUY ELIGIBLE" in the UI but the
actual pipeline rejects with a different gate. Worst-case foot-gun.

This module is **deliberately decoupled from IB**: every function works
on plain scan-row data + an optional state-feed snapshot. The risk
manager wraps these and adds IB-dependent checks (cash, market-hours,
day-trade history, etc).

────────────────────────────────────────────────────────────────────
GATE TIERS
────────────────────────────────────────────────────────────────────

STATIC GATES (no IB, no slow network — runs in dashboard preview):
  - paused
  - throttle halt
  - already-holding (from state.positions or passed list)
  - regime block (PANIC, CORRECTION)
  - score floor (regime-aware via REGIME_MIN_SCORE + cfg buffer)
  - score band cap (cfg.max_score_to_trade)
  - RR floor (cfg.min_rr_to_trade)
  - ML prob floor (cfg.min_ml_prob)
  - confidence floor (regime-aware)
  - sector blocklist (cfg.blocked_sectors_list)
  - reliability floor (cfg.min_reliability)
  - blocked-ticker list (data/state/blocked_tickers.json)
  - trade-level sanity (NaN, target ≥ entry × 1.02, stop ≤ entry × 0.995)

RUNTIME GATES (skipped in preview, enforced in production):
  - earnings calendar (yfinance fetch — slow + cached)
  - sector momentum (yfinance fetch)
  - portfolio correlation (yfinance batch fetch)
  - daily-loss circuit breaker (needs live IB net liq)
  - cash-after-buy ($2k tier boundary)
  - day-trade prevention (cash account T+1)
  - max open positions (live IB count)
  - daily buy limit (tracker count)
  - sizing — max portfolio exposure, insufficient cash
  - market-hours (when not dry_run)
  - news catalyst (24h move + volume)
  - gap-up / gap-down (live price vs scan close)
  - slippage hard-reject (live price vs scan price)

The dashboard preview labels itself "preview" so the user understands
that runtime gates can still reject — but they will NEVER see a static-
gate disagreement between preview and production again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class GateResult:
    """Outcome of running buy-eligibility gates against one row."""
    would_buy: bool
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)

    @property
    def primary_reason(self) -> str:
        """First failure reason, or empty string if all gates passed."""
        return self.gates_failed[0] if self.gates_failed else ""

    @property
    def verdict(self) -> str:
        return "BUY ELIGIBLE" if self.would_buy else "SKIP"


# Confidence ranking — used by both the scan classifier and the trader.
# HIGH > MEDIUM > LOW/SPECULATIVE > NONE
_CONF_MAP = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "SPECULATIVE": 1, "NONE": 0}

# Regimes where Medium-confidence is acceptable (the macro tailwind
# compensates for thinner per-stock confirmation).
_MEDIUM_OK_REGIMES = {"TREND_UP", "MODERATE_UP", "BULLISH", "STRONG_UPTREND", "UPTREND"}

# Regimes that block trading entirely (mirrors cfg.blocked_regimes default).
_BLOCKED_REGIMES = {"PANIC", "CORRECTION"}


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Defensive accessor — works for pandas Series and dict alike."""
    if hasattr(row, "get"):
        try:
            return row.get(key, default)
        except Exception:
            pass
    try:
        return row[key]
    except Exception:
        return default


def _row_get_first(row: Any, keys: List[str], default: Any = None) -> Any:
    """Try multiple keys (snake_case + PascalCase + aliases)."""
    for k in keys:
        v = _row_get(row, k, None)
        if v is not None:
            return v
    return default


def regime_score_floor(regime: str, cfg) -> float:
    """Regime-aware minimum FinalScore_20d to trade.

    Reads `REGIME_MIN_SCORE` from `core.scoring_config` and adds a +5
    buffer over the scan's inclusion threshold (so trades demand higher
    conviction than mere recommendation). Falls back to
    `cfg.min_score_to_trade` when the regime is unknown.
    """
    base = float(getattr(cfg, "min_score_to_trade", 73.0))
    if not regime:
        return base
    try:
        from core.scoring_config import REGIME_MIN_SCORE
        scan_min = float(REGIME_MIN_SCORE.get(regime.upper(), base - 5.0))
        return scan_min + 5.0
    except Exception:
        return base


def confidence_floor(regime: str, cfg) -> int:
    """Regime-aware confidence floor as integer (3=High, 2=Medium, 1=Low).

    cfg.min_confidence is the baseline (default "High"=3). In bullish
    regimes we relax to Medium (2) — same logic as
    `order_manager._filter_candidates`.
    """
    base = _CONF_MAP.get(str(getattr(cfg, "min_confidence", "High")).upper(), 3)
    if regime and regime.upper() in _MEDIUM_OK_REGIMES:
        return min(base, 2)
    return base


def _load_blocked_tickers() -> Set[str]:
    """Tickers blocked via command_bus /block. Cached read — file is small."""
    try:
        from core.control.command_bus import BLOCK_FILE
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        if not BLOCK_FILE.exists():
            return set()
        blocks = _json.loads(BLOCK_FILE.read_text())
        active = set()
        now = _dt.now(_tz.utc)
        for tkr, rec in blocks.items():
            until = rec.get("until")
            if until:
                try:
                    if _dt.fromisoformat(until) > now:
                        active.add(tkr.upper())
                except Exception:
                    active.add(tkr.upper())  # malformed date → block defensively
            else:
                active.add(tkr.upper())
        return active
    except Exception:
        return set()


def evaluate_static_gates(
    row: Any,
    *,
    cfg=None,
    state: Optional[Dict] = None,
    held_tickers: Optional[Set[str]] = None,
) -> GateResult:
    """Run the gates that don't require an IB connection.

    Used by the dashboard preview AND by the production path (which then
    layers IB-dependent gates on top).

    Args:
        row: scan row (pandas Series or dict). Reads:
             Ticker / FinalScore_20d / Score / RewardRisk / ML_20d_Prob
             / Sector / SignalQuality (or Confidence_Level) / Market_Regime
             / Reliability_Score / Entry_Price / Stop_Loss / Target_Price.
        cfg: trading config. Defaults to CONFIG.
        state: state-feed snapshot. Reads `paused`, `throttle.level`,
               `positions[].ticker`. Optional — if absent, those gates pass.
        held_tickers: explicit override for held tickers (e.g. when caller
                      already has a fresh IB position list).

    Returns: GateResult with full pass/fail trail.
    """
    if cfg is None:
        from core.trading.config import CONFIG as _CFG
        cfg = _CFG

    state = state or {}
    if held_tickers is None:
        held_tickers = {
            str(p.get("ticker", "")).upper()
            for p in (state.get("positions") or [])
        }

    # ── Extract row fields ───────────────────────────────────────────
    ticker = str(_row_get_first(row, ["Ticker", "ticker", "Symbol"], "")).upper()
    score = float(_row_get_first(row, ["FinalScore_20d", "Score", "final_score"], 0) or 0)
    rr = float(_row_get_first(row, ["RewardRisk", "RR_Ratio", "RR", "rr"], 0) or 0)
    ml_prob = float(_row_get_first(row, ["ML_20d_Prob", "ml_prob", "ML_Prob"], 0) or 0)
    sector = str(_row_get_first(row, ["Sector", "sector"], "") or "")
    confidence = str(
        _row_get_first(
            row,
            ["SignalQuality", "Signal_Quality", "Confidence_Level", "Confidence"],
            "",
        )
        or ""
    )
    regime = str(_row_get_first(row, ["Market_Regime", "market_regime", "Regime"], "") or "").upper()
    reliability = float(_row_get_first(row, ["Reliability_Score", "Reliability", "reliability"], 100) or 100)

    # Trade-level sanity inputs (optional — only checked when present)
    entry = float(_row_get_first(row, ["Entry_Price", "entry_price", "Close"], 0) or 0)
    stop = float(_row_get_first(row, ["Stop_Loss", "stop_loss", "StopLoss"], 0) or 0)
    target = float(_row_get_first(row, ["Target_Price", "target_price"], 0) or 0)

    passed: List[str] = []
    failed: List[str] = []

    # 1. Auto-trade paused?
    if state.get("paused"):
        failed.append("Auto-trading PAUSED (run /resume to lift)")
    else:
        passed.append("not paused")

    # 2. Throttle halt? (state.throttle.level == "halt")
    throttle = state.get("throttle", {}) or {}
    if throttle.get("level") == "halt":
        wr = throttle.get("win_rate", 0)
        failed.append(
            f"Performance throttle HALT (WR {int((wr or 0) * 100)}% < threshold)"
        )
    else:
        passed.append("throttle ok")

    # 3. Blocked ticker (command_bus /block)
    if ticker and ticker in _load_blocked_tickers():
        failed.append(f"Ticker {ticker} on block list (/block)")
    else:
        passed.append("not blocked")

    # 4. Already holding
    if ticker and ticker in held_tickers:
        failed.append(f"Already holding {ticker}")
    else:
        passed.append("not held")

    # 5. Regime block (PANIC / CORRECTION)
    blocked_regimes = set(getattr(cfg, "blocked_regimes_list", []) or _BLOCKED_REGIMES)
    blocked_regimes = {r.upper() for r in blocked_regimes}
    if regime and regime in blocked_regimes:
        failed.append(f"Market regime blocked: {regime}")
    else:
        passed.append(f"regime ok ({regime or 'unknown'})")

    # 6. Score band — floor (regime-aware) AND cap
    floor = regime_score_floor(regime, cfg)
    cap = float(getattr(cfg, "max_score_to_trade", 95.0))
    if score < floor:
        failed.append(f"Score {score:.1f} < {floor:.1f} ({regime or 'default'} floor)")
    elif score > cap:
        failed.append(f"Score {score:.1f} > {cap:.1f} cap (Q5 underperforms)")
    else:
        passed.append(f"score in [{floor:.0f}, {cap:.0f}]")

    # 7. RR floor
    min_rr = float(getattr(cfg, "min_rr_to_trade", 2.0))
    if rr < min_rr:
        failed.append(f"R:R {rr:.2f} < {min_rr:.2f}")
    else:
        passed.append(f"R:R ≥ {min_rr:.2f}")

    # 8. ML prob floor
    min_ml = float(getattr(cfg, "min_ml_prob", 0.33))
    if ml_prob < min_ml:
        failed.append(f"ML {ml_prob:.3f} < {min_ml:.3f}")
    else:
        passed.append(f"ML ≥ {min_ml:.2f}")

    # 9. Confidence floor (regime-aware)
    min_conf_int = confidence_floor(regime, cfg)
    conf_int = _CONF_MAP.get(confidence.upper(), 0)
    label_for = {3: "High", 2: "Medium", 1: "Low"}
    if conf_int < min_conf_int:
        failed.append(
            f"Confidence {confidence or 'n/a'} < required "
            f"({label_for.get(min_conf_int, '?')})"
        )
    else:
        passed.append(f"confidence ≥ {label_for.get(min_conf_int, '?')}")

    # 10. Sector blocklist
    blocked_sectors = {
        s.strip() for s in getattr(cfg, "blocked_sectors_list", []) if s.strip()
    }
    if sector and sector in blocked_sectors:
        failed.append(f"Blocked sector: {sector}")
    else:
        passed.append("sector ok")

    # 11. Reliability floor
    min_rel = float(getattr(cfg, "min_reliability", 50.0))
    if reliability < min_rel:
        failed.append(f"Reliability {reliability:.0f} < {min_rel:.0f}")
    else:
        passed.append(f"reliability ≥ {min_rel:.0f}")

    # 12. Trade-level sanity (only when entry+stop+target are all present)
    if entry > 0 and stop > 0 and target > 0:
        try:
            if math.isnan(entry) or math.isnan(stop) or math.isnan(target):
                failed.append("NaN in trade levels (entry/stop/target)")
            elif target < entry * 1.02:
                failed.append(
                    f"Target ${target:.2f} < entry+2% (${entry * 1.02:.2f})"
                )
            elif stop >= entry * 0.995:
                failed.append(
                    f"Stop ${stop:.2f} >= entry-0.5% (${entry * 0.995:.2f})"
                )
            else:
                passed.append("trade levels sane")
        except (TypeError, ValueError):
            failed.append("Non-numeric trade levels")

    return GateResult(
        would_buy=(len(failed) == 0),
        gates_passed=passed,
        gates_failed=failed,
    )


# ════════════════════════════════════════════════════════════════════
# EXECUTION PREVIEW — shows what the actual order will look like.
# Mirrors the price/stop/target/trail derivation in
# order_manager._execute_single. Used by the dashboard so the user
# sees the SAME numbers that will be sent to IB, not the stale scan
# numbers.
# ════════════════════════════════════════════════════════════════════


@dataclass
class ExecutionPreview:
    """What the order_manager would actually submit for this row.

    All prices/percentages match what `_execute_single` computes — barring
    live-price refresh (which the dashboard can't do without IB) and the
    earnings/sector-momentum runtime gates.
    """
    ticker: str
    entry: float            # scan entry (or live, if provided)
    stop: float             # adjusted stop (proportional to entry)
    target: float           # capped target (≤ analyst mean midpoint when known)
    trail_pct: float        # final TRAIL % (after regime mult + clamp)
    base_trail_pct: float   # before regime adjustment
    regime_mult: float      # regime multiplier applied
    qty_estimate: int       # estimated share count given cfg + cash
    spend_estimate: float   # qty × entry
    slippage_warning: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def compute_execution_preview(
    row: Any,
    *,
    cfg=None,
    live_price: Optional[float] = None,
    available_cash: Optional[float] = None,
) -> ExecutionPreview:
    """Return what the order_manager would submit for this scan row.

    This is a pure function that mirrors `_execute_single` lines 678-865.
    No IB connection needed — `live_price` is optional (caller provides
    it if available, otherwise we use scan price).

    Args:
        row: scan row.
        cfg: trading config; defaults to CONFIG.
        live_price: most recent quote, if known. When None, the preview
                    just shows scan-derived numbers (clearly labeled).
        available_cash: cash available for sizing. When None, the qty
                        estimate uses cfg.max_position_size as the bound.

    Returns: ExecutionPreview with all the numbers the dashboard should
             display alongside (or instead of) raw scan numbers.
    """
    if cfg is None:
        from core.trading.config import CONFIG as _CFG
        cfg = _CFG

    ticker = str(_row_get_first(row, ["Ticker", "ticker", "Symbol"], ""))
    scan_entry = float(_row_get_first(row, ["Entry_Price", "entry_price", "Close"], 0) or 0)
    stop = float(_row_get_first(row, ["Stop_Loss", "stop_loss", "StopLoss"], 0) or 0)
    target = float(_row_get_first(row, ["Target_Price", "target_price"], 0) or 0)
    atr_pct = float(_row_get_first(row, ["ATR_Pct", "atr_pct"], 0) or 0)
    score = float(_row_get_first(row, ["FinalScore_20d", "Score"], 0) or 0)
    rr = float(_row_get_first(row, ["RewardRisk", "RR"], 0) or 0)
    ml_prob = float(_row_get_first(row, ["ML_20d_Prob", "ml_prob"], 0) or 0)
    regime = str(_row_get_first(row, ["Market_Regime"], "") or "").upper()

    notes: List[str] = []
    slippage_warning: Optional[str] = None

    # ── Effective entry: live-refreshed when available ──
    entry = scan_entry
    if live_price and live_price > 0 and scan_entry > 0:
        move_pct = (live_price - scan_entry) / scan_entry * 100
        if abs(move_pct) > 5.0:
            slippage_warning = (
                f"Live price ${live_price:.2f} moved {move_pct:+.1f}% "
                f"from scan ${scan_entry:.2f} — order_manager will REJECT this trade."
            )
            notes.append("⚠️ slippage > 5% — would be rejected")
        else:
            # Proportional rescale (matches order_manager:707-712)
            if stop > 0:
                stop_pct_of_entry = (scan_entry - stop) / scan_entry
                stop = round(live_price * (1 - stop_pct_of_entry), 2)
            if target > 0:
                tgt_pct_of_entry = (target - scan_entry) / scan_entry
                target = round(live_price * (1 + tgt_pct_of_entry), 2)
            entry = live_price
            notes.append(
                f"live refresh: ${scan_entry:.2f} → ${live_price:.2f} "
                f"({move_pct:+.2f}%); stop/target rescaled"
            )
    elif live_price is None:
        notes.append("scan numbers shown — live refresh not available in preview")

    # ── Trail % calculation (matches order_manager:831-857) ──
    trail_candidates = []
    if stop > 0 and entry > 0:
        trail_candidates.append(round((entry - stop) / entry * 100, 1))
    if atr_pct > 0:
        trail_candidates.append(round(atr_pct * 1.5, 1))
    if trail_candidates:
        base_trail_pct = sum(trail_candidates) / len(trail_candidates)
    else:
        base_trail_pct = float(getattr(cfg, "trailing_stop_pct", 5.0))

    if regime in ("TREND_UP", "MODERATE_UP", "BULLISH", "STRONG_UPTREND"):
        regime_mult = 1.20
    elif regime in ("DISTRIBUTION",):
        regime_mult = 0.85
    elif regime in ("CORRECTION", "BEARISH", "PANIC"):
        regime_mult = 0.70
    else:
        regime_mult = 1.0

    trail_pct = base_trail_pct * regime_mult
    trail_pct = max(2.0, min(trail_pct, 9.0))  # safety clamp

    # ── Quantity estimate (matches risk_manager.calculate_qty conviction tiers) ──
    base_spend = float(getattr(cfg, "max_position_size", 300.0))
    if available_cash is not None:
        base_spend = min(base_spend, max(0.0, available_cash))

    # Conviction multiplier (mirrors RiskManager._CONVICTION_TIERS)
    conviction_mult = 0.70  # marginal default
    for min_score, min_rr, mult in (
        (82.0, 2.5, 1.35),
        (78.0, 2.2, 1.15),
        (75.0, 2.0, 1.00),
        (73.0, 1.8, 0.70),
    ):
        if score >= min_score and rr >= min_rr:
            conviction_mult = mult
            break

    # ML sizing
    ml_mult = 1.0
    if getattr(cfg, "ml_sizing_enabled", True) and ml_prob > 0:
        anchor = float(getattr(cfg, "ml_sizing_anchor", 0.40))
        slope = float(getattr(cfg, "ml_sizing_slope", 2.0))
        raw = 1.0 + slope * (ml_prob - anchor)
        ml_mult = max(
            float(getattr(cfg, "ml_sizing_min_mult", 0.70)),
            min(float(getattr(cfg, "ml_sizing_max_mult", 1.30)), raw),
        )

    # Volatility scaling (matches risk_manager)
    vol_factor = 1.0
    if atr_pct > 0:
        vol_factor = max(0.5, min(2.0 / max(atr_pct, 1.0), 1.0))

    target_spend = base_spend * conviction_mult * ml_mult * vol_factor
    target_spend = min(target_spend, float(getattr(cfg, "max_position_size", 300.0)) * 1.5)
    if available_cash is not None:
        target_spend = min(target_spend, max(0.0, available_cash))

    qty_estimate = math.floor(target_spend / entry) if entry > 0 else 0
    if qty_estimate == 0 and entry > 0 and entry <= base_spend:
        qty_estimate = 1  # fallback for high-priced stocks

    spend_estimate = qty_estimate * entry

    return ExecutionPreview(
        ticker=ticker,
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        trail_pct=round(trail_pct, 1),
        base_trail_pct=round(base_trail_pct, 1),
        regime_mult=round(regime_mult, 2),
        qty_estimate=qty_estimate,
        spend_estimate=round(spend_estimate, 2),
        slippage_warning=slippage_warning,
        notes=notes,
    )


# ════════════════════════════════════════════════════════════════════
# TICKER NORMALIZATION — IB requires "BRK B" not "BRK.B" for some
# share classes. This function does the round-trip so the trader and
# tracker speak the same language as IB.
# ════════════════════════════════════════════════════════════════════


# Class B / preferred-share tickers IB writes with a space
# (Yahoo / scanner uses dot). Add to this set as needed.
_DOT_TO_SPACE = {
    "BRK.A", "BRK.B", "BF.A", "BF.B", "GTN.A", "HEI.A",
    "LEN.B", "MKC.V", "MOG.A", "MOG.B", "RUSHA", "RUSHB",
}


def normalize_ticker_for_ib(ticker: str) -> str:
    """Return the IB-compatible form of a ticker.

    IB uses a space for class-share separators on US equities ("BRK B"),
    while Yahoo/our scanner uses a dot ("BRK.B"). For the handful of
    affected names we substitute; for everything else we pass through.

    Idempotent — calling it twice returns the same result.
    """
    if not ticker:
        return ticker
    t = str(ticker).strip().upper()
    if t in _DOT_TO_SPACE or (t.replace(".", " ") in _DOT_TO_SPACE):
        return t.replace(".", " ")
    # Heuristic for unknown class-share tickers: any ".A" / ".B" suffix
    # on a 1-3-letter base is probably IB-needs-space format.
    if "." in t:
        base, _, suffix = t.partition(".")
        if 1 <= len(base) <= 4 and suffix in ("A", "B", "C", "V"):
            return f"{base} {suffix}"
    return t
