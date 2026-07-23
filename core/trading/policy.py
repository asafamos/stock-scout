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


# ────────────────────────────────────────────────────────────────────
# CANONICAL TABLES — owned here, re-exported by scoring_config for
# backward compatibility. Both the trading layer (this file) and the
# scoring layer (core.scoring_config / core.pipeline.runner) need
# these floors. Originally defined in scoring_config and imported by
# trading; the trading layer ended up owning the semantics (it picks
# the +5 buffer, it has the dashboard preview, it has the runtime
# enforcement) so the audit (Cross-cut #1, 2026-05-01) recommended
# inverting the dependency. scoring_config now imports FROM here.
# ────────────────────────────────────────────────────────────────────

# Regime-aware minimum FinalScore_20d for inclusion in scan output.
# In weaker regimes, demand higher quality — prevents recommending
# stocks that merely "survived" a regime penalty multiplier.
REGIME_MIN_SCORE: Dict[str, float] = {
    "TREND_UP": 55.0,
    "BULLISH": 55.0,
    "MODERATE_UP": 60.0,
    "SIDEWAYS": 70.0,   # raised from 65 — in neutral markets demand higher quality
    "NEUTRAL": 70.0,    # raised from 65
    "DISTRIBUTION": 75.0,
    "CORRECTION": 80.0,
    "BEARISH": 80.0,
    "PANIC": 100.0,     # effectively blocks all recommendations
}


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

    Uses `REGIME_MIN_SCORE` (defined in this module) plus a +5 buffer
    over the scan's inclusion threshold. `cfg.min_score_to_trade` acts
    as a HARD FLOOR — regimes that would set a lower threshold get
    raised to CONFIG.

    2026-07-03 BUG FIX: previously regime table silently overrode
    CONFIG. In TREND_UP the floor was 60 even though CONFIG says 73,
    letting Score 60-72 picks through (AEO#2 score 50, IVZ score 68
    both bought during freeze week with 0% WR). CONFIG is now the
    contract, regime table only tightens above it.
    """
    base = float(getattr(cfg, "min_score_to_trade", 73.0))
    if not regime:
        return base
    scan_min = float(REGIME_MIN_SCORE.get(regime.upper(), base - 5.0))
    # CONFIG.min_score is the HARD FLOOR — regime can only raise it.
    return max(scan_min + 5.0, base)


def confidence_floor(regime: str, cfg) -> int:
    """Regime-aware confidence floor as integer (3=High, 2=Medium, 1=Low).

    cfg.min_confidence is the HARD floor. Regime can only TIGHTEN above
    it (never loosen). 2026-07-03 fix: same silent-override pattern as
    regime_score_floor. In TREND_UP/MODERATE_UP the previous code
    silently relaxed High → Medium, letting NEUTRAL-confidence picks
    through (e.g., PR bought with NEUTRAL confidence + score 66).
    Set cfg.min_confidence to "Medium" or lower if you WANT the relax
    behavior; default stays at "High" so bullish regimes still require
    High confidence.

    To OPT INTO the old regime-relax behavior, set env
    TRADE_CONFIDENCE_REGIME_RELAX=1. Default disables it.
    """
    base = _CONF_MAP.get(str(getattr(cfg, "min_confidence", "High")).upper(), 3)
    # Opt-in flag to preserve old bullish-regime relaxation
    relax_enabled = bool(getattr(cfg, "confidence_regime_relax", False))
    if relax_enabled and regime and regime.upper() in _MEDIUM_OK_REGIMES:
        return min(base, 2)
    # NEW 2026-07-21: Adaptive relaxation. After N consecutive dry cycles
    # blocked by Confidence < High in a bullish regime, adaptive_gates
    # module flips its own flag → we relax here too. Reset on any buy.
    # Env kill: TRADE_ADAPTIVE_GATES_ENABLED=0.
    if bool(getattr(cfg, "adaptive_gates_enabled", True)):
        try:
            from core.trading.adaptive_gates import get_adaptive_confidence_relaxed
            if (get_adaptive_confidence_relaxed()
                and regime and regime.upper() in _MEDIUM_OK_REGIMES):
                return min(base, 2)
        except Exception:
            pass  # never let adaptive-gate lookup break the trade path
    return base


_BLOCKED_TICKERS_CACHE: Dict[str, Any] = {
    "tickers": set(),  # last-known-good
    "fetched_at": 0.0,
    "mtime": 0.0,
}
_BLOCKED_TICKERS_TTL = 30.0  # seconds — matches dashboard polling cadence


def _load_blocked_tickers() -> Set[str]:
    """Tickers blocked via command_bus /block.

    Audit C5 (2026-05-01): the previous version read the JSON from disk
    on EVERY gate evaluation. With Streamlit polling state-feed every 30s
    over 2000 scan rows, that was 4000 disk reads/min during dashboard
    render — and worse, if the file was mid-write while command_bus was
    updating it, the JSON parse failed silently and the gate let through
    blocked tickers.

    Now: TTL cache (default 30s) + last-known-good fallback on parse
    failure so the gate never silently regresses to "no blocks" because
    of a transient read-during-write race.
    """
    import time as _time
    now_ts = _time.time()

    try:
        from core.control.command_bus import BLOCK_FILE
        import json as _json
        from datetime import datetime as _dt, timezone as _tz

        if not BLOCK_FILE.exists():
            _BLOCKED_TICKERS_CACHE["tickers"] = set()
            _BLOCKED_TICKERS_CACHE["fetched_at"] = now_ts
            return set()

        # mtime-aware refresh: re-read immediately if file has changed
        # (operator just ran /block) even within TTL window.
        try:
            current_mtime = BLOCK_FILE.stat().st_mtime
        except Exception:
            current_mtime = 0.0

        cache_age = now_ts - _BLOCKED_TICKERS_CACHE["fetched_at"]
        cache_mtime = _BLOCKED_TICKERS_CACHE["mtime"]
        if (
            cache_age < _BLOCKED_TICKERS_TTL
            and current_mtime == cache_mtime
            and _BLOCKED_TICKERS_CACHE["fetched_at"] > 0
        ):
            return _BLOCKED_TICKERS_CACHE["tickers"]

        # Read + parse. On parse failure, return last-known-good set
        # rather than silently degrading to "no blocks".
        try:
            text = BLOCK_FILE.read_text()
            blocks = _json.loads(text)
        except Exception as parse_err:
            # File mid-write or corrupt — keep the last-known set.
            # Don't update fetched_at, so we'll retry on next call.
            return _BLOCKED_TICKERS_CACHE["tickers"]

        active = set()
        now_dt = _dt.now(_tz.utc)
        for tkr, rec in blocks.items():
            until = rec.get("until")
            if until:
                try:
                    if _dt.fromisoformat(until) > now_dt:
                        active.add(tkr.upper())
                except Exception:
                    active.add(tkr.upper())  # malformed date → block defensively
            else:
                active.add(tkr.upper())

        _BLOCKED_TICKERS_CACHE["tickers"] = active
        _BLOCKED_TICKERS_CACHE["fetched_at"] = now_ts
        _BLOCKED_TICKERS_CACHE["mtime"] = current_mtime
        return active
    except Exception:
        # Any other error (import, file system) → return cached or empty.
        return _BLOCKED_TICKERS_CACHE.get("tickers", set())


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

    # 7. RR window — Supabase backtest shows sweet spot [3.0, 5.0]
    #    RR 2.0-2.5 = -0.62%, RR 3-4 = +2.78%, RR 4-5 = +3.81%, RR 7-10 = +0.70%
    # ADAPTIVE RR (task #145, 2026-07-23): honor adaptive_gates relax flag
    # here too, otherwise the pre-filter would let a candidate through and
    # then this SSOT would reject it — silent selection asymmetry.
    min_rr = float(getattr(cfg, "min_rr_to_trade", 3.0))
    max_rr = float(getattr(cfg, "max_rr_to_trade", 5.0))
    try:
        if bool(getattr(cfg, "adaptive_gates_enabled", True)):
            from core.trading.adaptive_gates import get_adaptive_rr_relaxed
            if get_adaptive_rr_relaxed():
                _relaxed_floor = float(getattr(cfg, "adaptive_rr_relaxed_floor", 2.0))
                if _relaxed_floor < min_rr:
                    min_rr = _relaxed_floor
    except Exception:
        pass
    if rr < min_rr:
        failed.append(f"R:R {rr:.2f} < {min_rr:.2f}")
    elif max_rr > 0 and rr > max_rr:
        failed.append(f"R:R {rr:.2f} > {max_rr:.2f} (sweet-spot cap)")
    else:
        passed.append(f"R:R in [{min_rr:.1f}, {max_rr:.1f}]")

    # 8. ML window — Supabase backtest shows sweet spot [0.40, 0.55]
    #    ML 0.20-0.30 = -1.16%, ML 0.45-0.50 = +5.07% (BEST), ML 0.55-0.60 = -4.33%
    min_ml = float(getattr(cfg, "min_ml_prob", 0.40))
    max_ml = float(getattr(cfg, "max_ml_prob", 0.55))
    if ml_prob < min_ml:
        failed.append(f"ML {ml_prob:.3f} < {min_ml:.3f}")
    elif max_ml > 0 and ml_prob > max_ml:
        failed.append(f"ML {ml_prob:.3f} > {max_ml:.3f} (sweet-spot cap)")
    else:
        passed.append(f"ML in [{min_ml:.2f}, {max_ml:.2f}]")

    # 8b. ATR floor — Supabase backtest shows volatility = opportunity
    #     ATR 2-3% = -0.64% (n=5844),  ATR 3-4% = +0.68% (n=5351, FIRST positive band),
    #     ATR 4-5% = +1.55%, ATR 5-7% = +6.13%
    # 2026-06-12 FIX: was 0.04 — too tight, blocked 58% of universe (127/304
    # passing other gates → only 4 with ATR ≥ 4%). The natural inflection
    # where returns turn positive is at 3% (0.03), not 4%. ALSO: handle
    # ATR=0 as MISSING DATA (not low-vol). Real stocks always have
    # non-zero ATR; 0.000 indicates the upstream data provider failed.
    # Fail-OPEN when ATR is missing (let other gates decide); fail-CLOSED
    # only when ATR is a real positive number below the floor.
    min_atr = float(getattr(cfg, "min_atr_pct", 0.03))
    atr_pct_val = float(_row_get_first(row, ["ATR_Pct", "atr_pct"], 0) or 0)
    if min_atr > 0 and atr_pct_val > 0 and atr_pct_val < min_atr:
        # Real positive ATR, but too low
        failed.append(f"ATR {atr_pct_val:.3f} < {min_atr:.2f} (low-vol drag)")
    elif min_atr > 0 and atr_pct_val <= 0:
        # Treat as missing data — fail-open with a warning note
        passed.append(f"ATR missing (pass-through)")
    elif min_atr > 0:
        passed.append(f"ATR ≥ {min_atr:.2f}")

    # 8a. Fundamental score floor (NEW 2026-06-26).
    # Soft filter on quality. Data on 1,748 trades:
    #   fund<30: mean +1.24% (n=12)  ← weakest cohort
    #   fund 30-40: +5.19% (n=129)
    #   fund 40-60: +7.7% mean (n=141) — sweet spot
    # Block fund < cfg.min_fundamental_score (default 30). Pass-through if
    # column missing or floor disabled (=0).
    min_fund = float(getattr(cfg, "min_fundamental_score", 30.0))
    if min_fund > 0:
        fund_val = float(_row_get_first(row, ["Fundamental_Score","fundamental_score","FundamentalScore"], -1) or -1)
        if fund_val >= 0 and fund_val < min_fund:
            failed.append(f"Fundamental {fund_val:.0f} < {min_fund:.0f} (weakest cohort)")
        elif fund_val >= 0:
            passed.append(f"Fundamental {fund_val:.0f} ≥ {min_fund:.0f}")

    # 8b. Volume surge filter (NEW 2026-06-26).
    # Counter-intuitive but data is clear (n=303 in tradeable universe,
    # p=0.04 SIG): HIGH volume_surge predicts LOWER fwd returns.
    # Buckets: vs<0.5=+11.15%, 0.5-1=+6.48%, 1-1.5=+4.97%,
    #          1.5-2=+0.70%, >=2=-6.12%.
    # Block vs >= cfg.max_volume_surge (default 1.5). Pass-through if
    # column missing or cap disabled (=0).
    max_vs = float(getattr(cfg, "max_volume_surge", 1.5))
    if max_vs > 0:
        vs_val = float(_row_get_first(row, ["Volume_Surge","volume_surge","VolumeSurge"], 0) or 0)
        if vs_val > 0 and vs_val >= max_vs:
            failed.append(f"Volume surge {vs_val:.2f} ≥ {max_vs:.1f} (low-return cohort)")
        elif vs_val > 0:
            passed.append(f"Volume surge {vs_val:.2f} < {max_vs:.1f}")

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

    # 11b. Liquidity floor — average daily dollar volume (2026-05-29).
    # Position-size impact at $300 is negligible even on a $10M-ADDV name
    # ($300/$10M = 0.003%), so this is NOT about market impact. It's about
    # the marketable-LIMIT entry (Stage A) filling cleanly: thin names have
    # wide NBBO spreads where ref×1.003 sits below the ask, so the limit
    # won't fill and we burn a candidate slot. A modest ADDV floor keeps the
    # candidate set to names where the limit fills at ~ref. ANDG/SOLS (the
    # +5.6%/+4.9% slippage disasters) were mostly stale-price MKT chases —
    # Stage A is the primary fix; this is defense-in-depth.
    # Fails OPEN on missing volume data (don't reject a good trade just
    # because the scan row lacked vol_avg). Env: TRADE_MIN_ADDV_USD.
    min_addv = float(getattr(cfg, "min_addv_usd", 0) or 0)
    if min_addv > 0:
        vol_avg = float(_row_get_first(row, ["vol_avg", "Vol_Avg", "AvgVolume", "avg_volume"], 0) or 0)
        px_for_liq = entry if entry > 0 else float(_row_get_first(row, ["Close", "close", "Price"], 0) or 0)
        if vol_avg > 0 and px_for_liq > 0:
            addv = vol_avg * px_for_liq
            if addv < min_addv:
                failed.append(
                    f"Liquidity ADDV ${addv/1e6:.1f}M < ${min_addv/1e6:.0f}M "
                    f"(thin — marketable limit may not fill at ref)"
                )
            else:
                passed.append(f"ADDV ${addv/1e6:.0f}M ≥ ${min_addv/1e6:.0f}M")
        else:
            # No volume data — fail open (pass) but note it.
            passed.append("liquidity unknown (vol_avg missing — passed)")

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
    throttle_mult: float = 1.0,
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
        throttle_mult: rolling-window-WR multiplier. 1.0 = normal, 0.5 =
                       throttle WARN level (halves position sizes), 0.0 =
                       throttle HALT (preview shows qty=0 with note).
                       Caller passes from state.throttle.size_multiplier.

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

    # 2026-05-15: "UPTREND" added for parity with _MEDIUM_OK_REGIMES (line 111)
    # and order_manager.py — all three sites must agree on which regimes
    # are "bullish" to keep dashboard preview ↔ live execution in sync.
    if regime in ("TREND_UP", "MODERATE_UP", "BULLISH", "STRONG_UPTREND", "UPTREND"):
        regime_mult = 1.20
    elif regime in ("DISTRIBUTION",):
        regime_mult = 0.85
    elif regime in ("CORRECTION", "BEARISH", "PANIC"):
        regime_mult = 0.70
    else:
        regime_mult = 1.0

    trail_pct = base_trail_pct * regime_mult
    # 2026-05-29 PARITY BUG FIX: this clamp used a hardcoded 2.0% floor while
    # order_manager._execute_single floors at min_initial_trail_pct. That
    # drift meant the preview (and any path relying on it) could submit a
    # trail as tight as 2% — exactly how ORCL got a 3.13% trail when the
    # intended floor was 4%, then got shaken out of a +16% run. Now we
    # mirror order_manager precisely: effective_floor = max(initial_floor,
    # atr_floor), then cap at 9%.
    _atr_floor_mult = float(getattr(cfg, "initial_trail_atr_floor_mult", 0.0))
    _atr_floor = atr_pct * _atr_floor_mult if (atr_pct > 0 and _atr_floor_mult > 0) else 0.0
    # Regime-aware floor (parity with order_manager): defensive regimes get a
    # lower floor so the regime_mult tightening isn't overridden by the wide
    # bull-market floor.
    _defensive = regime in ("CORRECTION", "BEARISH", "PANIC", "DISTRIBUTION")
    _init_floor = float(getattr(cfg,
        "min_initial_trail_pct_defensive" if _defensive else "min_initial_trail_pct",
        3.0 if _defensive else 5.5))
    _effective_floor = max(_init_floor, _atr_floor)
    trail_pct = max(_effective_floor, min(trail_pct, 9.0))

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

    # Throttle multiplier — rolling-window-WR safety brake.
    # Production code applies this in risk_manager.calculate_qty; the
    # preview must apply the same factor or qty estimates inflate by 2×
    # when throttle is at WARN (0.5×) level.
    throttle_mult = max(0.0, min(throttle_mult, 1.0))

    target_spend = base_spend * conviction_mult * ml_mult * vol_factor * throttle_mult
    target_spend = min(target_spend, float(getattr(cfg, "max_position_size", 300.0)) * 1.5)
    if available_cash is not None:
        target_spend = min(target_spend, max(0.0, available_cash))

    qty_estimate = math.floor(target_spend / entry) if entry > 0 else 0
    # Fallback: 1 share if we can afford it. Mirrors risk_manager.calculate_qty
    # exactly — bound is `available_cash` (when known) or `max_position_size × 2`,
    # NOT base_spend. This matters for $400-$600 stocks where the conviction-
    # weighted spend stays below the price but we DO have enough cash to buy 1.
    # Disabled when throttle is at HALT level (production blocks the trade
    # at the gate before reaching the sizing fallback).
    if qty_estimate == 0 and entry > 0 and throttle_mult > 0:
        max_affordable = (
            available_cash if available_cash is not None
            else float(getattr(cfg, "max_position_size", 300.0)) * 2
        )
        if entry <= max_affordable:
            qty_estimate = 1

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
