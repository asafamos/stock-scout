# Deferred Items — Next Session

**Owner:** Asaf
**Last updated:** 2026-05-01 (closeout of the cold-eyes audit session)

This file tracks improvements identified by the audit that were
**deliberately deferred** rather than rushed into Friday-night code.
Each item has a clear "what changes" + "why deferred" + "estimated
effort" so the next session picks up cleanly.

The system is **production-healthy** as of this write. Everything
listed here is improvement, not bug-fix.

---

## 🔴 Material EV decisions (need owner approval, not code review)

### H1 — Reduce concurrent positions: 3 × $300 → 2 × $400

**What changes:**
```python
# core/trading/config.py
max_open_positions: int = 2     # was 3
max_position_size: float = 400  # was 300
max_portfolio_exposure: float = 800  # was 900 (2 × 400)
```

**Why:** With $1k account, 3 concurrent positions = 90% deployed.
Commission is $1/leg = $2 round-trip = 0.67% drag on a $300 trade
vs 0.50% on a $400 trade. Two $400 positions has the same total
exposure but lets entries stagger by a day, decorrelating timing
risk on volatile open-prints.

**Why deferred:** This is a strategy-level decision, not a bug fix.
Worth backtesting both configurations against the last 30-60 closed
trades to see which produced better Sharpe. The current 29% WR is
borderline — concentration changes will affect WR-driven throttle
behavior in non-obvious ways.

**Effort:** 5 min code, ~30 min backtest.

### H2 — Recalibrate throttle: WR thresholds → expectancy

**What changes:** Replace win-rate-based throttle with expectancy-based:
```python
# Currently:
THROTTLE_WARN_WIN_RATE = 0.30  # halve if WR < 30%
THROTTLE_HALT_WIN_RATE = 0.20  # halt if WR < 20%

# Proposed:
THROTTLE_WARN_EXPECTANCY = 0.0   # halve if avg_pnl_pct ≤ 0
THROTTLE_HALT_EXPECTANCY = -1.5  # halt if avg_pnl_pct ≤ -1.5%
```

**Why:** A 2.0+ R:R strategy is profitable at 35% WR. The current
throttle at 30% WARN is punishing the system for behaving normally
(positive expectancy). Expectancy = `avg_win × WR − avg_loss × (1−WR)`
is what actually matters.

**Why deferred:** Same as H1 — strategy decision needing measurement
on the live trade history. The system has 7 closed trades; a 30-trade
window would give cleaner signal.

**Effort:** ~20 min code (touches `risk_manager.check_performance_throttle`),
~1 hr to validate against history.

### H7 — Selection ranking with live-WR awareness

**What changes:** Add a "this setup type's recent live performance"
multiplier to `_filter_candidates` ranking. E.g. lower rank for
sectors where the last 20 closed trades had WR < 25%.

**Why:** Current rank is forward-looking (score, RR, ML, sector mom,
insider). It has no memory — if the system has lost 5/5 in Energy
recently, the next Energy candidate ranks identically to a Tech
candidate with the same scores.

**Why deferred:** Needs historical-performance feature engineering
(per-sector WR, per-regime WR, rolling-window selection). Best done
with the same data analysis that informs H1+H2.

**Effort:** ~2 hrs.

---

## 🟠 Architectural cleanups

### H4 — Native IB bracket via `transmit=False/True`

**Status:** Reviewed and **not applicable to this design.**

The audit suggested using IB's atomic bracket-order pattern: place all
3 orders at once with `transmit=False` on the parent and `transmit=True`
on the last child, so IB's order server receives them as a unit.

**Why not applicable:** Native IB bracket sizes child legs from the
parent's `totalQuantity` rather than `filledQuantity`. With our market
buy + potential partial fill, the children would over-sell on
partials. The existing place-buy → wait-fill → place-protective flow
is the correct pattern; this commit (H4 follow-up in 9094137) tightened
the protective-order status-confirmation window to 5s, eliminating the
"PendingSubmit returned as success" risk.

The brief atomicity gap (~ms between buy fill and protective placement)
is covered by the monitor's resubmit path. Keep as-is.

### Cross-cutting #1 — REGIME_MIN_SCORE ownership

**What changes:** Move `REGIME_MIN_SCORE` (and `REGIME_RR_FLOOR` if
present) from `core/scoring_config.py` into `core/trading/policy.py`.
Have `scoring.py` import FROM there instead of the reverse.

**Why:** Trading layer currently treats `scoring_config` as a foreign
API. Inverting the dependency clarifies that the trading thresholds
own these tables — scoring just uses them for inclusion filtering.

**Why deferred:** Touches several import sites; medium-risk refactor.
**Effort:** ~30 min.

### Cross-cutting #4 — State-feed monotonic counter

**What changes:** Add a monotonic counter to each `system_state.json`
push so the dashboard can display "stale by N cycles" instead of
just absolute age.

**Why:** Today the dashboard shows "state age 22s" — fine for fresh.
But during a slow cycle the broadcaster might publish a snapshot from
BEFORE a scan completed, even though it's < 60s old. A counter would
let the dashboard show "snapshot is from cycle N, current is N+2".

**Why deferred:** Nice-to-have; current age display is good enough.
**Effort:** ~20 min.

### Drift-monitor bundle export pipeline

**What changes:** Update `train_ml_*.py` scripts to write
`feature_bins` and `training_bin_pct` into `metadata.json` (or as a
sibling `drift_bins.json`). Update `monitor_drift.py` to read from
there.

**Why:** Current state (after this session's C4 fix): drift monitor
recognizes the v3.6 bundle path but reports OFFLINE because the
bundle structure doesn't include the bins. The training pipeline
needs to start writing them.

**Why deferred:** Training pipeline change requires careful integration
testing to make sure model training itself isn't affected.
**Effort:** ~1 hr.

### Supabase trade_log mirror

See `docs/supabase_trade_log_mirror.md` for the full design.

**Status:** Documented, not implemented.
**Effort:** ~3.5 hrs.
**Priority:** Medium — duplicate-CLOSE guard (committed today) prevents
the specific incident from recurring; Supabase mirror prevents the
broader class of trade_log loss/corruption events.

### `monitor_positions.py` split

**What changes:** Split the 1437-line monitor into:
- `monitor/exit_logic.py` — close detection, fill matching
- `monitor/ratchet.py` — ratchet tier transitions
- `monitor/reconciliation.py` — drift detection + adopt + drop

**Why:** Current file is too long to navigate confidently.
**Why deferred:** Risky refactor without test coverage on the monitor;
the duplicate-CLOSE guard test we wrote today is the only meaningful
test of monitor logic.

**Effort:** ~3 hrs (split + test coverage).

---

## 🟡 Performance / cleanup

### H6 — Real live-price batching via `reqMktData` parallelism

**What changes:** Add `IBKRClient.get_live_prices_batch(tickers)` that
opens multiple market-data streams concurrently rather than serially.

**Why:** With max 3 candidates per session, current pattern adds ~18s
to trade window. If `max_daily_buys` ever rises to 5, that's 30+ s.
**Why deferred:** Requires IB stream-management code (subscribe → wait
all → unsubscribe atomically). Risk of leaking data subscriptions.
**Effort:** ~1 hr.

### N1 — Consolidate scan entry points

There are several near-duplicate scan scripts:
- `scan_500_live.py`
- `run_500_scan.py`
- `run_10_live_scan.py`
- `quick_scan.py`

Pick one canonical script (probably `run_500_scan.py` based on
structure), make others thin aliases or delete.

**Effort:** ~30 min.

### N2 — Archive legacy v3 model files

Move `models/model_20d_v3.pkl` and similar legacy bundles to
`archive/models/` so the active `ml/bundles/latest/` is unambiguously
the production model.

**Effort:** 5 min + verify no script references break.

### N3 — Multiple `risk_*` modules

`core/v2_risk_engine.py`, `core/risk_engine.py`, `core/risk_module.py`,
`core/risk/` — four risk-named places at the top level. Compose to
one or document why each exists.

**Effort:** ~2 hrs (need to read each).

### N4 — Document the "ratchet only tightens" invariant

`config.py:264-265` claims ratchet never widens TRAIL. Find the modify
call (probably `monitor_positions.py:1296+`) and add an inline assert
+ comment so a future reader can't widen TRAIL by accident.

**Effort:** 10 min.

### M6 — Measure BYPASS_DISABLED_ABOVE_MIN_SCORE impact

`core/scoring_config.py:564` sets `BYPASS_DISABLED_ABOVE_MIN_SCORE = 70`.
In SIDEWAYS regimes (current floor 70) the ML and pattern bypasses are
entirely off. Run an analysis on the last 30 scans: how often does the
candidate-pass rate hit zero? If >40%, the floors are too high.

**Why deferred:** Pure analysis, no code change. Need scan history
that's already in Supabase.
**Effort:** ~1 hr query + analysis.

### M8 — Reliability_Score weighting review

`core/scoring_config.py:RELIABILITY_WEIGHTS` is 40% completeness +
30% price-variance + 20% fundamental-coverage + 10% source-count.
None measure ACCURACY. For a 20-day strategy, price-variance is most
relevant — consider weighting it 50%+.

**Effort:** Tuning + re-backtest = ~1 hr.

---

## ✅ What's NOT deferred (fully done in this audit cycle)

- **C1**: max_drawdown_pct enforced via check_drawdown_breaker
- **C2**: IB executions day-trade check (GFV prevention)
- **C3**: daily-loss breaker fail-CLOSED on exception
- **C4**: monitor_drift reads v3.6 bundle path + clear OFFLINE status
- **C5**: _load_blocked_tickers TTL cache + last-known-good fallback
- **H3**: Earnings gate widened 3 → 5 days
- **H5**: Single-pass ranking weights (fixed silent 45% → 36% drift)
- **H8**: Ladder vs ratchet thresholds staggered (no more conflict at +10%)
- **H4 (verify-window)**: Protective-order status-confirmation window 2s → 5s
- **M2**: Removed dead code in `risk_manager` cash-check
- **M3**: `check_cash_after_buy` extended to sub-$2k tier
- **M4**: `tracker.get_trade_log()` cached (4× → 1× per gate eval)
- **M5**: Analyst-PT cache pre-warmed for top-N candidates
- **M7**: `/pnl` shows max DD + profit factor
- **Cross-cut #2**: yfinance health check (single point of failure)
- **Cross-cut #3**: `can_open_position` calls `evaluate_static_gates`
  for true runner-level parity with the dashboard

Plus the durability work from earlier in the same session:
- Duplicate-CLOSE guard
- VPS heartbeat (GitHub Action)
- 53 pytest tests
- Runtime files untracked from git
- `consecutive-miss` counter for monitor reconciliation
- Streamlit dispatch flow (GITHUB_TOKEN, command-poller systemd)

---

## How to use this file

When starting the next session: read this file first. Pick ONE
medium-effort item and finish it cleanly rather than starting many.
Mark it with ✅ in the "fully done" section above and update this
file in the same commit.
