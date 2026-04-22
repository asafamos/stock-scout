# StockScout System State — 2026-04-22 EOD

End-of-day snapshot after a full day of production-hardening.

## 🟢 What's Working

### Trading stack (VPS, 24/5)
| Service | Timer | Status |
|---|---|---|
| `stockscout-monitor` | every 5 min during market hours | ✅ active |
| `stockscout-statusbot` | always (Telegram poller) | ✅ active |
| `stockscout-trade.timer` | 14:30/15:55/18:30/20:55 UTC | ✅ active |
| `stockscout-healthcheck.timer` | every 15 min | ✅ active |
| `stockscout-daily-summary.timer` | 20:15 UTC (after close) | ✅ active |
| `stockscout-outcomes-record.timer` | 14:45/15:10/20:10/21:40 UTC | ✅ active |
| `stockscout-outcomes-resolve.timer` | 03:00 UTC nightly | ✅ active |
| `stockscout-state-backup.timer` | 23:00 UTC nightly | ✅ active |
| `stockscout-snapshot-sync.timer` | (existing) | ✅ active |

### Automation (GitHub Actions)
| Workflow | Cadence | Last Run | Status |
|---|---|---|---|
| `weekly-training.yml` (nightly ML) | Mon-Fri 22:00 UTC | 2026-04-21 | ✅ |
| `auto_scan.yml` | 4×/day | 2026-04-22 | ✅ |
| `track_outcomes.yml` | daily 22:00 UTC | 2026-04-21 | ✅ (but see bugs below) |
| `weekly_backtest.yml` | Sunday 06:30 UTC | 2026-04-19 | ✅ |

### Open positions (verified live in IBKR)
| Ticker | Qty | Entry | Stop/Trail | Target | OCA |
|---|---|---|---|---|---|
| CF   | 2 | $111.33 | STP $111.33 | LMT $145.46 | SS_CF_1776792064 |
| MUSA | 1 | $498.50 | TRAIL 3% | LMT $549.15 | SS_MUSA_1776778504 |
| TDW  | 2 | $85.83  | TRAIL 3% (activates tomorrow) | LMT $93.95 | SS_TDW_1776864914 |

### Phase 1-3 improvements deployed (16 items + 5 extras)

**Phase 1** — Reliability hardening: file locking, monitor timeouts, target/stop validation, stale-scan rejection.

**Phase 2** — Smarter selection: scan outcome tracker (regime-tagged JSONL), AUC-adaptive ML weights, sector momentum gate, analyst target cap.

**Phase 3** — Correlation & parity: portfolio correlation check, backtest gap-guard + intraday OHLC exits.

**Observability extras**: regime-tagged outcomes, dashboard summary, nightly git backup, tracker staleness alert, 8 unit tests (all pass).

## ⚠️ Known Issues (deferred to next sprint)

### 1. DuckDB outcome tracking is broken
On VPS: `recommendations` table is **empty** (0 rows). Scan pipeline fails silently when trying to `save_scan()` — likely a missing dependency or write-permission issue. Verification:
```sql
SELECT COUNT(*) FROM recommendations  -- returns 0 on VPS
```
Impact: the "official" DuckDB outcome system is not receiving data. **Mitigation**: our new JSONL tracker (`scripts/track_scan_outcomes.py`) is running in parallel and IS capturing all outcomes with regime tags. The JSONL is the source of truth until DuckDB is repaired.

### 2. Even the committed DuckDB has `market_regime=NULL`
Schema supports it; `_REC_COL_MAP` maps "Market_Regime" → "market_regime"; the column is preserved through the mapping (verified manually). Yet every recommendation has NULL regime. Root cause not identified — likely requires tracing `save_scan()` with a real scan to find where the value is getting dropped.
**Mitigation**: JSONL tracker captures regime correctly.

### 3. ML retraining doesn't feed from our actual outcomes
`weekly-training.yml` uses Polygon data on 1000 tickers — trains on generic market data, not on our scan history. This is fine as a baseline but we're leaving learning on the table.
**Next sprint**: wire `scan_outcomes.jsonl` → retraining pipeline.

### 4. Phase 3c (multi-regime model) not started
Requires 200+ resolved outcomes with regime tags to be meaningful. Will unlock ~2026-05-20 when our tracker matures.

### 5. Cash account $2000 rule
Account balance $977 — IBKR blocks new protective orders with "MINIMUM OF 2000 USD" error. System handles this gracefully (cooldowns, silent skip, existing orders untouched). Remediates once balance ≥ $2000 or account upgraded to margin.

### 6. Day Trade warning window
IBKR flagged 1 day-trade violation (SII). 5-business-day probation period active until 2026-04-28. A second violation in this window would cause account restrictions. **Code prevents this** via `goodAfterTime` + tracker guards.

## 📋 Telegram Commands

```
status              → portfolio + live orders + coverage
/pnl                → today's realized + unrealized + lifetime
/history            → last 10 closed trades
/panic              → two-step kill switch (preview → confirm)
help                → this list
```

## 🔮 Next Sprint Candidates (post ~2026-05-20)

1. Fix DuckDB regime persistence (root cause investigation)
2. Wire JSONL outcomes → ML retraining feedback loop
3. Build phase 3c: multi-regime model (3 separate model paths by regime)
4. Kelly-fraction sizing (needs 30+ real trades)
5. Add earnings calendar avoidance filter
6. Unit tests for scoring/scan pipeline (currently untested)
