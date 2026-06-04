# Architecture: IB-as-source-of-truth + event-sourced ledger

**Status:** implemented behind `TRADE_LEDGER_ENABLED` (default ON). Reversible
— set `TRADE_LEDGER_ENABLED=0` to fully restore the legacy `trade_log`
behavior.

## The problem this fixes (root cause, not symptom)

For months the bot suffered recurring **tracker↔IB drift**: `/status` (live
from IB) and `/pnl` + `/history` + the daily summary (derived from
`trade_log.json`) told **different stories**. Positions "vanished" (PAAS,
2026-06-03), losses disappeared, and the lifetime P&L did not reconcile with
the broker.

Every previous fix — grace windows (`9ca1a10`), drift suppression
(`eb0d92e`), consecutive-miss counters, `reconcile_adopt`/`reconcile_drop`,
snapshot-stale auto-recover — was **reconciliation logic**: a band-aid on a
**dual-source-of-truth** architecture. Two authoritative stores of the same
facts (the JSON tracker and IBKR) inevitably diverge. Drift was not a bug to
patch; it was the architecture.

### The smoking gun

Account opened at **$977.50** (no withdrawals). IB reported **NetLiq
$959.80**, open unrealized **−$17.05** → true realized ≈ **−$0.65**
(flat). The bot reported **lifetime realized +$74.05** — an overstatement of
~$75 (≈7.6% of the account), because:

1. `reconcile_drop` wrote `pnl=None`, and all stats counted only
   `action=="CLOSE"` → **real losses the monitor couldn't price simply
   vanished** from the books.
2. P&L was `(exit−entry)×qty` — **gross of commissions**.

## The model

**IB is the single source of truth.** The local JSON drops from
"authoritative shadow of IB" to a **metadata annotation layer**.

| Datum | Source of truth | Local JSON role |
|---|---|---|
| Hold X? qty? avg cost? unrealized? | IB `positions()`/`portfolio()` | none (derived) |
| Realized P&L per close (net of fees) | IB `commissionReport.realizedPNL` per `execId` → **ledger** | durable cache |
| Lifetime realized (headline) | `NetLiq − starting_capital − open_unrealized` (pure IB) | none |
| Total account value | IB `NetLiquidation` | none |
| Protective orders | IB `openOrders()` | OCA id hint only |
| Metadata: score, thesis, target_date, intended trail, opened_at | **us** | **authoritative** |

## The ledger (`core/trading/ledger.py`)

Append-only `data/trades/executions.jsonl`, **idempotent by `execId`** (IB's
globally-unique execution id). `ledger.ingest(client)` pulls `ib.fills()`
each monitor cycle and appends only unseen executions. Therefore:

- An execution can never be **double-counted** (kills the 2026-05-01
  double-CLOSE class).
- An execution is never **estimated/inferred** (kills the KNX/PAAS phantom
  class).
- `realized_pnl` is IB's own number, **net of commissions** → reconciles to
  NetLiquidation by construction.

`ib.fills()` is session-scoped (~today); the durable ledger accumulates
across sessions because we only ever **append** what IB reports.

## What changed in the monitor

`run_check()` now `ledger.ingest()`s at the top of each cycle and returns the
**new** executions. The close path no longer *infers* a close by diffing
positions and *hunting* for a price; in ledger mode it simply
`drop_metadata()` (no fabricated `trade_log` row) and reacts to any fresh
SELL execution (notify + opportunistic-buy). The entire
inference/`reconcile_drop` machinery is bypassed (`TRADE_LEDGER_ENABLED=0`
restores it verbatim).

## Reporting

`/pnl`, `/history`, and the daily summary read **IB-live + ledger** in ledger
mode:

- **Today realized** = Σ ledger SELL `realizedPNL`.
- **Lifetime realized** = `NetLiq − starting_capital − open_unrealized` (the
  always-true account identity; covers pre-ledger history too).
- **Record / Profit Factor / hold-time** = ledger round-trips.
- **Reconciliation line** = ledger-cumulative vs account-truth; surfaces any
  Δ instead of silently overstating. This is the single check that would
  have caught the $75 gap.

(Also fixed a latent dead-code bug: `/pnl` imported a nonexistent
`pair_buy_sell_events` → `ImportError` swallowed → Profit-Factor/Max-Drawdown
never displayed. Now uses `build_trade_pairs`.)

## Migration (run once, on the VPS, live IB)

```bash
TRADE_DRY_RUN=0 .venv/bin/python -m scripts.migrate_ledger
```

Ingests current fills and **anchors the reconciliation baseline to the
broker** at cutover (`pre_ledger_realized = account_truth − ledger_total`),
so `/pnl` reads Δ≈0 today and any future divergence is genuine new drift. The
unrecoverable historical gap (legacy gross-of-fee / dropped trades) is
captured in the baseline, not silently carried as fake profit.

Set your deposit: `TRADE_STARTING_CAPITAL=977.50` (env), so the lifetime
identity is correct.

## Config / env flags

| Env | Default | Meaning |
|---|---|---|
| `TRADE_LEDGER_ENABLED` | `1` | Master switch. `0` = legacy `trade_log` behavior. |
| `TRADE_STARTING_CAPITAL` | `977.50` | Initial deposit, for the lifetime identity. |
| `TRADE_RECONCILE_TOLERANCE_USD` | `5.0` | Δ above this flags drift in `/pnl`. |

## Implications / known limitations (by design)

- **IB availability:** live reads require the gateway. When it's down, reports
  should say so honestly rather than serve stale-but-confident JSON. (The
  status bot already surfaces "Cannot connect to IB".)
- **Pre-cutover per-trade history** cannot be reconstructed from the IB API
  (no months-deep execution feed). `/history` shows ledger trades from
  cutover forward; lifetime *total* remains correct via the NetLiq identity.
- **Healthcheck `tracker holds X but IB doesn't`** (`deploy/healthcheck.sh`):
  in ledger mode this is expected, short-lived churn — the monitor GC's stale
  metadata within ~2 cycles. The 30-min suppression already covers it; a
  follow-up can make that alert ledger-aware.
- **`open_positions.json`** is still written by `add_position` (it carries our
  metadata + drives the monitor's exit logic), but it no longer determines
  position *existence* or *P&L*. A future cleanup can rebuild it as a pure
  IB-join on read.
