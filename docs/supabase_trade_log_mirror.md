# Supabase Trade-Log Mirror — Implementation Plan

**Status:** Planned, not yet implemented.

**Why:** On 2026-05-01 a routine `git pull` on the VPS overwrote the
live `data/trades/trade_log.json` with a stale committed snapshot,
losing today's earlier events (SVM open/close, RDDT open, ORCL adopt,
TDW reconcile_drop). The `/pnl` command then reported wrong realized
totals until manual recovery.

This document describes how to add a Supabase-backed audit mirror so
the trade log survives any future overwrite, file corruption, or
disk-loss event.

---

## What we already do

- `data/trades/trade_log.json` — local append-only-ish JSON. Mutable
  on disk; can be lost.
- `data/trades/open_positions.json` — local current state. Now
  `.gitignore`d (commit `9bcc9d0`) so future pulls won't overwrite.
- `system_state.json` on the `state-feed` branch — broadcast every
  30s by the VPS. Includes `trade_log_today` (filtered to today only).
  Force-pushed, so old snapshots are not retained.
- Supabase table `scan_results` — already populated by the auto-scan
  pipeline. Connection working, secrets configured.

## What we still don't have

A **durable, append-only, off-VPS** record of every BUY/CLOSE/PARTIAL/
RECONCILE_DROP/RECONCILE_ADOPT event. Today the canonical record is
the local JSON, which is one `git reset --hard` (or one corrupt
write) away from being lost.

---

## Design

### Schema

```sql
create table trade_events (
  id              bigserial primary key,
  ts              timestamptz not null default now(),
  -- Mirror of the trade_log row's own timestamp (when the event occurred,
  -- not when it was written to Supabase). They differ if the mirror
  -- catches up after a downtime.
  event_ts        timestamptz not null,
  action          text not null,         -- OPEN / CLOSE / PARTIAL / RECONCILE_DROP / RECONCILE_ADOPT
  ticker          text not null,
  quantity        integer,
  price           numeric(10, 4),
  pnl             numeric(12, 2),         -- null for OPEN, RECONCILE_DROP, RECONCILE_ADOPT
  reason          text,
  oca_group       text,                   -- when applicable
  metadata        jsonb,                  -- the rest of the event verbatim
  source          text default 'vps',     -- vps / mac / backfill / manual
  -- Idempotency key: (ticker, action, event_ts) tuple is unique to one
  -- event. Re-running the mirror on the same trade_log.json must NOT
  -- create duplicate rows.
  unique (ticker, action, event_ts)
);

create index idx_trade_events_event_ts on trade_events (event_ts desc);
create index idx_trade_events_ticker_today
  on trade_events (ticker, event_ts)
  where event_ts > current_date;
```

### Write path: in `position_tracker._log_trade`

After the local JSON write succeeds, fire-and-forget a background
thread that pushes the same row to Supabase. Failures must NEVER
break the local write — Supabase being down is degraded, not
catastrophic. A background queue with bounded retry handles transient
failures. Persistent failures are logged to a `pending_supabase_writes.json`
file that a daily cron drains.

```python
# core/trading/position_tracker.py
def _log_trade(self, action, ticker, qty, price, extra):
    # ... existing local write (unchanged)
    row = {...}
    self._append_local(row)

    # NEW: best-effort Supabase mirror
    try:
        from core.db.supabase_audit import enqueue_trade_event
        enqueue_trade_event(row)  # non-blocking — runs in background thread
    except Exception:
        pass  # never crash the caller
```

### Read path: backfill + recovery

```python
# scripts/recover_trade_log.py
"""Rebuild local trade_log.json from Supabase for a given date range.
Used after a git-pull-overwrites-tracker incident."""

def recover(start_date, end_date, dry_run=True):
    rows = supabase.table("trade_events") \
        .select("*") \
        .gte("event_ts", start_date) \
        .lte("event_ts", end_date) \
        .order("event_ts") \
        .execute()
    local = json.load(open("data/trades/trade_log.json"))
    # Merge by (ticker, action, event_ts) tuple
    by_key = {(r["ticker"], r["action"], r["event_ts"][:19]): r for r in local}
    added = 0
    for remote in rows.data:
        k = (remote["ticker"], remote["action"], remote["event_ts"][:19])
        if k not in by_key:
            local.append(_supabase_row_to_log_format(remote))
            added += 1
    if not dry_run:
        json.dump(sorted(local, key=lambda r: r["timestamp"]),
                  open("data/trades/trade_log.json", "w"), indent=2)
    return added
```

### Daily reconciliation job

A new GitHub Action runs nightly (e.g. 23:30 UTC = 02:30 IL):

1. Fetch `trade_events` for the day from Supabase.
2. Fetch the VPS's `trade_log.json` for today's date via state-feed
   or a one-off SSH read.
3. Diff the two sets by `(ticker, action, event_ts)` key.
4. If counts differ → Telegram alert with the diff so the operator
   can verify which side is correct.

This catches:
- VPS wrote locally but Supabase mirror failed silently → Supabase
  has fewer rows.
- Someone manually edited `trade_log.json` → drift in either
  direction.
- Both sides agree → daily green check, peace of mind.

---

## Estimated effort

| Step | Effort |
|------|--------|
| Schema + Supabase migration | 15 min |
| `enqueue_trade_event` helper + background worker | 1 hr |
| Wire into `_log_trade` | 15 min |
| `scripts/recover_trade_log.py` | 30 min |
| GitHub Action for nightly diff | 30 min |
| Tests (mirror works, idempotent on replay, recovery merges correctly) | 1 hr |
| **Total** | **~3.5 hrs** |

Not done tonight because (a) the system is currently healthy, (b) the
duplicate-CLOSE guard + the runtime-files untrack already prevent the
two specific scenarios that caused today's incident, and (c) this
needs careful integration testing that's better done in daylight.

---

## Why this is worth doing eventually

The trading system makes real-money decisions based on `/pnl`,
`/today`, throttle win-rate calculations, ML feedback loops, and
daily summaries — all of which read from `trade_log.json`. Today
proved that file is fragile. Adding Supabase mirror turns it from a
single point of failure into a redundant pair: lose one, recover from
the other. The cost is one bounded background thread per write
(invisible at our trade rate of ~5/day).

The risk we're insuring against is real: today the loss was 4 events
worth $3.68 in unaccounted realized P&L. Next time it could be a
hundred events spanning weeks if a backup is restored, a disk
corrupts, or a config-typo wipes the file.
