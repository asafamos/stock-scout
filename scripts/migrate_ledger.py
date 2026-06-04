"""One-time migration to the event-sourced ledger (docs/architecture_ledger.md).

Run ONCE on the VPS after deploying the ledger code, with a live IB session:

    TRADE_DRY_RUN=0 .venv/bin/python -m scripts.migrate_ledger

What it does (all idempotent — safe to re-run):

  1. Ingests the current session's IB executions into the ledger
     (executions.jsonl), keyed by execId.
  2. Anchors the reconciliation baseline to the BROKER at cutover:
       pre_ledger_realized = account_truth − ledger_total
     where account_truth = NetLiq − starting_capital − open_unrealized.
     This makes /pnl's reconciliation read Δ≈0 today, so any FUTURE
     divergence is genuine new drift — not the (unrecoverable) historical
     gap from the legacy gross-of-fee / dropped-trade bookkeeping.

It does NOT touch open_positions.json or trade_log.json — the cutover is
non-destructive and reversible (set TRADE_LEDGER_ENABLED=0 to revert).
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    from core.trading.config import CONFIG
    from core.trading.ibkr_client import IBKRClient
    from core.trading import ledger

    if CONFIG.dry_run:
        logger.error("Refusing to migrate in DRY_RUN — need a LIVE IB session "
                     "for real fills/net-liq. Re-run with TRADE_DRY_RUN=0.")
        return 1

    client = IBKRClient()
    if not client.connect():
        logger.error("IB connection failed — start the gateway and retry.")
        return 2
    try:
        # 1. Ingest current-session executions.
        new = ledger.ingest(client)
        logger.info("Ingested %d new execution(s) into the ledger.", len(new))

        # 2. Anchor reconciliation baseline to the broker.
        net_liq = float(client.get_net_liquidation() or 0.0)
        unrealized = 0.0
        try:
            for p in client._ib.portfolio():
                if p.position != 0:
                    unrealized += float(p.unrealizedPNL or 0)
        except Exception as e:
            logger.warning("Could not read open unrealized: %s", e)
        unrealized = round(unrealized, 2)

        account_truth = ledger.lifetime_realized_truth(net_liq, unrealized)
        ledger_total = ledger.realized_ledger_total()
        baseline = round(account_truth - ledger_total, 2)
        ledger.set_pre_ledger_baseline(baseline)

        logger.info("─" * 56)
        logger.info("Starting capital     : $%.2f", CONFIG.starting_capital)
        logger.info("IB net liquidation   : $%.2f", net_liq)
        logger.info("Open unrealized      : $%.2f", unrealized)
        logger.info("Account-truth realized (NetLiq−start−open): $%.2f", account_truth)
        logger.info("Ledger realized      : $%.2f", ledger_total)
        logger.info("Pre-ledger baseline  : $%.2f  (anchored, Δ now ≈ 0)", baseline)
        logger.info("─" * 56)

        rec = ledger.reconcile(net_liq, unrealized)
        logger.info("Reconciliation: Δ $%.2f → %s",
                    rec["delta"], "OK ✅" if rec["ok"] else "DRIFT ⚠️")
        if CONFIG.starting_capital <= 0:
            logger.warning("starting_capital is 0 — set TRADE_STARTING_CAPITAL "
                           "to your initial deposit for a correct lifetime number.")
        return 0
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
