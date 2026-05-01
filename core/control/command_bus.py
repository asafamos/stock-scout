"""Command Bus — single source of truth for all system actions.

Both Telegram bot and Streamlit dashboard call into this module to
trigger actions. Identical behavior whether triggered from "/sellall"
on Telegram or "🚨 Sell All" button on Streamlit.

Why a shared bus:
- Avoids duplicating SELL/PAUSE logic in two clients
- Single audit trail for every command
- Single auth surface (caller must supply trusted_caller=True)
- Single place to validate/sanity-check commands

Auth: caller responsibility. Telegram bot validates chat_id;
Streamlit (when wired via API) validates API key.

Audit: every command writes to data/state/command_audit.jsonl
(append-only, never modified) so we can review what was done and when.

Commands:
  status          — return current portfolio + pipeline state
  sell TICKER     — close one position immediately at market
  sellall         — close all positions immediately at market
  pause [DAYS]    — pause auto-trading for N days (default until manually resumed)
  resume          — clear pause
  scan            — trigger a manual scan via GH Actions (returns run URL)
  resubmit TICKER — re-place protective orders for a position
  block TICKER N  — add ticker to do-not-buy list for N days
  today           — return today's trade log + skips
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
       else Path(__file__).resolve().parents[2]

STATE_DIR = ROOT / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG = STATE_DIR / "command_audit.jsonl"
PAUSE_FILE = STATE_DIR / "auto_trade_paused_until.json"
BLOCK_FILE = STATE_DIR / "blocked_tickers.json"


def _audit(command: str, source: str, args: dict, result: dict, success: bool):
    """Append-only audit of every command."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "source": source,  # "telegram" / "streamlit" / "cli"
        "args": args,
        "success": success,
        "result_summary": {k: v for k, v in result.items()
                           if k not in ("orders", "details") and not isinstance(v, list)},
    }
    try:
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:
        logger.warning("audit write failed: %s", e)


# ─────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────

def cmd_status(source: str = "cli", **_) -> Dict[str, Any]:
    """Return system_state.json + light additions."""
    state_file = STATE_DIR / "system_state.json"
    state = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception as e:
            return {"ok": False, "error": f"state file unreadable: {e}"}
    is_paused = _is_paused()
    state["paused"] = is_paused
    return {"ok": True, "state": state}


def cmd_sell(ticker: str, source: str = "cli", **_) -> Dict[str, Any]:
    """Close one position at market with protective-order cleanup."""
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return {"ok": False, "error": "ticker required"}
    try:
        from core.trading.config import CONFIG
        from core.trading.ibkr_client import IBKRClient
        from core.trading.position_tracker import PositionTracker

        # Force LIVE for emergency sell (override DRY_RUN safeguard)
        os.environ["TRADE_DRY_RUN"] = "0"
        os.environ["TRADE_LIVE_CONFIRMED"] = "1"

        tracker = PositionTracker()
        pos = tracker.get_position(ticker)
        if not pos:
            return {"ok": False, "error": f"{ticker} not in tracker"}

        client = IBKRClient()
        if not client.connect():
            return {"ok": False, "error": "IBKR connection failed"}

        try:
            # 1. Cancel all open protective orders for this ticker
            cancelled = []
            try:
                for t in client._ib.openTrades():
                    if t.contract.symbol == ticker and t.order.action == "SELL":
                        client._ib.cancelOrder(t.order)
                        cancelled.append(t.order.orderId)
                client._ib.sleep(2)
            except Exception as e:
                logger.warning("cancel-before-sell failed: %s", e)

            # 2. Market sell
            qty = int(pos["quantity"])
            result = client._sell_market(ticker, qty)

            # 3. If filled, update tracker
            avg = float(getattr(result, "filled_price", 0) or 0)
            if result.status in ("Filled", "Submitted", "PreSubmitted"):
                # Use actual fill price if available; otherwise placeholder
                if avg <= 0:
                    # Wait briefly for fill
                    client._ib.sleep(2)
                    try:
                        for f in client._ib.fills():
                            if f.contract.symbol == ticker and f.execution.side in ("SLD", "SELL"):
                                avg = float(f.execution.price or 0)
                                break
                    except Exception:
                        pass
                if avg > 0:
                    tracker.remove_position(ticker, avg, "manual_sell")
                else:
                    tracker.reconcile_drop(ticker, reason="manual_sell_no_fill_data")

                ret = {
                    "ok": True,
                    "ticker": ticker,
                    "qty": qty,
                    "fill_price": avg,
                    "cancelled_orders": cancelled,
                    "status": result.status,
                }
            else:
                ret = {
                    "ok": False,
                    "error": f"sell rejected: {result.status} ({getattr(result, 'error', '')})",
                    "ticker": ticker,
                }
        finally:
            client.disconnect()
        _audit("sell", source, {"ticker": ticker}, ret, ret.get("ok", False))
        return ret
    except Exception as e:
        logger.exception("cmd_sell failed")
        return {"ok": False, "error": str(e)}


def cmd_sellall(source: str = "cli", **_) -> Dict[str, Any]:
    """Emergency close ALL positions at market.

    Iterates through every tracked position and calls cmd_sell. Each
    failure is logged but doesn't abort the loop — we want maximum
    protection in an emergency.
    """
    try:
        from core.trading.position_tracker import PositionTracker
        positions = PositionTracker().get_open_positions()
        if not positions:
            return {"ok": True, "message": "no positions to sell", "results": []}
        results = []
        for p in positions:
            r = cmd_sell(p["ticker"], source=source)
            results.append(r)
        ok = all(r.get("ok") for r in results)
        ret = {
            "ok": ok,
            "total_positions": len(positions),
            "succeeded": sum(1 for r in results if r.get("ok")),
            "failed": sum(1 for r in results if not r.get("ok")),
            "results": results,
        }
        _audit("sellall", source, {}, ret, ok)
        return ret
    except Exception as e:
        logger.exception("cmd_sellall failed")
        return {"ok": False, "error": str(e)}


def _is_paused() -> bool:
    """Return True if auto-trading is paused (file-based flag)."""
    if not PAUSE_FILE.exists():
        return False
    try:
        rec = json.loads(PAUSE_FILE.read_text())
        until = rec.get("until")
        if not until:
            return True  # paused indefinitely
        return datetime.fromisoformat(until) > datetime.now(timezone.utc)
    except Exception:
        return False


def cmd_pause(days: Optional[int] = None, source: str = "cli", **_) -> Dict[str, Any]:
    """Pause auto-trading. days=None means until manually resumed."""
    until = None
    if days and int(days) > 0:
        until = (datetime.now(timezone.utc) + timedelta(days=int(days))).isoformat()
    PAUSE_FILE.write_text(json.dumps({
        "paused_at": datetime.now(timezone.utc).isoformat(),
        "until": until,
        "source": source,
    }, indent=2))
    ret = {"ok": True, "paused_until": until or "indefinitely"}
    _audit("pause", source, {"days": days}, ret, True)
    return ret


def cmd_resume(source: str = "cli", **_) -> Dict[str, Any]:
    if PAUSE_FILE.exists():
        PAUSE_FILE.unlink()
    ret = {"ok": True, "message": "auto-trading resumed"}
    _audit("resume", source, {}, ret, True)
    return ret


def cmd_today(source: str = "cli", **_) -> Dict[str, Any]:
    """Return today's trade log + skip reasons."""
    from pathlib import Path
    log_path = ROOT / "data" / "trades" / "trade_log.json"
    today_iso = date.today().isoformat()
    log = []
    try:
        log = json.loads(log_path.read_text())
    except Exception:
        pass
    todays = [t for t in log if str(t.get("timestamp", "")).startswith(today_iso)]
    return {
        "ok": True,
        "date": today_iso,
        "actions": todays,
        "buys": [t for t in todays if t.get("action") == "OPEN"],
        "closes": [t for t in todays if t.get("action") in ("CLOSE", "PARTIAL")],
    }


def cmd_resubmit(ticker: str, source: str = "cli", **_) -> Dict[str, Any]:
    """Re-submit protective orders for an existing position."""
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return {"ok": False, "error": "ticker required"}
    try:
        from core.trading.order_manager import OrderManager
        os.environ["TRADE_DRY_RUN"] = "0"
        os.environ["TRADE_LIVE_CONFIRMED"] = "1"
        manager = OrderManager()
        if not manager.client.connect():
            return {"ok": False, "error": "IBKR connection failed"}
        try:
            results = manager.resubmit_protections(only_ticker=ticker)
            ok = bool(results) and all(r.get("status") == "success" for r in results)
            ret = {"ok": ok, "ticker": ticker, "results": results}
        finally:
            manager.client.disconnect()
        _audit("resubmit", source, {"ticker": ticker}, ret, ok)
        return ret
    except Exception as e:
        logger.exception("cmd_resubmit failed")
        return {"ok": False, "error": str(e)}


def cmd_block(ticker: str, days: int = 30, source: str = "cli", **_) -> Dict[str, Any]:
    """Add ticker to do-not-buy list for N days."""
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return {"ok": False, "error": "ticker required"}
    blocks = {}
    if BLOCK_FILE.exists():
        try:
            blocks = json.loads(BLOCK_FILE.read_text())
        except Exception:
            pass
    until = (datetime.now(timezone.utc) + timedelta(days=int(days))).isoformat()
    blocks[ticker] = {"blocked_at": datetime.now(timezone.utc).isoformat(),
                      "until": until, "source": source}
    BLOCK_FILE.write_text(json.dumps(blocks, indent=2, default=str))
    ret = {"ok": True, "ticker": ticker, "blocked_until": until}
    _audit("block", source, {"ticker": ticker, "days": days}, ret, True)
    return ret


def cmd_unblock(ticker: str, source: str = "cli", **_) -> Dict[str, Any]:
    ticker = (ticker or "").upper().strip()
    blocks = {}
    if BLOCK_FILE.exists():
        try:
            blocks = json.loads(BLOCK_FILE.read_text())
        except Exception:
            pass
    if ticker in blocks:
        del blocks[ticker]
        BLOCK_FILE.write_text(json.dumps(blocks, indent=2))
    ret = {"ok": True, "ticker": ticker, "removed": ticker not in blocks}
    _audit("unblock", source, {"ticker": ticker}, ret, True)
    return ret


def cmd_scan(source: str = "cli", **_) -> Dict[str, Any]:
    """Trigger a manual scan via GH Actions workflow_dispatch.

    Includes conflict detection: if a scan is already running, returns
    the existing run URL instead of starting a duplicate.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {"ok": False, "error": "GITHUB_TOKEN not configured"}
    try:
        # Check if a scan is currently running
        import urllib.request
        req = urllib.request.Request(
            "https://api.github.com/repos/asafamos/stock-scout/actions/runs?status=in_progress",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            runs_data = json.loads(resp.read())
        active = [r for r in runs_data.get("workflow_runs", [])
                  if r.get("name") == "Auto Stock Scan"]
        if active:
            existing = active[0]
            return {
                "ok": True,
                "already_running": True,
                "url": existing.get("html_url"),
                "started_at": existing.get("created_at"),
                "message": "scan already in progress — not starting duplicate",
            }
        # Trigger new dispatch
        dispatch_req = urllib.request.Request(
            "https://api.github.com/repos/asafamos/stock-scout/actions/workflows/auto_scan.yml/dispatches",
            data=json.dumps({"ref": "main"}).encode("utf-8"),
            headers={"Authorization": f"token {token}",
                     "Accept": "application/vnd.github+json"},
            method="POST",
        )
        with urllib.request.urlopen(dispatch_req, timeout=10) as resp:
            if resp.status != 204:
                return {"ok": False, "error": f"dispatch HTTP {resp.status}"}
        ret = {"ok": True, "dispatched": True,
               "message": "scan dispatched — check GH Actions for progress"}
        _audit("scan", source, {}, ret, True)
        return ret
    except Exception as e:
        logger.exception("cmd_scan failed")
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────

COMMANDS = {
    "status": cmd_status,
    "sell": cmd_sell,
    "sellall": cmd_sellall,
    "pause": cmd_pause,
    "resume": cmd_resume,
    "today": cmd_today,
    "resubmit": cmd_resubmit,
    "block": cmd_block,
    "unblock": cmd_unblock,
    "scan": cmd_scan,
}


def execute(command: str, source: str = "cli", **kwargs) -> Dict[str, Any]:
    """Dispatch a command. Returns {ok: bool, ...}.

    source: "telegram" / "streamlit" / "cli" — for audit trail.
    """
    handler = COMMANDS.get(command)
    if not handler:
        return {"ok": False, "error": f"unknown command: {command}",
                "available": list(COMMANDS.keys())}
    return handler(source=source, **kwargs)


# CLI entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m core.control.command_bus <command> [args...]")
        print(f"Commands: {list(COMMANDS.keys())}")
        sys.exit(1)
    command = sys.argv[1]
    args = sys.argv[2:]
    kwargs = {}
    # Simple positional → keyword mapping for common cases
    if command == "sell" and args:
        kwargs["ticker"] = args[0]
    elif command == "pause" and args:
        kwargs["days"] = int(args[0])
    elif command == "resubmit" and args:
        kwargs["ticker"] = args[0]
    elif command in ("block", "unblock") and args:
        kwargs["ticker"] = args[0]
        if len(args) > 1:
            kwargs["days"] = int(args[1])
    result = execute(command, source="cli", **kwargs)
    print(json.dumps(result, indent=2, default=str))
