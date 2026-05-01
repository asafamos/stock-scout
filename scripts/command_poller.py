"""Command poller — watches the `commands` branch on GitHub for new
commands queued from Streamlit, processes them via command_bus, and
posts results back to Telegram.

Architecture:
  Streamlit button → repository_dispatch → command_dispatch.yml workflow →
  appends to commands/queue.jsonl on `commands` branch → THIS poller picks
  up → command_bus.execute(...) → result back to Telegram.

End-to-end latency: ~30-60 seconds (workflow_dispatch + 15s poll cycle).

Why through GitHub instead of a direct API:
- No need to expose VPS to public internet
- Free auth (GitHub token)
- Audit trail in git history
- Single source of truth for pending commands

Idempotency: each command has an `id`. We track processed IDs in
data/state/processed_command_ids.json so the same command never runs
twice even if the poller restarts.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
       else Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

PROCESSED_FILE = ROOT / "data" / "state" / "processed_command_ids.json"
PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL = 15  # seconds
QUEUE_URL = (
    "https://raw.githubusercontent.com/asafamos/stock-scout/"
    "commands/commands/queue.jsonl"
)


def _load_processed() -> Set[str]:
    if not PROCESSED_FILE.exists():
        return set()
    try:
        return set(json.loads(PROCESSED_FILE.read_text()))
    except Exception:
        return set()


def _save_processed(ids: Set[str]):
    # Cap at last 10,000 to prevent unbounded growth
    keep = list(ids)[-10000:]
    PROCESSED_FILE.write_text(json.dumps(keep))


def _fetch_queue() -> List[Dict]:
    try:
        req = urllib.request.Request(
            QUEUE_URL,
            headers={"User-Agent": "StockScoutCommandPoller/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                logger.warning("malformed queue line: %s", e)
        return out
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []  # branch or file doesn't exist yet
        logger.warning("queue fetch HTTP error: %s", e)
        return []
    except Exception as e:
        logger.warning("queue fetch error: %s", e)
        return []


def _telegram_reply(text: str):
    """Send a message to the Telegram chat (best-effort)."""
    token = os.environ.get("TRADE_TELEGRAM_TOKEN")
    chat_id = os.environ.get("TRADE_TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        from urllib.parse import urlencode
        params = urlencode({"chat_id": chat_id, "text": text,
                            "parse_mode": "HTML"})
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=params.encode("utf-8"),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        logger.warning("telegram reply failed: %s", e)


def process_command(rec: Dict) -> Dict:
    """Execute one command record via command_bus."""
    payload = rec.get("payload", {})
    command = payload.get("command", "")
    args = {k: v for k, v in payload.items() if k != "command"}
    logger.info("Processing %s (id=%s, args=%s)", command, rec.get("id"), args)
    try:
        from core.control.command_bus import execute
        result = execute(command, source="streamlit", **args)
        return result
    except Exception as e:
        logger.exception("command execution failed")
        return {"ok": False, "error": str(e)}


def main():
    logger.info("Command poller starting (poll every %ds)", POLL_INTERVAL)
    processed = _load_processed()
    while True:
        try:
            queue = _fetch_queue()
            new_cmds = [c for c in queue if c.get("id") not in processed]
            if new_cmds:
                logger.info("Found %d new command(s)", len(new_cmds))
                for cmd in new_cmds:
                    cmd_name = cmd.get("payload", {}).get("command", "?")
                    cmd_id = cmd.get("id", "?")
                    _telegram_reply(
                        f"⚙️ <b>Streamlit command:</b> <code>{cmd_name}</code>\n"
                        f"Processing..."
                    )
                    result = process_command(cmd)
                    processed.add(cmd_id)
                    _save_processed(processed)
                    ok = result.get("ok", False)
                    emoji = "✅" if ok else "❌"
                    summary = json.dumps(
                        {k: v for k, v in result.items()
                         if k in ("ok", "ticker", "qty", "fill_price",
                                   "succeeded", "failed", "error",
                                   "paused_until", "blocked_until",
                                   "message", "url")},
                        default=str,
                    )
                    _telegram_reply(
                        f"{emoji} <b>Command result:</b> <code>{cmd_name}</code>\n"
                        f"<pre>{summary[:300]}</pre>"
                    )
        except Exception as e:
            logger.error("poller iteration error: %s", e)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
