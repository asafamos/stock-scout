"""Static guard: no undefined names in the money-path modules (2026-09-29).

A NameError hid inside an error-handling branch of RiskManager.can_open_position (`row` instead of
`ticker`) — exactly the branch meant to block a bad trade, so it would have crashed instead. pyflakes
catches this class of bug without running the code.
"""
import subprocess
import sys

import pytest

pyflakes = pytest.importorskip("pyflakes")

MODULES = [
    "core/trading/risk_manager.py", "core/trading/order_manager.py", "core/trading/ibkr_client.py",
    "core/trading/position_tracker.py", "core/trading/policy.py", "core/trading/ledger.py",
    "core/trading/live_quote.py", "core/trading/notifications.py",
    "scripts/monitor_positions.py", "scripts/run_auto_trade.py", "scripts/reconcile_audit.py",
]


def test_money_paths_have_no_undefined_names():
    out = subprocess.run([sys.executable, "-m", "pyflakes", *MODULES], capture_output=True, text=True)
    bad = [l for l in (out.stdout + out.stderr).splitlines() if "undefined name" in l]
    assert not bad, "undefined names in money paths:\n" + "\n".join(bad)
