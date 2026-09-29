"""A function-local import of a name that is ALSO a module-level global makes the name local
for the whole function: any use on a path that did not execute the import raises
UnboundLocalError. Latent monitor bug found 2026-09-29 (`from datetime import datetime`
inside one branch of run_check). This guard fails on any such pattern in money-path modules."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/monitor_positions.py", "scripts/run_auto_trade.py",
    "core/trading/order_manager.py", "core/trading/risk_manager.py",
    "core/trading/ibkr_client.py", "core/trading/position_tracker.py",
    "core/trading/policy.py", "core/trading/ledger.py", "core/trading/notifications.py",
]


def _module_globals(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _scan(path):
    tree = ast.parse((ROOT / path).read_text())
    g = _module_globals(tree)
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local_imports = {}
        for n in ast.walk(fn):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    nm = (a.asname or a.name).split(".")[0]
                    local_imports.setdefault(nm, n.lineno)
        for nm, first in local_imports.items():
            if nm not in g:
                continue
            # a load of that name at an earlier line than the import, in this same function
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and n.id == nm and isinstance(n.ctx, ast.Load) and n.lineno < first:
                    bad.append(f"{path}:{fn.name}: '{nm}' used at line {n.lineno} before local import at {first}")
                    break
            # or a load OUTSIDE the block that holds the import (a path that may skip it)
            parent = None
            for n in ast.walk(fn):
                for field in ("body", "orelse", "finalbody", "handlers"):
                    blk = getattr(n, field, None)
                    if isinstance(blk, list) and any(
                        isinstance(x, (ast.Import, ast.ImportFrom)) and x.lineno == first for x in blk
                    ):
                        parent = blk
            if parent is None or parent is fn.body:
                continue
            lo = min(x.lineno for x in parent)
            hi = max(getattr(x, "end_lineno", x.lineno) for x in parent)
            for n in ast.walk(fn):
                if (isinstance(n, ast.Name) and n.id == nm and isinstance(n.ctx, ast.Load)
                        and not (lo <= n.lineno <= hi)):
                    bad.append(f"{path}:{fn.name}: '{nm}' loaded at line {n.lineno} outside the block "
                               f"({lo}-{hi}) that imports it locally at {first}")
                    break
    return bad


def test_no_shadowing_local_imports():
    bad = []
    for f in FILES:
        if (ROOT / f).exists():
            bad += _scan(f)
    assert not bad, "\n".join(bad)
