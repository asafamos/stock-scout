"""Summarise FMP usage written by core/fmp_meter.py.

    python -m scripts.fmp_usage_report [--file data/state/fmp_usage.jsonl] [--days 7]

Per process label (script name) and per endpoint: calls, MB, 429s. The VPS file lives on the VPS
(/home/stockscout/stock-scout-2/data/state/fmp_usage.jsonl); GitHub runs report in their job summary.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/state/fmp_usage.jsonl")
    ap.add_argument("--days", type=float, default=7)
    a = ap.parse_args()
    cut = time.time() - a.days * 86400
    by_label = defaultdict(lambda: {"calls": 0, "bytes": 0, "e429": 0, "runs": 0})
    by_ep = defaultdict(lambda: {"calls": 0, "bytes": 0, "e429": 0})
    try:
        lines = open(a.file).read().splitlines()
    except OSError:
        raise SystemExit(f"no usage file at {a.file}")
    for l in lines:
        try:
            r = json.loads(l)
            ts = datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        except (ValueError, KeyError):
            continue
        if ts < cut:
            continue
        g = by_label[r["label"]]
        g["calls"] += r["calls"]; g["bytes"] += r["bytes"]; g["e429"] += r["e429"]; g["runs"] += 1
        for ep, v in r.get("by_endpoint", {}).items():
            e = by_ep[ep]; e["calls"] += v["calls"]; e["bytes"] += v["bytes"]; e["e429"] += v["e429"]
    tot_b = sum(v["bytes"] for v in by_label.values()) or 1
    print(f"FMP usage, last {a.days:g} days\n\nBy process:")
    for k, v in sorted(by_label.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  {k:<28} {v['calls']:>8} calls  {v['bytes'] / 1e6:8.1f} MB ({v['bytes'] / tot_b * 100:4.0f}%)  429s {v['e429']:>6}  flushes {v['runs']}")
    print("\nBy endpoint:")
    for k, v in sorted(by_ep.items(), key=lambda kv: -kv[1]["bytes"])[:20]:
        print(f"  {k:<42} {v['calls']:>8} calls  {v['bytes'] / 1e6:8.1f} MB  429s {v['e429']:>6}")


if __name__ == "__main__":
    main()
