"""Collect extra per-candidate features for FUTURE testing — decisions unchanged.

Goal (2026-09-29, agreed with Asaf): use more of the paid/free data providers
"smartly". More data only helps if it predicts returns, and we can only test that
if it was recorded AT SCAN TIME. scan_outcomes.jsonl has no analyst-revision /
earnings-surprise / insider / earnings-date history, so those cannot be
backtested today. This script starts recording them, append-only, into its OWN
file (data/outcomes/extra_features.jsonl), keyed by (ticker, scan_date), to be
joined to resolved outcomes later. NOTHING here feeds any gate or ranker.

Source: Finnhub free tier (verified accessible 2026-09-29): recommendation
trends, earnings surprises, earnings calendar, insider sentiment. Top-N by score
only, ~1.1s between calls (free limit 60/min). Every call is failure-tolerant;
a 429 stops the run early. Idempotent per (ticker, scan_date). Run AFTER the
trade step so it can never delay a buy.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("collect_extra_features")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "outcomes" / "extra_features.jsonl"
SCAN = ROOT / "data" / "scans" / "latest_scan.parquet"
TOP_N = int(os.getenv("EXTRA_FEATURES_TOP_N", "30"))
SLEEP = 1.1


class RateLimited(Exception):
    pass


def _get(path: str, key: str):
    url = f"https://finnhub.io/api/v1/{path}{'&' if '?' in path else '?'}token={key}"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited()
        logger.warning("finnhub %s -> HTTP %s", path.split("?")[0], e.code)
    except Exception as e:
        logger.warning("finnhub %s failed: %s", path.split("?")[0], e)
    return None


def _features(ticker: str, key: str, today: date) -> dict:
    sym = urllib.parse.quote(ticker)
    f: dict = {}
    rec = _get(f"stock/recommendation?symbol={sym}", key); time.sleep(SLEEP)
    if isinstance(rec, list) and rec:
        cur = rec[0]
        f["rec_strong_buy"], f["rec_buy"] = cur.get("strongBuy"), cur.get("buy")
        f["rec_hold"], f["rec_sell"], f["rec_strong_sell"] = cur.get("hold"), cur.get("sell"), cur.get("strongSell")
        if len(rec) > 1:
            def bull(x):  # net bullishness = (SB*2+B) - (S+SS*2)
                return (x.get("strongBuy", 0) * 2 + x.get("buy", 0)) - (x.get("sell", 0) + x.get("strongSell", 0) * 2)
            f["rec_bull_change_1m"] = bull(cur) - bull(rec[1])
    eps = _get(f"stock/earnings?symbol={sym}&limit=4", key); time.sleep(SLEEP)
    if isinstance(eps, list) and eps:
        s = [x.get("surprisePercent") for x in eps if x.get("surprisePercent") is not None]
        if s:
            f["eps_surprise_last"] = s[0]
            f["eps_surprise_avg4"] = round(sum(s) / len(s), 2)
    cal = _get(f"calendar/earnings?from={today}&to={today + timedelta(days=60)}&symbol={sym}", key); time.sleep(SLEEP)
    if isinstance(cal, dict) and cal.get("earningsCalendar"):
        d = cal["earningsCalendar"][0].get("date")
        if d:
            f["next_earnings_date"] = d
            f["days_to_earnings"] = (date.fromisoformat(d) - today).days
    ins = _get(f"stock/insider-sentiment?symbol={sym}&from={today - timedelta(days=120)}&to={today}", key); time.sleep(SLEEP)
    if isinstance(ins, dict) and ins.get("data"):
        rows = ins["data"][-3:]
        f["insider_mspr_3m"] = round(sum(r.get("mspr", 0) for r in rows) / len(rows), 2)
        f["insider_net_change_3m"] = sum(r.get("change", 0) for r in rows)
    return f


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        logger.warning("FINNHUB_API_KEY missing — nothing collected")
        return 0
    try:
        import pandas as pd
        df = pd.read_parquet(SCAN)
    except Exception as e:
        logger.warning("cannot read scan parquet: %s", e)
        return 0
    if df.empty or "Ticker" not in df.columns or "As_Of_Date" not in df.columns:
        return 0
    scan_date = str(pd.Timestamp(df["As_Of_Date"].iloc[0]).date())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["ticker"], r["scan_date"]))
            except Exception:
                continue
    top = df.nlargest(TOP_N, "Score")["Ticker"].tolist()
    todo = [t for t in top if (t, scan_date) not in done]
    logger.info("scan_date=%s top=%d already=%d todo=%d", scan_date, len(top), len(top) - len(todo), len(todo))
    today = datetime.now(timezone.utc).date()
    n = 0
    with OUT.open("a") as fh:
        for t in todo:
            try:
                feats = _features(t, key, today)
            except RateLimited:
                logger.warning("Finnhub 429 — stopping early after %d tickers", n)
                break
            if feats:
                fh.write(json.dumps({"ticker": t, "scan_date": scan_date,
                                     "collected_at": datetime.now(timezone.utc).isoformat(), **feats}) + "\n")
                fh.flush()
                n += 1
    logger.info("collected extra features for %d tickers -> %s", n, OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
