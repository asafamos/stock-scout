"""Scan quality monitor (plan docs/plan_oct2026_costs_and_scan.md, P3/monitoring). Reads data/scans/latest_scan.parquet and records health
indicators that were silently drifting: ML-gate strictness (scale moves with every retrain), synthetic VIX, saturated/constant columns.

  python scripts/scan_quality_report.py [--no-telegram] [--no-write]
Appends one JSON line per run to data/scan_quality/history.jsonl; sends a Telegram message ONLY when the set of RED flags changes."""
import argparse, json, os, subprocess, sys, urllib.parse, urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HIST = os.path.join(ROOT, "data", "scan_quality", "history.jsonl")
ML_MIN, ML_MAX = float(os.getenv("TRADE_MIN_ML_PROB", "0.40")), float(os.getenv("TRADE_MAX_ML_PROB", "0.60"))


def evaluate(df: pd.DataFrame, vix_real=None, now=None, scan_date=None) -> dict:
    """Pure function: returns {metrics..., flags:[(level, code, text)]} for a scan DataFrame."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(dtype=float)
    m, flags = {}, []
    m["rows"] = int(len(df))
    if len(df) < 20: flags.append(("RED", "FEW_ROWS", f"only {len(df)} rows in the latest scan"))
    ml = num("ML_20d_Prob")
    if len(ml.dropna()):
        pr = float(ml.between(ML_MIN, ML_MAX).mean()); m["ml_gate_pass_rate"] = round(pr, 4); m["ml_median"] = round(float(ml.median()), 3)
        if pr < 0.03 or pr > 0.60:
            flags.append(("RED", "ML_GATE_STRICTNESS", f"ML gate [{ML_MIN},{ML_MAX}] passes {pr*100:.1f}% (median ML {ml.median():.2f}); model scale drifted — expected ≈15–45%"))
        atr = num("ATR_Pct")
        if len(atr.dropna()) > 30: m["spearman_ml_atr"] = round(float(pd.concat([ml, atr], axis=1).corr(method="spearman").iloc[0, 1]), 2)
    vix = num("VIX_Value"); src = str(df["VIX_Source"].iloc[0]) if "VIX_Source" in df.columns and len(df) else "unknown"
    m["vix_value"] = None if not len(vix.dropna()) else round(float(vix.dropna().iloc[0]), 2); m["vix_source"] = src
    if "SYNTH" in src.upper() or "PROXY" in src.upper():
        flags.append(("RED", "VIX_PROXY", "VIX comes from the SPY realised-vol proxy, not a real VIX"))
    if vix_real is not None and m["vix_value"] is not None:
        m["vix_real"] = round(float(vix_real), 2)
        if abs(m["vix_value"] - float(vix_real)) > 3:
            flags.append(("RED", "VIX_MISMATCH", f"scan VIX {m['vix_value']} vs real {vix_real:.1f}"))
    rel = num("Reliability_Score")
    if len(rel.dropna()) and rel.nunique() <= 2: m["reliability_unique"] = int(rel.nunique()); flags.append(("INFO", "RELIABILITY_SATURATED", f"Reliability_Score has {rel.nunique()} distinct values (no information)"))
    fund = num("Fundamental_Score")
    if len(fund) and fund.isna().mean() > 0.30: flags.append(("YELLOW", "FUNDAMENTAL_MISSING", f"{fund.isna().mean()*100:.0f}% of Fundamental_Score missing"))
    if scan_date is not None and now is not None:
        age_h = (now - scan_date).total_seconds() / 3600; m["scan_age_hours"] = round(age_h, 1)
        if now.weekday() < 5 and age_h > 72: flags.append(("RED", "STALE_SCAN", f"latest scan is {age_h:.0f}h old"))
    m["flags"] = flags
    return m


def _vix_real():
    key = os.getenv("FMP_API_KEY")
    if not key: return None
    try:
        with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/quote?symbol=%5EVIX&apikey={key}", timeout=20) as r: return float(json.loads(r.read())[0]["price"])
    except Exception: return None


def _telegram(text):
    tok, chat = os.getenv("TRADE_TELEGRAM_TOKEN") or os.getenv("TELEGRAM_TOKEN"), os.getenv("TRADE_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat: return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML"}).encode()
    try: urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=data), timeout=20)
    except Exception as e: print("telegram failed:", e, file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-telegram", action="store_true"); ap.add_argument("--no-write", action="store_true"); a = ap.parse_args()
    path = os.path.join(ROOT, "data", "scans", "latest_scan.parquet"); df = pd.read_parquet(path)
    cd = subprocess.run(["git", "log", "-1", "--format=%cI", "--", "data/scans/latest_scan.parquet"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sd = datetime.fromisoformat(cd).astimezone(timezone.utc) if cd else None
    res = evaluate(df, _vix_real(), datetime.now(timezone.utc), sd)
    red = sorted(c for l, c, _ in res["flags"] if l == "RED")
    print(json.dumps({k: v for k, v in res.items() if k != "flags"}), "\nFLAGS:", [(l, c) for l, c, _ in res["flags"]])
    prev_red = []
    if os.path.exists(HIST):
        try: prev_red = json.loads(open(HIST).read().strip().splitlines()[-1]).get("red", [])
        except Exception: pass
    if not a.no_write:
        os.makedirs(os.path.dirname(HIST), exist_ok=True)
        with open(HIST, "a") as f: f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **{k: v for k, v in res.items() if k != "flags"}, "red": red, "flags": [c for _, c, _ in res["flags"]]}) + "\n")
    if red != prev_red and not a.no_telegram:
        body = "\n".join(f"• <b>{c}</b>: {t}" for l, c, t in res["flags"] if l == "RED") or "all RED flags cleared ✅"
        _telegram("🔬 <b>Scan quality</b>\n" + body)
