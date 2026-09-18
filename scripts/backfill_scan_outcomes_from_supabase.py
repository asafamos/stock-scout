"""PHASE A: Backfill scan_outcomes.jsonl from Supabase.scan_recommendations.

Goal: expand our replay dataset from 2 months (42K records) to 6+ months
(150K+ records) by pulling historical Supabase data, matching to
portfolio_positions for realized returns, and merging into scan_outcomes.

Approach:
  1. Load current scan_outcomes → build dedupe set by (ticker, scan_date)
  2. Query Supabase.scan_recommendations for Mar-Jun 2026 (before our
     current coverage starts 2026-07-01)
  3. Parse full_row_json to extract score, ml, rr, fund, sector, etc.
  4. For each candidate: look up realized return in portfolio_positions
     joined by (ticker, close date within holding_days ± N)
  5. Only keep records that HAVE realized return (backtest requires it)
  6. Merge into scan_outcomes (append + dedupe) → save + backup

Safety: this script is IDEMPOTENT. If run twice, dedupe key ensures
no doubles. Original file backed up to .bak_pre_backfill_TIMESTAMP.
"""
import json, os, sys
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta
from pathlib import Path

url = os.getenv("SUPABASE_URL", "").strip()
key = os.getenv("SUPABASE_KEY", "").strip()
if not url or not key:
    with open('.env') as f:
        for line in f:
            if line.startswith('SUPABASE_URL='): url = line.split('=',1)[1].strip()
            elif line.startswith('SUPABASE_KEY='): key = line.split('=',1)[1].strip()

from supabase import create_client
sb = create_client(url, key)

ROOT = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
       else Path(__file__).resolve().parents[2]
OUTCOMES = ROOT / "data" / "outcomes" / "scan_outcomes.jsonl"

# 1. Load current scan_outcomes → build dedupe key
print("Step 1: Load current scan_outcomes for dedupe...")
existing = set()
current_count = 0
if OUTCOMES.exists():
    for line in OUTCOMES.read_text().splitlines():
        try:
            r = json.loads(line)
            tk = r.get('ticker'); d = (r.get('scan_date') or '')[:10]
            if tk and d:
                existing.add((tk, d))
                current_count += 1
        except: pass
print(f"  Current outcomes: {current_count} records, {len(existing)} unique (ticker,date)")

# 2. Load portfolio_positions for realized-return lookup
print()
print("Step 2: Load portfolio_positions for realized-return join...")
r = sb.table("portfolio_positions").select("ticker,entry_date,exit_date,realized_return_pct,exit_reason,sector,final_score,risk_class,holding_days,entry_price,stop_price,target_price,shares").execute()
positions = r.data or []
closed_positions = [p for p in positions if p.get("exit_date") and p.get("realized_return_pct") is not None]
print(f"  Total portfolio_positions: {len(positions)}, closed with realized_return: {len(closed_positions)}")

# Build lookup: for each ticker, list of (entry_date, position_data)
pos_by_ticker = defaultdict(list)
for p in closed_positions:
    pos_by_ticker[p["ticker"]].append(p)

# 3. Query scan_recommendations for backfill window (Mar-Jun 2026)
BACKFILL_START = "2026-03-01"
BACKFILL_END = "2026-07-01"  # exclusive; current scan_outcomes starts 2026-07-01
print()
print(f"Step 3: Query scan_recommendations for {BACKFILL_START} → {BACKFILL_END}...")

all_recs = []
offset = 0
BATCH = 1000
while True:
    resp = sb.table("scan_recommendations") \
        .select("id,ticker,scan_id,scan_timestamp,created_at,final_score,full_row_json") \
        .gte("created_at", BACKFILL_START) \
        .lt("created_at", BACKFILL_END) \
        .range(offset, offset + BATCH - 1) \
        .execute()
    if not resp.data: break
    all_recs.extend(resp.data)
    offset += BATCH
    if len(resp.data) < BATCH: break
    if offset > 200000: break
print(f"  Retrieved {len(all_recs)} scan_recommendations in window")

# 4. Parse, match to realized returns, build enriched records
print()
print("Step 4: Parse full_row_json + match to realized returns...")
enriched = []
skipped_dupe = 0
skipped_noreal = 0
skipped_noparse = 0

for rec in all_recs:
    ticker = rec.get("ticker") or ""
    scan_date = (rec.get("scan_timestamp") or rec.get("created_at") or "")[:10]
    if not ticker or not scan_date:
        skipped_noparse += 1
        continue
    if (ticker, scan_date) in existing:
        skipped_dupe += 1
        continue

    # Parse full_row_json (may be dict or JSON string)
    fr = rec.get("full_row_json")
    if isinstance(fr, str):
        try: fr = json.loads(fr)
        except: fr = {}
    if not isinstance(fr, dict):
        fr = {}

    # Extract features (case-tolerant)
    def _get(*names, default=None):
        for n in names:
            if n in fr and fr[n] is not None:
                return fr[n]
        return default

    score = _get("Score", "final_score", "FinalScore_20d") or rec.get("final_score") or 0
    ml_prob = _get("ML_Probability", "ml_prob", "ml_probability") or 0
    rr = _get("RR", "RR_Ratio", "rr_ratio", "RewardRisk") or 0
    fund = _get("Fundamental_Score", "fundamental_score", "fund_score") or 0
    sector = _get("Sector", "sector") or "Unknown"
    atr_pct = _get("ATR_Pct", "atr_pct") or 0
    market_regime = _get("Market_Regime", "market_regime", "mkt_market_regime") or ""
    reliability = _get("Reliability", "reliability", "reliability_score") or 0
    volume_surge = _get("Volume_Surge", "volume_surge") or 0
    risk_class = _get("Risk_Level", "risk_class", "Risk_Class") or ""

    # Find realized return from portfolio_positions (join on ticker + entry near scan_date)
    realized = None; hit_target = None; hit_stop = None; hold_days = None
    outcome = None; max_ret = None; min_ret = None; entry_price = None
    for p in pos_by_ticker.get(ticker, []):
        try:
            ed = p.get("entry_date", "")[:10]
            if not ed: continue
            delta = (date.fromisoformat(ed) - date.fromisoformat(scan_date)).days
            # Accept entries 0-3 days after scan
            if 0 <= delta <= 3:
                realized = float(p["realized_return_pct"])
                exit_reason = p.get("exit_reason", "")
                hit_target = (exit_reason == "target")
                hit_stop = (exit_reason == "stop")
                hold_days = p.get("holding_days")
                outcome = exit_reason or ("target_hit" if hit_target else ("stop_hit" if hit_stop else "expiry"))
                entry_price = p.get("entry_price")
                # max_return_pct, min_return_pct not available from portfolio_positions
                # Best proxy: realized = both (for trail-sim safety this is conservative)
                max_ret = realized
                min_ret = realized
                break
        except: pass

    if realized is None:
        skipped_noreal += 1
        continue

    row = {
        "ticker": ticker,
        "scan_date": scan_date,
        "recorded_at": rec.get("created_at") or rec.get("scan_timestamp") or "",
        "resolved": True,
        "resolved_at": datetime.now().isoformat(),
        "realized_return_pct": realized,
        "final_return_pct": realized,
        "max_return_pct": max_ret,
        "min_return_pct": min_ret,
        "score": float(score) if score else 0.0,
        "ml_prob": float(ml_prob) if ml_prob else 0.0,
        "rr": float(rr) if rr else 0.0,
        "fundamental_score": float(fund) if fund else 0.0,
        "atr_pct": float(atr_pct) if atr_pct else 0.0,
        "sector": sector,
        "mkt_market_regime": market_regime,
        "reliability": float(reliability) if reliability else 0.0,
        "volume_surge": float(volume_surge) if volume_surge else 0.0,
        "risk_class": risk_class,
        "hit_target": bool(hit_target),
        "hit_stop": bool(hit_stop),
        "holding_days": int(hold_days) if hold_days else 20,
        "outcome": outcome or "time_expired",
        "entry_price": float(entry_price) if entry_price else 0.0,
        "recovery_source": "supabase_backfill_phase_a",
    }
    enriched.append(row)

print(f"  Enriched (with realized): {len(enriched)}")
print(f"  Skipped:")
print(f"    duplicate of existing:   {skipped_dupe}")
print(f"    no realized return:      {skipped_noreal}")
print(f"    unparseable:             {skipped_noparse}")

if not enriched:
    print("Nothing to backfill.")
    sys.exit(0)

# 5. Backup + append
print()
print(f"Step 5: Backup + append {len(enriched)} records to {OUTCOMES}...")
backup_path = OUTCOMES.with_suffix(f".jsonl.bak_pre_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
if OUTCOMES.exists():
    backup_path.write_text(OUTCOMES.read_text())
    print(f"  Backup: {backup_path.name}")

with OUTCOMES.open("a") as f:
    for row in enriched:
        f.write(json.dumps(row, default=str) + "\n")
print(f"  ✅ Appended {len(enriched)} records to {OUTCOMES}")

# 6. Coverage report
print()
print("Step 6: Post-backfill coverage:")
new_count = 0; new_by_month = Counter()
for line in OUTCOMES.read_text().splitlines():
    try:
        r = json.loads(line)
        new_count += 1
        d = (r.get('scan_date') or '')[:10]
        if d: new_by_month[d[:7]] += 1
    except: pass
print(f"  Total records: {current_count} → {new_count} (+{new_count-current_count})")
for m in sorted(new_by_month.keys()):
    print(f"    {m}: {new_by_month[m]:>6d}")
