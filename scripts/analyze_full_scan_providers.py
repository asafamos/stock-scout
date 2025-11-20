#!/usr/bin/env python3
import csv
import ast
from collections import Counter

INPUT = 'full_v2_scan.csv'
OUT_BLOCKED = 'reports/blocked_tickers_diagnostics.csv'
OUT_MD = 'reports/full_scan_provider_summary.md'

rows = []
with open(INPUT, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Parse fund_sources_used_v2 which is stringified list like '["Fund_from_FMP", "Fund_from_Finnhub"]'
provider_counter = Counter()
price_provider_counter = Counter()
zero_fund_sources = []
blocked_rows = []

for r in rows:
    # fund sources
    raw = r.get('fund_sources_used_v2', '')
    try:
        parsed = ast.literal_eval(raw) if raw else []
    except Exception:
        # fall back to simple parsing
        parsed = []
    # normalize provider names (strip "Fund_from_")
    providers = [p.replace('Fund_from_', '') for p in parsed]
    if not providers:
        zero_fund_sources.append(r['Ticker'])
    for p in providers:
        provider_counter[p] += 1

    # price sources
    rawp = r.get('price_sources_used_v2', '')
    try:
        parsedp = ast.literal_eval(rawp) if rawp else []
    except Exception:
        parsedp = []
    price_providers = [pp.replace('Price_from_', '') for pp in parsedp]
    for pp in price_providers:
        price_provider_counter[pp] += 1

    if r.get('risk_gate_status_v2', '').lower() == 'blocked':
        blocked_rows.append(r)

# Write blocked diagnostics CSV
import os
os.makedirs('reports', exist_ok=True)
with open(OUT_BLOCKED, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Ticker','reward_risk_v2','fund_sources_used_v2','price_sources_used_v2','reliability_score_v2','risk_gate_reason_v2']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in blocked_rows:
        writer.writerow({k: r.get(k,'') for k in fieldnames})

# Write markdown summary
with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('# Full Scan Provider Summary\n\n')
    f.write(f'Total tickers analyzed: {len(rows)}\n\n')
    f.write('## Fund provider counts\n\n')
    for p, c in provider_counter.most_common():
        f.write(f'- {p}: {c}\n')
    f.write('\n')
    f.write('## Price provider counts\n\n')
    for p, c in price_provider_counter.most_common():
        f.write(f'- {p}: {c}\n')
    f.write('\n')
    f.write(f'- Tickers with zero fund sources: {len(zero_fund_sources)}\n')
    if zero_fund_sources:
        f.write('\nSample tickers with zero fund sources:\n\n')
        for t in zero_fund_sources[:20]:
            f.write(f'- {t}\n')
    f.write('\n')
    f.write(f'- Tickers blocked by risk gate: {len(blocked_rows)}\n')
    f.write('\n')
    f.write('## Notes and suggestions\n\n')
    f.write('- If many tickers have zero fund sources, check API keys or provider throttling.\n')
    f.write('- If most blocks are due to low `reward_risk_v2`, consider relaxing RR thresholds or ensuring ATR/52w are computed.\n')
    f.write('- Use `reports/blocked_tickers_diagnostics.csv` for per-ticker debugging.\n')

print('Wrote', OUT_MD)
print('Wrote', OUT_BLOCKED)
print('Summary: total rows=', len(rows), 'providers=', dict(provider_counter))
