# Full Scan Provider Summary

Total tickers analyzed: 500

## Fund provider counts

- FMP: 500
- Finnhub: 500

## Price provider counts


- Tickers with zero fund sources: 0

- Tickers blocked by risk gate: 53

## Notes and suggestions

- If many tickers have zero fund sources, check API keys or provider throttling.
- If most blocks are due to low `reward_risk_v2`, consider relaxing RR thresholds or ensuring ATR/52w are computed.
- Use `reports/blocked_tickers_diagnostics.csv` for per-ticker debugging.
