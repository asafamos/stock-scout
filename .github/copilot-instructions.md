## Quick orientation — Stock Scout (Asaf)

This repository is a single-file Streamlit app: `stock_scout.py`. The app
builds a US equities universe, downloads historical data (yfinance), computes
technical indicators and a combined score, optionally fetches fundamentals (Alpha
Vantage → Finnhub fallback), applies filters (earnings blackout, beta, sector
caps), verifies prices across external providers, allocates a small portfolio
and exposes a Hebrew/RTL UI with CSV export and quick charts.

Key files to scan for context
- `stock_scout.py` — everything lives here: config (`CONFIG`), data pipeline
  (`build_universe` → `fetch_history_bulk`), indicators (`rsi`, `atr`, `macd_line`,
  `adx`), scoring, externals (Alpha/Finnhub/Polygon/Tiingo), UI and CSV export.
- `requirements.txt` — runtime deps (Streamlit, yfinance, pandas, numpy, plotly).

Run / dev workflow
- Create an isolated env and install deps, then run the Streamlit app:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run stock_scout.py
```

Important environment variables
- ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY, POLYGON_API_KEY, TIINGO_API_KEY
- Values are loaded from `.env` (python-dotenv) or `st.secrets`.

Performance, rate limits and testing notes
- The app uses `st.cache_data` extensively (see many TTLs in `stock_scout.py`).
  Keep TTLs in mind when changing logic — cached results may hide changes.
- Alpha Vantage is rate-limited: the code uses `alpha_throttle()` and tracks
  `st.session_state['av_calls']`. When developing locally, either remove the
  key or set `CONFIG['SMART_SCAN']=False` and `CONFIG['UNIVERSE_LIMIT']=20` to
  avoid many external calls.
- For fast iterative development, disable expensive steps by editing `CONFIG`:
  e.g. set `EXTERNAL_PRICE_VERIFY=False`, lower `LOOKBACK_DAYS`, or use a
  trimmed `UNIVERSE_LIMIT`.

Concurrency and external API patterns
- `ThreadPoolExecutor` is used for `get_next_earnings_date` (`_earnings_batch`) and
  for external price verification in `_fetch_external_for`. Keep max workers small
  to avoid bursting API limits.
- HTTP helper `http_get_retry` centralizes retries/timeouts — prefer it when
  adding new external calls.

Project-specific conventions and pitfalls
- Single-file app: most changes are in `stock_scout.py`. Search for `CONFIG` to
  find tunable parameters and weights. Weighing logic is in `CONFIG['WEIGHTS']`.
- Column / UI localization: the app maps English columns to Hebrew via
  `hebrew_cols` and renders RTL text; tests or scripts that expect English
  columns should use the original DataFrame before renaming.
- Price validation merges multiple providers and writes `Price_Mean` / `Price_STD`.
  Add any new provider with a `get_<provider>_price` function and include it in
  `_fetch_external_for` and the provider checks at the top of the file.

Suggested first edits for contributors
- Add unit tests for pure functions: `rsi`, `atr`, `macd_line`, `fundamental_score`,
  `allocate_budget` and `_normalize_weights`. The current repo has no tests.
- Extract heavy logic into modules (e.g., `indicators.py`, `data.py`, `scoring.py`)
  if you plan to expand features — this will make testing and CI much easier.

Where to look when solving bugs or adding features
- Caching surprises: if a developer reports stale data, check `st.cache_data` TTLs
  and session state keys like `_alpha_last_call_ts` / `_alpha_ok`.
- Rate-limiting bugs: examine `alpha_throttle()` and `st.session_state['av_calls']`.
- UI localization issues: check `hebrew_cols`, `show_order` and the HTML blocks
  (search for `rtl` and `CARD_CSS`).

If you need more context
- Ask to expand the single-file app into a small module layout and I can
  refactor incrementally, add tests and CI. If you want the instructions in
  Hebrew or to include more examples, tell me which sections to expand.
