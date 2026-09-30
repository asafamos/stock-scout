# Research scripts (offline, not part of the trading path)

* `build_pit_universe.py <workdir>` — point-in-time universe: Tiingo supported-tickers list (public zip, includes
  delisted names) + FMP daily price/volume and historical market cap per symbol. Needs `FMP_API_KEY`,
  `<workdir>/tiingo_tickers.zip`. Writes `univ_px.pkl`, `univ_mcap.pkl` (~5k symbols, ~50 min).
* `sleeve_backtest_full.py <workdir> <all|surv> <noearn|block|earn> <maxpos> <S3|RANDOM>` — portfolio-level
  backtest of the exact v2 sleeve (S3_v1 rank, atr_wide exit, 4% risk cap, $1 commission + 0.10% slippage,
  ±6% gap guard, top-N-by-market-cap universe per day via `BT_TOPN`). Needs `bars_vol.pkl` (SPY + survivors).

Result that matters (2026-09-30): on the survivorship-reduced universe S3 is NOT better than random
(top-2000: −0.4% vs −0.3% CAGR; top-1000: −5.6% vs +1.7%; top-500: −3.6% vs −5.3%; SPY +15.8%), and worse with the
live earnings rules (−12.2%). The earlier +17–22% on a 431-ticker survivor list was survivorship + single-position
path noise. Lesson: never trust a backtest whose universe is today's winners, and a single-position sequential
backtest is dominated by which few lottery tickets it happens to catch.
