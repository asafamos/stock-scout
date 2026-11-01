# Night-scan price loss — diagnosis (2026-10-31; read from the GH logs of the 10-29 and 10-30 night runs)

## What the [HISTORY2] diagnostics show
| Run (start UTC) | Result | Evidence |
|---|---|---|
| 10-29 23:48 (id 40975587214) | integrity check FAILED (RS_63d NaN on 54 % of 84 rows) → last good scan kept | every history call after 00:00 UTC: all bars dated 10-29 (not a next-day bar), 251-row median (history complete), but **Close = NaN in the last bar for 48 → 189-195 of 200 tickers** |
| 10-30 23:28 (id 41144708621) | passed (128 rows) | same pattern from 00:00 UTC: Close NaN 119 → 190-196 of 200; passed only because most rows had been downloaded before midnight |
| 10-29 21:55, 10-30 21:38 (before midnight) | passed | no odd HISTORY2 lines at all (the line only warns when something is off) |

## Conclusions
1. The loss is NOT a missing-date / next-day-bar problem (all last bars carry the right date) and NOT a short history (rows normal). The final bar arrives with an empty Close while other fields exist, so `dropna(how="all")` keeps it and the downstream indicators read NaN.
2. It starts abruptly at ≈ 00:00 UTC (= 20:00 ET) and persists at least to 00:39 UTC. Calls made before midnight in the same run are clean.
3. It is not reproducible by replaying the same request with end = 23:50 / 00:05 / 12:00 / now (all returned a valid last Close, 0 NaN of 8 tickers). So it is a live data-source behaviour of the moment, not our request parameters. Whether it is "Yahoo rebuilds the daily bar after 20:00 ET" or "UTC date rollover" is not separable yet (see test below).
4. Why those runs exist at all: the 20:00 UTC cron ("end-of-day scan, no trade fires off this", auto_scan.yml) is serialized behind other scans by the concurrency group and by GitHub schedule delay, so it started at 21:38-23:48 UTC and ran across midnight.

## Cheap discriminating test (free, passive)
US clocks change on 2026-11-01: 20:00 ET becomes 01:00 UTC. If the NaN onset moves to ≈ 01:00 UTC on the first scan that crosses it, the cause is Yahoo's ET-based daily rebuild; if it stays at 00:00 UTC it is a UTC-date effect. The HISTORY2 line already records this; no code needed.

## Options (NOT applied — the scan workflow is shared; owner decides)
A. Make the end-of-day scan finish before 00:00 UTC (move/trim the 20:00 UTC cron, or drop it: the legacy channel is idle and CoreTrend does not use it) — removes the exposure, zero code risk.
B. In `fetch_history_bulk`, treat a NaN last Close as "not ready": re-request those tickers after a pause or fall back to another provider; if still NaN, drop the tail row and mark the scan stale. Fixes the cause for any hour but is a code change to the data path.
C. Leave as is: the integrity guard already prevents a bad scan from overwriting the last good one (it did on 10-28 and 10-29), at the cost of a missing night scan.

## Decision (2026-11-01, owner): option A applied
The 20:00 UTC cron was removed from `.github/workflows/auto_scan.yml` (the run that landed 22:55-23:53 UTC and crossed midnight). Remaining schedule: 13:30 and 17:30 UTC crons plus the VPS pipeline dispatches. The 00:00-UTC-vs-20:00-ET question can no longer be answered passively by a night scan; if it matters, re-test with a manual `workflow_dispatch` after 01:00 UTC.
