# Canonical Output Schema (Live & Precomputed)

This doc summarizes key columns produced by the app across live pipeline and precomputed runs. Prefer these names when consuming outputs.

- Core Identification
  - **Ticker**: stock symbol
  - **Sector**: sector name
  - **Industry**: industry name

- Prices & Sources
  - **Price_Yahoo**: primary price from Yahoo (fallback if no multi-source mean)
  - **Price_Mean** / **Price_STD**: mean/std from external verification subset
  - **Historical_StdDev**: recent (≤30d) close std dev for verified subset
  - **Source_List**: provider badges string (e.g., "🟡Yahoo - 🔵Finnhub")
  - **Price_Sources_Count**: number of distinct price sources used
  - **Price_Sources**: combined badges (fundamental + price)

- Technicals
  - **RSI**, **ATR**, **ATR_Pct**, **Near52w**: core indicators
  - **Return_5d/10d/20d/60d/120d**: multi-period returns
  - **MA20/MA50/MA200** and **MA50_Slope**: moving averages and slope

- ML (20d v3)
  - **ML_20d_Prob**: live_v3 adjusted probability
  - **TechScore_20d_v2** / **FinalScore_20d**: 20d scoring components (if present)

- Fundamentals
  - **Fund_Coverage_Pct**: coverage percent
  - **Fundamental_Sources_Count** / **fund_reliability** / **Fundamental_Reliability**
  - Provider flags: **from_fmp_full**, **from_fmp**, **from_simfin**, **from_eodhd**, **from_alpha**, **from_finnhub**, **from_tiingo**

- Reliability
  - **Price_Reliability**: source-count + variance blend
  - **Reliability_Score**: 40% price + 60% fundamentals

- Scoring & Classification
  - **Score**: final score used for ranking (may equal `conviction_v2_final`)
  - **Risk_Level**: `core` / `speculative`
  - **Data_Quality**: `high` / `medium` / `low`
  - **Confidence_Level**: `high` / `medium` / `low`

- Allocation (Hebrew UI)
  - **Unit_Price**: price per share (mean rounded or Yahoo/Close fallback)
  - **סכום קנייה ($)**: buy amount in USD per ticker
  - **מניות לקנייה**: shares to buy (floor of amount / unit price)
  - **עודף ($)**: leftover dollars per ticker
  - **position_value**: unit price × shares (used for budget totals)

Notes
- Live vs Precomputed: precomputed runs may require light renaming (e.g., `Close` → `Price_Yahoo`). The app applies a small renaming map when rendering; consumers should prefer the canonical names above.
- Providers Overview: the overview table derives from `SourcesOverview.PROVIDERS` (roles + keys) and session-tracked usage. Avoid hardcoding provider lists elsewhere.
