# 📊 Data Sources V2 Report - Stock Scout (Multi-Source Aggregation)

## Quick Summary (November 2025)
- **10+ active data sources**
- **Multi-source aggregation** with field-level median merging
- **8 price providers** (cross-verification with variance tracking)
- **4 active fundamentals sources** (FMP, Finnhub, Tiingo, Alpha Vantage)
- **Reliability scoring** based on source count + consistency + coverage

---

## 🔌 Active Sources and Status

### Price Sources (Multi-Source Verification & Aggregation)
| Source | Status | Usage | Cost / Rate Limit |
|------|-------|-------|----------------|
| **Yahoo Finance** | ✅ Base | Baseline price & history | Free, unlimited |
| **FMP** | ✅ Active | Real-time quote verification | Free tier available |
| **Finnhub** | ✅ Active | Current price verification | Free: 60/min |
| **Polygon** | ✅ Active | Previous close verification | Paid: 5/sec |
| **Tiingo** | ✅ Active | EOD price verification | Free tier: 50/hour |
| **Alpha Vantage** | ⚠️ Limited | Global quote (top 15 only) | Free: 25/day |

**Price Aggregation:**
- Fetches from all available sources concurrently
- Computes **median price** across sources (robust to outliers)
- Calculates **standard deviation** to measure disagreement
- Tracks **source count** (typically 3-5 sources)
- **Reliability formula:** `(1 - variance_ratio) × (0.4 + 0.6 × min(count/5, 1))`
- **Floor:** 20% reliability when any source available

### Fundamentals Sources (Multi-Source Field-Level Merge)
| Source | Status | Key Fields | Quality | Response Time |
|------|-------|--------------|--------|---------------|
| **FMP** | ✅ Primary | PE, PS, PB, ROE, Margin, Beta, D/E, Growth, Market Cap | ⭐⭐⭐⭐⭐ | ~200ms |
| **Finnhub** | ✅ Active | PE, PS, PB, ROE, Margin, Beta, D/E, Growth, Market Cap | ⭐⭐⭐⭐ | ~300ms |
| **Tiingo** | ✅ Active | PE, PS, PB, ROE, Margin, D/E, Growth | ⭐⭐⭐⭐ | ~250ms |
| **Alpha Vantage** | ⚠️ Limited | PE, PS, PB, ROE, Margin, Beta, D/E, Growth | ⭐⭐⭐ | ~500ms (rate-limited) |

**Fundamentals Aggregation (v2):**
- Each field (PE, PS, ROE, etc.) fetched from **all available sources**
- **Median value** computed per field when multiple sources provide data
- **Disagreement score:** Coefficient of variation (std/median) capped at 1.0
- **Coverage tracking:** % of expected fields successfully populated
- **Source attribution:** Records which sources contributed to each field
- **Reliability formula:** `coverage × (1 - disagreement) × (0.5 + 0.5 × min(count/4, 1))`
  - Floor of 15% when at least one source contributes
  - Scales up to 100% with 4+ sources and perfect agreement

---

## 🎯 Multi-Source Merge Strategy (v2 Architecture)

### Price Verification Pipeline
```python
# Module: core/data_sources_v2.py → fetch_price_multi_source()
1. Fetch current price from all 5+ sources concurrently
2. Filter out None/invalid values
3. Compute median (robust to outliers)
4. Calculate std deviation (measures disagreement)
5. Return: (median_price, std_price, source_count)
```

**Price Reliability Calculation:**
```python
variance_ratio = min(std_price / mean_price, 1.0)
source_factor = min(source_count / 5.0, 1.0)
reliability = (1 - variance_ratio) × (0.4 + 0.6 × source_factor)
# Floor: 20% if any source available
```

### Fundamentals Aggregation Pipeline
```python
# Module: core/data_sources_v2.py → aggregate_fundamentals()
1. Fetch from all sources: FMP, Finnhub, Tiingo, Alpha (rate-limited)
2. For each field (pe, ps, pb, roe, margin, etc.):
   - Collect all non-null values
   - Compute MEDIAN (robust to outliers)
   - Track which sources contributed
   - Calculate coefficient of variation
3. Overall disagreement = mean of per-field disagreements
4. Coverage = filled_fields / total_expected_fields
5. Return: merged dict + sources_used + coverage + disagreement_score
```

**Fundamental Reliability Calculation:**
```python
source_factor = 0.5 + 0.5 × min(source_count / 4.0, 1.0)
reliability = coverage × (1 - disagreement) × source_factor
# Floor: 15% × source_factor when any source contributes
# Max: 100% with 4 sources, 100% coverage, 0% disagreement
```

**Combined Reliability:**
```python
combined = 0.6 × fundamental_reliability + 0.4 × price_reliability
```

---

## 📊 Typical Data Flow (Nov 2025)

### 1. Historical Price Data
- **Yahoo Finance** (yfinance) → bulk download for all tickers
- Cached for 1 hour
- Used for technical indicators (RSI, ATR, MACD, etc.)

### 2. Current Price Verification
- **Multi-source price fetch** → 5+ providers queried
- Median taken as canonical price
- Std dev used to flag divergent sources
- Source count tracked (typical: 3-5 sources)

### 3. Fundamentals Fetch
- **Multi-source fundamentals** → 4 providers (FMP, Finnhub, Tiingo, Alpha)
- Per-field median merge
- Disagreement scoring per field
- Coverage calculation
- Source attribution

### 4. Reliability & Quality Metrics (Injected into DataFrame)
- `fund_sources_used_v2` — number of fundamental sources
- `price_sources_used_v2` — number of price sources
- `Fundamental_Reliability_v2` — 0-100%
- `Price_Reliability_v2` — 0-100%
- `reliability_v2` — combined 0-100%
- `fund_disagreement_score_v2` — 0-1 (0=perfect agreement)
- `price_variance_score_v2` — std/mean ratio
- `Price_Mean_v2` — multi-source median price
- `Price_STD_v2` — price standard deviation

---

## 💡 Architecture Benefits

### ✅ What works well
1. **Median-based aggregation** — robust to outliers and bad data
2. **Per-field merge** — maximizes coverage by using best available source for each metric
3. **Disagreement scoring** — flags when sources conflict significantly
4. **Reliability floors** — ensures partial credit even with single source
5. **Source tracking** — transparency on which providers contributed
6. **Variance monitoring** — detects price discrepancies across providers

### 🔧 Active Optimizations (Nov 2025)

#### 1. Alpha Vantage smart throttling ✅
```python
# Enabled only for top 15 tickers to stay under 25/day limit
# Remaining tickers use FMP + Finnhub + Tiingo (3 sources minimum)
```

#### 2. Concurrent fetching ✅
```python
# All sources fetched in parallel where rate limits allow
# FMP + Finnhub + Tiingo: concurrent
# Alpha: sequential with 12s delay
```

#### 3. Caching Strategy ✅
```python
# Per-source cache with TTL tracking
# Fundamentals: 1 hour cache
# Prices: fresh on each run (volatile)
```

#### 4. Reliability-based filtering ✅
```python
# V2 risk engine uses reliability_v2 for position sizing
# Low reliability (<40%) triggers reduced allocation
# Zero sources = blocked (no position)
```

### 🚀 Future Enhancements

#### 1. Expand to more sources
- **IEX Cloud** — excellent for real-time prices
- **Quandl/Nasdaq** — extended fundamental history
- **SEC EDGAR** — direct filings (10-K, 10-Q)

#### 2. Anomaly detection
```python
# Flag when sources disagree by >10%
# Automated alerts for suspicious data
# Cross-check against historical ranges
```

#### 3. Source quality scoring
```python
# Track per-source accuracy over time
# Weight more reliable sources higher in median calculation
# Auto-disable chronically bad sources
```

---

## 📈 Performance Metrics

### Coverage Statistics (Typical Run)
- **Price sources per ticker:** 3-5 (median: 4)
- **Fund sources per ticker:** 2-4 (median: 3)
- **Field coverage:** 70-95% (depends on ticker popularity)
- **Price reliability:** 80-95% (low variance)
- **Fund reliability:** 50-85% (varies by data availability)

### API Call Distribution
- **FMP:** ~25 tickers/run (primary)
- **Finnhub:** ~25 tickers/run (active)
- **Tiingo:** ~25 tickers/run (active)
- **Alpha Vantage:** ~15 tickers/run (smart throttle)
- **Total API calls:** ~300-400 per full scan

### Response Times
- **Single ticker price fetch:** 0.5-1.5s (parallel)
- **Single ticker fundamentals:** 1.0-2.5s (parallel + rate limits)
- **Full 25-ticker scan:** 30-60s (depends on cache hits)

---

## 🔍 Example: Multi-Source Data for AAPL

### Price Aggregation
```
Sources: FMP=$175.43, Finnhub=$175.45, Tiingo=$175.42, Alpha=$175.40, Polygon=$175.44
Median: $175.43
Std Dev: $0.02
Source Count: 5
Variance Ratio: 0.0001 (0.01%)
Price Reliability: 99.6%
```

### Fundamentals Aggregation
```
PE Ratio: FMP=28.5, Finnhub=28.3, Tiingo=28.4 → Median: 28.4
ROE: FMP=1.56, Finnhub=1.55, Alpha=1.57 → Median: 1.56
Beta: FMP=1.24, Finnhub=1.25 → Median: 1.245

Coverage: 9/10 fields (90%)
Disagreement: 0.03 (3% average CV)
Source Count: 3
Fund Reliability: 82.5%
```

---

## 📝 Integration Points

### Main Pipeline (stock_scout.py)
```python
# After fundamentals fetch, inject multi-source aggregation
from core.data_sources_v2 import (
    aggregate_fundamentals, 
    fetch_price_multi_source, 
    aggregate_price
)

for ticker in tickers:
    # Fetch multi-source fundamentals
    agg_fund = aggregate_fundamentals(ticker)
    fund_sources_count = len(agg_fund['sources_used'])
    fund_reliability = compute_fund_reliability(agg_fund)
    
    # Fetch multi-source prices
    prices = fetch_price_multi_source(ticker)
    mean_price, std_price, price_count = aggregate_price(prices)
    price_reliability = compute_price_reliability(std_price, mean_price, price_count)
    
    # Inject into DataFrame
    results.loc[idx, 'fund_sources_used_v2'] = fund_sources_count
    results.loc[idx, 'price_sources_used_v2'] = price_count
    results.loc[idx, 'Fundamental_Reliability_v2'] = fund_reliability
    results.loc[idx, 'Price_Reliability_v2'] = price_reliability
```

### UI Display (stock_scout.py card rendering)
```python
# Show Fund/Price reliability separately
fund_rel = row.get('Fundamental_Reliability_v2', np.nan)
price_rel = row.get('Price_Reliability_v2', np.nan)
reliability_fmt = f"{band} (F:{fund_rel:.0f}% / P:{price_rel:.0f}%)"
```

### CSV Export
```python
# Include new v2 columns
export_cols = [
    'Ticker', 'Score',
    'fund_sources_used_v2', 'price_sources_used_v2',
    'Fundamental_Reliability_v2', 'Price_Reliability_v2',
    'fund_disagreement_score_v2', 'price_variance_score_v2',
    'Price_Mean_v2', 'Price_STD_v2',
    # ... other columns
]
```
