# Stock Scout - Production Deployment Guide

## Problem: App hangs on GitHub but works locally

### Root Causes
1. **Too many stocks to scan** - Default `UNIVERSE_LIMIT=50` causes long processing
2. **Network timeouts** - `fetch_beta_vs_benchmark` had no timeout protection
3. **API rate limits** - Multiple providers (Alpha Vantage, Finnhub) throttle heavily
4. **No caching** - Beta calculations repeated unnecessarily

### Solutions Applied

#### 1. Timeout Protection for Beta Calculation
Added timeout and caching to `fetch_beta_vs_benchmark`:
```python
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_beta_vs_benchmark(...):
    # 10-second timeout via signal.alarm
    # Falls back to np.nan on timeout
```

#### 2. Environment Variable Configuration
Now you can control performance via environment variables:

```bash
# For GitHub/Streamlit Cloud - BALANCED settings
UNIVERSE_LIMIT=40        # Scan 40 stocks (balanced: enough candidates, reasonable time)
LOOKBACK_DAYS=90
SMART_SCAN=true
TOPK_RECOMMEND=5

# For local development - COMPREHENSIVE settings
UNIVERSE_LIMIT=200       # More thorough scan
LOOKBACK_DAYS=120
SMART_SCAN=false
```

### Deployment Steps for GitHub/Streamlit Cloud

#### Option A: Using Streamlit Secrets (Recommended)
1. Go to your Streamlit Cloud app settings
2. Add to "Secrets" section:
```toml
UNIVERSE_LIMIT = 20
LOOKBACK_DAYS = 90
SMART_SCAN = "true"
TOPK_RECOMMEND = 5

ALPHA_VANTAGE_API_KEY = "your_key"
FINNHUB_API_KEY = "your_key"
# ... other API keys
```

#### Option B: Using .env file
1. Copy `.env.example` to `.env`
2. Set `UNIVERSE_LIMIT=20` for production
3. Add your API keys
4. **DO NOT commit .env** (already in .gitignore)

### Performance Comparison

| Setting | Local Dev | GitHub Cloud |
|---------|-----------|--------------|
| UNIVERSE_LIMIT | 50-200 | **10-20** |
| Processing Time | 60-120s | **20-40s** |
| API Calls | ~100-200 | **~30-50** |
| Timeout Risk | Low | High if >30 |

### Monitoring & Debugging

If app still hangs, check:
1. Streamlit logs for timeout messages
2. `st.session_state.av_calls` - Alpha Vantage call counter
3. Universe size in spinner message: "Building stock universe..."
4. Beta calculation spinner - should timeout after 10s per stock

### Local Testing of Production Settings

Test GitHub performance locally:
```bash
export UNIVERSE_LIMIT=20
export SMART_SCAN=true
streamlit run stock_scout.py
```

Should complete in 20-40 seconds instead of 60-120 seconds.

### Rollback Plan

If issues persist:
1. Set `UNIVERSE_LIMIT=10` (minimal)
2. Set `EXTERNAL_PRICE_VERIFY=false` in code
3. Disable beta filtering temporarily
4. Check API key quotas (especially Alpha Vantage)

### Future Improvements
- [ ] Add async/parallel beta calculations
- [ ] Implement progressive loading (show results as they come)
- [ ] Add user-facing timeout warnings
- [ ] Pre-cache common tickers
- [ ] Add health check endpoint
