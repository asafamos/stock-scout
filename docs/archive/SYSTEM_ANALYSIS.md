# Stock Scout - System Analysis & Optimization Recommendations

## Current Status: Only 1 Recommendation Passing

### 📊 Current Filter Configuration

#### Core Stock Filters (Very Strict)
```python
MIN_QUALITY_SCORE_CORE = 27.0      # Fundamental quality (out of 100)
MAX_OVEREXTENSION_CORE = 0.10      # Max 10% above MA_Long
MAX_ATR_PRICE_CORE = 0.08          # Max 8% volatility
RSI_MIN_CORE = 45                  # Minimum RSI
RSI_MAX_CORE = 70                  # Maximum RSI
MIN_RR_CORE = 1.5                  # Minimum Reward/Risk ratio
```

#### Universal Filters (Applied to All)
```python
MIN_PRICE = 3.0
MIN_AVG_VOLUME = 500,000
MIN_DOLLAR_VOLUME = 5,000,000
MIN_MARKET_CAP = 100M
BETA_MAX_ALLOWED = 2.0
EARNINGS_BLACKOUT_DAYS = 7
SECTOR_CAP_MAX = 3 stocks per sector
MAX_SECTOR_ALLOCATION = 35%
```

---

## 🔍 Key Issues Identified

### 1. **Over-Restrictive Fundamental Requirements**
**Problem:**
- `MIN_QUALITY_SCORE_CORE = 27` might seem low but the scoring system is harsh
- Quality Score = 35% ROE/ROIC/GM + 30% Growth + 25% Valuation + 10% Leverage
- Many good growth stocks score <30 due to high P/E ratios (valuation penalty)
- Value stocks score <30 due to low growth rates

**Impact:** Eliminates ~70-80% of otherwise solid candidates

**Data Utilization:**
- ✅ Using: ROE, ROIC, Gross Margin, Revenue Growth, EPS Growth, P/E, P/S, D/E
- ❌ Missing: Free Cash Flow, Operating Margins, Asset Turnover, Dividend Yield
- ❌ Not weighted: Market conditions, sector-relative metrics

**Recommendation:**
```python
# Option A: Lower threshold
MIN_QUALITY_SCORE_CORE = 20.0  # Allow more candidates

# Option B: Separate growth vs value scoring
MIN_QUALITY_SCORE_GROWTH = 22.0  # Growth stocks (allow lower valuation scores)
MIN_QUALITY_SCORE_VALUE = 25.0   # Value stocks (require higher quality)

# Option C: Make fundamentals advisory, not disqualifying
FUNDAMENTAL_WEIGHT_IN_FINAL_SCORE = 0.20  # Use as boost, not filter
```

---

### 2. **Technical Filters Too Conservative**
**Problem:**
- `MAX_OVEREXTENSION_CORE = 0.10` eliminates stocks in strong uptrends
- `RSI_MAX_CORE = 70` eliminates momentum plays
- `MAX_ATR_PRICE_CORE = 0.08` eliminates growth/volatile stocks

**Impact:** Misses breakouts and trending stocks

**Data Utilization:**
- ✅ Using: RSI, ATR, Moving Averages, Overextension
- ✅ Using: MACD, ADX, Volume, Pullback detection
- ❌ Missing: Context - is market trending or ranging?
- ❌ Missing: Relative strength vs peers
- ❌ Missing: Chart patterns (consolidation, breakout)

**Recommendation:**
```python
# Make technical filters adaptive to market regime
if market_trending_up:
    MAX_OVEREXTENSION_CORE = 0.15  # Allow more extension in bull market
    RSI_MAX_CORE = 75
else:
    MAX_OVEREXTENSION_CORE = 0.08  # Be conservative in choppy market
    RSI_MAX_CORE = 70

# Add momentum exception
if RS_63d > 0.20 and Volume_Surge > 1.5:  # Strong relative strength + volume
    allow_higher_RSI = True  # Momentum play
```

---

### 3. **Scoring Weights Not Optimized**
**Current Technical Weights:**
```python
"ma": 0.22,              # Moving average positioning
"mom": 0.30,             # Momentum (1/3/6 month)
"rsi": 0.12,             # RSI range scoring
"near_high_bell": 0.10,  # Near 52-week high
"vol": 0.08,             # Volume consistency
"overext": 0.08,         # Overextension penalty
"pullback": 0.05,        # Pullback detection
"risk_reward": 0.03,     # Risk/reward ratio
"macd": 0.01,            # MACD signal
"adx": 0.01,             # ADX trend strength
```

**Issues:**
- MACD & ADX weighted too low (1% each) - strong trend indicators
- Overextension weighted same as volume (8%) - should differ by market regime
- Risk/Reward only 3% - should be higher for risk management
- Momentum 30% - might be too high in sideways market

**Data Utilization:**
- ✅ Using: All major technical indicators
- ❌ Missing: Relative volume (vs sector)
- ❌ Missing: Price action patterns
- ❌ Missing: Multi-timeframe analysis

**Recommendation:**
```python
# Regime-adaptive weights
if market_trending:
    weights = {
        "mom": 0.28,          # Keep momentum high
        "ma": 0.20,
        "macd": 0.08,         # Increase trend following
        "adx": 0.06,          # Increase trend strength
        "rsi": 0.10,
        "near_high_bell": 0.12,
        "vol": 0.07,
        "overext": 0.04,      # Lower penalty in trend
        "pullback": 0.03,
        "risk_reward": 0.02,
    }
else:  # Ranging market
    weights = {
        "ma": 0.25,           # Increase support/resistance
        "rsi": 0.20,          # Increase mean reversion
        "mom": 0.18,          # Lower momentum
        "pullback": 0.12,     # Higher pullback importance
        "risk_reward": 0.08,  # Higher risk management
        "vol": 0.07,
        "near_high_bell": 0.05,
        "overext": 0.03,
        "macd": 0.01,
        "adx": 0.01,
    }
```

---

### 4. **Sector/Industry Context Missing**
**Problem:**
- No sector-relative scoring (Tech vs Utilities have different "good" metrics)
- Sector cap (3 stocks per sector) is rigid
- No consideration of sector rotation/momentum

**Data Utilization:**
- ✅ Using: Sector limits for diversification
- ❌ Missing: Sector relative strength
- ❌ Missing: Industry-specific fundamentals
- ❌ Missing: Sector rotation signals

**Recommendation:**
```python
# Add sector-relative scoring
def normalize_by_sector(df):
    for sector in df['Sector'].unique():
        sector_mask = df['Sector'] == sector
        df.loc[sector_mask, 'Quality_Score_Sector_Relative'] = (
            (df.loc[sector_mask, 'Quality_Score'] - 
             df.loc[sector_mask, 'Quality_Score'].mean()) / 
            df.loc[sector_mask, 'Quality_Score'].std()
        )
    return df

# Dynamic sector allocation
if sector_has_momentum('Technology'):
    max_tech_stocks = 4  # Allow more in hot sector
else:
    max_tech_stocks = 2  # Reduce in weak sector
```

---

### 5. **Advanced Metrics Under-Utilized**
**Available but not heavily weighted:**
- Relative Strength (RS_63d, RS_126d)
- Volume Surge & Price-Volume Correlation
- Momentum Consistency & Acceleration
- MA Alignment Score
- Support/Resistance Distance
- Consolidation/Squeeze Detection

**Current Issues:**
- These are computed but not in main scoring formula
- Only used for classification (core vs speculative)
- Could significantly improve signal quality

**Recommendation:**
```python
# Add "Advanced Score" component
advanced_score = (
    0.30 * normalize(RS_63d) +              # Relative strength
    0.25 * normalize(Momentum_Consistency) + # Consistency
    0.20 * normalize(MA_Alignment_Score) +   # Trend alignment
    0.15 * normalize(Volume_Surge) +         # Institutional interest
    0.10 * normalize(1 - Distance_to_Resistance)  # Breakout proximity
)

# Integrate into final score
final_score = (
    0.50 * technical_score +
    0.25 * fundamental_score +
    0.25 * advanced_score  # New component
)
```

---

## 📈 Data Source Utilization Assessment

### Currently Used:
1. **Yahoo Finance (yfinance)** - Historical prices, volume ✅
2. **FMP** - Fundamentals (if Alpha unavailable) ✅
3. **Finnhub** - Fundamentals fallback + price verification ✅
4. **Polygon, Marketstack, NasdaqDL, EODHD** - Price verification ✅

### Under-Utilized:
1. **SimFin** - Available but not weighted in scoring
2. **Multiple price sources** - Used only for verification, not for consensus signal
3. **Fundamentals breadth** - Getting many fields but not all used

### Missing Opportunities:
1. **Earnings Calendar** - Fetch next earnings date (currently has placeholder)
2. **News Sentiment** - Not integrated (Finnhub has news API)
3. **Institutional Holdings** - Available in some sources
4. **Short Interest** - Could signal squeeze potential
5. **Insider Trading** - Strong signal for conviction

---

## 🎯 Recommended Action Plan

### Phase 1: Quick Wins (Immediate)
1. **Relax Core Filters** to allow 5-10 recommendations:
   ```python
   MIN_QUALITY_SCORE_CORE = 22.0  # Down from 27
   MAX_OVEREXTENSION_CORE = 0.12  # Up from 0.10
   RSI_MAX_CORE = 75              # Up from 70
   ```

2. **Increase Advanced Metrics Weight**:
   ```python
   # Add RS_63d and Momentum_Consistency to main score
   technical_score += 0.10 * normalize(RS_63d)
   ```

3. **Log Filter Elimination Stats**:
   ```python
   # Show how many stocks fail at each filter stage
   print(f"Failed Quality Score: {quality_fails}")
   print(f"Failed RSI: {rsi_fails}")
   print(f"Failed Overextension: {overext_fails}")
   ```

### Phase 2: Medium-Term (1-2 weeks)
1. **Implement Market Regime Detection**:
   - Detect if SPY is trending up/down/sideways
   - Adjust filters and weights accordingly

2. **Add Sector-Relative Scoring**:
   - Normalize fundamentals within sector
   - Allow different thresholds per sector

3. **Enhance Fundamental Scoring**:
   - Add Free Cash Flow, Operating Margin
   - Create separate growth vs value scores

### Phase 3: Long-Term (1 month+)
1. **Add News/Sentiment Integration**
2. **Machine Learning Score Calibration**:
   - Backtest current weights
   - Optimize for historical returns
3. **Multi-Timeframe Analysis**:
   - Daily + Weekly signals combined
4. **Portfolio-Level Optimization**:
   - Correlation analysis
   - Sharpe ratio maximization

---

## 💡 Specific Recommendations for Current System

### Immediate Changes (Copy-Paste Ready):

```python
# In stock_scout.py CONFIG
CONFIG = dict(
    # Relaxed thresholds - aim for 5-10 recommendations
    MIN_QUALITY_SCORE_CORE=22.0,      # Was 27.0
    MAX_OVEREXTENSION_CORE=0.12,      # Was 0.10
    MAX_ATR_PRICE_CORE=0.09,          # Was 0.08
    RSI_MIN_CORE=40,                  # Was 45
    RSI_MAX_CORE=75,                  # Was 70
    MIN_RR_CORE=1.3,                  # Was 1.5
    
    # Add momentum exception
    ALLOW_HIGH_RSI_IF_MOMENTUM=True,  # NEW
    MOMENTUM_RSI_THRESHOLD=0.20,      # If RS_63d > 20%, allow RSI up to 80
)

# In scoring weights
WEIGHTS = {
    "ma": 0.20,              # Reduced from 0.22
    "mom": 0.25,             # Reduced from 0.30
    "rsi": 0.12,             # Same
    "near_high_bell": 0.10,  # Same
    "vol": 0.08,             # Same
    "overext": 0.06,         # Reduced from 0.08
    "pullback": 0.05,        # Same
    "risk_reward": 0.05,     # Increased from 0.03
    "macd": 0.03,            # Increased from 0.01
    "adx": 0.03,             # Increased from 0.01
    "rs_strength": 0.03,     # NEW - Relative Strength
}
```

---

## 🔬 Testing Recommendations

1. **Create Backtest Module**:
   - Test different threshold combinations
   - Measure false positive rate
   - Track recommendation quality over time

2. **Add Diagnostic Dashboard**:
   - Show distribution of scores
   - Show where stocks fail filters
   - Show correlation between metrics

3. **A/B Testing Framework**:
   - Run conservative vs aggressive filters in parallel
   - Compare results over 30 days

---

## ⚠️ Risks & Considerations

1. **Lowering Filters = More False Positives**:
   - Need to clearly mark confidence levels
   - Consider "Core" vs "Aggressive" portfolios

2. **Market Regime Changes**:
   - Bull market settings will fail in bear market
   - Need adaptive system

3. **Overfitting to Current Market**:
   - Don't optimize only for recent winners
   - Keep some diversification

4. **Data Quality Variance**:
   - Not all tickers have full fundamental data
   - Need graceful degradation

---

## 📝 Summary: Yes, We Can Do Better!

**Current State:**
- ✅ Excellent data infrastructure (multiple sources, verification)
- ✅ Comprehensive technical analysis (10+ indicators)
- ✅ Strong risk management (blackout, beta, sector limits)
- ❌ Over-conservative filters (only 1 passing)
- ❌ Under-utilizing advanced metrics (RS, momentum quality, MA alignment)
- ❌ Missing sector/market context
- ❌ Fundamental scoring too harsh

**What to Do:**
1. **Immediate**: Relax filters by 15-20%
2. **Short-term**: Add relative strength and momentum quality to main score
3. **Medium-term**: Implement market regime detection
4. **Long-term**: Add sentiment, multi-timeframe, ML optimization

**Expected Outcome:**
- Current: 1 recommendation (maybe 1 good, 0 bad)
- After Phase 1: 5-8 recommendations (4-6 good, 1-2 uncertain)
- After Phase 2: 10-15 recommendations (8-10 good, 2-5 speculative)

The system has **excellent bones** - we just need to let more candidates through the door!
