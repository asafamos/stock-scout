# Stock Scout - סקירה מקיפה ומלאה של המערכת

**תאריך סקירה:** 3 בפברואר 2026
**גרסה נבדקת:** v3.0 (ML 20d Model)

---

## 📊 סיכום מנהלים

Stock Scout היא מערכת מתוחכמת לזיהוי מניות לפני עלייה, המשלבת ניתוח טכני, פונדמנטלי ולמידת מכונה (ML). המערכת עובדת גם ב-Streamlit Online וגם מקומית, עם אוטומציות של GitHub Actions.

### ⚠️ בעיות קריטיות שזוהו

| בעיה | חומרה | השפעה |
|------|-------|--------|
| **חוסר התאמה בין מטא-דאטה לפיצ'רים** | 🔴 קריטי | המודל מתאמן על 5 פיצ'רים בעוד המערכת מצפה ל-34 |
| **ML Boost מקסימלי קטן מדי** | 🟡 בינוני | ±10 נקודות זה רק ~10% מהציון - ML לא משפיע מספיק |
| **חוסר Backtesting אמיתי לפני Deploy** | 🔴 קריטי | אין validation אוטומטי של דיוק המודל |
| **Fallback Logic מוסתר** | 🟡 בינוני | כשה-bridge נכשל, המערכת עוברת ללוגיקה ישנה בשקט |
| **Feature Staleness** | 🟡 בינוני | פיצ'רים כמו Sector_RS תמיד 0 כי אין ETF mapping |

---

## 🏗️ ארכיטקטורה נוכחית

### מבנה הקבצים
```
stock-scout-2/
├── stock_scout.py          # UI ראשי (Streamlit) - 200K bytes!
├── core/                   # לוגיקה עסקית
│   ├── unified_logic.py    # ציון מאוחד - 1834 שורות
│   ├── pipeline_runner.py  # Pipeline ראשי - 2298 שורות
│   ├── ml_integration.py   # ML Boost - 836 שורות
│   ├── feature_registry.py # 34 פיצ'רים מוגדרים
│   └── scoring/
│       └── unified_scorer.py # Scorer מאוחד
├── scripts/
│   ├── run_full_scan.py    # סריקה אוטומטית
│   └── train_rolling_ml_20d.py # אימון ML
├── .github/workflows/
│   ├── auto_scan.yml       # 4 סריקות ביום
│   ├── daily-training.yml  # אימון יומי
│   └── ci.yml              # בדיקות CI
└── models/
    └── model_20d_v3.pkl    # מודל ML פעיל
```

### זרימת נתונים
```
Universe (2000 מניות)
    ↓
FMP/Polygon/Finnhub (מחירים + פונדמנטלים)
    ↓
build_technical_indicators() - חישוב 40+ אינדיקטורים טכניים
    ↓
compute_recommendation_scores() - ציון טכני + פונדמנטלי
    ↓
ML 20d Model - Boost של ±10 נקודות
    ↓
apply_classification() - סיווג סיכון
    ↓
allocate_budget() - הקצאת תקציב
    ↓
תוצאות סופיות
```

---

## 🔴 בעיות קריטיות - ניתוח מעמיק

### 1. **חוסר התאמה קריטי בין המודל לפיצ'רים**

**הבעיה:**
הקובץ `models/model_20d_v3.metadata.json` מראה:
```json
{
  "feature_list": ["RSI", "ATR_Pct", "Return_20d", "Return_10d", "Return_5d"]
}
```

בעוד `core/feature_registry.py` מגדיר **34 פיצ'רים**!

**השפעה:**
- המודל מקבל רק 5 פיצ'רים אבל המערכת מנסה להזין 34
- זה מוביל ל-fallback שקט לערכי ברירת מחדל
- דיוק המודל נפגע משמעותית

**פתרון מומלץ:**
```python
# בקובץ train_rolling_ml_20d.py - לאמן עם כל 34 הפיצ'רים
from core.feature_registry import get_feature_names
features_to_use = get_feature_names("v3")  # 34 features
```

### 2. **ML Boost מוגבל מדי**

**הבעיה:**
בקובץ `ml_integration.py`:
```python
def calculate_ml_boost(base_conviction, ml_probability, max_boost_pct=10.0):
    boost_fraction = (ml_probability - 0.5) * 2.0  # Range: -1 to +1
    ml_boost = boost_fraction * max_boost_pct  # Max ±10 points
```

**השפעה:**
- מניה עם ML prob של 0.9 מקבלת רק +8 נקודות
- מניה עם ML prob של 0.1 מקבלת רק -8 נקודות
- **זה לא מספיק להבדיל בין מניות "חמות" באמת**

**פתרון מומלץ:**
```python
# הגדל את ה-boost ל-20% לפחות
max_boost_pct = 20.0

# או השתמש בפונקציה א-לינארית שמעדיפה extreme probabilities
def calculate_ml_boost_v2(base_conviction, ml_probability, max_boost_pct=20.0):
    if ml_probability is None:
        return base_conviction, 0.0, "unavailable"

    # Non-linear boost: emphasize extreme probabilities
    deviation = ml_probability - 0.5
    # Exponential scaling: mild effect near 0.5, strong at extremes
    boost_factor = np.sign(deviation) * (abs(deviation) ** 0.7) * 2
    ml_boost = boost_factor * max_boost_pct

    return base_conviction + ml_boost, ml_boost, "applied"
```

### 3. **חוסר Backtesting אמיתי ב-CI/CD**

**הבעיה:**
ב-`daily-training.yml`:
```yaml
- name: Backtest Precision@K
  run: python -m scripts.backtest_precision_at_k || true  # || true = תמיד עובר!
```

**השפעה:**
- מודל גרוע יכול להיכנס לפרודקשן
- אין threshold מינימלי לדיוק
- אין rollback אוטומטי

**פתרון מומלץ:**
```yaml
- name: Backtest Precision@K with threshold
  run: |
    python -m scripts.backtest_precision_at_k > backtest_output.txt
    PRECISION=$(grep "Precision@20" backtest_output.txt | cut -d: -f2)
    if (( $(echo "$PRECISION < 0.55" | bc -l) )); then
      echo "❌ Model precision $PRECISION is below threshold 0.55"
      exit 1
    fi
    echo "✓ Model precision $PRECISION passed threshold"
```

### 4. **Sector Relative Strength תמיד 0**

**הבעיה:**
בקובץ `ml_integration.py`, שורות 576-583:
```python
# 18. Sector_RS - stock_return_20d - sector_etf_return_20d
sector_rs = safe_get(all_data, ['Sector_RS', 'sector_rs', ...], None)
if sector_rs is None:
    stock_ret = features['Return_20d']
    sector_ret = safe_get(all_data, ['sector_return_20d', ...], 0.0)  # תמיד 0!
    sector_rs = stock_ret - sector_ret  # = stock_ret
```

**השפעה:**
- Sector_RS = Return_20d במקום חישוב אמיתי
- אין יתרון יחסי לסקטור
- 3 פיצ'רים (Sector_RS, Sector_Momentum, Sector_Rank) לא עובדים

**פתרון מומלץ:**
```python
# הוסף mapping של סקטורים ל-ETFs
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

def compute_sector_features(ticker_data, technical_indicators):
    sector = ticker_data.get("sector", "Unknown")
    sector_etf = SECTOR_ETF_MAP.get(sector)

    if sector_etf:
        sector_df = yf.download(sector_etf, period="3mo", progress=False)
        sector_return = (sector_df["Close"].iloc[-1] / sector_df["Close"].iloc[-20] - 1)
        stock_return = technical_indicators.get("Return_20d", 0)
        return {
            "Sector_RS": stock_return - sector_return,
            "Sector_Momentum": sector_return,
            "Sector_Rank": 1.0 if stock_return > sector_return else 0.0
        }
    return {"Sector_RS": 0.0, "Sector_Momentum": 0.0, "Sector_Rank": 0.5}
```

---

## 🟡 בעיות בינוניות

### 5. **ציון VCP מחושב בצורה פשטנית**

**מצב נוכחי:**
```python
# בקובץ unified_logic.py שורות 999-1010
ratio = (atr_10 / atr_30)
vcp_raw = (1.0 - ratio.clip(lower=0.0)).where((atr_10 < atr_30) & ratio.notna(), 0.0)
```

**בעיה:**
- לא מזהה את ה-pattern המלא של VCP (contraction → tighter → breakout)
- לא משווה לממוצע נע
- לא בודק volume contraction

**פתרון מומלץ:**
```python
def compute_vcp_score_v2(df: pd.DataFrame, lookback: int = 30) -> float:
    """
    VCP Score based on Mark Minervini's criteria:
    1. Price consolidation with decreasing volatility
    2. Multiple contractions (T1, T2, T3)
    3. Price near 52-week high
    4. Volume drying up
    """
    if len(df) < lookback:
        return 0.0

    close = df["Close"].iloc[-lookback:]
    high = df["High"].iloc[-lookback:]
    low = df["Low"].iloc[-lookback:]
    volume = df["Volume"].iloc[-lookback:]

    # 1. Volatility Contraction: ATR decreasing
    atr_recent = (high[-5:] - low[-5:]).mean()
    atr_old = (high[:10] - low[:10]).mean()
    volatility_score = 1.0 if atr_recent < atr_old * 0.7 else 0.5

    # 2. Price near high: within 25% of 52w high
    current_price = close.iloc[-1]
    high_52w = df["High"].rolling(252).max().iloc[-1]
    price_position = current_price / high_52w if high_52w > 0 else 0
    price_score = 1.0 if price_position > 0.75 else price_position

    # 3. Volume drying up
    vol_recent = volume[-5:].mean()
    vol_old = volume[:10].mean()
    volume_score = 1.0 if vol_recent < vol_old * 0.6 else 0.5

    # 4. Tightness: recent range vs average
    range_recent = (close[-5:].max() - close[-5:].min()) / close[-5:].mean()
    range_old = (close[:20].max() - close[:20].min()) / close[:20].mean()
    tightness_score = 1.0 if range_recent < range_old * 0.5 else 0.5

    # Weighted combination
    vcp_score = (
        0.30 * volatility_score +
        0.25 * price_score +
        0.25 * volume_score +
        0.20 * tightness_score
    )

    return float(np.clip(vcp_score, 0, 1))
```

### 6. **Market Regime לא מעודכן בזמן אמת**

**מצב נוכחי:**
```python
# unified_logic.py - classify_regime
def classify_regime(row):
    if dd < -0.15 or vix_pct > 0.85:
        return 'PANIC'
    if dd < -0.08 or vix_pct > 0.70:
        return 'CORRECTION'
    ...
```

**בעיה:**
- VIX percentile מחושב על 252 ימים - לא תגובתי
- לא מתחשב ב-breadth (כמה מניות עולות/יורדות)
- לא משתמש ב-advance/decline line

**פתרון מומלץ:**
```python
def get_market_regime_v2():
    """Real-time regime based on multiple signals."""
    spy = yf.download("SPY", period="3mo", progress=False)
    vix = yf.download("^VIX", period="3mo", progress=False)

    # Quick calculations
    spy_20d_return = (spy["Close"].iloc[-1] / spy["Close"].iloc[-20] - 1)
    spy_50d_return = (spy["Close"].iloc[-1] / spy["Close"].iloc[-50] - 1)
    vix_current = vix["Close"].iloc[-1]
    vix_20d_avg = vix["Close"].iloc[-20:].mean()

    # Breadth approximation using RSP (equal-weight S&P) vs SPY
    rsp = yf.download("RSP", period="1mo", progress=False)
    breadth = (rsp["Close"].iloc[-1] / rsp["Close"].iloc[-20] - 1) - spy_20d_return

    # Scoring
    score = 0
    if spy_20d_return > 0.03: score += 2
    elif spy_20d_return < -0.03: score -= 2

    if spy_50d_return > 0.05: score += 1
    elif spy_50d_return < -0.05: score -= 1

    if vix_current < 20: score += 1
    elif vix_current > 30: score -= 2

    if breadth > 0.01: score += 1
    elif breadth < -0.02: score -= 1

    # Map to regime
    if score >= 3:
        return "TREND_UP", 1.10
    elif score <= -3:
        return "PANIC", 0.70
    elif score <= -1:
        return "CORRECTION", 0.85
    else:
        return "SIDEWAYS", 1.00
```

### 7. **Earnings Calendar לא משפיע מספיק**

**מצב נוכחי:**
ב-`risk.py` יש earnings blackout של 7 ימים, אבל:
- לא משפיע על הציון
- רק מסנן מניות
- לא מזהיר על surprise פוטנציאלי

**פתרון מומלץ:**
```python
def adjust_score_for_earnings(ticker, base_score, days_to_earnings):
    """
    Adjust score based on earnings proximity.

    Strategy:
    - 0-3 days: Very risky, heavy penalty
    - 4-7 days: Risky, moderate penalty
    - 8-14 days: Slight boost (potential run-up)
    - After earnings: Check surprise direction
    """
    if days_to_earnings is None:
        return base_score, "unknown"

    if days_to_earnings <= 3:
        return base_score * 0.7, "earnings_imminent"
    elif days_to_earnings <= 7:
        return base_score * 0.85, "earnings_soon"
    elif days_to_earnings <= 14:
        # Stocks often run up before earnings
        return base_score * 1.05, "pre_earnings_run"
    else:
        return base_score, "clear"
```

---

## 🟢 GitHub Actions - שיפורים מומלצים

### מצב נוכחי
```yaml
# auto_scan.yml - 4 סריקות ביום
schedule:
  - cron: '30 13 * * 1-5'  # Pre-market
  - cron: '0 15 * * 1-5'   # Early session
  - cron: '0 20 * * 1-5'   # Late session
  - cron: '30 21 * * 1-5'  # End of day
```

### שיפורים מומלצים

#### 1. הוסף Market Hours Check דינמי
```yaml
- name: Check market hours dynamically
  id: market_hours
  run: |
    HOUR=$(TZ=America/New_York date +%H)
    DAY=$(date +%u)

    if [ "$DAY" -gt 5 ]; then
      echo "is_trading=false" >> $GITHUB_OUTPUT
      echo "reason=weekend" >> $GITHUB_OUTPUT
    elif [ "$HOUR" -lt 9 ] || [ "$HOUR" -gt 16 ]; then
      echo "is_trading=false" >> $GITHUB_OUTPUT
      echo "reason=outside_hours" >> $GITHUB_OUTPUT
    else
      echo "is_trading=true" >> $GITHUB_OUTPUT
    fi
```

#### 2. הוסף Alert מיידי למניות חמות
```yaml
- name: Send high-confidence alerts to Telegram
  if: success()
  env:
    TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
    TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
  run: |
    python -c "
    import pandas as pd
    import requests

    df = pd.read_parquet('data/scans/latest_scan.parquet')
    hot_stocks = df[df['ML_20d_Prob'] > 0.75].head(5)

    if len(hot_stocks) > 0:
        msg = '🔥 מניות חמות:\n'
        for _, row in hot_stocks.iterrows():
            msg += f\"• {row['Ticker']}: Score {row['FinalScore_20d']:.1f}, ML {row['ML_20d_Prob']:.2f}\n\"

        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg}
        )
    "
```

#### 3. הוסף Validation לפני Deploy של מודל חדש
```yaml
# daily-training.yml - הוסף שלב validation
- name: Validate new model before deploy
  run: |
    python -c "
    import joblib
    import json

    # Load new model
    model_data = joblib.load('models/model_20d_v3.pkl')

    # Check feature count
    if 'feature_names' in model_data:
        n_features = len(model_data['feature_names'])
        if n_features < 30:
            print(f'❌ Model has only {n_features} features, expected 34')
            exit(1)

    # Check metrics
    if 'metrics' in model_data:
        auc = model_data['metrics'].get('auc_score', 0)
        if auc < 0.55:
            print(f'❌ Model AUC {auc} is below threshold 0.55')
            exit(1)

    print('✓ Model validation passed')
    "
```

---

## 📈 משקלות ציון - המלצות לשינוי

### מצב נוכחי (scoring_config.py)
```python
FINAL_SCORE_WEIGHTS = {
    "technical": 0.55,
    "fundamental": 0.25,
    "ml": 0.20,
}

TECH_WEIGHTS = {
    "vcp": 0.20,
    "relative_strength": 0.17,
    "ma": 0.14,
    "mom": 0.14,
    "rsi": 0.12,
    ...
}
```

### המלצות לשינוי
```python
# הגדל את משקל ה-ML ו-VCP
FINAL_SCORE_WEIGHTS_V2 = {
    "technical": 0.45,      # ↓ מ-0.55
    "fundamental": 0.20,    # ↓ מ-0.25
    "ml": 0.35,             # ↑ מ-0.20 - ML צריך להשפיע יותר!
}

# הדגש VCP ו-RS יותר
TECH_WEIGHTS_V2 = {
    "vcp": 0.25,            # ↑ מ-0.20 - VCP הכי חשוב לזיהוי breakouts
    "relative_strength": 0.20,  # ↑ מ-0.17
    "momentum_consistency": 0.15,
    "ma_alignment": 0.12,
    "volume_surge": 0.10,
    "rsi": 0.08,            # ↓ RSI פחות חשוב
    "overextension": 0.05,
    "reward_risk": 0.05,
}
```

---

## 🎯 המלצות ליישום מיידי (לפי עדיפות)

### עדיפות 1 - קריטי (השבוע)

1. **תקן את חוסר ההתאמה בפיצ'רים:**
   ```bash
   # אמן מחדש את המודל עם כל 34 הפיצ'רים
   python scripts/train_rolling_ml_20d.py --features all --horizon 20
   ```

2. **הגדל את ML Boost:**
   ```python
   # core/ml_integration.py שורה 272
   max_boost_pct = 20.0  # במקום 10.0
   ```

3. **הוסף validation אמיתי ל-CI:**
   ```yaml
   # .github/workflows/daily-training.yml
   - name: Backtest with threshold
     run: python -m scripts.backtest_precision_at_k --min-precision 0.55
   ```

### עדיפות 2 - חשוב (חודש)

4. **הוסף Sector ETF mapping:**
   - צור קובץ `core/sector_mapping.py` עם מיפוי מלא
   - עדכן `prepare_ml_features()` לחשב Sector_RS אמיתי

5. **שפר VCP calculation:**
   - יישם את `compute_vcp_score_v2()` מהמלצות למעלה

6. **הוסף Telegram alerts:**
   - הגדר Telegram bot
   - הוסף workflow step לשליחת התראות

### עדיפות 3 - שיפור (רבעון)

7. **Market Regime V2:**
   - הוסף breadth calculation
   - השתמש ב-RSP לחישוב participation

8. **Earnings Intelligence:**
   - הוסף adjustment לציון לפי קרבה לדוחות
   - שמור היסטוריה של surprises

9. **Backtesting Dashboard:**
   - צור dashboard לעקוב אחרי דיוק המודל לאורך זמן
   - התראות על model drift

---

## 📊 סיכום מספרי

| מדד | ערך נוכחי | ערך מומלץ |
|-----|----------|----------|
| פיצ'רים ב-ML | 5 | 34 |
| ML Boost מקסימלי | ±10 pts | ±20 pts |
| משקל ML בציון סופי | 20% | 35% |
| VCP weight | 20% | 25% |
| Sector features עובדים | 0/3 | 3/3 |
| Validation threshold | אין | AUC > 0.55 |

---

## 🏁 סיכום

המערכת בנויה טוב מבחינת ארכיטקטורה אבל יש כמה בעיות לוגיות קריטיות שפוגעות ביכולת שלה לזהות מניות לפני עלייה:

1. **ML לא עובד כמו שצריך** - מתאמן על 5 פיצ'רים במקום 34
2. **ML לא משפיע מספיק** - boost מוגבל ל-±10 נקודות
3. **Sector features לא מחושבים** - תמיד ערכי ברירת מחדל
4. **אין validation אוטומטי** - מודל גרוע יכול להיכנס לפרודקשן

אם תתקן את 4 הבעיות האלה, המערכת תשתפר משמעותית ביכולת לזהות מניות לפני עלייה.

---

*נכתב על ידי Claude - 3 בפברואר 2026*
