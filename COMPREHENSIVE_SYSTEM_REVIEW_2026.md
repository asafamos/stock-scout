# 📊 דוח סקירה מקיף - Stock Scout System
## תאריך: 1 בפברואר 2026

---

## ✅ בעיות שתוקנו

### תאריך: 2 בפברואר 2026

| בעיה | סטטוס | פתרון |
|------|-------|-------|
| מודל ML עם 5 features בלבד | ✅ תוקן | המודל v3 משתמש ב-34 features (core/feature_registry.py) |
| חסר Cross-Validation | ✅ תוקן | TimeSeriesSplit עם 5 folds |
| API Key חשוף | ✅ תוקן | נוסף APIKeyManager ב-core/api_keys.py |
| Calibration Hack | ✅ תוקן | Isotonic Regression בזמן training |
| נתיב מודל שגוי | ✅ תוקן | _find_model_path() עם fallbacks ב-ml_integration.py |
| Feature mismatch | ✅ תוקן | Feature Registry כ-Single Source of Truth |
| Scoring inconsistency | ✅ תוקן | UnifiedScorer ב-core/scoring/unified_scorer.py |
| GitHub Actions בזמנים לא רלוונטיים | ✅ תוקן | תזמון לפי שעות NYSE + בדיקת חגים |

---

## 📋 תקציר מנהלים

המערכת **Stock Scout** היא מערכת מתקדמת לזיהוי מניות לפני עליות משמעותיות, המשלבת ניתוח טכני, פונדמנטלי ו-ML. המערכת מבוססת על ארכיטקטורה מודולרית עם ריצות אוטומטיות ב-GitHub Actions.

**מצב נוכחי**: המערכת פונקציונלית אך יש פוטנציאל משמעותי לשיפור בכמה תחומים קריטיים.

---

## 🔴 בעיות קריטיות - דורשות טיפול מיידי

### 1. **מודל ML פשטני מדי**
**מיקום**: `scripts/train_rolling_ml_20d.py`

**הבעיה**:
```python
# רק 5 פיצ'רים! לא מספיק לזיהוי מניות מטאוריות
features = ['RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d']
```

**למה זה קריטי**:
- המודל משתמש רק ב-5 features בסיסיים
- חסרים features קריטיים: VCP_Ratio, RS vs SPY, Pocket Pivots, Volume Surge
- ה-README מציין 22 features אבל בפועל רק 5 בשימוש
- הסף לזיהוי "winner" הוא 20%+ תוך 20 יום - גבוה מדי וגורם ל-label imbalance

**תיקון נדרש**:
```python
# Feature set מורחב
FULL_FEATURES = [
    # Technical Base
    'RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d',
    # VCP/Volatility Contraction
    'VCP_Ratio', 'Tightness_Ratio', 'RangeRatio_5_20',
    # Relative Strength
    'RS_21d', 'RS_63d', 'Dist_From_52w_High',
    # Volume Analysis
    'Volume_Surge_Ratio', 'Pocket_Pivot_Ratio', 'PV_Correlation',
    # MA Alignment
    'MA_Alignment_Score', 'Above_MA50', 'Above_MA200',
    # Momentum
    'Momentum_Consistency', 'Mom_Acceleration',
    # Market Context
    'SPY_RSI', 'Market_Trend', 'Sector_RS',
]
```

---

### 2. **אין Cross-Validation תקין**
**מיקום**: `scripts/train_rolling_ml_20d.py`

**הבעיה**:
```python
# Training על כל הדאטה ואז בדיקת AUC על אותו דאטה!
model.fit(X, y)
auc = roc_auc_score(y, model.predict_proba(X)[:, 1])  # LEAK!
```

**למה זה קריטי**:
- ה-AUC המדווח הוא על ה-training set (overfitting!)
- אין train/test split
- אין time-series aware validation (עתיד לא יכול להשפיע על עבר)

**תיקון נדרש**:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    scores.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print(f"OOS AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

---

### 3. **Calibration Hack במקום תיקון שורש**
**מיקום**: `core/ml_integration.py` שורות 103-120

**הבעיה**:
```python
# CALIBRATION FIX: Spread probabilities to use full range
# Issue: Model returns very uniform 0.79-0.86 range
# Solution: Apply non-linear transformation to spread values
if 0.7 <= prob_win <= 0.9:
    prob_win = 0.5 + ((prob_win - 0.7) / 0.2) * 0.5
```

**למה זה קריטי**:
- ה-hack מסתיר את הבעיה האמיתית: המודל לא מזהה variance
- זה מוסיף שכבת אי-ודאות נוספת
- Calibration צריך להיעשות כחלק מה-training, לא post-hoc

**תיקון נדרש**:
- להשתמש ב-Platt Scaling או Isotonic Regression בזמן Training
- לאמן מודל עם יותר features שיוצרים variance טבעי
- להסיר את ה-hack לאחר תיקון המודל

---

### 4. **API Key Hardcoded בקוד**
**מיקום**: `scripts/train_rolling_ml_20d.py` שורה 17

**הבעיה**:
```python
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "HARDCODED_KEY_REMOVED")
```

**סיכון אבטחה**: API key חשוף בקוד פתוח!

**תיקון מיידי**:
```python
from core.api_keys import get_api_key
POLYGON_KEY = get_api_key("POLYGON_API_KEY", required=True)
```

---

## 🟠 בעיות משמעותיות - דורשות שיפור

### 5. **חוסר עקביות ב-Scoring Logic**
**מיקום**: קבצים מרובים

**הבעיה**:
יש 3 מערכות scoring שונות שלא תמיד מסונכרנות:
- `core/scoring_engine.py` - compute_overall_score
- `core/unified_logic.py` - compute_final_score_with_patterns
- `core/ml_integration.py` - integrate_ml_with_conviction

**תיקון נדרש**:
- ליצור Single Source of Truth אחד לחישוב Score
- להסיר code duplication
- לתעד את ה-flow בצורה ברורה

---

### 6. **GitHub Actions לא אופטימליים**

**auto_scan.yml**:
```yaml
- cron: '0 8 * * *'   # 08:00 UTC
- cron: '0 20 * * *'  # 20:00 UTC
```

**בעיות**:
- הסריקות לא מתואמות לשעות המסחר (NYSE 9:30-16:00 ET = 14:30-21:00 UTC)
- חסר trigger ל-pre-market (06:00 UTC)
- אין error alerting

**תיקון מומלץ**:
```yaml
on:
  schedule:
    - cron: '30 13 * * 1-5'  # 13:30 UTC = שעה לפני פתיחת NYSE
    - cron: '0 20 * * 1-5'   # 20:00 UTC = לפני סגירה
    - cron: '30 21 * * 1-5'  # 21:30 UTC = אחרי סגירה, EOD analysis
```

---

### 7. **Daily Training חסר Feature Engineering מתקדם**

**מיקום**: `scripts/run_daily_training.py`

**הבעיה**: הקובץ מפנה ל-train_rolling_ml_20d.py שלא כולל:
- Feature interactions
- Sector/Industry encoding
- Market regime features
- Seasonality features

**תוספות נדרשות**:
```python
# Feature Interactions
df['RSI_ATR_Interaction'] = df['RSI'] * df['ATR_Pct']
df['Momentum_Quality'] = df['Return_20d'] / (df['ATR_Pct'] + 0.001)

# Market Context
df['Days_Since_Market_High'] = ...
df['Sector_Rotation_Score'] = ...
df['VIX_Level'] = ...  # Need to fetch VIX data
```

---

### 8. **Test Coverage נמוכה**

**סטטיסטיקה**:
- סה"כ ~5,650 שורות קוד בטסטים
- ~50 קבצי טסט
- הרבה legacy tests שלא רצים

**חסרים טסטים ל**:
- ML inference edge cases
- Multi-source data aggregation
- Classification edge cases
- Full pipeline integration

**פעולות נדרשות**:
1. להריץ `pytest --cov=core --cov-report=html`
2. לזהות קבצים עם coverage נמוך
3. להוסיף טסטים לפונקציות קריטיות

---

## 🟡 שיפורים מומלצים - לטווח בינוני

### 9. **Meteor Mode לא מנוצל במלואו**

**מיקום**: `advanced_filters.py`

הלוגיקה של VCP + Pocket Pivots + RS קיימת אבל:
- לא משולבת במודל ה-ML
- חסר weight optimization
- אין backtesting ספציפי ל-Meteor signals

**המלצה**: ליצור Meteor-specific ML model

---

### 10. **חסר Real-time Alerting**

המערכת סורקת אבל אין התראות בזמן אמת:
- אין Webhook ל-Telegram/Slack/Discord
- אין Email alerts
- אין Push notifications

**הוספה מומלצת** ב-`scripts/run_full_scan.py`:
```python
def send_alert(signal: dict):
    webhook_url = os.getenv("ALERT_WEBHOOK")
    if webhook_url:
        requests.post(webhook_url, json={
            "ticker": signal['Ticker'],
            "score": signal['FinalScore_20d'],
            "ml_prob": signal['ML_20d_Prob'],
            "action": "BUY_SIGNAL"
        })
```

---

### 11. **Database Integration חסר**

כרגע הכל CSV/Parquet files. מומלץ:
- SQLite לפיתוח local
- PostgreSQL/TimescaleDB לפרודקשן
- Redis לcaching של API responses

---

### 12. **Backtesting Framework לא שלם**

**מיקום**: `backtest_recommendations.py`, `unified_backtest.py`

**חסר**:
- Position sizing simulation
- Slippage modeling
- Transaction costs
- Portfolio-level metrics (Sharpe, Sortino, Max Drawdown)

---

## 🟢 נקודות חזקות - לשמר

### ✅ **ארכיטקטורה מודולרית**
- הפרדה טובה בין core/, scripts/, tests/
- Dataclasses לtype safety
- Logging infrastructure

### ✅ **Multi-source Data Aggregation**
- 10 providers שונים עם fallback logic
- Price verification cross-source
- Reliability scoring

### ✅ **CI/CD Pipeline**
- GitHub Actions עובד
- Model versioning עם timestamps
- Automatic commits

### ✅ **Risk Management**
- Earnings blackout
- Sector diversification caps
- Position sizing logic

---

## 📋 תוכנית פעולה מומלצת

### שלב 1 - תיקונים קריטיים (שבוע 1-2)
1. [ ] הסרת API key מהקוד
2. [ ] הוספת Time-Series CV לtraining
3. [ ] הרחבת feature set ל-20+ features
4. [ ] הסרת calibration hack

### שלב 2 - שיפורי ML (שבוע 3-4)
5. [ ] Platt Scaling/Isotonic Regression
6. [ ] Feature importance analysis
7. [ ] Hyperparameter tuning (Optuna)
8. [ ] Model ensembling (XGBoost + LightGBM + CatBoost)

### שלב 3 - Infrastructure (שבוע 5-6)
9. [ ] Real-time alerting
10. [ ] Database integration
11. [ ] Improved GitHub Actions timing
12. [ ] Error monitoring (Sentry)

### שלב 4 - Advanced Features (שבוע 7-8)
13. [ ] Sector rotation detection
14. [ ] Market regime classification
15. [ ] Portfolio optimization
16. [ ] Full backtesting framework

---

## 📊 טבלת עדיפויות

| Priority | Issue | Impact | Effort | Status |
|----------|-------|--------|--------|--------|
| 🔴 P0 | API Key exposed | Security | Low | TODO |
| 🔴 P0 | No Cross-Validation | Model Quality | Medium | TODO |
| 🔴 P1 | Limited Features | Accuracy | High | TODO |
| 🔴 P1 | Calibration Hack | Reliability | Medium | TODO |
| 🟠 P2 | Scoring Inconsistency | Maintenance | Medium | TODO |
| 🟠 P2 | GH Actions Timing | Timeliness | Low | TODO |
| 🟡 P3 | Test Coverage | Quality | High | TODO |
| 🟡 P3 | Alerting | Usability | Medium | TODO |
| 🟢 P4 | Database | Scalability | High | FUTURE |

---

## 🎯 מטריקות יעד לאחר תיקונים

| Metric | Current | Target |
|--------|---------|--------|
| ML Features | 5 | 25+ |
| OOS AUC | ~0.53 (overfitted) | 0.60+ (validated) |
| Test Coverage | ~40% | 80% |
| Precision@20 | Unknown | 65%+ |
| Alert Latency | N/A | <5min |

---

## 📝 סיכום

המערכת יש לה בסיס טוב אבל דורשת שיפורים משמעותיים כדי להגיע לביצועים אמיתיים בזיהוי מניות מטאוריות. הבעיות הקריטיות ביותר הן:

1. **ML Model** - צריך יותר features ו-proper validation
2. **Training Pipeline** - cross-validation חסר לגמרי
3. **Code Quality** - API key exposed + calibration hacks

לאחר תיקון הבעיות הללו, המערכת יכולה להפוך לכלי רציני לזיהוי הזדמנויות.

---

*נוצר על ידי Claude | 1 בפברואר 2026*
