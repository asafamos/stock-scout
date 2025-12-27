# תוכנית איחוד הלוגיקה - Stock Scout 2025

## 🎯 מטרה
לרכז את כל הלוגיקה במבנה מסודר וברור, להסיר כפילויות, ולאחד גרסאות.

---

## 📋 שלב 1: ניתוח המצב הנוכחי

### קבצים פעילים (בשימוש):
1. ✅ `core/unified_logic.py` - אינדיקטורים טכניים + ציון טכני V2
2. ✅ `core/ml_20d_inference.py` - ML prediction V3
3. ✅ `core/ml_features_v3.py` - Feature engineering V3
4. ✅ `core/scoring_pipeline_20d.py` - מיזוג ציונים (20d)
5. ✅ `core/v2_risk_engine.py` - Risk gates + Reliability V2
6. ✅ `core/data_sources_v2.py` - משיכת נתונים
7. ✅ `core/classification.py` - סיווג Core/Speculative
8. ✅ `core/portfolio.py` - Budget allocation
9. ✅ `advanced_filters.py` - פילטרים מתקדמים
10. ✅ `core/pipeline_runner.py` - הפייפליין המרכזי

### קבצים חלקיים (שימוש מינימלי):
1. ⚠️ `core/scoring_engine.py` - רק `evaluate_rr_unified()` בשימוש
2. ⚠️ `core/ml_integration.py` - רק מ-v2_risk_engine (אופציונלי)

### קבצים מיותרים (להסרה/ארכיון):
1. ❌ `core/data_sources.py` - replaced by v2
2. ❌ קבצי ML ישנים (אם קיימים)

---

## 🏗️ שלב 2: מבנה מוצע חדש

```
core/
├── data/
│   ├── __init__.py
│   ├── sources.py              # משיכת נתונים (rename from data_sources_v2.py)
│   └── indicators.py           # אינדיקטורים טכניים (from unified_logic.py)
│
├── scoring/
│   ├── __init__.py
│   ├── technical.py            # ציון טכני (from unified_logic.py)
│   ├── ml_scoring.py           # ML scoring (from ml_20d_inference.py + ml_features_v3.py)
│   ├── fundamental.py          # ציון פונדמנטלי (קיים)
│   ├── final.py                # מיזוג ציונים סופי (from scoring_pipeline_20d.py)
│   └── risk.py                 # Risk scoring (from v2_risk_engine.py)
│
├── filters/
│   ├── __init__.py
│   ├── technical.py            # Technical filters (from unified_logic.py)
│   ├── advanced.py             # Advanced filters (from advanced_filters.py)
│   └── risk_gates.py           # Risk gates (from v2_risk_engine.py)
│
├── classification/
│   ├── __init__.py
│   └── classifier.py           # Core/Speculative classification (from classification.py)
│
├── allocation/
│   ├── __init__.py
│   └── budget.py               # Budget allocation (from portfolio.py)
│
├── pipeline.py                 # הפייפליין המרכזי (from pipeline_runner.py)
├── config.py                   # הגדרות (קיים)
└── models.py                   # מודלים (קיים)
```

---

## 🔧 שלב 3: תהליך האיחוד (Refactoring Plan)

### Phase 1: ארגון מחדש (1-2 שעות)
1. ✅ צור מבנה תיקיות חדש
2. ✅ העתק קבצים למיקומים החדשים
3. ✅ עדכן imports בכל הקבצים
4. ✅ הרץ tests לוודא שהכל עובד

### Phase 2: איחוד לוגיקה (2-3 שעות)
1. ✅ אחד אינדיקטורים טכניים ב-`data/indicators.py`
2. ✅ אחד ציון טכני ב-`scoring/technical.py`
3. ✅ אחד ML ב-`scoring/ml_scoring.py` (ml_20d_inference + ml_features_v3)
4. ✅ אחד פילטרים ב-`filters/`
5. ✅ פשט את הפייפליין ב-`pipeline.py`

### Phase 3: ניקוי (1 שעה)
1. ✅ הסר קבצים ישנים/מיותרים
2. ✅ הסר code דומה/מיותר
3. ✅ עדכן תיעוד
4. ✅ הוסף type hints
5. ✅ הוסף docstrings

### Phase 4: בדיקות (1 שעה)
1. ✅ הרץ את כל ה-tests
2. ✅ בדוק precomputed scan
3. ✅ בדוק live scan
4. ✅ ודא שהתוצאות זהות

---

## 📝 שלב 4: פרטי האיחוד לכל קובץ

### 1. data/indicators.py (אינדיקטורים טכניים)
**מקור**: `core/unified_logic.py::build_technical_indicators()`

```python
# כולל:
- MA (20, 50, 200)
- RSI
- ATR + ATR_Pct
- MACD
- ADX
- Volume indicators
- Momentum calculations
- All technical features for ML
```

### 2. scoring/technical.py (ציון טכני)
**מקור**: `core/unified_logic.py::compute_tech_score_20d_v2()`

```python
# כולל:
- compute_tech_score_20d_v2() - V2 scoring
- Legacy scoring functions (if needed)
- Normalization helpers
```

### 3. scoring/ml_scoring.py (ML)
**מקורות**: 
- `core/ml_20d_inference.py`
- `core/ml_features_v3.py`

```python
# כולל:
- ML model loading
- Feature engineering (V3)
- predict_20d_prob_from_row()
- apply_live_v3_adjustments()
- All ML helpers
```

### 4. scoring/final.py (ציון סופי)
**מקור**: `core/scoring_pipeline_20d.py`

```python
# כולל:
- compute_final_scores_20d()
- Blend tech + ML → FinalScore_20d
- Ranking logic
- Sorting policies
```

### 5. scoring/risk.py (Risk + Reliability)
**מקור**: `core/v2_risk_engine.py`

```python
# כולל:
- calculate_reliability_v2()
- score_ticker_v2_enhanced()
- conviction_v2 calculation
- buy_amount_v2 logic
- Risk gate logic
```

### 6. filters/technical.py (פילטרים טכניים)
**מקור**: `core/unified_logic.py::apply_technical_filters()`

```python
# כולל:
- apply_technical_filters() - 3 tiers (strict/normal/relaxed)
- RSI, ATR, Overext, RR, Momentum thresholds
```

### 7. filters/advanced.py (פילטרים מתקדמים)
**מקור**: `advanced_filters.py`

```python
# כולל:
- compute_advanced_score()
- should_reject_ticker()
- RS_63d, momentum_consistency, risk_reward_ratio
- Dynamic thresholds
```

### 8. filters/risk_gates.py (Risk Gates)
**מקור**: `core/v2_risk_engine.py` (חלק)

```python
# כולל:
- apply_risk_gates()
- Check reliability thresholds
- Block/reduce/pass decisions
```

### 9. pipeline.py (פייפליין מרכזי)
**מקור**: `core/pipeline_runner.py`

```python
# כולל:
- run_scan_pipeline() - הפונקציה הראשית
- Orchestration של כל השלבים
- Error handling
- Logging
```

---

## 🚀 שלב 5: תוכנית ביצוע

### Option A: Refactoring מלא (מומלץ אבל לוקח זמן)
- זמן: 5-7 שעות
- סיכון: בינוני-גבוה
- תועלת: מבנה מושלם, קל לתחזוקה

### Option B: Refactoring חלקי (מהיר יותר)
- זמן: 2-3 שעות
- סיכון: נמוך
- תועלת: שיפור משמעותי, לא מושלם

### Option C: ניקוי בלבד (המהיר ביותר) ✅ **מומלץ להתחיל**
- זמן: 1 שעה
- סיכון: נמוך מאוד
- תועלת: הסרת code מיותר, תיעוד ברור

---

## 🎯 אז מה עושים עכשיו?

### המלצה שלי: **Option C → B → A**

1. **עכשיו (15 דקות)**: 
   - ✅ תיעדתי את המפה המלאה ב-`docs/SCORING_FLOW_MAP.md`
   - ⏭️ תגיד לי אם אתה רוצה שאמשיך

2. **הצעד הבא (30 דקות)**:
   - צור `core/scoring/__init__.py` עם exports נקיים
   - העבר פונקציות קטנות ל-`scoring/` (כמו `evaluate_rr_unified`)
   - עדכן imports ב-stock_scout.py

3. **לאחר מכן (1-2 שעות)**:
   - אחד ML files (`ml_20d_inference` + `ml_features_v3` → `scoring/ml_scoring.py`)
   - אחד filters (`unified_logic` + `advanced_filters` → `filters/`)

---

## ❓ שאלות לך

1. **באיזו אופציה אתה רוצה להתחיל? A/B/C**
2. **יש משהו ספציפי שמפריע לך ברמה יומיומית?** (למשל: קשה למצוא איפה מחושב ציון X)
3. **כמה זמן אתה מוכן להשקיע בזה עכשיו?**
4. **אתה רוצה שאמשיך עם הרפקטורינג או רק התיעוד מספיק לך?**

תגיד לי ואני אמשיך בהתאם! 🚀
