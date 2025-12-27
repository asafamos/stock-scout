# מיפוי תהליך הציון - Stock Scout 2025

## 📊 סיכום: איפה מחושב מה

### גרסה נוכחית: **V3 (ML features) + V2 (Risk Engine)**

---

## 1️⃣ תהליך הציון המלא (Flow)

```
[Pipeline Start] → [Technical Indicators] → [ML Prediction] → [Scoring 20d] → [Fundamentals] 
    → [Risk Engine V2] → [Advanced Filters] → [Classification] → [Allocation] → [Display Filters]
```

---

## 2️⃣ קבצים ותפקידם

### 🎯 קבצי ליבה - Scoring & ML

| קובץ | תפקיד | ציונים שמחושבים | גרסה |
|------|-------|-----------------|------|
| `core/unified_logic.py` | אינדיקטורים טכניים + ציון טכני | `build_technical_indicators()`, `compute_tech_score_20d_v2()` | V2 |
| `core/ml_20d_inference.py` | ML prediction (20-day model) | `predict_20d_prob_from_row()`, `apply_live_v3_adjustments()` | **V3** |
| `core/ml_features_v3.py` | Feature engineering למודל ML | relative strength, volatility features | **V3** |
| `core/scoring_pipeline_20d.py` | מיזוג ציון טכני + ML | `compute_final_scores_20d()` → `TechScore_20d`, `ML_20d_Prob`, `FinalScore_20d` | V2+V3 |
| `core/v2_risk_engine.py` | Risk gates + Reliability scoring | `score_ticker_v2_enhanced()` → `conviction_v2_final`, `reliability_v2`, `buy_amount_v2` | **V2** |

### 🔧 קבצים משניים

| קובץ | תפקיד | מה הוא עושה |
|------|-------|-------------|
| `core/pipeline_runner.py` | ריצת הפייפליין המלא | מפעיל את כל הציונים + מיון + פילטרים |
| `advanced_filters.py` | פילטרים מתקדמים (RS, momentum, RR) | `compute_advanced_score()`, `should_reject_ticker()` |
| `core/classification.py` | סיווג Core vs Speculative | `apply_classification()` |
| `core/portfolio.py` | Budget allocation | `allocate_budget()` |
| `core/scoring/fundamental.py` | ציון פונדמנטלי | `compute_fundamental_score_with_breakdown()` |

---

## 3️⃣ הציונים שבשימוש (Score Hierarchy)

### 🏆 ציון עליון (Final Score)
```python
conviction_v2_final  # הציון הסופי ביותר (V2 Risk Engine)
├── conviction_v2_base  # ציון בסיס לפני ML boost
├── reliability_v2  # אמינות הנתונים (0-100)
└── ml_boost_v2  # בונוס מ-ML model
```

### 📈 ציון 20 יום (20-day Scoring)
```python
FinalScore_20d  # מיזוג 50/50 של טכני ו-ML (ranked)
├── TechScore_20d  # ציון טכני (percentile rank 0-100)
│   └── TechScore_20d_v2_raw  # ציון טכני גולמי (0-1)
└── ML_20d_Prob  # הסתברות ML (0-1)
    ├── ML_20d_Prob_live_v3  # V3 עם adjustments
    └── ML_20d_Prob_raw  # חיזוי גולמי מהמודל
```

### 🔢 ציונים טכניים
```python
Score_Tech  # ציון טכני מקורי (legacy, 0-100)
TechScore_20d_v2_raw  # ציון טכני V2 גולמי (0-1)
```

### 💰 ציונים פונדמנטליים
```python
Fundamental_S  # ציון פונדמנטלי כולל (0-100)
├── Quality_Score_F  # איכות (ROE, margins)
├── Growth_Score_F  # גדילה (revenue, earnings YoY)
└── Valuation_Score_F  # שווי (P/E, P/S, P/B)
```

---

## 4️⃣ תהליך הסינון (Filter Chain)

### שלב 1: Pipeline (core/pipeline_runner.py)
```python
1. apply_technical_filters()  # RSI, ATR, Overext, RR, Momentum
2. advanced_filters penalties  # RS_63d, momentum_consistency, risk_reward_ratio
3. Score > 0.1  # מסנן ציונים נמוכים מדי
```

### שלב 2: After Pipeline (stock_scout.py)
```python
4. Min score filter  # Score >= 10.0 או 2.0
5. Top-N filter  # רק 15 הטובות ביותר
6. Display cap  # מקסימום 15 לתצוגה
```

### שלב 3: Pre-Display (stock_scout.py)
```python
7. overall_score_20d >= 2  # ציון מינימום
8. buy_amount_v2 > 0  # רק עם הקצאת תקציב
9. risk_gate_status_v2 != "blocked"  # לא חסום על ידי risk gates
```

---

## 5️⃣ מה קורה בכל שלב

### 🏗️ Pipeline Flow (מפורט)

```python
# 1. Historical Data
data_map = fetch_history_bulk(tickers)  # yfinance download

# 2. Technical Indicators
for ticker, df in data_map:
    tech_df = build_technical_indicators(df)  # core/unified_logic.py
    row = tech_df.iloc[-1]
    
    # 3. Technical Filter
    if not apply_technical_filters(row, strict=False):
        continue  # דחיית מניה
    
    # 4. Technical Score V2
    tech_score = compute_tech_score_20d_v2(row) * 100.0  # 0-100
    
    # 5. ML Prediction V3
    ml_prob_raw = predict_20d_prob_from_row(row)  # core/ml_20d_inference.py
    
    rows.append({
        "Ticker": ticker,
        "TechScore_20d_v2_raw": tech_score / 100,
        "ML_20d_Prob_raw": ml_prob_raw,
        ...
    })

results = pd.DataFrame(rows)

# 6. ML V3 Adjustments
results["ML_20d_Prob_live_v3"] = apply_live_v3_adjustments(results)

# 7. Final Scores 20d
results = compute_final_scores_20d(results, include_ml=True)
# → TechScore_20d, ML_20d_Prob, FinalScore_20d

# 8. Advanced Filters
for ticker:
    signals = compute_advanced_score(ticker, df, benchmark_df)
    catastrophic, reason = should_reject_ticker(signals)
    
    if catastrophic:
        results[ticker]["Score"] = 0.1  # דחייה רכה
    else:
        penalty = calculate_penalties(signals)
        results[ticker]["Score"] = max(0.1, enhanced_score - penalty)

# Filter out Score < 0.1
results = results[results["Score"] >= 0.1]

# 9. Fundamentals
fund_df = fetch_fundamentals_batch(tickers)  # core/data_sources_v2.py
results = merge(results, fund_df)

for ticker:
    fund_score = compute_fundamental_score_with_breakdown(row)
    results[ticker]["Fundamental_S"] = fund_score.total

# 10. Risk Engine V2
for ticker:
    res = score_ticker_v2_enhanced(
        ticker, row, 
        budget_total=5000, 
        min_position=500, 
        enable_ml=True
    )
    # → reliability_v2, conviction_v2_base, conviction_v2_final,
    #    buy_amount_v2, shares_to_buy_v2, risk_gate_status_v2

# Use conviction_v2_final as final Score
results["Score"] = results["conviction_v2_final"]

# 11. Classification
results = apply_classification(results)
# → Risk_Level (core/speculative), Data_Quality (high/medium/low)

# 12. Allocation
results = allocate_budget(results, budget=5000, min_pos=500)
```

---

## 6️⃣ Score Priority (איזה ציון בשימוש?)

### ב-Pipeline Runner:
```python
# Priority order for main Score column:
1. conviction_v2_final  # אם קיים ו-meaningful (>20)
2. TechScore_20d        # fallback כשיש conviction נמוך
3. FinalScore_20d       # fallback אם אין conviction
```

### ב-Stock Scout (UI):
```python
score_candidates = [
    "conviction_v2_final",  # 1st priority
    "Score",                # 2nd priority
    "FinalScore_20d",       # 3rd priority
    "overall_score_20d",    # 4th priority
    "TechScore_20d"         # 5th priority (last resort)
]
```

---

## 7️⃣ גרסאות בשימוש

| רכיב | גרסה נוכחית | קובץ |
|------|-------------|------|
| ML Features | **V3** | `core/ml_features_v3.py` |
| ML Inference | V3 (live_v3_adjustments) | `core/ml_20d_inference.py` |
| Technical Scoring | **V2** | `core/unified_logic.py` → `compute_tech_score_20d_v2()` |
| Risk Engine | **V2** | `core/v2_risk_engine.py` |
| Scoring Pipeline | V2 (uses V3 ML) | `core/scoring_pipeline_20d.py` |
| Data Sources | V2 | `core/data_sources_v2.py` |

---

## 8️⃣ קבצים ישנים/לא בשימוש

❌ קבצים שכנראה לא בשימוש פעיל:
- `core/data_sources.py` (replaced by `data_sources_v2.py`)
- `core/scoring_engine.py` (replaced by `scoring_pipeline_20d.py`)
- Older ML feature files (if any v1/v2 exist)

⚠️ קבצים שצריך לבדוק:
- `core/scoring_engine.py` - האם עדיין בשימוש?
- `core/ml_integration.py` - האם עדיין בשימוש?

---

## 9️⃣ המלצות לאיחוד

### 🎯 מה כדאי לאחד:

1. **מיקום אחד לציון טכני**: הכל ב-`core/scoring/technical.py`
2. **מיקום אחד לציון ML**: הכל ב-`core/scoring/ml_20d.py`
3. **מיקום אחד לציון פונדמנטלי**: הכל ב-`core/scoring/fundamental.py` (כבר קיים)
4. **מיקום אחד לסינונים**: `core/filters/` (technical, advanced, risk)
5. **Pipeline מאוחד**: `core/pipeline.py` (single source of truth)

### 📁 מבנה מוצע:
```
core/
├── scoring/
│   ├── __init__.py
│   ├── technical.py      # כל הציון הטכני (V2)
│   ├── ml_20d.py         # כל ה-ML (V3)
│   ├── fundamental.py    # ציון פונדמנטלי (קיים)
│   └── final_score.py    # מיזוג הציונים הסופי
├── filters/
│   ├── __init__.py
│   ├── technical.py      # technical filters
│   ├── advanced.py       # advanced filters (RS, momentum, RR)
│   └── risk_gates.py     # risk gates from v2_risk_engine
├── pipeline.py           # הפייפליין המרכזי
└── allocation.py         # budget allocation + position sizing
```

---

## 🔟 שאלות לבירור

1. ✅ האם `core/scoring_engine.py` עדיין בשימוש?
2. ✅ האם `core/ml_integration.py` עדיין בשימוש?
3. ⚠️ האם יש קבצי ML ישנים (v1, v2) שצריך למחוק?
4. ⚠️ האם `advanced_filters.py` צריך להיות ב-`core/`?
5. ✅ האם `overall_score_20d` משמש רק כ-alias ל-`Score`?
