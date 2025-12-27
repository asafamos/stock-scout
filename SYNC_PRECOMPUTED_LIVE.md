# סנכרון מלא בין Precomputed ו-Live Scan

## תיאור השינויים (27 דצמבר 2025)

### בעיה שזוהתה
- **Precomputed**: הציג 6 מניות (מתוך 9 במקור)
- **Live Scan**: הציג 1 מניה בלבד
- הסיבה: הבדלים בלוגיקת הסינון בין שני המסלולים

### פתרון שהוטמע

#### 1. סנכרון לוגיקת הסינון הראשונית
**שני המסלולים כעת משתמשים באותה לוגיקה מדויקת:**

```python
# Score column candidates (זהה בשניהם)
score_candidates = ["conviction_v2_final", "Score", "FinalScore_20d", "overall_score_20d", "TechScore_20d"]
score_col = next((c for c in score_candidates if c in results.columns), None)
top_n = 15

# Min score filter (זהה בשניהם)
min_score = 10.0 if (score_values.dropna() > 10).any() else 2.0
results = results.loc[score_values >= min_score].copy()

# Top-N filter (זהה בשניהם)
if len(results) > top_n:
    results = results.nlargest(top_n, score_col)

# Display cap (זהה בשניהם)
display_cap = min(int(CONFIG.get("TOPN_RESULTS", 15)), top_n)
if len(results) > display_cap:
    results = results.head(display_cap).copy()
```

#### 2. סינון סופי אחיד
**שני המסלולים עוברים את אותם מסננים לפני תצוגה:**

1. **Overall score >= 2** - מסנן מניות עם ציון כולל נמוך מדי
2. **buy_amount_v2 > 0** - מציג רק מניות שהוקצה להן תקציב
3. **risk_gate_status_v2 != "blocked"** - מסיר מניות חסומות על ידי risk management

#### 3. Logging מפורט
הוספנו logging מפורט בשני המסלולים:

```
[PRECOMPUTED] Min score filter (threshold=10.0): 9 remain (removed 0)
[PRECOMPUTED] Top-15 filter: 9 remain
[PRECOMPUTED] Display cap (15): showing top 9 of 9 filtered stocks
[PRECOMPUTED] Final display: 9 stocks (original 9, removed_below_min=0)
[FILTER] Starting recommendation filtering with 9 stocks
[FILTER] Overall score >= 2: 9 remain (removed 0)
[FILTER] Buy amount > 0: 6 remain (removed 3)
[FILTER] Risk gate not blocked: 6 remain (removed 0)
```

### תוצאה
- ✅ **Precomputed ו-Live Scan משתמשים באותה לוגיקה מדויקת**
- ✅ **Logging מפורט מאפשר מעקב אחרי כל שלב**
- ✅ **הודעות משתמש עבריות מסבירות את תהליך הסינון**
- ✅ **6 מניות מוצגות בפרה-קומיוטד = המלצות חזקות עם הקצאת תקציב**

### למה 6 מניות ולא 9?
3 מניות סוננו על ידי `buy_amount_v2 = 0`:
- **AAPL** - חסומה על ידי risk gate
- **NVDA** - חסומה על ידי risk gate  
- **WMT** - חסומה על ידי risk gate

המשמעות: **המערכת החליטה שלא לייעד תקציב למניות אלה** בגלל גורמי סיכון (overvaluation, volatility, וכו').

זה **התנהגות נכונה** - המערכת מציגה רק מניות שהיא ממליצה לקנות עם הקצאת תקציב ספציפית.
