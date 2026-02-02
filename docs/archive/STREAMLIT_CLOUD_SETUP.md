# הגדרת Streamlit Cloud - מדריך מהיר 🚀

## הבעיה שתיקנו
המערכת תקעה ב-GitHub כי:
- ✅ ניסתה לסרוק יותר מדי מניות (50 במקום 20)
- ✅ חישובי Beta תקעו ללא timeout
- ✅ יותר מדי קריאות API

## השלבים להגדרה ב-Streamlit Cloud

### שלב 1: כניסה להגדרות
1. היכנס ל-[Streamlit Cloud](https://share.streamlit.io/)
2. בחר את האפליקציה `stock-scout`
3. לחץ על **⚙️ Settings** (גלגל השיניים)
4. בחר **Secrets**

### שלב 2: הוספת משתני הסביבה
העתק והדבק את התוכן הבא ל-**Secrets**:

```toml
# Performance Settings - חובה להוסיף!
UNIVERSE_LIMIT = "40"
LOOKBACK_DAYS = "90"
SMART_SCAN = "true"
TOPK_RECOMMEND = "5"
TOPN_RESULTS = "15"

# API Keys - הזן את המפתחות שלך (אין להשתמש בערכים לדוגמה!)
# ⚠️ SECURITY WARNING: Never commit real API keys to git!
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"
FINNHUB_API_KEY = "your_finnhub_key_here"
POLYGON_API_KEY = "your_polygon_key_here"
TIINGO_API_KEY = "your_tiingo_key_here"
FMP_API_KEY = "your_fmp_key_here"
OPENAI_API_KEY = "your_openai_key_here"
```

**⚠️ חשוב מאוד:**
- לאחר הוספת/עדכון Secrets, **חובה** ללחוץ **Save**
- אז ללחוץ **Reboot app** מהתפריט הראשי (לא מחלון Secrets)
- המתן 30-60 שניות עד שהאפליקציה עולה מחדש

### שלב 3: Reboot
1. לחץ **Save**
2. לחץ **Reboot app** (מחזור האפליקציה)
3. המתן 30-60 שניות

### שלב 4: בדיקה
האפליקציה צריכה עכשיו:
- ✅ לסרוק רק 20 מניות (במקום 50)
- ✅ לסיים תוך 30-60 שניות (במקום timeout)
- ✅ להציג לפחות המלצה אחת

## איך לבדוק שזה עובד?

### בדיקה מהירה:
בעמוד הראשי, **מתחת ל-Data Sources Overview**, תראה:
```
⚙️ Config: Universe=40 | Lookback=90d | Smart=True
```

אם תראה `Universe=50` - **הסביבה לא טעונה נכון!** חזור לשלב 2.

גם תראה בספינרים:
```
🔍 Building stock universe... [צריך להיות מהיר!]
📊 Fetching historical data for 40 stocks... [לא 50!]
```

### בדיקה מפורטת:
גלול למטה ולחץ על **"Check Secrets 🔐"**
תראה:
```
Alpha: ***xxxx ✓
Finnhub: ***xxxx ✓
...
```

## אם זה עדיין לא עובד

### אפשרות 1: הורד עוד יותר
```toml
UNIVERSE_LIMIT = "10"   # רק 10 מניות
SMART_SCAN = "false"    # ללא smart filtering
```

### אפשרות 2: בדוק logs
1. Settings → Manage app
2. לחץ **View logs**
3. חפש errors כמו:
   - `TimeoutError`
   - `API rate limit`
   - `fetch_beta_vs_benchmark`

### אפשרות 3: ניקוי cache
בתוך האפליקציה, לחץ **"🔄 Clear Cache & Reload"**

## השוואת ביצועים

| הגדרה | מקומי (Dev) | GitHub Cloud |
|-------|-------------|--------------|
| מניות לסריקה | 50-200 | **20** ✓ |
| זמן עיבוד | 60-120s | **30-60s** ✓ |
| קריאות API | ~100-200 | **~30-50** ✓ |
| סיכוי ל-timeout | נמוך | **נמוך** ✓ |

## טיפים נוספים

### להגדלת מהירות:
```toml
UNIVERSE_LIMIT = "15"      # עוד יותר מהיר
LOOKBACK_DAYS = "60"       # פחות היסטוריה
```

### לבדיקה מקומית של הגדרות GitHub:
```bash
export UNIVERSE_LIMIT=20
export SMART_SCAN=true
streamlit run stock_scout.py
```

## עזרה נוספת?

אם האפליקציה עדיין לא עובדת:
1. שלח לי screenshot של ה-logs
2. בדוק שכל ה-API keys תקינים
3. נסה להפעיל מקומית עם אותן הגדרות

---
**עודכן:** 22 נובמבר 2025
**גרסה:** 2.0 (עם timeout protection)
