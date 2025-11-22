# 🚨 תיקון מהיר - האפליקציה תקועה ב-GitHub

## מה עשינו עד עכשיו ✅

1. **תיקון Beta Timeout** - הוספנו timeout ל-`fetch_beta_vs_benchmark`
2. **תמיכה במשתני סביבה** - הוספנו `UNIVERSE_LIMIT`, `LOOKBACK_DAYS`, `SMART_SCAN`
3. **עדיפות ל-Streamlit Secrets** - Secrets של Streamlit Cloud נקראים לפני .env
4. **הודעת Debug** - מוצג בUI: `Config: Universe=20 | Lookback=90d | Smart=True`

## השלבים שצריך לעשות **עכשיו** 🎯

### 1️⃣ היכנס ל-Streamlit Cloud
```
https://share.streamlit.io/
```

### 2️⃣ בחר את האפליקציה
```
stock-scout-fm4iuuknxjwjbcg95inbcj.streamlit.app
```

### 3️⃣ לחץ Settings (⚙️) → Secrets

### 4️⃣ הוסף את השורות האלו (העתק הכל):
```toml
UNIVERSE_LIMIT = "20"
LOOKBACK_DAYS = "90"
SMART_SCAN = "true"
TOPK_RECOMMEND = "5"
```

### 5️⃣ לחץ **Save**

### 6️⃣ חזור לדף הראשי ולחץ **⋮ → Reboot app**

### 7️⃣ המתן 30-60 שניות

## איך לבדוק שזה עובד? ✓

### בדיקה 1: הודעת Config
מתחת ל-"Data Sources Overview" תראה:
```
⚙️ Config: Universe=20 | Lookback=90d | Smart=True
```

**אם תראה `Universe=50`** → חזור לשלב 3 ווודא ש-Secrets נשמרו!

### בדיקה 2: מספר מניות
בספינר תראה:
```
📊 Fetching historical data for 20 stocks...
```

**אם תראה 50** → ה-Secrets לא נקראו נכון!

### בדיקה 3: זמן עיבוד
האפליקציה צריכה לסיים תוך **30-60 שניות** (לא יותר!)

### בדיקה 4: יש המלצות
בסוף תראה לפחות **1-5 המלצות**.

## עדיין לא עובד? 🔧

### נסה את זה:
1. **נקה Cache**: לחץ "🔄 Clear Cache & Reload" באפליקציה
2. **בדוק Logs**: Settings → Manage app → View logs
3. **חפש בלוג**:
   - `fetch_beta_vs_benchmark` - אם מופיע הרבה = בעיית timeout
   - `Building stock universe` - אם לוקח >10 שניות = יותר מדי מניות
   - `HTTP 403/429` - בעיות API rate limit

### הורד עוד יותר את ההגדרות:
```toml
UNIVERSE_LIMIT = "10"    # רק 10 מניות!
LOOKBACK_DAYS = "60"     # פחות היסטוריה
SMART_SCAN = "false"     # ללא סינון חכם
```

## בדיקה מקומית 🧪

רוצה לבדוק מקומית עם אותן הגדרות כמו GitHub?

```bash
python test_github_settings.py
```

אם הכל תקין, תראה:
```
✅ All configuration values match expected GitHub settings!
```

## השוואה: לפני ואחרי

| מה | לפני | אחרי |
|----|------|------|
| מניות | 50 | **20** ✓ |
| זמן | timeout | **30-60s** ✓ |
| Beta | תקוע | **timeout 10s** ✓ |
| Secrets | לא נקרא | **עדיפות ראשונה** ✓ |
| Debug | אין | **Config line** ✓ |

## עזרה נוספת? 💬

אם אחרי כל זה זה עדיין לא עובד:

1. שלח צילום מסך של שורת ה-Config
2. העתק את ה-Logs (Settings → View logs, last 100 lines)
3. בדוק שכל ה-API keys תקינים (Check Secrets 🔐)

---
**עודכן:** 22 נובמבר 2025, 09:20
**גרסה:** 3.0 (Secrets priority fix)
