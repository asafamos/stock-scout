# Stock Scout - הנחיות פריסה לאינטרנט 🚀

## מה צריך להיות מוכן לפריסה?

### ✅ קבצים חובה
- [x] `stock_scout.py` - יישום ראשי
- [x] `requirements.txt` - תלויות Python
- [x] `runtime.txt` - גרסת Python (3.11)
- [x] `.streamlit/config.toml` - הגדרות Streamlit
- [x] `models/model_20d_v*.pkl` - מודלים מאומנים
- [x] `core/` - תיקיית המודול הראשי
- [x] `ONLINE_DEPLOYMENT_GUIDE.md` - הנחיות זה
- [x] `DEPLOYMENT_CHECKLIST.md` - רשימת בדיקה

---

## פריסה ל-Streamlit Cloud (מומלץ)

### שלב 1: הכנת GitHub

```bash
cd /workspaces/stock-scout-2

# בדקו שהכל מעודכן
git status

# שלחו לGitHub
git add -A
git commit -m "הכנה לפריסה אונליין"
git push origin main
```

### שלב 2: הפעלה בStreamlit Cloud

1. **היכנסו לstreamlit.io**
2. **לחצו: Deploy → Deploy an app**
3. **בחרו Repository: stock-scout-2**
4. **בחרו File: stock_scout.py**
5. **לחצו Deploy** (ממתינים 2-5 דקות)

### שלב 3: הוספת סודות (API Keys)

1. **לחצו ⋯ → Manage secrets**
2. **הוסיפו את ה-API keys (אופציונלי):**
```toml
ALPHA_VANTAGE_API_KEY = "המפתח_שלכם"
FINNHUB_API_KEY = "המפתח_שלכם"
POLYGON_API_KEY = "המפתח_שלכם"
TIINGO_API_KEY = "המפתח_שלכם"
OPENAI_API_KEY = "המפתח_שלכם"
```
3. **שמרו** (האפליקציה תופעל מחדש)

---

## לאחר הפריסה - בדיקות

### ✅ האפליקציה עולה?
- התחברו לקישור של Streamlit Cloud
- האפליקציה תעלה תוך 30-60 שניות בפעם הראשונה

### ✅ נתונים נטענים?
- לחצו "Live Scan" בתפריט
- תוך 10-30 שניות יופיעו המלצות

### ✅ ניבויי ML עובדים?
- בדקו שיש מספרים בעמודה "FinalScore_20d"
- צפויה התאמה של ~30% (תקן בחלק העליון)

### ✅ אין שגיאות?
- בדקו Logs בדשבורד של Streamlit Cloud
- צריך רק alerts, לא errors

---

## תשובות לשאלות תכופות

### Q: האפליקציה איטית מדי
**A**: תקטינו את `UNIVERSE_LIMIT` ל-30-50 במקום 500

### Q: אין נתונים
**A**: אתם בשימוש ב-Yahoo Finance (בחינם). API keys הם אופציונליים

### Q: זיכרון מלא (Out of Memory)
**A**: תשתמשו בـ"Precomputed Scan" בתפריט Advanced

### Q: איפה נשמרים הנתונים?
**A**: בשרת Streamlit Cloud, מתנקה בכל הפעלה מחדש

### Q: איך מעדכנים מודל?
**A**: תעדכנו `models/model_20d_v3.pkl` וכלחצו push - האפליקציה תתעדכן אוטומטית

---

## הגדרות מומלצות אונליין

### UNIVERSE_LIMIT (מספר מניות)
- **Free Tier Streamlit**: 30-50
- **Paid Streamlit**: 100-200
- **Local**: עד 500

### LOOKBACK_DAYS (ימי היסטוריה)
- **Online**: 60-90
- **Local**: 365

### Cache TTL
- **Online**: 1800 שניות (30 דקות)
- **Local**: 3600 שניות (שעה)

---

## עדכון מודל בין חיים

```bash
# אימנו מודל חדש בלוקל
python experiments/train_ml_20d.py \
    --input data/training_dataset_20d_v3.csv \
    --output-model models/model_20d_v3.pkl

# שלחנו לGitHub
git add models/model_20d_v3.pkl
git commit -m "עדכון ML model v3"
git push

# Streamlit Cloud תתעדכן אוטומטית תוך 2-5 דקות
```

---

## בעיות שכיחות ופתרונים

| בעיה | גורם | פתרון |
|------|------|--------|
| האפליקציה לא עולה | שגיאה בקוד | בדקו logs בdashboard |
| אין מלצות | חוסר נתונים | הוסיפו API keys ל-secrets |
| איטית מדי | יותר מדי מניות | הקטינו UNIVERSE_LIMIT |
| Out of Memory | זיכרון חסר | כבו "Live Scan", השתמשו בPrecomputed |

---

## אבטחה - חשוב!

### ✅ אמור להיות:
- API keys בـ"Manage Secrets" (לא בקוד)
- `.env` בـ`.gitignore`
- אין hardcoded passwords

### ❌ אסור:
- לשמור `OPENAI_API_KEY` בקוד
- לשמור `.env` ב-GitHub
- להעביר sensitive info בQR code

---

## הגדרות מתקדמות (אופציונלי)

### התאמת זמנים
`.streamlit/config.toml`:
```toml
[client]
maxUploadSize = 200
defaultCachedMessageExpirationTime = 3600

[server]
runOnSave = true
```

### ניטור ביצועים
- בדקו response time בלוגים
- צפויה עלייה של ~30% לעומת lokal

### Continuous Deployment
- כל commit ל-`main` מפעיל deployment אוטומטי
- בדקו סטטוס בdashboard של Streamlit Cloud

---

## קישורים שימושיים

- **Streamlit Cloud**: https://share.streamlit.io
- **Streamlit Docs**: https://docs.streamlit.io
- **Stock Scout Repo**: https://github.com/YOUR_USERNAME/stock-scout-2

---

## רשימת בדיקה סופית ✅

- [ ] כל הקבצים committed ב-GitHub
- [ ] האפליקציה עולה תוך 60 שניות
- [ ] Precomputed Scan עובד (תוך 5 שנייות)
- [ ] ML predictions מופיעות
- [ ] אין errors בlogs
- [ ] אפשר להוריד CSV
- [ ] גרפים מתרסמים

**אם הכל ✅ → אתם מוכנים לייצור!**

---

**עדכון אחרון**: 25 בדצמבר 2024  
**סטטוס**: ✅ מוכן לפריסה אונליין
