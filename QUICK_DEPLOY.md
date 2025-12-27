# 🚀 Stock Scout - Quick Deployment Reference Card

## ⏰ 5-Minute Deployment

### Step 1: Push to GitHub
```bash
cd /workspaces/stock-scout-2
git add -A
git commit -m "Ready for online deployment"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud
1. Go to **streamlit.io**
2. Click **"New app"** → **"Deploy an app"**
3. Select repository: **stock-scout-2**
4. Select file: **stock_scout.py**
5. Click **"Deploy"** (wait 2-5 min)

### Step 3: Add Secrets (Optional)
1. App dashboard → **⋯** menu → **"Manage secrets"**
2. Add your API keys (optional, app works without them)
3. Save → App restarts

✅ **Done!** Your app is live at:
```
https://share.streamlit.io/YOUR_USERNAME/stock-scout-2/main/stock_scout.py
```

---

## 📋 Pre-Deployment Checklist

```
☐ All changes committed: git status (clean)
☐ No .env file in repo: git grep .env (empty)
☐ Models exist: ls models/model_20d_*.pkl (3 files)
☐ Requirements updated: pip freeze > requirements.txt
☐ Config exists: .streamlit/config.toml
☐ Python version: runtime.txt (3.11)
☐ Deployment guides created: ONLINE_DEPLOYMENT_GUIDE.md
```

---

## 🧪 Validation After Deploy

| Check | Expected | Command |
|-------|----------|---------|
| **Load time** | < 60s | Visit app URL |
| **Precomputed Scan** | < 5s | Click "Use Precomputed" |
| **Live Scan** | < 30s | Live mode (50 tickers) |
| **ML Model** | Loaded | Check logs for "✓ Loaded ML" |
| **Scoring Policy** | Auto-selected | Check logs for policy name |
| **Errors** | None | Check Streamlit logs |

---

## 🛠️ Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| App won't load | Check `.streamlit/config.toml` exists |
| No ML predictions | Verify `models/` files committed to git |
| Out of memory | Reduce `UNIVERSE_LIMIT` to 30 in Advanced |
| Slow response | Use "Precomputed Scan" option |
| API keys not working | Add to "Manage Secrets", not `.env` |

---

## 🔄 Update Model Online

```bash
# Retrain locally
python experiments/train_ml_20d.py \
    --input data/training_dataset_20d_v3.csv \
    --output-model models/model_20d_v3.pkl

# Push to GitHub (auto-deploys)
git add models/model_20d_v3.pkl
git commit -m "Update ML model"
git push origin main

# App updates in 2-5 minutes automatically
```

---

## 📊 What Works Online

| Feature | Status | Notes |
|---------|--------|-------|
| Live Scan | ✅ | Uses Yahoo Finance (free) |
| ML Predictions | ✅ | Model bundled in repo |
| Technical Scoring | ✅ | 100% functional |
| Charts | ✅ | Renders in browser |
| CSV Export | ✅ | Download button works |
| Precomputed Scan | ✅ | Fastest (< 5s) |
| Fundamentals | ⚠️ | Need API keys |
| OpenAI Targets | ⚠️ | Need OPENAI_API_KEY |

---

## 💾 Recommended Settings Online

```
Advanced Options:
- Universe Limit: 30-50 (vs 500 local)
- Lookback Days: 60 (vs 90)
- Use Precomputed Scan: ON (faster)
- ML Threshold: 0 (no filtering)
```

---

## 📈 Expected Performance

```
First Visit:     30-60 seconds
Precomputed:     2-5 seconds  
Live Scan:       10-30 seconds
Charts:          1-2 seconds
```

---

## 🔐 Security Reminders

✅ **Good**: API keys in Streamlit "Manage Secrets"  
❌ **Bad**: API keys in `.env` file committed to git  
✅ **Good**: `.env` in `.gitignore`  
❌ **Bad**: Hardcoded credentials in Python files  

---

## 📚 Full Documentation

- **Complete Guide**: `ONLINE_DEPLOYMENT_GUIDE.md`
- **Checklist**: `DEPLOYMENT_CHECKLIST.md`
- **Hebrew**: `DEPLOYMENT_HEBREW.md`
- **Status**: `DEPLOYMENT_READY.md`

---

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io
- **GitHub Issues**: Report bugs in your repo
- **Streamlit Community**: https://discuss.streamlit.io

---

## ✅ SUCCESS CRITERIA

- [ ] App loads in < 60s
- [ ] Precomputed scan works
- [ ] ML predictions show
- [ ] No errors in logs
- [ ] Charts render
- [ ] Can download CSV

**All checked? 🚀 You're production-ready!**

---

**Status**: ✅ Ready to Deploy  
**Last Updated**: December 25, 2024
