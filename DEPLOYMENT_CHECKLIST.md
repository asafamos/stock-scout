# GitHub Deployment Checklist

## Pre-Deployment Verification

### Code Quality
- [ ] All Python files have no syntax errors: `python -m py_compile core/*.py stock_scout.py`
- [ ] No hardcoded paths (use relative paths instead)
- [ ] All imports work: `PYTHONPATH=$PWD python -c "import stock_scout; print('OK')"`
- [ ] No large binary files (> 100MB) in repo

### Dependencies
- [ ] `requirements.txt` is complete and up-to-date
- [ ] All required packages listed
- [ ] No version conflicts
- [ ] Test install works: `pip install -r requirements.txt`

### Configuration Files
- [ ] `.streamlit/config.toml` exists and is valid
- [ ] `.streamlit/packages.txt` exists (if system packages needed)
- [ ] `runtime.txt` specifies Python 3.11
- [ ] `.env.example` has correct template

### Model Files
- [ ] `models/model_20d_v3.pkl` exists (main model)
- [ ] `models/model_20d_v2.pkl` exists (fallback)
- [ ] Models are tracked in Git LFS (if > 50MB):
  ```bash
  git lfs install
  git lfs track "models/*.pkl"
  git add .gitattributes models/
  git commit -m "Add models to LFS"
  ```

### Documentation
- [ ] `README.md` updated with online link
- [ ] `ONLINE_DEPLOYMENT_GUIDE.md` reviewed
- [ ] `.env.example` has all necessary variables documented
- [ ] `STREAMLIT_CLOUD_SETUP.md` exists and up-to-date

### Git Repository
- [ ] All changes committed: `git status` shows clean
- [ ] Main branch is stable (no broken features)
- [ ] No `.env` file (should be in `.gitignore`)
- [ ] No local cache/logs (should be in `.gitignore`)
- [ ] Remote is set to GitHub: `git remote -v`

---

## Deployment Steps

### 1. Final Git Push
```bash
cd /workspaces/stock-scout-2

# Verify everything is committed
git status

# Push to GitHub
git push origin main
```

### 2. Verify on GitHub
1. Go to github.com/YOUR_USERNAME/stock-scout-2
2. Verify latest commit is visible
3. Check files structure is correct
4. Verify `models/*.pkl` files are present

### 3. Deploy to Streamlit Cloud
1. Go to streamlit.io
2. Click "New app" → "Deploy an app"
3. Connect GitHub account
4. Select repository: `stock-scout-2`
5. Select branch: `main`
6. Select file: `stock_scout.py`
7. Click "Deploy"
8. Wait for build (2-5 minutes)

### 4. Configure Secrets
1. App dashboard → "⋯" menu → "Manage secrets"
2. Add API keys (copy from `.env` if available):
```toml
ALPHA_VANTAGE_API_KEY = "your_key"
FINNHUB_API_KEY = "your_key"
POLYGON_API_KEY = "your_key"
TIINGO_API_KEY = "your_key"
OPENAI_API_KEY = "your_key"
```
3. Save and app will restart

---

## Post-Deployment Validation

### 1. App Loads
- [ ] App loads within 60 seconds
- [ ] No error messages in logs
- [ ] UI is responsive

### 2. Core Functionality
- [ ] Advanced options expand/collapse
- [ ] Scan button works
- [ ] Results display correctly
- [ ] Charts render without errors

### 3. Data Loading
- [ ] Uses precomputed scan initially (fast)
- [ ] Can switch to live scan (if API keys configured)
- [ ] ML predictions show (29.6% hit rate expected)
- [ ] No NaN values in key columns

### 4. Performance
- [ ] Precomputed scan: < 5s
- [ ] Live scan (50 tickers): < 30s
- [ ] No memory warnings

### 5. Logs Check
- [ ] Dashboard logs show no critical errors
- [ ] No warnings about missing models
- [ ] ML model loaded successfully
- [ ] Scoring mode auto-selected correctly

---

## Troubleshooting Checklist

### If app doesn't load:
- [ ] Check logs in Streamlit Cloud dashboard
- [ ] Verify `stock_scout.py` syntax: `python -m py_compile stock_scout.py`
- [ ] Check `requirements.txt` for conflicts
- [ ] Verify Python version is 3.11

### If ML predictions missing:
- [ ] Verify `models/model_20d_v*.pkl` files exist in repo
- [ ] Check `core/ml_20d_inference.py` paths are relative
- [ ] Look for errors in Streamlit logs about model loading

### If data doesn't load:
- [ ] Check API keys are in Secrets (if using external data)
- [ ] Verify network connectivity in logs
- [ ] Try precomputed scan first (no API needed)

### If out of memory:
- [ ] Reduce `UNIVERSE_LIMIT` (target: 30-50)
- [ ] Enable caching in advanced options
- [ ] Use precomputed scan instead of live

---

## Continuous Deployment

### Update Model Online
```bash
# Retrain locally (or in cloud)
python experiments/train_ml_20d.py \
    --input data/training_dataset_20d_v3.csv \
    --output-model models/model_20d_v3.pkl

# Commit and push (auto-deploys)
git add models/model_20d_v3.pkl
git commit -m "Update ML model"
git push origin main
```

### Update Code Online
```bash
# Make changes
# Test locally
# Commit and push (auto-deploys)
git add -A
git commit -m "Feature: description"
git push origin main
```

---

## Monitoring (Weekly)

- [ ] App loads successfully
- [ ] ML model selected correct policy (check logs)
- [ ] No critical errors in logs
- [ ] Response time < 10s for precomputed scan
- [ ] Dependencies are up-to-date

---

## Security Checklist

- [ ] No API keys in code (use Secrets only)
- [ ] No hardcoded credentials
- [ ] No `.env` file in repository
- [ ] `.gitignore` excludes sensitive files
- [ ] Public GitHub repo (no private data exposed)
- [ ] Secrets not logged to console

---

## Success Criteria

✅ All items checked above  
✅ App loads in < 60s on first visit  
✅ App responds in < 10s for precomputed scan  
✅ ML predictions available  
✅ No critical errors in logs  
✅ Can be accessed from any browser  

**Status**: 🚀 **READY FOR PRODUCTION**

---

**Last Updated**: December 25, 2024  
**GitHub URL Template**: https://share.streamlit.io/YOUR_USERNAME/stock-scout-2/main/stock_scout.py
