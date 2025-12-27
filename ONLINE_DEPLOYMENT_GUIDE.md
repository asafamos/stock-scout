# Stock Scout - Online Deployment Guide

## Deployment to Streamlit Cloud (Recommended)

### Prerequisites
- GitHub repository with Stock Scout code
- Streamlit Cloud account (free at streamlit.io)
- Environment variables configured in Streamlit secrets

### Step 1: Prepare GitHub Repository

1. **Commit all changes**:
```bash
cd /workspaces/stock-scout-2
git add -A
git commit -m "Prepare for online deployment"
git push origin main
```

2. **Verify key files exist**:
- ✅ `stock_scout.py` - Main app
- ✅ `requirements.txt` - Python dependencies
- ✅ `runtime.txt` - Python version (3.11)
- ✅ `.streamlit/config.toml` - Streamlit config
- ✅ `models/model_20d_v*.pkl` - Pre-trained ML models
- ✅ All files in `core/` directory
- ✅ `.env.example` - Template for secrets

### Step 2: Configure Streamlit Cloud Secrets

1. Go to **streamlit.io** → Deploy
2. Connect your GitHub repository
3. Select `stock_scout.py` as the main file
4. Click "Advanced settings"
5. Add secrets (copy from `.env` or use values):

```toml
# Financial Data APIs (Optional - used only if available)
ALPHA_VANTAGE_API_KEY = "your_key_here"
FINNHUB_API_KEY = "your_key_here"
POLYGON_API_KEY = "your_key_here"
TIINGO_API_KEY = "your_key_here"

# OpenAI (Optional - for target price predictions)
OPENAI_API_KEY = "your_key_here"

# Cache settings
CACHE_DIR = ".streamlit_cache"
CACHE_TTL = "3600"
```

### Step 3: Deploy

1. Click "Deploy"
2. Wait for build to complete (usually 2-5 minutes)
3. Your app is live at: `https://share.streamlit.io/YOUR_USERNAME/stock-scout-2/main/stock_scout.py`

---

## Key Environment Variables

### API Keys (Optional but Recommended)
- **ALPHA_VANTAGE_API_KEY**: Financial data (requires key from alphavantage.co)
- **FINNHUB_API_KEY**: Company fundamentals (requires key from finnhub.io)
- **POLYGON_API_KEY**: Stock data provider (requires key from polygon.io)
- **TIINGO_API_KEY**: Financial data (requires key from tiingo.com)
- **OPENAI_API_KEY**: For target price predictions using GPT

### App Settings
- **CACHE_DIR**: Directory for cached data (default: `.streamlit_cache`)
- **CACHE_TTL**: Cache time-to-live in seconds (default: 3600 = 1 hour)

---

## Online vs Local Differences

### Working Without API Keys
- ✅ App runs with Yahoo Finance data only
- ✅ ML predictions work (model bundled in repo)
- ✅ Technical analysis works
- ⚠️ Fundamentals disabled (shows N/A)
- ⚠️ Advanced filters limited
- ⚠️ OpenAI target prices disabled

### Data Storage
- **Online**: Cache stored in `/tmp` (ephemeral, clears on restart)
- **Local**: Cache stored in `.streamlit_cache` (persistent)

### Performance
- **Online**: First load takes 30-60s (model loading, data fetch)
- **Local**: 5-10s (cached data)
- **Subsequent loads**: 2-5s (both)

---

## Troubleshooting Online Deployment

### Issue: App crashes on startup
**Solution**: 
1. Check logs in Streamlit Cloud dashboard
2. Verify all dependencies in `requirements.txt`
3. Check that model files exist in `models/`

### Issue: "Module not found" errors
**Cause**: Python path issues in cloud environment  
**Solution**:
1. Ensure all imports use relative paths
2. Check that `__init__.py` exists in all packages
3. Verify `core/`, `experiments/` directories are in repo

### Issue: "Out of memory" error
**Cause**: Streamlit Cloud has limited RAM (~1GB)  
**Solution**:
1. Reduce `UNIVERSE_LIMIT` in advanced options
2. Use precomputed scans instead of live scans
3. Clear cache and restart app

### Issue: Data fetch times out
**Cause**: API rate limits or network issues  
**Solution**:
1. Enable caching (default: 1 hour)
2. Use fewer tickers (reduce UNIVERSE_LIMIT)
3. Retry after timeout

### Issue: Model file not found
**Cause**: Model not committed to git  
**Solution**:
```bash
# Check model exists
ls -lh models/*.pkl

# Add to git
git add models/*.pkl
git commit -m "Add ML models"
git push
```

---

## Optimizing for Online Performance

### 1. Use Precomputed Scans
```python
# Fastest option - uses cached data
"precomputed" mode in advanced options
Load time: 2-5 seconds
```

### 2. Cache Settings
```toml
# In .streamlit/config.toml
[client]
maxUploadSize = 200

[logger]
level = "warning"  # Reduce logging overhead
```

### 3. Limit Universe Size
```
Advanced Options → Universe Limit: 50-100 (instead of 500)
```

### 4. Enable Offline Mode
```
Advanced Options → Use Precomputed Scan: ON
Uses latest_scan.parquet for instant results
```

---

## Model Distribution

### Pre-trained Models Included
- `models/model_20d_v3.pkl` - Primary model (v3 with auto-selected policy)
- `models/model_20d_v2.pkl` - Fallback model (v2, if v3 unavailable)
- `models/model_20d_v1.pkl` - Legacy (if v2 unavailable)

### Model Loading Order
1. Try v3 (preferred: auto-selected scoring policy)
2. Fall back to v2 if v3 not found
3. Fall back to v1 if v2 not found
4. Disable ML if no models available

### Auto-Selected Scoring Policy
- Policy automatically chosen based on recent performance
- No user intervention required
- Stored in model bundle metadata
- Can be overridden by editing `core/ml_20d_inference.py`

---

## GitHub Repository Structure

```
stock-scout-2/
├── stock_scout.py              # Main Streamlit app (entry point)
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version (3.11)
├── .streamlit/
│   ├── config.toml             # Streamlit configuration
│   └── packages.txt            # System packages
├── core/                        # Core modules
│   ├── __init__.py
│   ├── ml_20d_inference.py     # ML model loading & inference
│   ├── unified_logic.py        # Technical indicators & scoring
│   ├── data_sources_v2.py      # Multi-source data fetching
│   └── ...
├── experiments/                 # Experiment & training scripts
│   ├── offline_recommendation_audit.py
│   ├── train_ml_20d.py
│   ├── validate_ml_improvements.py
│   └── ...
├── models/                      # Pre-trained ML models (COMMITTED)
│   ├── model_20d_v3.pkl
│   ├── model_20d_v2.pkl
│   └── model_xgboost_5d.pkl
├── data/                        # Data directory (created at runtime)
│   ├── scans/
│   │   ├── latest_scan.parquet
│   │   └── latest_scan.json
│   └── ...
├── reports/                     # Generated reports (optional)
└── .env.example                 # Template for environment variables
```

---

## Continuous Integration / Updates

### Update Model Online
To deploy an updated model without code changes:

1. Retrain locally:
```bash
PYTHONPATH=$PWD python experiments/train_ml_20d.py \
    --input data/training_dataset_20d_v3.csv \
    --output-model models/model_20d_v3.pkl
```

2. Commit new model:
```bash
git add models/model_20d_v3.pkl
git commit -m "Update ML model v3 with improved scoring policy"
git push
```

3. Streamlit Cloud automatically redeploys (2-5 min)

### Update Code Online
1. Make changes locally
2. Test thoroughly
3. Commit and push:
```bash
git add -A
git commit -m "Feature: description"
git push
```
4. Streamlit Cloud auto-redeploys (3-5 min)

---

## Monitoring & Maintenance

### Health Checks
1. **Weekly**: Verify app loads in < 10s
2. **Weekly**: Check ML model predictions make sense
3. **Monthly**: Review logs for errors
4. **Monthly**: Update dependencies

### Update Dependencies
```bash
# Check for updates
pip list --outdated

# Update requirements.txt
pip-audit

# Commit changes
git add requirements.txt
git commit -m "Update dependencies"
git push
```

---

## Cost & Limits

### Streamlit Cloud Free Plan
- ✅ Public apps: Unlimited
- ✅ Private apps: Up to 3
- ✅ Compute: Shared (~1 GB RAM)
- ✅ Bandwidth: Unlimited
- ⚠️ Runtime: App restarts daily or when inactive

### Optimization for Free Tier
- Use precomputed scans (faster loading)
- Limit universe to 50-100 tickers
- Enable caching (1 hour TTL)
- Avoid large data downloads

---

## Advanced: Custom Domain

To use custom domain (e.g., stock-scout.example.com):

1. Update DNS CNAME to `cname.streamlitapp.com`
2. Configure in Streamlit Cloud settings
3. See docs: streamlit.io/docs/streamlit-cloud

---

## Support & Resources

- **Streamlit Docs**: streamlit.io/docs
- **Stock Scout Repo**: github.com/YOUR_USERNAME/stock-scout-2
- **Streamlit Community**: discuss.streamlit.io
- **Report Issues**: github.com/YOUR_USERNAME/stock-scout-2/issues

---

**Last Updated**: December 25, 2024  
**Status**: ✅ Ready for Online Deployment
