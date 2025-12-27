# 🚀 Streamlit Cloud Deployment - Quick Fix Guide

## ⚠️ Problem: App not showing recommendations

If your Stock Scout app hangs or shows "No stocks currently pass..." on Streamlit Cloud but works locally, follow these steps:

---

## Step 1: Configure Secrets Properly

The app needs API keys configured in **Streamlit Cloud Secrets**.

### Navigate to Secrets:
1. Go to your Streamlit Cloud dashboard
2. Click on your app (`stock-scout-2`)
3. Click **Settings** → **Secrets**

### Add secrets in TOML format:

```toml
# API Keys (flat structure - NO [sections])
OPENAI_API_KEY = "sk-proj-..."
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_KEY"
FINNHUB_API_KEY = "YOUR_FINNHUB_KEY"
POLYGON_API_KEY = "YOUR_POLYGON_KEY"
TIINGO_API_KEY = "YOUR_TIINGO_KEY"
FMP_API_KEY = "YOUR_FMP_KEY"
```

**IMPORTANT:**
- Use **flat structure** (no `[api_keys]` section) - keys directly at top level
- Each key on a **single line** (no line breaks in the key value)
- Keys must be **exactly** as shown above (case-sensitive)
- Use **double quotes** around values

---

## Step 2: Verify Secrets Work

After configuring secrets, run the **debug script**:

1. In your Streamlit Cloud app, navigate to: `debug_secrets.py`
2. Check the output - all keys should show ✅ with `len>0`

If you see ❌ MISSING:
- Double-check secret names (must match exactly)
- Ensure no extra spaces or quotes
- Try removing and re-adding the secret

---

## Step 3: Clear Cache & Restart

After configuring secrets:

1. In Streamlit Cloud settings → **Reboot app**
2. Or click the menu (⋮) → **Clear cache** → **Reboot**

This ensures:
- Old cached data is cleared
- New secrets are loaded properly
- Beta calculations don't hang

---

## Step 4: Test Beta Calculation

Run `test_beta_timeout.py` to verify:
- Beta calculation completes within 10 seconds
- Invalid tickers return NaN (not hang)

If it still hangs:
- Check if yfinance is rate-limited
- Verify `concurrent.futures` is available in cloud

---

## Common Issues

### Issue: "Cannot list secrets"
**Cause:** Streamlit Cloud restricts direct secret enumeration  
**Solution:** Use direct key access (`st.secrets["KEY"]`) - already implemented in `_env()`

### Issue: Keys show len=0 but secrets exist
**Cause:** Keys may be in a nested section (e.g., `[api_keys]`)  
**Solution:** Move all keys to **top level** (no sections)

### Issue: App hangs on "running fetch_beta_vs_benchmark"
**Cause:** Old timeout method (signal.SIGALRM) doesn't work in containers  
**Solution:** Already fixed in commit ea3c811 - use ThreadPoolExecutor timeout

### Issue: "No stocks currently pass the threshold"
**Possible causes:**
1. All API keys missing → can't fetch data → no stocks pass filters
2. Beta filter blocks all stocks → disable beta filter temporarily
3. Risk gate too strict → lower risk thresholds in CONFIG

**Debug steps:**
- Check `debug_secrets.py` - verify all keys work
- Set `CONFIG['BETA_FILTER_ENABLED'] = False` temporarily
- Check logs for errors during universe building

---

## Minimal Working Configuration

If you want the app to work with **minimal API keys**:

```toml
# Minimum required (Stock Scout v2)
OPENAI_API_KEY = "sk-proj-..."
ALPHA_VANTAGE_API_KEY = "YOUR_KEY"
FINNHUB_API_KEY = "YOUR_KEY"
```

Optional but recommended:
- `POLYGON_API_KEY` - for price verification
- `TIINGO_API_KEY` - additional fundamentals
- `FMP_API_KEY` - primary fundamentals source

---

## After Deployment

Once secrets are configured and app is running:

1. **Check Data Sources table** - should show ✅ for configured providers
2. **Wait for beta calculation** - first run takes ~30-60 seconds (then cached for 6 hours)
3. **View recommendations** - should see 3-15 stock cards with:
   - Price data
   - Technical indicators (RSI, MACD, etc.)
   - Fundamental scores (if keys configured)
   - ML probability scores
   - Buy recommendations with target prices

---

## Need Help?

If issues persist after following this guide:

1. **Check Streamlit Cloud logs:**
   - Settings → Logs
   - Look for errors related to API keys, timeouts, or data fetching

2. **Run diagnostic scripts locally:**
   ```bash
   streamlit run debug_secrets.py
   streamlit run test_beta_timeout.py
   ```

3. **Verify commit is deployed:**
   - Check that commit `ea3c811` or later is deployed
   - Settings → App settings → Branch: `main`

4. **Contact support:**
   - Include screenshots of:
     - Secrets configuration
     - Error logs
     - `debug_secrets.py` output

---

## Summary Checklist

- [ ] Secrets configured in Streamlit Cloud (flat TOML format)
- [ ] All required API keys present (at minimum: OpenAI, Alpha, Finnhub)
- [ ] App rebooted after adding secrets
- [ ] `debug_secrets.py` shows ✅ for all keys
- [ ] `test_beta_timeout.py` completes successfully
- [ ] Data Sources table shows ✅ for configured providers
- [ ] Recommendations appear (3-15 cards)

If all checkboxes are ✅ - deployment is successful! 🎉
