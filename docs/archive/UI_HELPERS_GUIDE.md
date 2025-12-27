# UI Helpers Developer Guide

Quick reference for using the modernized UI components in Stock Scout.

## StatusManager

Centralized pipeline progress tracking with clear stage progression.

### Basic Usage

```python
from core.ui_helpers import StatusManager, get_pipeline_stages

# Initialize at app startup
status_manager = StatusManager(get_pipeline_stages())

# Advance to next stage with summary
status_manager.advance("Universe: 100 tickers loaded")

# Update detail within current stage
status_manager.update_detail("Processing ticker 50/100...")

# Mark pipeline complete
status_manager.complete("✅ Pipeline complete: 85 recommendations")
```

### API Reference

#### `__init__(stages: list[str])`
Initialize status manager with ordered list of stage names.

```python
stages = [
    "Market Regime Detection",
    "Universe Building",
    "Historical Data Fetch",
    # ... more stages
]
status_manager = StatusManager(stages)
```

#### `advance(detail: str = "")`
Move to next stage, optionally with detail message.

```python
# Just advance stage
status_manager.advance()

# Advance with summary
status_manager.advance(f"Technical scoring: {len(results)} stocks (12.3s)")
```

#### `update_detail(message: str)`
Update detail message within current stage (doesn't advance).

```python
# Show incremental progress
status_manager.update_detail("Computing indicators: 50/100 (50%)")
```

#### `complete(message: str = "✅ Pipeline complete")`
Mark pipeline as fully complete.

```python
status_manager.complete(f"✅ Pipeline complete: {len(rec_df)} recommendations")
```

### Best Practices

1. **One StatusManager per pipeline:** Initialize once at startup
2. **Advance at major milestones:** Use `advance()` for significant steps
3. **Update details for progress:** Use `update_detail()` for incremental feedback
4. **Include metrics in advance:** Show counts, timing, key stats
5. **Complete at end:** Always call `complete()` when pipeline finishes

### Example: Full Pipeline

```python
from core.ui_helpers import StatusManager, get_pipeline_stages
import time

# Initialize
status_manager = StatusManager(get_pipeline_stages())

# Stage 1: Market Regime
t0 = time.perf_counter()
status_manager.update_detail("Analyzing SPY/QQQ/VIX trends...")
regime_data = detect_market_regime()
status_manager.advance(f"Market regime: {regime_data['regime'].upper()}")

# Stage 2: Universe
status_manager.update_detail("Building stock universe...")
universe = build_universe(limit=100)
status_manager.advance(f"{len(universe)} tickers in universe")

# Stage 3: Historical Data
t0 = time.perf_counter()
status_manager.update_detail(f"Fetching history for {len(universe)} stocks...")
data_map = fetch_history_bulk(universe)
elapsed = time.perf_counter() - t0
status_manager.advance(f"{len(data_map)} stocks with data ({elapsed:.1f}s)")

# ... more stages ...

# Complete
status_manager.complete(f"✅ Pipeline complete: {len(recommendations)} recommendations")
```

---

## SourcesOverview

Dynamic provider tracking with real-time usage visibility.

### Basic Usage

```python
from core.ui_helpers import SourcesOverview

# Initialize
sources_overview = SourcesOverview()

# Render initial status
sources_overview.render(show_legend=True)

# Track usage during pipeline
sources_overview.mark_usage("Yahoo", "price")
sources_overview.mark_usage("Alpha Vantage", "fundamentals")

# Check for issues
warning = sources_overview.check_critical_missing()
if warning:
    st.error(warning)

# Render final summary
sources_overview.render(show_legend=True)
```

### API Reference

#### `__init__()`
Initialize sources tracker. No arguments needed.

```python
sources_overview = SourcesOverview()
```

#### `mark_usage(provider: str, category: str)`
Record that a provider was used for a specific category.

**Categories:** `"price"`, `"fundamentals"`, `"ml"`

```python
sources_overview.mark_usage("Yahoo", "price")
sources_overview.mark_usage("Alpha Vantage", "fundamentals")
sources_overview.mark_usage("OpenAI", "ml")
```

#### `render(show_legend: bool = True)`
Display markdown table with current provider status.

**Status indicators:**
- 🟢 Used this run
- 🟡 Available but not yet used
- ⚫ Missing key or not connected

```python
# Initial render (shows available)
sources_overview.render(show_legend=True)

# After pipeline (shows actual usage)
sources_overview.render(show_legend=True)
```

#### `get_active_providers() -> Dict[str, list[str]]`
Get lists of active providers by category.

```python
active = sources_overview.get_active_providers()
# Returns: {
#   "price": ["Yahoo", "Alpha Vantage", "Polygon"],
#   "fundamentals": ["Alpha Vantage", "Finnhub"],
#   "ml": ["OpenAI"]
# }
```

#### `check_critical_missing() -> Optional[str]`
Check if critical providers are missing. Returns warning message or None.

```python
warning = sources_overview.check_critical_missing()
if warning:
    st.error(warning)
```

### Adding New Providers

Edit `PROVIDERS` dict in `core/ui_helpers.py`:

```python
PROVIDERS = {
    "Yahoo": {"roles": {"price"}, "keys": []},  # No key needed
    "Alpha Vantage": {"roles": {"price", "fundamentals"}, "keys": ["ALPHA_VANTAGE_API_KEY"]},
    "NewProvider": {"roles": {"fundamentals"}, "keys": ["NEW_PROVIDER_KEY"]},
}
```

### Session State Integration

Sources overview checks `st.session_state` for provider status:

```python
# Set provider status (usually in main app)
st.session_state["_alpha_vantage_ok"] = True
st.session_state["_finnhub_ok"] = False

# SourcesOverview automatically checks these flags
sources_overview = SourcesOverview()
sources_overview.render()  # Alpha shows green, Finnhub shows gray
```

---

## Debug Helpers

Clean separation of production and development information.

### `create_debug_expander()`

Show debug info only when debug mode is enabled.

```python
from core.ui_helpers import create_debug_expander

# Only visible if DEBUG_MODE=True or st.session_state["debug_mode"]
create_debug_expander({
    "regime": "bullish",
    "confidence": 85,
    "spy_trend": 0.023,
    "qqq_trend": 0.031,
}, title="📊 Market Regime Details")
```

**Enabling debug mode:**

```python
# Environment variable
os.environ["DEBUG_MODE"] = "true"

# Or session state
st.session_state["debug_mode"] = True
```

### `show_config_summary()`

Display compact configuration summary.

```python
from core.ui_helpers import show_config_summary

show_config_summary(CONFIG)
# Output: ⚙️ Config: Universe=100 • Lookback=252d • Smart Scan=✅
```

### `get_pipeline_stages()`

Get ordered list of standard pipeline stages.

```python
from core.ui_helpers import get_pipeline_stages

stages = get_pipeline_stages()
# Returns: [
#   "Market Regime Detection",
#   "Universe Building",
#   "Historical Data Fetch",
#   ... 7 more
# ]
```

---

## Integration Examples

### Complete Pipeline Integration

```python
from core.ui_helpers import (
    StatusManager,
    SourcesOverview,
    get_pipeline_stages,
    show_config_summary,
    create_debug_expander,
)

# 1. Initialize
st.title("📈 Stock Scout — 2025")
show_config_summary(CONFIG)

status_manager = StatusManager(get_pipeline_stages())
sources_overview = SourcesOverview()

# 2. Show initial sources
sources_overview.render(show_legend=True)

# 3. Execute pipeline with tracking
status_manager.update_detail("Detecting market regime...")
regime_data = detect_market_regime()
status_manager.advance(f"Market regime: {regime_data['regime']}")

# Optional debug info
create_debug_expander(regime_data, title="📊 Regime Details")

# 4. Continue pipeline
status_manager.update_detail("Building universe...")
universe = build_universe(limit=100)
sources_overview.mark_usage("Yahoo", "price")
status_manager.advance(f"{len(universe)} tickers")

# ... more stages ...

# 5. Complete
status_manager.complete(f"✅ Complete: {len(results)} recommendations")
sources_overview.render(show_legend=True)
```

### Error Handling

```python
try:
    status_manager.update_detail("Fetching fundamentals...")
    results = fetch_fundamentals_batch(tickers)
    sources_overview.mark_usage("Alpha Vantage", "fundamentals")
    status_manager.advance(f"Fundamentals: {len(results)} enriched")
except Exception as e:
    status_manager.update_detail(f"⚠️ Fundamentals fetch failed: {str(e)}")
    # Continue with technical-only scoring
```

### Conditional Provider Usage

```python
# Check active providers
active = sources_overview.get_active_providers()

if "Alpha Vantage" in active["fundamentals"]:
    status_manager.update_detail("Fetching from Alpha Vantage...")
    data = fetch_alpha_fundamentals(ticker)
    sources_overview.mark_usage("Alpha Vantage", "fundamentals")
elif "Finnhub" in active["fundamentals"]:
    status_manager.update_detail("Falling back to Finnhub...")
    data = fetch_finnhub_fundamentals(ticker)
    sources_overview.mark_usage("Finnhub", "fundamentals")
else:
    status_manager.update_detail("⚠️ No fundamental sources available")
```

---

## Testing

### Unit Tests

```python
from unittest.mock import MagicMock, patch
from core.ui_helpers import StatusManager, SourcesOverview

def test_status_manager():
    with patch('streamlit.progress') as mock_progress, \
         patch('streamlit.empty') as mock_empty:
        
        mock_progress.return_value = MagicMock()
        mock_empty.return_value = MagicMock()
        
        manager = StatusManager(["Stage 1", "Stage 2"])
        manager.advance("Stage 1 complete")
        
        assert manager.current_stage == 1

def test_sources_overview():
    with patch('streamlit.empty'):
        sources = SourcesOverview()
        sources.mark_usage("Yahoo", "price")
        
        assert "Yahoo" in sources._usage
        assert "price" in sources._usage["Yahoo"]
```

### Integration Tests

```python
def test_full_pipeline_with_ui_helpers():
    status_manager = StatusManager(["Build", "Process", "Complete"])
    sources_overview = SourcesOverview()
    
    # Simulate pipeline
    status_manager.advance("Universe built")
    sources_overview.mark_usage("Yahoo", "price")
    
    status_manager.advance("Processing complete")
    status_manager.complete("All done")
    
    assert status_manager.current_stage == 2
    assert len(sources_overview._usage) == 1
```

---

## Migration Guide

### Replacing Scattered Status Calls

**Before:**
```python
st.success(f"Universe built: {len(universe)} tickers")
# ... 100 lines later ...
st.info("Fetching fundamentals...")
# ... 50 lines later ...
st.warning("Some stocks filtered out")
```

**After:**
```python
status_manager.advance(f"{len(universe)} tickers in universe")
# ... 100 lines later ...
status_manager.update_detail("Fetching fundamentals...")
# ... 50 lines later ...
status_manager.update_detail("⚠️ Some stocks filtered out")
```

### Replacing Provider Tracking

**Before:**
```python
usage = st.session_state.setdefault("provider_usage", {})
usage.setdefault("Alpha", set()).add("fundamentals")
```

**After:**
```python
sources_overview.mark_usage("Alpha Vantage", "fundamentals")
```

### Replacing Debug Output

**Before:**
```python
if CONFIG.get("DEBUG_MODE"):
    st.write(f"Debug info: {debug_data}")
```

**After:**
```python
create_debug_expander(debug_data, title="🔧 Debug Info")
```

---

## Troubleshooting

### Progress not updating
- Ensure `status_manager.advance()` is called at stage boundaries
- Check that stage list has enough stages for your pipeline

### Sources showing incorrect status
- Verify environment variables are set correctly
- Check session state flags: `st.session_state["_provider_ok"]`
- Ensure `mark_usage()` is called after actual provider use

### Debug info always visible
- Check `os.getenv("DEBUG_MODE")` returns "true" or "1"
- Verify `st.session_state.get("debug_mode")` is not accidentally set

### Tests failing with missing st.session_state
- Mock streamlit components properly:
  ```python
  with patch('streamlit.session_state', new_callable=dict):
      # test code
  ```

---

## Best Practices Summary

1. **Initialize once:** Create StatusManager and SourcesOverview at app startup
2. **Advance at milestones:** Use `advance()` for major steps, `update_detail()` for progress
3. **Track all usage:** Call `mark_usage()` whenever a provider is actually used
4. **Complete the pipeline:** Always call `status_manager.complete()` at end
5. **Use debug helpers:** Keep production UI clean with `create_debug_expander()`
6. **Include metrics:** Show counts, timing, and key stats in status messages
7. **Test thoroughly:** Write unit tests for UI components
8. **Document changes:** Update this guide when adding new providers or stages

---

## Additional Resources

- **Implementation:** `core/ui_helpers.py`
- **Tests:** `tests/test_ui_helpers.py`
- **Usage example:** `stock_scout.py` (search for "StatusManager" or "SourcesOverview")
- **Summary:** `UI_MODERNIZATION_SUMMARY.md`
- **Comparison:** `UI_BEFORE_AFTER.md`
