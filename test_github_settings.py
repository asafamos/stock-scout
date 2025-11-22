#!/usr/bin/env python3
"""
Test script to simulate GitHub/Streamlit Cloud settings locally.
Run this to verify the app works with production configuration.
"""
import os
import sys

# Override environment variables to match GitHub deployment
os.environ['UNIVERSE_LIMIT'] = '40'
os.environ['LOOKBACK_DAYS'] = '90'
os.environ['SMART_SCAN'] = 'true'
os.environ['TOPK_RECOMMEND'] = '5'
os.environ['TOPN_RESULTS'] = '15'

print("=" * 60)
print("🧪 Testing with GitHub/Streamlit Cloud settings")
print("=" * 60)
print(f"UNIVERSE_LIMIT    = {os.environ['UNIVERSE_LIMIT']}")
print(f"LOOKBACK_DAYS     = {os.environ['LOOKBACK_DAYS']}")
print(f"SMART_SCAN        = {os.environ['SMART_SCAN']}")
print(f"TOPK_RECOMMEND    = {os.environ['TOPK_RECOMMEND']}")
print(f"TOPN_RESULTS      = {os.environ['TOPN_RESULTS']}")
print("=" * 60)
print()

# Now import and test config
from core.config import get_config

cfg = get_config()
print("✅ Configuration loaded successfully!")
print(f"   universe_limit    = {cfg.universe_limit}")
print(f"   lookback_days     = {cfg.lookback_days}")
print(f"   smart_scan        = {cfg.smart_scan}")
print(f"   topk_recommend    = {cfg.topk_recommend}")
print(f"   topn_results      = {cfg.topn_results}")
print()

# Verify values match
issues = []
if cfg.universe_limit != 40:
    issues.append(f"❌ UNIVERSE_LIMIT should be 40, got {cfg.universe_limit}")
if cfg.lookback_days != 90:
    issues.append(f"❌ LOOKBACK_DAYS should be 90, got {cfg.lookback_days}")
if cfg.smart_scan != True:
    issues.append(f"❌ SMART_SCAN should be True, got {cfg.smart_scan}")
if cfg.topk_recommend != 5:
    issues.append(f"❌ TOPK_RECOMMEND should be 5, got {cfg.topk_recommend}")

if issues:
    print("⚠️  Configuration Issues:")
    for issue in issues:
        print(f"   {issue}")
    print()
    print("💡 This suggests environment variables are not being read correctly.")
    print("   Check core/config.py _get_config_value() function.")
    sys.exit(1)
else:
    print("✅ All configuration values match expected GitHub settings!")
    print()
    print("🚀 Ready to test with:")
    print("   streamlit run stock_scout.py")
    print()
    print("Expected behavior:")
    print("   • Should scan exactly 40 stocks (balanced)")
    print("   • Should complete in 60-90 seconds")
    print("   • Should produce 3-5+ recommendations")
    print("   • Should show: Config: Universe=40 | Lookback=90d | Smart=True")
    sys.exit(0)
