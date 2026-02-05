#!/usr/bin/env python
"""Debug why core.filters imports fail"""

import sys
import os
import traceback

# Test 1: Can we import core.unified_logic.apply_technical_filters directly?
print("[TEST 1] Import apply_technical_filters from core.unified_logic...")
try:
    from core.unified_logic import apply_technical_filters
    print("✅ SUCCESS: apply_technical_filters imported from core.unified_logic")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()

# Test 2: Can we import core.filters?
print("\n[TEST 2] Import core.filters...")
try:
    import core.filters as filters_module
    print("✅ SUCCESS: core.filters imported")
    print(f"   Available: {dir(filters_module)}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()

# Test 3: Can we import apply_technical_filters from core.filters?
print("\n[TEST 3] Import apply_technical_filters from core.filters...")
try:
    from core.filters import apply_technical_filters
    print("✅ SUCCESS: apply_technical_filters imported from core.filters")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()

# Test 4: Can we import from advanced_filters directly?
print("\n[TEST 4] Import from advanced_filters...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from advanced_filters import compute_advanced_score, should_reject_ticker, fetch_benchmark_data
    print("✅ SUCCESS: imported from advanced_filters")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()

# Test 5: Try pipeline_runner imports
print("\n[TEST 5] Import pipeline_runner...")
try:
    from core.pipeline_runner import run_scan_pipeline
    print("✅ SUCCESS: pipeline_runner imported")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()
