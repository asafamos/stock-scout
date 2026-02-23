#!/usr/bin/env python3
"""
Legacy direct pipeline runner script (not a pytest test).
"""
import logging
from core.config import get_config
from core.pipeline_runner import run_scan
from core.serialization import scanresult_to_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = get_config()
config_dict = config.__dict__ if hasattr(config, '__dict__') else config

test_universe = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
logger.info(f"Running pipeline on: {test_universe}")

sr = run_scan(universe=test_universe, config=config_dict)
results_df = scanresult_to_dataframe(sr)
print(f"Returned rows: {len(results_df)}")
