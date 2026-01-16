#!/usr/bin/env bash
set -euo pipefail

echo "=== Stock Scout Full Cycle (Local & Cloud) ==="
start_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Start: ${start_ts}"

# 1) Train Polygon-based ML model (20d)
echo "[1/2] Training ML model with Polygon data..."
python scripts/train_rolling_ml_20d.py

# 2) Run production scan pipeline (headless)
# Uses core/pipeline_runner.py main(), which fetches universe and runs the pipeline.
# Fresh results are saved to data/latest_scan_live.json & data/latest_scan_live.parquet.
echo "[2/2] Running unified scan pipeline (headless)..."
python -m core.pipeline_runner

end_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "End: ${end_ts}"
echo "Cycle Complete"
