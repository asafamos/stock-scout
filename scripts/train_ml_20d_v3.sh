#!/bin/bash
# End-to-end pipeline for ML 20d model v3 training
# This script:
# 1. Generates training dataset with enriched features
# 2. Trains GradientBoostingClassifier
# 3. Validates model performance
# 4. Saves feature importance analysis

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================"
echo "ML 20d Model v3 Training Pipeline"
echo "======================================"
echo ""

# Configuration
START_DATE="${START_DATE:-2023-01-01}"
END_DATE="${END_DATE:-2025-01-15}"
TICKERS="${TICKERS:-}"  # Optional: comma-separated list of tickers to focus on
DATASET_PATH="data/training_dataset_20d_v3.csv"
MODEL_PATH="models/model_20d_v3.pkl"
VALIDATION_REPORT="reports/ml_20d_v3_validation_report.txt"

echo "[1/4] Generating training dataset..."
echo "  - Date range: $START_DATE to $END_DATE"
if [ -n "$TICKERS" ]; then
    echo "  - Tickers: $TICKERS (explicit list)"
    TICKER_ARG="--tickers $TICKERS"
else
    echo "  - Tickers: S&P 500 universe (default)"
    TICKER_ARG=""
fi
echo "  - Output: $DATASET_PATH"
echo ""

python experiments/offline_recommendation_audit.py \
    --mode dataset \
    --start "$START_DATE" \
    --end "$END_DATE" \
    $TICKER_ARG \
    --output "$DATASET_PATH" \
    --drop-neutral

echo ""
echo "[2/4] Training GradientBoostingClassifier..."
echo "  - Model: GradientBoosting (200 estimators, depth=5)"
echo "  - Features: 26 (technical + returns + patterns + RS + volatility)"
echo "  - Output: $MODEL_PATH"
echo ""

python experiments/train_ml_20d.py \
    --input "$DATASET_PATH" \
    --output-model "$MODEL_PATH" \
    --min-return 0.15

echo ""
echo "[3/4] Validating model performance..."
echo "  - Output: $VALIDATION_REPORT"
echo ""

python experiments/validate_ml_improvements.py \
    --input "$DATASET_PATH" \
    --model-path "$MODEL_PATH" \
    > "$VALIDATION_REPORT"

echo ""
echo "[4/4] Pipeline complete!"
echo ""
echo "Outputs:"
echo "  - Training dataset: $DATASET_PATH"
echo "  - Trained model: $MODEL_PATH"
echo "  - Feature importance: ${MODEL_PATH%.pkl}_feature_importance.csv"
echo "  - Validation report: $VALIDATION_REPORT"
echo ""
echo "Quick validation summary:"
tail -n 30 "$VALIDATION_REPORT"
echo ""
echo "======================================"
echo "To use the new model in stock_scout.py,"
echo "update MODEL_PATH in core/config.py or"
echo "set environment variable:"
echo "  export ML_MODEL_PATH=$MODEL_PATH"
echo "======================================"
