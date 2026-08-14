#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/default.yaml}"
FEATURE_SETS_PATH="${2:-configs/feature_sets.json}"

echo "============================================"
echo "Retroviral Wall Challenge - Inference"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Config:       $CONFIG_PATH"
echo "Feature sets: $FEATURE_SETS_PATH"
echo

python -m src.inference \
    --config "$CONFIG_PATH" \
    --feature-sets "$FEATURE_SETS_PATH"

echo
echo "Inference completed successfully."
