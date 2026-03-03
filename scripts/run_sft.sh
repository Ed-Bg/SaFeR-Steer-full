#!/bin/bash
# =============================================================================
# SaFeR-Steer SFT Training Script
# Stage II: Synthetic Bootstrapping
# =============================================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "SaFeR-Steer SFT Training (Stage II)"
echo "============================================"

# Check LLaMA-Factory installation
if ! command -v llamafactory-cli &> /dev/null; then
    echo "LLaMA-Factory not found. Installing..."
    pip install llamafactory
fi

# Configuration file
CONFIG_FILE="$PROJECT_DIR/training/sft/llamafactory_config.yaml"

echo "Config: $CONFIG_FILE"
echo ""

# Run training
llamafactory-cli train "$CONFIG_FILE"

echo ""
echo "============================================"
echo "SFT Training completed!"
echo "============================================"
