#!/bin/bash
# =============================================================================
# Single-turn Safety Evaluation Script
# =============================================================================

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "SaFeR-Steer Single-turn Evaluation"
echo "============================================"

# Configuration
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/single_turn}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
BENCHMARKS="${BENCHMARKS:-mmsafety,vlguard,vlsbench}"

# Judge API
export JUDGE_API_KEY="${JUDGE_API_KEY:?Error: JUDGE_API_KEY not set}"
export JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-https://api.openai.com/v1}"

# vLLM settings
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-localhost}"

echo "Model: $MODEL_PATH"
echo "Benchmarks: $BENCHMARKS"
echo "Output: $OUTPUT_DIR"
echo ""

# Run evaluation
python -m evaluation.single_turn.runner \
    --benchmarks "$BENCHMARKS" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --vllm_port "$VLLM_PORT" \
    --vllm_host "$VLLM_HOST" \
    --stages "infer,eval,aggregate"

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "Results: $OUTPUT_DIR"
echo "============================================"
