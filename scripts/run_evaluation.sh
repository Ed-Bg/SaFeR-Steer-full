#!/bin/bash
# =============================================================================
# SaFeR-Steer Evaluation Script
# Multi-turn Safety Benchmark Evaluation
# =============================================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_CUMEM_ENABLE=0
export TRANSFORMERS_OFFLINE=0

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "SaFeR-Steer Multi-turn Evaluation"
echo "============================================"

# Configuration
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/eval}"
BENCHMARKS="${BENCHMARKS:-steer_bench,steer_beaver,steer_mmsafe,steer_vls,steer_spa,steer_dys}"

# Judge API
export JUDGE_API_KEY="${JUDGE_API_KEY:?Error: JUDGE_API_KEY not set}"
export JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-https://api.openai.com/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"

# vLLM Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"

echo "Model: $MODEL_PATH"
echo "Benchmarks: $BENCHMARKS"
echo "Judge: $JUDGE_MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start vLLM server
echo "[1/4] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --trust-remote-code &

VLLM_PID=$!
sleep 60  # Wait for server to start

# Health check
echo "Checking vLLM server health..."
curl -s "http://localhost:$VLLM_PORT/health" || {
    echo "vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    exit 1
}

echo "[2/4] Running inference..."
python -m evaluation.run_all \
    --model "$MODEL_PATH" \
    --benchmarks "$BENCHMARKS" \
    --output_dir "$OUTPUT_DIR" \
    --port "$VLLM_PORT" \
    --mode inference

echo "[3/4] Running evaluation..."
python -m evaluation.run_all \
    --model "$MODEL_PATH" \
    --benchmarks "$BENCHMARKS" \
    --output_dir "$OUTPUT_DIR" \
    --mode evaluation

echo "[4/4] Aggregating results..."
python -m evaluation.run_all \
    --model "$MODEL_PATH" \
    --benchmarks "$BENCHMARKS" \
    --output_dir "$OUTPUT_DIR" \
    --mode aggregation

# Stop vLLM server
kill $VLLM_PID 2>/dev/null

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "Results: $OUTPUT_DIR"
echo "============================================"
