#!/bin/bash
# =============================================================================
# SaFeR-Steer Data Construction Script
# Stage I: Intent Decomposition and Reconstruction
# =============================================================================

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# API Configuration (modify as needed)
export API_KEY="EMPTY"  # For local vLLM
export BASE_URL="http://localhost:8000/v1"
export MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"

# Data paths
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/data/seeds}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/steer_sft}"

echo "============================================"
echo "SaFeR-Steer Data Construction"
echo "============================================"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run data generation for each type
echo ""
echo "[1/3] Generating Strong Red-Team data..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from data_construction.pipeline import DataConstructionPipeline
import json

pipeline = DataConstructionPipeline(
    api_key='$API_KEY',
    base_url='$BASE_URL',
    model_name='$MODEL_NAME',
    num_workers=32
)

# Load seeds
with open('$INPUT_DIR/red_team_seeds.jsonl', 'r') as f:
    seeds = [json.loads(line) for line in f]

results = pipeline.run(
    input_data=seeds,
    data_type='red_team',
    num_turns=(4, 8),
    output_path='$OUTPUT_DIR/red_team.jsonl'
)
print(f'Generated {len(results)} red-team samples')
"

echo ""
echo "[2/3] Generating Progressive Disclosure data..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from data_construction.pipeline import DataConstructionPipeline
import json

pipeline = DataConstructionPipeline(
    api_key='$API_KEY',
    base_url='$BASE_URL',
    model_name='$MODEL_NAME',
    num_workers=32
)

with open('$INPUT_DIR/obfuscated_seeds.jsonl', 'r') as f:
    seeds = [json.loads(line) for line in f]

results = pipeline.run(
    input_data=seeds,
    data_type='obfuscated',
    num_turns=(3, 6),
    output_path='$OUTPUT_DIR/progressive.jsonl'
)
print(f'Generated {len(results)} progressive disclosure samples')
"

echo ""
echo "[3/3] Generating Benign data..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from data_construction.pipeline import DataConstructionPipeline
import json

pipeline = DataConstructionPipeline(
    api_key='$API_KEY',
    base_url='$BASE_URL',
    model_name='$MODEL_NAME',
    num_workers=32
)

with open('$INPUT_DIR/benign_seeds.jsonl', 'r') as f:
    seeds = [json.loads(line) for line in f]

results = pipeline.run(
    input_data=seeds,
    data_type='benign',
    num_turns=(2, 5),
    output_path='$OUTPUT_DIR/benign.jsonl'
)
print(f'Generated {len(results)} benign samples')
"

echo ""
echo "============================================"
echo "Data construction completed!"
echo "Output files in: $OUTPUT_DIR"
echo "============================================"
