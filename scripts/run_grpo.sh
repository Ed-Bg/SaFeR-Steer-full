#!/bin/bash
# =============================================================================
# SaFeR-Steer GRPO Training Script  
# Stage III: Tutor-in-the-loop Agentic RL with Feedback Dynamics
# =============================================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CUMEM_ENABLE=0
export VLLM_ATTENTION_BACKEND=XFORMERS

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "SaFeR-Steer GRPO Training (Stage III)"
echo "============================================"

# Configuration
MODEL_PATH="${MODEL_PATH:-/path/to/sft/checkpoint}"
DATA_PATH="${DATA_PATH:-$PROJECT_DIR/data/steer_rl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/grpo}"

# Judge API for Safety Tutor
export JUDGE_API_KEY="${JUDGE_API_KEY:-your-api-key}"
export JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-https://api.openai.com/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"

echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Judge: $JUDGE_MODEL"
echo ""

# Navigate to examples
cd "$PROJECT_DIR/examples/sglang_multiturn"

# Run GRPO training
python -m verl.trainer.main \
    --config config/saferdy_multiturn_grpo_w_interaction.yaml \
    data.train_files="$DATA_PATH/train.parquet" \
    data.val_files="$DATA_PATH/test.parquet" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.kl_ctrl.kl_coef=0.1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.project_name="safer-steer-grpo" \
    trainer.experiment_name="grpo-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "============================================"
echo "GRPO Training completed!"
echo "============================================"
