# run on 4xGPU (CUDA 4,5,6,7)
# make sure your current working directory is the root of the project
#     algorithm.norm_adv_by_std_in_grpo=False \
set -x

ulimit -n 65535

# export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
OFFLOAD=${OFFLOAD:-False}

export CUDA_VISIBLE_DEVICES=4,5,6,7

# ======== wandb 设置 ========
export WANDB_API_KEY="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"  # 或 wandb login
export WANDB_ENTITY="${WANDB_ENTITY:-harlan_h-huazhong-university-of-science-and-technology}"                    # 你的团队/用户名
export WANDB_MODE="${WANDB_MODE:-online}"                  # online/offline/disabled
export WANDB_NOTES="SafeDy multi-turn RL training 3b a0"
export WANDB_TAGS="safedy,multi-turn,grpo"
# ===========================

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='saferdy_multiturn_grpo_w_interaction' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$((1024 * 1)) \
    data.max_response_length=$((1024 * 6)) \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0.1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='saferdy_async_rl' \
    trainer.experiment_name='qwen2.5-3b_saferdy-sgl-multi-w-interaction-n4a0-newprompt' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    data.train_files=$PROJECT_DIR/data/safedy_multiturn/train.parquet \
    data.val_files=$PROJECT_DIR/data/safedy_multiturn/test.parquet \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/saferdy_interaction_config.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    "$@"
    

