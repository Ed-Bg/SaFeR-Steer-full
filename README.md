# SaFeR-Steer

**Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics**

> 🎉 **ICML 2026**
>
> 📄 [Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX) | 🤗 [Datasets (Coming Soon)](#datasets)
> 
> This is the complete codebase for reproducibility. For the anonymous review version, see the `SaFeR-Steer/` directory.

## Overview

SAFER-STEER is a progressive multi-turn alignment framework that combines staged synthetic bootstrapping with tutor-in-the-loop GRPO to train a single student under adaptive, on-policy attacks. We also introduce **TCSR** (Trajectory-Consistent Safety Reward), which uses trajectory minimum/average safety to propagate late-turn failures to earlier turns.

The framework consists of three stages:

- **Stage I**: Intent Decomposition and Reconstruction — multi-turn data synthesis with Red Agent (gpt4o-mini), Blue Agent (Gemini-3-Pro), and Dynamic Planner
- **Stage II**: Synthetic Bootstrapping — SFT on STEER-SFT (12,934 dialogues)
- **Stage III**: Tutor-in-the-loop Agentic RL with Feedback Dynamics — GRPO + TCSR with Safety Tutor (Qwen3-VL-32B)

## Repository Structure

```
SaFeR-Steer-Full/
├── data_construction/           # Stage I: Multi-turn data synthesis
│   ├── pipeline.py             # Main data generation pipeline
│   └── prompts/                # All prompt templates
│       ├── general_query.py    # Benign query generation
│       └── progressive_disclosure.py  # Medium-risk generation
│
├── evaluation/                  # Multi-turn Safety Evaluation
│   ├── run_all.py              # Main evaluation pipeline
│   ├── multi_turn/
│   │   ├── infer.py            # Multi-turn inference
│   │   ├── evaluate.py         # Judge-based evaluation
│   │   ├── aggregate.py        # Results aggregation
│   │   └── prompts.py          # Judge prompts
│   └── utils/                  # Utility functions
│
├── training/
│   ├── sft/                    # Stage II: SFT with LLaMA-Factory
│   │   └── llamafactory_config.yaml
│   └── grpo/                   # Stage III: GRPO with verl
│       └── reward_prompt.py    # Safety Tutor prompt
│
├── verl/                        # Modified verl framework for multi-turn GRPO
│   ├── interactions/           # Safety Tutor interaction module
│   ├── utils/reward_score/     # TCSR reward aggregation
│   ├── trainer/                # GRPO trainer
│   └── workers/                # Distributed workers
│
├── examples/                    # Training examples
│   ├── sglang_multiturn/       # Multi-turn GRPO configs
│   └── data_preprocess/        # Data preprocessing scripts
│
├── scripts/                     # Run scripts
├── configs/                     # Model and benchmark configs
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/SaFeR-Steer.git
cd SaFeR-Steer-Full

# Install dependencies
pip install -r requirements.txt

# Install verl framework
pip install -e .
```

## Quick Start

### Stage I: Data Construction

```bash
# Generate multi-turn training data
cd data_construction

# Configure API endpoint in pipeline.py
# API_KEY = "your-key"
# BASE_URL = "http://localhost:8000/v1"

python pipeline.py
```

### Stage II: SFT Training

```bash
# Using LLaMA-Factory
pip install llamafactory

llamafactory-cli train training/sft/llamafactory_config.yaml
```

### Stage III: GRPO Training

```bash
# Using verl framework
cd examples/sglang_multiturn

# Edit config to point to your SFT checkpoint
bash run_qwen2.5-7b_safer_multiturn_w_interaction.sh
```

### Evaluation

```bash
# Set Judge API key
export JUDGE_API_KEY="your-openai-key"
export JUDGE_API_BASE_URL="https://api.openai.com/v1"

# Run evaluation
python -m evaluation.run_all \
    --model /path/to/checkpoint \
    --benchmarks steer_bench,steer_beaver,steer_mmsafe \
    --output_dir outputs/
```

## Hyperparameters

### SFT (Stage II) - Table 5

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 |
| Batch size | 8 |
| Epochs | 2 |
| Warmup ratio | 0.1 |
| Dataset size | 12,934 |

### GRPO (Stage III) - Table 5

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-6 |
| Batch size | 64 |
| Mini-batch size | 2 |
| Rollouts per prompt (K) | 5 |
| KL penalty (β) | 0.1 |
| TCSR α | 0.3 |
| Dataset size | 2,000 |

## Datasets

We release **STEER**, a multi-turn multimodal safety dataset spanning 2–10 turns:

| Dataset | Size | Avg Turns | Usage |
|---------|------|-----------|-------|
| **STEER-SFT** | 12,934 | 6.35 | SFT training |
| **STEER-RL** | 2,000 | 8.33 | GRPO training |
| **STEER-BENCH** | 3,227 | 8.55 | Held-out evaluation |

**STEER-BENCH** consists of 5 subsets:

| Subset | Source |
|--------|--------|
| STEER-Beaver | BeaverTails-V |
| STEER-MMSafe | MM-SafetyBench |
| STEER-VLS | VLSBench |
| STEER-SPA | SPA-VL |
| STEER-DyS | Original dynamic scenarios |

Download links: [Coming Soon]

## Citation

```bibtex
@inproceedings{safersteer2026,
  title={SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics},
  author={...},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

Apache License 2.0

## Acknowledgments

- [verl](https://github.com/volcengine/verl) - GRPO training framework
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - SFT training
- [vLLM](https://github.com/vllm-project/vllm) - Fast inference
