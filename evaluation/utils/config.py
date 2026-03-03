"""
Evaluation Configuration

Centralized configuration for evaluation pipelines.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EvalConfig:
    """Configuration for evaluation pipeline."""
    
    # Model settings
    model_path: str = ""
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # vLLM server
    vllm_port: int = 8000
    vllm_host: str = "localhost"
    tensor_parallel: int = 4
    gpu_memory_utilization: float = 0.9
    
    # Inference parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = -1
    timeout: int = 120
    
    # Judge API (for evaluation)
    judge_model: str = "gpt-5-nano"
    judge_api_key: str = field(default_factory=lambda: os.environ.get("JUDGE_API_KEY", ""))
    judge_api_base: str = field(default_factory=lambda: os.environ.get("JUDGE_API_BASE_URL", "https://api.openai.com/v1"))
    
    # Processing
    num_workers: int = 16
    batch_size: int = 1
    skip_existing: bool = True
    
    # Paths
    output_dir: str = "outputs"
    data_dir: str = "data"
    
    # Image processing
    max_image_pixels: int = 512 * 512
    min_image_pixels: int = 338 * 338
    
    @property
    def vllm_url(self) -> str:
        return f"http://{self.vllm_host}:{self.vllm_port}/v1/chat/completions"
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
