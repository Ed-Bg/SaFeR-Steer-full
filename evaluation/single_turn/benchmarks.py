"""
Benchmark Implementations

Concrete implementations for each single-turn safety benchmark.
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path

from .base import SingleTurnEvaluator
from ..utils.config import EvalConfig


class MMSafetyBenchEvaluator(SingleTurnEvaluator):
    """MM-SafetyBench evaluator."""
    
    benchmark_name = "mmsafety"
    
    def __init__(self, config: EvalConfig, data_path: str = None):
        super().__init__(config)
        self.data_path = data_path or os.path.join(config.data_dir, "mm_safety_bench")
    
    def load_data(self) -> List[Dict]:
        data = []
        data_dir = Path(self.data_path)
        
        # Load from all subdirectories
        for jsonl_file in data_dir.rglob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    sample = json.loads(line)
                    sample["_source_file"] = str(jsonl_file)
                    data.append(sample)
        
        return data
    
    def get_question(self, sample: Dict) -> str:
        return sample.get("question", sample.get("prompt", ""))
    
    def get_image_path(self, sample: Dict) -> Optional[str]:
        img = sample.get("image", sample.get("image_path"))
        if img and not os.path.isabs(img):
            img = os.path.join(self.data_path, img)
        return img
    
    def get_sample_id(self, sample: Dict) -> str:
        return sample.get("id", sample.get("question_id", str(hash(self.get_question(sample)))))


class VLGuardEvaluator(SingleTurnEvaluator):
    """VLGuard benchmark evaluator."""
    
    benchmark_name = "vlguard"
    
    def __init__(self, config: EvalConfig, data_path: str = None):
        super().__init__(config)
        self.data_path = data_path or os.path.join(config.data_dir, "vlguard")
    
    def load_data(self) -> List[Dict]:
        data = []
        data_file = Path(self.data_path) / "test.jsonl"
        
        if data_file.exists():
            with open(data_file, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        
        return data
    
    def get_question(self, sample: Dict) -> str:
        return sample.get("question", sample.get("prompt", ""))
    
    def get_image_path(self, sample: Dict) -> Optional[str]:
        img = sample.get("image", sample.get("image_path"))
        if img and not os.path.isabs(img):
            img = os.path.join(self.data_path, img)
        return img
    
    def get_sample_id(self, sample: Dict) -> str:
        return sample.get("id", str(hash(self.get_question(sample))))


class VLSBenchEvaluator(SingleTurnEvaluator):
    """VLSBench evaluator."""
    
    benchmark_name = "vlsbench"
    
    def __init__(self, config: EvalConfig, data_path: str = None):
        super().__init__(config)
        self.data_path = data_path or os.path.join(config.data_dir, "vlsbench")
    
    def load_data(self) -> List[Dict]:
        data = []
        data_dir = Path(self.data_path)
        
        for jsonl_file in data_dir.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        
        return data
    
    def get_question(self, sample: Dict) -> str:
        return sample.get("question", sample.get("prompt", ""))
    
    def get_image_path(self, sample: Dict) -> Optional[str]:
        img = sample.get("image")
        if img and not os.path.isabs(img):
            img = os.path.join(self.data_path, img)
        return img
    
    def get_sample_id(self, sample: Dict) -> str:
        return sample.get("id", str(hash(self.get_question(sample))))


class SPAVLEvaluator(SingleTurnEvaluator):
    """SPA-VL benchmark evaluator."""
    
    benchmark_name = "spa_vl"
    
    def __init__(self, config: EvalConfig, data_path: str = None):
        super().__init__(config)
        self.data_path = data_path or os.path.join(config.data_dir, "spa_vl")
    
    def load_data(self) -> List[Dict]:
        data = []
        data_file = Path(self.data_path) / "test.jsonl"
        
        if data_file.exists():
            with open(data_file, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        
        return data
    
    def get_question(self, sample: Dict) -> str:
        return sample.get("question", sample.get("prompt", ""))
    
    def get_image_path(self, sample: Dict) -> Optional[str]:
        img = sample.get("image")
        if img and not os.path.isabs(img):
            img = os.path.join(self.data_path, "images", img)
        return img
    
    def get_sample_id(self, sample: Dict) -> str:
        return sample.get("id", str(hash(self.get_question(sample))))


class SIUOEvaluator(SingleTurnEvaluator):
    """SIUO benchmark evaluator."""
    
    benchmark_name = "siuo"
    
    def __init__(self, config: EvalConfig, data_path: str = None):
        super().__init__(config)
        self.data_path = data_path or os.path.join(config.data_dir, "siuo")
    
    def load_data(self) -> List[Dict]:
        data = []
        data_file = Path(self.data_path) / "test.jsonl"
        
        if data_file.exists():
            with open(data_file, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        
        return data
    
    def get_question(self, sample: Dict) -> str:
        return sample.get("question", sample.get("instruction", ""))
    
    def get_image_path(self, sample: Dict) -> Optional[str]:
        img = sample.get("image")
        if img and not os.path.isabs(img):
            img = os.path.join(self.data_path, img)
        return img
    
    def get_sample_id(self, sample: Dict) -> str:
        return sample.get("id", str(hash(self.get_question(sample))))


class BeaverTailsVEvaluator(SingleTurnEvaluator):
    """BeaverTails-V benchmark evaluator."""
    
    benchmark_name = "beavertails_v"
    
    def __init__(self, config: EvalConfig, data_path: str = None):
        super().__init__(config)
        self.data_path = data_path or os.path.join(config.data_dir, "beavertails_v")
    
    def load_data(self) -> List[Dict]:
        data = []
        data_file = Path(self.data_path) / "test.jsonl"
        
        if data_file.exists():
            with open(data_file, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        
        return data
    
    def get_question(self, sample: Dict) -> str:
        return sample.get("question", sample.get("prompt", ""))
    
    def get_image_path(self, sample: Dict) -> Optional[str]:
        img = sample.get("image")
        if img and not os.path.isabs(img):
            img = os.path.join(self.data_path, img)
        return img
    
    def get_sample_id(self, sample: Dict) -> str:
        return sample.get("id", str(hash(self.get_question(sample))))


class MSSBenchEvaluator(SingleTurnEvaluator):
    """MSSBench evaluator."""
    
    benchmark_name = "mssbench"
    
    def __init__(self, config: EvalConfig, data_path: str = None):
        super().__init__(config)
        self.data_path = data_path or os.path.join(config.data_dir, "mssbench")
    
    def load_data(self) -> List[Dict]:
        data = []
        data_file = Path(self.data_path) / "test.jsonl"
        
        if data_file.exists():
            with open(data_file, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        
        return data
    
    def get_question(self, sample: Dict) -> str:
        return sample.get("question", sample.get("prompt", ""))
    
    def get_image_path(self, sample: Dict) -> Optional[str]:
        img = sample.get("image")
        if img and not os.path.isabs(img):
            img = os.path.join(self.data_path, img)
        return img
    
    def get_sample_id(self, sample: Dict) -> str:
        return sample.get("id", str(hash(self.get_question(sample))))


# Registry of all benchmarks
BENCHMARK_REGISTRY = {
    "mmsafety": MMSafetyBenchEvaluator,
    "mm_safety": MMSafetyBenchEvaluator,
    "vlguard": VLGuardEvaluator,
    "vlsbench": VLSBenchEvaluator,
    "spa_vl": SPAVLEvaluator,
    "siuo": SIUOEvaluator,
    "beavertails_v": BeaverTailsVEvaluator,
    "beavertails": BeaverTailsVEvaluator,
    "mssbench": MSSBenchEvaluator,
}


def get_evaluator(benchmark_name: str, config: EvalConfig, **kwargs) -> SingleTurnEvaluator:
    """Get evaluator instance by benchmark name."""
    if benchmark_name.lower() not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARK_REGISTRY.keys())}")
    
    evaluator_cls = BENCHMARK_REGISTRY[benchmark_name.lower()]
    return evaluator_cls(config, **kwargs)
