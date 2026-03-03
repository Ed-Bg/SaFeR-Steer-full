"""
Single-turn Safety Evaluation

Modular evaluation pipeline for single-turn safety benchmarks.

Supported benchmarks (paper Table 3):
- BeaverTails-V
- MM-SafetyBench
- SPA-VL
- VLGuard  
- VLSBench
"""

from .base import SingleTurnEvaluator
from .runner import run_single_turn_evaluation

__all__ = [
    "SingleTurnEvaluator",
    "run_single_turn_evaluation",
]
