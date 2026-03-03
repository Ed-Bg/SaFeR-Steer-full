"""
Single-turn Evaluation Runner

Main entry point for running single-turn safety evaluations.
"""

import argparse
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from .benchmarks import BENCHMARK_REGISTRY, get_evaluator
from ..utils.config import EvalConfig


def run_single_turn_evaluation(
    benchmarks: List[str],
    config: EvalConfig,
    stages: List[str] = None,
    data_paths: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Run single-turn evaluation on multiple benchmarks.
    
    Args:
        benchmarks: List of benchmark names
        config: Evaluation configuration
        stages: Pipeline stages to run (infer, eval, aggregate)
        data_paths: Optional custom data paths per benchmark
    
    Returns:
        Results dictionary with metrics for each benchmark
    """
    if stages is None:
        stages = ["infer", "eval", "aggregate"]
    
    if data_paths is None:
        data_paths = {}
    
    all_results = {}
    
    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {benchmark}")
        print(f"{'='*60}")
        
        try:
            # Get custom data path if provided
            kwargs = {}
            if benchmark in data_paths:
                kwargs["data_path"] = data_paths[benchmark]
            
            # Create evaluator and run
            evaluator = get_evaluator(benchmark, config, **kwargs)
            results = evaluator.run(stages=stages)
            
            all_results[benchmark] = results
            
        except Exception as e:
            print(f"Error evaluating {benchmark}: {e}")
            all_results[benchmark] = {"error": str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    for benchmark, results in all_results.items():
        if "metrics" in results:
            metrics = results["metrics"]
            print(f"\n{benchmark}:")
            print(f"  Safe Rate: {metrics.get('safe_rate', 0):.2%}")
            print(f"  Avg Harmlessness: {metrics.get('avg_harmlessness', 0):.2f}")
            print(f"  Avg Helpfulness: {metrics.get('avg_helpfulness', 0):.2f}")
        elif "error" in results:
            print(f"\n{benchmark}: ERROR - {results['error']}")
    
    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Single-turn Safety Evaluation")
    
    # Required arguments
    parser.add_argument("--benchmarks", type=str, required=True,
                       help="Comma-separated list of benchmarks")
    parser.add_argument("--model", type=str, required=True,
                       help="Model path or ID")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="outputs/single_turn",
                       help="Output directory")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--stages", type=str, default="infer,eval,aggregate",
                       help="Comma-separated stages to run")
    
    # vLLM settings
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--vllm_host", type=str, default="localhost")
    
    # Inference settings
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=16)
    
    # Judge settings
    parser.add_argument("--judge_model", type=str, default="gpt-5-nano")
    
    args = parser.parse_args()
    
    # Build config
    config = EvalConfig(
        model_id=args.model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        vllm_port=args.vllm_port,
        vllm_host=args.vllm_host,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        judge_model=args.judge_model,
    )
    
    # Parse benchmarks and stages
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    stages = [s.strip() for s in args.stages.split(",")]
    
    # Run evaluation
    results = run_single_turn_evaluation(benchmarks, config, stages)
    
    # Save final results
    output_path = Path(args.output_dir) / "all_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
