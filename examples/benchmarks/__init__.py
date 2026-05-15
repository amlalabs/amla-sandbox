"""Long-context benchmarks for RLM evaluation.

Implements adapters for major industry benchmarks:
- RULER: NVIDIA's long-context benchmark (NIAH, Multi-hop, Aggregation)
- LongBench: THUDM's bilingual multi-task benchmark
- InfiniteBench: OpenBMB's 100K+ token benchmark
- ZeroSCROLLS: Zero-shot long document understanding
- BABILong: Reasoning-in-a-haystack at scale
- L-Eval: Standardized long-context evaluation
- SCROLLS: Long document NLP tasks
- LongEval: Topic/line retrieval benchmark

Usage:
    from benchmarks import BenchmarkRunner, list_benchmarks

    runner = BenchmarkRunner(llm_handler=my_handler)
    results = await runner.run_benchmark("ruler", subset="niah_single", num_samples=10)
"""

from .base import (
    Benchmark,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSample,
    list_benchmarks,
    register_benchmark,
)

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSample",
    "list_benchmarks",
    "register_benchmark",
]
