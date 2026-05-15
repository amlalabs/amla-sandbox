"""Base classes for benchmark evaluation."""

from __future__ import annotations

import contextlib
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add examples directory to path for local imports
_examples_dir = Path(__file__).parent.parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from rlm import LlmHandler  # noqa: E402
from rlm_sandbox import RlmSandbox  # noqa: E402

# Global registry of benchmarks
_BENCHMARK_REGISTRY: dict[str, type[Benchmark]] = {}


def register_benchmark(name: str):
    """Decorator to register a benchmark class."""

    def decorator(cls: type[Benchmark]) -> type[Benchmark]:
        _BENCHMARK_REGISTRY[name] = cls
        return cls

    return decorator


def list_benchmarks() -> list[str]:
    """List all registered benchmarks."""
    _ensure_benchmarks_registered()
    return list(_BENCHMARK_REGISTRY.keys())


def get_benchmark(name: str) -> type[Benchmark]:
    """Get a benchmark class by name."""
    if name not in _BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list_benchmarks()}")
    return _BENCHMARK_REGISTRY[name]


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""

    id: str
    """Unique identifier for the sample."""

    context: str
    """The long context to process."""

    query: str
    """The question or task."""

    expected: str | list[str] | int | float | None
    """Expected answer(s)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (task type, source, etc.)."""


@dataclass
class SampleResult:
    """Result from evaluating a single sample."""

    sample_id: str
    """ID of the sample."""

    passed: bool
    """Whether the answer was correct."""

    answer: str | None
    """The model's answer."""

    expected: str | list[str] | int | float | None
    """Expected answer."""

    tokens: int
    """Total tokens used."""

    subcalls: int
    """Number of llm_query() calls."""

    duration: float
    """Execution time in seconds."""

    error: str | None = None
    """Error message if failed."""


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""

    benchmark_name: str
    """Name of the benchmark."""

    subset: str | None
    """Subset or task within the benchmark."""

    method: str
    """'RLM' or 'Baseline'."""

    model: str
    """Model used."""

    samples: list[SampleResult]
    """Individual sample results."""

    total_duration: float
    """Total evaluation time."""

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def num_passed(self) -> int:
        return sum(1 for s in self.samples if s.passed)

    @property
    def accuracy(self) -> float:
        if not self.samples:
            return 0.0
        return self.num_passed / len(self.samples)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.samples)

    @property
    def total_subcalls(self) -> int:
        return sum(s.subcalls for s in self.samples)

    @property
    def avg_duration(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.duration for s in self.samples) / len(self.samples)

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"Subset: {self.subset or 'all'}",
            f"Method: {self.method} ({self.model})",
            f"Accuracy: {self.num_passed}/{self.num_samples} ({self.accuracy:.1%})",
            f"Total tokens: {self.total_tokens:,}",
            f"Sub-LLM calls: {self.total_subcalls}",
            f"Avg duration: {self.avg_duration:.2f}s",
        ]
        return "\n".join(lines)


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    name: str = "base"
    description: str = "Base benchmark class"
    source_url: str = ""
    paper_url: str = ""

    @abstractmethod
    def list_subsets(self) -> list[str]:
        """List available subsets/tasks within this benchmark."""
        ...

    @abstractmethod
    async def load_samples(
        self,
        subset: str | None = None,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> list[BenchmarkSample]:
        """Load benchmark samples.

        Args:
            subset: Specific subset/task to load (None for all).
            num_samples: Maximum samples to load (None for all).
            seed: Random seed for reproducibility.

        Returns:
            List of benchmark samples.
        """
        ...

    @abstractmethod
    def check_answer(
        self, answer: str | None, expected: str | list[str] | int | float | None
    ) -> bool:
        """Check if an answer is correct.

        Args:
            answer: Model's answer.
            expected: Expected answer(s).

        Returns:
            True if correct, False otherwise.
        """
        ...


class BenchmarkRunner:
    """Runs benchmarks against RLM or baseline models."""

    def __init__(
        self,
        llm_handler: LlmHandler,
        model: str = "gpt-4o-mini",
        max_iterations: int = 12,
        max_subcalls: int = 20,
    ):
        """Initialize the benchmark runner.

        Args:
            llm_handler: LLM handler for API calls.
            model: Model to use.
            max_iterations: Max RLM iterations.
            max_subcalls: Max llm_query() calls.
        """
        self.llm_handler = llm_handler
        self.model = model
        self.max_iterations = max_iterations
        self.max_subcalls = max_subcalls

        # Cached WASM sandbox (reused across samples)
        self._wasm_sandbox = None

    async def run_benchmark(
        self,
        benchmark_name: str,
        subset: str | None = None,
        num_samples: int | None = None,
        seed: int = 42,
        use_rlm: bool = True,
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """Run a benchmark evaluation.

        Args:
            benchmark_name: Name of the benchmark to run.
            subset: Specific subset/task to evaluate.
            num_samples: Maximum samples to evaluate.
            seed: Random seed for reproducibility.
            use_rlm: If True, use RLM. If False, use direct baseline.
            show_progress: Print progress during evaluation.

        Returns:
            BenchmarkResult with all sample results.
        """
        # Import benchmarks to ensure registration
        _ensure_benchmarks_registered()

        benchmark_cls = get_benchmark(benchmark_name)
        benchmark = benchmark_cls()

        # Load samples
        samples = await benchmark.load_samples(
            subset=subset,
            num_samples=num_samples,
            seed=seed,
        )

        if show_progress:
            print(f"Running {benchmark_name} ({subset or 'all'})")
            print(f"  Samples: {len(samples)}")
            print(f"  Method: {'RLM' if use_rlm else 'Baseline'}")
            print(f"  Model: {self.model}")

        start_time = time.time()
        sample_results: list[SampleResult] = []

        for i, sample in enumerate(samples):
            if show_progress:
                print(f"  [{i + 1}/{len(samples)}] {sample.id}...", end=" ", flush=True)

            if use_rlm:
                result = await self._run_rlm_sample(sample, benchmark)
            else:
                result = await self._run_baseline_sample(sample, benchmark)

            sample_results.append(result)

            if show_progress:
                status = "✓" if result.passed else "✗"
                error_info = f" [ERR: {result.error[:50]}]" if result.error else ""
                answer_info = (
                    f" [ans: {str(result.answer)[:30] if result.answer else 'None'}]"
                    if not result.passed
                    else ""
                )
                print(
                    f"{status} ({result.tokens:,} tok, {result.duration:.1f}s){error_info}{answer_info}"
                )

        total_duration = time.time() - start_time

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            subset=subset,
            method="RLM" if use_rlm else "Baseline",
            model=self.model,
            samples=sample_results,
            total_duration=total_duration,
        )

    async def _run_rlm_sample(
        self, sample: BenchmarkSample, benchmark: Benchmark
    ) -> SampleResult:
        """Run a single sample with RLM using WASM sandbox."""
        import inspect

        start_time = time.time()

        try:
            # Create fresh WASM sandbox for each sample to avoid state pollution
            if True:  # Always create fresh sandbox
                # Wrap LlmHandler to simple string->string function
                async def simple_llm(prompt: str) -> str:
                    result = self.llm_handler(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                    )
                    if inspect.iscoroutine(result):
                        result = await result
                    return result.content

                self._wasm_sandbox = RlmSandbox(
                    llm_handler=simple_llm,
                    max_iterations=self.max_iterations,
                    max_subcalls=self.max_subcalls,
                )

            result = await self._wasm_sandbox.run_rlm(
                context=sample.context,
                query=sample.query,
            )

            answer = result.answer
            passed = benchmark.check_answer(answer, sample.expected)

            return SampleResult(
                sample_id=sample.id,
                passed=passed,
                answer=answer,
                expected=sample.expected,
                tokens=0,  # WASM sandbox doesn't track tokens yet
                subcalls=result.subcalls,
                duration=time.time() - start_time,
            )

        except Exception as e:
            return SampleResult(
                sample_id=sample.id,
                passed=False,
                answer=None,
                expected=sample.expected,
                tokens=0,
                subcalls=0,
                duration=time.time() - start_time,
                error=str(e),
            )

    async def _run_baseline_sample(
        self, sample: BenchmarkSample, benchmark: Benchmark
    ) -> SampleResult:
        """Run a single sample with direct baseline (no RLM)."""
        import inspect

        start_time = time.time()

        try:
            system_prompt = """You are a precise document analyzer. Read the document carefully and answer questions accurately.
For retrieval tasks: search thoroughly and return exact values found.
For counting tasks: count methodically and return the exact number.
For reasoning tasks: think step by step and provide a clear answer.
Be concise - return only what is asked for."""

            messages = [
                {
                    "role": "user",
                    "content": f"Document:\n\n{sample.context}\n\nQuestion: {sample.query}",
                }
            ]

            response = self.llm_handler(
                messages=messages,
                system=system_prompt,
                model=self.model,
            )

            if inspect.iscoroutine(response):
                response = await response

            answer = response.content
            tokens = response.input_tokens + response.output_tokens
            passed = benchmark.check_answer(answer, sample.expected)

            return SampleResult(
                sample_id=sample.id,
                passed=passed,
                answer=answer,
                expected=sample.expected,
                tokens=tokens,
                subcalls=0,
                duration=time.time() - start_time,
            )

        except Exception as e:
            return SampleResult(
                sample_id=sample.id,
                passed=False,
                answer=None,
                expected=sample.expected,
                tokens=0,
                subcalls=0,
                duration=time.time() - start_time,
                error=str(e),
            )

    async def compare(
        self,
        benchmark_name: str,
        subset: str | None = None,
        num_samples: int | None = None,
        seed: int = 42,
        rlm_model: str | None = None,
        baseline_model: str | None = None,
        show_progress: bool = True,
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """Run both RLM and baseline on a benchmark for comparison.

        Args:
            benchmark_name: Name of the benchmark.
            subset: Specific subset/task.
            num_samples: Maximum samples.
            seed: Random seed.
            rlm_model: Model for RLM (default: self.model).
            baseline_model: Model for baseline (default: self.model).
            show_progress: Print progress.

        Returns:
            Tuple of (rlm_result, baseline_result).
        """
        original_model = self.model

        # Run RLM
        if rlm_model:
            self.model = rlm_model
        rlm_result = await self.run_benchmark(
            benchmark_name=benchmark_name,
            subset=subset,
            num_samples=num_samples,
            seed=seed,
            use_rlm=True,
            show_progress=show_progress,
        )

        # Run Baseline
        if baseline_model:
            self.model = baseline_model
        baseline_result = await self.run_benchmark(
            benchmark_name=benchmark_name,
            subset=subset,
            num_samples=num_samples,
            seed=seed,
            use_rlm=False,
            show_progress=show_progress,
        )

        self.model = original_model
        return rlm_result, baseline_result


def _ensure_benchmarks_registered():
    """Import all benchmark modules to ensure registration."""
    # Import each benchmark module - this triggers @register_benchmark decorators.
    # Some benchmarks may not be available (optional deps); ignore those.
    with contextlib.suppress(ImportError):
        from benchmarks import (  # noqa: F401
            babilong,
            infinitebench,
            leval,
            longbench,
            longeval,
            rlm_suite,
            ruler,
            scrolls,
            zeroscrolls,
        )
