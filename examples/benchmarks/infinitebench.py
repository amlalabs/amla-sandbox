"""InfiniteBench - 100K+ token evaluation benchmark.

Reference: https://github.com/OpenBMB/InfiniteBench
Paper: https://arxiv.org/abs/2402.13718
HuggingFace: https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench

Tasks:
- Retrieve: passkey, number_string, kv_retrieval
- Code: code_debug, code_run
- Book QA: longbook_choice_eng, longbook_qa_eng, longbook_qa_chn
- Math: math_calc, math_find
- Dialogue: longdialogue_qa_eng
- Summarization: longbook_sum_eng
"""

from __future__ import annotations

import random
import re
import string

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("infinitebench")
class InfiniteBenchBenchmark(Benchmark):
    """InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens."""

    name = "infinitebench"
    description = "OpenBMB's 100K+ token benchmark"
    source_url = "https://github.com/OpenBMB/InfiniteBench"
    paper_url = "https://arxiv.org/abs/2402.13718"

    SUBSETS = [
        "passkey",
        "number_string",
        "kv_retrieval",
        "code_debug",
        "code_run",
        "longbook_choice_eng",
        "longbook_qa_eng",
        "math_find",
        "longdialogue_qa_eng",
    ]

    def list_subsets(self) -> list[str]:
        return self.SUBSETS

    async def load_samples(
        self,
        subset: str | None = None,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        rng = random.Random(seed)

        subsets_to_load = [subset] if subset else self.SUBSETS[:4]

        for task in subsets_to_load:
            try:
                task_samples = await self._load_from_huggingface(task, num_samples)
            except Exception:
                task_samples = self._generate_synthetic(task, num_samples or 3, rng)
            samples.extend(task_samples)

        if num_samples and len(samples) > num_samples:
            rng.shuffle(samples)
            samples = samples[:num_samples]

        return samples

    async def _load_from_huggingface(
        self, subset: str, num_samples: int | None
    ) -> list[BenchmarkSample]:
        """Load from HuggingFace datasets."""
        try:
            from datasets import load_dataset

            ds = load_dataset(
                "xinrongzhang2022/InfiniteBench",
                subset,
                split="test",
                trust_remote_code=True,
            )

            samples = []
            limit = num_samples or min(10, len(ds))

            for i, item in enumerate(ds):
                if i >= limit:
                    break

                sample = BenchmarkSample(
                    id=f"{subset}_{i}",
                    context=item.get("context", item.get("input", "")),
                    query=item.get("question", item.get("query", "")),
                    expected=item.get("answer", ""),
                    metadata={"task": subset, "source": "huggingface"},
                )
                samples.append(sample)

            return samples
        except ImportError as e:
            raise ImportError("datasets library not installed") from e

    def _generate_synthetic(
        self, subset: str, num_samples: int, rng: random.Random
    ) -> list[BenchmarkSample]:
        """Generate synthetic samples."""
        samples = []

        for i in range(num_samples):
            if subset == "passkey":
                sample = self._gen_passkey(i, rng)
            elif subset == "number_string":
                sample = self._gen_number_string(i, rng)
            elif subset == "kv_retrieval":
                sample = self._gen_kv_retrieval(i, rng)
            elif subset == "code_debug":
                sample = self._gen_code_debug(i, rng)
            elif subset == "code_run":
                sample = self._gen_code_run(i, rng)
            elif subset == "math_find":
                sample = self._gen_math_find(i, rng)
            else:
                sample = self._gen_generic(subset, i, rng)
            samples.append(sample)

        return samples

    def _gen_passkey(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Passkey retrieval (classic needle-in-haystack)."""
        passkey = "".join(rng.choices(string.digits, k=5))

        # Generate long filler text (~50K chars)
        filler = (
            "The grass is green. The sky is blue. The sun is yellow. "
            "Here we go. There and back again. "
        ) * 2000

        # Insert passkey somewhere in the middle
        insert_pos = len(filler) // 2
        context = (
            filler[:insert_pos]
            + f"\n\nThe pass key is {passkey}. Remember it.\n\n"
            + filler[insert_pos:]
        )

        query = "What is the pass key?"

        return BenchmarkSample(
            id=f"passkey_{idx}",
            context=context,
            query=query,
            expected=passkey,
            metadata={
                "task": "passkey",
                "source": "synthetic",
                "context_len": len(context),
            },
        )

    def _gen_number_string(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Find a number in a long string of numbers."""
        # Generate long number string
        numbers = [str(rng.randint(0, 9)) for _ in range(50000)]
        target_pos = rng.randint(20000, 30000)
        target = "".join(rng.choices(string.digits, k=7))

        # Insert target
        for i, c in enumerate(target):
            numbers[target_pos + i] = c

        context = "".join(numbers)
        query = f"Find the sequence '{target}' in the number string. At what position does it start?"

        return BenchmarkSample(
            id=f"number_string_{idx}",
            context=context,
            query=query,
            expected=str(target_pos),
            metadata={"task": "number_string", "source": "synthetic"},
        )

    def _gen_kv_retrieval(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Key-value retrieval from a large dictionary."""
        # Generate many key-value pairs
        keys = [f"key_{i:04d}" for i in range(1000)]
        values = [
            "".join(rng.choices(string.ascii_lowercase, k=8)) for _ in range(1000)
        ]

        pairs = [f"{k}: {v}" for k, v in zip(keys, values, strict=False)]
        rng.shuffle(pairs)

        target_idx = rng.randint(0, 999)
        target_key = f"key_{target_idx:04d}"
        target_value = values[target_idx]

        context = "\n".join(pairs)
        query = f"What is the value for '{target_key}'?"

        return BenchmarkSample(
            id=f"kv_retrieval_{idx}",
            context=context,
            query=query,
            expected=target_value,
            metadata={"task": "kv_retrieval", "source": "synthetic"},
        )

    def _gen_code_debug(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Find bug in code."""
        bug_type = rng.choice(["off_by_one", "wrong_operator", "missing_return"])

        if bug_type == "off_by_one":
            code = (
                """
def sum_range(n):
    total = 0
    for i in range(n):  # BUG: should be range(n+1) to include n
        total += i
    return total

# Many other functions here...
"""
                + "\ndef helper():\n    pass\n" * 100
            )

            query = "What is the bug in the sum_range function?"
            expected = "range(n+1)"

        elif bug_type == "wrong_operator":
            code = (
                """
def multiply(a, b):
    return a + b  # BUG: should be a * b

# Many other functions here...
"""
                + "\ndef helper():\n    pass\n" * 100
            )

            query = "What is the bug in the multiply function?"
            expected = "a * b"

        else:
            code = (
                """
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for x in lst:
        if x > max_val:
            max_val = x
    # BUG: missing return statement

# Many other functions here...
"""
                + "\ndef helper():\n    pass\n" * 100
            )

            query = "What is missing in the find_max function?"
            expected = "return"

        return BenchmarkSample(
            id=f"code_debug_{idx}",
            context=code,
            query=query,
            expected=expected,
            metadata={
                "task": "code_debug",
                "source": "synthetic",
                "bug_type": bug_type,
            },
        )

    def _gen_code_run(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Trace code execution to find output."""
        a = rng.randint(1, 10)
        b = rng.randint(1, 10)
        expected = a + b

        code = f"""
x = {a}
y = {b}
z = x + y
print(z)
"""
        # Add filler code
        filler = "\n# Comment line\n" * 500
        context = code + filler

        query = "What will be printed when this code runs?"

        return BenchmarkSample(
            id=f"code_run_{idx}",
            context=context,
            query=query,
            expected=str(expected),
            metadata={"task": "code_run", "source": "synthetic"},
        )

    def _gen_math_find(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Find a number with specific mathematical property."""
        # Generate list of numbers, one is special
        numbers = [rng.randint(10, 1000) for _ in range(100)]
        special_idx = rng.randint(0, 99)

        # Make one number a perfect square
        root = rng.randint(5, 30)
        numbers[special_idx] = root * root

        context = "Numbers: " + ", ".join(map(str, numbers))
        query = "Which number in the list is a perfect square?"

        return BenchmarkSample(
            id=f"math_find_{idx}",
            context=context,
            query=query,
            expected=str(numbers[special_idx]),
            metadata={"task": "math_find", "source": "synthetic"},
        )

    def _gen_generic(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate generic sample."""
        answer = "".join(rng.choices(string.ascii_uppercase, k=6))
        context = f"The answer is {answer}. " + "Filler text. " * 5000
        query = "What is the answer?"

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=answer,
            metadata={"task": subset, "source": "synthetic"},
        )

    def check_answer(
        self, answer: str | None, expected: str | list[str] | int | float | None
    ) -> bool:
        if answer is None:
            return False

        answer_lower = answer.lower()

        if isinstance(expected, list):
            return any(str(e).lower() in answer_lower for e in expected)
        if isinstance(expected, int):
            numbers = re.findall(r"\d+", answer)
            return str(expected) in numbers
        return str(expected).lower() in answer_lower
