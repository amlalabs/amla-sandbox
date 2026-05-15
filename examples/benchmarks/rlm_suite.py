"""RLM Suite - Custom benchmark tasks for RLM evaluation.

This benchmark includes tasks specifically designed to test RLM capabilities:
- NIAH tasks (from RULER-style evaluation)
- Aggregation (code-solvable counting)
- Sentiment Analysis (requires llm_query for semantic reasoning)
- Claim Verification (requires llm_query for evidence comparison)

These tasks test both code-solvable and semantic reasoning capabilities.
"""

from __future__ import annotations

import random
import re
import string

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("rlm_suite")
class RlmSuiteBenchmark(Benchmark):
    """RLM Suite: Custom tasks for testing RLM capabilities."""

    name = "rlm_suite"
    description = "Custom RLM evaluation tasks (NIAH, aggregation, semantic reasoning)"
    source_url = "https://github.com/alexzhang13/rlm"
    paper_url = "https://arxiv.org/abs/2512.24601"

    FILLER_PARAGRAPHS = [
        "The system architecture employs a modular design pattern that facilitates maintenance and scalability across distributed environments.",
        "Performance metrics indicate consistent throughput across various load conditions with minimal latency variance.",
        "The implementation follows industry best practices for security and data integrity as defined by current standards.",
        "Documentation has been updated to reflect recent changes in the API specification and endpoint behaviors.",
        "Testing coverage exceeds target thresholds for all critical code paths and edge cases.",
        "The deployment pipeline includes automated quality gates and rollback capabilities for reliability.",
        "User feedback has been incorporated into the latest iteration of the interface design.",
        "The database schema supports both relational and document-based queries efficiently.",
        "Monitoring dashboards provide real-time visibility into system health metrics.",
        "The caching layer significantly reduces latency for frequently accessed data patterns.",
    ]

    def list_subsets(self) -> list[str]:
        return [
            "niah_single",
            "niah_multi",
            "aggregation",
            "sentiment",
            "claim_verification",
        ]

    async def load_samples(
        self,
        subset: str | None = None,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        rng = random.Random(seed)

        subsets_to_run = [subset] if subset else self.list_subsets()
        context_sizes = [8000, 16000, 32000, 50000]

        for task in subsets_to_run:
            if task in ("niah_single", "niah_multi", "aggregation"):
                # Generate samples at different context sizes
                for size in context_sizes:
                    sample = self._generate_sample(task, size, rng)
                    if sample:
                        samples.append(sample)
            else:
                # Semantic tasks don't need size variations
                sample = self._generate_sample(task, 0, rng)
                if sample:
                    samples.append(sample)

        if num_samples and len(samples) > num_samples:
            rng.shuffle(samples)
            samples = samples[:num_samples]

        return samples

    def _generate_sample(
        self, task: str, context_size: int, rng: random.Random
    ) -> BenchmarkSample | None:
        if task == "niah_single":
            return self._gen_niah_single(context_size, rng)
        if task == "niah_multi":
            return self._gen_niah_multi(context_size, 5, rng)
        if task == "aggregation":
            return self._gen_aggregation(context_size, rng)
        if task == "sentiment":
            return self._gen_sentiment(rng)
        if task == "claim_verification":
            return self._gen_claim_verification(rng)
        return None

    def _generate_haystack(self, target_chars: int, rng: random.Random) -> str:
        text = []
        current_len = 0
        while current_len < target_chars:
            para = rng.choice(self.FILLER_PARAGRAPHS)
            text.append(para)
            current_len += len(para) + 2
        return "\n\n".join(text)

    def _gen_niah_single(
        self, context_size: int, rng: random.Random
    ) -> BenchmarkSample:
        """Single needle retrieval."""
        needle_id = "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
        needle = f"The secret access code is {needle_id}."

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")
        insert_pos = len(lines) // 2
        lines.insert(
            insert_pos, f"\n[CLASSIFIED INFORMATION]\n{needle}\n[END CLASSIFIED]\n"
        )

        context = "\n\n".join(lines)
        query = "What is the secret access code mentioned in the document? Return ONLY the code itself."

        return BenchmarkSample(
            id=f"niah_single_{context_size}",
            context=context,
            query=query,
            expected=needle_id,
            metadata={"task": "niah_single", "context_size": context_size},
        )

    def _gen_niah_multi(
        self, context_size: int, num_needles: int, rng: random.Random
    ) -> BenchmarkSample:
        """Multi-needle retrieval."""
        cities = ["Paris", "Tokyo", "London", "Sydney", "Cairo"]
        codes: dict[str, str] = {}
        needles = []

        for city in cities[:num_needles]:
            code = "".join(rng.choices(string.digits, k=4))
            codes[city] = code
            needles.append(f"The access code for {city} office is {code}.")

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")

        for i, needle in enumerate(needles):
            pos = (i + 1) * len(lines) // (num_needles + 1)
            lines.insert(pos + i, f"\n[OFFICE CODE]\n{needle}\n[END CODE]\n")

        context = "\n\n".join(lines)
        query = "List ALL office access codes mentioned in the document. Format each as 'City: Code' on a new line."
        expected = [f"{city}: {code}" for city, code in codes.items()]

        return BenchmarkSample(
            id=f"niah_multi_{context_size}_{num_needles}",
            context=context,
            query=query,
            expected=expected,
            metadata={
                "task": "niah_multi",
                "context_size": context_size,
                "num_needles": num_needles,
            },
        )

    def _gen_aggregation(
        self, context_size: int, rng: random.Random
    ) -> BenchmarkSample:
        """Word counting test (code-solvable)."""
        target_word = "infrastructure"
        target_count = 7

        haystack = self._generate_haystack(context_size, rng)
        words = haystack.split()

        step = len(words) // (target_count + 1)
        for i in range(target_count):
            pos = (i + 1) * step
            words.insert(pos, target_word)

        context = " ".join(words)
        query = f"Count exactly how many times the word '{target_word}' appears in the document. Return ONLY the number."

        return BenchmarkSample(
            id=f"aggregation_{context_size}",
            context=context,
            query=query,
            expected=target_count,
            metadata={
                "task": "aggregation",
                "context_size": context_size,
                "target_word": target_word,
            },
        )

    def _gen_sentiment(self, rng: random.Random) -> BenchmarkSample:
        """Sentiment analysis - requires llm_query for semantic reasoning."""
        sections = [
            (
                "Product Quality",
                "positive",
                """Our customers absolutely love the new product line. Sales have exceeded expectations
by 40%, and customer satisfaction scores reached an all-time high of 94%. The quality
improvements we made last quarter are paying off tremendously.""",
            ),
            (
                "Customer Support",
                "negative",
                """Unfortunately, our customer support metrics have declined significantly this quarter.
Average response times increased to 48 hours, and resolution rates dropped to just 62%.
We received numerous complaints about unhelpful responses and long wait times.""",
            ),
            (
                "Market Position",
                "positive",
                """Our market share grew impressively from 23% to 31% this quarter, outpacing all competitors.
Brand recognition surveys show a 25-point improvement in positive associations.""",
            ),
            (
                "Financial Performance",
                "negative",
                """Despite revenue growth, profitability remains a serious concern. Operating margins contracted
by 8 percentage points due to rising costs. Cash flow turned negative for the first time.""",
            ),
            (
                "Employee Satisfaction",
                "positive",
                """Employee engagement scores improved dramatically to 87%, our highest ever recorded. Turnover
decreased by 40% following our new benefits program.""",
            ),
        ]

        rng.shuffle(sections)

        doc_parts = ["# Quarterly Business Review\n\n"]
        for title, _, content in sections:
            doc_parts.append(f"## {title}\n\n{content.strip()}\n\n")

        context = "".join(doc_parts)
        positive_count = sum(
            1 for _, sentiment, _ in sections if sentiment == "positive"
        )

        query = """Analyze each section of this business review and determine its sentiment (positive or negative).
Count how many sections have an overall POSITIVE sentiment.
Return ONLY the count of positive sections as a single number."""

        return BenchmarkSample(
            id="sentiment_analysis",
            context=context,
            query=query,
            expected=positive_count,
            metadata={"task": "sentiment", "positive_count": positive_count},
        )

    def _gen_claim_verification(self, rng: random.Random) -> BenchmarkSample:
        """Claim verification - requires semantic comparison."""
        evidence_sections = [
            """## Financial Results
The company reported annual revenue of $4.2 billion, representing 18% year-over-year growth.
Operating income reached $890 million with a margin of 21.2%.""",
            """## Workforce Update
As of December 31st, total headcount stood at 12,847 employees across all divisions.
The engineering team grew by 340 new hires, while sales expanded by 125 positions.""",
            """## Product Launches
Three major products were released this year: CloudSync Pro, DataVault Enterprise, and
SecureFlow Platform. The combined new products generated $180 million in first-year revenue.""",
            """## Geographic Expansion
Operations expanded to 8 new countries. International revenue now represents 34% of total revenue,
up from 28% last year. The APAC region showed particularly strong growth at 45%.""",
        ]

        rng.shuffle(evidence_sections)

        claims = [
            ("Annual revenue exceeded $4 billion", True),
            ("The company has more than 15,000 employees", False),
            ("Three new products were launched", True),
            ("International revenue is less than 30% of total", False),
        ]

        context = "# Annual Report 2024\n\n" + "\n\n".join(evidence_sections)
        context += (
            "\n\n## Claims to Verify\n\nPlease verify each of the following claims:\n"
        )
        for i, (claim, _) in enumerate(claims, 1):
            context += f"{i}. {claim}\n"

        query = """For each numbered claim, determine if it is TRUE or FALSE based on the evidence in the document.
List the TRUE claims by their numbers only (e.g., "1, 3" if claims 1 and 3 are true).
Return ONLY the comma-separated list of TRUE claim numbers."""

        true_claims = [str(i) for i, (_, is_true) in enumerate(claims, 1) if is_true]

        return BenchmarkSample(
            id="claim_verification",
            context=context,
            query=query,
            expected=true_claims,
            metadata={"task": "claim_verification", "true_claims": true_claims},
        )

    def check_answer(
        self, answer: str | None, expected: str | list[str] | int | float | None
    ) -> bool:
        """Check if answer matches expected."""
        if answer is None:
            return False

        answer_lower = answer.lower()

        if isinstance(expected, list):
            # Detect if this is a pure numeric list (claim verification)
            is_pure_numeric = all(re.fullmatch(r"\d+", str(e)) for e in expected)

            if is_pure_numeric:
                # Strict set matching for claim verification
                found_numbers = set(re.findall(r"\b\d+\b", answer))
                expected_set = {str(e) for e in expected}
                return found_numbers == expected_set
            # Substring matching for multi-needle
            return all(str(e).lower() in answer_lower for e in expected)

        if isinstance(expected, int):
            numbers = re.findall(r"\b\d+\b", answer)
            return str(expected) in numbers

        if isinstance(expected, float):
            numbers = re.findall(r"-?\d+\.?\d*", answer)
            return any(abs(float(n) - expected) < 0.01 for n in numbers)

        return str(expected).lower() in answer_lower
