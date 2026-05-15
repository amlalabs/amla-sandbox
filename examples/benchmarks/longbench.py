"""LongBench Benchmark - THUDM's bilingual multi-task benchmark.

Reference: https://github.com/THUDM/LongBench
Paper: https://arxiv.org/abs/2308.14508
HuggingFace: https://huggingface.co/datasets/THUDM/LongBench

Tasks (21 total across 6 categories):
- Single-Doc QA: narrativeqa, qasper, multifieldqa_en, multifieldqa_zh
- Multi-Doc QA: hotpotqa, 2wikimqa, musique
- Summarization: gov_report, qmsum, multi_news, vcsum
- Few-shot: trec, triviaqa, samsum, lsht
- Synthetic: passage_count, passage_retrieval_en, passage_retrieval_zh
- Code: lcc, repobench-p
"""

from __future__ import annotations

import random
import re

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("longbench")
class LongBenchBenchmark(Benchmark):
    """LongBench: Bilingual, Multitask Benchmark for Long Context Understanding."""

    name = "longbench"
    description = "THUDM's bilingual multi-task long-context benchmark"
    source_url = "https://github.com/THUDM/LongBench"
    paper_url = "https://arxiv.org/abs/2308.14508"

    # English subsets (most commonly used)
    SUBSETS = [
        # Single-doc QA
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        # Multi-doc QA
        "hotpotqa",
        "2wikimqa",
        "musique",
        # Summarization
        "gov_report",
        "qmsum",
        "multi_news",
        # Few-shot
        "trec",
        "triviaqa",
        "samsum",
        # Synthetic
        "passage_count",
        "passage_retrieval_en",
        # Code
        "lcc",
        "repobench-p",
    ]

    def list_subsets(self) -> list[str]:
        return self.SUBSETS

    async def load_samples(
        self,
        subset: str | None = None,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> list[BenchmarkSample]:
        """Load samples from HuggingFace or generate synthetic equivalents."""
        samples: list[BenchmarkSample] = []
        rng = random.Random(seed)

        subsets_to_load = [subset] if subset else self.SUBSETS[:5]  # Default to first 5

        for task in subsets_to_load:
            task_samples = await self._load_subset(task, num_samples, rng)
            samples.extend(task_samples)

        if num_samples and len(samples) > num_samples:
            rng.shuffle(samples)
            samples = samples[:num_samples]

        return samples

    async def _load_subset(
        self, subset: str, num_samples: int | None, rng: random.Random
    ) -> list[BenchmarkSample]:
        """Load a specific subset."""
        try:
            # Try loading from HuggingFace
            return await self._load_from_huggingface(subset, num_samples)
        except Exception:
            # Fall back to synthetic data
            return self._generate_synthetic(subset, num_samples or 5, rng)

    async def _load_from_huggingface(
        self, subset: str, num_samples: int | None
    ) -> list[BenchmarkSample]:
        """Load from HuggingFace datasets."""
        try:
            from datasets import load_dataset

            # Load dataset
            ds = load_dataset(
                "THUDM/LongBench", subset, split="test", trust_remote_code=True
            )

            samples = []
            limit = num_samples or len(ds)

            for i, item in enumerate(ds):
                if i >= limit:
                    break

                sample = BenchmarkSample(
                    id=f"{subset}_{i}",
                    context=item.get("context", ""),
                    query=item.get("input", ""),
                    expected=item.get("answers", item.get("answer", "")),
                    metadata={
                        "task": subset,
                        "source": "huggingface",
                        "length": item.get("length", len(item.get("context", ""))),
                    },
                )
                samples.append(sample)

            return samples

        except ImportError as e:
            raise ImportError(
                "datasets library not installed. Run: pip install datasets"
            ) from e

    def _generate_synthetic(
        self, subset: str, num_samples: int, rng: random.Random
    ) -> list[BenchmarkSample]:
        """Generate synthetic samples matching the task type."""
        samples = []

        for i in range(num_samples):
            if subset in ["narrativeqa", "qasper", "multifieldqa_en"]:
                sample = self._gen_single_doc_qa(subset, i, rng)
            elif subset in ["hotpotqa", "2wikimqa", "musique"]:
                sample = self._gen_multi_doc_qa(subset, i, rng)
            elif subset in ["gov_report", "qmsum", "multi_news"]:
                sample = self._gen_summarization(subset, i, rng)
            elif subset in ["passage_count", "passage_retrieval_en"]:
                sample = self._gen_synthetic_task(subset, i, rng)
            else:
                sample = self._gen_generic(subset, i, rng)

            samples.append(sample)

        return samples

    def _gen_single_doc_qa(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate single-document QA sample."""
        # Create a document with embedded facts
        topics = [
            "renewable energy",
            "machine learning",
            "climate change",
            "healthcare",
        ]
        topic = rng.choice(topics)

        facts = [
            f"The global market for {topic} reached $500 billion in 2023.",
            f"Research shows that {topic} adoption increased by 45% since 2020.",
            f"Experts predict {topic} will transform industries by 2030.",
            f"Major companies invested $10 billion in {topic} last year.",
        ]

        paragraphs = []
        for _ in range(rng.randint(10, 20)):
            paragraphs.append(
                f"This section discusses various aspects of {topic} and its implications. "
                f"The analysis covers multiple dimensions including economic, social, and technical factors. "
                f"Understanding these elements is crucial for stakeholders and policymakers alike."
            )
            if rng.random() < 0.3:
                paragraphs.append(rng.choice(facts))

        context = "\n\n".join(paragraphs)
        query = f"According to the document, what was the global market size for {topic} in 2023?"
        expected = "$500 billion"

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=expected,
            metadata={"task": subset, "source": "synthetic"},
        )

    def _gen_multi_doc_qa(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate multi-document QA sample."""
        # Multiple documents with facts that need to be connected
        person = rng.choice(["Dr. Smith", "Prof. Johnson", "Ms. Chen", "Mr. Williams"])
        org = rng.choice(["Stanford", "MIT", "Oxford", "Cambridge"])
        field = rng.choice(["physics", "biology", "chemistry", "computer science"])
        year = rng.randint(1990, 2020)

        doc1 = f"""Document 1: Biography
{person} is a renowned researcher in the field of {field}. Their groundbreaking work
has been recognized internationally. They received their PhD from a prestigious institution
and have published over 100 papers. Their current affiliation is with {org}.
"""

        doc2 = f"""Document 2: Awards and Recognition
The Nobel Prize committee announced in {year} that they would be honoring exceptional
contributions to {field}. The award recognized decades of innovative research that
fundamentally changed our understanding of the subject. The recipient was {person}.
"""

        doc3 = f"""Document 3: Institution History
{org} has a long tradition of excellence in {field}. The institution has been home
to many Nobel laureates over the years. Faculty members have made significant contributions
to both theoretical and applied aspects of their fields.
"""

        context = f"{doc1}\n\n{doc2}\n\n{doc3}"
        query = f"Where does the {year} Nobel Prize winner in {field} work?"
        expected = org

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=expected,
            metadata={"task": subset, "source": "synthetic"},
        )

    def _gen_summarization(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate summarization sample."""
        topics = [
            "quarterly earnings",
            "policy changes",
            "research findings",
            "market trends",
        ]
        topic = rng.choice(topics)

        paragraphs = []
        key_points = []

        for i in range(rng.randint(15, 25)):
            if i % 5 == 0:
                point = f"Key finding {i // 5 + 1}: {topic} showed {rng.randint(10, 50)}% improvement"
                paragraphs.append(point)
                key_points.append(point)
            else:
                paragraphs.append(
                    f"Paragraph {i}: This section provides additional context about {topic}. "
                    f"Various factors contribute to the overall trends observed. "
                    f"Stakeholders should consider multiple perspectives when analyzing results."
                )

        context = "\n\n".join(paragraphs)
        query = "Summarize the key findings from this document."

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=key_points,  # All key points should appear in summary
            metadata={"task": subset, "source": "synthetic"},
        )

    def _gen_synthetic_task(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate synthetic task (passage count/retrieval)."""
        if subset == "passage_count":
            # Count how many passages mention a specific topic
            target = rng.choice(
                ["innovation", "sustainability", "technology", "growth"]
            )
            count = rng.randint(3, 8)

            paragraphs = []
            for i in range(15):
                if i < count:
                    paragraphs.append(
                        f"Passage {i + 1}: This section discusses {target} and its importance. "
                        f"The topic of {target} is crucial for modern organizations."
                    )
                else:
                    paragraphs.append(
                        f"Passage {i + 1}: This section covers general business practices. "
                        f"Various methodologies are discussed in detail."
                    )

            rng.shuffle(paragraphs)
            context = "\n\n".join(paragraphs)
            query = f"How many passages specifically discuss '{target}'? Return only the number."
            expected = count

        else:  # passage_retrieval
            # Find which passage contains specific information
            target_passage = rng.randint(1, 10)
            secret = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=8))

            paragraphs = []
            for i in range(10):
                if i + 1 == target_passage:
                    paragraphs.append(
                        f"Passage {i + 1}: The secret code for the vault is {secret}. "
                        f"This information is highly confidential and should not be shared."
                    )
                else:
                    paragraphs.append(
                        f"Passage {i + 1}: This section contains general information. "
                        f"Nothing particularly sensitive is discussed here."
                    )

            context = "\n\n".join(paragraphs)
            query = "What is the secret code mentioned in the document?"
            expected = secret

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=expected,
            metadata={"task": subset, "source": "synthetic"},
        )

    def _gen_generic(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate generic sample for unhandled subsets."""
        value = "".join(rng.choices("0123456789", k=6))
        context = f"The answer to the question is {value}. " * 100
        query = "What is the answer mentioned in the document?"

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=value,
            metadata={"task": subset, "source": "synthetic"},
        )

    def check_answer(
        self, answer: str | None, expected: str | list[str] | int | float | None
    ) -> bool:
        """Check answer correctness using F1 or exact match."""
        if answer is None:
            return False

        answer_lower = answer.lower()

        if isinstance(expected, list):
            # For lists, check if any expected answer matches
            # (LongBench uses F1 score, we approximate with containment)
            matches = sum(1 for e in expected if str(e).lower() in answer_lower)
            return matches >= len(expected) * 0.5  # At least 50% match
        if isinstance(expected, int):
            numbers = re.findall(r"\d+", answer)
            return str(expected) in numbers
        # Normalize and check containment
        expected_lower = str(expected).lower()
        return expected_lower in answer_lower
