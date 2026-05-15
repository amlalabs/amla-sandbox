"""ZeroSCROLLS Benchmark - Zero-shot long document understanding.

Reference: https://github.com/tau-nlp/zero_scrolls
Paper: https://arxiv.org/abs/2305.14196
HuggingFace: https://huggingface.co/datasets/tau/zero_scrolls

Tasks:
- Summarization: gov_report, summ_screen_fd, qmsum, squality, book_sum_sort
- QA: qasper, narrative_qa, quality, musique
- Aggregation: space_digest
"""

from __future__ import annotations

import random
import re

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("zeroscrolls")
class ZeroScrollsBenchmark(Benchmark):
    """ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding."""

    name = "zeroscrolls"
    description = "Zero-shot long document understanding benchmark"
    source_url = "https://github.com/tau-nlp/zero_scrolls"
    paper_url = "https://arxiv.org/abs/2305.14196"

    SUBSETS = [
        "gov_report",
        "summ_screen_fd",
        "qmsum",
        "squality",
        "qasper",
        "narrative_qa",
        "quality",
        "musique",
        "space_digest",
        "book_sum_sort",
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

        subsets_to_load = [subset] if subset else self.SUBSETS[:3]

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
                "tau/zero_scrolls", subset, split="test", trust_remote_code=True
            )

            samples = []
            limit = num_samples or min(10, len(ds))

            for i, item in enumerate(ds):
                if i >= limit:
                    break

                sample = BenchmarkSample(
                    id=f"{subset}_{i}",
                    context=item.get("input", ""),
                    query=item.get("question", "Summarize this document."),
                    expected=item.get("output", ""),
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
            if subset in ["gov_report", "qmsum", "summ_screen_fd", "squality"]:
                sample = self._gen_summarization(subset, i, rng)
            elif subset in ["qasper", "narrative_qa", "quality"]:
                sample = self._gen_qa(subset, i, rng)
            elif subset == "musique":
                sample = self._gen_multi_hop(i, rng)
            elif subset == "space_digest":
                sample = self._gen_aggregation(i, rng)
            else:
                sample = self._gen_generic(subset, i, rng)
            samples.append(sample)

        return samples

    def _gen_summarization(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate summarization sample."""
        if subset == "gov_report":
            topic = rng.choice(
                [
                    "Environmental Protection",
                    "Healthcare Reform",
                    "Economic Policy",
                    "Infrastructure Investment",
                ]
            )
            key_points = [
                f"The report recommends increased funding for {topic.lower()}.",
                "Implementation should begin in fiscal year 2025.",
                "Expected benefits include improved outcomes and cost savings.",
            ]
        else:
            topic = "Meeting Discussion"
            key_points = [
                "The team discussed project milestones.",
                "Budget allocation was reviewed.",
                "Next steps were agreed upon.",
            ]

        paragraphs = []
        for point in key_points:
            paragraphs.append(point)
            # Add filler
            for _ in range(rng.randint(3, 6)):
                paragraphs.append(
                    f"Additional context regarding {topic.lower()} was provided. "
                    f"Various stakeholders contributed their perspectives on the matter."
                )

        context = "\n\n".join(paragraphs)
        query = "Summarize the key points of this document."

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=key_points,
            metadata={"task": subset, "source": "synthetic"},
        )

    def _gen_qa(self, subset: str, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate QA sample."""
        facts = {
            "author": rng.choice(["Smith", "Johnson", "Williams", "Brown"]),
            "year": str(rng.randint(2015, 2023)),
            "finding": rng.choice(
                ["positive correlation", "no significant effect", "negative impact"]
            ),
        }

        paragraphs = [
            f"This paper by {facts['author']} ({facts['year']}) examines the topic in depth.",
            f"The main finding was a {facts['finding']} between the variables studied.",
            "The methodology included surveys and statistical analysis.",
            "Conclusions suggest further research is needed.",
        ]

        # Add filler
        filler = [
            "Related work in this area has explored similar questions.",
            "The theoretical framework builds on established principles.",
            "Limitations of the study are acknowledged.",
        ] * 5

        all_paragraphs = paragraphs + filler
        rng.shuffle(all_paragraphs)

        context = "\n\n".join(all_paragraphs)
        query = "What was the main finding of the study?"
        expected = facts["finding"]

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=expected,
            metadata={"task": subset, "source": "synthetic"},
        )

    def _gen_multi_hop(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate multi-hop reasoning sample."""
        person = rng.choice(["Alice", "Bob", "Charlie", "Diana"])
        city = rng.choice(["Boston", "Seattle", "Chicago", "Denver"])
        company = rng.choice(["TechCorp", "DataInc", "AILabs", "CloudSys"])

        doc1 = f"{person} works at {company}."
        doc2 = f"{company} is headquartered in {city}."
        doc3 = f"The office in {city} has 500 employees."

        # Add distractors
        distractors = [
            "Many companies are growing rapidly.",
            "The tech industry is evolving.",
            "Remote work is becoming common.",
        ] * 10

        all_parts = [doc1, doc2, doc3, *distractors]
        rng.shuffle(all_parts)

        context = "\n\n".join(all_parts)
        query = f"In which city does {person} work?"
        expected = city

        return BenchmarkSample(
            id=f"musique_{idx}",
            context=context,
            query=query,
            expected=expected,
            metadata={"task": "musique", "source": "synthetic"},
        )

    def _gen_aggregation(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate aggregation sample (space_digest style)."""
        num_positive = rng.randint(3, 7)
        num_negative = rng.randint(2, 5)

        reviews = []
        for i in range(num_positive):
            reviews.append(
                f"Review {i + 1}: Great product! Highly recommend. Very satisfied. ★★★★★"
            )

        for i in range(num_negative):
            reviews.append(
                f"Review {num_positive + i + 1}: Disappointed. Poor quality. Would not buy again. ★☆☆☆☆"
            )

        rng.shuffle(reviews)
        context = "\n\n".join(reviews)
        query = "What percentage of reviews are positive (4-5 stars)?"

        total = num_positive + num_negative
        percentage = round(100 * num_positive / total)
        expected = str(percentage)

        return BenchmarkSample(
            id=f"space_digest_{idx}",
            context=context,
            query=query,
            expected=expected,
            metadata={
                "task": "space_digest",
                "source": "synthetic",
                "num_positive": num_positive,
                "num_negative": num_negative,
            },
        )

    def _gen_generic(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate generic sample."""
        answer = f"answer_{rng.randint(100, 999)}"
        context = f"The answer is {answer}. " + "More text here. " * 500
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
            # For summarization, check if key points appear
            matches = sum(1 for e in expected if str(e).lower() in answer_lower)
            return matches >= max(1, len(expected) // 2)
        if isinstance(expected, int):
            numbers = re.findall(r"\d+", answer)
            return str(expected) in numbers
        return str(expected).lower() in answer_lower
