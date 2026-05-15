"""L-Eval Benchmark - Standardized Long Context Evaluation.

Reference: https://github.com/OpenLMLab/LEval
Paper: https://arxiv.org/abs/2307.11088

Tasks (20 sub-tasks across multiple domains):
- Closed-ended: coursera, quality, topic_retrieval, sci_fi, etc.
- Open-ended: legal_contract, financial_qa, patent, etc.
- Summarization: meeting, news, paper, review, etc.
"""

from __future__ import annotations

import random
import re

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("leval")
class LEvalBenchmark(Benchmark):
    """L-Eval: Standardized Evaluation for Long Context Language Models."""

    name = "leval"
    description = "Standardized long-context LLM evaluation benchmark"
    source_url = "https://github.com/OpenLMLab/LEval"
    paper_url = "https://arxiv.org/abs/2307.11088"

    SUBSETS = [
        # Closed-ended tasks
        "coursera",
        "quality",
        "topic_retrieval_longchat",
        "sci_fi",
        "gsm100",
        "codeU",
        "tpo",
        # Open-ended tasks
        "legal_contract_qa",
        "financial_qa",
        "natural_question",
        "narrative_qa",
        "multidoc_qa",
        # Summarization
        "meeting_summ",
        "news_summ",
        "paper_summ",
        "review_summ",
        "gov_report_summ",
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

        subsets_to_load = [subset] if subset else self.SUBSETS[:5]

        for task in subsets_to_load:
            task_samples = self._generate_synthetic(task, num_samples or 3, rng)
            samples.extend(task_samples)

        if num_samples and len(samples) > num_samples:
            rng.shuffle(samples)
            samples = samples[:num_samples]

        return samples

    def _generate_synthetic(
        self, subset: str, num_samples: int, rng: random.Random
    ) -> list[BenchmarkSample]:
        """Generate synthetic samples."""
        samples = []

        for i in range(num_samples):
            if "summ" in subset:
                sample = self._gen_summarization(subset, i, rng)
            elif subset in ["coursera", "quality", "tpo"]:
                sample = self._gen_multiple_choice(subset, i, rng)
            elif subset == "topic_retrieval_longchat":
                sample = self._gen_topic_retrieval(i, rng)
            elif subset == "gsm100":
                sample = self._gen_math(i, rng)
            elif subset == "codeU":
                sample = self._gen_code(i, rng)
            elif "qa" in subset:
                sample = self._gen_qa(subset, i, rng)
            else:
                sample = self._gen_generic(subset, i, rng)
            samples.append(sample)

        return samples

    def _gen_multiple_choice(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate multiple choice question."""
        topics = {
            "coursera": [
                (
                    "What is machine learning?",
                    "A field of AI",
                    ["Statistics", "A field of AI", "Database", "Networking"],
                ),
                (
                    "What is gradient descent?",
                    "An optimization algorithm",
                    [
                        "A data structure",
                        "An optimization algorithm",
                        "A sorting method",
                        "A search algorithm",
                    ],
                ),
            ],
            "quality": [
                (
                    "What is the main theme?",
                    "Personal growth",
                    ["Revenge", "Personal growth", "Adventure", "Romance"],
                ),
            ],
            "tpo": [
                (
                    "According to the lecture, what is the main topic?",
                    "Climate change",
                    ["Economics", "Climate change", "History", "Literature"],
                ),
            ],
        }

        questions = topics.get(subset, topics["coursera"])
        question, answer, choices = rng.choice(questions)

        # Build context
        context_parts = [
            f"Today's lecture covers important concepts in {subset}.",
            "We will explore various aspects of this topic.",
        ]

        # Add the answer somewhere
        context_parts.append(f"The key point is that the answer is: {answer}.")

        # Add filler
        for _ in range(20):
            context_parts.append(
                "Additional material is covered in the supplementary readings."
            )

        rng.shuffle(context_parts)
        context = "\n\n".join(context_parts)

        choices_str = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
        query = f"{question}\n\n{choices_str}\n\nAnswer with the letter only."

        correct_letter = chr(65 + choices.index(answer))

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=correct_letter,
            metadata={"task": subset, "source": "synthetic", "answer_text": answer},
        )

    def _gen_topic_retrieval(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate topic retrieval task (LongChat style)."""
        topics = [
            "artificial intelligence",
            "climate change",
            "space exploration",
            "renewable energy",
            "quantum computing",
        ]

        num_topics = rng.randint(3, 5)
        selected_topics = rng.sample(topics, num_topics)
        first_topic = selected_topics[0]

        # Create conversation
        messages = []
        for topic in selected_topics:
            messages.append(f"User: Let's discuss {topic}.")
            messages.append(
                f"Assistant: {topic.title()} is a fascinating subject. There are many aspects to consider..."
            )
            # Add several turns per topic
            for _ in range(rng.randint(3, 6)):
                messages.append("User: Can you tell me more about this?")
                messages.append(
                    f"Assistant: Certainly! Here are more details about {topic}..."
                )

        context = "\n".join(messages)
        query = "What was the first topic discussed in this conversation?"

        return BenchmarkSample(
            id=f"topic_retrieval_{idx}",
            context=context,
            query=query,
            expected=first_topic,
            metadata={"task": "topic_retrieval", "source": "synthetic"},
        )

    def _gen_math(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate math problem (GSM100 style)."""
        a = rng.randint(10, 50)
        b = rng.randint(5, 20)
        c = rng.randint(2, 10)

        problem = f"""
Janet's ducks lay {a} eggs per day. She eats {b} for breakfast every morning and
bakes muffins for her friends every day with {c} eggs. She sells the remainder
at the farmers' market daily for $2 per egg. How much in dollars does she make
every day at the farmers' market?
"""
        remainder = a - b - c
        answer = remainder * 2

        # Add filler context
        filler = "More information about Janet's farm...\n" * 50
        context = problem + "\n" + filler

        query = "Answer the math problem. Provide just the final number."

        return BenchmarkSample(
            id=f"gsm100_{idx}",
            context=context,
            query=query,
            expected=str(answer),
            metadata={"task": "gsm100", "source": "synthetic"},
        )

    def _gen_code(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate code understanding task."""
        funcs = [
            ("def add(a, b): return a + b", "add", "adds two numbers"),
            ("def multiply(x, y): return x * y", "multiply", "multiplies two numbers"),
            ("def greet(name): return f'Hello, {name}'", "greet", "returns a greeting"),
        ]

        code, func_name, description = rng.choice(funcs)

        # Add filler code
        filler_code = "\n\n".join([f"def helper_{i}(): pass" for i in range(50)])

        context = code + "\n\n" + filler_code
        query = f"What does the function '{func_name}' do?"

        return BenchmarkSample(
            id=f"codeU_{idx}",
            context=context,
            query=query,
            expected=description,
            metadata={"task": "codeU", "source": "synthetic"},
        )

    def _gen_summarization(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate summarization task."""
        doc_type = subset.replace("_summ", "")

        key_points = [
            f"The {doc_type} discusses recent developments.",
            "Key stakeholders were consulted.",
            "Recommendations were proposed.",
        ]

        paragraphs = []
        for point in key_points:
            paragraphs.append(point)
            for _ in range(rng.randint(5, 10)):
                paragraphs.append(
                    f"Additional context about the {doc_type} is provided here. "
                    f"Various perspectives and considerations are explored in detail."
                )

        context = "\n\n".join(paragraphs)
        query = f"Summarize the main points of this {doc_type}."

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=key_points,
            metadata={"task": subset, "source": "synthetic"},
        )

    def _gen_qa(self, subset: str, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate QA task."""
        facts = {
            "legal_contract_qa": ("termination clause", "30 days notice"),
            "financial_qa": ("quarterly revenue", "$50 million"),
            "natural_question": ("capital of France", "Paris"),
            "narrative_qa": ("protagonist's goal", "find the treasure"),
            "multidoc_qa": ("main conclusion", "further research needed"),
        }

        topic, answer = facts.get(subset, ("key finding", "positive result"))

        context_parts = [
            f"This document discusses the {topic}.",
            f"The {topic} is: {answer}.",
        ]

        for _ in range(30):
            context_parts.append(
                "Additional information and context is provided throughout the document."
            )

        rng.shuffle(context_parts)
        context = "\n\n".join(context_parts)
        query = f"What is the {topic}?"

        return BenchmarkSample(
            id=f"{subset}_{idx}",
            context=context,
            query=query,
            expected=answer,
            metadata={"task": subset, "source": "synthetic"},
        )

    def _gen_generic(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate generic sample."""
        answer = f"answer_{rng.randint(100, 999)}"
        context = f"The answer is {answer}. " + "More content here. " * 200
        query = "What is the answer mentioned in the document?"

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

        answer_clean = answer.strip().upper()
        answer_lower = answer.lower()

        if isinstance(expected, list):
            # Summarization: check for key points
            matches = sum(1 for e in expected if str(e).lower() in answer_lower)
            return matches >= max(1, len(expected) // 2)
        if isinstance(expected, int):
            numbers = re.findall(r"\d+", answer)
            return str(expected) in numbers
        expected_str = str(expected)
        # Check for letter answers (A, B, C, D)
        if len(expected_str) == 1 and expected_str.isalpha():
            return expected_str.upper() in answer_clean[:5]
        return expected_str.lower() in answer_lower
