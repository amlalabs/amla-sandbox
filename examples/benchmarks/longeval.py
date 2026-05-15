"""LongEval Benchmark - Topic and line retrieval tasks.

Reference: https://lmsys.org/blog/2023-06-29-longchat/
Paper: Associated with LongChat models from LMSYS

Tasks:
- topic_retrieval: Retrieve the first topic from a multi-topic conversation
- line_retrieval: Retrieve a specific number from a long document
"""

from __future__ import annotations

import random
import re
import string

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("longeval")
class LongEvalBenchmark(Benchmark):
    """LongEval: Topic and line retrieval for long-context evaluation."""

    name = "longeval"
    description = "LMSYS topic and line retrieval benchmark"
    source_url = "https://lmsys.org/blog/2023-06-29-longchat/"
    paper_url = "https://lmsys.org/blog/2023-06-29-longchat/"

    SUBSETS = [
        "topic_retrieval",
        "line_retrieval",
    ]

    # Test at different lengths
    CONTEXT_LENGTHS = [2000, 4000, 8000, 16000, 32000]

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

        subsets_to_load = [subset] if subset else self.SUBSETS

        for task in subsets_to_load:
            for length in self.CONTEXT_LENGTHS:
                sample = self._generate_sample(task, length, rng)
                if sample:
                    samples.append(sample)

        if num_samples and len(samples) > num_samples:
            rng.shuffle(samples)
            samples = samples[:num_samples]

        return samples

    def _generate_sample(
        self, task: str, context_length: int, rng: random.Random
    ) -> BenchmarkSample | None:
        """Generate a sample for the given task."""
        if task == "topic_retrieval":
            return self._gen_topic_retrieval(context_length, rng)
        if task == "line_retrieval":
            return self._gen_line_retrieval(context_length, rng)
        return None

    def _gen_topic_retrieval(
        self, context_length: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate topic retrieval task.

        Creates a multi-topic conversation where the model must identify
        the first topic discussed.
        """
        topics = [
            "artificial intelligence",
            "climate change",
            "space exploration",
            "renewable energy",
            "quantum computing",
            "machine learning",
            "sustainable agriculture",
            "autonomous vehicles",
        ]

        num_topics = min(len(topics), max(3, context_length // 2000))
        selected_topics = rng.sample(topics, num_topics)
        first_topic = selected_topics[0]

        # Build conversation
        messages = []
        chars_per_topic = context_length // num_topics

        for topic in selected_topics:
            # Topic introduction
            messages.append(f"User: I'd like to discuss {topic}.")
            messages.append(
                f"Assistant: Great! {topic.title()} is a fascinating area. "
                f"There are many interesting aspects to explore."
            )

            # Add turns until we reach target length for this topic
            current_len = sum(len(m) for m in messages)
            while current_len < chars_per_topic * (selected_topics.index(topic) + 1):
                messages.append(f"User: Can you tell me more about {topic}?")
                messages.append(
                    f"Assistant: Certainly! {topic.title()} has evolved significantly "
                    f"over the years. Research continues to advance our understanding. "
                    f"Many experts believe this field will transform society. "
                    f"Let me share some key developments in {topic}."
                )
                current_len = sum(len(m) for m in messages)

        context = "\n".join(messages)
        query = "What was the FIRST topic discussed in this conversation? Answer with just the topic name."

        return BenchmarkSample(
            id=f"topic_retrieval_{context_length}",
            context=context,
            query=query,
            expected=first_topic,
            metadata={
                "task": "topic_retrieval",
                "context_length": context_length,
                "num_topics": num_topics,
            },
        )

    def _gen_line_retrieval(
        self, context_length: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate line retrieval task.

        Creates a document with numbered lines where the model must find
        a specific line's content.
        """
        # Generate lines
        num_lines = context_length // 50  # ~50 chars per line
        lines = []

        # Target line with special content
        target_line_num = rng.randint(num_lines // 4, 3 * num_lines // 4)
        target_value = "".join(rng.choices(string.digits, k=8))

        for i in range(num_lines):
            if i == target_line_num:
                lines.append(f"line {i + 1}: The special number is {target_value}")
            else:
                # Random filler content
                content = "".join(rng.choices(string.ascii_lowercase + " ", k=40))
                lines.append(f"line {i + 1}: {content.strip()}")

        context = "\n".join(lines)
        query = f"What is the special number mentioned on line {target_line_num + 1}?"

        return BenchmarkSample(
            id=f"line_retrieval_{context_length}",
            context=context,
            query=query,
            expected=target_value,
            metadata={
                "task": "line_retrieval",
                "context_length": context_length,
                "target_line": target_line_num + 1,
            },
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
