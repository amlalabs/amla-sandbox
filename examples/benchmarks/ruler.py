"""RULER Benchmark - NVIDIA's long-context evaluation.

Reference: https://github.com/NVIDIA/RULER
Paper: https://arxiv.org/abs/2404.06654

Tasks:
- NIAH (Needle-in-a-Haystack): Single/Multi/Multi-Key/Multi-Value/Multi-Query
- VT (Variable Tracking): Simple/Complex
- CWE (Common Words Extraction)
- FWE (Frequent Words Extraction)
- QA: Single-hop/Multi-hop

This implementation generates synthetic test data matching RULER's methodology.
"""

from __future__ import annotations

import random
import re
import string

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("ruler")
class RulerBenchmark(Benchmark):
    """RULER: What's the Real Context Size of Your Long-Context Language Models?"""

    name = "ruler"
    description = "NVIDIA's comprehensive long-context benchmark"
    source_url = "https://github.com/NVIDIA/RULER"
    paper_url = "https://arxiv.org/abs/2404.06654"

    # Realistic filler paragraphs (from Paul Graham essays style)
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
        "Integration tests verify correct behavior across service boundaries and failure scenarios.",
        "The configuration management system supports environment-specific overrides.",
        "Logging infrastructure captures detailed traces for debugging and audit purposes.",
        "Rate limiting protects backend services from excessive load during peak traffic.",
        "The authentication flow implements industry-standard security protocols.",
    ]

    def list_subsets(self) -> list[str]:
        return [
            "niah_single",
            "niah_multi",
            "niah_multikey",
            "niah_multivalue",
            "niah_multiquery",
            "vt_simple",
            "vt_complex",
            "cwe",
            "fwe",
            "qa_single",
            "qa_multi",
        ]

    async def load_samples(
        self,
        subset: str | None = None,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        rng = random.Random(seed)

        # Default context sizes to test
        context_sizes = [4000, 8000, 16000, 32000]

        subsets_to_run = [subset] if subset else self.list_subsets()

        for task in subsets_to_run:
            for size in context_sizes:
                sample = self._generate_sample(task, size, rng)
                if sample:
                    samples.append(sample)

        if num_samples and len(samples) > num_samples:
            rng.shuffle(samples)
            samples = samples[:num_samples]

        return samples

    def _generate_sample(
        self, task: str, context_size: int, rng: random.Random
    ) -> BenchmarkSample | None:
        """Generate a single sample for a task."""
        if task == "niah_single":
            return self._gen_niah_single(context_size, rng)
        if task == "niah_multi":
            return self._gen_niah_multi(context_size, 5, rng)
        if task == "niah_multikey":
            return self._gen_niah_multikey(context_size, rng)
        if task == "niah_multivalue":
            return self._gen_niah_multivalue(context_size, rng)
        if task == "niah_multiquery":
            return self._gen_niah_multiquery(context_size, rng)
        if task == "vt_simple":
            return self._gen_vt_simple(context_size, rng)
        if task == "vt_complex":
            return self._gen_vt_complex(context_size, rng)
        if task == "cwe":
            return self._gen_cwe(context_size, rng)
        if task == "fwe":
            return self._gen_fwe(context_size, rng)
        if task == "qa_single":
            return self._gen_qa_single(context_size, rng)
        if task == "qa_multi":
            return self._gen_qa_multi(context_size, rng)
        return None

    def _generate_haystack(self, target_chars: int, rng: random.Random) -> str:
        """Generate filler text of approximately target_chars length."""
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
        cities = ["Paris", "Tokyo", "London", "Sydney", "Cairo", "Berlin", "Madrid"]
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

    def _gen_niah_multikey(
        self, context_size: int, rng: random.Random
    ) -> BenchmarkSample:
        """Multi-key retrieval - find values for multiple different keys."""
        keys = {
            "project_id": "".join(rng.choices(string.ascii_uppercase, k=6)),
            "version": f"{rng.randint(1, 9)}.{rng.randint(0, 9)}.{rng.randint(0, 99)}",
            "api_key": "".join(rng.choices(string.ascii_letters + string.digits, k=16)),
        }

        needles = [f"The {k.replace('_', ' ')} is {v}." for k, v in keys.items()]

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")

        for i, needle in enumerate(needles):
            pos = (i + 1) * len(lines) // 4
            lines.insert(pos + i, f"\n[CONFIG]\n{needle}\n[/CONFIG]\n")

        context = "\n\n".join(lines)
        query = "Find all configuration values in the document. Return: project_id, version, and api_key."
        expected = list(keys.values())

        return BenchmarkSample(
            id=f"niah_multikey_{context_size}",
            context=context,
            query=query,
            expected=expected,
            metadata={"task": "niah_multikey", "context_size": context_size},
        )

    def _gen_niah_multivalue(
        self, context_size: int, rng: random.Random
    ) -> BenchmarkSample:
        """Multi-value retrieval - find multiple values for the same key."""
        values = ["".join(rng.choices(string.digits, k=6)) for _ in range(4)]
        needles = [f"Transaction ID: {v}" for v in values]

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")

        for i, needle in enumerate(needles):
            pos = (i + 1) * len(lines) // 5
            lines.insert(pos + i, f"\n[TRANSACTION]\n{needle}\n[/TRANSACTION]\n")

        context = "\n\n".join(lines)
        query = "List ALL transaction IDs found in the document."
        expected = values

        return BenchmarkSample(
            id=f"niah_multivalue_{context_size}",
            context=context,
            query=query,
            expected=expected,
            metadata={"task": "niah_multivalue", "context_size": context_size},
        )

    def _gen_niah_multiquery(
        self, context_size: int, rng: random.Random
    ) -> BenchmarkSample:
        """Multi-query - answer multiple questions about the same context."""
        data = {
            "ceo_name": rng.choice(["Alice Chen", "Bob Smith", "Carol Davis"]),
            "revenue": f"${rng.randint(10, 999)} million",
            "employees": str(rng.randint(1000, 50000)),
        }

        needles = [
            f"The CEO is {data['ceo_name']}.",
            f"Annual revenue reached {data['revenue']}.",
            f"The company employs {data['employees']} people.",
        ]

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")

        for i, needle in enumerate(needles):
            pos = (i + 1) * len(lines) // 4
            lines.insert(pos + i, f"\n{needle}\n")

        context = "\n\n".join(lines)
        query = "Answer: 1) Who is the CEO? 2) What is the annual revenue? 3) How many employees?"
        expected = list(data.values())

        return BenchmarkSample(
            id=f"niah_multiquery_{context_size}",
            context=context,
            query=query,
            expected=expected,
            metadata={"task": "niah_multiquery", "context_size": context_size},
        )

    def _gen_vt_simple(self, context_size: int, rng: random.Random) -> BenchmarkSample:
        """Simple variable tracking - track value through assignments."""
        var_name = rng.choice(["x", "value", "result", "data"])
        operations = []
        current_value = rng.randint(1, 100)
        operations.append(f"Set {var_name} = {current_value}")

        for _ in range(rng.randint(3, 6)):
            op = rng.choice(["add", "subtract", "multiply"])
            operand = rng.randint(1, 20)
            if op == "add":
                current_value += operand
                operations.append(f"Add {operand} to {var_name}")
            elif op == "subtract":
                current_value -= operand
                operations.append(f"Subtract {operand} from {var_name}")
            else:
                current_value *= operand
                operations.append(f"Multiply {var_name} by {operand}")

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")

        for i, op in enumerate(operations):
            pos = (i + 1) * len(lines) // (len(operations) + 1)
            lines.insert(pos + i, f"\n[INSTRUCTION] {op} [/INSTRUCTION]\n")

        context = "\n\n".join(lines)
        query = f"Follow all instructions in order. What is the final value of {var_name}? Return ONLY the number."

        return BenchmarkSample(
            id=f"vt_simple_{context_size}",
            context=context,
            query=query,
            expected=current_value,
            metadata={"task": "vt_simple", "context_size": context_size},
        )

    def _gen_vt_complex(self, context_size: int, rng: random.Random) -> BenchmarkSample:
        """Complex variable tracking - multiple variables with dependencies."""
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        operations = [f"Set A = {a}", f"Set B = {b}"]

        for _ in range(3):
            op = rng.choice(["swap", "add_to_a", "add_to_b"])
            if op == "swap":
                a, b = b, a
                operations.append("Swap A and B")
            elif op == "add_to_a":
                a += b
                operations.append("Add B to A")
            else:
                b += a
                operations.append("Add A to B")

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")

        for i, op in enumerate(operations):
            pos = (i + 1) * len(lines) // (len(operations) + 1)
            lines.insert(pos + i, f"\n[STEP] {op} [/STEP]\n")

        context = "\n\n".join(lines)
        query = "Follow all steps in order. What are the final values of A and B? Format: A=X, B=Y"

        return BenchmarkSample(
            id=f"vt_complex_{context_size}",
            context=context,
            query=query,
            expected=[str(a), str(b)],
            metadata={
                "task": "vt_complex",
                "context_size": context_size,
                "a": a,
                "b": b,
            },
        )

    def _gen_cwe(self, context_size: int, rng: random.Random) -> BenchmarkSample:
        """Common Words Extraction - find words that appear in multiple lists."""
        all_words = [
            "apple",
            "banana",
            "cherry",
            "date",
            "elderberry",
            "fig",
            "grape",
            "honeydew",
            "kiwi",
            "lemon",
            "mango",
            "nectarine",
        ]

        # Select 3-4 common words that will appear in all lists
        common_words = rng.sample(all_words, 3)

        # Create multiple lists with different additional words
        lists = []
        for i in range(4):
            other_words = [w for w in all_words if w not in common_words]
            list_words = common_words + rng.sample(other_words, rng.randint(2, 4))
            rng.shuffle(list_words)
            lists.append(f"List {i + 1}: {', '.join(list_words)}")

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")

        for i, lst in enumerate(lists):
            pos = (i + 1) * len(lines) // 5
            lines.insert(pos + i, f"\n[DATA]\n{lst}\n[/DATA]\n")

        context = "\n\n".join(lines)
        query = "Find the words that appear in ALL lists. Return them comma-separated."

        return BenchmarkSample(
            id=f"cwe_{context_size}",
            context=context,
            query=query,
            expected=common_words,
            metadata={"task": "cwe", "context_size": context_size},
        )

    def _gen_fwe(self, context_size: int, rng: random.Random) -> BenchmarkSample:
        """Frequent Words Extraction - find the most frequent word."""
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        target_word = rng.choice(words)
        target_count = rng.randint(7, 12)

        haystack = self._generate_haystack(context_size, rng)

        # Insert target word more frequently
        for word in words:
            count = target_count if word == target_word else rng.randint(1, 4)
            for _ in range(count):
                pos = rng.randint(0, len(haystack) - 1)
                haystack = haystack[:pos] + f" {word} " + haystack[pos:]

        context = haystack
        query = f"Among the words {', '.join(words)}, which one appears most frequently? Return ONLY the word."

        return BenchmarkSample(
            id=f"fwe_{context_size}",
            context=context,
            query=query,
            expected=target_word,
            metadata={
                "task": "fwe",
                "context_size": context_size,
                "target_count": target_count,
            },
        )

    def _gen_qa_single(self, context_size: int, rng: random.Random) -> BenchmarkSample:
        """Single-hop QA - answer requires finding one fact."""
        facts = [
            (
                "The capital of France is Paris.",
                "What is the capital of France?",
                "Paris",
            ),
            (
                "Mount Everest is 8,849 meters tall.",
                "How tall is Mount Everest?",
                "8,849",
            ),
            (
                "The Amazon River is the longest river in South America.",
                "What is the longest river in South America?",
                "Amazon",
            ),
            (
                "Marie Curie won the Nobel Prize in Physics in 1903.",
                "When did Marie Curie win the Nobel Prize in Physics?",
                "1903",
            ),
        ]

        fact, question, answer = rng.choice(facts)

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")
        insert_pos = len(lines) // 2
        lines.insert(insert_pos, f"\n[FACT]\n{fact}\n[/FACT]\n")

        context = "\n\n".join(lines)

        return BenchmarkSample(
            id=f"qa_single_{context_size}",
            context=context,
            query=question,
            expected=answer,
            metadata={"task": "qa_single", "context_size": context_size},
        )

    def _gen_qa_multi(self, context_size: int, rng: random.Random) -> BenchmarkSample:
        """Multi-hop QA - answer requires combining multiple facts."""
        # Fact 1: Person -> City
        # Fact 2: City -> Country
        # Question: What country does Person live in?

        people = ["Alice", "Bob", "Charlie", "Diana"]
        cities = [
            ("Paris", "France"),
            ("Tokyo", "Japan"),
            ("London", "England"),
            ("Berlin", "Germany"),
        ]

        person = rng.choice(people)
        city, country = rng.choice(cities)

        fact1 = f"{person} lives in {city}."
        fact2 = f"{city} is a city in {country}."

        haystack = self._generate_haystack(context_size, rng)
        lines = haystack.split("\n\n")

        pos1 = len(lines) // 3
        pos2 = 2 * len(lines) // 3
        lines.insert(pos1, f"\n{fact1}\n")
        lines.insert(pos2 + 1, f"\n{fact2}\n")

        context = "\n\n".join(lines)
        query = f"In which country does {person} live?"

        return BenchmarkSample(
            id=f"qa_multi_{context_size}",
            context=context,
            query=query,
            expected=country,
            metadata={
                "task": "qa_multi",
                "context_size": context_size,
                "person": person,
                "city": city,
            },
        )

    def check_answer(
        self, answer: str | None, expected: str | list[str] | int | float | None
    ) -> bool:
        """Check if answer matches expected."""
        if answer is None:
            return False

        answer_lower = answer.lower()

        if isinstance(expected, list):
            # All expected values must be present
            return all(str(e).lower() in answer_lower for e in expected)
        if isinstance(expected, int):
            # Extract numbers and check
            numbers = re.findall(r"-?\d+", answer)
            return str(expected) in numbers
        if isinstance(expected, float):
            numbers = re.findall(r"-?\d+\.?\d*", answer)
            return any(abs(float(n) - expected) < 0.01 for n in numbers)
        return str(expected).lower() in answer_lower
