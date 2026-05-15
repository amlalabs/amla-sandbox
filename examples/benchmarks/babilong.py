"""BABILong Benchmark - Reasoning-in-a-haystack at scale.

Reference: https://github.com/booydar/babilong
Paper: https://arxiv.org/abs/2406.10149

Tasks (20 bAbI tasks extended to long context):
- qa1: single supporting fact
- qa2: two supporting facts
- qa3: three supporting facts
- qa4: two arg relations
- qa5: three arg relations
- qa6: yes/no questions
- qa7: counting
- qa8: lists/sets
- qa9: simple negation
- qa10: indefinite knowledge
- qa11-qa20: more complex reasoning tasks
"""

from __future__ import annotations

import random
import re

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("babilong")
class BabilongBenchmark(Benchmark):
    """BABILong: Testing the Limits of LLMs with Long Context Reasoning."""

    name = "babilong"
    description = "Reasoning-in-a-haystack benchmark from bAbI tasks"
    source_url = "https://github.com/booydar/babilong"
    paper_url = "https://arxiv.org/abs/2406.10149"

    SUBSETS = [f"qa{i}" for i in range(1, 21)]

    # Background text from PG19 style
    FILLER_SENTENCES = [
        "The morning sun cast long shadows across the garden.",
        "She walked slowly through the empty corridors of the old building.",
        "The clock on the wall ticked steadily, marking each passing second.",
        "Outside, the wind rustled through the autumn leaves.",
        "He remembered the days when everything seemed simpler.",
        "The city lights flickered in the distance like earthbound stars.",
        "Time passed differently here, in this quiet place.",
        "The old photographs told stories of people long forgotten.",
        "Rain began to fall gently against the window panes.",
        "Somewhere in the house, a door creaked open.",
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

        subsets_to_load = [subset] if subset else ["qa1", "qa2", "qa3", "qa6", "qa7"]

        # Test at different context lengths
        context_lengths = [4000, 8000, 16000, 32000]

        for task in subsets_to_load:
            for length in context_lengths:
                sample = self._generate_sample(task, length, rng)
                if sample:
                    samples.append(sample)

        if num_samples and len(samples) > num_samples:
            rng.shuffle(samples)
            samples = samples[:num_samples]

        return samples

    def _generate_filler(self, target_chars: int, rng: random.Random) -> list[str]:
        """Generate filler sentences."""
        sentences = []
        current_len = 0
        while current_len < target_chars:
            sent = rng.choice(self.FILLER_SENTENCES)
            sentences.append(sent)
            current_len += len(sent) + 1
        return sentences

    def _generate_sample(
        self, task: str, context_length: int, rng: random.Random
    ) -> BenchmarkSample | None:
        """Generate a sample for the given task."""
        task_num = int(task.replace("qa", ""))

        if task_num == 1:
            return self._gen_qa1(context_length, rng)
        if task_num == 2:
            return self._gen_qa2(context_length, rng)
        if task_num == 3:
            return self._gen_qa3(context_length, rng)
        if task_num == 6:
            return self._gen_qa6(context_length, rng)
        if task_num == 7:
            return self._gen_qa7(context_length, rng)
        if task_num == 8:
            return self._gen_qa8(context_length, rng)
        # Default to qa1-style for unimplemented tasks
        return self._gen_qa1(context_length, rng)

    def _gen_qa1(self, context_length: int, rng: random.Random) -> BenchmarkSample:
        """Single supporting fact."""
        names = ["Mary", "John", "Sandra", "Daniel"]
        locations = ["garden", "kitchen", "bedroom", "office", "bathroom"]

        person = rng.choice(names)
        location = rng.choice(locations)

        fact = f"{person} went to the {location}."
        question = f"Where is {person}?"
        answer = location

        # Build context with fact hidden in filler
        filler = self._generate_filler(context_length, rng)
        insert_pos = len(filler) // 2
        filler.insert(insert_pos, fact)

        context = " ".join(filler)

        return BenchmarkSample(
            id=f"qa1_{context_length}",
            context=context,
            query=question,
            expected=answer,
            metadata={"task": "qa1", "context_length": context_length},
        )

    def _gen_qa2(self, context_length: int, rng: random.Random) -> BenchmarkSample:
        """Two supporting facts."""
        names = ["Mary", "John", "Sandra", "Daniel"]
        objects = ["football", "apple", "milk", "book"]
        locations = ["garden", "kitchen", "bedroom", "office"]

        person = rng.choice(names)
        obj = rng.choice(objects)
        location = rng.choice(locations)

        fact1 = f"{person} picked up the {obj}."
        fact2 = f"{person} went to the {location}."
        question = f"Where is the {obj}?"
        answer = location

        filler = self._generate_filler(context_length, rng)
        pos1 = len(filler) // 3
        pos2 = 2 * len(filler) // 3
        filler.insert(pos1, fact1)
        filler.insert(pos2 + 1, fact2)

        context = " ".join(filler)

        return BenchmarkSample(
            id=f"qa2_{context_length}",
            context=context,
            query=question,
            expected=answer,
            metadata={"task": "qa2", "context_length": context_length},
        )

    def _gen_qa3(self, context_length: int, rng: random.Random) -> BenchmarkSample:
        """Three supporting facts."""
        names = ["Mary", "John", "Sandra"]
        objects = ["football", "apple", "milk"]
        locations = ["garden", "kitchen", "bedroom", "office", "hallway"]

        person1 = names[0]
        person2 = names[1]
        obj = rng.choice(objects)
        loc1 = locations[0]
        loc2 = locations[1]
        loc3 = locations[2]

        facts = [
            f"{person1} picked up the {obj}.",
            f"{person1} went to the {loc1}.",
            f"{person1} gave the {obj} to {person2}.",
            f"{person2} went to the {loc2}.",
            f"{person2} dropped the {obj}.",
            f"{person2} went to the {loc3}.",
        ]

        question = f"Where is the {obj}?"
        answer = loc2  # Where it was dropped

        filler = self._generate_filler(context_length, rng)
        positions = sorted(rng.sample(range(len(filler)), len(facts)))
        for i, pos in enumerate(positions):
            filler.insert(pos + i, facts[i])

        context = " ".join(filler)

        return BenchmarkSample(
            id=f"qa3_{context_length}",
            context=context,
            query=question,
            expected=answer,
            metadata={"task": "qa3", "context_length": context_length},
        )

    def _gen_qa6(self, context_length: int, rng: random.Random) -> BenchmarkSample:
        """Yes/No questions."""
        names = ["Mary", "John", "Sandra", "Daniel"]
        locations = ["garden", "kitchen", "bedroom", "office"]

        person = rng.choice(names)
        actual_location = rng.choice(locations)
        asked_location = rng.choice(locations)

        fact = f"{person} is in the {actual_location}."
        question = f"Is {person} in the {asked_location}?"
        answer = "yes" if actual_location == asked_location else "no"

        filler = self._generate_filler(context_length, rng)
        insert_pos = len(filler) // 2
        filler.insert(insert_pos, fact)

        context = " ".join(filler)

        return BenchmarkSample(
            id=f"qa6_{context_length}",
            context=context,
            query=question,
            expected=answer,
            metadata={"task": "qa6", "context_length": context_length},
        )

    def _gen_qa7(self, context_length: int, rng: random.Random) -> BenchmarkSample:
        """Counting."""
        names = ["Mary", "John", "Sandra", "Daniel"]
        objects = ["apple", "orange", "football", "milk"]

        person = rng.choice(names)
        obj = rng.choice(objects)

        # Generate pickup/drop events
        count = rng.randint(1, 4)
        drops = rng.randint(0, count - 1)
        final_count = count - drops

        facts = []
        for _ in range(count):
            facts.append(f"{person} picked up the {obj}.")
        for _ in range(drops):
            facts.append(f"{person} dropped the {obj}.")

        rng.shuffle(facts)

        question = f"How many objects is {person} carrying?"
        answer = str(final_count)

        filler = self._generate_filler(context_length, rng)
        positions = sorted(rng.sample(range(len(filler)), len(facts)))
        for i, pos in enumerate(positions):
            filler.insert(pos + i, facts[i])

        context = " ".join(filler)

        return BenchmarkSample(
            id=f"qa7_{context_length}",
            context=context,
            query=question,
            expected=answer,
            metadata={
                "task": "qa7",
                "context_length": context_length,
                "final_count": final_count,
            },
        )

    def _gen_qa8(self, context_length: int, rng: random.Random) -> BenchmarkSample:
        """Lists/Sets."""
        names = ["Mary", "John", "Sandra", "Daniel"]
        all_objects = ["apple", "orange", "football", "milk", "book", "keys"]

        person = rng.choice(names)
        num_objects = rng.randint(2, 4)
        objects = rng.sample(all_objects, num_objects)

        facts = [f"{person} picked up the {obj}." for obj in objects]

        question = f"What is {person} carrying?"
        answer = objects  # All should be mentioned

        filler = self._generate_filler(context_length, rng)
        positions = sorted(rng.sample(range(len(filler)), len(facts)))
        for i, pos in enumerate(positions):
            filler.insert(pos + i, facts[i])

        context = " ".join(filler)

        return BenchmarkSample(
            id=f"qa8_{context_length}",
            context=context,
            query=question,
            expected=answer,
            metadata={"task": "qa8", "context_length": context_length},
        )

    def check_answer(
        self, answer: str | None, expected: str | list[str] | int | float | None
    ) -> bool:
        if answer is None:
            return False

        answer_lower = answer.lower()

        if isinstance(expected, list):
            # All items must be present
            return all(str(e).lower() in answer_lower for e in expected)
        if isinstance(expected, int):
            numbers = re.findall(r"\d+", answer)
            return str(expected) in numbers
        expected_lower = str(expected).lower()
        return expected_lower in answer_lower
