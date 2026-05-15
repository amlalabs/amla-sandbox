"""SCROLLS Benchmark - Long document NLP tasks.

Reference: https://github.com/tau-nlp/scrolls
Paper: EMNLP 2022 - "SCROLLS: Standardized CompaRison Over Long Language Sequences"
HuggingFace: https://huggingface.co/datasets/tau/scrolls

Tasks:
- gov_report: Government report summarization
- summ_screen_fd: TV episode summarization
- qmsum: Query-based meeting summarization
- narrative_qa: QA over stories
- qasper: QA over scientific papers
- quality: Multiple-choice QA over stories
- contract_nli: Legal NLI
"""

from __future__ import annotations

import random
import re

from .base import Benchmark, BenchmarkSample, register_benchmark


@register_benchmark("scrolls")
class ScrollsBenchmark(Benchmark):
    """SCROLLS: Standardized CompaRison Over Long Language Sequences."""

    name = "scrolls"
    description = "Long document NLP benchmark from EMNLP 2022"
    source_url = "https://github.com/tau-nlp/scrolls"
    paper_url = "https://aclanthology.org/2022.emnlp-main.823/"

    SUBSETS = [
        "gov_report",
        "summ_screen_fd",
        "qmsum",
        "narrative_qa",
        "qasper",
        "quality",
        "contract_nli",
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
                "tau/scrolls", subset, split="validation", trust_remote_code=True
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
            if subset == "gov_report":
                sample = self._gen_gov_report(i, rng)
            elif subset == "summ_screen_fd":
                sample = self._gen_tv_summary(i, rng)
            elif subset == "qmsum":
                sample = self._gen_meeting_summary(i, rng)
            elif subset == "narrative_qa":
                sample = self._gen_narrative_qa(i, rng)
            elif subset == "qasper":
                sample = self._gen_paper_qa(i, rng)
            elif subset == "quality":
                sample = self._gen_quality_mc(i, rng)
            elif subset == "contract_nli":
                sample = self._gen_contract_nli(i, rng)
            else:
                sample = self._gen_generic(subset, i, rng)
            samples.append(sample)

        return samples

    def _gen_gov_report(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate government report summarization."""
        topics = [
            "Infrastructure Investment",
            "Healthcare Reform",
            "Environmental Policy",
            "Economic Recovery",
        ]
        topic = rng.choice(topics)

        key_findings = [
            f"The report recommends $50 billion investment in {topic.lower()}.",
            "Implementation timeline spans 5 years.",
            "Bipartisan support is expected.",
        ]

        sections = []
        for i, finding in enumerate(key_findings):
            sections.append(f"Section {i + 1}: {finding}")
            for _ in range(rng.randint(10, 20)):
                sections.append(
                    f"Additional analysis of {topic.lower()} considerations. "
                    f"Various stakeholders have been consulted on this matter."
                )

        context = "\n\n".join(sections)
        query = "Summarize the key recommendations from this government report."

        return BenchmarkSample(
            id=f"gov_report_{idx}",
            context=context,
            query=query,
            expected=key_findings,
            metadata={"task": "gov_report", "source": "synthetic"},
        )

    def _gen_tv_summary(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate TV episode summarization."""
        characters = ["Sarah", "Mike", "Emma", "Jake"]
        plot_points = [
            f"{characters[0]} discovers a secret about {characters[1]}.",
            f"{characters[2]} makes a difficult decision.",
            f"The episode ends with {characters[3]} leaving town.",
        ]

        dialogue = []
        for _ in range(rng.randint(50, 100)):
            char = rng.choice(characters)
            dialogue.append(f"{char}: I can't believe what happened today.")
            dialogue.append(f"{char}: We need to figure this out.")

        # Insert plot points
        for i, point in enumerate(plot_points):
            pos = (i + 1) * len(dialogue) // 4
            dialogue.insert(pos, f"[{point}]")

        context = "\n".join(dialogue)
        query = "Summarize what happens in this TV episode."

        return BenchmarkSample(
            id=f"summ_screen_fd_{idx}",
            context=context,
            query=query,
            expected=plot_points,
            metadata={"task": "summ_screen_fd", "source": "synthetic"},
        )

    def _gen_meeting_summary(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate meeting summarization with query."""
        participants = ["Alice", "Bob", "Carol", "David"]
        topics = ["budget", "timeline", "resources", "risks"]

        topic = rng.choice(topics)
        decision = f"Approved {topic} increase of 20%"

        dialogue = []
        for _ in range(rng.randint(30, 60)):
            speaker = rng.choice(participants)
            dialogue.append(f"{speaker}: Let me add my thoughts on this matter.")

        # Add the key decision
        dialogue.insert(len(dialogue) // 2, f"[DECISION: {decision}]")

        context = "\n".join(dialogue)
        query = f"What was decided about the {topic}?"

        return BenchmarkSample(
            id=f"qmsum_{idx}",
            context=context,
            query=query,
            expected=decision,
            metadata={"task": "qmsum", "source": "synthetic"},
        )

    def _gen_narrative_qa(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate narrative QA."""
        protagonist = rng.choice(["John", "Mary", "Alex", "Sarah"])
        goal = rng.choice(
            [
                "find the treasure",
                "save the kingdom",
                "solve the mystery",
                "return home",
            ]
        )

        story_parts = [
            f"{protagonist} set out on a journey.",
            f"Their goal was to {goal}.",
            "Many challenges awaited along the way.",
            "But determination kept them going.",
        ]

        # Add filler
        for _ in range(50):
            story_parts.append("The path continued through unknown lands.")

        rng.shuffle(story_parts)
        context = "\n\n".join(story_parts)
        query = f"What was {protagonist}'s main goal?"

        return BenchmarkSample(
            id=f"narrative_qa_{idx}",
            context=context,
            query=query,
            expected=goal,
            metadata={"task": "narrative_qa", "source": "synthetic"},
        )

    def _gen_paper_qa(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate scientific paper QA."""
        method = rng.choice(["transformer", "CNN", "RNN", "attention mechanism"])
        metric = rng.choice(["F1 score", "accuracy", "BLEU", "perplexity"])
        result = f"{rng.randint(80, 99)}.{rng.randint(0, 9)}%"

        paper_sections = [
            f"Abstract: We propose a novel {method} for text classification.",
            f"Method: Our approach uses {method} with modifications.",
            f"Results: We achieve {result} {metric} on the benchmark.",
            "Conclusion: The proposed method shows promising results.",
        ]

        # Add filler
        for _ in range(30):
            paper_sections.append(
                "Related work has explored similar approaches with varying success."
            )

        rng.shuffle(paper_sections)
        context = "\n\n".join(paper_sections)
        query = f"What {metric} does the proposed method achieve?"

        return BenchmarkSample(
            id=f"qasper_{idx}",
            context=context,
            query=query,
            expected=result,
            metadata={"task": "qasper", "source": "synthetic"},
        )

    def _gen_quality_mc(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate multiple-choice story comprehension."""
        protagonist = rng.choice(["Elena", "Marcus", "Sophie", "James"])
        occupation = rng.choice(["scientist", "detective", "teacher", "artist"])

        story = f"""
{protagonist} was a renowned {occupation} who had dedicated their life to their work.
One day, something unexpected happened that would change everything.
The story unfolds as {protagonist} navigates this new challenge.
In the end, {protagonist} learned an important lesson about perseverance.
"""
        # Add filler
        filler = "\n".join(
            [f"Chapter {i}: More details about the journey..." for i in range(1, 21)]
        )

        context = story + filler
        query = f"What was {protagonist}'s occupation?\nA. Doctor\nB. {occupation.title()}\nC. Lawyer\nD. Engineer"

        return BenchmarkSample(
            id=f"quality_{idx}",
            context=context,
            query=query,
            expected="B",
            metadata={
                "task": "quality",
                "source": "synthetic",
                "answer_text": occupation,
            },
        )

    def _gen_contract_nli(self, idx: int, rng: random.Random) -> BenchmarkSample:
        """Generate contract NLI task."""
        clause_types = [
            (
                "termination requires 30 days notice",
                "The contract can be terminated immediately",
                "contradiction",
            ),
            (
                "payment is due within 60 days",
                "Payment must be made within 60 days",
                "entailment",
            ),
            (
                "the agreement covers software licensing",
                "Hardware warranties are included",
                "neutral",
            ),
        ]

        premise_fact, hypothesis, label = rng.choice(clause_types)

        contract = f"""
CONTRACT AGREEMENT

Section 1: Terms
This agreement establishes that {premise_fact}.

Section 2: Additional Terms
Various other standard clauses apply to this agreement.
"""
        # Add filler
        for i in range(3, 20):
            contract += f"\nSection {i}: Standard legal language and provisions.\n"

        context = contract
        query = f"Given the contract, does the following statement hold?\n\nStatement: {hypothesis}\n\nAnswer: entailment, contradiction, or neutral"

        return BenchmarkSample(
            id=f"contract_nli_{idx}",
            context=context,
            query=query,
            expected=label,
            metadata={"task": "contract_nli", "source": "synthetic"},
        )

    def _gen_generic(
        self, subset: str, idx: int, rng: random.Random
    ) -> BenchmarkSample:
        """Generate generic sample."""
        answer = f"answer_{rng.randint(100, 999)}"
        context = f"The answer is {answer}. " + "More content. " * 300
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
        answer_clean = answer.strip().upper()

        if isinstance(expected, list):
            # Summarization: check for key points
            matches = sum(1 for e in expected if str(e).lower() in answer_lower)
            return matches >= max(1, len(expected) // 2)
        if isinstance(expected, int):
            numbers = re.findall(r"\d+", answer)
            return str(expected) in numbers
        expected_str = str(expected)
        # Letter answer
        if len(expected_str) == 1 and expected_str.isalpha():
            return expected_str.upper() in answer_clean[:5]
        return expected_str.lower() in answer_lower
