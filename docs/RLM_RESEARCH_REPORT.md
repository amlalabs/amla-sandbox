# Recursive Language Models for Long-Context Processing: Implementation and Evaluation

**Version**: 1.0
**Date**: January 2026
**Authors**: Amla Labs

---

## Abstract

We implement and evaluate the Recursive Language Model (RLM) strategy proposed by Zhang, Kraska, and Khattab (2025) for inference-time scaling of long-context language model tasks. Our implementation exposes document context through a Python REPL environment, allowing models to programmatically search, slice, and analyze content rather than processing it in a single forward pass.

We conduct a fair comparison between RLM using gpt-4o-mini (~$0.24/1M tokens) and direct baseline using gpt-5 (~$2.95/1M tokens) on RULER-style benchmarks across five task types: single-needle retrieval, multi-needle retrieval, word aggregation, sentiment analysis, and claim verification.

**Key findings:**

- **Overall accuracy**: Baseline gpt-5 achieves **60%** vs RLM's **47%** (with original prompt)
- **Cost savings**: RLM is **85% cheaper** ($0.03 vs $0.21 per evaluation)
- **Code-solvable tasks**: RLM excels at aggregation (100% vs 0%) where code-based counting is precise
- **Semantic reasoning tasks**: Baseline wins decisively on sentiment analysis (100% vs 0%) and claim verification (67% vs 0%)
- **Retrieval tasks**: Comparable performance (ties on NIAH-Single and Multi)

The results reveal that **RLM's default behavior does not effectively utilize `llm_query()` for semantic reasoning**, defaulting to code-only approaches. We propose explicit task classification prompting as a potential solution, though this remains to be validated empirically.

---

## 1. Introduction

### 1.1 Background

Large language models face fundamental challenges with long-context processing. While context windows have expanded significantly (GPT-4 Turbo: 128K tokens, Claude 3: 200K tokens, Gemini 1.5: 1M+ tokens), effective utilization of this context remains problematic. Models exhibit degraded performance on retrieval and reasoning tasks as context length increases, a phenomenon documented extensively in benchmarks like RULER (Hsieh et al., 2024) and Needle-in-a-Haystack (Kamradt, 2023).

### 1.2 The RLM Approach

Zhang, Kraska, and Khattab (2025) propose Recursive Language Models (RLM) as an alternative paradigm. Instead of fitting the entire context into the model's forward pass, RLM treats the context as an **external environment** that the model can programmatically explore through:

1. **Lazy context access**: Efficient slicing and searching without loading full content
2. **Code execution**: Python REPL for programmatic analysis
3. **Recursive sub-queries**: `llm_query()` for delegating analysis of chunks to sub-LLMs
4. **Explicit termination**: `FINAL()` to return answers when ready

This approach converts the problem from "process everything at once" to "explore strategically and aggregate findings."

### 1.3 Research Questions

1. Can RLM with a cheap model match or exceed the performance of an expensive model with full context access?
2. Which task types benefit most from the RLM approach?
3. What are the cost-performance tradeoffs?
4. What are the failure modes and limitations?

---

## 2. Related Work

### 2.1 Long-Context Benchmarks

**RULER** (Hsieh et al., 2024): Comprehensive benchmark extending beyond simple needle-in-haystack to include multi-hop reasoning, variable tracking, and aggregation tasks. Our evaluation design draws heavily from RULER's task taxonomy.

**Needle-in-a-Haystack** (Kamradt, 2023): Classic benchmark testing retrieval of a single fact embedded in distractor text. Simple but effective for measuring basic long-context capabilities.

**ZeroSCROLLS** (Shaham et al., 2023): Document QA and summarization tasks requiring understanding of long documents.

**LongBench** (Bai et al., 2023): Multi-task benchmark covering various long-context scenarios.

### 2.2 Inference-Time Scaling

**Chain-of-Thought** (Wei et al., 2022): Prompting models to show reasoning steps, improving performance on complex tasks.

**Self-Consistency** (Wang et al., 2023): Sampling multiple reasoning paths and aggregating results.

**RLM** (Zhang et al., 2025): The approach we implement, treating context as explorable environment.

### 2.3 Code-Augmented LLMs

**Program-Aided Language Models** (Gao et al., 2023): Using code execution to improve reasoning accuracy.

**Toolformer** (Schick et al., 2023): Training models to use external tools including calculators and search.

---

## 3. Implementation

### 3.1 Architecture

Our RLM implementation consists of the following components:

```
┌──────────────────────────────────────────────────────────────────┐
│  RlmSession                                                       │
│  - Manages iteration loop (max 20 iterations default)             │
│  - Tracks token usage (root vs sub-LLM separately)                │
│  - Enforces limits (iterations, subcalls, token budget)           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  RlmEnvironment                                                   │
│  - Python code execution (RestrictedPython sandbox)               │
│  - LazyContext object for efficient context access                │
│  - llm_query() function for recursive sub-LLM calls               │
│  - FINAL() / FINAL_VAR() for returning answers                    │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 LazyContext API

The context is exposed through a `LazyContext` object providing:

| Method | Description | Complexity |
|--------|-------------|------------|
| `len(context)` | Total character count | O(1) |
| `context.slice(start, end)` | Extract substring | O(end - start) |
| `context.search(pattern, limit)` | Regex search returning matches | O(n) |
| `context.lines(start, end)` | Extract line range | O(lines) |
| `str(context)` | Full context (use sparingly) | O(n) |

### 3.3 Recursive Sub-Queries

The `llm_query(prompt, max_chars)` function allows the root LLM to delegate analysis:

```python
# Example: Summarize a section
section = context.slice(10000, 20000)
summary = llm_query(f"Summarize this section:\n{section}")
print(summary)
```

Sub-queries use a separate token budget and can optionally use a different (cheaper) model than the root orchestrator.

### 3.4 System Prompt

The system prompt instructs the model on environment usage (see Appendix A for full text). Key instructions include:

- Always use `print()` to observe output (not an interactive REPL)
- Use code blocks with ```python or```repl markers
- Call `FINAL("answer")` outside code blocks when done
- For small contexts (<10K chars), direct reading is acceptable
- For large contexts, use search/slice to find relevant sections

### 3.5 Configuration

```python
@dataclass
class RlmConfig:
    llm_handler: LlmHandler           # Primary LLM for root calls
    recursive_llm_handler: LlmHandler | None  # Optional cheaper model for sub-queries
    max_iterations: int = 20          # Maximum REPL iterations
    max_subcalls: int = 100           # Maximum llm_query() calls
    token_budget: int | None = None   # Optional total token limit
    max_context_chars: int = 500_000  # Maximum context size
    context_chunk_hint: int = 200_000 # Suggested chunk size for sub-queries
```

---

## 4. Experimental Design

### 4.1 Fairness Considerations

To ensure a rigorous comparison, we implemented the following controls:

1. **Identical test data**: Both RLM and baseline receive exactly the same context, query, and expected answer for each trial. Test data is generated deterministically using a fixed random seed.

2. **Multiple trials**: Each test runs 3 trials to measure variance and avoid cherry-picking favorable results.

3. **Equivalent prompting**: The baseline model receives a system prompt with clear instructions equivalent in spirit to the RLM system prompt:

   ```
   You are a precise document analyzer. Read the document carefully and answer questions accurately.
   For retrieval tasks: search thoroughly and return exact values found.
   For counting tasks: count methodically and return the exact number.
   Be concise - return only what is asked for.
   ```

4. **Same correctness criteria**: Both approaches are evaluated using identical answer-checking logic.

5. **Transparent reporting**: All results are reported, including failures.

### 4.2 Test Tasks

We implement five RULER-style tasks:

#### 4.2.1 Single Needle-in-Haystack (NIAH-Single)

- **Description**: Find a single 8-character alphanumeric code embedded in ~50K characters of distractor text
- **Needle format**: `[CLASSIFIED INFORMATION]\nThe secret access code is XXXXXXXX.\n[END CLASSIFIED]`
- **Query**: "What is the secret access code mentioned in the document? Return ONLY the code itself."
- **Correctness**: Expected code appears in answer (case-insensitive)

#### 4.2.2 Multi-Needle Retrieval (NIAH-Multi)

- **Description**: Find 5 office access codes for different cities embedded throughout ~50K characters
- **Needle format**: `[OFFICE CODE]\nThe access code for {City} office is {XXXX}.\n[END CODE]`
- **Query**: "List ALL office access codes mentioned in the document. Format each as 'City: Code' on a new line."
- **Correctness**: All 5 city:code pairs appear in answer

#### 4.2.3 Word Aggregation

- **Description**: Count exact occurrences of a target word ("infrastructure") inserted 7 times in ~50K characters
- **Query**: "Count exactly how many times the word 'infrastructure' appears in the document. Return ONLY the number."
- **Correctness**: The number "7" appears in answer

#### 4.2.4 Sentiment Analysis

- **Description**: Analyze multiple document sections and count how many express positive sentiment
- **Sections**: 5 analyst report sections covering topics like Product Quality, Market Position, Employee Satisfaction, etc.
- **Query**: "How many of the analyst report sections express POSITIVE sentiment? Return ONLY the number."
- **Correctness**: The correct count (3) appears in answer
- **Note**: Requires semantic understanding—code-based approaches (keyword matching) fail to capture nuance

#### 4.2.5 Claim Verification

- **Description**: Verify which of 3 factual claims are supported by numerical data in the document
- **Claims**: Statements like "Revenue exceeded $1M in Q3" that require extracting and comparing numerical values
- **Query**: "Which claims are TRUE based on the document? List the claim numbers."
- **Correctness**: Correct claim numbers (1, 3) appear in answer
- **Note**: May be solvable via code (numeric extraction) or semantic reasoning depending on claim complexity

### 4.3 Test Data Generation

All test data is generated deterministically:

```python
def generate_haystack(target_chars: int, rng: random.Random) -> str:
    """Generate filler text from a fixed set of paragraphs."""
    paragraphs = [
        "The system architecture employs a modular design pattern...",
        "Performance metrics indicate consistent throughput...",
        # ... 10 total paragraphs
    ]
    # Concatenate until target size reached
    ...
```

Needle insertion positions are calculated deterministically:

- NIAH-Single: Middle of document (50% depth)
- NIAH-Multi: Evenly distributed (20%, 40%, 60%, 80%, 100% positions)
- Aggregation: Evenly spaced throughout

### 4.4 Models and Parameters

| Parameter | RLM | Baseline |
|-----------|-----|----------|
| Model | gpt-4o-mini | gpt-5 |
| Temperature | 0 (deterministic) | Default (gpt-5 doesn't support temp=0) |
| Max tokens | 4096 (via iterations) | 500 |
| Context delivery | Lazy (via LazyContext) | Full (in prompt) |
| Max iterations | 12 | 1 |
| Token budget | 100,000 | Unlimited |

**Note on GPT-5 API**: GPT-5 requires `max_completion_tokens` instead of `max_tokens` and does not support `temperature=0`. These API differences were discovered during testing and accommodated.

### 4.5 Metrics

- **Pass rate**: Percentage of trials where correctness criteria is met
- **Token usage**: Total tokens consumed (input + output)
- **Cost**: Estimated cost based on published pricing
- **Duration**: Wall-clock time per trial

**Correctness Checking**:

- **Single value**: Substring presence (case-insensitive)
- **Numeric count**: Exact number match via regex extraction
- **Claim verification**: Strict set equality—extracted claim numbers must exactly match expected (no extra claims, no missing claims). This prevents inflated scores from answers like "1, 2, 3, 4" when only "1, 3" are correct.
- **Multi-needle retrieval**: All expected key:value pairs must appear as substrings

### 4.6 Reproducibility

All experiments can be reproduced using:

- Random seed: 42
- Haystack size: 50,000 characters
- Trials per test: 3
- Code: `examples/eval_ruler_fair.py`

---

## 5. Results

### 5.1 Summary Results

| Test | RLM (gpt-4o-mini) | Baseline (gpt-5) | Winner |
|------|-------------------|------------------|--------|
| NIAH-Single (retrieval) | 3/3 (100%) | 3/3 (100%) | Tie |
| NIAH-Multi (retrieval) | 1/3 (33%) | 1/3 (33%) | Tie |
| Aggregation (code-solvable) | **3/3 (100%)** | 0/3 (0%) | **RLM** |
| Sentiment (semantic reasoning) | 0/3 (0%) | **3/3 (100%)** | **Baseline** |
| Claim Verify (semantic reasoning) | 0/3 (0%) | **2/3 (67%)** | **Baseline** |
| **OVERALL** | **7/15 (47%)** | **9/15 (60%)** | **Baseline** |

### 5.2 Detailed Results

#### 5.2.1 NIAH-Single (Code-Solvable)

| Trial | RLM Result | RLM Tokens | Baseline Result | Baseline Tokens |
|-------|------------|------------|-----------------|-----------------|
| 1 | ✓ XAJI0Y6D | 2,093 | ✓ XAJI0Y6D | 7,429 |
| 2 | ✓ XAJI0Y6D | 2,075 | ✓ XAJI0Y6D | 7,429 |
| 3 | ✓ XAJI0Y6D | 2,083 | ✓ XAJI0Y6D | 7,429 |

**Analysis**: Both approaches achieved 100% accuracy. RLM used ~3.5x fewer tokens by using regex search rather than processing full context.

#### 5.2.2 NIAH-Multi (Code-Solvable)

| Trial | RLM Result | RLM Tokens | Baseline Result | Baseline Tokens |
|-------|------------|------------|-----------------|-----------------|
| 1 | ✓ All 5 codes | 11,779 | ✗ (partial) | 7,535 |
| 2 | ✗ (partial) | 14,553 | ✓ All 5 codes | 7,535 |
| 3 | ✗ (partial) | 2,316 | ✗ (partial) | 7,535 |

**Analysis**: Both showed variance on this task. RLM failures often involved returning variable names instead of values. Neither approach was reliable for complex multi-item retrieval.

#### 5.2.3 Aggregation (Code-Solvable)

| Trial | RLM Result | RLM Tokens | Baseline Result | Baseline Tokens |
|-------|------------|------------|-----------------|-----------------|
| 1 | ✓ 7 | 1,857 | ✗ (wrong count) | 7,558 |
| 2 | ✓ 7 | 1,857 | ✗ (wrong count) | 7,558 |
| 3 | ✓ 7 | 1,857 | ✗ (wrong count) | 7,558 |

**Analysis**: **RLM achieved perfect accuracy** using regex counting. GPT-5 failed all trials, confirming known LLM limitations on precise counting. This strongly validates RLM for aggregation tasks.

#### 5.2.4 Sentiment Analysis (Requires Semantic Reasoning)

| Trial | RLM Result | RLM Tokens | Baseline Result | Baseline Tokens |
|-------|------------|------------|-----------------|-----------------|
| 1 | ✗ (returned variable name) | 5,078 | ✓ 3 | 745 |
| 2 | ✗ (returned 0) | 3,836 | ✓ 3 | 809 |
| 3 | ✗ (returned 1) | 11,838 | ✓ 3 | 745 |

**Analysis**: **RLM completely failed** (0/3). The model attempted code-based sentiment detection (regex for positive/negative words) instead of using `llm_query()` for semantic analysis. This reveals a critical limitation: **RLM does not spontaneously use sub-LLM calls for tasks requiring language understanding**.

#### 5.2.5 Claim Verification (Requires Semantic Reasoning)

| Trial | RLM Result | RLM Tokens | Baseline Result | Baseline Tokens |
|-------|------------|------------|-----------------|-----------------|
| 1 | ✗ (no answer) | 29,881 | ✓ 1, 3 | 889 |
| 2 | ✗ (no answer) | 29,362 | ✓ 1, 3 | 697 |
| 3 | ✗ (empty) | 6,189 | ✗ (wrong) | 928 |

**Analysis**: **RLM failed all trials** despite using many tokens (up to 30K per trial). The model attempted regex-based fact extraction but couldn't perform the semantic comparison needed to verify claims. Baseline performed well (67%) with minimal tokens.

### 5.3 Cost Analysis

| Metric | RLM (gpt-4o-mini) | Baseline (gpt-5) |
|--------|-------------------|------------------|
| Total tokens | 131,413 | 72,533 |
| Estimated cost | $0.0315 | $0.2140 |
| Cost per correct answer | $0.0045 | $0.0238 |
| **Cost reduction** | **85.3%** | — |

**Pricing (from [OpenAI Pricing](https://openai.com/api/pricing/), 2025)**:

- gpt-4o-mini: $0.15/1M input + $0.60/1M output → ~$0.24/1M blended
- gpt-5: $1.25/1M input + $10.00/1M output → ~$2.95/1M blended

### 5.4 Latency Analysis

| Test | RLM Avg Duration | Baseline Avg Duration |
|------|------------------|----------------------|
| NIAH-Single | 4.0s | 8.3s |
| NIAH-Multi | 19.0s | 7.6s |
| Aggregation | 1.7s | 7.9s |
| Sentiment | 13.7s | 4.8s |
| Claim Verify | 30.2s | 6.5s |

**Analysis**: RLM is faster for simple code-solvable tasks but significantly slower for semantic tasks where it spins through many iterations without finding a working approach.

---

## 6. Analysis

### 6.1 Why RLM Excels at Aggregation

The aggregation task demonstrates RLM's core advantage: **code execution enables precise operations that LLMs approximate poorly**.

GPT-5's counting failures are not surprising. Neural networks process text through attention mechanisms that don't naturally support counting. The model must essentially "simulate" counting, which is error-prone for non-trivial counts.

RLM's approach:

```python
matches = context.search(r"\binfrastructure\b")
print(len(matches))  # Exact count
```

This is deterministic and guaranteed correct, regardless of context size.

### 6.2 Why RLM Fails on Semantic Reasoning Tasks

**This is the most significant finding of our evaluation.**

RLM failed completely (0%) on both sentiment analysis and claim verification. Analysis of execution traces reveals the cause: **the model does not spontaneously use `llm_query()` for semantic reasoning**.

Instead, the model attempts code-based approaches:

- **Sentiment**: Searches for positive/negative keywords with regex
- **Claim verification**: Extracts numbers with regex, compares strings

These approaches fail because:

1. Sentiment requires understanding context, tone, and nuance
2. Claim verification requires semantic comparison ("more than" vs actual numbers)
3. The model's system prompt doesn't sufficiently emphasize when to use `llm_query()`

**Root cause**: The RLM paper assumes models will naturally recognize when sub-LLM reasoning is needed. Our experiments show gpt-4o-mini defaults to code-only solutions even when inappropriate.

### 6.3 Comparison of Task Categories

| Task Type | RLM Advantage | RLM Limitation |
|-----------|---------------|----------------|
| **Code-solvable** (counting, regex) | Precise, deterministic | None observed |
| **Retrieval** (finding marked content) | Token-efficient | Variance in complex cases |
| **Semantic** (sentiment, reasoning) | N/A | Does not use llm_query() |

### 6.4 Token Efficiency

RLM used **more total tokens** than baseline (131K vs 73K), contrary to expectations:

- RLM: 131,413 tokens
- Baseline: 72,533 tokens

This is because RLM burned many tokens on failed semantic tasks (up to 30K per trial) while spinning through iterations trying code-based approaches.

For **code-solvable tasks only**, RLM is more efficient:

- NIAH-Single: RLM ~2K vs Baseline ~7.4K (3.7x more efficient)
- Aggregation: RLM ~1.9K vs Baseline ~7.6K (4x more efficient)

### 6.5 The llm_query() Gap

The RLM architecture provides `llm_query()` for delegating semantic analysis to sub-LLMs. However:

1. **Not used in our experiments**: Zero sub-LLM calls were made across all trials
2. **Model preference for code**: gpt-4o-mini prefers regex/string operations over LLM delegation
3. **System prompt insufficient**: Current prompt doesn't clearly indicate when llm_query() is appropriate

**Recommendation**: Future work should either:

- Train models specifically for RLM (knowing when to use llm_query())
- Provide explicit task-type hints in queries
- Use stronger orchestrator models (gpt-4o, gpt-5) that may better recognize when semantic reasoning is needed

---

## 7. Limitations

### 7.1 Task Coverage

Our evaluation covers five task types (NIAH-Single, NIAH-Multi, Aggregation, Sentiment Analysis, Claim Verification). The RLM approach may perform differently on:

- Multi-hop reasoning requiring synthesis across sections
- Summarization requiring holistic understanding
- Tasks requiring world knowledge combined with context

### 7.2 Model Selection

We tested only gpt-4o-mini vs gpt-5. Results may differ with:

- Other model families (Claude, Gemini, open-source models)
- Different model sizes within families
- Specialized long-context models

### 7.3 Prompt Sensitivity

Both RLM and baseline performance depend heavily on prompts. Our prompts were designed to be fair but may not be optimal for either approach.

### 7.4 Variance

Three trials per test provides limited statistical power. A production evaluation should include more trials and confidence intervals.

### 7.5 Synthetic Data

All tests use synthetic data with clear markers (`[CLASSIFIED]`, `[OFFICE CODE]`). Real-world documents may present different challenges.

### 7.6 API Constraints

GPT-5's inability to use temperature=0 introduces randomness in baseline results that may affect reproducibility.

---

## 8. Conclusion

### 8.1 Summary of Findings

1. **Baseline gpt-5 outperforms RLM (gpt-4o-mini) overall**: 60% vs 47% accuracy across all task types.

2. **RLM excels at code-solvable tasks**: Perfect 100% accuracy on aggregation (counting) where baseline achieved 0%.

3. **RLM fails on semantic reasoning tasks**: 0% accuracy on sentiment analysis and claim verification, where baseline achieved 100% and 67% respectively.

4. **Cost reduction of 85%** ($0.03 vs $0.21) despite lower accuracy, making RLM attractive for code-solvable workloads.

5. **Critical gap identified**: The model does not spontaneously use `llm_query()` for semantic reasoning, limiting RLM's applicability to code-solvable tasks only.

### 8.2 Recommendations

**Use RLM when**:

- Tasks are code-solvable (counting, regex matching, structured extraction)
- Precise, deterministic answers are required
- Cost efficiency is paramount
- Tasks have clear, structured markers in the text

**Use direct baseline when**:

- Tasks require semantic understanding (sentiment, reasoning, comparison)
- Holistic document comprehension is needed
- Task structure is unclear or requires judgment
- Accuracy is more important than cost

**Do NOT assume RLM will use llm_query()**: Current implementations default to code-only approaches. Tasks requiring semantic reasoning should use baseline or explicitly prompt for llm_query() usage.

### 8.3 Proposed Approach: Enhanced Prompting (Not Yet Implemented)

The experimental results above reveal that RLM does not spontaneously use `llm_query()` for semantic reasoning tasks. We propose **explicit task classification prompting** to address this gap. This approach has not yet been validated empirically.

#### Proposed Enhanced Prompt Additions

```
## CRITICAL: Task Classification

### CODE-SOLVABLE TASKS (use Python only):
- Counting occurrences (regex, string matching)
- Extracting exact values (IDs, numbers, names)
- Finding patterns (word frequency, formatting)
- Locating specific strings (needle-in-haystack)

### SEMANTIC TASKS (MUST use llm_query()):
- Sentiment analysis: Determining if text is positive/negative/neutral
- Claim verification: Checking if claims are supported by evidence
- Summarization: Condensing text while preserving meaning
- Reasoning tasks: Drawing conclusions from complex information

**RULE**: If the task requires UNDERSTANDING MEANING, you MUST call llm_query().
```

#### Hypothesis

We hypothesize that explicit task classification prompting would:

1. **Sentiment Analysis**: Trigger appropriate `llm_query()` usage for semantic understanding
2. **Claim Verification**: Enable intelligent routing to code (for numeric comparisons) or `llm_query()` (for semantic claims)
3. **Task Classification**: Allow the model to distinguish between code-solvable and semantic tasks

#### Future Work Required

To validate this hypothesis:

- Implement enhanced system prompt in `RLM_SYSTEM_PROMPT`
- Re-run benchmark evaluation with enhanced prompting
- Compare results against baseline measurements reported in Section 5

**Note**: The enhanced prompt shown in Appendix A represents the proposed design. Current implementation uses the original prompt without task classification guidance.

### 8.4 Future Work

1. **Further prompt refinement**:
   - More nuanced task classification examples
   - Domain-specific guidance for different task types

2. **Hybrid approaches**:
   - Automatic task classification to route to RLM vs baseline
   - Fallback to llm_query() when code approaches fail

3. **Extended evaluation**:
   - More trials for statistical significance
   - Additional semantic reasoning tasks
   - Cross-model comparison (Claude, Gemini)

4. **Architecture improvements**:
   - Automatic detection of code-vs-semantic task requirements
   - Mixed model configurations (strong orchestrator, cheap sub-queries)

---

## References

1. Zhang, Y., Kraska, T., & Khattab, O. (2025). Recursive Language Models: Inference-Time Scaling for Long Contexts. *arXiv:2512.24601*.

2. Hsieh, C.-Y., et al. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models? *arXiv:2404.06654*.

3. Kamradt, G. (2023). LLMTest_NeedleInAHaystack. GitHub repository. <https://github.com/gkamradt/LLMTest_NeedleInAHaystack>

4. Shaham, U., et al. (2023). ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding. *arXiv:2305.14196*.

5. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.

6. Gao, L., et al. (2023). PAL: Program-aided Language Models. *ICML 2023*.

---

## Appendix A: System Prompt (Proposed Enhanced Version)

The proposed enhanced system prompt with task classification (with placeholders for context length and chunk hint). **Note**: This version is not yet implemented in the codebase. Current implementation uses a simpler prompt without explicit task classification guidance. See Section 8.3 for the validation plan.

```
You have access to a Python REPL environment to process a large context.

## Environment

- `context`: A LazyContext object containing the input ({context_len:,} characters)
  - `len(context)` - Get total length
  - `context.slice(start, end)` - Get substring (returns string)
  - `context.search(pattern)` - Regex search (returns list of {start, end, match})
  - `context.lines(start_line, end_line)` - Get lines by number
  - `str(context)` - WARNING: Loads entire context, use only if small (<5K chars)

- `llm_query(prompt, max_chars={chunk_hint:,})` - Call a sub-LLM with a prompt
  - **CRITICAL**: You MUST use this for ANY task requiring semantic understanding!
  - Returns the model's response as a string

- `FINAL("your answer here")` - Return your final answer (call OUTSIDE code blocks!)
- `FINAL_VAR(var_name)` - Return the value of a variable as the final answer

## CRITICAL: Task Classification

Before writing any code, classify your task:

### CODE-SOLVABLE TASKS (use Python only):
- Counting occurrences (regex, string matching)
- Extracting exact values (IDs, numbers, names)
- Finding patterns (word frequency, formatting)
- Locating specific strings (needle-in-haystack)
- Mathematical operations on extracted data

### SEMANTIC TASKS (MUST use llm_query()):
- **Sentiment analysis**: Determining if text is positive/negative/neutral
- **Claim verification**: Checking if claims are supported by evidence
- **Summarization**: Condensing text while preserving meaning
- **Intent/tone analysis**: Understanding author's purpose or attitude
- **Reasoning tasks**: Drawing conclusions from complex information

**RULE**: If the task requires UNDERSTANDING MEANING, you MUST call `llm_query()`.
Code alone cannot determine sentiment, verify claims, or understand nuance.

## CRITICAL: Always use print()!

You MUST use `print()` to see any output. This is NOT an interactive REPL.

## Strategy

1. **Classify the task**: Is it code-solvable or semantic? (see above)
2. **Examine structure**: Use `context.search()` or `context.slice()` to find relevant sections
3. **For code-solvable tasks**: Use Python to extract/count/filter
4. **For semantic tasks**: Extract sections, then call `llm_query()` to analyze them
5. **When you have the answer**: Call `FINAL("your answer")` OUTSIDE the code block

## Example: Semantic Task (Sentiment Analysis)

Task: "What is the overall sentiment of section 3?"

\`\`\`python
# 1. Extract the section
text = str(context) if len(context) < 10000 else context.slice(0, 10000)

# 2. MUST use llm_query for sentiment - code cannot determine sentiment!
sentiment = llm_query(f"Analyze the sentiment of this text. Reply: positive, negative, or neutral.\n\n{text}")
print(f"Sentiment: {sentiment}")
\`\`\`

FINAL("positive")

## FINAL Answer Format

Your FINAL() answer must be COMPLETE and DETAILED:
- Include all findings, not just a summary
- For analysis tasks, include the full analysis
- NEVER say "see above" - repeat the information in FINAL()
```

---

## Appendix B: Reproduction Instructions

### Prerequisites

```bash
# Python 3.11+
# OpenAI API key with access to gpt-4o-mini and gpt-5

cd src/python/packages/amla-sandbox
```

### Environment Setup

```bash
# Create .env file with API key
echo "OPENAI_KEY=sk-..." > /path/to/.env

# Install dependencies
uv sync
```

### Running Evaluations

```bash
# Quick test (NIAH-Single only)
uv run python examples/eval_ruler_fair.py --quick

# Full evaluation (3 trials each)
uv run python examples/eval_ruler_fair.py --trials 3

# Custom configuration
uv run python examples/eval_ruler_fair.py \
    --trials 5 \
    --size 100000 \
    --seed 42 \
    --rlm-model gpt-4o-mini \
    --baseline-model gpt-5
```

### Expected Output

```
======================================================================
FAIR RULER COMPARISON: RLM vs Baseline
======================================================================
RLM Model:      gpt-4o-mini
Baseline Model: gpt-5
...
OVERALL                             7/15 (47%)      9/15 (60%)
...
Cost savings with RLM: 85.3%
```

**Note**: Results show 15 total trials (3 trials × 5 tasks). RLM achieves 47% accuracy vs baseline's 60%, with 85% cost savings.

---

## Appendix C: Raw Data

### Trial-Level Results (Seed 42, 3 trials)

```
NIAH-Single:
  Trial 1: RLM=✓(2093 tok), Baseline=✓(7429 tok)
  Trial 2: RLM=✓(2075 tok), Baseline=✓(7429 tok)
  Trial 3: RLM=✓(2083 tok), Baseline=✓(7429 tok)

NIAH-Multi:
  Trial 1: RLM=✓(11779 tok), Baseline=✗(7535 tok)
  Trial 2: RLM=✗(14553 tok), Baseline=✓(7535 tok)
  Trial 3: RLM=✗(2316 tok), Baseline=✗(7535 tok)

Aggregation:
  Trial 1: RLM=✓(1857 tok), Baseline=✗(7558 tok)
  Trial 2: RLM=✓(1857 tok), Baseline=✗(7558 tok)
  Trial 3: RLM=✓(1857 tok), Baseline=✗(7558 tok)

Sentiment (REQUIRES llm_query):
  Trial 1: RLM=✗(5078 tok, answer="positive_count"), Baseline=✓(745 tok, answer="3")
  Trial 2: RLM=✗(3836 tok, answer="0"), Baseline=✓(809 tok, answer="3")
  Trial 3: RLM=✗(11838 tok, answer="1"), Baseline=✓(745 tok, answer="3")

Claim Verification (REQUIRES llm_query):
  Trial 1: RLM=✗(29881 tok, no answer), Baseline=✓(889 tok, answer="1, 3")
  Trial 2: RLM=✗(29362 tok, no answer), Baseline=✓(697 tok, answer="1, 3")
  Trial 3: RLM=✗(6189 tok, empty), Baseline=✗(928 tok, wrong)
```

### Sub-LLM Usage

**Critical observation**: Across all 15 RLM trials, `llm_query()` was called **zero times**.

The model consistently attempted code-only solutions even for semantic reasoning tasks.

### Checksums

Test data generation is deterministic. Expected values for seed=42:

- NIAH-Single: `XAJI0Y6D`
- NIAH-Multi: Paris:0614, Tokyo:6744, London:0438, Sydney:5741, Cairo:8830
- Aggregation: 7 occurrences of "infrastructure"
- Sentiment: 3 positive sections (Product Quality, Market Position, Employee Satisfaction)
- Claim Verification: Claims 1 and 3 are TRUE
