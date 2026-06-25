"""
questions.py — The 10 benchmark questions for the DeepSeek-R1 shootout.

Questions are deliberately chosen to stress-test each retrieval paradigm:
  • Easy factual (baseline sanity check)
  • Numeric / table lookup (chunk-splitting risk)
  • Factual list (chunk-unfriendly tables)
  • Two-section conceptual (cross-section connecting)
  • Full-paper synthesis (distributed information)
  • Narrative / anecdote (localized, precise retrieval)
  • "What didn't work" (easy to miss in top-k)
  • Trade-off reasoning (why, not just what)
  • Multi-hop comparison (hardest — cross-table synthesis)

To benchmark a different paper, replace this list and update the PDF_URL
in config.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: int
    text: str
    category: str
    why: str  # explains what this question is designed to stress-test


QUESTIONS: list[Question] = [
    Question(
        id=1,
        text="What reinforcement learning algorithm was used to train DeepSeek-R1-Zero, and how does it differ from PPO?",
        category="Factual",
        why="Baseline sanity check — both approaches should pass. Tests single-section factual recall.",
    ),
    Question(
        id=2,
        text="What two main types of rewards make up DeepSeek-R1-Zero's rule-based reward system?",
        category="Factual",
        why="Another factual baseline. Defined in one concentrated section of the paper.",
    ),
    Question(
        id=3,
        text="What pass@1 score did DeepSeek-R1 achieve on the AIME 2024 benchmark, and how does that compare to OpenAI's o1-1217?",
        category="Numeric",
        why="Exact numbers often get split across chunk boundaries or buried deep in results tables.",
    ),
    Question(
        id=4,
        text="What is 'cold-start data', and why did the authors use it for DeepSeek-R1 but not for DeepSeek-R1-Zero?",
        category="Multi-hop",
        why="Requires connecting a definition from one section with a design decision from another.",
    ),
    Question(
        id=5,
        text="Summarize the full multi-stage training pipeline used to produce the final DeepSeek-R1 model, start to finish.",
        category="Synthesis",
        why="The pipeline is described piecemeal across multiple sections — the hardest case for top-k retrieval.",
    ),
    Question(
        id=6,
        text="Which dense models did the authors distill DeepSeek-R1's reasoning into, and what parameter sizes were used?",
        category="Factual list",
        why="Lists and tables are notoriously chunk-unfriendly — often split across boundaries.",
    ),
    Question(
        id=7,
        text="What was the 'aha moment' observed during DeepSeek-R1-Zero's training, and why did the authors highlight it?",
        category="Narrative",
        why="Localized in one place — tests whether precise retrieval can find a named, specific event.",
    ),
    Question(
        id=8,
        text="What approaches did the authors try that did NOT work well, according to the paper?",
        category="Synthesis",
        why="Failure modes are scattered and de-emphasized — easy to miss if not in the top-k retrieved chunks.",
    ),
    Question(
        id=9,
        text="What language-consistency problem did DeepSeek-R1-Zero have, and what trade-off did the authors accept to fix it in DeepSeek-R1?",
        category="Trade-off",
        why="Tests whether the model explains the *why* (the tradeoff), not just the *what* (the fix).",
    ),
    Question(
        id=10,
        text="Across the benchmarks reported, how does DeepSeek-R1 compare to its non-reasoning base model DeepSeek-V3, and what is the headline takeaway?",
        category="Multi-hop",
        why="True cross-document synthesis — requires reading multiple benchmark tables and drawing a conclusion.",
    ),
]


def get_questions() -> list[Question]:
    """Return the full list of benchmark questions."""
    return list(QUESTIONS)


def get_by_category(category: str) -> list[Question]:
    """Return all questions matching *category* (case-insensitive)."""
    cat = category.lower()
    return [q for q in QUESTIONS if q.category.lower() == cat]
