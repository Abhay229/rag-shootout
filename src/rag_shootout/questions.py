"""
questions.py — The 4 benchmark questions for the DeepSeek-R1 shootout.

Questions are deliberately chosen to stress-test each retrieval paradigm:
  • Easy factual (baseline sanity check)
  • Numeric / table lookup (chunk-splitting risk)
  • Full-paper synthesis (distributed information)
  • Narrative / anecdote (localized, precise retrieval)

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
        text="What pass@1 score did DeepSeek-R1 achieve on the AIME 2024 benchmark, and how does that compare to OpenAI's o1-1217?",
        category="Numeric",
        why="Exact numbers often get split across chunk boundaries or buried deep in results tables.",
    ),
    Question(
        id=3,
        text="Summarize the full multi-stage training pipeline used to produce the final DeepSeek-R1 model, start to finish.",
        category="Synthesis",
        why="The pipeline is described piecemeal across multiple sections — the hardest case for top-k retrieval.",
    ),
    Question(
        id=4,
        text="What was the 'aha moment' observed during DeepSeek-R1-Zero's training, and why did the authors highlight it?",
        category="Narrative",
        why="Localized in one place — tests whether precise retrieval can find a named, specific event.",
    ),
]


def get_questions() -> list[Question]:
    """Return the full list of benchmark questions."""
    return list(QUESTIONS)


def get_by_category(category: str) -> list[Question]:
    """Return all questions matching *category* (case-insensitive)."""
    cat = category.lower()
    return [q for q in QUESTIONS if q.category.lower() == cat]
