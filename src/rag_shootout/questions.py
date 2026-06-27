"""
questions.py - The 4 benchmark questions for the sample RAG shootout document.

Questions are deliberately chosen to expose where vector retrieval and
structure-aware retrieval differ on a small synthetic document.
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
        text="How does vector search retrieve relevant information?",
        category="Vector Match",
        why="Favors Vector RAG because the answer is a direct semantic match to one compact paragraph.",
    ),
    Question(
        id=2,
        text="How does tree-based retrieval navigate a document?",
        category="Structure Navigation",
        why="Favors PageIndex because the answer depends on document hierarchy and branch selection.",
    ),
    Question(
        id=3,
        text="What are the limitations of vector similarity search?",
        category="Limitations",
        why="Pushes beyond lookup into reasoning about failure modes, where structure-aware retrieval should do better.",
    ),
    Question(
        id=4,
        text="Which retrieval method better understands document structure?",
        category="Comparison",
        why="Forces a comparison answer grounded in the contrast between semantic similarity and document structure.",
    ),
]


def get_questions() -> list[Question]:
    """Return the full list of benchmark questions."""
    return list(QUESTIONS)


def get_by_category(category: str) -> list[Question]:
    """Return all questions matching *category* (case-insensitive)."""
    cat = category.lower()
    return [q for q in QUESTIONS if q.category.lower() == cat]
