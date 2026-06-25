"""
scoring.py — Manual scoring scaffolding and aggregate metrics.

Why manual scoring? LLM-grading-itself is a known bias source — the same
model that generated the answer will tend to rate itself highly. This
module provides the scaffolding for a human to score each answer 1-5
on three dimensions, then computes aggregate metrics.

Rubric per dimension:
  - accuracy:      1 = factually wrong, 5 = fully correct per the source
  - completeness:  1 = misses the point, 5 = covers everything asked
  - faithfulness:  1 = fabricated/hallucinated, 5 = fully grounded

Max score per question: 15. Max total across 4 questions: 60.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from rag_shootout.config import SCORE_DIMENSIONS, SCORE_MAX, SCORE_MIN


# ─── Types ────────────────────────────────────────────────────────────────────


@dataclass
class DimensionScore:
    accuracy: Optional[int] = None
    completeness: Optional[int] = None
    faithfulness: Optional[int] = None

    def is_complete(self) -> bool:
        return all(
            v is not None for v in [self.accuracy, self.completeness, self.faithfulness]
        )

    def total(self) -> Optional[float]:
        if not self.is_complete():
            return None
        return float(self.accuracy + self.completeness + self.faithfulness)  # type: ignore[operator]

    def validate(self) -> None:
        for attr in SCORE_DIMENSIONS:
            val = getattr(self, attr)
            if val is not None and not (SCORE_MIN <= val <= SCORE_MAX):
                raise ValueError(
                    f"{attr} score {val} out of range [{SCORE_MIN}, {SCORE_MAX}]"
                )


@dataclass
class QuestionScore:
    question_id: int
    vector: DimensionScore = field(default_factory=DimensionScore)
    pageindex: DimensionScore = field(default_factory=DimensionScore)

    def winner(self) -> str:
        vt = self.vector.total()
        pt = self.pageindex.total()
        if vt is None or pt is None:
            return "not scored yet"
        if vt > pt:
            return "Vector RAG"
        if pt > vt:
            return "PageIndex"
        return "Tie"


# ─── Scaffold ─────────────────────────────────────────────────────────────────


def make_empty_scorecard(n_questions: int = 4) -> list[QuestionScore]:
    """
    Return a list of empty QuestionScore objects (one per question).
    Fill in the DimensionScore values after reading each answer.
    """
    return [QuestionScore(question_id=i + 1) for i in range(n_questions)]


# ─── Metrics ──────────────────────────────────────────────────────────────────


def compute_summary(scores: list[QuestionScore]) -> dict:
    """
    Compute aggregate metrics across all scored questions.

    Returns a dict with:
      - per-pipeline totals and averages
      - per-dimension averages
      - win counts
      - latency averages (if passed in via results_df)
    """
    scored = [s for s in scores if s.vector.is_complete() and s.pageindex.is_complete()]

    if not scored:
        return {"error": "No questions have been fully scored yet."}

    n = len(scored)

    def avg_dim(pipeline: str, dim: str) -> float:
        vals = [getattr(getattr(s, pipeline), dim) for s in scored]
        return round(sum(v for v in vals if v is not None) / n, 2)

    def total_scores(pipeline: str) -> list[float]:
        return [getattr(s, pipeline).total() for s in scored]  # type: ignore[return-value]

    v_totals = total_scores("vector")
    p_totals = total_scores("pageindex")

    wins = {"Vector RAG": 0, "PageIndex": 0, "Tie": 0}
    for s in scored:
        wins[s.winner()] = wins.get(s.winner(), 0) + 1

    return {
        "n_scored": n,
        "vector": {
            "total": sum(v_totals),
            "avg_per_question": round(sum(v_totals) / n, 2),
            "avg_accuracy": avg_dim("vector", "accuracy"),
            "avg_completeness": avg_dim("vector", "completeness"),
            "avg_faithfulness": avg_dim("vector", "faithfulness"),
        },
        "pageindex": {
            "total": sum(p_totals),
            "avg_per_question": round(sum(p_totals) / n, 2),
            "avg_accuracy": avg_dim("pageindex", "accuracy"),
            "avg_completeness": avg_dim("pageindex", "completeness"),
            "avg_faithfulness": avg_dim("pageindex", "faithfulness"),
        },
        "wins": wins,
    }


def scores_to_dataframe(
    scores: list[QuestionScore],
    results: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Convert a list of QuestionScore objects into a flat DataFrame.

    Optionally merges in *results* (list of dicts from the benchmark runner)
    which contain answer text and latency.
    """
    rows = []
    for s in scores:
        row: dict = {
            "question_id": s.question_id,
            "vector_accuracy": s.vector.accuracy,
            "vector_completeness": s.vector.completeness,
            "vector_faithfulness": s.vector.faithfulness,
            "vector_total": s.vector.total(),
            "pageindex_accuracy": s.pageindex.accuracy,
            "pageindex_completeness": s.pageindex.completeness,
            "pageindex_faithfulness": s.pageindex.faithfulness,
            "pageindex_total": s.pageindex.total(),
            "winner": s.winner(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if results:
        results_df = pd.DataFrame(results)
        # Align on question_id (1-indexed)
        results_df["question_id"] = results_df.index + 1
        df = df.merge(results_df, on="question_id", how="left")

    return df


def print_summary(scores: list[QuestionScore]) -> None:
    """Pretty-print the scoring summary to stdout."""
    summary = compute_summary(scores)
    if "error" in summary:
        print(summary["error"])
        return

    print(f"\n{'='*60}")
    print(f"  RAG SHOOTOUT RESULTS — {summary['n_scored']} questions scored")
    print(f"{'='*60}")

    for pipeline in ("vector", "pageindex"):
        label = "Vector RAG" if pipeline == "vector" else "PageIndex"
        s = summary[pipeline]
        print(f"\n{label}:")
        print(f"  Total score:    {s['total']} / {summary['n_scored'] * 15}")
        print(f"  Avg per Q:      {s['avg_per_question']} / 15")
        print(f"  Accuracy avg:   {s['avg_accuracy']}")
        print(f"  Completeness:   {s['avg_completeness']}")
        print(f"  Faithfulness:   {s['avg_faithfulness']}")

    w = summary["wins"]
    print(f"\nWins → Vector RAG: {w.get('Vector RAG', 0)} | "
          f"PageIndex: {w.get('PageIndex', 0)} | "
          f"Tie: {w.get('Tie', 0)}")
    print(f"{'='*60}\n")
