"""
visualization.py — Benchmark result charts.

All chart functions take a pandas DataFrame (from scoring.scores_to_dataframe)
and return a matplotlib Figure so they can be rendered in a notebook,
saved to a file, or displayed in a script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent colour palette used across all charts
PALETTE = {
    "Vector RAG": "#4C72B0",
    "PageIndex": "#DD8452",
    "Tie": "#999999",
    "neutral": "#aaaaaa",
}


def _apply_style() -> None:
    """Apply a clean, minimal style to all charts."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )


def plot_scores_per_question(df: pd.DataFrame) -> plt.Figure:
    """
    Side-by-side bar chart of total score (out of 15) per question.
    Only plots questions that have been fully scored.
    """
    _apply_style()
    scored = df.dropna(subset=["vector_total", "pageindex_total"])
    if scored.empty:
        raise ValueError("No fully-scored questions in DataFrame.")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(scored))
    w = 0.35

    ax.bar(x - w / 2, scored["vector_total"], w, label="Vector RAG", color=PALETTE["Vector RAG"])
    ax.bar(x + w / 2, scored["pageindex_total"], w, label="PageIndex", color=PALETTE["PageIndex"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{qid}" for qid in scored["question_id"]])
    ax.set_ylabel("Total score (out of 15)")
    ax.set_title("Score per question — Vector RAG vs. PageIndex")
    ax.set_ylim(0, 16)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_win_counts(df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of how many questions each approach won.
    """
    _apply_style()
    scored = df.dropna(subset=["vector_total", "pageindex_total"])
    if scored.empty:
        raise ValueError("No fully-scored questions in DataFrame.")

    win_counts = scored["winner"].value_counts()
    labels = win_counts.index.tolist()
    colors = [PALETTE.get(lbl, PALETTE["neutral"]) for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, win_counts.values, color=colors)
    ax.set_ylabel("Questions won")
    ax.set_title("Win count by approach")

    for i, v in enumerate(win_counts.values):
        ax.text(i, v + 0.05, str(v), ha="center", fontweight="bold")

    fig.tight_layout()
    return fig


def plot_dimension_averages(df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart: average score per dimension (accuracy / completeness / faithfulness).
    """
    _apply_style()
    scored = df.dropna(subset=["vector_total", "pageindex_total"])
    if scored.empty:
        raise ValueError("No fully-scored questions in DataFrame.")

    dims = ["accuracy", "completeness", "faithfulness"]
    v_avgs = [scored[f"vector_{d}"].mean() for d in dims]
    p_avgs = [scored[f"pageindex_{d}"].mean() for d in dims]

    x = np.arange(len(dims))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, v_avgs, w, label="Vector RAG", color=PALETTE["Vector RAG"])
    ax.bar(x + w / 2, p_avgs, w, label="PageIndex", color=PALETTE["PageIndex"])

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dims])
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Average score (1–5)")
    ax.set_title("Average score by evaluation dimension")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_latency(df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart comparing average response latency of both pipelines.
    Requires 'vector_time' and 'pageindex_time' columns in *df*.
    """
    _apply_style()
    required = {"vector_time", "pageindex_time"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    avgs = [df["vector_time"].mean(), df["pageindex_time"].mean()]
    labels = ["Vector RAG", "PageIndex"]
    colors = [PALETTE["Vector RAG"], PALETTE["PageIndex"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, avgs, color=colors)
    ax.set_ylabel("Avg response time (seconds)")
    ax.set_title("Latency comparison (all questions)")

    for bar, avg in zip(bars, avgs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{avg:.2f}s",
            ha="center",
        )

    fig.tight_layout()
    return fig


def plot_wins_by_category(df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart of wins broken down by question category.
    Requires a 'category' column in *df*.
    """
    _apply_style()
    if "category" not in df.columns:
        raise ValueError("DataFrame must contain a 'category' column.")

    scored = df.dropna(subset=["vector_total", "pageindex_total"])
    if scored.empty:
        raise ValueError("No fully-scored questions in DataFrame.")

    cat_winner = (
        scored.groupby(["category", "winner"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cat_winner))
    w = 0.25
    columns = cat_winner.columns.tolist()

    for i, col in enumerate(columns):
        offset = (i - len(columns) / 2 + 0.5) * w
        color = PALETTE.get(col, PALETTE["neutral"])
        ax.bar(x + offset, cat_winner[col], w, label=col, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_winner.index, rotation=30, ha="right")
    ax.set_ylabel("Number of questions")
    ax.set_title("Wins by question category")
    ax.legend()

    fig.tight_layout()
    return fig


def save_all_charts(df: pd.DataFrame, output_dir: Path | str = ".") -> list[Path]:
    """
    Render and save all available charts to *output_dir*.
    Returns the list of paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    chart_fns = {
        "scores_per_question.png": plot_scores_per_question,
        "win_counts.png": plot_win_counts,
        "dimension_averages.png": plot_dimension_averages,
        "wins_by_category.png": plot_wins_by_category,
    }

    if "vector_time" in df.columns and "pageindex_time" in df.columns:
        chart_fns["latency.png"] = plot_latency

    saved: list[Path] = []
    for filename, fn in chart_fns.items():
        try:
            fig = fn(df)
            dest = output_dir / filename
            fig.savefig(dest, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(dest)
            print(f"[Chart] Saved {dest}")
        except Exception as exc:
            print(f"[Chart] Skipped {filename}: {exc}")

    return saved
