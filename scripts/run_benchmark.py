#!/usr/bin/env python
"""
run_benchmark.py — CLI runner for the RAG Shootout benchmark.

Runs all 4 questions through both pipelines and saves raw results to a CSV.
Scoring is still manual — open the CSV alongside the source PDF to score.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --output results/my_run.csv
    python scripts/run_benchmark.py --questions 1 3 5   # subset of questions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rag_shootout import config
from rag_shootout.pdf_utils import chunk_pages, download_pdf, extract_pages
from rag_shootout.pageindex_pipeline import PageIndexPipeline
from rag_shootout.questions import get_questions
from rag_shootout.vector_pipeline import VectorRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Shootout CLI runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR / "benchmark_results.csv",
        help="Path to write the results CSV (default: results/benchmark_results.csv)",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        type=int,
        default=None,
        help="Question IDs to run (default: all 4). E.g. --questions 1 3 4",
    )
    parser.add_argument(
        "--skip-pageindex",
        action="store_true",
        help="Run only the Vector RAG pipeline (skips PageIndex API calls)",
    )
    parser.add_argument(
        "--pageindex-doc-id",
        type=str,
        default=None,
        help="Re-use an existing PageIndex document ID to skip re-submission",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate env ──────────────────────────────────────────────────────────
    missing = config.validate_env()
    if args.skip_pageindex:
        missing = [m for m in missing if m != "PAGEINDEX_API_KEY"]
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("        Copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    # ── Load questions ─────────────────────────────────────────────────────────
    all_questions = get_questions()
    if args.questions:
        questions = [q for q in all_questions if q.id in args.questions]
        if not questions:
            print(f"[ERROR] No questions matched IDs: {args.questions}")
            sys.exit(1)
    else:
        questions = all_questions

    print(f"\n{'='*60}")
    print(f"  RAG SHOOTOUT — {len(questions)} question(s)")
    print(f"  Model:     {config.MODEL}")
    print(f"  Embedding: {config.EMBEDDING_MODEL}")
    print(f"  Chunks:    size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}, top_k={config.TOP_K}")
    print(f"{'='*60}\n")

    # ── Download & chunk PDF ──────────────────────────────────────────────────
    pdf_path = download_pdf()
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)

    # ── Build Vector RAG index ────────────────────────────────────────────────
    vector_pipeline = VectorRAGPipeline()
    vector_pipeline.index(chunks)

    # ── Submit to PageIndex ───────────────────────────────────────────────────
    pageindex_pipeline: PageIndexPipeline | None = None
    if not args.skip_pageindex:
        pageindex_pipeline = PageIndexPipeline()
        if args.pageindex_doc_id:
            pageindex_pipeline.set_doc_id(args.pageindex_doc_id)
        else:
            pageindex_pipeline.submit(pdf_path)

    # ── Run the benchmark ─────────────────────────────────────────────────────
    rows: list[dict] = []
    cfg = config.BenchmarkConfig()

    for i, q in enumerate(questions):
        print(f"\n{'─'*60}")
        print(f"  Q{q.id}/{len(questions)} [{q.category}]")
        print(f"  {q.text}")
        print(f"{'─'*60}")

        # Vector RAG
        vr = vector_pipeline.answer(q.text)
        print(f"\n  [Vector RAG] {vr['time_sec']}s | pages {vr['pages_retrieved']}")
        print(f"  {vr['answer'][:400]}")

        row: dict = {
            "question_id": q.id,
            "category": q.category,
            "question": q.text,
            "vector_answer": vr["answer"],
            "vector_time": vr["time_sec"],
            "vector_pages": json.dumps(vr["pages_retrieved"]),
            "vector_top_k_scores": json.dumps(vr["top_k_scores"]),
            # Scoring columns — fill in manually
            "vector_accuracy": None,
            "vector_completeness": None,
            "vector_faithfulness": None,
            "pageindex_answer": None,
            "pageindex_time": None,
            "pageindex_accuracy": None,
            "pageindex_completeness": None,
            "pageindex_faithfulness": None,
            # Config snapshot for reproducibility
            "config_model": cfg.model,
            "config_embedding": cfg.embedding_model,
            "config_chunk_size": cfg.chunk_size,
            "config_overlap": cfg.chunk_overlap,
            "config_top_k": cfg.top_k,
        }

        # PageIndex
        if pageindex_pipeline is not None:
            pi = pageindex_pipeline.answer(q.text)
            print(f"\n  [PageIndex] {pi['time_sec']}s")
            print(f"  {pi['answer'][:400]}")
            row["pageindex_answer"] = pi["answer"]
            row["pageindex_time"] = pi["time_sec"]

        rows.append(row)

        if i < len(questions) - 1:
            time.sleep(2)  # be polite to both APIs

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[OK] Results saved → {args.output}")
    print(
        "\nNext step: open the CSV alongside the source PDF and fill in the "
        "accuracy/completeness/faithfulness columns (1–5) for each answer."
    )


if __name__ == "__main__":
    main()
