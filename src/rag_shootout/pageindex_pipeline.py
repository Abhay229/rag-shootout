"""
pageindex_pipeline.py — PageIndex retrieval pipeline wrapper.

PageIndex is a vectorless retrieval approach: instead of embedding chunks
and doing similarity search, an LLM reasons over the document's tree
structure to decide which sections are relevant, then retrieves them.

This module wraps the pageindex SDK in the same interface as
vector_pipeline.py so the benchmark runner can call both identically.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

from rag_shootout import config


class PageIndexResult(TypedDict):
    answer: str
    time_sec: float


class PageIndexPipeline:
    """
    Thin wrapper around the PageIndex SDK.

    Usage:
        pipeline = PageIndexPipeline()
        pipeline.submit(pdf_path)    # one-time, takes ~1-3 min
        result = pipeline.answer("What RL algorithm was used?")
    """

    def __init__(self) -> None:
        try:
            from pageindex import PageIndexClient
        except ImportError as exc:
            raise ImportError(
                "pageindex is not installed. Run: pip install pageindex"
            ) from exc

        self._client = PageIndexClient(api_key=config.PAGEINDEX_API_KEY)
        self._doc_id: str | None = None

    # ── Submission ────────────────────────────────────────────────────────────

    def submit(
        self,
        pdf_path: Path | str,
        poll_interval: int = config.PAGEINDEX_POLL_INTERVAL_SEC,
    ) -> str:
        """
        Upload *pdf_path* to PageIndex and wait until tree generation completes.

        Returns the document ID, which is cached internally.
        Raises RuntimeError if processing fails.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"[PageIndex] Submitting {pdf_path.name} ...")
        result = self._client.submit_document(str(pdf_path))
        doc_id: str = result["doc_id"]
        print(f"[PageIndex] Document ID: {doc_id}")
        print("[PageIndex] Waiting for tree generation (1-3 min) ...")

        while True:
            status = self._client.get_document(doc_id)["status"]
            if status == "completed":
                print("[PageIndex] Tree generation complete.")
                break
            if status == "failed":
                raise RuntimeError(
                    f"PageIndex processing failed for doc_id={doc_id}"
                )
            print(f"[PageIndex] Status: {status} ...")
            time.sleep(poll_interval)

        self._doc_id = doc_id
        return doc_id

    def set_doc_id(self, doc_id: str) -> None:
        """Manually set a previously-submitted document ID to skip re-submission."""
        self._doc_id = doc_id

    # ── Generation ────────────────────────────────────────────────────────────

    def answer(self, query: str) -> PageIndexResult:
        """
        Run a query against the submitted document and return a PageIndexResult.
        Raises RuntimeError if submit() has not been called.
        """
        if self._doc_id is None:
            raise RuntimeError("Call submit() (or set_doc_id()) before answer().")

        t0 = time.perf_counter()

        response = self._client.chat_completions(
            messages=[{"role": "user", "content": query}],
            doc_id=self._doc_id,
        )
        answer_text: str = response["choices"][0]["message"]["content"]

        return PageIndexResult(
            answer=answer_text,
            time_sec=round(time.perf_counter() - t0, 3),
        )
