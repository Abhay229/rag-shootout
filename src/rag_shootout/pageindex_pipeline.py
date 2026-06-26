"""
pageindex_pipeline.py — PageIndex retrieval pipeline wrapper.

PageIndex is a vectorless retrieval approach: instead of embedding chunks
and doing similarity search, an LLM reasons over the document's tree
structure to decide which sections are relevant, then retrieves them.

This module wraps the pageindex SDK in the same interface as
vector_pipeline.py so the benchmark runner can call both identically.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TypedDict

from rag_shootout import config

logger = logging.getLogger(__name__)


class PageIndexResult(TypedDict, total=False):
    answer: str
    time_sec: float
    doc_id: str
    raw_response: str
    error: str


class PageIndexError(RuntimeError):
    """Raised when PageIndex fails to produce a valid answer."""


def _json_safe(value: Any) -> str:
    """Return a compact JSON/debug string for SDK responses and exceptions."""
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _extract_answer(response: Any) -> str:
    """
    Extract answer text from common PageIndex/OpenAI-compatible response shapes.

    The PageIndex SDK has historically exposed OpenAI-style chat completions.
    This helper accepts both dict-like and object-like SDK responses so API
    wrapper changes fail with a useful error instead of silently returning a
    placeholder score.
    """
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "").strip()
                return str(first.get("text") or first.get("content") or "").strip()

        for key in ("answer", "content", "message", "response", "output"):
            if key in response and response[key]:
                return str(response[key]).strip()

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            return str(getattr(message, "content", "") or "").strip()
        return str(getattr(first, "text", "") or getattr(first, "content", "") or "").strip()

    for attr in ("answer", "content", "message", "response", "output"):
        value = getattr(response, attr, None)
        if value:
            return str(value).strip()

    raise PageIndexError(
        "Could not extract an answer from PageIndex response. "
        f"Raw response: {_json_safe(response)}"
    )


def _looks_like_failure(answer: str) -> bool:
    lowered = answer.lower()
    failure_markers = [
        "skipped",
        "no credits",
        "insufficient credits",
        "unauthorized",
        "authentication",
        "api key",
        "rate limit",
        "quota",
        "error",
    ]
    return not answer.strip() or any(marker in lowered for marker in failure_markers)


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
        try:
            from importlib.metadata import version

            logger.info("Using pageindex SDK version %s", version("pageindex"))
        except Exception:
            logger.debug("Could not determine pageindex SDK version", exc_info=True)

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

        pdf_path = pdf_path.resolve()
        print(f"[PageIndex] Submitting {pdf_path} ...")
        logger.info("Submitting PDF to PageIndex: %s", pdf_path)

        try:
            result = self._client.submit_document(str(pdf_path))
        except Exception as exc:
            logger.exception("PageIndex document submission failed")
            raise PageIndexError(f"PageIndex document submission failed: {exc}") from exc

        logger.debug("PageIndex submit response: %s", _json_safe(result))
        if not isinstance(result, dict) or not result.get("doc_id"):
            raise PageIndexError(
                "PageIndex submit_document did not return a doc_id. "
                f"Raw response: {_json_safe(result)}"
            )

        doc_id: str = str(result["doc_id"])
        print(f"[PageIndex] Document ID: {doc_id}")
        print("[PageIndex] Waiting for tree generation (1-3 min) ...")

        while True:
            try:
                doc = self._client.get_document(doc_id)
            except Exception as exc:
                logger.exception("PageIndex document status check failed")
                raise PageIndexError(
                    f"PageIndex document status check failed for doc_id={doc_id}: {exc}"
                ) from exc

            logger.debug("PageIndex document response: %s", _json_safe(doc))
            status = doc.get("status") if isinstance(doc, dict) else getattr(doc, "status", None)
            if status == "completed":
                is_retrieval_ready = getattr(self._client, "is_retrieval_ready", None)
                if callable(is_retrieval_ready):
                    try:
                        if not is_retrieval_ready(doc_id):
                            print("[PageIndex] Status: completed, retrieval not ready yet ...")
                            time.sleep(poll_interval)
                            continue
                    except Exception as exc:
                        logger.exception("PageIndex retrieval readiness check failed")
                        raise PageIndexError(
                            f"PageIndex retrieval readiness check failed for doc_id={doc_id}: {exc}"
                        ) from exc
                print("[PageIndex] Tree generation complete.")
                break
            if status == "failed":
                raise PageIndexError(
                    f"PageIndex processing failed for doc_id={doc_id}. "
                    f"Raw response: {_json_safe(doc)}"
                )
            if status is None:
                raise PageIndexError(
                    f"PageIndex status response did not include status for doc_id={doc_id}. "
                    f"Raw response: {_json_safe(doc)}"
                )
            print(f"[PageIndex] Status: {status} ...")
            time.sleep(poll_interval)

        self._doc_id = doc_id
        return doc_id

    def set_doc_id(self, doc_id: str) -> None:
        """Manually set a previously-submitted document ID to skip re-submission."""
        doc_id = doc_id.strip()
        if not doc_id:
            raise ValueError("doc_id cannot be empty.")
        self._doc_id = doc_id
        logger.info("Using existing PageIndex doc_id=%s", doc_id)

    # ── Generation ────────────────────────────────────────────────────────────

    def answer(self, query: str) -> PageIndexResult:
        """
        Run a query against the submitted document and return a PageIndexResult.
        Raises RuntimeError if submit() has not been called.
        """
        if self._doc_id is None:
            raise RuntimeError("Call submit() (or set_doc_id()) before answer().")

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty.")

        t0 = time.perf_counter()
        logger.info("PageIndex query doc_id=%s question=%r", self._doc_id, query)

        try:
            response = self._client.chat_completions(
                messages=[{"role": "user", "content": query}],
                doc_id=self._doc_id,
            )
            raw_response = _json_safe(response)
            answer_text = _extract_answer(response)
            if _looks_like_failure(answer_text):
                raise PageIndexError(
                    "PageIndex returned a failure/placeholder answer. "
                    f"Answer: {answer_text!r}. Raw response: {raw_response}"
                )

            elapsed = round(time.perf_counter() - t0, 3)
            logger.debug("PageIndex raw response: %s", raw_response)
            logger.info("PageIndex answered in %.3fs", elapsed)
            return PageIndexResult(
                answer=answer_text,
                time_sec=elapsed,
                doc_id=self._doc_id,
                raw_response=raw_response,
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            logger.exception("PageIndex answer failed for doc_id=%s", self._doc_id)
            if isinstance(exc, PageIndexError):
                message = str(exc)
            else:
                message = f"{type(exc).__name__}: {exc}"
            raise PageIndexError(
                f"PageIndex answer failed after {elapsed}s for question={query!r}: {message}"
            ) from exc
