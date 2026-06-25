"""
vector_pipeline.py — Full Vector RAG pipeline.

Pipeline:
  1. Embed all chunks with sentence-transformers
  2. On query: embed the query, run cosine similarity, retrieve top-k
  3. Build a context string from retrieved chunks
  4. Generate an answer from an LLM conditioned on that context

The pipeline is intentionally self-contained so it can be swapped out
or extended without touching other modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from rag_shootout import config
from rag_shootout.pdf_utils import Chunk


# ─── Types ────────────────────────────────────────────────────────────────────


class VectorResult(TypedDict):
    answer: str
    time_sec: float
    pages_retrieved: list[int]
    top_k_scores: list[float]


# ─── Pipeline ─────────────────────────────────────────────────────────────────


@dataclass
class VectorRAGPipeline:
    """
    Embeddings-based retrieval pipeline.

    Usage:
        pipeline = VectorRAGPipeline()
        pipeline.index(chunks)
        result = pipeline.answer("What RL algorithm was used?")
    """

    embedding_model: str = config.EMBEDDING_MODEL
    top_k: int = config.TOP_K
    llm_model: str = config.MODEL
    temperature: float = config.LLM_TEMPERATURE
    max_tokens: int = config.LLM_MAX_TOKENS

    # Internal state — populated by index()
    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _embeddings: np.ndarray | None = field(default=None, repr=False)
    _embedder: SentenceTransformer | None = field(default=None, repr=False)
    _client: OpenAI | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk], show_progress: bool = True) -> None:
        """
        Embed *chunks* and store the resulting matrix.
        Must be called before answer().
        """
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        print(f"[VectorRAG] Loading embedder: {self.embedding_model}")
        self._embedder = SentenceTransformer(self.embedding_model)

        texts = [c["text"] for c in chunks]
        print(f"[VectorRAG] Embedding {len(texts)} chunks ...")
        self._embeddings = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        self._chunks = chunks
        print(f"[VectorRAG] Index ready — shape: {self._embeddings.shape}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """
        Return the top-k chunks most similar to *query*.

        Returns:
            List of (Chunk, cosine_score) tuples, descending by score.
        """
        if self._embedder is None or self._embeddings is None:
            raise RuntimeError("Call index() before retrieve().")

        q_emb = self._embedder.encode([query], normalize_embeddings=True)
        scores = np.dot(self._embeddings, q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]

    # ── Generation ────────────────────────────────────────────────────────────

    def answer(self, query: str) -> VectorResult:
        """
        Run the full RAG pipeline for *query* and return a VectorResult.
        """
        t0 = time.perf_counter()

        hits = self.retrieve(query)
        context = "\n\n".join(
            f"[Page {c['page']}] {c['text']}" for c, _ in hits
        )

        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say so explicitly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer_text = response.choices[0].message.content or ""

        return VectorResult(
            answer=answer_text,
            time_sec=round(time.perf_counter() - t0, 3),
            pages_retrieved=sorted({c["page"] for c, _ in hits}),
            top_k_scores=[round(score, 4) for _, score in hits],
        )
