"""Tests for vector_pipeline.py — retrieval logic (no API calls)."""

import numpy as np
import pytest

from rag_shootout.pdf_utils import Chunk
from rag_shootout.vector_pipeline import VectorRAGPipeline


def _make_chunks(n: int = 20) -> list[Chunk]:
    """Create synthetic chunks for testing."""
    topics = [
        "reinforcement learning and reward shaping",
        "transformer architecture and attention mechanism",
        "benchmark evaluation on AIME mathematics",
        "cold start data and training pipeline",
        "language consistency and multilingual issues",
    ]
    chunks: list[Chunk] = []
    for i in range(n):
        topic = topics[i % len(topics)]
        chunks.append({"page": (i // 5) + 1, "text": f"Chunk {i}: {topic}. " * 10})
    return chunks


@pytest.fixture(scope="module")
def indexed_pipeline() -> VectorRAGPipeline:
    """Build and index a pipeline once per test module (embedder is slow to load)."""
    pipeline = VectorRAGPipeline(top_k=3)
    pipeline.index(_make_chunks(n=20), show_progress=False)
    return pipeline


class TestVectorRAGPipelineIndexing:
    def test_index_sets_chunks(self, indexed_pipeline):
        assert len(indexed_pipeline._chunks) == 20

    def test_index_sets_embeddings(self, indexed_pipeline):
        assert indexed_pipeline._embeddings is not None
        assert indexed_pipeline._embeddings.shape[0] == 20

    def test_embeddings_normalised(self, indexed_pipeline):
        norms = np.linalg.norm(indexed_pipeline._embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(20), atol=1e-5)

    def test_index_empty_raises(self):
        pipeline = VectorRAGPipeline()
        with pytest.raises(ValueError, match="empty"):
            pipeline.index([])

    def test_retrieve_before_index_raises(self):
        pipeline = VectorRAGPipeline()
        with pytest.raises(RuntimeError, match="index"):
            pipeline.retrieve("some query")


class TestVectorRAGPipelineRetrieval:
    def test_retrieve_returns_top_k(self, indexed_pipeline):
        results = indexed_pipeline.retrieve("reinforcement learning")
        assert len(results) == indexed_pipeline.top_k

    def test_retrieve_returns_chunks_and_scores(self, indexed_pipeline):
        results = indexed_pipeline.retrieve("reinforcement learning")
        for chunk, score in results:
            assert "text" in chunk
            assert "page" in chunk
            assert isinstance(score, float)

    def test_scores_descending(self, indexed_pipeline):
        results = indexed_pipeline.retrieve("reinforcement learning")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self, indexed_pipeline):
        results = indexed_pipeline.retrieve("transformer attention")
        for _, score in results:
            assert -1.0 <= score <= 1.0

    def test_relevant_chunks_retrieved(self, indexed_pipeline):
        """The top result for a specific topic should mention that topic."""
        results = indexed_pipeline.retrieve("reinforcement learning reward")
        top_chunk, _ = results[0]
        assert "reinforcement" in top_chunk["text"].lower()

    def test_different_queries_different_top_results(self, indexed_pipeline):
        r1 = indexed_pipeline.retrieve("reinforcement learning")
        r2 = indexed_pipeline.retrieve("transformer architecture attention")
        # Top chunks should differ for clearly different queries
        top1_text = r1[0][0]["text"]
        top2_text = r2[0][0]["text"]
        assert top1_text != top2_text
