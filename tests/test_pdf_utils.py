"""Tests for pdf_utils.py — chunking and extraction logic."""

import pytest

from rag_shootout.pdf_utils import Chunk, PageText, chunk_pages


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_page(page: int, text: str) -> PageText:
    return {"page": page, "text": text}


def _make_pages(n: int = 3, chars_per_page: int = 2000) -> list[PageText]:
    return [
        _make_page(i + 1, f"Page {i + 1} content. " + ("x" * chars_per_page))
        for i in range(n)
    ]


# ─── chunk_pages ──────────────────────────────────────────────────────────────


class TestChunkPages:
    def test_produces_chunks(self):
        pages = _make_pages(n=2, chars_per_page=2000)
        chunks = chunk_pages(pages, chunk_size=800, overlap=150)
        assert len(chunks) > 0

    def test_chunks_carry_page_number(self):
        pages = _make_pages(n=2, chars_per_page=2000)
        chunks = chunk_pages(pages, chunk_size=800, overlap=150)
        assert all("page" in c for c in chunks)
        assert all(c["page"] in {1, 2} for c in chunks)

    def test_chunk_text_not_empty(self):
        pages = _make_pages(n=1, chars_per_page=500)
        chunks = chunk_pages(pages, chunk_size=200, overlap=50)
        assert all(c["text"].strip() for c in chunks)

    def test_chunk_size_respected(self):
        pages = _make_pages(n=1, chars_per_page=2000)
        chunks = chunk_pages(pages, chunk_size=300, overlap=50)
        # Chunks should not exceed chunk_size (may be smaller at end of page)
        assert all(len(c["text"]) <= 300 for c in chunks)

    def test_empty_pages_skipped(self):
        pages = [
            _make_page(1, ""),
            _make_page(2, "   "),
            _make_page(3, "Actual content here."),
        ]
        chunks = chunk_pages(pages, chunk_size=200, overlap=50)
        assert all(c["page"] == 3 for c in chunks)

    def test_overlap_greater_than_chunk_size_raises(self):
        pages = _make_pages(n=1, chars_per_page=500)
        with pytest.raises(ValueError, match="greater than overlap"):
            chunk_pages(pages, chunk_size=100, overlap=200)

    def test_short_text_produces_one_chunk(self):
        pages = [_make_page(1, "Short text.")]
        chunks = chunk_pages(pages, chunk_size=800, overlap=150)
        assert len(chunks) == 1
        assert chunks[0]["page"] == 1

    def test_whitespace_normalised(self):
        pages = [_make_page(1, "Hello   \n\n  world")]
        chunks = chunk_pages(pages, chunk_size=800, overlap=0)
        # Multiple spaces and newlines should be collapsed
        assert "  " not in chunks[0]["text"]
        assert "\n" not in chunks[0]["text"]

    def test_overlap_increases_chunk_count(self):
        pages = _make_pages(n=1, chars_per_page=3000)
        chunks_no_overlap = chunk_pages(pages, chunk_size=500, overlap=0)
        chunks_with_overlap = chunk_pages(pages, chunk_size=500, overlap=200)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)
