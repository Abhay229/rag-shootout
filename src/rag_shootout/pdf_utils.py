"""
pdf_utils.py — PDF download and text extraction utilities.

Handles downloading the benchmark PDF and extracting per-page text
in a clean, testable way that's decoupled from the retrieval logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import requests
from pypdf import PdfReader

from rag_shootout import config


class PageText(TypedDict):
    page: int       # 1-indexed page number
    text: str       # cleaned text content


class Chunk(TypedDict):
    page: int       # source page (1-indexed)
    text: str       # chunk text


def download_pdf(
    url: str = config.PDF_URL,
    dest: Path = config.PDF_PATH,
    force: bool = False,
) -> Path:
    """
    Download a PDF to *dest*. Skips the download if the file already exists
    (unless *force=True*).

    Returns the path to the downloaded file.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"[PDF] Using cached file: {dest} ({dest.stat().st_size // 1024} KB)")
        return dest

    print(f"[PDF] Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"[PDF] Saved to {dest} ({len(resp.content) // 1024} KB)")
    return dest


def extract_pages(pdf_path: Path | str) -> list[PageText]:
    """
    Extract per-page text from a PDF using pypdf.

    Returns a list of PageText dicts, one per page, in document order.
    Pages with no extractable text return an empty string (scan pages).
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        pages.append({"page": i + 1, "text": raw})

    print(f"[PDF] Extracted text from {len(pages)} pages ({pdf_path.name})")
    return pages


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Fixed-size character chunking with overlap.

    Each chunk carries its source *page* number so retrieved chunks can
    be grounded back to the original document.

    Args:
        pages:       Output of extract_pages().
        chunk_size:  Target characters per chunk.
        overlap:     Character overlap between consecutive chunks.

    Returns:
        List of Chunk dicts, ordered by page then position.
    """
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")

    chunks: list[Chunk] = []

    for p in pages:
        # Normalise whitespace — join hyphenated line-breaks etc.
        text = re.sub(r"\s+", " ", p["text"]).strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            chunk_text = text[start : start + chunk_size]
            if chunk_text.strip():
                chunks.append({"page": p["page"], "text": chunk_text})
            start += chunk_size - overlap

    print(
        f"[Chunk] Created {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap}) "
        f"from {len(pages)} pages"
    )
    return chunks
