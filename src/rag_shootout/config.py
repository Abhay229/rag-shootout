"""
config.py — Central configuration for the RAG Shootout benchmark.

All tuneable parameters live here so you can run ablation experiments
by changing a single value rather than hunting through the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Document ─────────────────────────────────────────────────────────────────

PDF_URL = "https://arxiv.org/pdf/2501.12948.pdf"
PDF_PATH = ROOT_DIR / "sample_paper.pdf"

# ─── LLM ──────────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")

# Any OpenRouter-supported model. Swap freely to compare cost/quality.
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# ─── PageIndex ────────────────────────────────────────────────────────────────

PAGEINDEX_API_KEY: str = os.environ.get("PAGEINDEX_API_KEY", "")
PAGEINDEX_POLL_INTERVAL_SEC = 10  # seconds between status checks

# ─── Vector RAG chunking ──────────────────────────────────────────────────────

CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks
TOP_K = 5               # number of chunks to retrieve per query

# Embedding model — any sentence-transformers compatible model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Scoring ──────────────────────────────────────────────────────────────────

SCORE_DIMENSIONS = ["accuracy", "completeness", "faithfulness"]
SCORE_MIN = 1
SCORE_MAX = 5


@dataclass
class BenchmarkConfig:
    """
    Snapshot of all config values at benchmark time.
    Saved alongside results so runs are fully reproducible.
    """
    pdf_url: str = PDF_URL
    model: str = MODEL
    embedding_model: str = EMBEDDING_MODEL
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K
    llm_temperature: float = LLM_TEMPERATURE
    llm_max_tokens: int = LLM_MAX_TOKENS
    score_dimensions: list[str] = field(default_factory=lambda: list(SCORE_DIMENSIONS))

    def to_dict(self) -> dict:
        return {
            "pdf_url": self.pdf_url,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "score_dimensions": self.score_dimensions,
        }


def validate_env() -> list[str]:
    """Return a list of missing environment variable names."""
    missing = []
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if not PAGEINDEX_API_KEY:
        missing.append("PAGEINDEX_API_KEY")
    return missing
