# Methodology Notes

## Why This Benchmark Exists

Most RAG comparisons suffer from at least one of:
- **Toy datasets** (5 hand-written documents, curated to make one approach look good)
- **LLM-as-judge scoring** (the same model grading itself is a known bias source)
- **Cherry-picked questions** (all factual lookups, no synthesis, no numeric retrieval)

This benchmark tries to fix all three. It runs against a real research paper, uses human scoring, and deliberately mixes question types to expose where each approach breaks.

---

## Vector RAG Pipeline

### 1. Text extraction
pypdf is used for text extraction. For papers with complex two-column layouts, some reflow artifacts may appear — this is a known limitation of PDF text extraction and affects both pipelines equally.

### 2. Chunking
Fixed-size character chunking with overlap:
- **Chunk size:** 800 characters
- **Overlap:** 150 characters
- **Rationale:** 800 chars ≈ 2-3 sentences for dense academic text, small enough to fit comfortably in LLM context, large enough to preserve sentence-level meaning. The overlap prevents key sentences from being split across chunk boundaries.

### 3. Embedding
`all-MiniLM-L6-v2` from sentence-transformers:
- 384-dimensional dense vectors
- Optimised for semantic similarity on English text
- Fast to run locally (no API dependency)
- Good balance of speed and quality at the scale of a single paper

### 4. Retrieval
Cosine similarity via dot product on L2-normalised vectors. Top-5 chunks are retrieved per query.

### 5. Generation
Retrieved chunks are concatenated with source page numbers into a context block. The LLM is prompted with a strict "answer from this context only" instruction to reduce hallucination.

---

## PageIndex Pipeline

PageIndex uses a different paradigm entirely: instead of embedding chunks and doing similarity search, an LLM reasons over a tree representation of the document to decide which sections to retrieve.

The key difference is **where intelligence is applied**:
- Vector RAG: intelligence is in the LLM's generation step; retrieval is dumb (nearest-neighbor)
- PageIndex: intelligence is applied at retrieval time; the LLM decides what's relevant

This is why PageIndex tends to do better on multi-hop synthesis (it can reason about which sections are relevant before retrieving) and worse on simple factual lookups (the overhead of tree reasoning isn't worth it for a single-section answer).

---

## Scoring Rubric

All scoring is done **manually by reading the source PDF**. This is intentional — LLM-grading-itself is a known bias (the model that generated the answer tends to rate itself highly).

### Dimensions

**Accuracy (1–5)**
- 1: Factually wrong
- 2: Mostly wrong, one correct detail
- 3: Partially correct, key errors
- 4: Mostly correct, minor errors
- 5: Fully correct per the source paper

**Completeness (1–5)**
- 1: Misses the point entirely
- 2: Addresses <25% of what was asked
- 3: Addresses ~50% — partial but useful
- 4: Addresses >75% — minor gaps
- 5: Covers everything the question asked

**Faithfulness (1–5)**
- 1: Clear fabrications / hallucinations
- 2: Significant speculation beyond the text
- 3: Mostly grounded, some inference
- 4: Almost fully grounded
- 5: Every claim traceable to the source

### Scoring tips

Keep the PDF open in another tab while scoring. The `vector_pages` column in the results CSV tells you exactly which pages Vector RAG retrieved — check those pages to see whether the answer was actually in the retrieved context.

For PageIndex answers, the approach doesn't expose which pages were retrieved, so check the full paper for each claim.

---

## Limitations

1. **Single paper.** Results may not generalise to other document types (contracts, textbooks, code docs). Run on your own documents to validate.

2. **Single LLM.** The model (default: Qwen 3.5 122B via OpenRouter) affects both pipelines. A weaker model may widen the gap; a stronger one may narrow it.

3. **Single evaluator.** Human scoring introduces personal judgment. Inter-rater reliability would require multiple scorers and a reconciliation process.

4. **PageIndex API version.** Results may change as PageIndex updates its tree-generation and retrieval logic.

5. **Chunking is not tuned.** The 800/150 chunk_size/overlap defaults are reasonable starting points, not optimal values. See the "Extending" section in the README.

---

## Ablation Ideas

- **Chunk size:** Try 400 and 1200. How do scores shift for numeric vs. synthesis questions?
- **Top-k:** Try k=3 and k=8. Does more context help synthesis, or hurt precision?
- **Embedding model:** Swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` or `bge-large-en-v1.5`.
- **BM25:** Add a third pipeline using keyword-based BM25 retrieval (`rank_bm25`). A three-way comparison makes for a much more interesting portfolio piece.
- **Hybrid retrieval:** Combine BM25 and vector scores (reciprocal rank fusion) and compare against either approach alone.
