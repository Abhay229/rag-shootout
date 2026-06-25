# Can LLMs Find Answers Without Vector Search?

I built two RAG pipelines from scratch and tested them on 10 questions about the same research paper — to find out exactly where each one breaks.

---

## What This Is

RAG (Retrieval-Augmented Generation) is the technique of feeding an LLM relevant text from a document before asking it a question. There are different ways to decide *which* text to retrieve. This project compares two of them:

**Vector RAG** — the classic approach. Break the PDF into chunks, convert them to numbers (embeddings), and find the chunks whose numbers are closest to the question's numbers. Simple and fast.

**PageIndex** — a newer approach. Instead of number-matching, an LLM actually reads a map of the document and *reasons* about which sections are relevant. No embeddings needed.

I ran both on the [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) using 10 questions designed to expose where each approach fails — not just easy lookups, but multi-hop reasoning, exact number retrieval, and synthesis questions that require reading multiple sections at once.

---

## Why I Built This

Most RAG comparisons I found online test on toy data (5 made-up documents) or use the LLM to grade its own answers. Both of those are unfair tests. I wanted:

- A real document (a dense 20-page research paper)
- Questions that actually stress-test retrieval, not just "what is X" lookups
- Human scoring against the source — no LLM judging itself

---

## The 10 Questions

Each question is designed to stress a different weakness:

| # | Type | What I'm testing |
|---|---|---|
| 1–2 | Simple factual | Sanity check — both should pass |
| 3 | Exact number | Numbers often get split across chunk boundaries |
| 4 | Two-section | Needs info from two different parts of the paper |
| 5 | Full summary | Answer is spread across many sections |
| 6 | List from a table | Tables are notoriously hard to chunk |
| 7 | Specific story | One paragraph buried in the paper |
| 8 | What didn't work | Failure modes aren't in the top results |
| 9 | Trade-off | Why did they make this choice, not just what |
| 10 | Cross-table comparison | Hardest — requires reading multiple benchmark tables |

---

## Results

*(Run the notebook and fill these in with your actual numbers)*

- **Overall winner:**
- **Vector RAG was better at:**
- **PageIndex was better at:**
- **Biggest surprise:**
- **Speed difference:**

---

## How It Works

```
PDF
 │
 ├─── Vector RAG ──────────────────────────────────────────────────►
 │     Step 1: Extract text from each page
 │     Step 2: Split into 800-character chunks with 150-char overlap
 │     Step 3: Convert chunks to numbers (embeddings)
 │     Step 4: On each question, find the 5 most similar chunks
 │     Step 5: Feed those chunks to the LLM and ask it to answer
 │
 └─── PageIndex ───────────────────────────────────────────────────►
       Step 1: Upload the PDF once (builds a document map)
       Step 2: On each question, the LLM reads the map and picks sections
       Step 3: Returns an answer grounded in those sections

Both answers → you score them 1–5 → charts show who won and where
```

---

## Project Layout

```
rag-shootout/
│
├── notebooks/
│   └── rag_shootout.ipynb       ← Start here
│
├── src/rag_shootout/
│   ├── config.py                ← Change models, chunk sizes here
│   ├── pdf_utils.py             ← Download + split the PDF
│   ├── vector_pipeline.py       ← Vector RAG logic
│   ├── pageindex_pipeline.py    ← PageIndex logic
│   ├── questions.py             ← The 10 test questions
│   ├── scoring.py               ← Score tracking + summary stats
│   └── visualization.py         ← All the charts
│
├── tests/                       ← Unit tests (no API keys needed)
├── scripts/run_benchmark.py     ← Run from terminal instead of notebook
├── results/                     ← Your CSVs land here
├── docs/methodology.md          ← Deeper explanation of design choices
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup (5 minutes)

**Step 1 — Clone and install**
```bash
git clone https://github.com/YOUR_USERNAME/rag-shootout.git
cd rag-shootout
pip install -r requirements.txt
```

**Step 2 — Get API keys**

You need two free accounts:
- [OpenRouter](https://openrouter.ai) — for the LLM (free tier works)
- [PageIndex](https://pageindex.ai) — for the tree-reasoning retrieval

**Step 3 — Add your keys**
```bash
cp .env.example .env
# Open .env and paste your keys in
```

**Step 4 — Open the notebook**
```bash
jupyter lab notebooks/rag_shootout.ipynb
```

Or run from terminal without a notebook:
```bash
python scripts/run_benchmark.py
```

---

## Tech Stack

| What | How |
|---|---|
| PDF reading | `pypdf` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| LLM calls | `openai` SDK → [OpenRouter](https://openrouter.ai) |
| Vector-free retrieval | [PageIndex](https://pageindex.ai) |
| Data | `pandas` |
| Charts | `matplotlib` |

---

## Try Your Own Document

Change `PDF_URL` in `src/rag_shootout/config.py` to any PDF link and rewrite the questions in `src/rag_shootout/questions.py`. Everything else stays the same.

---

## License

MIT
