# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI-driven financial research platform: a user asks a natural-language question about a company ("What are Apple's key risk factors this year?") and gets an answer grounded in real SEC EDGAR filings via RAG, plus live market data via tools, with citations resolving back to the exact filing and section.

**The project is mid-build and follows a strict incremental plan.** The full design and build order live in `docs/PLAN.md` — read it before making architectural changes, since most decisions here were made deliberately and have recorded rationale.

`docs/PLAN.md` is gitignored (local working copy). If it is missing after a fresh clone, the canonical original is at `~/.claude/plans/want-to-design-a-golden-barto.md`.

## Build status

Work proceeds module by module, and **each step stops for human verification before the next begins**. Do not run ahead of the current step.

| Module | Steps | Status |
|---|---|---|
| **A** — data pipeline (EDGAR → Postgres, retrieval) | A0–A8 | A0 done; A1 next |
| **B** — live data (Finnhub, Tavily) | B1–B2 | not started |
| **C** — LangGraph agent | C1–C2 | not started |
| **D** — FastAPI service | D1 | not started |

Module A step map: A0 schema · A1 `tickers.py` · A2/A3 `edgar.py` · A4 `parser.py` · A5 `chunker.py` · A6 `embeddings.py`+`store.py`+`scripts/ingest.py` · A7 dense retrieval · A8 hybrid retrieval.

Dependencies for unstarted modules are deliberately **not** in `pyproject.toml` — add them when the module is built, not before.

## Commands

```bash
# Postgres (pgvector). --wait blocks until the healthcheck passes.
docker compose up -d --wait
docker compose down          # add -v to also drop the data volume

# Environment. Must be Python 3.13 — see constraint below.
/usr/local/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# A0 gate: applies schema (twice, proving idempotency) and dumps structure.
.venv/bin/python scripts/check_db.py

# Tests
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_parser.py::test_name   # single test

# psql is NOT installed on the host; use the container's client
docker compose exec db psql -U postgres -d research
```

## Architecture

Data flows in one direction, and each stage is a separate file so it can be tested alone:

```
EDGAR API → tickers → edgar → parser → chunker → embeddings → store → Postgres
                                                                        ↓
                                                                    retriever
                                                                        ↓
Finnhub + Tavily ─────────────────────────→ tools.py ←──────────────────┘
                                                ↓
                                            agent.py  (LangGraph)
                                                ↓
                                             api.py
```

### Storage is Postgres + pgvector, and that choice drives the retrieval design

Filings are full of exact-match tokens that dense embeddings handle badly (`material weakness`, `going concern`, `fiscal 2024`). Retrieval is therefore **hybrid**: dense vector similarity + BM25 full-text, fused with Reciprocal Rank Fusion in a **single SQL query** over one table. RRF is used rather than weighted score blending because cosine distance and `ts_rank_cd` are on incomparable scales.

This is why the project uses pgvector rather than Chroma — Chroma has no real BM25, so hybrid there would mean a second index kept in sync. Postgres also serves as the ingestion ledger and, later, the session/eval store.

### The agent is a 2-node ReAct loop, not a router graph

`agent_node ⇄ tool_node`, looping until the model stops emitting tool calls. There is deliberately **no router node and no synthesis node**: tool-calling models already route, and the final tool-call-free turn *is* the synthesis.

The load-bearing consequence: **tools are list entries, not graph nodes.** Adding a capability means appending one `@tool` function — never modifying the graph. Retrieval is itself a tool (`search_filings`), which is what lets multi-hop questions work (the agent just calls it twice with different filters).

With five tools, three of which plausibly answer "what's going on with Apple," **tool docstrings are the routing logic** — they must say what the tool is for *and* when to prefer a sibling. If the agent picks the wrong source, fix the docstring, not the graph.

### Citations are assembled in code, never by the LLM

Models mangle URLs and invent accession numbers, while the retriever already knows ground truth. Citation objects are built programmatically as tool results flow through, and carry a `type` of `filing` / `news` / `web` so an audited SEC filing is visually distinguishable from a news article.

## Constraints that will silently break things

**Embedding dimension is coupled across three places.** `bge-small-en-v1.5` → 384 dims → `vector(384)` in `sql/schema.sql` → `settings.embedding_dim`. Changing the model means changing all three plus reindexing.

**Chunks must stay under 512 tokens.** `bge-small-en-v1.5` has a 512-token max sequence length and **truncates silently** — no error. `chunk_tokens` defaults to 400 for headroom. Raising it past ~450 means stored chunks are only partially embedded, degrading retrieval in a way that is very hard to trace. (This is why chunk size is 400 rather than the 800–1200 an earlier draft of the plan specified.)

**bge query/passage asymmetry.** bge models want the prefix `"Represent this sentence for searching relevant passages: "` on **queries only**, never on stored passages. Keep this inside `embed_query()` vs `embed_documents()` so callers cannot get it wrong.

**SEC requires a descriptive `User-Agent`** (`EDGAR_USER_AGENT`, format `"Name email@example.com"`) or it blocks requests. Set it centrally in `app/edgar.py` and rate-limit to ~2–3 req/sec (SEC's ceiling is 10) — never per call site.

**Python must be 3.13**, pinned in `pyproject.toml`. The machine's default `python3` is 3.14, which lacks reliable torch/sentence-transformers wheels.

**Re-ingestion must stay idempotent.** `UNIQUE (accession_number, chunk_index)` + `ON CONFLICT ... DO UPDATE` is the mechanism; `scripts/ingest.py` additionally skips accession numbers already present in `filings`. The standing test is that running ingestion twice leaves row counts unchanged.

**`chunks.content_tsv` is a `GENERATED` column** — Postgres maintains it automatically. Never write to it, and note that hybrid search needs no extra ingestion work as a result.

**`sql/schema.sql` is applied on every startup**, so every statement in it must remain idempotent (`IF NOT EXISTS`, named indexes).

## Parsing is the fragile part

`app/parser.py` (A4) splits filing HTML into named sections and is the step most likely to need iteration — EDGAR HTML varies by filer and year. Two defenses are required, not optional:

1. **Degrade, don't crash** — if no sections are confidently detected, fall back to chunking the whole document with `section="unknown"` and log a warning.
2. **Guard against the table of contents** — "Item 1A." and "Item 7." also appear in the TOC. Match the *last* occurrence or require a minimum section length, or you will extract a one-line TOC entry as your risk factors.

Sections are form-aware: 10-K uses Item 1 / 1A / 7; 10-Q uses Part I Item 2 (MD&A) and Part II Item 1A. Chunking never crosses a section boundary — citation precision depends on it.

## Testing approach

Per-step manual verification is the **primary** gate, because most failure modes here are quality judgments (does this chunk actually read like a risk factor?) rather than assertions. Each step in the plan file has an explicit test to run before proceeding.

Unit tests cover only the deterministic pieces and are pinned to saved filing fixtures in `data/raw/` so they run offline: parser section boundaries, chunker metadata completeness, retriever filter behavior and RRF math.
