# Data pipeline

Turns SEC filings into a searchable corpus. One direction, one file per stage, each testable alone.

```
tickers.py ──▶ edgar.py (+ sec_http.py) ──▶ parser.py ──▶ chunker.py ──▶ embeddings.py ──▶ store.py ──▶ Postgres
                                                                                                            │
                                                                                                       retriever.py
```

## Stages

| File | Job |
|---|---|
| `app/tickers.py` | Ticker → CIK, from SEC's `company_tickers.json` |
| `app/sec_http.py` | Shared HTTP transport for all SEC calls — required `User-Agent`, ~2-3 req/sec rate limit (SEC's ceiling is 10) |
| `app/edgar.py` | Filing lists and downloads via SEC's EDGAR API |
| `app/parser.py` | Filing HTML → named sections (`Item 1A Risk Factors`, `Item 7 MD&A`, ...) |
| `app/chunker.py` | Sections → ~512-token chunks, never crossing a section boundary |
| `app/embeddings.py` | Chunks → 768-dim vectors, via a local `google/embeddinggemma-300m` model |
| `app/store.py` | Vectors + metadata → Postgres, idempotently |
| `app/retriever.py` | Query → ranked passages, hybrid dense + BM25 |
| `scripts/ingest.py` | Runs the whole pipeline for one ticker |

## What's ingested

**10-K and 10-Q, last 2 fiscal years.** Current corpus: AAPL, META, MSFT, TSLA — 13 filings, 806 chunks. Exactly one 10-K per ticker (the rest are 10-Qs), so **no year-over-year filing comparison is answerable** for any covered company today.

Sections are form-aware:
- **10-K**: Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A)
- **10-Q**: Part I Item 2 (MD&A), Part II Item 1A (Risk Factors — often just "no material changes since our 10-K")

## Storage (`sql/schema.sql`)

Two tables:
- **`filings`** — one row per filing (accession number, ticker, form type, fiscal year, filing date, source URL). Doubles as the ingestion ledger: "have I already pulled this?" is a `SELECT`.
- **`chunks`** — one row per embeddable passage: text, a `vector(768)` embedding, a `GENERATED` `tsvector` column for full-text search, and section/ticker/form/year metadata for filtering.

Indexes: HNSW on the embedding column, GIN on the generated `tsvector`, plus btree indexes on `accession_number`, `section`, and `(ticker, form_type, fiscal_year)`.

**Applied on every startup** (`db.init_schema()`), and every statement is idempotent (`IF NOT EXISTS`) — safe to re-run.

## Retrieval

`retriever.hybrid_search(query, ticker=None, section=None, form_type=None, fiscal_year=None, k=6)` runs **one SQL query** that does both:
- **Dense**: cosine similarity over the embedding, via pgvector.
- **Sparse**: BM25-style ranking (`ts_rank_cd`) over the generated `tsvector`.

Fused with **Reciprocal Rank Fusion** (not weighted score blending — cosine distance and `ts_rank_cd` are on incomparable scales, but *ranks* aren't). This is why pgvector was chosen over Chroma: Chroma has no real BM25, so hybrid search there would mean a second index kept in sync by hand.

Filters (ticker, section, form type, fiscal year) narrow the SQL `WHERE`, not a post-filter — cheap, and precise.

## Two failure modes retrieval must distinguish

1. **Corpus gap** — the ticker was never ingested. Checked against the `filings` ledger *before* searching (`tools._covered_tickers()`), so it returns a distinct "not in our corpus" result rather than an empty one. Without this, an empty result looks identical to "the company genuinely never discussed this," and the agent silently falls back to news/web sources while presenting the answer as filing-grounded.
2. **Empty result from filters** — a real, narrow query (`ticker="AAPL", fiscal_year=1999`) that matches nothing. A nonsense *query* on an ingested ticker does **not** produce this — the dense half always returns its nearest neighbours, just irrelevant ones.

## Constraints worth knowing

- **Embedding dimension is coupled in three places**: the model, `vector(768)` in the schema, and `settings.embedding_dim`. Changing the model means changing all three plus a full re-embed.
- **Query and document embeddings use different prompt templates** (EmbeddingGemma's `query` vs `document` modes). Swapping them degrades retrieval silently — nothing errors, results just get worse. `embed_query()` / `embed_documents()` each pin their own template so no caller can pick wrong.
- **The embedding model is a process-wide singleton, locked twice** — once around construction (`@lru_cache` alone caches the *result*, not the *call*; concurrent first-callers built multiple models and crashed the interpreter), once around `encode()` itself (concurrent encoding on one loaded model is also unsafe on Apple Silicon). Both hazards are covered offline in `tests/test_embeddings.py`.
- **Chunk size (~512 tokens) is a quality choice, not a model limit.** A chunk spanning several risk factors averages into a vector that matches none of them well.
- **Re-ingestion is idempotent** — `UNIQUE (accession_number, chunk_index)` + `ON CONFLICT ... DO UPDATE`, and `scripts/ingest.py` skips accession numbers already in `filings`.

## Verifying it

```bash
.venv/bin/python scripts/check_db.py          # schema applies, twice, idempotently
.venv/bin/python scripts/check_tickers.py
.venv/bin/python scripts/check_edgar.py
.venv/bin/python scripts/check_parser.py      # risk factors are real prose, not a TOC line
.venv/bin/python scripts/check_chunker.py
.venv/bin/python scripts/check_retriever.py   # hybrid beats dense-only on exact-phrase queries
.venv/bin/python -m pytest tests/test_parser.py tests/test_chunker.py tests/test_retriever.py tests/test_embeddings.py -q
```
