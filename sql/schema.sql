-- Schema for the SEC filings RAG corpus.
-- Applied idempotently by app/db.py::init_schema() on startup, so every
-- statement here must be safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per SEC filing. Per-filing metadata lives here rather than being
-- repeated on every chunk, and doubles as the ingestion ledger: "have I already
-- pulled this accession number?" is a plain SELECT.
CREATE TABLE IF NOT EXISTS filings (
    accession_number text PRIMARY KEY,
    cik              text        NOT NULL,
    ticker           text        NOT NULL,
    company_name     text        NOT NULL,
    form_type        text        NOT NULL,          -- '10-K' | '10-Q'
    fiscal_year      int,
    filing_date      date        NOT NULL,
    source_url       text        NOT NULL,          -- becomes the citation link
    ingested_at      timestamptz NOT NULL DEFAULT now()
);

-- One row per embeddable chunk of filing text.
CREATE TABLE IF NOT EXISTS chunks (
    id               bigserial PRIMARY KEY,
    accession_number text NOT NULL
                     REFERENCES filings(accession_number) ON DELETE CASCADE,
    chunk_index      int  NOT NULL,
    section          text NOT NULL,                 -- 'Item 1A Risk Factors'
    content          text NOT NULL,

    -- 384 dims = BAAI/bge-small-en-v1.5. If you swap embedding models, this
    -- dimension must change with it (see app/config.py::embedding_dim).
    embedding        vector(384),

    -- GENERATED = Postgres maintains the full-text vector automatically on every
    -- insert/update. A8's hybrid search therefore needs zero extra ingestion
    -- work, and the BM25 side can never drift out of sync with content.
    content_tsv      tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- Makes re-ingestion idempotent: ON CONFLICT targets this pair.
    UNIQUE (accession_number, chunk_index)
);

-- Dense retrieval (A7). Cosine ops to match the normalized bge embeddings.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Sparse/BM25 retrieval (A8).
CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin_idx
    ON chunks USING gin (content_tsv);

-- Metadata filters (ticker/section/form) applied alongside both searches.
CREATE INDEX IF NOT EXISTS chunks_accession_idx ON chunks (accession_number);
CREATE INDEX IF NOT EXISTS chunks_section_idx   ON chunks (section);
CREATE INDEX IF NOT EXISTS filings_ticker_form_year_idx
    ON filings (ticker, form_type, fiscal_year);
