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

    -- 768 dims = google/embeddinggemma-300m. If you swap embedding models, this
    -- dimension must change with it (see app/config.py::embedding_dim).
    embedding        vector(768),

    -- GENERATED = Postgres maintains the full-text vector automatically on every
    -- insert/update. A8's hybrid search therefore needs zero extra ingestion
    -- work, and the BM25 side can never drift out of sync with content.
    content_tsv      tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- Makes re-ingestion idempotent: ON CONFLICT targets this pair.
    UNIQUE (accession_number, chunk_index)
);

-- Dense retrieval (A7). Cosine ops to match the normalized EmbeddingGemma vectors.
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

-- ---------------------------------------------------------------------------
-- F1 · XBRL company facts — the numeric spine.
--
-- Every filing is inline XBRL: each figure in the statements is wrapped in a
-- tag naming its us-gaap concept, unit and period. app/parser.py strips those
-- (get_text keeps "416,161" and discards the tag), which is why numbers in the
-- chunk corpus are prose. This table holds the same audited figures as data,
-- fetched from SEC's companyfacts API, with the accession they were reported
-- in so a number cites the exact filing.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS company_facts (
    id               bigserial PRIMARY KEY,
    cik              text NOT NULL,
    ticker           text NOT NULL,
    concept          text NOT NULL,      -- us-gaap tag, e.g. 'NetIncomeLoss'
    unit             text NOT NULL,      -- 'USD' | 'USD/shares' | 'shares'

    -- NULL for instant facts (balance-sheet items are a point in time).
    period_start     date,
    period_end       date NOT NULL,
    period_type      text NOT NULL,      -- 'annual' | 'quarterly' | 'instant'

    -- Calendar year of period_end, and named for exactly what it is. It
    -- coincides with the filer's own fiscal-year label for most companies but
    -- is not guaranteed to, and calling it fiscal_year would invite a
    -- confidently wrong number. Tools label periods by their end date instead.
    calendar_year    int  NOT NULL,

    value            numeric NOT NULL,
    form             text NOT NULL,      -- '10-K' | '10-Q'
    accession_number text NOT NULL,      -- becomes the citation
    filed_date       date NOT NULL,

    -- NULLS NOT DISTINCT needs Postgres 15+; we run pg16. Under plain UNIQUE,
    -- Postgres treats NULLs as distinct, so every instant fact would insert
    -- again on every backfill — silently, with no error and no duplicate key.
    UNIQUE NULLS NOT DISTINCT (cik, concept, unit, period_start, period_end)
);

-- The access pattern: one ticker, one concept, a period series, newest first.
CREATE INDEX IF NOT EXISTS company_facts_lookup_idx
    ON company_facts (ticker, concept, period_type, period_end DESC);

-- "Which companies have fundamentals?" — the numeric half of the coverage
-- question, answered the way tools._covered_tickers() answers the text half.
CREATE INDEX IF NOT EXISTS company_facts_ticker_idx ON company_facts (ticker);
