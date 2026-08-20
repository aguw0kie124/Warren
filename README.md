# Warren
AI powered financial research platform so I can stop gambling on robinhood...

Warren (you know who this is named after) allows you to ask a question about a publically traded company and get an answer grounded in real SEC EDGAR filings, live market data, and web search, with citations that resolve back to the exact filing and section.

## Why I built this

To learn how real agentic systems work: LangGraph for graph-based control flow, RAG over real financial data, and evals/observability.

## Stack

- **Agent orchestration**: LangGraph, LangChain tool calling, Claude (Haiku 4.6 for now)
- **Retrieval**: Postgres, pgvector, Hybrid search with BM25 and RRF
- **Data**: SEC EDGAR filings, Finnhub market data, Tavily web search
- **Embeddings**: sentence-transformers with `google/embeddinggemma-300m`
- **API**: FastAPI, Uvicorn, Pydantic

## Architecture

### The agent: a router in front of a loop

```
START -> router --- respond -> END          (simple | advisory | clarify)
              \\--- agent <-> tools -> END    (research)
```

The router is a single classifier call that sorts every question into one of four routes before any real work happens:

- **research**: needs filings, market data, news, or the web. The default.
- **simple**: answerable from general company knowledge alone (who runs it, what it does, where it's listed).
- **advisory**: asks for a pick, ranking, or buy/sell/hold call.
- **clarify**: can't be acted on as written.

Three of those routes go straight to a `respond` node (one model call, no tools, done). Only `research` enters the loop.

**The loop is the core of this project.** It's a plain ReAct cycle: `agent -> tools -> agent -> ...` until the model stops calling tools, and the last tool-free turn from the model is the answer. There is no separate synthesis step. Tools are plain entries in a list, not graph nodes, so adding a new capability (a new data source, a new API) means writing one function, never touching the graph.

State carried through the loop is deliberately small: the message history, the citations gathered so far, and the router's chosen route.

Every graph run, model call, and tool call is traced through **LangSmith**, which is how the loop, the router's decisions, and tool usage actually get observed and debugged instead of guessed at.

### The RAG pipeline

```
EDGAR API -> tickers -> edgar -> parser -> chunker -> embeddings -> store -> Postgres
                                                                          |
                                                                     retriever
```

- Filings are pulled from SEC EDGAR and split into named sections (Risk Factors, MD&A, etc).
- Sections are chunked and embedded, then stored in Postgres with pgvector.
- Retrieval is hybrid: dense vector similarity plus BM25 full text search, fused with Reciprocal Rank Fusion in one SQL query. Embeddings alone miss exact-match legal/financial language; BM25 alone misses semantic matches.
- Live market data (quotes, financials) and news/web search come from separate tools alongside the retriever, feeding the same agent loop.
- Retrieval quality itself is measured with evals (recall@k, MRR) rather than assumed to be good.

## Setup

Requires Python 3.13 and Docker. SEC ingestion also requires a real `EDGAR_USER_AGENT`, and the default local embedding model requires accepting the Hugging Face license for `google/embeddinggemma-300m` once before the first ingest.

```bash
# 1. Start Postgres (pgvector)
docker compose up -d --wait

# 2. Create a venv and install dependencies
/usr/local/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# 3. Copy env file and fill in your keys (Anthropic, Finnhub, Tavily, LangSmith)
cp .env.example .env

# 4. Run the service
.venv/bin/python -m uvicorn app.api:app --reload

# 5. Ingest a filing for a ticker (filings aren't ingested automatically)
.venv/bin/python scripts/ingest.py --ticker AAPL

# 6. Run tests (offline, no DB or API keys needed)
.venv/bin/python -m pytest
```

Query the API with:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are Apple'\''s key risk factors this year?"}'
```

## Verification

The unit tests are offline. The heavier checks live in `scripts/` and exercise the real integrations:

```bash
.venv/bin/python scripts/check_data.py
.venv/bin/python scripts/check_tools.py
.venv/bin/python scripts/check_api.py
.venv/bin/python scripts/check_retrieval_quality.py --local
```

## Still missing

- A frontend or CLI for asking questions outside raw HTTP.
- Automatic/background ingestion when a ticker has not been indexed yet.
- Broader corpus coverage beyond the filings you ingest locally.
- Auth, rate limiting, and deployment hardening.
- A stronger end-to-end answer quality eval, beyond retrieval metrics.
- Clear product disclaimers and guardrails for "not financial advice" use.

