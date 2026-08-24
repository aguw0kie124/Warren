# Warren

### Financial research you can actually trace.

Warren turns questions about public companies into research-backed answers grounded in SEC filings, live market data, and the web, with citations that lead back to the source.

> Stop gambling on a brokerage app's summary box. Ask a better question, see the evidence, and make up your own mind.

## Highlights

- **Ask questions in plain English** about a public company.
- **Search the filings, not just the internet**, including named sections such as Risk Factors and MD&A.
- **Combine durable and live context** from SEC EDGAR, market data, news, and web search.
- **Get citations with every research answer**, so claims can be checked against the underlying source.
- **Keep the conversation going** with persisted threads and streamed progress.
- **Measure retrieval quality** with offline evals instead of assuming the right context was found.

## What it feels like

```text
You: What are Apple's biggest risks this year?

Warren: [researches filings, market data, and the web]
        [returns an answer with source links and filing sections]
```

The project includes a FastAPI service and a small React interface. The API is also usable directly from `curl` or another client.

## How it works

Warren routes each question before doing expensive work:

```text
question -> router -> answer directly
                  +-> research loop -> tools -> cited answer
```

Research questions enter a ReAct-style loop managed by LangGraph. The agent can search filings, retrieve relevant passages, inspect market data, and search the web. Filings are parsed into sections, chunked, embedded, and stored in Postgres with pgvector. The retrieval-augmented generation pipeline uses hybrid search, combining semantic vector search with BM25 keyword search through reciprocal rank fusion.

The system is built to keep evidence visible: citations, route decisions, model calls, and tool calls are all observable through LangSmith when tracing is enabled.

## Quickstart

Requirements: Python 3.13, Docker, and API keys for Anthropic and Tavily.

```bash
# Start Postgres + pgvector
docker compose up -d --wait

# Install Warren
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# Configure credentials
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, TAVILY_API_KEY, and EDGAR_USER_AGENT

# Start the API
.venv/bin/python -m uvicorn app.api:app --reload
```

In another terminal, index a company before asking about its filings:

```bash
.venv/bin/python scripts/ingest.py --ticker AAPL
```

Then query it:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are Apple'"'"'s key risk factors this year?"}'
```

To run the web interface, see [`web/README.md`](web/README.md).

## Core technology

The core backend is a FastAPI service built around LangGraph for routing and agent orchestration. Its hybrid-search RAG pipeline uses Postgres, pgvector, BM25 full-text search, reciprocal rank fusion, and local sentence-transformer embeddings. The agent uses Claude and tool calls to work across SEC EDGAR filings, Yahoo Finance market data, and Tavily web search. A React frontend provides the chat interface.

## Project status

Warren is an active research project and learning vehicle for agentic systems, retrieval, and evaluation. It is not investment advice. The current focus is improving answer quality, expanding corpus coverage, and hardening the path from local prototype to a dependable product.

Run the offline test suite with:

```bash
.venv/bin/python -m pytest
```

## Why I built it

To understand what makes a research agent useful in practice: reliable retrieval over real documents, tool-using workflows, citations, observability, and evaluations that reveal when the system is wrong.
