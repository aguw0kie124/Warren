# Warren — web

React frontend for the research service. Chat in, sourced answer out.

```bash
npm install
npm run dev      # http://localhost:5173
```

The API is expected on `http://localhost:8000` (start it with
`.venv/bin/python -m uvicorn app.api:app --reload` from the repo root). Vite
proxies `/api/*` to it, so the backend needs no CORS middleware — change the
target in `vite.config.ts`, not in the app.

## What it talks to

| Call | Used for |
|---|---|
| `POST /query/stream` | every turn — SSE, so the run reports itself as it happens |
| `GET /threads/{id}` | restoring a session picked from History |

`POST /query` is the same run without the protocol. Nothing here calls it, but
`lib/api.ts` keeps `query()` because the blocking form is the easier one to
reach for from a script or a curl.

Four things the UI is careful about, because the service is careful about them:

- **`route` is rendered, not ignored.** A `research` answer with no citations
  found nothing; `simple` / `advisory` / `clarify` have none by design. The API
  reports the route precisely so those can look different, and they do.
- **Citations are the whole thread's, not the turn's.** Each answer shows only
  the sources it added (`App.tsx` diffs against what has already been shown).
- **`type` is on every source chip** — an audited SEC filing is not a news
  article is not a web page.
- **`reset` is honoured.** Tokens stream from a model turn before anyone knows
  whether it was the answer or a preamble to a tool call; when it was a
  preamble the server withdraws it. Ignoring that event glues *"Let me check
  the filings."* onto the front of every answer — wrong, and plausible enough
  that nobody files a bug about it.

Sources render **above** the answer, which is deliberate. During a run they
also exist first — the agent has read its filings well before it has written a
sentence — so the evidence is on screen while the answer is still arriving,
rather than being a footnote to a claim already made. The list collapses past
four; nothing is dropped, it is one click away.

History is `localStorage` only: the service has no thread index, just
`GET /threads/{id}`, so the client is the only thing that knows which ids are
this user's.
