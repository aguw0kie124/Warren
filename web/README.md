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
| `POST /query` | one turn; carries `thread_id` so follow-ups keep context |
| `GET /threads/{id}` | restoring a session picked from History |

Three things the UI is careful about, because the service is careful about them:

- **`route` is rendered, not ignored.** A `research` answer with no citations
  found nothing; `simple` / `advisory` / `clarify` have none by design. The API
  reports the route precisely so those can look different, and they do.
- **Citations are the whole thread's, not the turn's.** Each answer shows only
  the sources it added (`App.tsx` diffs against what has already been shown).
- **`type` is on every source chip** — an audited SEC filing is not a news
  article is not a web page.

History is `localStorage` only: the service has no thread index, just
`GET /threads/{id}`, so the client is the only thing that knows which ids are
this user's.
