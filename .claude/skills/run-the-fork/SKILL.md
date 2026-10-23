---
name: run-the-fork
description: Use when asked to run, start, launch, serve, smoke-test, or screenshot The Fork / Cerebrum Blocks — the FastAPI construction-intelligence platform. Covers the smoke driver, the uvicorn dev server, and browser screenshotting of the chat UI.
---

# Run The Fork (Cerebrum Blocks)

The Fork is a **FastAPI** app: a block/container "lego" platform for
construction-document intelligence, with a single static HTML/JS chat UI
served at `/`. There is no frontend build step — the React dashboard the
README mentions was deleted; `app/static/index.html` is the only frontend.

The app is driven programmatically by **`.claude/skills/run-the-fork/driver.py`**
— it launches uvicorn, smoke-tests the core endpoints, and tears the server
down. All paths below are relative to the repo root.

## Prerequisites

- Python 3.11 (verified with 3.11.9). Node is **not** required.
- A virtualenv at `.venv` with dependencies installed:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# POSIX: .venv/bin/python -m pip install -r requirements.txt
```

## Run — agent path (use this)

One command launches the app, exercises it, and stops it:

```bash
.venv/Scripts/python.exe .claude/skills/run-the-fork/driver.py
```

Expected output (exit code 0 when all pass):

```
[driver] launching uvicorn on :8000 ...
  [PASS] GET /v1/health  (35 blocks loaded)
  [PASS] GET /  (landing page)  (HTTP 200, 93301 bytes)
  [PASS] GET /v1/blocks  (34 blocks)
  [PASS] POST /v1/execute  (translate -> 'hola')
[driver] 4/4 checks passed
```

The driver launches its own uvicorn on `PORT` (default 8000) — nothing else
may hold that port. Set `PORT` to change it.

### Screenshot the chat UI

To verify the frontend visually, launch the server (human path below) and
drive a browser with the Playwright MCP tools:

```
browser_navigate  -> http://127.0.0.1:8000/
browser_take_screenshot  (fullPage)
```

A correct render shows the dark "Cerebrum" chat UI: left sidebar (New Chat,
Connect Drive, Projects list loaded from `GET /v1/projects`), a "Connected to
API (36 blocks available)" banner, and an "Ask anything…" composer.

## Run — human path

```bash
ENV=development PYTHONIOENCODING=utf-8 DATA_DIR="$PWD/data" \
  .venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

or `./start-local.sh` (bash; sets the same env and runs uvicorn). Then open
`http://localhost:8000/` (chat UI) or `http://localhost:8000/docs` (Swagger).
Ctrl-C to stop. `/v1/*` endpoints need header `Authorization: Bearer $CEREBRUM_DEV_KEY`
(the dev key works **only** when `ENV=development`).

## Test

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/browser
```

Direct invocation without the server (for block/internal changes):
`.venv/Scripts/python.exe -c "import app.main; print('imports OK')"`.

## Gotchas

- **`PYTHONIOENCODING=utf-8` is mandatory on Windows.** Startup logs contain
  emoji; without it the app crashes with a `cp1252` `UnicodeEncodeError`. The
  driver and `start-local.sh` both set it.
- **`ENV=development` is required for the `$CEREBRUM_DEV_KEY` API key.** With `ENV`
  unset/production the dev key is rejected (`401 "Dev key disabled in
  production"`) and every `/v1/*` call fails.
- **README is stale on the frontend.** It lists a React dashboard at
  `/dashboard/`; that frontend was deleted. The only UI is the static page
  at `/`.
- The app loads **35 blocks** at startup; `GET /v1/blocks` lists **34**
  (the `construction` container is registered but not surfaced in that list).
- `formula_executor` / `project_reasoner` / `/v1/project/ask` need a funded
  `KIMI_API_KEY` or `GROQ_API_KEY` in `.env` — the driver deliberately smoke-tests
  `translate` instead (pure compute, no LLM, no historical-benchmark dep).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Driver: "server did not become healthy" | Port 8000 already held. Stop the other process or run with a different `PORT`. |
| `UnicodeEncodeError` / `cp1252` on startup | `PYTHONIOENCODING` not set to `utf-8`. |
| `401 Dev key disabled in production` | Launch with `ENV=development`. |
