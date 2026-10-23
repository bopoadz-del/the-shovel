# AGENTS.md

## STANDING ORDER -- read before anything else (owner, 2026-10-02)

> NO HARDWIRING. A failing case is never fixed by name. No per-case branches,
> no literal document strings, no test-case IDs, no rescue functions, no
> per-case switches. Find the general cause and fix the mechanism so every
> case of that kind passes. If you cannot find a general fix, say so and stop
> -- a pass patch is a failure, not progress.

> A fix is accepted only if it raises the score on questions and corpora the
> builder has never seen. CI rejects any diff that adds `_rescue_*`,
> `*_NEEDLES`, `*_RESCUE` knobs, or a probe ID inside product code.

This REPLACES "loop until every test passes" and "after every fix, record the
case that would have caught it". Passing SEEN tests is not the target; working
on UNSEEN ones is. The CI tripwire is `scripts/scan_hardwiring.py` (blocking,
`lint.yml`); never add to `scripts/hardwiring_baseline.json` -- it may only
shrink. The acceptance is the owner-sealed unseen set: do not ask for it, do
not write it. The full order, the four tests, and the retriever surgery:
`docs/ORDER-no-hardwiring.md`.

## Cursor Cloud specific instructions

The Fork is a **FastAPI** backend (`app/`) plus a **React + Vite + TypeScript**
frontend (`frontend/`). Cloud Agent bootstrap is repo-managed in
`.cursor/environment.json` (Python 3.11 venv + `frontend` `npm ci`, uvicorn on
`:8000`). The notes below cover only non-obvious run/test caveats. For full
detail see `README.md`, `.env.example`, and `.claude/skills/run-the-fork/SKILL.md`.

### Runtime prerequisites (baked into the environment, not the update script)
- **Python 3.11** is required (`.python-version` / `runtime.txt`; `requirements.txt`
  is pinned for 3.11). The venv is built with `python3.11`, not the system 3.12.
  Cursor's default Cloud Agent image is Ubuntu 24.04 (only 3.12). `.cursor/Dockerfile`
  installs 3.11; the install script also bootstraps it via deadsnakes if missing.
- System packages the app needs at runtime: `tesseract-ocr` (+ `tesseract-ocr-ara`),
  `antiword`, `unrar`, `libgl1`, `libglib2.0-0`, `libmagic1` (see `Aptfile` /
  `render-build.sh`). These live in the base image, not the update script.

### Running the backend
- Dev launch: `./start-local.sh` (binds `0.0.0.0:8000`, sets `ENV=development`,
  `DATA_DIR=$PWD/data`, `PYTHONIOENCODING=utf-8`). It uses the system `uvicorn`,
  so activate `.venv` first or run uvicorn from the venv directly:
  `CEREBRUM_VIRGIN=false CEREBRUM_DOMAIN_KITS=construction ENV=development \
  DATA_DIR="$PWD/data" .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
- `ENV=development` is mandatory: it skips the production startup guard and
  enables the dev API key. All `/v1/*` calls need `Authorization: Bearer $CEREBRUM_DEV_KEY`
  (that key is rejected unless `ENV=development`).
- No database is required locally: with `DATABASE_URL` unset the app uses SQLite
  at `$DATA_DIR/the_fork.db`. Postgres 16 + pgvector is the production config only.
- The production block profile is `CEREBRUM_VIRGIN=false` +
  `CEREBRUM_DOMAIN_KITS=construction` (loads 48 blocks). Unset/`true` loads a
  smaller generic set; some tests only run under the full profile.
- First boot downloads the `model2vec` embedder (`minishlab/potion-base-8M`) from
  Hugging Face into `HF_HOME`; needs network once. For tests set
  `RAG_EMBEDDING_MODEL=fake` to skip the download.
- Registration auto-verifies in dev (no `RESEND_API_KEY`), so you can
  `POST /v1/users/register` then `/v1/users/login` and use the app without email.
- LLM chat (`/v1/chat/stream`, `/v1/agents/*/chat`) needs a funded `KIMI_API_KEY`
  or `GROQ_API_KEY`; everything else (auth, projects, uploads, block `/v1/execute`)
  works without any provider key.

### TIER-1 Drive ingest (`scripts/p1b_ingest_drive_server.py`)
- The run can no longer end without a final tally. Every observable exit —
 clean finish, `SIGTERM`/`SIGINT`/`SIGHUP`, uncaught exception, `SystemExit`,
 `atexit` — flushes the report once through `app/core/ingest_lifecycle`
 (`RunLifecycle`), and both the report JSON and the tally line carry
 `exit_reason=` + `complete=`. An unfinished run also prints
 `RUN INCOMPLETE` and exits `128+signal` (143 on SIGTERM). Nothing marks a
 partial run complete; the 4609 tier-1 gate still means what it meant.
- `SIGKILL` (cgroup OOM killer) cannot be handled, so a sidecar heartbeat is
 written after **every** file to `manifests/p1b_ingest_run_state.json`
 (gitignored, override with `--run-state` / `P1B_RUN_STATE`) carrying RSS,
 `memory.current`/`memory.max`, the cgroup `oom_kill` counter and
 `/proc/uptime`. The next start prints a `POSTMORTEM` line naming where the
 dead run stopped, and whether the box was replaced (uptime went backwards)
 or the kernel OOM-killed it (`oom_kill` counter went up).
- For a detached run prefer `--supervise` (or `nohup`/`setsid`): only a parent
 sees a `SIGKILL` wait status. The start banner logs `session_leader=` —
 `False` means closing the Render Shell will SIGHUP the run.
- `kill -USR1 <pid>` dumps every thread's stack without killing the run, and a
 run that stops making progress for `P1B_STALL_WARN_S` (default 900) dumps
 them by itself. Use this before assuming a stale log means a dead process:
 the forked extraction child in `app/core/extract_isolated` can wedge for
 `DOC_EXTRACT_TIMEOUT_S` per file.
- `StopFlag` / `install_signal_handlers` live in `app/core/ingest_lifecycle`
 and are re-exported by `scripts/rag_render_bulk_ingest.py`. Do not fork a
 second stop protocol; `tests/test_ingest_lifecycle.py` guards the drift.

### Frontend
- The chat UI is a **build artifact** — `frontend/dist` is gitignored and NOT
  committed, so the update script does not build it. uvicorn only serves the real
  UI at `/` once `frontend/dist` exists. Build it with
  `npm --prefix frontend run build`, then (re)start uvicorn. Without a build, `/`
  returns a tiny placeholder page while the API and `/docs` still work.
- For hot-reload dev use `npm --prefix frontend run dev` (Vite on `:5173`). There
  is no dev proxy; it calls the backend directly at `VITE_API_BASE`
  (default `http://localhost:8000`), so keep uvicorn running alongside it.

### Lint / test
- Backend fast gate: `.venv/bin/python -m pytest tests/e2e/ -q` (seconds). Full
  suite (`pytest tests/`) is ~35-40 min. Before pushing run
  `.venv/bin/python scripts/test_matrix.py` — CI runs the suite under BOTH the
  virgin and construction profiles and a plain `pytest tests/` covers only one.
- Frontend lint: `npm --prefix frontend run lint` (eslint; errors block CI,
  warnings are allowed). Context files (`AuthContext`, `ThemeContext`) keep a
  file-level `react-refresh/only-export-components` disable because the hook
  lives next to the provider.
- **GRANDFATHERED per-case behaviour -- surgery targets, NOT a pattern to copy.**
  The items below (Leftover L1 / L6, Live UI pack E4 / F1, the `clash` keyword
  switch, the named-calculator phrase override) describe switches and pins
  that exist in code TODAY. They are documented so you do not break a live
  kill-switch by accident. Under the STANDING ORDER at the top of this file,
  every one of them is a candidate for the general fix that removes it. Never
  add a new entry of this kind, and never cite one as justification for
  another.
- Leftover L1 named-file fetch: `_predispatch_file_tool` runs `fetch_document`
  for `.txt`/`.md`/`.docx`/`.doc` when the user names the file (full name or a
  ≥12-char stem, so `fenwick_waterproofing_spec` matches a timestamped upload).
  Empty RAG chunks fall through to disk `extract_document_text`. Do **not**
  re-ingest/reindex that docx; another agent owns RAG. Predispatch runs
  **before** the identifier RAG-miss short-circuit — a timestamped `.docx`
  looks like a drawing code and used to return "could not confirm this
  reference" in ~1s without fetching bytes.
- Clash detection stays **off** unless the user message contains `clash`.
  Predispatch then sets `run_clash_detection: True`. Default-on was hung on
  leftover L2 geom — do not flip the default.
- Named-calculator override (`_message_wants_named_calculator`) keeps
  "interim payment" + figures on `construction_calc`. A no-figure
  "issue/generate … payment certificate" is **not** stolen; it stays a
  `payment_certificate` deliverable (honest missing-`contract_value` is OK).
- Leftover L6 (trench bank volume): the smart-orchestrator hat calls
  `construction` with `action=construction_calc`. That action is aliased to
  the formula registry (`excavation_volume` for L×W×D). Extra LLM kwargs
  such as `text` are stripped. Do not send the model to `drawing_qto` when
  the user already supplied the dimensions.
- Live UI pack E4 (raft 30×20×1.5 including documented waste): a
  concrete/raft ask pins `concrete_volume` and the documented 5% waste
  (900 × 1.05 = 945 m³). Unnamed earthwork L×W×D stays `excavation_volume`.
  Kill-switch `APPLY_DOCUMENTED_WASTE=0` restores the FAIL (net 900).
- Live UI pack F1 (high-level WBS): a WBS / work-breakdown ask must
  render `wbs_tree` as a numbered hierarchy (phase headings + package
  codes), not only "Schedule built: N activities" + a workbook offer.
  `deliver_schedule` reads the tree already staged by `generate_wbs`.
  When the ask names demolition / site clearance / BOQ and measured
  CESMM D-rows are in the project document pool, elect those rows —
  do not print the building-template scaffold. Keep the template
  declaration only when no measured rows were retrieved. Do not invent
  BOQ items.
- Currency rate units (`AED/m2`, `USD/ft2`) are not document identifiers.
  Pinned self-coding conversions must not hit the RAG-miss short-circuit.
- Contract Data Q&A (Time for Completion, milestones, delay damages, DNP,
  performance bond, Engineer, Aconex) stays on project-assistant + RAG.
  Do not dispatch `generate_wbs` / predefined schedule.
- Python lint gates: `scripts/audit_stubs.py`, `scripts/scan_secrets.py`, and
  `scripts/scan_exception_pass.py` (stdlib only). `scan_secrets.py` is
  fail-closed on `SECRET_SCAN_PATTERNS` (env-fed client-pattern denylist).
  If that env is unset/empty the script exits non-zero and never reports
  clean — a missing denylist is a red gate, not a skip. Do not hardcode the
  patterns; they are a repository secret. The ruff S110 gate uses ruff,
  which is **not** in `requirements.txt` — CI installs `ruff==0.16.1` on
  demand; do the same locally if you need it (`pip install ruff==0.16.1`).
  S110 is bare `except: pass` only; `scan_exception_pass.py` is the
  `except Exception: pass` twin.

### Production Render (non-obvious)
- Live service is `the-fork` (`srv-d9s6l67avr4c73aiujsg`) at `https://theshovel.ai`. Auto-deploy is **After CI Checks Pass** (`autoDeployTrigger: checksPass` in `render.yaml`). A red required GitHub check skips the auto-deploy. This blueprint value does not change the running service until Dashboard Auto-Deploy is flipped (prefer that one toggle; Apply Blueprint is still dangerous — see `render.yaml` header). A Render API deploy **without** `commitId` ships whatever is currently on `main`, not the feature-branch SHA. Pass the pushed commit explicitly.
- `/v1/agents/*/chat` and `/v1/chat/stream` conversation ids must be `ws-{projectId}-{unix_ms}` only. Extra path segments 404.
- Kimi HTTP 400 on orphaned `tool_call_id`s ("must be followed by tool messages") or `tokenization failed` is retryable: skip any remaining Moonshot hop and take `LLM_FALLBACK_PROVIDER` (Groq). Generic 400/401/403/404 still do not fall back. User nudges wait until every tool in the assistant batch has a result so the pairing 400 is not generated in the first place.

### Stale docs to ignore
- `README.md` and `.claude/skills/run-the-fork/SKILL.md` claim the React frontend
  was deleted and only `app/static/index.html` is served. That is outdated: the
  Vite React app in `frontend/` is the real UI (served from `frontend/dist`).

### Before creating a module

Grep this repo's `main` **and** sibling repos (CerebrumDev.ai, Cerebrum-Blocks)
before adding a new Python module. Consume the existing module, or dual-register
it with a drift test. Never re-implement a module that already exists under
another name.

### Stage evidence never ships inside feature PRs

Do not commit Factory `build/stages/*.json` (or reread twins) in a feature PR.
Stage evidence is regenerable; emit it in a later single run at final HEAD after
the code merge. Canonical Factory S4 evidence is `S4_ship_kernel.json`
(`S4_kernel.json` is not a reader input).
