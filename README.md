# The First Product that intiated this Syetem - The Fork

A construction-intelligence platform: FastAPI backend + React frontend +
Postgres with pgvector + hybrid retrieval (BM25 + vector RRF) + an
agent runtime that handles project Q&A, BOQ extraction, drawing QTO,
WBS generation, and cost analysis.

**Live:** [the-fork-jn3t.onrender.com](https://the-fork-jn3t.onrender.com)

The deployed instance runs on Render (FastAPI web service + Postgres
16 with pgvector). Chat is provider-driven via `LLM_PROVIDER`: the
cloud ladder may be OpenRouter free models (`OPENROUTER_MODEL=openrouter/free`)
as primary, with Kimi and Groq optional — Ollama is the
on-prem provider only, not a cloud fallback. The canonical `chunks_v2`
store held ~10,500 indexed chunks across 53 docs at last count
(PILOT_READINESS.md, 2026-07-12); no "142k" figure is sourced anywhere
in the repo.

---

> ### Part of the CEREBRUM ecosystem — industrialized AI delivery
>
> **The Store — [Cerebrum-Blocks](https://github.com/bopoadz-del/Cerebrum-Blocks):** 94+ certified AI blocks, 17 industry kits, one universal API. Build a capability once; every sector inherits it.  
> **The Factory — [CerebrumDev.ai](https://github.com/bopoadz-del/CerebrumDev.ai):** the client-facing interface that assembles blocks into governed, deployable vertical platforms — evaluation gates in CI, release certification, honest closure reporting.  
> **The Products — [The Fork](https://github.com/bopoadz-del/The_Fork)** (construction AI — enterprise client pilot) **· [RetailOps](https://github.com/bopoadz-del/TEKsystems_GlobalRetailMNC)** (retail operations — assembled, CI-gated and deployed in under three days).  
> **The Edge:** sovereign deployment proven — zero-egress on-premise profile, executed air-gap acceptance test, signed sovereignty report.
>
> **You are here: A PRODUCT** — the flagship construction vertical, in enterprise client pilot. CerebrumDev's platform generator pins this repo as its reference baseline for new verticals.

---

## What it does

The Fork takes a construction project (RFP, BOD, drawings, BOQ, specs,
schedules, reports) and gives the operator a chat surface that:

- Answers document-grounded questions by retrieving from the project
  corpus and citing the source
- Extracts structured Bill of Quantities from PDF / XLSX / CSV
- Runs quantity takeoff on drawings (PDF + DXF)
- Generates CPM-validated work breakdown structures (200+ activities,
  full ES/EF/LS/LF, critical path)
- Reconciles drawing quantities against BOQ totals (variance math via
  sympy) and produces recommendations
- Answers engineering questions from a curated **construction knowledge
  base** (concrete, buildings, roads/earthworks/geotech, procurement) —
  every answer is a cited rule with provenance + a credibility tier, and
  formula rules can be evaluated against supplied values
- Cites real documents with confidence scores in every answer

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  React frontend (frontend/src/)                                   │
│  - 3-column workspace (left sidebar + chat + right sources)       │
│  - SSE chat streaming with theme + dark mode                      │
└───────────────────────────────┬──────────────────────────────────┘
                                │  /v1/agents/project-assistant/chat
┌───────────────────────────────▼──────────────────────────────────┐
│  FastAPI backend (app/)                                           │
│  - 41 blocks under app/blocks/ (boq_processor, drawing_qto, …)    │
│  - Agent runtime (app/agents/) with smart_orchestrator routing    │
│  - Hybrid retriever (BM25 + vector RRF) over the chunks table     │
│  - JWT auth + admin-gated /v1/admin/* endpoints                   │
└─────┬─────────────────────────┬──────────────────────────────────┘
      │                         │
      │  pgvector(384)          │  direct HTTPS
      │  + tsvector GIN         │  to LLM provider
      ▼                         ▼
┌──────────────┐         ┌──────────────────┐
│  Postgres 16 │         │  Kimi K2 primary │
│  + pgvector  │         │  Groq fallback   │
│  on Render   │         │                  │
└──────────────┘         └──────────────────┘
```

Storage:
- **Postgres** — users, projects, documents, conversations, messages,
  chunks (text + embedding + tsvector), agent_facts, hydration_runs.
  15 GB disk on Render basic_256mb plan, daily backups, 7-day
  retention. See [docs/backup-and-recovery.md](docs/backup-and-recovery.md).
- **Render persistent disk** (`/app/data`) — uploaded source documents,
  audit log, evidence vault, learning engine state, Google Drive OAuth
  refresh tokens, `.secret_key` fallback. Daily snapshots, 7-day
  retention.

Retrieval:
- Hybrid: 50 semantic + 50 BM25 candidates fused via Reciprocal Rank
  Fusion at the store layer, then re-ranked by cosine + identifier/GK
  signals and a hard project-scope filter. `app/core/rag/retriever.py`,
  `app/core/rag/vector_store.py`.
- Embeddings: shipped default `minishlab/potion-base-8M` (model2vec,
  256-dim, `app/core/rag/embeddings.py`), overridable via
  `RAG_EMBEDDING_MODEL` — prod currently overrides to
  `BAAI/bge-small-en-v1.5`, 384-dim. The store table width follows the
  embedder's actual dimension.
- RAG injection gate: only the `project-assistant` agent gets
  per-turn RAG context (`app/core/rag/inject.py`). Confidence
  threshold 0.4, daily token budget 500K, fallback prefix on miss.

---

## Construction knowledge base

A curated, general-purpose corpus of construction-engineering rules lives in
[`app/knowledge/construction_kb.json`](app/knowledge/construction_kb.json)
(human-readable mirror:
[`docs/knowledge/construction_kb.md`](docs/knowledge/construction_kb.md)).
It is deliberately **not** tied to any one project or region — entries cover
concrete, buildings, roads/earthworks/geotech, and procurement, each carrying
a credibility tier, provenance, and "verify against your spec" warnings for
region- or project-specific values.

- `app/blocks/_knowledge.py` loads/validates the corpus and exposes
  `evaluate(rule_id, **values)` (sympy) for formula rules and
  `search_knowledge(query, top_k, domain)` for token-overlap retrieval.
- The `construction_advisor` block turns a natural-language question into
  cited rule matches and evaluates the top formula when values are supplied.
- `smart_orchestrator` routes engineering queries (mass-concrete equilibrium
  time, compaction acceptance, dewatering, dynamic compaction, bitumen
  content, heavy-lift uplift FOS, diaphragm wall, …) to that block.

Adding rules: edit the JSON (keep entries general — `region_specific` /
`project_specific` flags and a null `provenance.project`), then run
`pytest tests/test_construction_kb.py` to validate every entry, formula, and
worked example before committing.

---

## Quickstart

### 1. Backend

```bash
git clone https://github.com/bopoadz-del/The_Fork.git
cd The_Fork

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in the required vars
                                 # SECRET_KEY at minimum

uvicorn app.main:app --reload
```

`requirements.txt` is enough for everything documented here, including
retrieval — the default embedder (`model2vec`) is a base dependency.
`requirements-rag.txt`, `-ml` and `-cv` are optional extras
(sentence-transformers, torch, OpenCV) and are **not** needed for the
Quickstart. They are large; only install them if you are switching
`RAG_EMBEDDING_MODEL` to a sentence-transformers model or running the
vision blocks.

No database setup is required to start. With `DATABASE_URL` unset the app
uses a local SQLite file under `DATA_DIR` (default `./data`). Postgres +
pgvector is the production configuration, not a local prerequisite.

Check it came up:

```bash
curl http://localhost:8000/health          # -> 200 with a status payload
open  http://localhost:8000/docs           # interactive API surface
```

### 2. Frontend

**The chat UI is a build artifact and is not committed.** The backend serves
it only once `frontend/dist` exists, so this step is required to see the UI —
without it you get the API and `/docs` but no chat surface.

```bash
cd frontend
npm ci
npm run build                    # produces frontend/dist
```

Restart uvicorn, then open `http://localhost:8000`.

For frontend development with hot reload, `npm run dev` serves on Vite's own
port (5173). There is no dev proxy — the app calls the backend directly at
`VITE_API_BASE`, which defaults to `http://localhost:8000`. Keep uvicorn
running alongside it, and set `VITE_API_BASE` only if your backend is
somewhere else.

### 3. Verify the install

These are the same gates CI runs, and they take seconds:

```bash
python scripts/audit_stubs.py           # no unregistered hollow functions
python scripts/scan_secrets.py          # no secret material in tracked files
python scripts/scan_exception_pass.py   # no except Exception: pass
python -m pytest tests/e2e/ -q          # the six demo flows, F1-F6
```

`tests/e2e/` is the honest description of what this platform does: boot and
health, the auth gate, a grounded citation that resolves to a real ingested
document, project isolation, refusal to invent a figure that is not in the
corpus, and a drawing going in with its schedules and title block coming out.

For production deploy, see [deploy/PILOT.md](deploy/PILOT.md) and
[docs/backup-and-recovery.md](docs/backup-and-recovery.md).

---

## API surface

The platform exposes:

- **Chat** — `POST /v1/agents/{agent}/chat` (and `/chat/stream` for SSE)
- **Projects** — `POST/GET/DELETE /v1/projects` and per-project
  `/documents`, `/conversations`, `/memory`, `/audit`
- **Document operations** — `/v1/projects/{id}/documents/search`,
  per-doc redline, exports
- **Drive integration** — `/v1/projects/{id}/drive/index-folder`,
  `/drive/import`
- **Admin (gated)** — `/v1/admin/debug/*`, `/v1/admin/corpus/collections`,
  `/v1/admin/corpus/bulk-insert`, `/v1/admin/training/*`
- **Observability** — `/metrics` (Prometheus, unauth, narrow
  request-counter set), `/v1/metrics` (admin, per-block execution
  data), `/health`

---

## Tests

```bash
python -m pytest tests/e2e/ -q   # the six demo flows — seconds
python -m pytest tests/ -q       # the full suite — ~35-40 minutes, 3100+ tests
python scripts/test_matrix.py    # BOTH CI profiles — what CI green actually means
```

Run the E2E flows first. They are fast and they fail loudly if the install is
wrong; the full suite is thorough but too slow to be a smoke test.

**Before pushing, run `scripts/test_matrix.py`, not just `pytest tests/`.** CI
runs the suite twice — once with `CEREBRUM_VIRGIN=true` (the generic block set)
and once with `CEREBRUM_VIRGIN=false` + `CEREBRUM_DOMAIN_KITS=construction` (the
full platform) — and some tests only run under one of them. A plain
`pytest tests/` covers one profile, so it can report a clean pass on a change
that CI rejects; that happened on 2026-08-11, where a green local run of 3302
tests missed an `ImportError` that only the production-like profile executes.

`test_matrix.py` reads the matrix and env straight out of
`.github/workflows/test.yml` rather than keeping its own copy, so it cannot
drift when the workflow changes. It exits non-zero if any profile fails.

```bash
python scripts/test_matrix.py --profile virgin   # one profile
python scripts/test_matrix.py -k chaining        # extra args go to pytest
```

CI gates, all blocking on every PR:

| gate | what it enforces |
|---|---|
| `tests` | full suite, two profiles (virgin / production-like) + a Postgres job |
| `diff-cover` | ≥50 % of newly-changed lines covered |
| coverage floor | 25 % overall, a regression floor |
| `audit_stubs` | no hollow function that is not registered in [docs/KNOWN_INCOMPLETE.md](docs/KNOWN_INCOMPLETE.md) |
| `scan_secrets` | no secret material in tracked files |
| `scan_exception_pass` | zero `except Exception: pass` handlers — S110 twin; empty allowlist |
| ruff S110 | zero silent `except: pass` handlers — baseline 0, never raise it |
| eslint | frontend Rules of Hooks and correctness errors |

[docs/KNOWN_INCOMPLETE.md](docs/KNOWN_INCOMPLETE.md) is the honest register of what is
not built. Nothing in this repository can be quietly hollow: a function is
either implemented or it is on that page with a reason.

---

## Key references

- [.env.example](.env.example) — every environment variable with
  REQUIRED-PROD / RECOMMENDED / OPTIONAL labels
- [docs/backup-and-recovery.md](docs/backup-and-recovery.md) — backup
  posture + restore procedure for Postgres + disk
- [deploy/PILOT.md](deploy/PILOT.md) — deployment runbook
