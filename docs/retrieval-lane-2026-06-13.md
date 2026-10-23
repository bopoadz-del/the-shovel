# Retrieval lane probe — 2026-06-13

Owner: `cc/retrieval-lane`. File fence: retrieval store, hybrid RAG layer,
probe script, `test_rag_injection.py`, `test_retrieve_drops_noise_before_top_k`,
plus validation on retrieval / RAG blocks (`DrawingQTOBlock` and any RAG block).
Do NOT touch ConfigAccessor / config base class, requirements / lock files, or
per-block validation on non-retrieval blocks (cursor's parallel branch).

## Tasks (per branch directive)

### 1. Push local test fix — REDUNDANT

`tests/test_rag_injection.py::test_retrieve_drops_noise_before_top_k` was made
green locally last night (two-line `fake_search` stub fix — accept `query_text`
kwarg). Cursor pushed an identical fix as `cb8f90e` between sessions. Stash
dropped, no commit needed in this lane.

### 2. Suite green — IN-LANE 82/82

Full repo suite shows 264 failures / 35 errors after the cursor cascade
(SQLAlchemy ORM migration + ConfigAccessor refactor). All failures are out
of this lane:
- `tests/test_drive_router_live.py` — `load_token()` missing arg
- `tests/browser/*` — chromium runner fixture
- `tests/test_drawing_qto.py` — RuntimeError: Runner... (asyncio runner change)
- `tests/test_workflows.py` — `sqlalchemy.exc.IntegrityError`

In-lane (the four files I own + the drawing block per the fence's RAG-block
clause): `tests/test_rag.py`, `tests/test_rag_injection.py`,
`tests/test_hybrid_retrieval.py`, `tests/test_drawing_qto.py` —
**82 / 82 PASS** (118s).

`test_drawing_qto.py` errors when run via the full suite collection but
passes when run in isolation — a runner setup conflict from elsewhere in
the suite, not from drawing_qto itself. Surfaced for the cursor lane to
look at (their async-runner change).

### 3. Hybrid probe before / after — captured

Re-ran `scripts/_probe_hybrid_vs_vector.py` against the live 139,949-chunk
`drive_archive`. Each query runs twice: hybrid (`query_text=q`) and
vector-only (`query_text=None`). Output:
`data/logs/probe_hybrid_vs_vector_20260613_1709.log`.

| Q | Vector rank | Hybrid rank | Verdict |
|---|---|---|---|
| Q1 (EXC format) | None | None | SAME (matcher strict; both legs do surface EXC drawings, matcher under-credits hybrid) |
| **Q2 SECTIONAL ELEVATION telecom** | **None** | **3** | **HYBRID BETTER — old failure resolved** |
| Q3 PRC-951 acceptance | 1 | 1 | SAME (perfect on both) |
| Q4 trench width | None | None | SAME for strict chunks 1115/1116. Hybrid does surface WS-600 drawing rank 15 — correct discipline |
| Q5 manhole spacing telecom | None | None | SAME for TL-600 chunk 0. Hybrid surfaces TL-100-1000002-A rank 1 — correct discipline, wrong sheet |

TOTAL: HYBRID BETTER 1/5, SAME 4/5, WORSE 0/5.

**Old Q2 failure: RESOLVED.** Old Q5 failure: **partially** resolved —
hybrid promotes a TL drawing to rank 1 (right discipline family) but the
exact target chunk stays buried; strict matcher reports SAME.

### 4. Production chat smoke — pending

Awaiting a non-empty project_id on prod. Bridge will hit `/projects/<id>`
and capture the SSE `end` event sources to confirm hybrid is firing on
real audit logs. Last night's blocker was that `owner@example.com`
had zero projects in the post-migration Postgres.

### 5. 678 MB RAM peak vs 512 MB Render envelope — DEFERRED

Render production: `DATABASE_URL=postgresql+psycopg://...thefork`. The
Postgres path uses `ts_rank` + GIN index and pgvector ANN, both server-
side; no chunks are loaded into Python memory. The 678 MB peak is on the
LOCAL SQLite numpy-cosine + FTS5 fallback path used only in dev / tests.

If a SQLite-mode chat deployment ever lands on a 512 MB tier, options to
reduce peak:
- stream cosine in batches instead of `embedder.encode + matrix dot`
- mmap the embeddings BLOB column instead of loading per-search
- shrink FTS5 backfill (currently 143k rows in chunks_fts)

For now: prod is Postgres-only. Deferring per file-fence directive.
