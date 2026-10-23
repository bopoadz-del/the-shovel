> **SUPERSEDED (2026-09-28).** Historical snapshot. Not a source of open items or current status. Current state lives in the fleet standing-order loop and the live measurements.

# Repository audit — 2026-08-03

Evidence-based sweep of the whole repo. Every claim below has a command or a
live probe behind it. Ordered by what would hurt a client deployment first.

Scope: 69,697 LOC application Python · 49,128 LOC tests (291 files) ·
5,912 LOC frontend · 85 scripts · 141 route decorators.

---

## 1. CRITICAL — the gate that let an outage ship (FIXED tonight)

`eslint` existed in the repo and ran in **no workflow**. A conditional
`useMemo` therefore shipped and crashed every project page into the
ErrorBoundary (React #310). `tsc -b && vite build` compiles a conditional
hook happily — Rules of Hooks is a runtime contract, not a type error.

Now a **blocking** CI job (PR #308). Verified against a minimal repro before
enabling. **Lesson generalises: a linter that is installed but unwired is
worth nothing, and its absence is invisible until it costs you.**

---

## 2. HIGH — silent error swallowing at scale

```
478  except Exception          (only 49 carry a # noqa rationale)
 72  except Exception: pass    (swallow entirely, no log)
```

Worst offenders: `containers/construction/documents.py` (5),
`routers/upload.py` (4), `core/rag/vector_store.py` (4), `core/doc_index.py`
(4), `agents/runtime.py` (4).

**This is not theoretical — it cost real time tonight, twice:**

- `_persist_failed_turn` referenced `agent_memory` at module scope. That
  raised `NameError`, which the helper's own `except Exception` swallowed,
  leaving a function that silently persisted nothing. Only a test caught it.
- The search `origin` field was added to an intermediate dict, never reached
  the wire, and shipped green. Only a live probe caught it.

**Recommendation:** a bare `except Exception: pass` should be a lint error.
Each of the 72 either needs a log line or a comment naming what it is
absorbing and why. This is the single highest-yield cleanup in the repo — it
is the class that hides every other class.

---

## 3. HIGH — CI depends on an unauthenticated third-party download

`test (virgin)` failed on PR #308 with **nine** errors, none of them code:

```
429 Too Many Requests for
https://huggingface.co/api/models/minishlab/potion-base-8M
3032 passed, 116 skipped, 2 xfailed, 9 errors
```

`HF_TOKEN` is set **nowhere** — not CI, not `render.yaml`, not the Dockerfile
(`grep -rn "HF_TOKEN|HUGGINGFACE"` returns empty). The same warning appears
in the production boot log.

So CI is flaky on someone else's rate limiter, and production cold starts
depend on unauthenticated Hub access. **Fix:** add `HF_TOKEN` as a repo
secret + a Render env var. Cheap, removes a whole class of red builds.

---

## 4. MEDIUM — unauthenticated capability disclosure

Live, no credentials:

| endpoint | status | leaks |
|---|---|---|
| `/v1/blocks` | **200** | every block: name, version, description, tags, ui_schema |
| `/api` | **200** | product name, version, block count, endpoint map |
| `/metrics` | 200 | documented public by design |
| `/openapi.json` | 404 | **closed tonight** (was 200) |

`/v1/blocks` and `/api` are the same class as the `/openapi.json` hole that
was just closed: they describe the whole capability surface to anyone. Four
routers import no auth helper at all: `blocks.py`, `health.py`, `static.py`,
`__init__.py`.

Not urgent — no data is exposed — but for a client-desk deployment the API
surface should not be enumerable anonymously.

---

## 5. MEDIUM — `runtime.py` is a 5,171-line god file

| file | LOC |
|---|---|
| `app/agents/runtime.py` | **5,171** |
| `app/containers/construction/boq.py` | 2,111 |
| `app/containers/construction/schedule.py` | 2,049 |
| `app/core/doc_index.py` | 1,949 |
| `app/containers/construction/documents.py` | 1,896 |

`runtime.py` holds the LLM ladder, tool schemas, the grounding directives,
citation parsing, history scrubbing, the routing gate and the streaming loop.
Every incident tonight touched it. It is the highest-churn, highest-risk file
in the repo and it has no internal seams.

Splitting it is a real project, not a tidy-up — but it should be on the plan.

---

## 6. MEDIUM — test suite: 116 skips and order-dependence

- **116 skipped** in the last clean CI run; 2 `xfail` markers.
- **Order-dependent failures**: a full local run shows ~28 red RAG tests
  purely from a contaminated `data/` directory (embedder-identity mismatch).
  Clean `DATA_DIR` → the same files pass 25/25. `test_backfill_layers` fails
  beside three GK files and passes alone.

The suite is green on CI and that is real, but **a local run is not
trustworthy without a clean `DATA_DIR`** — documented in
`CLIENT_DESK_READINESS_20260802.md` §4b after it produced a wrong conclusion
in this very session.

**Recommendation:** make the RAG tests hermetic (own tmp DATA_DIR per module)
so local and CI agree.

---

## 7. LOW — repo hygiene

- **52 local branches / 26 remote**; 3 remote branches already merged into
  `main` and never deleted.
- **`data/` holds 5.2 GB locally**, of which 50 MB is tracked (the
  `safety_world_v2.onnx` model at 50.3 MB — deliberate, it is baked into the
  image). The rest is untracked working data: two ~540 MB SQLite DBs and
  ~1.2 GB of Vol 3 drawing PDFs. Fine, but worth knowing before anyone
  clones onto a small disk.
- **`review_pack/` — 93 files, 907 KB tracked.** This is sweep evidence; it
  will grow every audit. Consider a retention rule.
- **`.git` is 141 MB** — healthy for the history size.
- **Frontend ships one 478 KB JS chunk** (145 KB gzip), no code splitting.
  Acceptable today; the first thing to look at if load time becomes a
  complaint.

---

## 8. Deliberate dormancy — correct, but know it is there

Flags shipped OFF in production, each intentionally:

| flag | prod | note |
|---|---|---|
| `RAG_LAYERED` | unset (off) | layered RAG, deferred to client deployment |
| `RAG_RERANKER` | unset (off) | shipped with a measured NEGATIVE verdict |
| `RAG_FOLLOWUP_CONTEXT` | unset (off) | added this session; needs a measured trial |
| `RAG_HYBRID_SEARCH` | unset but **defaults true** | hybrid IS on — verified |
| `RAG_K` | unset, defaults **5** | only five chunks per turn |
| `FORCE_CALC_ON_DIMENSIONS` | unset, defaults **on** | the calculation directive |

`RAG_K=5` is worth revisiting: several retrieval misses this session came
down to five chunks not being enough for a broad question.

---

## 9. What is genuinely healthy

Worth stating, because an audit that only lists problems misleads:

- **Golden gate 28/29**, the best recorded result, on a clean full sweep.
- **68-feature sweep: zero unexplained failures.** Every failure classified
  as oracle artifact, structure check, or the one real bug (now fixed).
- **Auth coverage is good**: 125 `Depends(require_*)` across 141 route
  decorators; the uncovered ones are health/static/blocks by design.
- **Secrets hygiene is correct**: `.env` is gitignored and untracked; the
  Fernet key is now backed up offline.
- **Deliverables genuinely work**: schedule export produces a valid 37 KB
  workbook — 5 sheets, 210 rows of real CPM data with predecessors and float.
- **Dependencies**: 1 open high, assessed unreachable with evidence.

---

## Priority order

1. **Ban `except Exception: pass`** (lint rule + triage the 72). Highest
   yield: it is the class that hides every other class.
2. **Add `HF_TOKEN`** to CI and Render. Removes a recurring red build and a
   cold-start dependency.
3. **Make the RAG tests hermetic** so a local run means something.
4. **Gate `/v1/blocks` and `/api`** for the client deployment.
5. **Plan the `runtime.py` split** — not urgent, but it is where every
   incident lands.
