# Live-fire convergence campaign — The Fork

**Target system:** The Fork, live deployment at `https://theshovel.ai`
(Render `srv-d9s6l67avr4c73aiujsg`, commit `8089edd`).

**Scope (operator, verbatim):** "the fork, is the system, and all the
construction codes and formulas" / "the whole application from A to Z from
log in".

**End-to-end path under test:**
`auth contract -> create project -> upload document -> index -> retrieve a
SYNTHETIC needle -> execute a construction formula -> export to docx`

**Standing condition during this campaign:** open registration was enabled on
this production instance today, and the instance has no per-project data
isolation (empty projects fall back to the Master Corpus by design). The
campaign therefore runs against a publicly-registerable box holding client
data. Flagged to the operator; awaiting their call on reverting the flag.

**Leg NOT covered:** a successful password login. I do not create accounts or
authenticate with passwords, so login-success is operator-verifiable only and
is recorded as UNTESTED rather than passed (rule 5: a claim the harness cannot
falsify is not tested). The login *rejection* path IS tested, and every
authenticated stage runs over the API-key path, which resolves to the same
`require_user` dependency.

**Real dependencies, no mocks:** Neon Postgres, BAAI/bge-small-en-v1.5 embedder,
Kimi k2.6 (Moonshot), Cloudflare R2, python-docx renderer, Cloudflare DNS +
Render edge.

## Enforcement rules active

| # | Claim the system makes | How the harness falsifies it |
|---|---|---|
| E1 | "document uploaded" | read the chunk count back from the index; >0 required |
| E2 | "answer grounded in YOUR document" | assert the Master-Corpus fallback note is ABSENT and sources name the uploaded file |
| E3 | "answer is correct" | question has a unique needle value present in no other corpus document |
| E4 | "exported" | unzip the docx and read the needle back out of the XML |
| E5 | "persisted" | re-read via a separate request, not the response body of the write |

## Wrap-around greens explicitly refused

- Chat `200` whose answer came from the Master Corpus while the question was about the uploaded doc.
- Upload `201` where indexing produced zero chunks.
- Export `200` returning a well-formed docx with no answer content in it.
- `sources: []` treated as acceptable alongside a confident answer.

## Stop rule

Same failure surviving two full runs -> stop running, switch to isolated
empirical probes against the real component.

---

## Ledger (append-only)

### Run 1 — 2026-08-14 16:50 UTC — VERDICT: FAIL (2 real defects, 1 test defect)

Target `https://theshovel.ai`, commit `8089edd`. Project created for the run:
`60cfc7bf` (`convergence-campaign-20260814-195036`) — **never delete**.

| Stage | Result | Note |
|---|---|---|
| S1a login rejects unknown user | PASS | HTTP 401, no user enumeration |
| S1b unauthenticated read refused | PASS | HTTP 401 on `/v1/agents` |
| S1c api key authenticates | PASS | 14 agents |
| S2 project persisted (independent read-back) | PASS | create 201, separate GET 200 |
| S3 upload actually indexed | **FAIL** | chunk_count 0 |
| S4 synthetic needle retrieved | **FAIL** | "could not confirm this reference in the indexed project sources" |
| S5 calculator executed | pass-but-unsound | see F3 |
| S5b arithmetic correct | PASS | 945 present |

**F1 — MAJOR: uploaded documents are never indexed.**
Mechanism (confirmed in code + `render.yaml:18-23`, not inferred): the arq
ingest worker was deleted 2026-08-08 to halve the compute bill, on the stated
assumption that "inline ingestion in the web request is acceptable". But
`app/routers/upload.py:124-132` only falls back to inline
`doc_index.maybe_eager_index` when `enqueue_ingest()` returns **False**, and
`app/worker/ingest_queue.py:12-21` returns False only when `REDIS_URL` is
unset or Redis errors. Redis was deliberately **kept** (sessions, rate
limiting). So the enqueue always succeeds, the inline fallback never runs,
and no worker drains the queue. Every document uploaded through the app since
the worker was removed sits at `chunk_count = 0` forever.
Why it escaped notice: the Master Corpus is loaded by bulk-insert over HTTPS,
a different path, so corpus retrieval kept working.
Root cause in one line: **"Redis is configured" was used as a proxy for "a
worker exists". They are different facts.**

**F2 — MINOR but real: exception-as-control-flow on every request.**
`app/main.py:390` calls `_jwt_auth.decode_token()` on whatever credential
arrives; an API key is not a JWT, so every API-key request raises
`DecodeError`, is swallowed, and logs a FULL TRACEBACK at WARNING. Confirmed
in live logs — one per request. Buries real warnings.

**F3 — TEST DEFECT (rule 4 violation, mine): a wrap-around green.**
S5 asserted `len(tool_calls) > 0` and passed on `tool_calls=[None]` — an event
whose name never resolved — while the answer visibly did the arithmetic in
prose and said the waste allowance was "Assumed". The server log independently
proves `tools=['construction_calc'] status=success`, so the SYSTEM is fine;
the ASSERTION could not tell an executed calculator from LLM prose. Must be
tightened to name the tool and match the tool_result payload before its green
means anything.

**F4 — MINOR: the SSE tool_call contract is inconsistent, and that is what
made F3 possible.** `runtime.py:2947` documents `tool_call` as
`{"name", "args", "id"}`; the streaming path at `runtime.py:3823-3827` emits
`{"tool", "args_preview"}`. Any consumer following the docstring reads `None`
for the tool name — which is exactly what the harness did. Recorded, not yet
fixed (rule 1: one defect at a time).

**Harness defect (corrected before it misled):** S3 first read chunk_count from
`/v1/projects/{id}/documents`, which never carries it; enrichment happens only
on the detail route (`app/routers/projects.py:339-345`). Fixed to poll detail.

### F1 fix — committed `9c9289e` on `fix/upload-ingest-reaches-index`

`enqueue_ingest` now requires an explicit `INGEST_WORKER_ENABLED` in addition
to `REDIS_URL`. Default is inline indexing, so a missing worker degrades to
slower ingestion rather than to none.

Red/green verified, and verified red for the RIGHT reason (a first attempt
failed on a foreign-key error in setup, which would have proved nothing):

| Test | old code | fixed |
|---|---|---|
| `test_upload_indexes_inline_when_no_worker_is_declared` | RED (no inline task scheduled) | GREEN |
| `test_reachable_redis_alone_never_implies_a_worker` | RED (`assert True is False`) | GREEN |
| `test_upload_uses_the_queue_when_a_worker_is_declared` | green | GREEN (fence holds both ways) |

Suite at the seam: 14 passed (`test_upload_reaches_the_index`,
`test_ingest_queue`, `test_ingest_worker`, `routers/test_upload`).

### F2 + F4 fixes — commit `98ef073`

- **F2** `_rate_limit_identity` now treats `InvalidTokenError` as the expected
  "this is an API key, not a JWT" case and stays silent; any *other* exception
  still logs loudly, so noise was not traded for blindness.
  `tests/test_rate_limit_identity_is_quiet.py` fences the observability
  property. Verified by stashing only the `main.py` change: RED on the old
  handler (`test_api_key_caller_is_identified_without_logging`), GREEN on the
  fix, with the still-loud fence passing in both states.
- **F4 corrected on the evidence.** My run-1 note called the SSE contract
  "inconsistent". It is not — both SSE events use `tool`, and the docstring I
  cited documents the separate `on_event` callback. The real defect is that the
  SSE shape is UNDOCUMENTED, and my harness read the wrong contract. Fixed
  additively: `name` is now emitted alongside `tool` on both events (frontend
  still reads `tool`, so non-breaking) and both contracts are documented in
  place.

### F5 — monitoring was blind to the client-facing hostname

health-watch probed only `the-fork-jn3t.onrender.com`. A dead Cloudflare
record, a dropped custom-domain binding, or an expired certificate all leave
the slug green — which is why theshovel.ai served 403/404 for five days
unnoticed. Added a separate `public-domain` job probing `theshovel.ai/livez`
on the same DB-free 15-minute cadence, kept separate so a red names the layer.

### Blast radius of F1 — MEASURED, and smaller than feared

`/v1/admin/corpus/collections` (note: the chunk field is `chunks`, not
`chunk_count` — a first parse using the wrong key wrongly reported five
zero-chunk collections, corrected here):

| Collection | Documents | Chunks |
|---|---|---|
| `drive_archive` (= `master_corpus` alias) | 7 | 2296 |
| `training_material` | 25 | 234 |
| `curated_kb` | 25 | 213 |
| `60cfc7bf` (campaign) | 1 | 1 |

58 documents, 2744 chunks — reconciles with `chunks_v2` = 2743 + the repaired
campaign doc. **Every document in the system is indexed.** So F1 destroyed no
existing data: nobody had uploaded through the app since the worker was
removed, which is also why it stayed invisible. It would have hit the client
on their first upload.

### Earlier row-count probe (retained)

`pilot-preflight`: 250 projects, 58 documents, `chunks_v2` = 2743,
`chunks` (legacy) = 0. The Master Corpus is intact because it was bulk-loaded
over HTTPS; it is specifically **app-uploaded** documents that went nowhere.
Repair path exists and is not destructive:
`POST /v1/admin/debug/project-reindex` and `/v1/admin/debug/doc-reindex`.

### F1 root cause CONFIRMED empirically (not just by code reading)

Ran `POST /v1/admin/debug/project-reindex?project_id=60cfc7bf` against the
still-unfixed production build. Result `{"indexed":1,"total_chunks":1}`, and an
independent read-back moved chunk_count 0 -> 1. Re-ran S4: the answer came back

> "The flange bolt torque specified for asset tag **QTX-7741** is **337 N.m**"
> ... *Source: COMMISSIONING RECORD - ASSET QTX-7741, chunk 0*

`needle_present=True, master_corpus_fallback=False`. The value 337 N.m is
fabricated and exists in no other corpus, so this answer could ONLY come from
the uploaded document.

**What this proves:** the extractor, chunker, embedder, pgvector store and
retriever were all healthy the whole time. The single broken link was the
routing decision in `enqueue_ingest`. That is the difference between a surface
symptom ("RAG is broken") and the mechanism, and it is why the fix is four
lines rather than a retrieval rewrite.

### S6 (export) exercised against production — PASS, first run

Run against the repaired project `60cfc7bf`, pre-deploy:

- A SECOND synthetic needle retrieved independently of the first: "The
  hydrotest pressure recorded for asset tag QTX-7741 is **18.5 bar**",
  correctly cited. Two distinct fabricated values now retrieved, so S4's pass
  was not a one-off.
- Export HTTP 200, **93,325 characters** extracted from the docx XML — not a
  well-formed-but-empty file, which is the wrap-around green E4 exists to
  refuse.
- Footer stamps `theshovel.ai` with **no** `onrender.com` anywhere, confirming
  `PUBLIC_BASE_URL` is genuinely in effect on the running process rather than
  merely set in the dashboard.

Note the trap this stage was designed around and avoided: the export renders
from `agent_memory`, and the predefined-orchestrator path does not write there,
so a workflow-shaped question yields a 404 that looks like a defect and is not
one. The stage deliberately asks a lookup-shaped question first.

**Local regression on the riskiest edit:** 227 passed across the streaming /
agent / chat suites (`-k "stream or tool_surface or agent_runtime or chat"`),
so the `runtime.py` tool-event alias broke nothing.

### F6 — Google Drive OAuth dead since 2026-08-09 — RESOLVED 2026-08-14

Not introduced by this campaign; found while tracing the domain migration.
`GOOGLE_REDIRECT_URI` pointed at the deleted `the-fork.onrender.com`, and that
was the ONLY callback registered on the OAuth client, so consent succeeded and
then dumped the user on a 404.

Falsified without authenticating: built the consent URL for three candidate
callbacks and fetched each. `redirect_uri_mismatch` vs the Google sign-in page
distinguishes registered from not, and grants no consent. Only the dead host
was accepted.

Operator registered both callbacks. Env now points at the onrender callback
(theshovel.ai was still inside Google's stated 5-minutes-to-hours propagation
window). Verified by fetching **the exact URL the server emits** from
`/v1/drive/connect` — Google returned `<title>Sign in - Google Accounts</title>`,
so the flow is live up to the human consent click.

A callback on the slug is safe: `/v1/drive/callback` takes no auth (the signed
`state` carries the user_id) and 302s back to `FRONTEND_URL`, so the SPA
session is untouched. **Follow-up:** flip `GOOGLE_REDIRECT_URI` to the
theshovel.ai callback once propagation completes.

### CI flake — recorded, deliberately NOT speculatively fixed

`tests/test_admin_corpus_reconcile.py::test_reconcile_reports_no_mismatches_when_clean`
failed with sqlite "database is locked" on this PR's first run, and on PR #333
before it. `app/core/db.py:79` already sets a 30s busy timeout, so the short
-timeout explanation is wrong and the real mechanism (likely reader/writer
contention under the rollback journal, which WAL would remove) is unproven.
Rule 2 forbids fixing a surface error, so this is logged rather than patched
with a guessed WAL pragma. If it recurs on this PR it gets a real reproduction
first.

### F7 — UptimeRobot: all three red monitors were FALSE ALARMS — FIXED

Same bug class as F5, third instance today. Read-only sweep of the account
showed 3 of 5 monitors DOWN. None was a real outage:

- `theshovel.ai` DOWN since **2026-08-04**, last event `405 Method Not Allowed`.
  UptimeRobot HTTP monitors default to **HEAD**, and FastAPI's `@router.get()`
  registers GET only — unlike raw Starlette it does not auto-add HEAD — so every
  path 405s. Verified directly: `HEAD /` and `HEAD /livez` return 405 while GET
  returns 200. Repointed to `/livez` with `http_method=2`.
- `Cerebrum-Blocks store` and `CerebrumDev backend` pointed at pre-rebuild
  hostnames returning 404; the current hosts return 200. Repointed.

Verified by outcome, not by the edit succeeding: all five now report UP with a
logged `type=2 ... 200 OK` event at 18:08. Note v2 `editMonitor` works on this
account even though `newMonitor` is plan-restricted.

**The pattern worth keeping:** health-watch, Cloudflare DNS and UptimeRobot were
ALL watching hostnames that no longer existed. A monitor stuck red is exactly as
useless as one stuck green — the operator had both simultaneously.

### My own test, three attempts — a lesson in measuring the right thing

`test_unexpected_failure_is_still_loud` passed in isolation and failed only in
the full CI suite, twice.

1. `caplog` — observes records only via propagation to root; measures ambient
   logging config, not the function.
2. A handler attached to `app.main`'s logger — still subject to logger level,
   `logging.disable()`, and 3000+ tests' leftovers. Failed identically.
3. **Substitute the module-level `logger` name itself.** The property is "does
   this function report the failure?", which is a question about the CALL, not
   about delivery. Immune to every ambient variable.

Hypotheses raised and *falsified before* acting on them, rather than pushed
blind: pytest-randomly reordering (not installed), a test replacing
`app_main.logger` (none does), `importlib.reload` of `jwt_auth` breaking patch
identity (reload preserves the module object), and `test_rate_limit.py`
interference (touches no logging). Rule 6 in practice: after the second
identical failure, stop paying for CI runs and reproduce locally.

The retry also carries a diagnostic: the test now records whether the patched
`decode_token` was actually invoked, so a third failure distinguishes "warning
not emitted" from "patch never reached the code" instead of just asserting False.

### PR #334 MERGED — squash `14124d1`, 2026-08-14 18:34 UTC

All 13 checks green including both test profiles and test-postgres. Local main
synced, branch deleted, the two deliberately-untracked files preserved.

### Run 2 — INVALID AS VERIFICATION (my harness bug), useful as a baseline

The deploy-watcher compared the live commit against `8089edd` (7 chars) while
Render returns `8089eddb` (8), so the mismatch read as "a new deploy landed"
and the run fired against the OLD build. Recorded rather than quietly re-run:
a run that verifies the wrong build is precisely the false green this campaign
exists to refuse. Fixed by matching the TARGET commit prefix positively
instead of negating the old one.

As a pre-deploy baseline it is genuinely useful, because the failures are
exactly predictive of what the deploy should fix:

| Stage | Run 2 (pre-deploy) | expected after deploy |
|---|---|---|
| S3 upload indexed | FAIL chunk_count=0 | PASS |
| S4 needle retrieved | FAIL | PASS |
| S5 calculator executed BY NAME | **PASS** `['construction_calc']` | PASS |
| S5b arithmetic 945 | PASS | PASS |
| S5c tool events aliased | FAIL `all_aliased=False` | PASS |
| S5d tool_result ok | PASS | PASS |
| S6 export has content | PASS 93381 chars | PASS |
| S6b footer stamps domain | PASS | PASS |
| S1a/b/c, S2 | PASS | PASS |

**F3 is now closed on the evidence:** with the tightened assertion the tool
resolves as `construction_calc` in BOTH the call and the result. The original
`[None]` was purely my parser reading the callback contract's key against the
SSE contract. The calculators were never dormant on this path.

### Run 3 — 2026-08-14, deploy `14124d15ab` — VERDICT: PASS, 12/12, CONVERGED

Full chain, real dependencies, every enforcement rule active.

| Stage | Result | Enforcement that made the green mean something |
|---|---|---|
| S1a login rejects unknown user | PASS | 401, no user enumeration |
| S1b unauthenticated read refused | PASS | 401 |
| S1c api key authenticates | PASS | 14 agents |
| S2 project persisted | PASS | read back by a SEPARATE request |
| **S3 upload actually indexed** | **PASS** | chunk_count read back from the index, not the upload's own 200 |
| **S4 synthetic needle retrieved** | **PASS** | fabricated value no corpus can produce; fallback note absent |
| S5 calculator executed | PASS | `construction_calc` named in BOTH call and result |
| **S5c tool events aliased** | **PASS** | `all_aliased=True` — F4 fence flipped red to green |
| S5d tool_result ok | PASS | `ok` flag asserted, not merely present |
| S5b arithmetic 945 | PASS | 900 x documented 1.05 |
| S6 export has content | PASS | 93,324 chars read back OUT of the docx XML |
| S6b footer stamps domain | PASS | theshovel.ai present, onrender absent |

The three stages that failed in the Run 2 baseline (S3, S4, S5c) are exactly
the three the deploy fixed, and nothing else regressed. That predictive match
is the strongest available evidence that the diagnosis was the mechanism and
not a coincidence.

**CAMPAIGN STATUS: CONVERGED.** Rule 8 satisfied — a run passed with all
enforcement active against the deployed fix.

### Defects found and fixed this campaign

| # | Defect | Fix | Fence |
|---|---|---|---|
| F1 | uploads enqueued to a queue nobody drained; every app upload unindexed since 2026-08-08 | `INGEST_WORKER_ENABLED`, default inline | `test_upload_reaches_the_index.py` |
| F2 | full traceback logged per request (API key is not a JWT) | expected `InvalidTokenError` is silent, anything else stays loud | `test_rate_limit_identity_is_quiet.py` |
| F4 | SSE tool events carried no `name` | additive alias, both contracts documented | live S5c |
| F5 | health-watch blind to the client hostname | separate `public-domain` job | workflow |
| F6 | Drive OAuth dead since 2026-08-09 | callback registered + env repointed | live consent-URL probe |
| F7 | all 3 red UptimeRobot monitors were false alarms | repointed to live URLs, HEAD -> GET | all 5 UP |

### Still open, owner-gated

- Open registration is ON with no per-project isolation (Master Corpus is
  reachable by any signup). One env flip reverts it.
- `test_admin_corpus_reconcile` sqlite lock flake — logged, unreproduced, not
  speculatively patched.
- Pre-existing owner items unchanged: golden-set corpus backfill, the two
  client-content fixture binaries, discipline-hat product decision,
  `RAG_LAYERED`. Rule 8 requires a run
that passes with all enforcement active; the fixes are not deployed, so S3/S4
still fail in production. S5 has been tightened (must name `construction_calc`
in both call and result, must carry both keys, must report ok) and S6 (export,
needle read back out of the docx XML) has been added but neither has been
exercised against a live run yet.
