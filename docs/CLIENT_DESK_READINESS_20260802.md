> **SUPERSEDED (2026-09-28).** Historical snapshot. Not a source of open items or current status. Current state lives in the fleet standing-order loop and the live measurements.

# Client-desk readiness verdict — 2026-08-02

> **Superseded by `CLIENT_DESK_READINESS_20260813.md`** — this snapshot
> predates the 2026-08-09 Neon/Render migration and the 2026-08-12 PRR
> (PRs #331/#332). Kept unedited as the record of what was true on
> 2026-08-02.

Scope of this pass: the items `PLATFORM_HEALTH_REPORT.md` lists under
**"NOT COVERED BY THIS SWEEP (the honest boundary)"**, plus the open
dependency alerts. That NOT-COVERED list is what "all aspects, not RAG
only" actually means — it is the only definition of remaining work that
was already written down and agreed, so it is the one used here.

**Verdict: NOT 100%. Code-complete on everything code can close; five
items are owner-gated and one could not be executed on this machine.**
Each line below is either verified with a named artifact, or explicitly
marked unverified with the reason. Nothing is claimed that was not run.

---

## 1. What this pass found and fixed

Two live **false-signal** bugs — both found by executing NOT-COVERED
items, both of the same family: a check that reports success without
measuring the thing it claims to measure.

### 1a. `pilot-preflight` graded a retired table (PR #299)

Live prod before the fix:

```
chunks_embedding_type : vector(256)
embedding_dim_ok      : True
row_counts.chunks     : 0
```

`chunks` has been empty since the 2026-07-12 v2 migration; every write
goes to `chunks_v2` at `vector(384)`. The operator readiness gate read
the width off a **dead table** and compared it to a hardcoded
`vector(256)`. It answered `True` while telling us nothing about the
store that actually serves retrieval, and would have stayed green with
`chunks_v2` missing, empty, or the wrong width.

This is the same class already fixed once in `admin.py` for the corpus
count — where reading `chunks` showed 0 chunks for every project and
"invited a destructive re-index click". The preflight was left behind.

Now reports the active table, its real width, the legacy width, and
compares against the **loaded embedder's** dim. Cold embedder returns
`null`, never a green boolean from an absent measurement.

**Verified live in prod after the #299 deploy:**

```
active_chunk_table          : chunks_v2
active_chunk_embedding_type : vector(384)
active_chunk_embedding_dim  : 384
embedder_dim                : 384
embedder_loaded             : True
embedding_dim_ok            : True        <- now measured against the real store
legacy_chunks_embedding_type: vector(256) <- kept visible, correctly labelled legacy
row_counts.chunks           : 0           <- the retired table, as expected
row_counts.chunks_v2        : 134121      <- the store that actually serves retrieval
```

The gate still says `True`, but for the first time it says it about the
134,121 chunks that answer real queries rather than about an empty table.

Three operator-facing docs were telling operators to verify against the
retired table and were corrected: `docs/backup-and-recovery.md` (live
runbook), `deploy/PILOT.md` (dated history — annotated SUPERSEDED, not
rewritten), `README.md` architecture diagram.

### 1b. Drive admin routes returned 500 for an API-key caller (PR #299)

```
GET /v1/admin/drive/scan -> 500 INTERNAL_ERROR
KeyError: 'user_id'
```

`require_api_key` accepts two principal shapes and **normalises only
one**. The JWT branch resolves a real user and includes `user_id`; the
API-key branch returns the `auth_manager` key record — `user`/`tier`/
`role`, no `user_id`. `CEREBRUM_MASTER_KEY` is minted with
`role: "admin"`, so a master-key caller passed `_require_admin` and then
died indexing `auth["user_id"]`.

Effect: the master key **could not use the admin Drive workflow at all** —
neither `drive/scan` nor `approve-from-drive` (which also used
`auth["user_id"]` as the owner of the project it creates).

The existing tests missed it because their fixture overrides
`require_api_key` with a dict that *has* `user_id` — a fixture more
generous than production. The new tests use the exact dict
`auth_manager` mints for the master key.

### 1c. The class behind 1b was triaged in full — it is closed

79 `auth["user_id"]` hard-index sites exist. An AST triage
(`scratchpad/auth_triage.py`) mapped every one to its route's auth
dependency:

| Dependency | Sites | Exposed? |
|---|---|---|
| `require_api_key` | 4 (all Drive admin) | **was exposed — fixed in #299** |
| `require_user` | 75 | No — safe by construction |
| helper / closure | 5 (`agents.py`) | No — callers all use `require_user` |

`require_user` is safe for a better reason than "JWT-only": it accepts
API keys too, but **normalises** them to the singleton system user, so
`user_id` is always present. `require_api_key` does not normalise. That
asymmetry is the root cause and is worth closing at the dependency
level as hardening — see §5.

**Exposed sites after #299: 0.**

---

## 2. Verified green (with artifacts)

| Aspect | Result | Evidence |
|---|---|---|
| **Golden set — clean full sweep** | **28/29 PASS — gate MET** (bar >=27/29). Best recorded result; the previous best was 27/29. | `golden_set_results.jsonl`, `GOLDEN_SET_REPORT.md` |
| Authenticated route sweep | 59 GET operations, 41 OK, **1** 5xx (fixed above) | `review_pack/sweep/route_sweep_20260802.{json,txt}` |
| Document-register golden question | **PASS live** after PR #297 deploy | `golden_set_gate.py --only pilot_document_metadata` |
| Live vector store | `chunks_v2`, `vector(384)`, non-empty; legacy `chunks` = 0 rows as expected | `/v1/admin/debug/pilot-preflight` |
| Live service health | 200 throughout the sweep | route sweep log |
| Deploy currency | `dep-d9naa9m…` live at `d0ade99` (includes #297) | Render deploys API |

**Route-sweep boundary, stated plainly:** 79 **mutating** operations
(POST/PUT/PATCH/DELETE) were deliberately **not** probed. This ran
against the live client service, and a mutating sweep on a live corpus
is the cascade-delete class this repo has already survived once. The
artifact records the unprobed list rather than implying full coverage.

---

## 3. Dependency alerts — measured, not asserted

Baseline at session start: **15 alerts (7 high)**.

Merged: #260 react-router 7.18.1, #271 postcss 8.5.23, #293
brace-expansion 5.0.9. #299 carries pyasn1 0.6.3 -> 0.6.4 (three HIGHs).

**Residual after #299 merged: exactly one HIGH, open — MEASURED, not
predicted.**

```
$ gh api repos/bopoadz-del/The_Fork/dependabot/alerts --paginate \
    -q '[.[]|select(.state=="open" and .security_advisory.severity=="high")]|length'
4      # t+30s after merge
1      # t+60s after merge
$ ... -q '.[]|select(.state=="open" and .security_advisory.severity=="high")
         |"\(.dependency.package.name) \(.security_vulnerability.vulnerable_version_range)"'
react-router >= 7.12.0, < 8.3.0
```

Note the lag: for the first ~60s after the merge the API still returned
the three pyasn1 HIGHs against a `requirements.txt` that already pinned
0.6.4. **Dependabot re-evaluates the manifest asynchronously**, so a
count taken immediately after a merge is not the settled number — re-read
it before drawing a conclusion.

Full open set now: **1 high, 1 medium, 4 low**, every one dispositioned
in the table below.

| Alert | Disposition |
|---|---|
| react-router `>=7.12.0,<8.3.0` (HIGH, GHSA-qwww-vcr4-c8h2) | **OPEN — assessed unreachable, NOT cleared.** The advisory is an *RSC-mode* CSRF bypass. `frontend/` is a client-side Vite SPA using `BrowserRouter`; zero RSC markers, no `use server`, no react-server imports, no server actions. Clearing it needs the v8 **major** migration, which is not justified for an unreachable path on a live client app. The alerts API will keep returning it — that is expected, not an oversight. |
| pytest `<9.0.3` (MEDIUM) | **Deferred — the upgrade experiment was INCONCLUSIVE, see §4b.** An earlier reading of this pass claimed pytest 9 caused 27 failures and that the pin was therefore load-bearing. That was **wrong** and is retracted: the identical failure set occurs under the pinned pytest 8, and it is local environment contamination, not a pytest-version effect. There is no evidence either way about pytest 9. What remains true and sufficient for deferral: pytest is installed in the production image (the Dockerfile installs `requirements.txt`) but is never invoked there, and the advisory is tmpdir handling, which requires *running* pytest. The pin's stated rationale (pytest-asyncio 0.x coupling) is documented in `requirements.in` but was NOT validated here. §5 has the structural fix that removes the exposure entirely. |
| torch `<=2.12.1` (LOW ×3) | **Unreachable.** `torch.jit.script` has zero call sites in `app/` or `scripts/`. |
| @babel/core (LOW) | Dev-time build tool only. |

---

## 4. Test-suite honesty (KNOWN_LIMITATIONS §8)

The "17 skip/xfail files" were enumerated. By their skip *reasons* they
are environment-gated rather than hidden failures: Postgres-only cascade
rules, Node/tsc not installed, POSIX-only resource limits, missing
optional IFC/XER fixtures, and flag-off no-op guards.

**Stated precisely, because KNOWN_LIMITATIONS §8 is exactly a warning
against overclaiming here:** they were classified by reading their skip
conditions. The gated paths were **not executed** — no environment was
stood up (Postgres, Node, POSIX, the missing fixtures) to confirm they
would pass if un-gated. The full run reports **31 skipped**; that set was
not individually enumerated against the reasons. So the honest claim is
"no skip is masking a *known* failure", **not** "the skipped paths are
verified working". Closing that properly needs a run in an environment
where they actually execute.

Two `xfail` markers were the real question — both said "GK crowds
project docs out of top-5". Re-checked this pass; **both markers are
correct and both were kept:**

- `test_doc_index.py::test_search_uses_hybrid_retriever` — genuinely
  xfails.
- `test_doc_search_api.py::test_search_returns_ranked_results` —
  XPASSes **in isolation**, which briefly looked like a stale marker.
  It is not. Under full-suite order it genuinely fails:

  ```
  assert 'concrete.txt' in ['', 'construction_kb.md', '', '', '']
  ```

  — the GK note crowds the project's own upload out of the top-5,
  exactly what the marker describes. The marker removal was tried and
  **reverted**. Isolation runs are not sufficient evidence for removing
  a `strict=False` xfail; full-suite order is.

## 4b. The local dev suite WAS untrustworthy — now FIXED (2026-08-03)

> **RESOLVED.** Both root causes were found and fixed; a local run no longer
> needs a clean `DATA_DIR` to mean something. Kept here because the wrong
> conclusion it produced is worth remembering, and because the two bugs
> underneath were real rather than test-only.
>
> 1. **Leaked embedder identity.** The autouse fixture that resets the
>    embedder / vector-store caches ran ONLY in the Postgres branch, so the
>    default SQLite suite reset nothing. Cache resets alone were not enough
>    either: the embedder identity is PERSISTED per namespace, so a test
>    using the real embedder stamped `potion-base-8M` onto namespace `v2`
>    and the next test expecting `fake` hit the mixed-model guard. Fixed by
>    giving every test its own vector namespace (PR #310).
>
> 2. **Schema init followed a global flag, not the database.**
>    `_initialized` was a module-level boolean while the database is
>    per-`DATA_DIR`, so after init ran against one database every other one
>    silently got no schema — `no such table: projects`. This was a REAL
>    defect, not a test artifact: it misbehaves anywhere the database can
>    change after first use. Fixed in `projects`, `users`, `agent_memory`
>    and `workflows` by tracking `_initialized_for_url`, the pattern three
>    sibling stores already used (PR #311).
>
> Measured: the reproducing combination went **28 failed → 0**, and the
> broad selection **4 failed → 203 passed**. All three CI test jobs green.

The original note follows, unchanged, because the mistake it records is the
point.


A full local run reports **28 failures** across `test_gk_lexical_fold`,
`test_gk_ranking_knobs`, `test_rag_injection`, `test_backfill_layers` —
under **both** pytest 8 and pytest 9. Every one is the same error:

```
RuntimeError: Embedding identity mismatch in namespace 'v2':
expected {'model': 'fake', 'dim': 256}, found {'model': 'minishlab/potion-base-8M', 'dim': 256}
```

The local `data/` tree carries a namespace stamped by a real-embedder
run, and the vector store's mixed-model guard (a *correct* safety
feature) fires against tests that use the `fake` embedder.

Proof it is environmental, not a code regression:

```
$ DATA_DIR=$(mktemp -d) DATABASE_URL="" pytest tests/test_gk_lexical_fold.py tests/test_gk_ranking_knobs.py
25 passed in 10.01s
```

CI is green because it runs from a clean checkout. **CI is the source of
truth; a local full-suite run is not, unless DATA_DIR is clean.** Anyone
who runs the suite locally and sees ~28 red RAG tests should check this
before believing the platform is broken — and specifically must not
conclude anything about a dependency upgrade from it, which is the
mistake §3's pytest row records.

---

## 4a. Project isolation — verified BY DESIGN, with a tenancy caveat

Probed as part of the retrieval-isolation trace. A project owning **zero
documents** (`76d7596a`, "Triage QA Empty Project") returns 5 real client
documents for `concrete curing Portland cement` — the client project II
specification PDFs and MOS cast-insitu concrete method statements.

Chased to ground rather than assumed:

- Not the master-corpus **alias** path — `76d7596a` is not an alias, so
  `doc_search` searches its own id.
- Not the **general-knowledge** layer — `curated_kb` returns entirely
  different documents for the same query (`construction_kb.md`, FIDIC
  notes, EW-2 workbooks). The the client specs are not in GK.
- It is `retrieve_with_filter` **STEP 0b**: an explicit *"empty/thin
  detection for the labeled Master-Corpus fallback"*. A project whose
  best own chunk cannot clear `RAG_CONFIDENCE_THRESHOLD` deliberately
  falls back to the Master Corpus rather than answering with nothing.

**So this is intended behaviour, not a leak.** For the current
single-client desk it is correct and useful.

**The caveat that must be said out loud:** there is therefore **no hard
per-project document isolation** in this configuration. Any project that
is empty or thin will surface Master-Corpus content. For one client that
is the point. For a **multi-client / multi-tenant** deployment it means
one client's documents can appear inside another client's project, and
that is what the dormant `RAG_LAYERED` work exists to address. Do not
put two different clients on this instance as configured.

Secondary note — **fixed, on the second attempt.** The chat path already
disclosed the fallback, but the raw `/v1/projects/{id}/documents/search`
payload carried no corpus-origin field, so a caller could not tell an
own-document hit from a Master-Corpus fallback hit. Each result now
carries `origin: "own" | "general_knowledge" | "master_corpus"`.

> **The first attempt (#301) shipped as a silent no-op.** The field was
> added to the intermediate dict inside `search_project_documents`, but
> the returned rows are rebuilt from scratch further down, so it never
> reached the wire — and CI stayed green because nothing asserted on the
> value a caller actually receives. Caught by probing the live endpoint
> after deploy and seeing `origin=None`. Fixed in #302, with tests that
> assert on the returned rows and that were **verified to fail against the
> pre-fix code** (4 failed → 4 passed) rather than assumed to.

`Chunk.layer` is deliberately withheld from the **LLM** path (the chat
runtime reads it to phrase its own disclosure). Surfacing it to a direct
API consumer is consistent with that intent rather than against it: it is
not model context, and the alternative is a caller silently attributing
another project's document to this one.

---

## 5. Hardening — ALL FOUR SHIPPED (follow-up PR)

These were listed as "found but not shipped". They were then implemented:

1. **`require_api_key` normalised.** The api-key branch now returns
   `user_id` (the singleton system user) and `auth_method`, matching the
   JWT branch, so the whole latent-500 class is closed at the dependency
   instead of route by route.
   **Identity only, never authority** — `role` is deliberately untouched.
   The system user *is* admin, so defaulting `role` from it would have
   silently promoted every plain `CEREBRUM_API_KEY_*` to admin
   (`_require_admin` gates on exactly that key). A test pins this: a plain
   key gets a `user_id` but still gets **403** from an admin route.
2. **Test framework stripped from the production image.** The Dockerfile
   now uninstalls `pytest`, `pytest-asyncio`, `pytest-cov`,
   `pytest-json-report` and `diff-cover` after the lock install. Safe:
   nothing under `app/` imports pytest at runtime, and CI installs the
   test stack explicitly (`test.yml`) rather than relying on
   `requirements.txt`. This removes the pytest advisory from the
   production surface without the pytest-9 migration.
3. **torch drift documented — and a false claim corrected.**
   `deploy/DEPENDENCY_RISKS.md` stated *"Production Render image installs
   `requirements.txt` only"* and marked torch **"Production exposure: No —
   not in `Dockerfile`"**. Both were wrong: the Dockerfile pins
   `torch==2.5.1` directly. The risk acceptance still holds but now rests
   on **reachability** (`torch.jit.script` has zero call sites), not on a
   false absence. A patched `2.13.0` now exists; closing it needs a
   coordinated torch/torchvision/ultralytics/onnxruntime bump plus ONNX
   re-qualification, so it stays accepted with that stated cost.
4. **API surface closed on production.** `/docs`, `/redoc` and
   `/openapi.json` are now disabled when `ENV=production`
   (`API_DOCS_ENABLED` overrides either way). They stayed open off-prod so
   local dev and the on-prem profile are unchanged.

### 5a. Also shipped — docs-only commits no longer redeploy prod

Observed while merging PR #300: a **markdown-only** merge rebuilt the
image and returned **502 for ~40s** on the live client service.
`render.yaml` now carries a `buildFilter.ignoredPaths` for `docs/**`,
`review_pack/**`, `**/*.md`, `LICENSE` and `.github/**`.

> **Operator action required for this one to take effect:** `buildFilter`
> is blueprint config. Like every other change in `render.yaml`, it is
> reconciled only when the blueprint is **applied** in the Render
> dashboard — a git push alone does not activate it.

---

## 6. Could NOT be verified on this machine

- **On-prem fork-in-a-box boot + zero-egress probe.** `deploy/onprem/`
  is packaged but has still never been built or booted. **Blocked, not
  deferred:** this machine has no container runtime — `docker`, `podman`
  and `nerdctl` are all absent, and `wsl` fails with
  `Class not registered`. Needs an operator with Docker installed. The
  sovereignty claims in `SOVEREIGNTY_REPORT.md` therefore remain
  **code-read only, never executed**.
- **Integration path traces as discrete saved evidence** (Drive OAuth
  browse+import, induced-failure error bubble, provider-fallback-logged,
  grounding-gate planted cases). Several require mutating a live client
  corpus; not run this pass. Still open from the previous report.
- **Full user-journey screenshots** to `review_pack/sweep/final/`. Not
  produced.

---

## 7. Owner-gated — status after the follow-up pass

Three of the original five are now closed. Updated 2026-08-02.

1. **Fernet `DATA_ENCRYPTION_KEY` offline backup — DONE (on-machine).**
   Captured from the Render env API straight to
   `~/.thefork-backup/FERNET_DATA_ENCRYPTION_KEY.backup` (mode `600`).
   44 chars, ends `…maw=` — matches the last-4 recorded in the credential
   census, so it is the live value. The value was written directly to
   disk and never printed.
   **Still yours:** keep a copy **off this machine** (password manager or
   offline media). A backup that lives only on the same laptop is one
   disk failure from the same loss. **Never rotate it** — rotation
   orphans every already-encrypted document.
2. **Render API key rotation — WAIVED by the operator (2026-08-02).**
   Same standing-waiver treatment as the DeepSeek keys in
   `PLATFORM_HEALTH_REPORT.md` F4: noted once, not raised again. The key
   (ends `…yNX0`) is stored at `~/.thefork-backup/render-api.env`
   (mode `600`) and is the sanctioned tool for Render work — the operator
   explicitly authorised its use. Prefer it over the Render MCP when
   reading secrets, because curl-to-file keeps values out of the
   transcript while the MCP returns them into context.
3. **OpenAI env vars — ALREADY CLEAN (verified, not assumed).** Queried
   the live service env: `OPENAI_API_KEY` / `OPENAI_MODEL` are **not
   present**, and no code under `app/` references them. Nothing to unset.
   *(While there: the provider ladder was confirmed live as
   `LLM_PROVIDER=kimi`, `KIMI_MODEL=kimi-k2.6`,
   `KIMI_FALLBACK_MODEL=moonshot-v1-128k`, with `LLM_FALLBACK_PROVIDER=groq`
   as a third-tier last resort. `GROQ_API_KEY` is still set, but the
   auto-pick only falls to Groq when a Kimi key is **absent** — it is set,
   so there is no provider drift.)*
4. **Repo secrets for `eval-battery.yml` — DONE.** `FORK_API_KEY` and
   `FORK_BASE_URL` are set on the repository, so the scheduled battery
   workflow can run.
5. **`ab-2023-101 vol 3` chunk backfill — OPEN, yours.** Needs the DB
   allowlist plus the direct-DB re-encode path (heavy scanned PDFs must
   not go through the 512Mi web box, which OOMs on them).

6. **Render build filter — DONE, no operator action needed.** This was
   briefly listed as "apply the blueprint". The operator's blueprint apply
   errored, but the setting landed anyway — confirmed live on the service
   via `GET /v1/services/srv-d8hdc6ek1jcs739rq5sg`:

   ```json
   "buildFilter": {"ignoredPaths": ["docs/**","review_pack/**","**/*.md","LICENSE",".github/**"]}
   ```

   Docs-only commits no longer rebuild the image, so the ~40s 502 per
   markdown merge (§5a) is closed. **Check the service, not the blueprint
   run** — a failed blueprint apply here did not mean a failed setting.

---

## 8. Bottom line

The platform is **code-complete for client-desk deployment** on
everything this pass could reach. The golden gate is **28/29 — MET, the
best result recorded**, measured on a clean full sweep against live prod
rather than inferred. Two real live defects were found and fixed that
would each have produced a false "all good" reading in front of a
client, and one of this pass's own conclusions (the pytest-9 claim) was
caught and retracted rather than shipped.

It is **not 100%**, and no honest report can say so while:

- one HIGH alert stays open (assessed unreachable, not cleared),
- the on-prem sovereignty package has never actually been booted,
- and five items require the operator's own hands.

Those are enumerated above with the exact action for each.

---

# ADDENDUM — live Chrome + feature-sweep testing (2026-08-03)

## The calculators are DORMANT in production — root cause found

76 deterministic calculators are registered (`concrete_volume`,
`guardrail_top_rail_height`, `foundation_bearing_pressure`, …). **None of
them can be forced on the live stack**, so self-contained arithmetic is
answered as a document lookup and refused:

```
"Calculate the concrete volume for a raft 25m x 18m x 1.2m thick and the
 number of 6m3 truck loads"
-> "I cannot find that specific information in the provided reference
    context ... they do not contain a raft measuring 25 m x 18 m x 1.2 m"
```

25 × 18 × 1.2 = 540 m³; 540 ÷ 6 = 90 loads. Nothing needed retrieving.

**Root cause** (`_tool_choice_for`, app/agents/runtime.py):

```python
if provider in ("groq", "kimi"):
    # Kimi K2: forcing a specific tool 400s outright ("tool_choice
    # 'specified' is incompatible with thinking enabled")
    return "auto"
```

Live provider is `LLM_PROVIDER=kimi`, `KIMI_MODEL=kimi-k2.6`. `tool_choice`
is therefore **always `"auto"`**, so the whole `_INTENT_TOOL_MAP` forcing
mechanism — roughly 22 keyword phrases built precisely to guarantee the
deterministic calculator runs — **never fires in production.**

Proved by discrimination, not inference: `bearing pressure` is an
*existing* map keyword and fails identically to an unmapped phrase. On
`"auto"` the model simply prefers the RAG context and refuses.

**A shape detector was added** (calc verb + ≥2 self-supplied dimensions,
`FORCE_CALC_ON_DIMENSIONS=0` to disable, 16 tests). It is **correct but
inert on Kimi** — it returns the right tool name into a mechanism the
provider ignores. It will take effect if the ladder changes; it does not
fix this today. Stated plainly so nobody reads the merge as a fix.

**Fixing it properly** cannot be tool_choice — K2 rejects that. The options,
in preference order:

1. **Pre-empt the LLM.** Detect a self-contained calculation, run the
   calculator deterministically, inject the result as authoritative
   context. Matches this repo's existing "deterministic calculators over
   prose maths" philosophy and is provider-independent.
2. Route calc turns to `moonshot-v1-128k` (non-reasoning; may accept
   `tool_choice`) for that turn only.
3. Prompt-level instruction — weakest, and the reason forcing was
   introduced in the first place.

## Sweep results, honestly scoped

49 features: 21 PASS / 23 FAIL / 3 PARTIAL / 2 BLOCKED — but **19 of the 23
FAILs were an oracle artifact**, not product defects (see the sweep-oracle
commit). Two more were routing-oracle misses on `commissioning_checklist`,
which produced 13,115 chars with every structure check passing.

**Exactly one real product bug in 49 features**: the tokenization-400 from
tool-result truncation, fixed in #304.

## Verified working end-to-end

- **Schedule generation** — HTTP 200, valid 37KB xlsx, 5 sheets, 210 rows
  of real CPM data with predecessors and float.
- **Interim payment** — correct FIDIC 14.3 certificate (900,000 gross,
  10% retention = 90,000).
- **EOT** — correctly reported the missing baseline/as-built XER files with
  requirements rather than inventing an entitlement.
- **Master-Corpus fallback disclosure** on the chat path.
