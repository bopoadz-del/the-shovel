> **SUPERSEDED (2026-09-28).** Historical snapshot. Not a source of open items or current status. Current state lives in the fleet standing-order loop and the live measurements.

# Client-desk readiness verdict — 2026-08-13

Supersedes `CLIENT_DESK_READINESS_20260802.md`. That pass ran against the
pre-migration service; this one covers the PRR that closed the audit branch
and ends with the current deploy. Same rule as before: every line is either
verified with a named artifact, or explicitly marked unverified with the
reason. Nothing is claimed that was not run.

**Verdict: shippable as it stands on `main` at `991e348`. Every workflow on
main is green — including the one that had been red for three days — the
deploy is live and probed, and the remaining items are owner decisions, not
code.**

---

## 1. What this pass found and fixed

Two PRs, both merged, deployed, and verified live.

### PR #331 — the three-bar audit (squash `8deb0b8`)

Every construction capability and the agents subsystem put through three
bars: not-a-stub, current-version, and control-delete (gut the body, run the
tests that claim coverage, require RED). Highlights, each with its fix on
main:

- **Three container actions were suggestible but undispatchable**; two had
  never been runnable (they called helpers that exist in no commit). All
  three implemented and payload-tested against real fixtures.
- **The confident zero**: `track_progress` manufactured
  `progress_percentage: 0, status: success` out of a missing detector. Now a
  refusal that names what is missing.
- **`procurement_optimizer` ranked suppliers nobody had scored** (every
  unscored supplier = 73.8, with a recommended winner). Now scores only
  supplied criteria; unscored carry `total_score: null` and sort last.
- **The agents package could not be imported by its package path** —
  Python-2 implicit relative imports, concealed by a `sys.path` hack in its
  own tests. Fixed and fenced.
- **Five runtime methods (702 lines) had zero effective coverage.**
  `_call_llm` — provider fallback, per-attempt tool_choice/temperature, the
  daily cost cap — was mocked 65 times across 23 files and tested by none of
  them. All five now have transport-boundary tests; a batched control-delete
  fails 67 of 78.
- **`/v1/recommend` reported a platform-wide provider outage on every call**
  (`emergency_mode: true, confidence: 100`) because its roster still listed
  providers removed 2026-07-25 and nothing ever fed it observations. Absence
  of data is now reported as absence of data.
- **The dormant hat system is registered, not wired**: 32 of its 35 declared
  actions dispatch to nothing. A conditional fence makes `FORK_HATS_ENABLED`
  unusable until they exist.

### PR #332 — the dead URL (squash `991e348`)

Found by chasing the one red workflow on main instead of shrugging it off.
**health-watch had failed every scheduled run since the 2026-08-09
migration** — it probed `the-fork.onrender.com`, deleted when the service
moved workspaces. Three days of red that meant nothing is how a real outage
gets ignored. The same dead URL was in the footer of every client-facing
Word export, every ops script default, and the recovery runbooks. The
export footer is now env-derived (`PUBLIC_BASE_URL` →
`RENDER_EXTERNAL_URL` → omitted), so the next migration cannot re-create
the bug.

### Also closed in this pass

- **PRR self-review caught my own defect**: the audit branch had silently
  replaced a 16-test file on main; six assertions had no replacement,
  including the only coverage of `llm_client.complete()`'s fallback.
  Restored before merge.
- **PR #330's identifier scrub missed the roman-numeral spelling of the project code** — 610 occurrences
  in a committed client Drive listing, plus the client project name in tool
  schemas and the system prompt shown to the LLM. Scrubbed; a dead one-off
  extractor with the client name in its filename retired. The tracked-file
  sweep is clean outside #330's four declared exceptions.
- **`MASTER_CORPUS_PROJECT_ID` on the live service** — #330 flagged it as a
  manual follow-up; set to `master_corpus` 2026-08-12 with operator
  approval. Closed.
- **PR #323** (react-router security bump) merged; the closed dependabot
  PRs (#317/#318/#321) verified superseded by manual pins already in
  `requirements.txt` — nothing was dropped.

---

## 2. Verified green (with artifacts)

| Aspect | Result | Evidence |
|---|---|---|
| Local matrix, both CI profiles, frozen tree | virgin **3377 passed** / production-like **3474 passed** | `scripts/test_matrix.py` at `825edf8`; an earlier run was discarded because the tree was edited mid-run |
| PR #331 CI | all 12 checks pass | github.com/bopoadz-del/The_Fork/pull/331 |
| PR #332 CI | all 13 checks pass | github.com/bopoadz-del/The_Fork/pull/332 |
| Main at `991e348` | tests, test-postgres, Docker Build and Publish, lint, CodeQL, dependency-audit, **health-watch** — all success | Actions, `head_sha=991e348` |
| health-watch deep dispatch | liveness + DB-touching readiness both success | run 31647554946 |
| Live health | `/livez` alive; `/ready` healthy — Neon 15-699 ms, embedder loaded, 37/37 blocks | live probes 2026-08-12 22:01–22:57 UTC |
| Live chat, lookup | FIDIC retention answer correct, `kimi-k2.6`, 25 s | `fork_cli` transcript |
| Live chat, calculation | raft 30×20×1.5 ran `construction_calc` — 945 m³ (900 × documented 1.05 waste factor), 119 trucks. The exact bug class the audit fixed, not refusing in prod | `fork_cli` transcript, tool timings in SSE |
| Live `/v1/recommend` | `recommended: null … "This is not an outage"`, `emergency_mode: false` | live response, post-deploy |
| Live `/v1/leaderboard` | roster exactly kimi/groq/ollama, all honestly `unknown/unproven` | live response, post-deploy |
| Export footer | with `RENDER_EXTERNAL_URL`: stamps the live URL; without: omits — never localhost | docx XML inspection, both directions |
| Stub gate | NO REACHABLE HOLLOW FUNCTIONS | `scripts/audit_stubs.py` |
| Secret gate | NO SECRET MATERIAL IN TRACKED FILES | `scripts/scan_secrets.py` |
| Live agents | 14 served | `GET /v1/agents` |
| Frontend | serving | `GET /` |

**Boundary, stated plainly:** the chat smokes are two questions, not a
battery. The golden-set gate was not re-run this pass — its corpus backfill
is owner-gated (below), and the best recorded clean sweep remains 28/29
(2026-08-02). No mutating live-API sweep was run, same cascade-delete
caution as the 08-02 pass.

---

## 3. Fences now standing

Each proven to bite by mutation before merge — the fence fails when the
thing it guards is broken, not merely when someone remembers to check.

| Fence | Bites on |
|---|---|
| `test_construction_actions_reachable.py` | suggestible-but-undispatchable actions; phantom helpers, transitively |
| `test_agent_tool_surface.py` | an advertised agent tool the runtime cannot dispatch |
| `test_no_implicit_relative_imports.py` | sibling-as-top-level imports anywhere under `app/`; the dual-module hazard |
| `test_hats_are_dormant.py` | enabling `FORK_HATS_ENABLED` while declared actions dispatch to nothing |
| `test_monitoring_roster_is_honest.py` | monitor roster drifting from what `_llm_config` can return; outage claims with zero observations |
| `test_container_delegations.py` | a delegation that stops delegating or names a block no profile declares |
| `test_llm_provider_fallback.py` | fallback firing on 4xx, or not firing on 429/5xx/timeout; forced tool_choice reaching kimi/groq |

---

## 4. Owner-gated — not code, and why

| Item | Status | What is needed |
|---|---|---|
| Golden-set corpus backfill | OPEN | ~15% of eval docs absent from the Neon corpus (PR #295 finding). Loading is an operator action over HTTPS bulk-insert; ISP blocks 5432. |
| Client-content fixture binaries (drawing PDFs, P6 baseline `.xer`, ingest-shard sample tree) | CLOSED | Replaced by synthetic files generated inside the tests (`tests/_synthetic_fixtures.py`); the payload suites run on the generated input in every profile, so coverage cannot silently drop to green skips. |
| Discipline-hat system | OPEN | Product decision: it overlaps the live 14-agent system. 32 actions would need building; fenced against accidental enablement meanwhile. |
| `RAG_LAYERED` flag | OPEN (unchanged) | Deferred to client deployment by decision; dormant. |
| `MASTER_CORPUS_PROJECT_ID` live env | **CLOSED 2026-08-12** | Set to `master_corpus` via Render API with operator approval. |

---

## 5. Bottom line

The 08-02 verdict was "code-complete on everything code can close; five
items owner-gated." This pass re-earned that sentence against a harder
standard — control-delete instead of code reading — found the places where
it had been false (unreachable actions, untested LLM core, a monitor crying
wolf, a three-days-red health check), fixed them, and shipped the fixes
through CI to the live service. What remains open is listed above with what
it needs, and none of it blocks putting `main` on a client desk.
