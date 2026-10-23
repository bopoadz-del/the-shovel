# Contributing to The Fork

Short guide for anyone (human or AI agent) opening a PR.

## Pull-request template

`.github/PULL_REQUEST_TEMPLATE.md` is loaded automatically when you open a PR. Fill out every section, including **RECEIPT**. PRs with an empty body are rejected by `pr-quality.yml`.

The body shape that worked best in this repo's history is **PR #5** ("Security hardening: IDOR, auth gaps, SSRF, path traversal"). Use it as a reference:

- A short summary at the top.
- A per-finding paragraph for each substantive change, naming the issue, the fix, and the regression test.
- An explicit **"What's NOT in this PR"** list of follow-ups deferred to a later PR.

The campaign that ran across PRs #5–#8 was reviewable because each PR named what it wasn't doing as clearly as what it was. PRs #1, #2, and #3 are the cautionary opposite — large, undocumented, permanently opaque to audit.

## PR sizing

There's no hard limit, but a few soft rules of thumb derived from review experience:

- **Under ~500 LOC of meaningful change** is comfortably reviewable.
- **500–3000 LOC** needs a structured body (per-finding paragraphs, file-by-file rationale).
- **Over 3000 LOC** is a red flag. If the body lists three independent workstreams (as PR #3 did), split into three PRs — each can be reverted independently and each is small enough for a real review.

The CI gate warns at 5000 LOC. If you need to ship past that, label the PR `large-by-design` and explain in the body why splitting doesn't work for this change.

## Test coverage

The CI gate is `--cov-fail-under=25` (regression floor, not a target). It exists to prevent further coverage regression, not to bless the current level.

For new code:

- **New files under `app/core/` should ship with ≥50% line coverage.** These are the platform's load-bearing layer; PR #8 introduced the gate after the security audit found seven endpoints splatting raw `dict` into block `execute()` calls with no coverage.
- **New blocks under `app/blocks/`** need at least the happy path tested. The resilient block loader (PR #8) means a broken block load is non-fatal at app startup, but it's still a hidden bug.
- **Tests for new code should use the `isolated_data_dir` fixture pattern** established by the hydration work — fresh `DATA_DIR` per test, module-level `_initialized` flags reset. Don't write tests that share state via `/tmp` or the live `data/` directory.

## Block output contracts

Blocks compose into chains via `OrchestratorBlock`. A chain step that produces a JSON dict (e.g. `translate` returning `{"translated": "...", ...}`) automatically gets its primary text unwrapped before flowing into a text-expecting next step (`chat`, `translate -> chat`, etc.).

The unwrap order is:

1. **Producing block's declared `text_output_field`** (class attribute on `UniversalBlock`). If set, this key wins.
2. **Global priority-ordered fallback list** in `app/blocks/orchestrator.py:_TEXT_OUTPUT_FIELDS` (`text`, `translated`, `response`, ...).
3. **Single-string heuristic**: if exactly one value in the dict is a non-empty string, return it.

If your block's canonical text lives under a key that's NOT in the global list, declare it:

```python
class MyBlock(UniversalBlock):
    name = "my_block"
    text_output_field = "my_canonical_key"  # add this
```

Even if the key IS in the global list (e.g. `translated`), declaring it explicitly is preferred — it locks the contract and survives reordering of the global tuple.

Test the override in `tests/test_chain_text_output_field.py` and the legacy global-list path in `tests/test_chain_json_text_coercion.py`.

## Security follow-ups

`docs/SECURITY_TRIAGE.md` captures the CodeQL dismissal rationales from PRs #11/#12/#14. Read it before re-triaging a CodeQL re-scan — many alerts are already-adjudicated false positives.

## Direct pushes to main

By default, all changes land via PRs. The repo owner has opted into allowing direct pushes to `main` in personal Claude Code sessions (see `.claude/settings.local.json`), but the team norm is still PR-first — direct pushes bypass CI, CodeQL, and any review.

`main` auto-deploys only after GitHub CI checks pass (`render.yaml`:
`autoDeployTrigger: checksPass` — Render "After CI Checks Pass"). A red
required check (`test-postgres`, `pip-audit`, `test (virgin)`,
`test (production-like)`, `integrity`, `eslint`, `ruff`,
`compose up (fresh clone) + health`) skips the deploy. Manual Dashboard
or API deploys still bypass the gate.

This blueprint field does **not** take effect from a git push alone.
Until the live `the-fork` service Auto-Deploy is set to
**After CI Checks Pass** (Dashboard → Settings — one toggle; do **not**
Apply Blueprint just to flip this), a red main SHA can still go live.

## Pre-deploy smoke

`main` auto-deploys after CI is green. After that deploy (or after a
manual deploy), run smoke for any change that touches `app/agents/runtime.py`, the `app/agents/` chat routes, or the frontend chat path (`frontend/src/pages/ProjectWorkspace.tsx`). The chat/deliverable path has a failure mode that unit tests don't catch: a contaminated message replayed to the provider 400s on every *tool-calling* turn, which the browser renders as a silent hang (see the reasoning-field bug, commits `d3cb9fc`/`dc2c2d1`). The tripwire for that whole class of bug is a live smoke run.

**Run the smoke before AND after any deploy that touches `app/agents/runtime.py`, the `app/agents/` chat routes, or the frontend chat path (`frontend/src/pages/ProjectWorkspace.tsx`).**

```powershell
# Windows / PowerShell
$env:FORK_API_KEY = "<master key>"     # env only — never commit a key
./scripts/smoke.ps1
```

```bash
# bash
export FORK_API_KEY="<master key>"
./scripts/smoke.sh
```

It runs `scripts/fork_cli.py` N times with a deliverable (tool-calling) prompt and prints a per-run summary line (including the **served model**, so a silent provider fallback is visible) plus a verdict. **PASS** requires every run to succeed — `answer_chars >= MIN_ANSWER_CHARS` (default 1500, so a degenerate short answer FAILs, not just an empty one) — AND at least `max(1, floor(N*0.3))` runs to exercise a `tool_call`/`tool_result` pair. Otherwise it **FAIL**s with a nonzero exit code. Zero-tool runs prove nothing — the tool path is what breaks.

- **Routine (default): `-Runs 3`** — the default. Keeps LLM-provider quota use light; heavy 10-run smokes before *and* after every deploy can exhaust a metered tier and mask real failures behind fallback.
- **Release: `-Runs 10`** (`SMOKE_RUNS=10` for `smoke.sh`) — the full discriminator, e.g. after a model/provider migration.

Point it at a branch preview or a staging instance with `$env:FORK_BASE_URL` before merging when you can; at minimum, run it against prod immediately after the deploy and be ready to roll back. If a run shows a model you didn't expect (a fallback), the smoke's PASS is weaker than it looks — investigate before trusting it.

## Fixture projects

Pilot test fixtures are **disposable and rebuildable as code**. Do not treat fixture project ids as stable constants — they change whenever the fixture is recreated.

- Canonical fixture names (resolved at runtime):
  - `FIXTURE — Fresh Upload Eval`
  - `FIXTURE — BOQ`
  - `FIXTURE — Programme+Drawings`
- Seeder: `python scripts/seed_fixtures.py`
  - Self-contained for `FIXTURE — Fresh Upload Eval` (uses the 12 CASES texts).
  - Dir-based fixtures (`FIXTURE — BOQ`, `FIXTURE — Programme+Drawings`) require `FIXTURES_DIR` to point at the files the owner provides.
- Harnesses resolve fixtures by name via the projects API. Hardcoded fixture ids in `tests/feature_matrix_manifest.yaml` or `scripts/rag_fresh_upload_eval.py` are a defect.
- Deleting a fixture project from the UI is safe: re-run `seed_fixtures.py` to recreate it.

Run `seed_fixtures.py` before any feature-matrix sweep; the sweep runner will BLOCK features whose fixture project is missing instead of failing with a 404.

## Trivial PRs

Single-typo fixes, dependency bumps, or label changes can skip the PR template by adding the `trivial` label. The `pr-quality.yml` gate honors it.
