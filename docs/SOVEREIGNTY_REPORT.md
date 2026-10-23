> **SUPERSEDED (2026-09-28).** Historical snapshot. Not a source of open items or current status. Current state lives in the fleet standing-order loop and the live measurements.

# SOVEREIGNTY REPORT — The Fork on-prem / air-gap track

Completion deliverable for the on-prem sovereignty program (owner directive
2026-07-17). The pilot deploys **fully on our own hardware**; the cloud
(Render / theshovel.ai) stays an untouched dev/demo profile. This report is the
single source of truth for what was **executed and verified** vs **prepared for
the target hardware**, so nothing here implies evidence that wasn't produced.

## Executed vs Prepared (read this first)

| Step | Deliverable | Status |
|------|-------------|--------|
| 0 | Retrieval isolation | **EXECUTED — shipped to prod + verified live** (PR #247) |
| 1 | EGRESS_LEDGER.md | **EXECUTED** — full read-only audit |
| 2 | DEPLOYMENT_PROFILE=onprem | **EXECUTED — shipped** (PR #248) + booted in a real process |
| 3 | LOCAL_MODEL_DECISION.md | **PARTIAL** — 8B-class referees executed on this box; 14B/30B prepared (need 24 GB GPU) |
| 4 | Fork-in-a-box | **PREPARED** — compose/scripts/Dockerfile authored + validated by inspection; **not built** (no Docker/GPU on this box) |
| 5 | Air-gap acceptance | **STEP 5-lite EXECUTED here** (local Ollama + SQLite); full air-gap run **prepared** for the egress-blocked target box |

The hardware honesty: this session ran on a Windows dev PC — local Ollama
(CPU/iGPU) with `qwen2.5:3b`/`7b`, no Docker, no 24 GB GPU, no separate
air-gapped box. Everything runnable here was run; the rest is a ready-to-run
harness + methodology for the owner's target box.

## STEP 0 — Retrieval isolation (SHIPPED + VERIFIED LIVE)

Full write-up: `docs/step0-retrieval-isolation.md`.

Root cause was not what the directive assumed: `store.search(project_id)` was
**already** a hard SQL `WHERE project_id = X`. The leak was the GK merge — the
live env declared the the client project client corpus (`drive_archive` + `example_infra_pack`)
as "general knowledge," silently blended into every project via ranking knobs.

Fix: two layers — general knowledge (`curated_kb`) always-eligible-but-disclosed;
the client/Master corpus structurally barred from the merge, surfacing only as
the labeled empty/thin fallback. `Chunk.layer` threads retriever → audit
(`fallback_used`) → answer banner + sources `layer_label`.

Live verification (post-deploy):
- `gk_project_ids = ['curated_kb']` only (drive_archive stripped by code, client by config).
- Strong project (curated_kb + FIDIC query): 8/8 own chunks, **zero** drive_archive.
- Empty project: answer banner "This project has no documents of its own… answering from the Master Corpus" + every source tagged "Master Corpus (fallback)".
- Tests: `tests/test_step0_retrieval_isolation.py` (golden + regression + disclosure); full RAG suite 138 green.

## STEP 1 — Egress ledger (EXECUTED)

Full ledger: `docs/EGRESS_LEDGER.md`. Every outbound call catalogued with call
site, trigger, and on-prem disposition. Air-gap headlines:
1. **RAG embedder auto-downloads from HuggingFace on every boot** (no offline flags anywhere) — the #1 blocker, closed in STEP 4's baked image.
2. **Cloud LLM ladder** (Groq/OpenAI/DeepSeek/Kimi) — closed by forcing Ollama in STEP 2.
Other egress (Sentry, Open-Meteo, Drive/OneDrive/R2, web/search/translate/webhook, MCP, Google Fonts, build-time PyPI/pytorch) each has a disposition.

## STEP 2 — DEPLOYMENT_PROFILE=onprem (SHIPPED)

Full write-up: `docs/onprem-profile.md`. One env switch; cloud path byte-for-byte
unchanged (merge-safe). Fail-loud boot assertion (refuses to start on a cloud LLM
provider, unset offline flags, Sentry on, or cloud-Ollama tunnel) +
per-disposition honest-unavailable gates (translate/web/search/onedrive/drive/
webhook/mcp/weather). Offline flags asserted, never set from Python. Tests:
`tests/test_deployment_profile_onprem.py` (20).

## STEP 3 — Local model decision (PARTIAL — EXECUTED where hardware allowed)

Full write-up + methodology: `docs/LOCAL_MODEL_DECISION.md`. Harness:
`scripts/bench_local_model.py`.

Executed on this box (local Ollama, CPU), `--runs 5`:

| Model | tool-call reliability | grounding | avg latency (CPU) |
|-------|----------------------|-----------|-------------------|
| qwen2.5:3b-instruct | 15/15 = 100% | 5/5 = 100% | 5.9 s |
| qwen2.5:7b-instruct | 15/15 = 100% | 5/5 = 100% | 50.9 s |

Key finding: the make-or-break referee (tool-call reliability through the agent
loop) is **100% even at 3B** — local tool-calling is not the risk it is for some
open models. The decision therefore hinges on answer quality + GPU latency, not
capability. Provisional recommendation: **qwen2.5:14b-instruct** primary on the
24 GB box, 7B as edge/Orin fallback, 32B as the quality ceiling if latency
allows. R3/R5/R6 + GPU latency (R7) are prepared to run on the target box.

## STEP 4 — Fork-in-a-box (PREPARED)

`deploy/onprem/`: `docker-compose.yml` (app + pgvector + Ollama, internal network
only), `Dockerfile.onprem` (bakes the embedder + sets offline flags + self-tests
the offline load at build), `install.sh` / `backup.sh` / `restore.sh`, `README.md`
(resource envelope per model, multi-arch x86/ARM64, air-gap transfer procedure),
`onprem.env.example`. Boot manifest + disk-survival canary + embedding-identity
assertion wired into the app. Compose YAML + shell syntax validated; **not built**
here (no Docker). Build + first boot happen on the target box via `install.sh`.

## STEP 5 — Air-gap acceptance

### STEP 5-lite — EXECUTED on this machine

The onprem app was booted in a real uvicorn process (local Ollama + SQLite +
offline flags + `RAG_EMBEDDING_MODEL=fake` to stand in for the baked embedder):

| Check | Result |
|-------|--------|
| Boots under onprem (leaky config would raise) | **PASS** — healthy in ~10 s, 39 blocks |
| Boot manifest surfaces | **PASS** — `profile=onprem, llm=ollama@localhost, embedder=fake/256, offline flags=true, sentry=false, db=sqlite` |
| Disk-survival canary | **PASS** — seeded on boot 1; boot 2 reported "volume persisted" with the same first_seen timestamp |
| Chat via LOCAL Ollama | **PASS** — answered via `qwen2.5:3b-instruct` (model confirmed in the END event), no cloud |
| **Real deliverable via natural chat** (7B) | **PASS** — "Generate a commissioning checklist for the HVAC system" routed via the LOCAL 7B intent classifier to `action=commissioning_checklist`, `mode=predefined`, and produced a **real structured deliverable** through the app (`commissioning_checklist_generated`: 24 tests, 5 systems, real test items) — not prose. Closes the owed caveat. |
| Egress snapshot during operation | **INCONCLUSIVE** — no non-loopback connections seen, but the Windows `netstat` filter matches PIDs not process names, so a null result can't distinguish "clean" from "filter missed." Real verification is the target-box tcpdump / firewall deny-log. |
| Live gate `/v1/drive/connect` | **PASS** — HTTP 503 under onprem |

Caveats:
- `fake` embedder stands in for the baked `bge-small` (the dev box has no bundled
  weights); the real offline embedder load is self-tested at image build in
  `Dockerfile.onprem`. CPU latency (chat ~70 s) is not representative of the GPU box.
- **On the owed "deliverable through the loop" caveat:** now closed via the
  **predefined orchestrator path** (`ORCHESTRATOR_PREDEFINED=1`, the prod
  deliverable path) — the local 7B did the intent classification and the app
  produced a real deterministic deliverable. Precise scope: this is NOT the raw
  `tool_choice` tool-agent loop — at runtime the chat agent (`project-assistant`)
  exposes **no** `tool_choice` tools; deliverables in prod flow through the
  predefined orchestrator, which is exactly what was exercised. Separately,
  `scripts/bench_local_model.py` proves the model tool-calls at 100% against
  Ollama's native tools API. The target-box run should confirm the same
  deliverable path at **14B** and grade output quality (R3/R5).

### Full air-gap acceptance — PREPARED (needs the egress-blocked target box)

Procedure for the target box:
1. `deploy/onprem/install.sh` on a staging box (network) → images + model pulled; transfer via `docker save` + volume export (README "Air-gap transfer").
2. Physically disconnect / firewall-block the target box.
3. `docker compose --env-file onprem.env up -d`; confirm the boot manifest + `/health`.
4. Ingest a fixture project (upload path), run the full golden set + feature matrix (`scripts/golden_set_gate.py`, `scripts/feature_matrix_sweep.py`) against the local instance.
5. PASS = all green **and** zero external connection attempts in an egress log (e.g. `tcpdump`/host firewall deny-log) for the duration.

## Residual items / for the owner

- **STEP 0 reconciliation (vetoable):** GK stays disclosed-eligible, not fallback-only. To make it fallback-only: `RAG_GENERAL_KNOWLEDGE_PROJECTS=""`.
- **Data hygiene:** `curated_kb` contains project-specific EW-2 files — prune (same leak class, minor).
- **Deferred egress gates (STEP 2 scope calls):** pdf/ocr/image URL-fetch fail naturally offline (not wholesale-gated to keep local processing); frontend Google Fonts self-hosting belongs to the STEP 4 image build.
- **Jetson/ARM64:** `deploy/edge/Dockerfile.jetson` references a missing `requirements-edge.txt` — add before an ARM64 build.
- **The bench numbers that decide the model** (R3/R5/R6/R7) need the 24 GB GPU box; harness is ready.

Cloud dev/demo remained running and untouched throughout.
