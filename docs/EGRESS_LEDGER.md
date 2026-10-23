# EGRESS LEDGER — The Fork

STEP 1 of the on-prem sovereignty program (owner directive 2026-07-17). Read-only
audit: **every** call the app makes to something outside the host, its call site,
its trigger, and its on-prem disposition. This is the checklist STEP 2
(`DEPLOYMENT_PROFILE=onprem`) implements and STEP 5 (air-gap acceptance) verifies.

Scope: the live tree (`app/`, `scripts/`, `frontend/`, Docker/build). `.venv/`,
`node_modules/`, and worktree copies excluded. Trigger legend: **Boot** =
import/startup, **Request** = per HTTP request, **Build** = image build only,
**Script** = manual CLI/eval only (not the running server).

## Headline for air-gap

Two things break a **running** air-gapped server on day one, both because **no
offline flags exist anywhere** (`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` /
ultralytics offline are unset in `.env.example`, app config, and every Dockerfile):

1. **RAG embedder auto-downloads from HuggingFace on every boot** (E-M1) — warm-loaded in `main.py` lifespan. Fail-closed with no network.
2. **LLM egress to cloud providers** (E-L1) — the default provider ladder is Groq/OpenAI/DeepSeek/Kimi. On-prem must pin `LLM_PROVIDER=ollama` + local `OLLAMA_URL`.

Everything else is either build-time (vendor it), a disable-able cloud connector, a user-triggered arbitrary-URL tool, or Google Fonts in the frontend.

---

## 1. LLM providers

Central config: `app/agents/runtime.py:1606–1684` (base-URL constants + `_llm_config`), precedence `LLM_PROVIDER` → GROQ → OPENAI → DeepSeek; fallback via `LLM_FALLBACK_PROVIDER` (`:1721`). Same resolution reused by `app/core/llm_client.py`, `app/blocks/chat.py`, `app/blocks/formula_executor_v2.py`, `app/blocks/project_reasoner.py`.

| ID | Destination | Env | Call sites | Trigger | On-prem disposition |
|----|-------------|-----|-----------|---------|---------------------|
| E-L1 | `api.groq.com`, `api.openai.com`, `api.deepseek.com`, `api.moonshot.ai` | `LLM_PROVIDER`, `*_API_KEY`, `*_MODEL` | runtime.py:3719/3752/3947/3977; llm_client.py:61; chat.py:355/413; formula_executor_v2.py:157; project_reasoner.py:155 | Request | **Force `LLM_PROVIDER=ollama` + `LLM_FALLBACK_PROVIDER=ollama`.** Cloud provider constants stay in code but are unreachable under the profile; the profile refuses to start if a cloud provider is selected. |
| E-L2 | Ollama — `http://localhost:11434` (default) **or** `OLLAMA_URL` cloud (`ollama.com`/tunnel) + `OLLAMA_API_KEY` | `OLLAMA_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL` | runtime.py:1670, chat.py:522/523 | Request | **Keep — this is the on-prem path.** Pin `OLLAMA_URL=http://localhost:11434` (or the in-compose ollama service), and assert `OLLAMA_API_KEY` is empty so no cloud tunnel is used. |

## 2. Model / weight downloads (HuggingFace / ultralytics)

| ID | Model | Call site | Trigger | On-prem disposition |
|----|-------|-----------|---------|---------------------|
| E-M1 | Embedder — `BAAI/bge-small-en-v1.5` (prod) / `minishlab/potion-base-8M` default | `app/core/rag/embeddings.py:98,103`; warm-loaded `app/main.py:186-188`; also `seed_knowledge()` | **Boot** | **BLOCKER.** Bundle weights into the image (`data/models/hf/…`), point `RAG_EMBEDDING_MODEL` at the local path, set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`. |
| ~~E-M2~~ | ~~zvec — `minishlab/potion-base-8M`, `all-MiniLM-L6-v2` (hardcoded)~~ | ~~`app/blocks/zvec.py:52,60`~~ | — | **RETIRED.** The zvec and vector_search blocks were deleted (in-memory TF-IDF, superseded by the pgvector path); this egress source no longer exists. ID kept, not reused, so the audit trail stays continuous. |
| E-M3 | Image block YOLO COCO — `yolov8n.pt` (auto-downloads from ultralytics assets) | `app/blocks/image.py:143-152` | Request (first `detect_objects`/`analyze`) | Bundle `yolov8n.pt` + `YOLO_MODEL=/app/models/…`, or set `YOLO_OFFLINE=1` and accept the COCO tier no-ops. |
| E-M4 | Safety Observation AI — `safety_world_v2.onnx` | `app/blocks/safety_world_detector.py:50`; warm-loaded `main.py:167-171` | Boot | **Already bundled** (committed, copied `Dockerfile:106`). Add `YOLO_OFFLINE=1` to suppress ultralytics version/settings check. |
| E-M5 | Local fine-tuned chat — base `Qwen/Qwen2.5-3B-Instruct` + local LoRA adapter | `app/core/learning/local_model.py:180-186` | Request (only if `use_local_model=true`) | Off by default; if enabled on-prem bundle the base weights + offline flags. |
| E-M6 | ultralytics telemetry (anonymous, on YOLO use) | ultralytics library default | Request | Set `YOLO_OFFLINE=1` / `yolo settings sync=False`. |

## 3. Observability

| ID | Destination | Env | Call sites | Trigger | On-prem disposition |
|----|-------------|-----|-----------|---------|---------------------|
| E-S1 | Sentry ingest (DSN host) | `SENTRY_DSN` | `app/infra/monitoring.py:108` (init, via `main.py:40` at boot); capture at monitoring.py:186, main.py:307/355/381, admin.py:757 | Boot + Request | **Leave `SENTRY_DSN` unset** → `init_sentry()` no-ops. Route errors to local structured logging (STEP 2). |

## 4. Weather

| ID | Destination | Call site | Trigger | On-prem disposition |
|----|-------------|-----------|---------|---------------------|
| E-W1 | `geocoding-api.open-meteo.com`, `api.open-meteo.com`, `archive-api.open-meteo.com` (keyless) | `app/containers/construction/__init__.py:775` (`_fetch_weather` :796/:808) | Request (construction tool) | Return an **honest "weather unavailable offline"** under the profile instead of attempting the fetch (which would hang until timeout). |

## 5. Cloud storage / ingestion connectors (all disable-able)

| ID | Destination | Env | Call sites | Trigger | On-prem disposition |
|----|-------------|-----|-----------|---------|---------------------|
| E-C1 | Google Drive + OAuth — `accounts.google.com`, `oauth2.googleapis.com`, `www.googleapis.com/drive/v3` | `GOOGLE_CLIENT_ID/SECRET/REDIRECT_URI/ACCESS_TOKEN` | drive.py:71/88/466; drive_auth.py:86; gdrive_service.py:134/188/322; google_drive.py:24/128/175 | Request | **Disable cloud Drive** under the profile; ingestion via **upload / local_drive only**, with an honest "cloud Drive not available on-prem" message. |
| E-C2 | Microsoft Graph / OneDrive — `graph.microsoft.com`, `login.microsoftonline.com` | `AZURE_CLIENT_ID/TENANT_ID/REDIRECT_URI/CLIENT_SECRET` | onedrive.py:106/143 | Request | Disable under the profile (same as Drive). |
| E-C3 | Cloudflare R2 object storage (S3-compatible) | `R2_ENDPOINT_URL`, `R2_*` keys | `app/core/r2_storage.py` (boto3/S3 client) — **present but unwired: no `app/` module imports it; dormant** | Request (only if wired + configured) | Leave R2 env unset → local disk storage under `DATA_DIR`; profile asserts R2 is off. |

## 6. User-triggered arbitrary-URL tools (egress vectors, not fixed hosts)

These fetch a **user-supplied URL** on request. Offline they fail/timeout rather than leak, but the profile should make them **honest-unavailable** (or SSRF-gate to loopback) so they don't hang.

| ID | Call sites | Purpose |
|----|-----------|---------|
| E-U1 | web.py:105/116 (SSRF-guarded `validate_public_url`) | Fetch/extract a web page |
| E-U2 | webhook.py:110 (HMAC-signed) | POST to a user-configured webhook |
| E-U3 | image.py:40-41; pdf.py:79; pdf_v2.py:85; ocr.py:89; ocr_v2.py:83; construction/__init__.py:136 | Download a user-supplied file URL to temp |
| E-U4 | search.py:40/41 — DuckDuckGo (`ddgs`), Serper (`google.serper.dev`) | Web search |
| E-U5 | translate.py:70 — `translate.googleapis.com` (keyless GTX) | Text translation |
| E-U6 | mcp_consumer.py:7 — spawns `npx -y @modelcontextprotocol/server-*` | External MCP servers make their own egress |

**On-prem disposition:** gate E-U1–E-U6 behind a profile flag that returns "not available in the on-prem profile." E-U6 (MCP consumer) must be **off** — it spawns external processes that phone home.

## 7. Frontend (browser runtime egress)

| ID | Destination | Call sites | On-prem disposition |
|----|-------------|-----------|---------------------|
| E-F1 | Google Fonts — `fonts.googleapis.com`, `fonts.gstatic.com` | `frontend/index.html:8-13`, `frontend/dist/index.html:8-13`, `frontend/src/theme/tokens.css:4` (survives into `dist/assets/*.css`) | **Self-host** the four families (Albert Sans, IBM Plex Sans, IBM Plex Mono, Inter) as local `@font-face`; remove the `<link>`/`preconnect`/`@import`. Without network the SPA still runs (system-font fallback), but self-hosting removes the only frontend egress. |

Everything else in the SPA is bundled (no CDN JS/CSS, no analytics/telemetry, no maps/iframes; `VITE_API_BASE=""` → same-origin).

## 8. Build-time fetches (vendor for an air-gapped build box; not a runtime concern)

`Dockerfile` + `render-build.sh` + `deploy/*`: PyPI (`requirements*.txt`), `download.pytorch.org/whl/cpu` (torch/torchvision/ultralytics/onnxruntime), `npm ci`, `apt-get`, and base images (`python:3.11-slim`, `node:20-slim` from Docker Hub, each **pinned by `@sha256` digest** in the Dockerfile since 2026-09-30 so a re-pointed tag cannot change the OS under a build; `pgvector/pgvector:pg16`; Jetson `nvcr.io/nvidia/l4t-pytorch`). `entrypoint.sh` does **no** start-time downloads (hardware probe + `alembic upgrade` + uvicorn only). Air-gapped builds need a local PyPI/npm/apt mirror + a pinned registry. (Note: `deploy/edge/Dockerfile.jetson` references a missing `requirements-edge.txt` — broken regardless; fix in STEP 4.)

## 9. Scripts (manual only — not the running server)

CLI/eval scripts default `FORK_BASE_URL=https://the-fork-jn3t.onrender.com` (the cloud dev/demo app) and hit it over httpx/urllib; some hit Ollama at `localhost:11434`, Google Drive directly, or the Render API (`api.render.com`). These never run at boot; for on-prem, point `FORK_BASE_URL` at the local instance and use the localhost-Ollama eval scripts. Not part of the deployed egress surface.

---

## Disposition summary → STEP 2 profile

Under `DEPLOYMENT_PROFILE=onprem` the app must, from a single env-selected switch:
- **LLM** (E-L1/L2): force Ollama; refuse to boot on a cloud provider.
- **Embeddings/models** (E-M1/M2/M3/M6): bundled local weights + `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `YOLO_OFFLINE=1`; E-M4 already bundled; E-M5 off.
- **Sentry** (E-S1): off → local structured logging.
- **Weather** (E-W1): honest-unavailable.
- **Connectors** (E-C1/C2/C3): cloud Drive/OneDrive/R2 disabled; ingestion via upload/local_drive only.
- **Arbitrary-URL tools + MCP** (E-U1–U6): gated to honest-unavailable; MCP consumer off.
- **Frontend** (E-F1): self-hosted fonts.

Each disposition gets a test in STEP 2 and is re-verified with a network egress monitor in STEP 5.
