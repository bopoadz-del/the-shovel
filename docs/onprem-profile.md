# DEPLOYMENT_PROFILE=onprem — the zero-egress profile (STEP 2)

STEP 2 of the on-prem sovereignty program. A single env switch,
`DEPLOYMENT_PROFILE`, selects how the app treats external services:

- **`cloud`** (default) — today's behaviour, byte-for-byte. The Render
  dev/demo. Every gate below is a strict no-op.
- **`onprem`** — zero-egress. Every would-egress feature is honest-unavailable,
  and the app **refuses to boot** on a config that would leak.

Implemented in `app/core/deployment_profile.py`; enforced at boot in
`app/main.py` lifespan (`assert_onprem_ready()`).

## Design boundary (deliberate)

**Network isolation itself is the deployment's job**, not the app's. The
air-gapped host has no route to the internet; STEP 5 verifies that with an
egress monitor. This profile's job is narrower: (a) don't *need* egress, and
(b) fail *gracefully* when a cloud-only feature is invoked. It does **not**
install an in-process socket blocker — that would risk breaking local Postgres
/ local Ollama and is the wrong layer (fail-closed blast radius).

## Boot assertion (fail-loud)

Under `onprem`, `check_onprem_ready()` returns — and `assert_onprem_ready()`
raises `RuntimeError` on — any of:

- `LLM_PROVIDER` unset or a cloud provider (groq/openai/deepseek/kimi/anthropic) → must be `ollama`.
- `LLM_FALLBACK_PROVIDER` a cloud provider.
- `OLLAMA_API_KEY` set (that means Ollama **Cloud** — egress).
- `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` not truthy.
- `SENTRY_DSN` set (telemetry egress).

**Offline model flags are asserted, never set from Python** — `transformers`
reads them at import time, so a late `os.environ[...] = "1"` is a silent no-op.
They live in the deployment env (`deploy/onprem/onprem.env.example`,
Dockerfile ENV, compose). The assertion guarantees they are present.

## Gate inventory (mapped to EGRESS_LEDGER)

| Ledger | Feature | Enforcement | Behaviour under onprem |
|--------|---------|-------------|------------------------|
| E-L1/L3 | Cloud LLM providers | boot assertion | refuse to boot |
| E-S1 | Sentry | boot assertion (DSN must be unset) | no telemetry |
| E-M1/M2/M3/M6 | HF/YOLO model downloads | boot assertion (offline flags) + STEP 4 bundling | offline load only |
| E-W1 | Weather (Open-Meteo) | `construction._fetch_weather` gate | honest "weather offline" |
| E-C1 | Google Drive (block + `/v1/drive/*` router) | block gate + `forbid_onprem` dep on connect/files/index-folder/import | 503 / unavailable |
| E-C2 | OneDrive | block gate | unavailable |
| E-C3 | R2 storage | unwired + env-off | local disk |
| E-U1 | Web fetch | block gate | unavailable |
| E-U4 | Web search (DuckDuckGo/Serper) | block gate | unavailable |
| E-U5 | Translate (Google GTX) | block gate | unavailable |
| E-U2 | Outbound webhook (send/trigger) | block gate (register/list stay local) | unavailable |
| E-U6 | MCP consumer | block gate | unavailable |

Each gate returns the standard `onprem_unavailable(feature)` payload
(`status=error`, `onprem_unavailable=True`) or, for the router, HTTP 503.

## Scope decisions (flagged, deferred — for the owner)

Two ledger items are **deliberately not wholesale-gated in STEP 2**, to avoid
collateral damage; each is handled elsewhere:

- **E-U3 — URL-download inside pdf/ocr/image blocks.** These blocks primarily
  process **uploaded** files locally (a keep feature). They only egress when a
  user passes an `http(s)` URL instead of a file — a niche path. Offline that
  fails with a natural network error (the block returns `status=error`).
  Wholesale-gating them would break local document processing, so it is not
  done. If a per-URL honest-unavailable is wanted, add a loopback check in each
  block's download helper (small follow-up).
- **E-F1 — Google Fonts in the frontend.** `index.html` + `tokens.css` load
  fonts from `fonts.googleapis.com`. The SPA runs fine offline (system-font
  fallback). Self-hosting the fonts belongs to the **STEP 4 on-prem image
  build** (swap fonts at build) — ripping them out here would also change the
  shared cloud-demo build's typography.

## Running on-prem locally

1. Copy `deploy/onprem/onprem.env.example` → `onprem.env`, fill secrets.
2. Ensure local Ollama is up with the target model
   (`ollama pull qwen2.5:7b-instruct`).
3. Bundle/pre-seed the embedder weights (STEP 4) or, for a smoke, set
   `RAG_EMBEDDING_MODEL=fake`.
4. `set -a; . onprem.env; set +a; uvicorn app.main:app` — the boot assertion
   confirms the profile is leak-free before serving.

## Tests

`tests/test_deployment_profile_onprem.py` (18): profile switch, boot assertion
(pass on valid onprem, raise on cloud provider / missing offline flags / Sentry
/ cloud tunnel), every block gate returns unavailable under onprem, and
**merge-safety** — under the default cloud profile every gate is inert (no
`onprem_unavailable`), so this ships to the cloud dev/demo unchanged.
