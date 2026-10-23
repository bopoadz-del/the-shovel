# Fork-in-a-box — on-prem, air-gapped deployment (STEP 4)

The whole platform as a self-contained stack that runs on your own hardware with
**zero external egress at runtime**: the app, a local Postgres/pgvector store, a
local Ollama LLM server, and bundled models. No cloud, no HuggingFace, no
telemetry.

> **Status:** authored + validated by inspection and unit tests in this session.
> The stack was **not built/run on a Docker host here** (this dev box has no
> Docker/GPU). Build + first-boot happen on the target box. The
> `DEPLOYMENT_PROFILE=onprem` app logic, boot assertion, gates, boot manifest,
> and disk canary are all unit-tested and were exercised in a real Python
> process (see `tests/test_deployment_profile_onprem.py`).

## Contents

| File | Purpose |
|------|---------|
| `docker-compose.yml` | app + postgres(pgvector) + ollama, internal network only |
| `Dockerfile.onprem` | extends the base image: **bakes the embedder + sets offline flags**, self-tests the offline load at build |
| `onprem.env.example` | the leak-free env the app asserts at boot |
| `install.sh` | one-shot bootstrap: secrets, build, pull model, up, health |
| `backup.sh` / `restore.sh` | Postgres + app-data volume backup/restore with a manifest |

## Quick start (on the target box)

```bash
cd deploy/onprem
./install.sh                      # generates secrets, builds, pulls the model, starts
curl http://localhost:8000/health # -> healthy
```

`install.sh` pulls the LLM once (needs network at install time). After that the
running stack makes no external calls — verified in STEP 5's air-gap acceptance.

## The on-prem guarantees (enforced, not just documented)

- **Boot refuses to start if the profile would leak** — cloud LLM provider
  selected, offline flags unset, Sentry on, or cloud-Ollama tunnel key
  (`app/core/deployment_profile.check_onprem_ready`).
- **Offline model flags are baked into the image** (`Dockerfile.onprem` ENV) and
  the image build **self-tests the offline embedder load** — a bad bake fails
  the build, not the customer.
- **Boot manifest** is logged on every on-prem start (profile, LLM wiring,
  embedder model+dim, offline flags, DB backend) so operators see exactly what
  is running.
- **Disk-survival canary** writes/reads a sentinel under `DATA_DIR` to prove the
  persistent volume is mounted and survives restarts.
- **Embedding-identity assertion** (`vector_store._verify_embedding_identity`)
  refuses to operate if the corpus was written by a different embedder — no
  silent mixed-model contamination.

## Resource envelope (per LLM choice)

Guidance for the 24 GB-VRAM x86 box (see `docs/LOCAL_MODEL_DECISION.md`). App +
Postgres are light; the LLM dominates VRAM/RAM.

| Model | ~VRAM (Q4) | Notes |
|-------|-----------|-------|
| qwen2.5:7b-instruct | ~5–6 GB | fast; also the AGX Orin default |
| qwen2.5:14b-instruct | ~9–11 GB | recommended primary on 24 GB |
| qwen2.5:32b-instruct | ~18–20 GB | quality ceiling if latency allows |

App container: ~1.5–2 GB RAM. Postgres: ~0.5–1 GB + corpus on disk. Embedder
(bge-small) is CPU and ~0.2 GB. Give the box 32 GB system RAM to be comfortable.

## Multi-arch (x86_64 now, ARM64 / AGX Orin later)

The compose `app.build.platforms` lists `linux/amd64`; uncomment `linux/arm64`
for the Jetson. Build with buildx:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t the-fork:onprem -f deploy/onprem/Dockerfile.onprem ..
```

Jetson caveat: `deploy/edge/Dockerfile.jetson` currently references a missing
`requirements-edge.txt` — that must be added before an ARM64/Jetson build
succeeds (tracked for the edge track; not needed for the x86 pilot box).

## Air-gap transfer (staging box -> air-gapped box)

When the target box truly has no network, do the network steps on a staging box
and ship the artifacts:

```bash
# On the staging box (has network):
./install.sh                                  # builds images, pulls the model
docker save the-fork:onprem pgvector/pgvector:pg16 ollama/ollama:latest \
  | gzip > fork-images.tar.gz
docker run --rm -v onprem_ollama_models:/m -v "$PWD":/out alpine \
  tar czf /out/ollama-models.tar.gz -C /m .   # the pulled LLM

# Ship fork-images.tar.gz + ollama-models.tar.gz + this deploy/onprem dir.
# On the air-gapped box:
gunzip -c fork-images.tar.gz | docker load
docker volume create onprem_ollama_models
docker run --rm -v onprem_ollama_models:/m -v "$PWD":/in alpine \
  tar xzf /in/ollama-models.tar.gz -C /m
docker compose --env-file onprem.env up -d    # no network needed
```

## Admin basics

- **Local users**: managed by the app's own user store (bootstrap admin via
  `BOOTSTRAP_USER_EMAIL` / `BOOTSTRAP_USER_PASSWORD` in `onprem.env`). No cloud
  identity provider.
- **Backups**: `./backup.sh` (cron it). `./restore.sh <dir>` to recover.
- **Health**: `docker compose ps` + `curl localhost:8000/health`; the boot
  manifest is in `docker compose logs app`.
