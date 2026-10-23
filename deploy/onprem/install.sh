#!/usr/bin/env bash
# Fork-in-a-box installer (STEP 4). Run on the target box (or a staging box with
# network, then ship the images + volumes for air-gap — see README.md "Air-gap
# transfer"). Idempotent: safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${OLLAMA_MODEL:-qwen2.5:14b-instruct}"
COMPOSE="docker compose --env-file onprem.env"

echo "==> Fork-in-a-box install (model: $MODEL)"

# 1. Secrets — generate onprem.env from the template on first run. Never reuse
#    cloud secrets; these are local-only.
if [ ! -f onprem.env ]; then
  echo "==> Generating onprem.env from template with fresh local secrets"
  cp onprem.env.example onprem.env
  gen() { python3 -c "import secrets;print(secrets.token_urlsafe(48))"; }
  {
    echo ""
    echo "# --- generated $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    echo "SECRET_KEY=$(gen)"
    echo "DATA_ENCRYPTION_KEY=$(gen)"
    echo "CEREBRUM_MASTER_KEY=$(gen)"
    echo "POSTGRES_USER=fork"
    echo "POSTGRES_PASSWORD=$(gen)"
    echo "POSTGRES_DB=fork"
    echo "OLLAMA_MODEL=$MODEL"
  } >> onprem.env
  echo "    wrote onprem.env (review it; secrets are generated)"
fi

# 2. Build the app image (base -> onprem, which bakes the embedder + offline flags).
echo "==> Building base image"
docker build -t the-fork:base ../..
echo "==> Building on-prem image (bakes embedder, sets offline flags)"
docker build -t the-fork:onprem --build-arg BASE_IMAGE=the-fork:base \
  -f Dockerfile.onprem ../..

# 3. Bring up postgres + ollama first so we can pull the model into the volume.
echo "==> Starting postgres + ollama"
$COMPOSE up -d postgres ollama
echo "==> Waiting for ollama..."
until docker exec fork-ollama ollama list >/dev/null 2>&1; do sleep 2; done

# 4. Pull the LLM into the ollama volume (network needed HERE; after this the
#    stack runs air-gapped). Skips if already present.
if docker exec fork-ollama ollama list | grep -q "${MODEL%%:*}"; then
  echo "==> Model $MODEL already present"
else
  echo "==> Pulling $MODEL into the ollama volume (one-time, needs network)"
  docker exec fork-ollama ollama pull "$MODEL"
fi

# 5. Start the app.
echo "==> Starting app"
$COMPOSE up -d app

# 6. Health.
echo "==> Waiting for app health..."
for i in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    echo "==> HEALTHY — http://localhost:8000"
    exit 0
  fi
  sleep 5
done
echo "!! app did not become healthy in time; check: $COMPOSE logs app" >&2
exit 1
