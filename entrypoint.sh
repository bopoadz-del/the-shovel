#!/bin/bash
echo "🚀 Cerebrum Blocks starting on $(uname -m) architecture"

# Hardware detection (future-proof for local OCR/voice/GPU blocks)
if command -v nvidia-smi &> /dev/null; then
    export HARDWARE="gpu"
    echo "✅ NVIDIA GPU detected"
elif [ -d "/proc/device-tree" ] && grep -q "nvidia" /proc/device-tree/model 2>/dev/null; then
    export HARDWARE="jetson"
    echo "✅ NVIDIA Jetson detected"
else
    export HARDWARE="cpu"
    echo "ℹ️  Running on CPU"
fi

# Render / cloud compatibility ($PORT)
if [ -n "$PORT" ]; then
    PORT=${PORT}
else
    PORT=8000
fi

echo "📡 Listening on 0.0.0.0:$PORT (HARDWARE=$HARDWARE)"

if [ -n "${DATABASE_URL:-}" ] && [[ "${DATABASE_URL}" == postgresql* ]]; then
  echo "🗄️  DATABASE_URL set — running alembic upgrade head"
  if ! python -m alembic upgrade head; then
    echo "❌ alembic upgrade failed — refusing to start with a stale schema"
    exit 1
  fi
fi

# Background-worker override: run the supplied long-lived command instead of uvicorn.
if [ -n "${WORKER_COMMAND:-}" ]; then
  echo "🔧 WORKER_COMMAND set — executing: $WORKER_COMMAND"
  # shellcheck disable=SC2086
  exec sh -c "$WORKER_COMMAND"
fi

# shellcheck source=scripts/uvicorn_worker_count.sh
. "$(dirname "$0")/scripts/uvicorn_worker_count.sh"

UVICORN_ARGS=(--host 0.0.0.0 --port "$PORT" --no-access-log --timeout-keep-alive 65)
echo "👷 uvicorn workers: ${UVICORN_WORKER_COUNT} (UVICORN_WORKERS)"
UVICORN_ARGS+=(--workers "$UVICORN_WORKER_COUNT")

exec uvicorn app.main:app "${UVICORN_ARGS[@]}"
