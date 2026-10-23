#!/bin/bash
# Start script for Cerebrum Blocks

# Use PORT from environment (Render sets this) or default to 8000
PORT=${PORT:-8000}

echo "🚀 Starting Cerebrum Blocks..."
echo "📍 Port: $PORT"
echo "📁 Data Directory: ${DATA_DIR:-/app/data}"

# Create data directory if it doesn't exist
mkdir -p ${DATA_DIR:-/app/data}

# shellcheck source=scripts/uvicorn_worker_count.sh
. "$(dirname "$0")/scripts/uvicorn_worker_count.sh"

UVICORN_ARGS=(--host 0.0.0.0 --port "$PORT" --timeout-keep-alive 65)
echo "👷 uvicorn workers: ${UVICORN_WORKER_COUNT} (UVICORN_WORKERS)"
UVICORN_ARGS+=(--workers "$UVICORN_WORKER_COUNT")

exec uvicorn app.main:app "${UVICORN_ARGS[@]}"
