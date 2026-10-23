#!/usr/bin/env bash
# Local launcher - boots the Cerebrum Blocks API + built-in UI on :8000.
# Usage:
#   ./start-local.sh            # start backend on :8000
set -euo pipefail

cd "$(dirname "$0")"

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'; exit 0 ;;
  esac
done

PORT="${PORT:-8000}"
export DATA_DIR="${DATA_DIR:-$PWD/data}"
export ENV="${ENV:-development}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
mkdir -p "$DATA_DIR"

# Load .env if present (so KIMI_API_KEY / GROQ_API_KEY etc. flow through)
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

echo
echo "Cerebrum Blocks - local mode"
echo "  App:       http://localhost:$PORT/"
echo "  API docs:  http://localhost:$PORT/docs"
echo

# shellcheck source=scripts/uvicorn_worker_count.sh
. "$(dirname "$0")/scripts/uvicorn_worker_count.sh"
echo "  Workers:   ${UVICORN_WORKER_COUNT} (UVICORN_WORKERS)"
echo

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers "$UVICORN_WORKER_COUNT"
