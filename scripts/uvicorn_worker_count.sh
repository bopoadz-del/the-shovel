# shellcheck shell=bash
# Sole uvicorn worker-count knob: UVICORN_WORKERS (default 1).
# Source from entrypoint.sh, start.sh, start-local.sh — do not branch on REDIS_URL etc.
UVICORN_WORKER_COUNT="${UVICORN_WORKERS:-1}"
