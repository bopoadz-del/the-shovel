#!/usr/bin/env bash
# Back up the on-prem stack: Postgres (the RAG corpus + app DB) + the app data
# volume (uploaded files, encrypted at rest). Writes a timestamped dir. STEP 4.
set -euo pipefail
cd "$(dirname "$0")"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${1:-./backups/$STAMP}"
mkdir -p "$OUT"

# shellcheck disable=SC1091
set -a; . onprem.env; set +a

echo "==> Postgres dump -> $OUT/postgres.sql.gz"
docker exec fork-postgres pg_dump -U "${POSTGRES_USER:-fork}" -d "${POSTGRES_DB:-fork}" \
  | gzip > "$OUT/postgres.sql.gz"

echo "==> App data volume -> $OUT/app_data.tar.gz"
docker run --rm -v onprem_app_data:/data -v "$(pwd)/$OUT":/backup alpine \
  tar czf /backup/app_data.tar.gz -C /data .

# Capture the exact model + image versions so a restore is reproducible.
echo "==> Manifest -> $OUT/manifest.txt"
{
  echo "backup_utc=$STAMP"
  echo "ollama_model=${OLLAMA_MODEL:-unknown}"
  docker image inspect the-fork:onprem --format 'app_image_id={{.Id}}' 2>/dev/null || true
  docker exec fork-ollama ollama list 2>/dev/null || true
} > "$OUT/manifest.txt"

echo "==> Backup complete: $OUT"
ls -la "$OUT"
