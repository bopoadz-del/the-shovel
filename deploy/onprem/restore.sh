#!/usr/bin/env bash
# Restore the on-prem stack from a backup dir produced by backup.sh. STEP 4.
# Usage: ./restore.sh ./backups/<STAMP>
set -euo pipefail
cd "$(dirname "$0")"

SRC="${1:?usage: restore.sh <backup-dir>}"
[ -f "$SRC/postgres.sql.gz" ] || { echo "no postgres.sql.gz in $SRC" >&2; exit 1; }

# shellcheck disable=SC1091
set -a; . onprem.env; set +a
COMPOSE="docker compose --env-file onprem.env"

echo "==> Ensuring postgres is up"
$COMPOSE up -d postgres
until docker exec fork-postgres pg_isready -U "${POSTGRES_USER:-fork}" >/dev/null 2>&1; do sleep 2; done

echo "==> Restoring Postgres from $SRC/postgres.sql.gz"
gunzip -c "$SRC/postgres.sql.gz" \
  | docker exec -i fork-postgres psql -U "${POSTGRES_USER:-fork}" -d "${POSTGRES_DB:-fork}"

if [ -f "$SRC/app_data.tar.gz" ]; then
  echo "==> Restoring app data volume"
  docker run --rm -v onprem_app_data:/data -v "$(pwd)/$SRC":/backup alpine \
    sh -c "rm -rf /data/* && tar xzf /backup/app_data.tar.gz -C /data"
fi

echo "==> Restarting app"
$COMPOSE up -d app
echo "==> Restore complete. Check: curl http://localhost:8000/health"
