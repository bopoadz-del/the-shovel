#!/usr/bin/env bash
# Post-deploy smoke checks for pilot / staging.
set -euo pipefail

BASE="${1:-http://localhost:8000}"
echo "Pilot smoke against ${BASE}"

curl -sf "${BASE}/health" | grep -q '"status"' && echo "✓ /health"

BLOCKS=$(curl -sf "${BASE}/blocks" -H "Authorization: Bearer $CEREBRUM_DEV_KEY")
echo "${BLOCKS}" | grep -q '"chat"' && echo "✓ /blocks lists chat"

if echo "${BLOCKS}" | grep -q '"construction"'; then
  echo "✓ construction kit loaded"
else
  echo "⚠ construction not in registry (set CEREBRUM_DOMAIN_KITS=construction)"
fi

if [ -n "${DATABASE_URL:-}" ]; then
  python3 -c "
from app.core.db import get_database_url
url = get_database_url()
assert url.startswith('postgresql'), url
print('✓ DATABASE_URL is PostgreSQL')
"
else
  echo "⚠ DATABASE_URL unset (SQLite fallback)"
fi

echo "Pilot smoke complete."
