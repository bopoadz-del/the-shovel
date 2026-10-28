"""New-domain-candidate log — S-7 improvisation harness (ORDERS_28OCT_0815Z).

When a construction calculation is NOT covered by any kernel, the agent
improvises an answer with an explicit label (never a silent invented number),
and each such uncovered ask appends one row here. The log is TENANT-SCOPED
JSONL at ``DATA_DIR/new_domain_candidates/<tenant>.jsonl``, append-only, read by
the Store later to decide which new domains deserve a real kernel.

It records only what happened — the question, the tenant, the improvisation
sources — and never fabricates. A write failure is logged, never raised: the
improvisation answer must not depend on the ledger succeeding.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")  # no '.', so no '..' path traversal


def _tenant_slug(tenant_id: Optional[str]) -> str:
    """A filesystem-safe, bounded slug for the tenant id (never empty)."""
    slug = _UNSAFE.sub("_", (tenant_id or "").strip()).strip("_")
    return (slug or "unknown")[:120]


def _log_path(tenant_id: Optional[str]) -> Path:
    root = Path(os.getenv("DATA_DIR", "./data")) / "new_domain_candidates"
    return root / f"{_tenant_slug(tenant_id)}.jsonl"


def log_new_domain_candidate(
    tenant_id: Optional[str],
    question: str,
    *,
    agent: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Append one uncovered-calculation row for ``tenant_id`` and return it."""
    row: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tenant": _tenant_slug(tenant_id),
        "question": (question or "").strip()[:2000],
        "agent": agent or None,
        "sources": list(sources or []),
    }
    path = _log_path(tenant_id)
    line = json.dumps(row, default=str, ensure_ascii=False) + "\n"
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except OSError:
        logger.warning(
            "new_domain_candidate not written for tenant=%s", row["tenant"], exc_info=True
        )
    return row


def read_new_domain_candidates(tenant_id: Optional[str]) -> List[Dict[str, Any]]:
    """Every candidate row for ``tenant_id`` (the Store's read path)."""
    path = _log_path(tenant_id)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError:
                    logger.debug("skipping a malformed candidate row", exc_info=True)
    except OSError:
        logger.warning("new_domain_candidates unreadable for tenant", exc_info=True)
        return []
    return rows
