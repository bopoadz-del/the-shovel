"""Append-only JSONL audit log for RAG retrieval activity.

One row per turn at ``${DATA_DIR}/logs/rag_audit.jsonl``. Used by both
the runtime injection path and the chat block to record retrieval
outcomes for offline tuning of K, the confidence threshold, the daily
budget, and the noise filter regex.

Best-effort: write failures are swallowed with a logger.warning. The
runtime must never refuse a chat turn because the audit log couldn't
write.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict

_LOG = logging.getLogger(__name__)
_LOCK = threading.RLock()


def _log_path() -> str:
    base = os.getenv("DATA_DIR", "./data")
    return os.path.join(base, "logs", "rag_audit.jsonl")


def write(record: Dict[str, Any]) -> None:
    """Append ``record`` as one JSON line. Best-effort, never raises."""
    path = _log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        _LOG.warning("rag_audit write failed: %s", exc)
