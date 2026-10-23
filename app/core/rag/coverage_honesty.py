"""Coverage honesty line + partial-index absence phrasing.

Every answer that knows its project must carry ``N of M project documents
indexed`` with live counts (or an explicit fixture tuple in tests). Below
100% the model is not allowed to say a clause ``does not exist`` or that
there is ``no such clause`` — those phrases claim a complete search of a
corpus that is still being ingested. Rewrite them to
``not found in the N indexed``.

The 2,935 / 6,206 pair used in tests is the historical coverage fixture,
not a claim about live Neon.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

COVERAGE_LINE_TEMPLATE = "{n} of {m} project documents indexed"
NOT_FOUND_IN_INDEXED = "not found in the {n} indexed"

# Phrases that assert a complete search. Forbidden on a partial index.
FORBIDDEN_ABSENCE_PHRASES: tuple[str, ...] = (
    "does not exist",
    "no such clause",
)

_COVERAGE_LINE_RE = re.compile(
    r"^\d+ of \d+ project documents indexed\s*$",
    re.MULTILINE,
)
_FORBIDDEN_RE = re.compile(
    r"does not exist|no such clause",
    re.IGNORECASE,
)


def format_coverage_line(indexed: int, total: int) -> str:
    return COVERAGE_LINE_TEMPLATE.format(n=int(indexed), m=int(total))


def format_not_found(indexed: int) -> str:
    return NOT_FOUND_IN_INDEXED.format(n=int(indexed))


def is_partial(indexed: int, total: int) -> bool:
    return int(total) > 0 and int(indexed) < int(total)


def rewrite_forbidden_absence_claims(text: str, indexed: int, total: int) -> str:
    """Replace complete-search claims when coverage is below 100%."""
    if not is_partial(indexed, total):
        return text
    replacement = format_not_found(indexed)
    return _FORBIDDEN_RE.sub(replacement, text or "")


def ensure_coverage_line(text: str, indexed: int, total: int) -> str:
    """Append the honesty line once. Does not invent counts."""
    line = format_coverage_line(indexed, total)
    body = text or ""
    if _COVERAGE_LINE_RE.search(body) or line in body:
        return body
    if body and not body.endswith("\n"):
        body += "\n"
    return body + line + "\n"


def live_coverage(project_id: str) -> Optional[tuple[int, int]]:
    """Indexed-doc count and project-doc count. None when we cannot tell.

    N = documents that have at least one RAG chunk (``count_by_doc``).
    M = ``count_documents`` for the same project. Read-only; does not
    write the ingest path.
    """
    if not (project_id or "").strip():
        return None
    try:
        from app.core.projects import count_documents
        from app.core.rag.vector_store import get_store

        total = int(count_documents(project_id) or 0)
        by_doc = get_store().count_by_doc(project_id) or {}
        indexed = sum(1 for n in by_doc.values() if int(n or 0) > 0)
        return indexed, total
    except Exception:  # honesty must not break an answer
        logger.warning(
            "live coverage lookup failed for project_id=%r", project_id, exc_info=True
        )
        return None


def apply_coverage_honesty(
    text: str,
    *,
    project_id: str | None = None,
    coverage: tuple[int, int] | None = None,
) -> str:
    """Rewrite + stamp the coverage line when counts are known.

    ``coverage`` is the fixture/override pair ``(indexed, total)``. When
    omitted, live counts are read for ``project_id``. No project and no
    fixture → the text is unchanged (tests of other gates stay stable).
    """
    counts = coverage
    if counts is None and project_id:
        counts = live_coverage(project_id)
    if counts is None:
        return text
    indexed, total = int(counts[0]), int(counts[1])
    out = rewrite_forbidden_absence_claims(text, indexed, total)
    return ensure_coverage_line(out, indexed, total)
