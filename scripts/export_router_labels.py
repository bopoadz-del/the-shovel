#!/usr/bin/env python3
"""Read-only export of stored document labels for a document-type router.

Exports every project in the connected database. There is no project allow
list and no project deny list: the deployment attached via ``DATABASE_URL``
is the scope.

``declared_type`` is the ingestion label stored on the document row. A null
or whitespace-only label is written blank. Filename and chunk text are never
consulted to fill or rewrite it.

Document text is chunk text in ``chunk_index`` order from the active RAG
namespace (the same table the Fork reads). ``doc_id`` is the stored
``content_sha256`` when that value is non-blank; otherwise it is the sha256
of the reconstructed text (chunks joined with a newline, UTF-8).

``first_512_tokens`` is the first 512 whitespace-separated tokens of that
text. Documents with no chunk text export an empty token list.

The database session is read-only. Nothing here is imported by the running
app.

Usage (from the repo root):

    DATABASE_URL=postgresql://... python scripts/export_router_labels.py --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOKEN_LIMIT = 512
_LABELS_NAME = "labels.jsonl"
_SUMS_NAME = "SHA256SUMS"


def active_chunk_table() -> str:
    """Chunk table for the active RAG namespace.

    Matches ``app.core.rag.vector_store._rag_vector_namespace``: empty
    namespace is the legacy ``chunks`` table, anything else is
    ``chunks_<namespace>`` via ``rag_chunk_table_name``.
    """
    from app.core.models import rag_chunk_table_name

    namespace = os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
    return rag_chunk_table_name(namespace)


def declared_type(stored: Any) -> str:
    """Ingestion label as stored. Blank stays blank. Never inferred."""
    if stored is None:
        return ""
    text = stored if isinstance(stored, str) else str(stored)
    if text.strip() == "":
        return ""
    return text


def _as_stored(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _content_id(stored_sha: Any) -> tuple[str, bool]:
    """Return ``(doc_id, needs_text_hash)``.

    A non-blank stored content hash is the id. Otherwise the caller hashes
    the reconstructed document text.
    """
    sha = _as_stored(stored_sha).strip()
    if sha:
        return sha, False
    return "", True


class _DocText:
    """Running token prefix and, when needed, a hash of the full text."""

    def __init__(self, needs_hash: bool) -> None:
        self.needs_hash = needs_hash
        self._hasher = hashlib.sha256() if needs_hash else None
        self._hashed_any = False
        self.tokens: list[str] = []

    def add_chunk(self, text: Any) -> None:
        piece = text if isinstance(text, str) else ("" if text is None else str(text))
        if self._hasher is not None:
            if self._hashed_any:
                self._hasher.update(b"\n")
            self._hasher.update(piece.encode("utf-8", errors="surrogateescape"))
            self._hashed_any = True
        if len(self.tokens) >= TOKEN_LIMIT:
            return
        for tok in piece.split():
            self.tokens.append(tok)
            if len(self.tokens) >= TOKEN_LIMIT:
                return

    def text_sha256(self) -> str:
        if self._hasher is None:
            raise RuntimeError("text hash requested when a stored content hash exists")
        return self._hasher.hexdigest()


def _read_only(conn: Any) -> None:
    """Mark the already-open transaction read-only.

    On PostgreSQL ``SET TRANSACTION READ ONLY`` has to be the first
    statement of the transaction. On SQLite the matching switch is
    ``PRAGMA query_only``.
    """
    name = conn.dialect.name
    if name == "postgresql":
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
    elif name == "sqlite":
        conn.exec_driver_sql("PRAGMA query_only = ON")


def _load_projects(conn: Any) -> list[str]:
    from sqlalchemy import text

    rows = conn.execute(text("SELECT id FROM projects ORDER BY id")).all()
    return [str(row[0]) for row in rows if row[0] is not None]


def _load_documents(conn: Any) -> list[Mapping[str, Any]]:
    from sqlalchemy import text

    result = conn.execute(
        text(
            "SELECT id, project_id, original_name, doc_type, uploaded_at, "
            "content_sha256 FROM documents "
            "ORDER BY project_id, uploaded_at, id"
        )
    )
    return list(result.mappings())


def _consume_chunks(conn: Any, table: str, states: dict[str, _DocText]) -> None:
    from sqlalchemy import text

    stream = conn.execution_options(yield_per=2000).execute(
        text(
            f"SELECT doc_id, text FROM {table} "
            "ORDER BY doc_id, chunk_index"
        )
    )
    for doc_id, chunk_text in stream:
        state = states.get(str(doc_id))
        if state is None:
            continue
        state.add_chunk(chunk_text)


def export_router_labels(engine: Any, dest: Path, *, chunk_table: str | None = None) -> dict[str, Any]:
    """Write ``labels.jsonl`` and ``SHA256SUMS`` under ``dest``.

    Returns a summary: export id, project ids, row count, counts by stored
    type, and the sha256 of ``labels.jsonl``. Does not write to the database.
    """
    from sqlalchemy.exc import SQLAlchemyError

    dest.mkdir(parents=True, exist_ok=True)
    table = chunk_table if chunk_table is not None else active_chunk_table()
    export_id = str(uuid.uuid4())
    tmp = dest / f".{_LABELS_NAME}.tmp"
    labels_path = dest / _LABELS_NAME
    sums_path = dest / _SUMS_NAME

    try:
        with engine.connect() as conn:
            with conn.begin():
                _read_only(conn)
                project_ids = _load_projects(conn)
                documents = _load_documents(conn)
                states: dict[str, _DocText] = {}
                order: list[Mapping[str, Any]] = []
                for doc in documents:
                    doc_id = doc["id"]
                    if doc_id is None:
                        continue
                    _stored, needs_hash = _content_id(doc["content_sha256"])
                    states[str(doc_id)] = _DocText(needs_hash)
                    order.append(doc)
                _consume_chunks(conn, table, states)
    except SQLAlchemyError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"label export failed: {exc.__class__.__name__}: {exc}") from exc

    by_type: Counter[str] = Counter()
    seen_projects: set[str] = set(project_ids)
    rows = 0
    without_text = 0
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for doc in order:
                key = str(doc["id"])
                state = states[key]
                stored_sha, needs_hash = _content_id(doc["content_sha256"])
                content_id = state.text_sha256() if needs_hash else stored_sha
                label = declared_type(doc["doc_type"])
                tokens = state.tokens
                if not tokens:
                    without_text += 1
                project_id = _as_stored(doc["project_id"])
                if project_id:
                    seen_projects.add(project_id)
                record = {
                    "doc_id": content_id,
                    "project_id": project_id,
                    "declared_type": label,
                    "filename": _as_stored(doc["original_name"]),
                    "first_512_tokens": tokens,
                    "ingested_at": _as_stored(doc["uploaded_at"]),
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                by_type[label] += 1
                rows += 1
        tmp.replace(labels_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    sums_path.write_text(f"{digest}  {_LABELS_NAME}\n", encoding="utf-8")

    summary = {
        "export_id": export_id,
        "projects": sorted(seen_projects),
        "row_count": rows,
        "by_declared_type": dict(sorted(by_type.items(), key=lambda item: (-item[1], item[0]))),
        "docs_without_text": without_text,
        "labels_sha256": digest,
        "chunk_table": table,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="directory for labels.jsonl and SHA256SUMS",
    )
    args = parser.parse_args(argv)
    if not os.getenv("DATABASE_URL", "").strip():
        print("DATABASE_URL is required; refusing to export an implicit local database", file=sys.stderr)
        return 2
    from app.core.db import _engine_for_url, get_database_url

    summary = export_router_labels(_engine_for_url(get_database_url()), args.out)
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
