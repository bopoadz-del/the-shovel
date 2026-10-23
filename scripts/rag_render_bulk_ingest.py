#!/usr/bin/env python3
"""Resumable, batch-committed Render RAG bulk ingestion.

Design guarantees
-----------------
* Never auto-wipes production. Destructive wipe requires ``--destructive-wipe``.
* Render Postgres is the source of truth; local checkpoint is an operational aid.
* Each embed batch is persisted in a short transaction and committed before the
  next batch. A crash loses at most the in-flight batch.
* Resume skips rows that already have a *valid* embedding (matching model, dim,
  normalized flag, and source text hash).
* Credentials are never logged; host/db names are redacted.

Usage examples
--------------
    .venv\\Scripts\\python.exe -u scripts\\rag_render_bulk_ingest.py --verify-only
    .venv\\Scripts\\python.exe -u scripts\\rag_render_bulk_ingest.py --resume --max-batches 2
    .venv\\Scripts\\python.exe -u scripts\\rag_render_bulk_ingest.py --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# StopFlag and the handler installer moved to app/core/ingest_lifecycle so this
# script and scripts/p1b_ingest_drive_server.py share ONE stop protocol instead
# of drifting into two. Re-exported here because the call sites and tests import
# them from this module. tests/test_ingest_lifecycle.py asserts they stay the
# same objects.
from app.core.ingest_lifecycle import (  # noqa: E402 - after sys.path bootstrap
    StopFlag,
    install_signal_handlers as _install_signal_handlers,
)

PG_ID = "dpg-d8m22mcm0tmc73b04elg-a"
# The project + expected totals were hardcoded to the first backfill
# (example_infra_pack, 27 docs / 7538 chunks). They are env-overridable so the
# same hardened writer can load another corpus -- e.g. the 2,924-doc
# drive_archive text recovered from the laptop vector store -- without forking
# the script. Defaults are unchanged, so existing invocations behave identically.
PROJECT_ID = os.getenv("INGEST_PROJECT_ID", "example_infra_pack")
DEFAULT_JSONL = Path(
    os.getenv("INGEST_JSONL", str(REPO / "rag_backfill_client_clean_all.jsonl"))
)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EXPECTED_DIM = 384
EXPECTED_DOCS = int(os.getenv("INGEST_EXPECTED_DOCS", "27"))
EXPECTED_CHUNKS = int(os.getenv("INGEST_EXPECTED_CHUNKS", "7538"))
DEFAULT_BATCH = 256
CHECKPOINT_PATH = REPO / "rag_render_bulk_ingest_checkpoint.json"
LOCK_PATH = REPO / "rag_render_bulk_ingest.lock"
NAMESPACE = "v2"

# Write resilience/pacing (Render Postgres drops connections under sustained
# bulk load). Tunable via env; defaults chosen for a starter-tier DB.
_RETRY_ATTEMPTS = max(1, int(os.getenv("INGEST_RETRY_ATTEMPTS", "5")))
_RETRY_BACKOFF_S = float(os.getenv("INGEST_RETRY_BACKOFF_S", "2.0"))
_PACING_MS = int(os.getenv("INGEST_PACING_MS", "150"))


# ── helpers ──────────────────────────────────────────────────────────────────


def _with_db_retry(fn: "Callable[[], Any]") -> Any:
    """Run a DB operation, reconnecting and retrying on a dropped/failed
    connection. Retries only transient connectivity errors (OperationalError /
    "server closed the connection" / connection failures); logic errors
    (IntegrityError, DataError, etc.) propagate immediately. Exponential
    backoff between attempts."""
    last: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classify below
            msg = str(exc).lower()
            transient = (
                exc.__class__.__name__ in ("OperationalError", "InterfaceError")
                or "server closed the connection" in msg
                or "connection failed" in msg
                or "connection is closed" in msg
                or "consuming input failed" in msg
                or "ssl connection has been closed" in msg
            )
            if not transient or attempt == _RETRY_ATTEMPTS:
                raise
            last = exc
            sleep_s = _RETRY_BACKOFF_S * (2 ** (attempt - 1))
            print(f"[retry] DB op failed ({exc.__class__.__name__}); "
                  f"attempt {attempt}/{_RETRY_ATTEMPTS}, backoff {sleep_s:.1f}s",
                  flush=True)
            time.sleep(sleep_s)
    raise last  # unreachable, satisfies type checkers


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id_for(doc_id: str, chunk_index: int, project_id: str = PROJECT_ID) -> str:
    return f"{project_id}:{doc_id}:{chunk_index}"


def redact(u: str) -> str:
    m = re.match(
        r"([a-zA-Z0-9+]+)://([^:@/]+)(:[^@]*)?@([^:/]+)(:\d+)?(/[^?]*)?",
        u,
    )
    if not m:
        return "<redacted>"
    scheme, user, _pw, host, port, path = m.groups()
    short = (
        host.split(".")[0][:14] + "….render.com"
        if "render" in host.lower()
        else host[:12] + "…"
    )
    db = ""
    if path:
        # redact db name to first char + …
        bare = path.lstrip("/").split("?")[0]
        db = "/" + (bare[:1] + "…" if bare else "")
    return f"{scheme}://{user}:***@{short}{port or ''}{db}"


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return -1.0


def _vec_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in vec.tolist()) + "]"


# ── data types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkRow:
    doc_id: str
    chunk_index: int
    text: str

    @property
    def chunk_id(self) -> str:
        return chunk_id_for(self.doc_id, self.chunk_index)

    @property
    def thash(self) -> str:
        return text_hash(self.text)


@dataclass
class ExistingMeta:
    chunk_id: str
    embedding_model: str
    embedding_dim: int
    embedding_normalized: bool
    text: str

    @property
    def thash(self) -> str:
        return text_hash(self.text)


@dataclass
class DocMeta:
    doc_id: str
    original_name: str
    source_path: Optional[str] = None
    drive_file_id: Optional[str] = None
    ext: Optional[str] = None
    content_sha256: Optional[str] = None


@dataclass
class IngestStats:
    total: int = 0
    valid_skipped: int = 0
    written: int = 0
    batches: int = 0
    rejected_wrong_identity: int = 0
    regenerated_text_hash: int = 0
    stopped_reason: str = ""


class ChunkStore(Protocol):
    def fetch_existing(self, project_id: str) -> Dict[str, ExistingMeta]: ...

    def upsert_batch(
        self,
        rows: Sequence[ChunkRow],
        embeddings: np.ndarray,
        model_name: str,
        dim: int,
        normalized: bool = True,
    ) -> int: ...

    def count_chunks(self, project_id: str) -> int: ...

    def upsert_documents(self, docs: Dict[str, DocMeta]) -> int: ...

    def count_documents(self, project_id: str) -> int: ...

    def wipe_all(self) -> Dict[str, int]: ...


# ── memory store (tests) ─────────────────────────────────────────────────────


@dataclass
class MemoryStore:
    """In-memory ChunkStore for unit tests. Supports fail-on-Nth-commit."""

    chunks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    documents: Dict[str, DocMeta] = field(default_factory=dict)
    fail_on_commit: Optional[int] = None  # 1-based commit index to raise
    _commit_n: int = 0
    writes_attempted: int = 0

    def fetch_existing(self, project_id: str) -> Dict[str, ExistingMeta]:
        out: Dict[str, ExistingMeta] = {}
        for cid, row in self.chunks.items():
            if row["project_id"] != project_id:
                continue
            out[cid] = ExistingMeta(
                chunk_id=cid,
                embedding_model=row["embedding_model"],
                embedding_dim=int(row["embedding_dim"]),
                embedding_normalized=bool(row["embedding_normalized"]),
                text=row["text"],
            )
        return out

    def upsert_batch(
        self,
        rows: Sequence[ChunkRow],
        embeddings: np.ndarray,
        model_name: str,
        dim: int,
        normalized: bool = True,
    ) -> int:
        self.writes_attempted += 1
        self._commit_n += 1
        if self.fail_on_commit is not None and self._commit_n == self.fail_on_commit:
            raise RuntimeError(f"simulated commit failure at batch {self._commit_n}")
        now = datetime.now(timezone.utc).isoformat()
        for i, row in enumerate(rows):
            self.chunks[row.chunk_id] = {
                "chunk_id": row.chunk_id,
                "project_id": PROJECT_ID,
                "doc_id": row.doc_id,
                "chunk_index": row.chunk_index,
                "text": row.text,
                "embedding": np.asarray(embeddings[i], dtype=np.float32).copy(),
                "created_at": now,
                "embedding_model": model_name,
                "embedding_dim": dim,
                "embedding_normalized": normalized,
            }
        return len(rows)

    def count_chunks(self, project_id: str) -> int:
        return sum(1 for r in self.chunks.values() if r["project_id"] == project_id)

    def upsert_documents(self, docs: Dict[str, DocMeta]) -> int:
        self.documents.update(docs)
        return len(docs)

    def count_documents(self, project_id: str) -> int:
        return len(self.documents)

    def wipe_all(self) -> Dict[str, int]:
        n_c = len(self.chunks)
        n_d = len(self.documents)
        self.chunks.clear()
        self.documents.clear()
        return {"chunks_v2": n_c, "documents": n_d}


# ── postgres store ───────────────────────────────────────────────────────────


class PostgresStore:
    def __init__(self, database_url: str):
        from sqlalchemy import create_engine

        # Keep the original URL string for psycopg — str(engine.url) can
        # mangle/hide passwords and break auth.
        self.url = database_url
        self._psycopg_url = database_url.replace("postgresql+psycopg://", "postgresql://")
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=0,
            connect_args={"connect_timeout": 30},
        )

    def fetch_existing(self, project_id: str) -> Dict[str, ExistingMeta]:
        from sqlalchemy import text

        out: Dict[str, ExistingMeta] = {}
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT chunk_id, embedding_model, embedding_dim, "
                    "embedding_normalized, text "
                    "FROM chunks_v2 WHERE project_id = :p"
                ),
                {"p": project_id},
            ).fetchall()
        for r in rows:
            out[r[0]] = ExistingMeta(
                chunk_id=r[0],
                embedding_model=r[1],
                embedding_dim=int(r[2]),
                embedding_normalized=bool(r[3]),
                text=r[4] or "",
            )
        return out

    def upsert_batch(
        self,
        rows: Sequence[ChunkRow],
        embeddings: np.ndarray,
        model_name: str,
        dim: int,
        normalized: bool = True,
    ) -> int:
        import psycopg

        if len(rows) != len(embeddings):
            raise ValueError(f"rows/emb mismatch {len(rows)} vs {len(embeddings)}")
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT INTO chunks_v2 (
                chunk_id, project_id, doc_id, chunk_index, text,
                embedding, created_at, embedding_model, embedding_dim, embedding_normalized
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s::vector, %s, %s, %s, %s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                created_at = EXCLUDED.created_at,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dim = EXCLUDED.embedding_dim,
                embedding_normalized = EXCLUDED.embedding_normalized
        """
        params = []
        for i, row in enumerate(rows):
            params.append(
                (
                    row.chunk_id,
                    PROJECT_ID,
                    row.doc_id,
                    row.chunk_index,
                    row.text,
                    _vec_literal(embeddings[i]),
                    now,
                    model_name,
                    dim,
                    normalized,
                )
            )
        def _insert() -> None:
            with psycopg.connect(self._psycopg_url, connect_timeout=60) as conn:
                try:
                    with conn.cursor() as cur:
                        cur.executemany(sql, params)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        def _confirm() -> int:
            with psycopg.connect(self._psycopg_url, connect_timeout=60) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM chunks_v2 WHERE chunk_id = ANY(%s)",
                        ([r.chunk_id for r in rows],),
                    )
                    return int(cur.fetchone()[0])

        # A struggling Render Postgres closes the connection mid-write under
        # sustained bulk load ("server closed the connection unexpectedly").
        # Raw psycopg has no pool_pre_ping safety net (unlike the app's
        # SQLAlchemy engine), so reconnect-and-retry here. INSERT is idempotent
        # (ON CONFLICT DO UPDATE) so a retried batch can't double-write.
        _with_db_retry(_insert)
        confirmed = _with_db_retry(_confirm)
        if confirmed != len(rows):
            raise RuntimeError(
                f"post-commit confirm failed: expected {len(rows)} got {confirmed}"
            )
        # Pace writes so sustained ingestion never starves live chat retrieval
        # sharing this DB (INGEST_PACING_MS, default 150ms between batches).
        if _PACING_MS:
            time.sleep(_PACING_MS / 1000.0)
        return confirmed

    def count_chunks(self, project_id: str) -> int:
        from sqlalchemy import text

        with self.engine.connect() as conn:
            return int(
                conn.execute(
                    text("SELECT COUNT(*) FROM chunks_v2 WHERE project_id=:p"),
                    {"p": project_id},
                ).scalar()
                or 0
            )

    def upsert_documents(self, docs: Dict[str, DocMeta]) -> int:
        from sqlalchemy import text

        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            proj = conn.execute(
                text("SELECT id FROM projects WHERE id=:p"), {"p": PROJECT_ID}
            ).fetchone()
            if proj is None:
                uid = conn.execute(text("SELECT id FROM users WHERE id='system'")).scalar()
                if not uid:
                    uid = conn.execute(text("SELECT id FROM users LIMIT 1")).scalar()
                if not uid:
                    raise RuntimeError("no users row for project FK")
                conn.execute(
                    text(
                        "INSERT INTO projects "
                        "(id, name, status, aconex_connected, user_id, created_at, "
                        "is_approved, origin) "
                        "VALUES (:id, :name, 'active', false, :uid, :created, true, "
                        "'admin_drive_approved')"
                    ),
                    {
                        "id": PROJECT_ID,
                        "name": "the client project",
                        "uid": uid,
                        "created": now,
                    },
                )
            rows = []
            for doc_id, meta in docs.items():
                metadata = json.dumps(
                    {
                        "drive_file_id": meta.drive_file_id,
                        "source_path": meta.source_path,
                        "ext": meta.ext,
                        "source": "rag_backfill_client_clean_all",
                    }
                )
                rows.append(
                    {
                        "id": doc_id,
                        "pid": PROJECT_ID,
                        "name": meta.original_name,
                        "path": meta.source_path,
                        "uploaded": now,
                        "sha": meta.content_sha256,
                        "meta": metadata,
                    }
                )
            conn.execute(
                text(
                    "INSERT INTO documents "
                    "(id, project_id, original_name, stored_as, file_path, doc_type, "
                    "doc_role, size, uploaded_at, content_sha256, metadata) "
                    "VALUES "
                    "(:id, :pid, :name, NULL, :path, 'document', 'other', 0, "
                    ":uploaded, :sha, CAST(:meta AS jsonb)) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "project_id=EXCLUDED.project_id, "
                    "original_name=EXCLUDED.original_name, "
                    "content_sha256=EXCLUDED.content_sha256, "
                    "metadata=EXCLUDED.metadata"
                ),
                rows,
            )
        return len(rows)

    def count_documents(self, project_id: str) -> int:
        from sqlalchemy import text

        with self.engine.connect() as conn:
            return int(
                conn.execute(
                    text("SELECT COUNT(*) FROM documents WHERE project_id=:p"),
                    {"p": project_id},
                ).scalar()
                or 0
            )

    def wipe_all(self) -> Dict[str, int]:
        from sqlalchemy import text

        deleted: Dict[str, int] = {}
        with self.engine.begin() as conn:
            for t in ("chunks_v2", "chunks", "photo_chunks", "doc_index", "documents"):
                exists = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema='public' AND table_name=:t"
                    ),
                    {"t": t},
                ).scalar()
                if not exists:
                    continue
                res = conn.execute(text(f"DELETE FROM {t}"))
                deleted[t] = int(res.rowcount or 0)
        return deleted

    def acceptance_report(self, project_id: str = PROJECT_ID) -> Dict[str, Any]:
        from sqlalchemy import text

        with self.engine.connect() as conn:
            docs = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM documents WHERE project_id=:p"),
                    {"p": project_id},
                ).scalar()
                or 0
            )
            chunks = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM chunks_v2 WHERE project_id=:p"),
                    {"p": project_id},
                ).scalar()
                or 0
            )
            models = [
                {"model": r[0], "dim": r[1], "n": r[2]}
                for r in conn.execute(
                    text(
                        "SELECT embedding_model, embedding_dim, COUNT(*) "
                        "FROM chunks_v2 WHERE project_id=:p GROUP BY 1,2"
                    ),
                    {"p": project_id},
                ).fetchall()
            ]
            null_emb = int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM chunks_v2 "
                        "WHERE project_id=:p AND embedding IS NULL"
                    ),
                    {"p": project_id},
                ).scalar()
                or 0
            )
            dup_ids = int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM ("
                        " SELECT chunk_id FROM chunks_v2 WHERE project_id=:p "
                        " GROUP BY chunk_id HAVING COUNT(*)>1"
                        ") t"
                    ),
                    {"p": project_id},
                ).scalar()
                or 0
            )
            wrong_dim = int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM chunks_v2 "
                        "WHERE project_id=:p AND embedding_dim <> :d"
                    ),
                    {"p": project_id, "d": EXPECTED_DIM},
                ).scalar()
                or 0
            )
            wrong_model = int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM chunks_v2 "
                        "WHERE project_id=:p AND embedding_model <> :m"
                    ),
                    {"p": project_id, "m": EMBED_MODEL},
                ).scalar()
                or 0
            )
            sample = conn.execute(
                text(
                    "SELECT chunk_id, left(text, 80) FROM chunks_v2 "
                    "WHERE project_id=:p ORDER BY chunk_id LIMIT 3"
                ),
                {"p": project_id},
            ).fetchall()
        return {
            "documents": docs,
            "chunks_v2": chunks,
            "models": models,
            "null_embeddings": null_emb,
            "dup_chunk_ids": dup_ids,
            "wrong_dim": wrong_dim,
            "wrong_model": wrong_model,
            "sample": sample,
        }


# ── validity / planning ──────────────────────────────────────────────────────


def is_valid_existing(
    existing: ExistingMeta,
    source: ChunkRow,
    expected_model: str,
    expected_dim: int,
) -> Tuple[bool, str]:
    if existing.embedding_model != expected_model:
        return False, "wrong_model"
    if int(existing.embedding_dim) != int(expected_dim):
        return False, "wrong_dim"
    if not existing.embedding_normalized:
        return False, "not_normalized"
    if existing.thash != source.thash:
        return False, "text_hash_changed"
    return True, "ok"


def plan_work(
    flat: Sequence[ChunkRow],
    existing: Dict[str, ExistingMeta],
    expected_model: str,
    expected_dim: int,
) -> Tuple[List[ChunkRow], IngestStats]:
    stats = IngestStats(total=len(flat))
    todo: List[ChunkRow] = []
    for row in flat:
        ex = existing.get(row.chunk_id)
        if ex is None:
            todo.append(row)
            continue
        ok, reason = is_valid_existing(ex, row, expected_model, expected_dim)
        if ok:
            stats.valid_skipped += 1
            continue
        if reason == "wrong_model" or reason == "wrong_dim":
            stats.rejected_wrong_identity += 1
        elif reason == "text_hash_changed":
            stats.regenerated_text_hash += 1
        todo.append(row)
    return todo, stats


# ── JSONL ────────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> Tuple[Dict[str, DocMeta], List[ChunkRow]]:
    """Load JSONL and re-index chunk_index sequentially per doc.

    The source JSONL has duplicate (doc_id, chunk_index) keys with *different*
    texts (7538 lines → ~4748 unique keys). Re-indexing preserves all texts.
    """
    docs: Dict[str, DocMeta] = {}
    per_doc: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("project_id") != PROJECT_ID:
                raise RuntimeError(f"line {line_no}: bad project_id {o.get('project_id')}")
            doc_id = o["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = DocMeta(
                    doc_id=doc_id,
                    original_name=o.get("original_name") or doc_id,
                    source_path=o.get("source_path"),
                    drive_file_id=o.get("drive_file_id"),
                    ext=o.get("ext"),
                    content_sha256=o.get("content_sha256"),
                )
                per_doc[doc_id] = []
            per_doc[doc_id].append((o["text"] or "").replace("\x00", ""))

    flat: List[ChunkRow] = []
    for doc_id in sorted(per_doc.keys()):
        for i, txt in enumerate(per_doc[doc_id]):
            flat.append(ChunkRow(doc_id=doc_id, chunk_index=i, text=txt))
    return docs, flat


# ── checkpoint ───────────────────────────────────────────────────────────────


def write_checkpoint(
    path: Path,
    *,
    written_ids: Sequence[str],
    stats: IngestStats,
    batch_size: int,
) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "project_id": PROJECT_ID,
        "model": EMBED_MODEL,
        "dim": EXPECTED_DIM,
        "namespace": NAMESPACE,
        "batch_size": batch_size,
        "valid_skipped": stats.valid_skipped,
        "written": stats.written,
        "batches": stats.batches,
        "last_chunk_ids": list(written_ids)[-32:],
        "stopped_reason": stats.stopped_reason,
        "note": "Render is source of truth; this file is operational aid only",
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# ── core ingest loop ─────────────────────────────────────────────────────────


def encode_batch(embedder: Any, texts: Sequence[str]) -> np.ndarray:
    """Encode and L2-normalize. Uses embedder.encode when available."""
    if hasattr(embedder, "encode"):
        emb = embedder.encode(list(texts))
    else:
        raise TypeError("embedder missing encode()")
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"bad embedding shape {arr.shape}")
    return arr


def run_ingest(
    store: ChunkStore,
    flat: Sequence[ChunkRow],
    docs: Dict[str, DocMeta],
    embedder: Any,
    *,
    batch_size: int = DEFAULT_BATCH,
    max_batches: Optional[int] = None,
    resume: bool = True,
    verify_only: bool = False,
    preview: bool = False,
    destructive_wipe: bool = False,
    stop_flag: Optional[StopFlag] = None,
    checkpoint_path: Optional[Path] = None,
    upsert_docs: bool = True,
    expected_model: str = EMBED_MODEL,
    expected_dim: int = EXPECTED_DIM,
) -> IngestStats:
    if stop_flag is None:
        stop_flag = StopFlag()
    if batch_size < 1 or batch_size > 256:
        raise ValueError(f"batch_size out of 1–256: {batch_size}")

    model_name = getattr(embedder, "model_name", expected_model)
    dim = int(getattr(embedder, "dim", expected_dim))
    if model_name != expected_model:
        raise RuntimeError(f"embedder model mismatch: {model_name} != {expected_model}")
    if dim != expected_dim:
        raise RuntimeError(f"embedder dim mismatch: {dim} != {expected_dim}")

    if destructive_wipe:
        if verify_only or preview:
            raise RuntimeError("refusing destructive wipe with verify-only/preview")
        wiped = store.wipe_all()
        log(f"[wipe] destructive wiped={wiped}")

    if upsert_docs and not verify_only and not preview:
        n_docs = store.upsert_documents(docs)
        log(f"[ingest] documents upserted={n_docs}")

    existing: Dict[str, ExistingMeta] = {}
    if resume or verify_only:
        existing = store.fetch_existing(PROJECT_ID)
    else:
        existing = {}

    todo, stats = plan_work(flat, existing, expected_model, expected_dim)
    log(
        f"[plan] total={stats.total} valid_skip={stats.valid_skipped} "
        f"todo={len(todo)} wrong_identity={stats.rejected_wrong_identity} "
        f"text_hash_regen={stats.regenerated_text_hash} "
        f"store_chunks={store.count_chunks(PROJECT_ID)}"
    )

    if verify_only:
        stats.stopped_reason = "verify_only"
        log(
            f"[verify] docs={store.count_documents(PROJECT_ID)} "
            f"chunks={store.count_chunks(PROJECT_ID)} "
            f"missing_or_stale={len(todo)}"
        )
        return stats

    if preview:
        stats.stopped_reason = "preview"
        log(f"[preview] would write {len(todo)} chunks in batches of {batch_size}")
        return stats

    if not todo:
        stats.stopped_reason = "nothing_to_do"
        log("[ingest] nothing to do — corpus already valid")
        return stats

    t0 = time.time()
    for bi, start in enumerate(range(0, len(todo), batch_size)):
        if stop_flag.stop:
            stats.stopped_reason = stop_flag.reason or "signal"
            log(f"[ingest] stop requested ({stats.stopped_reason}) after {stats.batches} batches")
            break
        if max_batches is not None and bi >= max_batches:
            stats.stopped_reason = "max_batches"
            log(f"[ingest] max-batches={max_batches} reached")
            break

        batch = todo[start : start + batch_size]
        texts = [r.text for r in batch]
        t_emb = time.time()
        emb = encode_batch(embedder, texts)
        emb_s = time.time() - t_emb
        if emb.shape != (len(batch), dim):
            raise RuntimeError(f"unexpected emb shape {emb.shape}")

        t_db = time.time()
        try:
            n = store.upsert_batch(batch, emb, model_name, dim, normalized=True)
        except Exception:
            # batch not committed (store must roll back); prior batches intact
            log(f"[ingest] BATCH FAILED at batch_index={bi}: {traceback.format_exc()}")
            stats.stopped_reason = "batch_failed"
            raise
        db_s = time.time() - t_db

        stats.written += n
        stats.batches += 1
        elapsed = time.time() - t0
        rate = stats.written / elapsed if elapsed > 0 else 0
        eta = (len(todo) - stats.written) / rate if rate > 0 else 0
        rss = _rss_mb()
        log(
            f"[ingest] batch={stats.batches} wrote={n} "
            f"done={stats.valid_skipped + stats.written}/{stats.total} "
            f"(+{stats.written} this run) "
            f"emb_s={emb_s:.1f} db_s={db_s:.1f} "
            f"rate={rate:.1f}/s eta={eta:.0f}s rss_mb={rss:.0f}"
        )
        if checkpoint_path is not None:
            write_checkpoint(
                checkpoint_path,
                written_ids=[r.chunk_id for r in batch],
                stats=stats,
                batch_size=batch_size,
            )

    if not stats.stopped_reason:
        stats.stopped_reason = "completed"
    return stats


# ── Render URL ───────────────────────────────────────────────────────────────


def fetch_database_url(api_key: Optional[str] = None) -> str:
    token = (
        api_key
        or os.environ.get("RENDER_API_KEY")
        or os.environ.get("RENDER_MCP_TOKEN")
        or ""
    )
    if not token:
        raise RuntimeError("RENDER_API_KEY / RENDER_MCP_TOKEN missing")
    req = urllib.request.Request(
        f"https://api.render.com/v1/postgres/{PG_ID}/connection-info",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        url = json.loads(resp.read().decode())["externalConnectionString"]
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    return url


def _tune_torch(threads: int) -> None:
    try:
        import torch

        n = max(1, threads)
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
        log(f"[config] torch threads={n} cuda={torch.cuda.is_available()}")
    except Exception as exc:  # noqa: BLE001
        log(f"[config] torch tune skipped: {exc}")


def install_signal_handlers(stop_flag: StopFlag) -> None:
    """One-arg shim so existing call sites keep working; the implementation is
    ``app.core.ingest_lifecycle.install_signal_handlers``. SIGHUP is included
    there — a detached Render Shell run that is HUPed on shell close used to
    die as silently as an OOM kill."""
    _install_signal_handlers(stop_flag, log=log)


class ProcessLock:
    """Exclusive lock so only one live ingest runs at a time (Windows-safe)."""

    def __init__(self, path: Path = LOCK_PATH):
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Stale lock cleanup: if pid in lock is dead, remove.
        if self.path.exists():
            try:
                text = self.path.read_text(encoding="utf-8").strip()
                old_pid = int(text.split("=", 1)[-1]) if text else -1
            except (OSError, ValueError):
                old_pid = -1
            alive = False
            if old_pid > 0:
                try:
                    import psutil

                    alive = psutil.pid_exists(old_pid)
                except Exception:  # noqa: BLE001
                    alive = False
            if alive:
                raise RuntimeError(
                    f"another ingest holds {self.path} (pid={old_pid}) — single process only"
                )
            try:
                self.path.unlink()
            except OSError:
                pass

        # Atomic create — fails if another process wins the race.
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"another ingest holds {self.path} — single process only"
            ) from exc
        self._fh = os.fdopen(fd, "w", encoding="utf-8")
        self._fh.write(f"pid={os.getpid()}\n")
        self._fh.flush()

    def release(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resumable Render RAG bulk ingest")
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true", help="Ignore existing rows (still upsert)")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--preview", action="store_true", help="Plan only; no writes")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("EMBED_BATCH", DEFAULT_BATCH)))
    p.add_argument("--threads", type=int, default=int(os.environ.get("TORCH_NUM_THREADS", "12")))
    p.add_argument("--destructive-wipe", action="store_true", help="EXPLICIT wipe — off by default")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--fake-embedder", action="store_true", help="Use fake embedder (tests/dev)")
    p.add_argument("--database-url", type=str, default=None, help="Override (tests); prefer Render API")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    resume = not args.no_resume

    if not args.jsonl.exists():
        log(f"missing jsonl {args.jsonl}")
        return 1
    if args.batch_size < 1 or args.batch_size > 256:
        log(f"batch-size out of 1–256: {args.batch_size}")
        return 1
    if args.destructive_wipe and (args.verify_only or args.preview):
        log("refusing --destructive-wipe with --verify-only/--preview")
        return 1
    if args.destructive_wipe and os.environ.get("CONFIRM_DESTRUCTIVE_WIPE") != "YES":
        log(
            "refusing --destructive-wipe without CONFIRM_DESTRUCTIVE_WIPE=YES "
            "(production protection)"
        )
        return 1

    _tune_torch(args.threads)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ["RAG_EMBEDDING_MODEL"] = EMBED_MODEL
    os.environ["RAG_VECTOR_NAMESPACE"] = NAMESPACE

    if args.database_url:
        url = args.database_url
    else:
        url = fetch_database_url()
    log(f"[target] Render Postgres {redact(url)}")
    log(
        f"[config] batch={args.batch_size} threads={args.threads} "
        f"model={EMBED_MODEL} resume={resume} verify_only={args.verify_only} "
        f"preview={args.preview} wipe={args.destructive_wipe}"
    )

    lock = ProcessLock()
    try:
        lock.acquire()
    except RuntimeError as exc:
        log(f"[fatal] {exc}")
        return 3

    try:
        store = PostgresStore(url)
        docs, flat = load_jsonl(args.jsonl)
        log(f"[ingest] jsonl docs={len(docs)} chunks={len(flat)}")

        stop_flag = StopFlag()
        install_signal_handlers(stop_flag)

        # verify/preview: no embedder needed — plan against Render only
        if args.verify_only or args.preview:
            class _MetaEmbedder:
                model_name = EMBED_MODEL
                dim = EXPECTED_DIM

                def encode(self, texts):  # pragma: no cover
                    raise RuntimeError("encode not used in verify/preview")

            embedder: Any = _MetaEmbedder()
            expected_model = EMBED_MODEL
            expected_dim = EXPECTED_DIM
        else:
            from app.core.rag.embeddings import get_embedder, reset_embedder_cache

            reset_embedder_cache()
            model = "fake" if args.fake_embedder else EMBED_MODEL
            if args.fake_embedder:
                log("[error] --fake-embedder cannot write to live Render (dim mismatch)")
                return 1
            embedder = get_embedder(model)
            if embedder.model_name != EMBED_MODEL or int(embedder.dim) != EXPECTED_DIM:
                log(f"[error] bad embedder identity {embedder.model_name}/{embedder.dim}")
                return 1
            expected_model = EMBED_MODEL
            expected_dim = EXPECTED_DIM
            log(f"[ingest] embedder={embedder.model_name} dim={embedder.dim}")

        try:
            stats = run_ingest(
                store,
                flat,
                docs,
                embedder,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                resume=resume,
                verify_only=args.verify_only,
                preview=args.preview,
                destructive_wipe=args.destructive_wipe,
                stop_flag=stop_flag,
                checkpoint_path=None
                if (args.verify_only or args.preview)
                else args.checkpoint,
                expected_model=expected_model,
                expected_dim=expected_dim,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[fatal] {exc}")
            return 1

        report = store.acceptance_report() if hasattr(store, "acceptance_report") else {}
        if report:
            log(f"[report] {json.dumps(report, default=str)}")

        log(
            f"[final] written={stats.written} skipped={stats.valid_skipped} "
            f"batches={stats.batches} reason={stats.stopped_reason}"
        )
        if args.verify_only:
            missing = stats.total - stats.valid_skipped
            ok = (
                report.get("documents") == EXPECTED_DOCS
                and report.get("chunks_v2") == EXPECTED_CHUNKS
                and missing == 0
                and report.get("dup_chunk_ids", 0) == 0
                and report.get("wrong_dim", 0) == 0
                and report.get("wrong_model", 0) == 0
                and report.get("null_embeddings", 0) == 0
            )
            log(f"[final] verify status={'PASS' if ok else 'INCOMPLETE'}")
            return 0 if ok else 2
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
