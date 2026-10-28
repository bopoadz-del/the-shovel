"""SQLAlchemy ORM models for the unified The Fork schema."""

from __future__ import annotations

import datetime
import os
import re
import uuid
import logging
import threading
from typing import Optional, Type

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    desc,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

def _configured_embedding_dim() -> int:
    """Width of the configured embedder: ``get_embedder().dim``.

    The ONE source for every width this module does not get passed explicitly
    (``EMBEDDING_DIM`` and the legacy ``EmbeddingVector``). Resolved on use, not
    at import, so importing the models (alembic, workers) never loads a model.
    """
    from app.core.rag.embeddings import get_embedder

    return int(get_embedder().dim)


def __getattr__(name: str):
    # ``models.EMBEDDING_DIM`` is DERIVED from the configured embedder on every
    # read (PEP 562). There is no hand-typed width in this module any more.
    if name == "EMBEDDING_DIM":
        return _configured_embedding_dim()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_embedding_vector(dim):
    """Return a TypeDecorator that stores ``dim``-dim float32 vectors.

    ``dim`` is an int, or a zero-arg callable resolved when the column is first
    compiled for a dialect (used by the legacy ``EmbeddingVector``, whose width
    follows the configured embedder).

    PostgreSQL uses pgvector ``vector(dim)``; SQLite stores raw float32 bytes.
    The dimension is bound at factory time so each namespace/table can have the
    correct width for its embedder.
    """

    class DynamicEmbeddingVector(TypeDecorator):
        impl = LargeBinary
        cache_ok = True

        def load_dialect_impl(self, dialect):
            if dialect.name == "postgresql":
                return dialect.type_descriptor(Vector(dim() if callable(dim) else dim))
            return dialect.type_descriptor(LargeBinary())

        def process_bind_param(self, value, dialect):
            if value is None:
                return None
            arr = np.asarray(value, dtype=np.float32)
            if dialect.name == "postgresql":
                return arr.tolist()
            return arr.tobytes()

        def process_result_value(self, value, dialect):
            if value is None:
                return None
            if dialect.name == "postgresql":
                return np.asarray(value, dtype=np.float32)
            return np.frombuffer(value, dtype=np.float32)

    return DynamicEmbeddingVector


class EmbeddingVector(make_embedding_vector(_configured_embedding_dim)):
    """Vector type of the original ``chunks`` table; its width follows the
    configured embedder (``get_embedder().dim``), never a constant."""

    pass


# Registry of namespaced chunk classes, filled ONLY by the deterministic factory
# ``_rag_chunk_class_factory`` and keyed by ``(namespace, dim)``. A namespace
# owns one table (``chunks_<ns>``) and so exactly one key; asking for a second
# width or a second embedding model on the same namespace RAISES (never a
# reuse of the existing class: that hands back a vector(N) column for a vector(M)
# caller). The configured namespace is registered at import, at the bottom of
# this module, so the active table is on ``Base.metadata`` before any caller.
_RAG_CHUNK_CLASSES: dict[tuple[str, int], type] = {}
_RAG_CHUNK_NAMESPACE_KEY: dict[str, tuple[str, int]] = {}
_RAG_CHUNK_LOCK = threading.RLock()



#: RAG namespaces are interpolated into raw SQL table identifiers, which cannot
#: be parameterised. Restrict them to a safe identifier charset. Matched with
#: ``fullmatch`` — NOT ``match`` + ``$`` — because ``$`` also matches just before
#: a trailing newline, so ``"foo\n"`` would slip a newline into the identifier.
_SAFE_NAMESPACE_RE = re.compile(r"[A-Za-z0-9_]+")


def rag_chunk_table_name(namespace: str) -> str:
    """Table name for a RAG vector namespace.

    The legacy ``chunks`` table uses namespace ``""``. Every other namespace
    gets ``chunks_<namespace>``. The old contaminated table is retired in
    place — new writes go to namespaced tables only.

    This is the single chokepoint for the RAG chunk table identifier, which is
    interpolated into raw SQL (``FROM {table}``) at ~15 call sites. Namespaces
    are internal today, but validate the identifier here so a malformed or
    hostile namespace can never produce an injectable table name — defence in
    depth for the dynamic-``FROM`` sites flagged in the repo audit.
    """
    if not namespace:
        return "chunks"
    if not _SAFE_NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(
            f"unsafe RAG namespace {namespace!r}: only [A-Za-z0-9_] allowed"
        )
    return f"chunks_{namespace}"


def rag_chunk_class_for(namespace: str):
    """The mapped class already registered for ``namespace``, or None.

    Lets a caller reuse whatever width a namespace was opened at instead of
    guessing one and tripping the dim-conflict guard in
    ``make_rag_chunk_class``. The legacy namespace is always ``RagChunk``.
    """
    if namespace == "":
        return RagChunk
    key = _RAG_CHUNK_NAMESPACE_KEY.get(namespace)
    return _RAG_CHUNK_CLASSES.get(key) if key is not None else None


class RagNamespaceIdentityConflict(ValueError):
    """A namespace is already registered at another width or embedding model."""

    def __init__(self, namespace: str, registered: dict, requested: dict):
        self.namespace = namespace
        self.registered = registered
        self.requested = requested
        super().__init__(
            f"namespace {namespace!r} is already mapped at dim "
            f"{registered['dim']} / model {registered['model']!r}; cannot remap "
            f"it to dim {requested['dim']} / model {requested['model']!r}. A "
            f"namespace has one table ({rag_chunk_table_name(namespace)}) and "
            f"therefore one embedding identity. Use a different namespace, or "
            f"change RAG_VECTOR_NAMESPACE together with the embedder."
        )


def make_rag_chunk_class(
    namespace: str,
    dim: int,
    model_name: str,
    normalized: bool = True,
) -> type:
    """Return the SQLAlchemy model class for ``chunks_<namespace>``.

    Thin front to the deterministic factory: the same ``(namespace, dim)`` and
    model always returns the same class, and a request whose ``dim`` or
    ``model_name`` differs from the registered one RAISES
    :class:`RagNamespaceIdentityConflict` (a ``ValueError``). The width guard is
    never bypassed by reusing whatever class is already bound to the table.

    The legacy namespace (empty string) is the static ``RagChunk`` class.
    """
    if namespace == "":
        return RagChunk
    return _rag_chunk_class_factory(namespace, int(dim), model_name, normalized)


def _rag_chunk_class_factory(
    namespace: str, dim: int, model_name: str, normalized: bool = True,
) -> type:
    """The ONE place a namespaced chunk class is built, keyed ``(namespace, dim)``.

    Deterministic: the class name, table name, columns and identity are a pure
    function of the key and model, so building namespaces in any order yields
    the same ``Base.metadata``. Each key is built once, under a lock.
    """
    requested = {"model": model_name, "dim": dim, "normalized": normalized}
    with _RAG_CHUNK_LOCK:
        key = _RAG_CHUNK_NAMESPACE_KEY.get(namespace)
        if key is not None:
            cls = _RAG_CHUNK_CLASSES[key]
            registered = dict(cls.embedding_identity)
            if registered != requested:
                raise RagNamespaceIdentityConflict(namespace, registered, requested)
            return cls
        cls = _build_rag_chunk_class(namespace, dim, model_name, normalized)
        _RAG_CHUNK_CLASSES[(namespace, dim)] = cls
        _RAG_CHUNK_NAMESPACE_KEY[namespace] = (namespace, dim)
        return cls


def _build_rag_chunk_class(
    namespace: str, dim: int, model_name: str, normalized: bool,
) -> type:
    table_name = rag_chunk_table_name(namespace)
    vector_type = make_embedding_vector(dim)
    class_name = f"RagChunk_{namespace or 'legacy'}_{dim}"

    def _repr(self):
        return f"<{class_name}(chunk_id={self.chunk_id!r})>"

    attrs = {
        "__tablename__": table_name,
        "__table_args__": (
            UniqueConstraint(
                "project_id", "doc_id", "chunk_index",
                name=f"uq_{table_name}_project_doc_index",
            ),
            Index(f"idx_{table_name}_project", "project_id"),
            Index(f"idx_{table_name}_doc", "project_id", "doc_id"),
            Index(f"idx_{table_name}_klayer", "knowledge_layer"),
        ),
        "__repr__": _repr,
        "chunk_id": mapped_column(String, primary_key=True),
        "project_id": mapped_column(String, nullable=False),
        "doc_id": mapped_column(String, nullable=False),
        "chunk_index": mapped_column(Integer, nullable=False),
        "text": mapped_column(Text, nullable=False),
        "embedding": mapped_column(vector_type, nullable=False),
        "created_at": mapped_column(String, nullable=False),
        # Layered RAG (docs/rag-deployment-plan.md) — persisted L1/L2A/L2B/L3
        # layer; distinct from the Chunk dataclass STEP 0 isolation tag.
        # Nullable so pre-migration rows and RAG_LAYERED=off read as unlayered.
        "knowledge_layer": mapped_column(String, nullable=True),
        "authority": mapped_column(String, nullable=True),
        # 1-based page of the source PDF where the chunk's text starts
        # (Alembic 0021). NULL for non-PDF sources and older rows.
        "page": mapped_column(Integer, nullable=True),
        # Embedding identity — stamped on every row and verified at startup.
        "embedding_model": mapped_column(String, nullable=False, default=model_name),
        "embedding_dim": mapped_column(Integer, nullable=False, default=dim),
        "embedding_normalized": mapped_column(Boolean, nullable=False, default=normalized),
        "embedding_identity": {
            "model": model_name,
            "dim": dim,
            "normalized": normalized,
        },
    }

    return type(class_name, (Base,), attrs)


class Base(DeclarativeBase):
    pass


class User(Base):
    """users table — see the_fork_schema.sql."""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    salt: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(String, nullable=False, default="user")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Has the address been proved to belong to whoever registered it?
    # Defaults FALSE so a new signup is unverified until they follow the
    # emailed link. Migration 0015 backfills TRUE for pre-existing rows —
    # they predate the feature and must not be locked out — and
    # ``_bootstrap_first_user`` sets it explicitly, because a fresh install
    # creates that admin AFTER the backfill has already run.
    email_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # Admin-set: this user's turns take driver mode (F-DRIVER Phase B) when
    # the platform runs DRIVER_MODE=request. Off for everyone by default.
    driver_mode: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_text("FALSE"),
    )


class Project(Base):
    """projects table — see the_fork_schema.sql."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    client: Mapped[str | None] = mapped_column(String, nullable=True)
    # Confirmed site location (city/site) — daily_site_report reads its weather
    # from here, never from the chat message (F4). Nullable: unset → no weather.
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    # The project's profile (JSON object: discipline, contract form, phase,
    # governing codes ...), filled in once by an admin. Driver mode gives it
    # to the model every turn. Nullable: no profile on record.
    profile: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The project's identity set (app.core.project_identity), JSON text:
    # written when the record, profile or documents change; read per turn.
    identity: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    aconex_connected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        default="system",
    )
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # PR A — admin-approved-projects: true for user-created + admin-
    # approved-from-Drive rows (the projects users should see); false
    # for detected-but-not-yet-approved rows the admin still needs to
    # rule on. Defaults to true so the column is safe to add to legacy
    # rows + safe to omit on inserts from older code paths.
    is_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=sa_text("TRUE"),
    )
    # PR B — discriminates how the project came into being. Admin page
    # filters on origin='admin_drive_approved' so user-created rows
    # (personal accounts) don't show in the admin's approved list.
    # Values: 'user_create' | 'admin_drive_approved' | 'user_drive_import'.
    origin: Mapped[str] = mapped_column(
        String(32), nullable=False, default="user_create",
        server_default=sa_text("'user_create'"),
    )
    # Layered RAG (Stage 5): hide a row from the sidebar WITHOUT archiving it, so
    # a RAG corpus / general-knowledge / eval project stays fully RETRIEVABLE
    # while disappearing from the user's project list. Distinct from status=
    # 'archived' (which also removes the project from retrieval). Defaults false.
    hidden_from_sidebar: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_text("FALSE"),
    )


class Document(Base):
    """documents table — see the_fork_schema.sql."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "doc_role IN ("
            "'baseline_schedule', 'daily_report', 'weekly_report', 'other'"
            ")",
            name="ck_documents_doc_role",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    original_name: Mapped[str] = mapped_column(String, nullable=False)
    stored_as: Mapped[str | None] = mapped_column(String, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    doc_type: Mapped[str] = mapped_column(String, nullable=False, default="document")
    doc_role: Mapped[str] = mapped_column(String, nullable=False, default="other")
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    uploaded_at: Mapped[str] = mapped_column(String, nullable=False)
    content_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    # ``metadata`` is a reserved word in SQLAlchemy Declarative; the DB column
    # is still named ``metadata`` (JSONB on Postgres), but the ORM attribute is
    # ``metadata_`` to avoid clashing with ``Base.metadata``.
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True
    )
    # ── ingest ledger (migration 0016) ────────────────────────────────────
    # A row here proves a file was REGISTERED. These prove what came out of
    # it. Without them "ingested" and "usable" are the same word, which is how
    # 1,873 one-chunk drawing sheets passed as INDEXED.
    ingest_status: Mapped[str] = mapped_column(
        String, nullable=False, default="UNVERIFIED"
    )
    ingest_status_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Drive's own content token. Distinct from ``content_sha256`` on purpose:
    # Drive publishes MD5, this codebase stores SHA-256, and comparing the two
    # can only ever report "changed". Resume compares md5-to-md5; the sha stays
    # the archive identity and is never a skip key.
    drive_md5: Mapped[str | None] = mapped_column(String, nullable=True)
    last_verified_at: Mapped[str | None] = mapped_column(String, nullable=True)
    # ── retrieval visibility (migration 0017) ─────────────────────────────
    # Supersede, don't delete. A stale extract stays on the row; retrieval
    # hides it when retrieval_visible is false. Reversible by flipping the
    # flag. Hard delete of a documents row remains forbidden.
    superseded_by: Mapped[str | None] = mapped_column(
        String, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    retrieval_visible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=sa_text("TRUE"),
    )
    # Extractor family that produced the current chunks. Tied to the
    # content-control fix (#550 / 8535199-sdt).
    extractor_version: Mapped[str | None] = mapped_column(String, nullable=True)


class ProjectFact(Base):
    """project_facts table — see the_fork_schema.sql."""

    __tablename__ = "project_facts"
    __table_args__ = (
        UniqueConstraint("project_id", "key", name="uq_project_facts_project_key"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    source_document: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class Workflow(Base):
    """workflows table — see the_fork_schema.sql."""

    __tablename__ = "workflows"
    __table_args__ = (
        Index("idx_workflows_owner_created", "owner_id", "created_at"),
        Index("idx_workflows_project_created", "project_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    owner_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    steps: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class Conversation(Base):
    """conversations table — see the_fork_schema.sql."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conversations_agent_updated", "agent_name", "updated_at"),
        Index("idx_conversations_project_updated", "project_id", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # Who wrote the session. NULL means the row predates owner tracking
    # and stays visible only to the project row owner.
    owner_id: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class Message(Base):
    """messages table — see the_fork_schema.sql."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="ck_messages_role",
        ),
        Index("idx_messages_conv", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Provenance record of an assistant answer (JSON): where each figure and
    # cited fact came from. Migration 0023.
    provenance: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocIndex(Base):
    """doc_index table — per-project text index blob (see the_fork_schema.sql)."""

    __tablename__ = "doc_index"

    project_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    index_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class DocIndexEntry(Base):
    """doc_index_entries table — one row per document (or skipped file) of a
    project's text index. ``index_json`` on ``doc_index`` is the header only;
    see ``app/core/doc_index.py`` (storage section) for why the entries left
    the blob."""

    __tablename__ = "doc_index_entries"

    project_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    entry_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        Index("idx_doc_index_entries_project", "project_id", "seq"),
    )


class UsageRun(Base):
    """runs table — see the_fork_schema.sql (usage tracker)."""

    __tablename__ = "runs"
    __table_args__ = (Index("idx_runs_user_created", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String, nullable=True)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class HydrationRun(Base):
    """hydration_runs table — see the_fork_schema.sql."""

    __tablename__ = "hydration_runs"
    __table_args__ = (
        CheckConstraint(
            "scope IN ('project', 'global')",
            name="ck_hydration_runs_scope",
        ),
        Index("idx_hydration_lookup", "scope", "project_id", desc("run_date")),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_date: Mapped[str] = mapped_column(String, nullable=False)
    scope: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
    )
    summary_md: Mapped[str] = mapped_column(Text, nullable=False)
    facts_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class RagBudget(Base):
    """rag_budget table — see the_fork_schema.sql."""

    __tablename__ = "rag_budget"

    day: Mapped[str] = mapped_column(String, primary_key=True)
    consumed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class RagChunk(Base):
    """chunks table — RAG vector store (see the_fork_schema.sql).

    ORM omits FK constraints so SQLite test DBs accept arbitrary
    project/doc ids; Alembic applies FK on PostgreSQL deployments.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "doc_id", "chunk_index", name="uq_chunks_project_doc_index"
        ),
        Index("idx_chunks_project", "project_id"),
        Index("idx_chunks_doc", "project_id", "doc_id"),
        Index("idx_chunks_knowledge_layer", "knowledge_layer"),
    )

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    # Nullable: the legacy chunks FK is ON DELETE SET NULL (Alembic 0012), so a
    # chunk survives its workspace's deletion as an orphan rather than being
    # cascade-deleted. Knowledge is decoupled from the project row.
    project_id: Mapped[str | None] = mapped_column(String, nullable=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[np.ndarray] = mapped_column(EmbeddingVector(), nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Layered RAG (docs/rag-deployment-plan.md). ``knowledge_layer`` is the
    # persisted L1/L2A/L2B/L3 layer of the chunk; distinct from the per-query
    # STEP 0 isolation tag on the Chunk dataclass (own/master_corpus/gk), which
    # the retriever recomputes every search. Nullable so pre-migration rows and
    # the RAG_LAYERED=off path read as unlayered. Values: app/core/rag/layers.py.
    knowledge_layer: Mapped[str | None] = mapped_column(String, nullable=True)
    authority: Mapped[str | None] = mapped_column(String, nullable=True)
    # 1-based page of the source PDF where the chunk's text starts (Alembic
    # 0021). NULL for non-PDF sources and rows indexed before pages existed.
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # NOTE: the hybrid BM25 leg uses a ``text_search`` tsvector column on
    # PostgreSQL — added by Alembic migration 0003 as GENERATED ALWAYS
    # AS STORED, with a GIN index. It is intentionally NOT declared on
    # the ORM model. Reason: a postgresql.TSVECTOR column with a
    # Computed expression breaks SQLite's
    # ``RagChunk.__table__.create(checkfirst=True)`` path (used by the
    # SQLite test/dev fallback to bootstrap a fresh DB), and SQLite's
    # BM25 leg goes through FTS5 anyway (see vector_store._ensure_fts5_sqlite).
    # vector_store._bm25_postgres references the column via raw SQL
    # (``c.text_search @@ plainto_tsquery(...)``), bypassing the ORM.


class AgentFact(Base):
    """agent_facts table — see the_fork_schema.sql.

    Project-less scope is stored as '' (NOT NULL per schema); the public API
    also uses ''.
    """

    __tablename__ = "agent_facts"
    __table_args__ = (
        UniqueConstraint(
            "agent_name", "project_id", "key", name="uq_agent_facts_scope_key"
        ),
        Index("idx_agent_facts_agent_project", "agent_name", "project_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    conversation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class IngestionJob(Base):
    """ingestion_jobs table — tracks async document ingestion status.

    Written by the upload endpoint and updated by the arq worker.
    """

    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    document_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending"
    )
    chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AuditEvent(Base):
    """audit_events — where an audit entry goes when the audit file cannot
    take it (app.core.audit): the entry is kept, never dropped."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("idx_audit_events_project_ts", "project_id", "ts"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[str] = mapped_column(String, nullable=False)
    event: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str | None] = mapped_column(String, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)


# ── Import-time registration of the configured namespace ──────────────────────
def register_configured_rag_namespace() -> Optional[type]:
    """Register ``chunks_<RAG_VECTOR_NAMESPACE>`` on ``Base.metadata`` through
    the factory, at the configured embedder's identity. Runs once at import, so
    the serving table never depends on which caller (or test shard) reaches
    ``make_rag_chunk_class`` first. The legacy namespace ``""`` is ``RagChunk``.

    If the embedder cannot be constructed here, nothing is registered and the
    failure is logged; ``get_store`` raises the same error loudly on first use.
    """
    namespace = os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
    if not namespace:
        return RagChunk
    try:
        from app.core.rag.embeddings import get_embedder

        ident = get_embedder().identity
    except Exception:  # noqa: BLE001 — reported again, loudly, by get_store()
        logging.getLogger(__name__).error(
            "import-time RAG namespace registration skipped: embedder "
            "unavailable", exc_info=True,
        )
        return None
    return make_rag_chunk_class(
        namespace, int(ident["dim"]), ident["model"], bool(ident["normalized"]),
    )


register_configured_rag_namespace()
