"""Stage 1 / Task 4 — knowledge outlives workspaces (decouple from projects).

The legacy ``chunks`` table historically had ``project_id ... ON DELETE
CASCADE`` — the coupling that made "delete a project" destroy its RAG. Prod
already writes to the namespaced ``chunks_v2`` table, which the ORM builds
with NO foreign key at all, so the live corpus was never FK-cascaded. This
task softens the legacy table's FK to ``SET NULL`` as belt-and-suspenders and
makes ``RagChunk.project_id`` nullable to match, so a chunk row can survive its
project's deletion (its text/embedding/knowledge_layer/authority intact).
"""
import os
import pytest

from app.core.models import RagChunk


def test_static_ragchunk_project_id_is_nullable():
    by = {c.name: c for c in RagChunk.__table__.columns}
    assert by["project_id"].nullable is True, (
        "legacy chunks.project_id must be nullable so ON DELETE SET NULL can "
        "orphan (not delete) a chunk when its workspace is removed")


def test_pg_chunks_project_fk_is_set_null():
    """Postgres-only: the live legacy chunks FK must be SET NULL, not CASCADE.

    Skips on the local SQLite dev DB, which does not model the FK the same way.
    """
    from app.core.db import get_database_url
    url = get_database_url()
    if not url.startswith("postgres"):
        pytest.skip("cascade delete_rule is Postgres-specific; SQLite dev skips")
    from sqlalchemy import create_engine, text
    eng = create_engine(url)
    with eng.connect() as c:
        rule = c.execute(text(
            "SELECT rc.delete_rule FROM information_schema.referential_constraints rc "
            "WHERE rc.constraint_name = 'chunks_project_id_fkey'"
        )).scalar()
    assert rule in (None, "SET NULL"), f"expected SET NULL, got {rule!r}"
