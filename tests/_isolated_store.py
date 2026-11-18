"""An isolated test store, given the documents table production always has.

A ``VectorStore`` built on its own test database has only the chunks table.
Such a store cannot compute the chat-session scope, so every project-wide
read of it returns no candidate (``VectorStore._session_hidden_ids`` fails
closed). A test of ranking, hydration or page plumbing on an isolated store
creates the (empty) projects and documents tables next to its chunks: no
document row means no chat-session file, as in a production database whose
project holds none.
"""
from __future__ import annotations


def give_documents_table(store):
    from app.core.db import _engine_for_url
    from app.core.models import Document, Project

    engine = _engine_for_url(store._database_url)
    try:
        for table in (Project.__table__, Document.__table__):
            table.create(engine, checkfirst=True)
    finally:
        engine.dispose()
    store._visibility_ready = None
    return store
