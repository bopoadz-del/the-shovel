"""A session for a URL always runs on the engine ``get_engine`` returns for it.

The engine cache holds eight URLs. When a process has used more, it evicts
one and builds a new engine for it on the next call; a session factory cached
by URL kept the evicted engine, so one URL ran on two pools and a listener on
``get_engine()`` saw none of the store's statements.
"""
from __future__ import annotations

from app.core import db


def test_sessions_follow_the_current_engine_after_eviction(tmp_path):
    url = f"sqlite:///{tmp_path / 'own.db'}"
    first = db._engine_for_url(url)
    assert db._session_factory_for_url(url)().get_bind() is first
    for index in range(db._engine_for_url.cache_info().maxsize + 1):
        db._engine_for_url(f"sqlite:///{tmp_path / f'other{index}.db'}")
    current = db._engine_for_url(url)
    assert current is not first
    assert db._session_factory_for_url(url)().get_bind() is current
