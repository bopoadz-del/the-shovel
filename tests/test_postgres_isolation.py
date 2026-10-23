"""Isolation TRUNCATE must retry Postgres deadlocks, not fail setup."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy.exc import OperationalError

from tests.conftest import _is_postgres_deadlock, _truncate_postgres_tables


class _FakeOrig:
    def __init__(self, sqlstate: str | None = None, message: str = ""):
        self.sqlstate = sqlstate
        self.pgcode = sqlstate
        self.args = (message,)

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def _deadlock_exc() -> OperationalError:
    orig = _FakeOrig("40P01", "deadlock detected")
    return OperationalError("TRUNCATE TABLE users", {}, orig)


def test_is_postgres_deadlock_reads_sqlstate():
    assert _is_postgres_deadlock(_deadlock_exc()) is True
    assert _is_postgres_deadlock(OperationalError("SELECT 1", {}, Exception("boom"))) is False


def test_truncate_retries_deadlock_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Conn:
        def execute(self, _sql):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _deadlock_exc()

    @contextmanager
    def _begin():
        yield _Conn()

    engine = type("E", (), {"begin": staticmethod(_begin)})()
    monkeypatch.setattr("time.sleep", lambda _s: None)
    _truncate_postgres_tables(engine, ("users",), attempts=5)
    assert calls["n"] == 3


def test_truncate_reraises_non_deadlock():
    class _Conn:
        def execute(self, _sql):
            raise OperationalError("TRUNCATE TABLE users", {}, Exception("disk full"))

    @contextmanager
    def _begin():
        yield _Conn()

    engine = type("E", (), {"begin": staticmethod(_begin)})()
    with pytest.raises(OperationalError, match="disk full"):
        _truncate_postgres_tables(engine, ("users",), attempts=3)
