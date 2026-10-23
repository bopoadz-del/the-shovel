# Reasoning Engine — Plan 2: Session State Store

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.
> Part of the Reasoning Engine — see `2026-05-20-reasoning-engine-INDEX.md`. Independent of Plans 1/1b/3.

**Goal:** A per-session project store that persists across conversation turns — documents, activities, CPM results, artifacts, history, generated-code cache — so a follow-up question ("now compress B1") can build on prior state.

**Architecture:** `app/schemas/project_session.py` holds the `ProjectSession` Pydantic model. `app/core/session_store.py` holds a `SessionStore` interface with two interchangeable backends: `InMemorySessionStore` (dict — dev/test, works now) and `RedisSessionStore` (production). A `get_session_store()` factory picks the backend from the environment. TTL expiry on both.

**Tech Stack:** Python 3.11, Pydantic v2. Redis backend uses the `redis` package **only if `REDIS_URL` is set** — it is an optional dependency; the in-memory backend needs nothing.

**Run tests:** `& .venv\Scripts\python.exe -m pytest <path> -q` from `C:\Users\example\The_Fork`.

---

### Task 1: Session schemas

**Files:**
- Create: `app/schemas/project_session.py`
- Test: `tests/test_session_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_session_store.py`:

```python
"""Tests for the session state store — Reasoning Engine Plan 2."""

import pytest

from app.schemas.project_session import Artifact, Message, ProjectSession


def test_new_session_is_empty():
    s = ProjectSession.new("sess1")
    assert s.id == "sess1"
    assert s.data == {} and s.history == [] and s.artifacts == []
    assert s.created_at and s.updated_at


def test_message_and_artifact_models():
    m = Message(role="user", content="hi", ts="2026-05-20T00:00:00Z")
    assert m.role == "user"
    a = Artifact(name="schedule.xlsx", path="/data/x.xlsx", type="excel")
    assert a.type == "excel"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_session_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schemas.project_session'`

- [ ] **Step 3: Write the schemas**

Create `app/schemas/project_session.py`:

```python
"""Session-state schemas — Reasoning Engine Plan 2."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Message(BaseModel):
    role: str       # 'user' | 'assistant'
    content: str
    ts: str = Field(default_factory=_now)


class Artifact(BaseModel):
    name: str
    path: str
    type: str       # 'excel' | 'chart' | 'pdf' | 'json' | ...


class ProjectSession(BaseModel):
    """All state for one project conversation. JSON-serialisable throughout."""
    id: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    # free-form computed state: activities, cpm_results, manpower, wbs, ...
    data: Dict[str, Any] = Field(default_factory=dict)
    history: List[Message] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)
    code_cache: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def new(cls, session_id: str) -> "ProjectSession":
        return cls(id=session_id)

    def touch(self) -> None:
        self.updated_at = _now()

    def add_message(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))
        self.touch()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_session_store.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/project_session.py tests/test_session_store.py
git commit -m "feat(session): ProjectSession schema (reasoning engine plan 2)"
```

---

### Task 2: SessionStore interface + in-memory backend

**Files:**
- Create: `app/core/session_store.py`
- Test: `tests/test_session_store.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session_store.py`:

```python
from app.core.session_store import InMemorySessionStore


def test_get_or_create_is_idempotent():
    store = InMemorySessionStore()
    a = store.get_or_create("s1")
    b = store.get_or_create("s1")
    assert a.id == b.id == "s1"
    assert store.get("s1") is not None


def test_get_missing_returns_none():
    assert InMemorySessionStore().get("nope") is None


def test_save_persists_mutations():
    store = InMemorySessionStore()
    s = store.get_or_create("s1")
    s.data["activities"] = [{"id": "A"}]
    s.add_message("user", "create a schedule")
    store.save(s)
    reloaded = store.get("s1")
    assert reloaded.data["activities"] == [{"id": "A"}]
    assert reloaded.history[0].content == "create a schedule"


def test_delete_removes_session():
    store = InMemorySessionStore()
    store.get_or_create("s1")
    assert store.delete("s1") is True
    assert store.get("s1") is None
    assert store.delete("s1") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_session_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.session_store'`

- [ ] **Step 3: Write the interface + in-memory backend**

Create `app/core/session_store.py`:

```python
"""Session state store — Reasoning Engine Plan 2.

Two interchangeable backends behind one interface: InMemorySessionStore
(dev/test) and RedisSessionStore (production, added in Task 4).
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from app.schemas.project_session import ProjectSession

DEFAULT_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "14400"))  # 4 hours


class SessionStore(ABC):
    """Persist and retrieve ProjectSession objects, with TTL expiry."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[ProjectSession]: ...

    @abstractmethod
    def save(self, session: ProjectSession) -> None: ...

    @abstractmethod
    def delete(self, session_id: str) -> bool: ...

    def get_or_create(self, session_id: str) -> ProjectSession:
        existing = self.get(session_id)
        if existing is not None:
            return existing
        session = ProjectSession.new(session_id)
        self.save(session)
        return session


class InMemorySessionStore(SessionStore):
    """Process-local dict backend. Fine for dev/test and single-process runs."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._data: Dict[str, Tuple[ProjectSession, float]] = {}

    def get(self, session_id: str) -> Optional[ProjectSession]:
        entry = self._data.get(session_id)
        if entry is None:
            return None
        session, expires_at = entry
        if time.time() > expires_at:
            del self._data[session_id]
            return None
        return session

    def save(self, session: ProjectSession) -> None:
        session.touch()
        self._data[session.id] = (session, time.time() + self._ttl)

    def delete(self, session_id: str) -> bool:
        return self._data.pop(session_id, None) is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_session_store.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add app/core/session_store.py tests/test_session_store.py
git commit -m "feat(session): SessionStore interface + in-memory backend"
```

---

### Task 3: TTL expiry

**Files:**
- Test: `tests/test_session_store.py` (TTL logic was written in Task 2)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session_store.py`:

```python
import time as _time


def test_session_expires_after_ttl():
    store = InMemorySessionStore(ttl_seconds=1)
    store.get_or_create("s1")
    assert store.get("s1") is not None
    _time.sleep(1.1)
    assert store.get("s1") is None          # expired and evicted


def test_save_refreshes_ttl():
    store = InMemorySessionStore(ttl_seconds=2)
    s = store.get_or_create("s1")
    _time.sleep(1.2)
    store.save(s)                            # resets the 2s window
    _time.sleep(1.2)
    assert store.get("s1") is not None       # still alive — refreshed
```

- [ ] **Step 2: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_session_store.py -q`
Expected: PASS (8 passed) — TTL was implemented in Task 2's `InMemorySessionStore`. If a test fails, fix the expiry logic in `get`/`save` before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_session_store.py
git commit -m "test(session): TTL expiry coverage"
```

---

### Task 4: Redis backend + factory

**Files:**
- Modify: `app/core/session_store.py` (add `RedisSessionStore` + `get_session_store`)
- Test: `tests/test_session_store.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session_store.py`:

```python
import os as _os

from app.core.session_store import get_session_store


def test_factory_returns_in_memory_when_no_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    store = get_session_store()
    from app.core.session_store import InMemorySessionStore as _IM
    assert isinstance(store, _IM)


@pytest.mark.skipif(not _os.getenv("REDIS_URL"), reason="no REDIS_URL configured")
def test_redis_backend_roundtrip():
    from app.core.session_store import RedisSessionStore
    store = RedisSessionStore(_os.getenv("REDIS_URL"), ttl_seconds=60)
    s = store.get_or_create("redis_test_sess")
    s.data["x"] = 1
    store.save(s)
    assert store.get("redis_test_sess").data["x"] == 1
    store.delete("redis_test_sess")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_session_store.py -q`
Expected: FAIL — `ImportError: cannot import name 'get_session_store'` (the Redis test is skipped when `REDIS_URL` is unset)

- [ ] **Step 3: Add the Redis backend and factory**

Append to `app/core/session_store.py`:

```python
class RedisSessionStore(SessionStore):
    """Redis backend. Sessions are JSON blobs keyed `session:<id>` with a TTL.

    Requires the `redis` package and a reachable server. Only instantiated
    when REDIS_URL is set — see get_session_store().
    """

    _PREFIX = "session:"

    def __init__(self, redis_url: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        import redis  # imported lazily so the dependency is optional
        self._ttl = ttl_seconds
        self._client = redis.from_url(redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"{self._PREFIX}{session_id}"

    def get(self, session_id: str) -> Optional[ProjectSession]:
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        return ProjectSession.model_validate_json(raw)

    def save(self, session: ProjectSession) -> None:
        session.touch()
        self._client.set(
            self._key(session.id), session.model_dump_json(), ex=self._ttl
        )

    def delete(self, session_id: str) -> bool:
        return self._client.delete(self._key(session_id)) > 0


def get_session_store() -> SessionStore:
    """Pick a backend: Redis when REDIS_URL is set and reachable, else
    in-memory. Falls back to in-memory if Redis cannot be reached."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            store = RedisSessionStore(redis_url)
            store._client.ping()
            return store
        except Exception:
            pass  # fall through to in-memory
    return InMemorySessionStore()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_session_store.py -q`
Expected: PASS — 10 passed, 1 skipped (the Redis round-trip skips without `REDIS_URL`)

- [ ] **Step 5: Commit**

```bash
git add app/core/session_store.py tests/test_session_store.py
git commit -m "feat(session): Redis backend + get_session_store factory"
```

---

### Task 5: Regression check

**Files:** none — verification only.

- [ ] **Step 1: Run the full suite**

Run: `& .venv\Scripts\python.exe -m pytest --ignore=tests/browser -q`
Expected: PASS — 284 passed, 86 skipped (Plan 1b's 274 passed + 10 new here; +1 skipped Redis test)

- [ ] **Step 2: Commit** — nothing to commit unless a regression was fixed (Tasks 1–4 already committed).

---

## Self-Review

**Spec coverage** (Reasoning Engine §5.2):
- `get(session_id)` / `save` / `delete` / `get_or_create` → Tasks 2 & 4 ✅
- Backends: Redis + in-memory, same interface → Tasks 2 & 4 ✅
- Session keys (`documents`, `activities`, `cpm_results`, …) → held in `data` dict, plus typed `history`/`artifacts`/`code_cache` → Task 1 ✅
- JSON-serialisable via Pydantic `model_dump_json` → used by the Redis backend, Task 4 ✅
- TTL default 4 h, configurable (`SESSION_TTL_SECONDS`) → Tasks 2 & 3 ✅

**Out of scope (noted):** the 50 MB session-size limit (spec §5.2) — deferred; enforce it in Plan 5 when the reasoner writes large activity lists, since that is where size is known. `redis` is **not** added to `requirements.txt` here — it stays optional; install it only when deploying with Redis.

**Placeholder scan:** none. **Type consistency:** `SessionStore.get -> Optional[ProjectSession]`, `save(ProjectSession) -> None`, `delete -> bool`, `get_or_create -> ProjectSession`; both backends implement the same ABC; `get_session_store() -> SessionStore`. Every task starts with a failing (or red-then-confirmed) test.

---

**Plan 2 complete.** Next: Plan 3 (Sandbox).
