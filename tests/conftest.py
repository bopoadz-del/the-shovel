"""Pytest configuration and fixtures."""

import pytest
import os
import uuid
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env before test collection so env-gated tests (e.g. the live
# LLM acceptance tests) see keys placed in .env. The app loads
# .env itself; conftest does it too so `skipif`s evaluated at collection time
# pick the key up without needing it exported in the shell.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# Disable per-caller rate limiting for the test suite — the full suite makes
# far more than a minute's quota of requests under one identity (the dev key).
# The rate limiter's own tests re-enable it explicitly via monkeypatch.
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"

# Isolate the suite's DATA_DIR so tests never write to the live ./data/ — the
# projects store, doc index, agent memory, and upload dir all derive their
# location from $DATA_DIR. Without this, every test that creates a project
# landed a row in the user's live projects.db (we found 2089 leaked rows
# from prior runs and 1806 synthetic `{hex}_name.pdf` fixture files). The
# override happens at collection time, BEFORE app modules import, so the
# resolved path on first read is the temp directory.
import tempfile as _tempfile
from tests._dev_key import DEV_KEY
_TEST_DATA_DIR = _tempfile.mkdtemp(prefix="thefork-tests-")
# Force the override — start-local.sh sets DATA_DIR=$PWD/data so a developer
# running pytest in the same shell after the server would otherwise inherit
# the live data dir.
_existing = os.environ.get("DATA_DIR", "")
if not _existing or os.path.abspath(_existing) == os.path.abspath("./data"):
    os.environ["DATA_DIR"] = _TEST_DATA_DIR

# Default suite uses isolated SQLite (the_fork.db under temp DATA_DIR).
# CI job test-postgres sets PYTEST_USE_POSTGRES=1 and DATABASE_URL explicitly.
if os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() not in ("1", "true", "yes"):
    os.environ.pop("DATABASE_URL", None)
else:
    # Fake embedder is 256-dim, aligned with pgvector schema (model2vec default).
    os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")


def _per_worker_postgres_database() -> None:
    """Under pytest-xdist, give each worker its own migrated database.

    The suite TRUNCATEs its tables between tests, so two workers sharing one
    database would wipe each other's rows mid-test. Each worker ``gwN`` gets
    ``<database>_gwN``: created if missing, extensions added, migrated to
    head, and DATABASE_URL pointed at it -- before any app module builds its
    engine. Without xdist (or on SQLite) nothing changes.
    """
    worker = os.getenv("PYTEST_XDIST_WORKER")
    url = os.getenv("DATABASE_URL", "")
    if not worker or not url:
        return
    import subprocess
    import sys

    import psycopg
    from sqlalchemy.engine import make_url

    base = make_url(url)
    name = f"{base.database}_{worker}"

    def plain(u):
        return u.set(drivername="postgresql").render_as_string(hide_password=False)

    with psycopg.connect(plain(base), autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone() is None:
            conn.execute(f'CREATE DATABASE "{name}"')
    target = base.set(database=name)
    with psycopg.connect(plain(target), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    os.environ["DATABASE_URL"] = target.render_as_string(hide_password=False)
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"],
                   cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   env=os.environ.copy(), check=True, capture_output=True)


if os.getenv("PYTEST_XDIST_WORKER"):
    # Each xdist worker writes its own uploads, indexes and SQLite files.
    os.environ["DATA_DIR"] = os.path.join(os.environ["DATA_DIR"], os.environ["PYTEST_XDIST_WORKER"])
    os.makedirs(os.environ["DATA_DIR"], exist_ok=True)
if os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes"):
    _per_worker_postgres_database()

def is_extended_boot() -> bool:
    """Legacy platform boot — extended blocks (drives, MCP, etc.) are loaded."""
    return os.getenv("CEREBRUM_VIRGIN", "true").strip().lower() in ("0", "false", "no")


def is_construction_kit_enabled() -> bool:
    from app.core.domain_kit_loader import active_kit_ids

    return "construction" in active_kit_ids()


def listable_block_count() -> int:
    """Non-container blocks exposed by GET /blocks (matches blocks router)."""
    from app.blocks import get_all_blocks
    from app.core.universal_base import UniversalContainer

    return sum(
        1
        for cls in get_all_blocks().values()
        if not issubclass(cls, UniversalContainer)
    )


_CONSTRUCTION_KIT_SKIP = pytest.mark.skipif(
    not is_construction_kit_enabled(),
    reason="requires CEREBRUM_DOMAIN_KITS=construction",
)
_EXTENDED_BOOT_SKIP = pytest.mark.skipif(
    not is_extended_boot(),
    reason="requires CEREBRUM_VIRGIN=false",
)

# Module-level: pytestmark = construction_kit_markers
construction_kit_markers = [pytest.mark.construction_kit, _CONSTRUCTION_KIT_SKIP]
extended_boot_markers = [pytest.mark.extended_boot, _EXTENDED_BOOT_SKIP]


def requires_construction_kit(func):
    func = pytest.mark.construction_kit(func)
    return _CONSTRUCTION_KIT_SKIP(func)


def requires_extended_boot(func):
    func = pytest.mark.extended_boot(func)
    return _EXTENDED_BOOT_SKIP(func)


def _postgres_test_mode() -> bool:
    return os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")


def _is_postgres_deadlock(exc: BaseException) -> bool:
    """True for a Postgres deadlock (sqlstate 40P01) wrapped by SQLAlchemy."""
    orig = getattr(exc, "orig", None)
    sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    if sqlstate == "40P01":
        return True
    return "deadlock detected" in str(exc).lower()


def _truncate_postgres_tables(engine, tables: tuple[str, ...], *, attempts: int = 8) -> None:
    """TRUNCATE the isolation set, retrying DeadlockDetected.

    Module-scoped TestClient keeps lifespan background work (knowledge-seed)
    on the same DB. TRUNCATE needs AccessExclusiveLock on every table;
    a concurrent SELECT/INSERT can invert lock order and fail setup.
    Retrying is the isolation-side counterpart of not re-entering init_db().
    """
    import time

    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    sql = text(
        "TRUNCATE TABLE " + ", ".join(tables) + " RESTART IDENTITY CASCADE"
    )
    delay = 0.05
    last: OperationalError | None = None
    for attempt in range(attempts):
        try:
            with engine.begin() as conn:
                conn.execute(sql)
            return
        except OperationalError as exc:
            last = exc
            if not _is_postgres_deadlock(exc) or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.8)
    if last is not None:
        raise last


@pytest.fixture(scope="session", autouse=True)
def _init_schema_once():
    """Create the whole unified schema exactly once, before any test,
    TestClient, or background task touches the database.

    Every test module that builds its own `TestClient(app)` re-runs the
    app's FastAPI `lifespan`, which calls each block's `init_db()`. Those
    functions are cheap-idempotent (an early-exit guard skips re-issuing
    DDL once the schema exists for the current DB URL — see
    app/core/{projects,users,agent_memory,doc_index,hydration_store}.py),
    but that guard only helps if the schema was already created by the
    time the first such call races a background task. Calling `init_db()`
    here, first, before collection even runs a test, guarantees the DDL
    happens once, quiescently, with nothing else on the database yet.
    """
    from app.core.projects import init_db as init_projects_db

    init_projects_db()
    from app.core.agent_memory import init_db as init_agent_memory_db

    init_agent_memory_db()
    from app.core.doc_index import init_db as init_doc_index_db

    init_doc_index_db()
    from app.core.hydration_store import init_db as init_hydration_db

    init_hydration_db()
    from app.core.workflows import init_db as init_workflows_db

    init_workflows_db()
    from app.core.usage_tracker import init_db as init_usage_tracker_db

    init_usage_tracker_db()
    yield


@pytest.fixture(autouse=True)
def _reset_rag_caches():
    """Drop the process-cached embedder + vector stores between EVERY test.

    These caches are module-level globals, so without this they leak across
    tests in the default SQLite mode. The failure that exposed it:

        RuntimeError: Embedding identity mismatch in namespace 'v2':
        expected {'model': 'fake', ...}, found {'model': 'minishlab/potion-base-8M', ...}

    A test that loads the REAL embedder leaves it cached; a later test that
    sets RAG_EMBEDDING_MODEL=fake then meets a store already stamped with the
    real model's identity and the mixed-model guard (correctly) refuses.

    Result was ~28 red RAG tests on a full local run that all pass in
    isolation — order-dependent, so it looked like flakiness rather than
    missing teardown, and it repeatedly cost time distinguishing "my change
    broke this" from "the suite leaks". CI missed it because the postgres job
    DID reset (below) and the virgin job happened not to hit the ordering.

    Was previously nested inside the postgres-only branch of
    `_isolate_postgres_db`, so it never ran for the default SQLite suite.
    """
    from app.core.rag import embeddings as _emb, vector_store as _vs

    # Caches alone are NOT enough. The embedder identity is PERSISTED per
    # namespace in the store, so a test that used the real embedder leaves
    # `potion-base-8M` stamped on namespace 'v2' in the shared suite DB, and
    # the next test using `fake` meets the mixed-model guard. Give every test
    # its own namespace so no two can ever share a stamp.
    #
    # Applied via os.environ (not monkeypatch) so a test's OWN
    # monkeypatch.setenv("RAG_VECTOR_NAMESPACE", ...) still wins — several
    # tests pin a namespace deliberately and must keep doing so.
    _prev_ns = os.environ.get("RAG_VECTOR_NAMESPACE")
    os.environ["RAG_VECTOR_NAMESPACE"] = f"t{uuid.uuid4().hex[:12]}"

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    yield
    # Reset on the way out too: a test that warms the real embedder must not
    # hand it to whatever runs next.
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    if _prev_ns is None:
        os.environ.pop("RAG_VECTOR_NAMESPACE", None)
    else:
        os.environ["RAG_VECTOR_NAMESPACE"] = _prev_ns


@pytest.fixture(autouse=True)
def _isolate_postgres_db():
    """Truncate unified schema between tests when running against PostgreSQL CI."""
    if not _postgres_test_mode():
        yield
        return

    from app.core.db import get_engine

    tables = (
        "chunks",
        "rag_budget",
        "hydration_runs",
        "runs",
        "doc_index",
        "agent_facts",
        "messages",
        "conversations",
        "workflows",
        "project_facts",
        "documents",
        "projects",
        "users",
        "ingestion_jobs",
    )
    _truncate_postgres_tables(get_engine(), tables)

    # Re-seed the system user row TRUNCATE just removed. Deliberately NOT
    # `users_store._initialized = False` + `init_db()`: that combination
    # forces a schema-creation (DDL) path to run again on every single
    # test, which is exactly what deadlocks Postgres when it lands
    # concurrently with a background task's read on the same tables.
    # `ensure_system_user()` only inserts the row — no DDL, no table lock
    # beyond a normal row write — and the schema itself is created exactly
    # once, at session start (see `_init_schema_once` below). A live
    # module-scoped TestClient may re-insert the same PK from knowledge
    # seed; ensure_system_user treats that UniqueViolation as success.
    from app.core import users as users_store

    users_store.ensure_system_user()

    from app.core.rag import embeddings as _emb, vector_store as _vs

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()

    yield


@pytest.fixture
def sample_text():
    return "Hello, this is a test document for Cerebrum Blocks."

@pytest.fixture
def sample_code():
    return """
def hello_world():
    print("Hello, World!")
    return 42

class MyClass:
    def __init__(self):
        self.value = 10
"""

@pytest.fixture
def data_dir():
    return "/app/data"


# NOTE: the previous autouse event-loop cleanup fixture was removed after the
# legacy Playwright browser suite was deleted. pytest-asyncio now manages loop
# lifecycle; forcibly closing the loop after every test broke module-scoped
# async fixtures (RuntimeError: Event loop is closed).


@pytest.fixture
def dev_key_is_admin(monkeypatch):
    """Give ``the dev key`` the admin role for one test.

    Only the admin path adds to a project's knowledge base
    (``app/core/privileges.caller_may_add_to_project_rag``); the dev key has no
    role, so a test of an indexing route runs as an admin caller. The non-admin
    side of that rule is tested in tests/test_project_rag_admin_only.py.
    """
    from app.core.auth import auth as auth_manager

    original = auth_manager.validate_key

    def _validate(credentials):
        principal = original(credentials)
        if getattr(credentials, "credentials", None) == DEV_KEY:
            principal = {**principal, "role": "admin"}
        return principal

    monkeypatch.setattr(auth_manager, "validate_key", _validate)


_CREDENTIAL_ROLES: dict = {}


def pytest_make_parametrize_id(config, val, argname):
    """A generated test credential appears in a test id by its role -- the
    fixture constant's name -- never by its value: values are random per
    process, and pytest-xdist needs every worker to collect the same ids."""
    # Only a value long enough to be told from an ordinary string: a short
    # generated value (kept at the length of the literal it replaced) can
    # equal a real parameter such as the unit "m" and must not rename it.
    if not isinstance(val, str) or len(val) < 8:
        return None
    if not _CREDENTIAL_ROLES:
        from tests import _fixture_values

        for name, value in vars(_fixture_values).items():
            if name.isupper() and isinstance(value, str):
                _CREDENTIAL_ROLES.setdefault(value, name)
        _CREDENTIAL_ROLES.setdefault(DEV_KEY, "DEV_KEY")
    return _CREDENTIAL_ROLES.get(val)

