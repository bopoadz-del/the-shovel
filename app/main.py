"""Cerebrum Blocks - Simple Block Execution API."""

import asyncio
import logging
import os
import sys
import shutil

logger = logging.getLogger(__name__)

# Force fresh bytecode on deployments (clear stale __pycache__)
for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
    for d in dirs:
        if d == "__pycache__":
            try:
                shutil.rmtree(os.path.join(root, d))
            except Exception:
                logger.warning(
                    "swallowed %s in <module>() — continuing",
                    "Exception", exc_info=True,
                )

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()

from app.infra.monitoring import (
    configure_structured_logging,
    init_sentry,
    observability_middleware,
    sentry_enabled,
)

configure_structured_logging()
logger = logging.getLogger(__name__)


_SENTRY_ENABLED = init_sentry()  # noqa: F841 — init side effect only

from app.blocks import BLOCK_REGISTRY
from app.dependencies import block_instances, _create_block_instance, init_blocks
from app.routers import (
    agents as agents_router,
    auth,
    blocks,
    chain,
    chat,
    debug,
    doc_search,
    doc_types,
    drive,
    execute,
    feedback as feedback_router,
    health,
    hydration as hydration_router,
    memory,
    mcp,
    monitoring,
    project,
    projects,
    connectors as connectors_router,
    rag as rag_router,
    redline,
    schedule as schedule_router,
    static,
    upload,
    users,
    workflows,
)
from app.agents import load_agents
def _bootstrap_first_user() -> None:
    """Create a single bootstrap user from env vars if no users exist yet.

    Avoids opening global registration on a public deploy: the operator sets
    BOOTSTRAP_USER_EMAIL + BOOTSTRAP_USER_PASSWORD once on the host, the first
    boot seeds that account, and subsequent boots no-op (idempotent).
    """
    email = os.getenv("BOOTSTRAP_USER_EMAIL", "").strip().lower()
    password = os.getenv("BOOTSTRAP_USER_PASSWORD", "")
    if not email or not password:
        return
    from app.core import users as users_store
    if users_store.get_user_by_email(email) is not None:
        return
    try:
        # email_verified=True: this account comes from operator-set env vars,
        # not a signup, so there is nobody to click a link. A fresh install
        # runs migration 0015's backfill against an empty table, so without
        # this the first admin would be created unverified and locked out of
        # its own login by the verification gate.
        users_store.create_user(
            email, password, role="admin", email_verified=True,
        )
        logger.info("bootstrap: created first user %s", email)
    except Exception as e:
        # Fail loud — a silent warning lets a broken bootstrap hide. If the
        # only intended admin can't be created (db lock, schema drift, etc),
        # the operator needs to see it in startup logs as an ERROR, not warn.
        logger.error("bootstrap: FAILED to create user %s: %s", email, e, exc_info=True)
        raise


def _validate_startup_env() -> None:
    """Fail fast on missing security config when ENV is explicitly production.

    SECRET_KEY is required: without it the JWT signing secret is generated
    per-process, so tokens are invalidated on every restart and differ across
    scaled instances. DATA_ENCRYPTION_KEY is required in production so
    uploaded documents are never stored as plaintext. The construction kit
    must be loaded from THIS repo (CEREBRUM_VIRGIN=false and
    CEREBRUM_DOMAIN_KITS=construction) — do not pull kits from the store.
    """
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "")).strip().lower()
    if env not in ("prod", "production"):
        return
    if not os.getenv("SECRET_KEY"):
        raise RuntimeError(
            "SECRET_KEY is required when ENV=production — without it the JWT "
            "signing secret is regenerated per process, invalidating all "
            "tokens on restart. Set SECRET_KEY in the environment."
        )
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError(
            "DATABASE_URL is required when ENV=production — without it the app "
            "silently falls back to sqlite:///{DATA_DIR}/the_fork.db, stranding "
            "the Postgres corpus (the 'empty SQLite' incident: the app read an "
            "empty local DB while the whole corpus sat in Postgres). Set "
            "DATABASE_URL to the Postgres connection string."
        )
    if not os.getenv("DATA_ENCRYPTION_KEY"):
        raise RuntimeError(
            "DATA_ENCRYPTION_KEY is required when ENV=production — without it "
            "uploaded documents are stored unencrypted at rest. Set "
            "DATA_ENCRYPTION_KEY in the environment."
        )
    virgin = os.getenv("CEREBRUM_VIRGIN", "true").strip().lower()
    kits = {
        k.strip()
        for k in os.getenv("CEREBRUM_DOMAIN_KITS", "").split(",")
        if k.strip()
    }
    if virgin not in ("0", "false", "no") or "construction" not in kits:
        raise RuntimeError(
            "Production Fork requires CEREBRUM_VIRGIN=false and "
            "CEREBRUM_DOMAIN_KITS=construction (this repo's construction kit)."
        )
    from app.blocks import BLOCK_REGISTRY
    if "construction" not in BLOCK_REGISTRY:
        raise RuntimeError(
            "Production boot failed: construction kit/container is missing "
            "from BLOCK_REGISTRY. Pin CEREBRUM_DOMAIN_KITS=construction and "
            "ensure this repo's construction container loaded."
        )


def _warn_when_the_llm_has_no_fallback() -> None:
    """Say it at boot: a chat with no usable fallback dies with its provider.

    A fallback named after a removed provider, or one whose key is unset,
    used to disable itself in silence (live 2026-09-19). This names the
    variable to fix; it never prints a value."""
    from app.core.health_probes import probe_llm

    state = probe_llm()
    if not state.get("primary_ready"):
        logger.warning("LLM: the primary provider has no API key; every chat turn will fail.")
    if not state.get("fallback_ready"):
        logger.warning(
            "LLM: NO USABLE FALLBACK -- check LLM_FALLBACK_PROVIDER names a supported "
            "provider other than the primary, and that its API key is set. A primary "
            "outage will fail every chat turn."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all blocks + load runtime agents at startup."""
    _validate_startup_env()
    _warn_when_the_llm_has_no_fallback()
    from app.core import rss_watchdog
    rss_watchdog.start()
    from app.core import loop_watchdog
    loop_watchdog.start()
    from app.core import health_probes as _health_probes
    _health_probes.start_db_keep_warm()
    from app.blocks.learning_engine import assert_learning_engine_hard_off
    assert_learning_engine_hard_off()
    await init_blocks()
    from app.core import redis_client as _redis_client
    await _redis_client.get_redis_client()
    from app.core.projects import init_db
    init_db()
    from app.core.users import init_db as init_users_db
    init_users_db()
    _bootstrap_first_user()
    from app.core.agent_memory import init_db as init_agent_memory_db
    init_agent_memory_db()
    from app.core.doc_index import init_db as init_doc_index_db
    init_doc_index_db()
    # NOTE: seed_knowledge() used to run HERE, inline. It is now part of the
    # background warm-up — see _seed_knowledge(). Measured 2026-08-11: on a
    # database that has not been seeded yet it takes 42s and +677 MB, because
    # indexing docs/knowledge/*.md constructs the embedder. Inline, that alone
    # kept the port shut long past the platform's scan window AND blew a 512Mi
    # instance before the app ever served a request.
    from app.core.hydration_store import init_db as init_hydration_db
    init_hydration_db()
    from app.core import rate_limit as _rate_limit_startup
    logger.info("Rate limiter backend: %s", _rate_limit_startup.init_rate_limiter())
    from app.core.session_store import get_session_store
    from app.routers import project as project_router
    app.state.project_store = get_session_store()
    project_router._store = app.state.project_store
    logger.info("Project session store ready: %s",
                type(app.state.project_store).__name__)
    loaded = load_agents()
    logger.info("Loaded %d runtime agents: %s", len(loaded), ", ".join(sorted(loaded.keys())))
    from app.core import hydration_scheduler
    hydration_scheduler.start()
    from app.core.cde.poll import start_event_poller
    start_event_poller()

    # Model warm-up runs AFTER the port is listening — see _warm_models.
    warm_task = None
    if os.getenv("WARM_MODELS_BLOCKING", "").strip().lower() in ("1", "true", "yes"):
        logger.info("WARM_MODELS_BLOCKING set — warming models before serving")
        await _warm_models()
    else:
        warm_task = asyncio.create_task(_warm_models())

    # Decrypted-temp sweeper: reap orphaned fork_dec_* plaintext files at
    # boot and hourly. Every decrypt path cleans up after itself, but a
    # crash mid-request (or a failed unlink) can leave CLIENT PLAINTEXT in
    # /tmp indefinitely — this bounds that exposure to one sweep interval.
    async def _plaintext_sweep_loop():
        from app.core.file_crypto import sweep_stale_plaintext
        interval = int(os.getenv("PLAINTEXT_SWEEP_INTERVAL_SECONDS", "3600"))
        while True:
            try:
                await asyncio.to_thread(sweep_stale_plaintext, interval)
            except Exception:  # noqa: BLE001 — the sweeper must outlive one bad pass
                logger.exception("plaintext sweep pass failed; will retry")
            await asyncio.sleep(interval)

    sweep_task = asyncio.create_task(_plaintext_sweep_loop())
    try:
        yield
    finally:
        sweep_task.cancel()
        try:
            await sweep_task
        except asyncio.CancelledError:
            logger.debug("plaintext sweeper cancelled at shutdown")
        if warm_task is not None:
            warm_task.cancel()
            try:
                await warm_task
            except asyncio.CancelledError:
                # Expected and benign: we cancelled it one line above because
                # shutdown began while a model was still loading. Not re-raised
                # because this task's cancellation is ours, not the caller's.
                logger.debug("model warm-up cancelled at shutdown")
            except Exception:
                # _warm_models logs each load's own failure; this only catches
                # something escaping that, and must not abort the remaining
                # shutdown steps (scheduler stop, redis close) below.
                logger.exception("model warm-up task failed during shutdown")
        await hydration_scheduler.stop()
        from app.core.cde.poll import stop_event_poller
        await stop_event_poller()
        await _redis_client.close_redis_client()


def _warm_safety_detector() -> None:
    """Load the Safety Observation AI v2 detector.

    Without this the first /v1/chat/analyze-photo pays a ~3-5s ONNX cold load
    and silently returns empty observations (the bug that hit PR #135's first
    deploy). When SAFETY_WORLD_WEIGHTS is unset, default_detector() returns
    None, and the chat route gracefully skips the safety_qaqc tier.
    """
    from app.blocks.safety_world_detector import default_detector
    det = default_detector()
    if det is not None:
        logger.info("Safety Observation AI v2 detector warm-loaded: %d classes",
                    len(det.class_names))
    else:
        logger.info("Safety Observation AI v2 detector NOT loaded "
                    "(SAFETY_WORLD_WEIGHTS unset or file missing)")


def _warm_embedder() -> None:
    """Load the RAG embedder (bge-small, ~8-40s cold; longer on first download).

    seed_knowledge() only loads it when it actually re-indexes, so on a restart
    where the notes are already seeded it would otherwise lazy-load inside the
    FIRST /v1/chat/stream RAG query — eating the cold load inside the stream
    deadline, which surfaces as an intermittent chat hang / empty bubble.
    """
    from app.core.rag import retrieval_worker

    if retrieval_worker.enabled():
        # Retrieval runs in the worker processes and the model in the model
        # server they share; start them all now, not inside the first turns.
        n = retrieval_worker.warm_all()
        logger.info("RAG embedder warm-loaded in the model server; %d retrieval workers ready", n)
        return

    from app.core.rag.embeddings import embedder_health, get_embedder

    # Probe first: embedder_health reports WHY a load failed instead of letting
    # the exception surface as a generic warm-up failure. The weights are baked
    # into the image and HF_HUB_OFFLINE is set, so a failure here means a broken
    # image (wrong RAG_EMBEDDING_MODEL vs the baked ARG, or a missing cache) —
    # not a transient network problem, and not something that will fix itself
    # on the first user question.
    health = embedder_health()
    if not health["ok"]:
        raise RuntimeError(
            f"embedder {health['model']!r} could not be loaded: {health['error']}"
        )
    emb = get_embedder()
    emb.encode(["warmup"])
    logger.info(
        "RAG embedder warm-loaded: %s dim=%d backend=%s",
        emb.model_name, emb.dim, emb.backend,
    )


def _seed_knowledge() -> None:
    """Ingest bundled docs/knowledge/*.md into the RAG general-knowledge project.

    Idempotent: on a database already seeded this is a cheap no-op (~0.02s
    measured), and it never raises. On a FRESH database it re-indexes, which
    constructs the embedder — 42s and +677 MB measured 2026-08-11.

    That is why it belongs here rather than in the lifespan. Inline, a first
    boot against an unseeded database could not open the port in time, the
    platform restarted the container, and the restart began the same 677 MB
    seed again — so the failure repeated on every deploy instead of clearing
    itself. Backgrounded, an incomplete seed costs GK retrieval quality until
    it finishes, but can never stop the service becoming reachable.
    """
    from app.core.knowledge_seed import seed_knowledge
    seed_knowledge()
    logger.info("Knowledge seed complete (docs/knowledge/*.md)")


def _warm_parse_libraries() -> None:
    """Import the document parse stack after the port opens. Their first
    import (numpy/pandas, pdfplumber) holds the GIL even in a worker thread
    and froze every user up to 0.43 s during a first BOQ export (CI
    test-timing); after this warm-up a live request never pays it. Kept out
    of startup itself: the boot check forbids pandas at import."""
    import importlib

    for name in ("pandas", "pdfplumber", "openpyxl"):
        try:
            importlib.import_module(name)
        except ImportError:
            logger.info("parse library %s not installed; the parse reports it when used", name)


async def _warm_models() -> None:
    """Warm the heavy models — and seed the knowledge base — WITHOUT blocking startup.

    These used to run inline in lifespan, before FastAPI opened the port. On a
    host with no persistent HF cache the embedder warm-up also downloads the
    model, so startup could exceed the platform's port-scan window. Render then
    restarted the container while the first boot was still loading, and TWO
    processes holding torch + bge-small at once exceeded the 512Mi instance —
    an out-of-memory kill whose actual cause was a slow startup, not a leak.

    Running them as a background task means the port opens as soon as the light
    initialisation is done. The trade-off is explicit: for a short window after
    boot the models may still be loading, so an early RAG query lazy-loads and
    pays the cold cost — exactly what happened before this warm-up existed, and
    strictly better than never becoming reachable at all.

    Set WARM_MODELS_BLOCKING=true to restore the old inline behaviour (on-prem
    deployments may prefer readiness to mean model-loaded). Each load is
    isolated: the detector failing must not stop the embedder from loading.
    """
    # Embedder first: the knowledge seed reuses the same cached instance, so
    # ordering it last means the seed never pays a second model load.
    for label, fn in (("safety detector", _warm_safety_detector),
                      ("RAG embedder", _warm_embedder),
                      ("knowledge seed", _seed_knowledge),
                      ("parse libraries", _warm_parse_libraries)):
        try:
            await asyncio.to_thread(fn)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s warm-load failed; it will lazy-load on first use", label)


# Interactive docs + the raw OpenAPI schema disclose the entire API surface
# (142 operations, including every /v1/admin/* route) to anyone unauthenticated
# — the 2026-08-02 route sweep confirmed /openapi.json returned 200 with no
# credentials on prod. Harmless for a public demo, needless attack-surface
# disclosure for a client-desk deployment.
#
# Default: EXPOSED off-production (local dev and the on-prem profile keep the
# docs a developer expects), CLOSED when ENV=production. `API_DOCS_ENABLED`
# overrides in either direction, so an operator who wants /docs on prod sets
# it to "true" rather than patching code.
def _api_docs_enabled() -> bool:
    override = os.getenv("API_DOCS_ENABLED", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False
    return os.getenv("ENV", "").strip().lower() != "production"


_DOCS_ON = _api_docs_enabled()

app = FastAPI(
    title="Cerebrum Blocks",
    description="Build AI Like Lego - Simple Block Execution API",
    version="2.0.0",
    docs_url="/docs" if _DOCS_ON else None,
    redoc_url="/redoc" if _DOCS_ON else None,
    openapi_url="/openapi.json" if _DOCS_ON else None,
    lifespan=lifespan,
)

_extra_origins = [
    o.strip()
    for o in os.getenv("CORS_EXTRA_ORIGINS", "").split(",")
    if o.strip()
]

# A browser reports EVERY blocked cross-origin request as the same opaque
# "Failed to fetch" — no status, no body — which is exactly how the production
# upload failure presented. The allow-list above is localhost-only, so a
# deployed frontend on its own domain is served entirely by CORS_EXTRA_ORIGINS;
# if that var is unset or misspelled in the environment, every upload from the
# real site fails with a message that looks like a network outage.
# CORS_ALLOW_ORIGIN_REGEX covers deploy-preview domains whose hostnames are not
# known ahead of time (e.g. r"https://.*\.vercel\.app"). Left unset it changes
# nothing.
_origin_regex = (os.getenv("CORS_ALLOW_ORIGIN_REGEX") or "").strip() or None

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:4173",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:4173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    *_extra_origins,
]

# NOTE: CORSMiddleware is registered LAST in this module (after the rate-limit
# and observability middleware), because Starlette runs the most recently added
# middleware OUTERMOST. It used to be registered here — first, and therefore
# innermost — which meant every response produced by an OUTER middleware
# (rate-limit 429s, and now oversize 413s) reached the browser with no CORS
# headers at all. A cross-origin caller cannot read such a response: the fetch
# rejects with the opaque "Failed to fetch", identical to the server being
# unreachable. See the registration site below.


# ── Oversize-body guard ────────────────────────────────────────────────────
# Reject a too-large upload from its Content-Length, BEFORE Starlette spools
# the body to disk. Without this the server accepts an entire 345 MB upload
# and only then answers 413 — minutes of transfer, and any proxy/gateway
# timeout along the way turns that into a connection reset, which the browser
# surfaces as that same opaque "Failed to fetch".
async def _drain_request_body(request: Request, *, seconds: float) -> bool:
    """Consume the client's body so it can actually READ our error response.

    An HTTP peer that is still streaming a body cannot receive a response on a
    socket the server closes underneath it: the send fails first, and both
    browsers and httpx report a bare transport error rather than the status we
    sent. Measured on the live service before this existed — a 60 MB POST to
    an upload route died after 34s with "Server disconnected without sending a
    response", which the browser shows as the opaque "Failed to fetch" this
    guard was written to prevent.

    Bounded by ``seconds`` so a genuinely enormous upload cannot pin a worker:
    if the budget runs out we answer anyway (best effort) and say so in the
    log. Returns True when the body was fully drained.
    """
    import time as _time

    deadline = _time.monotonic() + seconds
    try:
        async for _chunk in request.stream():
            if _time.monotonic() > deadline:
                return False
    except Exception:  # noqa: BLE001 — client vanished mid-drain; nothing to answer
        return False
    return True


@app.middleware("http")
async def _reject_oversize_bodies(request: Request, call_next):
    from app.core import upload_limits

    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            declared = int(raw_length)
        except ValueError:
            declared = 0
        limit = upload_limits.request_body_limit(request.url.path)
        # Multipart framing adds boundary/header overhead around the file, so
        # compare with headroom: the per-route check does the exact accounting
        # on the file itself. This guard only catches bodies no route could
        # possibly accept.
        if declared > limit + (1024 * 1024):
            # Drain BEFORE answering. Rejecting on Content-Length alone still
            # avoids spooling the file to disk (the point of this guard), but
            # the client must be allowed to finish sending or it never sees
            # the 413 — which turned a clear "file too large" into the very
            # "Failed to fetch" this middleware exists to eliminate.
            budget = float(os.getenv("OVERSIZE_DRAIN_SECONDS", "30"))
            if not await _drain_request_body(request, seconds=budget):
                logger.warning(
                    "oversize body (%s bytes) not fully drained within %ss — "
                    "the client may see a transport error instead of the 413",
                    declared, budget,
                )
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request body too large ({declared} bytes). The maximum "
                        f"upload is {limit} bytes."
                    )
                },
            )
    return await call_next(request)


# NOTE: a previous "file upload security" middleware was removed here. It
# called `await request.body()` on every /upload request, which buffered the
# entire multipart body (including the file) into memory — a memory-DoS that
# also defeated UploadFile's on-disk spooling. Its security check was dead
# code: it json.loads()'d a multipart body, always failed, and never ran the
# validator. Upload validation (size, extension, filename, path traversal)
# lives in app/routers/upload.py where the file is a proper UploadFile.


# ── Rate limiting ──────────────────────────────────────────────────────────
# Per-caller rate limiting on every request — including JWT sessions, which
# the per-API-key limiter never covered. Controlled by RATE_LIMIT_PER_MINUTE.
from app.core import rate_limit as _rate_limit
from app.core import jwt_auth as _jwt_auth

_RATE_LIMIT_EXEMPT_PREFIXES = ("/dashboard", "/assets")
_RATE_LIMIT_EXEMPT_EXACT = {
    # PR #98: /v1/metrics removed — it returns per-block execution counts +
    # latencies + error counts, which is operational data we should not
    # leak to anonymous callers. /metrics (Prometheus exposition) stays in
    # the exempt list because Prometheus scrapers typically don't auth and
    # the counter set there is intentionally limited to non-sensitive
    # request/response totals.
    "/", "/health", "/ready", "/v1/health", "/metrics", "/docs", "/redoc", "/openapi.json",
}


def _rate_limit_identity(request: Request) -> str:
    """Identify the caller for rate-limiting: user id (JWT), hashed API key,
    or client IP for an unauthenticated request."""
    authz = request.headers.get("Authorization", "")
    if authz.startswith("Bearer "):
        token = authz[7:].strip()
        payload = None
        try:
            payload = _jwt_auth.decode_token(token)
        except _jwt_auth.InvalidTokenError:
            # An API key is not a JWT. This is the NORMAL path for key-authed
            # callers, not an exceptional one — logging a traceback here fired
            # once per request in production and buried real warnings.
            pass
        except Exception:
            # Anything that is not "this simply isn't a JWT" is unexpected and
            # stays loud (e.g. a missing signing secret).
            logger.warning(
                "swallowed %s in _rate_limit_identity() — continuing",
                "Exception", exc_info=True,
            )
        if payload and payload.get("user_id"):
            return f"user:{payload['user_id']}"
        import hashlib
        return "key:" + hashlib.sha256(token.encode()).hexdigest()[:24]
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


@app.middleware("http")
async def observability_middleware_wrapper(request: Request, call_next):
    return await observability_middleware(request, call_next)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if (
        request.method == "OPTIONS"
        or path in _RATE_LIMIT_EXEMPT_EXACT
        or any(path.startswith(p) for p in _RATE_LIMIT_EXEMPT_PREFIXES)
    ):
        return await call_next(request)
    # In a thread: with Redis this is a network round trip, and on the event
    # loop it held every request (live: a 1.7 s stall at 15 users).
    from app.core.offload import off_loop

    if not await off_loop(_rate_limit.check_and_record, _rate_limit_identity(request)):
        return JSONResponse(
            status_code=429,
            content={"status": "error",
                     "error": "Rate limit exceeded — too many requests."},
        )
    response = await call_next(request)
    if (
        sentry_enabled()
        and response.status_code >= 500
        and not getattr(request.state, "sentry_captured", False)
    ):
        import sentry_sdk
        sentry_sdk.capture_message(
            f"5xx response: {request.method} {request.url.path} ({response.status_code})",
            level="error",
        )
    return response


# ── Browser hardening headers ─────────────────────────────────────────────
# Measured against live theshovel.ai on 2026-09-10: the only headers on `/`
# and `/v1/health` were `server: cloudflare` and `x-render-origin-server`.
# No HSTS, no CSP, no frame guard, no nosniff — on a platform serving a
# client's contract documents.
#
# Registered here, INSIDE CORS (which is added last and must stay outermost),
# so every response carries these — including the 401s and 413s the CORS
# layer short-circuits.
#
# The CSP ships REPORT-ONLY. Its allowlist is what the live SPA actually
# fetches, recorded in a browser on the same day rather than assumed: the
# login bundle pulls a stylesheet from fonts.googleapis.com and woff2 faces
# from fonts.gstatic.com, so a bare `default-src 'self'` would blank the
# site's typography the moment anyone enforced it. `'unsafe-inline'` for
# styles is what the Vite build needs today; removing it is the work that has
# to happen before the enforcing flip, and the report-only phase is how we
# find out what else is in the way.
_CSP_REPORT_ONLY = "; ".join((
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' data: https://fonts.gstatic.com",
    "img-src 'self' data: blob:",
    "connect-src 'self'",
    "object-src 'none'",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Stamp baseline security headers on every response.

    ``setdefault`` throughout, never assignment: a route that already chose a
    value had a reason, and this middleware does not know it.
    """

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        h = response.headers
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "no-referrer")
        h.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        h.setdefault("Content-Security-Policy-Report-Only", _CSP_REPORT_ONLY)
        # HSTS only where the origin really is https. Sent from a plain-HTTP
        # dev box it pins that browser to https for a year.
        if (os.getenv("ENV") or "").strip().lower() in {"production", "prod"}:
            h.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ── CORS — registered LAST so it is the OUTERMOST middleware ───────────────
# Starlette wraps the most recently added middleware around all earlier ones.
# Registering CORS last means EVERY response carries CORS headers, including
# ones short-circuited by the middleware above (rate-limit 429, oversize 413)
# and unhandled 500s. Previously CORS sat innermost, so exactly the responses
# a user most needs to read — "too large", "slow down" — arrived unreadable and
# surfaced in the browser as "Failed to fetch" with no status at all.
#
# Move this call and that breaks again; it is ordering-sensitive, not stylistic.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_origin_regex=_origin_regex,
    allow_credentials=True,
    # Enumerate methods/headers rather than "*" — a credentialed CORS config
    # should not reflect arbitrary methods/headers back to the browser.
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


# ── Unified error envelope ────────────────────────────────────────────────
# All API errors are returned as:
#   {"error": {"code": "<MACHINE>", "message": "<HUMAN>", "details"?: {...}},
#    "detail": "<HUMAN>"}      # legacy, kept so existing UI code doesn't break.
#
# Status code → code mapping is generic; routes can raise HTTPException with
# detail=<dict> to override (e.g. detail={"code":"INSUFFICIENT_QUOTA","message":...}).

_STATUS_CODE_NAME = {
    400: "BAD_REQUEST",
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    502: "UPSTREAM_ERROR",
    503: "UNAVAILABLE",
    504: "TIMEOUT",
}


def _envelope(status: int, message: str, code: str | None = None, details=None):
    body = {
        "error": {
            "code": code or _STATUS_CODE_NAME.get(status, "ERROR"),
            "message": message,
        },
        "detail": message,  # legacy field — keep until callers migrate
    }
    if details is not None:
        body["error"]["details"] = details
    return body


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    if sentry_enabled() and exc.status_code >= 500:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
        request.state.sentry_captured = True
    detail = exc.detail
    if isinstance(detail, dict):
        msg = str(detail.get("message") or detail.get("error") or detail.get("detail") or "")
        code = str(detail.get("code") or _STATUS_CODE_NAME.get(exc.status_code, "ERROR"))
        rest = {k: v for k, v in detail.items() if k not in {"code", "message", "error", "detail"}}
        body = _envelope(exc.status_code, msg, code, rest or None)
    else:
        body = _envelope(exc.status_code, str(detail))
    return JSONResponse(status_code=exc.status_code, content=body, headers=getattr(exc, "headers", None) or None)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(_request: Request, exc: RequestValidationError):
    # jsonable_encoder, not exc.errors() raw. A pydantic model_validator or
    # field_validator that raises ValueError puts the exception OBJECT into
    # each error's `ctx`, which JSONResponse cannot serialise -- so this
    # handler, whose whole job is to turn a 422 into an envelope, raised and
    # the caller got a 500 for sending a bad body. Field-level constraints
    # (min_length and friends) carry no ctx, which is why the defect stayed
    # invisible: every validation error the app happened to produce was of
    # the one shape that serialises.
    return JSONResponse(
        status_code=422,
        content=_envelope(
            422,
            "Request validation failed",
            "VALIDATION_ERROR",
            {"errors": jsonable_encoder(exc.errors())},
        ),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    if sentry_enabled():
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
        request.state.sentry_captured = True
    # Never leak internal stack traces in the response — log them, return generic.
    return JSONResponse(
        status_code=500,
        content=_envelope(500, "Internal server error", "INTERNAL_ERROR"),
    )


# Include all routers
app.include_router(blocks.router)
app.include_router(execute.router)
app.include_router(feedback_router.router)
app.include_router(chain.router)
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(auth.router)
app.include_router(memory.router)
app.include_router(monitoring.router)
app.include_router(projects.router)
app.include_router(connectors_router.router)
from app.routers import exports as exports_router  # noqa: E402 — late import to keep diff small
app.include_router(exports_router.router)
from app.routers import schedule as schedule_router_module  # noqa: E402
app.include_router(schedule_router_module.router)
from app.routers import usage as usage_router  # noqa: E402
app.include_router(usage_router.router)
from app.routers import chat_photos as chat_photos_router  # noqa: E402
app.include_router(chat_photos_router.router)
app.include_router(doc_search.router)
app.include_router(rag_router.router)
app.include_router(redline.router)
app.include_router(project.router)
app.include_router(users.router)
app.include_router(doc_types.router)
app.include_router(workflows.router)
app.include_router(health.router)
app.include_router(mcp.router)
# Mount the MCP SSE POST endpoint directly on the app — include_router does
# not propagate Starlette Mount routes (no-op if MCP SSE deps are absent).
mcp.mount_message_endpoint(app)
app.include_router(drive.router)
app.include_router(agents_router.router)
app.include_router(hydration_router.router)
# Debug routes — only in non-production environments. The allow-list lives
# in app.routers.debug beside the endpoint's own gate, so the mount decision
# and the payload gate cannot drift apart again.
if debug.is_dev_environment():
    app.include_router(debug.router)

# Admin diagnostic routes — mounted in ALL environments. Endpoints are
# role-gated (admin only); without them, a shell-less Render deploy has
# no way to inspect the production index.
from app.routers import admin as admin_router  # noqa: E402
app.include_router(admin_router.router)

# Mount the frontend bundle (frontend/dist). It is a build artifact absent in
# CI and fresh checkouts; StaticFiles raises RuntimeError at import time if its
# directory is missing, so mount each frontend path only when it exists. The
# legacy app/static dashboard was retired — see app/routers/static.py.
if os.path.isdir("frontend/dist"):
    app.mount("/dashboard", StaticFiles(directory="frontend/dist", html=True), name="dashboard")
if os.path.isdir("frontend/dist/assets"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

# static.router includes a catch-all SPA fallback. Register it LAST so
# specific API routes and StaticFiles mounts are matched first; only
# unmatched GETs fall through to serve frontend/dist/index.html.
app.include_router(static.router)
