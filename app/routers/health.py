import os
from datetime import datetime, timezone
from fastapi import APIRouter, Response

from app.blocks import BLOCK_REGISTRY, FAILED_BLOCKS
from app.core.health_probes import (
    probe_corpus_chunks,
    probe_database,
    probe_embedder,
    probe_llm,
)
from app.dependencies import block_instances, MONITORING_AVAILABLE, get_monitoring_block
from app.infra.monitoring import get_observability_health_payload

router = APIRouter()

# Deploy platforms inject the shipped commit. Render sets RENDER_GIT_COMMIT.
# GIT_SHA / SOURCE_VERSION are accepted aliases. Never invent a hash.
_BUILD_SHA_ENV_KEYS = ("RENDER_GIT_COMMIT", "GIT_SHA", "SOURCE_VERSION")


def _deployed_build_sha() -> str | None:
    """SHA from the already-deployed environment, or None if unset.

    Do not shell out to git or guess a hash. An unset env is an honest null
    so deploy verification can tell "not injected" from "this SHA is live".
    """
    for key in _BUILD_SHA_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _evaluate_health() -> dict:
    """Build the health payload from EVALUATED capability, not hardcoded state
    (audit §8.5/§8.6). ``checks.database`` is a real SELECT 1; ``checks.embedder``
    reports warm-load state without triggering a load. ``status`` is
    ``healthy`` only when the DB round-trips — otherwise ``degraded`` — instead
    of the previous hardcoded ``"healthy"``.
    """
    db = probe_database()
    emb = probe_embedder()
    corpus = probe_corpus_chunks()
    llm = probe_llm()
    return {
        # LIVENESS status: the process is up. DB reachability is reflected
        # truthfully as degraded rather than failing this endpoint — /health is
        # Render's liveness probe, so a transient DB blip must NOT restart a
        # live service mid-incident. Use /ready for a hard readiness gate.
        # Degraded (still 200, so Render liveness is not tripped) when the DB
        # is unreachable OR the model cannot currently be served -- health that
        # does not know the model is down is not health (19 Oct outage).
        "status": "healthy" if (db["ok"] and llm.get("serving", True)) else "degraded",
        "build_sha": _deployed_build_sha(),
        "checks": {
            "database": db,
            "embedder": emb,
            "llm": llm,
            "workflows": _probe_workflows(),
        },
        # Cached estimate of the active chunk table, not the stale
        # documents.chunk_count column. A probe error does not fail liveness.
        "corpus": corpus,
        # Model-call meter: total LLM call COUNT across all users. NAME-FREE and
        # COUNT-ONLY by design -- /health is public, so it carries no provider or
        # model name and NO tokens or cost (ORDERS_0715Z); the tokens/cost figures
        # and the per-provider,model breakdown are admin-only on /v1/metrics.
        # Reads an aggregate off the usage DB /health already touches; a meter
        # error returns a zero count, never failing liveness.
        "llm_usage": _llm_usage_meter(),
        "blocks_loaded": len(block_instances),
        "blocks_available": len(BLOCK_REGISTRY),
        "blocks_failed": {
            name: reason for name, reason in sorted(FAILED_BLOCKS.items())
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/livez")
async def livez():
    """Process liveness ONLY — deliberately touches no database.

    Exists because the database moved to Neon (2026-08-09), which bills by
    compute-time and suspends when idle. ``/health`` and ``/ready`` both run
    a real ``SELECT 1``, and Render polls its health path continuously — so
    pointing the platform health check at either would wake Neon on every
    probe and hold it awake around the clock, which is precisely the cost
    that motivated leaving Render Postgres.

    This answers the only question a liveness probe should ask: is the
    process up and serving HTTP? Dependency health is ``/health``
    (evaluated, always 200) and ``/ready`` (fail-closed 503). Keep this
    handler free of I/O — every dependency added here becomes a wake-up
    on a timer.
    """
    return {
        "status": "alive",
        "build_sha": _deployed_build_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/v1/upload-limits")
def upload_limits():
    """What the server will actually accept — published so the CLIENT can tell
    the user BEFORE attempting an upload that cannot succeed.

    Without this the browser had no way to know the cap, so an oversize file was
    discovered only by sending it. On a mobile connection that means minutes of
    uploading, and the failure surfaces as a bare ``TypeError: Failed to fetch``
    — because when a connection dies (or the server rejects and closes) while
    the body is still being sent, fetch() reports a network error and the real
    HTTP status never reaches JavaScript. "Failed to fetch" was therefore the
    one message the user could NOT act on, and the one they kept seeing.

    Unauthenticated and I/O-free on purpose: it is configuration, not data, and
    the composer needs it before any token work. Read live from the environment
    so raising the cap takes effect on restart without a rebuild.
    """
    from app.core import upload_limits as _limits

    return {
        "max_document_bytes": _limits.max_document_bytes(),
        "max_upload_bytes": _limits.max_upload_bytes(),
        "allowed_extensions": sorted(_limits.ALLOWED_UPLOAD_EXTENSIONS),
        "project_document_extensions": sorted(_limits.PROJECT_DOCUMENT_EXTENSIONS),
        "unsupported_type_status": _limits.UNSUPPORTED_TYPE_STATUS,
    }


@router.get("/health")
def health():
    """Liveness + evaluated component health.

    Reports a real DB round-trip and embedder warm-load state (not a hardcoded
    "healthy"). Always HTTP 200 — this is the liveness probe Render is wired to
    (``healthCheckPath: /health``); a DB blip surfaces as ``status:"degraded"``
    rather than a non-200 that would restart the live service. ``blocks_failed``
    still surfaces optional-dep drops.
    """
    return _evaluate_health()


@router.get("/ready")
def ready(response: Response):
    """Readiness gate: HTTP 503 when a required dependency (the database) is
    unreachable, 200 otherwise. Use this for deploy gates / load-balancer
    readiness / the smoke suite — it fails closed, unlike /health (liveness).
    """
    payload = _evaluate_health()
    # S1: refuse to serve if the live index and the resolved embedder disagree
    # on model/dim — retrieval would be silently wrong. Boot sets this flag.
    try:
        from app.core.rag.vector_store import identity_serving_blocked
        embedder_ok = not identity_serving_blocked()
    except Exception:  # noqa: BLE001 — a probe never breaks readiness evaluation
        embedder_ok = True
    payload["ready"] = payload["checks"]["database"]["ok"] and embedder_ok
    if not embedder_ok:
        payload["embedder_identity"] = "mismatch — serving blocked"
    if not payload["ready"]:
        response.status_code = 503
    return payload


@router.get("/stats")
def stats():
    """Platform stats."""
    return {
        "blocks": [name for name in BLOCK_REGISTRY.keys() if not name.startswith("container_")],
        "total_blocks": len(BLOCK_REGISTRY),
        "version": "2.0.0",
    }


@router.get("/v1/health")
async def health_v1():
    """Health check (v1 API) with observability enrichment."""
    payload = health()
    payload.update(await get_observability_health_payload())
    return payload


@router.get("/v1/system/health")
async def full_health():
    """Complete system health with predictions."""
    if not MONITORING_AVAILABLE:
        return await health_v1()
    block = get_monitoring_block()
    return await block.execute({"action": "health_report"})


def _llm_usage_meter() -> dict:
    """Name-free, COUNT-ONLY model-call meter for the public /health: the total
    LLM call count across all users and nothing else — no provider/model name,
    and no tokens or cost (ORDERS_0715Z); those stay on the admin /v1/metrics.
    Never breaks /health: on error returns a zero count with the exception type."""
    try:
        from app.core import usage_tracker
        return {"calls": usage_tracker.totals()["calls"]}
    except Exception as exc:  # noqa: BLE001 -- a meter never breaks /health
        return {"calls": 0, "error": type(exc).__name__}


def _probe_workflows() -> dict:
    """How many predefined workflows the model can reach. A drop shows here,
    not only as a missing answer (lanes 2 and 5: workflows no longer called)."""
    try:
        from app.core.predefined_reasoning import workflows_reachable
        return {"reachable": workflows_reachable()}
    except Exception as exc:  # noqa: BLE001 -- a probe never breaks /health
        return {"reachable": 0, "error": type(exc).__name__}
