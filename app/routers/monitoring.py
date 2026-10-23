from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app.dependencies import (
    MONITORING_AVAILABLE,
    get_monitoring_block,
    require_api_key,
)
from app.infra.monitoring import block_metrics
from app.routers.health import health_v1

router = APIRouter()


@router.get("/metrics")
def prometheus_metrics() -> Response:
    """Prometheus text-format exposition (PR #98).

    Intentionally unauthenticated — Prometheus scrapers typically don't
    auth, and the counters exposed here are non-sensitive aggregates:

      the_fork_requests_total              (method, status)
      the_fork_cm_template_match_total     (outcome)

    Both are counts with a fixed, low-cardinality label set. No user text,
    project identifier or document reference is exposed. Anything carrying
    those — including the raw phrasings behind a template miss, which are
    customer messages — stays out of here; the miss text is gated behind
    CM_LOG_TEMPLATE_MISSES and goes to logs, never to this endpoint.

    Keep this list current. It is the standing argument for why this route
    can stay open, so a counter added without updating it silently widens
    what an unauthenticated endpoint serves.

    Sensitive operational data stays behind /v1/metrics (admin-gated). When
    prometheus-client isn't installed, returns an empty 503 so the dev
    environment doesn't refuse to import this router.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    except Exception:
        raise HTTPException(status_code=503, detail="prometheus-client not installed")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


class RecordMetricsRequest(BaseModel):
    """Typed body for a provider-call metric. Only these fields reach the
    monitoring block — no arbitrary client dict is splatted into execute()."""
    provider: Optional[str] = None
    latency_ms: float = 0
    success: bool = True
    error_type: Optional[str] = None


@router.get("/v1/metrics")
def get_block_metrics(auth: dict = Depends(require_api_key)):
    """Per-block execution timings (from UniversalBlock.execute).

    Admin-only — the snapshot exposes execution counts, latencies, and
    error counts per block, which is operational data not safe to
    return to anonymous callers. PR #98 added the auth gate after the
    pilot-readiness audit flagged it. For unauthenticated scraping use
    the Prometheus ``/metrics`` endpoint instead, which intentionally
    exposes a narrower counter set.
    """
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return block_metrics.snapshot()


@router.get("/v1/leaderboard")
async def get_leaderboard(auth: dict = Depends(require_api_key)):
    """Provider reliability leaderboard"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    block = get_monitoring_block()
    return await block.execute({"action": "leaderboard"})


@router.get("/v1/recommend")
async def recommend_provider(auth: dict = Depends(require_api_key)):
    """Recommend a provider from observed reliability.

    Threshold sort over recorded latency/error rates, not a model. Returns
    `recommended: null` when no calls have been recorded -- that is a lack of
    evidence, not an outage.
    """
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    block = get_monitoring_block()
    return await block.execute({"action": "recommend"})


@router.get("/v1/predict")
async def predictive_failover(auth: dict = Depends(require_api_key)):
    """Predict potential failures before they happen"""
    if not MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    block = get_monitoring_block()
    return await block.execute({"action": "predictive_failover"})


@router.post("/v1/metrics/record")
async def record_metrics(
    request: RecordMetricsRequest, auth: dict = Depends(require_api_key)
):
    """Record call metrics for tracking"""
    if not MONITORING_AVAILABLE:
        return {"status": "no_op"}
    block = get_monitoring_block()
    if not (request.provider or "").strip():
        raise HTTPException(422, "provider is required")
    payload = {"action": "record_call", **request.model_dump(exclude_none=True)}
    result = await block.execute(payload)
    # The block reports an unknown provider as data. That is the caller's
    # fault and must not leave here as a 200 with an error inside it.
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(422, str(result["error"]))
    return result
