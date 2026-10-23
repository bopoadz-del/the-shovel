"""Platform observability utilities and provider monitoring block."""

from __future__ import annotations

import atexit
import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_sentry_enabled = False
_structured_logging_enabled = False

# Dead-tunnel / DNS / connect failures when calling the cloud LLM API.
_LLM_TRANSPORT_MARKERS = (
    "name or service not known",
    "errno -2",
    "gaierror",
    "not reachable at",
    "connecterror",
    "connection refused",
    "failed to resolve",
    "temporary failure in name resolution",
    "getaddrinfo failed",
)


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per log line for Render/log-drain ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        ctx_rid = request_id_ctx.get()
        if ctx_rid:
            payload["request_id"] = ctx_rid
        for key in (
            "request_id",
            "path",
            "method",
            "event",
            "block",
            "duration_ms",
            "provider",
            "failure_class",
        ):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _DropWhenFullQueueHandler(logging.handlers.QueueHandler):
    """Hands records to a bounded queue; never blocks the caller.

    Live 2026-10-07 (15 users): the event loop stalled 1.1 s inside a log
    line's stdout flush while the log driver was behind. Records now go to a
    queue a background thread writes out. When the queue is full a record is
    dropped (and counted, in ``the_fork_log_records_dropped_total``) rather
    than making the caller wait; the count is logged once there is room again.
    Diagnostic logs only: audit entries never pass through logging
    (app.core.audit).
    """

    def __init__(self, q: "queue.Queue") -> None:
        super().__init__(q)
        self.dropped = 0

    def enqueue(self, record: logging.LogRecord) -> None:
        if self.dropped:
            note = logging.LogRecord("app.infra.monitoring", logging.WARNING, __file__, 0,
                                     "dropped %d log records: the log queue was full", (self.dropped,), None)
            try:
                self.queue.put_nowait(self.prepare(note))
                self.dropped = 0
            except queue.Full:
                self.dropped += 1
                return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            _count_dropped_log_record()


_PROM_LOG_DROPPED = None
#: Records dropped since start (also exported to Prometheus when installed).
LOG_RECORDS_DROPPED = 0


def _count_dropped_log_record() -> None:
    global _PROM_LOG_DROPPED, LOG_RECORDS_DROPPED
    LOG_RECORDS_DROPPED += 1
    if _PROM_LOG_DROPPED is None:
        try:
            from prometheus_client import Counter
            _PROM_LOG_DROPPED = Counter(
                "the_fork_log_records_dropped_total",
                "Diagnostic log records dropped because the log queue was full",
            )
        except Exception:  # noqa: BLE001 -- prometheus-client is optional
            _PROM_LOG_DROPPED = False
    if _PROM_LOG_DROPPED:
        _PROM_LOG_DROPPED.inc()


_LISTENER: "Optional[logging.handlers.QueueListener]" = None


def _queued(handler: logging.Handler) -> logging.Handler:
    """``handler`` behind a bounded queue and a writer thread (LOG_QUEUE_MAX, default 10000)."""
    global _LISTENER
    try:
        size = max(100, int(os.getenv("LOG_QUEUE_MAX") or "10000"))
    except ValueError:
        size = 10000
    q: "queue.Queue" = queue.Queue(maxsize=size)
    if _LISTENER is None:
        atexit.register(_stop_listener)
    else:
        _LISTENER.stop()
    _LISTENER = logging.handlers.QueueListener(q, handler, respect_handler_level=True)
    _LISTENER.start()
    return _DropWhenFullQueueHandler(q)


def _stop_listener() -> None:
    """At exit: write out what is queued (the current listener, once)."""
    global _LISTENER
    if _LISTENER is not None:
        _LISTENER.stop()
        _LISTENER = None


def configure_structured_logging() -> bool:
    """Enable JSON logging in production (or when STRUCTURED_LOGS=true)."""
    global _structured_logging_enabled
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "production")).strip().lower()
    mode = os.getenv("STRUCTURED_LOGS", "auto").strip().lower()
    use_json = mode == "true" or (mode == "auto" and env in ("prod", "production"))
    if not use_json:
        _structured_logging_enabled = False
        return False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [_queued(handler)]
    # The old `if root.level == logging.NOTSET` guard never fired — root
    # defaults to WARNING, so the intended INFO level was never applied and
    # every module INFO line was dropped in prod (the whole illumination-table
    # investigation was blinded by this). Set the level outright; LOG_LEVEL dials it back
    # (e.g. WARNING) without a code change if CloudWatch volume bites.
    root.setLevel((os.getenv("LOG_LEVEL") or "INFO").strip().upper())
    _structured_logging_enabled = True
    return True


def structured_logging_enabled() -> bool:
    return _structured_logging_enabled


def init_sentry() -> bool:
    """Initialize Sentry when SENTRY_DSN is set. No-op otherwise."""
    global _sentry_enabled
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        _sentry_enabled = False
        return False

    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    sentry_sdk.init(
        dsn=dsn,
        integrations=[
            StarletteIntegration(),
            FastApiIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        environment=os.getenv("ENV", os.getenv("ENVIRONMENT", "production")),
        send_default_pii=False,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0") or 0),
    )
    _sentry_enabled = True
    return True


def sentry_enabled() -> bool:
    return _sentry_enabled


def get_request_id() -> str:
    rid = request_id_ctx.get()
    if rid:
        return rid
    rid = str(uuid.uuid4())[:12]
    request_id_ctx.set(rid)
    return rid


def is_llm_transport_failure(message: str) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _LLM_TRANSPORT_MARKERS)


def capture_llm_transport_failure(
    error_message: str,
    *,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    provider: str = "",
) -> Optional[str]:
    """Report dead-tunnel / DNS / connect LLM failures to Sentry with endpoint context."""
    if not is_llm_transport_failure(error_message):
        return None

    rid = request_id or get_request_id()
    endpoint_ctx = {
        "llm_provider": os.getenv("LLM_PROVIDER") or "deepseek",
        "llm_fallback_provider": os.getenv("LLM_FALLBACK_PROVIDER") or "",
    }
    logger.error(
        "llm_transport_failure",
        extra={
            "request_id": rid,
            "path": path,
            "provider": provider,
            "failure_class": "llm_transport",
            "event": "llm_transport_failure",
        },
    )

    if not _sentry_enabled:
        return None

    import sentry_sdk

    with sentry_sdk.push_scope() as scope:
        scope.set_tag("failure_class", "llm_transport")
        scope.set_tag("provider", provider)
        scope.set_tag("request_id", rid)
        if path:
            scope.set_tag("http_path", path)
        scope.set_context("llm_endpoint", endpoint_ctx)
        return sentry_sdk.capture_message(
            f"LLM transport failure ({provider}): {error_message[:500]}",
            level="error",
        )


class BlockMetricsRegistry:
    """In-process rolling stats from UniversalBlock.execute() timings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, Any]] = {}

    def record(self, block_name: str, duration_ms: int, status: str) -> None:
        if not block_name:
            return
        with self._lock:
            entry = self._stats.setdefault(
                block_name,
                {
                    "execution_count": 0,
                    "total_ms": 0,
                    "error_count": 0,
                    "last_ms": 0,
                    "last_status": status,
                },
            )
            entry["execution_count"] += 1
            entry["total_ms"] += duration_ms
            entry["last_ms"] = duration_ms
            entry["last_status"] = status
            if status == "error":
                entry["error_count"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            blocks: Dict[str, Dict[str, Any]] = {}
            for name, entry in self._stats.items():
                count = entry["execution_count"]
                blocks[name] = {
                    "execution_count": count,
                    "avg_ms": round(entry["total_ms"] / count, 2) if count else 0.0,
                    "last_ms": entry["last_ms"],
                    "error_count": entry["error_count"],
                    "last_status": entry["last_status"],
                }
            return {
                "blocks": blocks,
                "tracked_blocks": len(blocks),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


block_metrics = BlockMetricsRegistry()


def record_block_execution(block_name: str, duration_ms: int, status: str) -> None:
    try:
        block_metrics.record(block_name, duration_ms, status)
    except Exception:
        logger.debug("block_metrics.record failed for %s", block_name, exc_info=True)


async def get_observability_health_payload() -> Dict[str, Any]:
    from app.core.redis_client import redis_health
    snap = block_metrics.snapshot()
    return {
        "observability": {
            "sentry_enabled": sentry_enabled(),
            "structured_logging": structured_logging_enabled(),
            "request_tracing": True,
        },
        "block_metrics": snap["blocks"],
        "redis": await redis_health(),
    }


async def observability_middleware(request: Request, call_next) -> Response:
    """Assign/propagate request_id and emit structured access logs.

    PR #98: also bumps the Prometheus request counter so the new
    ``/metrics`` endpoint exposes survivable per-method/per-status
    request totals (the in-memory ``block_metrics`` resets on restart
    + per worker; this counter is the same per-process limitation but
    is intended as the seed for cumulative scraping).
    """
    incoming = request.headers.get("X-Request-ID", "").strip()
    rid = incoming or str(uuid.uuid4())[:12]
    token = request_id_ctx.set(rid)
    request.state.request_id = rid
    start = datetime.now(timezone.utc)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        _bump_prometheus_request_counter(request.method, response.status_code)
        return response
    except Exception:
        _bump_prometheus_request_counter(request.method, 500)
        logger.exception(
            "request_failed",
            extra={
                "request_id": rid,
                "path": request.url.path,
                "method": request.method,
                "event": "request_failed",
            },
        )
        raise
    finally:
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        logger.info(
            "request_complete",
            extra={
                "request_id": rid,
                "path": request.url.path,
                "method": request.method,
                "duration_ms": duration_ms,
                "event": "request_complete",
            },
        )
        request_id_ctx.reset(token)


# ── Prometheus counter (PR #98) ───────────────────────────────────────────
# Lazy import so the module stays importable in environments where
# prometheus-client isn't installed (e.g. a minimal CI tier). When the
# library is present, every request increments by (method, status).
_PROM_REQUESTS_TOTAL = None


def _bump_prometheus_request_counter(method: str, status: int) -> None:
    global _PROM_REQUESTS_TOTAL
    if _PROM_REQUESTS_TOTAL is None:
        try:
            from prometheus_client import Counter
            _PROM_REQUESTS_TOTAL = Counter(
                "the_fork_requests_total",
                "Total HTTP requests handled by the FastAPI app",
                ["method", "status"],
            )
        except Exception:
            return
    try:
        _PROM_REQUESTS_TOTAL.labels(method=method, status=str(status)).inc()
    except Exception:
        logger.warning(
            "swallowed %s in _bump_prometheus_request_counter() — continuing",
            "Exception", exc_info=True,
        )


# ── Provider monitoring block (Lego layer) ────────────────────────────────

from app.infra.lego_base import LegoBlock
from typing import Any, Dict, List, Optional
import time
import statistics
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
import json

class MonitoringBlock(LegoBlock):
    """
    Monitoring & Provider Leaderboard Block
    Tracks reliability scores, latency, auto-routes based on performance
    """
    
    name = "monitoring"
    version = "1.0.0"
    requires = ["config", "memory"]  # Uses Memory Block for metrics storage
    layer = 2  # Monitoring layer
    tags = ["monitoring", "observability", "core"]
    # The roster must match the ladder the platform can actually call:
    # DeepSeek (primary) and OpenRouter (fallback), the only providers
    # `_llm_config` can return. Removed providers (OpenAI, Anthropic, Kimi,
    # Groq, Ollama) must not reappear. Pinned against the providers
    # `_llm_config` can return by tests/test_monitoring_roster_is_honest.py,
    # so this cannot drift again.
    default_config = {
        "track_providers": ["openrouter", "deepseek"],
        "window_size": 100,
        "prediction_threshold": 0.3
    }
    
    def __init__(self, hal_block, config: Dict[str, Any]):
        super().__init__(hal_block, config)
        self.memory_block = None  # Wired by assembler
        
        # Provider tracking
        self.providers = {
            "openrouter": {"name": "OpenRouter (free)", "type": "cloud", "region": "global"},
            "deepseek": {"name": "DeepSeek", "type": "cloud", "region": "global"},
        }
        
        # Metrics window (last 100 calls per provider)
        self.metrics_window = 100
        self.latency_history = defaultdict(lambda: deque(maxlen=self.metrics_window))
        self.error_history = defaultdict(lambda: deque(maxlen=self.metrics_window))
        self.uptime_history = defaultdict(lambda: deque(maxlen=self.metrics_window))
        
        # Reliability scores (0-100)
        self.reliability_scores = {p: 100.0 for p in self.providers}
        
        # Auto-routing threshold
        self.degraded_threshold = 70  # Below 70% = avoid
        self.critical_threshold = 40   # Below 40% = emergency only
        
        self.last_leaderboard_update = 0
        self.leaderboard_cache = None
    
    async def initialize(self):
        """Initialize monitoring"""
        print("[monitoring] Monitoring Block initialized")
        print("   Tracking:", list(self.providers.keys()))
        print("   Auto-route thresholds: Degraded <70%, Critical <40%")
        return True
    
    async def execute(self, input_data: Dict) -> Dict:
        """Monitoring operations"""
        action = input_data.get("action")
        
        if action == "record_call":
            return await self._record_call(input_data)
        elif action == "leaderboard":
            return await self._get_leaderboard()
        elif action == "provider_status":
            return await self._get_provider_status(input_data.get("provider"))
        elif action == "recommend":
            return await self._recommend_provider()
        elif action == "health_report":
            return await self._health_report()
        elif action == "predictive_failover":
            return await self._predictive_analysis()
        
        return {"error": f"Unknown action: {action}"}
    
    async def _record_call(self, data: Dict) -> Dict:
        """Record API call metrics"""
        provider = data.get("provider")
        latency_ms = data.get("latency_ms", 0)
        success = data.get("success", True)
        error_type = data.get("error_type")
        timestamp = time.time()
        
        if provider not in self.providers:
            return {"error": "Unknown provider"}
        
        # Store in memory block for persistence
        if self.memory_block:
            await self.memory_block.execute({
                "action": "set",
                "key": f"metrics:{provider}:{int(timestamp)}",
                "value": {
                    "latency": latency_ms,
                    "success": success,
                    "error": error_type,
                    "timestamp": timestamp
                },
                "ttl": 86400  # 24 hour retention
            })
        
        # Update in-memory windows
        self.latency_history[provider].append(latency_ms)
        self.error_history[provider].append(0 if success else 1)
        self.uptime_history[provider].append(1 if success else 0)
        
        # Recalculate reliability score
        await self._calculate_reliability(provider)
        
        return {"recorded": True, "provider": provider}
    
    async def _calculate_reliability(self, provider: str):
        """Calculate provider reliability score (0-100)"""
        if not self.error_history[provider]:
            return
        
        # Error rate (weight: 50%)
        error_rate = sum(self.error_history[provider]) / len(self.error_history[provider])
        
        # Latency score (weight: 30%) - optimal is <500ms
        avg_latency = statistics.mean(self.latency_history[provider]) if self.latency_history[provider] else 0
        latency_score = max(0, 100 - (avg_latency / 10))  # 1000ms = 0 score
        
        # Uptime (weight: 20%)
        uptime_rate = statistics.mean(self.uptime_history[provider]) if self.uptime_history[provider] else 1
        
        # Composite score
        reliability = (uptime_rate * 50) + (latency_score * 0.30) + ((1 - error_rate) * 20)
        self.reliability_scores[provider] = round(reliability, 2)
        
        # Invalidate leaderboard cache
        self.leaderboard_cache = None
    
    async def _get_leaderboard(self) -> Dict:
        """Get provider leaderboard - ranked by reliability"""
        # Check cache (refresh every 60s)
        if self.leaderboard_cache and (time.time() - self.last_leaderboard_update) < 60:
            return self.leaderboard_cache
        
        leaderboard = []
        
        for provider_id, info in self.providers.items():
            score = self.reliability_scores[provider_id]
            latencies = list(self.latency_history[provider_id])
            errors = list(self.error_history[provider_id])
            total_calls = len(latencies)

            # Determine status. A provider with ZERO observed calls has not been
            # PROBED — reporting it "excellent"/100% (the seeded default) is a
            # capability claim the container hasn't earned (audit §8.6). Report
            # "unknown" until there is real evidence.
            if total_calls == 0:
                status = "unknown"
                color = "grey"
                reported_score = None
                recommendation = "unproven"
            else:
                reported_score = score
                if score >= 90:
                    status = "excellent"
                    color = "green"
                elif score >= 70:
                    status = "good"
                    color = "yellow"
                elif score >= 40:
                    status = "degraded"
                    color = "orange"
                else:
                    status = "critical"
                    color = "red"
                recommendation = (
                    "use" if score >= 70 else "avoid" if score >= 40 else "emergency_only"
                )

            leaderboard.append({
                "rank": 0,  # Set later
                "provider": provider_id,
                "name": info["name"],
                "type": info["type"],
                "region": info["region"],
                "reliability_score": reported_score,
                "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
                "error_rate_percent": round(sum(errors) / len(errors) * 100, 2) if errors else 0,
                "total_calls": total_calls,
                "status": status,
                "color": color,
                "recommendation": recommendation,
            })
        
        # Sort by reliability score (descending); "unknown" (None) providers
        # sort last — an unproven provider is never ranked above an observed one.
        leaderboard.sort(
            key=lambda x: x["reliability_score"] if x["reliability_score"] is not None else -1.0,
            reverse=True,
        )
        
        # Assign ranks
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        # Cache result
        # `top_provider` must name an OBSERVED winner. Taking leaderboard[0]
        # unconditionally promoted an unproven provider to "top" whenever
        # nothing had been recorded — which is the normal state — presenting an
        # arbitrary roster entry as a measured result.
        observed = [e for e in leaderboard if e["total_calls"]]
        self.leaderboard_cache = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "leaderboard": leaderboard,
            "top_provider": observed[0]["provider"] if observed else None,
            # Nothing routes on this. `_call_llm` picks its provider from
            # `_llm_config` and its fallback from `_llm_fallback_config`;
            # neither reads the leaderboard. Reporting True advertised a
            # capability the platform does not have.
            "auto_route_enabled": False,
        }
        self.last_leaderboard_update = time.time()
        
        return self.leaderboard_cache
    
    async def _recommend_provider(self) -> Dict:
        """Recommend a provider from OBSERVED reliability.

        Not "AI-powered" -- it is a threshold sort over recorded latency and
        error rates. The old wording promised a judgement the code does not
        make."""
        leaderboard = (await self._get_leaderboard())["leaderboard"]
        
        if not leaderboard:
            return {"recommended": None, "reason": "no_providers_configured"}

        # Find best available provider
        for entry in leaderboard:
            if entry["recommendation"] == "use":
                return {
                    "recommended": entry["provider"],
                    "confidence": entry["reliability_score"],
                    "reason": f"Best reliability at {entry['avg_latency_ms']}ms avg latency",
                    "fallback_sequence": [e["provider"] for e in leaderboard if e["recommendation"] in ["use", "avoid"]]
                }

        # NOTHING OBSERVED is not the same finding as ALL DEGRADED.
        #
        # Until 2026-08-12 this branch returned "All cloud providers degraded -
        # using edge fallback" with emergency_mode=True and confidence 100.0.
        # Nothing internal calls record_call — the only writer is the
        # POST /v1/metrics/record endpoint — so on a normal deployment every
        # provider has zero observations, every entry is "unproven", and this
        # branch fired on EVERY call. The endpoint reported a platform-wide
        # provider outage, at full confidence, as its steady state.
        #
        # That is the confident-zero pattern: absence of data rendered as a
        # diagnosis. Distinguish the two.
        if all(e["total_calls"] == 0 for e in leaderboard):
            return {
                "recommended": None,
                "confidence": None,
                "reason": (
                    "No provider calls have been recorded, so there is no "
                    "basis for a recommendation. This is not an outage: "
                    "nothing feeds POST /v1/metrics/record."
                ),
                "emergency_mode": False,
                "observed": False,
            }

        degraded = [e["provider"] for e in leaderboard if e["total_calls"]]
        return {
            "recommended": None,
            "confidence": None,
            "reason": (
                "Every observed provider is below the 'use' threshold: "
                + ", ".join(degraded)
            ),
            "emergency_mode": True,
            "observed": True,
        }
    
    async def _predictive_analysis(self) -> Dict:
        """Predict potential failures before they happen"""
        predictions = []
        
        for provider in self.providers:
            latencies = list(self.latency_history[provider])
            
            if len(latencies) >= 10:
                # Check for degrading trend (increasing latency)
                recent_avg = statistics.mean(latencies[-5:])
                older_avg = statistics.mean(latencies[-10:-5])
                
                if recent_avg > older_avg * 1.5:  # 50% increase
                    predictions.append({
                        "provider": provider,
                        "prediction": "latency_spike",
                        "severity": "medium",
                        "recommendation": f"Preemptively reroute {provider} traffic",
                        "confidence": 75
                    })
                
                # Check error rate trend
                recent_errors = sum(list(self.error_history[provider])[-10:])
                if recent_errors >= 3:  # 3 errors in last 10 calls
                    predictions.append({
                        "provider": provider,
                        "prediction": "impending_failure",
                        "severity": "high",
                        "recommendation": f"Switch {provider} to fallback immediately",
                        "confidence": 85
                    })
        
        return {
            "predictions": predictions,
            "preventive_actions_recommended": len(predictions) > 0,
            "analyzed_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _health_report(self) -> Dict:
        """Full system health report"""
        lb = await self._get_leaderboard()
        pred = await self._predictive_analysis()
        
        # Three states, not two. "degraded" is a claim about observed
        # providers; with nothing recorded the honest answer is "unknown".
        # Reporting degraded on no data cries wolf on every single call, which
        # is how a real degradation gets ignored.
        # Judge only providers that have actually SERVED calls. The old
        # top-2 slice mixed never-used fallbacks into the verdict: one
        # healthy kimi call + groq at 0 calls ("unproven") reported the
        # whole system degraded -- a false alarm on a perfectly healthy
        # deployment (observed live, phase-4 route census).
        observed = [e for e in lb["leaderboard"] if e["total_calls"] > 0]
        if not observed:
            overall = "unknown"
        elif all(e["recommendation"] == "use" for e in observed):
            overall = "healthy"
        else:
            overall = "degraded"

        return {
            "overall_status": overall,
            "leaderboard": lb,
            "predictions": pred,
            # Not a probe result. Nothing here tests that a failover would
            # actually succeed, so it states what it is.
            "failover_readiness": "not_probed",
            "recommendation": await self._recommend_provider()
        }
    
    async def _get_provider_status(self, provider: Optional[str]) -> Dict:
        """Get specific provider status"""
        if not provider:
            return {"providers": {p: self.reliability_scores[p] for p in self.providers}}
        
        if provider not in self.providers:
            return {"error": "Unknown provider"}
        
        return {
            "provider": provider,
            "reliability_score": self.reliability_scores[provider],
            "metrics": {
                "latency_history": list(self.latency_history[provider])[-10:],
                "error_history": list(self.error_history[provider])[-10:]
            }
        }
    
    def health(self) -> Dict:
        h = super().health()
        h["providers_tracked"] = len(self.providers)
        h["metrics_retention"] = f"{self.metrics_window} calls per provider"
        h["auto_route_enabled"] = False   # nothing reads the leaderboard to route
        return h
