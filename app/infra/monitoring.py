"""Platform observability utilities and provider monitoring block."""

from __future__ import annotations

import atexit
import json
import logging
import logging.handlers
import os
import queue
import re
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
#: True while an HTTP request is being served by this process (set by
#: :func:`observability_middleware`, carried into threadpool and task copies of
#: the context). ``request_id_ctx`` cannot answer this: ``get_request_id``
#: sets it outside requests too.
web_request_ctx: ContextVar[bool] = ContextVar("in_web_request", default=False)
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
    """Emit one JSON object per log line for log-drain ingestion."""

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


# ── Error-report scrubbing ────────────────────────────────────────────────
# DEFAULT-DENY. The free-form parts of an event -- ``extra``, the fields of a
# context, breadcrumb ``data``, request fields and header values -- are dropped
# unless their field is on a short allowed list of structural/diagnostic names.
# Values are also judged by shape (credential spellings, e-mail addresses,
# over-long passages), never by a particular name or secret value.
# ``send_default_pii=False`` alone still lets exception messages, log
# breadcrumbs and extra context through; this hook closes that gap.

_FILTERED = "[Filtered]"

#: A credential, session, identity or connection token, by spelling. Matched
#: with a letter boundary so a short spelling (``pin``, ``pw``, ``otp``) is a
#: credential while an ordinary word that merely contains it (``spinner``) is
#: not.
_CREDENTIAL_NAMES = (
    r"auth(?:orization|entic\w*)?|"
    r"cookie|token|secret|credential(?:s)?|session|signature|"
    r"pass(?:word|wd|code|phrase)?|pwd|pw|pin|otp|sid|"
    r"api[-_]?key|private[-_]?key|access[-_]?key|keys?|"
    r"dsn|conn(?:ection)?[-_]?str|(?:database|db)[-_]?ur[il]|e-?mail"
)
_CREDENTIAL_RE = re.compile(r"(?<![a-z])(" + _CREDENTIAL_NAMES + r")(?![a-z])", re.I)

#: Field names that never carry a diagnostic value and are dropped wherever
#: they appear (frame local variables -- ``include_local_variables`` is off,
#: but never trust that alone).
_DROP_FIELDS = frozenset({"vars", "locals"})

#: Free text longer than this is payload (a pasted page, document text), not a
#: diagnostic message, and is not carried.
_MAX_TEXT_CHARS = 200

#: Standard, safe HTTP header names (lowercase). Any other header keeps its
#: name (diagnostic) but loses its value.
_ALLOWED_HEADERS = frozenset({
    "accept", "accept-encoding", "accept-language", "allow", "cache-control",
    "connection", "content-encoding", "content-language", "content-length",
    "content-type", "date", "host", "location", "origin", "referer", "server",
    "user-agent", "vary",
})

#: Context names that are pure diagnostics and safe to carry; any other
#: context is dropped whole.
_ALLOWED_CONTEXTS = frozenset({
    "app", "browser", "device", "gpu", "os", "otel", "runtime", "trace",
})

#: Fields inside an allowed context that are safe to carry.
_ALLOWED_CONTEXT_FIELDS = frozenset({
    "build", "description", "name", "op", "parent_span_id", "span_id",
    "status", "trace_id", "version",
})

#: Request keys that are diagnosis, not payload. The request's other fields
#: (query string, body, env, cookies, data) are dropped.
_ALLOWED_REQUEST_FIELDS = frozenset({
    "api_target", "body_size", "finished", "fragment",
    "inferred_content_type", "method", "started", "url",
})

#: Breadcrumb keys that are diagnosis; ``data`` (free-form) and anything else
#: are dropped.
_ALLOWED_BREADCRUMB_FIELDS = frozenset({
    "category", "event_id", "level", "message", "timestamp", "type",
})

#: User identity fields that are safe to carry.
_ALLOWED_USER_FIELDS = frozenset({"id", "ip_address", "segment", "username"})

#: Shapes of secret material inside free text (messages, URLs, log lines).
_TEXT_SCRUBS = (
    # scheme://user:password@host -> scheme://[Filtered]@host
    (re.compile(r"(\b[a-z][a-z0-9+.\-]*://)[^\s/@]+@", re.I), r"\1" + _FILTERED + "@"),
    # name=value / name: value where the name is a credential spelling
    (re.compile(
        r"(?<![a-z])(" + _CREDENTIAL_NAMES + r")(?![a-z])"
        r"(\s*[=:]\s*)"
        r"(\"[^\"]*\"|'[^']*'|[^\s&,;]+)",
        re.I,
    ), r"\1\2" + _FILTERED),
    # Authorization schemes inside text
    (re.compile(r"\b(bearer|basic|token)\s+[A-Za-z0-9._~+/=\-]{6,}", re.I), r"\1 " + _FILTERED),
    # JSON web tokens
    (re.compile(r"\beyJ[\w-]+\.[\w-]+\.[\w-]*"), _FILTERED),
    # e-mail addresses
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[email]"),
    # long opaque strings that mix letters and digits (keys, tokens). Pure
    # hexadecimal is left alone: event, trace and span ids and commit shas are
    # the report's own identifiers (a hex secret is still caught by its field
    # kind or by its name=value shape above).
    (re.compile(
        r"\b(?![0-9a-fA-F]{32,}\b)(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{32,}\b"
    ), _FILTERED),
)


def _is_credential_field(name: str) -> bool:
    """Whether ``name`` is (or ends in) a credential, session or token kind."""
    return bool(_CREDENTIAL_RE.search(name))


def _scrub_text(text: str) -> str:
    if len(text) > _MAX_TEXT_CHARS:
        return f"[Filtered: {len(text)} chars]"
    for pattern, repl in _TEXT_SCRUBS:
        text = pattern.sub(repl, text)
    return text


def _scrub_value(value: Any) -> Any:
    """Scrub a scalar or a structural container (text, credential kinds)."""
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {k: _scrub_field(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_value(v) for v in value]
    return value


def _scrub_field(key: Any, value: Any) -> Any:
    """Scrub one structural field: drop local variables, mask credentials."""
    name = str(key)
    if name in _DROP_FIELDS or _is_credential_field(name):
        return _FILTERED if value not in (None, "", {}, []) else value
    return _scrub_value(value)


def _scrub_extra(value: Any) -> Any:
    """``extra`` is app-defined free-form: default-deny drops every field."""
    return {} if isinstance(value, dict) else _scrub_value(value)


def _scrub_contexts(value: Any) -> Any:
    if not isinstance(value, dict):
        return _scrub_value(value)
    out: Dict[str, Any] = {}
    for name, fields in value.items():
        if str(name) not in _ALLOWED_CONTEXTS:
            continue  # unknown context: dropped whole
        if isinstance(fields, dict):
            out[name] = {
                k: _scrub_value(v)
                for k, v in fields.items()
                if str(k) in _ALLOWED_CONTEXT_FIELDS
            }
        else:
            out[name] = _scrub_value(fields)
    return out


def _scrub_headers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            name: _scrub_value(v) if str(name).lower() in _ALLOWED_HEADERS
            else (_FILTERED if v not in (None, "", {}, []) else v)
            for name, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                name, v = item
                if name.lower() in _ALLOWED_HEADERS:
                    out.append([name, _scrub_value(v)])
                else:
                    out.append([name, _FILTERED if v not in (None, "", {}, []) else v])
            else:
                out.append(_scrub_value(item))
        return out
    return _scrub_value(value)


def _scrub_request(value: Any) -> Any:
    if not isinstance(value, dict):
        return _scrub_value(value)
    out: Dict[str, Any] = {}
    for key, v in value.items():
        name = str(key)
        if name in _ALLOWED_REQUEST_FIELDS:
            out[key] = _scrub_value(v)
        elif name == "headers":
            out[key] = _scrub_headers(v)
        # query string, body, data, env, cookies and unknown keys: dropped
    return out


def _scrub_breadcrumb(crumb: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, v in crumb.items():
        if str(key) in _ALLOWED_BREADCRUMB_FIELDS:
            out[key] = _scrub_value(v)
        # ``data`` and any other key are free-form: dropped
    return out


def _scrub_breadcrumbs(value: Any) -> Any:
    if not isinstance(value, dict):
        return _scrub_value(value)
    out: Dict[str, Any] = {}
    for key, v in value.items():
        if key == "values" and isinstance(v, list):
            out[key] = [_scrub_breadcrumb(c) for c in v if isinstance(c, dict)]
        else:
            out[key] = _scrub_field(key, v)
    return out


def _scrub_user(value: Any) -> Any:
    if not isinstance(value, dict):
        return _scrub_value(value)
    out: Dict[str, Any] = {}
    for key, v in value.items():
        name = str(key)
        if _is_credential_field(name):
            out[key] = _FILTERED if v not in (None, "", {}, []) else v
        elif name in _ALLOWED_USER_FIELDS:
            out[key] = _scrub_value(v)
        # anything else is free-form: dropped
    return out


_CONTAINER_HANDLERS = {
    "extra": _scrub_extra,
    "contexts": _scrub_contexts,
    "request": _scrub_request,
    "breadcrumbs": _scrub_breadcrumbs,
    "user": _scrub_user,
}


def _scrub_event(event: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in event.items():
        handler = _CONTAINER_HANDLERS.get(str(key))
        if handler is not None:
            out[key] = handler(value)
        else:
            out[key] = _scrub_field(key, value)
    return out


def scrub_error_event(event: Dict[str, Any], hint: Any = None) -> Optional[Dict[str, Any]]:
    """Sentry before_send / before_send_transaction hook.

    Returns a scrubbed copy of ``event``; the input is not mutated. If the
    scrub itself fails the event is dropped (None): an unscrubbed report must
    never leave.
    """
    try:
        return _scrub_event(event)
    except Exception as exc:  # noqa: BLE001 -- fail closed, never send unscrubbed
        logger.warning("error report dropped: scrub failed (%s)", type(exc).__name__)
        return None


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
        include_local_variables=False,
        max_request_body_size="never",
        before_send=scrub_error_event,
        before_send_transaction=scrub_error_event,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0") or 0),
    )
    _sentry_enabled = True
    return True


def in_web_request() -> bool:
    """Whether the caller runs on behalf of an HTTP request in this process."""
    return web_request_ctx.get()


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
    web_token = web_request_ctx.set(True)
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
        web_request_ctx.reset(web_token)


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


# ── Model-call meter on public /metrics ───────────────────────────────────
# The usage figures live in the usage DB (usage_tracker), not in an in-process
# counter, so a scrape-time collector reads the current aggregate each time
# /metrics is scraped. NAME-FREE: global calls/tokens/cost only, no provider or
# model label — provider+model breakdown stays on the admin /v1/metrics.
_LLM_METER_COLLECTOR_REGISTERED = False


def register_llm_meter_collector() -> None:
    """Register (once) the scrape-time collector exposing the global model-call
    meter on the public Prometheus /metrics:

      the_fork_llm_calls      LLM round-trips recorded (all users)
      the_fork_llm_tokens     LLM total tokens recorded (all users)
      the_fork_llm_cost_usd   LLM estimated cost in USD (all users)

    Gauges, read live from the usage DB at collect time. Idempotent; a missing
    prometheus-client, or a duplicate registration, is a quiet no-op. Carries no
    provider/model name, so it does not widen what the unauthenticated endpoint
    serves beyond non-sensitive aggregates."""
    global _LLM_METER_COLLECTOR_REGISTERED
    if _LLM_METER_COLLECTOR_REGISTERED:
        return
    try:
        from prometheus_client.core import REGISTRY, GaugeMetricFamily
    except ImportError:
        logger.debug("llm meter: prometheus-client unavailable", exc_info=True)
        return

    class _LLMMeterCollector:
        def collect(self):
            try:
                from app.core import usage_tracker
                t = usage_tracker.totals()
            except Exception:  # noqa: BLE001 -- a collector never breaks a scrape
                logger.debug("llm meter collect() swallowed an error", exc_info=True)
                t = {"calls": 0}
            # Public /metrics is NAME-FREE and COUNT-ONLY (ORDERS_0715Z): the
            # call count only -- no tokens, no cost_usd. Token and cost figures
            # are admin-only on /v1/metrics.
            yield GaugeMetricFamily(
                "the_fork_llm_calls",
                "LLM round-trips recorded (all users)",
                value=float(t.get("calls", 0) or 0),
            )

    try:
        REGISTRY.register(_LLMMeterCollector())
    except ValueError:
        # Already registered on a prior scrape — a duplicate registration is fine.
        logger.debug("llm meter collector already registered", exc_info=True)
    _LLM_METER_COLLECTOR_REGISTERED = True


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
