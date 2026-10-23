"""Per-request retrieval trace: which chunks were handed to the model.

One structured server-log line per ``rag_inject`` call, keyed by the request
id the observability middleware already assigns. It carries chunk ids,
document names, scores and ranks. It never carries chunk text, and it never
reaches an API response. ``RAG_RETRIEVAL_TRACE=0`` turns it off.

Every chunk, document name and message here is synthetic.
"""
from __future__ import annotations

import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.runtime import Agent
from app.core.rag.vector_store import Chunk
from app.infra.monitoring import request_id_ctx
from tests._fixture_values import PW_06, STUB_TOKEN_02

TRACE_LOGGER = "app.rag.retrieval_trace"
TRACE_EVENT = "rag_retrieval_trace"
BODY_MARK = "TRACEFIXTUREBODY"

# (chunk_id, doc_id, doc name, score)
_SYNTH = [
    ("synth-p1:doc-a:0", "doc-a", "FIXTURE-trace-Specification-Vol-9.pdf", 0.91),
    ("synth-p1:doc-b:3", "doc-b", "FIXTURE-trace-Drawing-XX-001.pdf", 0.74),
    ("synth-p1:doc-c:7", "doc-c", "FIXTURE-trace-Contract-Data.pdf", 0.52),
]

# G3: the log must not depend on how the question is worded.
PHRASINGS = [
    "What minimum cover does the synthetic specification require for pads?",
    "Per the synthetic spec, how much cover is required below pad footings?",
    "On the synthetic drawing, what cover sits below the pads?",
    "cover for pads cast against ground according to the fixture specification",
]


def _chunks(project_id, scale=1.0):
    return [
        Chunk(
            chunk_id=cid, project_id=project_id, doc_id=did, chunk_index=i,
            text=f"{BODY_MARK}-{i} invented clause wording number {i} for tests only.",
            score=round(score * scale, 6), source_name=name,
        )
        for i, (cid, did, name, score) in enumerate(_SYNTH)
    ]


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def trace_log(monkeypatch):
    """A handler on the trace logger itself, so the result does not depend on
    root-logger config or propagation set up by other tests."""
    lg = logging.getLogger(TRACE_LOGGER)
    cap = _Capture()
    monkeypatch.setattr(lg, "disabled", False)
    monkeypatch.setattr(lg, "level", logging.INFO)
    lg.addHandler(cap)
    yield cap
    lg.removeHandler(cap)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("MAX_RAG_TOKENS", "4000")
    monkeypatch.delenv("RAG_RETRIEVAL_TRACE", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


def _fake_retrieve(scale=1.0):
    def fake(query, project_id, k=5, **kwargs):
        return _chunks(project_id, scale)[:k], 0
    return fake


def _trace_lines(cap):
    return [r for r in cap.records if r.getMessage().startswith(TRACE_EVENT)]


def _payload(record):
    msg = record.getMessage()
    return json.loads(msg[len(TRACE_EVENT):].strip())


def _everything_logged(cap):
    out = []
    for r in cap.records:
        out.append(r.getMessage())
        out.append(json.dumps(r.__dict__, default=str))
    return "\n".join(out)


def _inject(ask, rid="rid-trace-0001", project_id="synth-p1"):
    from app.core.rag.inject import rag_inject

    token = request_id_ctx.set(rid)
    try:
        return rag_inject(
            user_message=ask, project_id=project_id, conversation_id="ws-synth",
            user_id="u-synth", agent_name="project-assistant",
        )
    finally:
        request_id_ctx.reset(token)


@pytest.mark.parametrize("ask", PHRASINGS)
def test_one_line_with_ids_names_scores_ranks(ask, trace_log, monkeypatch):
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", _fake_retrieve())
    sys_msg, _audit = _inject(ask)
    assert sys_msg is not None

    lines = _trace_lines(trace_log)
    assert len(lines) == 1, [r.getMessage() for r in trace_log.records]
    rec = lines[0]
    assert getattr(rec, "request_id", None) == "rid-trace-0001"
    assert getattr(rec, "event", None) == TRACE_EVENT
    p = _payload(rec)
    assert p["request_id"] == "rid-trace-0001"
    assert p["path"] == "pre_injection"
    assert p["query"] == "user_message"
    assert p["handed_k"] == 3
    got = [(c["rank"], c["chunk_id"], c["doc_id"], c["doc_name"], c["score"])
           for c in p["chunks"]]
    want = [(i + 1, cid, did, name, score)
            for i, (cid, did, name, score) in enumerate(_SYNTH)]
    assert got == want


@pytest.mark.parametrize("ask", PHRASINGS)
def test_no_chunk_text_and_no_message_text_in_log(ask, trace_log, monkeypatch):
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", _fake_retrieve())
    _inject(ask)
    assert _trace_lines(trace_log)
    blob = _everything_logged(trace_log)
    assert BODY_MARK not in blob
    assert "invented clause wording" not in blob
    assert ask not in blob


def test_threshold_miss_logs_nothing_handed(trace_log, monkeypatch):
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", _fake_retrieve(0.3))
    sys_msg, _audit = _inject(PHRASINGS[0], rid="rid-trace-miss")
    assert sys_msg is None
    lines = _trace_lines(trace_log)
    assert len(lines) == 1
    p = _payload(lines[0])
    assert p["request_id"] == "rid-trace-miss"
    assert p["handed_k"] == 0 and p["chunks"] == []
    assert p["threshold_fired"] is True
    assert [c["chunk_id"] for c in p["not_handed"]] == [s[0] for s in _SYNTH]
    assert [c["rank"] for c in p["not_handed"]] == [1, 2, 3]
    assert BODY_MARK not in _everything_logged(trace_log)


def test_followup_expansion_is_labelled_not_quoted(trace_log, monkeypatch):
    from app.core.rag.inject import rag_inject

    monkeypatch.setenv("RAG_FOLLOWUP_CONTEXT", "1")
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", _fake_retrieve())
    history = [{"role": "user", "content": "Synthetic pad footing cover per fixture spec"}]
    token = request_id_ctx.set("rid-trace-follow")
    try:
        rag_inject(user_message="and slabs?", project_id="synth-p1",
                   conversation_id="ws-synth", user_id="u", agent_name="a",
                   history=history)
    finally:
        request_id_ctx.reset(token)
    p = _payload(_trace_lines(trace_log)[0])
    assert p["query"] == "followup_context"
    assert "Synthetic pad footing" not in _everything_logged(trace_log)


@pytest.mark.parametrize("off", ["0", "false", "off", "no"])
def test_flag_off_emits_nothing(off, trace_log, monkeypatch):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", off)
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", _fake_retrieve())
    sys_msg, _ = _inject(PHRASINGS[1])
    assert sys_msg is not None
    assert _trace_lines(trace_log) == []


def test_default_is_on_in_code():
    from app.core.rag import inject

    assert inject.retrieval_trace_enabled() is True


def _strip_volatile(audit):
    return {k: v for k, v in audit.items()
            if k not in ("timestamp", "budget_remaining")}


@pytest.mark.parametrize("ask", PHRASINGS[:2])
def test_rag_inject_return_is_identical_on_and_off(ask, trace_log, monkeypatch):
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", _fake_retrieve())
    msg_on, audit_on = _inject(ask)
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "0")
    msg_off, audit_off = _inject(ask)
    assert msg_on == msg_off
    assert _strip_volatile(audit_on) == _strip_volatile(audit_off)
    assert TRACE_EVENT not in json.dumps(audit_on, default=str)
    # the trace line was emitted for the ON call only
    assert len(_trace_lines(trace_log)) == 1


# ── HTTP: the request id from the middleware reaches the line; the response
#    carries no trace ─────────────────────────────────────────────────────────

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _llm_ok(text="Synthetic mock answer."):
    async def fake(self, messages, api_key, project_id=None, **kwargs):
        return {"status": "success",
                "choice": {"message": {"content": text, "tool_calls": []}},
                "raw": {}}
    return fake


def _login(client, suffix):
    email = f"trace-{suffix}-{_RUN}@example.com"
    client.post("/v1/users/register", json={"email": email, "password": PW_06})
    tok = client.post("/v1/users/login",
                      json={"email": email, "password": PW_06}).json()["token"]
    return {"Authorization": f"Bearer {tok}"}


def _http_setup(client, monkeypatch, suffix):
    monkeypatch.setattr(Agent, "_call_llm", _llm_ok())
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", _fake_retrieve())
    headers = _login(client, suffix)
    pid = client.post("/v1/projects", json={"name": f"Trace {suffix}"},
                      headers=headers).json()["id"]
    return headers, pid


@pytest.mark.parametrize("ask", PHRASINGS[:2])
def test_http_chat_line_keyed_by_request_id_and_response_clean(
    ask, client, trace_log, monkeypatch,
):
    headers, pid = _http_setup(client, monkeypatch, "json")
    rid = f"rid-http-{uuid.uuid4().hex[:6]}"
    resp = client.post(
        "/v1/agents/smart-orchestrator/chat",
        json={"message": ask, "project_id": pid},
        headers={**headers, "X-Request-ID": rid},
    )
    assert resp.status_code == 200, resp.text
    lines = [r for r in _trace_lines(trace_log) if _payload(r)["request_id"] == rid]
    assert len(lines) == 1, [r.getMessage() for r in trace_log.records]
    assert [c["chunk_id"] for c in _payload(lines[0])["chunks"]] == [s[0] for s in _SYNTH]
    body = resp.text
    assert TRACE_EVENT not in body
    assert "retrieval_trace" not in body
    assert '"not_handed"' not in body and '"handed_k"' not in body


def test_http_stream_response_carries_no_trace(client, trace_log, monkeypatch):
    headers, pid = _http_setup(client, monkeypatch, "sse")
    rid = f"rid-sse-{uuid.uuid4().hex[:6]}"
    resp = client.post(
        "/v1/agents/smart-orchestrator/chat/stream",
        json={"message": PHRASINGS[2], "project_id": pid},
        headers={**headers, "X-Request-ID": rid},
    )
    assert resp.status_code == 200, resp.text
    body = resp.text
    lines = [r for r in _trace_lines(trace_log) if _payload(r)["request_id"] == rid]
    assert len(lines) == 1, [r.getMessage() for r in trace_log.records]
    assert TRACE_EVENT not in body
    assert "retrieval_trace" not in body
    assert '"not_handed"' not in body and '"handed_k"' not in body


def test_http_flag_off_no_line(client, trace_log, monkeypatch):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "0")
    headers, pid = _http_setup(client, monkeypatch, "off")
    resp = client.post(
        "/v1/agents/smart-orchestrator/chat",
        json={"message": PHRASINGS[3], "project_id": pid},
        headers={**headers, "X-Request-ID": "rid-http-off"},
    )
    assert resp.status_code == 200, resp.text
    assert _trace_lines(trace_log) == []
