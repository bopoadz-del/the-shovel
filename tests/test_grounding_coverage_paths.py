"""Market-readiness audit §3.2 — cost-grounding coverage on the two answer
paths that previously bypassed the gate: ``/v1/project/ask`` (ProjectReasoner)
and non-streaming ``/chat``.

The gate itself is proven in ``test_cost_grounding_gate.py``; this file proves
the *routing* — that each endpoint now runs its LLM/reasoner answer through the
SAME tested gate before returning. An ungrounded cost/rate figure must get the
canonical ``_CG_REFUSAL``; a grounded/cited figure and a plain non-cost answer
must pass through byte-for-byte unchanged (no over-refusal).

Mocking mirrors the established fixtures: the reasoner is swapped via
``project_router._reasoner_factory`` (as in test_project_ask_tenancy) and the
chat block via ``block_instances['chat']`` — no real LLM calls.
"""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents.runtime import _CG_REFUSAL
from app.routers import project as project_router
from app.core.session_store import InMemorySessionStore
from tests._dev_key import DEV_KEY

_RUN = uuid.uuid4().hex[:8]
_LEGACY = {"Authorization": "Bearer " + DEV_KEY}

# ── A rate-semantic retrieved chunk that grounds "220–280 SAR/m3" ──────────
_RATE_SNIPPET = (
    "## Detailed material prices — Saudi Arabia (SAR) | Material | Unit | Price "
    "range (SAR) | Ready-mix concrete (250 kg/cm2) | /m3 | 220 — 280 | "
    "Source: Turner & Townsend KSAMI 2025."
)
_GROUNDED_ANSWER = (
    "Ready-mix concrete (250 kg/cm2) in Saudi Arabia is about 220–280 SAR/m3 "
    "(indicative; Turner & Townsend KSAMI 2025)."
)
_UNGROUNDED_ANSWER = (
    "The unit rate for ready-mix concrete is 450 SAR per m3."
)
_NON_COST_ANSWER = (
    "The project has 12 work packages and 3 open RFIs across 2 zones."
)


# ══════════════════════════════════════════════════════════════════════════
# /v1/project/ask  (ProjectReasonerBlock)
# ══════════════════════════════════════════════════════════════════════════
class _MockReasoner:
    """Reasoner stand-in returning a controlled answer plus the internal
    ``_grounding`` evidence the router feeds to the cost gate."""

    def __init__(self, answer, excerpts=None, results_text=""):
        self._answer = answer
        self._grounding = {
            "excerpts": excerpts or [],
            "results_text": results_text,
        }

    async def process(self, inputs):
        return {
            "status": "success",
            "answer": self._answer,
            "understanding": "",
            "plan": None,
            "execution": None,
            "sources": [],
            "_grounding": self._grounding,
        }


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cov_data")
    os.environ.setdefault("DATA_DIR", str(tmp))
    store = InMemorySessionStore()
    with TestClient(app) as c:
        project_router._store = store
        yield c


def _ask(client, answer, *, excerpts=None, results_text=""):
    project_router._reasoner_factory = lambda: _MockReasoner(
        answer, excerpts=excerpts, results_text=results_text
    )
    sid = f"cov-{_RUN}-{uuid.uuid4().hex[:6]}"
    r = client.post(
        "/v1/project/ask",
        json={"session_id": sid, "request": "what is the concrete rate?"},
        headers=_LEGACY,
    )
    assert r.status_code == 200, r.text
    return r.json()["answer"]


def test_project_ask_ungrounded_cost_is_refused(client):
    out = _ask(client, _UNGROUNDED_ANSWER)
    assert out == _CG_REFUSAL
    assert "450" not in out


def test_project_ask_grounded_cost_passes(client):
    excerpts = [{"document_id": "rates_ksa", "snippet": _RATE_SNIPPET, "score": 0.81}]
    out = _ask(client, _GROUNDED_ANSWER, excerpts=excerpts)
    assert out == _GROUNDED_ANSWER  # cited, rate-semantic evidence -> untouched


def test_project_ask_computed_cost_is_grounded_by_step_results(client):
    # A real executed step result is authoritative: its numbers ground the
    # delivered prose even with no rate chunk (no false refusal of deliverables).
    answer = "Estimated total cost is SAR 1,250,000 for the concrete package."
    out = _ask(client, answer, results_text='{"total_cost": 1250000, "currency": "SAR"}')
    assert out == answer


def test_project_ask_non_cost_answer_passes_unchanged(client):
    out = _ask(client, _NON_COST_ANSWER)
    assert out == _NON_COST_ANSWER


# ══════════════════════════════════════════════════════════════════════════
# /chat  (non-streaming)
# ══════════════════════════════════════════════════════════════════════════
class _MockChatBlock:
    def __init__(self, text):
        self._text = text

    async def execute(self, message, opts):
        return {"result": {"text": self._text, "model": "mock", "provider": "mock"}}


@pytest.fixture
def chat_env(client, monkeypatch):
    """Neutralise the orchestrator hint so /chat stays hermetic; hand back a
    setter that installs a controlled chat block + optional doc grounding."""
    async def _passthrough_hint(prompt):
        return prompt

    monkeypatch.setattr("app.routers.chat._with_domain_hint", _passthrough_hint)

    from app.dependencies import block_instances

    def _install(answer, *, project_id=None, snippet=None):
        monkeypatch.setitem(block_instances, "chat", _MockChatBlock(answer))
        if project_id is not None:
            # Grant tenant access and serve one indexed snippet; keep project
            # memory empty so grounding comes solely from doc search.
            monkeypatch.setattr(
                "app.core.projects.get_project_accessible",
                lambda pid, uid: {"id": pid},
            )
            monkeypatch.setattr(
                "app.core.project_memory.build_memory_context",
                lambda pid, prompt: "",
            )

            async def _search(pid, prompt, top_k=5):
                return (
                    [{"document_id": "rates_ksa", "filename": "rates.md",
                      "snippet": snippet, "score": 0.81}]
                    if snippet else []
                )

            monkeypatch.setattr("app.core.doc_index.search_project_documents", _search)
        return _install

    return _install


def _chat(client, project_id=None):
    body = {"message": "what is the concrete unit rate?"}
    if project_id is not None:
        body["project_id"] = project_id
    r = client.post("/chat", json=body, headers=_LEGACY)
    assert r.status_code == 200, r.text
    return r.json()["text"]


def test_chat_ungrounded_cost_is_refused(client, chat_env):
    chat_env(_UNGROUNDED_ANSWER)  # no project -> no grounding
    out = _chat(client)
    assert out == _CG_REFUSAL
    assert "450" not in out


def test_chat_grounded_cost_passes(client, chat_env):
    pid = f"proj-{_RUN}"
    chat_env(_GROUNDED_ANSWER, project_id=pid, snippet=_RATE_SNIPPET)
    out = _chat(client, project_id=pid)
    assert out == _GROUNDED_ANSWER  # figures trace to the retrieved rate chunk


def test_chat_non_cost_answer_passes_unchanged(client, chat_env):
    chat_env(_NON_COST_ANSWER)
    out = _chat(client)
    assert out == _NON_COST_ANSWER
