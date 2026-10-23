"""The SPA generic banner is the last mile of a live chat 402.

Live tip 78bd9ca, isolation dual-browser: checks 1-6 PASS; both plain
users' project chat shows ``Something went wrong. Please try again.``
after repeated asks.

Path:

  POST /v1/chat/stream
    → Agent._call_llm returns ``deepseek HTTP 402: insufficient credit``
    → ``chat_watchdog.sanitize_error_frame`` rewrites plumbing to
      ``PROVIDER_FAILURE_MESSAGE``
    → ``ProjectWorkspace.friendlyErrorMessage`` has no ``includes()``
      needle for that sentence
    → ChatBubble renders the generic fallback

This file drives the same path through TestClient (two registered
users, no live credentials) and then checks the SPA mapper source so
the rewrite cannot silently become the generic banner again.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.chat_watchdog import PROVIDER_FAILURE_MESSAGE
from tests._fixture_values import PW_06, STUB_TOKEN_10, STUB_TOKEN_11

_SPA = Path("frontend/src/pages/ProjectWorkspace.tsx")
_GENERIC = "Something went wrong. Please try again."
_LIVE_402 = (
    'deepseek HTTP 402: insufficient credit — '
    '{"error":{"message":"Insufficient Balance",'
    '"type":"unknown_error","param":null,"code":"invalid_request_error"}}'
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client: TestClient, label: str) -> dict:
    email = f"spa-{label}-{uuid.uuid4().hex[:8]}@x.com"
    r = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert r.status_code in (200, 201, 409), r.text
    r = client.post("/v1/users/login", json={"email": email, "password": PW_06})
    assert r.status_code == 200, r.text
    body = r.json()
    return {"token": body["token"], "id": body["user"]["id"]}


def _headers(actor: dict) -> dict:
    return {"Authorization": f"Bearer {actor['token']}"}


def _project(client: TestClient, actor: dict) -> str:
    r = client.post(
        "/v1/projects",
        json={"name": f"SPA-{uuid.uuid4().hex[:6]}"},
        headers=_headers(actor),
    )
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _friendly_includes_needles() -> list[str]:
    """``includes('…')`` needles in ``friendlyErrorMessage`` before the generic return."""
    src = _SPA.read_text(encoding="utf-8")
    start = src.index("function friendlyErrorMessage")
    end = src.index(f"return '{_GENERIC}'", start)
    body = src[start:end]
    return re.findall(r"includes\('([^']+)'\)", body)


def _spa_would_show_generic(raw: str) -> bool:
    r = (raw or "").lower()
    return not any(needle in r for needle in _friendly_includes_needles())


def _stream_error(client: TestClient, actor: dict, project_id: str) -> dict:
    cid = f"ws-{project_id}-{int(uuid.uuid4().int % 10**12)}"
    with client.stream(
        "POST",
        "/v1/chat/stream",
        headers=_headers(actor),
        json={
            # Same shape as tests/test_a_provider_outage_is_survivable.py:
            # no project_id, so the empty-project RAG guardrail cannot
            # skip _call_llm and hide the 402.
            "message": "Explain in one sentence what a construction programme is.",
            "history": [],
            "conversation_id": cid,
        },
    ) as r:
        assert r.status_code == 200, r.text
        events = []
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            try:
                events.append(json.loads(line[5:].strip()))
            except ValueError:
                continue
    err = next((e for e in events if e.get("type") == "error"), None)
    assert err is not None, [e.get("type") for e in events]
    return err


@pytest.fixture
def deepseek_402(monkeypatch):
    """Keys present (as on live /health) — the 402 is injected at ``_call_llm``.

    Without the keys the stream errors earlier with ``No DEEPSEEK_API_KEY
    configured.`` The watchdog rewrite is the same sentence either way;
    setting the keys keeps this on the live hop.
    """
    from app.agents import runtime as rt

    for name in (
        "LLM_PROVIDER", "LLM_FALLBACK_PROVIDER",
        "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_10)
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_11)

    async def outage(self, *a, **k):
        return {"status": "error", "error": _LIVE_402}

    monkeypatch.setattr(rt.Agent, "_call_llm", outage, raising=True)


def test_two_plain_users_stream_402_is_rewritten_not_tenancy_404(client, deepseek_402):
    """Isolation checks 1-6 PASS; only the chat ask fails, for BOTH users."""
    shown = []
    for label in ("a", "b"):
        actor = _register(client, label)
        pid = _project(client, actor)
        err = _stream_error(client, actor, pid)
        msg = err.get("message") or ""
        assert "conversation not found" not in msg.lower()
        assert "404" not in msg
        assert msg == PROVIDER_FAILURE_MESSAGE, msg
        shown.append(msg)
    assert shown[0] == shown[1]


def test_provider_failure_message_is_not_the_spa_generic_banner():
    """No TestClient needed: the watchdog sentence itself is unclassified."""
    assert not _spa_would_show_generic(PROVIDER_FAILURE_MESSAGE), (
        f"SPA friendlyErrorMessage has no includes() needle for "
        f"{PROVIDER_FAILURE_MESSAGE!r}; ChatBubble will show {_GENERIC!r}. "
        f"needles={_friendly_includes_needles()}"
    )


@pytest.fixture
def deepseek_402_openrouter_ok(monkeypatch):
    """Live /health: both keys set. DeepSeek 402s; OpenRouter answers.

    Matches the additional UI/curl capture: POST /v1/chat returned the
    offline-mode template while fallback_ready was true.
    """
    import httpx

    from app.agents.runtime import DEEPSEEK_API_URL

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_10)
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_11)

    class _Resp:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text or ("" if payload is None else json.dumps(payload))

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None, **kwargs):
            if str(url).startswith(DEEPSEEK_API_URL.rsplit("/", 2)[0]):
                return _Resp(
                    402,
                    text='{"error":{"message":"Insufficient Balance"}}',
                )
            return _Resp(
                200,
                payload={"choices": [{"message": {"content": "openrouter recovered"}}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


def test_v1_chat_does_not_serve_offline_when_fallback_is_ready(
    client, deepseek_402_openrouter_ok,
):
    """Live curl: POST /v1/chat 200 body started with offline-mode text."""
    actor = _register(client, "curl")
    r = client.post(
        "/v1/chat",
        headers=_headers(actor),
        json={"message": "Hello, what is 2+2?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    text = (body.get("text") or "").lower()
    assert "offline mode" not in text, body
    assert "no language model is currently reachable" not in text, body
    assert body.get("provider") != "offline_template", body
    assert "openrouter recovered" in (body.get("text") or "")


def test_watchdog_rewrite_must_not_become_spa_something_went_wrong(client, deepseek_402):
    """The sentence the API actually sends must be classified by the SPA.

    Today ``PROVIDER_FAILURE_MESSAGE`` contains none of the mapper's
    ``includes()`` needles (the 402 / insufficient / deepseek tokens were
    stripped on purpose), so ``friendlyErrorMessage`` returns the generic
    Wave1 banner. That is the live isolation-user bubble.
    """
    actor = _register(client, "spa")
    pid = _project(client, actor)
    err = _stream_error(client, actor, pid)
    msg = err.get("message") or ""
    assert msg == PROVIDER_FAILURE_MESSAGE, msg
    assert not _spa_would_show_generic(msg), (
        f"SPA friendlyErrorMessage has no includes() needle for {msg!r}; "
        f"ChatBubble will show {_GENERIC!r}. needles={_friendly_includes_needles()}"
    )
