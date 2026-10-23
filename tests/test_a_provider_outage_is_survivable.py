"""What a USER sees when the LLM provider fails.

tests/test_llm_provider_fallback.py proves `_call_llm` walks its ladder. This
file drives a whole chat turn through POST /v1/chat/stream -- router, agent,
post-processing, SSE -- with DeepSeek (primary) and OpenRouter (fallback)
faked at the HTTP boundary, and asserts on the only thing that matters to the
person typing: the text on their screen.

For every failure of the primary:
  * the fallback is actually called, with ITS key and never the primary's;
  * the user gets the fallback's answer;
  * nothing about the plumbing reaches the screen -- no provider name, HTTP
    status, upstream body, host, exception text or key fragment.
When BOTH fail, the user gets one plain message and still none of the above.

Live context, 2026-09-19: LLM_FALLBACK_PROVIDER on Render said `kimi`, a
provider the code had removed. It resolved back to DeepSeek, so there was no
fallback at all and nothing said so.
"""
from __future__ import annotations

import json
import uuid

import httpx
import pytest

from app.agents.runtime import DEEPSEEK_API_URL, OPENROUTER_API_URL
from tests._fixture_values import PW_06, STUB_TOKEN_12, STUB_TOKEN_14

DS_KEY = STUB_TOKEN_12
OR_KEY = STUB_TOKEN_14
PRIMARY = "PRIMARY says: a programme sequences the works in time."
FALLBACK = "FALLBACK says: a programme sequences the works in time."
UPSTREAM = '{"error":{"message":"upstream exploded on node gpu-7 req=zz9","type":"server_error"}}'

# Nothing about the plumbing may reach the screen.
_PLUMBING = ("deepseek", "openrouter", "http 4", "http 5", "gpu-7", "zz9",
             "testkey", "api.deepseek", "connection refused", "errno",
             "server_error", "nginx", "readtimeout", "traceback")

_LLM_ENV = ("LLM_PROVIDER", "LLM_FALLBACK_PROVIDER", "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL", "OPENROUTER_ALLOW_PAID", "OPENROUTER_MAX_TOKENS",
            "OPENROUTER_PROMPT_TOKEN_CEILING", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
            "USAGE_DAILY_CAP_USD")


def _completion(text):
    return {"id": "x", "model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.headers = {}
        self.request = httpx.Request("POST", "https://llm.invalid")

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("upstream", request=self.request, response=self)


def _behave(kind, text):
    if kind == "ok":
        return _Resp(200, _completion(text))
    if kind == "refused":
        raise httpx.ConnectError("[Errno 111] Connection refused to api.deepseek.com")
    if kind == "timeout":
        raise httpx.ReadTimeout("ReadTimeout")
    if kind == "402":
        # Live DeepSeek body on tip 78bd9ca — not the OpenRouter in-flight
        # wording. Must still hop to the fallback.
        return _Resp(
            402,
            '{"error":{"message":"Insufficient Balance","type":"unknown_error","param":null,"code":"invalid_request_error"}}',
        )
    if kind in ("500", "503", "429"):
        return _Resp(int(kind), UPSTREAM)
    if kind == "empty":
        return _Resp(200, _completion(""))
    if kind == "garbage":
        return _Resp(200, "<html>502 Bad Gateway nginx</html>")
    raise AssertionError(kind)


class _Providers:
    """Fakes both providers by URL and records who was called with which key."""

    def __init__(self, primary, fallback):
        self.plan = {"ds": primary, "or": fallback}
        self.calls = []

    def _who(self, url):
        url = str(url)
        if url.startswith(DEEPSEEK_API_URL.rsplit("/", 2)[0]):
            return "ds"
        if url.startswith(OPENROUTER_API_URL.rsplit("/", 2)[0]):
            return "or"
        return "other"

    def respond(self, url, headers):
        who = self._who(url)
        self.calls.append((who, (headers or {}).get("Authorization", "")))
        if who == "other":
            return _Resp(200, _completion("x"))
        return _behave(self.plan[who], PRIMARY if who == "ds" else FALLBACK)


@pytest.fixture
def providers(monkeypatch):
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("DEEPSEEK_API_KEY", DS_KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_KEY)
    holder = {}

    def install(primary, fallback):
        fake = _Providers(primary, fallback)
        holder["fake"] = fake

        async def post(self, url, json=None, headers=None, **kw):
            return fake.respond(url, headers)

        class _Stream:
            def __init__(self, url, headers):
                self.url, self.headers = url, headers

            async def __aenter__(self):
                self.r = fake.respond(self.url, self.headers)
                self.status_code = self.r.status_code
                return self

            async def __aexit__(self, *a):
                return False

            async def aread(self):
                return self.r.text.encode()

            async def aiter_lines(self):
                if self.status_code == 200:
                    content = self.r.json()["choices"][0]["message"]["content"]
                    yield "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})
                yield "data: [DONE]"

        monkeypatch.setattr(httpx.AsyncClient, "post", post, raising=True)
        monkeypatch.setattr(httpx.AsyncClient, "stream",
                            lambda self, method, url, **kw: _Stream(url, kw.get("headers")),
                            raising=True)
        return fake

    return install


def _ask(message="Explain in one sentence what a construction programme is."):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        email = f"llm-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post("/v1/users/login",
                       json={"email": email, "password": PW_06}).json()["token"]
        r = c.post("/v1/chat/stream", headers={"Authorization": f"Bearer {token}"},
                   json={"message": message, "history": [],
                         "conversation_id": f"llm-{uuid.uuid4().hex[:8]}"})
    shown, errors = "", []
    for line in r.text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:])
        except ValueError:
            continue
        if ev.get("type") == "token":
            shown += ev.get("content") or ""
        elif ev.get("type") == "end" and not shown:
            shown = ev.get("content") or ""
        elif ev.get("type") == "error":
            errors.append(ev.get("message") or "")
    return shown, errors


def _keys_are_paired(fake):
    for who, auth in fake.calls:
        if who == "ds":
            assert OR_KEY not in auth, "OpenRouter's key was sent to DeepSeek"
        if who == "or":
            assert DS_KEY not in auth, "DeepSeek's key was sent to OpenRouter"


def _no_plumbing(text):
    low = text.lower()
    leaked = [w for w in _PLUMBING if w in low]
    assert not leaked, f"user saw plumbing {leaked}: {text[:300]!r}"


def test_a_healthy_primary_answers_and_the_fallback_is_not_called(providers):
    fake = providers("ok", "ok")
    shown, errors = _ask()
    assert "PRIMARY says" in shown, (shown, errors)
    assert not any(w == "or" for w, _ in fake.calls)


@pytest.mark.parametrize("failure", ["refused", "timeout", "500", "503", "429", "402", "empty", "garbage"])
def test_when_the_primary_fails_the_user_gets_the_fallbacks_answer(providers, failure):
    fake = providers(failure, "ok")
    shown, errors = _ask()
    assert any(w == "or" for w, _ in fake.calls), f"fallback never called: {fake.calls}"
    _keys_are_paired(fake)
    assert "FALLBACK says" in shown, (shown, errors)
    _no_plumbing(shown + " ".join(errors))


@pytest.mark.parametrize("primary,fallback", [("500", "500"), ("refused", "402"),
                                              ("timeout", "timeout"), ("503", "429")])
def test_when_both_fail_the_user_gets_one_plain_message(providers, primary, fallback):
    fake = providers(primary, fallback)
    shown, errors = _ask()
    _keys_are_paired(fake)
    text = (shown + " " + " ".join(errors)).strip()
    assert text, "the user was shown nothing at all"
    _no_plumbing(text)


# ── the net itself ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("meant_for_a_user", [
    "Agent 'self-coding' is not available.",
    "message exceeds 32000 characters",
    "The assistant is temporarily unavailable. Please try again.",
    "Response timeout — the turn produced no answer within 300s; the last event was 'start'.",
    "Hit 12-iteration cap.",
])
def test_a_message_written_for_the_user_passes_untouched(meant_for_a_user):
    from app.routers.chat_watchdog import user_safe_error

    assert user_safe_error(meant_for_a_user) == meant_for_a_user


@pytest.mark.parametrize("plumbing", [
    'openrouter HTTP 500: {"error":{"message":"boom"}}',
    "deepseek LLM call timed out (200s).",
    "chat_stream crashed: KeyError: 'choices'",
    "No OPENROUTER_API_KEY configured.",
    "deepseek LLM call failed: [Errno 111] Connection refused",
    "ReadTimeout: timed out",
])
def test_a_message_that_names_the_plumbing_is_replaced(plumbing):
    from app.routers.chat_watchdog import PROVIDER_FAILURE_MESSAGE, user_safe_error

    assert user_safe_error(plumbing) == PROVIDER_FAILURE_MESSAGE


def test_the_request_id_survives_the_rewrite():
    from app.routers.chat_watchdog import frame, sanitize_error_frame

    out = sanitize_error_frame(frame({"type": "error", "request_id": "rq-7",
                                      "message": "openrouter HTTP 503: {}"}))
    data = json.loads(out[5:])
    assert data["request_id"] == "rq-7"
    assert "openrouter" not in data["message"]
