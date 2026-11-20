"""BUILD 1: a calculation survives when no language model can be reached.

A turn that a registered formula can answer is answered by the deterministic
stages (calculator pre-dispatch, capability answer, fast paths) before any
model is asked. With NO provider configured those stages still run and the
user gets the computed figure, its working, and one plain line saying the
explanation is unavailable. A turn that needs the model gets the honest
"model unavailable" message, never a bare plumbing error.

Every test drives a whole chat turn through POST /v1/chat/stream and reads
only what the user sees. The provider HTTP boundary is replaced by a recorder
that fails the test if anything tries to reach it while no key is set.
"""
from __future__ import annotations

import json
import uuid

import httpx
import pytest

from tests._fixture_values import PW_06

_LLM_ENV = ("LLM_PROVIDER", "LLM_FALLBACK_PROVIDER", "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL", "OPENROUTER_ALLOW_PAID", "DEEPSEEK_API_KEY",
            "DEEPSEEK_MODEL")

#: The one line the user gets in place of the model's explanation.
_NOTE = "the language model could not be reached"


@pytest.fixture
def no_model(monkeypatch):
    """No provider key anywhere; any outbound provider call is recorded."""
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    calls: list[str] = []

    async def post(self, url, *a, **kw):
        calls.append(str(url))
        raise httpx.ConnectError("no network in this test")

    def stream(self, method, url, **kw):
        calls.append(str(url))
        raise httpx.ConnectError("no network in this test")

    monkeypatch.setattr(httpx.AsyncClient, "post", post, raising=True)
    monkeypatch.setattr(httpx.AsyncClient, "stream", stream, raising=True)
    return calls


def _ask(message: str) -> tuple[str, list[str]]:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        email = f"nomodel-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post("/v1/users/login",
                       json={"email": email, "password": PW_06}).json()["token"]
        r = c.post("/v1/chat/stream", headers={"Authorization": f"Bearer {token}"},
                   json={"message": message, "history": [],
                         "conversation_id": f"nomodel-{uuid.uuid4().hex[:8]}"})
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


# Two different registered formulas, each with every input given. The
# expected figure is computed here from the inputs, not copied from output.
FORMULA_ASKS = [
    pytest.param(
        "Calculate the excavation volume for a trench 12 m long, 2 m wide and 1.5 m deep.",
        f"{12 * 2 * 1.5:.2f}",
        id="excavation-volume",
    ),
    pytest.param(
        # The concrete calculator applies its stated 5% waste allowance.
        "Calculate the concrete volume for a slab 10 m long, 6 m wide and 0.25 m thick.",
        f"{10 * 6 * 0.25 * 1.05:g}",
        id="concrete-volume",
    ),
]


@pytest.mark.parametrize("message, figure", FORMULA_ASKS)
def test_a_registered_formula_answers_with_no_model(no_model, message, figure):
    shown, errors = _ask(message)
    assert not errors, (shown, errors)
    assert figure in shown, shown
    assert _NOTE in shown.lower(), shown
    # The figure comes first: the note is the last line, not the opening.
    assert shown.lower().index(_NOTE) > shown.index(figure), shown
    # The user is shown the formatted result, never the raw result record or
    # the instructions that were addressed to the model.
    assert '"status"' not in shown and "Answer from that result" not in shown, shown
    # Nothing tried to reach a provider: there was no key to send.
    assert no_model == [], no_model


@pytest.mark.parametrize("message", [
    "Explain in one sentence what a construction programme is.",
    "Why do contractors submit method statements before starting work?",
])
def test_a_question_that_needs_the_model_says_it_is_unavailable(no_model, message):
    from app.routers.chat_watchdog import PROVIDER_FAILURE_MESSAGE

    shown, errors = _ask(message)
    said = (shown + " " + " ".join(errors)).strip()
    assert PROVIDER_FAILURE_MESSAGE in said, said
    for leak in ("api_key", "deepseek", "openrouter", "configured"):
        assert leak not in said.lower(), said
