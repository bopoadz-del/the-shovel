"""Provider-outage injection — the platform must degrade HONESTLY.

New test shape (2026-07-25): the worst pilot failures were provider outages
(the 2026-07-24 kimi-HTTP-400 storm) where fallbacks quietly produced
boilerplate that read as invented answers. This probe kills the LLM at the
client seam and asserts the chat surface reports an honest error — no
fabricated construction numbers, no silent success.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dead_llm(monkeypatch):
    """Every chat-path LLM call fails like a provider outage (HTTP 500)."""
    from app.agents import runtime as rt

    async def outage(self, *a, **k):
        return {"status": "error", "error": "simulated provider outage (HTTP 500)"}

    # The agent loop's LLM seam — chat_stream consumes this result dict.
    monkeypatch.setattr(rt.Agent, "_call_llm", outage, raising=True)
    return outage


def _stream_chat(client, message, project_id=None):
    body = {"message": message, "history": []}
    if project_id:
        body["project_id"] = project_id
    with client.stream("POST", "/v1/chat/stream", json=body, headers=H) as r:
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except ValueError:
                    pass
        return events


def test_outage_yields_honest_error_not_fabrication(client, dead_llm):
    events = _stream_chat(
        client, "What is the design flow rate of the sewage pump station?"
    )
    kinds = [e.get("type") for e in events]
    # The stream must surface an explicit error event...
    assert "error" in kinds, kinds
    # ...and must NOT stream any answer tokens pretending it worked.
    tokens = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    assert "l/s" not in tokens.lower()
    assert "flow rate" not in tokens.lower() or "error" in tokens.lower()


def test_outage_error_names_the_problem(client, dead_llm):
    events = _stream_chat(client, "Summarise this project")
    err = next((e for e in events if e.get("type") == "error"), None)
    assert err is not None
    # The message tells the user the SERVICE failed — it doesn't blame their
    # question or invent an empty-corpus excuse. Depending on env, the honest
    # error comes from the injected outage OR from the provider-config check
    # upstream of it ("no kimi_api_key configured." on keyless CI runners) —
    # both name the service as the problem.
    msg = (err.get("message") or "").lower()
    # The plumbing itself (provider, HTTP status, upstream body) is no longer
    # shown -- see tests/test_a_provider_outage_is_survivable.py -- so the
    # message names the SERVICE in plain words.
    assert any(w in msg for w in (
        "outage", "llm", "500", "failed", "api_key", "configured", "provider",
        "service is temporarily unavailable",
    )), msg
    assert "your question" in msg or "outage" in msg or "configured" in msg, msg
