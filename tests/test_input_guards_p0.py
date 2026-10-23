"""A client fault must be a 4xx, never a 200 carrying an error.

Measured on 8535199, authenticated, one probe per mutating route:

  POST /v1/chat/stream {}              -> 200 text/event-stream, then
                                         {"type":"error","message":"The
                                         assistant is temporarily
                                         unavailable. Please try again."}
  POST /v1/chat/stream <200 KB msg>    -> 200, same
  POST /v1/schedule/generate {}        -> 200 {"status":"success",
                                         "project_type":"building",
                                         "actual_count":204, ...}
  POST /v1/metrics/record {}           -> 200 {"error":"Unknown provider"}
  POST /v1/document-types/classify {}  -> 200

Each is the same defect wearing different clothes: the caller sent
something the route cannot act on, and the route answered as though it
had. The schedule one is the worst of them -- an empty brief produces a
204-activity generic "building" programme, which is a fabrication with a
success code on it.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import jwt_auth
from app.core import users as users_store
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def h():
    user = users_store.create_user(
        f"guard-{uuid.uuid4().hex[:8]}@example.test",
        "Correct-Horse-Battery-9!",
        email_verified=True,
    )
    return {"Authorization": f"Bearer {jwt_auth.create_token(user['id'])}"}


# ── /v1/chat/stream ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body", [{}, {"message": ""}, {"message": "   "}, {"message": 42}]
)
def test_chat_stream_rejects_an_empty_or_non_string_message(client, h, body):
    r = client.post("/v1/chat/stream", headers=h, json=body)
    assert r.status_code == 422, r.text


def test_chat_stream_rejects_a_body_that_is_not_json(client, h):
    r = client.post(
        "/v1/chat/stream",
        headers={**h, "Content-Type": "application/json"},
        content=b"\x00\xff not json",
    )
    assert r.status_code == 422, r.text


def test_chat_stream_rejects_a_json_body_that_is_not_an_object(client, h):
    r = client.post("/v1/chat/stream", headers=h, json="just a string")
    assert r.status_code == 422, r.text


def test_chat_stream_rejects_an_oversize_message(client, h):
    r = client.post("/v1/chat/stream", headers=h, json={"message": "A" * 200_000})
    assert r.status_code == 413, r.text


def test_chat_stream_still_opens_for_a_real_message(client, h):
    """The guard must not cost the product its answer surface."""
    r = client.post("/v1/chat/stream", headers=h, json={"message": "hello"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")


def test_the_prompt_alias_still_works(client, h):
    """``prompt`` is the older key; both reach the same field."""
    r = client.post("/v1/chat/stream", headers=h, json={"prompt": "hello"})
    assert r.status_code == 200


# ── /v1/schedule/generate ─────────────────────────────────────────────────


@pytest.mark.parametrize("body", [{}, {"brief": ""}, {"brief": "  "}])
def test_schedule_generate_refuses_an_empty_brief(client, h, body):
    """An empty ask must not produce a 204-activity generic programme."""
    r = client.post("/v1/schedule/generate", headers=h, json=body)
    assert r.status_code == 422, r.text


def test_schedule_generate_still_works_with_a_real_brief(client, h):
    r = client.post(
        "/v1/schedule/generate",
        headers=h,
        json={"brief": "small office building", "target_count": 30},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "success"


# ── /v1/metrics/record ────────────────────────────────────────────────────


def test_metrics_record_requires_a_provider(client, h):
    r = client.post("/v1/metrics/record", headers=h, json={})
    assert r.status_code == 422, r.text


def test_metrics_record_unknown_provider_is_a_client_error(client, h):
    """The block answers "Unknown provider" as data; that is a 4xx."""
    r = client.post(
        "/v1/metrics/record",
        headers=h,
        json={"provider": "no-such-provider", "latency_ms": 1},
    )
    assert r.status_code == 422, r.text
    assert "provider" in r.text.lower()


def test_metrics_record_accepts_a_known_provider(client, h):
    r = client.post(
        "/v1/metrics/record",
        headers=h,
        json={"provider": "deepseek", "latency_ms": 12.5, "success": True},
    )
    assert r.status_code == 200, r.text


# ── /v1/document-types/classify ───────────────────────────────────────────


def test_classify_requires_a_filename_or_a_sample(client, h):
    r = client.post("/v1/document-types/classify", headers=h, json={})
    assert r.status_code == 422, r.text


def test_classify_still_works_from_a_filename_alone(client, h):
    r = client.post(
        "/v1/document-types/classify",
        headers=h,
        json={"filename": "Piling Method Statement.docx"},
    )
    assert r.status_code == 200, r.text


# ── the error envelope itself ─────────────────────────────────────────────


def test_a_custom_validator_error_is_a_422_not_a_500(client, h):
    """The handler that turns validation errors into envelopes used to crash.

    ``_validation_exception_handler`` passed ``exc.errors()`` straight to
    JSONResponse. A pydantic ``model_validator``/``field_validator`` raising
    ValueError puts the exception OBJECT into each error's ``ctx``, which is
    not JSON-serialisable — so the handler raised and the caller got a 500
    for sending a bad body.

    It stayed invisible because field-level constraints (``min_length`` and
    friends) carry no ``ctx``: every validation error the app happened to
    produce was of the one shape that serialises. The first model-level
    validator in a router surfaced it.
    """
    r = client.post("/v1/document-types/classify", headers=h, json={})
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    # The detail survives serialisation rather than exploding on the way out.
    assert "filename or content_sample is required" in r.text
