"""Untyped `request: dict` bodies were replaced with typed Pydantic models;
a malformed body is now rejected with 422 instead of being splatted into a
block's execute() call."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_memory_op_rejects_wrong_typed_field(client):
    """ttl must be an int — a non-numeric ttl is a 422, not a silent splat."""
    r = client.post("/v1/memory/set", headers=H,
                     json={"key": "k", "value": "v", "ttl": "not-a-number"})
    assert r.status_code == 422, r.text


def test_memory_op_accepts_valid_body(client):
    """A well-formed memory op still works."""
    r = client.post("/v1/memory/set", headers=H,
                     json={"key": "k", "value": "v"})
    assert r.status_code == 200, r.text


def test_record_metrics_rejects_wrong_typed_field(client):
    """latency_ms must be numeric."""
    r = client.post("/v1/metrics/record", headers=H,
                    json={"provider": "deepseek", "latency_ms": "abc"})
    assert r.status_code == 422, r.text


def test_record_metrics_accepts_valid_body(client):
    r = client.post("/v1/metrics/record", headers=H,
                    json={"provider": "deepseek", "latency_ms": 12.5, "success": True})
    assert r.status_code == 200, r.text


def test_auth_validate_rejects_non_object_body(client):
    """A JSON string where an object is expected is a 422."""
    r = client.post("/v1/auth/validate", headers=H, json="just-a-string")
    assert r.status_code == 422, r.text


def test_auth_validate_missing_key_is_422(client):
    """An object with neither api_key nor key is rejected."""
    r = client.post("/v1/auth/validate", headers=H, json={})
    assert r.status_code == 422, r.text
