"""F2 — protected routes reject unauthenticated callers, valid ones pass.

The brief's rule 3 is the one that matters here: a gate must DENY on missing
or unknown context, never fall through to allow. So these tests do not only
check that a good key works — they check the three ways a gate fails OPEN:

  * no credential at all
  * a syntactically valid but unknown credential
  * a valid credential without the required role

A gate that reads a missing context as "allow" is a vulnerability, so each
of those must be a rejection, and the rejection must be an auth status
(401/403) rather than a 500 that happens to keep the caller out.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY
from tests._fixture_values import STUB_KEY_03

VALID = {"Authorization": "Bearer " + DEV_KEY}
UNKNOWN = {"Authorization": "Bearer " + STUB_KEY_03}
MALFORMED = {"Authorization": DEV_KEY}          # no scheme

# Routes that must never serve an anonymous caller. Kept small and concrete:
# one read, one write, one admin-scoped.
PROTECTED_GET = ["/v1/memory/stats", "/v1/leaderboard"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", PROTECTED_GET)
def test_no_credential_is_rejected(client, path):
    assert client.get(path).status_code in (401, 403), \
        f"{path} served an anonymous caller"


@pytest.mark.parametrize("path", PROTECTED_GET)
def test_an_unknown_key_is_rejected(client, path):
    """Fails CLOSED on an unrecognised credential — the check must be
    membership, not merely presence."""
    assert client.get(path, headers=UNKNOWN).status_code in (401, 403), \
        f"{path} accepted an unknown key"


@pytest.mark.parametrize("path", PROTECTED_GET)
def test_a_malformed_header_is_rejected_not_crashed(client, path):
    """A header with no Bearer scheme must be an auth rejection, not a 500.
    A 500 keeps the caller out by accident, and accidents change."""
    status = client.get(path, headers=MALFORMED).status_code
    assert status in (401, 403), f"{path} -> {status} on a malformed header"


@pytest.mark.parametrize("path", PROTECTED_GET)
def test_a_valid_key_passes_the_gate(client, path):
    """Isolation must not be achieved by rejecting everyone."""
    assert client.get(path, headers=VALID).status_code not in (401, 403), \
        f"{path} rejected a valid key"


def test_an_authenticated_non_admin_is_still_denied_admin_actions(client):
    """`the dev key` authenticates but carries no admin role.

    This is the specific hole the api-key normalisation had to avoid:
    defaulting a missing `role` from the system user (which IS admin) would
    have silently promoted every standard key to admin. Authentication and
    authorisation stay separate.
    """
    assert client.post("/v1/memory/flush", json={}, headers=VALID).status_code == 403
    assert client.post("/v1/memory/keys", json={}, headers=VALID).status_code == 403
    # ...while an ordinary action with the same key still passes.
    assert client.post(
        "/v1/memory/get", json={"key": "x"}, headers=VALID
    ).status_code not in (401, 403)
    # Privileged code-execution blocks must stay 403 for a non-admin key.
    assert client.post(
        "/v1/execute", json={"block": "code", "input": "x"}, headers=VALID
    ).status_code == 403
    assert client.post(
        "/v1/chain",
        json={"steps": [{"block": "code", "params": {}}]},
        headers=VALID,
    ).status_code == 403
