"""Baseline browser hardening on every response, including errors.

Measured against live theshovel.ai on 2026-09-10: ``curl -I https://theshovel.ai/``
returned ``server: cloudflare`` and ``x-render-origin-server: uvicorn`` and
nothing else security-relevant. No HSTS, no CSP, no frame guard, no nosniff --
on a platform that serves a client's contract documents.

The CSP is REPORT-ONLY to start. The allowlist is not guesswork: it is what
the live SPA actually loads, recorded in a browser session on the same day --
the login bundle pulls its stylesheet from fonts.googleapis.com and its woff2
faces from fonts.gstatic.com. A ``default-src 'self'`` written without looking
would have blocked both the moment anyone flipped it to enforcing.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_baseline_headers_are_on_a_normal_response(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert "camera=()" in r.headers["Permissions-Policy"]


def test_error_responses_carry_them_too(client):
    """A 401 is the response an attacker sees most often."""
    r = client.get("/v1/projects")
    assert r.status_code == 401
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"


def test_hsts_only_in_production(client, monkeypatch):
    monkeypatch.setenv("ENV", "production")
    r = client.get("/livez")
    assert r.headers["Strict-Transport-Security"].startswith("max-age=31536000")
    assert "includeSubDomains" in r.headers["Strict-Transport-Security"]

    monkeypatch.setenv("ENV", "testing")
    r = client.get("/livez")
    assert "Strict-Transport-Security" not in r.headers, (
        "HSTS on a plain-HTTP dev origin pins the browser to https for a year"
    )


def test_csp_is_report_only_for_now(client):
    r = client.get("/livez")
    assert "Content-Security-Policy-Report-Only" in r.headers
    assert "Content-Security-Policy" not in r.headers, (
        "enforcing CSP is a separate, measured step — see the report-only note"
    )


def test_csp_allows_exactly_what_the_live_spa_loads(client):
    """Recorded in a browser against live theshovel.ai, not assumed.

    The login bundle issues:
      GET https://fonts.googleapis.com/css2?family=IBM+Plex+Sans...   (stylesheet)
      GET https://fonts.gstatic.com/s/ibmplexsans/...woff2            (font)
    A policy without those two hosts blocks the site's typography the moment
    it is enforced.
    """
    csp = client.get("/livez").headers["Content-Security-Policy-Report-Only"]
    assert "default-src 'self'" in csp
    assert "https://fonts.googleapis.com" in csp
    assert "https://fonts.gstatic.com" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'self'" in csp


def test_streaming_still_streams_through_the_middleware(client):
    """The risk this middleware brings, tested rather than assumed.

    ``BaseHTTPMiddleware`` wraps the response in an anyio stream, and the
    documented failure mode is a StreamingResponse that stops arriving
    incrementally. This app's whole answer surface is Server-Sent Events
    (``POST /v1/chat/stream``), so adding a header middleware without
    checking that is how you ship headers and lose the product.
    """
    import uuid

    from app.core import jwt_auth, users as users_store

    user = users_store.create_user(
        f"sse-{uuid.uuid4().hex[:8]}@example.test",
        "Correct-Horse-Battery-9!",
        email_verified=True,
    )
    headers = {"Authorization": f"Bearer {jwt_auth.create_token(user['id'])}"}

    r = client.post("/v1/chat/stream", headers=headers, json={"message": "hello"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream"), (
        "the middleware changed the streaming content type"
    )
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    # The frames still arrive: a `start` event is emitted before any model work.
    assert "data:" in r.text and '"type": "start"' in r.text
