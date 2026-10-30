"""The upload contract a browser actually experiences.

THE DEFECT THIS PINS
--------------------
The chat composer reported "Upload error: Failed to fetch" — the browser's
message for a request that never produced a readable HTTP response. Three
things made that reachable, none of them covered by any test:

1. Two upload routes carried DIFFERENT size caps (10 MB vs 50 MB) and
   DIFFERENT extension allowlists, so "what the product accepts" depended on
   which route the caller hit.
2. An oversize upload was accepted in full — the entire body spooled to disk —
   and only then answered 413. For a 345 MB corpus PDF that is minutes of
   transfer, and any proxy timeout in between becomes a connection reset, which
   the browser reports as "Failed to fetch".
3. CORSMiddleware was registered FIRST, which in Starlette makes it INNERMOST.
   Every response produced by an outer middleware (rate-limit 429s, oversize
   413s) therefore reached the browser with no CORS headers — unreadable, and
   indistinguishable from the server being down.

The old tests called ``upload_v1()`` as a plain Python function with the store
monkeypatched, so none of this was observable: no HTTP, no middleware, no
headers, no disk.
"""
from __future__ import annotations

import io
import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest
from fastapi.testclient import TestClient

from app.core import file_crypto, upload_limits
from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── 1. the two routes must agree ────────────────────────────────────────────

def test_both_upload_routes_share_one_allowlist():
    """A file accepted by one route and rejected by the other is, from the
    browser, indistinguishable from the product being broken. Two contracts:
    a file that becomes a PROJECT DOCUMENT (either route) must be a text
    format -- one declaration, ingest_status.TEXT_BEARING_EXTS (the /upload
    route's project refusal is pinned in test_ingest_text_only_rule); a
    session upload without a project keeps the sandbox allow-list."""
    from app.core.ingest_status import TEXT_BEARING_EXTS
    from app.routers import projects as projects_router
    from app.routers import upload as upload_router

    assert set(projects_router.ALLOWED_DOC_EXTENSIONS) == set(TEXT_BEARING_EXTS)
    assert set(upload_router.ALLOWED_UPLOAD_EXTENSIONS) == set(upload_limits.ALLOWED_UPLOAD_EXTENSIONS)


def test_limits_are_read_per_request_not_bound_at_import(monkeypatch):
    """The caps used to be module-level ints bound at import, so a
    deployment-time env change silently did nothing."""
    monkeypatch.setenv("MAX_DOC_UPLOAD_SIZE", "12345")
    assert upload_limits.max_document_bytes() == 12345


def test_a_garbled_limit_falls_back_instead_of_becoming_unlimited(monkeypatch):
    """A typo'd env var must not mean 'no limit' (OOM) or '0' (every upload
    rejected) — both are outages."""
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "50mb")
    assert upload_limits.max_upload_bytes() > 0
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "0")
    assert upload_limits.max_upload_bytes() > 0


# ── 2. oversize is refused early, and readably ──────────────────────────────

def test_oversize_body_is_refused_from_content_length_before_the_body_is_read(
    client, monkeypatch
):
    """The guard must fire on the declared length, not after spooling the file."""
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1024")
    monkeypatch.setenv("MAX_DOC_UPLOAD_SIZE", "1024")

    body = b"x" * (3 * 1024 * 1024)
    r = client.post(
        "/upload",
        files={"file": ("huge.pdf", body, "application/pdf")},
        headers=H,
    )
    assert r.status_code == 413, r.text


def test_the_413_is_readable_cross_origin(client, monkeypatch):
    """A rejection the browser cannot read IS 'Failed to fetch'.

    This is the assertion that would have caught the CORS ordering defect:
    the error response must carry Access-Control-Allow-Origin.
    """
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "1024")
    monkeypatch.setenv("MAX_DOC_UPLOAD_SIZE", "1024")

    origin = "http://localhost:5173"
    r = client.post(
        "/upload",
        files={"file": ("huge.pdf", b"x" * (3 * 1024 * 1024), "application/pdf")},
        headers={**H, "Origin": origin},
    )
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == origin, (
        "the 413 carries no CORS header, so a cross-origin caller sees only "
        "an opaque 'Failed to fetch' instead of the real reason"
    )


def test_cors_is_the_outermost_middleware():
    """Ordering-sensitive and easy to undo by moving one call: Starlette wraps
    the LAST-added middleware outermost, so CORS must be registered last for
    short-circuited responses to carry its headers."""
    names = [m.cls.__name__ for m in app.user_middleware]
    assert names[0] == "CORSMiddleware", (
        f"CORS must be outermost so every response carries CORS headers; "
        f"stack (outermost first) is {names}"
    )


def test_a_deployed_frontend_origin_can_be_allowed_by_env(monkeypatch):
    """The allowlist is localhost-only; a real deployment is served entirely by
    CORS_EXTRA_ORIGINS. If that is unset in prod, every browser call fails as
    'Failed to fetch' with no status — exactly the reported symptom."""
    monkeypatch.setenv("CORS_EXTRA_ORIGINS", "https://example-frontend.test")
    import importlib

    import app.main as main_module

    reloaded = importlib.reload(main_module)
    try:
        assert "https://example-frontend.test" in reloaded.CORS_ALLOWED_ORIGINS
    finally:
        monkeypatch.delenv("CORS_EXTRA_ORIGINS", raising=False)
        importlib.reload(main_module)


# ── 3. streaming, so a large file cannot OOM the shared worker ──────────────

def test_stream_write_returns_the_plaintext_size(tmp_path):
    body = b"a" * 100_000
    dest = tmp_path / "streamed.bin"
    written = file_crypto.write_document_stream(dest.as_posix(), io.BytesIO(body))
    assert written == len(body)
    assert file_crypto.read_document(dest.as_posix()) == body


def test_stream_write_aborts_over_budget_and_leaves_no_partial_file(tmp_path):
    """A rejected upload must not leave a truncated file behind — on disk it is
    indistinguishable from a complete one."""
    dest = tmp_path / "toobig.bin"
    with pytest.raises(file_crypto.UploadTooLarge):
        file_crypto.write_document_stream(
            dest.as_posix(), io.BytesIO(b"b" * 10_000), max_bytes=1_000,
        )
    assert not dest.exists(), "partial file survived a rejected upload"


def test_stream_write_round_trips_when_encrypted(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    body = b"c" * 50_000
    dest = tmp_path / "enc.bin"
    written = file_crypto.write_document_stream(dest.as_posix(), io.BytesIO(body))
    assert written == len(body)  # plaintext length, not ciphertext
    assert file_crypto.read_document(dest.as_posix()) == body


def test_a_project_document_is_accepted_by_both_routes_alike(client):
    """Replaces the link #773 dropped from the allow-list contract: /upload
    WITH a project accepts exactly what POST /v1/projects/{id}/documents
    accepts. It used to check the sandbox list first, so text formats outside
    it (an archive, a mail, HTML) were refused there and accepted here."""
    from app.core.ingest_status import TEXT_BEARING_EXTS

    pid = client.post("/v1/projects", json={"name": "Both routes"}, headers=H).json()["id"]
    # S-10d: the sandbox list now contains every project format, so compare
    # the WHOLE project rule across both routes (was: only the difference).
    project_rule = sorted(TEXT_BEARING_EXTS)
    assert project_rule
    for ext in project_rule:
        name = f"doc{ext}"
        via_project = client.post(
            f"/v1/projects/{pid}/documents", files={"file": (name, b"text body", "text/plain")}, headers=H,
        )
        via_upload = client.post(
            "/upload", files={"file": (name, b"text body", "text/plain")},
            data={"project_id": pid}, headers=H,
        )
        assert via_project.status_code == 201, (ext, via_project.text)
        assert via_upload.status_code == 200, (ext, via_upload.text)
