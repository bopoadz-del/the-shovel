"""A page must be able to tell that it runs a different build than its server.

17 Nov: a Word press under an answer reached the live server without the
scope the current bundle sends, which is what an older bundle sent. A tab
keeps the script it loaded until it is reloaded, and nothing on the page knew
its build, so nothing told the user.

The rule, server half (the client half is frontend/src/lib/buildCheck.test.mjs):
- every response names the server's build in ``X-Build-Id`` (never invented:
  an unset build sends no header);
- a cross-origin dev page may read that header;
- every HTML page shell is revalidated, whichever route or mount served it;
- the image stamps the bundle with the same commit the server reports.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from app.main import SecurityHeadersMiddleware, app
from app.routers.health import BUILD_ID_HEADER

_REPO = Path(__file__).resolve().parents[1]
_SERVER_BUILD = "synthetic-build-for-this-test"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _pin(monkeypatch, value: str | None) -> None:
    monkeypatch.delenv("SOURCE_VERSION", raising=False)
    if value is None:
        monkeypatch.delenv("GIT_SHA", raising=False)
    else:
        monkeypatch.setenv("GIT_SHA", value)


@pytest.mark.parametrize("server_build", [_SERVER_BUILD, "c" * 40])
def test_every_response_names_the_server_build(client, monkeypatch, server_build):
    _pin(monkeypatch, server_build)
    live = client.get("/livez")
    assert live.headers[BUILD_ID_HEADER] == server_build
    # The header and the liveness body name the same build.
    assert live.json()["build_sha"] == server_build
    # An error response from an API route carries it too: the press that
    # exposed the defect was a POST to an API route, not a page load.
    denied = client.post("/v1/projects/p/conversations/c/export?format=docx")
    assert denied.status_code in (401, 403)
    assert denied.headers[BUILD_ID_HEADER] == server_build


def test_an_unknown_server_build_is_not_invented(client, monkeypatch):
    _pin(monkeypatch, None)
    r = client.get("/livez")
    assert BUILD_ID_HEADER not in r.headers
    assert r.json()["build_sha"] is None


def test_a_cross_origin_dev_page_may_read_the_build(client, monkeypatch):
    _pin(monkeypatch, _SERVER_BUILD)
    r = client.get("/livez", headers={"Origin": "http://localhost:5173"})
    exposed = r.headers.get("access-control-expose-headers", "")
    assert BUILD_ID_HEADER.lower() in exposed.lower()


def _shell_app(media_type: str = "text/html") -> FastAPI:
    """A shell served with no cache header of its own, like a static mount."""
    shell = FastAPI()
    shell.add_middleware(SecurityHeadersMiddleware)

    @shell.get("/plain-shell")
    def plain_shell():
        return HTMLResponse("<!doctype html><title>t</title>", media_type=media_type)

    @shell.get("/chosen-shell")
    def chosen_shell():
        return HTMLResponse(
            "<!doctype html><title>t</title>",
            headers={"Cache-Control": "no-store"},
        )

    @shell.get("/data")
    def data():
        return {"ok": True}

    return shell


@pytest.mark.parametrize("media_type", ["text/html", "TEXT/HTML"])
def test_every_html_shell_is_revalidated(media_type):
    with TestClient(_shell_app(media_type)) as c:
        assert c.get("/plain-shell").headers["Cache-Control"] == "no-cache"
        # A route that already chose a stricter policy keeps it.
        assert c.get("/chosen-shell").headers["Cache-Control"] == "no-store"
        # JSON is not a page shell; its caching is the route's business.
        assert "Cache-Control" not in c.get("/data").headers


def _frontend_stage(dockerfile: str) -> str:
    m = re.search(r"^FROM [^\n]* AS frontend\n(.*?)(?=^FROM )", dockerfile, re.S | re.M)
    assert m, "Dockerfile has no frontend stage"
    return m.group(1)


def test_the_bundle_is_stamped_with_the_commit_the_server_reports():
    stage = _frontend_stage((_REPO / "Dockerfile").read_text(encoding="utf-8"))
    lines = [ln.strip() for ln in stage.splitlines()]
    assert "ARG GIT_SHA" in lines
    stamp = lines.index("ENV VITE_BUILD_ID=$GIT_SHA")
    build = lines.index("RUN npm run build")
    assert lines.index("ARG GIT_SHA") < stamp < build
    # The client reads exactly that variable.
    banner = (_REPO / "frontend/src/lib/buildWatch.ts").read_text(encoding="utf-8")
    assert "import.meta.env.VITE_BUILD_ID" in banner
