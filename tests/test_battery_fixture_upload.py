"""The battery seed's fixture upload must be accepted by the documents route.

The 9d28ad1 seeded battery failed with ``415 Unsupported Media Type`` on
``POST /v1/projects/{id}/documents`` for ``harbourline_boq.xlsx`` (the first
fixture, sorted). By source the route accepts ``.xlsx`` (it is in
``TEXT_BEARING_EXTS`` and ``compressed.is_compressed`` exempts ingestible
names), so this pins that the route accepts the seed's EXACT request shape --
multipart field ``file``, a real binary body, and the office/pdf content-type
the seed sends (``scripts/ui_battery_runner.py`` ``_SEED_CONTENT_TYPES``) -- for
BOTH a PDF and an XLSX. TestClient talks straight to the app with no proxy in
front, so if this passes the live 415 is environmental (a gateway/WAF), not the
application; if it fails it reproduces the defect with the body in the message.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}

# (filename, body, content-type) — the content types are exactly what the seed
# sends; the bodies carry a realistic head (a PDF marker, and the ZIP magic that
# every .xlsx starts with) so the compressed-folder guard is actually exercised.
_CASES = [
    ("plan.pdf", b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n", "application/pdf"),
    (
        "boq.xlsx",
        b"PK\x03\x04" + b"\x00" * 256,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("name,body,ctype", _CASES)
def test_documents_route_accepts_the_seed_fixture_shape(client, name, body, ctype):
    """A PDF and an XLSX, uploaded the way the battery seed uploads them, are
    accepted (201). A 415 here is the defect the live seed hit."""
    pid = client.post(
        "/v1/projects", json={"name": "battery fixture upload"}, headers=H,
    ).json()["id"]
    resp = client.post(
        f"/v1/projects/{pid}/documents",
        files={"file": (name, body, ctype)},
        headers=H,
    )
    assert resp.status_code == 201, (name, ctype, resp.status_code, resp.text[:300])
