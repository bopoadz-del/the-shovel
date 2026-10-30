"""A schedule export whose activities link by predecessors but lack ``code``
or ``duration_days`` is a client fault: 400 naming the missing fields, not 500.
The same list with those fields exports."""
from __future__ import annotations

import uuid

import pytest

import app.routers.exports as exports

from tests._fixture_values import PW_06


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        email = f"s10c-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        tok = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
        h = {"Authorization": f"Bearer {tok}"}
        pid = c.post("/v1/projects", json={"name": "s10c"}, headers=h).json()["id"]
        yield c, h, pid


MISSING = [{"name": "Dig", "predecessors": []}, {"name": "Pour", "predecessors": ["A1"]}]
COMPLETE = [
    {"code": "A1", "name": "Dig", "duration_days": 3, "predecessors": []},
    {"code": "A2", "name": "Pour", "duration_days": 2, "predecessors": ["A1"]},
]


@pytest.mark.parametrize("fmt", ["xlsx", "docx", "pdf"])
def test_missing_fields_is_400_naming_them(client, fmt):
    c, h, pid = client
    r = c.post(f"/v1/projects/{pid}/export/schedule?format={fmt}", headers=h,
               json={"activities": MISSING})
    assert r.status_code == 400, r.text
    assert "code" in r.text and "duration_days" in r.text


# PDF needs reportlab, which the CI image does not ship (it answers 503);
# the 200 path is pinned on the two formats every image renders.
@pytest.mark.parametrize("fmt", ["xlsx", "docx"])
def test_complete_activities_export(client, fmt):
    c, h, pid = client
    r = c.post(f"/v1/projects/{pid}/export/schedule?format={fmt}", headers=h,
               json={"activities": COMPLETE})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("acts,want", [
    ([{"name": "Dig", "predecessors": []}, {"name": "Pour", "predecessors": ["A1"]}],
     ["activity 1 (Dig): missing code, duration_days", "activity 2 (Pour): missing code, duration_days"]),
    ([{"id": "A1", "duration": 3, "predecessors": []}, {"code": "A2", "predecessors": ["A1"]}],
     ["activity 2 (unnamed): missing duration_days"]),
])
def test_link_gaps_are_named_per_activity(acts, want):
    assert exports._schedule_link_fields_missing(acts) == want


@pytest.mark.parametrize("acts", [
    [{"name": "Dig"}, {"name": "Pour"}],
    [{"name": "Dig", "predecessors": []}, {"name": "Pour", "predecessors": None}],
])
def test_nothing_linked_needs_nothing(acts):
    assert exports._schedule_link_fields_missing(acts) == []

