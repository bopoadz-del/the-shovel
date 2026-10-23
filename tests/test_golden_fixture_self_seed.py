"""Regression: missing named fixtures self-seed; display names are never ids.

The golden-set runner used to fall back to the unresolved display name
(``FIXTURE — BOQ``) as ``project_id``, which 404s. Missing non-master_corpus
fixtures must call the seed path; seed failure is fatal.
"""
from __future__ import annotations

import pytest

from scripts.golden_set_gate import (
    FixtureUnresolvedError,
    collect_named_fixtures,
    load_golden_set,
    preflight_named_fixtures,
    resolve_golden_project,
)
from scripts.seed_fixtures import (
    BOQ_FIXTURE_NAME,
    FIXTURES,
    FixtureSeedError,
    _generate_synthetic,
    ensure_named_fixture,
    fixture_key_for_name,
    seed_one,
)


# ── seed_fixtures: FIXTURE — BOQ + --synthetic ───────────────────────────────

def test_boq_fixture_name_is_exact():
    assert BOQ_FIXTURE_NAME == "FIXTURE — BOQ"
    assert FIXTURES["boq"]["name"] == BOQ_FIXTURE_NAME
    assert fixture_key_for_name(BOQ_FIXTURE_NAME) == "boq"
    assert fixture_key_for_name("master_corpus") is None


def test_synthetic_generates_boq_workbook(tmp_path):
    spec = FIXTURES["boq"]
    generated = _generate_synthetic(tmp_path, spec["required_any_files"])
    assert "synthetic_boq.xlsx" in generated
    assert (tmp_path / "synthetic_boq.xlsx").is_file()


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSeedClient:
    """Minimal /v1/projects + upload surface for seed_one."""

    def __init__(self):
        self.projects: list = []
        self.docs: dict = {}
        self.uploads: list = []

    def get(self, url, params=None):
        if url == "/v1/projects":
            return _FakeResponse({"projects": list(self.projects)})
        if url.startswith("/v1/projects/") and url.endswith("/documents"):
            pid = url.split("/")[3]
            return _FakeResponse({"documents": self.docs.get(pid, [])})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, json=None, files=None, params=None):
        if url == "/v1/projects":
            pid = "proj-boq-1"
            proj = {"id": pid, "name": json["name"]}
            self.projects.append(proj)
            self.docs[pid] = []
            return _FakeResponse(proj)
        if "/documents" in url:
            pid = url.split("/")[3]
            fname = files["file"][0]
            doc = {
                "id": f"doc-{len(self.uploads)}",
                "original_name": fname,
                "chunk_count": 3,
            }
            self.docs.setdefault(pid, []).append(doc)
            self.uploads.append(fname)
            return _FakeResponse({"document": doc})
        raise AssertionError(f"unexpected POST {url}")


def test_seed_one_synthetic_creates_fixture_boq(tmp_path):
    client = _FakeSeedClient()
    result = seed_one(client, "boq", tmp_path, synthetic=True)
    assert result["name"] == BOQ_FIXTURE_NAME
    assert result["project_id"] == "proj-boq-1"
    assert result["project_id"] != BOQ_FIXTURE_NAME
    assert "synthetic_boq.xlsx" in client.uploads
    assert client.projects[0]["name"] == BOQ_FIXTURE_NAME


def test_seed_one_blocked_without_synthetic_fails_loud(tmp_path):
    with pytest.raises(FixtureSeedError, match="blocked"):
        seed_one(_FakeSeedClient(), "boq", tmp_path, synthetic=False)


def test_ensure_named_fixture_rejects_master_corpus():
    with pytest.raises(FixtureSeedError, match="not a self-seedable"):
        ensure_named_fixture(
            "master_corpus",
            base="http://example.invalid",
            headers={},
            client=_FakeSeedClient(),
        )


# ── golden_set_gate: missing name triggers seed, never used as project_id ────

def test_collect_named_fixtures_skips_master_corpus():
    names = collect_named_fixtures([
        {"project": "master_corpus"},
        {"project": BOQ_FIXTURE_NAME},
        {"project": "FIXTURE — Fresh Upload Eval"},
        {"project": BOQ_FIXTURE_NAME},
        {"project": ""},
    ])
    assert names == [BOQ_FIXTURE_NAME, "FIXTURE — Fresh Upload Eval"]


def test_missing_named_fixture_triggers_seed():
    seeded: list = []

    def fake_lookup(client, value, **_kw):
        return None

    def fake_seeder(name, **_kw):
        seeded.append(name)
        return {"name": name, "project_id": "seeded-boq-id"}

    pid = resolve_golden_project(
        None, BOQ_FIXTURE_NAME,
        base="http://example.invalid",
        headers={},
        cache={},
        lookup=fake_lookup,
        seeder=fake_seeder,
    )
    assert seeded == [BOQ_FIXTURE_NAME]
    assert pid == "seeded-boq-id"
    assert pid != BOQ_FIXTURE_NAME


def test_existing_named_fixture_does_not_seed():
    seeded: list = []

    def fake_lookup(client, value, **_kw):
        return "already-there"

    def fake_seeder(name, **_kw):
        seeded.append(name)
        raise AssertionError("seed must not run when the fixture exists")

    pid = resolve_golden_project(
        None, BOQ_FIXTURE_NAME,
        base="http://example.invalid",
        headers={},
        cache={},
        lookup=fake_lookup,
        seeder=fake_seeder,
    )
    assert pid == "already-there"
    assert seeded == []


def test_master_corpus_is_not_seeded():
    seeded: list = []

    def fake_seeder(name, **_kw):
        seeded.append(name)
        raise AssertionError("master_corpus must not hit the seed path")

    pid = resolve_golden_project(
        None, "master_corpus",
        base="http://example.invalid",
        headers={},
        seeder=fake_seeder,
        lookup=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no lookup")),
    )
    assert pid == "master_corpus"
    assert seeded == []


def test_seed_failure_is_fatal_and_does_not_return_display_name():
    cache: dict = {}

    def fake_lookup(client, value, **_kw):
        return None

    def fake_seeder(name, **_kw):
        raise RuntimeError("synthetic builder exploded")

    with pytest.raises(FixtureUnresolvedError, match="seed of .* failed"):
        resolve_golden_project(
            None, BOQ_FIXTURE_NAME,
            base="http://example.invalid",
            headers={},
            cache=cache,
            lookup=fake_lookup,
            seeder=fake_seeder,
        )
    assert cache.get(BOQ_FIXTURE_NAME) != BOQ_FIXTURE_NAME
    assert BOQ_FIXTURE_NAME not in cache


def test_preflight_missing_name_is_not_a_valid_project_id():
    """Preflight must not cache the display name as a project_id."""
    cache: dict = {}
    queries = [
        {"project": "master_corpus"},
        {"project": BOQ_FIXTURE_NAME},
    ]

    def fake_lookup(client, value, **_kw):
        return None

    def fake_seeder(name, **_kw):
        raise FixtureSeedError("seed blocked")

    with pytest.raises(FixtureUnresolvedError):
        preflight_named_fixtures(
            None, queries,
            base="http://example.invalid",
            headers={},
            cache=cache,
            lookup=fake_lookup,
            seeder=fake_seeder,
        )
    assert cache.get(BOQ_FIXTURE_NAME) != BOQ_FIXTURE_NAME
    assert BOQ_FIXTURE_NAME not in cache


def test_preflight_seeds_missing_boq_and_caches_live_id():
    seeded: list = []
    cache: dict = {}

    def fake_lookup(client, value, **_kw):
        return None

    def fake_seeder(name, **_kw):
        seeded.append(name)
        return {"project_id": "live-boq"}

    out = preflight_named_fixtures(
        None,
        [{"project": BOQ_FIXTURE_NAME}, {"project": "master_corpus"}],
        base="http://example.invalid",
        headers={},
        cache=cache,
        lookup=fake_lookup,
        seeder=fake_seeder,
    )
    assert seeded == [BOQ_FIXTURE_NAME]
    assert out[BOQ_FIXTURE_NAME] == "live-boq"
    assert out[BOQ_FIXTURE_NAME] != BOQ_FIXTURE_NAME


def test_unknown_name_is_not_used_as_project_id():
    with pytest.raises(FixtureUnresolvedError, match="refusing to send"):
        resolve_golden_project(
            None, "not-a-fixture",
            base="http://example.invalid",
            headers={},
            lookup=lambda *_a, **_k: None,
            seeder=lambda *_a, **_k: {"project_id": "should-not-run"},
        )


def test_cached_display_name_is_refused_and_reseeded():
    seeded: list = []

    def fake_lookup(client, value, **_kw):
        return None

    def fake_seeder(name, **_kw):
        seeded.append(name)
        return {"project_id": "reseeded-id"}

    pid = resolve_golden_project(
        None, BOQ_FIXTURE_NAME,
        base="http://example.invalid",
        headers={},
        cache={BOQ_FIXTURE_NAME: BOQ_FIXTURE_NAME},
        lookup=fake_lookup,
        seeder=fake_seeder,
    )
    assert seeded == [BOQ_FIXTURE_NAME]
    assert pid == "reseeded-id"


def test_golden_set_pilot_boq_total_still_names_fixture_boq():
    golden = load_golden_set()
    q = next(x for x in golden["queries"] if x["id"] == "pilot_boq_total")
    assert q["project"] == BOQ_FIXTURE_NAME
    named = collect_named_fixtures(golden["queries"])
    assert BOQ_FIXTURE_NAME in named
    assert "master_corpus" not in named
