"""KS-3: a broken agent or hat setting fails readiness.

Before: load_agents() caught a config that did not parse, printed one line
and went on, so the service came up ready with an agent missing; the hat
catalog raised only at the first chat that wore a hat. Now a config that
does not parse is recorded (AGENT_LOAD_FAILURES), the hat catalog is read at
start-up, and /ready answers 503 naming the file and the error type.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.agents.runtime as rt
from app.agents import catalog
from app.main import app

CONFIGS = Path("app/agents/configs")
MANIFESTS = Path("app/agents/manifests")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def restore_agents():
    yield
    rt.load_agents()


# -- the loader records what it could not parse -----------------------------

@pytest.mark.parametrize("broken_name, broken_text", [
    ("zz-no-frontmatter.md", "just a body, no frontmatter at all\n"),
    ("aa-unterminated.md", "---\nname: half-open\ndescription: never closed\n"),
])
def test_a_config_that_does_not_parse_is_recorded_and_the_rest_load(
        tmp_path, restore_agents, broken_name, broken_text):
    good = sorted(CONFIGS.glob("*.md"))[:2]
    for md in good:
        shutil.copy(md, tmp_path / md.name)
    (tmp_path / broken_name).write_text(broken_text, encoding="utf-8")

    loaded = rt.load_agents(tmp_path)

    assert len(loaded) == len(good)
    assert rt.AGENT_LOAD_FAILURES == {broken_name: "ValueError"}


@pytest.mark.parametrize("count", [1, 3])
def test_a_clean_reload_clears_the_record(tmp_path, restore_agents, count):
    (tmp_path / "broken.md").write_text("no frontmatter\n", encoding="utf-8")
    rt.load_agents(tmp_path)
    assert rt.AGENT_LOAD_FAILURES

    clean = tmp_path / "clean"
    clean.mkdir()
    for md in sorted(CONFIGS.glob("*.md"))[:count]:
        shutil.copy(md, clean / md.name)
    assert len(rt.load_agents(clean)) == count
    assert rt.AGENT_LOAD_FAILURES == {}


# -- the hat catalog reports its own load error -----------------------------

def _broken_manifest_dir(tmp_path: Path, how: str) -> Path:
    d = tmp_path / how
    d.mkdir()
    for p in MANIFESTS.glob("fork.*.json"):
        shutil.copy(p, d / p.name)
    if how == "unparseable":
        (d / "fork.hat.planning.json").write_text("{ not json", encoding="utf-8")
    elif how == "unknown_base":
        data = json.loads((d / "fork.hat.qaqc.json").read_text(encoding="utf-8"))
        data["extends"] = "fork.no_such_base"
        (d / "fork.hat.qaqc.json").write_text(json.dumps(data), encoding="utf-8")
    return d


@pytest.mark.parametrize("how, error", [
    ("unparseable", "JSONDecodeError"),
    ("unknown_base", "ValueError"),
])
def test_a_broken_hat_manifest_is_a_catalog_load_error(tmp_path, monkeypatch, how, error):
    monkeypatch.setenv("AGENT_MANIFEST_DIR", str(_broken_manifest_dir(tmp_path, how)))
    monkeypatch.setattr(catalog, "_catalog", None)
    assert catalog.catalog_load_error() == error


@pytest.mark.parametrize("subset", [2, 9])
def test_a_sound_catalog_has_no_load_error(tmp_path, monkeypatch, subset):
    base = [p for p in MANIFESTS.glob("fork.*.json") if p.name == "fork.base.json"]
    hats = sorted(p for p in MANIFESTS.glob("fork.hat.*.json"))[: subset - 1]
    for p in base + hats:
        shutil.copy(p, tmp_path / p.name)
    monkeypatch.setenv("AGENT_MANIFEST_DIR", str(tmp_path))
    monkeypatch.setattr(catalog, "_catalog", None)
    assert catalog.catalog_load_error() is None
    assert len(catalog.get_agent_catalog()) == len(base + hats)


# -- /ready ------------------------------------------------------------------

def test_ready_reports_sound_agent_settings(client):
    r = client.get("/ready")
    agents = r.json()["checks"]["agents"]
    assert agents == {"ok": True, "failed_configs": {}, "hat_catalog_error": None}


@pytest.mark.parametrize("failures", [
    {"broken-one.md": "ValueError"},
    {"a.md": "KeyError", "b.md": "ValueError"},
])
def test_ready_is_503_when_an_agent_config_failed(client, monkeypatch, failures):
    monkeypatch.setattr(rt, "AGENT_LOAD_FAILURES", dict(failures))
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert body["checks"]["agents"]["ok"] is False
    assert body["checks"]["agents"]["failed_configs"] == dict(sorted(failures.items()))


@pytest.mark.parametrize("how, error", [
    ("unparseable", "JSONDecodeError"),
    ("unknown_base", "ValueError"),
])
def test_ready_is_503_when_the_hat_catalog_is_broken(client, tmp_path, monkeypatch, how, error):
    monkeypatch.setenv("AGENT_MANIFEST_DIR", str(_broken_manifest_dir(tmp_path, how)))
    monkeypatch.setattr(catalog, "_catalog", None)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["agents"] == {
        "ok": False, "failed_configs": {}, "hat_catalog_error": error}


@pytest.mark.parametrize("exc", [ImportError, RuntimeError])
def test_ready_fails_closed_when_the_settings_cannot_be_read(client, monkeypatch, exc):
    def boom():
        raise exc("unreadable")
    monkeypatch.setattr(catalog, "catalog_load_error", boom)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["agents"]["error"] == exc.__name__
