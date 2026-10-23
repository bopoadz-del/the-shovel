"""Tests for the artifact contract — Roadmap V2 · Epic 4."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import requires_construction_kit
from tests._dev_key import DEV_KEY
from app.core.artifacts import (
    code_artifact,
    make_artifact,
    result_to_artifacts,
    table_artifact,
)

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── builders ────────────────────────────────────────────────────────────────

def test_make_artifact_rejects_unknown_type():
    with pytest.raises(ValueError):
        make_artifact("hologram", "x", {})


def test_builders_produce_contract_shape():
    a = code_artifact("Snippet", "print(1)", "python")
    assert a["type"] == "code" and a["title"] == "Snippet"
    assert a["payload"]["language"] == "python"

    t = table_artifact("T", ["a", "b"], [[1, 2]])
    assert t["type"] == "table"
    assert t["payload"]["columns"] == ["a", "b"]
    assert t["payload"]["rows"] == [[1, 2]]


# ── result_to_artifacts ─────────────────────────────────────────────────────

def test_empty_result_yields_no_artifacts():
    assert result_to_artifacts({}) == []
    assert result_to_artifacts({"status": "success"}) == []


def test_generated_code_becomes_code_artifact():
    arts = result_to_artifacts({"generated_code": "result = 2 + 2"})
    assert len(arts) == 1 and arts[0]["type"] == "code"


def test_panel_line_items_become_table():
    result = {"panels": [{
        "type": "quantities", "title": "Quantities",
        "line_items": [
            {"item": "Concrete", "qty": 120, "unit": "m3"},
            {"item": "Rebar", "qty": 8, "unit": "t"},
        ],
    }]}
    arts = result_to_artifacts(result)
    assert len(arts) == 1
    assert arts[0]["type"] == "table"
    assert arts[0]["payload"]["columns"] == ["item", "qty", "unit"]
    assert len(arts[0]["payload"]["rows"]) == 2


def test_files_and_links_become_artifacts():
    arts = result_to_artifacts({
        "files": [{"name": "plan.pdf"}],
        "links": [{"label": "Spec", "url": "http://x/spec"}],
    })
    types = {a["type"] for a in arts}
    assert "file" in types and "link" in types


def test_text_is_fallback_only():
    # text alongside richer content → no text artifact
    arts = result_to_artifacts({"generated_code": "x=1", "text": "blah"})
    assert all(a["type"] != "text" for a in arts)
    # text alone → text artifact
    arts = result_to_artifacts({"text": "just some extracted text"})
    assert len(arts) == 1 and arts[0]["type"] == "text"


# ── /v1/execute attaches artifacts ──────────────────────────────────────────

@requires_construction_kit
def test_execute_response_carries_artifacts(client, monkeypatch):
    # formula_executor now delegates to the LLM-backed v2 block (Reasoning
    # Engine Plan 4). Mock the LLM seam so this plumbing test stays
    # deterministic without a live LLM API key.
    from app.blocks.formula_executor_v2 import FormulaExecutorV2Block

    async def _fake_llm(self, prompt):
        return "result = 2 + 2"

    monkeypatch.setattr(FormulaExecutorV2Block, "_call_llm", _fake_llm)

    r = client.post("/v1/execute", json={
        "block": "formula_executor_v2", "input": "2 + 2", "params": {},
    }, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "artifacts" in body
    assert isinstance(body["artifacts"], list)
    # formula_executor emits generated_code → a code artifact
    assert any(a["type"] == "code" for a in body["artifacts"])
