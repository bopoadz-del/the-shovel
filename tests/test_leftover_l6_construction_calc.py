"""Leftover L6: trench 14.5 × 3.2 × 1.75 m bank volume → 81.2 m³.

Live 4e0b485 (pinned smart-orchestrator on /v1/chat/stream):
  The hat called ``construction`` with ``action=construction_calc``.
  ``route()`` returned Unknown action; the model then narrated a
  ``formula_execute`` JSON stub and never emitted 81.2.

These tests pin the machinery (no live LLM). They must run under the
virgin CI profile: ``construction_calc`` is a method on the class, not a
kit-gated block, and the virgin job's diff-cover gate still scores the
new lines.
"""
from __future__ import annotations

import json

import pytest

from app.agents import runtime
from app.containers.construction import ConstructionContainer

L6_PROMPT = (
    "Stay on smart-orchestrator. Bank volume of a rectangular trench "
    "14.5 m long by 3.2 m wide by 1.75 m deep. Start with the "
    "smart_orchestrator tool then a calculator. Report Intent, Routed to, "
    "and the bank volume in m3."
)



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

def _bank(payload: dict):
    inner = payload.get("result") or payload
    if isinstance(inner, dict) and isinstance(inner.get("result"), dict):
        inner = inner["result"]
    return inner.get("bank_volume_m3") if isinstance(inner, dict) else None


@pytest.mark.asyncio
async def test_container_construction_calc_alias_returns_81_2():
    r = await ConstructionContainer().route(
        "construction_calc",
        {},
        {
            "action": "construction_calc",
            "name": "excavation_volume",
            "length_m": 14.5,
            "width_m": 3.2,
            "depth_m": 1.75,
        },
    )
    assert r.get("status") == "success", r
    assert _bank(r) == pytest.approx(81.2)


@pytest.mark.asyncio
async def test_construction_calc_infers_excavation_from_formula_string():
    r = await ConstructionContainer().route(
        "construction_calc",
        {},
        {"action": "construction_calc", "formula": "14.5 * 3.2 * 1.75", "unit": "m3"},
    )
    assert r.get("status") == "success", r
    assert _bank(r) == pytest.approx(81.2)


@pytest.mark.asyncio
async def test_construction_calc_ignores_extra_text_kwarg():
    """Live UI L6: model passed input text as an extra calculator kwarg."""
    r = await ConstructionContainer().route(
        "construction_calc",
        {"text": "Bank volume of a rectangular trench 14.5 m"},
        {
            "action": "construction_calc",
            "name": "excavation_volume",
            "length_m": 14.5,
            "width_m": 3.2,
            "depth_m": 1.75,
            "text": "Bank volume of a rectangular trench 14.5 m",
        },
    )
    assert r.get("status") == "success", r
    assert _bank(r) == pytest.approx(81.2)


@pytest.mark.asyncio
async def test_construction_calc_nested_params_and_length_aliases():
    r = await ConstructionContainer().construction_calc(
        {},
        {
            "action": "construction_calc",
            "params": {"length": 14.5, "width": 3.2, "depth": 1.75},
        },
    )
    assert r.get("status") == "success", r
    assert _bank(r) == pytest.approx(81.2)


@pytest.mark.asyncio
async def test_construction_calc_infers_from_lwd_keys_without_name():
    r = await ConstructionContainer().construction_calc(
        {"length_m": 14.5, "width_m": 3.2, "height_m": 1.75},
        {"action": "construction_calc"},
    )
    assert r.get("status") == "success", r
    assert _bank(r) == pytest.approx(81.2)


@pytest.mark.asyncio
async def test_construction_calc_rejects_empty_request():
    r = await ConstructionContainer().construction_calc({}, {"action": "construction_calc"})
    assert r.get("status") == "error", r
    assert "calculation name" in (r.get("error") or "")
    assert r.get("available")


@pytest.mark.asyncio
async def test_construction_execute_accepts_hat_shape():
    env = await ConstructionContainer().execute(
        {},
        {
            "action": "construction_calc",
            "name": "excavation_volume",
            "length_m": 14.5,
            "width_m": 3.2,
            "depth_m": 1.75,
        },
    )
    assert env.get("status") == "success", env
    blob = json.dumps(env, default=str)
    assert "81.2" in blob


@pytest.mark.asyncio
async def test_orchestrator_routes_l6_prompt_to_construction_calc():
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    r = await SmartOrchestratorBlock().process({"user_message": L6_PROMPT})
    assert r["status"] == "success"
    assert "construction_calc" in (r.get("action_queue") or []), r
    matched = [m.get("action") for m in (r.get("matched_actions") or [])]
    assert "construction_calc" in matched, r


class _FakeExtractor:
    def __init__(self):
        self.calls = []

    async def execute(self, input_data, params=None):
        self.calls.append(input_data)
        return {"status": "success", "areas_m2": 1}


@pytest.mark.asyncio
async def test_l6_self_contained_volume_skips_drawing_qto_predispatch(monkeypatch):
    """Project PDFs must not steal leftover L6 when dimensions are in the ask."""
    import app.core.projects as projects_mod

    fake = _FakeExtractor()
    monkeypatch.setattr(
        projects_mod,
        "list_documents",
        lambda pid: [{"original_name": "fenwick_drawing_TM9900001_20260819-081916.pdf"}],
        raising=False,
    )
    monkeypatch.setattr(runtime, "block_instances", {"drawing_qto": fake}, raising=False)
    agent = runtime.Agent(
        name="smart-orchestrator",
        description="",
        system_prompt="x",
        allowed_blocks=["drawing_qto", "construction"],
    )
    msgs = [{"role": "user", "content": L6_PROMPT}]
    rec = await runtime._predispatch_file_tool(agent, msgs, "p1")
    assert rec is None
    assert fake.calls == []
    assert len(msgs) == 1
