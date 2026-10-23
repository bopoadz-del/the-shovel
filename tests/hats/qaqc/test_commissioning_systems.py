"""When the user names commissioning systems, the lane tool returns only those."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.core.tool_registry import ToolCall
from app.agents.hats.qaqc.tools import handle_commissioning_checklist


class _Container:
    def __init__(self):
        self.seen = None

    async def commissioning_checklist(self, input_data, params):
        self.seen = list(params.get("systems") or [])
        systems = self.seen
        return {
            "status": "success",
            "checklists_by_system": {key: [{"test": key}] for key in systems},
            "summary": {
                "systems_covered": len(systems),
                "total_tests": len(systems),
                "pending": len(systems),
            },
        }


@pytest.mark.asyncio
async def test_named_systems_are_the_only_ones_returned(monkeypatch):
    container = _Container()
    monkeypatch.setattr(
        "app.dependencies.get_block_instance", lambda _name: container,
    )
    call = ToolCall(
        agent=SimpleNamespace(allowed_blocks=["construction"]),
        name="commissioning_checklist",
        args={"systems": ["electrical", "hvac", "fire", "elevator", "facade"]},
        user_message="commissioning checklist for the chillers and the fire alarm",
    )
    out = await handle_commissioning_checklist(call)
    assert out["ok"] is True
    systems = set(out["result"]["checklists_by_system"])
    assert systems == {"hvac", "fire"}
    assert container.seen == ["hvac", "fire"] or set(container.seen) == systems


@pytest.mark.asyncio
async def test_defaults_are_used_only_when_none_are_named(monkeypatch):
    container = _Container()
    monkeypatch.setattr(
        "app.dependencies.get_block_instance", lambda _name: container,
    )
    call = ToolCall(
        agent=SimpleNamespace(allowed_blocks=["construction"]),
        name="commissioning_checklist",
        args={},
        user_message="generate a commissioning checklist",
    )
    out = await handle_commissioning_checklist(call)
    assert out["ok"] is True
    systems = list(out["result"]["checklists_by_system"])
    assert len(systems) >= 2
    assert "electrical" in systems or "hvac" in systems


class _CatalogOnlyContainer:
    """Standard tests exist only for catalog keys, as the container does."""

    KEYS = {
        "hvac", "electrical", "fire", "plumbing", "elevator", "facade",
        "waterproofing", "reservoir", "bms",
    }

    async def commissioning_checklist(self, input_data, params):
        checklists = {}
        for key in params.get("systems") or []:
            if key in self.KEYS:
                checklists[key] = [
                    {"test": f"{key} functional test"},
                    {"test": f"{key} performance test"},
                ]
            else:
                checklists[key] = []
        return {
            "status": "success",
            "checklists_by_system": checklists,
            "summary": {
                "systems_covered": len(checklists),
                "total_tests": sum(len(v) for v in checklists.values()),
            },
        }


def _items_for(result, *names):
    by_sys = result.get("checklists_by_system") or {}
    lowered = {str(key).lower(): value for key, value in by_sys.items()}
    for name in names:
        items = lowered.get(name.lower())
        if isinstance(items, list) and items:
            return items
    return []


@pytest.mark.asyncio
@pytest.mark.parametrize("systems", (
    ["HVAC", "fire alarm"],
    "HVAC, fire alarm",
    "HVAC and fire alarm",
))
async def test_each_recognised_system_has_standard_tests(monkeypatch, systems):
    monkeypatch.setattr(
        "app.dependencies.get_block_instance",
        lambda _name: _CatalogOnlyContainer(),
    )
    call = ToolCall(
        agent=SimpleNamespace(allowed_blocks=["construction"]),
        name="commissioning_checklist",
        args={"systems": systems},
        user_message="",
    )
    out = await handle_commissioning_checklist(call)
    assert out["ok"] is True
    result = out["result"]
    hvac = _items_for(result, "hvac", "HVAC")
    fire = _items_for(result, "fire", "fire alarm", "fire_protection")
    assert len(hvac) > 0
    assert len(fire) > 0
    keys = {str(key).lower() for key in (result.get("checklists_by_system") or {})}
    assert "electrical" not in keys


@pytest.mark.asyncio
async def test_unrecognised_system_gets_a_plain_words_note(monkeypatch):
    monkeypatch.setattr(
        "app.dependencies.get_block_instance",
        lambda _name: _CatalogOnlyContainer(),
    )
    call = ToolCall(
        agent=SimpleNamespace(allowed_blocks=["construction"]),
        name="commissioning_checklist",
        args={"systems": ["kitchen extract"]},
        user_message="",
    )
    out = await handle_commissioning_checklist(call)
    result = out["result"]
    by_sys = result.get("checklists_by_system") or {}
    assert by_sys.get("kitchen extract") in (None, [])
    notes = " ".join(str(n) for n in (result.get("notes") or [result.get("note")]))
    assert "kitchen extract" in notes.lower() or "kitchen" in notes.lower()
    assert "no standard tests" in notes.lower()
    assert "[" not in notes
