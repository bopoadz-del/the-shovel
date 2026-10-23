"""Contract Data Q&A must not predispatch drawing_qto (UI-PHYS A5 / D2).

Live stall: delay-damages lookup predispatched drawing_qto, then a 3-iteration
tool loop with ~85s silence before SSE (xrid 11b957d3-3c5). Wrong first tool,
not a streaming/watchdog bug.

The collision is mechanical: a DXF named ``the whole of the Works.dxf`` has
a ≥12-char stem that is a substring of "…for the whole of the Works?".
"""
from __future__ import annotations

import pytest

from app.agents import runtime
from app.core.contract_lookup_intent import CONTRACT_LOOKUP_BLOCKED_ACTIONS


class _FakeQTO:
    def __init__(self):
        self.calls = []

    async def execute(self, input_data, params=None):
        self.calls.append(input_data)
        return {"status": "success", "areas_m2": 1200}



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

def _wire(monkeypatch, docs):
    import app.core.projects as projects_mod
    fake = _FakeQTO()
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: docs, raising=False)
    monkeypatch.setattr(runtime, "block_instances", {"drawing_qto": fake}, raising=False)
    monkeypatch.setattr(
        runtime, "_resolve_file_path",
        lambda pid, raw: "/app/data/" + raw,
    )
    return fake


def _agent():
    return runtime.Agent(
        name="project-assistant",
        description="",
        system_prompt="x",
        allowed_blocks=["drawing_qto", "construction"],
    )


DELAY_DAMAGES = "What are the Delay Damages for the whole of the Works?"
SCHEDULE_10 = "What does Schedule 10 of the contract contain?"
COLLIDING_DXF = "the whole of the Works.dxf"


@pytest.mark.asyncio
async def test_delay_damages_does_not_predispatch_colliding_dxf(monkeypatch):
    fake = _wire(monkeypatch, [{"original_name": COLLIDING_DXF}])
    msgs = [{"role": "user", "content": DELAY_DAMAGES}]
    rec = await runtime._predispatch_file_tool(_agent(), msgs, "p1")
    assert rec is None
    assert fake.calls == []
    assert len(msgs) == 1
    assert "PLATFORM PRE-DISPATCH" not in (msgs[-1].get("content") or "")


@pytest.mark.asyncio
async def test_schedule_n_lookup_does_not_predispatch_drawing_qto(monkeypatch):
    fake = _wire(monkeypatch, [{"original_name": COLLIDING_DXF}])
    msgs = [{"role": "user", "content": SCHEDULE_10}]
    rec = await runtime._predispatch_file_tool(_agent(), msgs, "p1")
    assert rec is None
    assert fake.calls == []


@pytest.mark.asyncio
async def test_real_qto_of_named_dxf_still_predispatches(monkeypatch):
    fake = _wire(monkeypatch, [{"original_name": "tower_b.dxf"}])
    msgs = [{"role": "user", "content": "QTO of tower_b.dxf please"}]
    rec = await runtime._predispatch_file_tool(_agent(), msgs, "p1")
    assert rec and rec["predispatched"] and rec["name"] == "drawing_qto"
    assert fake.calls == [{"file_path": "/app/data/tower_b.dxf"}]


def test_whole_of_the_works_dxf_is_a_real_filename_collision():
    """Precondition: without the lookup guard this DXF *would* predispatch."""
    assert runtime._user_names_project_file(
        DELAY_DAMAGES.lower(), COLLIDING_DXF,
    ) is True


def test_drawing_qto_is_blocked_on_contract_data_lookup():
    assert "drawing_qto" in CONTRACT_LOOKUP_BLOCKED_ACTIONS

