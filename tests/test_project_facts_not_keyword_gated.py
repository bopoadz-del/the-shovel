"""A stored project fact must reach the model even when the message
doesn't happen to mention its keywords.

Live-fire find 2026-08-15 (F23). The drawings project carried two
operator-stated facts: the floor-to-ceiling height and the door schedule.
A take-off request surfaced the height (the word "floor" in "floor area"
keyword-matched the fact key) but NOT the door schedule ("door" appears
nowhere in a request for wall paint and blockwork quantities) -- so the
agent asked the operator for a value the project already carried.
``build_memory_context`` keyword-gated the facts by the user's message,
which inverts the store's purpose: the fact you forget to name is exactly
the one that must be surfaced.

Matched facts now rank first and the remaining facts fill the limit.
"""
from __future__ import annotations

import pytest

from app.core import project_memory


@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import projects as store
    pid = store.create_project("facts-gate-test", user_id="system")["id"]
    store.set_fact(pid, "floor_to_ceiling_height_m", "3.10 (all floors)")
    store.set_fact(pid, "door_schedule", "one door per room, 0.90 x 2.10 m")
    return pid


def test_the_unmentioned_fact_still_reaches_the_context(project):
    """RED before the fix: the live request wording drops door_schedule."""
    ctx = project_memory.build_memory_context(
        project,
        "Give me the wall paint and blockwork quantities for this project's "
        "measured take-off: 192 rooms, 1947.87 square metres floor area, "
        "2342.20 metres perimeter.",
    )
    assert "3.10" in ctx, "height fact lost"
    assert "0.90 x 2.10" in ctx, "door schedule fact was keyword-gated out"


def test_matched_facts_rank_first(project):
    ctx = project_memory.build_memory_context(project, "what door sizes apply?")
    lines = [ln for ln in ctx.splitlines() if ln.startswith("- ")]
    assert "door" in lines[0].lower()


def test_no_query_still_lists_everything(project):
    ctx = project_memory.build_memory_context(project)
    assert "3.10" in ctx and "0.90 x 2.10" in ctx


def test_the_limit_still_caps_the_block(project):
    from app.core import projects as store
    for i in range(20):
        store.set_fact(project, f"minor_fact_{i}", f"value {i}")
    ctx = project_memory.build_memory_context(project, "anything", limit=8)
    assert len([ln for ln in ctx.splitlines() if ln.startswith("- ")]) == 8
