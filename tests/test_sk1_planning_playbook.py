"""
S-K1 -- the planning hat's skill playbooks (L1-L5, office skill
meeting-minutes-actions is L4), imported through the loader #507 merged.

Hat-agnostic where the decision is general: every hat that ships procedures
renders its binding rules, each refusal and each step in order into its own
prompt block, and the assembled agent prompt carries that block. A procedure
whose lineage reads "as <id>" shares the referenced procedure's method lineage,
so it carries the same licence obligation (attribution) -- read from the
playbook's own text, no procedure named in the rule.
"""
from __future__ import annotations

import re

import pytest

from app.agents import catalog
from app.agents.models import Discipline
from app.agents.procedures import render_procedures

_REF = re.compile(r"^as ([A-Za-z]\d+)\b")


def _hats_with_procedures():
    return [h for h in catalog.list_hats() if h.playbook.procedures]


def test_planning_hat_ships_procedures_beside_another_hat():
    ids = {h.id for h in _hats_with_procedures()}
    assert "fork.hat.planning" in ids
    assert len(ids) >= 2
    planning = catalog.get_hat(Discipline.PLANNING)
    assert len(planning.playbook.procedures) >= 5
    assert planning.playbook.rules


@pytest.mark.parametrize("hat", _hats_with_procedures(), ids=lambda h: h.id)
def test_every_step_and_refusal_renders_in_order(hat):
    out = render_procedures(hat)
    assert f"Binding rules: {hat.playbook.rules}" in out
    cursor = 0
    for proc in hat.playbook.procedures:
        assert proc.refusals in out
        for step in proc.steps:
            at = out.find(step, cursor)
            assert at >= 0, f"{hat.id}/{proc.id}: step missing or out of order: {step!r}"
            cursor = at


@pytest.mark.parametrize("hat", _hats_with_procedures(), ids=lambda h: h.id)
def test_lineage_reference_shares_attribution(hat):
    by_id = {p.id.lower(): p for p in hat.playbook.procedures}
    for proc in hat.playbook.procedures:
        m = _REF.match(proc.lineage)
        if not m:
            continue
        target = by_id.get(m.group(1).lower())
        assert target is not None, f"{hat.id}/{proc.id}: lineage names an unknown procedure"
        assert proc.attribution == target.attribution, f"{hat.id}/{proc.id}"


@pytest.mark.parametrize("hat", _hats_with_procedures(), ids=lambda h: h.id)
def test_agent_prompt_carries_the_hats_procedures(hat, tmp_path):
    from app.agents.runtime import _apply_hat_kernels

    body, ids = _apply_hat_kernels("base body", [hat.id], tmp_path / "a.md")
    assert hat.id in ids
    assert "## Procedures" in body
    for proc in hat.playbook.procedures:
        assert proc.steps[0] in body
    # Two hats may share binding rules (one playbook file split across hats),
    # so "only its own" is read on the procedure headings, not the rules.
    others = [h for h in _hats_with_procedures() if h.id != hat.id]
    for other in others:
        for proc in other.playbook.procedures:
            assert f"### {proc.id}. {proc.title}" not in body
