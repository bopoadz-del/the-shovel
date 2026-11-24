"""
S-K1 -- the safety hat's skill playbooks (S4 incident record, S5 job safety
analysis, S6 toolbox talk; the qaqc_safety file's S4-S6 half per the design
note), imported through the loader #507 merged.

Hat-agnostic: on every hat that ships procedures, the assembled agent prompt
keeps the hat's own guidance first and the procedures behind it (a playbook is
guidance under the hat, never in front of it), every refusal and step reaches
the prompt. A lineage reading "as <id>" is checked across every hat by
test_sk1_commercial_quantities_playbook (a playbook file may span two hats).
"""
from __future__ import annotations

import pytest

from app.agents import catalog
from app.agents.models import Discipline


def _hats_with_procedures():
    return [h for h in catalog.list_hats() if h.playbook.procedures]


def test_safety_hat_ships_procedures_beside_another_hat():
    ids = {h.id for h in _hats_with_procedures()}
    assert "fork.hat.safety" in ids
    assert len(ids) >= 2
    safety = catalog.get_hat(Discipline.SAFETY)
    assert len(safety.playbook.procedures) >= 3
    assert safety.playbook.rules


@pytest.mark.parametrize("hat", _hats_with_procedures(), ids=lambda h: h.id)
def test_procedures_sit_behind_the_hats_guidance(hat, tmp_path):
    from app.agents.runtime import _apply_hat_kernels

    body, _ = _apply_hat_kernels("base body", [hat.id], tmp_path / "a.md")
    head = hat.system_prompt_guidance[:80]
    assert head in body
    assert body.index(head) < body.index("## Procedures")
    for proc in hat.playbook.procedures:
        assert proc.refusals in body
        for step in proc.steps:
            assert step in body
