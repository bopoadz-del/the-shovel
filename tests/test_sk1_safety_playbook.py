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


# --- The safety hat's own content (OPUSGRADE S658 27e8382 test debt) -------
# The tests above prove the prompt renders whatever the manifest holds; these
# pin what the safety manifest must hold, so dropping a licence credit, a hard
# refusal or the stop-work step fails here.

_CC_BY_LINE = (
    "Attribution: CC BY 4.0 -- Constructelligence "
    "(https://creativecommons.org/licenses/by/4.0/)"
)


def _safety_section(proc_id, tmp_path):
    """The rendered text of one safety procedure, from its heading to the next."""
    from app.agents.runtime import _apply_hat_kernels

    safety = catalog.get_hat(Discipline.SAFETY)
    body, _ = _apply_hat_kernels("base body", [safety.id], tmp_path / "a.md")
    start = body.index(f"### {proc_id}. ")
    nxt = body.find("\n### ", start + 1)
    return body[start:] if nxt < 0 else body[start:nxt]


@pytest.mark.parametrize("proc_id", ["s5", "s6"])
def test_licensed_safety_procedures_carry_the_credit_line(proc_id, tmp_path):
    section = _safety_section(proc_id, tmp_path)
    assert section.count(_CC_BY_LINE) == 1


def test_common_practice_procedure_carries_no_credit_line(tmp_path):
    # s4 is common incident-record practice: no licence, so no credit line.
    assert "Attribution:" not in _safety_section("s4", tmp_path)


@pytest.mark.parametrize(
    "proc_id,clause",
    [
        ("s4", "assigning cause or fault"),
        ("s4", "delaying the statutory flag"),
        ("s4", "computing rates from incomplete hours"),
        ("s6", "naming or implying a person"),
        ("s6", "inventing statistics"),
    ],
)
def test_safety_refusals_keep_their_hard_clauses(proc_id, clause, tmp_path):
    section = _safety_section(proc_id, tmp_path)
    refusals = next(l for l in section.splitlines() if l.startswith("Refusals: "))
    assert clause in refusals


@pytest.mark.parametrize(
    "step",
    [
        "a stop-work trigger per high-risk step",
        "controls ordered eliminate, substitute, engineer, administer, protect",
    ],
)
def test_job_safety_analysis_keeps_its_numbered_step(step, tmp_path):
    section = _safety_section("s5", tmp_path)
    numbered = [l.split(". ", 1)[1] for l in section.splitlines()
                if l[:1].isdigit() and ". " in l]
    assert step in numbered
