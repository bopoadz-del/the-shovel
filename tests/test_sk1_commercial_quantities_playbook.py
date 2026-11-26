"""
S-K1 -- the commercial and quantities hats' skill playbooks, imported through
the loader #507 merged. The advisor's commercial_quantities file spans two
hats (design note split): Q1 unit-rate review, Q3 IPC check, Q4 change
pricing narrative -> commercial; Q2 takeoff review, Q5 bill-row consistency
-> quantities.

Hat-agnostic contracts:
- a lineage reading "as <id>" carries the referenced procedure's attribution,
  resolved across every hat (a playbook file split over two hats keeps its
  lineage: Q2 "as Q1" sits on quantities, Q1 on commercial);
- each hat's assembled prompt carries its own procedures and none of the other
  hat's procedure headings.
"""
from __future__ import annotations

import re

import pytest

from app.agents import catalog
from app.agents.models import Discipline

_REF = re.compile(r"^as ([A-Za-z]\d+)\b")


def _hats_with_procedures():
    return [h for h in catalog.list_hats() if h.playbook.procedures]


def _all_procedures():
    return {p.id.lower(): p for h in _hats_with_procedures() for p in h.playbook.procedures}


@pytest.mark.parametrize("discipline", [Discipline.COMMERCIAL, Discipline.QUANTITIES])
def test_hat_ships_rules_and_procedures(discipline):
    hat = catalog.get_hat(discipline)
    assert hat.playbook.rules
    assert len(hat.playbook.procedures) >= 2
    for proc in hat.playbook.procedures:
        assert proc.refusals and proc.steps and proc.done_when


@pytest.mark.parametrize("hat", _hats_with_procedures(), ids=lambda h: h.id)
def test_lineage_reference_resolves_across_hats(hat):
    every = _all_procedures()
    for proc in hat.playbook.procedures:
        m = _REF.match(proc.lineage)
        if m:
            ref = every.get(m.group(1).lower())
            assert ref is not None, f"{hat.id}/{proc.id} lineage names no shipped procedure"
            assert proc.attribution == ref.attribution, f"{hat.id}/{proc.id}"


@pytest.mark.parametrize(
    "own, other", [(Discipline.COMMERCIAL, Discipline.QUANTITIES), (Discipline.QUANTITIES, Discipline.COMMERCIAL)]
)
def test_prompt_carries_own_procedures_not_the_other_hats(own, other, tmp_path):
    from app.agents.runtime import _apply_hat_kernels

    mine, theirs = catalog.get_hat(own), catalog.get_hat(other)
    body, _ = _apply_hat_kernels("base body", [mine.id], tmp_path / "a.md")
    for proc in mine.playbook.procedures:
        assert f"### {proc.id}. {proc.title}" in body
        assert proc.refusals in body
        for step in proc.steps:
            assert step in body
    for proc in theirs.playbook.procedures:
        assert f"### {proc.id}. {proc.title}" not in body
