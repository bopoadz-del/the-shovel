"""
S-K1 -- every imported skill playbook satisfies one contract, on every hat.

The loader (#507) is hat-agnostic: whichever hat manifest carries
``playbook.procedures`` gets a PROCEDURES block in its prompt. These tests read
the shipped manifests themselves (no playbook id, clause or figure is named
here) and hold every hat that carries procedures to the same contract:

* the hat's binding rules travel with its procedures;
* each procedure's refusals and every step reach the rendered block, in order;
* a procedure id belongs to exactly one hat (no cross-hat reuse);
* the base manifest never inherits a hat's procedures;
* a hatted agent's assembled prompt carries the block for each of its hats.

Each decision is checked on at least two hats so a mutation that special-cases
one hat fails.
"""
from __future__ import annotations

import pytest

from app.agents import catalog
from app.agents.procedures import render_procedures


def _hats_with_procedures():
    return [h for h in catalog.list_hats() if h.playbook.procedures]


def test_at_least_two_hats_carry_procedures():
    # The contract below is only meaningful across hats; the import of the
    # office skills into a second hat is what this guards.
    assert len(_hats_with_procedures()) >= 2


@pytest.mark.parametrize("hat", _hats_with_procedures(), ids=lambda h: h.id)
def test_hat_procedures_render_rules_refusals_and_steps_in_order(hat):
    out = render_procedures(hat)
    assert hat.playbook.rules, f"{hat.id} carries procedures without binding rules"
    assert hat.playbook.rules in out
    cursor = 0
    for proc in hat.playbook.procedures:
        assert proc.refusals.strip()
        assert proc.steps
        assert proc.refusals in out
        for step in proc.steps:
            idx = out.find(step, cursor)
            assert idx >= 0, f"{hat.id}/{proc.id} step missing or out of order"
            cursor = idx + len(step)
        assert proc.done_when in out


def test_procedure_ids_unique_within_and_across_hats():
    seen: dict[str, str] = {}
    for hat in _hats_with_procedures():
        for proc in hat.playbook.procedures:
            assert proc.id not in seen, f"{proc.id} on both {seen.get(proc.id)} and {hat.id}"
            seen[proc.id] = hat.id
    assert len(set(seen.values())) >= 2


def test_base_never_inherits_any_hat_procedure():
    base = catalog.get_base()
    assert not base.playbook.procedures
    for hat in _hats_with_procedures():
        merged = catalog.resolve_hat_with_base(hat)
        assert [p.id for p in merged.playbook.procedures] == [p.id for p in hat.playbook.procedures]


def test_hatted_agent_prompt_carries_each_hats_block(tmp_path):
    from app.agents.runtime import _apply_hat_kernels

    hats = _hats_with_procedures()
    body, ids = _apply_hat_kernels("base body", [h.id for h in hats], tmp_path / "a.md")
    for hat in hats:
        assert hat.id in ids
        assert hat.playbook.rules in body
        for proc in hat.playbook.procedures:
            assert proc.refusals in body
    # One hat alone carries only its own procedures.
    first, second = hats[0], hats[1]
    solo, _ = _apply_hat_kernels("base body", [first.id], tmp_path / "b.md")
    assert first.playbook.rules in solo
    assert second.playbook.rules not in solo
