"""
S-K1 -- skill playbook import (contracts hat, first skill).

The mechanism under test: a hat manifest carries imported ``playbook.procedures``
and an optional ``playbook.rules`` preamble; ``render_procedures`` turns them into
a compact PROCEDURES block for the hat prompt; ``_apply_hat_kernels`` appends that
block behind the hat's guidance. A procedure belongs to exactly one hat -- the
base manifest never inherits one.
"""
from __future__ import annotations

import pytest

from app.agents import catalog
from app.agents.models import (
    ActivationConfig,
    ActivationTrigger,
    AgentManifest,
    ContextSource,
    Discipline,
    FailurePolicy,
    ManifestKind,
    Playbook,
    PlaybookFormula,
    PlaybookProcedure,
    Verification,
)
from app.agents.procedures import render_procedures, skill_playbooks_enabled


def _proc(pid: str, title: str, refusals: str, steps: list[str]) -> PlaybookProcedure:
    return PlaybookProcedure(
        id=pid,
        title=title,
        purpose=f"purpose of {pid}",
        inputs=f"inputs for {pid}",
        refusals=refusals,
        steps=steps,
        output=f"output of {pid}",
        done_when=f"done when {pid}",
        lineage=f"lineage {pid}",
    )


def _manifest(*procs: PlaybookProcedure) -> AgentManifest:
    return AgentManifest(
        manifest_version="1.0",
        id="test.hat.playbook",
        name="Playbook test hat",
        description="hat for procedure rendering tests",
        kind=ManifestKind.HAT,
        extends="fork.base",
        discipline=Discipline.CONTRACTS,
        system_prompt_guidance="guidance",
        activation=ActivationConfig(triggers=[ActivationTrigger(type="intent", value="test")]),
        context_sources=[ContextSource(kind="project_documents")],
        allowed_actions=[],
        available_tools=[],
        verification=Verification(),
        failure_policy=FailurePolicy(),
        playbook=Playbook(formulas=[PlaybookFormula(id="f1", description="d")], procedures=list(procs)),
    )


def _manifest_dict() -> dict:
    return _manifest(_proc("a1", "Alpha", "never guess A", ["step A1"])).model_dump()


def test_two_procedures_render_steps_and_refusals_in_order():
    manifest = _manifest(
        _proc("a1", "Alpha", "never guess A", ["step A1", "step A2"]),
        _proc("a2", "Beta", "never guess B", ["step B1"]),
    )
    out = render_procedures(manifest)
    assert "## Procedures" in out
    assert "never guess A" in out
    assert "never guess B" in out
    assert "step A1" in out
    assert "step A2" in out
    assert "step B1" in out
    assert out.index("step A1") < out.index("step A2") < out.index("step B1")


def test_skill_playbooks_off_renders_nothing(monkeypatch):
    monkeypatch.setenv("SKILL_PLAYBOOKS", "0")
    a = _manifest(_proc("a1", "Alpha", "never guess A", ["step A1"]))
    b = _manifest(_proc("b1", "Bravo", "never guess B", ["step B1"]))
    assert not skill_playbooks_enabled()
    assert not render_procedures(a)
    assert not render_procedures(b)


def test_no_procedures_renders_nothing():
    assert not render_procedures(_manifest())
    assert not render_procedures(None)


def test_procedure_missing_required_field_fails_validation():
    from pydantic import ValidationError

    no_refusals = _manifest_dict()
    del no_refusals["playbook"]["procedures"][0]["refusals"]
    no_steps = _manifest_dict()
    del no_steps["playbook"]["procedures"][0]["steps"]
    with pytest.raises(ValidationError):
        AgentManifest(**no_refusals)
    with pytest.raises(ValidationError):
        AgentManifest(**no_steps)


def test_base_never_inherits_hat_procedure():
    contracts = catalog.get_hat(Discipline.CONTRACTS)
    merged = catalog.resolve_hat_with_base(contracts)
    assert merged.playbook.procedures
    assert len(merged.playbook.procedures) >= 7
    assert "c1" in {p.id for p in merged.playbook.procedures}
    assert merged.playbook.rules
    assert not catalog.get_base().playbook.procedures
    assert catalog.get_base().playbook.rules is None


def test_contracts_hat_carries_every_playbook_steps_and_refusals():
    contracts = catalog.get_hat(Discipline.CONTRACTS)
    out = render_procedures(contracts)
    assert contracts.playbook.procedures
    assert "every figure, date, clause number and name" in out
    for proc in contracts.playbook.procedures:
        assert proc.refusals in out
        for step in proc.steps:
            assert step in out


def test_contracts_hat_attribution_recorded_and_rendered():
    contracts = catalog.get_hat(Discipline.CONTRACTS)
    procs = {p.id: p for p in contracts.playbook.procedures}
    assert procs["c2"].attribution is not None
    assert "CC BY" in procs["c2"].attribution.licence
    assert "Apache" in procs["c4"].attribution.licence
    assert procs["c1"].attribution is None
    assert procs["c3"].attribution is None
    assert procs["c7"].attribution is None
    out = render_procedures(contracts)
    assert "CC BY 4.0" in out
    assert "Apache-2.0" in out


def test_hat_kernels_appends_procedures(tmp_path):
    from app.agents.runtime import _apply_hat_kernels

    body, ids = _apply_hat_kernels("base body", ["fork.hat.contracts"], tmp_path / "a.md")
    assert "## Procedures" in body
    assert "never states entitlement" in body
    assert "fork.hat.contracts" in ids
