"""S-7 improvisation harness (ORDERS_28OCT_0815Z): an uncovered calculation
yields a labelled-improvisation nudge + a tenant-scoped new-domain-candidate
log row; a non-kernel error does not; the harness guidance reaches every hat.
"""
from __future__ import annotations

from pathlib import Path

from app.agents import runtime
from app.lib import new_domain_candidates as ndc
from tests.conftest import requires_construction_kit


def _calc_agent(name: str = "quantity-surveyor", can_delegate: bool = True) -> runtime.Agent:
    return runtime.Agent(
        name=name, description="t", system_prompt="t",
        allowed_blocks=["construction"], can_delegate=can_delegate,
    )


def test_uncovered_calc_improvises_with_label_and_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rec = {"name": "construction_calc",
           "result": {"status": "error", "error": "Unknown calculation 'psi_value'."}}
    msgs = [{"role": "user", "content": "compute the thermal bridging psi-value for this junction"}]
    nudge = runtime._nudge_for_failed_tool(rec, _calc_agent(), project_id="acme", messages=msgs)
    low = nudge.lower()
    assert "improvised" in low and "no kernel covers" in low
    assert "self-coding" not in low
    rows = ndc.read_new_domain_candidates("acme")
    assert len(rows) == 1
    assert "thermal bridging psi-value" in rows[0]["question"]
    assert rows[0]["tenant"] == "acme"


def test_a_non_kernel_error_does_not_improvise_or_log(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rec = {"name": "construction_calc",
           "result": {"status": "error", "error": "timeout talking to the kernel"}}
    nudge = runtime._nudge_for_failed_tool(rec, _calc_agent(), project_id="acme", messages=[])
    assert "improvised" not in nudge.lower()
    assert nudge == runtime._TOOL_ERROR_NUDGE
    assert ndc.read_new_domain_candidates("acme") == []


def test_two_uncovered_asks_log_two_distinct_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rec = {"name": "construction_calc", "result": {"error": "Unknown calculation 'x'."}}
    runtime._nudge_for_failed_tool(rec, _calc_agent(), project_id="t1",
                                   messages=[{"role": "user", "content": "size a soakaway for 200 m2"}])
    runtime._nudge_for_failed_tool(rec, _calc_agent(), project_id="t1",
                                   messages=[{"role": "user", "content": "psi-value of a parapet"}])
    rows = ndc.read_new_domain_candidates("t1")
    assert [r["question"] for r in rows] == ["size a soakaway for 200 m2", "psi-value of a parapet"]


@requires_construction_kit
def test_improvisation_harness_reaches_every_hat():
    body, hat_ids = runtime._apply_hat_kernels("BASE", "all", Path("x"))
    assert hat_ids, "the construction kit should resolve at least one hat"
    assert "Improvisation harness" in body
    assert "improvised" in body.lower() and "No kernel covers this" in body
