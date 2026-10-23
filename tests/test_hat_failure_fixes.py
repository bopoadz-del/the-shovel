"""Regression guards for the leftover hat-failure list (UI FAIL/PARTIAL).

Each case is a live miss: the model had the right tool or table and still
refused, hung, or got rerouted. These tests pin the *machinery* (prompts,
routing gates, timeouts, synthetic tools, file pre-dispatch) so a later
edit cannot silently reopen the same hole. They do not call a live LLM.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agents import runtime as runtime_module
from app.agents.runtime import (
    ROUTING_GENERALISTS,
    Agent,
    load_agents,
    select_agent_for_message,
)
from app.blocks.mcp_consumer import MCPConsumerBlock, _UNCONFIGURED_MCP_SERVERS
from tests.conftest import requires_construction_kit



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

def _run(coro):
    return asyncio.run(coro)


def _make_agent(name: str, blocks=(), can_delegate: bool = False) -> Agent:
    return Agent(
        name=name,
        description=f"{name} test stub",
        system_prompt="(test stub)",
        allowed_blocks=list(blocks),
        can_delegate=can_delegate,
    )


# ── Prompts: the numbers / rules must be in the kernel, not only RAG ────────


def test_safety_officer_kernel_carries_osha_type_c_slope():
    agent = load_agents()["safety-officer"]
    text = agent.system_prompt
    assert "Type C" in text
    assert "34°" in text or "34°" in text.replace(" ", "")
    assert "1½" in text or "1.5:1" in text or "1.5 : 1" in text
    assert "1926" in text


def test_quantity_surveyor_must_call_drawing_qto_on_pdf():
    agent = load_agents()["quantity-surveyor"]
    text = agent.system_prompt.lower()
    assert "drawing_qto" in text
    assert "pdf" in text
    assert "do not claim the tool is missing" in text or "never say you have no" in text


def test_supervision_proposal_does_not_require_the_md_filename():
    agent = load_agents()["supervision-proposal"]
    text = agent.system_prompt.lower()
    # The structure is inlined, and the prompt says so without naming any
    # corpus file -- a filename in a prompt is a photograph of one corpus.
    assert "already inlined" in text
    assert "do not search the corpus for a structure file" in text
    assert "supervision_proposal_structure" not in text


def test_supervision_proposal_does_not_loop_search_on_contractor_boq():
    """Leftover live FAIL: hat searched leftover BOQ instead of drafting §6."""
    agent = load_agents()["supervision-proposal"]
    text = agent.system_prompt.lower()
    assert "no enquiry pack is not a search loop" in text
    assert "methodology is always written" in text
    assert "at most one" in text and "search_project_documents" in text
    # Distinctive §6 obligations — full words, not bare "hold"/"itp" substrings.
    assert "inspection/wir" in text or "inspection requests/wir" in text or "wir flow" in text
    assert "hold and witness" in text
    assert "itp review" in text
    assert "ncr flow" in text


def test_learning_prompt_treats_invoice_as_a_correction():
    agent = load_agents()["learning"]
    text = agent.system_prompt.lower()
    assert "learning_engine" in text
    assert "invoice" in text
    assert "record_correction" in text


def test_external_mcp_must_not_spawn_weather():
    agent = load_agents()["external-mcp"]
    text = agent.system_prompt.lower()
    assert "weather" in text
    assert "do not call" in text or "do **not** call" in agent.system_prompt.lower()
    assert "npx" in text


def test_construction_pm_must_call_cash_flow_forecast():
    agent = load_agents()["construction-pm"]
    names = {
        (t.get("function") or {}).get("name")
        for t in agent.tool_definitions(project_id="fence")
    }
    assert "cash_flow_forecast" in names
    text = agent.system_prompt.lower()
    assert "cash_flow_forecast" in text
    assert "s-curve" in text


# ── Routing: specialists stay on the addressed hat ──────────────────────────


def test_routing_generalists_match_the_predefined_allowlist():
    from app.routers.agents import _PREDEFINED_GENERALISTS

    assert ROUTING_GENERALISTS is _PREDEFINED_GENERALISTS
    assert ROUTING_GENERALISTS == {
        "heavy-reasoning", "project-assistant", "smart-orchestrator",
    }


@requires_construction_kit
def test_learning_invoice_correction_is_not_stolen_by_heavy(monkeypatch):
    """'the actual invoice was 480 SAR/m3' used to classify as
    payment_certificate and redirect learning → heavy-reasoning."""
    learning = _make_agent("learning", ["learning_engine"])
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["learning"] = learning
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(
            "that's wrong, the actual invoice was 480 SAR/m3 — record the correction",
            learning,
        ))
        assert final is learning
        assert routing["final"] == "learning"
        assert routing["reason"] in (
            "specialist_passthrough", "below_routing_gate", "no-op",
        )
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@requires_construction_kit
def test_quantity_surveyor_qto_is_not_redirected_to_heavy(monkeypatch):
    qs = _make_agent("quantity-surveyor", ["drawing_qto"])
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["quantity-surveyor"] = qs
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(
            "Do a quantity takeoff from the infrastructure drawings",
            qs,
        ))
        assert final is qs
        assert routing["reason"] in (
            "specialist_passthrough", "below_routing_gate", "no-op",
        )
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@requires_construction_kit
def test_project_assistant_still_redirects_generative_intents(monkeypatch):
    pa = _make_agent("project-assistant")
    heavy = _make_agent("heavy-reasoning")
    sc = _make_agent("self-coding")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    runtime_module.AGENT_REGISTRY["self-coding"] = sc
    try:
        final, routing = _run(select_agent_for_message(
            "Create L2 schedule with 200 activities for the data center.",
            pa,
        ))
        assert final.name == "heavy-reasoning"
        assert routing["reason"] == "needs_planning"
        assert final is not sc
    finally:
        runtime_module.AGENT_REGISTRY.clear()


def test_bare_invoice_word_does_not_route_a_rate_correction():
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    result = _run(SmartOrchestratorBlock().process({
        "user_message": "the actual invoice rate was 480 SAR/m3, record that correction",
    }))
    matched = [m["action"] for m in (result.get("matched_actions") or [])]
    assert "payment_certificate" not in matched


# ── MCP: fail fast, never hang on weather npx ───────────────────────────────


@pytest.mark.asyncio
async def test_mcp_consumer_refuses_unconfigured_weather_without_npx():
    block = MCPConsumerBlock()
    result = await block.process({"server": "weather", "tool": "get_forecast"})
    assert result["status"] == "error"
    assert "weather" in result["error"].lower()
    assert "npx" in result["error"].lower() or "not configured" in result["error"].lower()
    assert "weather" in _UNCONFIGURED_MCP_SERVERS


@pytest.mark.asyncio
async def test_mcp_consumer_surfaces_timeout_error(monkeypatch):
    import sys
    import types

    async def _timeout(aw, timeout):
        if asyncio.iscoroutine(aw):
            aw.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("app.blocks.mcp_consumer.asyncio.wait_for", _timeout)

    mcp = types.ModuleType("mcp")

    class ClientSession:
        pass

    class StdioServerParameters:
        def __init__(self, *a, **k):
            pass

    mcp.ClientSession = ClientSession
    mcp.StdioServerParameters = StdioServerParameters
    client = types.ModuleType("mcp.client")
    stdio = types.ModuleType("mcp.client.stdio")
    stdio.stdio_client = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio)

    result = await MCPConsumerBlock().process(
        {"server": "github", "tool": "list_repos"}
    )
    assert result["status"] == "error"
    assert "timed out" in result["error"].lower()


# ── Synthetic cash_flow_forecast tool ───────────────────────────────────────


def _call(agent, name, args):
    return runtime_module.Agent.__dict__["_run_tool_call"](
        agent, {"function": {"name": name, "arguments": json.dumps(args)}},
    )


@pytest.mark.asyncio
async def test_cash_flow_forecast_synthetic_tool_hits_the_container(monkeypatch):
    calls = {}

    class FakeBlock:
        async def cash_flow_forecast(self, input_data, params):
            calls["input"] = input_data
            calls["params"] = params
            return {
                "status": "success",
                "action": "cash_flow_forecast",
                "monthly_forecast": [{"month": 1, "cumulative_percent": 6.67}],
            }

    import app.dependencies as deps
    monkeypatch.setattr(deps, "get_block_instance", lambda name: FakeBlock())
    agent = _make_agent("construction-pm", ["construction"])
    r = await _call(agent, "cash_flow_forecast", {
        "contract_value": "1152650", "duration_months": "12",
    })
    assert r["ok"], r
    assert r["name"] == "cash_flow_forecast"
    assert calls["params"]["contract_value"] == 1152650.0
    assert calls["params"]["duration_months"] == 12


# ── Drawing QTO pre-dispatch ────────────────────────────────────────────────


class _FakeExtractor:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def execute(self, input_data, params):
        self.calls.append(input_data)
        return self._result


def _wire_docs(monkeypatch, docs, result, tool="drawing_qto"):
    import app.core.projects as projects_mod
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: docs,
                        raising=False)
    fake = _FakeExtractor(result)
    monkeypatch.setattr(runtime_module, "block_instances", {tool: fake}, raising=False)
    monkeypatch.setattr(
        runtime_module, "_resolve_file_path",
        lambda pid, raw: "/app/data/" + raw)
    return fake


@pytest.mark.asyncio
async def test_named_dxf_is_predispatched_to_drawing_qto(monkeypatch):
    fake = _wire_docs(
        monkeypatch, [{"original_name": "tower_b.dxf"}],
        {"status": "success", "areas_m2": 1200},
    )
    agent = _make_agent("quantity-surveyor", ["drawing_qto"])
    msgs = [{"role": "user", "content": "QTO of tower_b.dxf please"}]
    rec = await runtime_module._predispatch_file_tool(agent, msgs, "p1")
    assert rec and rec["predispatched"] and rec["name"] == "drawing_qto"
    assert fake.calls == [{"file_path": "/app/data/tower_b.dxf"}]
    assert "PLATFORM PRE-DISPATCH" in msgs[-1]["content"]


@pytest.mark.asyncio
async def test_named_pdf_qto_is_predispatched(monkeypatch):
    fake = _wire_docs(
        monkeypatch, [{"original_name": "drawing_qz_0101.pdf"}],
        {"status": "success", "areas_m2": 88},
    )
    agent = _make_agent("quantity-surveyor", ["drawing_qto"])
    msgs = [{"role": "user",
             "content": "quantity takeoff from drawing_qz_0101.pdf"}]
    rec = await runtime_module._predispatch_file_tool(agent, msgs, "p1")
    assert rec and rec["name"] == "drawing_qto"
    assert fake.calls == [{"file_path": "/app/data/drawing_qz_0101.pdf"}]


@pytest.mark.asyncio
async def test_named_pdf_without_qto_intent_is_not_predispatched(monkeypatch):
    _wire_docs(
        monkeypatch, [{"original_name": "contract_vol2.pdf"}],
        {"status": "success"},
    )
    agent = _make_agent("quantity-surveyor", ["drawing_qto"])
    msgs = [{"role": "user", "content": "summarise contract_vol2.pdf"}]
    assert await runtime_module._predispatch_file_tool(agent, msgs, "p1") is None
    assert len(msgs) == 1


# ── Self-coding: one hop when no feature/formula exists ─────────────────────


@requires_construction_kit
def test_pinned_self_coding_is_not_stolen_by_generate_wbs(monkeypatch):
    sc = _make_agent("self-coding", ["formula_executor_v2"])
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["self-coding"] = sc
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(
            "Create L2 schedule with 200 activities for the data center.",
            sc,
        ))
        assert final is sc
        assert routing["reason"] == "specialist_passthrough"
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@requires_construction_kit
def test_unmatched_calc_from_assistant_goes_to_self_coding_once(monkeypatch):
    pa = _make_agent("project-assistant", can_delegate=True)
    sc = _make_agent("self-coding")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["self-coding"] = sc
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(
            "No registered formula exists. Convert 480 SAR/m3 to USD/yd3 "
            "using 3.75 SAR per 1 USD.",
            pa,
        ))
        assert final is sc
        assert routing["reason"] in (
            "self_coding_requested", "no_feature_or_formula",
        )
        assert routing["final"] == "self-coding"
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@requires_construction_kit
def test_named_calculator_stays_on_project_assistant(monkeypatch):
    pa = _make_agent("project-assistant")
    sc = _make_agent("self-coding")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["self-coding"] = sc
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(
            "calculate beam_moment_simple with udl_w_kn_m 10 and span_m 6",
            pa,
        ))
        assert final is pa, routing
        assert routing["final"] != "self-coding"
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@requires_construction_kit
def test_interim_payment_calculator_is_not_stolen_as_ipc(monkeypatch):
    """Live F5: project-assistant API was rerouted to heavy-reasoning and
    predefined intercept ran payment_certificate with empty params."""
    pa = _make_agent("project-assistant")
    sc = _make_agent("self-coding")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["self-coding"] = sc
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(
            "Interim payment certificate gross 750000 with 5 percent retention. "
            "Net payable? Use the registered interim payment calculator.",
            pa,
        ))
        assert final is pa, routing
        assert routing["final"] == "project-assistant"
        assert routing["action"] is None
        assert routing["reason"] == "named_calculator"
        assert runtime_module._message_wants_named_calculator(
            "Interim payment certificate gross 750000 with 5 percent retention."
        )
    finally:
        runtime_module.AGENT_REGISTRY.clear()


def test_ipc_issue_without_figures_is_not_named_calculator():
    """Negative for the F5 named-calculator override: a deliverable IPC
    request with no numbers must not be classified as construction_calc."""
    assert runtime_module._message_wants_named_calculator(
        "Interim payment certificate gross 750000 with 5 percent retention. "
        "Net payable? Use the registered interim payment calculator."
    )
    assert not runtime_module._message_wants_named_calculator(
        "Issue an interim payment certificate from the project's uploaded "
        "contract. Do not invent figures."
    )
    assert not runtime_module._message_wants_named_calculator(
        "Issue the interim payment certificate. Do not invent figures."
    )


@requires_construction_kit
def test_ipc_issue_without_figures_is_not_stolen_as_named_calculator(monkeypatch):
    """project-assistant 'issue the certificate' (no numbers) must keep
    the payment_certificate action so predefined intercept can run (and
    honestly error if contract_value is missing)."""
    pa = _make_agent("project-assistant")
    sc = _make_agent("self-coding")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["self-coding"] = sc
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(
            "Issue an interim payment certificate from the project's uploaded "
            "contract. Do not invent figures.",
            pa,
        ))
        assert routing["reason"] != "named_calculator", routing
        assert routing.get("action") == "payment_certificate", routing
        assert final.name in ("project-assistant", "heavy-reasoning"), routing
    finally:
        runtime_module.AGENT_REGISTRY.clear()


def test_unknown_calc_nudge_is_a_single_self_coding_handoff():
    agent = Agent(
        name="project-assistant",
        description="",
        system_prompt="x",
        allowed_blocks=[],
        can_delegate=True,
    )
    rec = {
        "name": "construction_calc",
        "ok": False,
        "result": {
            "status": "error",
            "error": "Unknown calculation 'mohr_coulomb'.",
        },
    }
    nudge = runtime_module._nudge_for_failed_tool(rec, agent)
    assert "self-coding" in nudge.lower()
    assert "EXACTLY ONCE" in nudge
    sc_agent = _make_agent("self-coding")
    generic = runtime_module._nudge_for_failed_tool(rec, sc_agent)
    assert generic == runtime_module._TOOL_ERROR_NUDGE


def test_self_coding_prompt_caps_formula_executor_at_one_call():
    text = load_agents()["self-coding"].system_prompt.lower()
    assert "exactly once" in text or "exactly ONCE" in load_agents()["self-coding"].system_prompt


@requires_construction_kit
def test_contracts_kernel_computes_pasted_word_numbers():
    text = load_agents()["contracts-manager"].system_prompt.lower()
    assert "four million" in text or "number-words" in text
    assert "do **not** search" in text or "do not search" in text


@requires_construction_kit
def test_contracts_kernel_writes_procedure_deliverables():
    """Leftover live FAIL: RFP chat stopped at 'let me search'."""
    text = load_agents()["contracts-manager"].system_prompt.lower()
    assert "procedure deliverables are written" in text
    assert "never end on" in text


@requires_construction_kit
def test_construction_pm_kernel_writes_wir_deliverables():
    """Leftover live FAIL: inspection_request stopped after search preamble."""
    text = load_agents()["construction-pm"].system_prompt.lower()
    assert "site procedures are drafted in full" in text
    assert "witness" in text
    assert "wir_form" in text


@requires_construction_kit
def test_validation_kernel_passes_claim_not_null_value():
    text = load_agents()["validation"].system_prompt
    assert "Never call `validation_pipeline` with `value: null`" in text
    assert "{claim:" in text
    assert "<function_calls>" in text
    assert "formula_executor_v2" in text


@requires_construction_kit
def test_heavy_kernel_uses_user_rate_and_expression():
    text = load_agents()["heavy-reasoning"].system_prompt.lower()
    assert "user typed" in text or "user's own message" in text
    assert "expression" in text


@requires_construction_kit
def test_bim_kernel_does_not_rerun_extractor_after_predispatch():
    text = load_agents()["bim-analyst"].system_prompt.lower()
    assert "pre-dispatch" in text or "pre-dispatched" in text
    assert "clash detection is off" in text or "clash" in text


@requires_construction_kit
def test_document_ingestion_kernel_requires_next_handoff():
    """Leftover L5: pinned document-ingestion must name the next hat."""
    text = load_agents()["document-ingestion"].system_prompt
    assert "Next:" in text
    assert "bim-analyst" in text
    assert "quantity-surveyor" in text
    assert "contracts-manager" in text
    assert "Never omit this line" in text or "never omit" in text.lower()
