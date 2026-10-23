"""Phase-4 full-surface campaign batch fixes (F31-F35).

Each test is new-shape and was red on the pre-fix code:

* F31 -- fasttrack_analyzer let a list-shaped ``reductions`` past its
  truthiness check and crashed deep in compress_schedule with
  "unhashable type: 'dict'" instead of stating the contract.
* F32 -- translate read the target language ONLY from params; a target in
  the input dict (the shape agents naturally produce) was silently ignored
  and the text came back in Spanish while reporting success.
* F33 -- cache_manager counted as a deliverable tool, so a live
  document-ingestion turn that ran search + a cache lookup ended on
  "Let me route this through the boq_processor" -- a narrated future
  dispatch as the final answer -- without force-synthesis firing.
* F34 -- the strict RAG grounding directive refused user-supplied
  corrections ("480 SAR/m3 does not appear in the reference context");
  the learning agent's entire job is to record exactly those figures.
* F35 -- external-mcp listed and chatted as a ghost: its only functional
  block (mcp_consumer) loads solely with CEREBRUM_VIRGIN=false, missing
  blocks are silently skipped at tool-build time, and the agent truthfully
  told the user it had no tools.
"""
from __future__ import annotations

import pytest

from app.agents import runtime


# ---------------------------------------------------------------- F31


@pytest.mark.asyncio
async def test_fasttrack_rejects_list_shaped_reductions_with_contract_error():
    from app.blocks.fasttrack_analyzer import FastTrackAnalyzerBlock
    acts = [
        {"id": "A", "name": "a", "duration": 10, "predecessors": []},
        {"id": "B", "name": "b", "duration": 6,
         "predecessors": [{"predecessor_id": "A"}]},
    ]
    res = await FastTrackAnalyzerBlock().process(
        {"activities": acts, "reductions": [{"B": 4}]}, {})
    assert res["status"] == "error"
    assert "mapping" in res["error"]
    assert "unhashable" not in res["error"]


@pytest.mark.asyncio
async def test_fasttrack_rejects_non_numeric_reduction_values():
    from app.blocks.fasttrack_analyzer import FastTrackAnalyzerBlock
    acts = [{"id": "A", "name": "a", "duration": 10, "predecessors": []}]
    res = await FastTrackAnalyzerBlock().process(
        {"activities": acts, "reductions": {"A": "four"}}, {})
    assert res["status"] == "error"
    assert "non-negative" in res["error"]


# ---------------------------------------------------------------- F32


@pytest.mark.asyncio
async def test_translate_honours_target_language_in_input_dict():
    from app.blocks.translate import TranslateBlock
    res = await TranslateBlock().process(
        {"text": "reinforced concrete slab", "target": "ar"},
        {"provider": "mock"},
    )
    assert res["status"] == "success"
    assert res["target_language"] == "ar"


@pytest.mark.asyncio
async def test_translate_params_still_win_over_input_dict():
    from app.blocks.translate import TranslateBlock
    res = await TranslateBlock().process(
        {"text": "hello", "target": "fr"},
        {"provider": "mock", "target": "de"},
    )
    assert res["status"] == "success"
    assert res["target_language"] == "de"


# ---------------------------------------------------------------- F33


def test_cache_manager_is_not_a_deliverable_tool():
    # A cache hit/miss is infrastructure. If this pin breaks, re-read the
    # live document-ingestion transcript in the F33 ledger entry before
    # deciding the set can shrink: search + cache lookup alone must leave
    # force-synthesis armed.
    assert "cache_manager" in runtime._NON_DELIVERABLE_TOOLS
    assert "search_project_documents" in runtime._NON_DELIVERABLE_TOOLS
    assert "smart_orchestrator" in runtime._NON_DELIVERABLE_TOOLS


# ---------------------------------------------------------------- F34


def _msgs(question: str):
    return [{"role": "user", "content": question}]


def test_correction_turn_directive_makes_user_figures_authoritative():
    msgs = _msgs("The actual supplier invoice came to 480 SAR/m3, "
                 "you predicted 420. Record it.")
    applied = runtime._apply_rag_context(
        msgs, {"role": "system", "content": "CTX: indicative rate 420 SAR/m3"},
        user_data_authoritative=True,
    )
    assert applied
    folded = msgs[-1]["content"]
    assert "AUTHORITATIVE first-party data" in folded
    assert "using ONLY the reference context" not in folded


def test_default_agents_keep_strict_grounding():
    msgs = _msgs("What is the retention percentage?")
    runtime._apply_rag_context(
        msgs, {"role": "system", "content": "CTX: retention is 10%"})
    assert "using ONLY the reference context" in msgs[-1]["content"]


def test_strict_grounding_directive_carves_out_tool_extraction():
    # F36 -- "say you don't have it" suppressed ALL tool calls: bim-analyst
    # refused an IFC element census with zero tool calls because the census
    # wasn't in the retrieved text, while bim_extract sat in its roster.
    # The strict directive must rank tools ABOVE refusal.
    msgs = _msgs("How many walls are in qa_building.ifc?")
    runtime._apply_rag_context(
        msgs, {"role": "system", "content": "CTX: unrelated BOQ text"})
    folded = msgs[-1]["content"]
    assert "CALL THE TOOL" in folded
    assert "neither the context nor your tools" in folded


def test_learning_agent_config_declares_user_data_authoritative():
    runtime.load_agents()
    agent = runtime.AGENT_REGISTRY.get("learning")
    assert agent is not None
    assert agent.user_data_authoritative is True
    # and nobody else silently inherits it
    others = [a.name for a in runtime.AGENT_REGISTRY.values()
              if a.user_data_authoritative and a.name != "learning"]
    assert others == []


# ---------------------------------------------------------------- F35


def _agent(blocks):
    return runtime.Agent(
        name="probe", description="", system_prompt="", allowed_blocks=blocks)


def test_agent_with_no_loaded_functional_blocks_is_unavailable(monkeypatch):
    monkeypatch.setattr(runtime, "BLOCK_REGISTRY", {"cache_manager": object()})
    reason = _agent(["mcp_consumer", "cache_manager"]).unavailable_reason()
    assert reason is not None
    assert "mcp_consumer" in reason
    assert "CEREBRUM_VIRGIN" in reason


def test_agent_with_any_loaded_functional_block_is_available(monkeypatch):
    monkeypatch.setattr(
        runtime, "BLOCK_REGISTRY",
        {"boq_processor": object(), "cache_manager": object()})
    assert _agent(["boq_processor", "mcp_consumer"]).unavailable_reason() is None


def test_prose_agent_with_no_blocks_stays_available(monkeypatch):
    monkeypatch.setattr(runtime, "BLOCK_REGISTRY", {})
    assert _agent([]).unavailable_reason() is None


def test_agent_referenced_blocks_load_on_every_deployment():
    # F35 part 2: surfacing the ghost was NOT the fix ("make everything
    # work is your mission, not surface honesty"). Every block an agent
    # config references must LOAD on every deployment: external-mcp's
    # mcp_consumer (runtime image gets node/npx in the same change),
    # document-ingestion's three drive connectors, self-coding's sandbox
    # pre-flight. Gated behind CEREBRUM_VIRGIN=false they silently
    # vanished from shipped agents' tool rosters.
    from app.blocks import BLOCK_REGISTRY, _GENERIC_BLOCK_SPECS
    generic = {s[0] for s in _GENERIC_BLOCK_SPECS}
    for name in ("mcp_consumer", "local_drive", "google_drive",
                 "onedrive", "sandbox"):
        assert name in generic, name
        assert name in BLOCK_REGISTRY, name


# ---------------------------------------------------------------- F37


_SCOPE_PROSE = (
    "The works comprise a three-storey reinforced concrete office building. "
    "Completion within 14 months. Chiller procurement lead time is 16 weeks."
)


@pytest.mark.asyncio
async def test_scope_extractor_surfaces_what_the_engine_extracted():
    # Wrap-around green: the shim read keys the engine never emits, so a
    # scope-rich brief returned success with EVERY field empty.
    from app.blocks.scope_extractor import ScopeExtractorBlock
    res = await ScopeExtractorBlock().process({"text": _SCOPE_PROSE}, {})
    assert res["status"] == "success"
    leads = res["equipment_lead_times"]
    assert any(s["equipment"].lower() == "chiller" for s in leads)
    # and never the unlabelled ontology defaults masquerading as extracted
    assert all(s.get("source") != "ontology_default" for s in leads)
    assert res["summary"]  # a real computed summary, not ""


@pytest.mark.asyncio
async def test_weeks_lead_times_convert_to_days():
    # "16 weeks" was recorded as lead_time_days=16.
    from app.blocks.scope_extractor import ScopeExtractorBlock
    res = await ScopeExtractorBlock().process({"text": _SCOPE_PROSE}, {})
    chiller = next(s for s in res["equipment_lead_times"]
                   if s["equipment"].lower() == "chiller")
    assert chiller["lead_time_days"] == 112


@pytest.mark.asyncio
async def test_duration_shaped_completion_target_is_extracted():
    # "Completion within 14 months" -- the standard contract phrasing --
    # matched no schedule_target pattern at all.
    from app.blocks.scope_extractor import ScopeExtractorBlock
    res = await ScopeExtractorBlock().process({"text": _SCOPE_PROSE}, {})
    durations = [t for t in res["schedule_targets"] if t.get("duration_days")]
    assert durations, res["schedule_targets"]
    assert durations[0]["duration_days"] == 14 * 30


# ---------------------------------------------------------------- F38


def _chain_block():
    # Platform-wired orchestrator, as app.dependencies builds it live.
    from app.dependencies import _create_block_instance
    from app.blocks.orchestrator import OrchestratorBlock
    return _create_block_instance(OrchestratorBlock)


@pytest.mark.asyncio
async def test_chain_threads_per_step_input_to_the_block(tmp_path):
    # The per-step "input" dict was silently dropped: boq_processor ran
    # with no file_path while the caller had supplied one, then the chain
    # wrapped the failure in a green envelope.
    p = tmp_path / "probe.txt"
    p.write_text("hello chain")
    res = await _chain_block().process(
        {"steps": [{"block": "file_hasher",
                    "input": {"file_path": str(p)}}]}, {})
    assert res["status"] == "success", res
    inner = res["results"][0]["result"]
    body = inner.get("result") if isinstance(inner.get("result"), dict) else inner
    assert body.get("hashes", {}).get("sha256"), res


@pytest.mark.asyncio
async def test_chain_with_a_failed_step_is_not_a_green_envelope():
    res = await _chain_block().process(
        {"steps": [{"block": "file_hasher",
                    "input": {"file_path": "/nope/missing.bin"}}]}, {})
    assert res["status"] == "error"
    assert "file_hasher" in res["error"]
    assert res["failed_steps"] == [0]


def test_agents_list_surfaces_availability():
    # Route-level shape check against the real registry: every listed agent
    # carries the availability fields, and any unavailable one names why.
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routers.agents import require_user
    app.dependency_overrides[require_user] = lambda: {"user_id": "test-user"}
    try:
        client = TestClient(app)
        data = client.get("/v1/agents").json()
        assert data["count"] >= 1
        for a in data["agents"]:
            assert "available" in a
            if not a["available"]:
                assert a["unavailable_reason"]
    finally:
        app.dependency_overrides.pop(require_user, None)


# ---------------------------------------------------------------- F39/F40


@pytest.mark.asyncio
async def test_villa_brief_classifies_as_building_not_data_center():
    # F39: "two-storey villa" fell through the keyword list to the
    # data_center default -- a data-hall WBS silently applied to a house.
    from app.containers.construction import ConstructionContainer
    w = await ConstructionContainer().generate_wbs(
        {}, {"brief": "two-storey villa, 8 month duration", "target_count": 20})
    assert w["project_type"] == "building"


@pytest.mark.asyncio
async def test_manpower_planner_maps_shorthand_crew_keys():
    # F40: "manpower": 8 was silently dropped by the schema and the
    # histogram came back ALL ZERO with status success.
    from app.blocks.manpower_planner import ManpowerPlannerBlock
    res = await ManpowerPlannerBlock().process({"activities": [
        {"id": "A", "name": "excavation", "duration": 10, "manpower": 8},
        {"id": "B", "name": "raft", "duration": 6, "manpower": 12,
         "predecessors": [{"predecessor_id": "A"}]}]}, {})
    assert res["status"] == "success"
    assert res["peak_total"] == 12.0
    assert sum(p["total"] for p in res["periods"]) > 0
    assert "warning" not in res


@pytest.mark.asyncio
async def test_manpower_planner_warns_when_nothing_is_resourced():
    from app.blocks.manpower_planner import ManpowerPlannerBlock
    res = await ManpowerPlannerBlock().process(
        {"activities": [{"id": "A", "name": "x", "duration": 5}]}, {})
    assert res["status"] == "success"
    assert "all zeros" in res.get("warning", "")


# ---------------------------------------------------------------- F41


@pytest.mark.asyncio
async def test_health_report_ignores_never_used_providers():
    # One healthy provider + an unproven fallback with ZERO calls reported
    # the whole system "degraded" -- crying wolf on a healthy deployment.
    from app.infra.monitoring import MonitoringBlock
    b = MonitoringBlock(hal_block=None, config={})
    await b.execute({"action": "record_call", "provider": "deepseek",
                     "latency_ms": 800, "success": True})
    rep = await b.execute({"action": "health_report"})
    assert rep["overall_status"] == "healthy", rep["overall_status"]


@pytest.mark.asyncio
async def test_health_report_unknown_with_no_observations():
    from app.infra.monitoring import MonitoringBlock
    rep = await MonitoringBlock(hal_block=None, config={}).execute(
        {"action": "health_report"})
    assert rep["overall_status"] == "unknown"
