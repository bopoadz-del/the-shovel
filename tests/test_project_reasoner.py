"""Tests for the Project Reasoner — Reasoning Engine Plan 5."""

import pytest

from app.schemas.execution_plan import ExecutionPlan, PlanStep
from tests.conftest import requires_construction_kit


def test_plan_step_defaults():
    s = PlanStep(type="compute_cpm")
    assert s.args == {} and s.output_key == ""


def test_execution_plan_construct():
    plan = ExecutionPlan(
        understanding="user wants the critical path",
        steps=[PlanStep(type="compute_cpm")],
    )
    assert len(plan.steps) == 1
    assert plan.understanding


def test_execution_plan_parses_from_llm_json():
    # The reasoner's PLAN call returns JSON — it must validate straight in.
    raw = {
        "understanding": "compute then compress",
        "steps": [
            {"type": "compute_cpm"},
            {"type": "compress", "args": {"reductions": {"B": 3}}},
        ],
    }
    plan = ExecutionPlan.model_validate(raw)
    assert plan.steps[1].type == "compress"
    assert plan.steps[1].args["reductions"] == {"B": 3}


def test_empty_step_type_rejected():
    with pytest.raises(Exception):
        PlanStep(type="")


from app.core.plan_executor import PlanExecutor
from app.core.session_store import InMemorySessionStore


_ACTIVITIES = [
    {"id": "A", "duration": 3, "predecessors": []},
    {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
    {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "B"}]},
]


async def _session_with_activities():
    s = await InMemorySessionStore().get_or_create("s1")
    s.data["activities"] = _ACTIVITIES
    return s


@pytest.mark.asyncio
async def test_executor_runs_compute_cpm_step():
    session = await _session_with_activities()
    plan = ExecutionPlan(steps=[PlanStep(type="compute_cpm")])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "success"
    assert session.data["cpm_results"]["project_duration"] == 10


@pytest.mark.asyncio
async def test_executor_runs_compress_step():
    session = await _session_with_activities()
    plan = ExecutionPlan(steps=[
        PlanStep(type="compute_cpm"),
        PlanStep(type="compress", args={"reductions": {"B": 3}}),
    ])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "success"
    assert session.data["compressed"]["revised"]["project_duration"] == 7
    assert session.data["compressed"]["days_saved"] == 3


@pytest.mark.asyncio
async def test_executor_runs_gantt_and_histogram_steps():
    session = await _session_with_activities()
    plan = ExecutionPlan(steps=[
        PlanStep(type="compute_cpm"),
        PlanStep(type="gantt"),
        PlanStep(type="resource_histogram", args={"period_unit": "week"}),
    ])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "success"
    assert len(session.data["gantt"]) == 3
    assert session.data["manpower"]["period_unit"] == "week"


@pytest.mark.asyncio
async def test_executor_reports_unknown_step_type():
    session = await _session_with_activities()
    plan = ExecutionPlan(steps=[PlanStep(type="teleport")])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "error"
    assert "teleport" in result.step_results[0].error


@pytest.mark.asyncio
async def test_executor_compute_cpm_without_activities_errors():
    session = await InMemorySessionStore().get_or_create("s1")  # no activities
    plan = ExecutionPlan(steps=[PlanStep(type="compute_cpm")])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "error"


from app.blocks.formula_executor_v2 import FormulaExecutorV2Block


class _MockCodeGen(FormulaExecutorV2Block):
    """Code-gen double — returns canned code, no DeepSeek call."""
    async def _call_llm(self, prompt):
        return "result = a + b"


@pytest.mark.asyncio
async def test_executor_runs_generate_code_step():
    session = await InMemorySessionStore().get_or_create("s1")
    plan = ExecutionPlan(steps=[PlanStep(
        type="generate_code",
        args={"task": "add a and b", "variables": {"a": 4, "b": 6}},
        output_key="sum",
    )])
    executor = PlanExecutor(code_block=_MockCodeGen([]))
    result = await executor.run(plan, session)
    assert result.status == "success"
    assert session.data["sum"]["result"] == 10


@pytest.mark.asyncio
async def test_generate_code_step_requires_a_task():
    session = await InMemorySessionStore().get_or_create("s1")
    plan = ExecutionPlan(steps=[PlanStep(type="generate_code", args={})])
    result = await PlanExecutor(code_block=_MockCodeGen([])).run(plan, session)
    assert result.status == "error"


from app.prompts.reasoner_system import build_reasoner_prompt


@pytest.mark.asyncio
async def test_reasoner_prompt_lists_the_phases():
    p = build_reasoner_prompt(await InMemorySessionStore().get_or_create("s"), "hi")
    for phase in ("UNDERSTAND", "PLAN", "EXECUTE", "DELIVER"):
        assert phase in p


@pytest.mark.asyncio
async def test_reasoner_prompt_advertises_step_types():
    p = build_reasoner_prompt(await InMemorySessionStore().get_or_create("s"), "hi")
    for t in ("compute_cpm", "resource_histogram", "gantt", "compress",
              "generate_code"):
        assert t in p


@pytest.mark.asyncio
async def test_reasoner_prompt_reflects_empty_session():
    p = build_reasoner_prompt(await InMemorySessionStore().get_or_create("s"), "hi")
    assert "no activities" in p.lower() or "not loaded" in p.lower()


@pytest.mark.asyncio
async def test_reasoner_prompt_reflects_loaded_state():
    s = await _session_with_activities()
    s.data["cpm_results"] = {"project_duration": 10}
    p = build_reasoner_prompt(s, "now compress B")
    # the prompt must tell the LLM CPM is already done, so it skips re-running
    assert "cpm" in p.lower()
    assert "now compress B" in p


import json

from app.blocks.project_reasoner import ProjectReasonerBlock
from app.blocks import BLOCK_REGISTRY, get_block


class _MockReasoner(ProjectReasonerBlock):
    """Reasoner double — scripts the PLAN call's JSON and the DELIVER call's
    answer. `_call_llm` is invoked twice per turn: first PLAN, then DELIVER."""

    def __init__(self, plan_json, answer, **kw):
        super().__init__(**kw)
        self._plan_json = plan_json
        self._answer = answer
        self.calls = 0

    async def _call_llm(self, prompt: str) -> str:
        self.calls += 1
        return self._plan_json if self.calls == 1 else self._answer


@pytest.mark.asyncio
async def test_reasoner_runs_full_loop():
    session = await _session_with_activities()
    plan_json = json.dumps({
        "understanding": "user wants the critical path",
        "steps": [{"type": "compute_cpm"}],
    })
    block = _MockReasoner(plan_json, "The critical path is A-B-C, 10 days.")
    out = await block.process({"request": "what is the critical path?",
                               "session": session})
    assert out["status"] == "success"
    assert out["answer"] == "The critical path is A-B-C, 10 days."
    assert out["understanding"] == "user wants the critical path"
    assert session.data["cpm_results"]["project_duration"] == 10
    assert block.calls == 2          # one PLAN call, one DELIVER call


@pytest.mark.asyncio
async def test_reasoner_records_turn_in_history():
    session = await _session_with_activities()
    plan_json = json.dumps({"understanding": "x",
                            "steps": [{"type": "compute_cpm"}]})
    block = _MockReasoner(plan_json, "done")
    await block.process({"request": "go", "session": session})
    roles = [m.role for m in session.history]
    assert roles == ["user", "assistant"]


_PLANNER_LEAKS = (
    "Could not build a plan",
    "no JSON object",
    "Traceback",
    "ExecutionPlan",
)


@pytest.mark.asyncio
async def test_reasoner_degrades_gracefully_without_context():
    """Planner returns unparsable JSON and the project has no indexed context:
    no hard error, a controlled user-facing message, sources=[], no leaked
    planner internals, and no wasted DELIVER call."""
    session = await _session_with_activities()
    block = _MockReasoner("not json at all", "unused")
    out = await block.process({"request": "go", "session": session})

    assert out["status"] != "error"
    assert not out.get("error")
    ans = out["answer"]
    assert ans                                   # user-facing, non-empty
    for leak in _PLANNER_LEAKS:
        assert leak not in ans
    assert out.get("sources") == []
    assert block.calls == 1                       # only the failed PLAN call


@pytest.mark.asyncio
async def test_reasoner_falls_back_to_rag_answer_with_sources(monkeypatch):
    """Planner fails but project docs are available: answer directly from the
    retrieved excerpts and surface their sources."""
    async def fake_search(project_id, query, top_k=5):
        return [
            {"document_id": "d1", "filename": "Spec-A.pdf",
             "snippet": "Concrete grade C40.", "score": 0.81},
            {"document_id": "d2", "filename": "BOQ-B.xlsx",
             "snippet": "250 m3 of concrete.", "score": 0.62},
        ]
    monkeypatch.setattr(
        "app.core.doc_index.search_project_documents", fake_search
    )
    session = await _session_with_activities()
    block = _MockReasoner("prose, no json here",
                          "The concrete grade is C40 (see Spec-A).")
    out = await block.process(
        {"request": "what concrete grade?", "session": session,
         "project_id": "p1"}
    )

    assert out["status"] == "success"
    assert out["answer"] == "The concrete grade is C40 (see Spec-A)."
    srcs = out.get("sources") or []
    assert len(srcs) == 2
    assert srcs[0]["doc_name"] == "Spec-A.pdf"
    for leak in _PLANNER_LEAKS:
        assert leak not in out["answer"]
    assert block.calls == 2                       # failed PLAN + fallback answer


@pytest.mark.asyncio
async def test_fallback_sources_do_not_leak_raw_filesystem_paths(monkeypatch):
    """Excerpt snippets can embed raw Drive/Windows paths from the source
    chunk text; the structured sources must not echo them."""
    async def fake_search(project_id, query, top_k=5):
        return [{
            "document_id": "d1", "filename": "Contract.docx",
            "snippet": r"[source: X:\Example Drive\600-Procurement\QRS-637.docx] text",
            "score": 0.7,
        }]
    monkeypatch.setattr(
        "app.core.doc_index.search_project_documents", fake_search
    )
    session = await _session_with_activities()
    block = _MockReasoner("no json", "Per Contract.docx, …")
    out = await block.process(
        {"request": "summary?", "session": session, "project_id": "p1"}
    )
    src = (out.get("sources") or [])[0]
    assert src["doc_name"] == "Contract.docx"
    blob = json.dumps(out["sources"])
    assert "G:\\" not in blob and ":\\" not in blob


class _DeliverFailReasoner(ProjectReasonerBlock):
    """PLAN call succeeds (valid JSON); the DELIVER call raises — exercises the
    answer-writing failure path."""

    def __init__(self, plan_json, exc=None, deliver_text=None, **kw):
        super().__init__(**kw)
        self._plan_json = plan_json
        self._exc = exc
        self._deliver_text = deliver_text
        self.calls = 0

    async def _call_llm(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return self._plan_json
        if self._exc is not None:
            raise self._exc
        return self._deliver_text


@pytest.mark.asyncio
async def test_deliver_failure_does_not_leak_internal_error():
    """A failed answer-writing call must not surface the raw exception text."""
    session = await _session_with_activities()
    plan = json.dumps({"understanding": "cp", "steps": [{"type": "compute_cpm"}]})
    block = _DeliverFailReasoner(
        plan, exc=RuntimeError("groq 500 boom token=secret-xyz")
    )
    out = await block.process({"request": "critical path?", "session": session})
    ans = out["answer"]
    assert ans                                          # controlled, non-empty
    for leak in ("Could not generate the written answer", "boom",
                 "secret-xyz", "RuntimeError", "Traceback"):
        assert leak not in ans
    assert block.calls == 2


@pytest.mark.asyncio
async def test_deliver_failure_preserves_sources(monkeypatch):
    """When context exists, a DELIVER failure still returns its sources."""
    async def fake_search(project_id, query, top_k=5):
        return [{"document_id": "d1", "filename": "Plan.pdf",
                 "snippet": "x", "score": 0.7}]
    monkeypatch.setattr(
        "app.core.doc_index.search_project_documents", fake_search
    )
    session = await _session_with_activities()
    plan = json.dumps({"understanding": "cp", "steps": [{"type": "compute_cpm"}]})
    block = _DeliverFailReasoner(plan, exc=RuntimeError("deliver down"))
    out = await block.process(
        {"request": "cp?", "session": session, "project_id": "p1"}
    )
    assert "Could not generate the written answer" not in out["answer"]
    srcs = out.get("sources") or []
    assert len(srcs) == 1 and srcs[0]["doc_name"] == "Plan.pdf"


@pytest.mark.asyncio
async def test_deliver_empty_answer_falls_back_controlled():
    """An empty DELIVER answer is treated as a failure, not surfaced blank."""
    session = await _session_with_activities()
    plan = json.dumps({"understanding": "cp", "steps": [{"type": "compute_cpm"}]})
    block = _DeliverFailReasoner(plan, deliver_text="   ")
    out = await block.process({"request": "cp?", "session": session})
    assert out["answer"].strip()                        # non-empty controlled msg
    assert "Could not generate the written answer" not in out["answer"]


@pytest.mark.asyncio
async def test_reasoner_valid_plan_is_unaffected_by_fallback():
    """The graceful-degradation path must not change the happy path: a valid
    plan still executes and returns status success with no fallback sources."""
    session = await _session_with_activities()
    plan_json = json.dumps({"understanding": "critical path",
                            "steps": [{"type": "compute_cpm"}]})
    block = _MockReasoner(plan_json, "Critical path is A-B-C.")
    out = await block.process({"request": "critical path?", "session": session})
    assert out["status"] == "success"
    assert out["answer"] == "Critical path is A-B-C."
    assert out.get("sources", []) == []
    assert block.calls == 2


@pytest.mark.asyncio
async def test_reasoner_reports_step_failure():
    session = await InMemorySessionStore().get_or_create("s1")   # no activities
    plan_json = json.dumps({"understanding": "x",
                            "steps": [{"type": "compute_cpm"}]})
    block = _MockReasoner(plan_json, "unused")
    out = await block.process({"request": "go", "session": session})
    assert out["status"] in ("error", "partial")


@requires_construction_kit
def test_reasoner_is_registered():
    assert "project_reasoner" in BLOCK_REGISTRY
    assert get_block("project_reasoner") is ProjectReasonerBlock


# ── code-review fixes ────────────────────────────────────────────────────


class _CapturingReasoner(_MockReasoner):
    """Like _MockReasoner but records the DELIVER prompt for inspection."""

    async def _call_llm(self, prompt: str) -> str:
        if self.calls == 1:                  # the DELIVER call is the 2nd
            self.deliver_prompt = prompt
        return await super()._call_llm(prompt)


@pytest.mark.asyncio
async def test_deliver_prompt_contains_this_turn_step_output():
    # The DELIVER prompt must be built from this turn's StepResult.output,
    # not from a blunt slice of the whole session blob.
    session = await _session_with_activities()
    plan_json = json.dumps({"understanding": "critical path",
                            "steps": [{"type": "compute_cpm"}]})
    block = _CapturingReasoner(plan_json, "answer")
    await block.process({"request": "critical path?", "session": session})
    prompt = block.deliver_prompt
    assert "STEP RESULTS (from this turn)" in prompt
    assert "compute_cpm" in prompt
    # the actual computed value (project_duration 10) must be in the prompt
    assert "project_duration" in prompt
    assert "10" in prompt
    # and the old whole-session dump phrasing must be gone
    assert "session data" not in prompt.lower()


@pytest.mark.asyncio
async def test_from_session_skips_non_allowlisted_key():
    # generate_code's `from_session` is LLM-controlled — a key outside the
    # allowlist must be silently skipped, not injected into the sandbox.
    session = await InMemorySessionStore().get_or_create("s1")
    session.data["cpm_results"] = {"project_duration": 10}   # allowlisted
    session.data["secret"] = "leak-me"                       # NOT allowlisted

    captured = {}

    class _SpyCodeGen(_MockCodeGen):
        async def process(self, input_data, params=None):
            captured["variables"] = dict(input_data.get("variables") or {})
            return await super().process(input_data, params)

    plan = ExecutionPlan(steps=[PlanStep(
        type="generate_code",
        args={"task": "x", "variables": {"a": 1, "b": 2},
              "from_session": ["cpm_results", "secret"]},
    )])
    await PlanExecutor(code_block=_SpyCodeGen([])).run(plan, session)
    assert "cpm_results" in captured["variables"]
    assert "secret" not in captured["variables"]


def test_extract_json_survives_trailing_prose_with_brace():
    from app.blocks.project_reasoner import _extract_json
    reply = ('{"understanding": "x", "steps": []}\n'
             "Note: this plan uses the {placeholder} convention. Thanks!")
    parsed = _extract_json(reply)
    assert parsed["understanding"] == "x"
    assert parsed["steps"] == []


def test_extract_json_parses_clean_json_directly():
    from app.blocks.project_reasoner import _extract_json
    parsed = _extract_json('{"understanding": "y", "steps": []}')
    assert parsed["understanding"] == "y"


@pytest.mark.asyncio
async def test_reasoner_handles_none_request():
    # request=None must yield the error dict, not an AttributeError.
    session = await _session_with_activities()
    block = _MockReasoner("unused", "unused")
    out = await block.process({"request": None, "session": session})
    assert out["status"] == "error"
    assert "request" in out["error"].lower()


@pytest.mark.asyncio
async def test_step_result_carries_output():
    session = await _session_with_activities()
    plan = ExecutionPlan(steps=[PlanStep(type="compute_cpm")])
    result = await PlanExecutor().run(plan, session)
    assert result.step_results[0].output is not None
    assert result.step_results[0].output["project_duration"] == 10
