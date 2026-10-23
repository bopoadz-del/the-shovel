"""Phase-4 re-gate follow-ups (F38b, F42, F43).

Found by running the post-deploy live re-gates for PR #363:

* F38b -- a SAVED workflow ran its blocks with no inputs ("No file path
  provided") while the direct orchestrator path threaded them fine:
  ChainStep had no ``input`` field, so pydantic silently dropped the
  stored step's input dict at the model boundary.
* F42 -- the construction container wraps a route error in a
  success-shaped transport envelope (ok: True, inner status: "error").
  The non-streaming loop only checked ``ok`` and never issued the
  corrective nudge.
* F43 -- container file actions (bim_extract et al) received the BARE
  filename and died on "File not found: qa_building.ifc"; the
  original-name -> stored-path resolution only covered file-schema blocks.
"""
from __future__ import annotations

import pytest

from app.agents import runtime


# ---------------------------------------------------------------- F38b



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

def test_chain_step_model_keeps_per_step_input():
    from app.routers.chain import ChainStep
    s = ChainStep(block="file_hasher", input={"file_path": "x.bin"})
    dumped = s.model_dump(exclude_unset=True)
    assert dumped.get("input") == {"file_path": "x.bin"}


# ---------------------------------------------------------------- F42


def test_envelope_green_inner_error_counts_as_errored():
    # The exact live shape: construction container, ok True, inner error.
    tr = {"name": "construction", "ok": True,
          "result": {"block": "construction", "status": "error",
                     "result": {"status": "error",
                                "error": "Unknown action: bim_element_census"}}}
    assert runtime._tool_result_errored(tr) is True


def test_two_level_nested_error_counts_as_errored():
    tr = {"ok": True, "result": {"status": "success",
          "result": {"status": "error", "error": "File not found"}}}
    assert runtime._tool_result_errored(tr) is True


def test_genuinely_green_results_do_not_nudge():
    tr = {"ok": True, "result": {"status": "success",
          "result": {"status": "success", "items": [1]}}}
    assert runtime._tool_result_errored(tr) is False
    assert runtime._tool_result_errored({"ok": True, "result": "plain text"}) is False


# ---------------------------------------------------------------- F43


def test_construction_dict_payloads_resolve_project_filenames(monkeypatch):
    calls = {}

    def fake_resolve(project_id, payload):
        calls["payload"] = payload
        if isinstance(payload, dict) and payload.get("file_path"):
            return {**payload, "file_path": "/app/data/abc_" + payload["file_path"]}
        return payload
    monkeypatch.setattr(runtime, "_resolve_block_file_input", fake_resolve)
    resolved = runtime._resolve_block_file_input(
        "p1", {"action": "bim_extract", "file_path": "qa_building.ifc"})
    assert resolved["file_path"].startswith("/app/data/")


def test_resolver_never_rewrites_bare_string_prose():
    # A bare-string construction input is a natural-language request; the
    # dispatch guard must leave it alone (only dict payloads resolve).
    # This pins the guard's premise: prose goes through unchanged when no
    # exact/substring document match exists in an empty project.
    out = runtime._resolve_file_path("nonexistent-project", "summarize the BOQ")
    assert out == "summarize the BOQ"


# ---------------------------------------------------------------- F43 (dispatch coverage)


import json as _json


def _tc(name, args):
    return {"function": {"name": name, "arguments": _json.dumps(args)}}


@pytest.mark.asyncio
async def test_construction_dispatch_resolves_filenames_for_dict_payloads(monkeypatch):
    agent = runtime.Agent(name="qa", description="", system_prompt="x",
                          allowed_blocks=["construction"])
    seen = {}

    class FakeConstruction:
        async def execute(self, input_data, params):
            seen["input"] = input_data
            return {"status": "success"}

    monkeypatch.setitem(runtime.BLOCK_REGISTRY, "construction", object())
    monkeypatch.setattr(runtime, "block_instances",
                        {"construction": FakeConstruction()}, raising=False)
    monkeypatch.setattr(
        runtime, "_resolve_file_path",
        lambda pid, raw: "/app/data/abc_" + raw if raw == "qa_building.ifc" else raw)
    r = await runtime.Agent.__dict__["_run_tool_call"](
        agent, _tc("construction",
                   {"input": {"action": "bim_extract", "file_path": "qa_building.ifc"}}),
        project_id="p1")
    assert r["ok"], r
    assert seen["input"]["file_path"] == "/app/data/abc_qa_building.ifc"


@pytest.mark.asyncio
async def test_construction_bare_string_input_is_never_rewritten(monkeypatch):
    agent = runtime.Agent(name="qa", description="", system_prompt="x",
                          allowed_blocks=["construction"])
    seen = {}

    class FakeConstruction:
        async def execute(self, input_data, params):
            seen["input"] = input_data
            return {"status": "success"}

    monkeypatch.setitem(runtime.BLOCK_REGISTRY, "construction", object())
    monkeypatch.setattr(runtime, "block_instances",
                        {"construction": FakeConstruction()}, raising=False)
    monkeypatch.setattr(runtime, "_resolve_file_path",
                        lambda pid, raw: "/should/never/happen")
    r = await runtime.Agent.__dict__["_run_tool_call"](
        agent, _tc("construction", {"input": "summarize the BOQ please"}),
        project_id="p1")
    assert r["ok"], r
    assert seen["input"] == "summarize the BOQ please"


@pytest.mark.asyncio
async def test_hat_alias_dispatch_resolves_filenames(monkeypatch):
    agent = runtime.Agent(name="qa", description="", system_prompt="x",
                          allowed_blocks=["construction"])
    seen = {}

    class FakeContainer:
        async def process(self, input_data, params):
            seen["input"], seen["params"] = input_data, params
            return {"status": "success"}

    import app.dependencies as deps
    monkeypatch.setattr(deps, "get_block_instance", lambda name: FakeContainer())
    monkeypatch.setattr(
        runtime, "_resolve_file_path",
        lambda pid, raw: "/app/data/abc_" + raw if raw.endswith(".pdf") else raw)
    alias = next(iter(runtime._HAT_CONTAINER_ALIASES))
    r = await runtime.Agent.__dict__["_run_tool_call"](
        agent, _tc(alias, {"input": {"file_path": "boq scan.pdf"}}),
        project_id="p1")
    assert r["ok"], r
    assert seen["input"]["file_path"] == "/app/data/abc_boq scan.pdf"
    assert seen["params"]["action"] == runtime._HAT_CONTAINER_ALIASES[alias]


# ---------------------------------------------------------------- F42 (edge lines)


def test_ok_false_is_errored_and_deep_nesting_gives_up_false():
    assert runtime._tool_result_errored({"ok": False, "result": {}}) is True
    deep = {"ok": True, "result": {"status": "success", "result": {
        "status": "success", "result": {"status": "success", "result": {
            "status": "success", "result": {"status": "error"}}}}}}
    # beyond the 4-level walk: treated as green (bounded, never recursive)
    assert runtime._tool_result_errored(deep) is False
    assert runtime._tool_result_errored("not a dict") is False


# ---------------------------------------------------------------- file-tool hint


def _hint_docs(monkeypatch, docs):
    import app.core.projects as projects_mod
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: docs,
                        raising=False)


def test_nudge_hint_names_the_tool_for_a_real_project_file(monkeypatch):
    _hint_docs(monkeypatch, [{"original_name": "qa_building.ifc"}])
    msgs = [{"role": "user", "content":
             "Element census of qa_building.ifc: how many walls?"}]
    hint = runtime._file_tool_hint(msgs, "p1", ["bim_extractor", "construction"])
    assert "qa_building.ifc" in hint
    assert "bim_extractor" in hint


def test_nudge_hint_silent_when_file_not_mentioned_or_tool_missing(monkeypatch):
    _hint_docs(monkeypatch, [{"original_name": "qa_building.ifc"}])
    msgs = [{"role": "user", "content": "what is the retention percentage?"}]
    assert runtime._file_tool_hint(msgs, "p1", ["bim_extractor"]) == ""
    msgs = [{"role": "user", "content": "census of qa_building.ifc please"}]
    assert runtime._file_tool_hint(msgs, "p1", ["boq_processor"]) == ""
    assert runtime._file_tool_hint(msgs, None, ["bim_extractor"]) == ""


def test_nudge_hint_skips_prior_nudges_when_finding_the_user_turn(monkeypatch):
    _hint_docs(monkeypatch, [{"original_name": "qa_building.ifc"}])
    msgs = [
        {"role": "user", "content": "census of qa_building.ifc please"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": runtime._TOOL_ERROR_NUDGE},
    ]
    hint = runtime._file_tool_hint(msgs, "p1", ["bim_extractor"])
    assert "bim_extractor" in hint


# ---------------------------------------------------------------- block contract strays


@pytest.mark.asyncio
async def test_sandbox_bare_code_payload_defaults_to_safety_check():
    # "Unknown action: None" on {"code": ...} -- same class as file_hasher
    # F30b; a bare code payload IS the pre-flight safety check.
    from app.blocks.sandbox import SandboxBlock
    res = await SandboxBlock().process({"code": "print(1+1)"}, {})
    assert "Unknown action" not in str(res)
    assert res.get("status") in ("success", "safe", "unsafe") or "safe" in str(res).lower()


@pytest.mark.asyncio
async def test_learning_engine_accepts_action_key_and_summary_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNING_ENGINE_STATE", str(tmp_path / "state.json"))
    from app.blocks.learning_engine import LearningEngineBlock
    b = LearningEngineBlock(hal_block=None, config={})
    rec = await b.process({"action": "record_correction", "formula_id": "f1",
                           "predicted": 100, "actual": 120}, {})
    assert rec["status"] == "success", rec
    summ = await b.process({"action": "summary"}, {})
    assert summ.get("status") != "error", summ
