"""Deterministic file pre-dispatch (IFC census gate).

Three live rounds on the same request produced three failure shapes --
refusal (F36), narrated intention (F27), and code-generation returning
zeros -- because K2 rejects forced tool_choice and the model kept
choosing wrong. The runtime now runs the matching file tool ITSELF when
the request literally names a project file of a pre-dispatch type and
the agent holds the tool, injecting the result before the model answers.
Kill-switch: AGENT_FILE_PREDISPATCH=0.
"""
from __future__ import annotations

import pytest

from app.agents import runtime


class _FakeExtractor:
    def __init__(self, result):
        self._result = result
        self.calls = []
        self.param_calls = []

    async def execute(self, input_data, params=None):
        self.calls.append(input_data)
        self.param_calls.append(params)
        return self._result



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

def _wire(monkeypatch, docs, result, tool="bim_extractor"):
    import app.core.projects as projects_mod
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: docs,
                        raising=False)
    fake = _FakeExtractor(result)
    monkeypatch.setattr(runtime, "block_instances", {tool: fake}, raising=False)
    monkeypatch.setattr(
        runtime, "_resolve_file_path",
        lambda pid, raw: "/app/data/e14_" + raw)
    return fake


def _agent(blocks=("bim_extractor", "construction")):
    return runtime.Agent(name="qa", description="", system_prompt="x",
                         allowed_blocks=list(blocks))


@pytest.mark.asyncio
async def test_named_ifc_is_predispatched_and_injected(monkeypatch):
    fake = _wire(monkeypatch, [{"original_name": "qa_building.ifc"}],
                 {"status": "success", "building_elements": [
                     {"ifc_type": "IfcWall"}] * 4})
    msgs = [{"role": "user",
             "content": "Element census of qa_building.ifc: how many walls?"}]
    rec = await runtime._predispatch_file_tool(_agent(), msgs, "p1")
    assert rec and rec["predispatched"] and rec["name"] == "bim_extractor"
    # the tool received the RESOLVED stored path, not the bare name
    assert fake.calls == [{"file_path": "/app/data/e14_qa_building.ifc"}]
    # and the result is folded into the turn as authoritative context
    assert "PLATFORM PRE-DISPATCH" in msgs[-1]["content"]
    assert "IfcWall" in msgs[-1]["content"]


@pytest.mark.asyncio
async def test_no_predispatch_when_file_not_named_or_tool_missing(monkeypatch):
    _wire(monkeypatch, [{"original_name": "qa_building.ifc"}],
          {"status": "success"})
    msgs = [{"role": "user", "content": "what is the retention percentage?"}]
    assert await runtime._predispatch_file_tool(_agent(), msgs, "p1") is None
    msgs = [{"role": "user", "content": "census of qa_building.ifc"}]
    assert await runtime._predispatch_file_tool(
        _agent(blocks=("boq_processor",)), msgs, "p1") is None
    assert await runtime._predispatch_file_tool(_agent(), msgs, None) is None
    assert len(msgs) == 1  # nothing injected on any silent path


@pytest.mark.asyncio
async def test_predispatch_failure_falls_through_to_the_normal_loop(monkeypatch):
    _wire(monkeypatch, [{"original_name": "qa_building.ifc"}],
          {"status": "error", "error": "boom"})
    msgs = [{"role": "user", "content": "census of qa_building.ifc"}]
    assert await runtime._predispatch_file_tool(_agent(), msgs, "p1") is None
    assert len(msgs) == 1


@pytest.mark.asyncio
async def test_kill_switch_disables_predispatch(monkeypatch):
    _wire(monkeypatch, [{"original_name": "qa_building.ifc"}],
          {"status": "success"})
    monkeypatch.setenv("AGENT_FILE_PREDISPATCH", "0")
    msgs = [{"role": "user", "content": "census of qa_building.ifc"}]
    assert await runtime._predispatch_file_tool(_agent(), msgs, "p1") is None


@pytest.mark.asyncio
async def test_named_txt_is_predispatched_via_fetch_document(monkeypatch):
    import app.core.projects as projects_mod
    monkeypatch.setattr(
        projects_mod, "list_documents",
        lambda pid: [{"original_name": "ui_fenwick_grout_spec.txt"}],
        raising=False,
    )

    def fake_fetch(pid, doc_id, filename):
        assert filename == "ui_fenwick_grout_spec.txt"
        return (
            {"text": "TOKEN UI-FENWICK-GROUT-TOKEN-88421", "truncated": False, "source": "file"},
            {"original_name": filename, "id": "d1"},
            None,
        )

    monkeypatch.setattr(runtime, "_fetch_document_content", fake_fetch)
    msgs = [{"role": "user",
             "content": "Find ui_fenwick_grout_spec.txt. Quote the unique token."}]
    rec = await runtime._predispatch_file_tool(_agent(blocks=()), msgs, "p1")
    assert rec and rec["predispatched"] and rec["name"] == "fetch_document"
    assert "UI-FENWICK-GROUT-TOKEN-88421" in msgs[-1]["content"]
    assert "PLATFORM PRE-DISPATCH" in msgs[-1]["content"]


def test_user_names_project_file_matches_timestamped_stem():
    """Leftover L1 names fenwick_waterproofing_spec; the stored file has a
    timestamp suffix. Short stems like 'spec' must not match."""
    named = runtime._user_names_project_file
    stored = "fenwick_waterproofing_spec_20260819-081916.docx"
    assert named("open fenwick_waterproofing_spec_20260819-081916.docx", stored)
    assert named("quote the unique token in fenwick_waterproofing_spec", stored)
    assert not named("what does the spec say about grout", stored)
    assert not named("quote the spec", "spec.docx")


@pytest.mark.asyncio
async def test_named_docx_stem_is_predispatched_via_fetch_document(monkeypatch):
    """Leftover L1: named .docx is fetch_document-predispatched from disk
    even when RAG chunks are empty. Ingest/index is not required."""
    import app.core.projects as projects_mod
    stored = "fenwick_waterproofing_spec_20260819-081916.docx"
    monkeypatch.setattr(
        projects_mod, "list_documents",
        lambda pid: [{"original_name": stored}],
        raising=False,
    )

    def fake_fetch(pid, doc_id, filename):
        assert filename == stored
        return (
            {"text": "TOKEN FENWICK-WP-SPEC-TOKEN-33107", "truncated": False, "source": "extracted"},
            {"original_name": filename, "id": "d-docx"},
            None,
        )

    monkeypatch.setattr(runtime, "_fetch_document_content", fake_fetch)
    msgs = [{"role": "user", "content": (
        "Open fenwick_waterproofing_spec. Quote the unique document token "
        "string from that file."
    )}]
    rec = await runtime._predispatch_file_tool(_agent(blocks=()), msgs, "p1")
    assert rec and rec["predispatched"] and rec["name"] == "fetch_document"
    assert "FENWICK-WP-SPEC-TOKEN-33107" in msgs[-1]["content"]
    assert "PLATFORM PRE-DISPATCH" in msgs[-1]["content"]


@pytest.mark.asyncio
async def test_named_ifc_clash_predispatch_sets_run_clash_detection(monkeypatch):
    """Clash stays off unless the user asks; predispatch then opts in."""
    fake = _wire(
        monkeypatch,
        [{"original_name": "sample_office.ifc"}],
        {"status": "success", "clash_report": {"clashes": []}},
    )
    msgs = [{"role": "user",
             "content": "Run clash detection on sample_office.ifc"}]
    rec = await runtime._predispatch_file_tool(_agent(), msgs, "p1")
    assert rec and rec["predispatched"] and rec["name"] == "bim_extractor"
    assert fake.param_calls == [{"run_clash_detection": True}]
    injected = msgs[-1]["content"]
    assert "clash_detection_ran=true" in injected
    assert injected.find("clash_report") != -1

    fake.calls.clear()
    fake.param_calls.clear()
    msgs = [{"role": "user",
             "content": "Parse sample_office.ifc and report IfcWall count."}]
    rec = await runtime._predispatch_file_tool(_agent(), msgs, "p1")
    assert rec and rec["predispatched"]
    assert fake.param_calls == [{"run_clash_detection": False}]


@pytest.mark.asyncio
async def test_named_ifc_do_not_clash_keeps_detection_off(monkeypatch):
    fake = _wire(
        monkeypatch,
        [{"original_name": "sample_office.ifc"}],
        {"status": "success", "clash_report": {"clashes": []}},
    )
    msgs = [{"role": "user",
             "content": "Parse sample_office.ifc. Do not run clash detection."}]
    rec = await runtime._predispatch_file_tool(_agent(), msgs, "p1")
    assert rec and rec["predispatched"]
    assert fake.param_calls == [{"run_clash_detection": False}]


@pytest.mark.asyncio
async def test_clash_predispatch_unwraps_execute_envelope(monkeypatch):
    """UniversalBlock.execute nests process output under result; the 6k
    inject must still lead with clash_report."""
    import app.core.projects as projects_mod
    monkeypatch.setattr(
        projects_mod, "list_documents",
        lambda pid: [{"original_name": "sample_office.ifc"}],
        raising=False,
    )
    monkeypatch.setattr(
        runtime, "_resolve_file_path",
        lambda pid, raw: "/app/data/" + raw,
    )

    class _Wrapped:
        async def execute(self, input_data, params=None):
            assert params == {"run_clash_detection": True}
            inner = {
                "status": "success",
                "ifc_schema": "IFC4",
                "quantities": {"walls": {"count": 8}},
                "building_elements": [{"ifc_type": "IfcWall"}] * 8,
                "clash_report": {
                    "clashes": [],
                    "detection_method": "aabb_intersection",
                    "detection_method_disclaimer": "bounding-box",
                },
            }
            return {"block": "bim_extractor", "status": "success", "result": inner}

    monkeypatch.setattr(
        runtime, "block_instances", {"bim_extractor": _Wrapped()}, raising=False,
    )
    msgs = [{"role": "user", "content": "Run clash detection on sample_office.ifc"}]
    rec = await runtime._predispatch_file_tool(_agent(), msgs, "p1")
    assert rec and rec["predispatched"]
    injected = msgs[-1]["content"]
    assert "clash_detection_ran=true" in injected
    assert "aabb_intersection" in injected
    assert injected.find("clash_report") < injected.find("building_elements")
