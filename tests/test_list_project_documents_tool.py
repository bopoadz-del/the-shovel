"""list_project_documents agent tool — the document REGISTER, deterministic.

Golden-set failure this closes (pilot_document_metadata, 2026-08-02 live):
"list the documents in this project and what type each one is" had NO tool
that could answer it — search_project_documents can only surface content
chunks, and no chunk IS the register — so the model honestly reported the
context held no document list. The new tool reads the project store
directly.

Shapes: planted-truth (a register the search path could never see must come
back), alias-resolution (the pilot's master-corpus project lists its backing
corpus), and outage (no project in scope -> controlled error, not a crash).
"""
import pytest

from app.agents.runtime import Agent


def _tool_call(name, args=None):
    import json
    return {"function": {"name": name, "arguments": json.dumps(args or {})}}


@pytest.fixture()
def runtime():
    return Agent(
        name="test-agent",
        description="Test agent for unit tests",
        system_prompt="You are a test agent.",
        allowed_blocks=[],
    )


@pytest.mark.asyncio
async def test_register_returned_deterministically(runtime, tmp_path, monkeypatch):
    from app.core import projects as store
    monkeypatch.setenv("FORK_DATA_DIR", str(tmp_path))
    p = store.create_project("Register Test")
    pid = p["id"]
    store.add_document(pid, "spec_vol2.pdf", "s_spec.pdf", str(tmp_path / "a"), 10)
    store.add_document(pid, "boq_final.xlsx", "s_boq.xlsx", str(tmp_path / "b"), 20)

    out = await runtime._run_tool_call(_tool_call("list_project_documents"),
                                       project_id=pid)
    assert out["ok"] is True
    res = out["result"]
    assert res["total_documents"] == 2
    names = {d["filename"] for d in res["documents"]}
    assert names == {"spec_vol2.pdf", "boq_final.xlsx"}
    # every row carries the fields the answer needs
    for d in res["documents"]:
        assert "doc_type" in d and "size_bytes" in d


@pytest.mark.asyncio
async def test_master_corpus_alias_lists_backing_corpus(runtime, tmp_path, monkeypatch):
    from app.core import projects as store
    monkeypatch.setenv("FORK_DATA_DIR", str(tmp_path))
    backing = store.create_project("Backing")
    store.add_document(backing["id"], "drawing_a101.pdf", "s_a101.pdf",
                       str(tmp_path / "c"), 30)
    monkeypatch.setattr(store, "MASTER_CORPUS_PROJECT_ID", "master_alias")
    monkeypatch.setattr(store, "MASTER_CORPUS_SOURCE_PROJECT_ID", backing["id"])

    out = await runtime._run_tool_call(_tool_call("list_project_documents"),
                                       project_id="master_alias")
    assert out["ok"] is True
    assert out["result"]["total_documents"] == 1
    assert out["result"]["documents"][0]["filename"] == "drawing_a101.pdf"


@pytest.mark.asyncio
async def test_no_project_scope_controlled_error(runtime):
    out = await runtime._run_tool_call(_tool_call("list_project_documents"),
                                       project_id=None)
    assert out["ok"] is False
    assert out["result"]["status"] == "error"


@pytest.mark.asyncio
async def test_limit_clamped(runtime, tmp_path, monkeypatch):
    from app.core import projects as store
    monkeypatch.setenv("FORK_DATA_DIR", str(tmp_path))
    p = store.create_project("Clamp")
    store.add_document(p["id"], "one.pdf", "s1.pdf", str(tmp_path / "d"), 1)
    out = await runtime._run_tool_call(
        _tool_call("list_project_documents", {"limit": "not-a-number"}),
        project_id=p["id"])
    assert out["ok"] is True   # bad limit degrades to default, never errors
    out = await runtime._run_tool_call(
        _tool_call("list_project_documents", {"limit": 99999}),
        project_id=p["id"])
    assert out["ok"] is True
    assert out["result"]["returned"] <= 200


def test_tool_offered_only_with_project_scope(runtime):
    with_project = runtime.tool_definitions(project_id="p1")
    names = {t["function"]["name"] for t in with_project}
    assert "list_project_documents" in names
    without = runtime.tool_definitions(project_id=None)
    names2 = {t["function"]["name"] for t in without}
    assert "list_project_documents" not in names2
