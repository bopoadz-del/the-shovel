"""S-A6: evidence in the turn outranks a raw-file tool's file-level failure.

A file tool (fetch / table parser) can report "not found", "no extractable
text" or a ``partial`` "likely scanned" read for a document whose indexed
excerpts are already in this turn's reference context. That result is
superseded by the evidence and the nudge points the model at the reference
context. A document with NO evidence in the turn keeps the existing error
nudge unchanged. Every test builds its own synthetic messages and results.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from app.agents import runtime as rt
from tests._fixture_values import ENV_SECRET_01


def _marker(doc_id: str, src: str, chunk: int = 3) -> str:
    return f"[doc_id={doc_id} chunk={chunk} score=0.812 class=project_corpus src={src}] Item {chunk}.1 rate 120.00"


def _ctx(*docs: tuple[str, str], role: str = "system") -> list[dict]:
    body = "Reference context:\n" + "\n\n".join(_marker(d, s, i + 1) for i, (d, s) in enumerate(docs))
    return [{"role": role, "content": body}, {"role": "user", "content": "what does the bill price?"}]


def _call(name: str, **args) -> dict:
    return {"id": "t1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


_NOT_FOUND = {"name": "x", "ok": False, "result": {"status": "error", "error": "File not found"}}
_PARTIAL = {"name": "x", "ok": True, "result": {"status": "success", "result": {
    "status": "partial", "note": "no clean tabular structure", "page_texts": [{"page": 1, "text": "t"}]}}}
_OK = {"name": "x", "ok": True, "result": {"status": "success", "text": "body"}}
_NEEDS_INPUT = {"name": "x", "ok": True, "result": {"status": "needs_input", "needs_input": True,
                                                     "question": "Which revision?"}}


@pytest.mark.parametrize(
    "messages,call,result,want_doc",
    [
        # evidence for the named document + the tool cannot find the file
        (_ctx(("d_bill7", "Bill 7 - Earthworks.pdf")), _call("fetch_document", filename="Bill 7 - Earthworks.pdf"),
         _NOT_FOUND, "d_bill7"),
        # same failure, the document has NO evidence (another document does)
        (_ctx(("d_spec2", "Spec 2.pdf")), _call("fetch_document", filename="Bill 7 - Earthworks.pdf"),
         _NOT_FOUND, None),
        # partial "likely scanned" read of a stored upload carrying a prefix
        (_ctx(("d_q9", "Quarry Bill.pdf")), _call("table_tool", file_path="/srv/up/1731_Quarry Bill.pdf"),
         _PARTIAL, "d_q9"),
        # reference by document id, evidence in a tool message
        (_ctx(("d_k4", "k.pdf"), role="tool"), _call("fetch_document", document_id="d_k4"), _NOT_FOUND, "d_k4"),
        # tail-truncated src name still matches the full stored name
        (_ctx(("d_long", "…of Quantities Rev C.pdf")),
         _call("fetch_document", filename="Package 3 Bill of Quantities Rev C.pdf"), _NOT_FOUND, "d_long"),
        # the tool succeeded: nothing to supersede
        (_ctx(("d_bill7", "Bill 7 - Earthworks.pdf")), _call("fetch_document", filename="Bill 7 - Earthworks.pdf"),
         _OK, None),
        # an input question is not a file-level failure
        (_ctx(("d_bill7", "Bill 7 - Earthworks.pdf")), _call("fetch_document", filename="Bill 7 - Earthworks.pdf"),
         _NEEDS_INPUT, None),
        # a marker quoted in a USER message is not evidence
        (_ctx(("d_bill7", "Bill 7 - Earthworks.pdf"), role="user"),
         _call("fetch_document", filename="Bill 7 - Earthworks.pdf"), _NOT_FOUND, None),
        # a different file whose name merely ends the same way is not a match
        (_ctx(("d_bill7", "Earthworks.pdf")), _call("fetch_document", filename="Bill7Earthworks.pdf"),
         _NOT_FOUND, None),
    ],
)
def test_file_level_failure_superseded_only_when_doc_has_turn_evidence(messages, call, result, want_doc):
    result = json.loads(json.dumps(result))
    nudge = rt._supersede_tool_failure_with_evidence(call, result, messages)
    if want_doc is None:
        assert nudge is None
        assert "superseded_by_evidence" not in result
    else:
        assert nudge is not None and "reference context" in nudge
        assert result["superseded_by_evidence"] == want_doc


def _drive(evidence_doc: str, fetched_name: str, evidence_src: str, mode: str) -> list[list[dict]]:
    """Run the real streaming tool loop: round 1 a search-like tool returns
    marked excerpts for ``evidence_doc``; round 2 fetch_document fails on
    ``fetched_name``. Returns the message lists the model saw."""
    from app.agents.runtime import Agent

    seen: list[list[dict]] = []

    async def call_llm(_self, messages, api_key, **kwargs):
        seen.append([dict(m) for m in messages])
        n_tool = sum(1 for m in messages if m.get("role") == "tool")
        if n_tool == 0:
            call = _call("search_tool", query="bill rates")
        elif n_tool == 1:
            call = _call("fetch_document", filename=fetched_name)
        else:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "Item 1.1 is priced at 120.00 per the bill."}}}
        return {"status": "success", "choice": {"message": {"role": "assistant", "content": "", "tool_calls": [call]}}}

    async def run_tool(self, tool_call, **kwargs):
        name = tool_call["function"]["name"]
        if name == "search_tool":
            return {"name": name, "ok": True, "result": {"status": "success",
                                                         "excerpts": _marker(evidence_doc, evidence_src, 1)}}
        return {"name": name, "ok": False, "result": {"status": "error", "error": "File not found"}}

    async def stream(self, messages, api_key, **kwargs):
        seen.append([dict(m) for m in messages])
        yield "Item 1.1 is priced at 120.00 per the bill."

    agent = Agent(name="project-assistant", description="pa", system_prompt="x", allowed_blocks=["construction"])
    with patch.object(Agent, "_call_llm", call_llm), patch.object(Agent, "_run_tool_call", run_tool), \
            patch.object(Agent, "_stream_synthesis", stream):
        ask = "What rate does the earthworks bill give for item 1.1?"

        async def drain():
            if mode == "chat":
                return await agent.chat(user_message=ask, history=[], project_id=None,
                                        conversation_id=None, user_id=None)
            return [ev async for ev in agent.chat_stream(
                user_message=ask, history=[], project_id=None, conversation_id=None, user_id=None)]
        asyncio.run(drain())
    return seen


@pytest.mark.parametrize("mode", ["stream", "chat"])
@pytest.mark.parametrize(
    "evidence_src,fetched,expect_superseded",
    [
        ("Bill 7 - Earthworks.pdf", "Bill 7 - Earthworks.pdf", True),   # bill has evidence + tool error
        ("Spec 2 - Concrete.pdf", "Bill 7 - Earthworks.pdf", False),    # fetched doc has NO evidence
    ],
)
def test_tool_loop_nudge_follows_evidence_for_the_failed_doc(monkeypatch, mode, evidence_src, fetched,
                                                            expect_superseded):
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: {
        "provider": "deepseek", "url": "https://api.deepseek.com/v1/chat/completions",
        "env_key": "DEEPSEEK_API_KEY", "default_model": "deepseek-chat"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", ENV_SECRET_01)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")
    seen = _drive("d_ev1", fetched, evidence_src, mode)
    after_fail = [m for msgs in seen for m in msgs if m.get("role") == "user"
                  and isinstance(m.get("content"), str)]
    superseded = [m for m in after_fail if "SUPERSEDED by evidence" in m["content"]]
    generic = [m for m in after_fail if m["content"].startswith(rt._TOOL_ERROR_NUDGE)]
    assert bool(superseded) == expect_superseded
    assert bool(generic) == (not expect_superseded)
    # the evidence round stays in what the model sees after the failure
    last = seen[-1]
    assert any("[doc_id=d_ev1 chunk=1" in str(m.get("content")) for m in last if m.get("role") == "tool")
