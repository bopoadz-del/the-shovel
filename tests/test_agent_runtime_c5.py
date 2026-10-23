"""Tests for agent-runtime core upgrades — Phase C5 · Stream C.

Covers:
- Backward compatibility (chat unchanged without new args)
- Project-context injection
- Conversation persistence + prior-history loading
- Inter-agent delegation (tool exposure, invocation, loop + depth guards)
- Document-search synthetic tool
- remember_fact synthetic tool

DeepSeek is always MOCKED — Agent._call_llm is monkeypatched to scripted
responses. No network calls.
"""

import importlib

import pytest

from app.agents import runtime as rt
from app.agents.runtime import Agent, MAX_DELEGATION_DEPTH


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_agent(name="test-agent", allowed_blocks=None, can_delegate=False):
    return Agent(
        name=name,
        description="Test agent for unit tests",
        system_prompt="You are a test agent.",
        allowed_blocks=allowed_blocks or [],
        can_delegate=can_delegate,
    )


def _isolate(tmp_path, monkeypatch):
    """Point DATA_DIR at tmp and reload agent_memory + projects so no test
    touches the real DB. Returns the fresh agent_memory module."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.core.agent_memory as _am
    import app.core.projects as _proj
    import app.core.project_memory as _pm
    importlib.reload(_proj)
    importlib.reload(_pm)
    importlib.reload(_am)
    return _am


def _llm_text(text):
    """A scripted _call_llm coroutine that returns a plain final answer."""
    async def fake(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {"message": {"content": text}},
            "raw": {},
        }
    return fake


def _llm_capture(box, text="ok"):
    """A scripted _call_llm that records the messages it received."""
    async def fake(self, messages, api_key, project_id=None, **kwargs):
        box["messages"] = messages
        box["project_id"] = project_id
        return {
            "status": "success",
            "choice": {"message": {"content": text}},
            "raw": {},
        }
    return fake


# ── 1. backward compatibility ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_unchanged_without_new_args(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(Agent, "_call_llm", _llm_text("plain answer"))
    agent = _make_agent()
    out = await agent.chat("hi", api_key="k")
    assert out["status"] == "success"
    assert out["answer"] == "plain answer"
    # No project / conversation → only the system prompt as system message
    sys_msgs = [m for m in out["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == agent.system_prompt


# ── 2. project context injection ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_injects_project_context(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    import app.core.projects as store
    proj = store.create_project("C5 Context Project")
    pid = proj["id"]
    store.set_fact(pid, "contract_value", "7777777")

    box = {}
    monkeypatch.setattr(Agent, "_call_llm", _llm_capture(box))
    agent = _make_agent()
    # query mentions "contract value" so build_project_context's fact search matches
    await agent.chat("what is the contract value", api_key="k", project_id=pid)

    sys_text = "\n".join(m["content"] for m in box["messages"] if m["role"] == "system")
    assert "7777777" in sys_text


# ── 3. conversation persistence ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_persists_conversation(tmp_path, monkeypatch):
    am = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(Agent, "_call_llm", _llm_text("the answer"))
    agent = _make_agent()
    await agent.chat("first", api_key="k", conversation_id="cv1")

    msgs = am.get_messages("cv1")
    contents = [(m["role"], m["content"]) for m in msgs]
    assert ("user", "first") in contents
    assert ("assistant", "the answer") in contents


# ── 4. loads prior history ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_loads_prior_history(tmp_path, monkeypatch):
    am = _isolate(tmp_path, monkeypatch)
    am.get_or_create_conversation("cv-hist", "test-agent")
    am.append_message("cv-hist", "user", "earlier question")
    am.append_message("cv-hist", "assistant", "earlier answer")

    box = {}
    monkeypatch.setattr(Agent, "_call_llm", _llm_capture(box))
    agent = _make_agent()
    await agent.chat("next", api_key="k", conversation_id="cv-hist")

    all_text = "\n".join(str(m.get("content")) for m in box["messages"])
    assert "earlier question" in all_text
    assert "earlier answer" in all_text


# ── 5. delegation tool exposure ──────────────────────────────────────────────

def test_delegating_agent_exposes_delegate_tool():
    delegating = _make_agent(can_delegate=True)
    names = [t["function"]["name"] for t in delegating.tool_definitions()]
    assert "delegate_to_agent" in names

    plain = _make_agent(can_delegate=False)
    plain_names = [t["function"]["name"] for t in plain.tool_definitions()]
    assert "delegate_to_agent" not in plain_names


# ── 6. delegation invokes the target ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_delegation_invokes_target(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    specialist = _make_agent(name="specialist")
    monkeypatch.setitem(rt.AGENT_REGISTRY, "specialist", specialist)
    monkeypatch.setattr(Agent, "_call_llm", _llm_text("specialist says hello"))

    caller = _make_agent(name="caller", can_delegate=True)
    tool_call = {
        "id": "d1",
        "function": {
            "name": "delegate_to_agent",
            "arguments": '{"agent_name": "specialist", "message": "help me"}',
        },
    }
    result = await caller._run_tool_call(tool_call, api_key="k", _call_stack=["caller"])
    assert result["ok"] is True
    assert result["result"]["agent"] == "specialist"
    assert result["result"]["answer"] == "specialist says hello"


# ── 7. delegation loop blocked ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delegation_loop_blocked(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    specialist = _make_agent(name="specialist")
    monkeypatch.setitem(rt.AGENT_REGISTRY, "specialist", specialist)

    caller = _make_agent(name="caller", can_delegate=True)
    tool_call = {
        "id": "d2",
        "function": {
            "name": "delegate_to_agent",
            "arguments": '{"agent_name": "specialist", "message": "loop"}',
        },
    }
    # specialist already in the call stack → loop
    result = await caller._run_tool_call(
        tool_call, api_key="k", _call_stack=["caller", "specialist"]
    )
    assert result["ok"] is False
    assert "loop" in result["result"]["hint"].lower()


# ── 8. delegation depth capped ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delegation_depth_capped(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    specialist = _make_agent(name="specialist")
    monkeypatch.setitem(rt.AGENT_REGISTRY, "specialist", specialist)

    caller = _make_agent(name="caller", can_delegate=True)
    tool_call = {
        "id": "d3",
        "function": {
            "name": "delegate_to_agent",
            "arguments": '{"agent_name": "specialist", "message": "deep"}',
        },
    }
    result = await caller._run_tool_call(
        tool_call, api_key="k", _depth=MAX_DELEGATION_DEPTH, _call_stack=["caller"]
    )
    assert result["ok"] is False
    assert "depth" in result["result"]["hint"].lower()


# ── 9. search tool present only with project_id ──────────────────────────────

def test_search_tool_present_only_with_project_id():
    agent = _make_agent()
    with_pid = [t["function"]["name"] for t in agent.tool_definitions(project_id="p")]
    assert "search_project_documents" in with_pid

    without = [t["function"]["name"] for t in agent.tool_definitions()]
    assert "search_project_documents" not in without


# ── 10. search tool calls doc_index ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_tool_calls_doc_index(monkeypatch):
    sentinel = [{"document_id": "d1", "filename": "spec.pdf",
                 "snippet": "found it", "score": 0.9}]
    captured = {}
    operator = "Per the project specification, what compaction is required?"

    async def fake_search(project_id, query, top_k=5):
        captured["project_id"] = project_id
        captured["query"] = query
        captured["top_k"] = top_k
        captured["args"] = (project_id, query, top_k)
        return sentinel

    import app.core.doc_index as di
    monkeypatch.setattr(di, "search_project_documents", fake_search)

    agent = _make_agent()
    tool_call = {
        "id": "s1",
        "function": {
            "name": "search_project_documents",
            "arguments": '{"query": "anything"}',
        },
    }
    result = await agent._run_tool_call(
        tool_call, project_id="proj-x", user_message=operator,
    )
    assert result["ok"] is True
    assert result["result"]["results"] == sentinel
    # One search: the model's query. The operator's message is not a second query.
    assert captured["project_id"] == "proj-x"
    assert captured["query"] == "anything"
    assert captured["args"] == ("proj-x", "anything", 5)
    assert operator not in captured["args"]


@pytest.mark.asyncio
async def test_search_tool_requires_project():
    agent = _make_agent()
    tool_call = {
        "id": "s2",
        "function": {
            "name": "search_project_documents",
            "arguments": '{"query": "anything"}',
        },
    }
    result = await agent._run_tool_call(tool_call)  # no project_id
    assert result["ok"] is False
    assert "project" in result["result"]["hint"].lower()


# ── 11. remember_fact persists ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_remember_fact_persists(tmp_path, monkeypatch):
    am = _isolate(tmp_path, monkeypatch)
    agent = _make_agent(name="rememberer")
    tool_call = {
        "id": "r1",
        "function": {
            "name": "remember_fact",
            "arguments": '{"key": "home_city", "value": "Dubai"}',
        },
    }
    result = await agent._run_tool_call(tool_call)
    assert result["ok"] is True

    facts = am.list_agent_facts("rememberer")
    assert any(f["key"] == "home_city" and f["value"] == "Dubai" for f in facts)


def test_remember_fact_tool_always_present():
    agent = _make_agent()
    names = [t["function"]["name"] for t in agent.tool_definitions()]
    assert "remember_fact" in names


# ── 12. chat_stream basic (unchanged without new args) ───────────────────────

@pytest.mark.asyncio
async def test_chat_stream_unchanged_without_new_args(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(Agent, "_call_llm", _llm_text("streamed answer"))
    agent = _make_agent()

    events = []
    async for ev in agent.chat_stream("hi", api_key="k"):
        events.append(ev)

    types = [ev["type"] for ev in events]
    assert types[0] == "start"
    assert "token" in types
    assert types[-1] == "end"

    token_text = "".join(ev["content"] for ev in events if ev["type"] == "token")
    assert token_text == "streamed answer"


# ── 13. chat_stream persists conversation ────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_stream_persists_conversation(tmp_path, monkeypatch):
    am = _isolate(tmp_path, monkeypatch)
    # Answer is >80 chars so chunking is actually exercised
    long_answer = "This is a long streamed answer that definitely exceeds eighty characters in length, ensuring chunks."
    monkeypatch.setattr(Agent, "_call_llm", _llm_text(long_answer))
    agent = _make_agent()

    async for _ in agent.chat_stream("hello", api_key="k", conversation_id="sv1"):
        pass

    msgs = am.get_messages("sv1")
    roles_contents = [(m["role"], m["content"]) for m in msgs]
    assert ("user", "hello") in roles_contents
    assert ("assistant", long_answer) in roles_contents


# ── 14. chat_stream injects project context ──────────────────────────────────

@pytest.mark.asyncio
async def test_chat_stream_injects_project_context(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    import app.core.projects as store
    proj = store.create_project("Stream Context Project")
    pid = proj["id"]
    store.set_fact(pid, "contract_value", "9999999")

    box = {}
    monkeypatch.setattr(Agent, "_call_llm", _llm_capture(box))
    agent = _make_agent()

    async for _ in agent.chat_stream("what is the contract value", api_key="k", project_id=pid):
        pass

    sys_text = "\n".join(m["content"] for m in box["messages"] if m["role"] == "system")
    assert "9999999" in sys_text


# ── 15. delegation propagates project_id ─────────────────────────────────────

@pytest.mark.asyncio
async def test_delegation_propagates_project_id(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    specialist = _make_agent(name="specialist")
    monkeypatch.setitem(rt.AGENT_REGISTRY, "specialist", specialist)

    box = {}
    monkeypatch.setattr(Agent, "_call_llm", _llm_capture(box, text="specialist answer"))
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True
    )

    caller = _make_agent(name="caller", can_delegate=True)
    tool_call = {
        "id": "d4",
        "function": {
            "name": "delegate_to_agent",
            "arguments": '{"agent_name": "specialist", "message": "sub-task"}',
        },
    }
    result = await caller._run_tool_call(
        tool_call, api_key="k", project_id="proj-xyz", _call_stack=["caller"]
    )
    assert result["ok"] is True
    assert box.get("project_id") == "proj-xyz"


# ── 16. forced final answer when tool-iteration cap is hit (chat) ─────────────

@pytest.mark.asyncio
async def test_chat_forces_final_answer_on_cap(tmp_path, monkeypatch):
    """When the LLM always returns tool calls and the 12-iteration cap is hit,
    chat() must make one more tool-free call and return status=='success' with
    the forced answer — NOT the old 'exceeded iterations' error."""
    _isolate(tmp_path, monkeypatch)

    calls = {"n": 0, "with_tools_seen": []}

    async def fake_llm(self, messages, api_key, project_id=None, with_tools=True, **kwargs):
        calls["n"] += 1
        calls["with_tools_seen"].append(with_tools)
        if calls["n"] <= 12:
            # Always return a tool call so the loop never naturally terminates
            return {
                "status": "success",
                "choice": {
                    "message": {
                        "tool_calls": [{
                            "id": f"tc{calls['n']}",
                            "function": {
                                "name": "remember_fact",
                                "arguments": '{"key": "k", "value": "v"}',
                            },
                        }]
                    }
                },
                "raw": {},
            }
        # 13th call: forced final answer (no tools)
        return {
            "status": "success",
            "choice": {"message": {"content": "forced final answer"}},
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    agent = _make_agent()

    out = await agent.chat("hi", api_key="k")

    # Must succeed — not an error
    assert out["status"] == "success", f"Expected success, got: {out}"
    assert out["answer"] == "forced final answer"

    # The 13th call must have been made with tools disabled
    assert calls["n"] == 13, f"Expected 13 LLM calls, got {calls['n']}"
    assert calls["with_tools_seen"][-1] is False, (
        f"Last call should have with_tools=False, got {calls['with_tools_seen'][-1]}"
    )


# ── 17. forced final answer when tool-iteration cap is hit (chat_stream) ──────

@pytest.mark.asyncio
async def test_chat_stream_forces_final_answer_on_cap(tmp_path, monkeypatch):
    """When the LLM always returns tool calls and the 12-iteration cap is hit,
    chat_stream() must yield token event(s) with the forced answer and an end
    event — NOT an error event."""
    am = _isolate(tmp_path, monkeypatch)

    calls = {"n": 0, "with_tools_seen": []}

    async def fake_llm(self, messages, api_key, project_id=None, with_tools=True, **kwargs):
        calls["n"] += 1
        calls["with_tools_seen"].append(with_tools)
        if calls["n"] <= 12:
            return {
                "status": "success",
                "choice": {
                    "message": {
                        "tool_calls": [{
                            "id": f"tc{calls['n']}",
                            "function": {
                                "name": "remember_fact",
                                "arguments": '{"key": "k", "value": "v"}',
                            },
                        }]
                    }
                },
                "raw": {},
            }
        # 13th call: forced final answer (no tools)
        return {
            "status": "success",
            "choice": {"message": {"content": "forced stream answer"}},
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    agent = _make_agent()

    events = []
    async for ev in agent.chat_stream("hello", api_key="k", conversation_id="cv-cap"):
        events.append(ev)

    types = [ev["type"] for ev in events]

    # Must NOT yield an error event
    assert "error" not in types, f"Got error event; all events: {events}"

    # Must yield token(s) with the forced answer
    token_text = "".join(ev["content"] for ev in events if ev["type"] == "token")
    assert token_text == "forced stream answer", f"Token text was: {token_text!r}"

    # Must end cleanly
    assert types[-1] == "end", f"Last event type was {types[-1]!r}"

    # The 13th call must have been made with tools disabled
    assert calls["n"] == 13, f"Expected 13 LLM calls, got {calls['n']}"
    assert calls["with_tools_seen"][-1] is False, (
        f"Last call should have with_tools=False, got {calls['with_tools_seen'][-1]}"
    )

    # Forced answer must be persisted to conversation memory
    msgs = am.get_messages("cv-cap")
    roles_contents = [(m["role"], m["content"]) for m in msgs]
    assert ("assistant", "forced stream answer") in roles_contents


# ── 18. DSML tool-call markup parsing (DeepSeek inline tool calls) ────────────

# The exact garbled markup observed live in the bug report.
_DSML_BUG_CONTENT = (
    "The procurement list is empty — there's no separate procurement document "
    "uploaded. Let me check if there are any other project documents. "
    '<｜｜DSML｜｜tool_calls> <｜｜DSML｜｜invoke name="search_project_documents"> '
    '<｜｜DSML｜｜parameter name="query" string="true">project document list all files'
    '</｜｜DSML｜｜parameter> <｜｜DSML｜｜parameter name="top_k" string="false">5'
    '</｜｜DSML｜｜parameter> </｜｜DSML｜｜invoke> </｜｜DSML｜｜tool_calls>'
)


def test_parse_dsml_tool_calls_extracts_call():
    import json as _json
    cleaned, tool_calls = rt._parse_dsml_tool_calls(_DSML_BUG_CONTENT)

    # No DSML markup left behind
    assert "DSML" not in cleaned
    assert "<" not in cleaned or "DSML" not in cleaned
    # Human-readable text survives
    assert "procurement list is empty" in cleaned

    # One parsed tool call, correct shape
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["type"] == "function"
    assert tc.get("id")
    assert tc["function"]["name"] == "search_project_documents"
    args = _json.loads(tc["function"]["arguments"])
    assert args["query"] == "project document list all files"
    assert str(args["top_k"]) == "5"


def test_parse_dsml_no_markup():
    text = "Just a normal final answer with no markup at all."
    cleaned, tool_calls = rt._parse_dsml_tool_calls(text)
    assert cleaned == text
    assert tool_calls == []


@pytest.mark.asyncio
async def test_chat_handles_dsml_tool_call(tmp_path, monkeypatch):
    """1st LLM call returns a DSML tool-call block in `content` with empty
    structured tool_calls; 2nd call returns a plain final answer. chat() must
    run the tool, continue the loop, and return a DSML-free answer."""
    am = _isolate(tmp_path, monkeypatch)

    calls = {"n": 0}

    async def fake_llm(self, messages, api_key, project_id=None, with_tools=True, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "status": "success",
                "choice": {
                    "message": {
                        "content": (
                            "Let me remember this. "
                            '<｜｜DSML｜｜tool_calls> <｜｜DSML｜｜invoke name="remember_fact"> '
                            '<｜｜DSML｜｜parameter name="key">topic</｜｜DSML｜｜parameter> '
                            '<｜｜DSML｜｜parameter name="value">dsml</｜｜DSML｜｜parameter> '
                            "</｜｜DSML｜｜invoke> </｜｜DSML｜｜tool_calls>"
                        ),
                        "tool_calls": [],
                    }
                },
                "raw": {},
            }
        return {
            "status": "success",
            "choice": {"message": {"content": "Done — final clean answer."}},
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    agent = _make_agent(name="dsml-agent")

    out = await agent.chat("do something", api_key="k")

    assert out["status"] == "success"
    # The loop continued: answer is the 2nd response
    assert out["answer"] == "Done — final clean answer."
    assert "DSML" not in out["answer"]
    # The DSML tool call actually ran
    assert calls["n"] == 2
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["name"] == "remember_fact"
    # The fact was persisted by the parsed tool call
    facts = am.list_agent_facts("dsml-agent")
    assert any(f["key"] == "topic" and f["value"] == "dsml" for f in facts)


@pytest.mark.asyncio
async def test_final_answer_strips_dsml(tmp_path, monkeypatch):
    """A final answer carrying a stray DSML fragment must be stripped before
    being returned to the user."""
    _isolate(tmp_path, monkeypatch)

    async def fake_llm(self, messages, api_key, project_id=None, with_tools=True, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {
                    "content": (
                        "Here is your answer. <｜｜DSML｜｜tool_calls> garbled "
                        "<｜｜DSML｜｜invoke name="
                    )
                }
            },
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    agent = _make_agent()
    out = await agent.chat("hi", api_key="k")
    assert out["status"] == "success"
    assert "DSML" not in out["answer"]
    assert "Here is your answer." in out["answer"]


# ── 19. DSML whole-block strip (the bug fix) ──────────────────────────────────

# Exact markup from the bug report: the inner text "snagging commissioning
# handover days 200 240 procurement 10" was reaching the UI.
_DSML_BUG_WHOLE_BLOCK = (
    'Real answer here. '
    '<｜｜DSML｜｜tool_calls>'
    '<｜｜DSML｜｜invoke name="search_project_documents">'
    '<｜｜DSML｜｜parameter name="query">snagging commissioning handover'
    '</｜｜DSML｜｜parameter>'
    '</｜｜DSML｜｜invoke>'
    '</｜｜DSML｜｜tool_calls>'
)


def test_strip_dsml_removes_whole_block():
    """_strip_dsml must discard everything from the first DSML marker onward,
    including any inner parameter text — not just the tags."""
    result = rt._strip_dsml(_DSML_BUG_WHOLE_BLOCK)
    assert result == "Real answer here."
    assert "DSML" not in result
    # The inner text of the parameter tag must NOT survive
    assert "snagging" not in result


def test_strip_dsml_plain_text():
    """Plain text with no DSML is returned unchanged."""
    text = "Just a plain final answer with no markup."
    assert rt._strip_dsml(text) == text


def test_parse_dsml_cleaned_content_has_no_inner_text():
    """_parse_dsml_tool_calls must return cleaned_content with no DSML and no
    parameter inner text; the parsed tool call itself must still be correct."""
    import json as _json
    cleaned, tool_calls = rt._parse_dsml_tool_calls(_DSML_BUG_WHOLE_BLOCK)

    # No DSML markup in cleaned content
    assert "DSML" not in cleaned
    # The parameter value must NOT be in cleaned content
    assert "snagging" not in cleaned
    # Prose before the DSML block survives
    assert "Real answer here." in cleaned

    # Tool call was still parsed correctly
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["function"]["name"] == "search_project_documents"
    args = _json.loads(tc["function"]["arguments"])
    assert args["query"] == "snagging commissioning handover"


@pytest.mark.asyncio
async def test_chat_empty_after_dsml_strip_forces_answer(tmp_path, monkeypatch):
    """When the first LLM call returns content that is ONLY unparseable DSML
    (nothing before the first marker), chat() must detect the empty stripped
    content, make ONE forced no-tools call, and return that answer rather than
    an empty bubble."""
    _isolate(tmp_path, monkeypatch)

    calls = {"n": 0, "with_tools_seen": []}

    async def fake_llm(self, messages, api_key, project_id=None, with_tools=True, **kwargs):
        calls["n"] += 1
        calls["with_tools_seen"].append(with_tools)
        if calls["n"] == 1:
            # Only DSML, no prose before it — strip will yield empty string.
            # No <｜｜DSML｜｜tool_calls> wrapper, so _parse_dsml_tool_calls
            # finds no structured tool calls either; final_text is empty.
            return {
                "status": "success",
                "choice": {
                    "message": {
                        "content": (
                            "<｜｜DSML｜｜garbled>"
                            "val"
                            "</｜｜DSML｜｜garbled>"
                        ),
                        "tool_calls": [],
                    }
                },
                "raw": {},
            }
        # 2nd call: forced no-tools call — returns a real plain-text answer
        return {
            "status": "success",
            "choice": {"message": {"content": "Here is a real answer."}},
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    agent = _make_agent()

    out = await agent.chat("hi", api_key="k")

    assert out["status"] == "success"
    # Must NOT return empty, raw DSML, or parameter inner text
    assert out["answer"] == "Here is a real answer."
    assert "DSML" not in out["answer"]
    assert "val" not in out["answer"]
    # Exactly two LLM calls: 1 original + 1 forced
    assert calls["n"] == 2
    # The forced call must have been made with tools disabled
    assert calls["with_tools_seen"][1] is False
