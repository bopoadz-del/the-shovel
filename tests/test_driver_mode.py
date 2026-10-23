"""Driver mode (F-DRIVER Phase B, slice 1): off by default; the model picks
the hat, tools come from the package registry, tool_choice is auto, nothing
is retrieved automatically, and the step limit holds. Synthetic data and a
scripted model."""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agents import driver
from app.agents.driver import context, llm, loop, tools
from tests._dev_key import DEV_KEY
from tests._fixture_values import STUB_TOKEN_09


@pytest.mark.parametrize("value,expected", [("", False), ("0", False), ("1", True), ("on", True)])
def test_driver_mode_is_off_unless_switched_on(monkeypatch, value, expected):
    monkeypatch.setenv("DRIVER_MODE", value)
    assert driver.enabled() is expected


_REAL_STREAM = llm.stream  # the streaming call itself, before any fixture replaces it


@pytest.fixture(autouse=True)
def _stream_through_call(monkeypatch):
    """The loop streams (llm.stream); these tests script the model through
    llm.call. Stream whatever llm.call answers, in small chunks."""
    async def fake_stream(agent, messages, offered, model="", timeout=120.0):
        reply = await llm.call(agent, messages, offered, model=model)
        if reply.get("status") == "success":
            text = reply["message"].get("content") or ""
            for i in range(0, len(text), 7):
                yield "content", text[i:i + 7]
        yield "done", reply

    monkeypatch.setattr(llm, "stream", fake_stream)


def _agent():
    from app.agents.runtime import Agent

    return Agent(name="driver-probe", description="probe", system_prompt="", allowed_blocks=[])


def _call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _scripted(replies, seen):
    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        seen.append({"messages": [dict(m) for m in messages],
                     "tools": [t["function"]["name"] for t in offered]})
        return {"status": "success", "message": replies.pop(0)}
    return fake_call


def _no_retrieval(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("driver mode must not retrieve automatically")
    import app.agents.runtime as rt
    import app.core.rag.inject as inject

    monkeypatch.setattr(inject, "rag_inject", boom)
    monkeypatch.setattr(rt, "rag_inject", boom)


def _run(agent, **kw):
    async def go():
        return [e async for e in loop.stream(agent, **kw)]
    return asyncio.run(go())


def test_the_model_picks_the_hat_and_gets_that_hats_tools(monkeypatch):
    _no_retrieval(monkeypatch)
    seen = []
    replies = [
        {"content": "", "tool_calls": [_call("select_hat", {"hat": "qaqc"}, "c1")]},
        {"content": "", "tool_calls": [_call("search_project_documents", {"query": "slump limits"}, "c2")]},
        {"content": "The specified slump is 75 mm (spec section 3)."},
    ]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    agent = _agent()
    ran = []

    async def fake_tool(call, api_key=None, project_id=None, conversation_id=None, **kw):
        ran.append(call["function"]["name"])
        # The real search tool's result shape: passages under result.results.
        return {"ok": True, "result": {"results": [{"text": "Slump: 75 mm, spec section 3.",
                                                     "doc_name": "spec.pdf", "page": 3}]}}

    monkeypatch.setattr(agent, "_run_tool_call", fake_tool)
    events = _run(agent, user_message="What slump does the spec allow?", project_id="synthetic-proj")

    assert events[0] == {"type": "start", "mode": "driver", "agent": "driver-probe"}
    end = events[-1]
    assert end["type"] == "end" and end["mode"] == "driver"
    assert end["hat"] == "qaqc" and end["tools"] == ["search_project_documents"]
    assert ran == ["search_project_documents"]  # select_hat is the driver's own
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "75 mm" in text
    # Before the hat: base tools only. After: the qaqc package's tools too, no other hat's.
    from app.agents.core import tool_registry
    from app.agents.driver import routes

    # The hat's own tools, less any that is a catalogue workflow (offered
    # through run_workflow instead).
    own = sorted(set(tool_registry.tools_of("qaqc")) - set(routes.catalogue()))
    assert not set(own) & set(seen[0]["tools"])
    assert set(own) <= set(seen[1]["tools"])
    assert "cash_flow_forecast" not in seen[1]["tools"]
    assert "select_hat" in seen[0]["tools"] and "search_project_documents" in seen[0]["tools"]
    assert "qaqc hat" in seen[1]["messages"][0]["content"]


def test_an_unknown_hat_is_refused_and_the_turn_continues(monkeypatch):
    _no_retrieval(monkeypatch)
    seen = []
    replies = [
        {"content": "", "tool_calls": [_call("select_hat", {"hat": "astrology"}, "c1")]},
        {"content": "A general answer."},
    ]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    events = _run(_agent(), user_message="hello")
    assert events[-1]["hat"] is None
    refused = json.loads(seen[1]["messages"][-1]["content"])
    assert refused["ok"] is False and "astrology" in refused["error"]


def test_the_step_limit_holds(monkeypatch):
    _no_retrieval(monkeypatch)
    monkeypatch.setenv("DRIVER_MAX_STEPS", "2")
    seen = []
    replies = [{"content": "", "tool_calls": [_call("select_hat", {"hat": "planning"}, f"c{i}")]}
               for i in range(5)]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    events = _run(_agent(), user_message="keep going")
    # Two tool steps, then one closing call that is offered no tools.
    assert len(seen) == 3 and seen[2]["tools"] == []
    assert events[-1]["steps"] == 2
    # The closing reply asked for another tool: nothing to show, so the stop line.
    assert "within the steps" in "".join(e["content"] for e in events if e["type"] == "token")


def test_out_of_steps_the_driver_answers_from_what_it_found(monkeypatch):
    """CORE: a turn that used its steps ended on "could not finish", throwing
    away what its tools returned. Now one tool-free reply answers from it."""
    _no_retrieval(monkeypatch)
    monkeypatch.setenv("DRIVER_MAX_STEPS", "2")
    seen = []
    replies = [{"content": "", "tool_calls": [_call("select_hat", {"hat": "planning"}, f"c{i}")]}
               for i in range(2)] + [{"content": "From what I found: the planning hat applies. "
                                                 "The duration is missing; please give it."}]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    events = _run(_agent(), user_message="keep going")
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "From what I found" in text and "duration is missing" in text
    assert "within the steps" not in text
    assert seen[-1]["tools"] == []
    assert any("No more tool calls" in str(m.get("content")) for m in seen[-1]["messages"])


def test_the_system_message_carries_rules_hats_and_profile():
    msg = context.system_message(None, None, None)["content"]
    assert context.standing_rules() in msg
    for h in context.hats():
        assert f"- {h['discipline']}: " in msg
    assert "No project is open" in msg


def test_the_model_call_lets_the_model_choose(monkeypatch):
    sent = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            sent.update(json)
            return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_09)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    offered = [tools.select_hat_schema(["qaqc", "planning"])]
    out = asyncio.run(llm.call(_agent(), [{"role": "user", "content": "hi"}], offered))
    assert out == {"status": "success", "message": {"content": "ok"}}
    assert sent["tool_choice"] == "auto"
    assert [t["function"]["name"] for t in sent["tools"]] == ["select_hat"]


def _stream(agent, **kw):
    async def go():
        return [e async for e in agent.chat_stream(**kw)]
    return asyncio.run(go())


def test_chat_stream_runs_the_driver_when_driver_mode_is_on(monkeypatch):
    _no_retrieval(monkeypatch)
    monkeypatch.setenv("DRIVER_MODE", "1")
    seen = []
    monkeypatch.setattr(llm, "call", _scripted([{"content": "Driver answer."}], seen))
    agent = _agent()

    async def old_path(*a, **k):
        raise AssertionError("the old path ran in driver mode")
        yield  # pragma: no cover

    monkeypatch.setattr(type(agent), "_chat_stream_impl", old_path)
    events = _stream(agent, user_message="hello")
    assert any(e.get("type") == "start" and e.get("mode") == "driver" for e in events)
    assert "Driver answer." in "".join(e.get("content", "") for e in events if e.get("type") == "token")


def test_chat_stream_keeps_the_old_path_when_driver_mode_is_off(monkeypatch):
    monkeypatch.delenv("DRIVER_MODE", raising=False)
    agent = _agent()

    async def old_path(self, **k):
        yield {"type": "token", "content": "old path"}
        yield {"type": "end", "complete": True}

    async def no_driver(*a, **k):
        raise AssertionError("the driver ran with DRIVER_MODE off")
        yield  # pragma: no cover

    monkeypatch.setattr(type(agent), "_chat_stream_impl", old_path)
    monkeypatch.setattr(loop, "stream", no_driver)
    events = _stream(agent, user_message="hello")
    assert "old path" in "".join(e.get("content", "") for e in events if e.get("type") == "token")


def test_general_knowledge_is_a_tool_the_driver_offers(monkeypatch):
    agent = _agent()
    names = [t["function"]["name"] for t in tools.offered(agent, "synthetic-proj", None, ["qaqc"])]
    assert "search_general_knowledge" in names


def test_search_general_knowledge_searches_the_library_only(monkeypatch):
    from app.agents.core import tool_registry
    from app.core import doc_index
    from app.core.rag import retriever

    asked = []

    async def fake_search(project_id, query, top_k):
        asked.append(project_id)
        return [{"text": f"{project_id} passage", "score": 0.9 if project_id == "gk-b" else 0.4}]

    monkeypatch.setattr(retriever, "_general_knowledge_project_ids", lambda: ["gk-a", "gk-b"])
    monkeypatch.setattr(doc_index, "search_project_documents", fake_search)
    spec = tool_registry.get("search_general_knowledge")
    call = tool_registry.ToolCall(agent=None, name="search_general_knowledge",
                                  args={"query": "retention release", "top_k": 1}, project_id="the-users-project")
    out = asyncio.run(spec.handler(call))
    assert asked == ["gk-a", "gk-b"]  # the library, never the user's project
    assert out["ok"] and [h["project_id"] for h in out["result"]["results"]] == ["gk-b"]
    assert out["result"]["results"][0]["layer"] == "general_knowledge"


def test_the_route_catalogue_is_a_tool_the_model_calls(monkeypatch):
    _no_retrieval(monkeypatch)
    from app.agents.driver import routes
    from app.core import predefined_reasoning

    monkeypatch.setattr(routes, "catalogue", lambda: ["generate_wbs", "resource_histogram"])
    ran = {}

    async def fake_run_workflow(action, ctx, session):
        ran.update(action=action, message=ctx["message"], deliverable=ctx.get("deliverable"))
        return {"handled": True, "answer": "WBS: 3 levels, 12 activities.",
                "exports": [{"label": "WBS (xlsx)", "format": "xlsx"}]}

    monkeypatch.setattr(predefined_reasoning, "run_workflow", fake_run_workflow)
    seen = []
    replies = [
        {"content": "", "tool_calls": [_call("run_workflow", {"workflow": "generate_wbs", "deliverable": True}, "w1")]},
        {"content": "Here is the WBS: 3 levels, 12 activities."},
    ]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    events = _run(_agent(), user_message="build the WBS for a two-storey clinic")
    assert "run_workflow" in seen[0]["tools"]
    assert ran == {"action": "generate_wbs", "message": "build the WBS for a two-storey clinic",
                   "deliverable": True}
    end = events[-1]
    assert end["tools"] == ["run_workflow:generate_wbs"]
    assert end["exports"] == [{"label": "WBS (xlsx)", "format": "xlsx"}]


def test_an_unknown_workflow_is_refused(monkeypatch):
    from app.agents.driver import routes

    monkeypatch.setattr(routes, "catalogue", lambda: ["generate_wbs"])
    out = asyncio.run(routes.run({"workflow": "invent_a_bridge"}, "x", None, None, None))
    assert out["ok"] is False and "invent_a_bridge" in out["error"]


def test_the_exit_check_keeps_backed_figures_and_removes_unbacked_ones():
    msgs = [
        {"role": "user", "content": "What slump does the spec allow?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("search_project_documents", {"query": "slump"}, "c1")]},
        {"role": "tool", "name": "search_project_documents", "tool_call_id": "c1",
         "content": json.dumps({"ok": True, "result": {"results": [{"text": "Slump shall be 75 mm."}]}})},
    ]
    kept, trail, _report = loop.exit_check("The specified slump is 75 mm.", msgs)
    assert "75 mm" in kept
    assert isinstance(trail, list)
    gone, _, report = loop.exit_check("The specified slump is 75 mm. The cube strength is 913 MPa.", msgs)
    assert "75 mm" in gone and "913" not in gone
    assert report["figures_removed"] == ["913 MPa"] and report["passages_read"] == 1


def test_a_driver_answer_passes_the_exit_check_before_it_streams(monkeypatch):
    _no_retrieval(monkeypatch)
    seen = []
    monkeypatch.setattr(llm, "call", _scripted([{"content": "It needs 4,217 bags."}], seen))
    calls = []
    real = loop.exit_check

    def spy(answer, msgs, **kw):
        calls.append(answer)
        return real(answer, msgs, **kw)

    monkeypatch.setattr(loop, "exit_check", spy)
    events = _run(_agent(), user_message="How many bags of cement?")
    assert calls == ["It needs 4,217 bags."]
    assert "provenance" in events[-1]


@pytest.mark.parametrize("mode,header,expected", [
    ("", "on", False), ("request", "", False), ("request", "on", True),
    ("1", "", True), ("request", "nope", False),
])
def test_driver_mode_can_be_asked_for_per_request(monkeypatch, mode, header, expected):
    monkeypatch.setenv("DRIVER_MODE", mode)
    driver.mark_request({driver.REQUEST_HEADER: header} if header else {})
    assert driver.enabled() is expected
    driver.mark_request({})


def test_only_the_request_that_asks_takes_the_driver(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    _no_retrieval(monkeypatch)
    monkeypatch.setenv("DRIVER_MODE", "request")
    seen = []
    monkeypatch.setattr(llm, "call", _scripted([{"content": "From the driver."}], seen))
    body = {"message": "hello", "project_id": None, "history": []}
    h = {"Authorization": "Bearer " + DEV_KEY}
    with TestClient(app) as client:
        asked = client.post("/v1/chat/stream", json=body, headers={**h, "X-Fork-Driver": "on"}).text
    events = [json.loads(line[len("data: "):]) for line in asked.split("\n") if line.startswith("data: ")]
    assert any(e.get("type") == "start" and e.get("mode") == "driver" for e in events)
    assert "From the driver." in "".join(e.get("content", "") for e in events if e.get("type") == "token")
    assert len(seen) == 1


def test_a_request_that_does_not_ask_keeps_the_old_path(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("DRIVER_MODE", "request")

    async def no_driver(*a, **k):
        raise AssertionError("the driver ran for a request that did not ask")
        yield  # pragma: no cover

    monkeypatch.setattr(loop, "stream", no_driver)
    with TestClient(app) as client:
        text = client.post("/v1/chat/stream", json={"message": "hello", "history": []},
                           headers={"Authorization": "Bearer " + DEV_KEY}).text
    assert '"mode": "driver"' not in text


def test_the_exit_check_reads_the_document_searchs_real_result_shape():
    """search_project_documents returns {document_id, filename, chunk, score,
    origin}; figures backed by such a passage stay (live 2026-10-08: they were
    all stripped while the reader looked for 'text')."""
    real = {"ok": True, "result": {"results": [
        {"document_id": "d-17", "filename": "structural spec.pdf", "score": 0.82, "origin": "own",
         "chunk": "Minimum cover to reinforcement in footings cast against earth: 75 mm."}]}}
    msgs = [
        {"role": "user", "content": "What cover does a footing need?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("search_project_documents", {"query": "cover"}, "c1")]},
        {"role": "tool", "name": "search_project_documents", "tool_call_id": "c1", "content": json.dumps(real)},
    ]
    kept, trail, _report = loop.exit_check("Footings cast against earth need 75 mm cover (structural spec).", msgs)
    assert "75 mm" in kept
    assert any(e.get("figure") == "75 mm" for e in trail)


def test_the_exit_check_reads_a_tool_result_whole_even_when_the_model_got_it_cut(monkeypatch):
    _no_retrieval(monkeypatch)
    filler = "x" * (loop._TOOL_RESULT_CHARS + 5000)
    big = {"ok": True, "result": {"results": [
        {"document_id": "d-1", "filename": "a.pdf", "chunk": filler, "score": 0.9},
        {"document_id": "d-2", "filename": "b.pdf", "chunk": "Retention is 5 % of each certificate.", "score": 0.8}]}}
    seen = []
    replies = [
        {"content": "", "tool_calls": [_call("search_project_documents", {"query": "retention"}, "c1")]},
        {"content": "Retention is 5 % of each certificate."},
    ]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    agent = _agent()

    async def fake_tool(call, *a, **k):
        return big

    monkeypatch.setattr(agent, "_run_tool_call", fake_tool)
    events = _run(agent, user_message="What is the retention?", project_id="synthetic-proj")
    assert len(seen[1]["messages"][-1]["content"]) == loop._TOOL_RESULT_CHARS  # the model's copy is cut
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "5 %" in text  # the exit check saw the whole result


def test_the_exit_check_reads_a_snippet_result():
    """Live 2026-10-08, the document search's results are
    {document_id, filename, snippet, score, origin}."""
    live = {"ok": True, "result": {"results": [
        {"document_id": "d-3", "filename": "spec.pdf", "score": 0.7, "origin": "own",
         "snippet": "Concrete cover to reinforcement in footings: 75 mm."}]}}
    msgs = [
        {"role": "user", "content": "What cover does a footing need?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("search_project_documents", {"query": "cover"}, "c1")]},
        {"role": "tool", "name": "search_project_documents", "tool_call_id": "c1", "content": json.dumps(live)},
    ]
    kept, _trail, report = loop.exit_check("Footings need 75 mm cover.", msgs)
    assert "75 mm" in kept and report["passages_read"] == 1


def test_the_driver_reads_the_project_profile_off_the_event_loop(monkeypatch):
    import threading

    _no_retrieval(monkeypatch)
    seen_threads = []

    def spy_profile(project_id, user_id):
        seen_threads.append(threading.current_thread())
        return "Project profile\nname: synthetic"

    monkeypatch.setattr(context, "project_profile", spy_profile)
    monkeypatch.setattr(llm, "call", _scripted([{"content": "ok"}], []))

    async def go():
        loop_thread = threading.current_thread()
        events = [e async for e in loop.stream(_agent(), user_message="hi", project_id="p")]
        return loop_thread, events

    loop_thread, _events = asyncio.run(go())
    assert seen_threads and all(t is not loop_thread for t in seen_threads)



def test_the_answer_streams_before_the_model_has_finished(monkeypatch):
    _no_retrieval(monkeypatch)
    order = []

    async def slow_stream(agent, messages, offered, model="", timeout=120.0):
        for piece in ("The first point stands. ", "The second point too. ", "And a third."):
            order.append(("model", piece))
            yield "content", piece
            await asyncio.sleep(0)
        order.append(("model", "done"))
        yield "done", {"status": "success", "message": {"content": "".join(p for k, p in order if k == "model" and p != "done")}}

    monkeypatch.setattr(llm, "stream", slow_stream)

    async def go():
        async for e in loop.stream(_agent(), user_message="explain"):
            if e["type"] == "token":
                order.append(("token", e["content"]))
    asyncio.run(go())
    first_token = next(i for i, (k, _v) in enumerate(order) if k == "token")
    done = order.index(("model", "done"))
    assert first_token < done  # a sentence reached the user while the model was still writing
    assert "".join(v for k, v in order if k == "token") == "The first point stands. The second point too. And a third."


def test_a_one_sentence_preamble_before_tool_calls_is_not_shown(monkeypatch):
    _no_retrieval(monkeypatch)
    replies = [
        {"content": "Let me look that up. ", "tool_calls": [_call("select_hat", {"hat": "qaqc"}, "c1")]},
        {"content": "Here it is."},
    ]
    monkeypatch.setattr(llm, "call", _scripted(replies, []))
    events = _run(_agent(), user_message="look it up")
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert text == "Here it is."


def test_the_streamed_call_assembles_content_and_tool_calls(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo."}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"search_","arguments":"{\\"q\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"project_documents","arguments":"\\"x\\"}"}}]}}]}',
        "data: [DONE]",
    ]

    class FakeResp:
        status_code = 200

        async def aiter_lines(self):
            for line in lines:
                yield line

    class FakeStreamCtx:
        async def __aenter__(self):
            return FakeResp()

        async def __aexit__(self, *a):
            return False

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, json=None, headers=None):
            assert json["stream"] is True and json.get("tool_choice") == "auto"
            return FakeStreamCtx()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_09)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    async def go():
        return [x async for x in _REAL_STREAM(_agent(), [{"role": "user", "content": "hi"}],
                                            [tools.select_hat_schema(["qaqc"])])]
    out = asyncio.run(go())
    assert out[:2] == [("content", "Hel"), ("content", "lo.")]
    kind, result = out[-1]
    assert kind == "done" and result["status"] == "success"
    call = result["message"]["tool_calls"][0]
    assert call["id"] == "t1" and call["function"]["name"] == "search_project_documents"
    assert json.loads(call["function"]["arguments"]) == {"q": "x"}


def test_a_streamed_answer_carries_the_calculator_credit_once(monkeypatch):
    _no_retrieval(monkeypatch)
    from app.lib import construction_formulas as cf

    calc = {"name": "construction_calc", "ok": True,  # the calculator's real envelope
            "result": cf.run_calculation("concrete_volume", {"length_m": 12, "width_m": 8, "thickness_m": 0.2})}
    replies = [
        {"content": "", "tool_calls": [_call("construction_calc",
                                             {"calculation": "concrete_volume", "length_m": 12, "width_m": 8,
                                              "thickness_m": 0.2}, "k1")]},
        {"content": "Net volume is 19.2 m3. With 5% waste it is 20.16 m3. Check openings before ordering."},
    ]
    monkeypatch.setattr(llm, "call", _scripted(replies, []))
    agent = _agent()

    async def fake_tool(call, *a, **k):
        return calc

    monkeypatch.setattr(agent, "_run_tool_call", fake_tool)
    events = _run(agent, user_message="slab 12 m long, 8 m wide, 200 mm thick", project_id="p")
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "19.2 m3" in text and "20.16 m3" in text
    assert text.count("construction_calc") <= 1  # the credit, at most once, never per sentence
    assert "m3.With" not in text and "m3.Check" not in text


def test_the_end_event_lists_sources_and_the_calculation_credit(monkeypatch):
    _no_retrieval(monkeypatch)
    search = {"name": "search_project_documents", "ok": True, "result": {"results": [
        {"document_id": "d-9", "filename": "Conditions of Contract.pdf", "score": 0.81, "origin": "own",
         "snippet": "Percentage of Retention: 10% of each Interim Payment Certificate."},
        {"document_id": "d-4", "filename": "Spec.pdf", "score": 0.42, "origin": "own", "snippet": "General."}]}}
    ipc = {"name": "payment_certificate", "ok": True,
           "result": {"status": "success", "net_due": 2160000, "retention": 240000, "gross": 2400000}}
    replies = [
        {"content": "", "tool_calls": [_call("search_project_documents", {"query": "retention"}, "s1"),
                                       _call("payment_certificate", {"gross_valuation": 2400000,
                                                                     "retention_percent": 10,
                                                                     "message": "the ask"}, "p1")]},
        {"content": "Net due is 2,160,000 after 10% retention."},
    ]
    monkeypatch.setattr(llm, "call", _scripted(replies, []))
    agent = _agent()

    async def fake_tool(call, *a, **k):
        return search if call["function"]["name"] == "search_project_documents" else ipc

    monkeypatch.setattr(agent, "_run_tool_call", fake_tool)
    events = _run(agent, user_message="gross 2,400,000, retention 10%, net payment?", project_id="p")
    end = events[-1]
    assert [s["doc_name"] for s in end["sources"]][0] == "Conditions of Contract.pdf"
    assert end["sources"][0]["confidence"] == "High" and end["sources"][1]["confidence"] == "Low"
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "Calculated with: Payment certificate (gross valuation 2,400,000, retention percent 10)" in text
    assert "payment_certificate" not in text
    assert "the ask" not in text


def test_a_shared_documents_name_is_shown_as_stored(monkeypatch):
    """The shared layer is cleaned at source; the panel shows its names as
    stored, with the layer label."""
    from app.agents.driver import sources

    trail = [{"role": "tool", "name": "search_general_knowledge", "content": json.dumps(
        {"ok": True, "result": {"results": [{"document_id": "g1", "filename": "general_specification.pdf",
                                              "score": 0.9, "origin": "master_corpus", "snippet": "x"}]}})}]
    out = sources.panel(trail, [])
    assert out and out[0]["doc_name"] == "general_specification.pdf" and out[0]["layer"] == "master_corpus"


def test_streamed_pieces_keep_their_line_breaks(monkeypatch):
    _no_retrieval(monkeypatch)
    answer = "Summary line.\n\n- First point.\n- Second point.\n\nClosing note."
    monkeypatch.setattr(llm, "call", _scripted([{"content": answer}], []))
    events = _run(_agent(), user_message="list it")
    assert "".join(e["content"] for e in events if e["type"] == "token") == answer
