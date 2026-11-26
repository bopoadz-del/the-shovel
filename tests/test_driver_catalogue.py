"""The catalogue (DESIGN V2 item 2): one entry in one format for every
registered formula, tool, workflow, skill and knowledge file; a short index
in a fixed order; a lookup that returns full entries for named ids only;
the optional declarations (when_to_use, when_not, examples) with their gaps.
Every formula and tool registered here is synthetic and removed afterwards."""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from app.agents.core import tool_registry
from app.agents.driver import catalogue, context, llm, loop, tools
from app.lib import formula_registry as fr
from app.lib.formula_registry import Param
from tests._dev_key import DEV_KEY

FIELDS = {"id", "kind", "name", "for", "inputs", "outputs", "when_to_use", "when_not",
          "standard", "examples", "feeds", "fed_by", "code", "source", "stamps", "gaps"}


@pytest.fixture
def registered():
    """Register synthetic formulas / tools for one test; remove them after."""
    names = {"formula": [], "tool": []}
    yield names
    for name in names["formula"]:
        fr._REGISTRY.pop(name, None)
    for name in names["tool"]:
        tool_registry._TOOLS.pop(name, None)
    catalogue._CACHE.clear()


def _formula(registered, name, inputs, outputs, **declared):
    import inspect

    def fn(**kw):  # the body is never run here
        return {}
    fn.__name__ = name
    # parameters() reads the inputs' type and required-ness from the signature.
    fn.__signature__ = inspect.Signature([inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, annotation=float)
                                          for k in inputs])
    fr.formula(owner="base", description=f"Synthetic {name}.", display_name=f"Synthetic {name}",
               inputs=inputs, outputs=outputs, **declared)(fn)
    registered["formula"].append(name)
    return fn


def _example(**over):
    out = {"invented": True, "given": "a 3 m by 4 m bay", "chain": ["formula:x(a=3, b=4)"],
           "result": "area 12 m2"}
    out.update(over)
    return out


# -- the whole registry, one format -------------------------------------------------

def test_every_registered_thing_has_one_entry():
    from app.agents import catalog
    from app.agents.driver import routes

    ids = [e["id"] for e in catalogue.entries()]
    assert len(ids) == len(set(ids))
    want = {f"formula:{s.name}" for s in fr.all_specs()}
    tool_registry.load()
    want |= {f"tool:{n}" for n in tool_registry._TOOLS}
    want |= {f"workflow:{n}" for n in routes.catalogue()}
    want |= {f"skill:{h.discipline.value}.{p.id}" for h in catalog.list_hats()
             for p in (h.playbook.procedures or [] if h.playbook else [])}
    root = catalogue.knowledge_dir()
    want |= {f"knowledge:{f[:-3]}" for f in os.listdir(root) if f.endswith(".md")}
    assert set(ids) == want
    # At least two kinds are populated, so the check above is not one registry.
    assert len({i.split(":")[0] for i in ids}) >= 2


def test_every_entry_has_the_one_format():
    for e in catalogue.entries():
        assert set(e) - {"needs"} == FIELDS, e["id"]
        assert e["kind"] in catalogue.KINDS
        assert e["id"].startswith(e["kind"] + ":")
        assert set(e["stamps"]) == {"verified", "tested", "approved"}
        # Nothing here stamps verified or approved: those are a tester's and the owner's.
        assert e["stamps"]["verified"] is None and e["stamps"]["approved"] is None
        assert e["source"].get("registry") == e["kind"]
        assert all(ex.get("invented") is True for ex in e["examples"])


def test_a_formula_entry_carries_its_declared_inputs(registered):
    for name, unit, hi in (("zz_cat_alpha", "m", 50.0), ("zz_cat_beta", "kN", 900.0)):
        _formula(registered, name, {"size": Param(unit, 0, hi, label="the size", required=True)},
                 {"out": unit})
        catalogue._CACHE.clear()
        entry = next(e for e in catalogue.entries() if e["id"] == f"formula:{name}")
        assert entry["inputs"][0]["unit"] == unit and entry["inputs"][0]["range"] == [0, hi]
        assert entry["inputs"][0]["label"] == "the size"
        assert entry["code"] == {"tool": "construction_calc", "calculation": name}
        assert entry["for"] == f"Synthetic {name}."
        assert entry["source"]["file"].endswith("test_driver_catalogue.py")


# -- declarations and gaps ----------------------------------------------------------

def test_declared_fields_fill_the_entry_and_missing_ones_are_gaps(registered):
    _formula(registered, "zz_cat_declared", {"a": "m"}, {"b": "m"}, when_to_use="a bay is rectangular",
             when_not="the bay is curved; use a polygon area instead", examples=(_example(),),
             standard="a published method")
    _formula(registered, "zz_cat_bare", {"a": "m"}, {"b": "m"})
    catalogue._CACHE.clear()
    held = {e["id"]: e for e in catalogue.entries()}
    full, bare = held["formula:zz_cat_declared"], held["formula:zz_cat_bare"]
    assert full["when_to_use"] == "a bay is rectangular"
    assert full["when_not"].startswith("the bay is curved")
    assert full["examples"] == [_example()]
    assert full["gaps"] == []
    assert bare["gaps"] == ["when_to_use", "when_not", "examples", "standard"]


def test_one_declaration_missing_is_that_one_gap(registered):
    _formula(registered, "zz_cat_no_when_not", {"a": "m"}, {"b": "m"}, when_to_use="x",
             examples=(_example(),), standard="s")
    _formula(registered, "zz_cat_no_examples", {"a": "m"}, {"b": "m"}, when_to_use="x",
             when_not="y", standard="s")
    catalogue._CACHE.clear()
    held = {e["id"]: e for e in catalogue.entries()}
    assert held["formula:zz_cat_no_when_not"]["gaps"] == ["when_not"]
    assert held["formula:zz_cat_no_examples"]["gaps"] == ["examples"]


@pytest.mark.parametrize("bad", [
    _example(invented=False),
    {k: v for k, v in _example().items() if k != "invented"},
    _example(result=""),
    _example(given=None),
    _example(chain="not a list"),
    _example(figure_from="a document"),
    "an example as text",
])
def test_an_example_that_is_not_invented_and_whole_is_refused(bad):
    examples = bad if isinstance(bad, str) else (bad,)
    with pytest.raises(ValueError):
        fr.declared_examples("formula", "zz", examples)
    with pytest.raises(ValueError):
        tool_registry.tool("zz_cat_tool_refused", owner="base", display_name="Z", examples=examples)


def test_a_whole_example_is_kept_and_its_chain_is_optional():
    kept = fr.declared_examples("formula", "zz", (_example(), {"invented": True, "given": " g ", "result": "r"}))
    assert kept[0] == _example()
    assert kept[1] == {"invented": True, "given": "g", "chain": [], "result": "r"}
    assert fr.declared_examples("formula", "zz", ()) == ()


def test_a_tool_declares_the_same_fields(registered):
    async def handle(call):
        return {"ok": True}

    tool_registry.tool("zz_cat_tool", owner="base", display_name="Synthetic tool",
                       when_to_use="w", when_not="n", examples=(_example(),),
                       params={"q": Param(label="q", kind="text", required=True)})(handle)
    registered["tool"].append("zz_cat_tool")
    catalogue._CACHE.clear()
    entry = next(e for e in catalogue.entries() if e["id"] == "tool:zz_cat_tool")
    assert (entry["when_to_use"], entry["when_not"]) == ("w", "n")
    # A tool with no schema has no description: its "for" is a gap.
    assert entry["gaps"] == ["for"]


# -- feeds / fed_by -----------------------------------------------------------------

def test_an_output_feeds_an_input_of_the_same_name_and_unit_only(registered):
    _formula(registered, "zz_cat_src", {"a": "m"}, {"zz_area": "m2"})
    _formula(registered, "zz_cat_dst", {"zz_area": "m2"}, {"c": "m3"})
    _formula(registered, "zz_cat_other_unit", {"zz_area": "ha"}, {"c": "m3"})
    catalogue._CACHE.clear()
    held = {e["id"]: e for e in catalogue.entries()}
    assert held["formula:zz_cat_src"]["feeds"] == ["formula:zz_cat_dst"]
    assert held["formula:zz_cat_dst"]["fed_by"] == ["formula:zz_cat_src"]
    assert held["formula:zz_cat_other_unit"]["fed_by"] == []
    assert held["formula:zz_cat_dst"]["feeds"] == []


# -- the index ----------------------------------------------------------------------

def test_the_index_is_one_line_per_entry_in_a_fixed_order():
    items = catalogue.entries()
    lines = catalogue.index_lines()
    assert len(lines) == len(items)
    assert all("\n" not in line and line.count(" | ") >= 3 for line in lines)
    keys = [(catalogue.KINDS.index(line.split(" | ")[1]), line.split(" | ")[0]) for line in lines]
    assert keys == sorted(keys)
    assert catalogue.index_lines() == lines


def test_the_index_order_does_not_follow_the_order_given():
    a = {"id": "tool:b", "kind": "tool", "name": "B", "for": "b"}
    b = {"id": "formula:z", "kind": "formula", "name": "Z", "for": "z"}
    c = {"id": "formula:a", "kind": "formula", "name": "A", "for": ""}
    assert catalogue.index_lines([a, b, c]) == catalogue.index_lines([c, a, b]) == [
        "formula:a | formula | A | -", "formula:z | formula | Z | z", "tool:b | tool | B | b"]


def test_a_long_for_line_is_cut_to_one_line():
    line = catalogue.index_line({"id": "formula:x", "kind": "formula", "name": "X",
                                 "for": catalogue._one_line("word\nword " * 100)})
    assert "\n" not in line and len(line.split(" | ")[3]) == catalogue.FOR_CHARS
    exact = "w" * catalogue.FOR_CHARS
    assert catalogue._one_line(exact) == exact
    assert catalogue._one_line(exact + "w").endswith("…")


def test_a_registered_formula_reaches_the_index_without_a_restart(registered):
    before = catalogue.index_lines()
    _formula(registered, "zz_cat_late", {"a": "m"}, {"b": "m"})
    after = catalogue.index_lines()
    assert len(after) == len(before) + 1
    assert any(line.startswith("formula:zz_cat_late | formula | ") for line in after)


# -- lookup -------------------------------------------------------------------------

def test_lookup_returns_the_named_entries_in_the_order_asked():
    ids = [e["id"] for e in catalogue.entries()]
    first, last = ids[0], ids[-1]
    for asked in ([first, last], [last, first]):
        out = catalogue.lookup(asked)
        assert out["ok"] is True and [e["id"] for e in out["entries"]] == asked
        assert out["unknown"] == [] and "not_returned" not in out


def test_lookup_names_unknown_ids_and_never_returns_everything():
    known = catalogue.entries()[1]["id"]
    out = catalogue.lookup([known, "formula:no_such_thing", known])
    assert [e["id"] for e in out["entries"]] == [known]
    assert out["unknown"] == ["formula:no_such_thing"]
    assert catalogue.lookup(known)["entries"][0]["id"] == known  # one id as text
    for nothing in ([], None, 7, [""], ["formula:no_such_thing"], {"ids": [known]}):
        out = catalogue.lookup(nothing)
        assert out["ok"] is False and out.get("entries", []) == [] and out["error"]


def test_lookup_stops_at_the_entry_and_character_limits(monkeypatch):
    ids = [e["id"] for e in catalogue.entries()]
    many = catalogue.lookup(ids[:catalogue.MAX_LOOKUP + 3])
    assert len(many["entries"]) <= catalogue.MAX_LOOKUP
    assert len(many["entries"]) + len(many["not_returned"]) == catalogue.MAX_LOOKUP + 3
    monkeypatch.setattr(catalogue, "LOOKUP_CHARS", 1)
    small = catalogue.lookup(ids[:3])
    # The first entry always comes back, whatever its size; the rest are named.
    assert [e["id"] for e in small["entries"]] == ids[:1]
    assert small["not_returned"] == ids[1:3]


def test_a_lookup_result_fits_the_loop_tool_result_limit():
    ids = [e["id"] for e in catalogue.entries()]
    out = catalogue.lookup(ids[:catalogue.MAX_LOOKUP])
    assert len(json.dumps(out, default=str)) <= loop._TOOL_RESULT_CHARS


def test_asked_ids_reads_a_list_or_one_id_only():
    assert catalogue.asked_ids(["a", " a ", "b", 3, None, ""]) == ["a", "b"]
    assert catalogue.asked_ids("tool:x") == ["tool:x"]
    assert catalogue.asked_ids(None) == [] and catalogue.asked_ids({"a": 1}) == []


# -- the driver: index in the system message, lookup as a tool ------------------------

def _agent():
    from app.agents.runtime import Agent

    return Agent(name="catalogue-probe", description="probe", system_prompt="", allowed_blocks=[])


def test_the_system_message_carries_the_index():
    text = context.system_message(None, None, None)["content"]
    lines = catalogue.index_lines()
    assert lines[0] in text and lines[-1] in text
    assert text.index(lines[0]) < text.index("No project is open")  # static lead, profile last


def test_the_lookup_tool_is_offered_last_with_or_without_a_hat(monkeypatch):
    from app.agents.driver import routes

    monkeypatch.setattr(routes, "catalogue", lambda: [])
    disciplines = [o for o in tool_registry.OWNERS if o != "base"]
    for hat in (None, disciplines[0]):
        names = [t["function"]["name"] for t in tools.offered(_agent(), "synthetic-proj", hat, disciplines)]
        assert names[-1] == catalogue.LOOKUP and names.count(catalogue.LOOKUP) == 1


@pytest.fixture
def _stream_through_call(monkeypatch):
    async def fake_stream(agent, messages, offered, model="", timeout=120.0):
        reply = await llm.call(agent, messages, offered, model=model)
        if reply.get("status") == "success":
            text = reply["message"].get("content") or ""
            if text:
                yield "content", text
        yield "done", reply

    monkeypatch.setattr(llm, "stream", fake_stream)


def _call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _scripted(replies, seen):
    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        seen.append({"messages": [dict(m) for m in messages],
                     "tools": [t["function"]["name"] for t in offered]})
        return {"status": "success", "message": replies.pop(0)}
    return fake_call


@pytest.mark.parametrize("pick", [0, -1])
def test_a_driver_turn_through_the_public_route_reads_an_entry(monkeypatch, _stream_through_call, pick):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("DRIVER_MODE", "request")
    wanted = catalogue.entries()[pick]["id"]
    seen = []
    replies = [{"content": "", "tool_calls": [_call(catalogue.LOOKUP, {"ids": [wanted, "tool:nope"]}, "c1")]},
               {"content": "Read it."}]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    body = {"message": "What can the platform work out?", "project_id": None, "history": []}
    with TestClient(app) as client:
        raw = client.post("/v1/chat/stream", json=body,
                          headers={"Authorization": "Bearer " + DEV_KEY, "X-Fork-Driver": "on"}).text
    events = [json.loads(line[len("data: "):]) for line in raw.split("\n") if line.startswith("data: ")]
    assert any(e.get("type") == "start" and e.get("mode") == "driver" for e in events)
    assert len(seen) == 2
    assert catalogue.LOOKUP in seen[0]["tools"]
    assert wanted in seen[0]["messages"][0]["content"]  # the index line
    result = json.loads(seen[1]["messages"][-1]["content"])
    assert [e["id"] for e in result["entries"]] == [wanted]
    assert result["unknown"] == ["tool:nope"]
    # A lookup is not a failed tool: no failure line, and the model's answer stands.
    text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    assert "Read it." in text and "nope" not in text


def test_the_turn_trace_records_the_ids_looked_up(monkeypatch, _stream_through_call, caplog):
    import logging

    ids = [e["id"] for e in catalogue.entries()][:2]
    replies = [{"content": "", "tool_calls": [_call(catalogue.LOOKUP, {"ids": ids}, "c1")]},
               {"content": "Done."}]
    monkeypatch.setattr(llm, "call", _scripted(replies, []))

    async def go():
        return [e async for e in loop.stream(_agent(), user_message="hi")]
    with caplog.at_level(logging.INFO):
        asyncio.run(go())
    turn = [r.getMessage() for r in caplog.records if "DRIVER_TURN" in r.getMessage()]
    assert turn and all(i in turn[-1] for i in ids)


@pytest.mark.parametrize("asked", [["tool:no_such_tool"], "formula:no_such_formula"])
def test_a_lookup_of_unknown_ids_is_told_to_the_model_not_the_user(monkeypatch, _stream_through_call, asked):
    seen = []
    replies = [{"content": "", "tool_calls": [_call(catalogue.LOOKUP, {"ids": asked}, "c1")]},
               {"content": "Answered without it."}]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))

    async def go():
        return [e async for e in loop.stream(_agent(), user_message="hi")]
    events = asyncio.run(go())
    told = json.loads(seen[1]["messages"][-1]["content"])
    assert told["ok"] is False and told["error"]
    text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    # The answer is the model's own words: no failure line is added for it.
    assert text.strip() == "Answered without it."
