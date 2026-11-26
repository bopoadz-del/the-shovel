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

def _registered_ids(kind):
    from app.agents import catalog
    from app.agents.driver import routes

    tool_registry.load()
    if kind == "formula":
        return {s.name for s in fr.all_specs()}
    if kind == "tool":
        return set(tool_registry._TOOLS)
    if kind == "workflow":
        return set(routes.catalogue())
    if kind == "skill":
        return {f"{h.discipline.value}.{p.id}" for h in catalog.list_hats()
                for p in (h.playbook.procedures or [] if h.playbook else [])}
    root = catalogue.knowledge_dir()
    return {f[:-3] for f in os.listdir(root) if f.endswith(".md")}


@pytest.mark.parametrize("kind", ["formula", "tool", "workflow", "skill", "knowledge"])
def test_every_registered_thing_has_one_entry(kind):
    ids = [e["id"] for e in catalogue.entries()]
    assert len(ids) == len(set(ids))
    want = _registered_ids(kind)
    assert want, kind  # the registry is populated, so the check is not empty
    assert {i.split(":", 1)[1] for i in ids if i.startswith(kind + ":")} == want


@pytest.mark.parametrize("kind", ["formula", "tool", "workflow", "skill", "knowledge"])
def test_every_entry_has_the_one_format(kind):
    of_kind = [e for e in catalogue.entries() if e["kind"] == kind]
    assert of_kind
    for e in of_kind:
        assert set(e) - {"needs"} == FIELDS, e["id"]
        assert e["id"].startswith(kind + ":")
        assert set(e["stamps"]) == {"verified", "tested", "approved"}
        # Nothing here stamps verified or approved: those are a tester's and the owner's.
        assert e["stamps"]["verified"] is None and e["stamps"]["approved"] is None
        assert e["source"].get("registry") == kind
        assert all(ex.get("invented") is True for ex in e["examples"])


@pytest.mark.parametrize("name,unit,hi", [("zz_cat_alpha", "m", 50.0), ("zz_cat_beta", "kN", 900.0)])
def test_a_formula_entry_carries_its_declared_inputs(registered, name, unit, hi):
    _formula(registered, name, {"size": Param(unit, 0, hi, label="the size", required=True)}, {"out": unit})
    catalogue._CACHE.clear()
    entry = next(e for e in catalogue.entries() if e["id"] == f"formula:{name}")
    assert entry["inputs"][0]["unit"] == unit and entry["inputs"][0]["range"] == [0, hi]
    assert entry["inputs"][0]["label"] == "the size" and entry["inputs"][0]["required"] is True
    assert entry["outputs"] == [{"name": "out", "unit": unit, "main": False}]
    assert entry["code"] == {"tool": "construction_calc", "calculation": name}
    assert entry["for"] == f"Synthetic {name}."
    assert entry["source"]["file"].endswith("test_driver_catalogue.py")


# -- declarations and gaps ----------------------------------------------------------

@pytest.mark.parametrize("declared,gaps", [
    ({"when_to_use": "a bay is rectangular", "when_not": "the bay is curved; use a polygon area",
      "examples": (_example(),), "standard": "a published method"}, []),
    ({}, ["when_to_use", "when_not", "examples", "standard"]),
    ({"when_to_use": "x", "examples": (_example(),), "standard": "s"}, ["when_not"]),
    ({"when_to_use": "x", "when_not": "y", "standard": "s"}, ["examples"]),
    ({"when_to_use": "x", "when_not": "y", "examples": (_example(result="r2"),)}, ["standard"]),
])
def test_declared_fields_fill_the_entry_and_each_missing_one_is_a_gap(registered, declared, gaps):
    _formula(registered, "zz_cat_declared", {"a": "m"}, {"b": "m"}, **declared)
    catalogue._CACHE.clear()
    entry = next(e for e in catalogue.entries() if e["id"] == "formula:zz_cat_declared")
    assert entry["gaps"] == gaps
    assert entry["when_to_use"] == declared.get("when_to_use", "")
    assert entry["when_not"] == declared.get("when_not", "")
    assert entry["examples"] == [dict(e) for e in declared.get("examples", ())]
    assert entry["standard"] == declared.get("standard")


@pytest.mark.parametrize("examples,why", [
    ((_example(invented=False),), "invented: True"),
    (({k: v for k, v in _example().items() if k != "invented"},), "invented: True"),
    ((_example(result=""),), "needs a result text"),
    ((_example(given=None),), "needs a given text"),
    ((_example(chain="not a list"),), "chain is a list of strings"),
    ((_example(chain=["ok", 3]),), "chain is a list of strings"),
    ((_example(figure_from="a document"),), "unknown example keys"),
    ("an example as text", "must be a sequence of dicts"),
    (_example(), "must be a sequence of dicts"),  # one dict, not a sequence of them
    (["an example as text"], "an example must be a dict"),
])
def test_an_example_that_is_not_invented_and_whole_is_refused(examples, why):
    with pytest.raises(ValueError, match=why):
        fr.declared_examples("formula", "zz", examples)
    with pytest.raises(ValueError, match=why):
        tool_registry.tool("zz_cat_tool_refused", owner="base", display_name="Z", examples=examples)


@pytest.mark.parametrize("given,kept", [
    ((_example(),), (_example(),)),
    (({"invented": True, "given": " g ", "result": "r"},), ({"invented": True, "given": "g", "chain": [], "result": "r"},)),
    ((), ()),
])
def test_a_whole_example_is_kept_and_its_chain_is_optional(given, kept):
    assert fr.declared_examples("formula", "zz", given) == kept


@pytest.mark.parametrize("declared,gaps", [
    ({"when_to_use": "w", "when_not": "n", "examples": (_example(),)}, ["for"]),
    ({"when_not": "n"}, ["for", "when_to_use", "examples"]),
])
def test_a_tool_declares_the_same_fields(registered, declared, gaps):
    async def handle(call):
        return {"ok": True}

    tool_registry.tool("zz_cat_tool", owner="base", display_name="Synthetic tool",
                       params={"q": Param(label="q", kind="text", required=True)}, **declared)(handle)
    registered["tool"].append("zz_cat_tool")
    catalogue._CACHE.clear()
    entry = next(e for e in catalogue.entries() if e["id"] == "tool:zz_cat_tool")
    assert (entry["when_to_use"], entry["when_not"]) == (declared.get("when_to_use", ""), declared["when_not"])
    # A tool with no schema has no description: its "for" is a gap.
    assert entry["gaps"] == gaps


# -- feeds / fed_by -----------------------------------------------------------------

@pytest.mark.parametrize("taker_unit,fed", [("m2", True), ("ha", False)])
def test_an_output_feeds_an_input_of_the_same_name_and_unit_only(registered, taker_unit, fed):
    _formula(registered, "zz_cat_src", {"a": "m"}, {"zz_area": "m2"})
    _formula(registered, "zz_cat_dst", {"zz_area": taker_unit}, {"c": "m3"})
    catalogue._CACHE.clear()
    held = {e["id"]: e for e in catalogue.entries()}
    assert held["formula:zz_cat_src"]["feeds"] == (["formula:zz_cat_dst"] if fed else [])
    assert held["formula:zz_cat_dst"]["fed_by"] == (["formula:zz_cat_src"] if fed else [])
    assert held["formula:zz_cat_dst"]["feeds"] == []


# -- the index ----------------------------------------------------------------------

@pytest.mark.parametrize("built", ["cold", "cached"])
def test_the_index_is_one_line_per_entry_in_a_fixed_order(built):
    if built == "cold":
        catalogue._CACHE.clear()
    items = catalogue.entries()
    lines = catalogue.index_lines()
    assert len(lines) == len(items)
    assert all("\n" not in line and line.count(" | ") >= 3 for line in lines)
    keys = [(catalogue.KINDS.index(line.split(" | ")[1]), line.split(" | ")[0]) for line in lines]
    assert keys == sorted(keys)
    assert catalogue.index_lines() == lines


_A = {"id": "tool:b", "kind": "tool", "name": "B", "for": "b"}
_B = {"id": "formula:z", "kind": "formula", "name": "Z", "for": "z"}
_C = {"id": "formula:a", "kind": "formula", "name": "A", "for": ""}
_D = {"id": "skill:q.1", "kind": "skill", "name": "Q", "for": "q"}


@pytest.mark.parametrize("given,want", [
    ([_A, _B, _C], ["formula:a | formula | A | -", "formula:z | formula | Z | z", "tool:b | tool | B | b"]),
    ([_C, _A, _B], ["formula:a | formula | A | -", "formula:z | formula | Z | z", "tool:b | tool | B | b"]),
    # by kind first: a skill comes after a tool although "skill" < "tool"
    ([_D, _A], ["tool:b | tool | B | b", "skill:q.1 | skill | Q | q"]),
])
def test_the_index_order_does_not_follow_the_order_given(given, want):
    assert catalogue.index_lines(given) == want


@pytest.mark.parametrize("text,cut", [
    ("w" * 160, False),
    ("w" * 161, True),
    ("word\nword " * 100, True),
])
def test_a_long_for_line_is_cut_to_one_line(text, cut):
    out = catalogue._one_line(text)
    assert "\n" not in out and len(out) <= catalogue.FOR_CHARS
    assert out.endswith("…") == cut


@pytest.mark.parametrize("name", ["zz_cat_late", "zz_cat_later"])
def test_a_registered_formula_reaches_the_index_without_a_restart(registered, name):
    before = catalogue.index_lines()
    _formula(registered, name, {"a": "m"}, {"b": "m"})
    after = catalogue.index_lines()
    assert len(after) == len(before) + 1
    assert [line for line in after if line.startswith(f"formula:{name} | ")] == [
        f"formula:{name} | formula | Synthetic {name} | Synthetic {name}."]


# -- lookup -------------------------------------------------------------------------

@pytest.mark.parametrize("reverse", [False, True])
def test_lookup_returns_the_named_entries_in_the_order_asked(reverse):
    ids = [e["id"] for e in catalogue.entries()]
    asked = [ids[-1], ids[0]] if reverse else [ids[0], ids[-1]]
    out = catalogue.lookup(asked)
    assert out["ok"] is True and [e["id"] for e in out["entries"]] == asked
    assert out["unknown"] == [] and "not_returned" not in out


@pytest.mark.parametrize("pick", [1, -2])
def test_lookup_names_unknown_ids_once(pick):
    known = catalogue.entries()[pick]["id"]
    out = catalogue.lookup([known, "formula:no_such_thing", known])
    assert [e["id"] for e in out["entries"]] == [known]
    assert out["unknown"] == ["formula:no_such_thing"]
    assert [e["id"] for e in catalogue.lookup(known)["entries"]] == [known]  # one id as text


@pytest.mark.parametrize("nothing,why", [
    ([], "Name the entry ids"), (None, "Name the entry ids"), (7, "Name the entry ids"),
    ([""], "Name the entry ids"), ({"ids": ["formula:x"]}, "Name the entry ids"),
    (["formula:no_such_thing"], "None of these ids"),
])
def test_lookup_of_nothing_it_holds_is_refused(nothing, why):
    out = catalogue.lookup(nothing)
    assert out["ok"] is False and out.get("entries", []) == [] and out["error"].startswith(why)


@pytest.mark.parametrize("extra", [1, 3])
def test_lookup_stops_at_the_entry_limit(extra):
    ids = [e["id"] for e in catalogue.entries()]
    many = catalogue.lookup(ids[:catalogue.MAX_LOOKUP + extra])
    assert len(many["entries"]) <= catalogue.MAX_LOOKUP
    assert len(many["entries"]) + len(many["not_returned"]) == catalogue.MAX_LOOKUP + extra


@pytest.mark.parametrize("asked", [2, 3])
def test_lookup_stops_at_the_character_limit(monkeypatch, asked):
    ids = [e["id"] for e in catalogue.entries()]
    monkeypatch.setattr(catalogue, "LOOKUP_CHARS", 1)
    small = catalogue.lookup(ids[:asked])
    # The first entry always comes back, whatever its size; the rest are named.
    assert [e["id"] for e in small["entries"]] == ids[:1]
    assert small["not_returned"] == ids[1:asked]


def test_a_lookup_result_fits_the_loop_tool_result_limit():
    ids = [e["id"] for e in catalogue.entries()]
    out = catalogue.lookup(ids[:catalogue.MAX_LOOKUP])
    assert len(json.dumps(out, default=str)) <= loop._TOOL_RESULT_CHARS


@pytest.mark.parametrize("given,want", [
    (["a", " a ", "b", 3, None, ""], ["a", "b"]),
    ("tool:x", ["tool:x"]),
    (None, []),
    ({"a": 1}, []),
])
def test_asked_ids_reads_a_list_or_one_id_only(given, want):
    assert catalogue.asked_ids(given) == want


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


# -- each builder and helper, named (the mutation gate selects tests by name) -------

@pytest.mark.parametrize("build,kind", [
    (catalogue._formula_entries, "formula"), (catalogue._tool_entries, "tool"),
    (catalogue._workflow_entries, "workflow"), (catalogue._skill_entries, "skill"),
    (catalogue._knowledge_entries, "knowledge"),
])
def test_each_builder_returns_its_registry_kind(build, kind):
    built = build()
    assert built and {e["kind"] for e in built} == {kind}
    assert {e["id"].split(":", 1)[1] for e in built} == _registered_ids(kind)
    assert all(e["for"] == catalogue._one_line(e["for"]) for e in built)


@pytest.mark.parametrize("unit", ["m", "kN"])
def test_formula_entries_read_the_registered_formula(registered, unit):
    _formula(registered, "zz_cat_built", {"a": Param(unit, 1, 9, label="aa", required=True)}, {"b": unit},
             standard="std")
    entry = next(e for e in catalogue._formula_entries() if e["id"] == "formula:zz_cat_built")
    assert entry["inputs"] == [{"name": "a", "label": "aa", "type": "number", "unit": unit, "range": [1, 9],
                                "required": True, "basis": ""}]
    assert entry["standard"] == "std" and entry["source"]["owner"] == "base"


@pytest.mark.parametrize("obj,ends", [(_example, "test_driver_catalogue.py"), (len, "")])
def test_source_file_is_the_repo_path_or_empty(obj, ends):
    path = catalogue._source_file(obj)
    assert path.endswith(ends) and "\\" not in path
    assert (path == "") == (ends == "")


@pytest.mark.parametrize("param,want", [
    (Param("m", 0, 5, label="len", kind="number", required=True),
     {"name": "x", "label": "len", "type": "number", "unit": "m", "range": [0, 5], "required": True, "basis": ""}),
    (Param("-", label="", kind="text", required=False),
     {"name": "x", "label": "x", "type": "text", "unit": "", "range": None, "required": False, "basis": ""}),
])
def test_input_reads_a_declared_param(param, want):
    assert catalogue._input("x", param) == want


@pytest.mark.parametrize("entry,extra,want", [
    ({"for": "f", "when_to_use": "u", "when_not": "n", "examples": [1]}, (), []),
    ({"for": "", "when_to_use": "", "when_not": "n", "examples": []}, (), ["for", "when_to_use", "examples"]),
    ({"for": "f", "when_to_use": "u", "when_not": "n", "examples": [1]}, ("card",), ["card"]),
])
def test_gaps_name_each_missing_declaration(entry, extra, want):
    assert catalogue._gaps(entry, extra) == want


@pytest.mark.parametrize("kind,key", [("formula", "a_b"), ("skill", "qaqc.q1")])
def test_entry_has_every_field_and_its_id(kind, key):
    entry = catalogue._entry(kind, key, name="N")
    assert set(entry) == FIELDS
    assert (entry["id"], entry["kind"], entry["name"]) == (f"{kind}:{key}", kind, "N")
    assert entry["stamps"] == {"verified": None, "tested": None, "approved": None}


def _linked(out_unit, in_unit):
    a = catalogue._entry("formula", "a", inputs=[], outputs=[{"name": "q", "unit": out_unit}])
    b = catalogue._entry("formula", "b", inputs=[{"name": "q", "unit": in_unit}], outputs=[])
    catalogue._link_formulas([a, b])
    return a, b


@pytest.mark.parametrize("in_unit,fed", [("m3", True), ("m2", False), ("", False)])
def test_link_formulas_by_name_and_unit(in_unit, fed):
    a, b = _linked("m3", in_unit)
    assert a["feeds"] == (["formula:b"] if fed else [])
    assert b["fed_by"] == (["formula:a"] if fed else [])


@pytest.mark.parametrize("card,needs,gaps", [
    ({"produces": "A register", "use_when": "asked for one", "needs": "a list",
      "inputs": {"items": "Items"}}, "a list", ["when_not", "examples"]),
    (None, None, ["for", "when_to_use", "when_not", "examples", "card"]),
])
def test_workflow_entries_read_the_card(monkeypatch, card, needs, gaps):
    from app.agents.driver import routes

    monkeypatch.setattr(routes, "catalogue", lambda: ["zz_register"])
    monkeypatch.setattr(routes, "cards", lambda: {"zz_register": card} if card else {})
    (entry,) = catalogue._workflow_entries()
    assert entry["id"] == "workflow:zz_register" and entry.get("needs") == needs
    assert entry["gaps"] == gaps
    assert entry["code"] == {"tool": "run_workflow", "workflow": "zz_register"}
    assert entry["inputs"] == ([{"name": "items", "label": "Items", "type": "", "unit": "", "range": None,
                                 "required": False, "basis": ""}] if card else [])


@pytest.mark.parametrize("text,want", [
    ("# Title\n\n| a | b |\n\nFirst prose line.\nSecond.", ("Title", "First prose line.")),
    ("Prose before.\n## Later heading\n", ("Later heading", "Prose before.")),
    ("", ("", "")),
])
def test_knowledge_summary_is_the_heading_and_first_prose(text, want):
    assert catalogue._knowledge_summary(text) == want


@pytest.mark.parametrize("files", [{"zz_one.md": "# One\n\nAbout one."},
                                   {"zz_a.md": "# A\n\nAbout a.", "zz_b.md": "no heading", "x.txt": "skip"}])
def test_knowledge_entries_read_each_markdown_file(monkeypatch, tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(catalogue, "knowledge_dir", lambda: str(tmp_path))
    built = catalogue._knowledge_entries()
    md = sorted(n for n in files if n.endswith(".md"))
    assert [e["id"] for e in built] == [f"knowledge:{n[:-3]}" for n in md]
    assert [e["source"]["file"] for e in built] == [f"docs/knowledge/{n}" for n in md]


@pytest.mark.parametrize("pick", [0, -1])
def test_skill_entries_carry_purpose_and_refusals(pick):
    from app.agents import catalog

    procs = [(h.discipline.value, p) for h in sorted(catalog.list_hats(), key=lambda h: h.discipline.value)
             for p in (h.playbook.procedures or [] if h.playbook else [])]
    hat, proc = procs[pick]
    entry = next(e for e in catalogue._skill_entries() if e["id"] == f"skill:{hat}.{proc.id}")
    assert (entry["when_to_use"], entry["when_not"]) == (proc.purpose.strip(), proc.refusals.strip())
    assert entry["code"] == {"hat": hat, "procedure": proc.id}


@pytest.mark.parametrize("pick", [0, -1])
def test_index_block_lists_the_index_and_the_lookup(pick):
    block = catalogue.index_block()
    assert catalogue.LOOKUP in block.split("\n", 1)[0]
    assert catalogue.index_lines()[pick] in block.split("\n")
