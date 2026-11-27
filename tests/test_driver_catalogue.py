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
    from app.blocks import _knowledge

    root = catalogue.knowledge_dir()
    return ({f[:-3] for f in os.listdir(root) if f.endswith(".md")}
            | {e["id"] for e in _knowledge.load_knowledge()})


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
    ((_example(invented=1),), "must say invented: True"),  # truthy is not True
    ((_example(invented="yes"),), "must say invented: True"),
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
    # The knowledge kind has two builders: the knowledge files and the knowledge base.
    (lambda: catalogue._knowledge_entries() + catalogue.kb_entries(), "knowledge"),
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


@pytest.mark.parametrize("note", ["This ask names a platform calculator.", ""])
def test_stream_adds_the_named_calculator_note_only_when_there_is_one(monkeypatch, _stream_through_call, note):
    """loop.stream: a note is a system message before the question; no note, no message."""
    seen = []
    monkeypatch.setattr(loop, "_named_calculator_note", lambda message: note)
    monkeypatch.setattr(llm, "call", _scripted([{"content": "Done."}], seen))

    async def go():
        return [e async for e in loop.stream(_agent(), user_message="hi")]
    asyncio.run(go())
    sent = seen[0]["messages"]
    assert [m["role"] for m in sent] == ["system"] + (["system"] if note else []) + ["user"]
    assert [m["content"] for m in sent[1:-1]] == ([note] if note else [])


# -- the whole lookup result is held to the loop's limit ------------------------------

@pytest.mark.parametrize("ordering", ["index", "reversed"])
def test_a_lookup_of_every_index_id_fits_the_loop_limit(ordering):
    ids = [e["id"] for e in catalogue.entries()]
    if ordering == "reversed":
        ids = ids[::-1]
    out = catalogue.lookup(ids)
    assert len(json.dumps(out, default=str)) <= loop._TOOL_RESULT_CHARS
    returned = [e["id"] for e in out["entries"]]
    assert returned == ids[:len(returned)] and 1 <= len(returned) <= catalogue.MAX_LOOKUP
    # Every found id past the returned ones is named or counted, in order.
    named = out.get("not_returned", [])
    assert named == ids[len(returned):len(returned) + len(named)]
    assert len(returned) + len(named) + out.get("not_returned_more", 0) == len(ids)


@pytest.mark.parametrize("unknown_count,id_chars", [(40, 200), (90, 120)])
def test_long_unknown_ids_do_not_push_a_lookup_past_the_loop_limit(unknown_count, id_chars):
    ids = [e["id"] for e in catalogue.entries()][:catalogue.MAX_LOOKUP]
    unknown = [f"formula:{i:03d}" + "x" * id_chars for i in range(unknown_count)]
    alone = catalogue.lookup(ids)
    out = catalogue.lookup(ids + unknown)
    assert len(json.dumps(out, default=str)) <= loop._TOOL_RESULT_CHARS
    assert out["entries"] == alone["entries"]  # the entries are never what gives way
    assert out["unknown"] == unknown[:len(out["unknown"])]
    assert len(out["unknown"]) + out.get("unknown_more", 0) == unknown_count
    assert out.get("unknown_more", 0) > 0


@pytest.mark.parametrize("room", [60, 200])
def test_lookup_names_what_fits_and_counts_the_rest(monkeypatch, room):
    ids = [e["id"] for e in catalogue.entries()][:6]
    monkeypatch.setattr(catalogue, "LOOKUP_CHARS", 1)  # one entry back; five to name
    base = len(json.dumps(catalogue.lookup(ids[:1]), default=str))
    out = catalogue.lookup(ids, limit=base + room)
    assert len(json.dumps(out, default=str)) <= base + room
    named = out.get("not_returned", [])
    assert named == ids[1:1 + len(named)]
    assert len(named) + out.get("not_returned_more", 0) == 5
    assert out.get("not_returned_more", 0) > 0


@pytest.mark.parametrize("names", [["a", "bb", "ccc"], ["formula:x", "tool:y"]])
def test_cap_names_leaves_a_list_that_fits_as_it_is(names):
    result = {"ok": True, "entries": [], "unknown": list(names)}
    catalogue._cap_names(result, "unknown", 10_000)
    assert result == {"ok": True, "entries": [], "unknown": names}  # no count key when all fit
    empty = {"ok": True, "unknown": []}
    catalogue._cap_names(empty, "unknown", 1)
    assert empty == {"ok": True, "unknown": []}  # nothing to cap


@pytest.mark.parametrize("names,keep", [(["aaaa", "bbbb", "cccc"], 2), (["aaaa", "bbbb", "cccc", "dddd"], 1)])
def test_cap_names_keeps_names_in_order_up_to_the_limit(names, keep):
    result = {"unknown": list(names)}
    want = {"unknown": names[:keep], "unknown_more": len(names) - keep}
    limit = len(json.dumps(want))
    catalogue._cap_names(result, "unknown", limit)
    assert result == want and len(json.dumps(result)) <= limit


@pytest.mark.parametrize("unknown", [["formula:zz_none_a", "formula:zz_none_b"],
                                     ["tool:zz_none_c", "tool:zz_none_d", "tool:zz_none_e"]])
def test_found_ids_to_ask_again_are_named_before_unknown_ones(monkeypatch, unknown):
    ids = [e["id"] for e in catalogue.entries()][:3]
    monkeypatch.setattr(catalogue, "LOOKUP_CHARS", 1)
    base = len(json.dumps(catalogue.lookup(ids[:1]), default=str))
    want_rest = len(json.dumps(ids[1:])) + len(', "not_returned": ')
    # Room is held for the unknown ids' count before any name is kept.
    held_count = len(f', "unknown_more": {len(unknown)}')
    out = catalogue.lookup(ids + unknown, limit=base + want_rest + held_count + 25)
    assert out["not_returned"] == ids[1:]
    assert len(out["unknown"]) < len(unknown) and out["unknown_more"] >= 1


@pytest.mark.parametrize("ordering", ["index", "reversed"])
def test_every_index_id_through_the_public_route_reaches_the_model_uncut(monkeypatch, _stream_through_call, ordering):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("DRIVER_MODE", "request")
    ids = [e["id"] for e in catalogue.entries()]
    if ordering == "reversed":
        ids = ids[::-1]
    seen = []
    replies = [{"content": "", "tool_calls": [_call(catalogue.LOOKUP, {"ids": ids}, "c1")]},
               {"content": "Read them."}]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    body = {"message": "What can the platform work out?", "project_id": None, "history": []}
    with TestClient(app) as client:
        client.post("/v1/chat/stream", json=body,
                    headers={"Authorization": "Bearer " + DEV_KEY, "X-Fork-Driver": "on"})
    shown = json.loads(seen[1]["messages"][-1]["content"])
    assert "truncated" not in shown
    assert [e["id"] for e in shown["entries"]] == [e["id"] for e in catalogue.lookup(ids)["entries"]]


# -- the cache follows every kind it builds -------------------------------------------

@pytest.fixture
def _counted_builds(monkeypatch):
    calls = []

    def build():
        calls.append(1)
        return []
    monkeypatch.setattr(catalogue, "_BUILDERS", (build,))
    catalogue._CACHE.clear()
    yield calls
    catalogue._CACHE.clear()


@pytest.mark.parametrize("first,then", [([], ["zz_wf_a"]), (["zz_wf_a"], ["zz_wf_a", "zz_wf_b"])])
def test_a_workflow_that_appears_later_reaches_the_index_without_a_restart(monkeypatch, first, then):
    from app.agents.driver import routes

    names, no_cards = {"now": first}, {}
    monkeypatch.setattr(routes, "catalogue", lambda: list(names["now"]))
    monkeypatch.setattr(routes, "cards", lambda: no_cards)  # one cards object: only the workflows change
    catalogue._CACHE.clear()
    before = [e["id"] for e in catalogue.entries() if e["kind"] == "workflow"]
    names["now"] = then
    after = [e["id"] for e in catalogue.entries() if e["kind"] == "workflow"]
    catalogue._CACHE.clear()
    assert before == [f"workflow:{n}" for n in first]
    assert after == [f"workflow:{n}" for n in then]


@pytest.mark.parametrize("produces", ["A first register.", "A second register."])
def test_a_reloaded_card_reaches_the_index_without_a_restart(monkeypatch, produces):
    from app.agents.driver import routes

    held = {"cards": {}}
    monkeypatch.setattr(routes, "catalogue", lambda: ["zz_wf_c"])
    monkeypatch.setattr(routes, "cards", lambda: held["cards"])
    catalogue._CACHE.clear()
    before = next(e for e in catalogue.entries() if e["id"] == "workflow:zz_wf_c")["for"]
    held["cards"] = {"zz_wf_c": {"produces": produces}}  # a new cards object, as after cache_clear
    after = next(e for e in catalogue.entries() if e["id"] == "workflow:zz_wf_c")["for"]
    catalogue._CACHE.clear()
    assert (before, after) == ("", produces)


def _knowledge_file_entries():
    """The knowledge entries built from the knowledge files (the knowledge
    base's entries share the kind; they name their entry in ``source``)."""
    return [e for e in catalogue.entries()
            if e["kind"] == "knowledge" and "entry" not in e["source"]]


@pytest.mark.parametrize("text", ["# Zz\n\ny", "# Zz\n\nOther words, longer."])  # the same size, then not
def test_a_changed_or_added_knowledge_file_reaches_the_index_without_a_restart(monkeypatch, tmp_path, text):
    monkeypatch.setattr(catalogue, "knowledge_dir", lambda: str(tmp_path))
    catalogue._CACHE.clear()
    assert not _knowledge_file_entries()
    path = tmp_path / "zz_k.md"
    path.write_text("# Zz\n\nx", encoding="utf-8")
    assert next(iter(_knowledge_file_entries()))["for"] == "x"
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # a later write, whatever the clock
    got = next(iter(_knowledge_file_entries()))["for"]
    catalogue._CACHE.clear()
    assert got == text.split("\n\n", 1)[1]


@pytest.mark.parametrize("hats", [1, 2])
def test_a_reloaded_hat_set_is_a_new_state(monkeypatch, hats):
    from app.agents import catalog

    held = {"hats": [object()]}
    monkeypatch.setattr(catalog, "list_hats", lambda: list(held["hats"]))
    before = catalogue._state_key()
    assert catalogue._state_key() == before  # the same objects: the same state
    held["hats"] = [object() for _ in range(hats)]
    assert catalogue._state_key() != before


@pytest.mark.parametrize("calls_after", [2, 3])
def test_an_unchanged_state_is_built_once(_counted_builds, calls_after):
    for _ in range(calls_after):
        catalogue.entries()
    assert len(_counted_builds) == 1


@pytest.mark.parametrize("files", [{}, {"zz_a.md": "a", "zz_b.md": "bb"}])
def test_knowledge_state_names_each_markdown_file_with_its_size(monkeypatch, tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "skip.txt").write_text("no", encoding="utf-8")
    monkeypatch.setattr(catalogue, "knowledge_dir", lambda: str(tmp_path))
    state = catalogue._knowledge_state()
    assert [(n, size) for n, _, size in state] == [(n, len(t)) for n, t in sorted(files.items())]
    monkeypatch.setattr(catalogue, "knowledge_dir", lambda: str(tmp_path / "missing"))
    assert catalogue._knowledge_state() == ()


# -- what the tests did not pin -------------------------------------------------------

@pytest.mark.parametrize("names", [("q", "q"), ("r", "r")])
def test_an_output_with_no_unit_feeds_nothing(names):
    a = catalogue._entry("formula", "a", inputs=[], outputs=[{"name": names[0], "unit": ""}])
    b = catalogue._entry("formula", "b", inputs=[{"name": names[1], "unit": ""}], outputs=[])
    catalogue._link_formulas([a, b])
    assert (a["feeds"], b["fed_by"]) == ([], [])


@pytest.mark.parametrize("unit", ["m3", "kN"])
def test_a_formula_never_feeds_itself(unit):
    a = catalogue._entry("formula", "a", inputs=[{"name": "q", "unit": unit}], outputs=[{"name": "q", "unit": unit}])
    b = catalogue._entry("formula", "b", inputs=[{"name": "q", "unit": unit}], outputs=[])
    catalogue._link_formulas([a, b])
    assert a["feeds"] == ["formula:b"] and a["fed_by"] == []
    assert b["fed_by"] == ["formula:a"]


@pytest.mark.parametrize("pick", [0, -1])
def test_a_caller_changing_its_copy_leaves_the_catalogue_as_built(pick):
    first = catalogue.entries()
    name = first[pick]["name"]
    first[pick]["name"] = "changed by a caller"
    first[pick]["inputs"].append({"name": "added"})
    again = catalogue.entries()
    assert again[pick]["name"] == name
    assert {"name": "added"} not in again[pick]["inputs"]


@pytest.mark.parametrize("certified", [True, False])
def test_a_certified_formula_is_stamped_tested_by_the_registry(registered, monkeypatch, certified):
    _formula(registered, "zz_cat_cert", {"a": "m"}, {"b": "m2"})
    monkeypatch.setattr(fr, "certified", lambda name: certified if name == "zz_cat_cert" else False)
    entry = next(e for e in catalogue._formula_entries() if e["id"] == "formula:zz_cat_cert")
    assert entry["stamps"]["tested"] == ({"by": "formula_registry.certified"} if certified else None)
    assert entry["stamps"]["verified"] is None and entry["stamps"]["approved"] is None


@pytest.mark.parametrize("use,not_", [("  use me  ", " not me "), ("\tfor sums\n", "  never for rates")])
def test_a_tool_declaration_is_stored_without_its_padding(registered, use, not_):
    async def handle(call):
        return {"ok": True}

    tool_registry.tool("zz_cat_pad", owner="base", display_name="Padded tool",
                       params={"q": Param(label="q", kind="text", required=True)},
                       when_to_use=use, when_not=not_)(handle)
    registered["tool"].append("zz_cat_pad")
    catalogue._CACHE.clear()
    entry = next(e for e in catalogue.entries() if e["id"] == "tool:zz_cat_pad")
    assert (entry["when_to_use"], entry["when_not"]) == (use.strip(), not_.strip())


def _driver_turn_line(monkeypatch, caplog, asked, user_message="hi"):
    import logging

    replies = [{"content": "", "tool_calls": [_call(catalogue.LOOKUP, {"ids": asked}, "c1")]},
               {"content": "Done."}]
    monkeypatch.setattr(llm, "call", _scripted(replies, []))

    async def go():
        return [e async for e in loop.stream(_agent(), user_message=user_message)]
    with caplog.at_level(logging.INFO):
        asyncio.run(go())
    return [r.getMessage() for r in caplog.records if "DRIVER_TURN" in r.getMessage()][-1]


def _looked_up(line):
    turn = json.loads(line.split("DRIVER_TURN ", 1)[1])
    return [c for s in turn["steps"] for c in s.get("calls", []) if c.get("tool") == catalogue.LOOKUP]


@pytest.mark.parametrize("pick", [(0, 1), (-1, 2)])
def test_the_turn_trace_records_held_ids_as_read(monkeypatch, _stream_through_call, caplog, pick):
    held = [e["id"] for e in catalogue.entries()]
    first, second = held[pick[0]], held[pick[1]]
    line = _driver_turn_line(monkeypatch, caplog, [f" {first} ", first, second])
    [call] = _looked_up(line)
    assert (call["ids"], call["not_returned"], call["unknown"]) == ([first, second], 0, 0)


@pytest.mark.parametrize("words", ["my card is 4111 1111 1111 1111",
                                   "ring the site office on 050 123 4567 about the bill"])
def test_the_turn_trace_never_keeps_an_id_the_catalogue_does_not_hold(monkeypatch, _stream_through_call,
                                                                       caplog, words):
    # The model may send the user's own words as an id: the trace keeps a
    # count of such ids, never their text.
    held = catalogue.entries()[0]["id"]
    line = _driver_turn_line(monkeypatch, caplog, [words, held, words + " again"], user_message=words)
    [call] = _looked_up(line)
    assert (call["ids"], call["not_returned"], call["unknown"]) == ([held], 0, 2)
    assert words.split()[-1] not in line.split("DRIVER_TURN ", 1)[1]


@pytest.mark.parametrize("count, chars", [(3000, 110), (500, 2000)])
def test_the_turn_trace_stays_small_for_many_long_unknown_ids(monkeypatch, _stream_through_call, caplog,
                                                              count, chars):
    asked = [f"{i:06d}" + "x" * (chars - 6) for i in range(count)]
    line = _driver_turn_line(monkeypatch, caplog, asked)
    [call] = _looked_up(line)
    assert (call["ids"], call["not_returned"], call["unknown"]) == ([], 0, count)
    assert len(line) < 5_000


_NOTHING_READ = {"ids": [], "not_returned": 0, "unknown": 0}


@pytest.mark.parametrize("ids, want", [(None, _NOTHING_READ), ([], _NOTHING_READ), (["", "  "], _NOTHING_READ),
                                       (["zz:none", "zz:none"], {**_NOTHING_READ, "unknown": 1})])
def test_read_ids_of_nothing_or_one_unknown(ids, want):
    assert catalogue.read_ids(catalogue.lookup(ids)) == want


@pytest.mark.parametrize("result", [None, [], 7, "", True, {}, {"entries": None}, {"entries": ["x", 3]},
                                    {"ok": False, "error": "Name the entry ids"}])
def test_read_ids_of_something_that_is_no_lookup_result_is_nothing_read(result):
    assert catalogue.read_ids(result) == _NOTHING_READ


@pytest.mark.parametrize("asked_count", [13, 40])
def test_the_turn_trace_records_only_the_ids_the_model_was_given(monkeypatch, _stream_through_call, caplog,
                                                                 asked_count):
    """More held ids than one lookup returns: the model receives some entries
    and the rest by name or by count. The trace says the same, id for id."""
    asked = [e["id"] for e in catalogue.entries()][:asked_count]
    given = catalogue.lookup(asked)
    received = [e["id"] for e in given["entries"]]
    assert 0 < len(received) <= catalogue.MAX_LOOKUP < asked_count
    line = _driver_turn_line(monkeypatch, caplog, asked)
    [call] = _looked_up(line)
    assert call["ids"] == received
    assert (call["not_returned"], call["unknown"]) == (asked_count - len(received), 0)


@pytest.mark.parametrize("asked_count, unknown_count", [(3, 2), (20, 400)])
def test_read_ids_retells_the_lookup_result_without_reading_the_catalogue_again(monkeypatch, asked_count,
                                                                                unknown_count):
    """The trace is written on the event loop: it reads the result it is
    given and never the catalogue (which is built off the loop)."""
    asked = [e["id"] for e in catalogue.entries()][:asked_count] + [f"zz:none-{i}" * 9 for i in range(unknown_count)]
    result = catalogue.lookup(asked)

    def no_second_pass():
        raise AssertionError("the trace read the catalogue again")

    monkeypatch.setattr(catalogue, "entries", no_second_pass)
    monkeypatch.setattr(catalogue, "_state_key", no_second_pass)
    read = catalogue.read_ids(result)
    assert read["ids"] == [e["id"] for e in result["entries"]]
    assert read["not_returned"] == asked_count - len(result["entries"])
    assert read["unknown"] == unknown_count


@pytest.mark.parametrize("names", [["a" * 50, "b" * 5, "c" * 5], ["x" * 40, "y", "z"]])
def test_the_names_kept_are_a_prefix_nothing_after_the_first_that_does_not_fit(names):
    """A long name that does not fit ends the list: a shorter one after it is
    counted, not kept, so the kept names are always the first ones asked."""
    base = len(json.dumps({"unknown": [], "unknown_more": len(names)}))
    limit = base + len(json.dumps(names[1])) + len(json.dumps(names[2])) + 4   # room for both short names
    assert limit < base + len(json.dumps(names[0]))                           # and not for the long one
    result = {"unknown": list(names)}
    catalogue._cap_names(result, "unknown", limit)
    assert result == {"unknown": [], "unknown_more": len(names)}


@pytest.mark.parametrize("param,want", [
    (Param("m", 0, None, label="len", kind="number"), [0, None]),
    (Param("kN", None, 50, label="load", kind="number"), [None, 50]),
])
def test_a_one_sided_range_keeps_its_one_bound(param, want):
    assert catalogue._input("x", param)["range"] == want


@pytest.mark.parametrize("examples", [5, 2.5])
def test_examples_that_are_not_a_sequence_are_refused_at_declaration(examples):
    with pytest.raises(ValueError, match="examples must be a sequence"):
        fr.declared_examples("formula", "zz", examples)


@pytest.mark.parametrize("name_chars, count", [(3, 150), (7, 1200)])
def test_cap_names_fits_the_limit_when_the_count_has_several_digits(name_chars, count):
    # Every limit across a span is met, including those that leave the
    # result within a few characters of it with a multi-digit count.
    names = [f"{i:0{name_chars}d}"[-name_chars:] for i in range(count)]
    base = len(json.dumps({"unknown": [], "unknown_more": count}))
    for limit in range(base, base + 200):
        result = {"unknown": list(names)}
        catalogue._cap_names(result, "unknown", limit)
        assert len(json.dumps(result)) <= limit, limit
        assert len(result["unknown"]) + result.get("unknown_more", 0) == count


@pytest.mark.parametrize("first, then", [("Area of a bay.", "Area of a bay, gross."),
                                         ("Volume of a pit.", "Volume of a pit, net of blinding.")])
def test_a_formula_re_declared_under_its_name_is_a_new_state(registered, first, then):
    import inspect

    def make(description):
        def fn(**kw):
            return {}
        fn.__name__ = "zz_cat_redeclared"
        fn.__signature__ = inspect.Signature([inspect.Parameter("a", inspect.Parameter.KEYWORD_ONLY,
                                                                annotation=float)])
        fr.formula(owner="base", description=description, display_name="Redeclared",
                   inputs={"a": "m"}, outputs={"b": "m"})(fn)
    registered["formula"].append("zz_cat_redeclared")
    make(first)
    assert {e["id"]: e for e in catalogue.entries()}["formula:zz_cat_redeclared"]["for"] == first
    fr._REGISTRY.pop("zz_cat_redeclared")  # the registry refuses a second declaration
    make(then)  # no cache clear: the state key alone must see the new spec
    assert {e["id"]: e for e in catalogue.entries()}["formula:zz_cat_redeclared"]["for"] == then


@pytest.mark.parametrize("first, then", [("use for sums", "use for totals"), ("before a bill", "after a bill")])
def test_a_tool_spec_replaced_under_its_name_is_a_new_state(registered, first, then):
    import dataclasses

    async def handle(call):
        return {"ok": True}

    tool_registry.tool("zz_cat_replaced", owner="base", display_name="Replaced tool",
                       params={"q": Param(label="q", kind="text", required=True)},
                       when_to_use=first)(handle)
    registered["tool"].append("zz_cat_replaced")
    assert {e["id"]: e for e in catalogue.entries()}["tool:zz_cat_replaced"]["when_to_use"] == first
    spec = tool_registry._TOOLS["zz_cat_replaced"]
    tool_registry._TOOLS["zz_cat_replaced"] = dataclasses.replace(spec, when_to_use=then)
    assert {e["id"]: e for e in catalogue.entries()}["tool:zz_cat_replaced"]["when_to_use"] == then


# -- the whole lookup result fits the loop's limit, whatever is asked -------------------

@pytest.mark.parametrize("unknown_count", [1, 5, 12, 40])
def test_every_id_and_unknown_ids_fit_the_loops_limit(unknown_count):
    held = [e["id"] for e in catalogue.entries()]
    asked = held + [f"unknown:{i:010d}" for i in range(unknown_count)]
    out = catalogue.lookup(asked)
    assert len(json.dumps(out, default=str)) <= loop._TOOL_RESULT_CHARS
    # Nothing is lost: every asked id is returned, named or counted.
    named = len(out["entries"]) + len(out.get("not_returned", [])) + out.get("not_returned_more", 0)
    assert named == len(held)
    assert len(out["unknown"]) + out.get("unknown_more", 0) == unknown_count


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_random_lookups_fit_the_loops_limit(seed):
    import random

    rng = random.Random(seed)
    held = [e["id"] for e in catalogue.entries()]
    for _ in range(60):
        asked = rng.sample(held, rng.randint(len(held) // 2, len(held)))
        asked += [f"unknown:{rng.random():.12f}" for _ in range(rng.randint(0, 40))]
        rng.shuffle(asked)
        out = catalogue.lookup(asked)
        assert len(json.dumps(out, default=str)) <= loop._TOOL_RESULT_CHARS


@pytest.mark.parametrize("first, second", [(["a" * 5] * 30, ["b" * 7] * 30), (["c" * 9] * 50, ["d" * 3] * 80)])
def test_cap_names_holds_room_for_a_later_count(first, second):
    first = [f"{x}{i}" for i, x in enumerate(first)]
    second = [f"{x}{i}" for i, x in enumerate(second)]
    base = len(json.dumps({"p": [], "p_more": len(first), "q": [], "q_more": len(second)}))
    for limit in range(base, base + 400, 7):
        result = {"p": list(first), "q": list(second)}
        catalogue._cap_names(result, ("p", "q"), limit)
        assert len(json.dumps(result)) <= limit, limit
        assert len(result["p"]) + result.get("p_more", 0) == len(first)
        assert len(result["q"]) + result.get("q_more", 0) == len(second)


@pytest.mark.parametrize("p, q", [(["x", "y"], ["z"]), ([], ["a", "b", "c"]), (["d"], [])])
def test_cap_names_keeps_every_name_of_both_lists_when_they_fit(p, q):
    result = {"p": list(p), "q": list(q)}
    catalogue._cap_names(result, ("p", "q"), 10_000)
    assert result == {"p": p, "q": q}


@pytest.mark.parametrize("name", ["N" * 200, "word " * 40])
def test_an_index_line_cuts_a_long_name_to_80_characters(name):
    entry = {"id": "zz:long", "kind": "tool", "name": name, "for": "x"}
    shown = catalogue.index_line(entry).split(" | ")[2]
    assert len(shown) <= 80
    assert shown[:20] == name.strip()[:20]


@pytest.mark.parametrize("found, rest", [(13, 1), (20, 8)])
def test_a_lookup_of_more_than_twelve_found_ids_returns_twelve(found, rest):
    ids = [e["id"] for e in catalogue.entries()][:found]
    out = catalogue.lookup(ids)
    assert len(out["entries"]) == 12
    assert out["not_returned"] == ids[12:] and len(out["not_returned"]) == rest
