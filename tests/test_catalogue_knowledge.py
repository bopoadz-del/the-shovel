"""Knowledge entries of the catalogue: one per knowledge base entry, and the
sections of each knowledge file, all read from the knowledge as it is."""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from app.agents.driver import catalogue, catalogue_knowledge as ck
from app.blocks import _knowledge


def _kb(path, entries):
    path.write_text(json.dumps({"schema_version": "1", "kb_version": "t", "entries": entries}),
                    encoding="utf-8")
    return str(path)


ALPHA = {"id": "zz.alpha_rule", "domain": "d.one", "type": "formula", "title": "Alpha rule",
         "statement": "Alpha   holds\nfor  a span.", "expression": "2 * L",
         "variables": {"L": {"unit": "meter", "desc": "Span length"}},
         "thresholds": {"unit": "meter"}, "applicability": {"applies_to": ["d.one"]},
         "remediation": [], "provenance": {"source": "s-a", "course_reference": "c-a"},
         "credibility_tier": 2, "needs_review": False}
GAMMA = {"id": "zz.gamma_load", "domain": "d.three", "type": "formula", "title": "Gamma load",
         "statement": "Gamma is a load.", "expression": "q * A",
         "variables": {"q": {"unit": "kilopascal", "desc": "Pressure"},
                       "A": {"unit": "meter**2"}},
         "thresholds": {"unit": "kilonewton"}, "table": [{"k": 1}],
         "provenance": {"source": "s-g"}, "credibility_tier": 1, "needs_review": False}
BETA = {"id": "zz.beta_flow", "domain": "d.two", "type": "workflow", "title": "Beta flow",
        "statement": "Beta moves through states.", "expression": "",
        "variables": {}, "thresholds": {}, "states": ["open", "closed"],
        "transitions": [{"from": "open", "to": "closed"}], "checklist": ["x"],
        "provenance": {"source": "s-b"}, "credibility_tier": 3, "needs_review": True}
DELTA = {"id": "zz.delta_check", "domain": "d.four", "type": "rule", "title": "Delta check",
         "statement": "Delta is checked.", "checklist": ["a", "b"],
         "provenance": {"source": "s-d"}, "credibility_tier": 2}

# Entries the tool does not evaluate although they hold what an evaluation
# would read: variables without an expression, an expression on a type that
# is not a formula, a formula with no expression, a variable that is a list.
THETA = {"id": "zz.theta_limit", "domain": "d.five", "type": "threshold", "title": "Theta limit",
         "statement": "Theta stays under a limit.",
         "variables": {"T_a": {"unit": "degC", "desc": "One side"}, "T_b": {"unit": "degC"}},
         "thresholds": {"unit": "degC", "max": 20}, "provenance": {"source": "s-t"},
         "credibility_tier": 2}
KAPPA = {"id": "zz.kappa_gap", "domain": "d.six", "type": "threshold", "title": "Kappa gap",
         "statement": "Kappa is a gap.", "expression": "a - b",
         "variables": {"a": {"unit": "meter"}, "b": {"unit": "meter"}},
         "thresholds": {"unit": "meter"}, "provenance": {"source": "s-k"}, "credibility_tier": 2}
LAMDA = {"id": "zz.lamda_blank", "domain": "d.seven", "type": "formula", "title": "Lamda blank",
         "statement": "Lamda has no expression.", "expression": "",
         "variables": {"n": {"unit": "", "desc": "Count"}}, "thresholds": {"unit": "meter"},
         "provenance": {"source": "s-l"}, "credibility_tier": 1}
SIGMA = {"id": "zz.sigma_steps", "domain": "d.eight", "type": "workflow", "title": "Sigma steps",
         "statement": "Sigma moves in steps.", "expression": "x + 1",
         "variables": {"x": {"unit": "meter"}, "layout": ["widths"]},
         "thresholds": {"unit": "meter"}, "states": ["a", "b"],
         "provenance": {"source": "s-s"}, "credibility_tier": 3}
# An entry that holds only a part a call does not return.
OMEGA = {"id": "zz.omega_table", "domain": "d.nine", "type": "table", "title": "Omega table",
         "table": {"columns": [{"name": "k"}], "rows": [{"k": 1}]}, "checklist": ["x"],
         "provenance": {"source": "s-o"}, "credibility_tier": 2}


def _match(entry, query):
    """What the tool's own block returns for the entry when asked for it."""
    import asyncio

    from app.blocks.construction_advisor import ConstructionAdvisorBlock

    answer = asyncio.run(ConstructionAdvisorBlock().process({"query": query}, {"top_k": 20}))
    found = [m for m in answer["matches"] if m["id"] == entry["id"]]
    assert len(found) == 1, (entry["id"], query, [m["id"] for m in answer["matches"]])
    return found[0]


def _asked(entry, value):
    """The tool's own answer to the entry's code with a value for every
    variable the knowledge base entry declares (one value when it declares
    none, so an evaluation is asked for either way)."""
    import asyncio

    from app.blocks.construction_advisor import ConstructionAdvisorBlock

    declared = entry.get("variables") if isinstance(entry.get("variables"), dict) else {}
    values = {k: value for k in declared} or {"x": value}
    code = ck.kb_entry(entry, "kb.json")["code"]
    return asyncio.run(ConstructionAdvisorBlock().process(
        {"query": entry["title"]}, {"rule_id": code["rule_id"], "values": values}))


@pytest.fixture
def kb_file(tmp_path, monkeypatch):
    def use(entries, name="kb.json"):
        path = _kb(tmp_path / name, entries)
        monkeypatch.setenv("CONSTRUCTION_KB_FILE", path)
        catalogue._CACHE.clear()
        return path
    yield use
    catalogue._CACHE.clear()


# --- sections of a markdown file ---------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("# Title\nprose\n## One\n### One.a\n## Two ##\n", ["One", "One.a", "Two"]),
    ("# T\n## Rates\n```\n## not a heading\n```\n## Units\n", ["Rates", "Units"]),
    ("# T\n~~~\n# fenced\n~~~\n## After\n", ["After"]),
    ("# Only a title\nprose only\n", []),
    ("", []),
    ("no heading at all\n#hashtag is not a heading\n", []),
    ("#\n# Real title\n## S\n", ["S"]),
    ("# T\n## ##\n## S\n", ["S"]),                      # a heading with no text is not a section
    ("# T\n####### x\n## S\n", ["S"]),                  # seven hashes are not a heading
    ("# T\n## Using C#\n## F# and C# #\n", ["Using C#", "F# and C#"]),  # only a spaced run closes
    ("# T\n~~~\n```\n## in\n~~~\n## After\n", ["After"]),         # another fence does not close
    ("# T\n````\n```\n## in\n````\n## Out\n", ["Out"]),           # nor a shorter one
    ("# T\n```\n``` py\n## in\n```\n## Out\n", ["Out"]),          # nor one with text after it
    ("# T\n``` not `a` fence\n## Seen\n", ["Seen"]),      # inline code is not a fence
    ("# T\n    ## code\n\t## code\n   ## Three spaces\n", ["Three spaces"]),  # four spaces: code
    ("#hashtag\n## Real\n### Sub\n", ["Sub"]),            # "Real" is the first heading: the name
])
def test_markdown_sections_are_the_headings_after_the_title(text, want):
    assert ck.markdown_sections(text) == want


@pytest.mark.parametrize("text,want", [
    ("# Title\nprose\n## One\n### One.a\n", ["Title", "One", "One.a"]),
    ("#hashtag\n## Real\n```\n# code\n```\n### Sub #\n", ["Real", "Sub"]),
    ("no heading\n####### seven\n    # code\n", []),
    ("", []),
    (None, []),
])
def test_markdown_headings_are_every_heading_with_text_outside_code(text, want):
    assert ck.markdown_headings(text) == want


@pytest.mark.parametrize("text", [
    "#hashtag\n## Real\n## Next\n",
    "####### seven\n## Using C#\n### Two\n",
    "~~~\n# fenced\n```\n~~~\n# Shown\n## S\n",
    "    # code\n# T #\n## ##\n## S\n",
    "prose only\n",
])
def test_a_file_name_and_its_sections_are_one_reading_of_its_headings(text):
    # The name is the first heading and the sections are the rest: no heading
    # is both, and none is dropped as "the title" while another one is the name.
    title = catalogue._knowledge_summary(text)[0]
    assert ([title] if title else []) + ck.markdown_sections(text) == ck.markdown_headings(text)


@pytest.mark.parametrize("files,want", [
    ({"a_file.md": "# A\nFor a.\n## Scope\n## Table 1\n", "b_file.md": "# B\nFor b.\n"},
     {"knowledge:a_file": ["Scope", "Table 1"], "knowledge:b_file": []}),
    ({"c_file.md": "# C\n## Only\n"}, {"knowledge:c_file": ["Only"]}),
])
def test_a_knowledge_file_entry_carries_its_sections(monkeypatch, tmp_path, files, want):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(catalogue, "knowledge_dir", lambda: str(tmp_path))
    built = {e["id"]: e["source"]["sections"] for e in catalogue._knowledge_entries()}
    assert built == want


# --- one entry per knowledge base entry ----------------------------------------------

@pytest.mark.parametrize("entry,want", [
    (ALPHA, ["statement", "expression", "thresholds"]),   # its variables are not returned
    (BETA, ["statement"]),                                # nor states, transitions, a checklist
    (GAMMA, ["statement", "expression", "thresholds"]),   # nor a table
    (THETA, ["statement", "thresholds"]),
    (OMEGA, []),                                          # it holds parts; a call returns none
    ({"id": "x", "title": "t", "statement": "", "table": None, "checklist": []}, []),
    ({"id": "y", "thresholds": {"max": 3}, "statement": "S.", "expression": None}, ["thresholds", "statement"]),
])
def test_kb_sections_are_the_parts_of_the_entry_a_call_returns(entry, want):
    assert ck.kb_sections(entry) == want


@pytest.mark.parametrize("entries", [
    [ALPHA, BETA, THETA, OMEGA],      # a formula, a workflow, a threshold, a table
    [GAMMA, SIGMA, DELTA, KAPPA],     # a formula with a table, a workflow, a rule, a threshold
])
def test_an_entry_names_only_the_parts_the_tool_returns_for_it(kb_file, entries):
    kb_file(entries)
    named = 0
    for entry in entries:
        e = ck.kb_entry(entry, "kb.json")
        match = _match(entry, entry["title"])
        sections = e["source"]["sections"]
        # Every part named is in the tool's answer, with the entry's content;
        # every part of the entry the answer carries with content is named.
        assert all(match[part] == entry[part] for part in sections), (entry["id"], sections, match)
        assert sections == [k for k in entry if k not in ck.RECORD_FIELDS and ck._held(match.get(k))]
        # The rest of what the entry says of itself is in the answer too.
        assert (e["name"], e["source"]["type"], e["source"]["domain"]) == \
            (match["title"], match["type"], match["domain"])
        assert e["for"] == catalogue._one_line(match["statement"])
        assert e["source"]["provenance"] == match["provenance"]
        assert e["source"]["credibility_tier"] == match["credibility_tier"]
        named += len(sections)
    assert named  # the sets hold parts that are returned, so the check is not empty


@pytest.mark.parametrize("asked_by", ["title", "id"])
def test_no_shipped_kb_entry_names_a_part_the_tool_does_not_return(monkeypatch, asked_by):
    monkeypatch.delenv("CONSTRUCTION_KB_FILE", raising=False)
    catalogue._CACHE.clear()
    built = {e["id"]: e for e in catalogue.entries() if e["kind"] == "knowledge"}
    shipped = _knowledge.load_knowledge()
    held_back = 0
    for kb in shipped:
        e = built["knowledge:" + kb["id"]]
        # Asked as a caller would: by the entry's title, or by its id the way
        # the tool's handler turns an id into a search.
        match = _match(kb, kb["title"] if asked_by == "title" else kb["id"].replace(".", " "))
        for part in e["source"]["sections"]:
            assert ck._held(match.get(part)) and match[part] == kb[part], (kb["id"], part)
        held_back += len([k for k in kb if k not in ck.RECORD_FIELDS and ck._held(kb[k])
                          and k not in e["source"]["sections"]])
    assert shipped and held_back  # the base holds parts a call does not return: none is named


@pytest.mark.parametrize("entry,file,name,for_,inputs,outputs", [
    (ALPHA, "kb.json", "Alpha rule", "Alpha holds for a span.",  # whitespace folded, no word changed
     [{"name": "L", "label": "Span length", "type": "number", "unit": "meter",
       "range": None, "required": True, "basis": ""}],
     [{"name": "result", "unit": "meter", "main": True}]),
    (GAMMA, "other.json", "Gamma load", "Gamma is a load.",
     [{"name": "q", "label": "Pressure", "type": "number", "unit": "kilopascal",
       "range": None, "required": True, "basis": ""},
      {"name": "A", "label": "A", "type": "number", "unit": "meter**2",
       "range": None, "required": True, "basis": ""}],
     [{"name": "result", "unit": "kilonewton", "main": True}]),
])
def test_a_formula_kb_entry_is_read_as_it_is(entry, file, name, for_, inputs, outputs):
    e = ck.kb_entry(entry, file)
    assert e["id"] == "knowledge:" + entry["id"] and e["kind"] == "knowledge"
    assert (e["name"], e["for"]) == (name, for_)
    assert e["inputs"] == inputs and e["outputs"] == outputs
    assert e["code"] == {"tool": "construction_advisor", "rule_id": entry["id"]}
    assert e["source"]["provenance"] == entry["provenance"]
    assert e["source"]["credibility_tier"] == entry["credibility_tier"]
    assert (e["source"]["type"], e["source"]["domain"]) == ("formula", entry["domain"])
    assert e["source"]["file"] == file and e["source"]["entry"] == entry["id"]
    assert e["stamps"] == {"verified": None, "tested": None, "approved": None}


@pytest.mark.parametrize("entry,sections", [
    (BETA, ["statement"]),
    (DELTA, ["statement"]),
])
def test_an_entry_without_an_expression_has_no_output_and_no_inputs(entry, sections):
    e = ck.kb_entry(entry, "kb.json")
    assert e["outputs"] == [] and e["inputs"] == []
    assert e["name"] == entry["title"] and e["for"] == entry["statement"]
    assert e["source"]["sections"] == sections


@pytest.mark.parametrize("entry", [THETA, KAPPA, LAMDA, SIGMA])
def test_an_entry_the_tool_does_not_evaluate_offers_no_inputs_and_no_output(entry):
    e = ck.kb_entry(entry, "kb.json")
    assert e["inputs"] == [] and e["outputs"] == []
    # Its variables are not a part a call returns, and its code still finds it.
    assert "variables" not in e["source"]["sections"] and "statement" in e["source"]["sections"]
    assert e["code"] == {"tool": "construction_advisor", "rule_id": entry["id"]}
    assert e["source"]["type"] == entry["type"]


@pytest.mark.parametrize("entries,value", [
    ([ALPHA, THETA, KAPPA, BETA], 3.0),
    ([GAMMA, LAMDA, SIGMA, DELTA], 2.5),
])
def test_an_entry_offers_inputs_and_an_output_exactly_when_the_tool_evaluates_it(kb_file, entries, value):
    kb_file(entries)
    evaluated = 0
    for entry in entries:
        e = ck.kb_entry(entry, "kb.json")
        answer = _asked(entry, value)
        # The offer is the tool's behaviour: an output iff an evaluation came
        # back, a refusal otherwise; inputs iff those values were taken.
        assert bool(e["outputs"]) == ("evaluation" in answer), (entry["id"], answer)
        assert bool(e["outputs"]) != ("evaluation_error" in answer), (entry["id"], answer)
        if e["outputs"]:
            evaluated += 1
            assert [i["name"] for i in e["inputs"]] == list(entry["variables"])
            assert e["outputs"] == [{"name": "result", "unit": answer["evaluation"]["unit"], "main": True}]
            assert isinstance(answer["evaluation"]["result"], float)
        else:
            assert e["inputs"] == []
    assert 0 < evaluated < len(entries)  # both sides of the rule were exercised


@pytest.mark.parametrize("value", [2.0, 7.5])
def test_no_shipped_kb_entry_offers_what_the_tool_refuses(monkeypatch, value):
    monkeypatch.delenv("CONSTRUCTION_KB_FILE", raising=False)
    catalogue._CACHE.clear()
    built = {e["id"]: e for e in catalogue.entries() if e["kind"] == "knowledge"}
    offered, refused = 0, 0
    for kb in _knowledge.load_knowledge():
        e = built["knowledge:" + kb["id"]]
        declared = kb.get("variables") if isinstance(kb.get("variables"), dict) else {}
        if _knowledge.evaluation_refusal(kb) is None:
            offered += 1
            assert [i["name"] for i in e["inputs"]] == list(declared)
            assert e["outputs"] == [{"name": "result", "unit": (kb.get("thresholds") or {}).get("unit") or "",
                                     "main": True}]
        else:
            refused += 1
            assert e["inputs"] == [] and e["outputs"] == [], kb["id"]
            with pytest.raises(ValueError):
                _knowledge.evaluate(kb["id"], **({k: value for k in declared} or {"x": value}))
    assert offered and refused  # the shipped base has both, so neither side is empty


@pytest.mark.parametrize("entry", [ALPHA, GAMMA])
def test_an_entry_does_not_share_the_knowledge_base_provenance(entry):
    before = entry["provenance"]["source"]
    e = ck.kb_entry(entry, "kb.json")
    e["source"]["provenance"]["source"] = "changed"
    assert entry["provenance"]["source"] == before


@pytest.mark.parametrize("entry,name,gaps", [
    ({"id": "zz.no_statement", "title": "T"}, "T", ["for", "when_to_use", "when_not", "examples"]),
    ({"id": "zz.stated", "title": "U", "statement": "S."}, "U", ["when_to_use", "when_not", "examples"]),
    ({"id": "zz.untitled"}, "zz.untitled", ["for", "when_to_use", "when_not", "examples"]),
    ({"id": "zz.blank_title", "title": ""}, "zz.blank_title", ["for", "when_to_use", "when_not", "examples"]),
])
def test_a_kb_entry_names_its_gaps_and_falls_back_to_its_id(entry, name, gaps):
    e = ck.kb_entry(entry, "kb.json")
    assert e["gaps"] == gaps and e["name"] == name


@pytest.mark.parametrize("entries,want", [
    ([BETA, {"title": "no id"}, "not an entry", ALPHA], ["knowledge:zz.beta_flow", "knowledge:zz.alpha_rule"]),
    ([GAMMA, {"id": ""}, DELTA], ["knowledge:zz.gamma_load", "knowledge:zz.delta_check"]),
    ([], []),
])
def test_kb_entries_follow_the_file_and_leave_out_an_entry_without_an_id(kb_file, entries, want):
    kb_file(entries)
    assert [e["id"] for e in ck.kb_entries()] == want


@pytest.mark.parametrize("content", ["not json", json.dumps({"no_entries": []})])
def test_kb_entries_of_an_unreadable_or_missing_file_are_none(monkeypatch, tmp_path, content):
    bad = tmp_path / "bad.json"
    bad.write_text(content, encoding="utf-8")
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(bad))
    assert ck.kb_entries() == []
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(tmp_path / "missing.json"))
    assert ck.kb_entries() == []


@pytest.mark.parametrize("name", ["kb.json", "nested_kb.json"])
def test_kb_file_names_the_file_the_tool_reads(kb_file, monkeypatch, name):
    path = kb_file([ALPHA], name)
    root = os.path.realpath(catalogue._ROOT)
    assert os.path.normcase(os.path.normpath(os.path.join(root, ck.kb_file()))) == \
        os.path.normcase(os.path.realpath(path))
    monkeypatch.delenv("CONSTRUCTION_KB_FILE")
    assert ck.kb_file() == "app/knowledge/construction_kb.json"


@pytest.mark.parametrize("root,served,want", [
    ("C:\\repo\\the-shovel", "D:\\served\\kb.json", "D:/served/kb.json"),
    ("E:\\work", "C:\\data\\sub\\..\\other_kb.json", "C:/data/other_kb.json"),
    ("C:\\repo\\the-shovel", "C:\\repo\\the-shovel\\app\\k\\kb.json", "app/k/kb.json"),
])
def test_kb_file_on_another_drive_is_its_full_path(monkeypatch, root, served, want):
    # Windows path rules on every platform: a file on another drive letter has
    # no path relative to the repository, so its full path is named, with "/".
    import ntpath
    import types

    monkeypatch.setattr(ck, "os", types.SimpleNamespace(path=ntpath, sep=ntpath.sep))
    monkeypatch.setattr(catalogue, "_ROOT", root)
    monkeypatch.setattr(_knowledge, "_kb_path", lambda: served)
    assert ck.kb_file() == want


@pytest.mark.parametrize("entries,later", [([ALPHA], 5_000_000_000), ([GAMMA, DELTA], 9_000_000_000)])
def test_kb_identity_changes_with_a_later_mtime_at_the_same_size(kb_file, entries, later):
    path = kb_file(entries)
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    a = _knowledge.kb_identity()
    os.utime(path, ns=(later, later))
    b = _knowledge.kb_identity()
    assert a[2] == b[2] and a[1] < b[1] and a != b


@pytest.mark.parametrize("one,two", [([ALPHA], [ALPHA, BETA]), ([GAMMA, DELTA], [DELTA])])
def test_kb_identity_changes_with_another_size_at_the_same_mtime(kb_file, one, two):
    path = kb_file(one)
    os.utime(path, ns=(3_000_000_000, 3_000_000_000))
    a = _knowledge.kb_identity()
    _kb(pathlib.Path(path), two)
    os.utime(path, ns=(3_000_000_000, 3_000_000_000))
    b = _knowledge.kb_identity()
    assert a[1] == b[1] and a[2] != b[2] and a != b


@pytest.mark.parametrize("spelling", [("sub", "..", "kb.json"), ("sub", "deeper", "..", "..", "kb.json")])
def test_kb_identity_names_the_resolved_path_whatever_the_spelling(kb_file, monkeypatch, tmp_path, spelling):
    path = kb_file([ALPHA])
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    spelled = os.path.join(str(tmp_path), *spelling)
    assert spelled != path
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", spelled)
    identity = _knowledge.kb_identity()
    assert identity[0] == os.path.realpath(path) and ".." not in identity[0]
    assert identity[1] is not None


# --- the catalogue carries them --------------------------------------------------------

@pytest.mark.parametrize("entries,line,looked_up,sections", [
    ([ALPHA, BETA], "knowledge:zz.alpha_rule | knowledge | Alpha rule | Alpha holds for a span.",
     "knowledge:zz.beta_flow", ["statement"]),
    ([GAMMA, DELTA], "knowledge:zz.delta_check | knowledge | Delta check | Delta is checked.",
     "knowledge:zz.gamma_load", ["statement", "expression", "thresholds"]),
])
def test_kb_entries_reach_the_catalogue_index_and_lookup(kb_file, entries, line, looked_up, sections):
    kb_file(entries)
    assert line in catalogue.index_lines()
    found = catalogue.lookup([looked_up])
    assert found["entries"][0]["source"]["sections"] == sections


@pytest.mark.parametrize("first,then", [([ALPHA], [BETA, ALPHA]), ([GAMMA, DELTA], [DELTA])])
def test_a_changed_knowledge_base_reaches_the_index_without_a_restart(kb_file, first, then):
    path = kb_file(first)
    ids = {line.split(" | ")[0] for line in catalogue.index_lines()}
    assert {"knowledge:" + e["id"] for e in first} <= ids
    _kb(pathlib.Path(path), then)
    os.utime(path, ns=(1, 1))  # a new stat identity even inside the clock's resolution
    ids = {line.split(" | ")[0] for line in catalogue.index_lines()}
    kb_ids = {i for i in ids if i.startswith("knowledge:zz.")}
    assert kb_ids == {"knowledge:" + e["id"] for e in then}


@pytest.mark.parametrize("one,two", [([ALPHA], [ALPHA, BETA]), ([GAMMA], [DELTA])])
def test_kb_identity_changes_with_the_file(kb_file, monkeypatch, tmp_path, one, two):
    first = kb_file(one, "one.json")
    a = _knowledge.kb_identity()
    second = kb_file(two, "two.json")
    b = _knowledge.kb_identity()
    assert a[0] == os.path.realpath(first) and b[0] == os.path.realpath(second) and a != b
    assert all(isinstance(v, int) for v in a[1:] + b[1:])
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(tmp_path / "gone.json"))
    assert _knowledge.kb_identity()[1:] == (None, None, None)


@pytest.mark.parametrize("field,read", [("title", "name"), ("statement", "for")])
def test_every_shipped_kb_entry_has_one_entry_read_as_it_is(monkeypatch, field, read):
    monkeypatch.delenv("CONSTRUCTION_KB_FILE", raising=False)
    catalogue._CACHE.clear()
    shipped = _knowledge.load_knowledge()
    assert shipped  # the shipped base is populated, so the check is not empty
    built = {e["id"]: e for e in catalogue.entries() if e["kind"] == "knowledge"}
    for kb in shipped:
        e = built["knowledge:" + kb["id"]]
        assert e[read] == (kb[field] if read == "name" else catalogue._one_line(kb[field]))
        assert e["code"] == {"tool": "construction_advisor", "rule_id": kb["id"]}
        assert e["source"]["file"] == "app/knowledge/construction_kb.json"
        assert "statement" in e["source"]["sections"]
