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
])
def test_markdown_sections_are_the_headings_after_the_title(text, want):
    assert ck.markdown_sections(text) == want


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
    (ALPHA, ["statement", "expression", "variables", "thresholds"]),
    (BETA, ["statement", "states", "transitions", "checklist"]),
    (GAMMA, ["statement", "expression", "variables", "thresholds", "table"]),
    ({"id": "x", "title": "t", "statement": "", "table": None, "checklist": []}, []),
])
def test_kb_sections_are_the_fields_with_content_minus_the_record(entry, want):
    assert ck.kb_sections(entry) == want


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
    (BETA, ["statement", "states", "transitions", "checklist"]),
    (DELTA, ["statement", "checklist"]),
])
def test_an_entry_without_an_expression_has_no_output_and_no_inputs(entry, sections):
    e = ck.kb_entry(entry, "kb.json")
    assert e["outputs"] == [] and e["inputs"] == []
    assert e["name"] == entry["title"] and e["for"] == entry["statement"]
    assert e["source"]["sections"] == sections


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


# --- the catalogue carries them --------------------------------------------------------

@pytest.mark.parametrize("entries,line,looked_up,sections", [
    ([ALPHA, BETA], "knowledge:zz.alpha_rule | knowledge | Alpha rule | Alpha holds for a span.",
     "knowledge:zz.beta_flow", ["statement", "states", "transitions", "checklist"]),
    ([GAMMA, DELTA], "knowledge:zz.delta_check | knowledge | Delta check | Delta is checked.",
     "knowledge:zz.gamma_load", ["statement", "expression", "variables", "thresholds", "table"]),
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
