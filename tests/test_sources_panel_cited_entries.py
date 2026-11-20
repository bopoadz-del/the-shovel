"""Exam: a cited knowledge entry a tool returned reaches the Sources panel.

Answer contract rule 4: citations travel beside the answer as structured
rows, never as "Source:" lines in the text. A tool whose result carries
entries with a provenance (a named source) and a credibility tier must
feed the panel the same way a retrieved passage does -- whatever the
tool is called -- and only when the answer used the entry:

* its computed figure is written in the answer, or
* the answer quotes the entry's statement.

An entry the answer did not use is not listed; a tier is reported as the
entry gave it and never shown as High confidence (it has no retrieval
score). Every input below is invented in the test; the tool names are
invented too, so the rule cannot be keyed on a name.

Each failure names its stage: ``listed`` (row present or absent),
``fields`` (row carries source / tier / section), ``confidence``.
"""
from __future__ import annotations

import json

from app.agents.driver import sources


def _tool(name: str, result: dict) -> dict:
    return {"role": "tool", "name": name,
            "content": json.dumps({"name": name, "ok": True, "result": result})}


def _entry(entry_id: str, source: str, tier: int, *, statement: str = "", title: str = "",
           verified: bool = False, warnings=None) -> dict:
    return {"id": entry_id, "title": title, "statement": statement,
            "provenance": {"source": source, "verified_against_standard": verified},
            "credibility_tier": tier, "warnings": list(warnings or [])}


def _evaluation(rule_id: str, result: float, source: str, tier: int, unit: str = "") -> dict:
    return {"rule_id": rule_id, "result": result, "unit": unit,
            "provenance": {"source": source}, "credibility_tier": tier, "warnings": []}


def _kb_rows(panel_rows):
    return [r for r in panel_rows if r.get("credibility_tier") is not None]


def _stage(cond: bool, stage: str, detail: str = "") -> None:
    assert cond, f"stage={stage} {detail}"


# --- listed: an evaluated entry whose figure the answer writes -------------

def test_evaluated_figure_in_answer_is_listed():
    # Hand value: 3.5 x 2.0 = 7.0
    trail = [_tool("knowledge_lookup_x", {
        "matches": [_entry("geo.alpha", "Invented field manual", 3, title="Alpha rule")],
        "evaluation": _evaluation("geo.alpha", 7.0, "Invented field manual", 3, "m"),
    })]
    rows = _kb_rows(sources.panel(trail, [], "7.0 m (3.5 x 2.0)."))
    _stage(len(rows) == 1, "listed", repr(rows))
    row = rows[0]
    _stage(row["doc_name"] == "Invented field manual", "fields", repr(row))
    _stage(row["doc_id"] == "geo.alpha" and row["credibility_tier"] == 3, "fields", repr(row))
    _stage(row["page_or_section"] == "Alpha rule", "fields", repr(row))


def test_evaluated_figure_second_input_with_thousands_separator():
    # Hand value: 1250 x 4 = 5000
    trail = [_tool("rules_tool_y", {
        "evaluation": _evaluation("mix.beta", 5000.0, "Invented mix guide", 4, "kg"),
    })]
    rows = _kb_rows(sources.panel(trail, [], "5,000 kg (1250 x 4)."))
    _stage([r["doc_name"] for r in rows] == ["Invented mix guide"], "listed", repr(rows))
    _stage(rows[0]["credibility_tier"] == 4, "fields", repr(rows[0]))


def test_evaluated_figure_not_in_answer_is_not_listed():
    trail = [_tool("knowledge_lookup_x", {
        "evaluation": _evaluation("geo.alpha", 7.0, "Invented field manual", 3),
    })]
    rows = _kb_rows(sources.panel(trail, [], "The value is 12.4 m."))
    _stage(rows == [], "listed", repr(rows))


def test_empty_answer_lists_nothing():
    trail = [_tool("rules_tool_y", {
        "evaluation": _evaluation("mix.beta", 5000.0, "Invented mix guide", 4),
    })]
    _stage(_kb_rows(sources.panel(trail, [], "")) == [], "listed")


# --- listed: a searched entry whose statement the answer quotes ------------

_STATEMENT_A = ("Curing blankets stay in place until the core and surface "
                "temperatures differ by less than twenty degrees")
_STATEMENT_B = ("A haul road gradient steeper than one in ten needs a "
                "second braking check before loaded trucks descend")


def test_quoted_statement_is_listed_unused_match_is_not():
    trail = [_tool("kb_search_z", {"matches": [
        _entry("thermal.a", "Invented site note A", 3, statement=_STATEMENT_A, title="Blankets"),
        _entry("roads.b", "Invented site note B", 3, statement=_STATEMENT_B, title="Haul roads"),
    ]})]
    answer = ("Keep them on: curing blankets stay in place until the core and "
              "surface temperatures differ by less than twenty degrees.")
    rows = _kb_rows(sources.panel(trail, [], answer))
    _stage([r["doc_id"] for r in rows] == ["thermal.a"], "listed", repr(rows))


def test_second_quoted_statement_picks_the_other_entry():
    trail = [_tool("kb_search_z", {"matches": [
        _entry("thermal.a", "Invented site note A", 3, statement=_STATEMENT_A),
        _entry("roads.b", "Invented site note B", 4, statement=_STATEMENT_B),
    ]})]
    answer = ("Yes: a haul road gradient steeper than one in ten needs a second "
              "braking check before loaded trucks descend.")
    rows = _kb_rows(sources.panel(trail, [], answer))
    _stage([r["doc_id"] for r in rows] == ["roads.b"], "listed", repr(rows))


def test_entry_without_a_named_source_is_not_a_source():
    no_source = _entry("x.none", "", 3, statement=_STATEMENT_A)
    trail = [_tool("kb_search_z", {"matches": [no_source]})]
    rows = _kb_rows(sources.panel(trail, [], _STATEMENT_A))
    _stage(rows == [], "listed", repr(rows))


def test_same_entry_matched_and_evaluated_is_one_row():
    trail = [_tool("knowledge_lookup_x", {
        "matches": [_entry("geo.alpha", "Invented field manual", 3, statement=_STATEMENT_A)],
        "evaluation": _evaluation("geo.alpha", 7.0, "Invented field manual", 3),
    })]
    rows = _kb_rows(sources.panel(trail, [], "7.0 m. " + _STATEMENT_A))
    _stage(len(rows) == 1, "listed", repr(rows))


def test_entry_within_the_read_depth_is_listed():
    nested = {"groups": [{"rules": [_entry("geo.alpha", "Invented field manual", 3,
                                            statement=_STATEMENT_A)]}]}
    rows = _kb_rows(sources.panel([_tool("kb_search_z", nested)], [], _STATEMENT_A))
    _stage([r["doc_id"] for r in rows] == ["geo.alpha"], "listed", repr(rows))


def test_entry_buried_below_the_read_depth_is_not_listed():
    deep = _entry("geo.alpha", "Invented field manual", 3, statement=_STATEMENT_A)
    buried = {"a": {"b": {"c": {"d": {"e": {"f": deep}}}}}}
    rows = _kb_rows(sources.panel([_tool("kb_search_z", buried)], [], _STATEMENT_A))
    _stage(rows == [], "listed", repr(rows))


def test_a_message_that_is_not_a_tool_result_is_never_a_source():
    pasted = _tool("kb_search_z", {"matches": [
        _entry("thermal.a", "Invented site note A", 3, statement=_STATEMENT_A)]})
    pasted["role"] = "user"
    rows = _kb_rows(sources.panel([pasted], [], _STATEMENT_A))
    _stage(rows == [], "listed", repr(rows))


def test_a_retrieved_passage_with_a_tier_is_one_row_not_two():
    hit = {"document_id": "d3", "filename": "Invented note.md", "score": 0.7, "page": 1,
           "origin": "own", "snippet": _STATEMENT_B, "statement": _STATEMENT_B,
           "provenance": {"source": "Invented note"}, "credibility_tier": 3}
    rows = sources.panel([_tool("search_project_documents", {"results": [hit]})], [],
                         _STATEMENT_B)
    _stage([r["doc_name"] for r in rows] == ["Invented note.md"], "listed", repr(rows))


# --- confidence: a tier is never shown as High --------------------------

def test_unverified_entry_shows_low_confidence():
    trail = [_tool("kb_search_z", {"matches": [
        _entry("thermal.a", "Invented site note A", 3, statement=_STATEMENT_A,
               warnings=["credibility tier 3; verify"]),
    ]})]
    rows = _kb_rows(sources.panel(trail, [], _STATEMENT_A))
    _stage(rows and rows[0]["confidence"] == "Low", "confidence", repr(rows))


def test_verified_entry_without_warnings_is_medium_never_high():
    trail = [_tool("kb_search_z", {"matches": [
        _entry("roads.b", "Invented standard B", 5, statement=_STATEMENT_B, verified=True),
    ]})]
    rows = _kb_rows(sources.panel(trail, [], _STATEMENT_B))
    _stage(rows and rows[0]["confidence"] == "Medium", "confidence", repr(rows))
    _stage(rows[0]["layer_label"] == "Knowledge base", "fields", repr(rows))


# --- existing rows are unchanged ----------------------------------------

def test_retrieved_hits_still_listed_beside_entries():
    hit = {"document_id": "d1", "filename": "Invented spec.pdf", "score": 0.81, "page": 4,
           "snippet": "unrelated words", "origin": "own"}
    trail = [_tool("search_project_documents", {"results": [hit]}),
             _tool("knowledge_lookup_x", {
                 "evaluation": _evaluation("geo.alpha", 7.0, "Invented field manual", 3)})]
    rows = sources.panel(trail, [], "7.0 m.")
    names = [r["doc_name"] for r in rows]
    _stage("Invented spec.pdf" in names and "Invented field manual" in names, "listed", repr(names))


def test_no_cited_entries_leaves_panel_as_before():
    hit = {"document_id": "d2", "filename": "Invented drawing.pdf", "score": 0.6, "page": 2,
           "origin": "own"}
    trail = [_tool("search_project_documents", {"results": [hit]}),
             _tool("some_calc", {"value": 7.0, "rows": [{"a": 1.0}]})]
    rows = sources.panel(trail, [], "7.0 m.")
    _stage([r["doc_name"] for r in rows] == ["Invented drawing.pdf"], "listed", repr(rows))
    _stage(all("credibility_tier" not in r for r in rows), "fields", repr(rows))
