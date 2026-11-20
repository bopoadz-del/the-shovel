"""Offline answer-against-its-sources scorer: each rule on two inputs
(one that holds, one that does not). Every input is built here."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "answer_source_score.py"
_spec = importlib.util.spec_from_file_location("answer_source_score", _PATH)
ass = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = ass
_spec.loader.exec_module(ass)


def _chunk(cid: str, name: str, text: str = "") -> dict:
    return {"chunk_id": cid, "doc_id": "d-" + cid, "doc_name": name, "text": text}


# --- (a) numbers --------------------------------------------------------------

@pytest.mark.parametrize("answer, expected, flag", [
    ("The wall is 250 mm thick [source: Spec A.pdf].", (1, 0), "ok"),
    ("The wall is 300 mm thick [source: Spec A.pdf].", (0, 1), "low"),
])
def test_number_in_cited_chunk_is_found_or_not(answer, expected, flag):
    row = {"id": "r", "answer": answer,
           "handed": [_chunk("c1", "Spec A.pdf", "Walls shall be 250 mm thick.")]}
    s = ass.score_row(row)
    assert (s.numbers_found, s.numbers_not_found) == expected
    assert s.flag == flag
    assert ("number_not_found" in s.reasons) == (expected[1] > 0)


@pytest.mark.parametrize("tool_result, expected", [
    ('{"result": 945.0}', (1, 0)),
    ('{"result": 900.0}', (0, 1)),
])
def test_number_from_tool_result(tool_result, expected):
    row = {"id": "r", "answer": "Volume is 945 m3.", "tool_results": [tool_result]}
    s = ass.score_row(row)
    assert (s.numbers_found, s.numbers_not_found) == expected


@pytest.mark.parametrize("answer, expected_not_found", [
    # handed but not cited: the rule reads cited chunks, not the whole turn
    ("Volume is 945 m3.", 1),
    ("Volume is 945 m3 [source: X.pdf].", 0),
])
def test_number_only_in_uncited_chunk_is_not_found(answer, expected_not_found):
    row = {"id": "r", "answer": answer, "handed": [_chunk("c1", "X.pdf", "945")]}
    assert ass.score_row(row).numbers_not_found == expected_not_found


@pytest.mark.parametrize("text, expected", [
    ("SAR 4,984,120 and 0.125 l", [4984120.0, 0.125]),
    ("4984120", [4984120.0]),
])
def test_thousands_commas_and_decimals_compare_as_values(text, expected):
    assert ass.numbers_in(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("ref IP-INF-053 item I233.3 sheet D/13/2", []),
    ("ref 53 item 233.3", [53.0, 233.3]),
])
def test_identifier_digits_are_not_figures(text, expected):
    assert ass.numbers_in(text) == expected


@pytest.mark.parametrize("sources, expected", [
    (["A.pdf", "B.pdf"], [0.125, 30.0]),
    # no panel: the trailing 2 is an ordinary number
    ([], [0.125, 30.0, 2.0]),
])
def test_numbered_marker_is_not_a_figure_when_panel_numbers_it(sources, expected):
    answer = "Leakage is 0.125 litres per 30 minutes 2.\nNext line."
    assert ass.answer_numbers(answer, sources) == expected


@pytest.mark.parametrize("answer, expected", [
    ("Rate 585 [source: Bill 13 page 2.pdf]", [585.0]),
    ("Rate 585 per Bill 13", [585.0, 13.0]),
])
def test_number_inside_a_named_citation_is_not_a_figure(answer, expected):
    assert ass.answer_numbers(answer) == expected


# --- (b) citations handed -----------------------------------------------------

@pytest.mark.parametrize("cited, expected, reason", [
    ("Spec A.pdf", (1, 0), False),
    ("Spec B.pdf", (0, 1), True),
])
def test_named_citation_counts_handed_or_not(cited, expected, reason):
    row = {"id": "r", "answer": f"Grade C40 [source: {cited}].",
           "handed": [_chunk("c1", "Spec A.pdf", "grade C40")]}
    s = ass.score_row(row)
    assert (s.citations_handed, s.citations_not_handed) == expected
    assert ("citation_not_handed" in s.reasons) == reason


@pytest.mark.parametrize("answer, expected", [
    ("Rate 585 [source: Bill (Priced).pdf].", (1, 0)),
    ("Rate 585 (source: Bill (Priced).pdf).", (1, 0)),
    ("Rate 585 [source: Bill (Draft).pdf].", (0, 1)),
    ("Rate 585 (source: Bill (Draft).pdf).", (0, 1)),
])
def test_named_citation_keeps_parentheses_inside_the_name(answer, expected):
    row = {"id": "r", "answer": answer, "handed": [_chunk("c1", "Bill (Priced).pdf", "rate 585")]}
    s = ass.score_row(row)
    assert (s.citations_handed, s.citations_not_handed) == expected


def test_ellipsis_name_matches_on_both_ends():
    assert ass.name_matches("IP-BOQ-07…ities (Priced).xlsx", "IP-BOQ-07 Bill of Quantities (Priced).xlsx")
    assert not ass.name_matches("IP-BOQ-07…ities (Priced).xlsx", "IP-BOQ-07 Bill of Quantities (Draft).xlsx")


@pytest.mark.parametrize("answer, expected", [
    ("Value 5 (chunk_id=c7).", 1),
    ("Value 5 (chunk_id=c8).", 0),
])
def test_keyed_citation_matches_chunk_id(answer, expected):
    handed = [_chunk("c7", "A.pdf", "text")]
    cites = ass.find_citations(answer)
    assert len(ass.cited_chunks(cites[0], handed)) == expected


@pytest.mark.parametrize("answer, expected", [
    ("Tested at 2 m head 1.\n", [("marker", "Method.pdf")]),
    # a marker beyond the panel's numbering is not a citation
    ("Tested at 2 m head 3.\n", []),
])
def test_marker_resolves_through_sources_list(answer, expected):
    cites = ass.find_citations(answer, ["Method.pdf"])
    assert [(c.kind, c.target) for c in cites] == expected


@pytest.mark.parametrize("line, kind, expected", [
    # a panel line is "<name>, <section>": the marker names the handed document
    ("Method Statement.pdf, Section 4 – Testing", "marker", 1),
    # the same line held by a named citation is not stripped
    ("Method Statement.pdf, Section 4 – Testing", "name", 0),
    # a longer name that only begins with the handed name is another document
    ("Method Statement.pdf.bak, Section 4", "marker", 0),
    # the handed name must open the line, not sit inside it
    ("Old Method Statement.pdf, Section 4", "marker", 0),
    ("Method Statement.pdf", "marker", 1),
])
def test_panel_line_with_section_names_the_handed_document(line, kind, expected):
    handed = [_chunk("c1", "Method Statement.pdf", "text")]
    cite = ass.Citation(kind, line, "Tested 1.")
    assert len(ass.cited_chunks(cite, handed)) == expected


@pytest.mark.parametrize("answer, expected", [
    ("Test at 2 m head above soffit.", 0),
    ("Test above soffit 2.", 1),
])
def test_mid_sentence_small_number_is_not_a_marker(answer, expected):
    assert len(ass.find_citations(answer, ["A.pdf", "B.pdf"])) == expected


@pytest.mark.parametrize("answer, expected", [
    # the answer contract's bracketed form, adjacent markers included
    ("Grade C40 [1][2].", [("marker", "A.pdf"), ("marker", "B.pdf")]),
    # a bracketed marker the panel does not number is not a citation
    ("Grade C40 [3].", []),
])
def test_bracketed_marker_resolves_through_sources_list(answer, expected):
    cites = ass.find_citations(answer, ["A.pdf", "B.pdf"])
    assert [(c.kind, c.target) for c in cites] == expected


@pytest.mark.parametrize("answer, expected", [
    ("Walls are 250 mm thick [3].", [250.0]),
    # parentheses are not the marker form: the 3 stays a figure
    ("Walls are 250 mm thick (3).", [250.0, 3.0]),
])
def test_bracketed_marker_is_never_a_figure(answer, expected):
    assert ass.answer_numbers(answer, ["A.pdf"]) == expected


@pytest.mark.parametrize("sources, handed, not_handed, numbers, unresolved", [
    (["Spec A.pdf"], 1, 0, (1, 0), 0),
    # the panel numbers no entry: the marker is unresolved, the figure unbacked
    ([], 0, 0, (0, 1), 1),
])
def test_bracketed_marker_row_score(sources, handed, not_handed, numbers, unresolved):
    row = {"id": "r", "answer": "Walls are 250 mm thick [1].", "sources_list": sources,
           "handed": [_chunk("c1", "Spec A.pdf", "Walls shall be 250 mm thick.")]}
    s = ass.score_row(row)
    assert (s.citations_handed, s.citations_not_handed) == (handed, not_handed)
    assert (s.numbers_found, s.numbers_not_found) == numbers
    assert s.markers_unresolved == unresolved
    assert s.as_dict()["markers_unresolved"] == unresolved


@pytest.mark.parametrize("panel, expected", [
    ("SOURCES CITED\n2\n\n31 of 32 documents indexed\n\n1\nA.pdf\n2\nB.pdf", ["A.pdf", "B.pdf"]),
    ("SOURCES CITED\n1\n\n31 of 32 documents indexed\n\n1\nA.xlsx\nProject document", ["A.xlsx"]),
])
def test_parse_sources_list_skips_the_count_line(panel, expected):
    assert ass.parse_sources_list(panel) == expected


@pytest.mark.parametrize("panel, expected", [
    # one source, no coverage line, chip-headed entry: the count is not entry 1
    ("SOURCES CITED\n1\nHIGH\nA.pdf\nProject document", []),
    ("SOURCES CITED\n2\nLOW\nA.pdf\nMEDIUM\nB.pdf", []),
    # one numbered citation, no coverage line: the entry still parses
    ("SOURCES CITED\n1\n1\nA.pdf\nProject document", ["A.pdf"]),
    ("SOURCES CITED\n2\n1\nA.pdf\n2\nB.pdf", ["A.pdf", "B.pdf"]),
    # an empty panel has no count line
    ("SOURCES CITED\n\nNo citations for the current answer.", []),
])
def test_parse_sources_list_count_line_is_never_an_entry(panel, expected):
    assert ass.parse_sources_list(panel) == expected


@pytest.mark.parametrize("panel, numbers, not_handed", [
    # chip panel: a trailing "1" is a figure, not a marker naming the chip
    ("SOURCES CITED\n1\nHIGH\nA.pdf", (0, 1), 0),
    # numbered panel: the same "1" is a marker resolving to the handed doc
    ("SOURCES CITED\n1\n1\nA.pdf", (0, 0), 0),
])
def test_one_source_panel_row_score(panel, numbers, not_handed):
    row = {"id": "r", "answer": "The slab is cast in one pour 1.",
           "handed": [{"chunk_id": "c1", "doc_id": "d1", "doc_name": "A.pdf"}],
           "sources_list": ass.parse_sources_list(panel)}
    s = ass.score_row(row)
    assert (s.numbers_found, s.numbers_not_found) == numbers
    assert s.citations_not_handed == not_handed


# --- (c) overlap ----------------------------------------------------------------

@pytest.mark.parametrize("chunk_text, expected", [
    ("sulphate resisting cement concrete grade", 1.0),
    ("pipe bedding granular backfill", 0.0),
])
def test_overlap_of_sentence_words_in_chunk(chunk_text, expected):
    assert ass.overlap("Concrete grade sulphate resisting cement", chunk_text) == expected


def test_low_overlap_flags_the_row_and_floor_is_configurable():
    row = {"id": "r", "answer": "Pipes bedded on granular material [source: A.pdf].",
           "handed": [_chunk("c1", "A.pdf", "Concrete sulphate cement")]}
    assert "low_overlap" in ass.score_row(row, overlap_floor=0.2).reasons
    assert "low_overlap" not in ass.score_row(row, overlap_floor=0.0).reasons


def test_overlap_is_none_without_chunk_text():
    row = {"id": "r", "answer": "Pipes bedded [source: A.pdf].", "handed": [_chunk("c1", "A.pdf")]}
    s = ass.score_row(row)
    assert s.overlap_mean is None and s.text_available is False
    row["handed"][0]["text"] = "pipes bedded"
    assert ass.score_row(row).text_available is True


# --- output -------------------------------------------------------------------

@pytest.mark.parametrize("chunk_text, found, not_found", [
    ("Secret words 12", 1, 0),
    ("Secret words 13", 0, 1),
])
def test_output_carries_ids_and_counts_only(tmp_path, capsys, chunk_text, found, not_found):
    rows = [{"id": "Q9", "answer": "Secret words 12 [source: Private Name.pdf].",
             "question": "secret question", "handed": [{"chunk_id": "c1", "doc_name": "Private Name.pdf"}]}]
    p = tmp_path / "rows.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    t = tmp_path / "text.json"
    t.write_text(json.dumps({"c1": chunk_text}), encoding="utf-8")
    assert ass.main([str(p), "--chunk-text", str(t)]) == 0
    out = capsys.readouterr().out
    rec = json.loads(out.strip())
    assert rec["id"] == "Q9" and rec["citations_handed"] == 1
    assert (rec["numbers_found"], rec["numbers_not_found"]) == (found, not_found)
    for leaked in ("Secret", "Private", "secret question"):
        assert leaked not in out


@pytest.mark.parametrize("rep,shown", [(3, 3), (True, None), ("3", None), (None, None)])
def test_output_carries_row_and_rep_the_trace_page_joins_on(rep, shown):
    out = ass.score_row({"id": "Q7", "rep": rep, "answer": ""}).as_dict()
    assert (out["row"], out["rep"], out["id"]) == ("Q7", shown, "Q7")
