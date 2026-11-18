"""Offline answer-against-its-sources scorer: each rule on two inputs
(one that holds, one that does not). Every input is built here."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "answer_source_score.py"
_spec = importlib.util.spec_from_file_location("answer_source_score", _PATH)
ass = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = ass
_spec.loader.exec_module(ass)


def _chunk(cid: str, name: str, text: str = "") -> dict:
    return {"chunk_id": cid, "doc_id": "d-" + cid, "doc_name": name, "text": text}


# --- (a) numbers --------------------------------------------------------------

def test_number_in_cited_chunk_is_found():
    row = {"id": "r", "answer": "The wall is 250 mm thick [source: Spec A.pdf].",
           "handed": [_chunk("c1", "Spec A.pdf", "Walls shall be 250 mm thick.")]}
    s = ass.score_row(row)
    assert (s.numbers_found, s.numbers_not_found) == (1, 0)
    assert s.flag == "ok"


def test_number_absent_from_cited_chunk_is_not_found():
    row = {"id": "r", "answer": "The wall is 300 mm thick [source: Spec A.pdf].",
           "handed": [_chunk("c1", "Spec A.pdf", "Walls shall be 250 mm thick.")]}
    s = ass.score_row(row)
    assert (s.numbers_found, s.numbers_not_found) == (0, 1)
    assert "number_not_found" in s.reasons and s.flag == "low"


def test_number_from_tool_result_is_found():
    row = {"id": "r", "answer": "Volume is 945 m3.", "tool_results": ['{"result": 945.0}']}
    assert ass.score_row(row).numbers_found == 1


def test_number_only_in_uncited_chunk_is_not_found():
    # handed but not cited: the rule reads cited chunks, not the whole turn
    row = {"id": "r", "answer": "Volume is 945 m3.", "handed": [_chunk("c1", "X.pdf", "945")]}
    assert ass.score_row(row).numbers_not_found == 1


def test_thousands_commas_and_decimals_compare_as_values():
    assert ass.numbers_in("SAR 4,984,120 and 0.125 l") == [4984120.0, 0.125]
    assert ass.numbers_in("4984120") == [4984120.0]


def test_identifier_digits_are_not_figures():
    assert ass.numbers_in("ref IP-INF-053 item I233.3 sheet D/13/2") == []
    assert ass.numbers_in("ref 53 item 233.3") == [53.0, 233.3]


def test_numbered_marker_is_not_a_figure_when_panel_numbers_it():
    answer = "Leakage is 0.125 litres per 30 minutes 2.\nNext line."
    assert ass.answer_numbers(answer, ["A.pdf", "B.pdf"]) == [0.125, 30.0]
    # no panel: the trailing 2 is an ordinary number
    assert ass.answer_numbers(answer, []) == [0.125, 30.0, 2.0]


def test_number_inside_a_named_citation_is_not_a_figure():
    assert ass.answer_numbers("Rate 585 [source: Bill 13 page 2.pdf]") == [585.0]
    assert ass.answer_numbers("Rate 585 per Bill 13") == [585.0, 13.0]


# --- (b) citations handed -----------------------------------------------------

def test_named_citation_of_handed_document_counts_handed():
    row = {"id": "r", "answer": "Grade C40 [source: Spec A.pdf].",
           "handed": [_chunk("c1", "Spec A.pdf", "grade C40")]}
    s = ass.score_row(row)
    assert (s.citations_handed, s.citations_not_handed) == (1, 0)


def test_named_citation_of_unhanded_document_counts_not_handed():
    row = {"id": "r", "answer": "Grade C40 [source: Spec B.pdf].",
           "handed": [_chunk("c1", "Spec A.pdf", "grade C40")]}
    s = ass.score_row(row)
    assert (s.citations_handed, s.citations_not_handed) == (0, 1)
    assert "citation_not_handed" in s.reasons


def test_ellipsis_name_matches_on_both_ends():
    assert ass.name_matches("IP-BOQ-07…ities (Priced).xlsx", "IP-BOQ-07 Bill of Quantities (Priced).xlsx")
    assert not ass.name_matches("IP-BOQ-07…ities (Priced).xlsx", "IP-BOQ-07 Bill of Quantities (Draft).xlsx")


def test_keyed_citation_matches_chunk_id():
    handed = [_chunk("c7", "A.pdf", "text")]
    hit = ass.find_citations("Value 5 (chunk_id=c7).")
    miss = ass.find_citations("Value 5 (chunk_id=c8).")
    assert len(ass.cited_chunks(hit[0], handed)) == 1
    assert ass.cited_chunks(miss[0], handed) == []


def test_marker_resolves_through_sources_list():
    cites = ass.find_citations("Tested at 2 m head 1.\n", ["Method.pdf"])
    assert [(c.kind, c.target) for c in cites] == [("marker", "Method.pdf")]
    # a marker beyond the panel's numbering is not a citation
    assert ass.find_citations("Tested at 2 m head 3.\n", ["Method.pdf"]) == []


def test_mid_sentence_small_number_is_not_a_marker():
    assert ass.find_citations("Test at 2 m head above soffit.", ["A.pdf", "B.pdf"]) == []
    assert len(ass.find_citations("Test above soffit 2.", ["A.pdf", "B.pdf"])) == 1


def test_parse_sources_list_skips_the_count_line():
    panel = "SOURCES CITED\n2\n\n31 of 32 documents indexed\n\n1\nA.pdf\n2\nB.pdf"
    assert ass.parse_sources_list(panel) == ["A.pdf", "B.pdf"]
    single = "SOURCES CITED\n1\n\n31 of 32 documents indexed\n\n1\nA.xlsx\nProject document"
    assert ass.parse_sources_list(single) == ["A.xlsx"]


# --- (c) overlap ----------------------------------------------------------------

def test_overlap_high_when_sentence_words_are_in_chunk():
    v = ass.overlap("Concrete grade sulphate resisting cement", "sulphate resisting cement concrete grade")
    assert v == 1.0


def test_overlap_low_when_sentence_words_are_not_in_chunk():
    v = ass.overlap("Concrete grade sulphate resisting cement", "pipe bedding granular backfill")
    assert v == 0.0


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

def test_output_carries_ids_and_counts_only(tmp_path, capsys):
    rows = [{"id": "Q9", "answer": "Secret words 12 [source: Private Name.pdf].",
             "question": "secret question", "handed": [{"chunk_id": "c1", "doc_name": "Private Name.pdf"}]}]
    p = tmp_path / "rows.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    t = tmp_path / "text.json"
    t.write_text(json.dumps({"c1": "Secret words 12"}), encoding="utf-8")
    assert ass.main([str(p), "--chunk-text", str(t)]) == 0
    out = capsys.readouterr().out
    rec = json.loads(out.strip())
    assert rec["id"] == "Q9" and rec["numbers_found"] == 1 and rec["citations_handed"] == 1
    for leaked in ("Secret", "Private", "secret question"):
        assert leaked not in out
