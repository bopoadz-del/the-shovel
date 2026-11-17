"""Count/sum slice 2: the rows a count or sum selects come from the user's ask
by a deterministic rule, never from the words the model put in its call, and
the answer's statement names the rows used and carries the tool's own figure.

Every bill here is invented in the test; nothing names a real bill, row or
question. Each decision is driven by at least two inputs with different
outcomes.
"""

from __future__ import annotations

import re

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g


def _row(path, page, ref, desc, unit="", qty="", amount=""):
    fields = {"description": desc}
    for k, v in (("unit", unit), ("quantity", qty), ("total", amount)):
        if v:
            fields[k] = v
    return g.BillRow(section=path[-1], source_page=page, fields=fields,
                     extra={"Item": ref} if ref else {}, path=tuple(path))


RUNS = ("Drainage", "Runs")
CHAMBERS = ("Drainage", "Chambers")
FENCES = ("External", "Fencing")

#: A pipe run is split across pages 2 and 3 (description on 2, figures on 3);
#: one pipe row prints no quantity; one chamber row mentions a pipe.
BILL = [
    _row(RUNS, 1, "R1", "Clay pipe 150 diameter laid in trench", "m", "73", "2190"),
    _row(RUNS, 1, "R2", "Clay pipe 225 diameter laid in trench", "m", "18.5", "925"),
    _row(RUNS, 1, "R3", "Plastic pipe 110 diameter to gullies", "m", "40", "800"),
    _row(RUNS, 2, "R4", "Clay pipe 300 diameter laid in trench with", "m", "", ""),
    _row(RUNS, 3, "", "granular bed and surround", "", "12", "720"),
    _row(RUNS, 3, "R5", "Clay pipe bends, rate only", "m", "", ""),
    _row(CHAMBERS, 4, "C1", "Inspection chamber with pipe connections", "nr", "6", "3600"),
    _row(CHAMBERS, 4, "C2", "Rodding eye", "nr", "3", "450"),
    _row(FENCES, 5, "F1", "Timber post and rail fence", "m", "120", "3000"),
]


@pytest.fixture
def rows(tmp_path):
    path = tmp_path / "bill.xlsx"
    g.write_bill_workbook(BILL, str(path))
    return agg.join_page_continuations(agg.read_sheet_rows(str(path)))


@pytest.mark.parametrize("ask, terms, section, ignored", [
    # a heading word is a section word; a description word is a term
    ("What is the total length of clay pipes in the runs?", ["clay", "pipes"], ["runs"], []),
    # the same words in another order and case: the same selection
    ("RUNS: total length of PIPES, clay, please", ["clay", "pipes"], ["runs"], []),
    # a word the bill never prints selects nothing and is said to be ignored
    ("How many uPVC pipes are there?", ["pipes"], [], ["upvc"]),
    # a section word alone
    ("How many items in the chambers section?", [], ["chambers"], []),
], ids=["section+terms", "reworded", "ignored", "section-only"])
def test_selection_is_a_rule_over_the_ask_and_the_bill(rows, ask, terms, section, ignored):
    picked = agg.selection_from_ask(rows, ask)
    assert picked == {"terms": terms, "section_terms": section, "ignored": ignored}


@pytest.mark.parametrize("ask, measure", [
    ("What is the total cost of clay pipe?", agg.MEASURE_AMOUNT),
    ("What is the total value of the fencing?", agg.MEASURE_AMOUNT),
    ("What is the total length of clay pipe?", agg.MEASURE_QUANTITY),
    ("Sum the clay pipe", "kept"),
])
def test_measure_from_ask(ask, measure):
    expected = "kept" if measure == "kept" else measure
    assert agg.measure_from_ask(ask, default="kept") == expected


#: Two different wordings a model might put in its call for the same ask.
_MODEL_CALLS = [
    {"terms": "clay pipe trench", "section": "", "measure": agg.MEASURE_QUANTITY},
    {"terms": "pipe", "section": "drainage", "measure": agg.MEASURE_AMOUNT},
]


@pytest.mark.parametrize("ask, refs, total, unit, without", [
    # plural in the ask, singular in the bill; the split run counts once; the
    # chamber row mentions a pipe but is outside the runs section
    ("What is the total length of clay pipes in the runs?",
     ["R1", "R2", "R4", "R5"], 73 + 18.5 + 12, "m", ["R5"]),
    # no section: the chamber row that mentions a pipe is included
    ("How many pipe rows are there in total?",
     ["R1", "R2", "R3", "R4", "R5", "C1"], None, None, ["R5"]),
], ids=["sum-in-section", "count-everywhere"])
def test_same_ask_selects_same_rows_whatever_the_model_passes(rows, ask, refs, total, unit, without):
    outs = []
    for call in _MODEL_CALLS:
        out = agg._aggregate_bill(rows, ask=ask, **call)
        assert out["selected_by"] == "ask"
        outs.append(out)
    assert outs[0] == outs[1]
    out = outs[0]
    assert [r["ref"] for r in out["rows"]] == refs
    assert [w["ref"] for w in out["without_figure"]] == without
    if total is not None:
        assert out["sum"] == pytest.approx(total)
        assert out["unit"] == unit


@pytest.mark.parametrize("ask, call, selected_by, refs", [
    # an ask that names no word of the bill: the call's terms are used, and said so
    ("and how many of those?", {"terms": "fence", "section": None}, "call", ["F1"]),
    # an ask that does name the bill's words wins over the call
    ("how many fence rows?", {"terms": "pipe", "section": None}, "ask", ["F1"]),
])
def test_call_terms_only_when_the_ask_selects_nothing(rows, ask, call, selected_by, refs):
    out = agg._aggregate_bill(rows, ask=ask, measure=agg.MEASURE_QUANTITY, **call)
    assert out["selected_by"] == selected_by
    assert [r["ref"] for r in out["rows"]] == refs


@pytest.mark.parametrize("terms, fold, refs", [
    # the call's own words match exactly: a plural is not its singular
    ("pipes", False, []),
    # the ask's words fold plurals: the same word now matches
    ("pipes", True, ["R1", "R2", "R3", "R4", "R5", "C1"]),
    ("trench", False, ["R1", "R2", "R4"]),
])
def test_plural_folding_only_for_the_asks_selection(rows, terms, fold, refs):
    out = agg.aggregate(rows, terms=terms, fold=fold)
    assert [r["ref"] for r in out["rows"]] == refs


@pytest.mark.parametrize("value, shown", [(3000.0, "3,000"), (103.5, "103.5"), (0.125, "0.125"), (0.0, "0")])
def test_figure_format(value, shown):
    assert agg._fmt(value) == shown


def _numbers(text):
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


@pytest.mark.parametrize("ask, why, figure, listed, absent", [
    ("total length of clay pipes in the runs",
     'row(s) whose description contains "clay", "pipes" in the section "runs": ',
     "103.5 m", ["R1 (p. 1)", "R4 (p. 2-3)", "R5 (p. 3)"], ["C1"]),
    ("total cost of the fencing", 'row(s) in the section "fencing": ',
     "3,000", ["F1 (p. 5)"], ["R1"]),
], ids=["quantity", "amount"])
def test_statement_names_the_rows_and_carries_the_tools_figure(rows, ask, why, figure, listed, absent):
    out = agg._aggregate_bill(rows, ask=ask, terms="", section="", measure=agg.MEASURE_QUANTITY)
    text = out["statement"]
    assert text.startswith(f"{out['count']} {why}")
    assert f"Total {out['measure']} {figure}" in text
    assert out["sum"] in _numbers(text)
    for label in listed:
        assert label in text
    for ref in absent:
        assert ref not in text


@pytest.mark.parametrize("ask, expect", [
    ("how many paint rows?", "no row of this bill matches"),
    ("how many rows in the drainage section?", "Total quantity by unit: "),
])
def test_statement_for_no_match_and_mixed_units(rows, ask, expect):
    out = agg._aggregate_bill(rows, ask=ask, terms="paint", section="", measure=agg.MEASURE_QUANTITY)
    assert expect in out["statement"]


@pytest.fixture
def one_bill(tmp_path, monkeypatch):
    from app.core import bill_derived, projects

    path = str(tmp_path / "bill.xlsx")
    g.write_bill_workbook(BILL, path)
    docs = [{"id": "d1", "original_name": "bill.xlsx", "path": path,
             "metadata": {"derivation": bill_derived.DERIVATION, "derived_from": "s1"}}]
    monkeypatch.setattr(projects, "list_documents", lambda pid: docs)
    monkeypatch.setattr(projects, "materialize_document_path", lambda d: d["path"])


@pytest.mark.asyncio
@pytest.mark.parametrize("ask, refs", [
    ("What is the total length of clay pipes in the runs?", ["R1", "R2", "R4", "R5"]),
    ("How many items in the chambers section?", ["C1", "C2"]),
])
async def test_tool_handler_selects_from_the_users_message(one_bill, ask, refs):
    from app.agents.base.tools import handle_bill_rows_aggregate
    from app.agents.core.tool_registry import ToolCall

    results = []
    for args in _MODEL_CALLS:
        out = await handle_bill_rows_aggregate(ToolCall(
            agent=None, name="bill_rows_aggregate", args=dict(args), project_id="p", user_message=ask,
        ))
        assert out["ok"] is True
        results.append(out["result"]["bills"][0])
    assert results[0]["rows"] == results[1]["rows"]
    assert results[0]["statement"] == results[1]["statement"]
    assert [r["ref"] for r in results[0]["rows"]] == refs


@pytest.mark.parametrize("ask, by", [
    ("total length of clay pipes in the runs", "by=ask"),
    ("and those?", "by=call"),
])
def test_log_line_says_where_the_selection_came_from(one_bill, caplog, ask, by):
    import logging

    with caplog.at_level(logging.INFO, logger=agg.__name__):
        agg.aggregate_project_bills("p", terms="pipe", ask=ask)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("bill_rows_aggregate"))
    assert by in line
