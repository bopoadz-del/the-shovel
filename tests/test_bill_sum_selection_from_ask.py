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
    # Arabic: the article and letter variants do not hide a measure word
    ("ما هي التكلفة الإجمالية للأنابيب؟", agg.MEASURE_AMOUNT),
    ("ما قيمه بنود السياج", agg.MEASURE_AMOUNT),
    ("ما هو الطول الكلي للأنابيب؟", agg.MEASURE_QUANTITY),
    ("كم عدد الأنابيب؟", "kept"),
])
def test_measure_from_ask(ask, measure):
    expected = "kept" if measure == "kept" else measure
    assert agg.measure_from_ask(ask, default="kept") == expected


@pytest.mark.parametrize("word, forms", [
    ("pipes", {"pipes", "pipe", "pip"}),
    ("trench", {"trench"}),
    # a word ending in a double s is not a plural of one ending in one s
    ("glass", {"glass"}),
    ("glasses", {"glasses", "glasse", "glass"}),
])
def test_plural_fold(word, forms):
    assert agg._forms(word) == forms


SLEEVES = [
    _row(RUNS, 1, "P1", "Pipeline sleeve", "m", "5", "50"),
    _row(RUNS, 1, "P2", "Pipe sleeve", "m", "7", "70"),
]


@pytest.mark.parametrize("ask, refs, total", [
    # a word inside a longer word does not match: "pipe" is not "pipeline"
    ("total length of pipe sleeves", ["P2"], 7),
    ("total length of pipeline sleeves", ["P1"], 5),
])
def test_a_word_inside_a_longer_word_does_not_match(tmp_path, ask, refs, total):
    path = tmp_path / "sleeves.xlsx"
    g.write_bill_workbook(SLEEVES, str(path))
    rows = agg.join_page_continuations(agg.read_sheet_rows(str(path)))
    out = agg._aggregate_bill(rows, ask=ask, terms="", section="", measure=agg.MEASURE_AMOUNT)
    assert [r["ref"] for r in out["rows"]] == refs
    assert out["sum"] == total


ARABIC = [
    _row(("صرف", "مواسير"), 1, "A1", "مواسير فخار قطر 150", "م", "20", "600"),
    _row(("صرف", "مواسير"), 1, "A2", "مواسير بلاستيك قطر 110", "م", "10", "200"),
    _row(("صرف", "غرف"), 2, "A3", "غرفة تفتيش", "عدد", "2", "900"),
]


@pytest.mark.parametrize("ask, refs, measure, total, ignored, unread", [
    # an Arabic cost question sums the amounts of the rows it names
    ("ما هي تكلفة مواسير فخار؟", ["A1"], agg.MEASURE_AMOUNT, 600, [], False),
    # an Arabic length question sums the quantities; an unknown word is named
    ("ما هو طول مواسير الحديد", ["A1", "A2"], agg.MEASURE_QUANTITY, 30, ["الحديد"], False),
    # no measure word: the quantity, said plainly to be the quantity
    ("كم مجموع مواسير فخار", ["A1"], agg.MEASURE_QUANTITY, 20, [], True),
], ids=["cost", "length", "no-measure"])
def test_an_arabic_ask(tmp_path, ask, refs, measure, total, ignored, unread):
    path = tmp_path / "ar.xlsx"
    g.write_bill_workbook(ARABIC, str(path))
    rows = agg.join_page_continuations(agg.read_sheet_rows(str(path)))
    out = agg._aggregate_bill(rows, ask=ask, terms="", section="", measure=agg.MEASURE_QUANTITY)
    assert out["selected_by"] == "ask"
    assert [r["ref"] for r in out["rows"]] == refs
    assert out["measure"] == measure
    assert out["sum"] == total
    assert out["ignored_ask_words"] == ignored
    said = "names no column, so this is the total quantity, not the amount" in out["statement"]
    assert said is unread
    assert out["measure_unread"] is unread
    assert [f["value"] for f in out["stated_figures"]] == [total]


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
    # an ask that names no word of the bill selects nothing, whatever the call
    # carries: the model's words never stand in for the user's
    ("and how many of those?", {"terms": "fence", "section": None}, "none", []),
    ("", {"terms": "pipe", "section": "runs"}, "none", []),
    # an ask that does name the bill's words wins over the call
    ("how many fence rows?", {"terms": "pipe", "section": None}, "ask", ["F1"]),
    # no ask at all (not a chat turn): the caller's own terms
    (None, {"terms": "fence", "section": None}, "call", ["F1"]),
], ids=["follow-up", "empty", "ask-wins", "no-ask"])
def test_the_calls_terms_never_stand_in_for_the_ask(rows, ask, call, selected_by, refs):
    out = agg._aggregate_bill(rows, ask=ask, measure=agg.MEASURE_QUANTITY, **call)
    assert out["selected_by"] == selected_by
    assert [r["ref"] for r in out["rows"]] == refs
    said = "No row was selected from the question" in out["statement"]
    assert said is (selected_by == "none")
    if selected_by == "none":
        assert out["count"] == 0 and out["sum"] is None and out["stated_figures"] == []


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
    ("how many paint rows?", "No row was selected from the question"),
    ("total quantity of rows in the drainage section?", "Total quantity by unit: "),
    ("how many clay pipes in the runs?", "Total quantity 103.5 m. The question names no column"),
])
def test_statement_for_no_match_mixed_units_and_no_measure(rows, ask, expect):
    out = agg._aggregate_bill(rows, ask=ask, terms="paint", section="", measure=agg.MEASURE_QUANTITY)
    assert expect in out["statement"]


@pytest.mark.parametrize("ask, named", [
    # the words that matched nothing are named, not dropped
    ("How many uPVC pipes are there?", ['"upvc"']),
    ("How many fence and paint rows?", ['"paint"']),
    ("How many pipes are there?", []),
])
def test_statement_names_the_words_it_did_not_use(rows, ask, named):
    text = agg._aggregate_bill(rows, ask=ask, terms="", section="", measure=agg.MEASURE_QUANTITY)["statement"]
    assert ("so not used:" in text) is bool(named)
    for word in named:
        assert word in text


@pytest.mark.parametrize("ask, figures", [
    ("total length of clay pipes in the runs", [(103.5, "103.5 m")]),
    ("total quantity in the drainage section", [(73 + 18.5 + 40 + 12, "143.5 m"), (9, "9 nr")]),
    ("how many fence rows?", [(120.0, "120 m")]),
    ("how many paint rows?", []),
])
def test_stated_figures_are_the_tools_totals(rows, ask, figures):
    out = agg._aggregate_bill(rows, ask=ask, terms="", section="", measure=agg.MEASURE_QUANTITY)
    assert [(f["value"], f["shown"]) for f in out["stated_figures"]] == figures


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
    ("and those?", "by=none"),
])
def test_log_line_says_where_the_selection_came_from(one_bill, caplog, ask, by):
    import logging

    with caplog.at_level(logging.INFO, logger=agg.__name__):
        agg.aggregate_project_bills("p", terms="pipe", ask=ask)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("bill_rows_aggregate"))
    assert by in line


# -- the exit check: the answer carries the tool's own figure ----------------

def _turn(*figures):
    from app.agents import answer_exit

    turn = answer_exit.Turn()
    answer_exit.note_tool_result("bill_rows_aggregate", {
        "name": "bill_rows_aggregate", "ok": True,
        "result": {"bills": [{"stated_figures": [{"value": v, "shown": s} for v, s in figures]}]},
    }, turn)
    return turn


@pytest.mark.parametrize("answer, figures, kept", [
    ("The clay pipes in the runs total 103.5 m (R1, R2, R4, R5).", [(103.5, "103.5 m")], True),
    ("The clay pipes in the runs total 112 m.", [(103.5, "103.5 m")], False),
    ("The fencing costs 3,000.", [(3000.0, "3,000")], True),
    ("The fencing costs 3,500.", [(3000.0, "3,000")], False),
    ("Item R1 is clay pipe.", [(1.0, "1 m")], False),
    ("The total is 112 m.", [], True),
], ids=["same", "different", "thousands", "thousands-different", "ref-is-not-a-figure", "no-tool"])
def test_exit_check_holds_the_tools_figure(answer, figures, kept):
    from app.agents import answer_exit

    out = answer_exit.hold_stated_figures(answer, _turn(*figures))
    assert (out == answer) is kept
    if not kept:
        assert out.startswith(answer)
        for _v, shown in figures:
            assert f"The tool's own figure is {shown}." in out
        assert "the tool's figure stands" in out


@pytest.mark.parametrize("answer, kept", [
    ("Total: 103.5 m.", True),
    ("Total: 99 m.", False),
])
def test_exit_check_runs_on_the_turns_result_and_end_event(answer, kept):
    from app.agents import answer_exit

    turn = _turn((103.5, "103.5 m"))
    out = answer_exit.check_result({"answer": answer, "sources": []}, turn)
    assert ("The tool's own figure is 103.5 m." in out["answer"]) is (not kept)
    end = answer_exit.check_end_event({"type": "end", "content": answer, "sources": []}, "", turn)
    assert ("The tool's own figure is 103.5 m." in end["content"]) is (not kept)


@pytest.mark.parametrize("payload, values", [
    ({"result": {"bills": [{"stated_figures": [{"value": 5, "shown": "5 m"}, {"value": True}]}]}}, [5.0]),
    ({"result": {"bills": [{"statement": "no figure"}]}}, []),
])
def test_only_numeric_stated_figures_are_recorded(payload, values):
    from app.agents import answer_exit

    assert [f["value"] for f in answer_exit._stated_figures(payload)] == values


def _nest(depth: int, leaf: dict) -> dict:
    node: dict = leaf
    for _ in range(depth):
        node = {"result": node}
    return node


@pytest.mark.parametrize("depth, values", [
    (3, [7.0]),
    (6, [7.0]),
    (8, []),
    (12, []),
])
def test_stated_figures_are_read_only_to_a_bounded_depth(depth, values):
    """The payload walk stops at a fixed depth: a figure nested deeper than
    any tool result is not read, and a deep (or self-referencing) payload
    cannot recurse without end."""
    from app.agents import answer_exit

    payload = _nest(depth, {"stated_figures": [{"value": 7, "shown": "7 m"}]})
    assert [f["value"] for f in answer_exit._stated_figures(payload)] == values


def test_a_self_referencing_payload_ends():
    from app.agents import answer_exit

    payload: dict = {"stated_figures": [{"value": 2, "shown": "2 nr"}]}
    payload["self"] = payload
    assert {f["value"] for f in answer_exit._stated_figures(payload)} == {2.0}


@pytest.mark.parametrize("result, values", [
    ({"selected_by": "ask", "rows": [{"q": 1}], "sum": 4.0, "unit": "m"}, [4.0]),
    ({"selected_by": "none", "rows": [{"q": 1}], "sum": 4.0, "unit": "m"}, []),
    ({"selected_by": "ask", "rows": [], "sum": 4.0, "unit": "m"}, []),
    ({"selected_by": "ask", "rows": [{"q": 1}], "sum": None, "sum_by_unit": {"m": 2.0, "nr": 3.0}}, [2.0, 3.0]),
    ({"selected_by": "none", "rows": [], "sum": None, "sum_by_unit": {"m": 2.0}}, []),
])
def test_a_result_that_selected_no_row_states_no_figure(result, values):
    """A bill result whose ask selected nothing, or that holds no row, states
    no figure even when a sum field is present; otherwise each total is
    stated (one per unit when the units differ)."""
    assert [f["value"] for f in agg.stated_figures(result)] == values


@pytest.mark.parametrize("texts, expected", [
    (("Cost  VALUE",), frozenset({"cost", "value"})),
    (("القيمة", "أسعار كلفة"), frozenset({"قيمه", "اسعار", "كلفه"})),
])
def test_word_tables_fold_case_article_and_letter_variants(texts, expected):
    """The ask-word tables hold each word in its looked-up form: case folded,
    the Arabic article dropped and letter variants made one."""
    assert agg._table(*texts) == expected
