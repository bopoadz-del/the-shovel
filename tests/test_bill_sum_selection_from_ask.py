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
    assert picked == {"terms": terms, "section_terms": section, "sizes": [], "ignored": ignored}


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


@pytest.mark.parametrize("ask, refs, measure, total", [
    # the article on both words: "المواسير الفخار" selects the clay pipes
    ("ما هي تكلفة المواسير الفخار؟", ["A1"], agg.MEASURE_AMOUNT, 600),
    # a one-letter prefix and the article: "والبلاستيك" selects the plastic pipe
    ("ما طول مواسير والبلاستيك", ["A2"], agg.MEASURE_QUANTITY, 10),
], ids=["article", "prefix-and-article"])
def test_an_arabic_ask_word_with_its_article_or_a_prefix_selects(tmp_path, ask, refs, measure, total):
    """Before this fold an ask word carrying the article or a prefix matched
    no bill word and selected nothing (it failed closed)."""
    path = tmp_path / "ar.xlsx"
    g.write_bill_workbook(ARABIC, str(path))
    rows = agg.join_page_continuations(agg.read_sheet_rows(str(path)))
    out = agg._aggregate_bill(rows, ask=ask, terms="", section="", measure=agg.MEASURE_QUANTITY)
    assert out["selected_by"] == "ask"
    assert [r["ref"] for r in out["rows"]] == refs
    assert out["measure"] == measure
    assert out["sum"] == total
    assert out["ignored_ask_words"] == []


@pytest.mark.parametrize("word, has, lacks", [
    ("بالفخار", {"بالفخار", "فخار"}, set()),
    ("والبلاستيك", {"والبلاستيك", "بلاستيك"}, set()),
    # a word that starts with a prefix letter keeps itself; a stem too short
    # to be a word is not added
    ("بلاستيك", {"بلاستيك"}, set()),
    ("وزن", {"وزن"}, {"زن"}),
], ids=["article", "prefix-article", "starts-with-prefix-letter", "too-short"])
def test_arabic_forms_fold_the_article_and_a_prefix(word, has, lacks):
    forms = agg._forms(word)
    assert has <= forms
    assert not (lacks & forms)


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


# -- the exit check: the tool's statement leads; the model's arithmetic never --

ASK = "What is the total length of clay pipes in the runs?"
NO_SELECT_ASK = "What is the total length of copper ducts?"


def _bill_payload(bill_rows, ask):
    """A ``bill_rows_aggregate`` tool payload as the handler returns it,
    computed by the real aggregation over ``bill_rows`` for ``ask``."""
    res = agg._aggregate_bill(bill_rows, terms=None, section=None, measure="quantity", ask=ask)
    return {"name": "bill_rows_aggregate", "ok": True, "result": {"status": "success", "bills": [
        {"document_id": "d1", "filename": "bill.xlsx", "status": "success", **res}]}}


def _turn_for(payloads, ask=ASK):
    from app.agents import answer_exit

    turn = answer_exit.Turn()
    turn.question = ask
    for payload in payloads:
        answer_exit.note_tool_result("bill_rows_aggregate", payload, turn)
    return turn


def _lead(turn):
    from app.agents import answer_exit

    return "\n\n".join(answer_exit.computed_leads(turn))


def _leads(content, lead):
    """``content`` opens with ``lead``; the exit contract may turn the lead's
    ``(source: ...)`` into a numbered citation marker, nothing else."""
    norm = lambda t: re.sub(r" \(source: [^)]*\)| \[\d+\]", "", t)  # noqa: E731
    return norm(content).startswith(norm(lead))


@pytest.mark.parametrize("reply, removed", [
    ("The clay pipes in the runs come to 103.5 m over rows R1, R2 and R4.", False),
    ("Clay pipe runs total 112 m. Rows R1, R2 and R4 were used.", True),
], ids=["right", "different-total"])
def test_the_tools_statement_leads_and_a_foreign_total_is_removed(rows, reply, removed):
    from app.agents import answer_exit

    turn = _turn_for([_bill_payload(rows, ASK)])
    lead = _lead(turn)
    assert "103.5" in lead
    out = answer_exit.hold_stated_figures(reply, turn)
    assert out.startswith(lead)
    assert ("112" in out) is False
    assert out.count(answer_exit.FIGURE_REMOVED) == (1 if removed else 0)
    if not removed:
        assert out == f"{lead}\n\n{reply}"


@pytest.mark.parametrize("reply", [
    "Here is what the bill says about drainage.",
    "No figure could be given for that.",
])
def test_a_turn_without_a_computed_result_is_untouched(reply):
    from app.agents import answer_exit

    turn = answer_exit.Turn()
    turn.question = ASK
    answer_exit.note_tool_result("search_documents", {"ok": True, "result": {"passages": ["p 12"]}}, turn)
    assert answer_exit.hold_stated_figures(reply, turn) == reply
    assert answer_exit.hold_stated_figures(reply + " Total 99 m.", turn) == reply + " Total 99 m."


@pytest.mark.parametrize("reply, replaced", [
    ("The copper ducts total 64 m.", True),
    ("No copper ducts were found in this bill.", False),
], ids=["states-a-total", "states-none"])
def test_a_total_when_the_tool_selected_nothing_gets_the_plain_statement(rows, reply, replaced):
    from app.agents import answer_exit

    payload = _bill_payload(rows, NO_SELECT_ASK)
    assert payload["result"]["bills"][0]["selected_by"] == "none"
    turn = _turn_for([payload], NO_SELECT_ASK)
    out = answer_exit.hold_stated_figures(reply, turn)
    if replaced:
        assert out.startswith("No row was selected from the question")
        assert "64" not in out
    else:
        assert out == reply


@pytest.mark.parametrize("first_ask, leads", [
    (NO_SELECT_ASK, ["103.5"]),
    ("fencing", ["120", "103.5"]),
    (ASK, ["103.5"]),
], ids=["first-selected-nothing", "two-selections", "same-ask-twice"])
def test_every_figure_stating_call_leads_in_call_order_once(rows, first_ask, leads):
    """A call that selected nothing does not lead; every call that stated a
    figure leads, in call order; an identical statement is shown once."""
    from app.agents import answer_exit

    turn = _turn_for([_bill_payload(rows, first_ask), _bill_payload(rows, ASK)])
    stated = answer_exit.computed_leads(turn)
    assert len(stated) == len(leads)
    for lead, figure in zip(stated, leads):
        assert figure in lead
    out = answer_exit.hold_stated_figures("The runs carry 103.5 m of clay pipe.", turn)
    assert out.startswith("\n\n".join(stated))
    assert sum(out.count(lead) for lead in stated) == len(stated)
    assert answer_exit.FIGURE_REMOVED not in out


@pytest.mark.parametrize("with_turn", [False, True], ids=["no-turn", "turn-without-results"])
def test_no_turn_or_no_computed_result_has_no_lead(with_turn):
    """Outside a turn, or on a turn no computed result reached, nothing leads."""
    from app.agents import answer_exit

    turn = answer_exit.Turn() if with_turn else None
    assert answer_exit.computed_leads(turn) == []


@pytest.mark.parametrize("reply, kept", [
    ("The clay runs are on rows R1, R2 and R4. That is 103.5 m.", True),
    ("The clay runs are on rows R1, R2 and R4. Re-added they make 112 m.", False),
], ids=["all-figures-the-tools", "one-foreign-figure"])
def test_commentary_is_whole_or_nothing(rows, reply, kept):
    """The model's text is kept whole when every figure in it is the tool's
    or the user's, and withheld whole (never cut sentence by sentence) when
    any one figure is its own."""
    from app.agents import answer_exit

    turn = _turn_for([_bill_payload(rows, ASK)])
    out = answer_exit.hold_stated_figures(reply, turn)
    assert (reply in out) is kept
    assert ("rows R1, R2 and R4" in out) is kept
    assert (answer_exit.FIGURE_REMOVED in out) is (not kept)


@pytest.mark.parametrize("answer, kept", [
    ("Total: 103.5 m.", True),
    ("Total: 99 m.", False),
])
def test_exit_check_runs_on_the_turns_result_and_end_event(rows, answer, kept):
    from app.agents import answer_exit

    turn = _turn_for([_bill_payload(rows, ASK)])
    out = answer_exit.check_result({"answer": answer, "sources": []}, turn)
    assert _leads(out["answer"], _lead(turn))
    assert (answer_exit.FIGURE_REMOVED in out["answer"]) is (not kept)
    end = answer_exit.check_end_event({"type": "end", "content": answer, "sources": []}, "", turn)
    assert _leads(end["content"], _lead(turn))
    assert (answer_exit.FIGURE_REMOVED in end["content"]) is (not kept)


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


@pytest.mark.parametrize("value, as_list", [
    (2, False),
    (9.5, True),
])
def test_a_self_referencing_payload_ends(value, as_list):
    """A payload that holds itself (in a dict field or a list) ends and still
    reads its figure."""
    from app.agents import answer_exit

    payload: dict = {"stated_figures": [{"value": value, "shown": f"{value} nr"}]}
    payload["self"] = [payload] if as_list else payload
    assert {f["value"] for f in answer_exit._stated_figures(payload)} == {float(value)}


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


# -- the live shape through the real stream route (TESTER 18 Nov: the headline
# -- said 5,858, the reply's own check line and the tool's sum said 4,858) ----

_LIVE_ROWS = (633, 488, 828, 957, 844, 221, 552, 200, 135)
_OTHER_ROWS = (1250, 740, 310)


def _invented_bill(tmp_path, quantities):
    bill = [_row(RUNS, 1 + i // 4, f"P{i + 1}", f"Clay pipe run {i + 1} laid in trench", "m", str(q), "")
            for i, q in enumerate(quantities)]
    bill.append(_row(FENCES, 9, "F1", "Timber post and rail fence", "m", "120", "3000"))
    path = tmp_path / "invented.xlsx"
    g.write_bill_workbook(bill, str(path))
    return agg.join_page_continuations(agg.read_sheet_rows(str(path)))


def _live_reply(quantities, headline):
    return f"The total length of clay pipe in the runs is {headline:,} m."


async def _through_the_stream_route(payload, ask, reply):
    """The reply streamed token by token through ``with_answer_exit`` (the
    wrapper every chat stream route returns), the tool's result recorded on
    the turn as the agent loop records it; returns the decoded end frame."""
    from app.agents import answer_exit
    from app.routers.exit_frames import with_answer_exit
    from app.routers.hat_frames import _decode, _encode

    async def frames():
        turn = answer_exit.current_turn()
        turn.question = ask
        answer_exit.note_tool_result("bill_rows_aggregate", payload, turn)
        for piece in reply.split(" "):
            yield _encode({"type": "token", "content": piece + " "})
        yield _encode({"type": "end", "sources": []})

    decoded = [_decode(f) async for f in with_answer_exit(frames())]
    ends = [e for e in decoded if e and e.get("type") == "end"]
    assert len(ends) == 1
    return ends[0]


@pytest.mark.parametrize("quantities, headline", [
    (_LIVE_ROWS, lambda q: sum(q) + 1000),
    (_OTHER_ROWS, lambda q: sum(q) + 10),
    # a wrong total that equals a ROW's quantity: a payload number, not a figure the tool stated
    (_LIVE_ROWS, lambda q: max(q)),
    (_OTHER_ROWS, lambda q: max(q)),
    (_LIVE_ROWS, sum),
    (_OTHER_ROWS, sum),
], ids=["live-off-by-a-digit", "other-off", "live-row-quantity", "other-row-quantity",
        "live-right", "other-right"])
async def test_the_live_shape_through_the_stream_route(tmp_path, quantities, headline):
    from app.agents import answer_exit

    bill_rows = _invented_bill(tmp_path, quantities)
    payload = _bill_payload(bill_rows, ASK)
    right, said = sum(quantities), headline(quantities)
    end = await _through_the_stream_route(payload, ASK, _live_reply(quantities, said))
    content = end["content"]
    lead = _lead(_turn_for([payload]))
    assert f"{right:,}" in lead
    assert _leads(content, lead)
    wrong = said != right
    assert (_live_reply(quantities, said) in content) is (not wrong)
    assert content.count(answer_exit.FIGURE_REMOVED) == (1 if wrong else 0)


@pytest.mark.parametrize("correction, withheld", [
    # the right figure, then a second figure for the same claim: withheld whole
    (lambda right: f"The runs total {right + 7:,} m. Correction: the runs total {right:,} m.", True),
    (lambda right: f"The runs total {right:,} m. Correction: the runs total {right + 7:,} m.", True),
    # the one figure stated once: kept
    (lambda right: f"The runs total {right:,} m.", False),
], ids=["wrong-then-right", "right-then-wrong", "stated-once"])
@pytest.mark.parametrize("quantities", [_LIVE_ROWS, _OTHER_ROWS], ids=["live", "other"])
async def test_a_self_correcting_reply_is_withheld_whole_through_the_stream_route(
        tmp_path, quantities, correction, withheld):
    from app.agents import answer_exit

    payload = _bill_payload(_invented_bill(tmp_path, quantities), ASK)
    reply = correction(sum(quantities))
    content = (await _through_the_stream_route(payload, ASK, reply))["content"]
    assert (reply in content) is (not withheld)
    assert content.count(answer_exit.FIGURE_REMOVED) == (1 if withheld else 0)


@pytest.mark.parametrize("text, numbers", [
    ("1,234.5 m", {1234.5}),
    ("1 234.5 m", {1234.5}),
    ("1 234 m and 1 234 m", {1234.0}),
    ("١٬٢٣٤٫٥ m", {1234.5}),
    ("12 and 345", {12.0, 345.0}),
    ("1.8 and 18", {1.8, 18.0}),
], ids=["comma", "space", "thin-spaces", "arabic", "not-grouped", "two-numbers"])
def test_every_thousands_separator_reads_as_one_number(text, numbers):
    from app.agents import answer_exit

    assert answer_exit._answer_numbers(text) == numbers


@pytest.mark.parametrize("reply", [
    "The drainage section lists 9 runs of clay pipe and a fence.",
    "The bill has no copper ducts; 64 m was mentioned in the spec.",
])
async def test_a_non_computed_turn_through_the_stream_route_is_untouched(reply):
    from app.agents import answer_exit
    from app.routers.exit_frames import with_answer_exit
    from app.routers.hat_frames import _decode, _encode

    async def frames():
        answer_exit.current_turn().question = "What does the bill hold?"
        for piece in reply.split(" "):
            yield _encode({"type": "token", "content": piece + " "})
        yield _encode({"type": "end", "sources": []})

    decoded = [_decode(f) async for f in with_answer_exit(frames())]
    ends = [e for e in decoded if e and e.get("type") == "end"]
    assert ends[0]["content"].strip() == reply
    assert answer_exit.FIGURE_REMOVED not in ends[0]["content"]


# -- the exit check's own bounds and the renderer branch ---------------------

@pytest.mark.parametrize("depth, statements", [
    (2, ["two rows: total 7 m."]),
    (5, ["two rows: total 7 m."]),
    (9, []),
    (14, []),
])
def test_plain_statements_are_read_only_to_a_bounded_depth(depth, statements):
    from app.agents import answer_exit

    leaf = {"statement": "two rows: total 7 m.", "stated_figures": []}
    assert answer_exit._plain_statements(_nest(depth, leaf)) == statements


@pytest.mark.parametrize("figures, reply, restated", [
    ([{"value": 18.0, "shown": "18 m"}], "18 m, or rather 20 m", True),
    ([{"value": 18.0, "shown": "18 m"}], "18 m over 4 rows", False),
    # the tool stated two figures in the unit: two figures are not a correction
    ([{"value": 18.0, "shown": "18 m"}, {"value": 20.0, "shown": "20 m"}], "18 m and 20 m", False),
], ids=["two-for-one", "one-for-one", "two-for-two"])
def test_restates_a_claim(figures, reply, restated):
    from app.agents import answer_exit

    assert answer_exit._restates_a_claim(reply, figures) is restated


@pytest.mark.parametrize("tool, rendered", [
    # a declared computed tool: the #392 renderer words the lead
    ("bill_rows_aggregate", True),
    # a tool with no computed declaration: the tool's own statement sentence
    ("an_undeclared_tool", False),
], ids=["declared", "undeclared"])
def test_the_lead_is_rendered_from_the_tools_declared_statement(rows, tool, rendered):
    from app.agents import answer_exit

    payload = _bill_payload(rows, ASK)
    res = {"tool": tool, "payload": payload,
           "figures": answer_exit._stated_figures(payload),
           "statements": answer_exit._plain_statements(payload)}
    lead = answer_exit._computed_lead(res, ASK)
    assert ("**103.5 m**" in lead) is rendered
    assert (lead == "\n".join(res["statements"])) is (not rendered)
