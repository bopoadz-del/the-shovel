"""One answer per turn from several computed calls (live reading of #392).

A turn may call one computed tool several times. The answer LEADS with a
result; a result identical to one already shown is one result; a call with no
figure is mentioned only when no result of the same tool answered the same
part of the question (same tool, its argument words contained in the
result's), and then at the end in one plain line; with no result at all the
no-figure calls are the answer. Every bill here is invented; two inputs per
decision; through the real recovery (``_recover_answer_from_tool_messages``
and ``_postprocess_answer``).
"""
from __future__ import annotations

import json

import pytest

from app.agents import runtime
from app.agents.core import computed_answer
from app.agents.core.computed_answer import LABELS, Call, covers, selection

EN = LABELS["en"]
TOOL = "bill_rows_aggregate"


def _bill(filename: str, terms: list[str], unit: str, rows: list[tuple[str, int, float]]) -> dict:
    entries = [{"ref": ref, "pages": [page], "section": ["Invented section"],
                "description": " ".join(terms) + f" item {ref}", "unit": unit,
                "quantity": qty} for ref, page, qty in rows]
    total = sum(q for _r, _p, q in rows)
    return {"status": "success", "bills": [{
        "document_id": f"doc-{filename}", "filename": filename, "derived_from": None,
        "status": "success", "measure": "quantity", "terms": terms, "section_terms": [],
        "rows_scanned": 40, "count": len(rows), "sum": total, "unit": unit,
        "sum_by_unit": {unit: total}, "without_figure": [], "rows": entries}]}


NO_FIGURE = {"status": "success", "bills": []}

BILLS = [
    (_bill("bill-alpha.xlsx", ["kerb"], "m", [("K.1", 3, 120.5), ("K.4", 4, 79.5)]),
     {"terms": "kerb"}, {"terms": "kerb", "section": "drainage"}, "**200 m**"),
    (_bill("bill-beta.xlsx", ["duct", "pvc"], "nr", [("D.2", 7, 11.0), ("D.3", 7, 6.0)]),
     {"terms": ["PVC", "duct"]}, {"terms": ["duct", "pvc"], "section": "ducts"}, "**17 nr**"),
]


def _turn(question: str, *calls: tuple[dict, dict]) -> list[dict]:
    """A turn: the question, then one assistant call + one tool reply per
    (arguments, result) pair, each with its own call id."""
    msgs: list[dict] = [{"role": "system", "content": "You answer questions."},
                        {"role": "user", "content": question}]
    for i, (args, result) in enumerate(calls):
        cid = f"call-{i}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": cid, "type": "function",
            "function": {"name": TOOL, "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "name": TOOL,
                     "content": json.dumps({"name": TOOL, "ok": True, "result": result})})
    return msgs


def _answers(messages: list[dict]) -> list[str]:
    return [runtime._recover_answer_from_tool_messages("", messages),
            runtime._postprocess_answer("", None, messages)]


@pytest.mark.parametrize("result,first_args,result_args,figure", BILLS)
def test_no_figure_then_a_result_for_the_same_part_shows_the_result_alone(
        result, first_args, result_args, figure):
    question = "how many and what total?"
    for out in _answers(_turn(question, (first_args, NO_FIGURE), (result_args, result))):
        assert out.startswith("Bill row count and total:"), out
        assert figure in out
        assert EN["not_returned"] not in out and EN["no_figure"] not in out, out
        assert out.count("Bill row count and total") == 1, out
        assert "computed results in this turn" not in out


@pytest.mark.parametrize("result,_a,args,figure", BILLS)
def test_the_same_result_twice_is_shown_once(result, _a, args, figure):
    for out in _answers(_turn("total please", (args, result), (dict(reversed(list(args.items()))), result))):
        assert out.count(figure) == 1, out
        assert out.count("Bill row count and total") == 1
        assert "computed results in this turn" not in out


def test_two_different_results_are_both_shown_labelled_by_what_was_asked():
    (a, a_args, _x, a_fig), (b, b_args, _y, b_fig) = BILLS
    question = "kerb total, and the pvc duct total?"
    for first, second in (((a_args, a), (b_args, b)), ((b_args, b), (a_args, a))):
        for out in _answers(_turn(question, first, second)):
            assert out.startswith(EN["several"].format(n=2)), out
            assert a_fig in out and b_fig in out
            # A string argument the user typed is echoed; a list argument is
            # not (pinned in test_computed_answer_parts), and that result is
            # labelled by its own statement of what was selected.
            assert f"{EN['asked']}: kerb" in out
            assert "holds duct, pvc: **2**" in out
            assert EN["not_returned"] not in out


@pytest.mark.parametrize("args_one,args_two,expected_blocks", [
    ({"terms": "gully"}, {"terms": "Gully "}, 1),
    ({"terms": "gully"}, {"terms": "manhole"}, 2),
])
def test_no_figure_twice_is_the_plain_answer_once_per_part(args_one, args_two, expected_blocks):
    question = "count the gully and manhole rows"
    for out in _answers(_turn(question, (args_one, NO_FIGURE), (args_two, NO_FIGURE))):
        assert out.count(f"{EN['not_returned']}: {EN['no_figure']}") == expected_blocks, out
        # The no-figure calls are the answer itself, not a trailing line.
        assert out.startswith("Bill row count and total:"), out
        assert EN["unanswered"].split("{")[0] not in out, out
        assert "unable to generate" not in out.lower()
        assert "computed results in this turn" not in out


@pytest.mark.parametrize("result,_a,args,figure", BILLS)
def test_an_unanswered_part_comes_last_in_one_plain_line(result, _a, args, figure):
    question = "kerb, pvc duct and gully totals?"
    turn = _turn(question, ({"terms": "gully"}, NO_FIGURE), (args, result),
                 ({"terms": "gully"}, NO_FIGURE))
    for out in _answers(turn):
        assert out.startswith("Bill row count and total:"), out
        last = out.rstrip().split("\n")[-1]
        assert last == EN["unanswered"].format(parts=(
            f"Bill row count and total ({EN['asked']}: gully), "
            f"{EN['not_returned']}: {EN['no_figure']}")), out
        assert out.index(figure) < out.index(last)
        assert out.count(EN["no_figure"]) == 1


def test_inside_one_call_the_bills_that_returned_lead():
    good = _bill("bill-gamma.xlsx", ["gully"], "nr", [("G.1", 2, 3.0)])["bills"][0]
    failed = {"status": "error", "reason": "derived_sheet_is_not_a_readable_workbook",
              "filename": "bill-failed.xlsx", "terms": ["gully"]}
    for bills in ([failed, good], [good, failed]):
        out = runtime._recover_answer_from_tool_messages("", _turn(
            "count the gullies", ({"terms": "gully"}, {"status": "success", "bills": bills})))
        assert out.index("**3 nr**") < out.index(EN["not_returned"]) and "bill-failed.xlsx" in out


@pytest.mark.parametrize("asked,expected", [
    ({"terms": ["Kerb", "PVC"], "n": 2}, frozenset({"kerb", "pvc", "2"})),
    ({"x": {"y": ["1.8 x 1.8m"]}, "flag": True, "none": None}, frozenset({"1.8", "x", "1.8m"})),
])
def test_a_selection_is_the_case_folded_words_of_the_argument_values(asked, expected):
    assert selection(asked) == expected


@pytest.mark.parametrize("other_tool,other_args,covered", [
    (TOOL, {"terms": "kerb"}, True),
    (TOOL, {"terms": "kerb", "section": "roads"}, False),
    ("zz_other_invented_tool", {"terms": "kerb"}, False),
    (TOOL, {}, True),
])
def test_a_result_covers_a_call_of_the_same_tool_whose_words_it_holds(other_tool, other_args, covered):
    result = Call(tool=TOOL, heading="H", asked={"terms": "Kerb", "section": "drainage"})
    assert covers(result, Call(tool=other_tool, heading="H", asked=other_args)) is covered


FAILED = {"status": "error", "reason": "no_derived_bill_sheet"}


@pytest.mark.parametrize("first,answers", [
    # Its part answered by the result: nothing waits.
    (({"terms": "kerb"}, NO_FIGURE), True),
    # A part no result answered, and the call said nothing: wait.
    (({"terms": "gully"}, NO_FIGURE), False),
    # A part no result answered, but the call stated its failure: answer.
    (({"terms": "gully"}, FAILED), True),
])
@pytest.mark.parametrize("result,_x,args,_f", BILLS)
def test_require_stated_waits_only_for_a_call_no_result_answered(first, answers, result, _x, args, _f):
    turn = _turn("kerb and gully?", first, ({"terms": "kerb"}, result))
    out = runtime._computed_turn_answer(turn, require_stated=True)
    assert bool(out) is answers, out
    if answers:
        assert out.startswith("Bill row count and total:"), out


@pytest.mark.parametrize("question", ["total?", "kerb and gully totals?"])
def test_a_turn_with_no_computed_call_has_no_computed_answer(question):
    msgs = [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]
    assert runtime._computed_turn_answer(msgs, require_stated=True) == ""
    assert computed_answer.answer_turn([], "en", question) == ""


# -- the same four shapes through the real stream route (``drive``) ---------

STREAM_BILLS = [
    # (terms A, terms B, invented rows: (ref, description, unit, qty))
    ("kerb", "duct", [("A1", "Precast kerb type B", "m", "120"), ("A2", "Precast kerb type C", "m", "35.5"),
                      ("A3", "uPVC duct 110 dia", "m", "9")]),
    ("gully", "manhole", [("G1", "Road gully grating", "nr", "4"), ("M1", "Manhole cover", "nr", "2"),
                          ("M2", "Manhole ring", "nr", "5"), ("G2", "Yard gully", "nr", "1")]),
]


def _install_stream_bill(monkeypatch, rows, readable):
    """An invented bill the tool reads; ``readable`` says, call by call,
    whether the derived sheet exists yet (False: the tool returns no figure)."""
    from app.core import bill_aggregate, projects

    sheet = [bill_aggregate.SheetRow(path=("Bill 1",), ref=ref, description=desc, unit=unit,
                                     quantity=qty, rate="", amount="", pages=[i + 1])
             for i, (ref, desc, unit, qty) in enumerate(rows)]
    seq = iter(readable)
    monkeypatch.setattr(bill_aggregate, "derived_bill_documents",
                        lambda project_id, document_id=None: (
                            [{"id": "doc-1", "original_name": "invented.xlsx", "metadata": {}}]
                            if next(seq) else []))
    monkeypatch.setattr(projects, "materialize_document_path", lambda doc: "invented.xlsx")
    monkeypatch.setattr(bill_aggregate, "read_sheet_rows", lambda path: list(sheet))


def _stream(monkeypatch, rows, terms, readable) -> str:
    from tests.lane5_core_support import answer_text, call, drive, install_driver_stream

    _install_stream_bill(monkeypatch, rows, readable)
    install_driver_stream(monkeypatch)
    events, _ = drive(monkeypatch, f"how many {' and '.join(dict.fromkeys(terms))} items, and the totals?", [
        {"content": "", "tool_calls": [call(TOOL, {"terms": t}, f"s{i}") for i, t in enumerate(terms)]},
        {"content": ""},
    ], project_id="p-invented")
    return answer_text(events)


def _count(rows, terms) -> str:
    return f"holds {terms}: **{sum(terms in d.lower() for _r, d, _u, _q in rows)}**"


_UNANSWERED = EN["unanswered"].split("{")[0]


@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_stream_no_figure_then_result_shows_the_result_alone(monkeypatch, a, b, rows):
    out = _stream(monkeypatch, rows, [a, a], [False, True])
    assert out.startswith("Bill row count and total:"), out
    assert _count(rows, a) in out
    assert EN["not_returned"] not in out and EN["several"].format(n=2) not in out, out


@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_stream_the_same_result_twice_is_shown_once(monkeypatch, a, b, rows):
    out = _stream(monkeypatch, rows, [a, a], [True, True])
    assert out.count(_count(rows, a)) == 1, out
    assert out.count("Bill row count and total:") == 1, out


@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_stream_two_different_results_both_shown_in_call_order(monkeypatch, a, b, rows):
    out = _stream(monkeypatch, rows, [a, b], [True, True])
    assert out.startswith(EN["several"].format(n=2)), out
    assert (out.index(f"{EN['asked']}: {a}") < out.index(_count(rows, a))
            < out.index(f"{EN['asked']}: {b}") < out.index(_count(rows, b))), out


@pytest.mark.parametrize("same_part,blocks", [(True, 1), (False, 2)])
@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_stream_no_figure_twice_is_the_plain_answer(monkeypatch, a, b, rows, same_part, blocks):
    out = _stream(monkeypatch, rows, [a, a] if same_part else [a, b], [False, False])
    # With no result the no-figure calls ARE the answer, one per part, each
    # under its heading -- not the trailing "not answered" line.
    assert out.startswith("Bill row count and total:"), out
    assert out.count(f"{EN['not_returned']}:") == blocks, out
    assert _UNANSWERED not in out, out
    assert "unable to generate" not in out.lower()


@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_stream_an_unanswered_part_follows_the_result_in_one_line(monkeypatch, a, b, rows):
    out = _stream(monkeypatch, rows, [b, a], [False, True])
    assert out.startswith("Bill row count and total:"), out
    line = next(ln for ln in out.splitlines() if ln.startswith(_UNANSWERED))
    assert f"{EN['asked']}: {b}" in line
    assert out.index(_count(rows, a)) < out.index(line)
