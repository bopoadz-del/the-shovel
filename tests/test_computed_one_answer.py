"""One answer per turn from several computed calls (live reading of #392).

A turn may call one computed tool several times. The answer LEADS with a
result; a result identical to one already shown is one result; a call with no
figure is mentioned only when no result of the same tool answered the same
part of the question (same tool, the SAME selection of words, or a
statement of the result that names exactly that selection; a narrower or a
broader selection is a different part, and an empty one is never answered),
and then at the end in one plain line; with no result at all the
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
from app.agents.core.tool_registry import Statement

EN = LABELS["en"]
TOOL = "bill_rows_aggregate"


def _bill(filename: str, terms: list[str], unit: str, rows: list[tuple[str, int, float]],
          section_terms: tuple[str, ...] = ()) -> dict:
    entries = [{"ref": ref, "pages": [page], "section": ["Invented section"],
                "description": " ".join(terms) + f" item {ref}", "unit": unit,
                "quantity": qty} for ref, page, qty in rows]
    total = sum(q for _r, _p, q in rows)
    return {"status": "success", "bills": [{
        "document_id": f"doc-{filename}", "filename": filename, "derived_from": None,
        "status": "success", "measure": "quantity", "terms": terms, "section_terms": list(section_terms),
        "rows_scanned": 40, "count": len(rows), "sum": total, "unit": unit,
        "sum_by_unit": {unit: total}, "without_figure": [], "rows": entries}]}


NO_FIGURE = {"status": "success", "bills": []}

BILLS = [
    (_bill("bill-alpha.xlsx", ["kerb"], "m", [("K.1", 3, 120.5), ("K.4", 4, 79.5)]),
     {"terms": "kerb"}, {"terms": "Kerb "}, "**200 m**"),
    (_bill("bill-beta.xlsx", ["duct", "pvc"], "nr", [("D.2", 7, 11.0), ("D.3", 7, 6.0)]),
     {"terms": ["PVC", "duct"]}, {"terms": ["duct", "pvc"]}, "**17 nr**"),
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


COVER_CASES = [
    # (the result's arguments, its statements' (terms, within), the same
    #  selection asked again, a broader one, a narrower one)
    ({"terms": "Kerb", "section": "drainage"}, [(("kerb",), ("drainage",))],
     {"section": "DRAINAGE ", "terms": "kerb"}, {"terms": "kerb"},
     {"terms": "kerb", "section": "drainage", "type": "B"}),
    ({"terms": ["pipes", "100mm"]}, [(("pipes", "100mm"), ())],
     {"terms": ["100MM", "Pipes"]}, {"terms": "pipes"},
     {"terms": ["pipes", "100mm", "ductile"]}),
]


def _result(args: dict, stated: list, figure: float | None = 4.0) -> Call:
    return Call(tool=TOOL, heading="H", asked=args, statements=tuple(
        Statement(figure=f, what="quantity", basis="count", count=None if f is None else 4,
                  terms=t, within=w)
        for t, w, f in ((*x, figure) if len(x) == 2 else x for x in stated)))


@pytest.mark.parametrize("args,stated,same,broader,narrower", COVER_CASES)
def test_a_result_covers_only_the_same_selection(args, stated, same, broader, narrower):
    result = _result(args, stated)
    assert covers(result, Call(tool=TOOL, heading="H", asked=same)) is True
    # A narrower success does not answer a broader failure, and a broader
    # success does not answer a narrower one it does not state.
    assert covers(result, Call(tool=TOOL, heading="H", asked=broader)) is False
    assert covers(result, Call(tool=TOOL, heading="H", asked=narrower)) is False
    assert covers(result, Call(tool="zz_other_invented_tool", heading="H", asked=same)) is False


@pytest.mark.parametrize("args,stated,_s,_b,_n", COVER_CASES)
@pytest.mark.parametrize("empty", [{}, {"terms": None, "flag": True}])
def test_an_empty_selection_is_never_covered(args, stated, _s, _b, _n, empty):
    assert covers(_result(args, stated), Call(tool=TOOL, heading="H", asked=empty)) is False


@pytest.mark.parametrize("terms,sections,part", [
    ("pipes", ["Bill 1", "Bill 2", "Bill 3"], "Bill 2"),
    ("gully", ["Section A", "Section C"], "Section C"),
])
def test_a_broader_success_covers_a_narrower_part_only_when_its_statement_names_it(
        terms, sections, part):
    args = {"terms": terms, "section": sections}
    asked = Call(tool=TOOL, heading="H", asked={"terms": terms, "section": part})
    named = [((terms,), (s,)) for s in sections]
    assert covers(_result(args, named), asked) is True
    # The part is not among the stated ones: not answered.
    assert covers(_result(args, [n for n in named if n[1] != (part,)]), asked) is False
    # The part is stated, but with no figure: not answered.
    failed = [(t, w, None if w == (part,) else 4.0) for t, w in named]
    assert covers(_result(args, failed), asked) is False


FAILED = {"status": "error", "reason": "no_derived_bill_sheet"}


@pytest.mark.parametrize("first,answers", [
    # Its part answered by the result: nothing waits.
    (({"terms": "kerb"}, NO_FIGURE), True),
    # A part no result answered, and the call said nothing: wait.
    (({"terms": "gully"}, NO_FIGURE), False),
    # A part no result answered, but the call stated its failure: answer.
    (({"terms": "gully"}, FAILED), True),
    # Empty arguments name no part, so no result answered it: wait.
    (({}, NO_FIGURE), False),
    # A narrower part than the result's, said nothing: wait.
    (({"terms": "kerb", "section": "roads"}, NO_FIGURE), False),
])
@pytest.mark.parametrize("result,_x,args,_f", BILLS)
def test_require_stated_waits_only_for_a_call_no_result_answered(first, answers, result, _x, args, _f):
    turn = _turn("kerb and gully?", first, ({"terms": "kerb"}, result))
    out = runtime._computed_turn_answer(turn, require_stated=True)
    assert bool(out) is answers, out
    if answers:
        assert out.startswith("Bill row count and total:"), out


SUBSET_TURNS = [
    # (terms, unit, rows, section): a bill matched on ``terms`` within ``section``
    (["kerb"], "m", [("K.1", 3, 120.5), ("K.4", 4, 79.5)], "drainage"),
    (["duct", "pvc"], "nr", [("D.2", 7, 11.0), ("D.3", 7, 6.0)], "ducts"),
]


def _unanswered_line(out: str) -> str:
    return next((ln for ln in out.splitlines() if ln.startswith(EN["unanswered"].split("{")[0])), "")


@pytest.mark.parametrize("terms,unit,rows,section", SUBSET_TURNS)
@pytest.mark.parametrize("failed_first", [True, False])
def test_a_narrower_success_does_not_answer_a_broader_failure(terms, unit, rows, section, failed_first):
    narrow = _bill("bill-n.xlsx", terms, unit, rows, section_terms=(section,))
    calls = [({"terms": " ".join(terms)}, FAILED),
             ({"terms": " ".join(terms), "section": section}, narrow)]
    question = f"{' '.join(terms)} total, and within {section}?"
    for out in _answers(_turn(question, *(calls if failed_first else calls[::-1]))):
        assert out.startswith("Bill row count and total:"), out
        line = _unanswered_line(out)
        assert line and f"{EN['asked']}: {' '.join(terms)})" in line, out


@pytest.mark.parametrize("terms,unit,rows,section", SUBSET_TURNS)
def test_a_broader_success_does_not_answer_a_narrower_failure(terms, unit, rows, section):
    broad = _bill("bill-b.xlsx", terms, unit, rows)
    question = f"{' '.join(terms)} total, and within {section}?"
    turn = _turn(question, ({"terms": " ".join(terms), "section": section}, FAILED),
                 ({"terms": " ".join(terms)}, broad))
    for out in _answers(turn):
        line = _unanswered_line(out)
        assert line and section in line, out


@pytest.mark.parametrize("result,_a,args,figure", BILLS)
@pytest.mark.parametrize("empty", [{}, {"flag": True}])
def test_a_failed_call_with_empty_arguments_is_still_named(result, _a, args, figure, empty):
    for out in _answers(_turn("totals?", (empty, FAILED), (args, result))):
        line = _unanswered_line(out)
        assert line and out.index(figure) < out.index(line), out
        assert EN["no_figure"] not in out  # its own declared reason, not a generic one


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


# -- the fifth shape (live A3 count row, 18 Nov): a result cut to a preview
# that does not parse, then a result. The cut call returned; its figure cannot
# be read in this turn. It is STATED (the turn does not wait for it) and named
# at the end in the one plain line; the readable result leads.

_CUT_REASON = EN["result_cut"]


def _cut(result: dict, keep: int) -> str:
    """A tool message cut to a preview of its first ``keep`` characters."""
    text = json.dumps({"name": TOOL, "ok": True, "result": result})
    return json.dumps({"truncated": True, "note": "cut", "preview": text[:keep]})


def _cut_turn(question: str, cut_args: dict, cut_content: str, args: dict, result: dict) -> list[dict]:
    msgs = _turn(question, (cut_args, NO_FIGURE), (args, result))
    msgs[3]["content"] = cut_content
    return msgs


@pytest.mark.parametrize("keep", [40, 120])
@pytest.mark.parametrize("result,_a,args,figure", BILLS)
def test_a_cut_unreadable_result_is_stated_and_named_after_the_readable_one(keep, result, _a, args, figure):
    cut_args = {"terms": "gully"}
    turn = _cut_turn("gully and kerb totals?", cut_args, _cut(result, keep), args, result)
    # The cut call can never state a figure: the turn must not wait for it.
    assert runtime._computed_turn_answer(turn, require_stated=True), "waited on a cut call"
    for out in _answers(turn) + [runtime._computed_turn_answer(turn, require_stated=True)]:
        assert out.startswith("Bill row count and total:"), out
        line = _unanswered_line(out)
        assert f"{EN['asked']}: gully" in line and _CUT_REASON in line, out
        assert out.index(figure) < out.index(line)
        assert out.count(_CUT_REASON) == 1, out


@pytest.mark.parametrize("result,_a,args,figure", BILLS)
def test_a_cut_preview_that_is_whole_json_is_read_not_called_cut(result, _a, args, figure):
    whole = json.dumps({"name": TOOL, "ok": True, "result": result})
    content = json.dumps({"truncated": True, "note": "cut", "preview": whole})
    turn = _turn("totals?", (args, NO_FIGURE))
    turn[3]["content"] = content
    out = runtime._computed_turn_answer(turn, require_stated=True)
    assert figure in out and _CUT_REASON not in out, out


def test_a_cut_call_alone_is_the_answer_with_its_reason():
    for result, _a, args, _f in BILLS:
        turn = _turn("totals?", (args, NO_FIGURE))
        turn[3]["content"] = _cut(result, 60)
        out = runtime._computed_turn_answer(turn, require_stated=True)
        assert out.startswith("Bill row count and total:") and _CUT_REASON in out, out
        assert _UNANSWERED not in out, out


@pytest.mark.parametrize("result,_a,args,figure", BILLS)
def test_the_real_serializer_cut_then_a_zero_result(result, _a, args, figure):
    """Live order (core loop): call 1's result is over the tool-result cap,
    and the core serializer (``_tool_result_content``) cuts it to a preview
    that does not parse; call 2 returns a count of 0. The 0 leads; call 1 is
    named last with the reason; the turn does not wait for call 1."""
    zero = {**result, "bills": [{**result["bills"][0], "count": 0, "sum": 0, "rows": [],
                                 "sum_by_unit": {}}]}
    full = {"name": TOOL, "ok": True, "result": result}
    cap = len(json.dumps(full)) // 2
    cut = runtime._tool_result_content(full, max_chars=cap)
    assert json.loads(cut).get("truncated"), cut
    turn = _turn("how many gully items, and the other part?", ({"terms": "gully"}, NO_FIGURE), (args, zero))
    turn[3]["content"] = cut
    for out in _answers(turn):
        assert out.startswith("Bill row count and total:"), out
        assert "**0**" in out, out
        line = _unanswered_line(out)
        assert f"{EN['asked']}: gully" in line and _CUT_REASON in line, out
        assert out.index("**0**") < out.index(line), out


# -- a result over the size cap (Advisor 06:33Z): its statement is never cut.
# The figure, the unit, the selection and the row count ride whole beside the
# window (``whole_result_fields``); the answer reads the whole result by
# reference (``full_result``), never by re-parsing a preview.

def _long_bill(filename: str, word: str, unit: str, n: int) -> dict:
    """An invented bill with ``n`` matching rows of quantity 1..n."""
    return _bill(filename, [word], unit,
                 [(f"{word[0].upper()}.{i}", 1 + i // 10, float(i + 1)) for i in range(n)])


LONG_BILLS = [
    (_long_bill("bill-gamma.xlsx", "kerb", "m", 60), {"terms": "kerb"}, 60, 1830),
    (_long_bill("bill-delta.xlsx", "duct", "nr", 85), {"terms": "duct"}, 85, 3655),
]


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_a_result_over_the_cap_keeps_its_figure_selection_and_row_count(result, args, rows, total):
    full = {"name": TOOL, "ok": True, "result": result}
    cap = 4000
    assert len(json.dumps(full)) > cap
    env = json.loads(runtime._tool_result_content(full, max_chars=cap))
    assert env["truncated"] and env["next_char_offset"], env  # still paged
    whole = env[runtime.OUTLINE_KEY]
    bill = whole["result"]["bills"][0]
    assert bill["sum"] == total and bill["count"] == rows and bill["terms"] == result["bills"][0]["terms"]
    assert bill["rows"] == []  # only the rows go
    assert whole[runtime.ROWS_TOTAL_KEY] == {"result.bills[0].rows": rows}


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_a_cut_result_held_by_reference_states_its_figure(result, args, rows, total):
    full = {"name": TOOL, "ok": True, "result": result}
    cut = runtime._tool_result_content(full, max_chars=4000)
    turn = _turn("how many items, and the total?", (args, result))
    turn[3]["content"] = cut
    turn[3].update(runtime._tool_message_extra(full, cut, TOOL))
    assert runtime._computed_turn_answer(turn, require_stated=True), "waited on a cut call it holds"
    for out in _answers(turn):
        assert f"**{rows}**" in out, out
        assert _CUT_REASON not in out and _UNANSWERED not in out, out


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_a_cut_result_without_its_reference_is_too_large_not_no_figure(result, args, rows, total):
    full = {"name": TOOL, "ok": True, "result": result}
    cut = runtime._tool_result_content(full, max_chars=60)
    turn = _turn("how many items, and the total?", (args, result))
    turn[3]["content"] = cut
    for out in _answers(turn):
        assert _CUT_REASON in out, out
        assert "too large to show" in _CUT_REASON


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_the_live_fifth_shape_a_cut_result_then_a_zero_result(result, args, rows, total):
    """Live A3 (18 Nov): call 1 over the cap, call 2 a count of 0. Call 1's
    figure survives and leads."""
    zero = {**result, "bills": [{**result["bills"][0], "terms": ["gully"], "count": 0, "sum": 0,
                                 "rows": [], "sum_by_unit": {}}]}
    full = {"name": TOOL, "ok": True, "result": result}
    cut = runtime._tool_result_content(full, max_chars=4000)
    turn = _turn("how many items, and gully?", (args, result), ({"terms": "gully"}, zero))
    turn[3]["content"] = cut
    turn[3].update(runtime._tool_message_extra(full, cut, TOOL))
    for out in _answers(turn):
        assert f"**{rows}**" in out, out
        assert _CUT_REASON not in out, out
        # The 0 answers a different part (gully): one plain line, at the end.
        line = _zero_line(out)
        assert "gully" in line and out.rstrip().endswith(line), out
        assert out.index(f"**{rows}**") < out.index(line), out
        assert EN["several"].split("}", 1)[1] not in out and "**0**" not in out, out


@pytest.mark.parametrize("content,held", [
    (json.dumps({"truncated": True, "preview": "{"}), True),
    (json.dumps({"ok": True, "result": {}}), False),
])
def test_only_a_cut_result_is_held_by_reference(content, held):
    payload = {"ok": True, "result": {"rows": [1]}}
    extra = runtime._tool_message_extra(payload, content, TOOL)
    assert (runtime.FULL_RESULT_KEY in extra) is held
    if held:
        assert extra[runtime.FULL_RESULT_KEY] is payload


def test_the_whole_result_never_reaches_the_provider():
    msg = {"role": "tool", "tool_call_id": "c1", "name": TOOL, "content": "{}",
           runtime.FULL_RESULT_KEY: {"ok": True}}
    assert runtime.FULL_RESULT_KEY not in runtime._sanitize_messages_for_provider([msg])[-1]


@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_stream_a_result_over_the_cap_reaches_the_model_whole_and_the_answer(monkeypatch, a, b, rows):
    """The real stream route with the model's per-result cap lowered: the
    model is handed a VALID envelope whose outline carries the count of rows
    and the selection, and the answer states the figure."""
    from app.agents.driver import loop as driver_loop
    from tests.lane5_core_support import answer_text, call, drive, install_driver_stream

    many = [(f"{r}{i}", d, u, q) for i in range(40) for r, d, u, q in rows]
    monkeypatch.setattr(driver_loop, "_TOOL_RESULT_CHARS", 3000)
    _install_stream_bill(monkeypatch, many, [True])
    install_driver_stream(monkeypatch)
    events, seen = drive(monkeypatch, f"how many {a} items, and the total?", [
        {"content": "", "tool_calls": [call(TOOL, {"terms": a}, "s0")]},
        {"content": ""},
    ], project_id="p-invented")
    tool_msgs = [m for m in seen[-1]["messages"] if m.get("role") == "tool"]
    env = json.loads(tool_msgs[-1]["content"])  # whole JSON, not a raw slice
    assert env["truncated"], env
    want = sum(a in d.lower() for _r, d, _u, _q in many)
    totals = env[runtime.OUTLINE_KEY][runtime.ROWS_TOTAL_KEY]
    assert want in totals.values(), totals
    assert f"**{want}**" in answer_text(events)


# -- a zero beside a figure (Advisor 06:33Z): the zero result is the one plain
# line at the end, and only when it answers a DIFFERENT selection; a zero for
# the selection a figure already answered is not shown at all.

_ZERO = EN["zero_line"].split("{")[0]


def _zero_line(out: str) -> str:
    return next((ln for ln in out.splitlines() if ln.startswith(_ZERO)), "")


def _zero_of(result: dict, terms: list[str]) -> dict:
    return {**result, "bills": [{**result["bills"][0], "terms": terms, "count": 0, "sum": 0,
                                 "rows": [], "sum_by_unit": {}}]}


@pytest.mark.parametrize("zero_first", [True, False])
@pytest.mark.parametrize("result,_a,args,figure", BILLS)
def test_a_zero_for_the_part_a_figure_answered_is_not_shown(zero_first, result, _a, args, figure):
    zero = (args, _zero_of(result, result["bills"][0]["terms"]))
    lead = (args, result)
    turn = _turn("how many, and the total?", *((zero, lead) if zero_first else (lead, zero)))
    for out in _answers(turn):
        assert figure in out, out
        assert _zero_line(out) == "" and "**0**" not in out, out
        assert EN["several"].split("}", 1)[1] not in out, out


@pytest.mark.parametrize("zero_first", [True, False])
@pytest.mark.parametrize("result,_a,args,figure", BILLS)
def test_a_zero_for_a_different_part_is_one_plain_line_at_the_end(zero_first, result, _a, args, figure):
    zero = ({"terms": "gully"}, _zero_of(result, ["gully"]))
    lead = (args, result)
    turn = _turn("how many gully, and the total?", *((zero, lead) if zero_first else (lead, zero)))
    for out in _answers(turn):
        line = _zero_line(out)
        assert "gully" in line and out.rstrip().endswith(line), out
        assert out.index(figure) < out.index(line), out
        assert "**0**" not in out and out.count(_ZERO) == 1, out


@pytest.mark.parametrize("terms", [["gully"], ["manhole", "cover"]])
def test_zeros_alone_are_the_answer(terms):
    result = BILLS[0][0]
    turn = _turn("how many?", ({"terms": terms}, _zero_of(result, terms)))
    for out in _answers(turn):
        assert "**0**" in out and _zero_line(out) == "", out


def test_the_zero_line_is_worded_in_every_language():
    assert all("zero_line" in lab and "{parts}" in lab["zero_line"] for lab in LABELS.values())


# -- the mechanics of the cap, each rule with two inputs.

@pytest.mark.parametrize("payload", [
    [{"ref": "A.1", "quantity": 2}, {"ref": "A.2", "quantity": 3}],
    [[1, 2], [3, 4], [5, 6]],
])
def test_only_an_object_result_has_an_outline(payload):
    assert runtime._result_outline(payload, 10_000) is None


@pytest.mark.parametrize("content", [b'{"truncated": true}', bytearray(b'{"truncated": true, "preview": "{"}')])
def test_content_that_is_not_text_is_not_a_cut_envelope(content):
    assert runtime._tool_result_was_cut(content) is False


@pytest.mark.parametrize("cap", [0, -40])
@pytest.mark.parametrize("result,_a,_b,_f", BILLS)
def test_a_cap_below_one_falls_back_to_the_default_cap(cap, result, _a, _b, _f):
    full = {"name": TOOL, "ok": True, "result": result}
    assert len(json.dumps(full)) < runtime._TOOL_RESULT_MAX_CHARS
    assert json.loads(runtime._tool_result_content(full, max_chars=cap)) == full


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_compacting_a_cut_result_keeps_its_outline_whole(result, args, rows, total):
    full = {"name": TOOL, "ok": True, "result": result}
    cut = runtime._tool_result_content(full, max_chars=4000)
    first = json.loads(cut)[runtime.OUTLINE_KEY]
    again = json.loads(runtime._compacted_tool_payload(cut, 500))
    assert again["truncated"] and again[runtime.OUTLINE_KEY] == first, again
    bill = again[runtime.OUTLINE_KEY]["result"]["bills"][0]
    assert bill["sum"] == total and again[runtime.OUTLINE_KEY][runtime.ROWS_TOTAL_KEY] == {
        "result.bills[0].rows": rows}


@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_stream_the_live_fifth_shape_a_result_over_the_cap_then_a_zero(monkeypatch, a, b, rows):
    """The real stream route: call 1 is over the model's cap, call 2 asks
    for a word no row holds. Call 1's figure leads; the zero is one line at
    the end; nothing says the tool did not return."""
    from app.agents.driver import loop as driver_loop
    from tests.lane5_core_support import answer_text, call, drive, install_driver_stream

    absent = "zzinvented"
    many = [(f"{r}{i}", d, u, q) for i in range(40) for r, d, u, q in rows]
    monkeypatch.setattr(driver_loop, "_TOOL_RESULT_CHARS", 3000)
    _install_stream_bill(monkeypatch, many, [True, True])
    install_driver_stream(monkeypatch)
    events, seen = drive(monkeypatch, f"how many {a} items, and {absent}?", [
        {"content": "", "tool_calls": [call(TOOL, {"terms": a}, "s0")]},
        {"content": "", "tool_calls": [call(TOOL, {"terms": absent}, "s1")]},
        {"content": ""},
    ], project_id="p-invented")
    cut = []
    for m in seen[-1]["messages"]:
        if m.get("role") == "tool":
            env = json.loads(m["content"])
            assert len(m["content"]) <= 3000, len(m["content"])
            if env.get("truncated"):
                cut.append(env)
    # Exactly one result was over the cap, and exactly what was cut: the
    # rest counted, every row of the call-1 list counted in the outline.
    assert len(cut) == 1, cut
    env = cut[0]
    assert env["chars_shown"] + env["chars_dropped"] == env["chars_total"] > 3000, env
    matching = sum(a in d.lower() for _r, d, _u, _q in many)
    assert list(env[runtime.OUTLINE_KEY][runtime.ROWS_TOTAL_KEY].values()) == [matching], env
    out = answer_text(events)
    want = sum(a in d.lower() for _r, d, _u, _q in many)
    assert f"**{want}**" in out, out
    assert EN["not_returned"] not in out and _CUT_REASON not in out, out
    line = _zero_line(out)
    assert absent in line and out.index(f"**{want}**") < out.index(line), out


# ── #428 grader 0840Z: (a) the model's copy never loses the outline ─────────

def _many_bills(n: int, rows_each: int) -> dict:
    """An invented result stating ``n`` figures: ``n`` bills, each with rows."""
    bills = [_bill(f"bill-{i}.xlsx", ["kerb"], "m",
                   [(f"K.{i}.{j}", 1, float(j + 1)) for j in range(rows_each)])["bills"][0]
             for i in range(n)]
    return {"status": "success", "bills": bills}


def _one_bill_wide(rows: int, pages: int) -> dict:
    """An invented result stating ONE figure whose non-row fields are large."""
    result = _long_bill("bill-wide.xlsx", "duct", "nr", rows)
    result["bills"][0]["pages_read"] = list(range(1, pages + 1))
    result["bills"][0]["working"] = "summed per unit " * 400
    return result


@pytest.mark.parametrize("result,figures", [
    (_many_bills(120, 4), 120),     # many stated figures
    (_one_bill_wide(30, 3000), 1),  # one stated figure, wide fields
], ids=["many_figures", "one_figure"])
def test_an_outline_over_its_budget_keeps_the_statement_fields_never_dropped(result, figures):
    full = {"name": TOOL, "ok": True, "result": result}
    cap = 4000
    env = json.loads(runtime._tool_result_content(full, max_chars=cap))
    outline = env.get(runtime.OUTLINE_KEY)
    assert outline is not None, "the outline was dropped"
    assert outline["name"] == TOOL and outline["ok"] is True
    bills = outline["result"]["bills"]
    assert bills, outline
    for got, want in zip(bills, result["bills"]):
        assert (got["count"], got["sum"], got["terms"], got["filename"]) == (
            want["count"], want["sum"], want["terms"], want["filename"])
        assert "rows" not in got or got["rows"] == []
        # Every statement shown is the whole statement, never a cut copy.
        assert {k: got[k] for k, v in want.items() if isinstance(v, str)} == {
            k: v for k, v in want.items() if isinstance(v, str)}
    if len(bills) < figures:
        # The parts shown are said to be N of M.
        assert outline[runtime.PARTS_TOTAL_KEY] == {"result.bills": figures}, outline
    else:
        assert runtime.PARTS_TOTAL_KEY not in outline
        assert len(bills[0].get("pages_read", [])) <= runtime._OUTLINE_LIST_ITEMS
    assert all(isinstance(v, int) for v in outline[runtime.ROWS_TOTAL_KEY].values())


@pytest.mark.parametrize("payload", [
    {"ok": True, "result": {"bills": [{"count": 3, "rows": [{"r": "x" * 900}] * 3}]}},
    {"ok": True, "rows": [{"r": "y" * 900}] * 5, "sum": 7},
])
def test_an_outline_never_returns_none_for_a_result_with_rows(payload):
    outline = runtime._result_outline(payload, 40)
    assert outline is not None and outline["ok"] is True, outline


# ── (b) N of M rows shown ─────────────────────────────────────────────────────

def _whole_rows_in(preview: str, rows: list[dict]) -> int:
    cursor, n = 0, 0
    for r in rows:
        piece = json.dumps(r)
        at = preview.find(piece, cursor)
        if at < 0:
            break
        cursor, n = at + len(piece), n + 1
    return n


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
@pytest.mark.parametrize("cap", [3000, 6000])
def test_a_cut_result_says_how_many_rows_are_shown_of_how_many(result, args, rows, total, cap):
    full = {"name": TOOL, "ok": True, "result": result}
    env = json.loads(runtime._tool_result_content(full, max_chars=cap))
    path = "result.bills[0].rows"
    shown = env[runtime.ROWS_SHOWN_KEY][path]
    assert shown == _whole_rows_in(env["preview"], result["bills"][0]["rows"])
    assert 0 <= shown < rows
    assert f"rows shown: {shown} of {rows} ({path})" in env["rows_note"], env["rows_note"]


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_the_compacted_copy_also_says_how_many_rows_are_shown(result, args, rows, total):
    text = json.dumps({"name": TOOL, "ok": True, "result": result})
    env = json.loads(runtime._compacted_tool_payload(text, 2500, outline_budget=2000))
    shown = env[runtime.ROWS_SHOWN_KEY]["result.bills[0].rows"]
    assert shown == _whole_rows_in(env["preview"], result["bills"][0]["rows"]) and shown < rows
    assert f"{shown} of {rows}" in env["rows_note"]


@pytest.mark.parametrize("result", [BILLS[0][0], BILLS[1][0]])
def test_a_whole_result_carries_no_rows_shown(result):
    full = {"name": TOOL, "ok": True, "result": result}
    assert runtime.ROWS_SHOWN_KEY not in runtime._tool_result_content(full, max_chars=8000)


# ── (c) the reference: only a declared computed result, never stored or logged ──

def _fetch_result(n: int) -> dict:
    return {"status": "success", "document_id": "d-1", "filename": "invented-spec.pdf",
            "content": "Cure under wet hessian. " * n,
            "pages": [{"page": i, "text": "p"} for i in range(n)]}


@pytest.mark.parametrize("n", [600, 1500])
def test_a_cut_document_read_is_never_held_by_reference(n):
    payload = _fetch_result(n)
    cut = runtime._tool_result_content(payload, max_chars=4000)
    assert runtime._tool_result_was_cut(cut)
    assert runtime._tool_message_extra(payload, cut, "fetch_document") == {}
    assert runtime._tool_message_extra(payload, cut, "search_project_documents") == {}


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_a_cut_computed_result_is_held_by_reference(result, args, rows, total):
    full = {"name": TOOL, "ok": True, "result": result}
    cut = runtime._tool_result_content(full, max_chars=4000)
    assert runtime._tool_message_extra(full, cut, TOOL)[runtime.FULL_RESULT_KEY] is full
    assert runtime._tool_message_extra(full, cut, "") == {}


_DEEPSEEK = {"provider": "deepseek", "url": "https://api.deepseek.com/v1/chat/completions",
             "env_key": "DEEPSEEK_API_KEY", "default_model": "deepseek-chat"}


def _core_stream(monkeypatch, caplog, rows, term, cap, conversation_id=None):
    """The CORE stream route (``Agent.chat_stream``), the provider scripted:
    hop 1 asks for the bill tool, which runs for real on an invented bill;
    hop 2 says nothing, so the answer is the computed one. Returns the
    events and what the provider was sent, hop by hop."""
    import asyncio
    import logging
    from unittest.mock import patch

    from app.agents.runtime import Agent

    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "invented-key")
    monkeypatch.setenv("SYNTHESIS_STREAMING", "0")
    monkeypatch.setattr(runtime, "_TOOL_RESULT_MAX_CHARS", cap)
    _install_stream_bill(monkeypatch, rows, [True])

    async def _is_ready(_pid):
        return True

    # The project is indexed and retrieval hands nothing: the turn is the tool's.
    monkeypatch.setattr("app.agents.runtime._project_is_rag_ready_off_process", _is_ready)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setattr("app.agents.runtime.rag_inject",
                        lambda **kwargs: (None, {"project_id": "p-invented", "chunks": []}))
    sent: list[list[dict]] = []

    async def fake_call(_self, messages, api_key, **kwargs):
        sent.append(runtime._sanitize_messages_for_provider(messages))
        if len(sent) == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "s0", "type": "function", "function": {
                    "name": TOOL, "arguments": json.dumps({"terms": term})}}]}}}
        return {"status": "success", "choice": {"message": {"role": "assistant", "content": ""}}}

    agent = Agent(name="project-assistant", description="pa", system_prompt="x",
                  allowed_blocks=["construction"])

    async def go():
        return [ev async for ev in agent.chat_stream(
            user_message=f"how many {term} items, and the total?", history=[],
            project_id="p-invented", conversation_id=conversation_id, user_id=None)]

    caplog.set_level(logging.DEBUG)
    with patch.object(Agent, "_call_llm", fake_call):
        events = asyncio.run(go())
    return events, sent


def _tokens(events) -> str:
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_core_stream_a_computed_result_over_the_cap_states_its_figure(monkeypatch, caplog, a, b, rows):
    """(d) through the CORE stream route, which holds the reference: the
    model gets a valid envelope within the cap (outline and N of M rows);
    the answer states the whole figure; the provider never gets the key."""
    many = [(f"{r}{i}", d, u, q) for i in range(40) for r, d, u, q in rows]
    events, sent = _core_stream(monkeypatch, caplog, many, a, 3000)
    want = sum(a in d.lower() for _r, d, _u, _q in many)
    out = _tokens(events)
    assert f"**{want}**" in out, (out, events[-3:])
    assert _CUT_REASON not in out and EN["not_returned"] not in out, out
    assert len(sent) >= 2, sent
    tool_msgs = [m for m in sent[1] if m.get("role") == "tool"]
    assert tool_msgs, sent[1]
    for m in tool_msgs:
        assert runtime.FULL_RESULT_KEY not in m
        env = json.loads(m["content"])
        assert len(m["content"]) <= 3000 and env["truncated"], m["content"][:200]
        # The core loop hands the tool's own result (no name/ok wrapper).
        outline = env[runtime.OUTLINE_KEY]
        assert outline["bills"][0]["count"] == want
        assert "rows shown:" in env["rows_note"] and f"of {want} (bills[0].rows)" in env["rows_note"]


@pytest.mark.parametrize("a,b,rows", STREAM_BILLS)
def test_core_stream_the_reference_is_never_stored_reloaded_or_logged(monkeypatch, caplog, a, b, rows):
    """(c) the turn is persisted by ``agent_memory.append_message`` (role and
    text only) and reloaded by ``agent_memory.get_messages``: neither carries
    the key or a row past the model's window, and no log line of the turn
    does."""
    import uuid

    from app.core import agent_memory

    marker = "zzpastwindow"
    many = [(f"{r}{i}", d, u, q) for i in range(40) for r, d, u, q in rows]
    many.append(("Z.999", f"{a} {marker} final row", many[0][2], "1"))
    cid = f"ws-p-invented-{uuid.uuid4().int % 10**13}"
    agent_memory.get_or_create_conversation(cid, "project-assistant")
    events, sent = _core_stream(monkeypatch, caplog, many, a, 3000, conversation_id=cid)
    held = [m for m in sent[1] if m.get("role") == "tool"]
    assert held and marker not in held[0]["content"], "the marker row must lie past the window"
    stored = agent_memory.get_messages(cid, limit=None)
    assert stored, "the turn was not persisted"
    assert {m["role"] for m in stored} <= {"user", "assistant"}
    for m in stored:
        assert runtime.FULL_RESULT_KEY not in json.dumps(m, default=str)
    assert runtime.FULL_RESULT_KEY not in caplog.text
    assert f"{marker} final row" not in caplog.text


# ── CI-M on b1c728e: each new rule pinned by its own inputs ─────────────────

@pytest.mark.parametrize("n,budget", [(120, 1000), (60, 700)])
def test_an_outline_over_its_budget_fits_it_and_names_the_parts_it_holds(n, budget):
    full = {"name": TOOL, "ok": True, "result": _many_bills(n, 4)}
    outline = runtime._result_outline(full, budget)
    assert len(json.dumps(outline)) <= budget, len(json.dumps(outline))
    assert outline[runtime.PARTS_TOTAL_KEY] == {"result.bills": n}
    shown = len(outline["result"]["bills"])
    assert 0 < shown < n
    # Only the parts still shown keep their row counts.
    assert all(runtime._part_index(k, "result.bills") < shown
               for k in outline[runtime.ROWS_TOTAL_KEY])


@pytest.mark.parametrize("key,path,index", [
    ("bills[7].rows", "bills", 7),
    ("x[7].rows", "b", -1),          # a key under another list
    ("result.bills[12].rows", "result.bills", 12),
    ("result.extra[3].rows", "result.bills", -1),
])
def test_a_part_index_is_read_only_under_its_own_list(key, path, index):
    assert runtime._part_index(key, path) == index


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_rows_are_counted_per_list_of_rows_never_per_list_of_parts(result, args, rows, total):
    full = {"name": TOOL, "ok": True, "result": result}
    text = json.dumps(full)
    shown = runtime._rows_in_window(full, text, 0, len(text))
    assert shown == {"result.bills[0].rows": rows}, shown


@pytest.mark.parametrize("result,args,rows,total", LONG_BILLS)
def test_a_cut_result_compacted_again_is_not_recounted(monkeypatch, result, args, rows, total):
    """A cut envelope no longer holds its rows: compacting it again keeps
    its outline and never scans it for rows it cannot see."""
    full = {"name": TOOL, "ok": True, "result": result}
    cut = runtime._tool_result_content(full, max_chars=4000)

    def scanned(*_a, **_k):
        raise AssertionError("a cut envelope was scanned for rows")

    monkeypatch.setattr(runtime, "_rows_in_window", scanned)
    again = json.loads(runtime._compacted_tool_payload(cut, 500))
    assert runtime.ROWS_SHOWN_KEY not in again and again[runtime.OUTLINE_KEY]


# -- #428 follow-up: a statement is never cut in the model's copy ----------

def _statement(word: str, n: int) -> str:
    return " ".join(f"{word}{i}" for i in range(n))


@pytest.mark.parametrize("where", ["in_a_part", "top_level"])
@pytest.mark.parametrize("n", [80, 400])
def test_a_statement_over_two_hundred_characters_is_never_cut(where, n):
    long = _statement("per-unit", n)
    assert len(long) > 200
    result = _many_bills(12, 4)
    if where == "in_a_part":
        result["bills"][0]["working"] = long
    else:
        result["basis"] = long
    full = {"name": TOOL, "ok": True, "result": result}
    outline = runtime._result_outline(full, 900)
    got = (outline["result"]["bills"][0]["working"] if where == "in_a_part"
           else outline["result"]["basis"])
    assert got == long
    # Parts gave way before any statement did.
    assert outline[runtime.PARTS_TOTAL_KEY] == {"result.bills": 12}
    assert 1 <= len(outline["result"]["bills"]) < 12


@pytest.mark.parametrize("n,budget", [(400, 300), (900, 1000)])
def test_top_level_statements_over_the_budget_hold_at_a_stated_floor(n, budget):
    long = _statement("basis", n)
    assert len(long) > budget
    full = {"name": TOOL, "ok": True, "result": _many_bills(6, 3), "basis": long}
    outline = runtime._result_outline(full, budget)
    assert outline["basis"] == long and outline["name"] == TOOL
    assert len(outline["result"]["bills"]) == 1
    assert outline["result"]["bills"][0]["count"] == 3
    assert outline[runtime.PARTS_TOTAL_KEY] == {"result.bills": 6}
    floor = outline[runtime.OUTLINE_FLOOR_KEY]
    body = {k: v for k, v in outline.items() if k != runtime.OUTLINE_FLOOR_KEY}
    assert floor["budget"] == budget
    assert floor["chars"] == len(json.dumps(body)) > budget
    assert floor["kept"] == ["name", "ok", "result", "basis", "result.bills[0]"]


@pytest.mark.parametrize("n,budget", [(120, 1000), (60, 700)])
def test_an_outline_that_fits_its_budget_has_no_floor(n, budget):
    full = {"name": TOOL, "ok": True, "result": _many_bills(n, 4)}
    outline = runtime._result_outline(full, budget)
    assert runtime.OUTLINE_FLOOR_KEY not in outline
    assert len(json.dumps(outline)) <= budget


@pytest.mark.parametrize("budget", [1, 40])
def test_a_list_of_parts_keeps_its_first_part_at_any_budget(budget):
    full = {"name": TOOL, "ok": True, "result": _many_bills(9, 2)}
    outline = runtime._result_outline(full, budget)
    first = outline["result"]["bills"]
    assert len(first) == 1 and first[0]["filename"] == "bill-0.xlsx", first
    assert outline[runtime.OUTLINE_FLOOR_KEY]["kept"][-1] == "result.bills[0]"
