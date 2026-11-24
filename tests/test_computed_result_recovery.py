"""A computed result never ends as 'unable to generate a response'.

R-A: an empty synthesis (or the fallback sentence) after a successful
deterministic tool goes to the same recovery as a provider failure.
R-B: the recovery reads the tool's DECLARED result kind from the registry --
any ``kind="computed"`` tool states its own figures, with what was counted or
summed and the source rows; no tool names are listed in the recovery.
R-C (G-8): a ``kind="search"`` result is never shown as the answer.

Every input is invented here; each decision is checked with two inputs.
"""
from __future__ import annotations

import json

import pytest

from app.agents import runtime
from app.agents.base.tools import _row_label, bill_rows_statement
from app.agents.core import tool_registry
from app.agents.core.tool_registry import Statement
from app.agents.driver.sources import RETRIEVAL_TOOLS

FALLBACK = runtime._EMPTY_RESPONSE_FALLBACK


def _tool_msg(name: str, result: dict, ok: bool = True) -> dict:
    return {"role": "tool", "tool_call_id": f"call-{name}", "name": name,
            "content": json.dumps({"name": name, "ok": ok, "result": result})}


def _turn(ask: str, *tool_msgs: dict) -> list[dict]:
    return [{"role": "system", "content": "You answer questions."},
            {"role": "user", "content": ask}, *tool_msgs]


def _bill(filename: str, terms: list[str], unit: str, rows: list[tuple[str, int, float]]) -> dict:
    """An invented bill_rows_aggregate result over the given (ref, page, qty) rows."""
    entries = [{"ref": ref, "pages": [page], "section": ["Invented section"],
                "description": " ".join(terms) + f" item {ref}", "unit": unit,
                "quantity": qty} for ref, page, qty in rows]
    total = sum(q for _r, _p, q in rows)
    return {"status": "success", "bills": [{
        "document_id": f"doc-{filename}", "filename": filename, "derived_from": None,
        "status": "success", "measure": "quantity", "terms": terms, "section_terms": [],
        "rows_scanned": 40, "count": len(rows), "sum": total, "unit": unit,
        "sum_by_unit": {unit: total}, "without_figure": [], "rows": entries}]}


BILLS = [
    ("count the kerb runs", _bill("bill-alpha.xlsx", ["kerb"], "m",
                                  [("K.1", 3, 120.5), ("K.4", 4, 79.5)]), "200", "m"),
    ("total the duct lengths", _bill("bill-beta.xlsx", ["duct", "pvc"], "nr",
                                     [("D.2", 7, 11.0), ("D.3", 7, 6.0), ("D.9", 8, 4.0)]),
     "21", "nr"),
]


@pytest.mark.parametrize("synthesis", ["", FALLBACK])
@pytest.mark.parametrize("ask,result,total,unit", BILLS)
def test_empty_synthesis_after_the_bill_tool_states_its_figure(synthesis, ask, result, total, unit):
    messages = _turn(ask, _tool_msg("bill_rows_aggregate", result))
    out = runtime._postprocess_answer(synthesis, None, messages)
    bill = result["bills"][0]
    assert "unable to generate" not in out.lower()
    assert f"**{total} {unit}**" in out
    assert f"Number of rows whose description holds {', '.join(bill['terms'])}: **{bill['count']}**" in out
    assert bill["filename"] in out
    for row in bill["rows"]:
        assert row["ref"] in out
    for term in bill["terms"]:
        assert term in out


@pytest.mark.parametrize("ask,result,total,unit", BILLS)
def test_provider_failure_and_empty_synthesis_recover_the_same_statement(ask, result, total, unit):
    messages = _turn(ask, _tool_msg("bill_rows_aggregate", result))
    after_failure = runtime._recover_after_provider_failure(messages)
    after_empty = runtime._recover_answer_from_tool_messages("", messages)
    assert after_failure == after_empty
    assert f"**{total} {unit}**" in after_empty


@pytest.fixture
def invented_tools():
    """Two deterministic tools no product code names, declared computed."""
    tool_registry.load()
    added = []

    def declare(name: str, display: str, stater):
        async def handle(call):  # pragma: no cover - never called here
            return {}
        tool_registry.tool(name, owner="base", display_name=display,
                           kind="computed", statement=stater)(handle)
        added.append(name)

    declare("zz_invented_span_ratio", "Span ratio check",
            lambda r: [Statement(figure=r["ratio"], what="span to depth ratio",
                                 source=r["drawing"], rows=tuple(r["members"]))])
    declare("zz_invented_pallet_count", "Pallet counter",
            lambda r: [Statement(figure=float(len(r["lots"])), what="pallets on the delivery note",
                                 count=len(r["lots"]), source=r["note"],
                                 rows=tuple(r["lots"]))])
    yield
    for name in added:
        tool_registry._TOOLS.pop(name, None)


@pytest.mark.parametrize("name,result,expect", [
    ("zz_invented_span_ratio", {"ratio": 18.75, "drawing": "S-invented-12",
                                "members": ["B3", "B7"]}, ["**18.75**", "S-invented-12", "B3; B7",
                                                           "Span to depth ratio", "Span ratio check"]),
    ("zz_invented_pallet_count", {"lots": ["L-01", "L-02", "L-05"], "note": "DN-invented-4"},
     ["**3**", "DN-invented-4", "L-01; L-02; L-05", "Pallets on the delivery note",
      "Pallet counter"]),
])
def test_any_tool_declared_computed_is_recovered_by_its_kind(invented_tools, name, result, expect):
    out = runtime._recover_answer_from_tool_messages("", _turn("how many?", _tool_msg(name, result)))
    for piece in expect:
        assert piece in out


@pytest.mark.parametrize("name,error", [("zz_invented_span_ratio", "input missing"),
                                        ("zz_invented_pallet_count", "note not found")])
def test_a_failed_computed_tool_says_what_was_asked_and_that_it_failed(invented_tools, name, error):
    """The tool's own error text is never shown (it may carry a table, a
    path or an id): the plain sentence stands in its place."""
    from app.agents.core.computed_answer import LABELS

    ask = {"role": "assistant", "content": "", "tool_calls": [{
        "id": f"call-{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps({"drawing": "S-invented-9"})}}]}
    for ok in (False, True):
        failed = _tool_msg(name, {"status": "error", "error": error}, ok=ok)
        messages = _turn("how many on S-invented-9?", ask, failed)
        out = runtime._recover_answer_from_tool_messages("", messages)
        assert "unable to generate" not in out.lower()
        assert f"did not return: {LABELS['en']['tool_error']}" in out and "Asked: S-invented-9" in out
        assert error not in out, out
        assert FALLBACK not in runtime._postprocess_answer("", None, messages)


@pytest.mark.parametrize("name", ["zz_invented_span_ratio", "zz_invented_pallet_count"])
def test_a_result_the_tool_cannot_state_says_it_returned_no_figure(invented_tools, name):
    odd = _tool_msg(name, {"unexpected": "shape"})
    out = runtime._recover_answer_from_tool_messages("", _turn("how many?", odd))
    assert "did not return: the tool returned no figure" in out
    assert "unable to generate" not in out.lower()


SEARCH_RESULTS = [
    ("search_project_documents", {"message": "Found 4 passages for kerb in 2 documents",
                                  "summary": "Kerb clauses appear in section 7",
                                  "results": [{"text": "Kerb shall be precast", "score": 0.8}]}),
    ("search_general_knowledge", {"note": "3 knowledge passages on duct bedding",
                                  "message": "Duct bedding is sand 100 mm",
                                  "results": [{"text": "Bed ducts on sand", "score": 0.7}]}),
]


@pytest.mark.parametrize("synthesis", ["", FALLBACK])
@pytest.mark.parametrize("name,result", SEARCH_RESULTS)
def test_a_search_only_turn_with_empty_synthesis_keeps_the_honest_fallback(synthesis, name, result):
    messages = _turn("what does the spec say?", _tool_msg(name, result))
    out = runtime._postprocess_answer(synthesis, None, messages)
    assert FALLBACK in out
    for text in (result.get("message"), result.get("summary"), result.get("note")):
        if text:
            assert text not in out


@pytest.mark.parametrize("name,result", [
    ("search_project_documents", {"calculation": "invented_area", "note": "Passage: area 41 m2",
                                  "results": [{"text": "area 41 m2", "score": 0.9}]}),
    ("fetch_document", {"calculation": "invented_length", "summary": "Excerpt: 12 m of kerb",
                        "content": "12 m of kerb"}),
])
def test_a_search_result_never_speaks_whatever_fields_it_carries(name, result):
    """The declared kind outranks the shape: calculator-looking fields on a
    search or reader result are passages, not a computed figure."""
    messages = _turn("what does the drawing say?", _tool_msg(name, result))
    assert runtime._recover_answer_from_tool_messages("", messages) == ""
    out = runtime._postprocess_answer("", None, messages)
    assert (result.get("note") or result.get("summary")) not in out


@pytest.mark.parametrize("name,result", SEARCH_RESULTS)
@pytest.mark.parametrize("ask,bill,total,unit", BILLS)
def test_the_computed_result_wins_over_a_later_search_result(name, result, ask, bill, total, unit):
    messages = _turn(ask, _tool_msg("bill_rows_aggregate", bill), _tool_msg(name, result))
    out = runtime._recover_answer_from_tool_messages("", messages)
    assert f"**{total} {unit}**" in out
    assert result["message"] not in out


@pytest.mark.parametrize("kind,stater", [("computing", None), ("search", lambda r: [])])
def test_a_tool_declares_a_known_kind_and_only_computed_states(kind, stater):
    async def handle(call):  # pragma: no cover
        return {}
    with pytest.raises(ValueError):
        tool_registry.tool("zz_invented_bad", owner="base", display_name="Bad",
                           kind=kind, statement=stater)(handle)
    assert "zz_invented_bad" not in tool_registry._TOOLS


def test_a_computed_tool_without_a_statement_is_refused():
    async def handle(call):  # pragma: no cover
        return {}
    with pytest.raises(ValueError):
        tool_registry.tool("zz_invented_bare", owner="base", display_name="Bare",
                           kind="computed")(handle)
    assert "zz_invented_bare" not in tool_registry._TOOLS


@pytest.mark.parametrize("kind,agrees", [
    ("search", lambda got: got == set(RETRIEVAL_TOOLS)),
    ("computed", lambda got: "bill_rows_aggregate" in got and not got & set(RETRIEVAL_TOOLS)),
])
def test_search_kind_and_the_driver_retrieval_list_agree(kind, agrees):
    assert agrees(set(tool_registry.tools_of_kind(kind)))


@pytest.mark.parametrize("row,label", [
    ({"ref": "K.1", "pages": [3, 4], "description": "Precast kerb"}, "K.1 (p.3)"),
    ({"ref": " ", "pages": [], "description": "  Duct bedding in sand " + "x" * 80},
     ("Duct bedding in sand " + "x" * 80)[:60]),
])
def test_a_row_is_labelled_by_its_ref_and_first_page_else_its_description(row, label):
    assert _row_label(row) == label


@pytest.mark.parametrize("failed,reason", [
    ({"status": "error", "reason": "derived_sheet_is_not_a_readable_workbook",
      "filename": "bill-failed.xlsx", "terms": ["gully"]},
     "the bill sheet is not a readable workbook"),
    ({"status": "error", "error": "sheet locked by the invented host",
      "filename": "bill-locked.xlsx", "terms": ["gully"]}, "the tool reported an error"),
])
def test_a_bill_that_did_not_return_is_named_never_skipped(failed, reason):
    good = _bill("bill-gamma.xlsx", ["gully"], "nr", [("G.1", 2, 3.0)])["bills"][0]
    stated = bill_rows_statement({"bills": [failed, good, "not a bill"]})
    assert [s.source for s in stated] == [failed["filename"], "bill-gamma.xlsx", "bill-gamma.xlsx"]
    assert stated[0].figure is None and stated[0].failure
    assert stated[1].count == 1 and stated[2].figure == 3.0 and stated[2].unit == "nr"
    out = runtime._recover_answer_from_tool_messages("", _turn(
        "count the gullies", _tool_msg("bill_rows_aggregate", {"status": "success",
                                                              "bills": [failed, good]})))
    assert f"did not return ({reason}) (source: {failed['filename']})" in out
    # The bill's own words are never shown: a declared code reads as its
    # wording, anything else as the plain sentence.
    assert str(failed.get("error") or failed.get("reason")) not in out
    assert "**1**" in out and "**3 nr**" in out


@pytest.mark.parametrize("ask,result,total,unit", BILLS)
def test_each_bill_states_its_count_and_its_sum_per_unit_with_the_rows_behind(ask, result, total, unit):
    bill = result["bills"][0]
    count, summed = bill_rows_statement(result)
    labels = tuple(f"{r['ref']} (p.{r['pages'][0]})" for r in bill["rows"])
    assert (count.figure, count.count, count.rows) == (float(len(bill["rows"])), len(bill["rows"]), labels)
    assert (summed.figure, summed.unit, summed.rows) == (float(total), unit, labels)
    assert count.terms == summed.terms == tuple(bill["terms"])
    assert (count.basis, summed.basis, summed.what) == ("count", "sum", bill["measure"])


@pytest.mark.parametrize("terms,filename", [(["kerb"], "bill-empty.xlsx"),
                                             (["duct", "pvc"], "bill-bare.xlsx")])
def test_a_zero_row_result_says_no_matching_rows(terms, filename):
    result = {"status": "success", "bills": [{"status": "success", "filename": filename,
                                              "terms": terms, "rows": [], "sum": None}]}
    messages = _turn("count them", _tool_msg("bill_rows_aggregate", result))
    out = runtime._postprocess_answer("", None, messages)
    assert FALLBACK not in out
    assert f"holds {', '.join(terms)}: **0** (no matching rows) (source: {filename})" in out


@pytest.mark.parametrize("result,reason", [
    ({"status": "error", "reason": "no_derived_bill_sheet"},
     "no bill sheet has been derived for this project"),
    ({"status": "success", "bills": []}, "the tool returned no figure"),
])
def test_a_computed_call_with_nothing_to_state_says_why(result, reason):
    ask = {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call-bill_rows_aggregate", "type": "function",
        "function": {"name": "bill_rows_aggregate", "arguments": {"terms": "kerb", "section": "7"}}}]}
    messages = _turn("count the kerb runs in section 7", ask, _tool_msg("bill_rows_aggregate", result))
    out = runtime._postprocess_answer("", None, messages)
    assert FALLBACK not in out
    assert f"did not return: {reason}" in out and "Asked: kerb, 7" in out


HOSTILE_ERRORS = [
    ("relation invented_bill_rows_internal does not exist",
     ["invented_bill_rows_internal", "relation"]),
    (r"could not open C:\Program Files\Invented Data\rows store.db (/srv/invented x/y.db)",
     ["Program Files", "Invented Data", "rows store", "/srv/invented x"]),
]


@pytest.mark.parametrize("error,absent", HOSTILE_ERRORS)
def test_a_mixed_turn_never_pastes_the_failed_calls_error(invented_tools, error, absent):
    """One computed call succeeds with no figure, another fails with an error
    holding a table name and a path: the answer states both calls and holds
    neither the table nor the path (no old formatter on the way)."""
    from app.agents.core.computed_answer import LABELS

    messages = _turn("how many?",
                     _tool_msg("zz_invented_span_ratio", {"unexpected": "shape"}),
                     _tool_msg("zz_invented_pallet_count", {"status": "error", "error": error}, ok=False))
    for out in (runtime._recover_answer_from_tool_messages("", messages),
                runtime._postprocess_answer("", None, messages)):
        assert "Span ratio check" in out and "Pallet counter" in out, out
        assert LABELS["en"]["tool_error"] in out
        assert "unable to generate" not in out.lower()
        for piece in absent:
            assert piece not in out, (piece, out)


@pytest.mark.parametrize("error,absent", HOSTILE_ERRORS)
def test_a_tool_error_json_answer_becomes_the_plain_sentence(error, absent):
    """The other caller of the error formatter: a reply that is a tool-error
    envelope reads as the plain sentence, never the error text."""
    raw = json.dumps({"status": "error", "error": error, "request_id": "req-invented-1"})
    out = runtime._sanitize_final_text(raw)
    assert out.startswith("The tool could not complete that request.")
    assert "req-invented-1" in out
    for piece in absent:
        assert piece not in out, (piece, out)


@pytest.mark.parametrize("error,absent", HOSTILE_ERRORS)
def test_a_failed_calculation_not_declared_computed_reads_the_plain_sentence(error, absent):
    messages = _turn("work it out", _tool_msg("zz_undeclared_invented_calc", {
        "status": "error", "error": error, "calculation": "invented_volume"}))
    out = runtime._recover_answer_from_tool_messages("", messages)
    assert out.startswith("The tool could not complete that request.")
    for piece in absent:
        assert piece not in out, (piece, out)


@pytest.mark.parametrize("rid,shown", [("req-invented-2", True), (r"C:\Invented Dir\x", False)])
def test_a_request_id_is_shown_only_when_it_is_an_id(rid, shown):
    out = runtime._plain_tool_error_sentence({"status": "error", "error": "x", "request_id": rid})
    assert (rid in out) is shown


@pytest.mark.parametrize("first,second", [(0, 1), (1, 0)])
def test_every_computed_call_of_the_turn_is_shown_in_call_order(first, second):
    a, b = BILLS[first], BILLS[second]
    messages = _turn("both please", _tool_msg("bill_rows_aggregate", a[1]),
                     _tool_msg("bill_rows_aggregate", b[1]))
    out = runtime._recover_answer_from_tool_messages("", messages)
    assert "2 computed results in this turn" in out.rstrip().splitlines()[-1], out
    first = out.splitlines()[0]
    assert first.startswith("Bill row count and total: ") and any(ch.isdigit() for ch in first), out
    one, two = a[1]["bills"][0]["filename"], b[1]["bills"][0]["filename"]
    assert one in out and two in out and out.index(one) < out.index(two)


@pytest.mark.parametrize("before,now", [(0, 1), (1, 0)])
def test_a_computed_call_of_an_earlier_turn_does_not_answer_this_one(before, now):
    earlier = _turn(BILLS[before][0], _tool_msg("bill_rows_aggregate", BILLS[before][1]))
    thread = [*earlier, {"role": "assistant", "content": "earlier answer"},
              {"role": "user", "content": BILLS[now][0]},
              _tool_msg("bill_rows_aggregate", BILLS[now][1])]
    out = runtime._recover_answer_from_tool_messages("", thread)
    assert BILLS[now][1]["bills"][0]["filename"] in out
    assert BILLS[before][1]["bills"][0]["filename"] not in out


ARABIC_QUESTION = "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0628\u0646\u0648\u062f\u061f"


@pytest.mark.parametrize("ask,result,total,unit", BILLS)
def test_the_recovered_answer_follows_the_language_of_the_question(ask, result, total, unit):
    from app.agents.core.computed_answer import LABELS
    bill = result["bills"][0]
    english = runtime._recover_answer_from_tool_messages("", _turn(ask, _tool_msg(
        "bill_rows_aggregate", result)))
    arabic = runtime._recover_answer_from_tool_messages("", _turn(
        ARABIC_QUESTION, _tool_msg("bill_rows_aggregate", result)))
    ar, en = LABELS["ar"], LABELS["en"]
    assert en["count"] in english and en["count"] not in arabic
    for key in ("count", "source", "rows", "heading"):
        assert ar[key] in arabic
    assert ar["sum"].format(what=ar["quantity"]) in arabic
    for latin_label in ("source:", "Rows:", "summed", "whose description", "Bill row"):
        assert latin_label not in arabic
    assert f"**{total} {unit}**" in arabic and bill["filename"] in arabic


@pytest.mark.parametrize("question,language", [
    (ARABIC_QUESTION, "ar"), ("how long are the kerbs?", "en"), ("", "en"), ("12 / 7", "en"),
    ("kerb " + ARABIC_QUESTION, "ar")])
def test_the_question_language_is_the_script_most_of_its_letters_are_in(question, language):
    from app.agents.core.computed_answer import question_language
    assert question_language(question) == language


def test_every_language_carries_every_label():
    from app.agents.core.computed_answer import LABELS
    keys = set(LABELS["en"])
    assert all(set(table) == keys for table in LABELS.values())


@pytest.mark.parametrize("name", ["zz_invented_twice_a", "zz_invented_twice_b"])
def test_a_second_handler_under_one_name_is_refused_and_the_first_kept(name):
    async def first(call):  # pragma: no cover
        return {}

    async def second(call):  # pragma: no cover
        return {}
    register = tool_registry.tool(name, owner="base", display_name="Twice")
    try:
        register(first)
        assert register(first) is first  # the same handler again is not a second one
        with pytest.raises(ValueError):
            register(second)
        assert tool_registry.get(name).handler is first
    finally:
        tool_registry._TOOLS.pop(name, None)
