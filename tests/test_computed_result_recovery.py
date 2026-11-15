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
    assert f"**{bill['count']} rows**" in out
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
     ["**3 rows**", "DN-invented-4", "L-01; L-02; L-05", "Pallets on the delivery note",
      "Pallet counter"]),
])
def test_any_tool_declared_computed_is_recovered_by_its_kind(invented_tools, name, result, expect):
    out = runtime._recover_answer_from_tool_messages("", _turn("how many?", _tool_msg(name, result)))
    for piece in expect:
        assert piece in out


@pytest.mark.parametrize("name", ["zz_invented_span_ratio", "zz_invented_pallet_count"])
def test_a_failed_computed_tool_states_nothing(invented_tools, name):
    failed = _tool_msg(name, {"status": "error", "error": "input missing"}, ok=False)
    assert runtime._recover_answer_from_tool_messages("", _turn("how many?", failed)) == ""
    refused = _tool_msg(name, {"status": "error", "error": "input missing"})
    assert runtime._recover_answer_from_tool_messages("", _turn("how many?", refused)) == ""


@pytest.mark.parametrize("name", ["zz_invented_span_ratio", "zz_invented_pallet_count"])
def test_a_result_the_tool_cannot_state_keeps_the_honest_fallback(invented_tools, name):
    odd = _tool_msg(name, {"unexpected": "shape"})
    assert runtime._recover_answer_from_tool_messages("", _turn("how many?", odd)) == ""


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


def test_search_kind_and_the_driver_retrieval_list_agree():
    assert set(tool_registry.tools_of_kind("search")) == set(RETRIEVAL_TOOLS)
    assert "bill_rows_aggregate" in tool_registry.tools_of_kind("computed")
