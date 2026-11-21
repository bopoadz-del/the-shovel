"""A stated total the turn's computed tool stated is never lowered to the sum
of an incomplete printed table (answer_row_totals rule 5), and a markdown
table's rule line and bold Total row are read as layout, not as rows.

Every row set, reference, label and figure below is constructed here; none is
copied from any document or live answer. Each decision runs on two unrelated
inputs (a count set and a length set).
"""
from __future__ import annotations

import json

import pytest

from app.agents.answer_counted_rows import name_dropped_rows
from app.agents.answer_row_totals import computed_totals, reconcile
from app.agents.base.tools import bill_rows_statement
from app.agents.core.tool_registry import Statement

# (ref, page, figure, unit): two unrelated sets.
SET_COUNT = (
    ("J 21", 4, 9.0, "No"),
    ("J 22", 4, 14.0, "No"),
    ("J 23", 5, 2.0, "No"),
    ("J 24", 6, 7.0, "No"),
)
SET_LENGTH = (
    ("V3.1", 30, 212.0, "m"),
    ("V3.2", 30, 47.5, "m"),
    ("V3.3", 31, 105.0, "m"),
)


def _bill(rows) -> dict:
    unit = rows[0][3]
    total = sum(r[2] for r in rows)
    return {"status": "success", "bills": [{
        "document_id": "doc-y", "filename": "made-up schedule.xlsx", "status": "success",
        "measure": "quantity", "terms": ["run"], "section_terms": [],
        "count": len(rows), "sum": total, "unit": unit, "sum_by_unit": {unit: total},
        "rows": [{"ref": ref, "pages": [page], "section": ["S"], "description": "d",
                  "unit": u, "quantity": fig} for ref, page, fig, u in rows],
    }]}


def _fmt(value: float) -> str:
    return f"{value:g}"


def _total(rows) -> str:
    return _fmt(sum(r[2] for r in rows))


def _markdown_answer(rows, printed, *, total: str | None = None) -> str:
    """A headline that states the total, then a markdown table (rule line and
    a bold Total row) that prints only the rows in ``printed``."""
    unit = rows[0][3]
    total = total or _total(rows)
    lines = [
        f"The sum of the matching rows is {total} {unit}.",
        "",
        f"| Item | Ref | Qty ({unit}) |",
        "|---|:---|---:|",
    ]
    lines += [f"| {chr(65 + i)} | {rows[i][0]} | {_fmt(rows[i][2])} |" for i in printed]
    lines.append(f"| **Total** | | **{total}** |")
    return "\n".join(lines)


CASES = [
    pytest.param(SET_COUNT, (0, 1, 3), (2,), id="count-one-row-left-out"),
    pytest.param(SET_LENGTH, (1, 2), (0,), id="length-first-row-left-out"),
]


@pytest.mark.parametrize("rows,printed,_missing", CASES)
def test_a_total_the_computed_tool_stated_is_not_lowered(rows, printed, _missing):
    answer = _markdown_answer(rows, printed)
    statements = bill_rows_statement(_bill(rows))
    assert reconcile(answer, statements) == answer


@pytest.mark.parametrize("rows,printed,_missing", CASES)
def test_without_the_computed_statement_the_same_table_is_still_checked(
    rows, printed, _missing,
):
    """The control: the answer's text alone cannot tell a dropped row from a
    wrong total, so without the turn's statement the old rule still applies."""
    answer = _markdown_answer(rows, printed)
    printed_sum = _fmt(sum(rows[i][2] for i in printed))
    out = reconcile(answer)
    assert out != answer
    assert f"{printed_sum} {rows[0][3]}" in out.split("\n")[0]


@pytest.mark.parametrize("rows,printed,_missing", CASES)
def test_a_computed_total_that_differs_vouches_for_nothing(rows, printed, _missing):
    answer = _markdown_answer(rows, printed)
    other = (Statement(figure=sum(r[2] for r in rows) + 3.0, what="quantity",
                       unit=rows[0][3], basis="sum"),)
    assert reconcile(answer, other) != answer


@pytest.mark.parametrize("rows,printed,_missing", CASES)
def test_a_computed_total_in_another_unit_vouches_for_nothing(rows, printed, _missing):
    answer = _markdown_answer(rows, printed)
    other_unit = "m" if rows[0][3] == "No" else "No"
    other = (Statement(figure=sum(r[2] for r in rows), what="quantity",
                       unit=other_unit, basis="sum"),)
    assert reconcile(answer, other) != answer


@pytest.mark.parametrize("rows,printed,_missing", CASES)
def test_a_computed_total_with_no_unit_still_vouches(rows, printed, _missing):
    answer = _markdown_answer(rows, printed)
    bare = (Statement(figure=sum(r[2] for r in rows), what="quantity", basis="sum"),)
    assert reconcile(answer, bare) == answer


@pytest.mark.parametrize("rows,printed,_missing", CASES)
def test_a_count_is_not_a_total_of_rows(rows, printed, _missing):
    """A count statement whose figure happens to equal the stated total is a
    number of rows, not the rows' sum, so it vouches for nothing."""
    answer = _markdown_answer(rows, printed)
    count = (Statement(figure=sum(r[2] for r in rows), what="quantity",
                       unit=rows[0][3], basis="count"),)
    assert reconcile(answer, count) != answer
    assert computed_totals(count) == []


@pytest.mark.parametrize("statements", [
    pytest.param((Statement(figure=None, what="", basis="sum", failure="timeout"),),
                 id="sum-that-failed"),
    pytest.param((Statement(figure=None, what="", unit="m", basis="sum"),
                  Statement(figure=4.0, what="", unit="m", basis="")), id="no-figure-or-no-sum"),
    pytest.param(None, id="no-statements"),
])
def test_a_statement_that_did_not_return_states_no_total(statements):
    assert computed_totals(statements) == []


@pytest.mark.parametrize("rows,printed,missing", CASES)
def test_the_answer_exit_keeps_the_total_and_names_the_rows_left_out(rows, printed, missing):
    """Through the exit the user reads: the total stays the tool's, and the
    rows the table left out are named from the same computed set."""
    from app.agents.runtime import _postprocess_answer

    payload = {"name": "bill_rows_aggregate", "ok": True, "result": _bill(rows)}
    messages = [
        {"role": "user", "content": "What is the total billed?"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {
            "name": "bill_rows_aggregate", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "bill_rows_aggregate",
         "content": json.dumps(payload)},
    ]
    unit = rows[0][3]
    out = _postprocess_answer(_markdown_answer(rows, printed), None, messages)
    assert f"is {_total(rows)} {unit}" in out
    assert "Row total checked" not in out
    for i in missing:
        ref, page, fig, _u = rows[i]
        assert f"{ref} (p.{page}): {_fmt(fig)} {unit}" in out
    assert f"prints {len(printed)} of them" in out


@pytest.mark.parametrize("rows,printed,missing", CASES)
def test_a_bold_total_row_is_read_as_the_total_when_naming_left_out_rows(
    rows, printed, missing,
):
    out = name_dropped_rows(_markdown_answer(rows, printed), bill_rows_statement(_bill(rows)))
    for i in missing:
        assert f"{rows[i][0]} (p.{rows[i][1]})" in out


@pytest.mark.parametrize("answer,computed,unit,rows", [
    pytest.param(
        "| Item | Qty (No) |\n|---|---|\n| A | 18 |\n| B | 26 |\n| C | 64 |\n"
        "| **Total** | **117** |",
        "108", "No", 3, id="bold-count",
    ),
    pytest.param(
        "| Run | Length (m) |\n| :--- | ---: |\n| P | 633 |\n| Q | 488 |\n"
        "| __Total__ | __1,900__ |",
        "1,121", "m", 2, id="underscore-length-aligned-rule",
    ),
])
def test_a_markdown_rule_line_and_bold_total_are_layout_not_rows(answer, computed, unit, rows):
    out = reconcile(answer)
    assert f"sum to {computed} {unit} over {rows} rows" in out
    # Neither the rule line nor the Total row is a row printed without a figure.
    assert "Printed without a readable figure" not in out
    assert f"| {computed} |" in out


@pytest.mark.parametrize("answer,stated,unit", [
    pytest.param(
        "The rows sum to 40 No.\n\n| Item | Qty (No) |\n|---|---|\n| A | 7 |\n"
        "| B | 9 |\n| C | 11 |",
        "40", "No", id="prose-count",
    ),
    pytest.param(
        "Altogether 2,000 m is billed.\n\n| Run | Length (m) |\n|:--|--:|\n| P | 633 |\n"
        "| Q | 488 |",
        "2,000", "m", id="prose-length",
    ),
])
def test_a_rule_line_is_not_named_under_a_prose_total(answer, stated, unit):
    out = reconcile(answer)
    assert out != answer
    assert "Printed without a readable figure" not in out
    held = (Statement(figure=float(stated.replace(",", "")), what="q", unit=unit,
                      basis="sum"),)
    assert reconcile(answer, held) == answer


@pytest.mark.parametrize("answer,label", [
    pytest.param("| Item | Qty (No) |\n|---|---|\n| A | 7 |\n| B | - |\n| C | 5 |\n"
                 "| Total | 20 |", "B", id="dash-cell"),
    pytest.param("| Run | Length (m) |\n|---|---|\n| P | 633 |\n| Q | -- |\n| R | 120 |\n"
                 "| Total | 900 |", "Q", id="double-dash-cell"),
])
def test_a_row_with_a_dash_for_its_figure_is_still_a_row(answer, label):
    out = reconcile(answer)
    assert "Printed without a readable figure, so not in the total:" in out
    assert label in out.split("**Row total checked")[1]
    assert "---" not in out.split("**Row total checked")[1]


@pytest.mark.parametrize("answer,stated,unit", [
    pytest.param("We have 18 + 26 + 64 = 117 No in all.", 117.0, "No", id="count-chain"),
    pytest.param("Runs: 633 + 488 + 120 = 1,400 m.", 1400.0, "m", id="length-chain"),
])
def test_a_chain_whose_total_the_tool_stated_is_not_lowered(answer, stated, unit):
    assert reconcile(answer) != answer
    held = (Statement(figure=stated, what="q", unit=unit, basis="sum"),)
    assert reconcile(answer, held) == answer


@pytest.mark.parametrize("answer,stated,unit", [
    pytest.param("We have 18 + 26 + … = 117 No in all.", 117.0, "No", id="count-ellipsis"),
    pytest.param("Runs: 633 + 488 + ... = 1,400 m.", 1400.0, "m", id="length-ellipsis"),
])
def test_a_chain_that_admits_it_is_incomplete_is_still_said_to_be(answer, stated, unit):
    held = (Statement(figure=stated, what="q", unit=unit, basis="sum"),)
    out = reconcile(answer, held)
    assert "row list is incomplete" in out


@pytest.mark.parametrize("rows,printed,_missing", CASES)
def test_the_held_total_leaves_an_operator_trace(rows, printed, _missing, caplog):
    answer = _markdown_answer(rows, printed)
    with caplog.at_level("INFO", logger="app.agents.answer_row_totals"):
        reconcile(answer, bill_rows_statement(_bill(rows)))
    assert any("row_total_held_by_computed columns=1 chains=0" in r.getMessage()
               for r in caplog.records)
    caplog.clear()
    with caplog.at_level("INFO", logger="app.agents.answer_row_totals"):
        reconcile(answer)
    assert not any("row_total_held_by_computed" in r.getMessage() for r in caplog.records)
