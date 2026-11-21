"""A table drawn from the turn's computed row set prints every row of it, or
the rows it left out are named from that set (app/agents/answer_counted_rows).

Every row set, reference, label and figure below is constructed here; none
is copied from any document or live answer.
"""
from __future__ import annotations

import json

import pytest

from app.agents.answer_counted_rows import name_dropped_rows, row_sets
from app.agents.base.tools import bill_rows_statement
from app.agents.core.tool_registry import Statement

# Two unrelated row sets: (ref, page, description words, figure, unit).
SET_COUNT = (
    ("Q 41", 7, "band one", 12.0, "No"),
    ("Q 42", 7, "band two", 3.0, "No"),
    ("Q 43", 8, "band three", 20.0, "No"),
    ("Q 44", 8, "band four", 5.0, "No"),
)
SET_LENGTH = (
    ("W7.1", 21, "run alpha", 140.5, "m"),
    ("W7.2", 21, "run beta", 66.0, "m"),
    ("W7.3", 22, "run gamma", 310.25, "m"),
)


def _bill(rows) -> dict:
    """A bill_rows_aggregate result, shaped as the tool returns it."""
    unit = rows[0][4]
    return {"status": "success", "bills": [{
        "document_id": "doc-x", "filename": "invented bill.xlsx", "status": "success",
        "measure": "quantity", "terms": ["band"], "section_terms": [],
        "count": len(rows), "sum": sum(r[3] for r in rows), "unit": unit,
        "sum_by_unit": {unit: sum(r[3] for r in rows)},
        "rows": [{"ref": ref, "pages": [page], "section": ["S"], "description": desc,
                  "unit": u, "quantity": fig} for ref, page, desc, fig, u in rows],
    }]}


def _fmt(value: float) -> str:
    return f"{value:g}"


def _answer(rows, printed, *, total_line: bool = True) -> str:
    unit = rows[0][4]
    total = sum(r[3] for r in rows)
    head = f"{_fmt(total)} {unit} in total across the matching rows.\n\n" if total_line else ""
    lines = [f"Item\tRef\tQty ({unit})"]
    lines += [f"{chr(65 + i)}\t{rows[i][0]}\t{_fmt(rows[i][3])}" for i in printed]
    return head + "\n".join(lines) + "\n\nThese are the rows the bill prints."


def _statements(rows):
    return bill_rows_statement(_bill(rows))


@pytest.mark.parametrize("rows,printed,dropped", [
    pytest.param(SET_COUNT, (0, 1, 3), (2,), id="count-one-dropped"),
    pytest.param(SET_LENGTH, (1, 2), (0,), id="length-first-dropped"),
    pytest.param(SET_COUNT, (0, 3), (1, 2), id="count-two-dropped"),
])
def test_a_row_the_table_dropped_is_named_from_the_row_set(rows, printed, dropped):
    answer = _answer(rows, printed)
    out = name_dropped_rows(answer, _statements(rows))
    assert out != answer
    unit = rows[0][4]
    note = next(line for line in out.split("\n") if line.startswith("The "))
    assert f"counts {len(rows)} rows" in note
    assert f"prints {len(printed)} of them" in note
    for i in dropped:
        ref, page, _d, fig, _u = rows[i]
        assert f"{ref} (p.{page}): {_fmt(fig)} {unit}" in note
    for i in printed:
        assert f"{rows[i][0]} (p." not in note
    # The table and the answer's own words are kept as they were.
    assert out.startswith(answer.split("\n\nThese")[0])
    assert out.endswith("These are the rows the bill prints.")


@pytest.mark.parametrize("rows", [SET_COUNT, SET_LENGTH])
def test_a_table_that_prints_every_row_is_returned_byte_for_byte(rows):
    answer = _answer(rows, tuple(range(len(rows))))
    assert name_dropped_rows(answer, _statements(rows)) == answer


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 1)), (SET_LENGTH, (0, 2))])
def test_an_answer_that_does_not_state_the_set_total_is_left_alone(rows, printed):
    """A table of some rows, with no claim to the set's total, may be a chosen
    subset: nothing is named."""
    answer = _answer(rows, printed, total_line=False)
    assert name_dropped_rows(answer, _statements(rows)) == answer


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 1)), (SET_LENGTH, (1, 2))])
def test_a_table_with_a_row_the_set_does_not_hold_is_not_a_rendering_of_it(rows, printed):
    answer = _answer(rows, printed).replace(
        "\n\nThese", "\nZ\tX 999\t4\n\nThese",
    )
    assert name_dropped_rows(answer, _statements(rows)) == answer


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 1)), (SET_LENGTH, (0, 1))])
def test_a_table_whose_figures_are_not_the_sets_is_left_alone(rows, printed):
    i = printed[0]
    wrong = _fmt(rows[i][3] + 1)
    answer = _answer(rows, printed).replace(
        f"\t{rows[i][0]}\t{_fmt(rows[i][3])}", f"\t{rows[i][0]}\t{wrong}",
    )
    assert name_dropped_rows(answer, _statements(rows)) == answer


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 0, 1)), (SET_LENGTH, (2, 1, 2))])
def test_a_row_printed_twice_is_left_alone(rows, printed):
    answer = _answer(rows, printed)
    assert name_dropped_rows(answer, _statements(rows)) == answer


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 1)), (SET_LENGTH, (0, 2))])
def test_a_set_that_states_no_per_row_figures_is_not_checked(rows, printed):
    """Without the figure each row carries there is nothing to name a dropped
    row's contribution with, so nothing is named."""
    bare = [Statement(figure=st.figure, what=st.what, unit=st.unit, rows=st.rows,
                      basis=st.basis) for st in _statements(rows)]
    answer = _answer(rows, printed)
    assert name_dropped_rows(answer, bare) == answer


@pytest.mark.parametrize("rows", [SET_COUNT, SET_LENGTH])
def test_the_bill_stater_carries_each_rows_figure_in_row_order(rows):
    sums = [st for st in _statements(rows) if st.basis == "sum"]
    assert len(sums) == 1
    assert sums[0].row_figures == tuple(r[3] for r in rows)
    assert len(sums[0].rows) == len(rows)


@pytest.mark.parametrize("rows", [SET_COUNT, SET_LENGTH])
def test_a_set_repeated_by_several_calls_is_read_once(rows):
    assert len(row_sets(_statements(rows) * 3)) == 1


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 2, 3)), (SET_LENGTH, (0, 1))])
def test_a_pipe_delimited_table_is_checked_too(rows, printed):
    unit = rows[0][4]
    total = sum(r[3] for r in rows)
    body = "\n".join([f"| Ref | Qty ({unit}) |", "|---|---|"]
                     + [f"| {rows[i][0]} | {_fmt(rows[i][3])} |" for i in printed])
    answer = f"Total {_fmt(total)} {unit}.\n\n{body}"
    out = name_dropped_rows(answer, _statements(rows))
    missing = [i for i in range(len(rows)) if i not in printed]
    for i in missing:
        assert f"{rows[i][0]} (p.{rows[i][1]})" in out


@pytest.mark.parametrize("statements", [None, [], "not statements"])
def test_no_statements_returns_the_answer_byte_for_byte(statements):
    answer = _answer(SET_COUNT, (0, 1))
    assert name_dropped_rows(answer, statements) == answer


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (1, 2, 3)), (SET_LENGTH, (0, 2))])
def test_the_answer_exit_names_the_dropped_row_from_the_turns_tool_result(rows, printed):
    """Through the exit the user reads: the row set is the turn's own
    bill_rows_aggregate result, read from its tool message."""
    from app.agents.runtime import _postprocess_answer

    payload = {"name": "bill_rows_aggregate", "ok": True, "result": _bill(rows)}
    messages = [
        {"role": "user", "content": "How many are billed in total?"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {
            "name": "bill_rows_aggregate", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "bill_rows_aggregate",
         "content": json.dumps(payload)},
    ]
    out = _postprocess_answer(_answer(rows, printed), None, messages)
    missing = [i for i in range(len(rows)) if i not in printed]
    for i in missing:
        assert f"{rows[i][0]} (p.{rows[i][1]}): {_fmt(rows[i][3])} {rows[i][4]}" in out
    assert f"prints {len(printed)} of them" in out


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 1, 2)), (SET_LENGTH, (0, 2))])
def test_a_table_that_carries_its_own_total_row_still_names_the_dropped_rows(rows, printed):
    unit = rows[0][4]
    total = _fmt(sum(r[3] for r in rows))
    answer = _answer(rows, printed).replace("\n\nThese", f"\nTotal\t\t{total}\n\nThese")
    out = name_dropped_rows(answer, _statements(rows))
    for i in (i for i in range(len(rows)) if i not in printed):
        assert f"{rows[i][0]} (p.{rows[i][1]}): {_fmt(rows[i][3])} {unit}" in out
    assert f"Total\t\t{total}" in out


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 1)), (SET_LENGTH, (1, 2))])
def test_a_table_cell_equal_to_the_total_is_not_the_answer_stating_it(rows, printed):
    """A figure inside the table (a line or chainage column) that happens to
    equal the set's total is a cell of a row, not a claim to the total."""
    total = _fmt(sum(r[3] for r in rows))
    plain = _answer(rows, printed, total_line=False)
    lines = plain.split("\n")
    lines[0] = "Line\t" + lines[0]
    for n in range(1, len(printed) + 1):
        lines[n] = f"{total}\t" + lines[n]
    answer = "\n".join(lines)
    assert name_dropped_rows(answer, _statements(rows)) == answer


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 1)), (SET_LENGTH, (0, 2))])
def test_a_count_is_not_a_summed_row_set(rows, printed):
    """A count of rows names how many there are, not a total their figures
    make: a table is checked only against a sum."""
    labels = tuple(f"{r[0]} (p.{r[1]})" for r in rows)
    count = Statement(figure=float(len(rows)), what="quantity", rows=labels, count=len(rows),
                      basis="count", row_figures=tuple(r[3] for r in rows))
    answer = _answer(rows, printed, total_line=False).replace(
        "These are", f"{len(rows)} rows match; these are",
    )
    assert name_dropped_rows(answer, [count]) == answer


@pytest.mark.parametrize("first,second,printed", [
    (SET_LENGTH, SET_COUNT, (0, 2)),
    (SET_COUNT, SET_LENGTH, (1, 2)),
])
def test_each_table_is_read_on_its_own(first, second, printed):
    """A table that renders no set does not stop a later table that drops a
    row of its set from being checked."""
    unrelated = "Note\tValue\nalpha\tone\nbeta\ttwo"
    answer = unrelated + "\n\n" + _answer(second, printed)
    statements = _statements(second) + _statements(first)
    out = name_dropped_rows(answer, statements)
    assert out.startswith(unrelated)
    for i in (i for i in range(len(second)) if i not in printed):
        assert f"{second[i][0]} (p.{second[i][1]})" in out


@pytest.mark.parametrize("rows,printed", [(SET_COUNT, (0, 1, 2)), (SET_LENGTH, (1, 2))])
def test_only_an_answer_with_named_rows_leaves_the_operator_trace(rows, printed, caplog):
    """The trace line counts the tables and rows named; an answer left alone
    (every row printed) writes no trace at all."""
    import logging

    caplog.set_level(logging.INFO, logger="app.agents.answer_counted_rows")
    complete = _answer(rows, tuple(range(len(rows))))
    assert name_dropped_rows(complete, _statements(rows)) == complete
    assert "counted_rows_named" not in caplog.text
    name_dropped_rows(_answer(rows, printed), _statements(rows))
    missing = len(rows) - len(printed)
    assert f"counted_rows_named tables=1 rows={missing}" in caplog.text
