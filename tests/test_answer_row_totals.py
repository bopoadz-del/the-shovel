"""A stated row total equals the sum of the rows the answer itself prints.

Every test builds its own answer text; there is no fixture file, no document,
no item name and no figure taken from a real bill. Each test runs on two
independently built inputs, so a rule that only holds for the wording it was
written against fails here.
"""
from __future__ import annotations

import pytest

from app.agents.answer_row_totals import reconcile


def _table(header: tuple[str, ...], rows: tuple[tuple[str, ...], ...],
           total_label: str, total: str, pad: str = "") -> str:
    """An answer table built from its parts.

    ``pad`` is the empty columns the Total row prints before its figure, so a
    table whose figure sits in the last column reads like the answer's own.
    """
    body = ["\t".join(header)]
    body.extend("\t".join(row) for row in rows)
    body.append(f"{total_label}\t{pad}\t{total}")
    return "\n".join(body)


#: Two tables that agree with their own column: same rule, different shapes
#: (two rows and three rows, metres and counts, different separators).
AGREEING = [
    pytest.param(
        _table(("Item", "Length (m)"), (("A", "633"), ("B", "488")), "Total", "1,121"),
        id="two-row-metres",
    ),
    pytest.param(
        "| Item | Qty (No) |\n| A | 18 |\n| B | 26 |\n| C | 64 |\n| Total | 108 |",
        id="pipe-three-row-count",
    ),
]


@pytest.mark.parametrize("answer", AGREEING)
def test_a_total_that_equals_its_column_is_returned_byte_for_byte(answer):
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,stated,computed,unit,rows", [
    pytest.param(
        _table(("Item", "Length (m)"), (("A", "633"), ("B", "488")), "Total", "1,900"),
        "1,900", "1,121", "m", 2, id="two-row-metres",
    ),
    pytest.param(
        _table(("Item", "Qty (No)"), (("A", "18"), ("B", "26"), ("C", "64")), "Total", "117"),
        "117", "108", "No", 3, id="three-row-count",
    ),
])
def test_a_total_that_disagrees_is_the_sum_of_the_rows_printed(
    answer, stated, computed, unit, rows,
):
    out = reconcile(answer)
    assert f"Total\t\t{computed}" in out
    assert f"sum to {computed} {unit} over {rows} rows" in out
    assert "the total is incomplete" in out


@pytest.mark.parametrize("answer,stated,computed,unit", [
    pytest.param(
        _table(("Item", "Length (m)"), (("A", "633"), ("B", "488")), "Total", "1,900")
        + "\n\nPage 1,900 records the survey date.",
        "1,900", "1,121", "m", id="page-number-has-no-unit",
    ),
    pytest.param(
        _table(("Item", "Qty (No)"), (("A", "18"), ("B", "26")), "Total", "50")
        + "\n\nThe drawing's sheet number is 50 in the appendix.",
        "50", "44", "No", id="sheet-number-has-no-unit",
    ),
])
def test_a_figure_without_the_column_unit_is_never_rewritten(answer, stated, computed, unit):
    out = reconcile(answer)
    # The prose sentence is untouched; only the Total cell is corrected.
    assert f" {stated} " in out or out.count(stated) == 0
    assert f"Total\t\t{computed}" in out
    assert out.count(stated) <= 1


@pytest.mark.parametrize("answer,computed,bucket", [
    pytest.param(
        _table(("Item", "Ref", "Qty (No)"),
               (("A", "K 132", "18"), ("B", "K 133", "26"),
                ("C", "K 134", "64"),
                ("\u2014", "remaining rows (K 134)", "9")),
               "Total", "117"),
        "108", "remaining rows (K 134)", id="bucket-repeats-a-counted-reference",
    ),
    pytest.param(
        _table(("Item", "Ref", "Qty (No)"),
               (("A", "K 7", "7"), ("B", "K 8", "5"),
                ("plus the others (K 8)", "", "8")),
               "Total", "20"),
        "12", "the others (K 8)", id="others-bucket-repeats-a-reference",
    ),
])
def test_a_remainder_bucket_that_restates_a_counted_row_is_not_added_twice(
    answer, computed, bucket,
):
    out = reconcile(answer)
    assert f"Total\t\t{computed}" in out
    assert f"{bucket} restates rows the answer has already counted" in out
    assert "not added a second time" in out


@pytest.mark.parametrize("answer", [
    pytest.param(
        _table(("Item", "Depth", "Qty (No)"),
               (("A", "1.5-2.0 m", "18"), ("B", "2.0-2.5 m", "26"),
                ("C", "2.5-3.0 m", "64"),
                ("\u2014", "remaining depth bands", "9")),
               "Total", "117"),
        id="bucket-prints-its-figure",
    ),
    pytest.param(
        _table(("Item", "Qty (No)"),
               (("A", "7"), ("B", "5"), ("plus the others in the section", "8")),
               "Total", "20"),
        id="others-bucket-prints-its-figure",
    ),
    pytest.param(
        _table(("Item", "Depth", "Qty (No)"),
               (("A", "1.5-2.0 m", "18"), ("B", "2.0-2.5 m", "26"),
                ("\u2014", "remaining rows (13 in total)", "\u2014")),
               "Total", "44"),
        id="bucket-prints-no-figure",
    ),
])
def test_a_bucket_the_answer_does_not_tie_to_a_counted_row_never_rewrites_its_total(
    answer,
):
    # The answer says in its own words that rows it did not itemise exist and
    # that its total is larger than the rows it printed. That is a disclosure,
    # not a disagreement: the printed evidence cannot settle the total, so the
    # check leaves the answer alone rather than inventing a smaller figure.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,computed", [
    pytest.param(
        _table(("Item", "Ref", "Qty (No)"),
               (("A", "K 1", "18"), ("B", "K 2", "\u2014"),
                ("\u2014", "remaining rows (K 2)", "9")),
               "Total", "27"),
        "18", id="reference-printed-without-a-figure",
    ),
    pytest.param(
        _table(("Item", "Ref", "Qty (No)"),
               (("A", "K 4", "6"), ("B", "K 5", "not read"),
                ("plus the others (K 5)", "", "3")),
               "Total", "11"),
        "6", id="words-instead-of-a-figure",
    ),
])
def test_a_bucket_naming_a_row_the_answer_could_not_count_stays_a_disclosure(
    answer, computed,
):
    # The named row printed no figure, so it was never counted: the bucket is
    # not a restatement of it, and the total above the printed rows is left to
    # the answer's own disclosure.
    assert reconcile(answer) == answer
    assert computed in answer


@pytest.mark.parametrize("answer,computed,label", [
    pytest.param(
        _table(("Bore", "Qty (m)"), (("400 mm", "633"), ("500 mm", "\u2014"), ("600 mm", "120")),
               "Total", "800"),
        "753", "500 mm", id="one-row-without-figure",
    ),
    pytest.param(
        _table(("Item", "Qty (No)"), (("A", "7"), ("B", "n/a"), ("C", "5")), "Total", "20"),
        "12", "B", id="one-row-marked-na",
    ),
])
def test_a_row_printed_without_a_figure_is_named_and_not_summed(answer, computed, label):
    out = reconcile(answer)
    assert f"Total\t\t{computed}" in out
    assert "Printed without a readable figure, so not in the total:" in out
    assert label in out.split("**Row total checked")[1]


@pytest.mark.parametrize("answer,computed", [
    pytest.param(
        _table(("Item", "Amount (SAR)"), (("A", "772,260.00"), ("B", "732,000.00")),
               "Total", "1,400,000.00"),
        "1,504,260.00", id="money-two-decimals",
    ),
    pytest.param(
        _table(("Item", "Length (m)"), (("A", "633.5"), ("B", "488.25")), "Total", "1,000.00"),
        "1,121.75", id="length-two-decimals",
    ),
])
def test_a_column_printed_to_two_decimals_keeps_its_format(answer, computed):
    assert f"Total\t\t{computed}" in reconcile(answer)


@pytest.mark.parametrize("answer", [
    pytest.param(
        _table(("Bore", "Qty (m)"), (("400 mm", "633"), ("500 mm", "488")),
               "Total", "10,573 [source: the bill's carried-to-collection line]"),
        id="cell-credits-source",
    ),
    pytest.param(
        _table(("Item", "Qty (No)"), (("A", "7"), ("B", "5")),
               "Total", "as printed on the bill's collection line"),
        id="cell-says-as-printed",
    ),
])
def test_a_total_the_answer_credits_to_a_document_is_left_alone(answer):
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,computed", [
    pytest.param("| Item | Qty (No) |\n| A | 7 |\n| B | 5 |\n| Total | 20 |", "12", id="two-rows"),
    pytest.param(
        "| Item | Qty (m) |\n| A | 633 |\n| B | 488 |\n| C | 120 |\n| Total | 1,000 |",
        "1,241", id="three-rows",
    ),
])
def test_a_pipe_delimited_table_is_reconciled_too(answer, computed):
    assert f"| Total | {computed} |" in reconcile(answer)


@pytest.mark.parametrize("answer,computed", [
    pytest.param(
        _table(("Item", "Qty (No)"), (("A", "1 + 1 + 1 + 1"), ("B", "6")), "Total", "30"),
        "10", id="four-ones",
    ),
    pytest.param(
        _table(("Item", "Qty (No)"), (("A", "2 + 3"), ("B", "7")), "Total", "20"),
        "12", id="two-plus-three",
    ),
])
def test_a_row_printed_as_a_sum_of_figures_counts_as_its_sum(answer, computed):
    assert f"Total\t\t{computed}" in reconcile(answer)


@pytest.mark.parametrize("answer", [
    pytest.param("4,858 m of pipe is billed across nine bores, summed from the bill's rows.",
                 id="prose-claim-no-table"),
    pytest.param("", id="empty"),
    pytest.param("12 No of manholes.", id="short-prose"),
    pytest.param("| only one pipe |", id="single-pipe-not-a-table"),
])
def test_an_answer_with_no_table_is_returned_byte_for_byte(answer):
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,first,second", [
    pytest.param(
        _table(("Bore", "Qty (m)"), (("400 mm", "633"), ("500 mm", "488")), "Total", "1,121")
        + "\n\n" + _table(("Item", "Qty (No)"), (("A", "18"), ("B", "26")), "Total", "50"),
        "Total\t\t1,121", "Total\t\t44", id="second-table-wrong",
    ),
    pytest.param(
        _table(("Item", "Qty (No)"), (("A", "7"), ("B", "5")), "Total", "20")
        + "\n\n" + _table(("Bore", "Length (m)"), (("400 mm", "633"), ("500 mm", "488")),
                          "Total", "1,121"),
        "Total\t\t12", "Total\t\t1,121", id="first-table-wrong",
    ),
])
def test_two_tables_are_each_checked_against_their_own_rows(answer, first, second):
    out = reconcile(answer)
    assert first in out
    assert second in out
    assert out.count("**Row total checked against the rows printed above:**") == 1
    assert out.count("- The rows printed above sum to") == 1


@pytest.mark.parametrize("answer", [
    pytest.param(
        "Item\tQty (No)\nremaining rows\t9\nothers\t3\nTotal\t\t12\n", id="two-buckets",
    ),
    pytest.param(
        "Item\tQty (m)\nthe rest of the band\t40\nTotal\t\t40\n", id="one-bucket",
    ),
])
def test_a_table_whose_every_row_is_a_remainder_states_no_total_of_its_own(answer):
    # No row states a figure of its own, so there is nothing to sum: the check
    # states no total rather than reading the buckets as rows.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,stated,computed", [
    pytest.param(
        "The sum of the priced rows is 400 No.\n"
        "Item\tRef\tQty (No)\nA\tK 9\t120\nB\tK 10\t180\nremaining rows (K 10)\t\t60\n",
        "400", "300", id="no-total-row-counts",
    ),
    pytest.param(
        "The rows sum to 1,400 m in this band.\n"
        "Item\tLength (m)\nA\t633\nB\t488\n",
        "1,400", "1,121", id="no-total-row-metres",
    ),
])
def test_a_table_with_no_total_row_is_checked_against_the_sum_the_answer_claims(
    answer, stated, computed,
):
    out = reconcile(answer)
    assert stated not in out
    assert computed in out


@pytest.mark.parametrize("answer", [
    pytest.param("The rows sum to 300 No.\nItem\tQty (No)\nA\t120\nB\t180\n", id="two-rows"),
    pytest.param("The rows sum to 44 No.\nItem\tQty (No)\nA\t18\nB\t26\n", id="other-count"),
])
def test_a_claim_that_already_equals_its_rows_is_returned_byte_for_byte(answer):
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,computed", [
    pytest.param(_table(("Item", "Qty (m)"), (("A", "633"), ("B", "488")), "Total", "1,400"),
                 "1,121", id="two-row-length"),
    pytest.param(_table(("Item", "Qty (No)"), (("A", "18"), ("B", "26")), "Total", "50"),
                 "44", id="two-row-count"),
])
def test_the_answer_exit_checks_the_rows_it_prints(answer, computed):
    """The check runs on the answer the user reads, not only as a module."""
    from app.agents.runtime import _postprocess_answer

    out = _postprocess_answer(
        answer, None, [{"role": "user", "content": "What is the bill's total?"}],
    )
    assert f"Total\t\t{computed}" in out
    assert "Row total checked against the rows printed above" in out


@pytest.mark.parametrize("answer,computed", [
    pytest.param("The rows read: 120 + 180 + 60 = 400 No, all depths together.",
                 "360", id="three-terms"),
    pytest.param("Lengths: 633 + 488 + 828 + 957 = 3,000 m.",
                 "2,906", id="four-terms"),
])
def test_a_printed_sum_chain_that_disagrees_with_its_own_result_is_corrected(
    answer, computed,
):
    out = reconcile(answer)
    assert f"= {computed}" in out
    assert "corrected to the sum of the terms printed" in out


@pytest.mark.parametrize("answer", [
    pytest.param("The rows read: 120 + 180 + 60 = 360 No, all depths together.", id="count"),
    pytest.param("Lengths: 633 + 488 + 828 = 1,949 m.", id="length"),
])
def test_a_printed_sum_chain_that_already_agrees_is_returned_byte_for_byte(answer):
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,terms", [
    pytest.param("The rows read: 18 + 26 + 64 + \u2026 = 168 No across every depth band.",
                 "3", id="three-then-ellipsis"),
    pytest.param("Rows: 7 + 5 + 2 + ... = 40 No in the band.", "3", id="ascii-ellipsis"),
])
def test_a_chain_the_answer_marks_incomplete_is_reported_not_guessed(answer, terms):
    out = reconcile(answer)
    # No figure is guessed: the answer is told its own list is incomplete.
    assert "prints an ellipsis over" in out
    assert f"over {terms} terms" in out


@pytest.mark.parametrize("answer", [
    pytest.param("SAR 5,000 + 3,000 = 8,000 for the two items.", id="two-terms-money"),
    pytest.param("The collection line is 10,000 + 573 = 10,573 m [source: the bill].",
                 id="two-terms-length"),
])
def test_a_sum_of_two_terms_is_prose_not_a_row_chain(answer):
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer", [
    pytest.param(
        "1,000 m \u2014 the sum of the 17 billed trench rows in this band.\n"
        "Item\tQty (m)\nA\t633\nB\t267\n\n"
        "The remaining rows continue on the next page and make up the balance to 1,000 m.\n",
        id="remaining-named",
    ),
    pytest.param(
        "40 No \u2014 the manholes in this section.\n"
        "Item\tQty (No)\nA\t18\nB\t10\n\n"
        "The others are listed on the following pages and make up the balance to 40 No.\n",
        id="others-named",
    ),
])
def test_a_claim_the_answer_itself_discloses_as_partial_is_left_alone(answer):
    # The answer has named the rows it did not print, so its figure is the
    # band's total and is not rewritten.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,prose,computed", [
    pytest.param(
        _table(("Item", "Qty (m)"), (("A", "633"), ("B", "488")), "Total", "1,400")
        + "\n\nThe rows were read once; the survey ran over two days.\n",
        "The rows were read once; the survey ran over two days.",
        "1,121", id="prose-with-no-figure",
    ),
    pytest.param(
        _table(("Item", "Qty (No)"), (("A", "18"), ("B", "26")), "Total", "44")
        + "\n\nThe bill prints 44 against this band on page 300.\n",
        "The bill prints 44 against this band on page 300.",
        "44", id="prose-with-a-bare-figure",
    ),
])
def test_a_wrong_total_is_corrected_by_its_cell_and_the_prose_is_untouched(
    answer, prose, computed,
):
    # Only a figure stated with the column's own unit is rewritten; a bare
    # figure or a sentence with no figure is left as the answer wrote it.
    out = reconcile(answer)
    assert prose in out
    assert f"Total\t\t{computed}" in out


@pytest.mark.parametrize("answer", [
    pytest.param(
        "The rows sum to 9,000 in this band.\n"
        "Item\tQty\nA\t633\nB\t488\n",
        id="header-with-no-unit",
    ),
    pytest.param(
        "The rows sum to 400 here.\n"
        "Item\tQuantity\nA\t120\nB\t180\n",
        id="header-with-no-unit-counts",
    ),
])
def test_a_prose_claim_against_a_header_with_no_unit_is_left_alone(answer):
    # The unit a figure carries is read from the answer's own header; without
    # one there is no way to tell the claimed figure from any other number, so
    # the claim is not corrected.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer", [
    pytest.param(
        "The rows sum to 9,000 m in this band.\n"
        "Item\tQty (m)\tAmount (SAR)\nA\t633\t772,260\nB\t488\t732,000\n",
        id="two-figure-columns-metres",
    ),
    pytest.param(
        "The rows sum to 5,000 No in this section.\n"
        "Item\tQty (No)\tRate (SAR)\nA\t18\t17,000\nB\t26\t19,200\n",
        id="two-figure-columns-counts",
    ),
])
def test_a_prose_claim_against_a_two_figure_table_states_nothing_of_its_own(answer):
    # Two figure columns: which one the claim is about cannot be read from the
    # answer, so the claim is not corrected against either.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,stated,computed", [
    pytest.param(
        "40 No - the sum of the priced bands in this section.\n\n"
        "The rows read: 1.5-2.0 m 7; 2.0-2.5 m 9; 2.5-3.0 m 12; "
        "plus the two further bands on the next sheet [source: the bill].\n",
        "40 No", "28 No", id="a-bucket-names-rows-and-no-figure",
    ),
    pytest.param(
        "The rows sum to 1,400 m across the band.\n\n"
        "Rows: 400 mm 633; 500 mm 488; 600 mm 120; 700 mm 60 "
        "- plus the remaining bores in the same section.\n",
        "1,400 m", "1,301 m", id="a-dash-bucket-names-rows-and-no-figure",
    ),
])
def test_a_prose_row_list_total_is_the_sum_of_the_rows_it_prints(
    answer, stated, computed,
):
    # Some answers print their rows as a sentence rather than a table. The same
    # rule reads them: the total is the sum of the figures the rows state, and
    # a bucket that only names further rows accounts for no part of it.
    out = reconcile(answer)
    assert stated not in out
    assert computed in out
    assert "Row total checked against the rows printed above" in out


@pytest.mark.parametrize("answer", [
    pytest.param(
        "The rows sum to 28 No.\n\n"
        "The rows read: 1.5-2.0 m 7; 2.0-2.5 m 9; 2.5-3.0 m 12.\n",
        id="counts",
    ),
    pytest.param(
        "The rows sum to 1,241 m.\n\n"
        "Rows: 400 mm 633; 500 mm 488; 600 mm 120.\n",
        id="lengths",
    ),
])
def test_a_prose_row_list_that_agrees_with_its_total_is_returned_byte_for_byte(answer):
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer", [
    pytest.param(
        "The rows sum to 36 No.\n\n"
        "The rows read: 1.5-2.0 m 7; 2.0-2.5 m 9; 2.5-3.0 m 12 "
        "- plus the remaining rows, which make up the balance to 36 No.\n",
        id="balance-in-the-listing-line",
    ),
    pytest.param(
        "The rows sum to 1,301 m.\n\n"
        "Rows: 400 mm 633; 500 mm 488; 600 mm 120.\n\n"
        "The remaining bores continue on the next sheet and make up the balance to 1,301 m.\n",
        id="balance-in-its-own-sentence",
    ),
])
def test_a_prose_row_list_the_answer_ties_to_the_balance_is_left_alone(answer):
    # The bucket says it is the rest of the stated total, so the total is
    # printed rows plus that bucket by the answer's own definition.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer", [
    pytest.param(
        "The rows sum to 36 No.\n\n"
        "The rows read: 1.5-2.0 m 7; 2.0-2.5 m 9; 2.5-3.0 m 12; "
        "plus the remaining rows 8 No.\n",
        id="bucket-figure-in-the-listing-line",
    ),
    pytest.param(
        "The rows sum to 1,301 m.\n\n"
        "Rows: 400 mm 633; 500 mm 488; 600 mm 120.\n\n"
        "The remaining bores add 60 m.\n",
        id="bucket-figure-in-its-own-sentence",
    ),
])
def test_a_prose_row_list_bucket_that_states_its_figure_reconciles_the_total(answer):
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,computed,label", [
    pytest.param(
        "The rows sum to 40 No.\n\n"
        "The rows read: 1.5-2.0 m 7; 2.0-2.5 m 9; 2.5-3.0 m not read.\n",
        "16 No", "2.5-3.0 m not read", id="words-instead-of-a-quantity",
    ),
    pytest.param(
        "The rows sum to 1,400 m.\n\n"
        "Rows: 400 mm 633; 500 mm 488; 600 mm (not measured).\n",
        "1,121 m", "600 mm", id="parenthetical-instead-of-a-quantity",
    ),
])
def test_a_prose_row_without_a_quantity_is_named_and_not_summed(
    answer, computed, label,
):
    out = reconcile(answer)
    assert computed in out
    assert "Printed without a readable figure, so not in the total:" in out
    assert label in out.split("**Row total checked")[1]


@pytest.mark.parametrize("answer", [
    pytest.param(
        "The rows sum to 40 No.\n\n"
        "The rows read: 1.5-2.0 m 7; 2.0-2.5 m 9; 2.5-3.0 m not read; "
        "plus the remaining rows on the next sheet.\n",
        id="counts-with-a-bucket",
    ),
    pytest.param(
        "The rows sum to 1,400 m.\n\n"
        "Rows: 400 mm 633; 500 mm 488; 600 mm (not measured) "
        "- plus the remaining bores in this band.\n",
        id="lengths-with-a-bucket",
    ),
])
def test_a_prose_row_without_a_quantity_plus_a_bucket_leaves_the_total_alone(answer):
    # One row's figure is unknown and the answer says further rows exist, so
    # the printed evidence cannot settle the total: nothing is rewritten.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer", [
    pytest.param(
        "The rows sum to 300 No.\n\nThe rows read: K 137; K 138; K 139.\n",
        id="reference-shaped-segments",
    ),
    pytest.param(
        "The rows sum to 40 No.\n\nThe rows read: 1.5-2.0 m; 2.0-2.5 m; 2.5-3.0 m.\n",
        id="dimension-shaped-segments",
    ),
])
def test_a_prose_segment_that_states_no_quantity_is_not_read_as_a_row(answer):
    # A reference ("K 137") and a depth range ("1.5-2.0 m") end in figures that
    # are not quantities, so the line is not a row list and states no sum.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer,stated,computed", [
    pytest.param(
        _table(("Item", "Qty (No)"),
               (("A", "18"), ("B", "26"), ("remaining rows (not itemised)", "\u2014")),
               "Total", "50"),
        "50", "44", id="tab-table",
    ),
    pytest.param(
        "| Item | Qty (No) |\n| A | 7 |\n| B | 5 |\n| remaining rows (not itemised) | \u2014 |\n"
        "| Total | 20 |",
        "20", "12", id="pipe-table",
    ),
])
def test_a_bucket_that_states_no_figure_does_not_excuse_a_larger_total(
    answer, stated, computed,
):
    # The bucket names further rows but states none of their figures and makes
    # no claim to the balance, so nothing the answer prints accounts for the
    # difference: the total is the sum of the rows that do.
    out = reconcile(answer)
    assert computed in out
    assert "Row total checked against the rows printed above" in out


@pytest.mark.parametrize("answer", [
    pytest.param(
        _table(("Item", "Qty (No)"),
               (("A", "18"), ("B", "26"), ("\u2014", "remaining rows 6 No", "\u2014")),
               "Total", "50"),
        id="figure-written-into-the-wording",
    ),
    pytest.param(
        "| Item | Qty (m) |\n| A | 633 |\n| B | 488 |\n"
        "| the rest of the band, 120 m | \u2014 |\n| Total | 1,241 |",
        id="figure-written-into-the-wording-pipe",
    ),
])
def test_a_bucket_that_writes_its_figure_into_its_wording_reconciles_the_total(answer):
    # The bucket states the figure for the rows it stands for, and the total is
    # the printed rows plus that figure, so the total is the answer's own sum.
    assert reconcile(answer) == answer


@pytest.mark.parametrize("answer", [
    pytest.param(
        "The rows sum to 40 in this section.\n\n"
        "The rows read: 1.5-2.0 m 7; 2.0-2.5 m 9; 2.5-3.0 m 12.\n",
        id="preposition-after-the-figure",
    ),
    pytest.param(
        "The rows sum to 1,400 of the priced rows.\n\n"
        "Rows: 400 mm 633; 500 mm 488; 600 mm 120.\n",
        id="partitive-after-the-figure",
    ),
])
def test_a_claim_figure_followed_by_an_ordinary_word_states_no_unit(answer):
    # "40 in this section", "1,400 of the priced rows": the word after the
    # figure is the sentence's own, not a unit the answer gives that figure. A
    # figure with no unit cannot be matched to a column, so nothing is
    # rewritten - the same silence the check keeps for a table with no unit.
    assert reconcile(answer) == answer
