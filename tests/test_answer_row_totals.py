"""A table's stated total equals the sum of the rows the table prints.

Every test builds its own answer text; there is no fixture file, no document,
no item name and no figure taken from a real bill. The passages are written
to make one rule at a time the only thing the check can decide on, and each
rule is exercised on two inputs (a disagreement and an agreement).
"""
from __future__ import annotations

from app.agents.answer_row_totals import reconcile

PLAIN = (
    "The rows read from the bill:\n"
    "Item\tBore\tLength (m)\n"
    "A\t400 mm\t633\n"
    "B\t500 mm\t488\n"
    "Total\t\t1,121\n"
    "\n"
    "The bill prints no subtotal, so the figure is the sum of the rows above."
)


def test_a_total_that_equals_its_column_is_returned_byte_for_byte():
    assert reconcile(PLAIN) == PLAIN


def test_a_total_that_disagrees_is_the_sum_of_the_rows_printed():
    wrong = PLAIN.replace("1,121", "1,900").replace(
        "The bill prints no subtotal", "1,900 m is the billed length. The bill prints no subtotal",
    )
    out = reconcile(wrong)
    # (1) the Total cell is the column's own sum
    assert "Total\t\t1,121" in out
    # (1) and the answer's prose statement of the wrong figure with its unit
    assert "1,900 m" not in out
    assert "1,121 m is the billed length" in out
    # the total and the rows agree, and the check says what it covers
    assert "sum to 1,121 m over 2 rows" in out


def test_a_figure_without_the_column_unit_is_never_rewritten():
    # "1,900" here is a page number, not a length: it carries no unit.
    wrong = PLAIN.replace("1,121", "1,900").replace(
        "the figure is the sum", "page 1,900 states no subtotal; the figure is the sum",
    )
    out = reconcile(wrong)
    assert "page 1,900 states no subtotal" in out
    assert "Total\t\t1,121" in out


def test_a_remainder_bucket_is_not_a_row_and_never_enters_the_total():
    answer = (
        "Item\tDepth\tQty (No)\n"
        "A\t1.5-2.0 m\t18\n"
        "B\t2.0-2.5 m\t26\n"
        "C\t2.5-3.0 m\t64\n"
        "\u2014\tremaining depth bands\t9\n"
        "Total\t\t117\n"
    )
    out = reconcile(answer)
    assert "Total\t\t108" in out           # 18 + 26 + 64; the bucket's 9 is not a row
    assert "remaining depth bands stands for rows this answer does not itemise" in out
    assert "the total is therefore incomplete" in out


def test_a_row_printed_without_a_figure_is_named_and_not_summed():
    answer = (
        "Bore\tQty (m)\n"
        "400 mm\t633\n"
        "500 mm\t\u2014\n"
        "600 mm\t120\n"
        "Total\t\t800\n"
    )
    out = reconcile(answer)
    assert "Total\t\t753" in out
    assert "Printed without a readable figure, so not in the total:" in out
    assert "500 mm" in out.split("**Row total checked")[1]


def test_a_column_printed_to_two_decimals_keeps_its_format():
    answer = (
        "Item\tAmount (SAR)\n"
        "A\t772,260.00\n"
        "B\t732,000.00\n"
        "Total\t\t1,400,000.00\n"
    )
    out = reconcile(answer)
    assert "Total\t\t1,504,260.00" in out


def test_a_total_the_answer_credits_to_a_document_is_left_alone():
    answer = (
        "Bore\tQty (m)\n"
        "400 mm\t633\n"
        "500 mm\t488\n"
        "Total\t\t10,573 [source: the bill's carried-to-collection line]\n"
    )
    assert reconcile(answer) == answer


def test_a_pipe_delimited_table_is_reconciled_too():
    answer = (
        "| Item | Qty (No) |\n"
        "| A | 7 |\n"
        "| B | 5 |\n"
        "| Total | 20 |\n"
    )
    out = reconcile(answer)
    assert "| Total | 12 |" in out


def test_a_row_printed_as_a_sum_of_figures_counts_as_its_sum():
    answer = (
        "Item\tQty (No)\n"
        "A\t1 + 1 + 1 + 1\n"
        "B\t6\n"
        "Total\t\t30\n"
    )
    out = reconcile(answer)
    assert "Total\t\t10" in out


def test_an_answer_with_no_table_is_returned_byte_for_byte():
    for text in (
        "4,858 m of pipe is billed across nine bores, summed from the bill's rows.",
        "",
        "12 No of manholes.",
    ):
        assert reconcile(text) == text


def test_two_tables_are_each_checked_against_their_own_rows():
    answer = (
        "Bore\tQty (m)\n"
        "400 mm\t633\n"
        "500 mm\t488\n"
        "Total\t\t1,121\n"
        "\n"
        "Item\tQty (No)\n"
        "A\t18\n"
        "B\t26\n"
        "Total\t\t50\n"
    )
    out = reconcile(answer)
    assert "Total\t\t1,121" in out        # the first table already agreed
    assert "Total\t\t44" in out           # the second did not
    assert out.count("**Row total checked against the rows printed above:**") == 1
    assert out.count("- The rows printed above sum to") == 1


def test_a_table_whose_every_row_is_a_remainder_states_no_total_of_its_own():
    answer = (
        "Item\tQty (No)\n"
        "remaining rows\t9\n"
        "others\t3\n"
        "Total\t\t12\n"
    )
    # No row states a figure of its own, so there is nothing to sum: the check
    # states no total rather than reading the buckets as rows.
    assert reconcile(answer) == answer


def test_a_table_with_no_total_row_is_checked_against_the_sum_the_answer_claims():
    answer = (
        "The sum of the priced rows is 400 No [source: the bill].\n"
        "Item\tQty (No)\n"
        "A\t120\n"
        "B\t180\n"
        "remaining rows\t60\n"
    )
    out = reconcile(answer)
    assert out.startswith("The sum of the priced rows is 300 No")
    assert "remaining rows stands for rows this answer does not itemise" in out


def test_a_claim_that_already_equals_its_rows_is_returned_byte_for_byte():
    answer = (
        "The rows sum to 300 No.\n"
        "Item\tQty (No)\n"
        "A\t120\n"
        "B\t180\n"
    )
    assert reconcile(answer) == answer


def test_the_answer_exit_checks_the_rows_it_prints():
    """The check runs on the answer the user reads, not only as a module."""
    from app.agents.runtime import _postprocess_answer
    answer = (
        "Item\tQty (m)\n"
        "A\t633\n"
        "B\t488\n"
        "Total\t\t1,400\n"
    )
    out = _postprocess_answer(
        answer, None, [{"role": "user", "content": "How much pipe is billed?"}],
    )
    assert "Total\t\t1,121" in out
    assert "sum to 1,121 m over 2 rows" in out


def test_a_printed_sum_chain_that_disagrees_with_its_own_result_is_corrected():
    answer = (
        "The rows read: 120 + 180 + 60 = 400 No, all depths together."
    )
    out = reconcile(answer)
    assert "120 + 180 + 60 = 360 No" in out
    assert "sum to 360 No" in out


def test_a_printed_sum_chain_that_already_agrees_is_returned_byte_for_byte():
    answer = "The rows read: 120 + 180 + 60 = 360 No, all depths together."
    assert reconcile(answer) == answer


def test_a_chain_the_answer_marks_incomplete_is_reported_not_guessed():
    answer = "The rows read: 18 + 26 + 64 + \u2026 = 168 No across every depth band."
    out = reconcile(answer)
    # No figure is guessed: the answer is told its own list is incomplete.
    assert "= 168 No" in out
    assert "prints an ellipsis over 3 terms" in out


def test_a_sum_of_two_terms_is_prose_not_a_row_chain():
    for answer in (
        "SAR 5,000 + 3,000 = 8,000 for the two items.",
        "The band's carried-to-collection line is 10,000 + 573 = 10,573 m [source: the bill].",
    ):
        assert reconcile(answer) == answer


def test_a_claim_the_answer_itself_discloses_as_partial_is_left_alone():
    answer = (
        "900 m — the sum of the 17 billed trench rows in this band.\n"
        "Item\tQty (m)\n"
        "A\t633\n"
        "B\t267\n"
        "\n"
        "The remaining rows continue on the next page and make up the balance to 900 m.\n"
    )
    # The answer has named the rows it did not print, so its figure is the
    # band's total and is not rewritten.
    assert reconcile(answer) == answer
