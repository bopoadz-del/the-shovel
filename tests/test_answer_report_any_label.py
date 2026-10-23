"""An answer-report export names its answers by any one-letter label.

The range is the user's own numbering of the turns ("Q1-Q5", "B2 to B4",
"answers 3-7"); no label letter is special.
"""
from app.core.answer_report_intent import message_wants_answer_report, parse_answer_report_range


def test_any_single_letter_label_gives_the_range():
    assert parse_answer_report_range("Export Q1-Q5 answers as a docx report") == (1, 5)
    assert parse_answer_report_range("save answers B2 to B4 as docx") == (2, 4)
    assert parse_answer_report_range("export answers 3-7 as a report") == (3, 7)


def test_any_label_is_an_answer_report_ask():
    assert message_wants_answer_report("Export Q1-Q5 answers as a docx report")


def test_a_lone_code_is_still_not_a_range():
    assert parse_answer_report_range("what does Q1 of the RFP say") is None


def test_a_labelled_range_without_answers_is_not_an_answer_report():
    assert not message_wants_answer_report("Export drawing sheets S1-S5 as a Word report")
