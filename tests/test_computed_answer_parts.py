"""#392: each part of the computed-answer text (app.agents.core.computed_answer)
decides one thing; each decision is checked here with two inputs the test
builds itself (invented tools, figures, files and words).
"""
from __future__ import annotations

import pytest

from app.agents.core.computed_answer import (
    LABELS,
    REASONS,
    _asked_line,
    _figure_text,
    _statement_lines,
    _what,
    _word,
    failure_text,
    has_words,
    heading_for,
    join_calls,
    labels_for,
    render_call,
    reason_text,
    MAX_CODE_CHARS,
    tool_call_args,
)
from app.agents.core.tool_registry import Statement

EN, AR = LABELS["en"], LABELS["ar"]


@pytest.mark.parametrize("text, expected", [
    ("Bank volume 42 m3", True),
    ("٤٢ متر", True),
    ("**\n---\n- *", False),
    ("", False),
])
def test_has_words_is_false_for_markdown_alone(text, expected):
    assert has_words(text) is expected


@pytest.mark.parametrize("language, expected", [
    ("ar", AR),
    ("en", EN),
    ("zz-unknown", EN),
])
def test_labels_for_falls_back_to_the_default_language(language, expected):
    assert labels_for(language) == expected


@pytest.mark.parametrize("value, expected", [
    (1234.5, "1,234.5"),
    (2.0, "2"),
    (0.1255, "0.126"),
])
def test_a_figure_is_shown_without_trailing_zeros(value, expected):
    assert _figure_text(value) == expected


@pytest.mark.parametrize("lab, key, expected", [
    (EN, "no_rows", EN["no_rows"]),
    (AR, "no_rows", AR["no_rows"]),
    (EN, "m3 of invented fill", "m3 of invented fill"),
])
def test_a_known_code_reads_through_the_labels_and_data_as_is(lab, key, expected):
    assert _word(lab, key) == expected


@pytest.mark.parametrize("st, lab, expected", [
    (Statement(figure=3.0, what="bank volume"), EN, "Bank volume"),
    (Statement(figure=3.0, what="loose volume", basis=""), AR, "Loose volume"),
    (Statement(figure=2.0, what="", basis="count", terms=("pile cap",)), EN,
     "Number of rows whose description holds pile cap"),
    (Statement(figure=9.0, what="quantity", basis="sum", within=("B2",)), EN,
     "Quantity summed over the rows in the section B2"),
])
def test_what_a_statement_is_reads_its_basis(st, lab, expected):
    assert _what(st, lab) == expected


@pytest.mark.parametrize("st, lab, needle", [
    (Statement(figure=None, what="quantity", basis="sum", failure="no_derived_bill_sheet"),
     EN, f"{EN['not_returned']} ({REASONS['en']['no_derived_bill_sheet']})"),
    (Statement(figure=None, what="quantity", basis="sum"), AR,
     f"{AR['not_returned']} ({AR['failed']})"),
    (Statement(figure=0.0, what="", basis="count", count=0), EN, f"**0** ({EN['no_rows']})"),
    (Statement(figure=12.5, what="bank volume", unit="m3", source="BS 6031"), EN,
     f"- Bank volume: **12.5 m3** ({EN['source']}: BS 6031)"),
])
def test_a_statement_line_says_returned_none_or_its_figure(st, lab, needle):
    assert needle in _statement_lines(st, lab)[0]


@pytest.mark.parametrize("asked, lab, question, expected", [
    ({"terms": "invented kerb", "unit": "m", "flag": None}, EN,
     "Count the Invented Kerb runs in M", "Asked: invented kerb, m"),
    ({"calculation": "x_volume", "depth": 2}, AR, "x_volume بعمق 2", f"{AR['asked']}: x_volume{AR['sep']}2"),
    ({"skip": [1, 2], "blank": "  "}, EN, "1 2", ""),
])
def test_the_asked_line_lists_the_values_the_user_typed(asked, lab, question, expected):
    assert _asked_line(asked, lab, question) == expected


@pytest.mark.parametrize("asked, question, absent, expected", [
    ({"file": r"C:\Program Files\Invented Works\bill sheet.xlsx", "terms": "kerb"},
     "count the kerb runs", ["Program Files", "Invented Works", "bill sheet", "C:"], "Asked: kerb"),
    ({"vendor": "Zephyrine Invented Holdings Ltd", "qty": 12},
     "total of 12 pallets", ["Zephyrine", "Holdings"], "Asked: 12"),
])
def test_an_argument_the_user_did_not_type_is_never_echoed(asked, question, absent, expected):
    line = _asked_line(asked, EN, question)
    assert line == expected
    for piece in absent:
        assert piece not in line


@pytest.mark.parametrize("asked, question", [
    ({"table": "invented_rows_internal", "path": "/srv/invented data/x.db"}, "how many?"),
    ({"company": "Invented Contracting LLC"}, "count them"),
])
def test_no_value_from_the_question_drops_the_asked_line(asked, question):
    assert _asked_line(asked, EN, question) == ""
    out = render_call("Invented tool", [], asked=asked, error="", language="en", question=question)
    assert EN["asked"] not in out
    for value in asked.values():
        assert value not in out


@pytest.mark.parametrize("length, shown", [(64, True), (65, False)])
def test_a_declared_code_is_shown_only_up_to_64_characters(length, shown):
    code = ("invented_reason_" + "x" * 80)[:length]
    out = reason_text(code, EN, "en", (code,))
    assert (code in out) is shown
    assert out.startswith(EN["tool_error"])
    assert MAX_CODE_CHARS == 64


@pytest.mark.parametrize("error, language, question, last", [
    ("missing depth", "en", "a trench 7 long", f"- {EN['not_returned']}: {EN['tool_error']}"),
    ("", "ar", "خندق بطول 7", f"- {AR['not_returned']}: {AR['no_figure']}"),
])
def test_a_call_with_no_statement_says_what_was_asked_and_why(error, language, question, last):
    out = render_call("Invented calculator", [], asked={"length": 7}, error=error, language=language,
                      question=question)
    lines = out.splitlines()
    assert lines[0] == "Invented calculator:"
    assert lines[-1] == last
    assert "7" in lines[1]
    assert "missing depth" not in out


@pytest.mark.parametrize("declared, language, last", [
    (("invented_gauge_offline",), "en",
     f"- {EN['not_returned']}: {EN['tool_error']} ({EN['code']}: invented_gauge_offline)"),
    (("invented_gauge_offline",), "ar",
     f"- {AR['not_returned']}: {AR['tool_error']} ({AR['code']}: invented_gauge_offline)"),
    ((), "en", f"- {EN['not_returned']}: {EN['tool_error']}"),
    (("another_declared_code",), "ar", f"- {AR['not_returned']}: {AR['tool_error']}"),
])
def test_a_code_is_shown_only_when_the_tool_declares_it(declared, language, last):
    out = render_call("Invented gauge", [], asked={}, error="invented_gauge_offline",
                      language=language, declared=declared)
    assert out.splitlines()[-1] == last


@pytest.mark.parametrize("declared, shown", [
    (("invented_meter_jammed",), f"({EN['tool_error']} ({EN['code']}: invented_meter_jammed))"),
    ((), f"({EN['tool_error']})"),
])
def test_a_failed_part_shows_a_code_only_when_declared(declared, shown):
    st = Statement(figure=None, what="", basis="count", failure="invented_meter_jammed")
    assert _statement_lines(st, EN, "en", declared)[0].endswith(shown)


@pytest.mark.parametrize("language", ["en", "ar"])
def test_every_reason_code_is_worded_in_every_language(language):
    assert set(REASONS[language]) == set(REASONS["en"])
    assert all(v.strip() for v in REASONS[language].values())


def test_every_worded_reason_code_is_declared_by_a_tool():
    from app.agents.core import tool_registry

    tool_registry.load()
    declared = set()
    for spec in list(tool_registry._TOOLS.values()) + list(tool_registry._RESULTS.values()):
        declared.update(spec.reason_codes)
    assert set(REASONS["en"]) <= declared, set(REASONS["en"]) - declared


@pytest.mark.parametrize("figure, shown", [(4.0, "**4 m3**"), (17.25, "**17.25 m3**")])
def test_a_call_with_a_statement_shows_its_figure_and_no_failure(figure, shown):
    out = render_call("Invented calculator", [Statement(figure=figure, what="volume", unit="m3")],
                      asked={"length": 7}, error="ignored", language="en")
    assert out == f"Invented calculator:\n- Volume: {shown}"


@pytest.mark.parametrize("name, language, expected", [
    ("Invented trench tool", "en", "Invented trench tool"),
    ("Invented trench tool", "ar", AR["heading"]),
    ("  ", "en", EN["heading"]),
])
def test_the_heading_is_the_tool_name_only_in_the_default_language(name, language, expected):
    assert heading_for(name, language) == expected


@pytest.mark.parametrize("parts, language, last", [
    (["A:\n- 1", "B:\n- 2"], "en", EN["several"].format(n=2)),
    (["A:\n- 1", "", "B:\n- 2", "C:\n- 3"], "ar", AR["several"].format(n=3)),
])
def test_several_calls_close_with_how_many_are_shown(parts, language, last):
    # The count is not the answer, so it follows the calls; the first call opens.
    out = join_calls(parts, language)
    assert out.split("\n\n")[-1] == last
    assert out.split("\n\n")[0] == parts[0]
    assert all(p in out for p in parts if p)


@pytest.mark.parametrize("parts", [["Only:\n- 1"], ["", "Only:\n- 2"]])
def test_one_call_is_shown_alone(parts):
    assert join_calls(parts, "en") == [p for p in parts if p][0]


def _assistant(call_id: str, arguments) -> dict:
    return {"role": "assistant", "tool_calls": [
        {"id": call_id, "function": {"name": "invented", "arguments": arguments}}]}


@pytest.mark.parametrize("messages, call_id, expected", [
    ([_assistant("c1", '{"depth": 2}')], "c1", {"depth": 2}),
    ([_assistant("c1", '{"depth": 2}'), _assistant("c2", {"length": 9})], "c2", {"length": 9}),
    ([_assistant("c1", "not json")], "c1", {}),
    ([_assistant("c1", '["a list"]')], "c1", {}),
    ([_assistant("", '{"depth": 2}')], "", {}),
    ([_assistant("c1", '{"depth": 2}')], "c9", {}),
])
def test_tool_call_args_are_the_ones_sent_with_that_call(messages, call_id, expected):
    assert tool_call_args(messages, call_id) == expected


@pytest.mark.parametrize("payload, expected", [
    ({"error": "unknown calculation {zz}"}, "unknown calculation zz"),
    ({"error": "  ", "reason": "[sheet] missing"}, "sheet missing"),
    ({"status": "error"}, ""),
    (None, ""),
])
def test_failure_text_is_the_plain_error_or_reason(payload, expected):
    assert failure_text(payload) == expected



# --- a declared reason code must be a snake_case code (tool and result) ---

def _stater(_result):
    return []


def _declare_tool(name, codes):
    from app.agents.core import tool_registry
    return tool_registry.tool(name, owner="base", display_name="Probe tool",
                              kind="computed", statement=_stater,
                              reason_codes=codes)


def _declare_result(name, codes):
    from app.agents.core import tool_registry
    tool_registry.declare_result(name, owner="base", display_name="Probe result",
                                 kind="computed", statement=_stater,
                                 reason_codes=codes)


@pytest.mark.parametrize("declare", [_declare_tool, _declare_result])
@pytest.mark.parametrize("bad_code", [
    "Table Name",          # spaces and capitals: free text, not a code
    "orders",              # one bare word: a table name, not a code
    "/srv/data/x.db",      # a path
    42,                    # not text at all
])
def test_a_reason_code_that_is_not_a_snake_case_code_is_refused(declare, bad_code):
    from app.agents.core import tool_registry
    name = f"probe_bad_code_{declare.__name__}_{abs(hash(repr(bad_code)))}"
    with pytest.raises(ValueError, match="reason code"):
        declare(name, ("rows_missing", bad_code))
    assert name not in tool_registry._TOOLS
    assert name not in tool_registry._RESULTS


@pytest.mark.parametrize("codes", [("rows_missing",), ("term_not_found", "bill_not_found")])
def test_declared_snake_case_codes_are_kept_in_order(codes):
    from app.agents.core import tool_registry
    name = f"probe_good_codes_{len(codes)}"
    try:
        _declare_result(name, codes)
        assert tool_registry._RESULTS[name].reason_codes == codes
    finally:
        tool_registry._RESULTS.pop(name, None)


@pytest.mark.parametrize("value, question", [
    (12000.0, "excavate 12,000 m3 of soil"),
    (4500, "a slab of 4,500 m2"),
])
def test_a_figure_typed_with_thousands_separators_is_the_users(value, question):
    from app.agents.core.computed_answer import _typed_by_user

    assert str(value) not in question  # only the figure form appears
    assert _typed_by_user(value, question) is True
