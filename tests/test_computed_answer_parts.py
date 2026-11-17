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


@pytest.mark.parametrize("asked, lab, expected", [
    ({"terms": "invented kerb", "unit": "m", "flag": None}, EN, "Asked: invented kerb, m"),
    ({"calculation": "x_volume", "depth": 2}, AR, f"{AR['asked']}: x_volume{AR['sep']}2"),
    ({"skip": [1, 2], "blank": "  "}, EN, ""),
])
def test_the_asked_line_lists_the_values_sent(asked, lab, expected):
    assert _asked_line(asked, lab) == expected


@pytest.mark.parametrize("error, language, last", [
    ("missing depth", "en", f"- {EN['not_returned']}: {EN['tool_error']}"),
    ("", "ar", f"- {AR['not_returned']}: {AR['no_figure']}"),
])
def test_a_call_with_no_statement_says_what_was_asked_and_why(error, language, last):
    out = render_call("Invented calculator", [], asked={"length": 7}, error=error, language=language)
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


@pytest.mark.parametrize("parts, language, first", [
    (["A:\n- 1", "B:\n- 2"], "en", EN["several"].format(n=2)),
    (["A:\n- 1", "", "B:\n- 2", "C:\n- 3"], "ar", AR["several"].format(n=3)),
])
def test_several_calls_open_with_how_many_are_shown(parts, language, first):
    out = join_calls(parts, language)
    assert out.split("\n\n")[0] == first
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

