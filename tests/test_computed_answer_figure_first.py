"""The first line of a computed answer carries its figure (answer style,
first-line class).

A heading says which tool spoke, not what it found; a count of results says
how many there are, not what they are. Neither stands alone as the first line:
the first result's figure shares the heading's line, and the count of results
follows them. Every input here is written by the test (invented tools,
invented words); two inputs per decision.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.core.computed_answer import LABELS, Call, answer_turn, join_calls

EN, AR = LABELS["en"], LABELS["ar"]


def _st(figure, *, what="quantity", unit="m", count=None, rows=(), working="", source=""):
    return SimpleNamespace(figure=figure, what=what, unit=unit, count=count, rows=tuple(rows),
                           working=working, source=source, basis="", terms=(), within=(),
                           failure="")


def _call(tool, heading, *statements, asked=None):
    return Call(tool=tool, heading=heading, statements=tuple(statements), asked=asked or {})


@pytest.mark.parametrize("heading,figure,shown", [
    ("Invented trench tool", 41.5, "41.5"),
    ("Invented slab tool", 1250.0, "1,250"),
])
def test_one_result_opens_on_its_figure(heading, figure, shown):
    out = answer_turn([_call("zz_invented", heading, _st(figure))], "en")
    first = out.splitlines()[0]
    assert first.startswith(f"{heading}: ") and f"**{shown} m**" in first, out
    assert f"{heading}:\n" not in out, out


@pytest.mark.parametrize("rows,working", [
    (("row alpha", "row beta"), ""),
    ((), "2 x 3 = 6"),
])
def test_lines_after_the_figure_follow_it_unchanged(rows, working):
    out = answer_turn([_call("zz_invented", "Invented tool", _st(6.0, rows=rows, working=working))], "en")
    lines = out.splitlines()
    assert "**6 m**" in lines[0], out
    if rows:
        assert lines[1] == f"  {EN['rows']}: " + EN["list_sep"].join(rows), out
    if working:
        assert lines[1] == f"  {EN['working']}: {working}", out


@pytest.mark.parametrize("language,lab", [("en", EN), ("ar", AR)])
def test_the_count_of_results_follows_them(language, lab):
    calls = [_call("zz_one", "Invented one", _st(3.0)), _call("zz_two", "Invented two", _st(7.0))]
    out = answer_turn(calls, language)
    assert "**3 m**" in out.splitlines()[0], out
    assert out.rstrip().endswith(lab["several"].format(n=2)), out
    assert out.index("**3 m**") < out.index("**7 m**") < out.index(lab["several"].format(n=2))


@pytest.mark.parametrize("one,two", [("kerbwidget", "ductwidget"), ("gullything", "manholething")])
def test_what_was_asked_shares_the_heading_line_before_the_figure(one, two):
    question = f"totals for {one} and {two}?"
    calls = [_call("zz_same", "Invented tool", _st(2.0), asked={"terms": one}),
             _call("zz_same", "Invented tool", _st(5.0), asked={"terms": two})]
    out = answer_turn(calls, "en", question)
    first = out.splitlines()[0]
    assert first.startswith(f"Invented tool ({EN['asked']}: {one}): "), out
    assert first.index(one) < first.index("**2 m**"), out
    assert f"- {EN['asked']}:" not in out, out


@pytest.mark.parametrize("figure", [None, 0.0])
def test_a_block_with_no_figure_line_keeps_its_heading_line(figure):
    # No result at all: the calls are the answer (render_call), heading first.
    # A zero count: its statement is still the line beside the heading.
    st = _st(figure, count=0 if figure is not None else None)
    out = answer_turn([_call("zz_invented", "Invented tool", st)], "en")
    first = out.splitlines()[0]
    if figure is None:
        assert first == "Invented tool:", out
    else:
        assert first.startswith("Invented tool: ") and "**0**" in first, out


@pytest.mark.parametrize("parts", [["A:\n- 1", "B:\n- 2"], ["X:\n- 9", "", "Y:\n- 8", "Z:\n- 7"]])
def test_joined_calls_open_on_the_first_call(parts):
    out = join_calls(parts, "en")
    kept = [p for p in parts if p]
    assert out.startswith(kept[0]), out
    assert out.endswith(EN["several"].format(n=len(kept))), out


@pytest.mark.parametrize("parts", [["A:\n- 1"], ["", "B:\n- 2"]])
def test_a_single_call_has_no_count_line(parts):
    out = join_calls(parts, "en")
    assert EN["several"].split("{n}")[1] not in out, out
