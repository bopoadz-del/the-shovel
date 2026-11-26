"""A page that opens on its date is filed under its heading, not the date.

Every input is invented here; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import _is_date_line, _is_heading_line, page_heading_lines


@pytest.mark.parametrize("line", ["14 October 2031", "2 NOV 2029", "9th May", "AUGUST 2030"])
def test_a_date_line_is_not_a_heading(line):
    assert _is_date_line(line)
    assert not _is_heading_line(line)


@pytest.mark.parametrize(
    "line",
    ["4 Programme for October 2031", "7 MAY ROAD DRAINAGE", "12 Formwork", "3.1 PILING"],
)
def test_a_heading_that_names_a_month_or_starts_on_a_number_stays_a_heading(line):
    assert not _is_date_line(line)
    assert _is_heading_line(line)


@pytest.mark.parametrize(
    "date, heading",
    [("14 October 2031", "RE: CLAIM FOR EXTENSION OF TIME"), ("2 NOV 2029", "5 Site Access")],
)
def test_a_page_opening_on_its_date_is_filed_under_the_heading_below(date, heading):
    text = f"{date}\n{heading}\nWe refer to the meeting held on site.\n"
    assert page_heading_lines(text, [(1, 0)]) == {1: heading}


@pytest.mark.parametrize("date", ["14 October 2031", "AUGUST 2030"])
def test_a_page_with_only_a_date_has_no_heading(date):
    text = f"{date}\nwe write further to our letter of last week.\n"
    assert page_heading_lines(text, [(1, 0)]) == {}
