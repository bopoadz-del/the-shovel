"""Pins the figure-before-label parse (G2 post-hoc of #40, M7).

The empty ``()`` group in ``_figure_before_label``'s regex was dead: removing
it changes no match span, no captured figure and no extracted parameter. These
tests pin that a label written straight after a figure still labels that
figure, never the next one.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf


@pytest.mark.parametrize(
    ("ask", "days"),
    [
        ("60 days at 5,000 per day", 60.0),
        ("45 days at 2,000 per day", 45.0),
    ],
)
def test_a_label_after_a_figure_labels_that_figure(ask, days):
    fn = cf.CALCULATORS["delay_damages_daily"]
    params = cf.extract_calculation_params_from_text(fn, ask)
    assert params["days"] == days
