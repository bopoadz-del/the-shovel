"""The discrepancy sweep compares only a figure stated FOR the shared
identifier (A2, discrepancy sweep precision).

A candidate passage that names the identifier in one sentence and carries an
unrelated figure in another ("... comply with <code>. The slab is 200 mm")
states no value for the identifier, so it is no difference. A figure is
stated for the identifier when its sentence names it, or when it sits under a
heading line led by it, down to the next line led by a designation.

Every input is invented here; two inputs per decision.
"""

from __future__ import annotations

import pytest

from app.agents.discrepancy_sweep import _identifier_scope, sweep_differences


def _hits(stated: str, passage: str) -> list[str]:
    return [h.differing_value for h in sweep_differences(stated, [("doc", passage)])]


@pytest.mark.parametrize("stated, passage", [
    ("Under XQ 412 the minimum cover is 40 mm.",
     "Cover shall comply with XQ 412. The slab thickness is 200 mm throughout."),
    ("Per Spec 77 the joint width is 12 mm.",
     "Sealants per Spec 77 apply. Paving thickness 150 mm on the access road."),
])
def test_a_figure_in_another_sentence_is_not_stated_for_the_identifier(stated, passage):
    assert _hits(stated, passage) == []


@pytest.mark.parametrize("stated, passage, differing", [
    ("Under XQ 412 the minimum cover is 40 mm.",
     "General notes apply. Under XQ 412 the minimum cover is 50 mm.", "50 mm"),
    ("Per Spec 77 the joint width is 12 mm.",
     "Per Spec 77 the joint width is 20 mm for all bays.", "20 mm"),
])
def test_a_figure_in_a_sentence_naming_the_identifier_is_compared(stated, passage, differing):
    assert _hits(stated, passage) == [differing]


@pytest.mark.parametrize("stated, passage, differing", [
    ("Under Clause 6.3.2 the minimum cover is 40 mm.",
     "Clause 6.3.2 Minimum cover\nThe minimum cover shall be 50 mm.", "50 mm"),
    ("Under Clause 9.1.4 the lap length is 600 mm.",
     "9.1.4 Laps\nLaps shall be 750 mm long.", "750 mm"),
])
def test_the_body_under_a_heading_led_by_the_identifier_is_compared(stated, passage, differing):
    assert _hits(stated, passage) == [differing]


@pytest.mark.parametrize("stated, passage", [
    ("Under Clause 6.3.2 the minimum cover is 40 mm.",
     "Clause 6.3.2 Minimum cover\nClause 6.3.3 Spacers\nSpacers at 900 mm centres."),
    ("Under Clause 9.1.4 the lap length is 600 mm.",
     "9.1.4 Laps\nTable 9.2 Bar sizes\nBars of 450 mm are cut on site."),
])
def test_the_heading_scope_ends_at_the_next_line_led_by_a_designation(stated, passage):
    assert _hits(stated, passage) == []


@pytest.mark.parametrize("stated, passage", [
    ("Under XQ 412 the minimum cover is 40 mm.",
     "XQ 412 governs all concrete.\nThe slab thickness is 200 mm."),
    ("Per Spec 77 the joint width is 12 mm.",
     "Spec 77 applies to sealants.\nKerb height 150 mm."),
])
def test_a_sentence_led_by_the_identifier_is_not_a_heading(stated, passage):
    assert _hits(stated, passage) == []


@pytest.mark.parametrize("passage, shared, kept", [
    ("Intro text. Per XQ 412 cover is 50 mm. Slab is 200 mm.", {"xq 412"},
     "Per XQ 412 cover is 50 mm."),
    ("Clause 6.3.2 Cover\nCover 50 mm.\nClause 7.1 Laps\nLap 600 mm.", {"clause 6.3.2"},
     "Clause 6.3.2 Cover\nCover 50 mm."),
])
def test_scope_keeps_only_what_is_stated_for_the_identifier(passage, shared, kept):
    assert _identifier_scope(passage, shared) == kept
