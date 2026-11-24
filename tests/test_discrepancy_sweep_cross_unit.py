"""The discrepancy sweep recalls a differing figure written in ANOTHER unit of
the same kind, read and compared by the conflict statement's own reader.

Every test invents its own strings and runs at least two distinct inputs; no
case, document, standard, value or row is named in product code. The hard
negatives: a figure in another unit that converts to the stated value is the
same value, a figure of another kind is never compared, and another
identifier is never recalled.
"""

import pytest

from app.agents.discrepancy_sweep import sweep_differences


def _differing(hits):
    return [h.differing_value for h in hits]


@pytest.mark.parametrize("stated,passage,expected", [
    ("The cover is 40 mm per ACI 318.", "The cover is 0.05 m per ACI 318.", "0.05 m"),
    ("The curing period is 2 weeks per Clause 9.1.",
     "The curing period is 21 days per Clause 9.1.", "21 days"),
    ("The wall is 0.3 m thick per Table 6.2.", "The wall is 250 mm thick per Table 6.2.", "250 mm"),
])
def test_other_unit_same_kind_different_value_recalls(stated, passage, expected):
    hits = sweep_differences(stated, [("other.pdf", passage)])
    assert _differing(hits) == [expected]
    assert hits[0].source == "other.pdf" and hits[0].passage == passage


@pytest.mark.parametrize("stated,passage", [
    ("The cover is 50 mm per ACI 318.", "The cover is 0.05 m per ACI 318."),
    ("The curing period is 2 weeks per Clause 9.1.",
     "The curing period is 14 days per Clause 9.1."),
    ("The wall is 0.3 m thick per Table 6.2.", "The wall is 30 cm thick per Table 6.2."),
])
def test_other_unit_converting_to_the_stated_value_is_not_recalled(stated, passage):
    assert sweep_differences(stated, [("other.pdf", passage)]) == []


@pytest.mark.parametrize("stated,passage,recalled", [
    # One millimetre apart is a difference at the scale the draft wrote it,
    # though 0.001 m sits inside the tolerance's absolute floor in metres.
    ("The cover is 40 mm per ACI 318.", "The cover is 0.041 m per ACI 318.", True),
    ("The cover is 25 mm per BS 8110.", "The cover is 0.026 m per BS 8110.", True),
    # A rounding of the stated value in another unit is the same value.
    ("The span is 4.5 m per BS 8110.", "The span is 4500.2 mm per BS 8110.", False),
    ("The cover is 40 mm per ACI 318.", "The cover is 4.0 cm per ACI 318.", False),
])
def test_tolerance_applies_at_the_stated_scale(stated, passage, recalled):
    assert bool(sweep_differences(stated, [("other.pdf", passage)])) is recalled


@pytest.mark.parametrize("stated,passage", [
    ("The cover is 40 mm per ACI 318.", "The cover is 0.05 t per ACI 318."),
    ("The curing period is 2 weeks per Clause 9.1.", "The load is 21 kN per Clause 9.1."),
])
def test_another_kind_is_never_compared(stated, passage):
    assert sweep_differences(stated, [("other.pdf", passage)]) == []


@pytest.mark.parametrize("stated,passage", [
    ("The cover is 40 mm per ACI 318.", "The cover is 0.05 m per BS 8110."),
    ("The curing period is 2 weeks per Clause 9.1.",
     "The curing period is 21 days per Clause 9.2."),
])
def test_other_unit_under_another_identifier_is_not_recalled(stated, passage):
    assert sweep_differences(stated, [("other.pdf", passage)]) == []


@pytest.mark.parametrize("passage,recalled", [
    ("The cover is 0.18 m per Table 3.1.", False),   # inside the stated range
    ("The cover is 0.25 m per Table 3.1.", True),    # outside it
    ("The cover is 17 cm per Table 3.1.", False),
    ("The cover is 12 cm per Table 3.1.", True),
])
def test_stated_range_is_one_value(passage, recalled):
    stated = "The cover is 150-200 mm per Table 3.1."
    assert bool(sweep_differences(stated, [("other.pdf", passage)])) is recalled


@pytest.mark.parametrize("passage,recalled", [
    ("The joint gap is 0.005 m per Clause 2.1.4.", True),
    ("The joint gap is 0 m per Clause 2.1.4.", False),
])
def test_stated_zero_compares_base_values(passage, recalled):
    stated = "The joint gap is 0 mm per Clause 2.1.4."
    assert bool(sweep_differences(stated, [("other.pdf", passage)])) is recalled


@pytest.mark.parametrize("stated,passage", [
    ("Handover is 15 March 2027 per Clause 8.2.", "Handover is 20 March 2027 per Clause 8.2."),
    ("The concrete is grade C40 per Table 2.3.", "The concrete is grade C35 per Table 2.3."),
])
def test_unmeasured_values_are_not_swept(stated, passage):
    assert sweep_differences(stated, [("other.pdf", passage)]) == []


@pytest.mark.parametrize("stated,passage", [
    ("The cover is 40 mm per Clause 8.4.",
     "Cover shall comply with Clause 8.4. The slab is 0.25 m thick throughout."),
    ("The curing period is 2 weeks per Table 3.1.",
     "Table 3.1 applies to curing generally. Formwork stays 5 days per Table 7.2."),
])
def test_other_unit_figure_outside_the_identifier_scope_is_not_recalled(stated, passage):
    # The converted figure is stated for something else (another sentence,
    # or another identifier), so the cross-unit recall reads only the
    # identifier's own scope, the same as the same-unit recall.
    assert sweep_differences(stated, [("other.pdf", passage)]) == []


@pytest.mark.parametrize("stated,passage,expected", [
    ("The cover is 40 mm per Clause 8.4.",
     "General notes apply. Clause 8.4 sets the cover at 0.05 m.", "0.05 m"),
    ("The curing period is 2 weeks per Table 3.1.",
     "Table 3.1\nThe curing period is 21 days.\nTable 3.2\nStripping after 5 days.",
     "21 days"),
])
def test_other_unit_figure_inside_the_identifier_scope_recalls(stated, passage, expected):
    assert _differing(sweep_differences(stated, [("other.pdf", passage)])) == [expected]
