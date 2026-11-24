"""Two-input tests for the discrepancy sweep's pure detection half.

Each test invents its own strings and runs at least two distinct inputs; no
case, document, standard, value or row is named. The readers are
``citation_provenance``'s typed readers (the same ones the conflict statement
uses), so these tests exercise the sweep against recorded strings, not a mock
of the retriever.
"""

import pytest

from app.agents.discrepancy_sweep import SweepHit, _designations, sweep_differences


def _differing(hits: list[SweepHit]) -> list[str]:
    return [h.differing_value for h in hits]


@pytest.mark.parametrize("stated,candidates,expected", [
    (
        "The cover is 50 mm per ACI 318.",
        [("drawing.pdf", "The cover is 60 mm per ACI 318.")],
        ["60 mm"],
    ),
    (
        "The span is 4.5 m per BS 8110.",
        [("report.pdf", "The span is 5.0 m per BS 8110.")],
        ["5.0 m"],
    ),
])
def test_same_identifier_different_figure_recalls(stated, candidates, expected):
    hits = sweep_differences(stated, candidates)
    assert _differing(hits) == expected
    assert hits and hits[0].identifier  # a real identifier, not a name list


@pytest.mark.parametrize("stated,candidates", [
    (
        "The cover is 50 mm per ACI 318.",
        [("drawing.pdf", "The cover is 50 mm per ACI 318.")],
    ),
    (
        "The span is 4.5 m per BS 8110.",
        [("report.pdf", "The span is 4.5 m per BS 8110.")],
    ),
])
def test_same_figure_is_not_recalled(stated, candidates):
    assert sweep_differences(stated, candidates) == []


@pytest.mark.parametrize("stated,candidates", [
    (
        "The cover is 50 mm per ACI 318.",
        [("drawing.pdf", "The cover is 60 mm per BS 8110.")],
    ),
    (
        "The span is 4.5 m per BS 8110.",
        [("report.pdf", "The span is 5.0 m per ACI 318.")],
    ),
])
def test_different_identifier_is_not_recalled(stated, candidates):
    assert sweep_differences(stated, candidates) == []


@pytest.mark.parametrize("stated,candidates", [
    (
        "The cover is 50 mm per ACI 318.",
        [("drawing.pdf", "The cover is 60 kg per ACI 318.")],
    ),
    (
        "The span is 4.5 m per BS 8110.",
        [("report.pdf", "The span is 5.0 kN per BS 8110.")],
    ),
])
def test_different_unit_is_not_recalled(stated, candidates):
    # Same designation, but the candidate's figure is in a unit of another
    # kind (a mass, a force against a length): never compared. A unit of the
    # same kind is compared after conversion
    # (tests/test_discrepancy_sweep_cross_unit.py).
    assert sweep_differences(stated, candidates) == []


@pytest.mark.parametrize("stated,candidates", [
    (
        "The cover is 50 mm per ACI 318.",
        [("drawing.pdf", "The cover is 50.00 mm per ACI 318.")],
    ),
    (
        "The span is 4.5 m per BS 8110.",
        [("report.pdf", "The span is 4.50 m per BS 8110.")],
    ),
])
def test_rounding_to_same_figure_is_not_recalled(stated, candidates):
    assert sweep_differences(stated, candidates) == []


@pytest.mark.parametrize("stated,candidates,expected", [
    (
        "The cover is 50 mm per ACI 318 for the slab.",
        [
            ("drawing.pdf", "The cover is 60 mm per ACI 318."),
            ("other.pdf", "The cover is 70 mm per BS 8110."),
        ],
        ["60 mm"],
    ),
    (
        "The span is 4.5 m per BS 8110 for the beam.",
        [
            ("report.pdf", "The span is 5.0 m per BS 8110."),
            ("other.pdf", "The span is 6.0 m per ACI 318."),
        ],
        ["5.0 m"],
    ),
])
def test_two_designations_shared_one_recalls(stated, candidates, expected):
    # only the candidate sharing the stated designation recalls
    hits = sweep_differences(stated, candidates)
    assert _differing(hits) == expected


@pytest.mark.parametrize("text,expected", [
    ("ACI 318 and ACI 318 both apply.", ["aci 318"]),
    ("Vol 6 and Vol 6 govern.", ["vol 6"]),
])
def test_designations_dedupe_and_normalise(text, expected):
    assert _designations(text) == expected


@pytest.mark.parametrize("stated", [
    "The cover is 50 mm.",
    "The span is 4.5 m.",
])
def test_no_designation_is_not_swept(stated):
    # A figure with no distinctive identifier cannot be matched to a passage.
    candidates = [("drawing.pdf", "The cover is 60 mm per ACI 318.")]
    assert sweep_differences(stated, candidates) == []


@pytest.mark.parametrize("stated", [
    "The cover per ACI 318.",
    "The span per BS 8110.",
])
def test_no_figure_is_not_swept(stated):
    # A distinctive identifier with no stated figure has nothing to compare.
    candidates = [("drawing.pdf", "The cover is 60 mm per ACI 318.")]
    assert sweep_differences(stated, candidates) == []
