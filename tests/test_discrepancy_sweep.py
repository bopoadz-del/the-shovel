"""Two-input tests for the discrepancy sweep's pure detection half.

Each test invents its own strings; no case, document, standard, value or row
is named. The readers are ``citation_provenance``'s typed readers (the same
ones the conflict statement uses), so these tests exercise the sweep against
recorded strings, not a mock of the retriever.
"""

from app.agents.discrepancy_sweep import SweepHit, sweep_differences


def _differing(hits: list[SweepHit]) -> list[str]:
    return [h.differing_value for h in hits]


def test_same_identifier_different_figure_recalls():
    stated = "The cover is 50 mm per ACI 318."
    candidates = [
        ("drawing.pdf", "The cover is 60 mm per ACI 318."),
    ]
    hits = sweep_differences(stated, candidates)
    assert _differing(hits) == ["60 mm"]
    assert hits[0].identifier  # a real identifier, not a name list


def test_same_figure_is_not_recalled():
    stated = "The cover is 50 mm per ACI 318."
    candidates = [
        ("drawing.pdf", "The cover is 50 mm per ACI 318."),
    ]
    assert sweep_differences(stated, candidates) == []


def test_different_identifier_is_not_recalled():
    stated = "The cover is 50 mm per ACI 318."
    candidates = [
        ("drawing.pdf", "The cover is 60 mm per BS 8110."),
    ]
    assert sweep_differences(stated, candidates) == []


def test_different_unit_is_not_recalled():
    # Same designation, but the candidate's figure is a different unit; the
    # unit equivalence is the conflict statement's judgement, not this recall.
    stated = "The cover is 50 mm per ACI 318."
    candidates = [
        ("drawing.pdf", "The cover is 60 m per ACI 318."),
    ]
    assert sweep_differences(stated, candidates) == []


def test_rounding_to_same_figure_is_not_recalled():
    stated = "The cover is 50 mm per ACI 318."
    candidates = [
        ("drawing.pdf", "The cover is 50.00 mm per ACI 318."),
    ]
    assert sweep_differences(stated, candidates) == []


def test_two_designations_shared_one_recalls():
    stated = "The cover is 50 mm per ACI 318 for the slab."
    candidates = [
        ("drawing.pdf", "The cover is 60 mm per ACI 318."),
        ("other.pdf", "The cover is 70 mm per BS 8110."),
    ]
    hits = sweep_differences(stated, candidates)
    # only the candidate sharing ACI 318 recalls
    assert _differing(hits) == ["60 mm"]
