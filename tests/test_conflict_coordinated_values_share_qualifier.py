"""Values written as one coordination share the words around them.

A passage that names two levels of one thing in one coordination ("rolled to
ninety percent (90%) and ninety-three percent (93%) of the maximum dry
density") states both levels; the qualifier after the run belongs to every
value in it, not to the last one. Read value by value, the last value took the
whole qualifier, so that document was reported as giving the other level and
the answer said the documents differ when every other document agreed with it.

Every test builds its own passages (no fixture file, no case id). Each
decision has two inputs, and the other half is pinned too: a value that its
own words qualify ("85% beneath footways and 93% of ...") is still read alone,
and a document stating only the other level is still a conflict.
"""

from __future__ import annotations

import pytest

from app.agents.evidence_conflicts import _candidates, _coordinated, _groups, _terms, detect, extract


def _rows(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"doc_name": doc, "page_or_section": "NOTES", "snippet": text} for doc, text in pairs]


_AGREEING = [
    ("Earthworks Specification", "Subbase shall be rolled to 90% of the maximum dry density."),
    ("Priced Bill of Quantities", "Trench backfill compacted to 90 % of maximum dry density, complete as specified."),
]


@pytest.mark.parametrize(
    "coordinated",
    [
        # Spelled-out levels beside their digits, joined by "and".
        ("Soaked samples compacted to ninety percent (90%) and ninety-three percent (93%) "
         "of the maximum dry density are summarised in Table 4."),
        # A comma list closed by "or", three levels.
        "Test specimens compacted to 88%, 90% or 93% of the maximum dry density are listed in Table 4.",
    ],
)
def test_a_coordinated_run_is_not_one_document_value(coordinated):
    rows = _rows(*_AGREEING, ("Ground Investigation Report", coordinated))
    answer = "Pass: 91.2% of maximum dry density meets the 90% of maximum dry density requirement."
    assert detect(answer, rows, "Field density 1.79, maximum dry density 1.96, 90% required. Pass?") == []
    assert detect(answer, rows, None) == []


@pytest.mark.parametrize(
    "other",
    [
        # Each level carries its own qualifier: not a coordination, read alone.
        "Subbase shall be rolled to 85% beneath footways and 93% of the maximum dry density.",
        # A document that states only the other level.
        "Subbase shall be rolled to 93% of the maximum dry density.",
    ],
)
def test_a_value_its_own_words_qualify_still_differs(other):
    rows = _rows(*_AGREEING, ("Ground Investigation Report", other))
    answer = "Pass: 91.2% of maximum dry density meets the 90% of maximum dry density requirement."
    found = detect(answer, rows, None)
    assert found, "a document stating 93% for the same thing must still be a conflict"
    assert {v["value"] for v in found[0]["values"]} >= {"90%", "93%"}


@pytest.mark.parametrize(
    "gap, joined",
    [
        (" and ", True),
        (") and ninety-three percent (", True),
        (", ", True),
        (" or two hundred and five mm ", True),
        (" under verges and ", False),
        (" & ", True),
        (" ", False),
        (" to the cap and ", False),
    ],
)
def test_coordination_gap(gap, joined):
    assert _coordinated(gap) is joined


@pytest.mark.parametrize(
    "clause, runs",
    [
        # One kind joined by a coordination: one run.
        ("rolled to 90% and 93% of the maximum dry density", [(0, 1)]),
        ("rolled to 88%, 90% or 93% of the maximum dry density", [(0, 2)]),
        # Two kinds side by side are never one coordination.
        ("a 150 mm layer and 93% of the maximum dry density", [(0, 0), (1, 1)]),
        ("a 40 kN load or 25 mm deflection", [(0, 0), (1, 1)]),
        # A bare coordinator between two kinds: the gap alone would join
        # them, the kinds keep them apart.
        ("a layer of 150 mm and 93% of the maximum dry density", [(0, 0), (1, 1)]),
        ("a load of 40 kN or 25 mm under it", [(0, 0), (1, 1)]),
    ],
)
def test_runs_join_one_kind_only(clause, runs):
    vals = [v for v in extract(clause) if v.kind != "text"]
    assert _groups(clause, vals) == runs


@pytest.mark.parametrize(
    "passage, before_word, after_word",
    [
        # The run is long: the word before it is more than a window's width
        # from the last value, so only the run's own window reaches it.
        ("Specimens were rolled to eighty-eight percent (88%), ninety percent (90%), "
         "ninety-one percent (91%), ninety-two percent (92%) or ninety-three percent (93%) "
         "of modified Proctor density.", "rolled", "Proctor"),
        ("Bar cover was set at forty millimetres (40 mm), forty-five millimetres (45 mm), "
         "fifty millimetres (50 mm), fifty-five millimetres (55 mm) and sixty millimetres "
         "(60 mm) to the outer face of the wall.", "cover", "outer"),
    ],
)
def test_every_value_of_a_run_shares_the_run_neighbourhood(passage, before_word, after_word):
    cands = _candidates(_rows(("Test Report", passage)))
    assert len(cands) == 5
    shared = cands[0].sent_terms
    assert all(c.sent_terms == shared for c in cands)
    assert _terms(before_word) <= shared and _terms(after_word) <= shared
    assert all(_terms(before_word) <= c.local_terms and _terms(after_word) <= c.local_terms
               for c in cands)
