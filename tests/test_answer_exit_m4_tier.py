"""M4: the server envelope names the tier that answered and caps confidence.

``ORDERS_08NOV_1300Z_SHOVEL_M4_cap_on_the_server``: when no cited source is
project_corpus, the envelope itself carries ``tier`` "general" and confidence
no higher than MEDIUM. The API, exports and any other client must never see
HIGH on a general-only answer; the client renders what the envelope says. A
project_corpus-cited answer -- and a mixed answer citing one project_corpus
source -- is not general, and its rows keep their real confidence.

The rule lives in ``answer_exit``, the single exit choke point every chat
answer passes through (``check_result`` for the non-streaming result,
``check_end_event`` for the streamed end event). Structural only: the source
rows here are synthetic and name no real project or question.
"""
from __future__ import annotations

import pytest

from app.agents.answer_exit import check_end_event, check_result


@pytest.mark.parametrize(
    "rows",
    [
        # general_knowledge layer: the platform calculator row.
        [
            {
                "doc_name": "Platform calculator",
                "doc_id": "",
                "layer": "general_knowledge",
                "layer_label": "Platform calculator",
                "confidence": "High",
                "score": 1.0,
                "page_or_section": "Calculated",
            }
        ],
        # master_corpus layer: a fallback chunk from the shared corpus.
        [
            {
                "doc_name": "MasterCorpus.pdf",
                "doc_id": "mc_1",
                "layer": "master_corpus",
                "confidence": "High",
                "score": 0.8,
                "page_or_section": "p. 2",
            }
        ],
    ],
)
def test_general_only_end_event_is_general_and_capped(rows):
    """Two distinct general layers: both read GENERAL with confidence below
    HIGH (the server caps High at Medium on the envelope)."""
    event = check_end_event({"type": "end", "content": "A general answer.", "sources": rows})

    assert event["tier"] == "general"
    assert event["sources"]
    assert all(row.get("confidence") != "High" for row in event["sources"])
    assert event["sources"][0]["confidence"] == "Medium"


@pytest.mark.parametrize(
    "rows",
    [
        # a project_corpus-cited answer: one own-layer chunk.
        [
            {
                "doc_name": "Spec.pdf",
                "doc_id": "doc_1",
                "layer": "own",
                "confidence": "High",
                "score": 0.9,
                "page_or_section": "p. 4",
            }
        ],
        # a mixed answer: one project_corpus source among general rows.
        [
            {
                "doc_name": "Spec.pdf",
                "doc_id": "doc_1",
                "layer": "own",
                "confidence": "High",
                "score": 0.9,
                "page_or_section": "p. 4",
            },
            {
                "doc_name": "MasterCorpus.pdf",
                "doc_id": "mc_1",
                "layer": "master_corpus",
                "confidence": "High",
                "score": 0.8,
                "page_or_section": "p. 2",
            },
        ],
    ],
)
def test_project_cited_end_event_is_not_general_and_not_capped(rows):
    """A project_corpus-cited answer and a mixed answer are not general, and
    HIGH confidence survives the envelope untouched."""
    event = check_end_event({"type": "end", "content": "A project answer.", "sources": rows})

    assert event["tier"] == "project"
    assert all(row.get("confidence") == "High" for row in event["sources"])


@pytest.mark.parametrize(
    "rows",
    [
        # negative control: no sources at all still reads general, never breaks.
        [],
        # negative control: a general row already below HIGH stays where it is.
        [
            {
                "doc_name": "MasterCorpus.pdf",
                "doc_id": "mc_1",
                "layer": "master_corpus",
                "confidence": "Low",
                "score": 0.3,
                "page_or_section": "p. 2",
            }
        ],
    ],
)
def test_general_only_envelope_never_raises_and_preserves_below_high(rows):
    """The cap only ever lowers HIGH; empty and already-capped envelopes pass
    through intact, and the exit never breaks an answer."""
    event = check_end_event({"type": "end", "content": "Answer.", "sources": rows})

    assert event["tier"] == "general"
    assert all(row.get("confidence") != "High" for row in event["sources"])


@pytest.mark.parametrize(
    "rows",
    [
        [
            {
                "doc_name": "Platform calculator",
                "doc_id": "",
                "layer": "general_knowledge",
                "confidence": "High",
                "score": 1.0,
                "page_or_section": "Calculated",
            }
        ],
        [
            {
                "doc_name": "MasterCorpus.pdf",
                "doc_id": "mc_1",
                "layer": "master_corpus",
                "confidence": "High",
                "score": 0.8,
                "page_or_section": "p. 2",
            }
        ],
    ],
)
def test_general_only_non_stream_result_is_general_and_capped(rows):
    """The non-streaming path (check_result) applies the same rule: GENERAL
    with confidence below HIGH on the envelope the API returns."""
    result = check_result({"answer": "A general answer.", "sources": rows})

    assert result["tier"] == "general"
    assert all(row.get("confidence") != "High" for row in result["sources"])
    assert result["sources"][0]["confidence"] == "Medium"


@pytest.mark.parametrize(
    "provenance,expected_tier",
    [
        # A figure capped to general_knowledge (its chunk held a competing
        # figure) lowers the banner to general even though the retrieval
        # source is an own project chunk.
        (
            [{"figure": "28 days", "source": "general_knowledge",
              "competing": ["42 days"]}],
            "general",
        ),
        # An answer whose figures are all project-credited keeps the project
        # banner.
        (
            [{"figure": "28 days", "source": "project_document",
              "sentence": "The deadline is 28 days."}],
            "project",
        ),
    ],
)
def test_answer_tier_reflects_a_capped_figure(provenance, expected_tier):
    """B11: the answer-tier banner reflects the lowest credited tier among
    the answer's figures; the source-row confidence stays retrieval-driven."""
    rows = [
        {
            "doc_name": "Spec.pdf",
            "doc_id": "doc_1",
            "layer": "own",
            "confidence": "High",
            "score": 0.9,
            "page_or_section": "p. 4",
        }
    ]
    event = check_end_event({"type": "end", "content": "Answer.",
                              "sources": rows, "provenance": provenance})

    assert event["tier"] == expected_tier
    # The figure judgement never caps the retrieval confidence: HIGH survives
    # a project-cited answer even when one of its figures was capped.
    assert event["sources"][0]["confidence"] == "High"
