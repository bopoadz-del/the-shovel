"""When the question names the governing document, ranking must honour it.

Live SET4.1 on 3403cb5, asked five times each:

    "PER THE PROJECT SPECIFICATION for SYN-2024-001, to what degree must
     structural backfill under foundations be compacted?"     -> 1/5

The answer came from an *MTS & Risk Assessment for Site Office Mobilization*
(95% MDD) instead of the specification (98%, modified Proctor). The lux
questions did the same. Both figures are real and both documents are real --
the question said which one governs and ranking treated every document as
equally eligible.

A question that does NOT name a class must rank exactly as before.
Synthetic document names throughout.
"""
import pytest

from app.core.rag import retriever as r
from app.core.rag.vector_store import Chunk

SPEC_ASK = ("Per the project specification for SYN-2024-001, to what degree must "
            "structural backfill under foundations be compacted?")
HSE_ASK = ("Per the project HSE lighting requirements, what minimum illumination "
           "is required for concrete placement at night?")
PLAIN_ASK = "What is the Defects Notification Period under this contract?"

SPEC_DOC = "SYN-SPEC-001 Particular Specification - Earthworks Rev 02.pdf"
MOBILIZATION_DOC = "MTS & Risk Assessment for Site Office Mobilization Rev-00.pdf"
HSE_DOC = "SYN-PLN-HS-001 HSE Plan Rev 00.pdf"
NEUTRAL_DOC = "Contract Data SYN-2024-001.pdf"


# ── which class the question names ─────────────────────────────────────────

@pytest.mark.parametrize("ask,expected", [
    (SPEC_ASK, "specification"),
    (HSE_ASK, "hse"),
    ("Per the project lifting plan, when does a lift count as critical?", "lifting"),
    (PLAIN_ASK, ""),
    # Mentioning a word is not naming a governing source.
    ("What does the specification say about concrete?", ""),
    ("Is safety training required?", ""),
])
def test_the_question_names_the_class_or_it_does_not(ask, expected):
    assert r.source_class_named_by(ask) == expected


# ── which document belongs to that class ───────────────────────────────────

@pytest.mark.parametrize("name,class_name,expected", [
    (SPEC_DOC, "specification", True),
    (MOBILIZATION_DOC, "specification", False),
    (HSE_DOC, "hse", True),
    (SPEC_DOC, "hse", False),
])
def test_the_document_name_says_what_it_is(name, class_name, expected):
    assert r.filename_is_source_class(name, class_name) is expected


def test_the_named_class_is_lifted_and_an_off_scope_document_demoted():
    assert r.source_class_adjustment(SPEC_DOC, "specification") > 0
    assert r.source_class_adjustment(MOBILIZATION_DOC, "specification") < 0
    # Neither: untouched, so an ordinary contract document is not penalised
    # merely for not being the named class.
    assert r.source_class_adjustment(NEUTRAL_DOC, "specification") == 0.0


def test_nothing_moves_when_no_class_is_named():
    for name in (SPEC_DOC, MOBILIZATION_DOC, HSE_DOC, NEUTRAL_DOC):
        assert r.source_class_adjustment(name, "") == 0.0


# ── the ranking itself ─────────────────────────────────────────────────────

def _scored():
    mob = Chunk(chunk_id="m1", project_id="p1", doc_id="dmob", chunk_index=0,
                text="density tests should not be less than 95% Maximum Dry Density", score=0.81)
    spec = Chunk(chunk_id="s1", project_id="p1", doc_id="dspec", chunk_index=0,
                 text="Compaction of the backfill to minimum 98% of maximum dry density "
                      "of the modified proctor test", score=0.78)
    return [(0.81, mob), (0.78, spec)], {"dmob": MOBILIZATION_DOC, "dspec": SPEC_DOC}


def test_the_specification_outranks_the_mobilization_statement():
    scored, names = _scored()
    r._apply_source_class_preference(SPEC_ASK, scored, names)
    top = max(scored, key=lambda pair: pair[0])[1]
    assert top.doc_id == "dspec", [(s, c.doc_id) for s, c in scored]


def test_a_plain_question_leaves_the_order_untouched():
    scored, names = _scored()
    before = [(s, c.doc_id) for s, c in scored]
    r._apply_source_class_preference(PLAIN_ASK, scored, names)
    assert [(s, c.doc_id) for s, c in scored] == before


def test_the_chunk_score_follows_the_adjustment():
    scored, names = _scored()
    r._apply_source_class_preference(SPEC_ASK, scored, names)
    spec_chunk = next(c for _s, c in scored if c.doc_id == "dspec")
    assert spec_chunk.score == pytest.approx(0.78 + r._SOURCE_CLASS_BONUS)
