"""A document's FILENAME can be the only place a fact exists.

Live case (Infra Pack UI eval, 2026-08-23, Q5). Asked when the Street Lighting
NOC expires, retrieval returned the correct document — and the answer was still
wrong, because "17 July 2025" appears nowhere in the extracted text. It exists
only in the filename::

    AM Rev Design NOC PRJ2 Infra Package 1 - Street Lighting (exp. 17Jul25).pdf

The chunk itself is OCR-degraded: "NOC Reference 0095-PRJ2-PR Issue Date
13\02\2025 ... Time Duration 6 months". So the model did date arithmetic on
noise to reach a date the document never states in extractable form.

No amount of ranking work fixes that — the model simply never saw the name.
Construction documents routinely carry expiry dates, revision letters,
discipline codes and status in the filename alone, so the name travels with
each excerpt now.

Deliberately NOT embedded: putting the filename into the indexed text would
mean re-embedding all 172,809 chunks, and would let filename tokens influence
similarity. This is a context change only.
"""
from __future__ import annotations

from app.core.rag.inject import _MAX_SOURCE_NAME_CHARS, format_chunks_as_system_message
from app.core.rag.vector_store import Chunk

NOC_NAME = "AM Rev Design NOC PRJ2 Infra Package 1 - Street Lighting (exp. 17Jul25).pdf"


def _chunk(name: str = "", text: str = "NOC Reference 0095-PRJ2-PR") -> Chunk:
    c = Chunk(
        chunk_id="f1c78383:26cf9357:0", project_id="f1c78383", doc_id="26cf9357",
        chunk_index=0, text=text, score=0.702,
    )
    c.source_name = name
    return c


def test_the_expiry_that_only_exists_in_the_filename_reaches_the_model():
    """The whole point, in one assertion."""
    msg = format_chunks_as_system_message([_chunk(NOC_NAME)], 12)
    assert "17Jul25" in msg["content"]


def test_marker_carries_the_source_name():
    msg = format_chunks_as_system_message([_chunk(NOC_NAME)], 12)
    assert "src=" in msg["content"]


def test_model_is_told_filenames_are_evidence():
    """Without the instruction the tag reads as decoration, and a model that
    has been told to answer only from the excerpts may refuse to use it."""
    msg = format_chunks_as_system_message([_chunk(NOC_NAME)], 12)
    content = msg["content"]
    assert "FILENAMES ARE EVIDENCE" in content
    assert "disagree" in content, "must say what to do when name and text conflict"


def test_long_names_keep_the_TAIL_not_the_head():
    """Revision letters and parenthesised dates sit at the END of engineering
    filenames. Truncating from the right would throw away the fact this
    feature exists to surface."""
    long_name = ("QZ-ENG-310-0000-EXC-SPC-IF-000013-B_Scope_of_Package_"
                 "Requirements_Infrastructure (exp. 17Jul25).pdf")
    assert len(long_name) > _MAX_SOURCE_NAME_CHARS
    msg = format_chunks_as_system_message([_chunk(long_name)], 12)
    assert "17Jul25" in msg["content"], "the tail carries the fact"
    assert "QZ-ENG-310-0000-EXC" not in msg["content"], "head is what gets dropped"


def test_long_contract_filename_keeps_the_PREFIX_YEAR_SEQ():
    """A3: tail-only truncation dropped AB-2023-101 from long Package 1
    Conditions of Contract names, so the model could not name the cited
    contract. The id stays in src=; the tail is still kept."""
    long_name = (
        "AB-2023-101_the client project II Infrastructure Package 1_"
        "Vol 1 - Conditions of Contract.pdf"
    )
    assert len(long_name) > _MAX_SOURCE_NAME_CHARS
    msg = format_chunks_as_system_message([_chunk(long_name)], 12)
    assert "AB-2023-101" in msg["content"]
    assert "CONTRACT ATTRIBUTION" in msg["content"]
    assert "Conditions of Contract.pdf" in msg["content"]


def _marker_of(msg: dict) -> str:
    """Just the excerpt line. The HEADER also contains the literal "src=..."
    inside its instruction text, so asserting against the whole message would
    pass or fail for the wrong reason."""
    return msg["content"].rsplit("\n\n", 1)[-1]


def test_no_marker_noise_when_the_name_is_unknown():
    """A chunk whose document name could not be resolved must not emit an
    empty src= tag on every excerpt."""
    msg = format_chunks_as_system_message([_chunk("")], 12)
    assert "src=" not in _marker_of(msg)
    assert "src=" in msg["content"], "…but the header instruction still mentions it"


def test_source_name_stays_off_the_wire():
    """Internal context signal, like revision/superseded. Adding it to the API
    payload would change the response shape the Sources panel consumes."""
    c = _chunk(NOC_NAME)
    assert "source_name" not in c.to_dict()


def test_source_name_does_not_affect_chunk_equality():
    """compare=False, matching the other filename-derived fields — otherwise
    every equality assertion in the suite starts depending on name resolution."""
    a, b = _chunk(NOC_NAME), _chunk("")
    assert a == b
