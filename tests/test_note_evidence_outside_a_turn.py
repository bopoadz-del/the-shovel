"""``note_evidence`` records the places a turn read; outside a turn it is a
quiet no-op.

Kept out of the reverted #438 file: this behaviour predates #438 and stays
after its revert. Every document and passage here is invented for this file;
each decision is shown on two inputs.
"""

from __future__ import annotations

import pytest

from app.agents import answer_exit


def _excerpts(*parts: tuple[str, int, str, str]) -> dict:
    """A pre-injected excerpt message in the marker form the model sees."""
    return {"role": "system", "content": "\n".join(
        f"[doc_id={doc} chunk={idx} score=0.800 class=project_corpus src={name}] {text}"
        for doc, idx, name, text in parts)}


_PARTS = [
    pytest.param([("d-e", 0, "Truss Elevation.pdf", "Top chord grade S355.")], id="one-passage"),
    pytest.param([("d-f", 4, "Truss Method.pdf", "Bottom chord grade S275."),
                  ("d-g", 5, "Truss Notes.pdf", "Ties at 600 mm.")], id="two-passages"),
]


@pytest.mark.parametrize("parts", _PARTS)
def test_no_open_turn_records_nothing_and_warns_nothing(parts, caplog):
    """Outside a turn there is nothing to record into: a quiet no-op, never a
    logged failure."""
    token = answer_exit._TURN.set(None)
    try:
        with caplog.at_level("WARNING", logger=answer_exit._LOG.name):
            answer_exit.note_evidence(_excerpts(*parts), [], None)
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []
    finally:
        answer_exit._TURN.reset(token)


@pytest.mark.parametrize("parts", _PARTS)
def test_an_open_turn_records_each_place_read(parts):
    token = answer_exit._TURN.set(None)
    try:
        turn = answer_exit.open_turn()
        answer_exit.note_evidence(_excerpts(*parts), [], turn)
        got = [(p.doc_id, p.chunk) for p in turn.places]
        assert got == [(doc, idx) for doc, idx, _name, _text in parts]
    finally:
        answer_exit._TURN.reset(token)
