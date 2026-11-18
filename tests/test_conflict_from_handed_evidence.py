"""A conflict between two documents the turn was HANDED is reported, even
when the answer made no tool call and cited only one of them.

The rule: conflict detection reads every passage the turn put in front of
the model (pre-injected excerpts and tool passages), not only the Sources
rows. A Sources row built from a pre-injected chunk carries the document and
page but not the passage, and Sources lists only what the answer cited -- so
a check that reads Sources alone never sees the second document's value.

Every document, value and question here is invented for this file; the
expected values are written by hand. Each decision is shown on two inputs
or more.
"""

from __future__ import annotations

import pytest

from app.agents import answer_exit
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk


def _chunk(doc_id: str, idx: int, name: str, text: str, page: int) -> Chunk:
    c = Chunk(chunk_id=f"{doc_id}:{idx}", project_id="proj-invented", doc_id=doc_id,
              chunk_index=idx, text=text, score=0.8)
    c.source_name = name
    c.page = page
    return c


def _sources_row(doc_id: str, name: str, page: int) -> dict:
    """A Sources row as the pre-injected path builds it: no passage."""
    return {"doc_id": doc_id, "doc_name": name, "page_or_section": f"p. {page}", "page": page,
            "score": 0.8, "confidence": "High", "layer": "own", "layer_label": "Project"}


# (question, drawing note, method statement, answer, the value each states)
_CASES = [
    pytest.param(
        "What grade is the steel for the gantry columns?",
        ("dwg-1", "GN-410 Gantry General Arrangement.pdf", 12,
         "General notes. 4. Gantry columns and rafters shall be grade S275 hot-rolled sections."),
        ("ms-1", "Method Statement for Gantry Erection.pdf", 3,
         "Materials. The gantry columns are fabricated from grade S355 steel delivered to site."),
        "The gantry columns are grade S275 steel [source: GN-410 Gantry General Arrangement.pdf].",
        ("S275", "S355"),
        id="steel-grade",
    ),
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        ("dwg-2", "PC-220 Pile Cap Details.pdf", 7,
         "Note 6. Minimum concrete cover to pile cap reinforcement is 75 mm."),
        ("ms-2", "Method Statement Pile Caps.pdf", 5,
         "Fixing. Spacers give a cover to pile cap reinforcement of 50 mm throughout."),
        "The cover to the pile cap reinforcement is 75 mm [source: PC-220 Pile Cap Details.pdf].",
        ("75 mm", "50 mm"),
        id="cover",
    ),
]


def _run_turn(question, drawing, method, answer):
    token = answer_exit._TURN.set(None)
    try:
        turn = answer_exit.open_turn()
        turn.question = question
        handed = [_chunk(drawing[0], 0, drawing[1], drawing[3], drawing[2]),
                  _chunk(method[0], 0, method[1], method[3], method[2])]
        rag_sys_msg = format_chunks_as_system_message(handed, total_candidates=2, query=question)
        # No tool call: the only messages are the system excerpt and the user.
        answer_exit.note_evidence(rag_sys_msg, [rag_sys_msg, {"role": "user", "content": question}], turn)
        event = {"type": "end", "content": answer,
                 "sources": [_sources_row(drawing[0], drawing[1], drawing[2])]}
        return answer_exit.check_end_event(event, turn=turn)
    finally:
        answer_exit._TURN.reset(token)


@pytest.mark.parametrize("question, drawing, method, answer, values", _CASES)
def test_handed_documents_that_differ_are_reported(question, drawing, method, answer, values):
    out = _run_turn(question, drawing, method, answer)
    conflicts = out.get("conflicts") or []
    assert len(conflicts) == 1, out.get("content")
    got = {(v["value"], v["document"]) for v in conflicts[0]["values"]}
    assert got == {(values[0], drawing[1]), (values[1], method[1])}
    text = out["content"]
    assert "differ" in text
    for value in values:
        assert value in text
    # Each value carries a marker whose citation names its own document.
    by_marker = {c["marker"]: c["document"] for c in out.get("citations") or []}
    for v in conflicts[0]["values"]:
        assert by_marker[v["marker"]] == v["document"]


@pytest.mark.parametrize("question, drawing, method, answer, values", _CASES)
def test_handed_documents_that_agree_are_not_a_conflict(question, drawing, method, answer, values):
    agreeing = (method[0], method[1], method[2], method[3].replace(values[1], values[0]))
    out = _run_turn(question, drawing, agreeing, answer)
    assert not out.get("conflicts")
    assert "differ" not in out["content"]


@pytest.mark.parametrize("question, drawing, method, answer, values", _CASES)
def test_a_value_the_answer_did_not_state_is_not_a_conflict(question, drawing, method, answer, values):
    """The answer anchors it: two handed documents that differ on something
    the answer never stated raise nothing."""
    unrelated = "The erection sequence follows the approved programme [source: %s]." % drawing[1]
    out = _run_turn(question, drawing, method, unrelated)
    assert not out.get("conflicts")
