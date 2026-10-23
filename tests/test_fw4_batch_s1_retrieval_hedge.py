"""S1 through retrieval and the first-line hedge together.

Agent E pools the specification clause that defers cover, and the drawing
that states it. The rebuilt hedge verifies the model's first line. It
does not invent both cover figures from a narrative that commits to
neither, and it does not ask which document when the subjects differ.
A body that commits to 75 mm is annotated with that drawing, not the
specification.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.agents.first_line_hard_rule import apply_first_line_hard_rule

PID = "FIXTURE-e-20260927-project"
FIXTURE = Path(__file__).parent / "fixtures" / "fw4_spec_deferral_chunks.json"
SPEC_DOC = "FIXTURE-e-20260927-spec-vol-a"
FOOT_DOC = "FIXTURE-e-20260927-dwg-footing-cover"
SPEC_NAME = "FIXTURE-e-20260927 Specification Volume A.pdf"
SOIL_NAME = "FIXTURE-e-20260927 DWG soil contact note A.pdf"
S1_ASK = (
    "Per the project specification, what is the minimum concrete cover for "
    "foundations cast directly against soil?"
)
NARRATIVE = "I will answer from the retrieved context."


def _load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["docs"]


def _corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_K", raising=False)
    monkeypatch.delenv("MAX_RAG_TOKENS", raising=False)
    monkeypatch.delenv("FIRST_LINE_HARD_RULE", raising=False)
    from app.core.rag import embeddings as emb
    from app.core.rag import vector_store as vs

    emb.reset_embedder_cache()
    vs.reset_store_cache()
    from app.core.rag import retriever as ret
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    embedder = get_embedder()
    store = get_store(dim=embedder.dim)
    names = {}
    for doc in _load():
        texts = [c["text"] for c in doc["chunks"]]
        store.upsert_chunks(PID, doc["doc_id"], texts, embedder.encode(texts))
        names[doc["doc_id"]] = doc["name"]
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda d: names.get(d, ""))
    return emb, vs, names


def test_s1_fixture_states_both_cover_figures_from_drawings(tmp_path, monkeypatch):
    """Retrieval still pools the deferring spec and the footing drawing.

    An uncommitted narrative is left alone: different cover subjects do
    not produce "which document's figure is meant?". A body that commits
    to 75 mm is annotated with that figure and a drawing, and the
    specification volume is not the credited source. 100 mm stays off
    the first line.
    """
    from app.core.rag.inject import rag_inject

    emb, vs, _names = _corpus(tmp_path, monkeypatch)
    try:
        _assert_s1(rag_inject)
    finally:
        emb.reset_embedder_cache()
        vs.reset_store_cache()


def _assert_s1(rag_inject):
    msg, audit = rag_inject(
        user_message=S1_ASK,
        project_id=PID,
        conversation_id="ws-FIXTURE-e-20260927",
        user_id="fixture",
        agent_name="project-assistant",
        history=[],
    )
    injected = [c["doc_id"] for c in audit.get("chunks") or []]
    assert SPEC_DOC in injected, injected
    assert FOOT_DOC in injected, injected
    msgs = [{"role": "user", "content": S1_ASK}]
    untouched = apply_first_line_hard_rule(NARRATIVE, msg, msgs)
    assert untouched == NARRATIVE, untouched
    assert "which document" not in untouched.lower(), untouched

    committed = (
        "I will answer from the retrieved context.\n\n"
        "Nominal concrete cover is 75 mm for concrete cast against soil "
        f"in {SOIL_NAME}.\n"
        "The bottom of footings is 100 mm.\n"
    )
    out = apply_first_line_hard_rule(committed, msg, msgs)
    assert "which document" not in out.lower(), out
    first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    assert re.search(r"(?i)\b75\s*mm\b", first), first
    assert not re.search(r"(?i)\b100\s*mm\b", first), first
    assert "DWG" in first, first
    assert SPEC_NAME not in first, first
    assert "3.1.25.8" not in first, first
