"""Revision currency (audit §5.2) — ALWAYS ON, independent of RAG_LAYERED.

Two behaviours, both keyed on the uploader's filename (resolved per chunk):
  1. a SUPERSEDED document is firmly down-ranked so a current document of
     comparable relevance outranks it — but survives as a last resort;
  2. among retrieved chunks that share a DRAWING NUMBER, a lower revision is
     suppressed when a strictly-higher, same-kind revision is also retrieved.

The retriever tests reuse the fixture shape of test_layered_retrieval_rerank:
a fake VectorStore.search returning controlled chunks, with ``_doc_name_for_id``
monkeypatched to map each doc id to a filename. RAG_LAYERED is left UNSET
throughout — revision currency must hold with layered RAG off.
"""
import pytest

from app.core.rag import revision as rev


# ── pure helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("A-101_superseded.pdf", True),
    ("old drawing.pdf", True),
    ("plan_OBSOLETE.pdf", True),
    ("archive/A-101.pdf", True),
    ("A-101_Rev C.pdf", False),
    ("current_spec.pdf", False),
    ("", False),
    (None, False),
])
def test_is_superseded(name, expected):
    assert rev.is_superseded(name) is expected


@pytest.mark.parametrize("name,expected", [
    ("A-101_RevC.pdf", "A-101"),
    ("STR2100 rev 2.pdf", "STR2100"),
    ("M_401_current.pdf", "M_401"),
    ("meeting notes.pdf", ""),      # no sheet code
    ("", ""),
])
def test_drawing_number(name, expected):
    assert rev.drawing_number(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("A-101_RevC.pdf", "C"),
    ("A-101_Rev2.pdf", "2"),
    ("A-101_R2.pdf", "2"),
    ("A-101-Rev10.pdf", "10"),        # multi-digit (the §5.2 follow-up)
    ("A-101-Rev9.pdf", "9"),
    ("A-101-RevAA.pdf", "AA"),        # multi-letter
    ("STR2100 rev 2.pdf", "2"),       # sheet-code digits are NOT the revision
    ("STR2100.pdf", ""),              # no revision marker -> no token
    ("plan.pdf", ""),                 # no revision token
    ("", ""),
])
def test_revision_token(name, expected):
    assert rev.revision_token(name) == expected


def test_revision_rank_orders_within_kind_only():
    # alphabetic order
    assert rev.revision_rank("A") < rev.revision_rank("C")
    # numeric order
    assert rev.revision_rank("0") < rev.revision_rank("2")
    # cross-kind: different buckets (kind differs), never ordered as supersede
    assert rev.revision_rank("2")[0] != rev.revision_rank("B")[0]
    # unparseable -> None
    assert rev.revision_rank("") is None


def test_multi_digit_revision_ordering_the_hazard():
    """The bug this fixes: Rev 10 is the CURRENT sheet and must outrank Rev 9.
    The old single-char parser read '10' as '1', ranking it below '9' and
    letting a stale revision suppress the current one."""
    assert rev.revision_rank("10") > rev.revision_rank("9")
    assert rev.revision_rank("11") > rev.revision_rank("2")


def test_multi_letter_revision_supersedes_z():
    # Standard drawing sequence: A < ... < Z < AA < AB.
    assert rev.revision_rank("AA") > rev.revision_rank("Z")
    assert rev.revision_rank("AB") > rev.revision_rank("AA")
    # cross-kind still never ordered
    assert rev.revision_rank("10")[0] != rev.revision_rank("AA")[0]


def test_end_to_end_rev10_beats_rev9_by_filename():
    """Full path: two chunks of the same sheet, filenames only."""
    r10 = rev.revision_rank(rev.revision_token("A-101-Rev10.pdf"))
    r9 = rev.revision_rank(rev.revision_token("A-101-Rev9.pdf"))
    assert r10 is not None and r9 is not None
    assert r10 > r9  # current (Rev10) preferred, not suppressed


# ── retriever integration ────────────────────────────────────────────────────

ACTIVE = "p_active"


def _chunk(cid, score):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=cid, project_id=ACTIVE, doc_id=f"doc-{cid}",
                 chunk_index=0, text="filler", score=score)


def _install(monkeypatch, chunks, names):
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in chunks if c.project_id == project_id]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None)
    monkeypatch.setattr(ret, "_doc_name_for_id",
                        lambda _id: names.get(_id, ""), raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    for v in ("RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
              "RAG_GK_LEXICAL_FOLD", "RAG_AUTHORITY_WEIGHT", "RAG_LAYER_WEIGHT",
              "RAG_LAYERED"):
        monkeypatch.delenv(v, raising=False)
    return ret


def test_superseded_demoted_below_current_at_equal_cosine(monkeypatch):
    ret = _install(
        monkeypatch,
        [_chunk("sup", 0.80), _chunk("cur", 0.80)],
        {"doc-sup": "spec_superseded.pdf", "doc-cur": "spec_current.pdf"},
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    ids = [c.chunk_id for c in chunks]
    assert ids[0] == "cur", "current doc must outrank the superseded one"
    sup = next(c for c in chunks if c.chunk_id == "sup")
    assert sup.superseded is True


def test_superseded_survives_as_last_resort(monkeypatch):
    # When the ONLY match is superseded it is still returned (a flagged stale
    # answer beats no answer) — the penalty demotes, it does not drop.
    ret = _install(monkeypatch, [_chunk("sup", 0.80)],
                   {"doc-sup": "A-101_superseded.pdf"})
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert [c.chunk_id for c in chunks] == ["sup"]
    assert chunks[0].superseded is True


def test_lower_revision_of_same_drawing_is_suppressed(monkeypatch):
    ret = _install(
        monkeypatch,
        [_chunk("revA", 0.80), _chunk("revC", 0.80)],
        {"doc-revA": "A-101_RevA.pdf", "doc-revC": "A-101_RevC.pdf"},
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    ids = [c.chunk_id for c in chunks]
    assert ids == ["revC"], "only the highest revision of A-101 should survive"


def test_different_drawings_both_kept(monkeypatch):
    ret = _install(
        monkeypatch,
        [_chunk("a", 0.80), _chunk("b", 0.80)],
        {"doc-a": "A-101_RevA.pdf", "doc-b": "A-102_RevA.pdf"},
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert {c.chunk_id for c in chunks} == {"a", "b"}


def test_incomparable_revisions_both_kept(monkeypatch):
    # A letter rev and a numeric rev of the same sheet are NOT ordered against
    # each other — keep both rather than risk hiding the current one.
    ret = _install(
        monkeypatch,
        [_chunk("alpha", 0.80), _chunk("num", 0.80)],
        {"doc-alpha": "A-101_RevB.pdf", "doc-num": "A-101_Rev2.pdf"},
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert {c.chunk_id for c in chunks} == {"alpha", "num"}


def test_revision_currency_holds_with_layered_off(monkeypatch):
    # Explicit: no RAG_LAYERED in env, yet suppression + demotion both act.
    ret = _install(
        monkeypatch,
        [_chunk("revA", 0.80), _chunk("revC", 0.80), _chunk("sup", 0.80)],
        {"doc-revA": "A-101_RevA.pdf", "doc-revC": "A-101_RevC.pdf",
         "doc-sup": "notes_old.pdf"},
    )
    import os
    assert "RAG_LAYERED" not in os.environ
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    ids = [c.chunk_id for c in chunks]
    assert "revA" not in ids               # lower rev suppressed
    assert ids[0] == "revC"                # current rev leads
    assert "sup" in ids and ids[-1] == "sup"  # superseded kept but last


# ── disclosure ───────────────────────────────────────────────────────────────

def test_context_marker_exposes_revision():
    from app.core.rag.inject import format_chunks_as_system_message
    from app.core.rag.vector_store import Chunk
    c = Chunk(chunk_id="x", project_id=ACTIVE, doc_id="d", chunk_index=0,
              text="Slab thickness 250mm", score=0.9)
    c.revision = "C"
    msg = format_chunks_as_system_message([c], total_candidates=1)
    assert "rev=C" in msg["content"]


def test_superseded_disclosure_note_in_context():
    from app.core.rag.inject import format_chunks_as_system_message
    from app.core.rag.vector_store import Chunk
    c = Chunk(chunk_id="x", project_id=ACTIVE, doc_id="d", chunk_index=0,
              text="old detail", score=0.9)
    c.superseded = True
    msg = format_chunks_as_system_message([c], total_candidates=1)
    assert "SUPERSEDED" in msg["content"]
    assert "confirm" in msg["content"].lower()
