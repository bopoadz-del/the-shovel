"""Retrieval must recognise a word it already knows in another form.

The term rescue matches each query term as a SUBSTRING of the chunk text, so
the query's word and the document's word have to share a prefix. Live 23 Sep
2026 on 589e637, the same content was rank 1 for a keyword query and absent
from the top 40 for the user's own question:

    "To what degree must structural backfill be compacted?"
      -> never retrieved the chunk reading "Compaction of the backfill to
         minimum 98% of maximum dry density of the modified proctor test"

compacted / compaction diverge after "compact", so no pair co-occurred and the
rescue stayed silent. Stemming the query term (never the chunk) closes it.
Synthetic corpus and synthetic figures throughout.
"""
import pytest

from app.core.rag import retriever
from app.core.rag.vector_store import Chunk

ASKED = "To what degree must structural backfill be compacted?"
DOC_TEXT = ("Earthworks clause 7.4: Compaction of the backfill to a minimum of "
            "SYN% of maximum dry density of the modified proctor test.")


# ── the stem ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("word,stem", [
    ("compacted", "compact"),
    ("compaction", "compact"),
    ("striking", "strik"),
    ("illumination", "illumin"),
    ("placement", "placement"),   # "plac" would fall below the floor
    ("requirements", "requir"),
    ("minimum", "minimum"),      # no suffix to strip
    ("work", "work"),            # too short to strip: never becomes "wor"
    ("backfill", "backfill"),
])
def test_a_term_reduces_to_the_prefix_of_its_word_family(word, stem):
    assert retriever.stem_query_term(word) == stem


def test_one_suffix_at_most():
    # "applications" loses "ations" and stops there -- the stem is not stripped
    # again down to "app", which would co-occur with half the corpus.
    assert retriever.stem_query_term("applications") == "applic"
    # "processes" loses "es" and stops: stripping again would give "proces",
    # which is not a prefix of "processes" and so matches nothing at all.
    assert retriever.stem_query_term("processes") == "process"


def test_the_stem_never_drops_below_the_floor():
    for word in ("cases", "rings", "tied", "used"):
        assert retriever.stem_query_term(word) == word


# ── the pairs the rescue searches with ─────────────────────────────────────

def test_the_query_pairs_match_the_documents_own_wording():
    terms = retriever.distinctive_query_terms(ASKED)
    pairs = retriever.cooccurrence_pair_phrases(terms)
    lowered = DOC_TEXT.lower()
    matching = [p for p in pairs if all(tok in lowered for tok in p.split())]
    assert matching, f"no pair of {pairs} co-occurs in the chunk"


def test_pairs_are_deduplicated_by_stem():
    # "compacted" and "compaction" in one query are one term after stemming.
    pairs = retriever.cooccurrence_pair_phrases(["compacted", "compaction", "backfill"])
    assert pairs == ["compact backfill"]


# ── end to end through retrieve_with_filter ────────────────────────────────

def _install(monkeypatch, semantic, lexical_corpus):
    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search",
                        lambda self, project_id, qvec, k, query_text=None:
                        [c for c in semantic if c.project_id == project_id][:k])

    def fake_identifier_search(self, project_id, identifiers, k=20):
        out = []
        for chunk in lexical_corpus:
            if chunk.project_id != project_id:
                continue
            low = chunk.text.lower()
            if any(all(tok in low for tok in ident.split()) for ident in identifiers):
                out.append(chunk)
        return out[:k]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.identifier_search",
                        fake_identifier_search)
    monkeypatch.setattr(retriever, "_doc_name_for_id", lambda did: "synthetic spec.pdf",
                        raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")


def _chunk(cid, text, score):
    return Chunk(chunk_id=cid, project_id="p1", doc_id=f"d{cid}", chunk_index=0,
                 text=text, score=score)


def test_the_answers_chunk_is_recovered_though_the_wording_differs(monkeypatch):
    unrelated = [_chunk("u1", "Cable ladder schedule for the substation.", 0.81),
                 _chunk("u2", "Road alignment setting out data.", 0.80)]
    target = _chunk("t1", DOC_TEXT, 0.10)
    _install(monkeypatch, semantic=unrelated, lexical_corpus=[target])

    chunks, _noise = retriever.retrieve_with_filter(ASKED, "p1", k=5)
    assert any(c.chunk_id == "t1" for c in chunks), (
        "the chunk the question is about was not recovered: "
        f"{[c.chunk_id for c in chunks]}")


def test_a_healthy_retrieval_is_left_alone(monkeypatch):
    # The chunk is already in the semantic top-k: the rescue must be a no-op,
    # scores included, or it perturbs every ordinary query.
    target = _chunk("t1", DOC_TEXT, 0.91)
    _install(monkeypatch, semantic=[target], lexical_corpus=[target])
    chunks, _noise = retriever.retrieve_with_filter(ASKED, "p1", k=5)
    assert [c.chunk_id for c in chunks] == ["t1"]
    assert chunks[0].score == pytest.approx(0.91, abs=1e-6), "score was perturbed"
