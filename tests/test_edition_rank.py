"""Edition-aware candidate demotion — general, no hardcoded standards/years.

Two behaviours (order 11:25Z, 11 Nov):

  1. when the ask names a standard AND an edition/version/year token, a
     candidate whose text or filename carries a DIFFERENT edition of the same
     standard ranks BELOW candidates that match it or carry no edition;
  2. a candidate with no edition token is unaffected, and an ask without an
     edition token triggers no demotion at all.

The standard and the edition are read from the ask at run time; nothing here
names FIDIC / a specific year / a clause number. Synthetic fixtures stand in
for them ("ACME 2017" vs "ACME 1999") so no real standard or edition enters
product code.
"""
import pytest

from app.core.rag import edition_rank as er

ACTIVE = "p_active"


# ── pure: extraction ────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,standard,edition", [
    ("Under ACME 2017, what is the notice period?", "ACME", "2017"),
    ("What does ACME 1999 require here?", "ACME", "1999"),
    ("Per ZZZ 51B (2019), how long is the watch?", "ZZZ 51B", "2019"),
    ("ISO 9001:2015 says what about audits?", "ISO 9001", "2015"),
    ("2017 ACME conditions apply?", "ACME", "2017"),   # edition before standard
])
def test_extract_edition_standards_finds_pair(query, standard, edition):
    pairs = er.extract_edition_standards(query)
    assert (standard, edition) in [(p.standard, p.edition) for p in pairs]


@pytest.mark.parametrize("query", [
    "What is the notice period?",                    # no standard, no edition
    "What does ACME require here?",                  # standard, no edition
    "How long is the watch?",                        # neither
    "The agreement was signed in 1999 and filed.",   # year, but no standard near it
])
def test_extract_edition_standards_returns_empty(query):
    assert er.extract_edition_standards(query) == []


# ── pure: conflict detection ────────────────────────────────────────────────

@pytest.mark.parametrize("text,source_name,standard,edition,expected", [
    ("ACME 1999 notice is 28 days.", "", "ACME", "2017", True),   # conflicting year
    ("ACME 2017 notice is 14 days.", "", "ACME", "2017", False),  # matching year
    ("ACME notice is 28 days.", "", "ACME", "2017", False),       # no edition -> unaffected
    ("Other org 1999 notice", "", "ACME", "2017", False),         # different standard + year
    ("", "ACME_1999_conditions.pdf", "ACME", "2017", True),       # edition in filename
    ("", "ACME_2017_conditions.pdf", "ACME", "2017", False),      # matching filename
    ("ISO 9001:2008 audits", "", "ISO 9001", "2015", True),       # code + conflicting year
    ("ISO 9001 audits", "", "ISO 9001", "2015", False),           # code, no year
    ("ISO 14001:2008 audits", "", "ISO 9001", "2015", False),     # different code + year
    ("Regional Standard 2019 rules", "", "Regional Standard", "2017", True),  # non-acronym standard
])
def test_candidate_conflicts(text, source_name, standard, edition, expected):
    assert er.candidate_conflicts(text, source_name, standard, edition) is expected


# ── retriever integration ───────────────────────────────────────────────────

def _chunk(cid, score, text):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=cid, project_id=ACTIVE, doc_id=f"doc-{cid}",
                 chunk_index=0, text=text, score=score)


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
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    for v in ("RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
              "RAG_GK_LEXICAL_FOLD", "RAG_AUTHORITY_WEIGHT", "RAG_LAYER_WEIGHT",
              "RAG_LAYERED"):
        monkeypatch.delenv(v, raising=False)
    return ret


@pytest.mark.parametrize("query,conflict_text,match_text,conflict_name,match_name", [
    ("Under ACME 2017, what is the notice period?",
     "ACME 1999 notice is 28 days.", "ACME 2017 notice is 14 days.",
     "ACME_1999_conditions.pdf", "ACME_2017_conditions.pdf"),
    ("ZZZ 51B 2019 watch duration?",
     "ZZZ 51B 2005 watch is 60 min.", "ZZZ 51B 2019 watch is 30 min.",
     "ZZZ_51B_2005.pdf", "ZZZ_51B_2019.pdf"),
])
def test_apply_edition_conflict_demotion_demotes_only_conflicting(
    query, conflict_text, match_text, conflict_name, match_name,
):
    # Direct unit: a conflicting-edition candidate is demoted, a matching one
    # is not, and a query with no edition pair leaves every candidate alone.
    conflict = _chunk("c", 0.80, conflict_text)
    match = _chunk("m", 0.80, match_text)
    scored = [(0.80, conflict), (0.80, match)]
    names = {"doc-c": conflict_name, "doc-m": match_name}
    er.apply_edition_conflict_demotion(query, scored, names)
    assert conflict.score == round(0.80 - er.EDITION_CONFLICT_PENALTY, 6)
    assert match.score == 0.80


@pytest.mark.parametrize("query", [
    "Under ACME 2017, what is the notice period?",
    "ACME 2017 edition notice period",
])
def test_conflicting_edition_demoted_below_matching(monkeypatch, query):
    ret = _install(
        monkeypatch,
        [
            _chunk("match", 0.80, "ACME 2017 notice is 14 days."),
            _chunk("old", 0.80, "ACME 1999 notice is 28 days."),
        ],
        {"doc-match": "conditions_current.pdf", "doc-old": "conditions_1999.pdf"},
    )
    chunks, _ = ret.retrieve_with_filter(query, ACTIVE, k=5)
    ids = [c.chunk_id for c in chunks]
    assert ids[0] == "match", "matching edition must outrank the conflicting one"
    old = next(c for c in chunks if c.chunk_id == "old")
    assert old.score < 0.80 - (er.EDITION_CONFLICT_PENALTY / 2), "conflicting candidate demoted"


@pytest.mark.parametrize("query", [
    "What does ACME require for the notice period?",
    "ACME notice period",
])
def test_no_edition_in_ask_leaves_conflicting_candidate_in_place(monkeypatch, query):
    # Same two sources; without an edition token in the ask, neither is demoted,
    # so the original order (equal cosine -> stable) is preserved.
    ret = _install(
        monkeypatch,
        [
            _chunk("a", 0.80, "ACME 2017 notice is 14 days."),
            _chunk("b", 0.80, "ACME 1999 notice is 28 days."),
        ],
        {"doc-a": "conditions_2017.pdf", "doc-b": "conditions_1999.pdf"},
    )
    chunks, _ = ret.retrieve_with_filter(query, ACTIVE, k=5)
    ids = [c.chunk_id for c in chunks]
    assert ids == ["a", "b"], "no edition in ask -> stable order, no demotion"


@pytest.mark.parametrize("text,name", [
    ("ACME 1999 notice is 28 days.", "ACME_1999_conditions.pdf"),  # conflicting
    ("ACME 2017 notice is 14 days.", "ACME_2017_conditions.pdf"),  # matching
])
def test_conflicting_candidate_survives_as_last_resort(monkeypatch, text, name):
    # A conflicting edition is demoted, not dropped: when it is the only match
    # it is still returned (a flagged stale answer beats no answer).
    ret = _install(
        monkeypatch,
        [_chunk("only", 0.80, text)],
        {"doc-only": name},
    )
    chunks, _ = ret.retrieve_with_filter("Under ACME 2017, what is the notice period?", ACTIVE, k=5)
    assert [c.chunk_id for c in chunks] == ["only"]
