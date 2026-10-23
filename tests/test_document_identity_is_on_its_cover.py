"""Who prepared a document, its number and its revision are on its cover.

Live on 24d1c0c, master_corpus, 0/3:

    "What is the document number and revision of the priced Bill of
     Quantities, and who prepared it?"

The cover is indexed, in a file whose NAME is the title the question uses:

    IP-...-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf  [chunk 3]
    Document no: ... 000007  Revision no: B  Prepared by: <firm>  Date: ...

Searched by its document number, that chunk ranks first at 2.07. Asked the
way a person asks, the top five are contract templates describing HOW a bill
of quantities should be identified. Nothing in the question resembles a cover
sheet: a control block is labels and codes, and "document number", "revision"
and "prepared" are in every template in the corpus.

The filename rescue that should have caught it keeps the five most
"distinctive" words of the question -- capitalised first -- and "priced", the
one word that tells this bill from the unpriced one, came sixth.

These tests pin the shape: an ask about a document's IDENTITY is answered by
the control block of the document the question names.
"""
from __future__ import annotations

import pytest

from app.core.rag.vector_store import Chunk

ACTIVE = "p_master"
PRICED = "boqpriced"
UNPRICED = "boqplain"
TEMPLATE = "tmpl"

PRICED_NAME = "XX-INF-000-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf"
UNPRICED_NAME = "XX-INF-000-BOQ-CA-000008-B_Bill of Quantities.pdf"

COVER_PRICED = (
    "Example Infrastructure Client reference: XX-2099-001 Project no: EX335 "
    "Document no: XX-INF-000-BOQ-CA- 000007 Project director: A. Example "
    "Revision no: B Prepared by: EXAMPLEQS Date: June 2, 2099 "
    "File name: XX-INF-000-BOQ-CA- 000007 Doc status: For Client's review"
)
COVER_UNPRICED = COVER_PRICED.replace("000007", "000008").replace(
    "EXAMPLEQS", "OTHERFIRM")
REVISION_TABLE = (
    "Revision Date Description Author Checked Reviewed Approved "
    "A 12/05/2099 Bill of Quantities (Priced) EQS EQS AB PK "
    "B 02/06/2099 Bill of Quantities (Priced) EQS EQS AB PK"
)
BOQ_ITEMS = (
    "D110 General site clearance ha 158 2,500.00 395,000.00 "
    "D290.1 Removal of trees Nr 48 220.00 10,560.00"
)
TEMPLATE_CLAUSE = (
    "Where a Bill of Quantities is annexed to the Contract it shall be "
    "identified by its author, title, date, document number and revision "
    "number, and the person who prepared it shall be named."
)
PROGRAMME_CLAUSE = (
    "Each revision of the Programme shall state the revision number, the "
    "date prepared and the document number of the programme it supersedes."
)

B6 = ("What is the document number and revision of the priced Bill of "
      "Quantities, and who prepared it?")


def _chunk(cid, doc_id, score, text, index=0):
    return Chunk(chunk_id=cid, project_id=ACTIVE, doc_id=doc_id,
                 chunk_index=index, text=text, score=score)


@pytest.fixture
def ret(monkeypatch):
    """Live shape: cosine returns the templates; the covers are never pooled."""
    from app.core.rag import retriever

    semantic = [
        _chunk("t1", TEMPLATE, 0.73, TEMPLATE_CLAUSE, 72),
        _chunk("t2", TEMPLATE, 0.72, PROGRAMME_CLAUSE, 25),
    ]
    docs = {
        PRICED: [_chunk("pc", PRICED, 0.0, COVER_PRICED, 3),
                 _chunk("pr", PRICED, 0.0, REVISION_TABLE, 4),
                 _chunk("pi", PRICED, 0.0, BOQ_ITEMS, 40)],
        UNPRICED: [_chunk("uc", UNPRICED, 0.0, COVER_UNPRICED, 3)],
    }
    names = {PRICED: PRICED_NAME, UNPRICED: UNPRICED_NAME,
             TEMPLATE: "QRS-637_CONTRACT TEMPLATE.docx"}
    seen = {}

    def fake_filename_terms(pid, terms, *, min_terms=2, require_letter=False,
                            limit=8, require_all=False):
        seen["terms"], seen["require_all"] = list(terms), require_all
        out = []
        for did in (UNPRICED, PRICED):  # the wrong bill is listed FIRST
            blob = names[did].lower()
            hits = sum(1 for t in terms if t.lower() in blob)
            if (hits == len(terms)) if require_all else (hits >= min_terms):
                out.append({"id": did, "original_name": names[did], "file_path": ""})
        return out[:limit]

    def fake_chunks_for_docs(self, pid, doc_ids, k_per_doc=12, **_kw):
        return [c for d in doc_ids for c in docs.get(d, [])][: k_per_doc * len(doc_ids)]

    vs = "app.core.rag.vector_store.VectorStore."
    monkeypatch.setattr(
        vs + "search", lambda self, pid, qvec, k, query_text=None: semantic[:k])
    monkeypatch.setattr(vs + "identifier_search",
                        lambda self, pid, identifiers, k=20: [])
    monkeypatch.setattr(vs + "chunks_for_docs", fake_chunks_for_docs)
    monkeypatch.setattr(vs + "chunks_containing_all",
                        lambda self, pid, needles, k=20, **_kw: [])
    monkeypatch.setattr(vs + "count", lambda self, pid=None: 11)
    monkeypatch.setattr(vs + "_verify_embedding_identity", lambda self: None)
    monkeypatch.setattr(retriever, "_doc_name_for_id",
                        lambda did: names.get(did, ""), raising=False)
    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase",
                        lambda pid, phrase, limit=8: [])
    monkeypatch.setattr("app.core.projects.documents_matching_filename_terms",
                        fake_filename_terms)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    retriever._seen_filename_lookup = seen
    return retriever


def _texts(ret, question):
    chunks, _ = ret.retrieve_with_filter(question, ACTIVE, k=5)
    return [c.text for c in chunks]


def test_b6_the_cover_of_the_named_document_leads(ret):
    texts = _texts(ret, B6)
    assert texts and "Prepared by: EXAMPLEQS" in texts[0]
    assert "000007" in texts[0] and "Revision no: B" in texts[0]


def test_the_word_that_tells_two_documents_apart_is_kept(ret):
    """The unpriced bill shares every other word of the title, is listed
    first, and was prepared by someone else."""
    blob = "\n".join(_texts(ret, B6))
    assert "OTHERFIRM" not in blob
    assert "priced" in ret._seen_filename_lookup["terms"]
    assert ret._seen_filename_lookup["require_all"] is True


def test_the_words_that_ask_are_not_the_words_that_name(ret):
    """"document", "number", "revision", "prepared" describe the ask. No file
    is called that, so searching filenames for them finds nothing."""
    _texts(ret, B6)
    assert not {"document", "number", "revision", "prepared"} & set(
        ret._seen_filename_lookup["terms"])


def test_the_body_of_the_document_is_not_dragged_in(ret):
    assert not any("General site clearance" in t for t in _texts(ret, B6))


@pytest.mark.parametrize(
    "question",
    [
        "Who prepared the priced Bill of Quantities?",
        "What revision is the priced Bill of Quantities?",
        "What is the document no. of the priced Bill of Quantities?",
        "Who checked and approved the priced Bill of Quantities?",
    ],
)
def test_every_way_of_asking_about_identity_reaches_the_cover(ret, question):
    assert any("Document no:" in t or "Author Checked" in t
               for t in _texts(ret, question)), question


@pytest.mark.parametrize(
    "question",
    [
        # About the CONTENT of the same document.
        "What is the rate for general site clearance in the priced Bill of Quantities?",
        # Owned by the specification-title path.
        "Which specification document covers the Variation Procedure, and what is its number?",
        "What is the value of the Performance Bond?",
    ],
)
def test_a_question_about_something_else_never_starts_this_lookup(ret, question):
    """Asserted on the lookup itself: the older filename rescue may still
    bring a cover along on its own account, and that is not this change."""
    assert not ret.query_asks_for_document_identity(question)
    _texts(ret, question)
    assert ret._seen_filename_lookup.get("require_all") is not True


def test_identity_words_with_no_document_named_fetch_nothing(ret):
    """"What revision number must each Programme state?" -- identity
    vocabulary, but the words left over name no file."""
    texts = _texts(ret, "What revision number must each Programme state?")
    assert not any("Prepared by:" in t for t in texts)


def test_the_richest_control_block_wins_not_the_first_one(ret, monkeypatch):
    """Live 39d6b8d: number, revision and date came back -- and "the excerpt
    does not name the party that prepared it". A cover has a title block
    (number, revision) BEFORE the full control block (number, revision,
    prepared by, status, file name). Page order picked the thin one."""
    title_block = (
        "Example City 4000 Bill of Quantities (Priced) Document no: "
        "XX-INF-000-BOQ-CA-000007 Revision no: B Example Client Limited"
    )
    contents = "Contents Revision no: see cover. Document no: see cover. Page 2"
    pages = [
        _chunk("p0", PRICED, 0.0, title_block, 1),
        _chunk("p1", PRICED, 0.0, contents, 2),
        _chunk("pc", PRICED, 0.0, COVER_PRICED, 3),
    ]
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, pid, doc_ids, k_per_doc=12, **_kw: (
            pages if PRICED in doc_ids else []),
    )
    chunks, _ = ret.retrieve_with_filter(B6, ACTIVE, k=5)
    lifted = [c for c in chunks if (c.score or 0.0) >= ret._DOC_IDENTITY_BONUS]
    assert any("Prepared by: EXAMPLEQS" in c.text for c in lifted)


# ── the title is the phrase after "of the", not every other word ──────────
#
# Unseen Set 3, live: "What is the date of the priced Bill of Quantities and
# the Employer's contract reference on it?" Two misses. "date of" was not an
# identity ask at all. And the title was taken as every word that is not an
# ask-word -- here that adds "employer" and "contract", every one of which the
# filename then has to contain, and it contains neither.

C1 = ("What is the date of the priced Bill of Quantities and the Employer's "
      "contract reference on it?")


def test_a_date_question_is_an_identity_question(ret):
    assert ret.query_asks_for_document_identity(C1)
    assert ret.query_asks_for_document_identity("When was the priced Bill of Quantities issued?")


def test_the_title_is_the_phrase_the_question_points_at(ret):
    assert ret.document_identity_title_terms(C1) == ["bill", "priced", "quantities"]
    # The Set 1 wording still yields the same title.
    assert ret.document_identity_title_terms(B6) == ["bill", "priced", "quantities"]


def test_c1_reaches_the_cover(ret):
    texts = _texts(ret, C1)
    assert texts and "Date: June 2, 2099" in texts[0]
    assert "Client reference: XX-2099-001" in texts[0]


def test_a_question_with_no_of_the_phrase_falls_back_to_the_old_rule(ret):
    """"Who prepared the priced Bill of Quantities?" has no "of the"."""
    assert ret.document_identity_title_terms(
        "Who prepared the priced Bill of Quantities?") == ["bill", "priced", "quantities"]


def test_one_leftover_word_does_not_name_a_document(ret):
    """"Who prepared the quantities?" leaves one word. Every bill in the
    corpus has it in its name; that is a topic, not a title."""
    ret._seen_filename_lookup.clear()
    _texts(ret, "Who prepared the quantities?")
    assert ret._seen_filename_lookup.get("require_all") is not True


def test_only_the_control_block_gets_the_lift(ret, monkeypatch):
    """The body of the named document may still arrive by other routes, but
    it must not be LIFTED as if it answered an identity question -- even when
    it sits before the cover in the file."""
    body_first = [
        _chunk("pi0", PRICED, 0.0, BOQ_ITEMS, 0),
        _chunk("pc", PRICED, 0.0, COVER_PRICED, 3),
    ]
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, pid, doc_ids, k_per_doc=12, **_kw: (
            body_first if PRICED in doc_ids else []),
    )
    chunks, _ = ret.retrieve_with_filter(B6, ACTIVE, k=5)
    assert "Prepared by: EXAMPLEQS" in chunks[0].text
    for c in chunks:
        if "General site clearance" in c.text:
            assert (c.score or 0.0) < ret._DOC_IDENTITY_BONUS, c.score


def test_a_cover_is_recognised_by_its_labels_not_its_position(ret):
    from tests.test_document_identity_is_on_its_cover import (
        BOQ_ITEMS, COVER_PRICED, REVISION_TABLE, TEMPLATE_CLAUSE,
    )
    assert ret.chunk_states_document_control_block(COVER_PRICED)
    assert ret.chunk_states_document_control_block(REVISION_TABLE)
    assert not ret.chunk_states_document_control_block(BOQ_ITEMS)
    # Prose ABOUT identifying a document is not a control block.
    assert not ret.chunk_states_document_control_block(TEMPLATE_CLAUSE)


@pytest.fixture
def project_store(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod

    importlib.reload(db_mod)
    from app.core import projects as projects_mod

    pm = importlib.reload(projects_mod)
    pm._initialized = False
    pm.init_db()
    return pm


def test_the_lookup_finds_one_file_behind_a_hundred_near_misses(project_store):
    """Real SQL. The OR-match is cut to 80 rows BEFORE it is ranked; on a
    corpus where a hundred names contain "bill", the one file holding every
    word can be cut away. ``require_all`` puts the test in the query."""
    pid = project_store.create_project("identity corpus")["id"]
    for i in range(100):
        project_store.add_document(pid, f"Bill of Quantities - Part {i:03d}.pdf", size=1)
    priced = project_store.add_document(pid, PRICED_NAME, size=1)
    terms = ["bill", "priced", "quantities"]

    found = project_store.documents_matching_filename_terms(
        pid, terms, require_all=True, limit=3,
    )
    assert [d["id"] for d in found] == [priced["id"]]

    # And the behaviour being replaced, so this test means something:
    loose = project_store.documents_matching_filename_terms(pid, terms, min_terms=3)
    assert priced["id"] not in [d["id"] for d in loose], (
        "the OR-match found it this time; the 80-row cut is no longer the "
        "hazard this test was written against"
    )


def test_require_all_still_skips_a_retired_document(project_store):
    pid = project_store.create_project("identity corpus 2")["id"]
    old = project_store.add_document(pid, PRICED_NAME, size=1)
    new = project_store.add_document(pid, PRICED_NAME.replace("-B_", "-C_"), size=1)
    project_store.supersede_document(old["id"], new["id"])

    found = project_store.documents_matching_filename_terms(
        pid, ["bill", "priced", "quantities"], require_all=True,
    )
    assert [d["id"] for d in found] == [new["id"]]
