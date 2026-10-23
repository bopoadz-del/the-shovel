"""Named-document recall: the document a question names reaches the top-k.

General mechanisms in the retriever:

* ``recall_titled_documents`` -- "which <kind of document> covers <Title
  Phrase>, and what is its number?": documents whose upload name carries the
  title phrase, and register lines that print a reference code right before
  the phrase, are pooled and lifted over long volumes that only share the
  topic. A section heading that shares the topic is not a register line.
* ``recall_issue_stamps`` -- "on what date was <named document> issued, and
  under which <label> number?": the stamp that prints a labelled date and a
  "<Label> No. <code>" and repeats the title is pooled, and when two
  contracts' stamps match, the year-lock election keeps the later one.

They replace paths keyed on one document code prefix, one document kind and
one reference label, with title words photographed from a corpus. Every name,
code and date below is invented.
"""
from __future__ import annotations

import importlib

import pytest

from app.core.rag.vector_store import Chunk

PID = "synthetic-named-doc-project"

TITLE_ASK = (
    "Which procedure document covers the Snagging Procedure, and what is its "
    "number?"
)
TITLED_NAME = "XY-QA-PRC-009410-2.0 Snagging Procedure.pdf"
TITLED_TEXT = (
    "Document number: XY-QA-PRC-009410-2.0 Snagging Procedure. This procedure "
    "sets out how snags are recorded, assigned and closed."
)
VOLUME_NAME = "Riverside Works Volume 5 Quality Management.pdf"
VOLUME_TEXT = (
    "Quality management volume. This document covers inspection procedures, "
    "snagging of finishes, and the procedure for closing defects."
)
REGISTER_TEXT = (
    "XY-QA-PRC-009410-2.0 Snagging Procedure\n"
    "XY-QA-PRC-009420-1.0 Handover Procedure"
)
SECTION_TEXT = (
    "004410 - Snagging and Defects\n"
    "SECTION 004410 - SNAGGING AND DEFECTS\n"
    "This section covers snagging procedures and the closing of defects."
)

STAMP_ASK = (
    "On what date was the Kerb Replacement schedule issued, and under which "
    "tender number?"
)
LATER_NAME = "AB-2091-007 Riverside Works Volume 4 Schedules.pdf"
EARLIER_NAME = "AB-2089-003 Kerb Replacement Works Schedules.pdf"
LATER_STAMP = (
    "Date: May 2, 2091 Procurement of Contractor for Riverside Works Tender No. "
    "AB-2091-007 BILL OF QUANTITIES KERB REPLACEMENT SCHEDULE PART 2"
)
EARLIER_STAMP = (
    "Date: June 9, 2089 Procurement of Contractor for Kerb Works Tender No. "
    "AB-2089-003 BILL OF QUANTITIES KERB REPLACEMENT SCHEDULE PART 1"
)
EARLIER_ITEMS = "Kerb Replacement schedule. Item K10 precast kerb 300 m 45.00 13,500.00"


def _chunk(cid, doc_id, score, text, index=0):
    return Chunk(chunk_id=cid, project_id=PID, doc_id=doc_id, chunk_index=index,
                 text=text, score=score)


def _install(monkeypatch, *, semantic, by_doc, names, title_docs=()):
    from app.core.rag import retriever as ret
    from app.core.rag import vector_store as vs

    every = [c for chunks in by_doc.values() for c in chunks]

    def containing(self, project_id, needles, k=20, doc_ids=None):
        want = [n.lower() for n in needles]
        return [Chunk(**{**c.__dict__}) for c in every
                if all(n in (c.text or "").lower() for n in want)][:k]

    def for_docs(self, project_id, doc_ids, k_per_doc=12, **_kw):
        return [Chunk(**{**c.__dict__}) for d in doc_ids for c in by_doc.get(d, [])]

    patches = {
        "search": lambda self, project_id, qvec, k, query_text=None: [
            Chunk(**{**c.__dict__}) for c in semantic][:k],
        "identifier_search": lambda self, project_id, identifiers, k=20: [],
        "chunks_for_docs": for_docs,
        "chunks_containing_all": containing,
        "count": lambda self, pid=None: len(every),
        "_verify_embedding_identity": lambda self: None,
    }
    for name, fn in patches.items():
        monkeypatch.setattr(vs.VectorStore, name, fn)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""))

    def title_match(pid, phrase, limit=8):
        return [{"id": d, "original_name": names[d], "file_path": ""}
                for d in title_docs if phrase.lower() in names[d].lower()][:limit]

    def filename_terms(pid, terms, *, min_terms=2, require_letter=False,
                       limit=8, require_all=False):
        out = []
        for did, name in names.items():
            hits = sum(1 for t in terms if t.lower() in name.lower())
            if (hits == len(terms)) if require_all else (hits >= min_terms):
                out.append({"id": did, "original_name": name, "file_path": ""})
        return out[:limit]

    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase", title_match)
    monkeypatch.setattr("app.core.projects.documents_matching_filename_terms", filename_terms)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


# ── titled documents ──────────────────────────────────────────────────────

def test_a_titled_document_beats_a_volume_that_shares_the_topic(monkeypatch):
    ret = _install(
        monkeypatch,
        semantic=[_chunk("vol", "vol5", 0.88, VOLUME_TEXT)],
        by_doc={"vol5": [_chunk("vol", "vol5", 0.0, VOLUME_TEXT)],
                "prc": [_chunk("prc", "prc", 0.0, TITLED_TEXT)]},
        names={"vol5": VOLUME_NAME, "prc": TITLED_NAME},
        title_docs=("prc",),
    )
    chunks, _ = ret.retrieve_with_filter(TITLE_ASK, PID, k=5)
    assert chunks and chunks[0].doc_id == "prc", [c.doc_id for c in chunks]


def test_a_register_line_beats_a_section_heading_on_the_same_topic(monkeypatch):
    ret = _install(
        monkeypatch,
        semantic=[_chunk("sec", "vol5", 0.91, SECTION_TEXT)],
        by_doc={"vol5": [_chunk("sec", "vol5", 0.0, SECTION_TEXT),
                         _chunk("reg", "vol5", 0.0, REGISTER_TEXT, 1)]},
        names={"vol5": VOLUME_NAME},
    )
    chunks, _ = ret.retrieve_with_filter(TITLE_ASK, PID, k=5)
    assert chunks and "XY-QA-PRC-009410-2.0" in chunks[0].text, [c.text for c in chunks]


def test_register_line_needs_a_reference_code_before_the_title():
    from app.core.rag.retriever import chunk_states_document_register_line

    phrases = ["snagging procedure"]
    assert chunk_states_document_register_line(REGISTER_TEXT, phrases)
    assert chunk_states_document_register_line(TITLED_TEXT, phrases)
    assert not chunk_states_document_register_line(SECTION_TEXT, phrases)
    assert not chunk_states_document_register_line(VOLUME_TEXT, phrases)


@pytest.mark.parametrize("ask, expected", [
    (TITLE_ASK, True),
    ("Which specification section sets out the Handover Procedure?", True),
    ("Which plan document covers the Traffic Management Plan?", True),
    ("Who is the Engineer under this contract?", False),
    ("What is the Time for Completion for the whole of the Works?", False),
    ("What scheduling method does Specification 0042 require?", False),
])
def test_which_document_questions_are_read_from_their_shape(ask, expected):
    from app.core.rag.retriever import query_asks_which_document

    assert query_asks_which_document(ask) is expected


def test_title_phrases_are_title_case_runs():
    from app.core.rag.retriever import extract_document_title_phrases

    assert extract_document_title_phrases(TITLE_ASK) == ["snagging procedure"]
    assert extract_document_title_phrases("Which section covers snagging?") == []


def test_title_filename_bonus_only_for_the_named_title():
    from app.core.rag.retriever import title_filename_bonus

    assert title_filename_bonus(TITLED_NAME, ["snagging procedure"]) == 2.0
    assert title_filename_bonus(VOLUME_NAME, ["snagging procedure"]) == 0.0


# ── issue stamps ──────────────────────────────────────────────────────────

def _stamp_corpus(monkeypatch):
    return _install(
        monkeypatch,
        semantic=[_chunk("e-s", "early", 0.91, EARLIER_STAMP, 2),
                  _chunk("e-i", "early", 0.84, EARLIER_ITEMS, 40)],
        by_doc={"early": [_chunk("e-s", "early", 0.0, EARLIER_STAMP, 2),
                          _chunk("e-i", "early", 0.0, EARLIER_ITEMS, 40)],
                "later": [_chunk("l-s", "later", 0.0, LATER_STAMP, 3)]},
        names={"early": EARLIER_NAME, "later": LATER_NAME},
    )


def test_the_later_contracts_stamp_answers_an_issue_question(monkeypatch):
    ret = _stamp_corpus(monkeypatch)
    chunks, _ = ret.retrieve_with_filter(STAMP_ASK, PID, k=5)
    texts = [c.text for c in chunks]
    assert texts and "May 2, 2091" in texts[0], texts
    assert "Tender No. AB-2091-007" in texts[0]
    assert not any("June 9, 2089" in t for t in texts), texts


def test_an_issue_stamp_is_a_labelled_date_and_a_labelled_code():
    from app.core.rag.retriever import (
        chunk_states_document_control_block,
        chunk_states_issue_stamp,
    )

    assert chunk_states_issue_stamp(LATER_STAMP, ["tender"])
    assert chunk_states_issue_stamp(LATER_STAMP)
    assert not chunk_states_issue_stamp(LATER_STAMP, ["contract"])
    assert not chunk_states_issue_stamp(EARLIER_ITEMS)
    assert not chunk_states_document_control_block(LATER_STAMP)


def test_issue_questions_and_their_neighbours():
    from app.core.rag.retriever import (
        asked_reference_labels,
        document_identity_title_terms,
        query_asks_for_issue_identity,
    )

    assert query_asks_for_issue_identity(STAMP_ASK)
    assert asked_reference_labels(STAMP_ASK) == ["tender"]
    assert document_identity_title_terms(STAMP_ASK) == ["kerb", "replacement", "schedule"]
    for neighbour in (
        "What is the document number and revision of the priced Bill of "
        "Quantities, and who prepared it?",
        "What is the date of the priced Bill of Quantities and the Employer's "
        "contract reference on it?",
        TITLE_ASK,
        "What is the Time for Completion for the whole of the Works?",
    ):
        assert not query_asks_for_issue_identity(neighbour), neighbour


# ── store / SQL helpers the recall relies on ──────────────────────────────

@pytest.fixture
def project_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod

    importlib.reload(db_mod)
    from app.core import projects as projects_mod

    pm = importlib.reload(projects_mod)
    pm._initialized = False
    pm.init_db()
    return pm


def test_title_sql_finds_the_titled_document_among_decoys(project_store):
    p = project_store.create_project("Titled corpus")
    pid = p["id"]
    titled = project_store.add_document(pid, TITLED_NAME, file_path=f"QA/{TITLED_NAME}", size=12)
    project_store.add_document(pid, VOLUME_NAME, size=40)
    found = project_store.documents_matching_title_phrase(pid, "snagging procedure")
    assert {d["id"] for d in found} == {titled["id"]}
    assert project_store.documents_matching_title_phrase(pid, "ab cd") == []
    assert project_store.documents_matching_title_phrase(pid, "snagging%") == []
    assert project_store.documents_matching_title_phrase(pid, "snagging") == []


def test_chunks_containing_all_finds_the_register_line(project_store, monkeypatch):
    from app.core.rag import embeddings as _emb, vector_store as _vs

    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store

    e = Embedder(model_name="fake")
    store = get_store(dim=e.dim)
    p = project_store.create_project("Register SQL")
    pid = p["id"]
    vol = project_store.add_document(pid, VOLUME_NAME, size=40)
    store.upsert_chunks(pid, vol["id"], [SECTION_TEXT, REGISTER_TEXT],
                        e.encode([SECTION_TEXT, REGISTER_TEXT]))
    hits = store.chunks_containing_all(pid, ["snagging procedure"], k=8)
    assert REGISTER_TEXT in [c.text for c in hits]
