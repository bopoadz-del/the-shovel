"""Who is a party: the particulars row that names it beats the definition.

General rule inside labelled-row recall: when the question asks WHO a defined
party is (the Engineer, the Employer, the Contractor, a Representative ...),
the particulars row that names that party -- a label row whose value is a
proper name, in the Contract Data / Appendix to Tender / Particular
Conditions -- must outrank the General Conditions clause that DEFINES the
same term ("'Employer' means the person named as employer in ...").

The definition shares every word of the question and wins on cosine; the
named row shares one. And a text search on the role word alone hits every
clause that mentions the role, so a store's LIMIT cuts the row off: the row
is fetched by the role together with the words a legal person's name ends in
(Ltd, Limited, Company, Authority ...), a naming lexicon, not any name.

Every firm, document and clause below is invented.
"""
from __future__ import annotations

import pytest

from app.core.rag.vector_store import Chunk

PID = "synthetic-party-project"
PREFIX = "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"

ENGINEER_ASK = "Who is the Engineer under this contract?"
EMPLOYER_ASK = "Who is the Employer under this contract?"

ENGINEER_DEFINITION = (
    '1.1.2.4 "Engineer" means the person appointed by the Employer to act as '
    "the Engineer for the purposes of the Contract and named in the Contract "
    "Data, or other person appointed from time to time by the Employer."
)
EMPLOYER_DEFINITION = (
    '1.1.2.2 "Employer" means the person named as employer in the Contract '
    "Data and the legal successors in title to this person."
)
ENGINEER_ROW = PREFIX + "1.1.2.4 | Engineer | Acme Consulting Ltd |"
EMPLOYER_ROW = PREFIX + "1.1.2.2 | Employer | Harbourview Port Authority Ltd |"
REPRESENTATIVE_ROW = PREFIX + "3.2 | Engineer's Representative | Jane Example |"


def _decoys(role):
    return [
        f"{n}.1 The {role} shall give notice to the other Party within 28 days "
        f"of becoming aware of the event or circumstance, clause {n}."
        for n in range(2, 32)
    ]


def _chunk(cid, doc_id, score, text, index=0):
    return Chunk(chunk_id=cid, project_id=PID, doc_id=doc_id, chunk_index=index,
                 text=text, score=score)


def _install(monkeypatch, *, role, definition, row, row_doc_name):
    from app.core.rag import retriever as ret
    from app.core.rag import vector_store as vs

    decoys = [_chunk(f"gc{i}", "gc", 0.0, t, i + 1) for i, t in enumerate(_decoys(role))]
    defn = _chunk("defn", "gc", 0.93, definition, 0)
    named = _chunk("row", "part", 0.0, row)
    # The row is stored last, so a LIMIT on a role-word-only search cuts it off.
    corpus = [defn] + decoys + [named]
    semantic = [defn, decoys[0], decoys[1]]
    semantic = [Chunk(**{**c.__dict__, "score": s}) for c, s in zip(semantic, (0.93, 0.88, 0.86))]

    def containing(self, project_id, needles, k=20, doc_ids=None):
        want = [" ".join(n.lower().split()) for n in needles]
        hits = [c for c in corpus
                if (not doc_ids or c.doc_id in doc_ids)
                and all(n in (c.text or "").lower() for n in want)]
        return [Chunk(**{**c.__dict__}) for c in hits][:k]

    patches = {
        "search": lambda self, project_id, qvec, k, query_text=None: [
            Chunk(**{**c.__dict__}) for c in semantic][:k],
        "identifier_search": lambda self, project_id, identifiers, k=20: [],
        "chunks_for_docs": lambda self, project_id, doc_ids, **kw: [],
        "chunks_containing_all": containing,
        "count": lambda self, pid=None: len(corpus),
        "_verify_embedding_identity": lambda self: None,
    }
    for name, fn in patches.items():
        monkeypatch.setattr(vs.VectorStore, name, fn)
    names = {"gc": "Harbourview General Conditions.pdf", "part": row_doc_name}
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""))
    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase",
                        lambda pid, phrase, limit=8: [])
    monkeypatch.setattr("app.core.projects.documents_matching_filename_terms",
                        lambda *a, **k: [])
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


@pytest.mark.parametrize("ask, role, definition, row, doc_name", [
    (ENGINEER_ASK, "Engineer", ENGINEER_DEFINITION, ENGINEER_ROW,
     "Harbourview Appendix to Tender.pdf"),
    (EMPLOYER_ASK, "Employer", EMPLOYER_DEFINITION, EMPLOYER_ROW,
     "Harbourview Particular Conditions.pdf"),
], ids=["engineer", "employer"])
def test_the_naming_row_outranks_the_definition(monkeypatch, ask, role, definition,
                                                row, doc_name):
    ret = _install(monkeypatch, role=role, definition=definition, row=row,
                   row_doc_name=doc_name)
    chunks, _ = ret.retrieve_with_filter(ask, PID, k=5)
    ids = [c.chunk_id for c in chunks]
    assert "row" in ids, ids
    if "defn" in ids:
        assert ids.index("row") < ids.index("defn"), ids


@pytest.mark.parametrize("ask, role", [
    (ENGINEER_ASK, "engineer"),
    (EMPLOYER_ASK, "employer"),
    ("Who is the Engineer's Representative?", "engineer's representative"),
    ("Which company is the Contractor?", "contractor"),
    ("What does Employer mean?", ""),
    ("What is the Time for Completion?", ""),
])
def test_the_asked_role_is_read_from_the_question(ask, role):
    from app.core.rag.retriever import asked_party_role

    assert asked_party_role(ask) == role


def test_a_naming_row_is_a_role_label_and_a_proper_name():
    from app.core.rag.retriever import chunk_defines_role, chunk_names_party

    assert chunk_names_party(ENGINEER_ROW, "engineer")
    assert chunk_names_party(EMPLOYER_ROW, "employer")
    assert not chunk_names_party(ENGINEER_DEFINITION, "engineer")
    assert not chunk_names_party(EMPLOYER_DEFINITION, "employer")
    # The Representative is a different party from the Engineer.
    assert not chunk_names_party(REPRESENTATIVE_ROW, "engineer")
    assert chunk_names_party(REPRESENTATIVE_ROW, "engineer's representative")
    assert chunk_defines_role(EMPLOYER_DEFINITION, "employer")
    assert not chunk_defines_role(EMPLOYER_ROW, "employer")


# ── the answer path follows the asked party, not the Engineer only ────────

EMPLOYER_RAG = (
    "[doc_id=part] " + EMPLOYER_ROW + "\n\n[doc_id=gc] " + EMPLOYER_DEFINITION
)


def test_the_graft_states_the_asked_party_from_its_naming_row(monkeypatch):
    monkeypatch.setenv("RAG_WITHHOLD_PARTY_NAMES", "0")
    from app.agents.runtime import _graft_asked_contract_particular

    out = _graft_asked_contract_particular(
        "The excerpts do not state the Employer.",
        {"content": EMPLOYER_RAG},
        [{"role": "user", "content": EMPLOYER_ASK}],
    )
    assert out.startswith("The Employer is Harbourview Port Authority Ltd."), out


def test_the_party_name_is_read_for_any_role():
    from app.core.rag.retriever import extract_party_name

    assert extract_party_name(EMPLOYER_RAG, "employer") == "Harbourview Port Authority Ltd"
    assert extract_party_name(ENGINEER_ROW, "engineer") == "Acme Consulting Ltd"
    assert extract_party_name(EMPLOYER_DEFINITION, "employer") is None


def test_inject_names_the_asked_party_in_its_hint(monkeypatch):
    monkeypatch.setenv("RAG_WITHHOLD_PARTY_NAMES", "0")
    from app.core.rag.inject import format_chunks_as_system_message

    msg = format_chunks_as_system_message(
        [_chunk("row", "part", 0.9, EMPLOYER_ROW)], 4, query=EMPLOYER_ASK,
    )
    assert "EMPLOYER IDENTITY" in msg["content"], msg["content"][:400]
