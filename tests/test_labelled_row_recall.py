"""Labelled-row recall: the filled row for the label a question names.

General mechanism (``recall_labelled_rows`` in the retriever). A particular
-- an amount, a duration, a party, "Not required", a register entry -- is a
labelled row: a label cell and a value cell. Asked plainly, the row is not in
the pool (a scanned table embeds badly and shares a word or two with the
question) while long prose about the same topic is. The row is found by the
question's own labels:

* in the particulars documents, whichever particulars kind the upload name
  says it is (contract data, appendix to tender, contract particulars);
* for a particulars-shaped question, as text across the project, keeping a
  chunk only when a line opens with the label and states a filled value --
  a figure, a party, a date or a stated absence; a blank template and a
  pointer elsewhere are not filled.

It replaces seven per-particular rescues with phrase lists (some carrying one
contract's clause numbers). Every name and figure below is invented.
"""
from __future__ import annotations

import pytest

from app.core.rag.vector_store import Chunk

PID = "synthetic-rows-project"
PREFIX = "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"

APPENDIX_NAME = "Lakeview Works Appendix to Tender.pdf"
APPENDIX_ROWS = PREFIX + (
    "| 14.2 | Advance Payment: 12% of the Accepted Contract Amount | |\n"
    "| 14.3 | Percentage of retention: 5% | |\n"
)
GC_ADVANCE = (
    "14.2 Advance Payment. The Employer shall make an advance payment, as an "
    "interest-free loan for mobilisation, when the Contractor submits a "
    "guarantee in accordance with this Sub-Clause."
)

SECURITY_ROW = "4.2 Performance Security\nNot required for this package."
SECURITY_FORM = (
    "Annex C Form of Performance Security. The Guarantor irrevocably "
    "undertakes to pay the Employer any sum not exceeding [insert amount] "
    "upon receipt of a demand."
)
SECURITY_CLAUSE = (
    "4.2 Performance Security. The Contractor shall obtain at his cost a "
    "Performance Security for proper performance, in the amount and currency "
    "stated in the Contract Data."
)


def _chunk(cid, doc_id, score, text, index=0):
    return Chunk(chunk_id=cid, project_id=PID, doc_id=doc_id, chunk_index=index,
                 text=text, score=score)


def _install(monkeypatch, *, semantic, docs, names, kind_titles):
    from app.core.rag import retriever as ret
    from app.core.rag import vector_store as vs

    every = [c for chunks in docs.values() for c in chunks]

    def containing(self, project_id, needles, k=20, doc_ids=None):
        want = [" ".join(n.lower().split()) for n in needles]
        return [Chunk(**{**c.__dict__}) for c in every
                if all(n in " ".join((c.text or "").lower().split()) for n in want)][:k]

    def for_docs(self, project_id, doc_ids, k_per_doc=12, **_kw):
        return [Chunk(**{**c.__dict__}) for d in doc_ids for c in docs.get(d, [])]

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
                for d in kind_titles if phrase.lower() in names[d].lower()][:limit]

    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase", title_match)
    monkeypatch.setattr("app.core.projects.documents_matching_filename_terms",
                        lambda *a, **k: [])
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_a_row_in_an_appendix_to_tender_is_found_by_its_label(monkeypatch):
    ret = _install(
        monkeypatch,
        semantic=[_chunk("gc", "gc", 0.82, GC_ADVANCE)],
        docs={"att": [_chunk("att0", "att", 0.0, APPENDIX_ROWS)],
              "gc": [_chunk("gc", "gc", 0.0, GC_ADVANCE)]},
        names={"att": APPENDIX_NAME, "gc": "Lakeview Works Conditions.pdf"},
        kind_titles=("att",),
    )
    chunks, _ = ret.retrieve_with_filter("What is the Advance Payment?", PID, k=5)
    assert chunks and "Advance Payment: 12%" in chunks[0].text, [c.text for c in chunks]


def test_a_not_required_row_outside_the_particulars_beats_the_form(monkeypatch):
    ret = _install(
        monkeypatch,
        semantic=[_chunk("form", "annexc", 0.91, SECURITY_FORM),
                  _chunk("gc", "gc", 0.86, SECURITY_CLAUSE)],
        docs={"sched": [_chunk("row", "sched", 0.0, SECURITY_ROW)],
              "annexc": [_chunk("form", "annexc", 0.0, SECURITY_FORM)],
              "gc": [_chunk("gc", "gc", 0.0, SECURITY_CLAUSE)]},
        names={"sched": "Lakeview Works Volume 4 Schedules.pdf",
               "annexc": "Lakeview Works Annex C forms.pdf",
               "gc": "Lakeview Works Conditions.pdf"},
        kind_titles=(),
    )
    chunks, _ = ret.retrieve_with_filter(
        "What is the value of the Performance Security?", PID, k=5,
    )
    assert chunks and chunks[0].chunk_id == "row", [c.chunk_id for c in chunks]


def test_an_ordinary_question_does_not_search_its_word_runs(monkeypatch):
    """A question that is not particulars-shaped: its word runs are not labels."""
    minutes = "Site Induction Workshop\nChaired by the safety lead."
    ret = _install(
        monkeypatch,
        semantic=[_chunk("gc", "gc", 0.86, SECURITY_CLAUSE)],
        docs={"mins": [_chunk("row", "mins", 0.0, minutes)],
              "gc": [_chunk("gc", "gc", 0.0, SECURITY_CLAUSE)]},
        names={"mins": "Lakeview Works meeting minutes.pdf",
               "gc": "Lakeview Works Conditions.pdf"},
        kind_titles=(),
    )
    chunks, _ = ret.retrieve_with_filter(
        "Who chaired the site induction workshop?", PID, k=5,
    )
    assert "row" not in [c.chunk_id for c in chunks]


@pytest.mark.parametrize("text, filled", [
    (SECURITY_ROW, True),
    ("4.2 Performance Security: 10% of the Accepted Contract Amount", True),
    ("| 4.2 | Performance Security | No |", True),
    ("4.2 Performance Security: [insert amount]", False),
    ("4.2 Performance Security: as stated in the Contract Data", False),
    (SECURITY_CLAUSE, False),
    (SECURITY_FORM, False),
])
def test_a_filled_row_is_a_label_and_a_value(text, filled):
    from app.core.rag.retriever import chunk_states_labelled_row

    assert chunk_states_labelled_row(text, ["performance security"]) is filled


def test_row_labels_come_from_the_question():
    from app.core.rag.retriever import asked_row_labels

    labels = asked_row_labels("What does Schedule 7 of the contract contain?")
    assert labels[0] == "schedule 7"
    assert "performance security" in asked_row_labels(
        "What is the value of the Performance Security?"
    )
    assert asked_row_labels("What does Performance Security mean?") == []
