"""Rows past a volume's early windows: the asked row and composition operands.

General mechanisms in the retriever:

* ``recall_rows_deep_in_pooled_documents`` -- when no pooled chunk states the
  asked particular, the pooled particulars / conditions volumes (or those
  that mention the particular) are read whole and the row that states it is
  pooled. A bound volume mentions a particular early and states it in a late
  appendix; first-N fetches never reach it.
* ``recall_composition_operands`` -- a computed answer needs every operand
  row; a missing operand is read from the pooled volumes, and the text match
  uses the words of the operand's label one by one, because a scanned label
  is split across lines and carries no clause number.

They replace two rescues whose text matches carried one contract's clause
number, its VAT wording and one expected rate. Every name and figure below is
invented.
"""
from __future__ import annotations

from app.core.rag.vector_store import Chunk

PID = "synthetic-deep-project"
VOLUME = "vol-bound"
VOLUME_NAME = "Lakeview Works Conditions of Contract (bound).pdf"
PREFIX = "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"

DNP_ASK = "What is the Defects Notification Period?"
DNP_MENTION = (
    '1.1.27 "Defects Notification Period" means the period for notifying '
    "defects in the Works, as stated in the Contract Data, calculated from "
    "the date of completion."
)
DNP_ROW = PREFIX + (
    "1.1.27 | Defects Notification Period | 545 days from the date of the "
    "Taking-Over Certificate"
)
FILLER = "General conditions continued. The Parties shall act in good faith."

DAILY_ASK = (
    "Calculate the delay damages per calendar day in SAR for the whole of the "
    "Works."
)
RATE_ROW = PREFIX + (
    "8.8 | Delay Damages for the whole of the Works | 0.05% of the Contract "
    "Price per calendar day"
)
BASE_ROW = PREFIX + (
    "2.4 | Accepted\nContract\nAmount | SAR 4,200,000.00 (exclusive of tax)"
)


def _chunk(cid, text, index, score=0.0, doc=VOLUME):
    return Chunk(chunk_id=cid, project_id=PID, doc_id=doc, chunk_index=index,
                 text=text, score=score)


def _volume(late_index, late_text, early_text, early_index):
    rows = [_chunk(f"f{i}", FILLER, i) for i in range(late_index + 20)
            if i not in (early_index, late_index)]
    rows.append(_chunk("early", early_text, early_index))
    rows.append(_chunk("late", late_text, late_index))
    return sorted(rows, key=lambda c: c.chunk_index)


class _WholeReadStore:
    """Reads a document whole on ``all_rows``; otherwise first-N only."""

    def __init__(self, rows):
        self.rows = rows

    def chunks_for_docs(self, project_id, doc_ids, k_per_doc=12, from_end=False,
                        all_rows=False):
        rows = [c for c in self.rows if c.doc_id in (doc_ids or [])]
        if all_rows:
            return rows
        n = min(int(k_per_doc or 12), 12)
        return rows[-n:] if from_end else rows[:n]


class _TextOnlyStore:
    """Caps every document fetch at its first rows; matches text inside documents."""

    def __init__(self, rows):
        self.rows = rows

    def chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
        return [c for c in self.rows if c.doc_id in (doc_ids or [])][:12]

    def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
        want = [str(n).lower() for n in needles]
        return [c for c in self.rows
                if (not doc_ids or c.doc_id in doc_ids)
                and all(n in (c.text or "").lower() for n in want)][:k]


def _names(monkeypatch):
    from app.core.rag import retriever as ret

    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: VOLUME_NAME)
    monkeypatch.setattr(ret, "_doc_owner_project_ids", lambda doc_ids: [])
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    return ret


def test_a_particular_stated_late_in_a_pooled_volume_is_pooled(monkeypatch):
    ret = _names(monkeypatch)
    rows = _volume(450, DNP_ROW, DNP_MENTION, 2)
    mention = next(c for c in rows if c.chunk_id == "early")
    fused = {"early": (mention, 0.88, 0.0)}
    added = ret.recall_rows_deep_in_pooled_documents(
        DNP_ASK, PID, fused, _WholeReadStore(rows),
    )
    assert added >= 1
    assert any("545 days" in c.text for c, _s, _b in fused.values())


def test_a_pooled_statement_leaves_the_volume_unread(monkeypatch):
    ret = _names(monkeypatch)
    rows = _volume(450, DNP_ROW, DNP_MENTION, 2)
    stated = _chunk("stated", DNP_ROW, 3)
    fused = {"stated": (stated, 0.7, 0.0)}
    assert ret.recall_rows_deep_in_pooled_documents(
        DNP_ASK, PID, fused, _WholeReadStore(rows),
    ) == 0
    assert set(fused) == {"stated"}


def test_a_scanned_operand_label_is_matched_word_by_word(monkeypatch):
    ret = _names(monkeypatch)
    rows = _volume(500, BASE_ROW, RATE_ROW, 9)
    rate = next(c for c in rows if c.chunk_id == "early")
    fused = {"early": (rate, 0.95, 0.0)}
    added = ret.recall_composition_operands(DAILY_ASK, PID, fused, _TextOnlyStore(rows))
    assert added >= 1 and "late" in fused, sorted(fused)
    assert "SAR 4,200,000.00" in fused["late"][0].text


def test_only_a_composing_question_has_operands():
    from app.core.rag.retriever import composition_operands_for

    assert [name for name, _p, _a in composition_operands_for(DAILY_ASK)] == ["base", "rate"]
    assert composition_operands_for(DNP_ASK) == []
    assert composition_operands_for("What is the Time for Completion?") == []


def test_label_words_survive_a_line_split():
    from app.core.rag.retriever import label_word_needle_sets

    assert label_word_needle_sets(["accepted contract amount", "is"]) == [
        ("accepted", "contract", "amount"),
    ]
