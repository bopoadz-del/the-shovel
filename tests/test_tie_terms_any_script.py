"""The question's words that break an exact score tie are read in ANY script.

A Latin-only word pattern gave an Arabic question no words at all, so every
tie the cut split fell to the chunk id. Words are now runs of Unicode letters
and digits, folded on both sides with the document-typing rule's Arabic
folding (marks dropped, alef / ya / ta marbuta variants unified).

Every test builds its own invented passages and questions. No real document,
project, question or value is named.
"""
from __future__ import annotations

import pytest

from app.core.rag.inject import _terms, apply_token_cap, tie_break_key
from app.core.rag.vector_store import Chunk

# "what is the pump maintenance period" / an invented note that answers it /
# an invented note about something else entirely.
_AR_Q = "ما هي مدة صيانة المضخة"
_AR_ON = "ملاحظة مخترعة مدة صيانة المضخة مذكورة هنا "
_AR_OFF = "ملاحظة مخترعة عن موضوع آخر تماما الآن "

# The same ask spelt with a shadda, a damma and hamza alefs; the on-topic
# note spells those words bare.
_AR_Q_MARKED = "ما مُدَّة إنجاز الأعمال"
_AR_ON_BARE = "ملاحظة مخترعة مدة انجاز الاعمال مذكورة هنا "


def _chunk(doc: str, score: float, text: str) -> Chunk:
    c = Chunk(chunk_id=f"p:{doc}:0", project_id="p", doc_id=doc,
              chunk_index=0, text=text, score=score)
    c.source_name = f"invented-{doc}.pdf"
    return c


def _same_size(*texts: str) -> list[str]:
    n = max(len(t) for t in texts) + 8
    return [(t + " pad" * n)[:n] for t in texts]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("RAG_HANDED_CHAR_BUDGET", "RAG_HANDED_POOL_FACTOR",
                 "RAG_HANDED_DOC_SHARE", "RAG_HANDED_DOC_MARGIN", "RAG_K"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_RAG_TOKENS", "100000")


@pytest.mark.parametrize("text,expected", [
    (_AR_Q, {"ما", "هي", "مده", "صيانه", "المضخه"}),
    ("Kerb_backing grade C20", {"kerb", "backing", "grade", "c20"}),
])
def test_words_are_read_in_any_script(text, expected):
    assert _terms(text) == expected


@pytest.mark.parametrize("marked,bare", [
    ("مُدَّة", "مده"), ("الأعمال", "الاعمال")])
def test_one_word_spelt_two_ways_is_one_term(marked, bare):
    assert _terms(marked) == _terms(bare) == {bare}


@pytest.mark.parametrize("question,on_topic", [
    (_AR_Q, _AR_ON), (_AR_Q_MARKED, _AR_ON_BARE)])
def test_an_arabic_question_counts_its_words_in_the_passage(question, on_topic):
    on, off = _chunk("beta", 0.61, on_topic), _chunk("alpha", 0.61, _AR_OFF)
    terms = _terms(question)
    assert tie_break_key(on, terms)[0] < tie_break_key(off, terms)[0] <= 0


# The off-topic note always holds the LOWER chunk id ("alpha" < "beta"), so a
# tie decided by the chunk id alone hands the wrong passage.
@pytest.mark.parametrize("question,on_topic,on_first", [
    (_AR_Q, _AR_ON, False), (_AR_Q_MARKED, _AR_ON_BARE, True)])
def test_a_split_tie_on_an_arabic_question_hands_the_passage_on_topic_at_the_token_cap(
        monkeypatch, question, on_topic, on_first):
    t_on, t_off = _same_size(on_topic, _AR_OFF)
    on, off = _chunk("beta", 0.61, t_on), _chunk("alpha", 0.61, t_off)
    monkeypatch.setenv("MAX_RAG_TOKENS", str(len(t_on) // 4 + 2))
    why: dict = {}
    kept, _ = apply_token_cap([on, off] if on_first else [off, on],
                              query=question, reasons=why)
    assert [c.chunk_id for c in kept] == [on.chunk_id]
    assert why == {off.chunk_id: "score_tie_overlap"}

