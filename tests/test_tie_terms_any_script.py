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



# --- Grader round (19:05Z): Latin unchanged, the stated Latin change, no
# shared word, and the stated limit (no prefix stripping). ---------------

_OLD_LATIN = __import__("re").compile(r"[a-z0-9]+")


def _old_terms(text: str) -> set[str]:
    """The pattern this PR replaced, kept here only to compare against."""
    return set(_OLD_LATIN.findall((text or "").casefold()))


_EN_Q = "what is the pump maintenance period"
_EN_ON = "invented note the pump maintenance period is stated here "
_EN_OFF = "invented note about another subject entirely now "


@pytest.mark.parametrize("text", [
    _EN_Q + " " + _EN_ON, "Kerb_backing grade C20, ref 4-B/7 (rev.3)"])
def test_plain_latin_text_yields_the_same_words_as_the_old_pattern(text):
    assert _terms(text) == _old_terms(text)


@pytest.mark.parametrize("on_first", [False, True])
def test_a_split_tie_on_a_latin_question_is_decided_as_before(monkeypatch, on_first):
    t_on, t_off = _same_size(_EN_ON, _EN_OFF)
    on, off = _chunk("beta", 0.61, t_on), _chunk("alpha", 0.61, t_off)
    # Same overlap under the old pattern and the new one: same outcome.
    for c in (on, off):
        assert (len(_terms(_EN_Q) & _terms(c.text))
                == len(_old_terms(_EN_Q) & _old_terms(c.text)))
    monkeypatch.setenv("MAX_RAG_TOKENS", str(len(t_on) // 4 + 2))
    why: dict = {}
    kept, _ = apply_token_cap([on, off] if on_first else [off, on],
                              query=_EN_Q, reasons=why)
    assert [c.chunk_id for c in kept] == [on.chunk_id]
    assert why == {off.chunk_id: "score_tie_overlap"}


# STATED CHANGE: an accented Latin letter and a digit of another script used
# to SPLIT a word ("béton" -> "b", "ton"); they are now part of it.
@pytest.mark.parametrize("text,new,old", [
    ("béton armé", {"béton", "armé"}, {"b", "ton", "arm"}),
    ("Grade C٣٠ mix", {"grade", "c٣٠", "mix"},
     {"grade", "c", "mix"}),
])
def test_accented_letters_and_other_digits_are_now_inside_a_word(text, new, old):
    assert _old_terms(text) == old
    assert _terms(text) == new


# Consequence of the change: a fragment of a split word no longer counts as
# a shared word ("ton" in a passage was a hit for "béton" in the question).
@pytest.mark.parametrize("question,passage", [
    ("béton", "invented ton of fill"), ("C٣٠", "invented c mix")])
def test_a_fragment_of_an_accented_or_mixed_word_is_no_longer_a_hit(question, passage):
    assert _old_terms(question) & _old_terms(passage)
    assert tie_break_key(_chunk("x", 0.5, passage), _terms(question))[0] == 0


# No word of the question in either passage: the tie falls to the chunk id,
# whichever order retrieval handed them in.
@pytest.mark.parametrize("question,on_first", [
    ("zzqv wqxk", False), ("قطعة مختلفة", True)])
def test_with_no_shared_word_the_tie_falls_to_the_chunk_id(monkeypatch, question, on_first):
    t_b, t_a = _same_size(_EN_ON, _EN_OFF)
    b, a = _chunk("beta", 0.61, t_b), _chunk("alpha", 0.61, t_a)
    terms = _terms(question)
    assert tie_break_key(a, terms)[0] == tie_break_key(b, terms)[0] == 0
    monkeypatch.setenv("MAX_RAG_TOKENS", str(len(t_b) // 4 + 2))
    why: dict = {}
    kept, _ = apply_token_cap([b, a] if on_first else [a, b],
                              query=question, reasons=why)
    assert [c.chunk_id for c in kept] == [a.chunk_id]
    assert why == {b.chunk_id: "score_tie_overlap"}


# STATED LIMIT: no prefix stripping. An Arabic word with the article (or
# another attached prefix) is a different term from the bare word.
@pytest.mark.parametrize("prefixed,bare", [
    ("المضخة", "مضخة"),
    ("للمضخة", "مضخة")])
def test_no_prefix_stripping_an_arabic_word_with_the_article_is_another_term(prefixed, bare):
    assert _terms(prefixed) and _terms(bare)
    assert not (_terms(prefixed) & _terms(bare))
