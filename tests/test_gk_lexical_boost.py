"""General-knowledge lexical-boost helpers (retriever ranking)."""
from app.core.rag import retriever


def test_significant_terms_drops_stopwords_and_short_words():
    t = retriever._significant_terms("What is the standard unit for excavation and concrete?")
    assert "excavation" in t and "concrete" in t and "unit" in t
    for w in ("what", "the", "for", "and", "is", "standard"):
        assert w not in t


def test_gk_lexical_bonus_positive_on_overlap():
    q = retriever._significant_terms("unit of measurement for excavation and concrete")
    text = "BOQ Standard Units of Measurement: excavation m3, concrete m3, formwork m2"
    assert retriever._gk_lexical_bonus(q, text) > 0


def test_gk_lexical_bonus_caps():
    q = frozenset({"excavation", "concrete", "formwork", "reinforcement",
                   "pipe", "manhole", "kerb", "masonry"})
    text = "excavation concrete formwork reinforcement pipe manhole kerb masonry"
    assert retriever._gk_lexical_bonus(q, text) == retriever._GK_BONUS_CAP


def test_gk_lexical_bonus_zero_when_no_overlap():
    q = retriever._significant_terms("banana smoothie recipe")
    assert retriever._gk_lexical_bonus(q, "concrete m3 and excavation earthworks") == 0.0


def test_gk_boost_stays_below_identifier_bonus():
    # exact-reference (identifier) lookups must still outrank the GK boost.
    assert retriever._GK_BONUS_CAP < 2.0
