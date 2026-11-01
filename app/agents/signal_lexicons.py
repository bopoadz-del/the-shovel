"""Signal lexicons: vocabulary CLASSES a hat signal recognises as a category
(S-10e), not per-phrase keyword literals.

A ``TriggerType.LEXICON`` trigger names one of these classes; the hat fires when
ANY term of the class appears in the message. Adding a term widens the class in
one place -- there is no per-phrase trigger to hand-maintain per hat, and no
discipline name is special-cased in the scorer.
"""
from __future__ import annotations

import re

# Structural-design vocabulary: members, elements, actions and quantities that
# mark a structural-design ask. A category, maintained here as one set.
STRUCTURAL: frozenset[str] = frozenset({
    # members / elements
    "beam", "column", "slab", "footing", "foundation", "pile", "pile cap",
    "raft", "girder", "truss", "lintel", "shear wall", "retaining wall",
    "masonry wall", "steel member", "rebar", "reinforcement", "stirrup",
    "bracing", "diaphragm", "pier", "pedestal", "corbel",
    # actions / analysis
    "bending moment", "shear force", "deflection", "load path", "load combination",
    "bearing capacity", "structural design", "rebar design", "footing design",
    "retaining wall design", "slenderness", "buckling", "moment capacity",
    # spans / geometry carrying structural intent
    "span", "cantilever", "clear span",
})

LEXICONS: dict[str, frozenset[str]] = {
    "structural": STRUCTURAL,
}


def _term_in(term: str, message_lower: str) -> bool:
    # Whole-word (or whole-phrase) match with a simple optional plural, so
    # "slab"/"slabs" and "load path"/"load paths" hit while "columnar" (a
    # different word) and other embedded substrings do not.
    return re.search(r"(?<![a-z])" + re.escape(term) + r"s?(?![a-z])", message_lower) is not None


def lexicon_matches(name: str, message_lower: str) -> bool:
    """True when any term of lexicon ``name`` appears in the lowercased message.
    Unknown lexicon name -> False (never raises)."""
    terms = LEXICONS.get((name or "").lower())
    if not terms:
        return False
    return any(_term_in(t, message_lower) for t in terms)
