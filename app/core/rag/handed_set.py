"""Which retrieved chunks are handed to the model, and when to plan for more.

Three general rules over the first-pass retrieval (S-R5). None of them knows a
project, a document or a question; they read only scores, document ids,
source names, chunk text lengths and the question's own words.

(a) The handed set is bounded by a CHARACTER budget, not a fixed count. A
    count of two hands two long chunks of one document and nothing else; a
    character budget lets several short excerpts from several sources fit
    the same spend. The budget is the same on every turn: the daily RAG
    token counter (app/core/rag/budget.py) is measured and traced, never
    used to shorten what is handed (owner order, 13 Nov).

(b) The handed set is diverse across documents. One document may hold at
    most a share of the handed slots while another document's best chunk
    scores within a margin of the top; a chunk past its document's share is
    deferred behind the other documents, never dropped, so a corpus with one
    relevant document still hands all of it. Copies of one document (the same
    title in two folders, a duplicate upload) are one document family and
    count once; a chunk whose text repeats one already handed from the same
    family is not handed twice.

(c) A document ask (not only a figure ask) gets the evidence loop's plan when
    the first pass covers fewer distinct sources than the kinds of document
    the question names: a question naming a bill, a drawing and a method
    statement needs three sources, not two chunks of one.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Sequence

#: Character budget for the handed first-pass set on a normal turn. Roughly
#: MAX_RAG_TOKENS' default spend at 4 chars per token, leaving the token cap
#: as the backstop it already is.
DEFAULT_CHAR_BUDGET = 16_000
#: Retrieval pool depth as a multiple of the turn's k: selection needs
#: candidates beyond the top k to have other documents to choose from.
DEFAULT_POOL_FACTOR = 2
#: Largest share of the handed set one document family may take while
#: another family scores within the margin.
DEFAULT_DOC_SHARE = 0.4
#: Score margin (absolute, same scale as chunk scores) inside which another
#: document counts as a competitor.
DEFAULT_DOC_MARGIN = 0.15


def _env_number(name: str, default: float, *, minimum: float) -> float:
    raw = (os.getenv(name) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def char_budget() -> int:
    """The handed set's character budget. It does not depend on the day's
    token spend: the daily counter never changes what is handed."""
    return int(_env_number("RAG_HANDED_CHAR_BUDGET", DEFAULT_CHAR_BUDGET, minimum=1))


def pool_k(k: int) -> int:
    """How many candidates to retrieve so selection has a choice."""
    factor = int(_env_number("RAG_HANDED_POOL_FACTOR", DEFAULT_POOL_FACTOR, minimum=1))
    return max(1, int(k)) * factor


def doc_share() -> float:
    share = _env_number("RAG_HANDED_DOC_SHARE", DEFAULT_DOC_SHARE, minimum=0.0)
    return share if 0.0 < share <= 1.0 else DEFAULT_DOC_SHARE


def doc_margin() -> float:
    return _env_number("RAG_HANDED_DOC_MARGIN", DEFAULT_DOC_MARGIN, minimum=0.0)


# --- document family -------------------------------------------------------

_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")
# Upload / copy decorations that do not change which document it is: a
# leading or trailing timestamp, "(1)", "copy", "- Copy (2)".
_STAMP_RE = re.compile(r"(?:^|[_\s-])\d{9,}(?=$|[_\s-])")
_COPY_RE = re.compile(r"(?:[\s_-]*\(\d{1,3}\)|[\s_-]+copy(?:[\s_-]*\(?\d{1,3}\)?)?)\s*$",
                      re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def family_key(chunk: Any) -> str:
    """One key per document family: the source title without its folder,
    extension, upload timestamp or copy suffix. A chunk with no source name
    is its own family (its doc_id)."""
    name = (getattr(chunk, "source_name", "") or "").strip()
    if name:
        base = re.split(r"[\\/]", name)[-1]
        base = _EXT_RE.sub("", base)
        base = _STAMP_RE.sub(" ", base)
        prev = None
        while prev != base:
            prev = base
            base = _COPY_RE.sub("", base)
        key = _NON_ALNUM_RE.sub("", base.lower())
        if key:
            return "t:" + key
    return "d:" + str(getattr(chunk, "doc_id", "") or id(chunk))


def _score(chunk: Any) -> float:
    raw = getattr(chunk, "score", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    return float(raw)


def _text_key(chunk: Any) -> str:
    return " ".join((getattr(chunk, "text", "") or "").split()).lower()


# --- (b) diversity ---------------------------------------------------------

def diversify(chunks: Sequence[Any], *, slots: int, share: float, margin: float) -> list:
    """Reorder ``chunks`` (best first) so no family takes more than
    ``ceil(share * slots)`` of the leading positions while another family's
    best chunk scores within ``margin`` of the overall top. Deferred chunks
    follow, in score order; nothing is dropped except a repeated text inside
    one family (the duplicate copy)."""
    ordered = sorted(chunks, key=_score, reverse=True)
    if not ordered:
        return []
    top = _score(ordered[0])
    best_by_family: dict[str, float] = {}
    for c in ordered:
        best_by_family.setdefault(family_key(c), _score(c))
    competitors = {f for f, s in best_by_family.items() if s >= top - margin}
    cap = max(1, math.ceil(share * max(1, slots)))
    seen_text: set[tuple[str, str]] = set()
    taken: dict[str, int] = {}
    lead: list = []
    deferred: list = []
    for c in ordered:
        fam = family_key(c)
        tkey = (fam, _text_key(c))
        if tkey[1] and tkey in seen_text:
            continue
        seen_text.add(tkey)
        crowded = len(competitors) > 1 and taken.get(fam, 0) >= cap
        if crowded:
            deferred.append(c)
            continue
        taken[fam] = taken.get(fam, 0) + 1
        lead.append(c)
    return lead + deferred


# --- (a) character budget --------------------------------------------------

def fit_char_budget(chunks: Sequence[Any], budget: int) -> list:
    """Whole chunks, in order, while they fit ``budget`` characters. The first
    chunk is always kept (an over-long top excerpt is still the best one);
    a chunk that does not fit is skipped so a shorter later one may."""
    kept: list = []
    used = 0
    for c in chunks:
        size = len(getattr(c, "text", "") or "")
        if kept and used + size > budget:
            continue
        kept.append(c)
        used += size
    return kept


def select_handed(
    chunks: Sequence[Any], *, k: int,
) -> tuple[list, dict[str, Any]]:
    """The handed set: diversified, then fitted to the character budget.

    Returns ``(handed, allotment)`` where ``allotment`` is what the trace
    prints: the budget, the characters used, the per-layer and per-family
    counts."""
    budget = char_budget()
    ordered = diversify(chunks, slots=k, share=doc_share(), margin=doc_margin())
    handed = fit_char_budget(ordered, budget)
    layers: dict[str, int] = {}
    families: dict[str, int] = {}
    for c in handed:
        layer = getattr(c, "layer", None) or "own"
        layers[layer] = layers.get(layer, 0) + 1
        fam = family_key(c)
        families[fam] = families.get(fam, 0) + 1
    allotment = {
        "char_budget": budget,
        "chars_handed": sum(len(getattr(c, "text", "") or "") for c in handed),
        "layers": layers,
        "families": len(families),
        "max_family_share": max(families.values()) if families else 0,
    }
    return handed, allotment


# --- (c) named source kinds vs first-pass coverage --------------------------

#: Kinds of project document a question can name. The vocabulary is the
#: construction document classes themselves, not any document's title.
_SOURCE_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("bill", r"\b(?:bill(?:s)?\s+of\s+quantit(?:y|ies)|boq|priced\s+bill|bill\s+no\.?)\b"),
    ("drawing", r"\b(?:drawings?|dwgs?|plans?\s+and\s+sections?|layout\s+plans?)\b"),
    ("method_statement", r"\bmethod\s+statements?\b"),
    ("specification", r"\b(?:specifications?|specs?)\b"),
    ("programme", r"\b(?:programmes?|baseline\s+schedule|construction\s+schedule)\b"),
    ("contract", r"\b(?:contract\s+data|conditions\s+of\s+contract|particular\s+conditions|"
                 r"agreement)\b"),
    ("report", r"\breports?\b"),
    ("correspondence", r"\b(?:letters?|correspondence|transmittals?|emails?)\b"),
    ("rfi", r"\b(?:rfis?|requests?\s+for\s+information)\b"),
    ("submittal", r"\b(?:submittals?|material\s+approvals?)\b"),
    ("minutes", r"\bminutes\b"),
    ("register", r"\bregisters?\b"),
)
_SOURCE_KIND_RES = tuple((kind, re.compile(p, re.IGNORECASE)) for kind, p in _SOURCE_KIND_PATTERNS)
_MARKER_DOC_RE = re.compile(r"\[doc_id=([^\s\]]+)\s+chunk=")


def named_source_kinds(question: str) -> tuple[str, ...]:
    """The distinct kinds of document the question names, in a fixed order."""
    text = question or ""
    return tuple(kind for kind, rx in _SOURCE_KIND_RES if rx.search(text))


def first_pass_sources(evidence_text: str) -> int:
    """Distinct documents in the first-pass excerpts (their doc_id markers)."""
    return len(set(_MARKER_DOC_RE.findall(evidence_text or "")))


def wants_source_coverage_retrieval(question: str, evidence_text: str) -> bool:
    """True when the question names more kinds of document than the first
    pass has sources: the evidence loop should plan one retrieval per need.
    A question naming one kind or none never triggers it."""
    kinds = named_source_kinds(question)
    if len(kinds) < 2:
        return False
    return first_pass_sources(evidence_text) < len(kinds)
