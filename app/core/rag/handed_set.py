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
    family is not handed twice. The share holds in CHARACTERS too: slots are
    not equal, and two long chunks of one document can take most of the
    budget in two slots. While another family competes, a family's later
    chunk that would take the family past the share of the character budget
    is deferred behind the other documents the same way (never dropped). A
    family's first chunk is never deferred for its size.

(c) A document ask (not only a figure ask) gets the evidence loop's plan when
    the first pass covers fewer distinct sources than the kinds of document
    the question names: a question naming a bill, a drawing and a method
    statement needs three sources, not two chunks of one.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Callable, Sequence

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

def _competing_families(chunks: Sequence[Any], margin: float) -> set[str]:
    """Families whose best chunk scores within ``margin`` of the top chunk."""
    ordered = sorted(chunks, key=_score, reverse=True)
    if not ordered:
        return set()
    top = _score(ordered[0])
    best_by_family: dict[str, float] = {}
    for c in ordered:
        best_by_family.setdefault(family_key(c), _score(c))
    return {f for f, s in best_by_family.items() if s >= top - margin}


def diversify(
    chunks: Sequence[Any], *, slots: int, share: float, margin: float,
    reasons: dict[str, str] | None = None, deferred_ids: set | None = None,
) -> list:
    """Reorder ``chunks`` (best first) so no family takes more than
    ``ceil(share * slots)`` of the leading positions while another family's
    best chunk scores within ``margin`` of the overall top. Deferred chunks
    follow, in score order; nothing is dropped except a repeated text inside
    one family (the duplicate copy).

    When given, ``reasons`` gets ``duplicate`` for each copy dropped and
    ``deferred_ids`` the id of each chunk deferred, at the point each is
    decided."""
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
            if reasons is not None:
                reasons[getattr(c, "chunk_id", None) or ""] = "duplicate"
            continue
        seen_text.add(tkey)
        crowded = len(competitors) > 1 and taken.get(fam, 0) >= cap
        if crowded:
            if deferred_ids is not None:
                deferred_ids.add(getattr(c, "chunk_id", None))
            deferred.append(c)
            continue
        taken[fam] = taken.get(fam, 0) + 1
        lead.append(c)
    return lead + deferred


def share_characters(
    ordered: Sequence[Any], *, budget: int, share: float, competing: bool,
    deferred_ids: set | None = None,
) -> list:
    """Rule (b) counted in characters. Walks ``ordered`` as the budget walk
    will (a chunk that does not fit the room left is passed over in place);
    a chunk whose family already holds a chunk, and that would take the
    family past ``share`` of ``budget`` characters, is deferred behind the
    rest, in order. Chunks already in ``deferred_ids`` stay deferred. Nothing
    is dropped, and nothing moves when no other family competes."""
    if not competing:
        return list(ordered)
    already = set(deferred_ids or ())
    limit = share * budget
    room = budget
    held: dict[str, int] = {}
    lead: list = []
    tail: list = []
    for c in ordered:
        size = len(getattr(c, "text", "") or "")
        cid = getattr(c, "chunk_id", None)
        if cid in already:
            tail.append(c)
            continue
        if size > room and lead:
            lead.append(c)
            continue
        fam = family_key(c)
        if fam in held and held[fam] + size > limit:
            if deferred_ids is not None:
                deferred_ids.add(cid)
            tail.append(c)
            continue
        lead.append(c)
        room -= size
        held[fam] = held.get(fam, 0) + size
    return lead + tail


# --- (a) character budget --------------------------------------------------

def _walk_chars(band: Sequence[Any], room: int, keep_first: bool) -> tuple[list, list]:
    """Greedy walk of one tie band through ``room`` characters: (kept,
    skipped). With ``keep_first`` the band's first chunk is kept whatever
    its size (the top excerpt of the whole set)."""
    kept: list = []
    skipped: list = []
    for c in band:
        size = len(getattr(c, "text", "") or "")
        if size > room and not (keep_first and not kept):
            skipped.append(c)
            continue
        kept.append(c)
        room -= size
    return kept, skipped


def fit_char_budget(
    chunks: Sequence[Any], budget: int, skipped: list | None = None,
    *, tie_key: Callable[[Any], Any] | None = None,
    band_of: Callable[[Any], Any] = _score, tie_lost: set | None = None,
) -> list:
    """Whole chunks, in order, while they fit ``budget`` characters. The first
    chunk is always kept (an over-long top excerpt is still the best one);
    a chunk that does not fit is skipped so a shorter later one may (and
    appended to ``skipped`` when given).

    Each run of adjacent chunks with one ``band_of`` value is an exact tie.
    A tie keeps its order unless the walk in that order keeps one member and
    skips another; only then, and only when ``tie_key`` is given, is that tie
    reordered by ``tie_key`` and walked again. The id of each member skipped
    after a reorder that would have fit the room alone goes to ``tie_lost``."""
    order = list(chunks)
    kept: list = []
    used = 0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and band_of(order[j]) == band_of(order[i]):
            j += 1
        band = order[i:j]
        room = budget - used
        band_kept, band_skipped = _walk_chars(band, room, not kept)
        if tie_key is not None and band_kept and band_skipped:
            band = sorted(band, key=tie_key)
            band_kept, band_skipped = _walk_chars(band, room, not kept)
            if tie_lost is not None:
                tie_lost.update(
                    getattr(c, "chunk_id", None) for c in band_skipped
                    if len(getattr(c, "text", "") or "") <= room)
        if skipped is not None:
            skipped.extend(band_skipped)
        kept.extend(band_kept)
        used += sum(len(getattr(c, "text", "") or "") for c in band_kept)
        i = j
    return kept


def select_handed(
    chunks: Sequence[Any], *, k: int, reasons: dict[str, str] | None = None,
    tie_key: Callable[[Any], Any] | None = None,
) -> tuple[list, dict[str, Any]]:
    """The handed set: diversified, then fitted to the character budget.

    Returns ``(handed, allotment)`` where ``allotment`` is what the trace
    prints: the budget, the characters used, the per-layer and per-family
    counts. When ``reasons`` is a dict, each chunk left out gets its code
    (chunk_id -> ``duplicate`` | ``family_share`` | ``char_budget`` |
    ``score_tie_overlap``) where it is left out: ``family_share`` is a chunk
    the diversity pass deferred behind other documents that then did not fit
    the character budget; ``score_tie_overlap`` a chunk of an exact score
    tie the budget split, that would have fit alone, and that ``tie_key``
    (when given; the question's words in the passage, then the chunk id)
    put behind the tie's other members."""
    budget = char_budget()
    deferred_ids: set = set()
    ordered = diversify(chunks, slots=k, share=doc_share(), margin=doc_margin(),
                        reasons=reasons, deferred_ids=deferred_ids)
    ordered = share_characters(
        ordered, budget=budget, share=doc_share(),
        competing=len(_competing_families(chunks, doc_margin())) > 1,
        deferred_ids=deferred_ids)
    skipped: list = []
    tie_lost: set = set()
    # A tie band never spans a deferral: the tie-break must not move a chunk
    # the diversity pass deferred ahead of the other documents.
    handed = fit_char_budget(
        ordered, budget, skipped, tie_key=tie_key, tie_lost=tie_lost,
        band_of=lambda c: (_score(c), getattr(c, "chunk_id", None) in deferred_ids))
    if reasons is not None:
        for c in skipped:
            cid = getattr(c, "chunk_id", None)
            reasons[cid or ""] = (
                "score_tie_overlap" if cid in tie_lost
                else "family_share" if cid in deferred_ids else "char_budget")
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


class SearchTurn:
    """What the model's own search tool may still do in one chat turn (S-R9).

    One object per turn, made by the chat loop and passed to every tool call:

    * ``cap`` is the declared number of searches a turn may run
      (AGENT_SEARCH_TOOL_CAP). ``admit()`` counts a search and refuses the
      ones past the cap, so a batch of parallel calls cannot run more
      searches than declared (the loops used to count them and run them all).
    * ``handed`` holds the ids of the chunks already in front of the model
      this turn: the first-pass excerpts, then every chunk a search handed.
      A search skips them, so its rows carry passages the model has not
      read yet instead of repeating the ones it has.

    It reads ids and counts only; it never sees a question or chunk text.
    """

    def __init__(self, cap: int, handed: Sequence[Any] = ()) -> None:
        self.cap = max(0, int(cap))
        self.calls = 0
        self.handed: set[str] = {str(cid) for cid in handed if cid}
        #: Chunks the latest search reached but left out as already handed.
        self.last_skipped = 0

    @classmethod
    def from_audit(cls, cap: int, audit: dict[str, Any] | None) -> SearchTurn:
        """A turn whose handed set starts as the pre-injected chunks of
        ``audit`` (the record ``_rag_inject`` returns)."""
        chunks = (audit or {}).get("chunks") or []
        return cls(cap, [c.get("chunk_id") for c in chunks if isinstance(c, dict)])

    @property
    def spent(self) -> bool:
        return self.calls >= self.cap

    def admit(self) -> bool:
        """Count one search call; False when the turn's cap is already used."""
        if self.spent:
            return False
        self.calls += 1
        return True

    def record_search(self, handed_ids: Sequence[Any], skipped: int) -> None:
        """What one search did: the chunks it handed join the handed set;
        ``skipped`` is how many it left out because they were handed before."""
        self.handed.update(str(cid) for cid in handed_ids if cid)
        self.last_skipped = max(0, int(skipped))
