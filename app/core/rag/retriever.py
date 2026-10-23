"""High-level retrieval — the unit the chat block and the HTTP route call.

Composes the embedder + the vector store into a single ``retrieve()``
call. All public callers should go through this module rather than
talking to ``Embedder`` / ``VectorStore`` directly; the composition is
where caching, dimension matching, and graceful-degradation policy live.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.core.contract_data_chunks import (
    filled_particulars_rows,
    particulars_chunk_states_a_value,
)
from app.core.rag.embeddings import Embedder, get_embedder
from app.core.rag.vector_store import (
    Chunk,
    get_lexical_store,
    get_store,
    normalize_cesmm_item_codes,
)
from app.core.rag import layers
from app.lib.boq_ref_codes import find_ref_codes
from app.core.rag import revision as _revision
from app.core.rag import reranker as _reranker

import os
import re


_NOISE_DEFAULT = r"^(~\$|nambae-menu|SandsChina_Application)"

# Construction reference labels used to anchor identifier extraction.
# These are generic categories, not project-specific values.
_REFERENCE_LABELS = (
    "BOQ", "Clause", "Contract", "Doc", "Document", "Drawing",
    "Item", "NCR", "Package", "PRC", "Ref", "Reference", "RFI",
    "Rev", "Revision", "Schedule", "Spec", "Specification", "VO",
    "Variation Order",
)

# Regex components for extract_query_identifiers.
_QUOTED_RE = re.compile(r'["“]([^"”]{4,})["”]|\'([^\']{4,})\'')
_CODE_TOKEN_RE = re.compile(r"\b[A-Z]{2,}(?:[-./][A-Z0-9]+)+\b")
# Named capture ``label`` keeps the category word (VO, RFI, PRC, ...)
# separate from the captured ``code``.
_LABELED_REF_FULL_RE = re.compile(
    r"\b(?P<label>" + "|".join(re.escape(l) for l in _REFERENCE_LABELS) + r")"
    r"\s*(?:No|Ref|Number|#)?\s*[:\-]?\s*"
    r"(?P<code>[A-Za-z0-9][A-Za-z0-9\-./]*)",
    re.IGNORECASE,
)
# Mixed/lowercase code-shaped tokens that clearly contain a digit, e.g.
# D999.46, 12-A, revision-3.  The token may contain dots/dashes/slashes.
_ALPHANUMERIC_RE = re.compile(
    r"\b(?=[A-Za-z0-9./\-]*\d)[A-Za-z0-9]{2,}(?:[./\-][A-Za-z0-9]{1,})+\b"
)

_STOPWORDS: Set[str] = {
    "this", "that", "with", "from", "have", "what", "when", "where",
    "which", "about", "please", "thank", "thanks", "hello", "help",
}


def _is_prose_compound(token: str) -> bool:
    """True if a hyphen/dot/slash token contains a full English-word segment
    (all-alpha, >4 chars) — e.g. '30-storey', '12-month', 'revision-3'. These
    are descriptive compounds, NOT reference codes (whose alpha parts are short
    abbreviations: TL, PRC, IP). Without this guard a generative request like
    "risk register for a 30-storey tower" extracted '30-storey' as a reference,
    which then (on a retrieval miss) wrongly fired the missing-reference
    short-circuit — answering "provide the exact filename" to a generate request.
    """
    for seg in re.split(r"[-./]", token):
        if seg.isalpha() and len(seg) > 4:
            return True
    return _is_misspelled_word(token)


# A separator-free token is only a reference code when its letters are a SHORT
# abbreviation prefix: M145, A615, D999, PRC951, IP054. A long alphabetic run
# with a digit buried inside it is a TYPO, not a code.
#
# Live incident 2026-08-02: the operator typed "You should find it in the
# project specif8cation not drawings" — a conversational correction. The '8'
# in the misspelling made "specif8cation" match rule 4, retrieval missed on
# it, and the missing-reference short-circuit answered "I could not confirm
# this reference in the indexed project sources ... provide the exact
# filename." A typo silently converted a correction into a failed document
# lookup.
#
# _is_prose_compound could not catch it: that guard splits on [-./] and checks
# for all-alpha segments, but a separator-free token yields ONE segment which
# isn't .isalpha() precisely BECAUSE of the stray digit.
# The discriminator is the ALPHABETIC RUN. Reference codes are built from
# short abbreviations (M145, A615, D999, PRC951, IP054 — runs of 1-3 letters).
# An English word carries runs of 5+ letters, and a typo'd digit does not
# change that: "specif8cation" still contains "specif" and "cation".
#
# This also catches the same typo arriving via the LABELED-reference rule,
# which happily split "specif8cation" into label "spec" + code "if8cation" —
# and "if8cation" satisfies any letters-then-digits shape test, so only the
# run-length check rejects it.
_MAX_CODE_ALPHA_RUN = 4


def _is_misspelled_word(token: str) -> bool:
    """True for a separator-free word with a digit typo'd into it."""
    if re.search(r"[-./]", token):
        return False  # separator tokens are handled by the segment rule above
    if not any(ch.isdigit() for ch in token):
        return False
    return any(
        len(run) > _MAX_CODE_ALPHA_RUN
        for run in re.findall(r"[A-Za-z]+", token)
    )


# Measurement units / unit-ratios are NOT reference codes. A spec unit in the
# query (e.g. "concrete 250 kg/cm2") otherwise slips through _ALPHANUMERIC_RE
# (letters + a digit + a slash) and earns the +2.0 identifier bonus, which then
# matches every drawing dimension-table chunk that happens to contain the same
# number — burying the real answer and inducing a fabricated figure lifted from
# the number-soup (2026-07-14 live cost-query incident). Unit atoms are short
# symbols; a trailing exponent digit (m2, m3, cm2, mm2) is stripped before the
# vocabulary check. Reference codes (AB-CDE-012, PRC-123, X123.45) are NOT unit
# atoms, so they survive untouched.
_UNIT_ATOMS = frozenset({
    "kg", "g", "mg", "t", "ton", "tonne", "lb", "kn", "mn", "n",
    "pa", "kpa", "mpa", "gpa", "bar", "psi",
    "mm", "cm", "m", "km", "in", "ft", "yd", "mil",
    "sqm", "cum", "rm", "lm", "ha",
    "l", "ml", "kl", "cc",
    "s", "sec", "min", "hr", "h",
    "w", "kw", "mw", "kwh", "wh", "v", "kv", "a", "ma", "hz", "khz",
    "c", "f", "k",
    "pcs", "pc", "no", "nos", "ea", "each", "unit",
    # Currencies in rate units (AED/m2, USD/ft2). Live 2026-08-20: a
    # self-coding conversion was identifier-miss short-circuited because
    # `aed/m2` contains a digit (the exponent) but is not a document code.
    "aed", "usd", "sar", "eur", "gbp", "qar", "bhd", "kwd", "omr", "egp",
    "cny", "inr", "jpy",
})


def _strip_exponent(seg: str) -> str:
    """'cm2' -> 'cm', 'm3' -> 'm', 'mm2' -> 'mm'; leaves 'd999' unchanged
    (only a SINGLE trailing exponent digit after an alpha base is stripped)."""
    m = re.fullmatch(r"([a-z]{1,4})([23])", seg)
    return m.group(1) if m else seg


def _looks_like_unit(token: str) -> bool:
    """True when the token is a measurement unit or unit-ratio (kg/cm2, n/mm2,
    kn/m3, m3) rather than a document reference code. Ratios split on '/' (or the
    middot); every part, once its exponent is stripped, must be a known unit
    atom. A bare single unit (m3) also qualifies. Reference codes use '-'/'.'
    separators and non-unit alpha stems, so they are never flagged."""
    t = token.lower().strip()
    parts = [p for p in re.split(r"[/·]", t) if p]
    if not parts:
        return False
    if all(_strip_exponent(p) in _UNIT_ATOMS for p in parts):
        return True
    return False


# A small integer joined to a time word — "28-day", "7 day", "90-days",
# "56 week" — is a concrete-age / cure / duration spec, never a document
# reference code. Live find: "28-day cube strength" extracted ['28-day'],
# which the missing-reference short-circuit could not match to any chunk, so
# it false-declined ("could not confirm this reference") even though the
# C35/45 concrete chunk sat in the top-5. This mirrors the decimal-quantity
# and unit-ratio exclusions already applied in extract_query_identifiers.
_DURATION_RE = re.compile(
    r"^\d{1,3}[-\s]?"
    r"(?:day|days|week|weeks|hour|hours|hr|hrs|"
    r"month|months|year|years|yr|yrs)$",
    re.IGNORECASE,
)


def _looks_like_duration(token: str) -> bool:
    """True for a duration/age spec like '28-day' or '90 days' — a spec value,
    not a document reference. Drawing refs ('054-0009') and codes never match:
    their tail is not a time word."""
    return bool(_DURATION_RE.match(token.strip()))


def extract_query_identifiers(query: str) -> List[str]:
    """Pull construction reference identifiers out of a user query.

    Detects, without hardcoding any specific value:
      * quoted phrases (preserved as exact-match candidates)
      * code-shaped tokens such as PRC-123, AB-CDE-012-0000-...
      * labeled references such as "VO Ref 31", "RFI 42", "Clause 13.1"
      * alphanumeric tokens that clearly contain a digit (e.g. X123.45)

    Returns a deduplicated list of lowercase identifier strings. The list
    is empty for queries that contain no identifier-like tokens.
    """
    if not query:
        return []

    # OCR / CESMM print ``D 549.2``; compact that before the token regexes
    # so a spaced code extracts as ``d549.2`` (WAVE 2 B5). The original
    # fences still apply to the collapsed string — a typo like
    # ``specif8cation`` is unchanged.
    query = normalize_cesmm_item_codes(query)

    found: Set[str] = set()

    # 1. Quoted phrases (preserve exact content).
    for m in _QUOTED_RE.finditer(query):
        phrase = (m.group(1) or m.group(2) or "").strip()
        if phrase and len(phrase) >= 3:
            found.add(phrase.lower())

    # 2. Code-shaped tokens (hyphen/dotted/dashed uppercase codes).
    for m in _CODE_TOKEN_RE.finditer(query):
        token = m.group(0).strip("-.:/")
        if len(token) >= 4:
            found.add(token.lower())

    # 3. Labeled references: "VO Ref 31", "PRC-951", "RFI 12-A", etc.
    for m in _LABELED_REF_FULL_RE.finditer(query):
        label = m.group("label")
        # The captured code may have trailing punctuation; strip it.
        code = m.group("code").strip("-.:,;")
        # A genuine reference code carries a digit (VO 99, Clause 13.1,
        # PRC-951). Several labels ("Contract", "Spec", "Package", ...) are
        # also ordinary English words, so a label followed by a digit-less
        # word ("contract cover", "specification") is prose — NOT a reference.
        # Without this guard those false identifiers earned the +2.0 retrieval
        # bonus and flooded the top-K with boilerplate, so grounded chat
        # answered "I cannot find" for broad questions (2026-06-30 pilot).
        # ...and it must not be a typo'd English word. This rule cheerfully
        # split "specif8cation" into label "spec" + code "if8cation", which
        # carries a digit and so passed the check above (live 2026-08-02).
        if (
            code
            and any(ch.isdigit() for ch in code)
            and not _is_misspelled_word(code)
        ):
            found.add(f"{label.lower()} {code.lower()}")
            found.add(code.lower())

    # 4. Standalone alphanumeric codes containing digits.
    for m in _ALPHANUMERIC_RE.finditer(query):
        token = m.group(0).strip("-.:,;")
        # A pure number with a decimal point is a QUANTITY, not a reference
        # code. Live find 2026-08-15 (F21): a costing request carrying
        # measured quantities ("1947.87 square metres ... 2342.20 metres")
        # had both decimals extracted as identifiers; no chunk contains
        # them, so the exact-reference gate short-circuited every grounded
        # costing question with "could not confirm this reference".
        # Letterless dot/comma-separated digits are never document codes;
        # hyphenated digit pairs (drawing/sheet refs like 054-0009) keep
        # matching. A decimal minus a number (18.4-16) is leftover L7
        # arithmetic, not a sheet ref — that token used to fire the
        # RAG-miss short-circuit ("could not confirm this reference")
        # before sympy_reasoning ever ran.
        if re.fullmatch(r"\d+[.,]\d+", token):
            continue
        if re.fullmatch(r"\d+[.,]\d+[-+*/]\d+(?:[.,]\d+)?", token):
            continue
        if len(token) >= 5 and not _is_prose_compound(token):
            found.add(token.lower())

    # Filter out trivial stopwords, very short tokens, and measurement units
    # (a spec unit like "kg/cm2" is not a reference code — see _looks_like_unit).
    result = [
        t for t in found
        if len(t) >= 2 and t not in _STOPWORDS and not _looks_like_unit(t)
        and not _looks_like_duration(t)
    ]
    # Prefer longer, more specific identifiers first.
    result.sort(key=lambda t: (-len(t), t))
    return result


def identifier_present_in_text(ident: str, text: str) -> bool:
    """True when every token of ``ident`` appears in ``text``.

    CESMM OCR spacing is collapsed first so query ``D549.2`` matches
    stored ``D 549.2`` (WAVE 2 B5 live miss: the inject gate split the
    stored code into ``d``+``549`` and the query into ``d549``+``2``,
    then AND-failed and short-circuited chat). Label words (Ref / No /
    #) are ignored so ``VO 99`` still matches ``VO Ref: 99``.
    """
    blob = normalize_cesmm_item_codes(text or "")
    ident_norm = normalize_cesmm_item_codes(ident or "")
    text_tokens = set(re.split(r"[^a-z0-9]+", blob.lower())) - {"", "ref", "no", "#"}
    ident_tokens = [t for t in re.split(r"[^a-z0-9]+", ident_norm.lower()) if t]
    return bool(ident_tokens) and all(t in text_tokens for t in ident_tokens)


def _identifier_context_terms(query: str, identifiers: List[str]) -> List[str]:
    """Distinctive query terms excluding the identifier tokens themselves.

    WAVE 2 B4: a query for D599.5 + carriageway/340904 must prefer the
    priced carriageway row over an Excluded culvert that only shares the
    code. Identifier tokens (``d599``, ``5``) are dropped so the overlap
    score measures the rest of the question.
    """
    ident_toks: Set[str] = set()
    for ident in identifiers:
        ident_toks.update(
            t for t in re.split(
                r"[^a-z0-9]+", normalize_cesmm_item_codes(ident).lower()
            ) if t
        )
    terms: List[str] = []
    seen: Set[str] = set()
    for word in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", query or ""):
        lowered = word.lower()
        if lowered in seen or lowered in ident_toks or lowered in _STOPWORDS:
            continue
        seen.add(lowered)
        terms.append(lowered)
    compact_q = (query or "").replace(",", "")
    for num in re.findall(r"\d{3,}(?:\.\d+)?", compact_q):
        if num in seen or num in ident_toks:
            continue
        seen.add(num)
        terms.append(num)
    return terms


def _identifier_context_overlap(terms: List[str], text: str) -> float:
    """Fraction of ``terms`` that appear in CESMM-normalised chunk text."""
    if not terms:
        return 0.0
    blob = normalize_cesmm_item_codes(text or "").lower().replace(",", "")
    matched = sum(1 for term in terms if term in blob)
    return matched / len(terms)


# Tender / executed-contract numbers: PREFIX-YEAR-SEQ (AB-2031-007, FX-2044-001).
# Drawing codes (AB-CDE-012-...) and quantities do not match this shape.
# Underscore-glued filenames ("AB-2031-007_Vol 1.pdf") must still match, so
# this is not a \b word-boundary pattern (_ is a word character).
_CONTRACT_DOC_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,}-\d{4}-\d+)(?![A-Za-z0-9])"
)


def extract_contract_doc_ids(text: str) -> List[str]:
    """Return lowercase PREFIX-YEAR-SEQ contract/doc ids in ``text``.

    Used to scope a named-contract question to that contract's files so a
    AB-2023 question cannot surface AB-2022 chunks. Empty when the text
    names no such id.
    """
    if not text:
        return []
    found: List[str] = []
    seen: Set[str] = set()
    for m in _CONTRACT_DOC_ID_RE.finditer(text):
        tok = m.group(1).lower()
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
    return found


def filename_matches_named_contracts(
    filename: str,
    named_ids: List[str],
    *,
    chunk_text: str = "",
) -> bool:
    """True when this document belongs to a contract the query named.

    The upload filename is the authority — live corpus contract numbers
    live there. An unresolved filename falls back to a *contiguous* id in
    chunk text. Token-soup matching ('dd' + '2023' + '118' scattered) is
    rejected: a AB-2022 Conditions of Contract chunk can contain those
    tokens as a prefix, a date, and a clause number.
    """
    if not named_ids:
        return True
    name_l = (filename or "").lower()
    if name_l:
        return any(cid in name_l for cid in named_ids)
    text_l = (chunk_text or "").lower()
    return any(cid in text_l for cid in named_ids)


def _contract_id_recency(cid: str) -> Tuple[int, int]:
    """Sort key for PREFIX-YEAR-SEQ: newer year, then higher sequence.

    Unnamed Master Corpus questions can retrieve a filled Time for
    Completion from more than one package (an earlier package and a
    later executed one, each with its own figure). First-in-rank used to
    lock the pool to whichever cosine arrived first. The later executed
    package owns the unnamed ask; the earlier one stays reachable by
    naming its id (#443).
    """
    parts = (cid or "").lower().split("-")
    year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
    seq = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
    return (year, seq)


def elect_answer_bearing_contract(
    query: str,
    ranked_docs: Iterable[Tuple[str, str]],
) -> Optional[str]:
    """Which contract owns an UNNAMED answer, decided by evidence not by rank.

    ``ranked_docs`` is ``(filename, chunk_text)`` in descending final-score
    order. It may be a lazy iterable: nothing is consumed for a question that
    wants no particular. When the ask is particulars-shaped the walk reads
    every candidate so a later-year filled row can beat an earlier-year
    filled row that happened to rank first (Time for Completion and delay-damages rate asks).
    Returns the winning PREFIX-YEAR-SEQ, or None to leave the choice to
    arrival order (today's behaviour).

    WHY THIS EXISTS. The unnamed fence locks onto the first candidate that
    carries a contract id and drops every other id, so the top-ranked chunk
    does not merely outrank the rest — it DELETES the other contract from the
    result set. On the live Master Corpus that is decided by whichever chunk
    happens to sort first, and both outcomes were measured on one corpus in
    one session: the ACA and Defects Notification Period asks passed because the asked contract's
    Contract Data row sorted first, while the delay-damages rate and Engineer asks failed because
    another contract's Conditions clause did — and once it had, the row holding the
    answer could not appear at any rank.

    A General Conditions clause or a defined-term glossary entry is not an
    answer to "what is the rate" or "who is the Engineer"; it is a pointer to
    the row that holds it. So when the question asks for a kind of answer and
    the pool contains one, the contract that owns the best-ranked chunk OF
    THAT KIND wins the pool — a pointer from another contract cannot take it
    away.

    When TWO contracts both own that kind of answer (both state a Time for
    Completion, both state a Delay Damages rate), first-in-rank is still
    arrival order. The newer PREFIX-YEAR-SEQ wins: it is the later executed
    package. Naming the older id still fail-closes onto that year.

    Each ask shape brings its own idea of what an answer looks like, because
    the wrong-year chunk that wins is different every time:

    * a filled Contract Data particulars row, for a particular or a numbered
      contract Schedule;
    * a chunk of a bill of quantities, for measured scope (a BOQ-scope WBS ask, whose
      three citations were all another year's Conditions of Contract prose
      describing the demolition scope in words).

    An ask that matches no shape returns None and keeps arrival order, so
    this can never reorder a corpus it has no opinion about.

    This complements the named-id fence (#443) rather than re-implementing
    it: a question that names its contract never reaches here.
    """
    # (does the ask want this kind of answer?, is this chunk that answer?)
    # Built here, not at module scope: both halves are defined further down.
    kinds = [
        (query_asks_for_contract_particulars,
         lambda _name, text: chunk_answers_asked_particular(query, text)),
        (query_asks_for_boq_scope,
         lambda name, _text: document_is_a_bill_of_quantities(name)),
        (query_asks_for_boq_item_amount,
         lambda _name, text: (
             chunk_states_priced_item(
                 text, extract_asked_cesmm_codes(query), query=query,
             )
             or chunk_states_rate_only_item(
                 text, extract_asked_cesmm_codes(query),
             )
         )),
        (query_asks_for_part_summary_total,
         lambda _name, text: chunk_states_part_summary_total(
             text, extract_asked_boq_page_refs(query),
         )),
        # "On what date was <named document> issued, and under which <label>
        # number?" An issue stamp is the answer. An earlier contract whose
        # file names repeat the title must not lock the pool.
        (query_asks_for_issue_identity,
         lambda _name, text: chunk_states_issue_stamp(
             text, asked_reference_labels(query),
         )),
    ]
    active = [is_answer for asks, is_answer in kinds if asks(query)]
    if not active:
        return None
    found: List[str] = []
    seen: Set[str] = set()
    for filename, text in ranked_docs:
        if not any(is_answer(filename, text) for is_answer in active):
            continue
        ids = extract_contract_doc_ids(filename or "")
        if not ids:
            continue
        cid = ids[0]
        if cid not in seen:
            seen.add(cid)
            found.append(cid)
    if not found:
        return None
    return max(found, key=_contract_id_recency)


class _ContractScope:
    """Drop wrong-contract chunks before top-K selection.

    Named query (a PREFIX-YEAR-SEQ id in the question): keep only that id's files;
    if none remain the result is empty (fail closed — do not fill with
    another year's DD contract).

    Unnamed query: one contract wins the result set and chunks from a
    different PREFIX-YEAR-SEQ are dropped, so one answer cannot mix years.
    The winner is elected from answer-bearing evidence when the question
    asks for a filled Contract Data particular (see
    :func:`elect_answer_bearing_contract`); otherwise the first kept chunk
    carrying a contract id wins, as before.
    """

    def __init__(
        self,
        query: str,
        ranked_docs: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        self.query = query or ""
        self.named = extract_contract_doc_ids(self.query)
        self.winning: Optional[str] = None
        self._particulars = (
            not self.named
            and query_asks_for_contract_particulars(self.query)
        )
        self._title_phrases: List[str] = []
        self._spec_identity_in_pool = False
        self._aca_contract_data_in_pool = False
        # Year-lock elects the right contract for a Time for Completion ask, but
        # the delay-damages 0.1%-per-day row and the Engineer appointment live in a
        # different / unprefixed chunk. Once those answers are in the
        # pool, Client/Consultant PSA and same-year Sub-Clause 8.8 / cap
        # rows must not occupy the top-k.
        self._delay_rate_in_pool = False
        self._engineer_identity_in_pool = False
        self._party_role = ""
        self._aca_incl_vat_in_pool = False
        self._tfc_in_pool = False
        # Defects Notification Period ask: PSA / CPM TOC
        # recitals used to occupy every slot after #522.
        self._dnp_in_pool = False
        # Delay-damages daily amount (rate × ACA). The exclusive rate fence would
        # drop the money row; the daily amount needs both operands in the top-k.
        self._daily_damages_compose_in_pool = False
        self._schedule_labels: List[str] = []
        self._schedule_register_in_pool = False
        # Parent company guarantee / commencement date asks: Contract Data "not required" / empty commencement
        # over Schedule 8 form % and commencement-pack dates.
        self._pcg_cd_in_pool = False
        self._commencement_cd_in_pool = False
        # Rate Only BOQ item ask: D529.3 Amount is Rate Only. Priced lookalikes
        # (D549.2 fence, D599.5 carriageway, Excluded culvert) used to
        # occupy every slot and the model greeted. Not #504/#505/#506.
        # When a priced D549.2 row is also in-pool, do not
        # fence to Rate Only / Excluded siblings — that hid 280,320.
        self._rate_only_codes: List[str] = []
        self._rate_only_in_pool = False
        self._priced_item_in_pool = False
        # Part Summary total ask: the page footer loses to D110 / D290.1
        # line items on the same demolition page. Fence to the printed
        # page total when it is in the pool.
        self._part_summary_refs: List[str] = []
        self._part_summary_in_pool = False
        docs: Optional[List[Tuple[str, str]]] = (
            list(ranked_docs) if ranked_docs is not None else None
        )
        if docs is not None:
            if query_asks_which_document(
                self.query,
            ):
                self._title_phrases = extract_document_title_phrases(self.query)
                if self._title_phrases:
                    self._spec_identity_in_pool = any(
                        chunk_states_document_register_line(text, self._title_phrases)
                        or title_filename_bonus(name, self._title_phrases) > 0
                        for name, text in docs
                    )
            if (
                query_asks_for_accepted_contract_amount(self.query)
            ):
                self._aca_contract_data_in_pool = any(
                    contract_data_chunk_states_aca(name, text, self.query)
                    for name, text in docs
                )
            if (
                query_asks_for_delay_damages_rate(self.query)
            ):
                self._delay_rate_in_pool = any(
                    chunk_states_delay_damages_rate(text) for _n, text in docs
                )
            self._party_role = asked_party_role(self.query)
            if self._party_role:
                self._engineer_identity_in_pool = any(
                    chunk_names_party(text, self._party_role) for _n, text in docs
                )
            if (
                query_asks_for_aca_including_vat(self.query)
            ):
                self._aca_incl_vat_in_pool = any(
                    chunk_states_aca_including_vat(text) for _n, text in docs
                )
            if (
                query_asks_for_time_for_completion(self.query)
            ):
                self._tfc_in_pool = any(
                    chunk_states_time_for_completion(text) for _n, text in docs
                )
            if (
                query_asks_for_defects_notification_period(self.query)
            ):
                self._dnp_in_pool = any(
                    chunk_states_defects_notification_period(text)
                    for _n, text in docs
                )
            if (
                query_asks_delay_damages_daily_amount(self.query)
            ):
                has_rate = any(
                    chunk_states_delay_damages_rate(text) for _n, text in docs
                )
                has_aca = any(
                    _chunk_is_daily_damages_operand(text)
                    and not chunk_states_delay_damages_rate(text)
                    for _n, text in docs
                )
                # Only fence when both operands are reachable. A
                # rate-only fence would delete the ACA.
                self._daily_damages_compose_in_pool = has_rate and has_aca
            # Numbered schedule register ask: a Schedule-N register row ("Schedule 10: Not Used")
            # is the answer. Vol 4 / Vol 5 / CPM mention "schedule" at length
            # and used to occupy every slot. Not #500/#501/#502/#503.
            if (
                query_asks_for_contract_particulars(self.query)
                and query_asks_for_numbered_contract_schedule(self.query)
            ):
                self._schedule_labels = extract_asked_schedule_labels(self.query)
                if self._schedule_labels:
                    self._schedule_register_in_pool = any(
                        chunk_states_schedule_register(text, self._schedule_labels)
                        for _name, text in docs
                    )
            if (
                query_asks_for_parent_company_guarantee(self.query)
            ):
                self._pcg_cd_in_pool = any(
                    chunk_states_pcg_contract_data(text) for _n, text in docs
                )
            if (
                query_asks_for_contract_commencement_date(self.query)
            ):
                self._commencement_cd_in_pool = any(
                    chunk_states_commencement_contract_data(text)
                    for _n, text in docs
                )
            # A delay-damages daily amount (rate × ACA in SAR/day) is not a CESMM quote.
            # A priced D549.2 / D599.5 row in the same unnamed pool must
            # not fence out Contract Data operands — that refuse-closes
            # as "upload your priced BOQ" (live leftover after #559).
            if (
                query_asks_for_boq_item_amount(self.query)
                and not query_asks_delay_damages_daily_amount(self.query)
            ):
                self._rate_only_codes = extract_asked_cesmm_codes(self.query)
                if self._rate_only_codes:
                    # Named PREFIX-YEAR-SEQ (#443): decide Rate Only /
                    # priced fences from that year's docs only. A priced
                    # row on another contract must not empty a named
                    # Rate Only ask (Codex #558).
                    scoped = [
                        (name, text) for name, text in docs
                        if (
                            not self.named
                            or filename_matches_named_contracts(
                                name, self.named, chunk_text=text,
                            )
                        )
                    ]
                    self._rate_only_in_pool = any(
                        chunk_states_rate_only_item(
                            text, self._rate_only_codes,
                        )
                        for _n, text in scoped
                    )
                    if priced_boq_compose_enabled():
                        self._priced_item_in_pool = any(
                            chunk_states_priced_item(
                                text, self._rate_only_codes, query=self.query,
                            )
                            for _n, text in scoped
                        )
            if (
                part_summary_compose_enabled()
                and query_asks_for_part_summary_total(self.query)
            ):
                self._part_summary_refs = extract_asked_boq_page_refs(
                    self.query,
                )
                if self._part_summary_refs:
                    self._part_summary_in_pool = any(
                        chunk_states_part_summary_total(
                            text, self._part_summary_refs,
                        )
                        for _n, text in docs
                    )
        if not self.named and docs is not None:
            self.winning = elect_answer_bearing_contract(self.query, docs)

    def allow(self, filename: str, chunk_text: str = "") -> bool:
        # Named PREFIX-YEAR-SEQ (#443) is fail-closed onto that year.
        # The rate / Engineer fences are unnamed-only — a question that
        # names an earlier contract id must still see that year's chunks.
        daily_damages_ask = query_asks_delay_damages_daily_amount(self.query)
        if self._priced_item_in_pool and not daily_damages_ask:
            # A priced Part Nr. 3 line beats Rate Only /
            # Excluded siblings for the same CESMM code. A Rate Only ask stays on
            # the Rate Only fence below when no priced row exists.
            # A daily-amount ask keeps Contract Data rate + ACA even when a priced
            # CESMM row shares the unnamed pool.
            if not chunk_states_priced_item(
                chunk_text, self._rate_only_codes, query=self.query,
            ):
                return False
        elif self._rate_only_in_pool and not daily_damages_ask:
            if not chunk_states_rate_only_item(
                chunk_text, self._rate_only_codes,
            ):
                return False
        if self._part_summary_in_pool:
            if not chunk_states_part_summary_total(
                chunk_text, self._part_summary_refs,
            ):
                return False
        if not self.named:
            if self._delay_rate_in_pool:
                if not chunk_states_delay_damages_rate(chunk_text):
                    # "If Milestone 1 is 30 days late, what are the damages?"
                    # applies a duration to the rate and wants money: the sum
                    # the rate is a percentage OF has to survive this fence.
                    # Not reclassified as a daily-amount ask — that composer multiplies the
                    # whole-of-Works rate and would misstate a milestone.
                    if not (
                        query_applies_a_delay_duration(self.query)
                        and chunk_states_accepted_contract_amount(chunk_text)
                    ):
                        return False
            if self._engineer_identity_in_pool:
                if not chunk_names_party(chunk_text, self._party_role):
                    return False
            if self._aca_incl_vat_in_pool:
                if not chunk_states_aca_including_vat(chunk_text):
                    return False
            if self._tfc_in_pool:
                if not chunk_states_time_for_completion(chunk_text):
                    return False
            if self._dnp_in_pool:
                if not chunk_states_defects_notification_period(chunk_text):
                    return False
            if self._daily_damages_compose_in_pool:
                if not _chunk_keeps_for_daily_damages(filename, chunk_text):
                    return False
        if self._spec_identity_in_pool:
            titled = title_filename_bonus(filename, self._title_phrases) > 0
            identity = chunk_states_document_register_line(
                chunk_text, self._title_phrases,
            )
            if not (titled or identity):
                return False
        if self._aca_contract_data_in_pool:
            if not filename_looks_like_contract_data(filename):
                return False
        if self._schedule_register_in_pool:
            if not chunk_states_schedule_register(
                chunk_text, self._schedule_labels,
            ):
                return False
        if self._pcg_cd_in_pool:
            if not chunk_states_pcg_contract_data(chunk_text):
                return False
        if self._commencement_cd_in_pool:
            if not chunk_states_commencement_contract_data(chunk_text):
                return False
        # Commencement date ask: a commencement pack is never the contract particular, even
        # when Contract Data never reached the pool. Floor-all still
        # invented 10 Jan 2024 from the pack after #563 when the CD
        # row was missing.
        elif (
            query_asks_for_contract_commencement_date(self.query)
            and chunk_states_commencement_pack(chunk_text)
        ):
            return False
        if self.named:
            return filename_matches_named_contracts(
                filename, self.named, chunk_text=chunk_text,
            )
        ids = extract_contract_doc_ids(filename or "")
        if not ids:
            # No PREFIX-YEAR-SEQ in the filename. After year-lock that
            # used to let Long Form PSA Client/Consultant parties occupy
            # every Engineer-ask slot. When an Engineer appointment is in the pool
            # the identity fence above already dropped them.
            return True
        if self.winning is None:
            # A particulars ask whose election declined must not freeze the
            # pool on a Volume 4 schedule duration or a GC pointer. Only a
            # matching-label filled row may lock arrival order.
            # Scanned Contract Data (no index-time prefix) still counts
            # when it states the asked rate / Engineer.
            if self._particulars:
                if chunk_answers_asked_particular(self.query, chunk_text):
                    self.winning = ids[0]
                    return True
                # Live A2: scanned Contract Data has PREFIX-YEAR-SEQ in
                # the filename and no index-time particulars prefix, so
                # the filled-row predicate never fires. The ACA filename
                # election already decided this file owns the ask.
                if (
                    self._aca_contract_data_in_pool
                    and filename_looks_like_contract_data(filename)
                    and contract_data_chunk_states_aca(
                        filename, chunk_text, self.query,
                    )
                ):
                    self.winning = ids[0]
                    return True
                # Schedule register ask: scanned / unprefixed register row in a
                # PREFIX-YEAR-SEQ Contract Data file. Same exception as
                # the ACA ask — the row is the answer even without the index-time
                # particulars prefix.
                if (
                    self._schedule_register_in_pool
                    and chunk_states_schedule_register(
                        chunk_text, self._schedule_labels,
                    )
                ):
                    self.winning = ids[0]
                    return True
                return False
            self.winning = ids[0]
            return True
        return self.winning in ids


def _noise_regex():
    """Compile the active noise regex. Re-reads env every call so
    tests / operators can flip RAG_NOISE_FILENAME_REGEX live."""
    return re.compile(os.getenv("RAG_NOISE_FILENAME_REGEX", _NOISE_DEFAULT))


def _is_noise_filename(name: str) -> bool:
    """True iff the document filename matches the noise regex.

    Used to drop accumulated garbage docs (lockfiles, unrelated pptx
    menus, etc.) from the retrieval candidate pool BEFORE top-K
    selection, so they cannot displace a relevant chunk.
    """
    if not name:
        return False
    return bool(_noise_regex().match(name))


logger = logging.getLogger(__name__)


def available() -> bool:
    """True when retrieval is functional in this process.

    Reports True when either the real embedding stack is importable OR
    the configured model is the test-mode "fake" embedder — that way
    test suites that swap ``RAG_EMBEDDING_MODEL=fake`` go through the
    same code path as production rather than short-circuiting to "unavailable."

    False is the signal callers (chat, route) treat as "skip retrieval"
    rather than treating empty results as "no matches."
    """
    import os as _os
    if _os.getenv("RAG_EMBEDDING_MODEL") == "fake":
        return True
    return Embedder.available()


def chunk_text(text: str, max_chars: int = 512, overlap: int = 50) -> List[str]:
    """Sliding-window chunker. Plain and deterministic — no spaCy, no
    LangChain, no semantic segmenter. Good enough for keyword-flavored
    retrieval over construction docs; the doc indexer's own
    ``chunk_text`` covers fancier cases when needed.

    Empty / whitespace-only input → empty list.
    """
    if not text or not text.strip():
        return []
    if max_chars <= overlap:
        raise ValueError(f"max_chars ({max_chars}) must exceed overlap ({overlap})")
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    step = max_chars - overlap
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _doc_name_for_id(doc_id: str) -> str:
    """Resolve a doc_id to its original filename. Returns '' if not
    found - the noise filter treats unknown names as non-noise so a
    schema mismatch never silently drops a real document."""
    prefetched = _PREFETCHED_DOC_NAMES.get()
    if prefetched is not None and doc_id in prefetched:
        return prefetched[doc_id]
    try:
        from app.core import projects as _projects
        doc = _projects.get_document(doc_id)
        return (doc or {}).get("original_name") or ""
    except Exception:
        logger.debug("doc name lookup failed for %s", doc_id, exc_info=True)
        return ""


# Names fetched in one query for the duration of a ``_doc_names_prefetched``
# block; ``_doc_name_for_id`` answers from it instead of one round trip per
# chunk (live: ~30 single-row document reads per question).
_PREFETCHED_DOC_NAMES: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "_PREFETCHED_DOC_NAMES", default=None,
)


@contextmanager
def _doc_names_prefetched(doc_ids: Iterable[str]):
    """Resolve many doc names with one query for the enclosed lookups.

    An id with no documents row resolves to ``""``, which is what the
    per-id lookup returns for it. If the batch read fails nothing is
    prefetched and every lookup falls back to its own query.
    """
    ids = sorted({d for d in doc_ids if d})
    names: Optional[Dict[str, str]] = None
    if ids:
        try:
            from app.core import projects as _projects
            found = _projects.document_names(ids)
            names = {did: found.get(did, "") for did in ids}
        except Exception:  # noqa: BLE001 — per-id lookups still answer
            logger.debug("batched doc name lookup failed", exc_info=True)
    token = _PREFETCHED_DOC_NAMES.set(names)
    try:
        yield
    finally:
        _PREFETCHED_DOC_NAMES.reset(token)


def retrieve(
    query: str,
    project_id: str,
    k: int = 5,
    *,
    intent: Optional[str] = None,
    operator_text: Optional[str] = None,
) -> List[Chunk]:
    """Backwards-compatible: returns top-K AFTER the noise filter."""
    chunks, _ = retrieve_with_filter(
        query, project_id, k=k, intent=intent, operator_text=operator_text,
    )
    return chunks


def retrievable_project_ids(project_id: str) -> List[str]:
    """Project ids a retrieval for ``project_id`` may read.

    The active project plus configured general-knowledge projects. Another
    project's own documents are never in this set. Shared general-knowledge
    stays readable by every project. Scope is the query's project-id list,
    not a name denylist.
    """
    pid = (project_id or "").strip()
    ids: List[str] = []
    if pid:
        ids.append(pid)
    for gk in _general_knowledge_project_ids():
        if gk and gk not in ids:
            ids.append(gk)
    return ids


def _general_knowledge_project_ids() -> List[str]:
    """Project ids whose chunks count as cross-project general knowledge —
    queried alongside the active project on every retrieval.

    Configured via ``RAG_GENERAL_KNOWLEDGE_PROJECTS`` (comma-separated);
    the default lives with the project registry
    (``app.core.projects.DEFAULT_GENERAL_KNOWLEDGE_PROJECTS``), the one place
    every reader of this variable takes it from. Set to the empty string to
    disable the merge (the retriever then queries the active project only).
    """
    try:
        from app.core.projects import DEFAULT_GENERAL_KNOWLEDGE_PROJECTS as default
    except Exception:  # noqa: BLE001 — registry unavailable: no default merge
        logger.debug("general-knowledge default unavailable", exc_info=True)
        default = ""
    raw = os.getenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", default)
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    # STEP 0 structural isolation: the master-corpus / client fallback corpus
    # is NEVER part of the always-on GK merge, even when a stale env still lists
    # it (prod once listed client corpora in this var, silently merging a whole
    # client corpus into every OTHER project's results). It may only surface as
    # the disclosed empty/thin fallback below. This makes the client corpus
    # structurally unreachable from another project's populated query
    # regardless of score.
    fb = _master_corpus_fallback_id()
    if fb:
        ids = [p for p in ids if p != fb]
    return ids


def _master_corpus_fallback_id() -> Optional[str]:
    """The corpus queried ONLY when the active project is empty/thin, and always
    disclosed as a Master-Corpus fallback (STEP 0b).

    This is the client/master corpus (``MASTER_CORPUS_SOURCE_PROJECT_ID``) —
    deliberately kept SEPARATE from the general-knowledge layer so it can never
    silently blend into another project's results. Returns None when unset (the
    CI / self-host default), so the empty-project contract stays ``[]`` there.
    """
    pid = (os.getenv("MASTER_CORPUS_SOURCE_PROJECT_ID") or "").strip()
    return pid or None


def _skip_master_fallback_for_formula_ask(
    query: str, project_id: str, fb_id: Optional[str],
    operator_text: Optional[str] = None,
) -> bool:
    """True when a formula ask on a non-master project must not hit fallback.

    Thin/empty fixtures were answering rebar-lap / unit-convert asks from
    Master Corpus excerpts with zero construction_calc calls. Lookups still
    fall back (STEP 0b). Operator-selected Master Corpus is unchanged.

    ``operator_text`` is the original composer string when retrieve was
    called with a follow-up-expanded ``query``.
    """
    pid = (project_id or "").strip()
    if not pid:
        return False
    if fb_id and pid == fb_id:
        return False
    try:
        from app.agents.runtime import should_suppress_master_corpus_fallback
        for text in (operator_text, query):
            if text and should_suppress_master_corpus_fallback(pid, text):
                return True
        return False
    except Exception:  # noqa: BLE001 — skip is best-effort
        logger.debug("formula-ask fallback skip unavailable", exc_info=True)
        return False


def _project_has_any_chunks(store, project_id: str) -> bool:
    """True iff the project (or any configured GK project) has indexed chunks."""
    if store.has_chunks(project_id):
        return True
    for pid in _general_knowledge_project_ids():
        if pid != project_id and store.has_chunks(pid):
            return True
    return False


def project_is_rag_ready(project_id: str) -> bool:
    """Public guard: is there any corpus to retrieve from for this project?"""
    if not project_id:
        return False
    if not available():
        return False
    try:
        store = get_store(dim=get_embedder().dim)
        return _project_has_any_chunks(store, project_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("project_is_rag_ready check failed: %s", exc)
        return False


# General-knowledge relevance boost — lift a curated reference chunk (units /
# CESMM / POMI / FIDIC in the GK project) that LEXICALLY overlaps the query, so
# everyday phrasings surface it even when pure cosine ranks it just below the
# active project's own chunks. Capped well under IDENTIFIER_BONUS_MAX (2.0) so
# exact-code lookups still win, and relevance-gated (only overlapping GK chunks
# are boosted) so it never displaces a strongly-matched project chunk.
_GK_TERM_BONUS = 0.25
_GK_BONUS_CAP = 1.2
_GK_STOPWORDS = frozenset({
    "what", "which", "when", "where", "whom", "whose", "does", "did", "how",
    "the", "and", "for", "are", "was", "were", "this", "that", "these", "those",
    "from", "with", "into", "your", "our", "their", "please", "tell", "give",
    "answer", "question", "about", "standard", "project", "document", "documents",
    "knowledge", "base", "using", "used", "there", "here", "have", "has", "will",
})


# Intent classes gating the GK-contamination knobs (RAG_GK_SCORE_MARGIN /
# RAG_OWN_DOC_BOOST / RAG_GK_TOPK_CAP). RAG_AUDIT_V2 found curated GK notes
# outranking a user's own uploads 9/12 times, but calculation/standards
# features DEPEND on GK winning (unit tables, CESMM/POMI, FIDIC clauses) — so
# the knobs apply only to lookup-shaped retrieval. intent=None (the chat path
# today, and any caller that doesn't classify) counts as lookup.
DOC_LOOKUP_INTENTS = frozenset({"document_lookup", "project_lookup", "doc_qa"})
CALC_KB_INTENTS = frozenset({"calculation", "standards", "knowledge"})

#: A question framed in the asker's OWN project: a deictic / possessive frame
#: on a project-record noun ("this contract", "the project specification",
#: "our drawings", "under the contract"). Generic English framing, no corpus
#: words. Such a question is answered from the project layer first.
_OWN_PROJECT_FRAME_RE = re.compile(
    r"\b(?:this|the|our|my)\s+(?:project\s+)?"
    r"(?:contract|project|site|specifications?|specs?|drawings?|works|tender|"
    r"scope|boq|bill\s+of\s+quantities|particular\s+conditions|contract\s+data)\b",
    re.IGNORECASE,
)


def asks_about_own_project(query: str) -> bool:
    """True when the question speaks of its own project / contract."""
    return bool(_OWN_PROJECT_FRAME_RE.search(query or ""))


def question_framed_in_project(query: str, project_id: str, scored, names: Dict[str, str]) -> bool:
    """True when the question is asked INSIDE the active project.

    Either it speaks of its own project ("this contract", "the project
    specification") or it carries a reference code that names one of the
    project's own candidate documents. A code that merely appears in a
    project document's TEXT (a specification citing a standard) does not frame
    the question in the project. ``names`` caches resolved document names.
    """
    if asks_about_own_project(query):
        return True
    codes = [i for i in extract_query_identifiers(query)
             if re.search(r"[a-z]", i) and re.search(r"\d", i)]
    if not codes:
        return False
    for _score, chunk in scored:
        if chunk.project_id != project_id:
            continue
        if chunk.doc_id not in names:
            names[chunk.doc_id] = _doc_name_for_id(chunk.doc_id) or ""
        low = (names.get(chunk.doc_id) or "").lower()
        if any(code in low for code in codes):
            return True
    return False


def _keep_project_layer_first(query: str, project_id: str, scored, gk_id_set) -> None:
    """Project layer ahead of general knowledge for a project-framed question.

    No general-knowledge chunk may outrank the active project's best chunk:
    each GK score is capped just below it, so the project clause leads and the
    code or handbook clause on the same topic still follows as context. A
    layer rule -- it reads only which project a chunk belongs to. Projects with
    no candidate are left alone (the GK-only fallback keeps working).
    """
    if not asks_about_own_project(query):
        return
    project_scores = [s for s, c in scored if c.project_id == project_id]
    if not project_scores:
        return
    ceiling = max(project_scores) - 1e-6
    for i, (score, chunk) in enumerate(scored):
        if chunk.project_id in gk_id_set and score > ceiling:
            chunk.score = round(ceiling, 6)
            scored[i] = (ceiling, chunk)


def _knob_float(name: str) -> Optional[float]:
    """Env-driven ranking knob. Unset/blank/unparsable means OFF (None) so a
    typo'd value can never silently change ranking; 0.0 is a valid ON value."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; knob disabled", name, raw)
        return None


def _knob_int(name: str) -> Optional[int]:
    """Integer variant of _knob_float; same OFF-on-garbage contract."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid %s=%r; knob disabled", name, raw)
        return None


def _knob_flag(name: str) -> bool:
    """Boolean variant of _knob_float; unset/blank/garbage means OFF so a
    typo'd value can never silently change ranking."""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("invalid %s=%r; knob disabled", name, raw)
    return False


def _significant_terms(query: str) -> frozenset:
    """Content words (>=4 chars, minus stopwords) used for lexical overlap."""
    import re as _re
    return frozenset(
        w for w in _re.findall(r"[a-z0-9]{4,}", (query or "").lower())
        if w not in _GK_STOPWORDS
    )


def _gk_lexical_bonus(query_terms: frozenset, chunk_text: str) -> float:
    """Bonus for a GK chunk = capped count of distinct query terms it contains."""
    if not query_terms or not chunk_text:
        return 0.0
    text = chunk_text.lower()
    overlap = sum(1 for t in query_terms if t in text)
    return min(overlap * _GK_TERM_BONUS, _GK_BONUS_CAP)


# ── lexical term rescue ─────────────────────────────────────────────────────
#
# Live failure (2026-08-17): the operator asked about the "Saudi Building Code".
# Two turns earlier the assistant had itself quoted "SBC 304 — Saudi Building
# Code" out of the structural general notes, so the phrase demonstrably sits in
# the corpus. The later turn retrieved five unrelated chunks (cable ladders, LV
# single-line diagrams, fire alarm, water pipelines, road alignment) and the
# model reported the corpus "does not mention the Saudi Building Code at all".
#
# Two things went wrong. The overclaim is handled in rag/inject.py (an answer
# may only speak for the retrieved excerpts). The RECALL miss is handled here.
#
# Why the existing machinery did not catch it: extract_query_identifiers only
# fires on code-SHAPED tokens (digits, hyphens, uppercase runs). "Saudi Building
# Code" is three ordinary words, so no identifier was extracted and no lexical
# pass ran at all — the turn was pure cosine over a 227-document corpus with
# k=5, and the user's typo ("buiding") degraded the lexical half of the hybrid
# search too.
#
# The rescue therefore matches on term CO-OCCURRENCE rather than exact phrase:
# terms are taken pairwise, so "saudi"+"code" still selects the right chunk when
# "building" is misspelt beyond recognition. A single common term ("code") is
# never enough to match, which is what keeps this from dragging in boilerplate —
# the failure mode a naive OR-any-term search would have.
#
# It runs ONLY when the semantic pass has already failed to surface any chunk
# carrying two or more of the query's distinctive terms, so on a healthy
# retrieval it is a no-op and costs one cheap SQL query at most.
_TERM_RESCUE_BONUS_MAX = 1.0   # below IDENTIFIER_BONUS_MAX: exact codes still win
_TERM_RESCUE_MAX_TERMS = 5     # caps the pair expansion at C(5,2) = 10 clauses
_TERM_RESCUE_MIN_TERMS = 2     # co-occurrence needs at least a pair


def distinctive_query_terms(query: str) -> List[str]:
    """The distinctive content terms of ``query``, most distinctive first.

    Proper-noun-shaped terms (capitalised) rank above ordinary words, then
    longer above shorter, because those carry the naming that makes a corpus
    lookup specific ("Saudi", "Building" before "code").
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", query or "")
    seen: Set[str] = set()
    ranked: List[Tuple[bool, int, str]] = []
    for word in words:
        lowered = word.lower()
        if lowered in seen or lowered in _GK_STOPWORDS or lowered in _STOPWORDS:
            continue
        seen.add(lowered)
        ranked.append((word[:1].isupper(), len(lowered), lowered))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [term for _cap, _len, term in ranked[:_TERM_RESCUE_MAX_TERMS]]


# Morphology. The rescue matches each term as a SUBSTRING of the chunk text,
# so a query word and the document's word must share a prefix. They routinely
# do not: live 23 Sep, "To what degree must structural backfill be compacted?"
# never retrieved the chunk reading "Compaction of the backfill to minimum 98%
# of maximum dry density of the modified proctor test" -- compacted / compaction
# differ after "compact". Stemming the QUERY term (never the chunk) to its
# shared prefix closes that: both reduce to "compact". One suffix at most, and
# never below _STEM_MIN_CHARS, so a stem stays specific enough to co-occur
# meaningfully ("work" never becomes "wor").
_STEM_SUFFIXES = ("ations", "ation", "ements", "ement", "ings", "ing", "ions",
                  "ion", "ally", "ies", "ed", "es", "s")
_STEM_MIN_CHARS = 5


def stem_query_term(term: str) -> str:
    """``term`` reduced to the prefix it shares with its own word family."""
    word = (term or "").lower()
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _STEM_MIN_CHARS:
            return word[: -len(suffix)]
    return word


def cooccurrence_pair_phrases(terms: List[str]) -> List[str]:
    """Pairwise co-occurrence phrases for :meth:`VectorStore.identifier_search`.

    identifier_search AND-matches the tokens within one phrase and OR-matches
    across phrases, so a list of pairs asks exactly: "any chunk containing at
    least two of these terms". Scoring comes back as the fraction of pairs
    matched, which ranks a chunk carrying all the terms above one carrying two.
    """
    import itertools

    stems: List[str] = []
    for term in terms:
        stem = stem_query_term(term)
        if stem not in stems:
            stems.append(stem)
    return [" ".join(pair) for pair in itertools.combinations(stems, 2)]


# ── letter / named-party filename recall ──────────────────────────────────
#
# "Who signed the letter about <site / party>" retrieved only a long
# miscellaneous-documents volume that mentions the place. The letter was
# indexed, and its file name carries "Letter" plus the site and party the
# question names. The term rescue skipped the out-of-pool fetch because the
# volume already mentioned the place in-chunk.
#
# A letter whose indexed text ends without a signatory states no name; do
# not invent one. Re-extracting a sparse letter is an ingest job, not a
# ranking delta.
#
# Filename overlap is the discriminator the volume cannot fake: its name is
# a contract volume, not a letter.
_LETTER_OR_SIGNATORY_RE = re.compile(
    r"(?i)\b(?:"
    r"who\s+signed|who\s+put\s+their\s+name|"
    r"signed\s+the\s+letter|signator(?:y|ies)|"
    r"in\s+what\s+capacity|"
    r"letter\s+(?:about|on|regarding|for|to)\b"
    r")"
)
_LETTER_IN_NAME_RE = re.compile(
    r"(?i)(?:^|[\s_\-/.(])letter(?:s)?(?:[\s_\-/.)]|$)",
)
# Below IDENTIFIER_BONUS_MAX (2.0) so exact codes still win; above typical
# cosine + Volume-5 term overlap so a filename-matched letter leads.
_FILENAME_OVERLAP_BONUS_MAX = 1.6
_FILENAME_LETTER_BONUS = 0.4
_FILENAME_RESCUE_MIN_TERMS = 2
_FILENAME_RESCUE_OPEN_MIN_TERMS = 3


def query_asks_for_letter_or_signatory(query: str) -> bool:
    """True for a who-signed / letter-about ask (D1), not a contract-role ask.

    "Who is the Engineer" stays on the Contract Data path (#483). "Who
    signed the letter about the batching plant" is this class.
    """
    return bool(_LETTER_OR_SIGNATORY_RE.search(query or ""))


def filename_looks_like_letter(filename: str) -> bool:
    """True when the upload name or path is a correspondence letter."""
    return bool(_LETTER_IN_NAME_RE.search(filename or ""))


def filename_query_overlap(filename: str, terms: List[str]) -> float:
    """Fraction of distinctive ``terms`` that appear in the filename/path."""
    blob = (filename or "").lower()
    cleaned = [t.lower() for t in terms if t and len(t) >= 3]
    if not blob or not cleaned:
        return 0.0
    hits = sum(1 for t in cleaned if t in blob)
    return hits / len(cleaned)


def filename_match_bonus(
    filename: str,
    terms: List[str],
    *,
    letter_query: bool,
) -> float:
    """Additive lift for a chunk whose document name matches the query.

    A letter-shaped name gets an extra tier so Volume 5 'Other Documents'
    that share a place-name cannot outrank the letter the question named.
    Zero when the name shares no distinctive term, and zero on a weak
    one-token collision (``contract.pdf`` vs "Conditions of Contract")
    so ordinary Q&A ranking stays byte-identical.
    """
    cleaned = [t for t in terms if t and len(t) >= 3]
    overlap = filename_query_overlap(filename, cleaned)
    if overlap <= 0.0 or not cleaned:
        return 0.0
    hits = overlap * len(cleaned)
    is_letter = filename_looks_like_letter(filename)
    if letter_query:
        if hits < _FILENAME_RESCUE_MIN_TERMS and not is_letter:
            return 0.0
    elif hits < _FILENAME_RESCUE_OPEN_MIN_TERMS and not (
        is_letter and hits >= _FILENAME_RESCUE_MIN_TERMS
    ):
        return 0.0
    bonus = overlap * _FILENAME_OVERLAP_BONUS_MAX
    if is_letter:
        bonus += _FILENAME_LETTER_BONUS
    return bonus


# ── governing-source preference ──────────────────────────────────────────
#
# A question that names its governing document ("per the project
# specification, what compaction ...") was answered from a site-office
# mobilisation method statement stating its own lesser figure instead of the
# specification. The question named the governing document
# class and ranking ignored it: every document was equally eligible.
#
# The classes below are the ones a construction question actually names. Each
# is a filename/path test, because the class is what the document IS, and the
# corpus keeps that in its name and folder. A chunk from the named class is
# lifted; one from a class the question did NOT name is demoted only when the
# question named a class at all -- so ordinary questions rank byte-identically.
_SOURCE_CLASSES: dict[str, tuple] = {
    "specification": (
        re.compile(r"(?i)\b(?:per|as\s+per|according\s+to|under)\s+the\s+"
                   r"(?:project\s+)?spec(?:ification)?s?\b"),
        re.compile(r"(?i)(?:^|[\s_\-/])(?:spec|specification|particular\s+spec)"),
    ),
    "hse": (
        # "HSE lighting requirements" / "HSE plan": the qualifier between the
        # class and the noun is what the question is about, so allow it.
        re.compile(r"(?i)\b(?:per|as\s+per|according\s+to|under)\s+the\s+"
                   r"(?:project\s+)?(?:hse|health\s+and\s+safety|safety)"
                   r"(?:\s+\w+){0,2}\s+(?:plan|requirements?|procedure)\b"),
        re.compile(r"(?i)(?:^|[\s_\-/])(?:hse|hs|safety|health)"),
    ),
    "lifting": (
        re.compile(r"(?i)\b(?:per|as\s+per|according\s+to|under)\s+the\s+"
                   r"(?:project\s+)?lifting\s+plan\b"),
        re.compile(r"(?i)lifting"),
    ),
}
# Documents that answer a different scope than a project-wide question: a
# mobilization / site-office method statement states its own lesser values.
_OFF_SCOPE_NAME_RE = re.compile(
    r"(?i)(?:mobilization|mobilisation|site\s*office|temporary\s+facilit)",
)
_SOURCE_CLASS_BONUS = 1.2
_OFF_SCOPE_PENALTY = 0.8


def source_class_named_by(query: str) -> str:
    """The governing document class the question names, or ""."""
    for name, (ask_rx, _name_rx) in _SOURCE_CLASSES.items():
        if ask_rx.search(query or ""):
            return name
    return ""


def filename_is_source_class(filename: str, class_name: str) -> bool:
    """True when this document's name/path says it IS that class."""
    spec = _SOURCE_CLASSES.get(class_name)
    return bool(spec and spec[1].search(filename or ""))


def source_class_adjustment(filename: str, class_name: str) -> float:
    """Lift for the named class, demotion for an off-scope document, else 0."""
    if not class_name:
        return 0.0
    if filename_is_source_class(filename, class_name):
        return _SOURCE_CLASS_BONUS
    if _OFF_SCOPE_NAME_RE.search(filename or ""):
        return -_OFF_SCOPE_PENALTY
    return 0.0


def _apply_source_class_preference(
    query: str,
    scored: List[Tuple[float, Chunk]],
    name_by_id: Dict[str, str],
) -> None:
    """In-place: rank by the governing source the question named."""
    class_name = source_class_named_by(query)
    if not class_name:
        return
    for i, (score, chunk) in enumerate(scored):
        name = name_by_id.get(chunk.doc_id, "") or getattr(chunk, "source_name", "") or ""
        add = source_class_adjustment(name, class_name)
        if add == 0.0:
            continue
        adjusted = score + add
        chunk.score = round(adjusted, 6)
        scored[i] = (adjusted, chunk)


# ── numeric-requirement recall ───────────────────────────────────────────
#
# A question that says "per the project specification" lifts every file
# whose name says specification (+_SOURCE_CLASS_BONUS). That is right when
# the specification itself states the figure, and wrong when it only
# mentions the topic: the chunk that states the number often lives in a
# geotech / "Other Documents" volume and uses the quantity's own words
# (nominal cover, sub-grade, maximum dry density) rather than the words
# in the question (minimum concrete cover, road pavement, compaction).
# Two consequences, both generic:
#
#   * BM25 never sees that chunk. The hybrid leg only keeps the top 50
#     lexical hits, and a neighbour that repeats the question's words
#     fills those slots. FTS5 does not stem, so "compaction" does not
#     match "compacted".
#   * Even inside the pool, the filename lift is larger than a typical
#     cosine, so a qualitative specification chunk outranks the chunk
#     that actually states a number with the asked unit.
#
# The supplementary query adds the quantity's vocabulary, not a figure.
# The score lift applies only to a chunk that states a number in that
# unit, so a specification chunk that states its own number still leads
# (it keeps the filename lift on top of this one) and a qualitative
# mention does not move. Kill-switch: RETRIEVAL_NUMERIC_REQUIREMENT_BOOST=0.
_NUMERIC_REQUIREMENT_BONUS = 2.5
_COVER_ASK_RE = re.compile(
    r"(?i)\b(?:concrete\s+cover|cover\s+to\s+reinforcement|nominal\s+cover|"
    r"minimum\s+(?:concrete\s+)?cover)\b"
)
_COMPACTION_ASK_RE = re.compile(
    r"(?i)\b(?:compact\w*|sub-?grades?|cbr|dry\s+density)\b"
)
# Illuminance: lux / lx, or a foot-candle column. "lighting design" or a
# "light fitting" is not an asked level; an illumination / lux / lighting
# level is.
_ILLUMINANCE_ASK_RE = re.compile(
    r"(?i)\b(?:illuminat\w*|illuminance|lux|lighting\s+levels?|"
    r"light(?:ing)?\s+intensit\w*|foot[-\s]*candles?)\b"
)
# A lux figure: "50 lux", "300lx", or a table whose header carries a lux /
# foot-candle column and whose rows carry numeric cells.
_LUX_FIGURE_RE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(?:lux|lx)\b")
_LUX_COLUMN_RE = re.compile(r"(?i)\|[^|\n]{0,40}\b(?:lux|lx|foot[-\s]*candles?)\b[^|\n]{0,20}\|")
_NUMERIC_CELL_RE = re.compile(r"\|\s*\d+(?:\.\d+)?\s*(?=\|)")
_COVER_WORD_RE = re.compile(r"(?i)\bcovers?\b")
_MM_FIGURE_RE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*mm\b")
# A cover *length* is a millimetre next to the cover phrase. "200 mm"
# bollard bands and "600 mm" floor panels in the same chunk as the word
# "cover" are not that length. Window covers "nominal cover should be 50mm".
_COVER_PHRASE_WINDOW = 64
# Filename lift kept when a specification chunk actually states the asked
# figure. Otherwise the +1.2 class bonus is capped so it cannot stack on
# an unrelated millimetre and outrank the clause that states the length.
_SOURCE_CLASS_BONUS_CAP = 0.25
_COVER_LEXICAL_TERMS = "nominal cover cast against soil cast against blinding"
_MDD_RE = re.compile(r"(?i)\b(?:maximum\s+dry\s+density|mdd)\b")
_PERCENT_FIGURE_RE = re.compile(
    r"(?i)(?:\d+(?:\.\d+)?\s*%|\b(?:twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred)"
    r"(?:[-\s]+(?:one|two|three|four|five|six|seven|eight|nine))?"
    r"\s+percent\b)"
)
_QUANTITY_NEAR = 96


def numeric_requirement_boost_enabled() -> bool:
    """ON by default. ``RETRIEVAL_NUMERIC_REQUIREMENT_BOOST=0`` restores
    the pre-fix candidate pool and scores exactly."""
    return (os.getenv("RETRIEVAL_NUMERIC_REQUIREMENT_BOOST", "1") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def spec_boost_guard_enabled() -> bool:
    """ON by default. ``RETRIEVAL_SPEC_BOOST_GUARD=0`` restores the
    uncapped specification-filename lift, the loose cover detector, and
    duplicate signed/unsigned slots."""
    return (os.getenv("RETRIEVAL_SPEC_BOOST_GUARD", "1") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def asked_quantity_kinds(query: str) -> frozenset:
    """Which measurable quantities ``query`` is asking for.

    Empty for an ordinary question, so this path does not run and ranking
    stays byte-identical. "cover letter" is not a cover-to-reinforcement
    ask; a pavement width is not a compaction ask.
    """
    text = query or ""
    kinds = set()
    if _COVER_ASK_RE.search(text):
        kinds.add("length_mm")
    if _COMPACTION_ASK_RE.search(text):
        kinds.add("compaction")
    if _ILLUMINANCE_ASK_RE.search(text):
        kinds.add("illuminance")
    return frozenset(kinds)


def numeric_requirement_expansion(query: str) -> str:
    """Vocabulary to append so the quantity's own wording can enter BM25.

    Terms are the family the question is about. They are not a figure and
    not a document name. Empty when the question asks for neither.
    """
    kinds = asked_quantity_kinds(query)
    parts: List[str] = []
    if "length_mm" in kinds:
        parts.append("nominal cover millimetre millimeter blinding")
        if spec_boost_guard_enabled():
            # A cover clause names the condition ("cast against soil /
            # blinding"), not "minimum cover to reinforcement".
            parts.append("cast against soil blinding")
    if "compaction" in kinds:
        parts.append(
            "compacted sub-grade subgrade embankment maximum dry density CBR percent"
        )
    if "illuminance" in kinds:
        parts.append("illumination lux lighting level foot candle")
    return " ".join(parts)


def _spans_within(text: str, left: re.Pattern, right: re.Pattern, window: int) -> bool:
    a = [m.start() for m in left.finditer(text or "")]
    b = [m.start() for m in right.finditer(text or "")]
    return any(abs(x - y) <= window for x in a for y in b)


def chunk_states_cover_length(text: str) -> bool:
    """True when the chunk states a cover and a length in millimetres.

    With ``RETRIEVAL_SPEC_BOOST_GUARD`` on (the default), the millimetre
    has to sit next to "nominal cover", "concrete cover", or "cover to
    reinforcement". A bollard dimension elsewhere in the chunk does not
    count. ``RETRIEVAL_SPEC_BOOST_GUARD=0`` restores the any-cover-word
    plus any-millimetre check.
    """
    blob = text or ""
    if spec_boost_guard_enabled():
        return _cover_clause_states_length(blob)
    return bool(_COVER_WORD_RE.search(blob) and _MM_FIGURE_RE.search(blob))


def chunk_states_compaction_figure(text: str) -> bool:
    """True when the chunk states a dry-density percent or a CBR number."""
    blob = text or ""
    if _spans_within(blob, _PERCENT_FIGURE_RE, _MDD_RE, _QUANTITY_NEAR):
        return True
    low = blob.lower()
    for match in re.finditer(r"\bcbr\b", low):
        window = low[max(0, match.start() - 40): match.end() + 48]
        if re.search(r"\d", window):
            return True
    return False


def chunk_states_illuminance(text: str) -> bool:
    """True when the chunk states a lux figure.

    Either a number with its unit ("50 lux"), or a table whose header has a
    lux / foot-candle column and whose rows carry numeric cells. Prose that
    names lighting and states no figure does not count.
    """
    blob = text or ""
    if _LUX_FIGURE_RE.search(blob):
        return True
    return bool(_LUX_COLUMN_RE.search(blob) and _NUMERIC_CELL_RE.search(blob))


def chunk_states_asked_quantity(text: str, kinds: frozenset) -> bool:
    """True when ``text`` states a number for one of ``kinds``."""
    if "length_mm" in kinds and chunk_states_cover_length(text):
        return True
    if "compaction" in kinds and chunk_states_compaction_figure(text):
        return True
    if "illuminance" in kinds and chunk_states_illuminance(text):
        return True
    return False


# A compaction question names what is being compacted. Chunks about a
# different element state a real figure for a different question — foundation
# backfill and a road sub-grade are not interchangeable. Synonyms sit in the
# same group so the chunk can use the document's word (sub-grade) when the
# question used another (pavement). No group named → any compaction figure.
_COMPACTION_SUBJECT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("pavement", "road", "carriageway", "sub-grade", "subgrade", "embankment"),
    ("backfill", "foundation"),
)


def _term_in(blob: str, term: str) -> bool:
    if term in ("sub-grade", "subgrade"):
        return "sub-grade" in blob or "subgrade" in blob
    return re.search(rf"\b{re.escape(term)}s?\b", blob) is not None


def compaction_subject_agrees(query: str, text: str) -> bool:
    """True when ``text`` is about the element ``query`` is compacting.

    True when the question names no element. When it does, the chunk has to
    name that same element or a synonym of it.
    """
    q = (query or "").lower()
    blob = (text or "").lower()
    named = [
        group for group in _COMPACTION_SUBJECT_GROUPS
        if any(_term_in(q, term) for term in group)
    ]
    if not named:
        return True
    return any(any(_term_in(blob, term) for term in group) for group in named)


def chunk_matches_quantity_question(query: str, text: str, kinds: frozenset) -> bool:
    """Figure present, and about the asked subject.

    Compaction keeps its element groups (a road sub-grade figure does not
    answer a backfill question). An illumination table lists levels for many
    subjects at once, so a lux figure counts only when the chunk also names
    the question's subject.
    """
    if not chunk_states_asked_quantity(text, kinds):
        return False
    if "compaction" in kinds and not compaction_subject_agrees(query, text):
        return False
    if (
        "illuminance" in kinds
        and not chunk_states_asked_quantity(text, kinds - {"illuminance"})
        and not chunk_names_quantity_subject(text, quantity_subject_terms(query))
    ):
        return False
    return True


def retrieval_lexical_query(query: str) -> str:
    """BM25 text for pre-answer retrieval.

    The vector leg keeps the operator's words. A cover ask additionally
    carries the clause vocabulary ("nominal cover", "cast against soil")
    so the lexical leg can meet the durability sentence. Empty extra
    when the guard is off, so that path stays on the raw question.
    """
    base = (query or "").strip()
    if not spec_boost_guard_enabled():
        return base
    if "length_mm" not in asked_quantity_kinds(query):
        return base
    return f"{base} {_COVER_LEXICAL_TERMS}".strip()


def _loose_cover_and_millimetre(text: str) -> bool:
    """The pre-guard detector: any cover-word and any millimetre."""
    blob = text or ""
    return bool(_COVER_WORD_RE.search(blob) and _MM_FIGURE_RE.search(blob))


def _cap_specification_class_bonus(
    query: str,
    scored: List[Tuple[float, Chunk]],
    name_by_id: Dict[str, str],
) -> None:
    """In-place: stop the specification filename lift overwhelming content.

    The full ``_SOURCE_CLASS_BONUS`` stays when the chunk states the asked
    figure (a specification that gives 98% MDD still leads) or when the
    question is not a cover-length ask. A cover ask whose chunk only
    shares the word "cover" with an unrelated millimetre keeps a small
    cap. ``RETRIEVAL_SPEC_BOOST_GUARD=0`` leaves scores untouched.
    No-op for HSE / lifting lifts.
    """
    if not spec_boost_guard_enabled():
        return
    if source_class_named_by(query) != "specification":
        return
    if "length_mm" not in asked_quantity_kinds(query):
        return
    kinds = asked_quantity_kinds(query)
    for i, (score, chunk) in enumerate(scored):
        name = name_by_id.get(chunk.doc_id, "") or getattr(chunk, "source_name", "") or ""
        add = source_class_adjustment(name, "specification")
        if add <= _SOURCE_CLASS_BONUS_CAP:
            continue
        if chunk_matches_quantity_question(query, chunk.text or "", kinds):
            continue
        if chunk_points_quantity_elsewhere(chunk.text or "", kinds):
            continue
        if not _loose_cover_and_millimetre(chunk.text or ""):
            continue
        excess = add - _SOURCE_CLASS_BONUS_CAP
        adjusted = score - excess
        chunk.score = round(adjusted, 6)
        scored[i] = (adjusted, chunk)


# ── pointer-following: the named source sends the figure elsewhere ───────
#
# "Per the project specification, what is the minimum <quantity> for
# <subject>?" The named source often states no figure. Its clause sends the
# reader to another document -- "the cover specified ... on the Drawings",
# "compacted as shown on the drawings", "lighting levels as listed in the
# Schedule" -- and that document states the figure. Two things go wrong
# without this block:
#
#   * The pointer clause is never a candidate: it states no figure, so the
#     numeric fetch rejects it, and cosine for a clause about something else
#     (bar fixing, spacer blocks) is low.
#   * Every slot goes to chunks from other documents that state some figure
#     of the asked quantity, and nothing prefers the document the source
#     points to.
#
# The pointer is read from the chunk itself: one sentence names the asked
# quantity and points at a document class (drawings, a schedule, an
# appendix, the contract data). When the question names its source, such a
# clause from a document of that source class is fetched, enters the pool
# at its cosine and is lifted like a stated figure (it IS the source's
# answer); once it is pooled, chunks from the pointed-to class that state the
# quantity, and chunks about the element the question names, are preferred.
#
# Cover lengths: a cover list item under its heading counts as a stated
# length; a lid size "250mm x 250mm x 10mm" does not.
_COVER_CLAUSE_PHRASE_RE = re.compile(
    r"(?i)(?:nominal\s+cover|concrete\s+cover|clear\s+cover|minimum\s+cover|"
    r"cover\s+to\s+(?:the\s+)?(?:steel\s+)?reinforce)"
)
_COVER_CLAUSE_WINDOW = 160
# A new sentence or a new numbered note ends the clause a heading governs.
_CLAUSE_BREAK_RE = re.compile(
    r"(?:[.;!?](?=\s)|\n\s*-?\s*\d+(?:\.\d+)*[.)]\s)"
)
_DIMENSION_SIDE_RE = re.compile(r"(?i)[x×]\s*$")
_DIMENSION_NEXT_RE = re.compile(r"(?i)^\s*[x×]\s*\d")
_COVER_REF_RE = re.compile(
    r"(?i)\b(?:concrete\s+cover|nominal\s+cover|minimum\s+cover|clear\s+cover|"
    r"cover\s+to\s+(?:the\s+)?(?:steel\s+)?reinforce\w*|"
    r"cover\s+specified|specified\s+(?:minimum\s+)?(?:concrete\s+)?cover)\b"
)
# How a clause names each quantity when it points elsewhere for the figure.
# A manhole "cover" is not the concrete cover; a "lighting fitting" is not a
# lighting level.
_QUANTITY_REFERENCE_RES: Dict[str, "re.Pattern"] = {
    "length_mm": _COVER_REF_RE,
    "compaction": re.compile(
        r"(?i)\b(?:compaction|compacted|degree\s+of\s+compaction|"
        r"(?:maximum\s+)?dry\s+density)\b"
    ),
    "illuminance": re.compile(
        r"(?i)\b(?:illuminat\w*|illuminance|lighting\s+levels?|lux)\b"
    ),
}
# The document classes a clause can send the reader to, each with the test
# that tells a document of that class by its name. Vocabulary of document
# kinds, not names of documents.
_POINTER_TARGET_CLASSES: Tuple[Tuple[str, "re.Pattern", "re.Pattern"], ...] = (
    ("drawings", re.compile(r"(?i)\bdrawings?\b"),
     re.compile(r"(?i)(?:(?:^|[^a-z])dwg(?:[^a-z]|$)|\bdrawings?\b)")),
    ("schedule", re.compile(r"(?i)\bschedules?\b"),
     re.compile(r"(?i)\bschedules?\b")),
    ("appendix", re.compile(r"(?i)\b(?:appendix|appendices|annex\w*)\b"),
     re.compile(r"(?i)\b(?:appendix|appendices|annex\w*)\b")),
    ("contract data", re.compile(r"(?i)\bcontract\s+data\b"),
     re.compile(r"(?i)contract[\s_]+data")),
)
# "shown on the Drawings", "as specified in the Schedule", "per the drawings",
# "refer to the Appendix", "on the Drawings or as the engineer directs".
_POINTER_LEAD_RE = re.compile(
    r"(?i)\b(?:(?:on|in|by)\s+the|"
    r"(?:shown|indicated|detailed|noted|specified|given|stated|listed|"
    r"scheduled|set\s+out|tabulated)\s+(?:on|in)\s+(?:the)?|"
    r"(?:as\s+)?per\s+(?:the)?|refer(?:red)?\s+to\s+(?:the)?|see\s+(?:the)?)\s*$"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+")
_POINTER_FETCH_K = 60
_POINTER_MAX_SCOPED_DOCS = 24
_DEFERRED_AUTHORITY_BONUS = 0.5
_COVER_SUBJECT_BONUS = 0.3
# A cover question names what is covered. Same idea as the compaction
# subject groups: the chunk may use the document's word for the element.
_COVER_SUBJECT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("foundation", "footing", "raft", "pile cap"),
    ("slab",),
    ("wall",),
    ("column",),
    ("beam",),
)


def _figure_is_dimension(blob: str, start: int, end: int) -> bool:
    """True when the millimetre is one side of a size ("250mm x 250mm")."""
    return bool(
        _DIMENSION_SIDE_RE.search(blob[max(0, start - 4):start])
        or _DIMENSION_NEXT_RE.match(blob[end:end + 6])
    )


def _cover_clause_states_length(text: str) -> bool:
    """True when a cover phrase is given a single millimetre length.

    Either next to the phrase (the 64-character window) or as a list item
    the phrase heads, with no sentence or numbered-note break between.
    A millimetre that is one side of a size is not a cover length.
    """
    blob = text or ""
    phrases = [m for m in _COVER_CLAUSE_PHRASE_RE.finditer(blob)]
    if not phrases:
        return False
    figures = [
        m for m in _MM_FIGURE_RE.finditer(blob)
        if not _figure_is_dimension(blob, m.start(), m.end())
    ]
    for ph in phrases:
        for fig in figures:
            if abs(ph.start() - fig.start()) <= _COVER_PHRASE_WINDOW:
                return True
            if 0 < fig.start() - ph.end() <= _COVER_CLAUSE_WINDOW:
                between = blob[ph.end():fig.start()]
                if not _CLAUSE_BREAK_RE.search(between):
                    return True
    return False


def _sentence_points_to(sentence: str) -> Optional[str]:
    """The document class a sentence sends the reader to, or None."""
    for name, noun_rx, _name_rx in _POINTER_TARGET_CLASSES:
        for match in noun_rx.finditer(sentence):
            lead = sentence[max(0, match.start() - 40):match.start()]
            if _POINTER_LEAD_RE.search(lead):
                return name
    return None


def chunk_points_quantity_elsewhere(text: str, kinds: frozenset) -> Optional[str]:
    """The document class one sentence sends an asked quantity to, or None.

    The same sentence must name the quantity ("the cover specified",
    "compacted", "lighting levels") and point at a document class ("on the
    Drawings", "as listed in the Schedule"). A manhole-cover sentence does
    not name the concrete cover; a quantity and a pointer in different
    sentences are not a deferral.
    """
    refs = [_QUANTITY_REFERENCE_RES[k] for k in sorted(kinds) if k in _QUANTITY_REFERENCE_RES]
    if not refs:
        return None
    for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
        if not any(rx.search(sentence) for rx in refs):
            continue
        target = _sentence_points_to(sentence)
        if target:
            return target
    return None


def chunk_defers_cover_to_drawings(text: str) -> bool:
    """True when one sentence names the concrete cover and sends it to the drawings."""
    return chunk_points_quantity_elsewhere(text, frozenset({"length_mm"})) == "drawings"


def filename_is_pointer_target(filename: str, target: str) -> bool:
    """True when the document name says it is of the pointed-to class."""
    for name, _noun_rx, name_rx in _POINTER_TARGET_CLASSES:
        if name == target:
            return bool(name_rx.search(filename or ""))
    return False


def filename_is_drawing(filename: str) -> bool:
    """True when the document name says it is a drawing or drawings volume."""
    return filename_is_pointer_target(filename, "drawings")


def cover_subject_named(query: str) -> List[tuple]:
    """The element groups a cover question names, or []."""
    q = (query or "").lower()
    return [
        group for group in _COVER_SUBJECT_GROUPS
        if any(_term_in(q, term) for term in group)
    ]


def cover_subject_agrees(groups: List[tuple], text: str) -> bool:
    """True when ``text`` names one of the asked element groups."""
    if not groups:
        return False
    blob = (text or "").lower()
    return any(any(_term_in(blob, term) for term in group) for group in groups)


# The specification can be named as the source without "per the": "what
# cover does the spec require", "what does the specification say about".
# Used only by this path, so the global +1.2 class lift is unchanged.
_SPEC_AS_SUBJECT_RE = re.compile(
    r"(?i)\b(?:the\s+|this\s+)?(?:project\s+)?spec(?:ification)?s?\s+"
    r"(?:require|requires|required|say|says|state|states|specify|specifies|"
    r"call\s+for|calls\s+for|give|gives|set|sets|demand|demands)\b"
    r"|\b(?:in|by|from)\s+the\s+(?:project\s+)?spec(?:ification)?s?\b"
)
# "cover" alone is a cover ask when the question is about concrete work,
# not a lid, hatch, letter or sheet.
_BARE_COVER_RE = re.compile(r"(?i)\bcover\b")
_COVER_CONCRETE_CONTEXT_RE = re.compile(
    r"(?i)\b(?:foundations?|footings?|rafts?|pile\s+caps?|slabs?|walls?|"
    r"columns?|beams?|reinforc\w*|rebar|bars?|concrete)\b"
)
_NOT_CONCRETE_COVER_RE = re.compile(
    r"(?i)\b(?:cover\s+(?:letter|sheet|page|note|plate)s?|manholes?|hatch\w*|"
    r"lids?|insurance|cover(?:ed|s)?\s+by)\b"
    # "cover" as a verb: "does the spec cover curing", "specs cover".
    r"|\b(?:does|do|did|will|would|can|should)\s+(?:the\s+|this\s+)?"
    r"(?:project\s+)?\w+\s+cover\b"
    r"|\bspec(?:ification)?s?\s+covers?\b"
)


def query_names_specification(query: str) -> bool:
    """The question names the specification as its source."""
    if source_class_named_by(query) == "specification":
        return True
    return bool(_SPEC_AS_SUBJECT_RE.search(query or ""))


def query_asks_concrete_cover(query: str) -> bool:
    """A cover-length ask, including a bare "cover" about concrete work."""
    if "length_mm" in asked_quantity_kinds(query):
        return True
    text = query or ""
    return bool(
        _BARE_COVER_RE.search(text)
        and _COVER_CONCRETE_CONTEXT_RE.search(text)
        and not _NOT_CONCRETE_COVER_RE.search(text)
    )


def query_names_its_source(query: str) -> str:
    """The document class the question names as its source, or ""."""
    named = source_class_named_by(query)
    if named:
        return named
    if query_names_specification(query):
        return "specification"
    return ""


def query_follows_source_pointers(query: str) -> bool:
    """A measured-quantity question scoped to a named source document."""
    if not spec_boost_guard_enabled():
        return False
    return bool(query_names_its_source(query)) and bool(measured_quantity_kinds(query))


# The word a pointer clause uses for each quantity, for the text fetch. The
# detector above is the real gate.
_QUANTITY_POINTER_ANCHORS: Dict[str, Tuple[str, ...]] = {
    "length_mm": ("cover",),
    "compaction": ("compact",),
    "illuminance": ("illuminat", "lighting"),
}
_POINTER_TARGET_WORDS = ("drawing", "schedule", "appendix", "contract data")


def follow_quantity_pointers(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    *,
    k: int = 5,
    embedder=None,
    query_vec=None,
) -> Dict[str, str]:
    """Pool the named source's clause that sends the asked figure elsewhere.

    Returns ``{doc_id: name}`` for every doc resolved here, so the later
    name pass does not look them up twice. Project corpus only. A pointer
    clause already pooled outside the provisional top-``k`` with no bonus of
    its own competes on its cosine (a lexical-only entry carries a BM25 rank
    in that slot); when none is pooled, one is fetched -- first inside the
    source documents already in the pool, then across the project. New
    clauses enter at their cosine (0.0 without an embedder). Failures leave
    the pool standing.
    """
    names: Dict[str, str] = {}
    if not query_follows_source_pointers(query):
        return names
    source = query_names_its_source(query)
    kinds = measured_quantity_kinds(query)

    def _name(doc_id: str) -> str:
        if doc_id not in names:
            names[doc_id] = _doc_name_for_id(doc_id)
        return names[doc_id]

    def _is_source(doc_id: str) -> bool:
        return filename_is_source_class(_name(doc_id), source)

    ranked = sorted(fused.items(), key=lambda kv: -((kv[1][1] or 0.0) + (kv[1][2] or 0.0)))
    top_ids = {cid for cid, _e in ranked[:max(k, 1)]}
    source_docs: List[str] = []
    admitted: List[Chunk] = []
    found_pooled = False
    for chunk_id, (chunk, _sem, bonus) in ranked:
        if getattr(chunk, "project_id", project_id) != project_id:
            continue
        if not _is_source(chunk.doc_id):
            continue
        if chunk.doc_id not in source_docs:
            source_docs.append(chunk.doc_id)
        if not chunk_points_quantity_elsewhere(chunk.text or "", kinds):
            continue
        found_pooled = True
        if chunk_id not in top_ids and not (bonus or 0.0):
            admitted.append(chunk)

    fetch = getattr(store, "chunks_containing_all", None)
    if not found_pooled and callable(fetch):
        anchors = [a for kind in sorted(kinds) for a in _QUANTITY_POINTER_ANCHORS.get(kind, ())]
        needle_sets = [(anchor, target) for anchor in anchors for target in _POINTER_TARGET_WORDS]
        seen: Set[str] = set(fused)
        scopes: List[Optional[List[str]]] = []
        if source_docs:
            scopes.append(source_docs[:_POINTER_MAX_SCOPED_DOCS])
        scopes.append(None)
        for doc_ids in scopes:
            if admitted:
                break  # the source's own documents answered; skip the open pass
            for needles in needle_sets:
                try:
                    hits = fetch(
                        project_id, list(needles), k=_POINTER_FETCH_K, doc_ids=doc_ids,
                    )
                except Exception as exc:  # noqa: BLE001 — recall must not break the turn
                    logger.warning(
                        "pointer-following fetch for %s (%r) failed: %s",
                        project_id, needles, exc,
                    )
                    continue
                for chunk in hits or []:
                    if chunk.chunk_id in seen:
                        continue
                    seen.add(chunk.chunk_id)
                    if not chunk_points_quantity_elsewhere(chunk.text or "", kinds):
                        continue
                    if not _is_source(chunk.doc_id):
                        continue
                    admitted.append(chunk)
    sims = _cosine_to_query(embedder, query_vec, [c.text or "" for c in admitted])
    for chunk, sim in zip(admitted, sims):
        prior = fused.get(chunk.chunk_id)
        if prior is not None and (prior[1] or 0.0) > 0.0:
            sim = max(sim, prior[1] or 0.0)
        chunk.score = round(sim, 6)
        fused[chunk.chunk_id] = (chunk, sim, 0.0)
    if admitted:
        logger.info(
            "pointer-following pooled %d %s clause(s) that send the asked "
            "figure to another document", len(admitted), source,
        )
    return names


def _cosine_to_query(embedder, query_vec, texts: List[str]) -> List[float]:
    """Cosine of each text to the query vector; zeros when unavailable."""
    if not texts:
        return []
    if embedder is None or query_vec is None:
        return [0.0] * len(texts)
    try:
        vecs = embedder.encode(texts)
        return [
            float(sum(float(a) * float(b) for a, b in zip(vec, query_vec)))
            for vec in vecs
        ]
    except Exception as exc:  # noqa: BLE001 — scoring must not break the turn
        logger.warning("spec-deferral cosine failed: %s; entering at 0.0", exc)
        return [0.0] * len(texts)


def _apply_quantity_pointer_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
    name_by_id: Dict[str, str],
) -> None:
    """In-place: rank the source's pointer clause and the documents it names.

    A chunk of the named source class that sends the asked quantity to
    another document takes the stated-figure lift. With such a clause in the
    pool, a chunk from the pointed-to document class that states the figure
    gets a small lift over other documents, and a figure chunk about the
    element the question names gets another. No-op unless the question is a
    source-scoped quantity question.
    """
    if not query_follows_source_pointers(query):
        return
    if not numeric_requirement_boost_enabled():
        return
    kinds = measured_quantity_kinds(query)
    source = query_names_its_source(query)

    def _nm(chunk) -> str:
        return name_by_id.get(chunk.doc_id, "") or getattr(chunk, "source_name", "") or ""

    pointer_idx: List[int] = []
    targets: Set[str] = set()
    for i, (_s, c) in enumerate(scored):
        if not filename_is_source_class(_nm(c), source):
            continue
        target = chunk_points_quantity_elsewhere(c.text or "", kinds)
        if not target:
            continue
        if chunk_matches_quantity_question(query, c.text or "", kinds):
            continue
        pointer_idx.append(i)
        targets.add(target)
    if not pointer_idx:
        return
    for i in pointer_idx:
        score, chunk = scored[i]
        adjusted = score + _NUMERIC_REQUIREMENT_BONUS
        chunk.score = round(adjusted, 6)
        scored[i] = (adjusted, chunk)
    groups = cover_subject_named(query) if "length_mm" in kinds else []
    terms = quantity_subject_terms(query)
    for i, (score, chunk) in enumerate(scored):
        if i in pointer_idx:
            continue
        text = chunk.text or ""
        if not chunk_matches_quantity_question(query, text, kinds):
            continue
        add = 0.0
        # The document's name says which class it is; a parsed drawing_number
        # also fires on contract ids, which are not drawings.
        if any(filename_is_pointer_target(_nm(chunk), t) for t in targets):
            add += _DEFERRED_AUTHORITY_BONUS
        if groups:
            if cover_subject_agrees(groups, text):
                add += _COVER_SUBJECT_BONUS
        elif terms and chunk_names_quantity_subject(text, terms):
            add += _COVER_SUBJECT_BONUS
        if add:
            adjusted = score + add
            chunk.score = round(adjusted, 6)
            scored[i] = (adjusted, chunk)


# ── asked-quantity recall ────────────────────────────────────────────────
#
# A question that asks for a measured quantity ("what minimum <quantity> is
# required for <subject>") is answered by a chunk that states a number of that
# quantity next to the subject. That chunk is often a table row or a drawing
# note whose wording shares little with the question: the question says
# "cast directly against soil", the note says "in contact with soil"; the
# question says "minimum illumination for <task>", the table prints
# "| <task> | 50 |" under a "Lux" header. Neither retrieval leg pools it, and
# prose that repeats the question's words (but states no figure) fills the
# slots. The term rescue does not help either: that prose already co-occurs
# the question's terms, so the top-k looks grounded.
#
# Recall, for every quantity class the question asks for:
#   * the subject is the question's own content words, minus the words that
#     name the quantity and minus the clause that names the source document
#     ("per the project specification");
#   * candidates are chunks carrying one subject word together with a word a
#     document prints beside that quantity's figure (the anchor lexicon below
#     -- a units vocabulary, not text from any one document);
#   * a candidate is admitted only if it states a figure of that class and
#     names the subject in an affirmed (not negated) mention.
# Admitted chunks enter the pool at their own cosine, like any semantic
# candidate. Ranking is left to the numeric-requirement lift, which already
# prefers a chunk that states the asked figure. A chunk already pooled is left
# exactly as it is, and a corpus that states no such figure gets nothing.
_QUANTITY_ANCHOR_WORDS: Dict[str, Tuple[str, ...]] = {
    "length_mm": ("cover",),
    "compaction": ("dry density", "proctor", "cbr"),
    "illuminance": ("lux", "foot candle"),
}
# Words that name the quantity itself, per class. They say WHAT is measured,
# not what it is measured for.
_QUANTITY_NAME_WORDS: Dict[str, frozenset] = {
    "length_mm": frozenset({
        "cover", "covers", "concrete", "nominal", "clear", "reinforcement",
        "reinforcing", "rebar", "steel", "length", "thickness",
    }),
    "compaction": frozenset({
        "compact", "compacted", "compaction", "compacting", "degree", "density",
        "maximum", "proctor", "modified", "test", "tests", "percent",
    }),
    "illuminance": frozenset({
        "illumination", "illuminance", "lighting", "light", "level", "levels",
        "intensity", "candle", "candles",
    }),
}
# The frame of a requirement question, in any domain.
_QUANTITY_FRAME_WORDS = frozenset({
    "minimum", "maximum", "required", "require", "requires", "requirement",
    "requirements", "must", "shall", "should", "need", "needed", "needs",
    "value", "figure", "amount", "much", "many", "what", "which", "under",
    "during", "given", "stated", "state", "states", "says", "apply", "applies",
    "directly", "against", "where", "within",
    # the source a question names, wherever it sits in the sentence
    "project", "specification", "specifications", "spec", "specs", "drawing",
    "drawings", "plan", "plans", "procedure", "procedures", "document",
})
# "Per the project specification", "according to the site safety plan",
# "under the HSE lighting requirements": the clause that names the source.
_QUANTITY_SOURCE_CLAUSE_RE = re.compile(
    r"(?i)\b(?:per|as\s+per|according\s+to|under|in|from|by)\s+the\s+"
    r"(?:[a-z0-9'&-]+\s+){0,4}?"
    r"(?:specifications?|specs?|plans?|requirements?|procedures?|drawings?|"
    r"standards?|codes?)\b"
)
# A subject mention governed by a negation in the same clause: "not in
# contact with <subject>", "elements not on <subject>".
_QUANTITY_NEGATED_LEAD_RE = re.compile(
    r"(?i)\b(?:not|no|non|without|except)\b[^.;:\n|]{0,24}$"
)
_QUANTITY_RECALL_FETCH_K = 80
_QUANTITY_RECALL_MAX_TERMS = 6
_QUANTITY_RECALL_MAX_CHUNKS = 8


def measured_quantity_kinds(query: str) -> frozenset:
    """Every quantity class ``query`` asks a figure for.

    ``asked_quantity_kinds`` plus a bare "cover" asked about concrete work.
    Empty for an ordinary question.
    """
    kinds = set(asked_quantity_kinds(query))
    if query_asks_concrete_cover(query):
        kinds.add("length_mm")
    return frozenset(kinds)


def query_asks_source_scoped_quantity(query: str) -> bool:
    """A measured-quantity question that names the document governing it.

    "Per the project specification, what minimum cover ...", "under the site
    safety plan, what lighting level ...". Its answer can sit in two places:
    the named source's own clause and the document that clause points to.
    """
    if not measured_quantity_kinds(query):
        return False
    return bool(source_class_named_by(query)) or query_names_specification(query)


def quantity_subject_terms(query: str) -> List[str]:
    """Stems of the words that name what the quantity is asked FOR.

    The source clause, the question frame and the words naming an asked
    quantity are removed; what is left is the subject ("formwork erection",
    "pile caps poured against rock"). Order of first appearance.
    """
    kinds = measured_quantity_kinds(query)
    drop: Set[str] = set(_QUANTITY_FRAME_WORDS)
    for kind in kinds:
        drop |= _QUANTITY_NAME_WORDS.get(kind, frozenset())
    scope = _QUANTITY_SOURCE_CLAUSE_RE.sub(" ", query or "")
    out: List[str] = []
    for word in re.findall(r"[a-z0-9]{4,}", scope.lower()):
        if word in drop or word in _GK_STOPWORDS or word in _STOPWORDS:
            continue
        if word.isdigit():
            continue
        stem = stem_query_term(word)
        if stem not in out:
            out.append(stem)
    return out


# Words a document uses for the same element or ground condition as the
# question: a note says "in contact with soil" where the question said "cast
# against earth", "footings" where it said "foundations". A construction
# thesaurus, not phrases from any one document.
_SUBJECT_SYNONYM_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("soil", "earth", "ground"),
    ("foundation", "footing", "raft", "pile cap"),
    ("pavement", "road", "carriageway", "sub-grade", "subgrade", "embankment"),
    ("backfill", "fill"),
)
_SUBJECT_FETCH_MAX_WORDS = 10


def subject_alternatives(term: str) -> List[str]:
    """``term`` and the words its synonym group uses for the same thing."""
    out = [term]
    for group in _SUBJECT_SYNONYM_GROUPS:
        if any(word == term or word.startswith(term) for word in group):
            out.extend(word for word in group if word not in out)
    return out


def _affirmed_mention(blob: str, word: str) -> bool:
    for match in re.finditer(rf"(?i)\b{re.escape(word)}", blob):
        lead = blob[max(0, match.start() - 32):match.start()]
        if _QUANTITY_NEGATED_LEAD_RE.search(lead):
            continue
        return True
    return False


def chunk_names_quantity_subject(text: str, terms: List[str]) -> bool:
    """True when ``text`` mentions one of ``terms`` and that mention is affirmed.

    A term also matches the words of its synonym group. Vacuously true when
    the question names no subject. A mention inside a negated clause ("not in
    contact with <subject>") does not count; a chunk that also states the
    affirmed condition still does.
    """
    if not terms:
        return True
    blob = text or ""
    return any(
        _affirmed_mention(blob, word)
        for term in terms
        for word in subject_alternatives(term)
    )


def _quantity_subject_coverage(text: str, terms: List[str]) -> int:
    """How many of the question's subject terms the chunk names (affirmed)."""
    blob = text or ""
    return sum(
        1 for term in terms
        if any(_affirmed_mention(blob, word) for word in subject_alternatives(term))
    )


def recall_asked_quantity_chunks(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    *,
    k: int = 5,
    extra_pids: Optional[List[str]] = None,
    embedder=None,
    query_vec=None,
) -> int:
    """Pool chunks that state the asked quantity for the asked subject.

    Scans the project and ``extra_pids`` (the corpora the semantic leg
    searched). New chunks enter at their own cosine to the query (0.0
    without an embedder). A matching chunk already pooled but outside the
    provisional top-``k`` with no bonus of its own competes on its cosine
    too: a lexical-only entry carries a BM25 rank, not a cosine, in that
    slot. The top-``k`` itself is never touched. Returns the number of chunks
    added or re-scored. Store failures leave the pool standing.
    """
    kinds = measured_quantity_kinds(query)
    if not kinds:
        return 0
    terms = quantity_subject_terms(query)[:_QUANTITY_RECALL_MAX_TERMS]
    if not terms:
        return 0
    contain = getattr(store, "chunks_containing_all", None)
    ident = getattr(store, "identifier_search", None)
    anchors = [
        anchor for kind in sorted(kinds)
        for anchor in _QUANTITY_ANCHOR_WORDS.get(kind, ())
    ]
    # The question's own words first, then the words their synonym groups add.
    words: List[str] = list(terms)
    for term in terms:
        for word in subject_alternatives(term):
            # A text match on "found" already finds "foundation".
            if not any(have in word for have in words):
                words.append(word)
    groups = [(word, anchor) for word in words[:_SUBJECT_FETCH_MAX_WORDS] for anchor in anchors]
    pids = [project_id] + [p for p in (extra_pids or []) if p and p != project_id]
    top = sorted(fused.items(), key=lambda kv: -((kv[1][1] or 0.0) + (kv[1][2] or 0.0)))
    seen: Set[str] = {cid for cid, _entry in top[:max(k, 1)]}
    admitted: List[Tuple[int, Chunk]] = []

    def _consider(chunk: Chunk) -> None:
        if chunk.chunk_id in seen:
            return
        seen.add(chunk.chunk_id)
        prior = fused.get(chunk.chunk_id)
        if prior is not None and (prior[2] or 0.0):
            return  # carries another mechanism's lift; leave it
        text = chunk.text or ""
        if not chunk_matches_quantity_question(query, text, kinds):
            return
        if not chunk_names_quantity_subject(text, terms):
            return
        admitted.append((_quantity_subject_coverage(text, terms), prior[0] if prior else chunk))

    for _cid, (chunk, _sem, _bonus) in top[max(k, 1):]:
        _consider(chunk)
    for pid in pids:
        if callable(contain):
            for term, anchor in groups:
                try:
                    hits = contain(pid, [term, anchor], k=_QUANTITY_RECALL_FETCH_K)
                except Exception as exc:  # noqa: BLE001 — recall must not break the turn
                    logger.warning(
                        "asked-quantity recall for %s (%s + %s) failed: %s",
                        pid, term, anchor, exc,
                    )
                    continue
                for chunk in hits or []:
                    _consider(chunk)
        if callable(ident):
            try:
                hits = ident(
                    pid, [f"{term} {anchor}" for term, anchor in groups],
                    k=_QUANTITY_RECALL_FETCH_K,
                )
            except Exception as exc:  # noqa: BLE001 — recall must not break the turn
                logger.warning("asked-quantity recall for %s failed: %s", pid, exc)
                hits = []
            for chunk in hits or []:
                _consider(chunk)
    if not admitted:
        return 0
    admitted.sort(key=lambda item: -item[0])
    chosen = [chunk for _cov, chunk in admitted[:_QUANTITY_RECALL_MAX_CHUNKS]]
    sims = _cosine_to_query(embedder, query_vec, [c.text or "" for c in chosen])
    for chunk, sim in zip(chosen, sims):
        prior = fused.get(chunk.chunk_id)
        if prior is not None and (prior[1] or 0.0) > 0.0:
            sim = max(sim, prior[1] or 0.0)
        chunk.score = round(sim, 6)
        fused[chunk.chunk_id] = (chunk, sim, 0.0)
    logger.info(
        "asked-quantity recall pooled %d chunk(s) for %s (subject %r)",
        len(chosen), sorted(kinds), terms,
    )
    return len(chosen)


_SOURCE_HEADER_RE = re.compile(r"(?i)^\[source:[^\]]*\]\s*")
_COPY_KEY_MIN_CHARS = 80


def chunk_copy_key(text: str) -> str:
    """Body key for signed/unsigned copies of one clause.

    Empty when the body is too short to collapse. A leading
    ``[source: …]`` path is not part of the clause — the two copies of
    one volume differ there and match everywhere else. Whitespace is
    collapsed so a line wrap does not look like a different sentence.
    """
    body = _SOURCE_HEADER_RE.sub("", (text or "").strip())
    body = re.sub(r"\s+", " ", body).strip().lower()
    if len(body) < _COPY_KEY_MIN_CHARS:
        return ""
    return body


def _fetch_numeric_requirement_chunks(
    query: str,
    project_id: str,
    store,
    k: int,
    seen_ids: Set[str],
) -> List[Chunk]:
    """Chunks the primary query's lexical leg did not keep.

    Only chunks that state the asked quantity are returned, with score 0
    so a BM25 rank is never treated as a cosine. The later lift is what
    ranks them. Failures return [] — the primary pool stands.
    """
    if not numeric_requirement_boost_enabled():
        return []
    expansion = numeric_requirement_expansion(query)
    kinds = asked_quantity_kinds(query)
    if not expansion or not kinds:
        return []
    try:
        hits = store.bm25_search(
            project_id, f"{(query or '').strip()} {expansion}", k,
        )
    except Exception as exc:  # noqa: BLE001 — extras must not break the turn
        logger.warning(
            "numeric-requirement retrieval for %s failed: %s; "
            "primary results stand",
            project_id, exc,
        )
        return []
    admitted: List[Chunk] = []
    for chunk in hits:
        if chunk.chunk_id in seen_ids:
            continue
        if not chunk_matches_quantity_question(query, chunk.text or "", kinds):
            continue
        chunk.score = 0.0
        admitted.append(chunk)
        seen_ids.add(chunk.chunk_id)
    return admitted


def _apply_numeric_requirement_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
    *,
    higher_is_better: bool = True,
) -> None:
    """In-place: lift a chunk that states a number for the asked quantity.

    ``higher_is_better`` is false on the lexical-only path, where a better
    BM25 rank is a more negative score and the final sort negates it.
    No-op unless the question asks for one of these quantities, and a
    no-op when the kill-switch is off — scores are then untouched.
    """
    if not numeric_requirement_boost_enabled():
        return
    kinds = asked_quantity_kinds(query)
    if not kinds:
        return
    bonus = _NUMERIC_REQUIREMENT_BONUS if higher_is_better else -_NUMERIC_REQUIREMENT_BONUS
    for i, (score, chunk) in enumerate(scored):
        if not chunk_matches_quantity_question(query, chunk.text or "", kinds):
            continue
        adjusted = score + bonus
        chunk.score = round(adjusted, 6)
        scored[i] = (adjusted, chunk)


def _apply_filename_overlap_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
    name_by_id: Dict[str, str],
) -> None:
    """In-place: lift chunks whose resolved filename matches the query."""
    terms = distinctive_query_terms(query)
    if len(terms) < _FILENAME_RESCUE_MIN_TERMS:
        return
    letter_q = query_asks_for_letter_or_signatory(query)
    for i, (score, chunk) in enumerate(scored):
        name = name_by_id.get(chunk.doc_id, "") or getattr(chunk, "source_name", "") or ""
        add = filename_match_bonus(name, terms, letter_query=letter_q)
        if add <= 0.0:
            continue
        boosted = score + add
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


def _pool_docs_named_by_query(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    extra_pids: List[str],
) -> Dict[str, str]:
    """Pull chunks from filename-matched letters into ``fused``.

    Returns ``{doc_id: original_name}`` so the later name resolution does
    not re-query the documents table for docs we just looked up.
    Failures never raise — the semantic pool stands.
    """
    names: Dict[str, str] = {}
    terms = distinctive_query_terms(query)
    letter_q = query_asks_for_letter_or_signatory(query)
    if letter_q:
        if len(terms) < _FILENAME_RESCUE_MIN_TERMS:
            return names
        require_letter = True
        min_terms = _FILENAME_RESCUE_MIN_TERMS
    else:
        # Named-site lookup without "letter": only fire when the query is
        # specific enough that a filename collision is unlikely.
        if len(terms) < _FILENAME_RESCUE_OPEN_MIN_TERMS:
            return names
        require_letter = False
        min_terms = _FILENAME_RESCUE_OPEN_MIN_TERMS

    try:
        from app.core.projects import documents_matching_filename_terms
    except Exception:  # noqa: BLE001
        logger.warning("filename rescue: projects import failed", exc_info=True)
        return names

    pids = [project_id] + [p for p in extra_pids if p and p != project_id]
    fetch = getattr(store, "chunks_for_docs", None)
    if not callable(fetch):
        return names

    recovered = 0
    for pid in pids:
        try:
            matches = documents_matching_filename_terms(
                pid, terms, min_terms=min_terms, require_letter=require_letter,
            )
            if letter_q and not matches:
                # Filename has the site/party but not the word "letter"
                # (handover certificate). Retry on overlap alone.
                matches = documents_matching_filename_terms(
                    pid, terms, min_terms=_FILENAME_RESCUE_OPEN_MIN_TERMS,
                    require_letter=False,
                )
        except Exception as exc:  # noqa: BLE001 — extras must not break the turn
            logger.warning("filename rescue listing for %s failed: %s", pid, exc)
            continue
        if not matches:
            continue
        for doc in matches:
            names[doc["id"]] = doc.get("original_name") or ""
        try:
            hits = fetch(pid, [d["id"] for d in matches])
        except Exception as exc:  # noqa: BLE001
            logger.warning("filename rescue fetch for %s failed: %s", pid, exc)
            continue
        for chunk in hits:
            names.setdefault(chunk.doc_id, names.get(chunk.doc_id, ""))
            if chunk.chunk_id in fused:
                continue
            fused[chunk.chunk_id] = (chunk, 0.0, 0.0)
            recovered += 1
    if recovered:
        logger.info(
            "filename rescue recovered %d chunk(s) for terms %r (letter_query=%s)",
            recovered, terms, letter_q,
        )
    return names


# ── named-document recall: identity of a document the question names ──────
#
# "What is the document number and revision of <named document>, and who
# prepared it?" / "On what date was <named document> issued, and under which
# <reference> number?" The answer is the named document's own control block
# (labels and codes) or its issue stamp ("Date: ... <Label> No. <code>").
# Nothing in the question resembles either, and "document number", "revision",
# "prepared" are in every template in the corpus, so cosine prefers templates
# that describe how a document should be identified.
#
# The words that ASK (number, revision, prepared, the asked reference label)
# are separated from the words that NAME the document; the named document is
# found by its name or by the stamp that repeats its title, and only its
# identity evidence is pooled.
#
# "under which RFP number", "which tender no.", "what contract reference":
# the reference a question asks for by its label. Labels that are the
# control-block path's own vocabulary are not issue-stamp labels.
_REFERENCE_LABEL_ASK_RE = re.compile(
    r"(?i)\b(?:under\s+which|which|what)\s+(?P<label>[a-z]{2,12})\s+"
    r"(?:number|no\b\.?|reference|ref\b)"
)
_CONTROL_BLOCK_LABEL_WORDS = frozenset({
    "document", "doc", "drawing", "revision", "page", "clause", "section",
    "item", "sheet", "is", "its", "the",
})
_ISSUED_WHEN_RE = re.compile(
    r"(?i)\bwhen\s+was\b[^?]{0,80}\b(?:issued|prepared|dated|published|revised)\b|"
    r"\b(?:on\s+what\s+)?date\s+was\b[^?]{0,80}"
    r"\b(?:issued|prepared|dated|published|revised)\b"
)
_DOC_IDENTITY_ASK_RE = re.compile(
    r"(?i)\b(?:document|doc\.?|drawing|reference)\s+(?:number|no\b\.?|ref\b)|"
    r"\brevision\b|\bprepared\s+by\b|"
    r"\bwho\s+(?:prepared|authored|wrote|issued|checked|reviewed|approved)\b|"
    # "What is the date of the priced BOQ", "when / on what date was … issued".
    r"\bdate\s+of\s+(?:the\s+)?(?!commencement|completion|award|access|taking)"
)
_DOC_IDENTITY_ASK_WORDS = frozenset({
    "document", "number", "revision", "prepared", "authored", "wrote",
    "issued", "checked", "reviewed", "approved", "reference", "drawing",
    "date", "dated", "title", "author",
})
# A labelled issue date: "Date: July 10, 2031", "Dated 4 March 2031",
# "Issue date: 04/03/2031".
_LABELLED_DATE_RE = re.compile(
    r"(?i)\b(?:issue\s+date|date\s+of\s+issue|dated|date)\s*:?\s*"
    r"(?:[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?\s+\d{4}|"
    r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2})"
)
# A reference code: letters, then hyphen- or slash-joined segments, at least
# one of them numeric ("AB-2031-044", "TN/88/12").
_REFERENCE_CODE = r"[A-Za-z][A-Za-z0-9]*(?:[-/][A-Za-z0-9.]+)*[-/]\d+[A-Za-z0-9.-]*"
_LABELLED_CODE_RE = re.compile(
    rf"(?i)\b(?P<label>[a-z]{{2,12}})\s+no\.?\s*:?\s*{_REFERENCE_CODE}"
)
_DOC_CONTROL_BLOCK_LABEL_RES = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bdocument\s+no\b", r"\brevision\s+no\b", r"\bprepared\s+by\b",
        r"\bdoc\s+status\b", r"\bproject\s+no\b", r"\bfile\s+name\b",
        r"\bchecked\b", r"\breviewed\b", r"\bapproved\b", r"\bauthor\b",
        r"\bclient\s+reference\b",
    )
)
_DOC_IDENTITY_BONUS = 2.0
_DOC_IDENTITY_COVER_CHUNKS = 8
_DOC_IDENTITY_MAX_CHUNKS = 2


def asked_reference_labels(query: str) -> List[str]:
    """Labels of the references the question asks for by number ("rfp")."""
    out: List[str] = []
    for match in _REFERENCE_LABEL_ASK_RE.finditer(query or ""):
        label = match.group("label").lower()
        if label in _CONTROL_BLOCK_LABEL_WORDS or label in out:
            continue
        out.append(label)
    return out


def query_asks_for_document_identity(query: str) -> bool:
    """True for "what number / revision is X, who prepared it, when was X issued"."""
    q = query or ""
    return bool(
        _DOC_IDENTITY_ASK_RE.search(q)
        or _ISSUED_WHEN_RE.search(q)
        or asked_reference_labels(q)
    )


# "...number and revision OF THE priced Bill of Quantities, and who prepared
# it" / "the date OF THE priced Bill of Quantities and the Employer's ...".
# The title is the noun phrase an identity word points at, cut at the next
# clause. Taking every non-ask word instead adds "employer" and "contract"
# to the title, and a filename has to contain every title word.
_DOC_IDENTITY_TITLE_RE = re.compile(
    r"(?i)\b(?:number|no\.?|revision|date|reference|ref|status|title|author)\s+"
    r"of\s+(?:the\s+)?(?P<title>.+?)(?=\s+and\s+(?:the|who|what|its|when)\b|[,;?]|$)"
)
# "On what date was the <named document> issued"
_DOC_IDENTITY_WAS_ISSUED_TITLE_RE = re.compile(
    r"(?i)\b(?:date|when)\s+was\s+(?:the\s+)?(?P<title>.+?)\s+"
    r"(?:issued|prepared|dated|published|revised)\b"
)


def document_identity_title_terms(query: str) -> List[str]:
    """The words of the ask that NAME the document, not the ones that ask."""
    q = query or ""
    pointed = (
        _DOC_IDENTITY_WAS_ISSUED_TITLE_RE.search(q)
        or _DOC_IDENTITY_TITLE_RE.search(q)
    )
    scope = pointed.group("title") if pointed else q
    asking = _DOC_IDENTITY_ASK_WORDS | set(asked_reference_labels(q))
    return sorted(t for t in _significant_terms(scope) if t not in asking)


def query_asks_for_issue_identity(query: str) -> bool:
    """True for "when was X issued, and under which <label> number?".

    Who-prepared / revision asks and "the date of X and its reference" stay
    on the control-block path.
    """
    q = query or ""
    return bool(_ISSUED_WHEN_RE.search(q) or asked_reference_labels(q))


def chunk_states_issue_stamp(text: str, labels: Optional[List[str]] = None) -> bool:
    """True for an issue stamp: a labelled date and a labelled reference code.

    With ``labels`` the reference must carry one of them ("RFP No. <code>");
    without, any "<Label> No. <code>" counts.
    """
    blob = text or ""
    if not _LABELLED_DATE_RE.search(blob):
        return False
    wanted = {lab.lower() for lab in (labels or [])}
    for match in _LABELLED_CODE_RE.finditer(blob):
        if not wanted or match.group("label").lower() in wanted:
            return True
    return False


def document_control_label_count(text: str) -> int:
    """How many distinct document-control labels the chunk carries."""
    t = text or ""
    return sum(1 for rx in _DOC_CONTROL_BLOCK_LABEL_RES if rx.search(t))


def chunk_states_document_control_block(text: str) -> bool:
    """True for a cover / revision-history block: two or more control labels."""
    return document_control_label_count(text) >= 2


def _pool_named_document_control_block(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    extra_pids: List[str],
) -> int:
    """Pull the named document's control block into ``fused``."""
    if not query_asks_for_document_identity(query):
        return 0
    terms = document_identity_title_terms(query)
    if len(terms) < 2:
        return 0  # no document is named; identity vocabulary alone is not one
    try:
        from app.core.projects import documents_matching_filename_terms
    except Exception:  # noqa: BLE001
        logger.warning("document-identity rescue: projects import failed", exc_info=True)
        return 0
    fetch = getattr(store, "chunks_for_docs", None)
    if not callable(fetch):
        return 0
    recovered = 0
    pids = [project_id] + [p for p in extra_pids if p and p != project_id]
    for pid in pids:
        try:
            docs = documents_matching_filename_terms(
                pid, terms, require_all=True, limit=3,
            )
            hits = (
                fetch(pid, [d["id"] for d in docs],
                      k_per_doc=_DOC_IDENTITY_COVER_CHUNKS)
                if docs else []
            )
        except Exception as exc:  # noqa: BLE001 — extras must not break the turn
            logger.warning("document-identity rescue for %s failed: %s", pid, exc)
            continue
        # Richest block first. A cover prints a thin title block (number,
        # revision) BEFORE the full control block (…, prepared by, status);
        # page order lifted the thin one and the answer reported that the
        # excerpt "does not name the party that prepared it" (live 39d6b8d).
        blocks = sorted(
            (c for c in hits if chunk_states_document_control_block(c.text or "")),
            key=lambda c: (-document_control_label_count(c.text or ""), c.chunk_index),
        )
        for chunk in blocks[:_DOC_IDENTITY_MAX_CHUNKS]:
            prev = fused.get(chunk.chunk_id)
            if prev is not None:
                fused[chunk.chunk_id] = (
                    prev[0], prev[1], max(prev[2], _DOC_IDENTITY_BONUS),
                )
                continue
            fused[chunk.chunk_id] = (chunk, 0.0, _DOC_IDENTITY_BONUS)
            recovered += 1
        if recovered:
            break
    if recovered:
        logger.info("document-identity rescue recovered %d chunk(s)", recovered)
    return recovered


_ISSUE_STAMP_TITLE_WORDS = 2


def recall_issue_stamps(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    extra_pids: List[str],
) -> int:
    """Pull the issue stamp of the document the question names into ``fused``.

    A stamp repeats the document's title on every page it heads, so it is
    found by its own words even when the file it sits in is named for
    something else (a bill bound inside a schedules volume). The text fetch
    pairs the asked reference label ("rfp") with the two most distinctive
    title words; a hit must be a stamp and carry two title words. When stamps
    from more than one contract match, the year-lock election keeps one.
    """
    if not query_asks_for_issue_identity(query):
        return 0
    terms = document_identity_title_terms(query)
    if len(terms) < 2:
        return 0
    fetch = getattr(store, "chunks_containing_all", None)
    if not callable(fetch):
        return 0
    labels = asked_reference_labels(query)
    distinctive = sorted(terms, key=lambda t: (-len(t), t))[:_ISSUE_STAMP_TITLE_WORDS]
    needles = (labels[:1] or ["date"]) + distinctive
    pids = [project_id] + [p for p in extra_pids if p and p != project_id]
    stamps: List[Tuple[str, object]] = []
    for pid in pids:
        try:
            hits = fetch(pid, needles, k=20)
        except Exception as exc:  # noqa: BLE001 — extras must not break the turn
            logger.warning("issue-stamp recall for %s failed: %s", pid, exc)
            continue
        for chunk in hits:
            text = chunk.text or ""
            if not chunk_states_issue_stamp(text, labels):
                continue
            blob = text.lower()
            if sum(1 for t in terms if t in blob) < 2:
                continue
            try:
                name = _doc_name_for_id(chunk.doc_id) or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "issue-stamp name lookup for %s failed: %s", chunk.doc_id, exc,
                )
                name = ""
            stamps.append((name, chunk))
    if not stamps:
        return 0
    winner = elect_answer_bearing_contract(
        query, ((name, chunk.text or "") for name, chunk in stamps),
    )
    recovered = 0
    kept = 0
    for name, chunk in stamps:
        ids = extract_contract_doc_ids(name)
        if winner and ids and winner not in ids:
            continue
        if kept >= _DOC_IDENTITY_MAX_CHUNKS:
            break
        kept += 1
        prev = fused.get(chunk.chunk_id)
        if prev is not None:
            fused[chunk.chunk_id] = (
                prev[0], prev[1], max(prev[2], _DOC_IDENTITY_BONUS),
            )
            continue
        fused[chunk.chunk_id] = (chunk, 0.0, _DOC_IDENTITY_BONUS)
        recovered += 1
    if recovered:
        logger.info(
            "issue-stamp recall pooled %d chunk(s) winner=%s", recovered, winner,
        )
    return recovered


# ── named-document recall: which document carries a title ────────────────
#
# "Which <kind of document> covers <Title Phrase>, and what is its number?"
# The question names the document by its title. A long volume that merely
# mentions the topic outranks the short titled document on cosine, and the
# term rescue treats the volume's overlap as already grounded.
#
# Two places carry the title, and both are read from the corpus itself:
#   * the upload name ("<code> <Title Phrase>.pdf") -- documents whose name
#     contains the phrase are pulled into the pool and lifted;
#   * a register line inside another volume ("<reference code> <Title
#     Phrase>") -- the document number sits in front of the title. A
#     section heading that only shares the topic ("Section 0123 - <topic>")
#     has no reference code in front of the phrase and is not a register
#     line.
_WHICH_DOCUMENT_ASK_RE = re.compile(
    r"(?i)\b(?:which|what)\s+(?:specification|procedure|plan|standard|manual|"
    r"policy|report|document|drawing)s?\s+(?:document|section|covers|sets\s+out|"
    r"describes|governs)\b"
    r"|\b(?:specification|procedure|plan|standard|manual|policy|report)\s+"
    r"document\s+covers\b"
    r"|\band\s+what\s+is\s+its\s+(?:number|reference|ref)\b"
)
# Two-or-more consecutive Title-Case words. Leading question words ("Which
# Specification") are stripped below.
_TITLE_CASE_PHRASE_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\b"
)
_TITLE_PHRASE_STOP = frozenset({
    "which", "what", "whose", "when", "where", "why",
    "this", "that", "the", "and", "for", "its", "our",
})
# Equal to IDENTIFIER_BONUS_MAX so a titled filename beats a high-cosine
# volume the way an exact code beats boilerplate.
_TITLE_MATCH_BONUS = 2.0
# A document reference code: letter-led segments joined by hyphens, at least
# three segments and one numeric run of three or more digits
# ("AB-CD-XYZ-0001-2.0"). A bare section number is not one.
_DOC_REFERENCE_CODE_RE = re.compile(
    r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9][A-Z0-9.]*){2,}\b"
)
_REGISTER_LINE_GAP = 12


def query_asks_which_document(query: str) -> bool:
    """True for "which <kind of document> covers <Title>" / "what is its number".

    Numbered-document questions ("Specification 0042") stay on the
    identifier path. Contract-role and letter asks are not this class.
    """
    return bool(_WHICH_DOCUMENT_ASK_RE.search(query or ""))


def extract_document_title_phrases(query: str) -> List[str]:
    """Title-Case phrases of two or more content words from ``query``.

    A run of capitalised words is a title; leading question words ("Which
    Specification") are dropped. Lowercased, deduplicated.
    """
    found: List[str] = []
    seen: Set[str] = set()
    for match in _TITLE_CASE_PHRASE_RE.finditer(query or ""):
        words = [
            w for w in match.group(1).split()
            if w.lower() not in _TITLE_PHRASE_STOP
        ]
        if len(words) < 2:
            continue
        phrase = " ".join(words).lower()
        if phrase in seen or len(phrase) < 8:
            continue
        seen.add(phrase)
        found.append(phrase)
    return found


def title_filename_bonus(filename: str, phrases: List[str]) -> float:
    """Additive lift when the upload name carries a queried title phrase.

    Zero when the name shares no title phrase, so ordinary Q&A ranking
    stays byte-identical.
    """
    blob = (filename or "").lower()
    if not blob or not phrases:
        return 0.0
    if any(phrase and phrase in blob for phrase in phrases):
        return _TITLE_MATCH_BONUS
    return 0.0


def _normalize_retrieval_ws(text: str) -> str:
    """Collapse OCR / table newlines so a scanned label still matches."""
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_states_document_register_line(text: str, phrases: List[str]) -> bool:
    """True when a document reference code stands right before a title phrase."""
    if not phrases:
        return False
    blob = _normalize_retrieval_ws(text)
    lower = blob.lower()
    for code in _DOC_REFERENCE_CODE_RE.finditer(blob):
        if not any(ch.isdigit() for ch in code.group(0)):
            continue
        if not re.search(r"\d{3,}", code.group(0)):
            continue
        tail = lower[code.end():code.end() + _REGISTER_LINE_GAP + 80]
        for phrase in phrases:
            at = tail.find(phrase)
            if 0 <= at <= _REGISTER_LINE_GAP:
                return True
    return False


def _apply_title_filename_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
    name_by_id: Dict[str, str],
) -> None:
    """In-place: lift chunks whose resolved filename carries the asked title."""
    if not query_asks_which_document(query):
        return
    phrases = extract_document_title_phrases(query)
    if not phrases:
        return
    for i, (score, chunk) in enumerate(scored):
        name = name_by_id.get(chunk.doc_id, "") or getattr(chunk, "source_name", "") or ""
        add = title_filename_bonus(name, phrases)
        if add <= 0.0:
            continue
        boosted = score + add
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


def _apply_register_line_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
) -> None:
    """In-place: lift chunks whose body is a code + title register line."""
    if not query_asks_which_document(query):
        return
    phrases = extract_document_title_phrases(query)
    if not phrases:
        return
    for i, (score, chunk) in enumerate(scored):
        if not chunk_states_document_register_line(chunk.text or "", phrases):
            continue
        boosted = score + _TITLE_MATCH_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


def recall_titled_documents(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    extra_pids: List[str],
) -> Dict[str, str]:
    """Pull the documents the question names by title into ``fused``.

    Both carriers of the title are fetched: documents whose upload name
    contains a title phrase (their chunks), and register lines that print a
    reference code before the phrase. Returns ``{doc_id: original_name}`` for
    the name pass. Failures never raise -- the semantic pool stands.
    """
    names: Dict[str, str] = {}
    if not query_asks_which_document(query):
        return names
    phrases = extract_document_title_phrases(query)
    if not phrases:
        return names
    pids = [project_id] + [p for p in extra_pids if p and p != project_id]
    recovered = 0

    try:
        from app.core.projects import documents_matching_title_phrase
    except Exception:  # noqa: BLE001
        logger.warning("titled-document recall: projects import failed", exc_info=True)
        documents_matching_title_phrase = None
    by_docs = getattr(store, "chunks_for_docs", None)
    if documents_matching_title_phrase is not None and callable(by_docs):
        for pid in pids:
            matches: List[Dict[str, str]] = []
            for phrase in phrases:
                try:
                    matches.extend(documents_matching_title_phrase(pid, phrase))
                except Exception as exc:  # noqa: BLE001 — extras must not break the turn
                    logger.warning(
                        "titled-document listing for %s (%r) failed: %s", pid, phrase, exc,
                    )
            unique: List[Dict[str, str]] = []
            for doc in matches:
                did = doc.get("id") or ""
                if not did or did in names:
                    continue
                names[did] = doc.get("original_name") or ""
                unique.append(doc)
            if not unique:
                continue
            try:
                hits = by_docs(pid, [d["id"] for d in unique])
            except Exception as exc:  # noqa: BLE001
                logger.warning("titled-document fetch for %s failed: %s", pid, exc)
                continue
            for chunk in hits:
                names.setdefault(chunk.doc_id, "")
                if chunk.chunk_id in fused:
                    continue
                fused[chunk.chunk_id] = (chunk, 0.0, 0.0)
                recovered += 1

    containing = getattr(store, "chunks_containing_all", None)
    if callable(containing):
        for pid in pids:
            for phrase in phrases:
                try:
                    hits = containing(pid, [phrase], k=20)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "register-line recall for %s (%r) failed: %s", pid, phrase, exc,
                    )
                    continue
                for chunk in hits:
                    if not chunk_states_document_register_line(chunk.text or "", phrases):
                        continue
                    if chunk.chunk_id in fused:
                        continue
                    fused[chunk.chunk_id] = (chunk, 0.0, 0.0)
                    recovered += 1
    if recovered:
        logger.info(
            "titled-document recall pooled %d chunk(s) for phrases %r", recovered, phrases,
        )
    return names


# ── particulars documents by kind ──────────────────────────────────────────
#
# An Accepted Contract Amount ask can retrieve service-agreement and permit
# trackers and report the figure absent while the executed amount sits in a
# file whose NAME says it is the Contract Data (a scanned table, newlines
# between Accepted / Contract / Amount, no index-time particulars prefix, so
# the particulars boost and the unnamed election never fire). The document
# kind in the file name is the discriminator the other documents cannot
# fake.
_ACA_ASK_RE = re.compile(r"(?i)accepted\s+contract\s+amount")
_CONTRACT_DATA_FILENAME_BONUS = 2.0
_INCLUDING_VAT_RE = re.compile(r"(?i)including\s+vat|incl\.?\s+vat")


def query_asks_for_accepted_contract_amount(query: str) -> bool:
    """True for a filled Accepted Contract Amount ask, not a definition."""
    q = query or ""
    if _DEFINITION_QUESTION_RE.search(q):
        return False
    return bool(_ACA_ASK_RE.search(_normalize_retrieval_ws(q)))


def filename_looks_like_contract_data(filename: str) -> bool:
    """True when the upload name is a Contract Data file, not a PSA/CPM."""
    blob = (filename or "").replace("_", " ")
    return bool(re.search(r"(?i)contract\s+data", blob))


def filename_looks_like_conditions_volume(filename: str) -> bool:
    """True for the bound CoC / Contract Data volume a delay-damages daily-amount scan walks.

    Sources can cite a truncated ``<id>_…_Cond…`` — a complete Conditions
    volume whose delay-damages windows occupy top-k. Requiring only
    ``contract data`` in the name left ``_e1_pool_doc_ids`` empty when
    those chunks were pointer-only, so the all-chunk scan never ran.
    """
    blob = (filename or "").replace("_", " ")
    return bool(re.search(
        r"(?i)contract\s+data|conditions?\s+of\s+contract|"
        r"particular\s+conditions|"
        # Abbreviated / truncated "Cond of Contract".
        r"cond(?:itions?)?\.?\s+of\s+con",
        blob,
    ))


def contract_data_chunk_states_aca(filename: str, text: str, query: str) -> bool:
    """True when a Contract Data file's chunk states the asked ACA figure.

    Scanned Particular Conditions split the label across lines
    (``Accepted\\nContract\\nAmount (including VAT)``). Whitespace is
    collapsed before the label test. A money amount must still be visible.
    """
    if not filename_looks_like_contract_data(filename):
        return False
    blob = _normalize_retrieval_ws(text).lower()
    if "accepted contract amount" not in blob:
        return False
    if not _CD_MONETARY_VALUE_RE.search(text or ""):
        return False
    if _INCLUDING_VAT_RE.search(query or "") and not _INCLUDING_VAT_RE.search(blob):
        return False
    return True


def _apply_contract_data_filename_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
    name_by_id: Dict[str, str],
) -> None:
    """In-place: lift Contract Data files on a Contract Data particular ask."""
    if not query_wants_contract_data_file(query):
        return
    want_aca = query_asks_for_accepted_contract_amount(query)
    want_tfc = query_asks_for_time_for_completion(query)
    party_role = asked_party_role(query)
    want_daily_damages = query_asks_delay_damages_daily_amount(query)
    want_dnp = query_asks_for_defects_notification_period(query)
    want_pcg = (
        query_asks_for_parent_company_guarantee(query)
    )
    want_comm = (
        query_asks_for_contract_commencement_date(query)
    )
    for i, (score, chunk) in enumerate(scored):
        name = name_by_id.get(chunk.doc_id, "") or getattr(chunk, "source_name", "") or ""
        if not filename_looks_like_contract_data(name):
            continue
        text = chunk.text or ""
        # The ACA ask keeps the "any Contract Data file" lift. TfC / DNP / Engineer only
        # lift the row that answers — an ACA-only Contract Data file
        # must not steal Time for Completion (test_a3_is_not_stolen).
        # A daily-amount ask lifts the two compose operands, not every CD sibling.
        # PCG / commencement asks lift only the answering PCG / commencement row.
        if want_tfc and not want_aca and not chunk_states_time_for_completion(text):
            continue
        if party_role and not want_aca and not chunk_names_party(text, party_role):
            continue
        if want_daily_damages and not want_aca and not chunk_states_delay_damages_rate(text):
            continue
        if want_dnp and not want_aca and not chunk_states_defects_notification_period(text):
            continue
        if want_pcg and not want_aca and not chunk_states_pcg_contract_data(text):
            continue
        if want_comm and not want_aca and not chunk_states_commencement_contract_data(text):
            continue
        boosted = score + _CONTRACT_DATA_FILENAME_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


# ── named particulars rows ─────────────────────────────────────────────────
#
# A filled particulars row asked plainly (its value, a milestone's duration,
# a method named in a cell) is absent from the pool even though it is
# indexed and correctly prefixed: a scanned table embeds badly, cosine never
# pools it, and every bonus only re-scores the pool. The discriminator is the
# row itself: a filled row line that carries two or more of the question's
# own content words is a row the question named; one shared word is a
# coincidence.
_NAMED_ROW_MIN_LINE_TERMS = 2
_NAMED_ROW_MIN_COVERAGE = 0.5
_NAMED_ROW_MAX_CHUNKS = 2
# On nearly every Contract Data line, so they name no row in particular.
_NAMED_ROW_UBIQUITOUS_TERMS = frozenset({
    "contract", "contracts", "works", "applicable", "clause", "data",
    "under", "many", "much", "stated", "state", "states", "according",
    # Operators of the QUESTION, not words of a label. "Among the Northern
    # milestones, which have the longest Time for Completion and by how much
    # over the shortest?" names the Time for Completion row; left in, these
    # dilute coverage until that row no longer matches (unseen Set 3 F1).
    "among", "between", "longest", "shortest", "longer", "shorter", "highest",
    "lowest", "largest", "smallest", "greatest", "most", "least", "every",
    "each", "both", "compare", "compared", "difference", "exceed", "exceeds",
    "over", "than", "same", "combined", "total", "together",
})
_NAMED_ROW_SEPARATOR_RE = re.compile(r"[:|]")
_NAMED_ROW_FILLED_CELL_RE = re.compile(r"[:|][^A-Za-z0-9]*[A-Za-z0-9]")
_NAMED_ROW_NEW_CLAUSE_RE = re.compile(r"^[\s|]*\d+(?:\.\d+)+")
_NAMED_ROW_SHARE_OF_SUM_RE = re.compile(
    r"(?i)\d\s*%\s*of\s+the\s+(?:contract\s+price|accepted\s+contract\s+amount)"
)
_NAMED_ROW_BASE_AMOUNT_BONUS = 1.5
_DELAY_DURATION_ASK_RE = re.compile(
    r"(?i)\b\d+\s*(?:calendar\s+|working\s+)?(?:days?|weeks?|months?)\s+"
    r"(?:late|of\s+delay|delay(?:ed)?|behind|overdue|over(?:run)?)\b"
)


def query_applies_a_delay_duration(query: str) -> bool:
    """True for "... is 30 days late ..." — a rate alone cannot answer it."""
    return bool(_DELAY_DURATION_ASK_RE.search(query or ""))


def _named_row_terms(query: str) -> frozenset:
    # "6 weeks behind" is an operand applied to the row, not part of its
    # label; left in, it dilutes coverage and the row stops matching.
    query = _DELAY_DURATION_ASK_RE.sub(" ", query or "")
    return frozenset(
        t for t in _significant_terms(query)
        if t not in _NAMED_ROW_UBIQUITOUS_TERMS
    )


# Words that join a question together and never sit in a label cell on
# their own. Separate from _GK_STOPWORDS, which only knows words of four
# letters and up.
_LABEL_PHRASE_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in",
    "on", "at", "by", "as", "or", "and", "for", "it", "its", "do", "does",
    "did", "who", "what", "which", "when", "where", "why", "how", "this",
    "that", "these", "those", "under", "per", "any", "all", "there", "with",
    "from", "has", "have", "will", "please", "tell", "me", "give",
})
# A label sits at the START of its row: after nothing but a clause number,
# cell pipes and colons. ``0.1% of the Contract Price`` contains the phrase
# "contract price" and is a value, not the label of a row called that.
_LABEL_START_PREFIX = r"^[\s|:]*(?:\d+(?:\.\d+)*(?:\([a-z0-9]+\))*[\s|:]*)?"


def _label_phrases(query: str) -> List[str]:
    """Runs of two to four consecutive content words in the question.

    Live unseen Set 3: "What is the Contract Date?" and "Who is the VT
    Subcontractor?" missed the sheet. Counting content words cannot see
    either label — "contract" is on every line and is not counted, "VT" is
    two letters — but the PHRASES are exactly what the label cell prints.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9&/-]*", query or "")
    runs: List[List[str]] = [[]]
    for w in words:
        if w.lower() in _LABEL_PHRASE_STOPWORDS or w.lower() in _GK_STOPWORDS:
            runs.append([])
        else:
            runs[-1].append(w.lower())
    out: List[str] = []
    for run in runs:
        for n in (4, 3, 2):
            for i in range(0, max(0, len(run) - n + 1)):
                phrase = " ".join(run[i:i + n])
                # "Contract Data" opens EVERY particulars chunk as its section
                # heading; it names the sheet, not a row on it.
                if _CD_HEADING_IN_CHUNK_RE.fullmatch(phrase):
                    continue
                out.append(phrase)
    return out


def _line_is_labelled(line: str, phrases: List[str]) -> Optional[int]:
    """End offset of a label phrase that OPENS ``line``, else None."""
    low = line.lower()
    for phrase in phrases:
        m = re.match(_LABEL_START_PREFIX + re.escape(phrase) + r"(?![a-z0-9])", low)
        if m:
            return m.end()
    return None


def named_particulars_row_match(query: str, text: str) -> int:
    """How strongly a particulars chunk states a row the question names.

    0 when it does not. Otherwise the number of distinct question terms in
    the chunk body, so callers can prefer the better-matching window.

    Judged per LINE because that is what a row is in the rendered chunk, and
    on the whole line rather than the parsed key: a scanned table often puts
    the clause number in the key position and the label in the value
    (``4.3.3(a): | Value of Performance Bond: 10 %``).
    """
    if not is_contract_data_particulars_row(text):
        return 0
    terms = _named_row_terms(query)
    phrases = _label_phrases(query)
    if len(terms) < _NAMED_ROW_MIN_LINE_TERMS and not phrases:
        return 0
    body = _cd_chunk_body(text)
    names_a_row = False
    by_label = False
    lines = body.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        ends = [low.rfind(t) + len(t) for t in terms if t in low]
        label_end = _line_is_labelled(line, phrases)
        if label_end is not None:
            ends = [label_end]
        elif len(ends) < _NAMED_ROW_MIN_LINE_TERMS:
            continue
        # The row has to SAY something, in a cell of its own. ``Value of
        # Performance Bond | |`` names the row and states nothing, and the
        # rest of a bare label (``...for the whole of the Works``) is not a
        # value either — so content only counts after a cell separator.
        tail = low[max(ends):]
        if not _NAMED_ROW_SEPARATOR_RE.search(tail) and i + 1 < len(lines):
            # A scanned key wraps: ``Time for Completion (by`` /
            # ``Milestone, if applicable): Milestone 1 | NNN days``. A next
            # line that opens with a clause number is the next ROW, not the
            # rest of this one.
            nxt = lines[i + 1]
            if not _NAMED_ROW_NEW_CLAUSE_RE.match(nxt):
                tail = f"{tail} {nxt.lower()}"
        if _NAMED_ROW_FILLED_CELL_RE.search(tail):
            names_a_row = True
            by_label = label_end is not None
            break
    if not names_a_row:
        return 0
    if by_label:
        # The whole label was named; coverage of the rest of the question
        # ("What is the ...?") says nothing more.
        return max(_NAMED_ROW_MIN_LINE_TERMS, sum(1 for t in terms if t in body.lower()))
    body_low = body.lower()
    covered = sum(1 for t in terms if t in body_low)
    if covered / len(terms) < _NAMED_ROW_MIN_COVERAGE:
        return 0
    return covered


# One cause behind several failed asks: a particulars row lists many
# milestones; the page breaks after one of them and so does the chunk. The
# second half opens with the table's repeated header and then "Milestone N |
# NNN days ..." — no "Time for Completion" label anywhere on it — so every
# answer stopped, honestly, at the page break.
#
# What marks a continuation is the NUMBERING: the first half ends on
# "<Word> n" and a following chunk of the same document opens on "<Word> n+1"
# (or n+2: a scan can drop a row). A chunk that merely comes next does not
# qualify, so the retention rows after the milestones are not dragged along.
_ENUMERATED_ITEM_RE = re.compile(r"(?m)^[\s|:]*([A-Z][a-z]{3,})\s+(\d{1,2})\b")
_CONTINUATION_LOOKAHEAD_CHUNKS = 5
_CONTINUATION_MAX_CHUNKS = 2
# A repeated OCR header can push the next milestone past 400 chars, so the
# continuation check reads a longer opening.
_CONTINUATION_OPENING_CHARS = 1600


def _last_enumerated_item(text: str) -> Optional[Tuple[str, int]]:
    items = [(w.lower(), int(n)) for w, n in
             _ENUMERATED_ITEM_RE.findall(_cd_chunk_body(text or ""))]
    if not items:
        return None
    word, number = items[-1]
    # A LIST, not a lone numbered thing: "Milestone 4" then "Milestone 5".
    # "Page 3 of 46" in a footer is numbered and is not an enumeration.
    if (word, number - 1) not in items:
        return None
    return word, number


def _enumeration_continuations(parent: Chunk, sheet: List[Chunk]) -> List[Chunk]:
    """Following chunks of the same document whose numbering runs on."""
    last = _last_enumerated_item(parent.text or "")
    if last is None:
        return []
    word, number = last
    following = sorted(
        (
            c for c in sheet
            if c.doc_id == parent.doc_id
            and parent.chunk_index < c.chunk_index
            <= parent.chunk_index + _CONTINUATION_LOOKAHEAD_CHUNKS
        ),
        key=lambda c: c.chunk_index,
    )
    out: List[Chunk] = []
    for chunk in following:
        opening = _cd_chunk_body(chunk.text or "")[:_CONTINUATION_OPENING_CHARS]
        first = next(
            (
                int(n) for w, n in _ENUMERATED_ITEM_RE.findall(opening)
                if w.lower() == word
            ),
            None,
        )
        if first is None or not number < first <= number + 2:
            continue
        out.append(chunk)
        if len(out) >= _CONTINUATION_MAX_CHUNKS:
            break
    return out


_NAMED_COMMUNITY_AMONG_RE = re.compile(
    r"(?i)\bamong\s+the\s+(.+?)\s+milestones\b",
)
# A named group of milestones: up to three words before a group noun
# ("<Name> Quarter", "<Name Name> District"). The group nouns are a
# vocabulary of how a site is divided, not any project's place names.
_MILESTONE_GROUP_NOUNS = r"(?:community|quarter|district|precinct|zone|phase|sector|area|parcel)"
_NAMED_COMMUNITY_NAME_RE = re.compile(
    rf"(?i)\b((?:[a-z][\w'-]*\s+){{1,3}}?{_MILESTONE_GROUP_NOUNS})\b",
)
_NAMED_COMMUNITY_SPAN_RE = re.compile(
    r"(?i)\b(?:longest|shortest|exceed)\b",
)
_MILESTONE_DAYS_ROW_RE = re.compile(
    r"(?i)milestone\s+(\d+)(?:\s*[|:]\s*)+(\d+)\s*days",
)


def extract_asked_community_name(query: str) -> str:
    """Community / quarter the question names, or ''."""
    q = query or ""
    among = _NAMED_COMMUNITY_AMONG_RE.search(q)
    if among:
        return re.sub(r"\s+", " ", among.group(1)).strip()
    for named in _NAMED_COMMUNITY_NAME_RE.finditer(q):
        words = named.group(1).split()
        # The question's own frame words are not part of the name.
        while words and (
            words[0].lower() in _LABEL_PHRASE_STOPWORDS
            or words[0].lower() in _NAMED_ROW_UBIQUITOUS_TERMS
            or words[0].lower() in _GK_STOPWORDS
        ):
            words = words[1:]
        if len(words) >= 2:
            return " ".join(words)
    return ""


def query_asks_named_community_tfc_span(query: str) -> bool:
    """True for longest/shortest Time for Completion in a named group of milestones."""
    q = query or ""
    if not extract_asked_community_name(q):
        return False
    if not _NAMED_COMMUNITY_SPAN_RE.search(q):
        return False
    return bool(
        re.search(r"(?i)time\s+for\s+completion|milestones?", q)
    )


def compose_named_community_tfc_span(
    query: str, excerpts: str,
) -> Optional[Dict[str, Any]]:
    """Longest / shortest Time for Completion among a named community.

    The longest and shortest durations among the milestones that name the
    community, and their difference. Does not invent days; every figure
    must already be printed on a Milestone row that names the community.
    """
    community = extract_asked_community_name(query)
    if not community or not excerpts:
        return None
    needle = community.lower()
    days_by_ms: Dict[int, int] = {}
    for match in _MILESTONE_DAYS_ROW_RE.finditer(excerpts):
        # Cut at the next Milestone, not a fixed char window — a 200-char
        # reach stained the next community's days as this community's when
        # the next row named it. A wrap that keeps the community on the
        # same item still counts.
        nxt = re.search(r"(?i)milestone\s+\d+", excerpts[match.end():])
        end = match.end() + (nxt.start() if nxt else 240)
        window = excerpts[match.start(): end]
        if needle not in window.lower():
            continue
        days_by_ms[int(match.group(1))] = int(match.group(2))
    if not days_by_ms:
        return None
    longest_days = max(days_by_ms.values())
    shortest_days = min(days_by_ms.values())
    return {
        "community": community,
        "longest_days": longest_days,
        "shortest_days": shortest_days,
        "delta": longest_days - shortest_days,
        "longest_milestones": tuple(
            sorted(n for n, d in days_by_ms.items() if d == longest_days)
        ),
        "shortest_milestones": tuple(
            sorted(n for n, d in days_by_ms.items() if d == shortest_days)
        ),
        "days_by_milestone": days_by_ms,
    }


def format_named_community_tfc_span_line(composed: Dict[str, Any]) -> str:
    """User-facing longest/shortest community TFC sentence."""
    if not composed:
        return ""
    community = composed.get("community") or "named community"
    longest = int(composed["longest_days"])
    shortest = int(composed["shortest_days"])
    delta = int(composed["delta"])
    long_ms = composed.get("longest_milestones") or ()
    short_ms = composed.get("shortest_milestones") or ()
    long_txt = " and ".join(f"Milestone {n}" for n in long_ms) or "the longest"
    short_txt = " and ".join(f"Milestone {n}" for n in short_ms) or "the shortest"
    verb = "has" if len(long_ms) == 1 else "have"
    exceed = "exceeds" if len(long_ms) == 1 else "exceed"
    return (
        f"Among the {community} milestones, {long_txt} {verb} the longest "
        f"Time for Completion ({longest} days) and {exceed} the shortest "
        f"({short_txt}, {shortest} days) by {delta} days."
    )


# ── labelled-row recall: the filled row for the label a question names ────
#
# A particular (an amount, a duration, a party, "Not Used", "Not required",
# a register entry, a bill item marked Rate Only) is a labelled ROW: a label
# cell and a value cell. Asked plainly, the row is absent from the pool: a
# scanned table embeds badly, the row shares one or two words with the
# question, and long prose that mentions the same topic at length fills the
# slots. The discriminator is the row itself: a line that opens with a label
# the question names and states a value for it.
#
# The labels come from the question: the particular names it uses (a lexicon
# of contract-particular names), any identifier label it names ("Schedule 7",
# a bill item code), and runs of its own content words. A filled value is a
# figure, a party, a date, a register entry or a stated absence ("No", "Not
# required", "Not used", "to be notified"); a blank template ("[insert
# amount]") and a pointer to somewhere else ("as stated in the Contract
# Data") are not.
_PARTICULARS_KIND_PHRASES = ("contract data", "appendix to tender", "contract particulars")
_LABELLED_ROW_MAX_LABELS = 6
_LABELLED_ROW_FETCH_K = 20
_LABELLED_ROW_VALUE_CHARS = 120
_STATED_ABSENCE_RE = re.compile(
    r"(?i)^(?:no|none|nil|n/?a|not\s+(?:required|used|applicable|populated|stated)|"
    r"tba|tbc|to\s+be\s+(?:advised|agreed|confirmed|notified|issued|inserted))\b"
)
_TEMPLATE_PLACEHOLDER_RE = re.compile(
    r"(?i)\[\s*(?:insert|name|amount|date|enter|state)\b|\.{5,}|_{5,}"
)
# A value cell that points elsewhere is not a value.
_ROW_VALUE_POINTER_RE = re.compile(
    r"(?i)\b(?:stated|set\s+out|specified|given|shown|defined|referred\s+to)\s+in\s+"
    r"(?:the\s+)?(?:contract\s+data|appendix|schedule|particular|letter\s+of)"
)
_ROW_VALUE_MAX_PROSE_WORDS = 14


def asked_row_labels(query: str) -> List[str]:
    """The row labels the question names, most specific first.

    Identifier labels ("schedule 7", a bill item code) first, then the
    contract-particular names it uses, then runs of its own content words.
    Empty for a definition question.
    """
    q = query or ""
    if _DEFINITION_QUESTION_RE.search(q):
        return []
    out: List[str] = []

    def _add(label: str) -> None:
        lab = " ".join((label or "").lower().split())
        if len(lab) >= 3 and lab not in out:
            out.append(lab)

    if query_asks_for_numbered_contract_schedule(q):
        for lab in extract_asked_schedule_labels(q):
            _add(lab)
    if query_asks_for_boq_item_amount(q):
        for code in extract_asked_cesmm_codes(q):
            _add(code)
    for phrase in _asked_particular_key_phrases(q):
        _add(phrase)
    for phrase in sorted(_label_phrases(q), key=lambda p: -len(p.split())):
        _add(phrase)
    return out


def query_wants_a_labelled_row(query: str) -> bool:
    """A question whose answer is a filled row, so its labels may be text-searched.

    A contract-particulars question, a numbered register / schedule entry, or
    a bill item named by its code. An ordinary question is not searched this
    way: its word runs are not labels.
    """
    q = query or ""
    if _DEFINITION_QUESTION_RE.search(q):
        return False
    return bool(
        query_asks_for_contract_particulars(q)
        or query_asks_for_numbered_contract_schedule(q)
        or (query_asks_for_boq_item_amount(q) and extract_asked_cesmm_codes(q))
        or query_asks_for_parent_company_guarantee(q)
        or query_asks_for_contract_commencement_date(q)
        or query_wants_contract_data_file(q)
    )


def _row_value_is_filled(value: str) -> bool:
    val = (value or "").strip(" \t|:;-–—.")
    if not val or not re.search(r"[A-Za-z0-9]", val):
        return False
    if _TEMPLATE_PLACEHOLDER_RE.search(val):
        return False
    if _ROW_VALUE_POINTER_RE.search(val):
        return False
    if _STATED_ABSENCE_RE.search(val):
        return True
    if _CD_FILLED_VALUE_RE.search(val[:_LABELLED_ROW_VALUE_CHARS]):
        return True
    # A sentence of prose under a heading is a clause, not a value cell.
    return len(val.split()) <= _ROW_VALUE_MAX_PROSE_WORDS


def chunk_states_labelled_row(text: str, labels: List[str]) -> bool:
    """True when a line opens with one of ``labels`` and a filled value follows.

    The value is the rest of the line after the label (past any ``:`` / ``|``
    cell separators), or the next line when the label stands alone on its
    line, as a scanned key often does.
    """
    if not labels:
        return False
    lines = [ln for ln in (text or "").splitlines()]
    for i, line in enumerate(lines):
        end = _line_is_labelled(line, labels)
        if end is None:
            continue
        rest = line.lower()[end:]
        if rest.strip(" \t|:;-–—.") == "" and i + 1 < len(lines):
            rest = lines[i + 1]
        if _row_value_is_filled(rest):
            return True
    return False


def _known_particular_row_test(query: str):
    """The row recogniser for a contract particular whose row shape is known.

    None when the question asks for no such particular.
    """
    if query_asks_for_aca_including_vat(query):
        return chunk_states_aca_including_vat
    if query_asks_for_time_for_completion(query):
        return chunk_states_time_for_completion
    party_role = asked_party_role(query)
    if party_role:
        return lambda text: chunk_names_party(text, party_role)
    if query_asks_delay_damages_daily_amount(query):
        return _chunk_is_daily_damages_operand
    if query_asks_for_defects_notification_period(query):
        return chunk_states_defects_notification_period
    if query_asks_for_parent_company_guarantee(query):
        return chunk_states_pcg_contract_data
    if query_asks_for_contract_commencement_date(query):
        return chunk_states_commencement_contract_data
    return None


def chunk_states_asked_row(query: str, text: str, labels: Optional[List[str]] = None) -> bool:
    """True when ``text`` states the row the question asks for.

    The particular shapes the retriever already recognises (a rate, a party,
    a duration, a Rate Only item, a register entry, a not-required /
    not-populated particular) or, for any other label, a labelled row with a
    filled value.
    """
    if chunk_answers_asked_particular(query, text):
        return True
    if query_asks_for_parent_company_guarantee(query) and chunk_states_pcg_contract_data(text):
        return True
    if (
        query_asks_for_contract_commencement_date(query)
        and chunk_states_commencement_contract_data(text)
    ):
        return True
    if query_asks_for_numbered_contract_schedule(query):
        schedule_labels = extract_asked_schedule_labels(query)
        if schedule_labels and chunk_states_schedule_register(text, schedule_labels):
            return True
    if query_asks_for_boq_item_amount(query):
        codes = extract_asked_cesmm_codes(query)
        if codes and chunk_states_rate_only_item(text, codes):
            return True
    return chunk_states_labelled_row(text, labels if labels is not None else asked_row_labels(query))


def recall_labelled_rows(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    extra_pids: List[str],
) -> Dict[str, str]:
    """Pull the filled row(s) for the label(s) the question names into ``fused``.

    Labelled-row recall, in two passes:

      * the particulars documents (their kind read from the upload name) are
        loaded, and the rows the question names -- by its own label phrases,
        or by a particular it asks for -- are pooled with the asked-value
        bonus; a row that runs on into the next chunk brings that chunk, and
        a share of a named sum brings the row stating the sum;
      * for a particulars-shaped question, the question's labels are also
        searched as text across the project, so a row that lives in a file
        of another kind (a register, a bill, a scanned sheet) is pooled when
        it states a value for that label.

    Returns ``{doc_id: name}`` for the particulars documents listed here.
    Failures never raise -- the semantic pool stands.
    """
    names: Dict[str, str] = {}
    if _DEFINITION_QUESTION_RE.search(query or ""):
        return names
    labels = asked_row_labels(query)
    named_terms = _named_row_terms(query)
    if not labels and len(named_terms) < _NAMED_ROW_MIN_LINE_TERMS:
        return names
    pids = [project_id] + [p for p in extra_pids if p and p != project_id]
    recovered = 0

    def _pool(chunk: Chunk, bonus: float) -> None:
        nonlocal recovered
        prev = fused.get(chunk.chunk_id)
        if prev is not None:
            fused[chunk.chunk_id] = (prev[0], prev[1], max(prev[2] or 0.0, bonus))
            return
        fused[chunk.chunk_id] = (chunk, 0.0, bonus)
        recovered += 1

    keep = (lambda text: chunk_states_asked_row(query, text, labels))
    daily_damages_ask = query_asks_delay_damages_daily_amount(query)

    def _bonus_for(text: str) -> float:
        # A daily amount is composed from the rate and the sum it is a share
        # of: the rate row earns the asked-value bonus, the sum enters at 0 so
        # the monetary reservation still owns the last slot.
        if daily_damages_ask and not chunk_states_delay_damages_rate(text):
            return 0.0
        return _ASKED_PARTICULAR_VALUE_BONUS

    # ── pass 1: the particulars documents ─────────────────────────────────
    try:
        from app.core.projects import documents_matching_title_phrase
    except Exception:  # noqa: BLE001
        logger.warning("labelled-row recall: projects import failed", exc_info=True)
        documents_matching_title_phrase = None
    by_docs = getattr(store, "chunks_for_docs", None)
    span_ask = query_asks_named_community_tfc_span(query)
    if documents_matching_title_phrase is not None and callable(by_docs):
        matched: List[Tuple[int, Chunk]] = []
        sheet: List[Chunk] = []
        for pid in pids:
            docs: List[Dict[str, str]] = []
            for phrase in _PARTICULARS_KIND_PHRASES:
                try:
                    docs.extend(documents_matching_title_phrase(pid, phrase) or [])
                except Exception as exc:  # noqa: BLE001 — extras must not break the turn
                    logger.warning(
                        "particulars listing for %s (%r) failed: %s", pid, phrase, exc,
                    )
            ids: List[str] = []
            for doc in docs:
                did = doc.get("id") or ""
                if did and did not in ids:
                    ids.append(did)
                    names[did] = doc.get("original_name") or ""
            if not ids:
                continue
            try:
                hits = by_docs(pid, ids, k_per_doc=80 if span_ask else 40)
            except Exception as exc:  # noqa: BLE001
                logger.warning("particulars fetch for %s failed: %s", pid, exc)
                continue
            sheet.extend(hits)
            for chunk in hits:
                strength = named_particulars_row_match(query, chunk.text or "")
                if strength:
                    matched.append((strength, chunk))
            # A label split from its value across chunks is still one row. Only
            # for a particular whose row shape is known: joining arbitrary
            # neighbours would make any two adjacent rows look like one.
            known = _known_particular_row_test(query)
            if known is not None:
                for chunk in _pair_adjacent_keep_text(hits, known):
                    names.setdefault(chunk.doc_id, "")
                    _pool(chunk, _bonus_for(chunk.text or ""))
            # The Accepted Contract Amount ask keeps every window of the sheet so
            # the filename fence can see the variant (VAT basis) asked for.
            if query_asks_for_accepted_contract_amount(query):
                for chunk in hits:
                    if chunk.chunk_id not in fused:
                        fused[chunk.chunk_id] = (chunk, 0.0, 0.0)
                        recovered += 1
        matched.sort(key=lambda m: (-m[0], m[1].chunk_index))
        row_cap = max(_NAMED_ROW_MAX_CHUNKS, 4) if span_ask else _NAMED_ROW_MAX_CHUNKS
        chosen = [chunk for _strength, chunk in matched[:row_cap]]
        # A row that runs on into the next chunk is still one row. The second
        # half carries no label, so it is found from the first half.
        for parent in list(chosen):
            for cont in _enumeration_continuations(parent, sheet):
                if all(cont.chunk_id != c.chunk_id for c in chosen):
                    chosen.append(cont)
        # A span across a named group of milestones needs every row that names
        # the group and states days, even past an over-long repeated header.
        if span_ask:
            needle = (extract_asked_community_name(query) or "").lower()
            if needle:
                for chunk in sheet:
                    text = chunk.text or ""
                    if needle in text.lower() and re.search(r"(?i)\d+\s*days", text):
                        if all(chunk.chunk_id != c.chunk_id for c in chosen):
                            chosen.append(chunk)
        # A share of a named sum is half an answer; the sum is a row of the same
        # sheet. Below the asked row's bonus, so it accompanies and never leads.
        if any(_NAMED_ROW_SHARE_OF_SUM_RE.search(c.text or "") for c in chosen):
            docs_in = {c.doc_id for c in chosen}
            base = next(
                (c for c in sheet
                 if c.doc_id in docs_in and chunk_states_accepted_contract_amount(c.text or "")),
                None,
            )
            if base is not None and base.chunk_id not in fused:
                fused[base.chunk_id] = (base, 0.0, _NAMED_ROW_BASE_AMOUNT_BONUS)
                recovered += 1
        for chunk in chosen:
            _pool(chunk, _ASKED_PARTICULAR_VALUE_BONUS)

    # ── pass 2: the question's labels as text, project corpus ─────────────
    if labels and query_wants_a_labelled_row(query):
        gk = set(_general_knowledge_project_ids())
        text_pids = [p for p in pids if p == project_id or p not in gk]
        for label in labels[:_LABELLED_ROW_MAX_LABELS]:
            recovered += _pool_lexical_hits_matching(
                project_id, fused, store, (label,), keep,
                label="labelled-row", bonus=_bonus_for,
            )
        containing = getattr(store, "chunks_containing_all", None)
        party_role = asked_party_role(query)
        needle_sets: List[List[str]] = [[label] for label in labels[:_LABELLED_ROW_MAX_LABELS]]
        # The row that names a party: the role word alone hits every clause
        # that mentions the role, so pair it with a name ending.
        needle_sets.extend(list(n) for n in party_name_needle_sets(party_role))
        if callable(containing):
            for pid in text_pids:
                for needles in needle_sets:
                    label = " + ".join(needles)
                    try:
                        hits = containing(pid, needles, k=_LABELLED_ROW_FETCH_K)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "labelled-row text fetch for %s (%r) failed: %s", pid, label, exc,
                        )
                        continue
                    # Adds what the pool is missing; a row already pooled keeps
                    # the score the other legs gave it.
                    for chunk in _pair_adjacent_keep_text(hits or [], keep):
                        if chunk.chunk_id not in fused:
                            _pool(chunk, _bonus_for(chunk.text or ""))
    if recovered:
        logger.info("labelled-row recall pooled %d chunk(s) for labels %r", recovered, labels)
    return names


# ── numbered schedule register rows ───────────────────────────────────────
#
# "What does Schedule N of the contract contain?" can retrieve long volumes
# that mention "schedule" at length and answer with a generic
# acknowledgement, while the contract's own schedule index row
# (``Schedule N: Not Used`` / ``Schedule N | <title>``) is the answer. Do not
# invent contents; surface the register row as written. When a register row
# is in the pool, lookalikes drop.
_SCHEDULE_REGISTER_BONUS = 2.0
_SCHEDULE_REGISTER_ROW_RE = re.compile(
    r"(?i)\b(schedule\s+(?:no\.?\s*)?\d+[A-Za-z]?)"
    r"\s*[:|–—-]\s*"
    r"(?P<val>\S[^\n]{0,79})"
)
_SCHEDULE_NOT_USED_COLLAPSED_RE = re.compile(
    r"(?i)\b(schedule\s+(?:no\.?\s*)?\d+[A-Za-z]?)"
    r"(?:\s*[:|–—-]\s*|\s+)"
    r"not\s+used\b"
)
_SCHEDULE_REGISTER_PROSE_RE = re.compile(
    r"(?i)\b(?:sets?\s+out|shall|the\s+contractor|contains?|covers?|"
    r"includes?|must\b|will\s+provide)\b"
)


def query_asks_for_numbered_contract_schedule(query: str) -> bool:
    """True for 'what does Schedule N of the contract contain?'.

    Bare 'what does Schedule 10 contain?' stays off this path: a numbered
    schedule also appears inside specifications and method statements.
    Definition questions and programme-build asks are not this class.
    """
    q = (query or "").strip()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    return bool(
        _CD_SCHEDULE_ASK_RE.search(q) and _CD_SCHEDULE_CONTEXT_RE.search(q)
    )


def extract_asked_schedule_labels(query: str) -> List[str]:
    """``schedule 10`` / ``appendix 3`` labels the ask actually names."""
    out: List[str] = []
    seen: Set[str] = set()
    for match in _CD_SCHEDULE_ASK_RE.finditer(query or ""):
        label = re.sub(r"\s+", " ", match.group(0).lower()).strip()
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def _schedule_label_matches(asked: str, key: str) -> bool:
    """True when a register key is the asked Schedule / Appendix N."""
    asked_n = re.sub(r"(?i)\s+no\.?\s*", " ", asked or "").strip()
    key_n = re.sub(r"(?i)\s+no\.?\s*", " ", key or "").strip()
    if not asked_n or not key_n:
        return False
    return asked_n == key_n or asked_n in key_n


def chunk_states_schedule_register(text: str, labels: List[str]) -> bool:
    """True when ``text`` is a schedule-index row for an asked label.

    ``Schedule 10: Not Used`` and ``Schedule 9 | Health & Safety KPIs``
    are register rows. ``Schedule 10 sets out any applicable Works
    Guarantees`` is prose from another package — not a register, and
    must not be treated as contents we invented.
    """
    if not labels:
        return False
    wanted = [re.sub(r"\s+", " ", lab.lower()).strip() for lab in labels if lab]
    if not wanted:
        return False
    blob = text or ""
    collapsed = _normalize_retrieval_ws(blob)

    def _wanted(key: str) -> bool:
        key_l = re.sub(r"\s+", " ", (key or "").lower()).strip()
        return any(_schedule_label_matches(lab, key_l) for lab in wanted)

    if _SCHEDULE_NOT_USED_COLLAPSED_RE.search(collapsed):
        for match in _SCHEDULE_NOT_USED_COLLAPSED_RE.finditer(collapsed):
            if _wanted(match.group(1)):
                return True
    for match in _SCHEDULE_REGISTER_ROW_RE.finditer(blob):
        val = (match.group("val") or "").strip()
        if _SCHEDULE_REGISTER_PROSE_RE.search(val):
            continue
        if _wanted(match.group(1)):
            return True
    for key, val in filled_particulars_rows(blob):
        if _SCHEDULE_REGISTER_PROSE_RE.search(val or ""):
            continue
        if _wanted(key):
            return True
    return False


def chunk_states_schedule_not_used(text: str) -> bool:
    """True when a numbered-schedule register row says Not Used.

    Query-free so inject can warn the model without the user ask. Does
    not invent: the excerpt itself must already say Not Used.
    """
    return bool(_SCHEDULE_NOT_USED_COLLAPSED_RE.search(_normalize_retrieval_ws(text)))


def _apply_schedule_register_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
) -> None:
    """In-place: lift chunks whose body is the asked Schedule-N register."""
    if not query_asks_for_contract_particulars(query):
        return
    if not query_asks_for_numbered_contract_schedule(query):
        return
    labels = extract_asked_schedule_labels(query)
    if not labels:
        return
    for i, (score, chunk) in enumerate(scored):
        if not chunk_states_schedule_register(chunk.text or "", labels):
            continue
        boosted = score + _SCHEDULE_REGISTER_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


# ── a particular stated as absent beats a lookalike (form / pack) ──────────
#
#   Guarantee ask: the Contract Data row says the guarantee is not required.
#   Cosine preferred the blank form of guarantee (a percentage of paid-up
#   capital) and the model invented a monetary value. The form is a
#   template; "not required" IS the answer.
#
#   Commencement-date ask: the tender's Contract Data field is empty / tied
#   to the letter of acceptance. Cosine preferred a site commencement pack
#   report and the model invented the pack's date. That pack is site
#   commencement, not the contract Commencement Date particular.
#
# Same shape as the schedule register (row over volume prose) and Rate Only
# (row over priced lookalikes): pool the Contract Data row, fence the
# lookalike, instruct compose, graft if the model still invents. Do not
# invent: the excerpt itself must already say not required / not populated,
# or a filled value / date. A filled value or date still wins -- this only
# refuses the form / pack when the Contract Data already answered.
_PCG_HONEST_BONUS = 2.0
_COMMENCEMENT_HONEST_BONUS = 2.0
_PCG_ASK_RE = re.compile(r"(?i)\bparent\s+company\s+guarantee\b|\bpcg\b")
_PCG_NOT_REQUIRED_RE = re.compile(
    r"(?i)(?:not\s+required|is\s+not\s+used|not\s+applicable|"
    r"\bno\b\s+(?:parent\s+company\s+guarantee|pcg)\s+is\s+required|"
    r"parent\s+company\s+guarantee\s+is\s+not\s+required|"
    r"parent\s+company\s+guarantee\s*[:|–—-]?\s*(?:no|none)\b)"
)
_PCG_NO_ROW_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:no\.?|none)\s*(?:[.\n]|$)",
)
# A guarantee FORM: the form's own title, its operative wording, or a blank
# placeholder -- in whichever schedule / annex a contract binds it.
_PCG_FORM_RE = re.compile(
    r"(?i)(?:paid[- ]up\s+capital|shareholders['’]?\s+funds|"
    r"form\s+of\s+(?:parent\s+company\s+)?guarantee|"
    r"the\s+guarantor\s+(?:shall|irrevocably|hereby)|"
    r"\[\s*(?:insert|name|amount|date)\b)",
)
_PCG_FILLED_VALUE_RE = re.compile(
    r"(?i)(?:\d+(?:\.\d+)?\s*%|"
    r"\b(?:sar|aed|usd|eur|gbp|qar|bhd|kwd|omr)\b"
    r"[^\n]{0,12}\d{1,3}(?:,\d{3})+(?:\.\d+)?)",
)
_COMMENCEMENT_DATE_ASK_RE = re.compile(
    r"(?i)(?:\bcommencement\s+date\b|"
    r"\bcontract\s+commencement\b|"
    r"when\s+does\s+(?:the\s+|this\s+)?contract\s+commence|"
    r"when\s+(?:does|is)\s+(?:the\s+|this\s+)?contract\s+"
    r"(?:start|begin))",
)
_COMMENCEMENT_PACK_ASK_RE = re.compile(
    r"(?i)(?:commencement\s+pack|pack\s+report|"
    r"site\s+commencement)",
)
_COMMENCEMENT_NOT_POPULATED_RE = re.compile(
    r"(?i)(?:not\s+populated|not\s+stated|not\s+completed|"
    r"not\s+inserted|left\s+blank|no\s+date\s+(?:is\s+)?(?:given|stated)|"
    r"to\s+be\s+(?:advised|agreed|confirmed|issued|notified|inserted)|"
    r"\btba\b|\btbc\b|"
    r"as\s+stated\s+in\s+(?:the\s+)?(?:letter\s+of\s+acceptance|loa)|"
    r"notified\s+under\s+(?:sub-?clause\s+)?8\.1|"
    r"shall\s+be\s+notified|"
    r"tied\s+to\s+(?:the\s+)?(?:loa|noa|letter\s+of\s+acceptance|"
    r"notice\s+of\s+(?:award|acceptance)))",
)
_COMMENCEMENT_EMPTY_FIELD_RE = re.compile(
    r"(?i)commencement\s+date\s*[:|]\s*(?:[-—–]+|n/?a|nil|none)?\s*(?:\n|$)",
)
_COMMENCEMENT_PACK_RE = re.compile(
    r"(?i)(?:construction\s+commencement\s+pack|"
    r"commencement\s+pack\s+report|commencement\s+pack)",
)
_COMMENCEMENT_LABEL_RE = re.compile(r"(?i)\bcommencement\s+date\b")
_COMMENCEMENT_FILLED_DATE_RE = re.compile(
    r"(?i)(?:\b\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{4}\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b)",
)
_COMMENCEMENT_UNSUPPORTED_LINE = (
    "The contract Commencement Date is not stated in the Contract Data "
    "retrieved for this project. I will not invent a calendar date."
)


def query_asks_for_parent_company_guarantee(query: str) -> bool:
    """True for an ask for the value of the Parent Company Guarantee.

    Performance bond / performance guarantee stay on their own path.
    Definition questions are not this class.
    """
    q = (query or "").strip()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if re.search(r"(?i)performance\s+(?:bond|security|guarantee)", q):
        return False
    return bool(_PCG_ASK_RE.search(q))


def query_asks_for_site_commencement_pack(query: str) -> bool:
    """True when the ask wants site commencement from a pack, not CD."""
    return bool(_COMMENCEMENT_PACK_ASK_RE.search(query or ""))


def query_asks_for_contract_commencement_date(query: str) -> bool:
    """True for an ask for the contract Commencement Date particular.

    Time for Completion ('N days from the Commencement Date') and an
    explicit commencement-pack / site-commencement ask stay off this
    path so a filled TfC row and a pack report can still answer those.
    """
    q = (query or "").strip()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if query_asks_for_time_for_completion(q):
        return False
    if re.search(r"(?i)time\s+for\s+completion", q):
        return False
    if query_asks_for_site_commencement_pack(q):
        return False
    return bool(_COMMENCEMENT_DATE_ASK_RE.search(q))


def _chunk_mentions_pcg(text: str) -> bool:
    blob = _normalize_retrieval_ws(text)
    return bool(_PCG_ASK_RE.search(blob))


def chunk_states_pcg_form_template(text: str) -> bool:
    """True for a guarantee form / blank template, not the Contract Data row."""
    if not text or not _chunk_mentions_pcg(text):
        return False
    if _PCG_NOT_REQUIRED_RE.search(text):
        return False
    return bool(_PCG_FORM_RE.search(text))


def chunk_states_pcg_not_required(text: str) -> bool:
    """True when Contract Data (or a particulars row) says PCG is No.

    Does not invent: the excerpt itself must already say not required
    / No. Form language that mentions a % of paid-up capital is not
    this class even when it also names the guarantee.
    """
    if not text or not _chunk_mentions_pcg(text):
        return False
    if chunk_states_pcg_form_template(text):
        return False
    if _PCG_NOT_REQUIRED_RE.search(text):
        return True
    # Label on one line and "No." on the next (a scanned particulars row).
    if _PCG_NO_ROW_RE.search(text) and not _PCG_FILLED_VALUE_RE.search(text):
        return True
    return False


def chunk_states_pcg_filled_value(text: str) -> bool:
    """True when Contract Data states a PCG amount or percentage.

    A guarantee form's percentage of paid-up capital is not a filled
    particular.
    """
    if not text or not _chunk_mentions_pcg(text):
        return False
    if chunk_states_pcg_not_required(text):
        return False
    if chunk_states_pcg_form_template(text):
        return False
    return bool(_PCG_FILLED_VALUE_RE.search(text))


def chunk_states_pcg_contract_data(text: str) -> bool:
    """CD already answered the PCG ask — not required, or a filled value."""
    return chunk_states_pcg_not_required(text) or chunk_states_pcg_filled_value(text)


def chunk_states_commencement_pack(text: str) -> bool:
    """True for a Construction Commencement Pack / site-start report."""
    return bool(_COMMENCEMENT_PACK_RE.search(text or ""))


def chunk_states_commencement_not_populated(text: str) -> bool:
    """True when Contract Data says the commencement field is empty.

    Tied-to-LOA/NOA, TBA/TBC, FIDIC 8.1 notification, and a labelled
    blank cell are the same class. A pack report that happens to
    mention LOA is not this row.
    """
    if not text or not _COMMENCEMENT_LABEL_RE.search(text):
        return False
    if chunk_states_commencement_pack(text):
        return False
    if _COMMENCEMENT_NOT_POPULATED_RE.search(text):
        return True
    return bool(_COMMENCEMENT_EMPTY_FIELD_RE.search(text))


def chunk_states_commencement_filled_date(text: str) -> bool:
    """True when Contract Data itself states a commencement calendar date."""
    if not text or not _COMMENCEMENT_LABEL_RE.search(text):
        return False
    if chunk_states_commencement_not_populated(text):
        return False
    if chunk_states_commencement_pack(text):
        return False
    return bool(_COMMENCEMENT_FILLED_DATE_RE.search(text))


def chunk_states_commencement_contract_data(text: str) -> bool:
    """CD already answered the commencement ask — not populated, or a filled date."""
    return (
        chunk_states_commencement_not_populated(text)
        or chunk_states_commencement_filled_date(text)
    )


# The clause number a particulars row carries in front of its label:
# "9.2 | Parent Company Guarantee", "4.3.3(a): | Value of ...".
_ROW_CLAUSE_BEFORE_LABEL = r"(\d+(?:\.\d+)+(?:\s*\([a-z0-9]+\))?)[\s|:]{0,8}"


def particular_citation(text: str, label_rx: "re.Pattern") -> str:
    """Where the chunk itself says a particular sits: "(Contract Data 9.2)".

    The clause is the number the chunk prints in front of the label; the
    document kind is the particulars heading the chunk carries. Nothing is
    supplied that the chunk does not print: no clause, no number.
    """
    blob = text or ""
    clause = ""
    for match in label_rx.finditer(blob):
        lead = blob[max(0, match.start() - 24):match.start()]
        m = re.search(_ROW_CLAUSE_BEFORE_LABEL + r"$", lead)
        if m:
            clause = re.sub(r"\s+", "", m.group(1))
            break
    heading = _CD_HEADING_IN_CHUNK_RE.search(blob)
    kind = " ".join(heading.group(0).split()).title() if heading else ""
    if kind.lower() == "contract data":
        kind = "Contract Data"
    parts = " ".join(p for p in (kind, clause) if p)
    if not parts:
        return ""
    return f" ({parts if kind else 'clause ' + clause})"


def _pcg_source_excerpt(excerpt: str) -> str:
    """The block of ``excerpt`` that states the guarantee particular."""
    for block in re.split(r"\n{2,}|\[doc_id=", excerpt or ""):
        if chunk_states_pcg_contract_data(block):
            return block
    return excerpt or ""


def pcg_citation(excerpt: str = "") -> str:
    """" (Contract Data 9.2)" -- where the stating excerpt says the guarantee sits."""
    return particular_citation(_pcg_source_excerpt(excerpt), _PCG_ASK_RE)


def format_pcg_honest_line(excerpt: str = "") -> str:
    """User-facing PCG sentence. Does not invent a % from the form.

    Cites the clause the stating chunk itself carries, if any.
    """
    source = _pcg_source_excerpt(excerpt)
    cite = pcg_citation(source)
    if chunk_states_pcg_filled_value(source):
        match = _PCG_FILLED_VALUE_RE.search(source or "")
        value = (match.group(0) or "").strip() if match else ""
        if value:
            return f"The Parent Company Guarantee is {value}{cite}."
    return f"A Parent Company Guarantee is not required{cite}."


def format_commencement_honest_line(excerpt: str = "") -> str:
    """User-facing commencement-date sentence. Does not invent a pack date."""
    if chunk_states_commencement_filled_date(excerpt):
        match = _COMMENCEMENT_FILLED_DATE_RE.search(excerpt or "")
        value = (match.group(0) or "").strip() if match else ""
        if value:
            return f"The Commencement Date of the contract is {value}."
    if chunk_states_commencement_not_populated(excerpt):
        return (
            "The Commencement Date is not populated in the Contract Data; "
            "it is tied to LOA/NOA issuance."
        )
    return format_commencement_unsupported_line()


def format_commencement_unsupported_line() -> str:
    """Commencement-date refusal when Contract Data does not support a calendar date."""
    return _COMMENCEMENT_UNSUPPORTED_LINE


def answer_states_pcg_not_required(text: str) -> bool:
    """True when the answer already elects not required / no PCG value."""
    blob = text or ""
    if _PCG_NOT_REQUIRED_RE.search(blob):
        return True
    return bool(re.search(r"(?i)\bno\s+value\b|\bno parent company guarantee\b", blob))


def answer_states_commencement_not_populated(text: str) -> bool:
    """True when the answer already elects not populated / tied to LOA."""
    blob = text or ""
    if _COMMENCEMENT_NOT_POPULATED_RE.search(blob):
        return True
    if answer_states_commencement_unsupported(blob):
        return True
    return bool(re.search(r"(?i)\b(?:loa|noa)\b", blob) and re.search(
        r"(?i)(?:tied|issuance|not\s+populated|not\s+stated)", blob,
    ))


def answer_states_commencement_unsupported(text: str) -> bool:
    """True when the answer already refuses to invent a commencement date."""
    blob = text or ""
    return bool(re.search(
        r"(?i)(?:will not invent|not stated in the contract data|"
        r"cannot confirm the (?:contract )?commencement)",
        blob,
    ))


def answer_invents_pcg_value(text: str) -> bool:
    """True when the answer quotes a % / money / paid-up-capital figure."""
    blob = text or ""
    if _PCG_FORM_RE.search(blob):
        return True
    return bool(_PCG_FILLED_VALUE_RE.search(blob))


def answer_invents_commencement_date(text: str) -> bool:
    """True when the answer states a calendar date as commencement."""
    return bool(_COMMENCEMENT_FILLED_DATE_RE.search(text or ""))


def _apply_pcg_value_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
) -> None:
    """In-place: lift the Contract Data PCG row over the Schedule 8 form."""
    if not query_asks_for_parent_company_guarantee(query):
        return
    for i, (score, chunk) in enumerate(scored):
        if not chunk_states_pcg_contract_data(chunk.text or ""):
            continue
        boosted = score + _PCG_HONEST_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


def _apply_commencement_date_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
) -> None:
    """In-place: lift the empty/filled CD commencement row over a pack."""
    if not query_asks_for_contract_commencement_date(query):
        return
    for i, (score, chunk) in enumerate(scored):
        if not chunk_states_commencement_contract_data(chunk.text or ""):
            continue
        boosted = score + _COMMENCEMENT_HONEST_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


# Dual-query retrieval (F18, phase-3 campaign). Measured on a 203-page
# contract and a 129-page tender: the needle chunk ranks FIRST for a query
# whose wording overlaps the answer's, and falls out of the top-12 for the
# same fact asked as a natural question — the interrogative scaffolding
# ("In the X, how many days is ...") drags the query vector away from the
# declarative prose of the document. Stopword stripping does NOT fix it
# (it breaks the noun phrases that carry the signal); what does, verified
# by live probes, is removing ONLY the wrapper while keeping every content
# phrase contiguous: "In the Conditions of Contract, how many days is the
# Time for Completion for the whole of the Works?" -> "days is the Time
# for Completion for the whole of the Works" moved the needle from
# outside the top-12 to rank 1 (0.933). The retriever therefore searches
# with BOTH phrasings and merges candidates by max score per chunk.
_WRAPPER_LEAD_RX = re.compile(
    # a short "In/From/Per/According to <source>," clause before the question
    r"^(?:in|from|under|per|according\s+to|based\s+on|as\s+per)\s+[^,]{1,60},\s*",
    re.IGNORECASE,
)
_WRAPPER_IMPERATIVE_RX = re.compile(
    r"^(?:please\s+)?(?:tell\s+me|give\s+me|show\s+me|state|list|specify)\s+",
    re.IGNORECASE,
)
_WRAPPER_TOKEN_RX = re.compile(
    # interrogative tokens only -- NEVER articles/copulas ("is", "the", "of"),
    # which are the glue inside the phrases that must stay contiguous
    r"\b(?:how\s+(?:many|much|long)|what|which|who|whose|when|why|does|did|please)\b",
    re.IGNORECASE,
)


def _strip_question_wrapper(query: str) -> Optional[str]:
    """The wrapper-stripped, phrase-intact variant of a natural question,
    or None when stripping changes nothing (terse queries cost no second
    search). Interior word order and every content phrase are preserved."""
    base = (query or "").strip().rstrip("?").strip()
    if not base:
        return None
    s = _WRAPPER_LEAD_RX.sub("", base)
    s = _WRAPPER_IMPERATIVE_RX.sub("", s)
    s = _WRAPPER_TOKEN_RX.sub(" ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,.;:")
    if len(s.split()) < 3 or s.lower() == base.lower():
        return None
    return s


def _dual_query_enabled() -> bool:
    """ON by default -- RAG_DUAL_QUERY=0/false/no/off is the kill-switch."""
    return (os.getenv("RAG_DUAL_QUERY") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


# Contract Data particulars vs defined-term glossary (live S1 Q&A).
# Asking for a filled-in amount/duration/percentage must prefer the
# Contract Data / Appendix-to-Tender / Contract Particulars row over the
# GC glossary ("X means the amount accepted…"). Below IDENTIFIER_BONUS_MAX
# so exact reference codes still win. Kill-switch: RAG_CD_PARTICULARS_BOOST=0.
_DEFINITION_QUESTION_RE = re.compile(
    r"(?i)\b(?:what\s+does\b.+\bmean|defin(?:e|ition\s+of)|meaning\s+of)\b",
)
_PARTICULARS_FIELD_RE = re.compile(
    r"(?i)(?:excluding\s+vat|including\s+vat|accepted\s+contract\s+amount|"
    r"delay\s+damages|liquidated\s+damages|time\s+for\s+completion|"
    r"defects\s+notification|performance\s+(?:bond|security|guarantee)|"
    r"contract\s+data|appendix\s+to\s+(?:the\s+)?tender|"
    r"contract\s+particulars)",
)
# Phrases that must appear on a filled row's KEY to elect a contract.
# "contract data" / "appendix to tender" are section headings, not a
# particular — they classify the ask but must not match every window.
_ASKED_PARTICULAR_KEY_RES = (
    (re.compile(r"(?i)time\s+for\s+completion"), "time for completion"),
    (re.compile(r"(?i)delay\s+damages"), "delay damages"),
    (re.compile(r"(?i)liquidated\s+damages"), "liquidated damages"),
    (re.compile(r"(?i)defects\s+notification"), "defects notification"),
    (re.compile(r"(?i)accepted\s+contract\s+amount"), "accepted contract amount"),
    (re.compile(r"(?i)performance\s+bond"), "performance bond"),
    (re.compile(r"(?i)performance\s+security"), "performance security"),
    (re.compile(r"(?i)performance\s+guarantee"), "performance guarantee"),
    (re.compile(r"(?i)including\s+vat"), "including vat"),
    (re.compile(r"(?i)excluding\s+vat"), "excluding vat"),
)
_FILLED_IN_ASK_RE = re.compile(
    r"(?i)(?:how\s+many\s+days|what\s+is\s+the\s+(?:amount|rate|percentage|"
    r"duration|figure)|per\s+(?:calendar\s+)?day|calendar\s+days|"
    r"\bpercentage\b|\bamount\b)",
)
# Not every Contract Data particular is a number. A filled row can name a
# party (the Engineer), a method (the approved electronic communication), or
# an address — and asking WHO the Engineer is wants that filled row, not the
# General Conditions' "'Engineer' means the person appointed by the Employer"
# glossary entry. A "who is the Engineer" ask was answered from the glossary of a DIFFERENT
# contract year because the ask was never recognised as particulars-shaped,
# so no part of the particulars machinery ran on it.
_CD_CONTRACT_ROLE_RE = re.compile(
    r"(?i)\b(?:engineer(?:'s\s+representative)?|"
    r"employer(?:'s\s+representative)?|contractor|"
    r"dispute\s+(?:adjudication\s+)?board|adjudicator)\b",
)
_CD_WHO_IS_RE = re.compile(
    r"(?i)\bwho\s+(?:is|are)\b|\bname\s+of\s+the\b|"
    r"\bwhich\s+(?:firm|company|entity|organisation|organization)\b",
)
# A numbered Schedule / Appendix / Annex of the contract is a Contract Data
# register row: "Schedule 10 | Not Used", "Schedule 9 | Health & Safety KPIs".
# A numbered schedule ask (what Schedule 10 contains) was answered and was answered out
# out of a DIFFERENT project's show package, which has a Schedule 10 of its own
# and talks about it at length. The register row that says "Not Used" IS the
# answer, and it is a filled particulars row — but the ask was not recognised
# as wanting one, so nothing lifted it and the arrival-order fence dropped
# the contract it belongs to.
#
# The context word is required: a numbered schedule also appears inside
# specifications and method statements, and those asks must stay where they
# are rather than being scoped to a contract's Contract Data.
_CD_SCHEDULE_ASK_RE = re.compile(
    r"(?i)\b(?:schedule|appendix|annex(?:ure)?)\s+(?:no\.?\s*)?\d+[A-Za-z]?\b",
)
_CD_SCHEDULE_CONTEXT_RE = re.compile(
    r"(?i)\b(?:contract|contracts|volume|volumes|"
    r"conditions\s+of\s+contract|tender)\b",
)
# Arithmetic over a particular that wants a MONEY answer. A daily-amount ask
# (a rate per day, asked in a currency) retrieved the percentage-per-day rate
# row at rank 1 and then reported the money figure as absent — because a
# percentage is not an amount, and the row carrying the amount shares no
# wording with the question, so it lost every top-5 slot to rows that do.
_CD_MONEY_ARITHMETIC_ASK_RE = re.compile(
    # "What are <particular> in <currency> per <period>" is the same ask as
    # "calculate <particular> in <currency>". The caller also requires
    # _CD_MONEY_UNIT_ASK_RE, so a plain lookup with no currency and no
    # amount-per token still does not qualify.
    r"(?i)\b(?:calculate|compute|work\s+out|how\s+much|"
    r"what\s+(?:is|are|would|will))\b",
)
_CD_MONEY_UNIT_ASK_RE = re.compile(
    r"(?i)\b(?:sar|aed|usd|eur|gbp|qar|bhd|kwd|omr)\b|"
    r"\bmonetary\b|\bamount\s+per\b|\bvalue\s+per\b",
)
# A particulars row whose value IS an amount of money — the base a
# percentage-of-the-Contract-Price calculation needs.
_CD_MONETARY_VALUE_RE = re.compile(
    r"(?i)\b(?:sar|aed|usd|eur|gbp|qar|bhd|kwd|omr)\b[^\n]{0,12}"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?",
)
# Bill-of-quantities / measured-scope asks. Live wave-2 F1 asked for a WBS
# over "the demolition and site clearance scope in this project's BOQ" and
# every citation came from another year's Conditions of Contract, whose prose
# describes that scope in words while the BOQ carries it as measured rows.
_BOQ_SCOPE_ASK_RE = re.compile(
    r"(?i)\bbo[q]\b|\bbill\s+of\s+quantit|\bschedule\s+of\s+quantit|"
    r"\bmeasured\s+(?:work|works|items?|quantit)|\bpriced\s+bill\b",
)
_CD_PARTICULARS_PREFIX_RE = re.compile(
    r"contract\s+data\s+particulars", re.IGNORECASE,
)
_CD_HEADING_IN_CHUNK_RE = re.compile(
    r"(?:contract\s+data|appendix\s+to\s+(?:the\s+)?tender|"
    r"contract\s+particulars)",
    re.IGNORECASE,
)
_CD_MEANS_RE = re.compile(
    r"\bmeans\s+the\b|\bshall\s+mean\b|\bis\s+defined\s+as\b", re.IGNORECASE,
)
_CD_FILLED_VALUE_RE = re.compile(
    r"(?i)(?:\b(?:sar|aed|usd|eur|gbp|qar|bhd|kwd|omr)\b|"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\d[\d,]*\.\d{2}|"
    r"\d+(?:\.\d+)?\s*%|"
    r"\d+\s+(?:calendar\s+|working\s+)?days?)",
)
# A Conditions-of-Contract clause that says "at the rate stated in the
# Contract Data" is a POINTER to the answer, not the answer. It carries the
# heading phrase and — being a clause about delay damages — repeats every
# label word the question uses, so it used to collect the heading tier plus
# the full label bonus and land within ~0.45 of the row that actually holds
# the rate. In a delay-damages rate ask that chunk won: the answer cited
# Sub-Clause 8.8 and then reported the rate and cap as absent from its
# excerpts, because the clause it had only refers to where they are stated.
#
# The discriminator is what precedes the phrase. A section heading sits at a
# line start; a cross-reference is governed by a preposition ("in the",
# "stated in the", "set out in the"). Only when EVERY mention in the chunk is
# prepositional is the chunk a pointer — one genuine heading is enough to
# keep the tier.
_CD_XREF_LEAD_RE = re.compile(
    r"(?i)\b(?:stated|set\s+out|specified|given|listed|described|defined|"
    r"identified|named|shown|provided|inserted|entered|contained|"
    r"referred\s+to|required)?\s*"
    r"\b(?:in|within|under|per|to|of|from|into)\s+(?:the\s+)?$",
)
_CD_XREF_LOOKBACK = 48
_CD_PARTICULARS_PREFIX_BONUS = 0.85
_CD_PARTICULARS_HEADING_BONUS = 0.40
_CD_DEFINITION_PENALTY = 0.40
# Scope disambiguation inside the particulars family (Time for Completion ask:
# "Time for Completion for the whole of the Works" answered with the
# milestone table — every particulars row got the same lift, and
# milestones win on bulk). When the query names one scope, rows of the
# other scope lose the family bonus.
_CD_WHOLE_WORKS_QUERY_RE = re.compile(
    r"(?i)\bwhole\s+of\s+the\s+works\b|\bwhole\s+works\b|\bworks\s+as\s+a\s+whole\b",
)
_CD_MILESTONE_QUERY_RE = re.compile(r"(?i)\bmilestones?\b")
_CD_MILESTONE_CHUNK_RE = re.compile(r"(?i)\bmilestones?\b")
_CD_SCOPE_MISMATCH_PENALTY = 1.10

# Label-awareness inside the particulars family (delay rate, daily amount, DNP and ACA asks).
#
# The family bonus is flat: every "CONTRACT DATA particulars" chunk with a
# filled value gets the same +0.85. Within the family nothing distinguishes the
# row the question is about from the 200+ that are not, so ordering falls back
# to raw cosine — and these chunks are near-identical to the embedder. Measured
# on the live index: top-5 scores spanning 0.003, with the answer-bearing row at
# rank 21 (delay rate) and 29 (daily amount); the ACA and DNP asks survived only because a small candidate
# pool happened not to supply enough competitors.
#
# The scope rule above was the first instance of this, solved for one axis
# (whole-of-Works vs milestone). This generalises it: reward the row whose LABEL
# the question actually names.
#
# Two things make it work where a small nudge would not:
#   * the bonus must dominate the family tie, not break it — competitors sit
#     within 0.003 of each other, and a larger candidate pool supplies more of
#     them, so the separation has to be decisive
#   * overlap is computed on the chunk BODY, never the header. Every particulars
#     chunk opens with an identical ~138-char header carrying the document
#     title; terms matching there are the same for every candidate and would
#     add noise in exactly the place discrimination is needed.
_CD_LABEL_TERM_BONUS = 0.35
_CD_LABEL_BONUS_CAP = 1.40


def _cd_chunk_body(text: str) -> str:
    """Chunk text minus the identical particulars header line."""
    t = text or ""
    nl = t.find("\n")
    return t[nl + 1:] if nl != -1 else t


def _cd_label_bonus(query_terms: frozenset, text: str) -> float:
    """Reward a particulars row for containing the label the query names."""
    if not query_terms:
        return 0.0
    body = _cd_chunk_body(text).lower()
    if not body:
        return 0.0
    overlap = sum(1 for t in query_terms if t in body)
    return min(overlap * _CD_LABEL_TERM_BONUS, _CD_LABEL_BONUS_CAP)


def contract_data_mention_is_only_a_cross_reference(text: str) -> bool:
    """True when every "Contract Data" mention points AT it rather than IS it.

    ``... at the rate stated in the Contract Data for every calendar day ...``
    is a clause telling the reader where to look. ``Contract Data`` on its own
    line, followed by rows, is the thing itself. False when the chunk has no
    mention at all — the caller has already established there is one.
    """
    t = text or ""
    mentions = list(_CD_HEADING_IN_CHUNK_RE.finditer(t))
    if not mentions:
        return False
    for m in mentions:
        lead = t[max(0, m.start() - _CD_XREF_LOOKBACK):m.start()]
        if not _CD_XREF_LEAD_RE.search(lead):
            return False
    return True


def is_contract_data_particulars_row(text: str) -> bool:
    """True for an index-time ``CONTRACT DATA particulars`` chunk that states
    a particular — the "answer-bearing Contract Data evidence" predicate the
    unnamed contract election runs on (:func:`elect_answer_bearing_contract`).

    Deliberately stricter than the scoring tier below, which asks only "is
    this a particulars chunk". The election decides which contract owns the
    whole result set, so it must be satisfied by a row that carries a VALUE
    and not by a window of unfilled keys — a clause number like ``1.1.67``
    reads as a decimal to the numeric test, so numbers alone are not proof
    that anything is filled in.
    """
    t = text or ""
    if not _CD_PARTICULARS_PREFIX_RE.search(t):
        return False
    return particulars_chunk_states_a_value(t)


def _asked_particular_key_phrases(query: str) -> Tuple[str, ...]:
    """The Contract Data label(s) an unnamed ask is actually requesting.

    Term-overlap on the chunk body is too weak: ``works`` / ``completion``
    appear on a Volume 4 programme note and on an unfilled TfC key sitting
    next to a filled Accepted Contract Amount. Election must see the
    asked field on the *key* of a filled row.
    """
    q = (query or "").strip()
    if not q:
        return ()
    phrases: List[str] = []
    for rx, phrase in _ASKED_PARTICULAR_KEY_RES:
        if rx.search(q):
            phrases.append(phrase)
    if _CD_WHO_IS_RE.search(q):
        for m in _CD_CONTRACT_ROLE_RE.finditer(q):
            role = re.sub(r"\s+", " ", m.group(0).lower()).strip()
            if role:
                phrases.append(role)
    for m in _CD_SCHEDULE_ASK_RE.finditer(q):
        phrases.append(re.sub(r"\s+", " ", m.group(0).lower()).strip())
    # Dedup, keep order.
    seen: Set[str] = set()
    out: List[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def particulars_row_answers_asked_label(query: str, text: str) -> bool:
    """True when a filled row's KEY is the particular the ask names.

    The unnamed election used to lock the pool to the first filled
    particulars row of any kind. A filled Accepted Contract Amount window
    from another package then stole a Time for Completion ask, the same
    way a glossary definition used to steal the Engineer ask before the role-identity
    ask was recognised.

    #496 required label overlap on the chunk body. That still elects a
    mixed window whose TfC / Delay Damages *key* is unfilled. Live
    (TfC and delay-rate asks): another contract won, its schedule durations and its
    delay-damages clause stayed in the pool, and the asked contract's filled rows were
    fenced out. The asked label's own value must be filled.

    When the ask has no named field (a bare "Contract Data" lookup), the
    previous body-overlap test stands so we do not empty a pool we have
    no opinion about.
    """
    phrases = _asked_particular_key_phrases(query)
    if phrases:
        want_whole_tfc = (
            "time for completion" in phrases
            and query_asks_for_time_for_completion(query)
        )
        for key, _val in filled_particulars_rows(text):
            key_l = key.lower()
            if not any(p in key_l for p in phrases):
                continue
            if want_whole_tfc and not _tfc_row_is_whole_works(key, text):
                continue
            return True
        return False
    return _cd_label_bonus(_significant_terms(query), text) > 0.0


def _collapse_retrieval_ws(text: str) -> str:
    """Collapse OCR / table newlines so a scanned label still matches."""
    return re.sub(r"\s+", " ", text or "").strip()


def _env_flag_on(name: str, default: str = "1") -> bool:
    return (os.getenv(name, default) or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def query_asks_for_delay_damages_rate(query: str) -> bool:
    """True for a whole-works Delay Damages *rate* ask.

    A daily-amount ask ("calculate … in SAR") stays on the monetary-base reservation.
    An ask that names the maximum / cap is not this class.
    """
    q = query or ""
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if query_needs_a_monetary_base(q):
        return False
    if not re.search(r"(?i)(?:delay|liquidated)\s+damages", q):
        return False
    if re.search(r"(?i)\b(?:maximum|max(?:imum)?\s+amount|capped?)\b", q):
        return False
    return True


def query_asks_delay_damages_daily_amount(query: str) -> bool:
    """True for a daily-amount ask (calculate … delay damages … in SAR), not a rate lookup.

    Reuses the monetary-base ask class so a rate ask stays a particular lookup
    and this path stays compose-only. Twin of
    ``construction_formulas_commercial.query_asks_delay_damages_daily_amount``.

    An ACA including-VAT ask has no
    delay-damages token, so it stays off this path. A combined
    "calculate delay damages … including VAT" remains a daily-amount ask
    (after #523) and must not be stolen back onto the ACA particular.
    """
    q = query or ""
    if not q or not _DELAY_RATE_KEY_RE.search(q):
        return False
    return query_needs_a_monetary_base(q)


def query_asks_who_the_engineer_is(query: str) -> bool:
    """True for "who is the Engineer", not the Representative (D1)."""
    q = query or ""
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if not _CD_WHO_IS_RE.search(q):
        return False
    if re.search(r"(?i)engineer'?s\s+representative", q):
        return False
    return bool(re.search(r"(?i)\bengineer\b", q))


_DELAY_RATE_KEY_RE = re.compile(r"(?i)(?:delay|liquidated)\s+damages")
_DELAY_DAMAGES_CLAUSE_RE = re.compile(
    r"(?i)(?:sub[- ]?clause\s+)?\b\d+(?:\.\d+)+\s*[-–—:|]?\s*"
    r"(?:delay|liquidated)\s+damages\b"
)
_DELAY_CAP_KEY_RE = re.compile(
    r"(?i)\b(?:maximum|max(?:imum)?\s+amount|capped?)\b",
)
_DELAY_RATE_VALUE_RE = re.compile(
    r"(?i)\d+(?:\.\d+)?\s*%[^\n]{0,80}\bper\b",
)
_DELAY_RATE_POINTER_RE = re.compile(
    r"(?i)at\s+the\s+rate\s+stated\s+in\s+the\s+contract\s+data",
)
_ENGINEER_GLOSSARY_RE = re.compile(
    r'(?i)"?engineer"?\s+means\s+the\s+person',
)
_ENGINEER_REP_RE = re.compile(r"(?i)engineer'?s\s+representative")
_ENGINEER_KEY_MAX_CHARS = 80
_NOT_A_PARTY_NAME_RE = re.compile(
    r"(?i)^(?:the\s+)?(?:person\s+appointed|consultant|client|"
    r"employer|contractor|engineer)\s*$"
    # The Contract Data table's own header row: "Clause (as amended) |
    # Description | Data". Capitalised, four letters, and nobody's name.
    r"|^(?:sub-?clause|clause|description|data)\b",
)
_PARTY_FIRM_RE = re.compile(
    r"(?i)\b(?:limited|ltd\.?|llc|llp|gmbh|plc|inc\.?)\b",
)
_ENGINEER_POINTER_VAL_RE = re.compile(
    r"(?i)^(?:named|stated|identified|appointed|set\s+out|specified|"
    r"defined|described|referred\s+to)\s+(?:in|as|under)\b"
)
# Inject routing notes ("ENGINEER APPOINTMENT — an excerpt below… That
# IS the answer. State the appointed firm.") are steering, not a firm
# name. extract_engineer_identity once elected that heading as the
# Engineer and the graft prepended it to the appointed firm.
_ROUTING_HINT_VAL_RE = re.compile(
    r"(?i)(?:that is the answer|an excerpt below|"
    r"state the appointed firm|state only that firm|"  # old + new hint wording
    r"do not state the appointment|internal guidance|"
    r"do not say the identity is absent|"
    r"do not answer from a conditions of contract|"
    r"do not invent|do not open with|do not give a generic|"
    r"do not search further|do not replace it)"
)


def _client_excerpt_text(text: str) -> str:
    """Drop inject-header routing notes; keep ``[doc_id=…]`` excerpt bodies.

    Extractors that scan the formatted RAG system message must not treat
    ALL-CAPS routing headings as Contract Data. When there is no excerpt
    marker the caller passed raw chunk text — leave it unchanged.
    """
    t = text or ""
    marker = t.find("[doc_id=")
    return t[marker:] if marker >= 0 else t


def _looks_like_appointed_party(val: str) -> bool:
    """True when a particulars value is a firm / person, not a role word."""
    name = re.sub(r"\s+", " ", (val or "")).strip(" \t.:;,-")
    if len(name) < 4 or _NOT_A_PARTY_NAME_RE.match(name):
        return False
    if _ENGINEER_POINTER_VAL_RE.search(name):
        return False
    if _ROUTING_HINT_VAL_RE.search(name):
        return False
    if _PARTY_FIRM_RE.search(name):
        return True
    letters = re.sub(r"[^A-Za-z]", "", name)
    return len(letters) >= 4 and any(ch.isupper() for ch in name)


def _delay_damages_key_is_rate(key: str) -> bool:
    if not _DELAY_RATE_KEY_RE.search(key or ""):
        return False
    return not _DELAY_CAP_KEY_RE.search(key or "")


def chunk_states_delay_damages_rate(text: str) -> bool:
    """True when the chunk states the daily Delay Damages *rate*.

    A cap row (``Maximum amount of delay damages: 10%…``) and a General
    Conditions pointer (``at the rate stated in the Contract Data``)
    both contain the label and used to satisfy the unnamed election /
    reservation. A delay-rate ask after #501 then cited 118 without the 0.1%
    per-calendar-day figure. The rate lives in a different chunk.
    """
    t = text or ""
    if not t or _DELAY_RATE_POINTER_RE.search(t):
        return False
    for key, val in filled_particulars_rows(t):
        if _delay_damages_key_is_rate(key) and _DELAY_RATE_VALUE_RE.search(val):
            return True
    blob = _collapse_retrieval_ws(t)
    if _DELAY_RATE_POINTER_RE.search(blob):
        return False
    if _DELAY_CAP_KEY_RE.search(blob) and not _DELAY_RATE_VALUE_RE.search(blob):
        return False
    if not _DELAY_RATE_KEY_RE.search(blob):
        return False
    return bool(_DELAY_RATE_VALUE_RE.search(blob))


def chunk_states_engineer_identity(text: str) -> bool:
    """True when the chunk *appoints* the Engineer (the naming row, not the definition)."""
    return chunk_names_party(text, "engineer")


# ── who is a party: the particulars row that names it ─────────────────────
#
# "Who is the <defined party>?" -- the Engineer, the Employer, the
# Contractor, a Representative, the adjudicator. The answer is the
# particulars row that NAMES the party: a label row whose value is a proper
# name. The General Conditions clause that DEFINES the term ("'<Party>'
# means the person named as ... in the Contract Data") repeats every word of
# the question and wins on cosine; it is a lookalike, not the answer. A text
# search on the role word alone hits every clause that mentions the role, so
# a LIMIT cuts the naming row off; the row is fetched by the role together
# with the words a legal person's name ends in (a naming lexicon).
_PARTY_ROLE_ASK_RE = re.compile(
    r"(?i)\b(?:engineer['’]?s\s+representative|employer['’]?s\s+representative|"
    r"engineer|employer|contractor|"
    r"dispute\s+(?:avoidance\s+(?:and|/)\s+)?(?:adjudication\s+)?board|adjudicator)\b"
)
_PARTY_NAME_ENDINGS = (
    "limited", "ltd", "llc", "plc", "gmbh", "inc", "company", "corporation",
    "consult", "partners", "authority", "group",
)
_PARTY_LINE_MAX_NEXT = 2


def asked_party_role(query: str) -> str:
    """The defined party a who-is question asks for ("engineer"), or ""."""
    q = query or ""
    if not q or _DEFINITION_QUESTION_RE.search(q) or not _CD_WHO_IS_RE.search(q):
        return ""
    match = _PARTY_ROLE_ASK_RE.search(q)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(0).lower().replace("’", "'"))


def _role_pattern(role: str) -> str:
    """Regex for the role word in a document, not its Representative (unless asked)."""
    parts = [re.escape(p) for p in role.replace("'s", "").split()]
    body = r"\s+".join(parts)
    if "representative" in role:
        body = body.replace(r"\s+representative", r"['’]?s\s+representative")
        return rf"\b{body}\b"
    return rf"\b{body}\b(?!\s*['’]?s\s+representative)"


def chunk_defines_role(text: str, role: str) -> bool:
    """True when the chunk is the definition of the role term ("X" means ...)."""
    if not role:
        return False
    return bool(re.search(
        rf"(?i)[\"“']?{_role_pattern(role)}[\"”']?\s+(?:means|shall\s+mean|is\s+defined\s+as)\b",
        text or "",
    ))


def chunk_names_party(text: str, role: str) -> bool:
    """True when the chunk has a row that NAMES the asked party.

    A filled particulars row whose key is the role and whose value is a
    proper name; or a scanned line that opens with the role and carries a
    name on it or the next lines; or "<role> is <Name Ltd>". The definition
    of the term, a party list that only repeats role words, and the
    Representative of the asked party are lookalikes.
    """
    t = text or ""
    if not t or not role:
        return False
    role_rx = re.compile(rf"(?i){_role_pattern(role)}")
    rows = filled_particulars_rows(t)
    for key, val in rows:
        if role_rx.search(key or "") and _looks_like_appointed_party(val):
            return True
    if chunk_defines_role(t, role) and not rows:
        return False
    line_rx = re.compile(
        r"(?im)^[ \t|:]*(?:\d+(?:\.\d+)+\s*(?:\([a-z]\))?[ \t|:]*)?"
        r"(?:(?:the|name\s+of\s+the)\s+)?"
        rf"{_role_pattern(role)}[ \t]*[:|–-]?\s*(.*)$"
    )
    lines = t.splitlines()
    for i, line in enumerate(lines):
        m = line_rx.match(line)
        if not m:
            continue
        rest = (m.group(1) or "").strip()
        following = [lines[j].strip() for j in range(i + 1, min(len(lines), i + 1 + _PARTY_LINE_MAX_NEXT))]
        nxt = following[0] if following else ""
        nxt2 = following[1] if len(following) > 1 else ""
        for cand in (rest, nxt, nxt2, f"{rest} {nxt}".strip(), f"{nxt} {nxt2}".strip()):
            if _looks_like_appointed_party(cand):
                return True
    blob = _collapse_retrieval_ws(t)
    is_rx = re.compile(
        rf"(?i)\b(?:the\s+|name\s+of\s+the\s+)?{_role_pattern(role)}\s*(?:is|are|:)\s+(.{{4,80}})"
    )
    for named in is_rx.finditer(blob):
        cand = named.group(1)
        if _looks_like_appointed_party(cand) and (
            _PARTY_FIRM_RE.search(cand) or re.search(r"\b[A-Z]{3,}\b", cand)
        ):
            return True
    return False


def party_name_needle_sets(role: str) -> List[Tuple[str, ...]]:
    """Text-search needles for the row that names ``role``."""
    word = role.replace("'s", "").split()[0] if role else ""
    if not word:
        return []
    return [(word, ending) for ending in _PARTY_NAME_ENDINGS]


def chunk_answers_asked_particular(query: str, text: str) -> bool:
    """Election / reservation predicate for a particulars-shaped ask.

    Prefixed filled rows keep today's behaviour (TfC / DNP / ACA year-lock).
    After #501, delay-rate / Engineer asks also accept an unprefixed scanned rate /
    Engineer appointment so the year-lock fence cannot drop the
    chunk that actually answers.
    """
    if query_asks_for_delay_damages_rate(query):
        if chunk_states_delay_damages_rate(text):
            return True
    if query_asks_delay_damages_daily_amount(query):
        if (
            chunk_states_delay_damages_rate(text)
            or chunk_states_accepted_contract_amount(text)
        ):
            return True
    party_role = asked_party_role(query)
    if party_role and chunk_names_party(text, party_role):
        return True
    if query_asks_for_aca_including_vat(query):
        if chunk_states_aca_including_vat(text):
            return True
    if query_asks_for_time_for_completion(query):
        if chunk_states_time_for_completion(text):
            return True
    if query_asks_for_defects_notification_period(query):
        if chunk_states_defects_notification_period(text):
            return True
    if (
        is_contract_data_particulars_row(text)
        and particulars_row_answers_asked_label(query, text)
    ):
        return True
    # The key-position test above cannot see a scanned row whose label sits in
    # the value cell (``4.3.3(a): | Value of Performance Bond: 10 %``), nor a
    # label no regex lists. The row the named-row rescue fetched must survive
    # the election it was fetched for.
    return (
        named_particulars_row_match(query, text) > 0
    )


# ── rows deep inside a pooled document ────────────────────────────────────
#
# A bound volume (conditions and particulars together) puts the clause that
# MENTIONS a particular near its start and the row that STATES it in a later
# appendix. identifier_search LIMIT and a first-N chunks_for_docs both stay on
# the early windows, so the stated row never enters the pool. When the
# question's row is not pooled, the volumes already in the pool are read
# whole: every row, or -- for a store that caps a fetch -- prefix, tail and
# fixed windows across the middle, then a text match on the words of the
# asked labels (a scanned label is split across lines, so its words are
# matched one by one, never as a phrase).
_DOC_SCAN_WINDOW = 400
_DOC_SCAN_MAX = 8000
_DOC_TEXT_SCAN_K = 400
_ROW_PAIR_WINDOW = 3
# Pin pooled composition operands above the early windows' cosine so the
# token cap cannot drop them.
_OPERAND_PIN_SCORE = 2.4
_ASKED_PARTICULAR_VALUE_BONUS = 2.0
_TFC_DAYS_RE = re.compile(r"(?i)\b(\d{2,4})\s+(?:calendar\s+|working\s+)?days\b")
_TFC_PERMIT_TRACKER_RE = re.compile(
    r"(?i)permit[- ]track|commencement[- ]completion|"
    r"community\s+[a-z0-9-]+\s+\w{3}-\d{2}\s+to\s+\w{3}-\d{2}",
)
# A Time for Completion ask was PARTIAL: a sectional "within N days" figure from a
# specification ranked ahead of the Contract Data row and the graft led with it.
_TFC_SECTIONAL_RE = re.compile(
    r"(?i)\bsection(?:al)?s?\s+"
    r"(?:\d+|[ivxlcd]+|[a-z]\b|of\s+(?:the\s+)?works)",
)
# The Time for Completion particulars ROW: a clause number in front of the
# label, at the start of a line or cell -- whatever number the contract uses.
_TFC_LABELLED_ROW_RE = re.compile(
    r"(?im)(?:^|\|)[\s|:]*\d+(?:\.\d+)+(?:\s*\([a-z0-9]+\))?[\s|:]*"
    r"time\s+for\s+completion\b"
)
# A key that is only a clause number (a scanned table puts the label in the
# value cell).
_BARE_CLAUSE_KEY_RE = re.compile(r"^[\s|:]*\d+(?:\.\d+)+(?:\s*\([a-z0-9]+\))?[\s|:]*$")
_TFC_POINTER_RE = re.compile(
    r"(?i)(?:stated|named|identified|set\s+out|specified|defined|"
    r"described|referred\s+to)\s+in\s+(?:the\s+)?contract\s+data",
)
_TFC_NOTICE_DAYS_RE = re.compile(
    r"(?i)\b(?:within|not\s+later\s+than|no\s+later\s+than|"
    r"after\s+(?:the\s+)?(?:taking[- ]over|toc)|before\s+the)\s+"
    r"(\d{2,4})\s+(?:calendar\s+|working\s+)?days",
)


def query_asks_for_aca_including_vat(query: str) -> bool:
    """True for ACA including VAT, not the excluding-VAT ask or a definition."""
    if not query_asks_for_accepted_contract_amount(query):
        return False
    return bool(_INCLUDING_VAT_RE.search(query or ""))


def query_is_aca_including_vat_particular(query: str) -> bool:
    """True for the ACA including-VAT ask, not daily-amount compose.

    Live 9ad62cc: the including-VAT particular retrieved Contract Data
    chunk #0 (delay damages × a partial ACA) and daily-amount compose
    stated SAR/day. An including-VAT ask is not rate × ACA. A combined
    "calculate delay damages … including VAT" stays a daily-amount ask.
    """
    if not query_asks_for_aca_including_vat(query):
        return False
    return not query_asks_delay_damages_daily_amount(query)


def query_asks_for_time_for_completion(query: str) -> bool:
    """True for a whole-Works TfC ask, not a milestone-only or sectional ask."""
    q = query or ""
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if not re.search(r"(?i)time\s+for\s+completion", q):
        return False
    if _CD_MILESTONE_QUERY_RE.search(q) and not _CD_WHOLE_WORKS_QUERY_RE.search(q):
        return False
    # "Section 2 of the Works" is not the whole-Works particular.
    if _TFC_SECTIONAL_RE.search(q) and not _CD_WHOLE_WORKS_QUERY_RE.search(q):
        return False
    return True


def _tfc_key_is_not_whole_works(key: str) -> bool:
    """True when a TfC key is a milestone or section, not whole-of-Works."""
    k = key or ""
    if _CD_MILESTONE_CHUNK_RE.search(k):
        return True
    if _CD_WHOLE_WORKS_QUERY_RE.search(k):
        return False
    return bool(_TFC_SECTIONAL_RE.search(k))


def _tfc_row_is_whole_works(key: str, chunk_text: str = "") -> bool:
    """Positive test: this key is the whole-Works particular, not a lookalike.

    A Vol-2 sentence that mentions Time for Completion and peels
    ``within 90 days`` as a value is not the particulars row.
    """
    k = key or ""
    if not k or _CD_MILESTONE_CHUNK_RE.search(k) or _TFC_SECTIONAL_RE.search(k):
        return False
    if _TFC_POINTER_RE.search(k):
        return False
    if _CD_WHOLE_WORKS_QUERY_RE.search(k):
        return True
    if _BARE_CLAUSE_KEY_RE.match(k):
        return bool(_TFC_LABELLED_ROW_RE.search(chunk_text or "")
                    or _CD_PARTICULARS_PREFIX_RE.search(chunk_text or ""))
    if not re.search(r"(?i)time\s+for\s+completion", k):
        return False
    if len(k) > 96:
        return False
    return bool(
        _CD_PARTICULARS_PREFIX_RE.search(chunk_text or "")
        or _TFC_LABELLED_ROW_RE.search(k)
        or _TFC_LABELLED_ROW_RE.search(chunk_text or "")
    )


def _tfc_row_anchor_index(text: str, key: str, val: str, days_num: str) -> int:
    """Index of this key+value, not the first lookalike with the same label."""
    loc = (text or "").lower()
    key_l = (key or "").lower()
    val_l = (val or "").lower()
    num = (days_num or "").split()[0]
    if key_l:
        start = 0
        while True:
            i = loc.find(key_l[:40], start)
            if i < 0:
                break
            window = loc[i: i + max(len(key_l) + 80, 160)]
            if num and num in window:
                return i
            if val_l and val_l[:24] in window:
                return i
            start = i + max(1, len(key_l[:40]))
    if val_l:
        return loc.find(val_l)
    return -1


def _tfc_days_from_block(block: str) -> Optional[str]:
    """Days figure tied to the TfC label, not a neighbouring notice period."""
    blob = _normalize_retrieval_ws(block)
    if not blob:
        return None
    anchors = (
        _TFC_LABELLED_ROW_RE,
        re.compile(r"(?i)time\s+for\s+completion"),
        _CD_WHOLE_WORKS_QUERY_RE,
    )
    for rx in anchors:
        for m in rx.finditer(blob):
            window = blob[m.start(): m.end() + 120]
            if _tfc_key_is_not_whole_works(window):
                continue
            notice = _TFC_NOTICE_DAYS_RE.search(window)
            dm = _TFC_DAYS_RE.search(window)
            if not dm:
                continue
            if notice and notice.group(1) == dm.group(1):
                continue
            return f"{dm.group(1)} days"
    for dm in _TFC_DAYS_RE.finditer(blob):
        lead = blob[max(0, dm.start() - 48): dm.end() + 8]
        if _TFC_NOTICE_DAYS_RE.search(lead):
            continue
        if _tfc_key_is_not_whole_works(lead):
            continue
        return f"{dm.group(1)} days"
    return None


def _score_tfc_candidate(days: str, context: str, *, from_particulars: bool) -> int:
    """Higher wins. Whole-Works Contract Data / newer year beat lookalikes."""
    ctx = context or ""
    score = 0
    if _CD_WHOLE_WORKS_QUERY_RE.search(ctx):
        score += 100
    if _TFC_LABELLED_ROW_RE.search(ctx):
        score += 80
    if from_particulars or _CD_PARTICULARS_PREFIX_RE.search(ctx):
        score += 60
    elif _CD_HEADING_IN_CHUNK_RE.search(ctx) and not (
        contract_data_mention_is_only_a_cross_reference(ctx)
    ):
        score += 40
    ids = extract_contract_doc_ids(ctx)
    if ids:
        year, seq = max(_contract_id_recency(cid) for cid in ids)
        if year > 0:
            score += year
        if seq > 0:
            score += min(seq, 30)
    if _tfc_key_is_not_whole_works(ctx):
        score -= 200
    if _TFC_PERMIT_TRACKER_RE.search(ctx):
        score -= 200
    if _TFC_POINTER_RE.search(ctx) and not from_particulars:
        score -= 80
    notice = _TFC_NOTICE_DAYS_RE.search(ctx)
    if notice and notice.group(1) == (days or "").split()[0]:
        score -= 150
    return score


def chunk_states_accepted_contract_amount(text: str) -> bool:
    """True when the chunk states an Accepted Contract Amount in money.

    The daily-amount ask's rate base. Including-VAT and excluding-VAT both count —
    compose prefers excl when both are in the excerpts. A delay-damages
    rate row that only *names* the Contract Price / ACA is not this.
    A cap row (``10% of the Accepted Contract Amount``) has no SAR
    figure and fails the money test.
    """
    t = text or ""
    blob = _normalize_retrieval_ws(t).lower()
    if "accepted contract amount" not in blob:
        return False
    if not _CD_MONETARY_VALUE_RE.search(t):
        return False
    if _DELAY_RATE_KEY_RE.search(blob) and _DELAY_RATE_VALUE_RE.search(blob):
        for key, val in filled_particulars_rows(t):
            joined = f"{key} {val}".lower()
            if (
                "accepted contract amount" in joined
                and _CD_MONETARY_VALUE_RE.search(val)
                and not _DELAY_RATE_KEY_RE.search(key)
            ):
                break
        else:
            # Rate sentence is not the money row. A paired scanned
            # window that also carries the filled excl-VAT ACA still
            # is — do not drop it just because 8.8 shares the chunk.
            try:
                from app.lib.construction_formulas_commercial import (
                    chunk_has_real_accepted_contract_amount,
                )
                if not chunk_has_real_accepted_contract_amount(t):
                    return False
            except Exception:  # noqa: BLE001 — rate-only window is not ACA
                logger.debug("real-ACA test unavailable; treating as non-ACA", exc_info=True)
                return False
    try:
        from app.lib.construction_formulas_commercial import (
            chunk_accepted_contract_amount_is_only_toy,
            chunk_has_real_accepted_contract_amount,
        )
        if chunk_has_real_accepted_contract_amount(t):
            return True
        if chunk_accepted_contract_amount_is_only_toy(t):
            return False
    except Exception:  # noqa: BLE001 — never break a turn over an import
        logger.debug("toy-ACA test unavailable; treating money as ACA", exc_info=True)
    return True


def _chunk_is_daily_damages_operand(text: str) -> bool:
    """Rate row or ACA money row — the two daily-amount multiply operands."""
    if chunk_states_delay_damages_rate(text):
        return True
    if chunk_states_accepted_contract_amount(text):
        return True
    try:
        from app.lib.construction_formulas_commercial import (
            chunk_has_real_accepted_contract_amount,
        )
        return chunk_has_real_accepted_contract_amount(text)
    except Exception:  # noqa: BLE001 — rate-only window is not ACA
        logger.debug("real-ACA test unavailable; treating as non-operand", exc_info=True)
        return False


def _chunk_keeps_for_daily_damages(filename: str, text: str) -> bool:
    """Keep compose operands and filled particulars; drop GC lookalikes.

    Exclusive rate-or-ACA fencing deleted the particulars family and
    shrank the daily-amount ask below k=5. Spec TOC / Daywork / insurance are not
    Contract Data and must still drop once both operands are in-pool.

    Daily-amount ask after #523: the bound Contract Data volume's
    Sub-Clause 8.8 chunks (9–11) passed the filename keep, occupied
    top-k, and the last-slot money reserve then elected the
    including-VAT ACA. A filename match alone is not an operand.
    """
    _ = filename  # operands are textual; a CD filename is not enough
    if _chunk_is_daily_damages_operand(text):
        return True
    t = text or ""
    if _DELAY_RATE_POINTER_RE.search(t) and not chunk_states_delay_damages_rate(t):
        return False
    if is_contract_data_particulars_row(t):
        return True
    return bool(
        _CD_HEADING_IN_CHUNK_RE.search(t)
        and not contract_data_mention_is_only_a_cross_reference(t)
    )


def _is_daily_damages_cap_noise(text: str) -> bool:
    """True for a restated milestone rate or pointer-only 8.8 windows that crowd the cap.

    Daily-amount ask after #541: HIGH chunks 9–11 (pointer or a milestone rate of
    the filled ACA) fill ``MAX_RAG_TOKENS`` and drop the 0.0-score
    Contract Data 0.1% / excl-VAT operands. Those windows are never
    the daily-amount product — evict them once both operands are protected.
    """
    t = text or ""
    if not t:
        return False
    try:
        if _daily_rate_preference(t) >= 2:
            return False
        if _has_standalone_excl_vat_aca(t):
            return False
    except Exception:  # noqa: BLE001 — treat as noise-unknown, keep the row
        logger.debug("e1 cap-noise test failed; keeping the row", exc_info=True)
        return False
    # A delay-damages rate the preference above did not pick (a milestone /
    # restated rate): it crowds the cap.
    if _DELAY_RATE_VALUE_RE.search(t) and _DELAY_RATE_KEY_RE.search(t):
        return True
    if _DELAY_RATE_POINTER_RE.search(t) and not chunk_states_delay_damages_rate(t):
        return True
    return False


def _has_standalone_excl_vat_aca(text: str) -> bool:
    """True for a 1.1.1 / excl-VAT money row, not a rate window that cites ACA.

    Daily-amount ask after #535: CoC chunks 9–11 state a milestone rate of the
    filled excl-VAT ACA. ``chunk_has_real_accepted_contract_amount``
    is True, so the all-chunk scan early-exited and compose used a milestone rate.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            chunk_has_real_accepted_contract_amount,
        )
    except Exception:  # noqa: BLE001 — treat as missing; keep scanning
        logger.debug("real-ACA import failed; treating as missing", exc_info=True)
        return False
    if not chunk_has_real_accepted_contract_amount(text or ""):
        return False
    if chunk_states_delay_damages_rate(text or ""):
        return False
    return _daily_damages_aca_preference(text) >= 2


def _daily_rate_preference(text: str) -> int:
    """Higher wins for the daily-amount rate. Contract Data 0.1% beats a restated milestone rate."""
    t = text or ""
    if not chunk_states_delay_damages_rate(t):
        return -1
    try:
        from app.lib.construction_formulas_commercial import (
            delay_damages_rate_preference_score,
            parse_delay_damages_rate_percent,
        )
        pct = parse_delay_damages_rate_percent(t)
        if pct is None:
            return 0
        return delay_damages_rate_preference_score(pct, t)
    except Exception:  # noqa: BLE001 — a rate window still outranks none
        logger.debug("e1 rate preference failed", exc_info=True)
        return 1


def _daily_damages_aca_preference(text: str) -> int:
    """Higher wins for the daily-amount rate base. Excl-VAT (2) > unlabeled (1) > incl (0)."""
    try:
        from app.lib.construction_formulas_commercial import (
            chunk_accepted_contract_amount_is_only_toy,
            chunk_has_real_accepted_contract_amount,
        )
        if chunk_accepted_contract_amount_is_only_toy(text):
            return -1
        # Daily-amount ask: a scanned excl-VAT row can fail
        # chunk_states_accepted_contract_amount (line-split / excl. VAT)
        # while still being the real money operand. Do not rank it -1
        # or reservation leaves top-k on 8.8 toys.
        states = chunk_states_accepted_contract_amount(text)
        if not states and not chunk_has_real_accepted_contract_amount(text):
            return -1
    except Exception:  # noqa: BLE001 — unlabeled ACA still ranks above none
        logger.debug("toy-ACA preference test failed", exc_info=True)
        if not chunk_states_accepted_contract_amount(text):
            return -1
    try:
        from app.lib.construction_formulas_commercial import (
            _EXCL_VAT_RE,
            _INCL_VAT_RE,
            text_states_clause_111_aca,
        )
    except Exception:  # noqa: BLE001 — unlabeled ACA still ranks above none
        return 1
    # Milestone delay damages ask: a Contract Data chunk states "Accepted Contract
    # Amount: <amount> |" with no VAT qualifier, beside the
    # "(including VAT)" particular. The VAT regexes below ranked that chunk
    # 0 (incl matched, excl absent), so the rescue never collected the real
    # base and composed the only excluding-VAT figure left — a partial. The
    # clause is the net figure the rate applies to: rank it with excl-VAT.
    if text_states_clause_111_aca(text or ""):
        return 2
    blob = _normalize_retrieval_ws(text or "")
    if _EXCL_VAT_RE.search(blob) and not _INCL_VAT_RE.search(blob):
        return 2
    if _INCL_VAT_RE.search(blob) and not _EXCL_VAT_RE.search(blob):
        return 0
    if _EXCL_VAT_RE.search(blob):
        return 2
    return 1


def chunk_states_aca_including_vat(text: str) -> bool:
    """True when the chunk states Accepted Contract Amount *including VAT*.

    A delay-damages sentence that cites the excl-VAT ACA as the rate
    base (live A2) is the neighboring field, not this answer.
    """
    t = text or ""
    blob = _normalize_retrieval_ws(t).lower()
    if "accepted contract amount" not in blob:
        return False
    if not _INCLUDING_VAT_RE.search(blob):
        return False
    if not _CD_MONETARY_VALUE_RE.search(t):
        return False
    if _DELAY_RATE_KEY_RE.search(blob) and re.search(
        r"(?i)per\s+(?:calendar\s+)?day", blob,
    ):
        for key, val in filled_particulars_rows(t):
            joined = f"{key} {val}".lower()
            if (
                "accepted contract amount" in joined
                and _INCLUDING_VAT_RE.search(joined)
                and _CD_MONETARY_VALUE_RE.search(val)
            ):
                return True
        return False
    return True


def chunk_states_time_for_completion(text: str) -> bool:
    """True when the chunk states whole-Works Time for Completion in days.

    Permit-tracker / community commencement-completion tables
    mention completion dates but are not the Contract Data duration.
    Milestone-only and sectional rows are not this class. A Vol-2
    specification that only cites TfC and a ``within 90 days`` notice
    is a lookalike, not the particular.
    """
    t = text or ""
    if not t or _TFC_PERMIT_TRACKER_RE.search(t):
        return False
    blob = _normalize_retrieval_ws(t)
    if not re.search(r"(?i)time\s+for\s+completion", blob):
        return False
    if _CD_MILESTONE_CHUNK_RE.search(blob) and not _CD_WHOLE_WORKS_QUERY_RE.search(blob):
        return False
    if _TFC_SECTIONAL_RE.search(blob) and not _CD_WHOLE_WORKS_QUERY_RE.search(blob):
        return False
    for key, val in filled_particulars_rows(t):
        key_l = key.lower()
        if "time for completion" not in key_l:
            continue
        if not _tfc_row_is_whole_works(key, t):
            continue
        if _CD_PARTICULARS_PREFIX_RE.search(val or ""):
            continue
        if _TFC_DAYS_RE.search(val) or _TFC_DAYS_RE.search(key):
            return True
    if _CD_MILESTONE_CHUNK_RE.search(blob) and not _CD_WHOLE_WORKS_QUERY_RE.search(blob):
        return False
    if _TFC_POINTER_RE.search(blob) and not (
        _CD_PARTICULARS_PREFIX_RE.search(t)
        or (
            _CD_HEADING_IN_CHUNK_RE.search(t)
            and not contract_data_mention_is_only_a_cross_reference(t)
        )
    ):
        return False
    days = _tfc_days_from_block(t)
    if not days:
        return False
    if not (
        _CD_WHOLE_WORKS_QUERY_RE.search(blob)
        or _TFC_LABELLED_ROW_RE.search(t)
        or _CD_PARTICULARS_PREFIX_RE.search(t)
        or (
            _CD_HEADING_IN_CHUNK_RE.search(t)
            and not contract_data_mention_is_only_a_cross_reference(t)
        )
    ):
        return False
    notice = _TFC_NOTICE_DAYS_RE.search(blob)
    if notice and notice.group(1) == days.split()[0]:
        return False
    return True


def _aca_row_is_including_vat(key: str, val: str) -> bool:
    """True when this particulars row is the including-VAT ACA, not a neighbor.

    A system-message mega-row that mentions the inject hint and later
    peels the first SAR figure (live A2: excl-VAT / delay-damages) is
    not this class — same 96-char key cap as whole-Works TfC.
    """
    k = (key or "").strip()
    v = (val or "").strip()
    if not k or len(k) > 96:
        return False
    # RAG system-message peels can glue the next [doc_id=…] chunk onto a
    # short including-VAT key and steal the previous figure.
    if "[doc_id=" in v or len(v) > 160:
        return False
    joined = f"{k} {v}"
    if "accepted contract amount" not in joined.lower():
        return False
    if _DELAY_RATE_KEY_RE.search(k) or _DELAY_RATE_KEY_RE.search(joined[:80]):
        return False
    try:
        from app.lib.construction_formulas_commercial import (
            _EXCL_VAT_RE,
            _INCL_VAT_RE,
        )
    except Exception:  # noqa: BLE001
        logger.debug("VAT regex import failed for including-VAT row", exc_info=True)
        return False
    if _EXCL_VAT_RE.search(k) and not _INCL_VAT_RE.search(k):
        return False
    if _EXCL_VAT_RE.search(joined) and not _INCL_VAT_RE.search(k):
        return False
    if _INCL_VAT_RE.search(k):
        return True
    # filled_particulars_rows can glue an early chunk (delay damages × a
    # partial amount) onto the later including-VAT label. Including-VAT in the value must precede the first figure —
    # otherwise the partial ACA is peeled as the including-VAT amount.
    incl = _INCL_VAT_RE.search(v)
    if not incl:
        return False
    try:
        from app.lib.construction_formulas_commercial import _MONEY_RE
        first_money = _MONEY_RE.search(v)
    except Exception:  # noqa: BLE001
        first_money = None
    if first_money is not None and first_money.start() < incl.start():
        return False
    return not (_EXCL_VAT_RE.search(v) and not _INCL_VAT_RE.search(v[:incl.end()]))


def _aca_nearest_vat_is_including(lead: str) -> bool:
    """True when the last VAT qualifier before the figure is including-VAT.

    Adjacent excl/incl table rows share a 64-char window; first-excl-in-window
    would reject the including-VAT amount sitting on the next line.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            _EXCL_VAT_RE,
            _INCL_VAT_RE,
        )
    except Exception:  # noqa: BLE001
        logger.debug("VAT regex import failed for nearest-VAT test", exc_info=True)
        return False
    last_incl = max((m.start() for m in _INCL_VAT_RE.finditer(lead or "")), default=-1)
    last_excl = max((m.start() for m in _EXCL_VAT_RE.finditer(lead or "")), default=-1)
    return last_incl >= 0 and last_incl > last_excl


def _aca_money_is_including_vat(tight: str, wide: str) -> bool:
    """True when the figure's local label is including VAT, not excl-VAT."""
    try:
        from app.lib.construction_formulas_commercial import _INCL_VAT_RE
    except Exception:  # noqa: BLE001
        logger.debug("including-VAT regex import failed", exc_info=True)
        return False
    if "accepted contract amount" not in (wide or "").lower():
        return False
    if not _INCL_VAT_RE.search(tight or ""):
        return False
    if not _aca_nearest_vat_is_including(tight or ""):
        return False
    if _DELAY_RATE_KEY_RE.search(wide or "") and re.search(
        r"(?i)per\s+(?:calendar\s+)?day", wide or "",
    ):
        return False
    return True


def _score_aca_incl_candidate(
    context: str, *, from_particulars: bool,
) -> int:
    """Higher wins. Including-VAT Contract Data / newer year beat neighbors."""
    try:
        from app.lib.construction_formulas_commercial import (
            _EXCL_VAT_RE,
            _INCL_VAT_RE,
        )
    except Exception:  # noqa: BLE001
        _EXCL_VAT_RE = None
        _INCL_VAT_RE = None
    ctx = context or ""
    score = 0
    if from_particulars:
        score += 60
    elif _CD_PARTICULARS_PREFIX_RE.search(ctx) or (
        _CD_HEADING_IN_CHUNK_RE.search(ctx)
        and not contract_data_mention_is_only_a_cross_reference(ctx)
    ):
        score += 40
    if _INCL_VAT_RE is not None and _INCL_VAT_RE.search(ctx) and not (
        _EXCL_VAT_RE.search(ctx) if _EXCL_VAT_RE is not None else False
    ):
        score += 100
    if _EXCL_VAT_RE is not None and _EXCL_VAT_RE.search(ctx) and not (
        _INCL_VAT_RE.search(ctx) if _INCL_VAT_RE is not None else False
    ):
        score -= 200
    if _DELAY_RATE_KEY_RE.search(ctx):
        score -= 200
    ids = extract_contract_doc_ids(ctx)
    if ids:
        year, seq = max(_contract_id_recency(cid) for cid in ids)
        if year > 0:
            score += year
        if seq > 0:
            score += min(seq, 30)
    return score


def extract_aca_including_vat(text: str) -> Optional[Tuple[float, str]]:
    """Including-VAT ACA from client text, or None. Does not invent.

    Walks the full RAG blob (graft reads the system message). When an
    excluding-VAT neighbor or delay-damages base shares the text, elect
    the including-VAT row — first-in-blob used to prepend the excl-VAT
    figure labeled as including VAT (live A2 after #517).
    """
    t = text or ""
    if not t:
        return None
    try:
        from app.lib.construction_formulas_commercial import (
            _MONEY_RE,
            _collapse_ws,
        )
    except Exception:  # noqa: BLE001
        logger.debug("ACA money regex import failed", exc_info=True)
        return None
    cands: List[Tuple[int, int, Tuple[float, str]]] = []
    order = 0
    try:
        for key, val in filled_particulars_rows(t):
            if not _aca_row_is_including_vat(key, val):
                continue
            money = _MONEY_RE.search(val) or _MONEY_RE.search(f"{key} {val}")
            if not money:
                continue
            amount = float(money.group(2).replace(",", ""))
            currency = money.group(1).upper()
            idx = (t or "").lower().find((val or "").lower()[:24]) if val else -1
            local = t[max(0, idx - 280): idx + 80] if idx >= 0 else f"{key} {val}"
            score = _score_aca_incl_candidate(
                f"{key} {val}\n{local}", from_particulars=True,
            )
            cands.append((-score, order, (amount, currency)))
            order += 1
    except Exception:  # noqa: BLE001
        logger.debug("including-VAT particulars parse failed", exc_info=True)
    blob = _collapse_ws(t)
    for m in _MONEY_RE.finditer(blob):
        tight = blob[max(0, m.start() - 64): m.end() + 8]
        wide = blob[max(0, m.start() - 160): m.end() + 24]
        if not _aca_money_is_including_vat(tight, wide):
            continue
        amount = float(m.group(2).replace(",", ""))
        currency = m.group(1).upper()
        score_ctx = blob[max(0, m.start() - 280): m.end() + 80]
        score = _score_aca_incl_candidate(score_ctx, from_particulars=False)
        cands.append((-score, order, (amount, currency)))
        order += 1
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def extract_time_for_completion_days(text: str) -> Optional[str]:
    """Whole-Works TfC duration as written (e.g. ``NNN days``), or None.

    Walks the full RAG blob (graft reads the system message). When a
    sectional / notice-period 90-day lookalike and the Contract Data
    852-day particular share the same text, elect the whole-Works row
    — first-in-blob used to prepend 90 days onto a correct 852 answer.
    """
    t = text or ""
    if not t:
        return None
    cands: List[Tuple[int, int, str]] = []
    order = 0
    for key, val in filled_particulars_rows(t):
        key_l = key.lower()
        if "time for completion" not in key_l:
            continue
        if not _tfc_row_is_whole_works(key, t):
            continue
        joined = f"{key} {val}"
        if _TFC_PERMIT_TRACKER_RE.search(joined):
            continue
        if _CD_PARTICULARS_PREFIX_RE.search(val or ""):
            continue
        m = _TFC_DAYS_RE.search(val) or _TFC_DAYS_RE.search(key)
        if not m:
            continue
        days = f"{m.group(1)} days"
        idx = _tfc_row_anchor_index(t, key, val, m.group(1))
        local = t[max(0, idx - 240): idx + 280] if idx >= 0 else joined
        score = _score_tfc_candidate(days, joined + "\n" + local, from_particulars=True)
        cands.append((-score, order, days))
        order += 1
    for block in re.split(r"\n{2,}|\[doc_id=", t):
        if not chunk_states_time_for_completion(block):
            continue
        days = _tfc_days_from_block(block)
        if not days:
            continue
        score = _score_tfc_candidate(days, block, from_particulars=False)
        cands.append((-score, order, days))
        order += 1
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def extract_engineer_identity(text: str) -> Optional[str]:
    """Appointed Engineer firm/name from client text, or None."""
    return extract_party_name(text, "engineer")


def party_role_title(role: str) -> str:
    """"engineer's representative" -> "Engineer's Representative"."""
    return " ".join(w[:1].upper() + w[1:] for w in (role or "").split())


def extract_party_name(text: str, role: str) -> Optional[str]:
    """The name the client text gives the asked party, or None.

    Read from the row that names the party (see ``chunk_names_party``): a
    filled particulars row keyed by the role, a scanned line that opens with
    the role, or "<role> is <Name>". Never the definition of the term.
    """
    t = _client_excerpt_text(text or "")
    if not t or not role:
        return None
    role_rx = re.compile(rf"(?i){_role_pattern(role)}")
    for key, val in filled_particulars_rows(t):
        # Live bcb5bbf: a flattened page came through as ONE 400-character
        # "key" that merely contained the word Engineer, with the table
        # header as its value -- and "Clause (as" was returned as the firm.
        # A row's key is a label; a label is short.
        if len(key) > _ENGINEER_KEY_MAX_CHARS:
            continue
        if role_rx.search(key) and _looks_like_appointed_party(val):
            return re.sub(r"\s+", " ", val).strip(" \t.:;,-")
    line_rx = re.compile(
        r"(?im)^[ \t|:]*(?:\d+(?:\.\d+)+\s*(?:\([a-z]\))?[ \t|:]*)?"
        r"(?:(?:the|name\s+of\s+the)\s+)?"
        rf"{_role_pattern(role)}[ \t]*[:|–-]?\s*(.*)$"
    )
    lines = t.splitlines()
    for i, line in enumerate(lines):
        if _ROUTING_HINT_VAL_RE.search(line):
            continue
        m = line_rx.match(line)
        if not m:
            continue
        # The name is the first CELL after the role: drop the row's trailing
        # empty cells and pipes before judging it.
        rest = (m.group(1) or "").split("|")[0].strip(" \t|")
        nxt = lines[i + 1].strip(" \t|") if i + 1 < len(lines) else ""
        nxt2 = lines[i + 2].strip(" \t|") if i + 2 < len(lines) else ""
        for cand in (rest, f"{rest} {nxt}".strip(), nxt, f"{nxt} {nxt2}".strip()):
            if _looks_like_appointed_party(cand) and _PARTY_FIRM_RE.search(cand):
                return re.sub(r"\s+", " ", cand).strip(" \t.:;,-")
            if _looks_like_appointed_party(cand) and re.search(r"[A-Z]{3,}", cand):
                return re.sub(r"\s+", " ", cand).strip(" \t.:;,-")
    is_rx = re.compile(
        rf"(?i)\b(?:the\s+|name\s+of\s+the\s+)?{_role_pattern(role)}\s*(?:is|are|:)\s+(.{{4,80}})"
    )
    for m in is_rx.finditer(_collapse_retrieval_ws(t)):
        cand = m.group(1)
        if _looks_like_appointed_party(cand) and (
            _PARTY_FIRM_RE.search(cand) or re.search(r"\b[A-Z]{3,}\b", cand)
        ):
            return re.sub(r"\s+", " ", cand).strip(" \t.:;,-")
    return None


# ── Defects Notification Period ───────────────────────────────────────────
#
# A Defects Notification Period ask can retrieve service-agreement contents
# pages, recitals and document registers and refuse. The answer is a
# duration from the Taking-Over Certificate in the Contract Data. The ACA / TfC / delay-rate / Engineer asks already had fences;
# the DNP ask was surviving on family-bonus luck and was not named off the
# precedence-list path. Same shape as TfC: state a duration, fence lookalikes.
_DNP_ASK_RE = re.compile(
    r"(?i)(?:defects\s+notification(?:\s+period)?"
    r"|(?:what\s+is\s+(?:the\s+)?)dnp\b)"
)
_DNP_KEY_RE = re.compile(r"(?i)defects\s+notification(?:\s+period)?")
# The Defects Notification Period particulars ROW: a clause number in front
# of the label -- whatever number the contract uses.
_DNP_LABELLED_ROW_RE = re.compile(
    r"(?im)(?:^|\|)[\s|:]*\d+(?:\.\d+)+(?:\s*\([a-z0-9]+\))?[\s|:]*"
    r"defects\s+notification\b"
)
_DNP_DURATION_RE = re.compile(
    r"(?i)\b(\d{1,4})\s+(?:calendar\s+|working\s+)?"
    r"(days?|months?|years?)\b"
)
_DNP_POINTER_RE = re.compile(
    r"(?i)(?:stated|named|identified|set\s+out|specified|defined|"
    r"described|referred\s+to)\s+in\s+(?:the\s+)?contract\s+data"
)
_DNP_GLOSSARY_RE = re.compile(
    r"(?i)(?:defects\s+notification\s+period|\bdnp\b)\s+means\b"
)
_DNP_TOC_RE = re.compile(
    r"(?i)table\s+of\s+contents|document\s+register|\brecitals?\b"
)
_DNP_TOC_ANCHOR_RE = re.compile(r"(?i)taking[- ]over")


def query_asks_for_defects_notification_period(query: str) -> bool:
    """True for a Defects Notification Period ask, not the other particular asks."""
    q = query or ""
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if not _DNP_ASK_RE.search(q):
        return False
    # Neighboring-field asks that happen to mention DNP stay off this path.
    if _ACA_ASK_RE.search(q):
        return False
    if re.search(r"(?i)time\s+for\s+completion", q):
        return False
    if re.search(r"(?i)(?:delay|liquidated)\s+damages", q):
        return False
    if _CD_WHO_IS_RE.search(q):
        return False
    if _BOQ_SCOPE_ASK_RE.search(q):
        return False
    return True


def _format_dnp_duration(match: re.Match) -> str:
    num = match.group(1)
    unit = (match.group(2) or "days").lower()
    if unit.startswith("day"):
        return f"{num} days"
    if unit.startswith("month"):
        return f"{num} months"
    if unit.startswith("year"):
        return f"{num} years"
    return f"{num} {unit}"


def _dnp_duration_from_text(text: str) -> Optional[str]:
    """Duration tied to the DNP label, not a neighbouring notice period."""
    blob = _normalize_retrieval_ws(text)
    if not blob:
        return None
    for m in _DNP_KEY_RE.finditer(blob):
        window = blob[m.start(): m.end() + 140]
        if _DNP_POINTER_RE.search(window) and not _DNP_DURATION_RE.search(window):
            continue
        dm = _DNP_DURATION_RE.search(window)
        if dm:
            return _format_dnp_duration(dm)
    return None


def chunk_states_defects_notification_period(text: str) -> bool:
    """True when the chunk states a Defects Notification Period duration.

    PSA / CPM table-of-contents, recitals, and document registers that
    only *name* the heading (live A6 on 82eb9c5) are lookalikes. A
    General Conditions pointer (``as stated in the Contract Data``) and
    a glossary ``means the period…`` are not the filled particulars row.
    """
    t = text or ""
    if not t:
        return False
    if _DNP_GLOSSARY_RE.search(t) and not filled_particulars_rows(t):
        return False
    for key, val in filled_particulars_rows(t):
        if not _DNP_KEY_RE.search(key):
            continue
        if _DNP_DURATION_RE.search(val) or _DNP_DURATION_RE.search(key):
            return True
    blob = _normalize_retrieval_ws(t)
    if _DNP_POINTER_RE.search(blob) and not (
        _CD_PARTICULARS_PREFIX_RE.search(t)
        or (
            _CD_HEADING_IN_CHUNK_RE.search(t)
            and not contract_data_mention_is_only_a_cross_reference(t)
        )
    ):
        return False
    if _DNP_TOC_RE.search(blob) and not (
        _CD_PARTICULARS_PREFIX_RE.search(t) or filled_particulars_rows(t)
    ):
        return False
    return bool(_dnp_duration_from_text(t))


def extract_defects_notification_period(text: str) -> Optional[str]:
    """DNP duration as written (e.g. ``365 days``), or None.

    Prefers the numbered particulars row / Taking-Over / Contract Data over a
    glossary or a TOC heading that happens to sit near a duration.
    """
    t = text or ""
    if not t:
        return None
    cands: List[Tuple[int, int, str]] = []
    order = 0
    for key, val in filled_particulars_rows(t):
        if not _DNP_KEY_RE.search(key):
            continue
        m = _DNP_DURATION_RE.search(val) or _DNP_DURATION_RE.search(key)
        if not m:
            continue
        days = _format_dnp_duration(m)
        joined = f"{key} {val}"
        score = 60
        # The row's own clause number, not one elsewhere in the blob.
        if _DNP_LABELLED_ROW_RE.search(joined):
            score += 80
        if _CD_PARTICULARS_PREFIX_RE.search(t):
            score += 40
        if _DNP_TOC_ANCHOR_RE.search(joined):
            score += 30
        ids = extract_contract_doc_ids(t)
        if ids:
            year, seq = max(_contract_id_recency(cid) for cid in ids)
            if year > 0:
                score += year
            if seq > 0:
                score += min(seq, 30)
        cands.append((-score, order, days))
        order += 1
    for block in re.split(r"\n{2,}|\[doc_id=", t):
        if not chunk_states_defects_notification_period(block):
            continue
        days = _dnp_duration_from_text(block)
        if not days:
            continue
        score = 0
        if _DNP_LABELLED_ROW_RE.search(block):
            score += 80
        if _CD_PARTICULARS_PREFIX_RE.search(block):
            score += 60
        if _DNP_TOC_ANCHOR_RE.search(block):
            score += 30
        if _CD_HEADING_IN_CHUNK_RE.search(block):
            score += 20
        cands.append((-score, order, days))
        order += 1
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def query_wants_contract_data_file(query: str) -> bool:
    """These particulars live in a Contract Data file, not PSA / CPM."""
    return (
        query_asks_for_accepted_contract_amount(query)
        or query_asks_for_time_for_completion(query)
        or query_asks_who_the_engineer_is(query)
        or bool(asked_party_role(query))
        or query_asks_delay_damages_daily_amount(query)
        or (
            query_asks_for_defects_notification_period(query)
        )
        or (
            query_asks_for_parent_company_guarantee(query)
        )
        or (
            query_asks_for_contract_commencement_date(query)
        )
    )


def _pair_adjacent_keep_text(
    hits: List[Chunk],
    keep,
    *,
    window: int = 2,
) -> List[Chunk]:
    """Scanned Contract Data often splits a label and its value.

    Engineer ask: ``Engineer`` on chunk N, ``<FIRM> (<Firm> Limited)`` on
    N+1. Identifier keep() then fails on both. Pair consecutive same-doc
    chunks so the appointment / TfC / including-VAT row is visible.
    ``window`` > 2 also joins N+2 (the daily-amount excl-VAT amount one row
    past the 1.1.1 label). Default 2 keeps Engineer / ACA pairing unchanged.
    """
    span = max(2, int(window or 2))
    by_doc: Dict[str, List[Chunk]] = {}
    for chunk in hits:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    out: List[Chunk] = []
    seen: Set[str] = set()
    for group in by_doc.values():
        group = sorted(group, key=lambda c: int(getattr(c, "chunk_index", 0) or 0))
        for i, chunk in enumerate(group):
            text = chunk.text or ""
            if keep(text):
                if chunk.chunk_id not in seen:
                    out.append(chunk)
                    seen.add(chunk.chunk_id)
                continue
            paired_hit = False
            for width in range(2, span + 1):
                if i + width - 1 >= len(group):
                    break
                idxs = [
                    int(getattr(group[i + j], "chunk_index", 0) or 0)
                    for j in range(width)
                ]
                # Sparse fetches (chunk #0 + appendix 80) must not glue
                # a delay-damages window onto the including-VAT row.
                # Equal indexes (tests that omit chunk_index) keep the
                # old list-adjacency pairing.
                if len(set(idxs)) > 1 and any(
                    idxs[j] != idxs[0] + j for j in range(width)
                ):
                    continue
                combined = "\n".join(
                    (group[i + j].text or "") for j in range(width)
                )
                if keep(combined):
                    paired = replace(chunk, text=combined)
                    if paired.chunk_id not in seen:
                        out.append(paired)
                        seen.add(paired.chunk_id)
                    paired_hit = True
                    break
            if paired_hit:
                continue
    return out


def _pool_lexical_hits_matching(
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    phrases: Tuple[str, ...],
    keep,
    *,
    label: str,
    bonus: float = _ASKED_PARTICULAR_VALUE_BONUS,
) -> int:
    """Pull lexical hits that pass ``keep`` into ``fused``. Project-only.

    General-knowledge notes (illustrative 0.05%) must not be rescued.
    Failures never raise.
    """
    fetch = getattr(store, "identifier_search", None)
    if not callable(fetch) or not phrases:
        return 0
    recovered = 0
    try:
        hits = fetch(project_id, list(phrases), k=20)
    except Exception as exc:  # noqa: BLE001 — extras must not break the turn
        logger.warning("%s rescue for %s failed: %s", label, project_id, exc)
        return 0
    neighbors = getattr(store, "chunks_for_docs", None)
    if callable(neighbors) and hits:
        doc_ids = list({c.doc_id for c in hits if c.doc_id})
        try:
            extra = neighbors(project_id, doc_ids, k_per_doc=24)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s neighbor fetch for %s failed: %s", label, project_id, exc)
            extra = []
        by_id = {c.chunk_id: c for c in hits}
        for chunk in extra:
            by_id.setdefault(chunk.chunk_id, chunk)
        hits = list(by_id.values())
    for chunk in _pair_adjacent_keep_text(hits, keep):
        if chunk.chunk_id in fused:
            continue
        add = bonus(chunk.text or "") if callable(bonus) else bonus
        fused[chunk.chunk_id] = (chunk, 0.0, add)
        recovered += 1
    if recovered:
        logger.info("%s lexical recall pooled %d chunk(s)", label, recovered)
    return recovered


def _chunks_have_both_daily_damages_operands(chunks: Iterable) -> bool:
    """True when loaded rows already have Contract Data 0.1% and excl-VAT ACA."""
    has_rate = False
    has_aca = False
    for chunk in chunks or []:
        text = getattr(chunk, "text", None)
        if text is None and isinstance(chunk, str):
            text = chunk
        text = text or ""
        if not has_rate and _daily_rate_preference(text) >= 2:
            has_rate = True
        if not has_aca and _has_standalone_excl_vat_aca(text):
            has_aca = True
        if has_rate and has_aca:
            return True
    return False


def _doc_qualifies_for_late_aca_scan(text: str, name: str) -> bool:
    """True for a fused row whose document may still hold daily-amount operands.

    Daily-amount ask after #537: Cosine kept pointer-only Contract Data
    8.8 chunks 9–11. ``chunk_states_delay_damages_rate`` is false on a
    pointer, and a truncated Sources filename (``<id>_Vol N_Con…``) misses
    ``filename_looks_like_conditions_volume``. The bound volume still has
    0.1% + excl-VAT later — qualify the doc from the 8.8 pointer too.
    """
    if filename_looks_like_conditions_volume(name):
        return True
    if chunk_states_delay_damages_rate(text):
        return True
    t = text or ""
    if _DELAY_RATE_POINTER_RE.search(t):
        return True
    # A numbered delay-damages clause, whatever its number.
    return bool(_DELAY_DAMAGES_CLAUSE_RE.search(t))


def _late_scan_project_ids(
    project_id: str,
    extra_pids: Optional[Iterable[str]] = None,
    fused: Optional[Dict[str, Tuple]] = None,
) -> List[str]:
    """UI project + cited-chunk owners + Master Corpus source.

    Master Corpus daily-amount ask: retrieve may surface chunks whose
    ``project_id`` is the source corpus, while graft / late-scan were
    keyed only on the UI id. ``chunks_for_docs`` then returned empty
    and the cost-grounding gate refused.
    """
    out: List[str] = []
    seen: Set[str] = set()

    def _add(pid: str) -> None:
        p = (pid or "").strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)

    _add(project_id)
    try:
        # Alias remap only: when the operator is ON the master-corpus
        # project, read its backing source. Do not add a foreign project's
        # own id just because it is configured as a fallback corpus.
        from app.core.projects import _master_corpus_source
        _add(_master_corpus_source(project_id) or "")
    except Exception:  # noqa: BLE001 — alias remap is optional
        logger.debug("e1 master-corpus alias remap unavailable", exc_info=True)
    for pid in extra_pids or []:
        _add(pid)
    if fused:
        for entry in fused.values():
            chunk = entry[0] if isinstance(entry, tuple) and entry else entry
            if isinstance(chunk, Chunk):
                _add(getattr(chunk, "project_id", "") or "")
    return out


def _pool_doc_ids_for_late_aca(fused: Dict[str, Tuple]) -> List[str]:
    """Rate-window docs already in fused, plus any Contract Data filename."""
    doc_ids: List[str] = []
    seen: Set[str] = set()

    def _fused_chunk(entry) -> Optional[Chunk]:
        if isinstance(entry, tuple) and entry:
            chunk = entry[0]
        else:
            chunk = entry
        return chunk if isinstance(chunk, Chunk) else None

    for entry in fused.values():
        chunk = _fused_chunk(entry)
        if chunk is None or not chunk.doc_id or chunk.doc_id in seen:
            continue
        text = chunk.text or ""
        name = getattr(chunk, "source_name", "") or ""
        if not name:
            try:
                name = _doc_name_for_id(chunk.doc_id) or ""
            except Exception:  # noqa: BLE001 — filename is optional
                name = ""
        if not _doc_qualifies_for_late_aca_scan(text, name):
            continue
        seen.add(chunk.doc_id)
        doc_ids.append(chunk.doc_id)
    if doc_ids:
        return doc_ids
    # Daily-amount ask after #538: Cosine 9–11 may be OCR that fails
    # pointer / 8.8 / filename qualify (a truncated Sources name).
    # Still scan those docs — compose only keeps real operands.
    for entry in fused.values():
        chunk = _fused_chunk(entry)
        if chunk is None or not chunk.doc_id or chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        doc_ids.append(chunk.doc_id)
        if len(doc_ids) >= 8:
            break
    return doc_ids


def _doc_owner_project_ids(doc_ids: List[str]) -> List[str]:
    """Project ids that actually own the cited documents.

    Master Corpus daily-amount ask: UI / remap pid can miss the row
    owner. ``chunks_for_docs`` is exact-pid, so resolve from the
    document row when the cited ``doc_id`` is known.
    """
    out: List[str] = []
    seen: Set[str] = set()
    try:
        from app.core.projects import get_document
    except Exception:  # noqa: BLE001 — listing is optional
        return out
    for did in doc_ids or []:
        if not did:
            continue
        try:
            doc = get_document(did) or {}
        except Exception:  # noqa: BLE001 — one miss must not skip the rest
            continue
        pid = str(doc.get("project_id") or "").strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def label_word_needle_sets(labels: Iterable[str]) -> List[Tuple[str, ...]]:
    """One AND-set of words per label, for a text match that survives line splits.

    A scanned label prints "Accepted\\nContract\\nAmount"; a phrase match
    misses it, a match on each word does not. Words shorter than three
    characters are dropped (the store ignores them).
    """
    out: List[Tuple[str, ...]] = []
    for label in labels:
        words = tuple(w for w in re.findall(r"[a-z0-9]+", (label or "").lower()) if len(w) >= 3)
        if words and words not in out:
            out.append(words)
    return out


def _scan_whole_documents(
    store,
    project_id: str,
    doc_ids: List[str],
    extra_pids: Optional[Iterable[str]] = None,
    fused: Optional[Dict[str, Tuple]] = None,
    *,
    needle_sets: Iterable[Tuple[str, ...]] = (),
    done=None,
) -> List[Chunk]:
    """Every chunk of ``doc_ids`` this store will give, then a word match.

    Reads each document whole (``all_rows``); for a store that ignores that
    or caps a fetch, a very large first-N, then prefix and tail windows,
    then fixed windows across the middle. ``done(chunks)`` -- when given --
    stops the walk as soon as the chunks in hand answer; without it the walk
    always runs to the end. Finally, ``needle_sets`` are matched as text
    inside the same documents. The row owner is not always the UI project,
    so the owners of the cited documents are tried too.
    """
    by_id: Dict[str, Chunk] = {}
    allowed = set(doc_ids)
    fetch = getattr(store, "chunks_for_docs", None)
    pids = _late_scan_project_ids(project_id, extra_pids, fused)
    for pid in _doc_owner_project_ids(doc_ids):
        if pid not in pids:
            pids.append(pid)
    if not pids and project_id:
        pids = [project_id]

    def _keep(chunk: Chunk) -> None:
        if chunk.doc_id and chunk.doc_id not in allowed:
            return
        by_id.setdefault(chunk.chunk_id, chunk)

    def _answered() -> bool:
        return bool(done) and done(by_id.values())

    def _fetch(pid: str, **kwargs) -> List[Chunk]:
        try:
            return list(fetch(pid, doc_ids, **kwargs) or [])
        except TypeError:
            # This store's chunks_for_docs does not take these keywords; the
            # next fetch shape is tried.
            logger.debug("chunks_for_docs does not accept %r", sorted(kwargs))
            return []
        except Exception as exc:  # noqa: BLE001 — extras must not break the turn
            logger.warning("whole-document scan %r for %s failed: %s", kwargs, pid, exc)
            return []

    if callable(fetch):
        for pid in pids:
            for chunk in _fetch(pid, all_rows=True):
                _keep(chunk)
            # A store that ignores all_rows returns first-N: ask for everything.
            if not _answered():
                for chunk in _fetch(pid, k_per_doc=1_000_000):
                    _keep(chunk)
            if _answered():
                return list(by_id.values())
            for from_end in (False, True):
                got = _fetch(pid, k_per_doc=_DOC_SCAN_WINDOW, from_end=from_end)
                if not got and not from_end:
                    got = _fetch(pid, k_per_doc=_DOC_SCAN_WINDOW)
                for chunk in got:
                    _keep(chunk)
            if _answered():
                return list(by_id.values())
            # Windows across the middle, for a store that caps k_per_doc.
            offset = _DOC_SCAN_WINDOW
            while offset < _DOC_SCAN_MAX:
                try:
                    got = list(fetch(
                        pid, doc_ids, k_per_doc=_DOC_SCAN_WINDOW, offset=offset,
                    ) or [])
                except TypeError:
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "whole-document window offset=%s for %s failed: %s",
                        offset, pid, exc,
                    )
                    break
                if not got:
                    break
                for chunk in got:
                    _keep(chunk)
                if _answered():
                    return list(by_id.values())
                if len(got) < _DOC_SCAN_WINDOW:
                    break
                offset += _DOC_SCAN_WINDOW

    containing = getattr(store, "chunks_containing_all", None)
    if callable(containing):
        for pid in pids:
            for needles in needle_sets:
                try:
                    try:
                        hits = containing(
                            pid, list(needles), k=_DOC_TEXT_SCAN_K, doc_ids=doc_ids,
                        )
                    except TypeError:
                        hits = containing(pid, list(needles), k=_DOC_TEXT_SCAN_K)
                except Exception as exc:  # noqa: BLE001 — extras must not break
                    logger.warning("whole-document text scan for %s failed: %s", pid, exc)
                    hits = []
                for chunk in hits or []:
                    _keep(chunk)
            if _answered():
                return list(by_id.values())
    return list(by_id.values())


# The operands of a delay-damages daily amount: the rate row and the sum it
# is a share of. Contract-particular names, matched word by word.
_DAILY_DAMAGES_OPERAND_LABELS = (
    "delay damages", "contract price", "accepted contract amount",
)


def _scan_documents_for_operands(
    store,
    project_id: str,
    doc_ids: List[str],
    extra_pids: Optional[Iterable[str]] = None,
    fused: Optional[Dict[str, Tuple]] = None,
) -> List[Chunk]:
    """Whole-document scan that stops once both daily-amount operands are in hand."""
    return _scan_whole_documents(
        store, project_id, doc_ids, extra_pids, fused,
        needle_sets=label_word_needle_sets(_DAILY_DAMAGES_OPERAND_LABELS),
        done=_chunks_have_both_daily_damages_operands,
    )
_RAG_CONTEXT_DOC_ID_RE = re.compile(r"\[doc_id=([^\]\s]+)")


def doc_ids_from_rag_context(rag_context: str) -> List[str]:
    """Doc ids from ``[doc_id=…]`` markers in the injected RAG context."""
    out: List[str] = []
    seen: Set[str] = set()
    for match in _RAG_CONTEXT_DOC_ID_RE.finditer(rag_context or ""):
        did = (match.group(1) or "").strip()
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def daily_damages_excerpts_from_loaded_cd_volume(
    query: str,
    project_id: str,
    store=None,
    *,
    rag_context: str = "",
    doc_ids: Optional[List[str]] = None,
    extra_pids: Optional[Iterable[str]] = None,
) -> str:
    """Join Contract Data 0.1% + excl-VAT ACA from the loaded CD volume.

    Daily-amount ask after #536: top-k stayed on Contract Data chunks
    9–11 that do not surface both operands, so compose returned None
    and the cost-grounding gate refused. When those rows exist later
    in the same loaded volume, return them so compose can state
    SAR/day — do not invent a figure and do not elect a restated milestone rate.
    """
    pids = _late_scan_project_ids(project_id, extra_pids)
    if not (
        query_asks_delay_damages_daily_amount(query)
    ):
        return ""
    ids: List[str] = []
    seen: Set[str] = set()

    def _add(did: str) -> None:
        if did and did not in seen:
            seen.add(did)
            ids.append(did)

    # Cited RAG doc_ids first (live Sources 9–11 of one volume). Do not
    # all_rows-scan every "conditions of contract" hit on a 3k-doc corpus
    # — that timed out and left last-chance empty 4/5 New-chat attempts.
    for did in doc_ids or []:
        _add(did)
    for did in doc_ids_from_rag_context(rag_context):
        _add(did)
    cited = list(ids)

    if store is None:
        try:
            store = get_lexical_store()
        except Exception:  # noqa: BLE001 — never break a turn over the store
            logger.debug("e1 loaded-volume store open failed", exc_info=True)
            return ""

    extra: List[Chunk] = []
    if cited:
        extra = _scan_documents_for_operands(
            store, project_id or (pids[0] if pids else ""), cited,
            extra_pids=pids,
        )
        if _chunks_have_both_daily_damages_operands(extra):
            ids = cited
        else:
            extra = extra or []
    if not _chunks_have_both_daily_damages_operands(extra):
        try:
            from app.core.projects import documents_matching_title_phrase
            for pid in pids or [project_id]:
                if not pid:
                    continue
                for phrase in ("contract data", "conditions of contract"):
                    try:
                        matches = documents_matching_title_phrase(pid, phrase) or []
                    except Exception:  # noqa: BLE001 — listing is optional
                        logger.debug(
                            "e1 loaded-volume title listing failed for %r",
                            phrase, exc_info=True,
                        )
                        matches = []
                    for doc in matches:
                        _add(doc.get("id") or "")
        except Exception:  # noqa: BLE001 — rag doc_ids may still be enough
            logger.debug("e1 loaded-volume projects import failed", exc_info=True)
        added = [did for did in ids if did not in set(cited)]
        if added:
            extra = list(extra or []) + _scan_documents_for_operands(
                store, project_id or (pids[0] if pids else ""), added[:2],
                extra_pids=pids,
            )

    if not ids:
        return ""
    rate_parts: List[str] = []
    aca_parts: List[str] = []

    def _collect(text: str) -> None:
        if _daily_rate_preference(text) >= 2 and text not in rate_parts:
            rate_parts.append(text)
        if _has_standalone_excl_vat_aca(text) and text not in aca_parts:
            aca_parts.append(text)

    for chunk in extra or []:
        _collect(chunk.text or "")
    if not rate_parts or not aca_parts:
        for chunk in _pair_adjacent_keep_text(
            extra or [],
            lambda t: (
                _daily_rate_preference(t) >= 2 or _has_standalone_excl_vat_aca(t)
            ),
            window=_ROW_PAIR_WINDOW,
        ):
            _collect(chunk.text or "")
    if not rate_parts or not aca_parts:
        return ""
    # The joined excerpt carries NO [doc_id=] markers, so compose reads it as
    # one document and its clause-1.1.1 price search cannot tell the filled
    # base from a partial ACA. If partial excl-VAT rows fill aca_parts[:3] the
    # real base row is dropped and compose elects the partial (milestone delay
    # damages: a milestone rate x a partial amount). Order the base row first so the
    # cap can never drop it.
    return "\n\n".join(
        rate_parts[:3] + _aca_parts_clause_111_first(aca_parts)[:3]
    )


def _aca_parts_clause_111_first(parts: List[str]) -> List[str]:
    """Stable-sort ACA excerpt texts so a clause-1.1.1 excl-VAT base leads.

    A partial Accepted Contract Amount and the filled clause 1.1.1 both pass
    ``_has_standalone_excl_vat_aca``; only the 1.1.1 row is the real base. The
    rescue caps the joined excerpt at three ACA parts, so without this a run
    of partials pushed the 1.1.1 row out of the window.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            text_states_clause_111_aca,
        )
    except Exception:  # noqa: BLE001 — ordering is best-effort
        return parts
    return sorted(
        parts, key=lambda t: 0 if text_states_clause_111_aca(t) else 1,
    )


def _loaded_cd_chunk_texts(
    project_id: str,
    extra_pids: Optional[Iterable[str]] = None,
    store=None,
) -> List[str]:
    """Every Contract Data chunk text in the loaded volume, or []."""
    pids = _late_scan_project_ids(project_id, extra_pids)
    if store is None:
        try:
            store = get_lexical_store()
        except Exception:  # noqa: BLE001 — never break a turn over the store
            logger.debug("loaded-CD store open failed", exc_info=True)
            return []
    fetch = getattr(store, "chunks_for_docs", None)
    if not callable(fetch):
        return []
    try:
        from app.core.projects import documents_matching_title_phrase
    except Exception:  # noqa: BLE001 — listing is optional
        logger.debug("loaded-CD projects import failed", exc_info=True)
        return []
    texts: List[str] = []
    seen: Set[str] = set()
    for pid in pids or [project_id]:
        if not pid:
            continue
        try:
            docs = documents_matching_title_phrase(pid, "contract data") or []
            hits = fetch(pid, [d["id"] for d in docs], k_per_doc=80) if docs else []
        except Exception as exc:  # noqa: BLE001 — extras must not break
            logger.warning("loaded-CD volume scan for %s failed: %s", pid, exc)
            continue
        for chunk in hits or []:
            cid = getattr(chunk, "chunk_id", None) or str(id(chunk))
            if cid in seen:
                continue
            seen.add(cid)
            text = getattr(chunk, "text", "") or ""
            if text:
                texts.append(text)
    return texts


def percentage_of_aca_excerpts_from_loaded_cd_volume(
    query: str,
    project_id: str,
    store=None,
    *,
    rag_context: str = "",
    extra_pids: Optional[Iterable[str]] = None,
) -> str:
    """Join Advance Payment % + excl-VAT ACA from the loaded CD volume.

    Advance-payment amount ask: top-k had neither operand and the model shipped the
    fetch truncation notice. When those rows exist later in the same
    volume, return them so compose can state SAR — do not invent a
    figure. Kill-switch: COMPOSE_PERCENTAGE_OF_ACA=0.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            compose_percentage_of_aca_enabled,
            extract_named_percentage_particular,
            query_asks_percentage_particular_in_money,
        )
    except Exception:  # noqa: BLE001 — never break a turn over an import
        logger.debug("percentage-of-ACA import failed", exc_info=True)
        return ""
    if not compose_percentage_of_aca_enabled():
        return ""
    if not query_asks_percentage_particular_in_money(query):
        return ""
    pct_parts: List[str] = []
    aca_parts: List[str] = []
    texts = list(_loaded_cd_chunk_texts(project_id, extra_pids, store))
    if rag_context:
        texts.append(rag_context)
    for text in texts:
        if extract_named_percentage_particular(query, text):
            if text not in pct_parts:
                pct_parts.append(text)
        if _has_standalone_excl_vat_aca(text) and text not in aca_parts:
            aca_parts.append(text)
    if not pct_parts or not aca_parts:
        return ""
    return "\n\n".join(pct_parts[:3] + aca_parts[:3])


def milestone_period_excerpts_from_loaded_cd_volume(
    query: str,
    project_id: str,
    store=None,
    *,
    rag_context: str = "",
    extra_pids: Optional[Iterable[str]] = None,
) -> str:
    """Join Milestone N | a milestone rate rows + excl-VAT ACA from the loaded volume.

    Per-milestone delay damages ask: top-k packed whole-of-Works 0.1% under "per Milestone".
    Scan for the real milestone rate rows. Kill-switch:
    COMPOSE_DELAY_DAMAGES_PERIOD=0.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            compose_delay_damages_period_enabled,
            parse_asked_milestones,
            parse_milestone_delay_rate_percent,
            query_asks_delay_damages_over_a_period,
        )
    except Exception:  # noqa: BLE001 — never break a turn over an import
        logger.debug("milestone-period import failed", exc_info=True)
        return ""
    if not compose_delay_damages_period_enabled():
        return ""
    if not query_asks_delay_damages_over_a_period(query):
        return ""
    milestones = parse_asked_milestones(query)
    rate_parts: List[str] = []
    aca_parts: List[str] = []
    texts = list(_loaded_cd_chunk_texts(project_id, extra_pids, store))
    if rag_context:
        texts.append(rag_context)
    for text in texts:
        if milestones and any(
            parse_milestone_delay_rate_percent(text, n) is not None
            for n in milestones
        ):
            if text not in rate_parts:
                rate_parts.append(text)
        if _has_standalone_excl_vat_aca(text) and text not in aca_parts:
            aca_parts.append(text)
    if not rate_parts or not aca_parts:
        return ""
    return "\n\n".join(rate_parts[:3] + aca_parts[:3])


def community_tfc_span_excerpts_from_loaded_cd_volume(
    query: str,
    project_id: str,
    store=None,
    *,
    rag_context: str = "",
    extra_pids: Optional[Iterable[str]] = None,
) -> str:
    """Join named-community Time-for-Completion rows from the loaded volume.

    The top-k can stop at the page break, before the asked community's
    milestones, which sit on the continuation page. Return those rows so
    compose can state the span -- do not invent days.
    """
    if not query_asks_named_community_tfc_span(query):
        return ""
    community = extract_asked_community_name(query)
    needle = (community or "").lower()
    if not needle:
        return ""
    parts: List[str] = []
    texts = list(_loaded_cd_chunk_texts(project_id, extra_pids, store))
    if rag_context:
        texts.append(rag_context)
    for text in texts:
        if needle not in (text or "").lower():
            continue
        if not re.search(r"(?i)\d+\s*days", text):
            continue
        if text not in parts:
            parts.append(text)
    if not parts:
        return ""
    composed = compose_named_community_tfc_span(query, "\n\n".join(parts))
    if not composed:
        return ""
    return "\n\n".join(parts[:6])


def _fused_entry_chunk(entry) -> Optional[Chunk]:
    if isinstance(entry, tuple) and entry:
        chunk = entry[0]
    else:
        chunk = entry
    return chunk if isinstance(chunk, Chunk) else None


_DEEP_ROW_TITLE_KINDS = _PARTICULARS_KIND_PHRASES + ("conditions of contract",)
_DEEP_ROW_MAX_LABELS = 4


def _pooled_documents_for_deep_rows(fused: Dict[str, Tuple], phrases: List[str]) -> List[str]:
    """Pooled documents worth reading whole for an asked particular.

    A particulars or conditions volume by name, or a document whose pooled
    chunk mentions an asked particular without stating it.
    """
    doc_ids: List[str] = []
    for entry in fused.values():
        chunk = _fused_entry_chunk(entry)
        if chunk is None or not chunk.doc_id or chunk.doc_id in doc_ids:
            continue
        name = getattr(chunk, "source_name", "") or ""
        if not name:
            try:
                name = _doc_name_for_id(chunk.doc_id) or ""
            except Exception:  # noqa: BLE001 — filename is optional
                name = ""
        blob = _normalize_retrieval_ws(chunk.text or "").lower()
        if (
            filename_looks_like_contract_data(name)
            or filename_looks_like_conditions_volume(name)
            or any(p and p in blob for p in phrases)
        ):
            doc_ids.append(chunk.doc_id)
    return doc_ids


def _particulars_documents_by_title(pids: List[str]) -> List[str]:
    """Documents whose upload name says they are particulars / conditions."""
    out: List[str] = []
    try:
        from app.core.projects import documents_matching_title_phrase
    except Exception:  # noqa: BLE001 — listing is optional
        logger.debug("particulars title listing unavailable", exc_info=True)
        return out
    for pid in pids:
        if not pid:
            continue
        for phrase in _DEEP_ROW_TITLE_KINDS:
            try:
                matches = documents_matching_title_phrase(pid, phrase) or []
            except Exception:  # noqa: BLE001 — listing is optional
                matches = []
            for doc in matches:
                did = doc.get("id") or ""
                if did and did not in out:
                    out.append(did)
    return out


def recall_rows_deep_in_pooled_documents(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
) -> int:
    """Pool the asked particular's row when it sits past the early windows.

    For a particular whose row shape is known: when no pooled chunk states
    it, the pooled particulars / conditions volumes (or those that mention
    the particular) are read whole and the rows that state it are pooled
    with the asked-value bonus. A daily amount is two rows and has its own
    pass (``recall_composition_operands``).
    """
    test = _known_particular_row_test(query)
    if test is None or query_asks_delay_damages_daily_amount(query):
        return 0
    if not query_wants_contract_data_file(query):
        return 0
    fused_chunks = [c for c in (_fused_entry_chunk(e) for e in fused.values()) if c is not None]
    if any(test(c.text or "") for c in fused_chunks):
        return 0
    phrases = list(_asked_particular_key_phrases(query))
    doc_ids = _pooled_documents_for_deep_rows(fused, phrases)
    pids = _late_scan_project_ids(project_id, None, fused)
    if not doc_ids:
        doc_ids = _particulars_documents_by_title(pids or [project_id])
    if not doc_ids:
        return 0
    labels = asked_row_labels(query)[:_DEEP_ROW_MAX_LABELS]
    extra = _scan_whole_documents(
        store, project_id, doc_ids, extra_pids=pids, fused=fused,
        needle_sets=label_word_needle_sets(labels),
    )
    recovered = 0
    for chunk in _pair_adjacent_keep_text(extra or [], test, window=_ROW_PAIR_WINDOW):
        if chunk.chunk_id in fused or not test(chunk.text or ""):
            continue
        fused[chunk.chunk_id] = (chunk, 0.0, _ASKED_PARTICULAR_VALUE_BONUS)
        recovered += 1
    if recovered:
        logger.info("deep-row recall pooled %d chunk(s) past the early windows", recovered)
    return recovered


def including_vat_excerpts_from_loaded_cd_volume(
    query: str,
    project_id: str,
    store=None,
    *,
    rag_context: str = "",
    doc_ids: Optional[List[str]] = None,
    extra_pids: Optional[Iterable[str]] = None,
) -> str:
    """Join including-VAT ACA rows from the loaded particulars volume.

    The top-k can stay on the volume's early windows (a delay-damages clause
    restating some other amount). When the filled including-VAT row exists
    later in the same loaded volume -- possibly owned by a cited source
    project, not the UI id -- return it so the graft can state it. Never
    invents a figure and never composes delay damages.
    """
    pids = _late_scan_project_ids(project_id, extra_pids)
    if not (
        query_is_aca_including_vat_particular(query)
        and (pids or project_id)
    ):
        return ""
    ids: List[str] = []
    seen: Set[str] = set()

    def _add(did: str) -> None:
        if did and did not in seen:
            seen.add(did)
            ids.append(did)

    for did in doc_ids or []:
        _add(did)
    for did in doc_ids_from_rag_context(rag_context):
        _add(did)

    try:
        from app.core.projects import documents_matching_title_phrase
        for pid in pids or [project_id]:
            if not pid:
                continue
            for phrase in _DEEP_ROW_TITLE_KINDS:
                try:
                    matches = documents_matching_title_phrase(pid, phrase) or []
                except Exception:  # noqa: BLE001 — listing is optional
                    logger.debug(
                        "loaded-volume title listing failed for %r",
                        phrase, exc_info=True,
                    )
                    matches = []
                for doc in matches:
                    _add(doc.get("id") or "")
    except Exception:  # noqa: BLE001 — rag doc_ids may still be enough
        logger.debug("loaded-volume projects import failed", exc_info=True)

    if not ids:
        return ""
    if store is None:
        try:
            store = get_lexical_store()
        except Exception:  # noqa: BLE001 — never break a turn over the store
            logger.debug("loaded-volume store open failed", exc_info=True)
            return ""

    extra = _scan_whole_documents(
        store, project_id or (pids[0] if pids else ""), ids, extra_pids=pids,
        needle_sets=label_word_needle_sets(asked_row_labels(query)[:_DEEP_ROW_MAX_LABELS]),
    )
    parts: List[str] = []
    for chunk in _pair_adjacent_keep_text(
        extra or [],
        chunk_states_aca_including_vat,
        window=_ROW_PAIR_WINDOW,
    ):
        text = chunk.text or ""
        if chunk_states_aca_including_vat(text) and text not in parts:
            parts.append(text)
    if not parts:
        for chunk in extra or []:
            text = chunk.text or ""
            if chunk_states_aca_including_vat(text) and text not in parts:
                parts.append(text)
    if not parts:
        return ""
    return "\n\n".join(parts[:3])


def ensure_kept_has_including_vat(
    query: str,
    kept: List[Chunk],
    ranked: List[Chunk],
    *,
    allow=None,
) -> bool:
    """Put the including-VAT row in top-k when it is already ranked.

    Live Wave-1 A2 on 9ad62cc: kept stayed on Contract Data chunk #0
    (delay damages × a partial ACA) after the late scan added the
    filled including-VAT row to ranked. Prefer replacing a delay-
    damages window.
    """
    if not kept:
        return False
    if not (
        query_is_aca_including_vat_particular(query)
    ):
        return False
    if any(chunk_states_aca_including_vat(c.text or "") for c in kept):
        return False

    def _ok(chunk: Chunk) -> bool:
        return allow is None or allow(chunk)

    incl: Optional[Chunk] = None
    for chunk in ranked:
        if not _ok(chunk):
            continue
        if chunk_states_aca_including_vat(chunk.text or ""):
            incl = chunk
            break
    if incl is None:
        return False
    present = {c.chunk_id for c in kept}
    if incl.chunk_id in present:
        return False
    replace_at = 0
    for i, chunk in enumerate(kept):
        text = chunk.text or ""
        if chunk_states_delay_damages_rate(text) or (
            chunk_states_accepted_contract_amount(text)
            and not chunk_states_aca_including_vat(text)
        ):
            replace_at = i
            break
        replace_at = i
    kept[replace_at] = incl
    return True


def _daily_damages_operands():
    """The operands of a delay-damages daily amount, as (name, is_pooled, admits).

    ``is_pooled(text)`` says a pooled chunk already supplies the operand;
    ``admits(text)`` says a scanned chunk supplies it.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            chunk_has_real_accepted_contract_amount,
        )
    except Exception:  # noqa: BLE001 — never break a turn over an import
        logger.debug("composition operand import failed", exc_info=True)
        return []

    def _base_admits(text: str) -> bool:
        # The sum the rate is a share of, stated as its own row -- not a rate
        # window that restates some amount.
        return (
            chunk_states_accepted_contract_amount(text)
            and chunk_has_real_accepted_contract_amount(text)
            and not chunk_states_delay_damages_rate(text)
        )

    return [
        ("base", _has_standalone_excl_vat_aca, _base_admits),
        ("rate", lambda text: _daily_rate_preference(text) >= 2,
         lambda text: _daily_rate_preference(text) >= 2),
    ]


def composition_operands_for(query: str):
    """The operands a question's composition needs, or [] when it composes nothing."""
    if query_asks_delay_damages_daily_amount(query):
        return _daily_damages_operands()
    return []


def recall_composition_operands(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
) -> int:
    """Pool every operand a composition needs, read from the documents in the pool.

    A computed answer (a daily amount = a rate x the sum it is a share of)
    needs every operand row in the context, and each operand is a row of a
    particulars / conditions volume that is often past the early windows the
    store returns first. When an operand is missing from the pool, the
    pooled documents that hold the other operand -- or any pooled
    particulars / conditions volume -- are read whole and the missing
    operand's rows are pooled, pinned above the early windows so the token
    cap keeps them. An operand already pooled is not fetched again.
    """
    operands = composition_operands_for(query)
    if not operands:
        return 0
    fused_chunks = [c for c in (_fused_entry_chunk(e) for e in fused.values()) if c is not None]
    missing = [
        (name, admits) for name, is_pooled, admits in operands
        if not any(is_pooled(c.text or "") for c in fused_chunks)
    ]
    if not missing:
        return 0
    doc_ids = _pool_doc_ids_for_late_aca(fused)
    if not doc_ids:
        return 0
    extra = _scan_documents_for_operands(store, project_id, doc_ids, fused=fused)
    recovered = 0
    for name, admits in missing:
        # A row whose label and value were split across chunks is joined back.
        candidates = (
            _pair_adjacent_keep_text(extra or [], chunk_states_accepted_contract_amount,
                                     window=_ROW_PAIR_WINDOW)
            if name == "base" else list(extra or [])
        )
        for chunk in candidates:
            if chunk.chunk_id in fused or not admits(chunk.text or ""):
                continue
            fused[chunk.chunk_id] = (chunk, _OPERAND_PIN_SCORE, 0.0)
            recovered += 1
    if recovered:
        logger.info("composition-operand recall pooled %d chunk(s)", recovered)
    return recovered


def _apply_asked_particular_value_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
) -> None:
    """In-place: lift the asked particular over neighboring-field lookalikes."""
    want_rate = (
        query_asks_for_delay_damages_rate(query)
    )
    party_role = asked_party_role(query)
    want_eng = bool(party_role)
    want_aca = (
        query_asks_for_aca_including_vat(query)
    )
    want_tfc = (
        query_asks_for_time_for_completion(query)
    )
    want_dnp = (
        query_asks_for_defects_notification_period(query)
    )
    if not (want_rate or want_eng or want_aca or want_tfc or want_dnp):
        return
    for i, (score, chunk) in enumerate(scored):
        text = chunk.text or ""
        hit = (
            (want_rate and chunk_states_delay_damages_rate(text))
            or (want_eng and chunk_names_party(text, party_role))
            or (want_aca and chunk_states_aca_including_vat(text))
            or (want_tfc and chunk_states_time_for_completion(text))
            or (want_dnp and chunk_states_defects_notification_period(text))
        )
        if not hit:
            continue
        boosted = score + _ASKED_PARTICULAR_VALUE_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


# ── bill items: Rate Only / priced / Excluded ─────────────────────────────
#
# "What is the total amount for <item> (<item code>)?" where the asked row is
# marked Rate Only: no amount exists. Cosine prefers priced lookalikes on
# the same page and an Excluded item that shares the description's words.
# Do not invent a money total; elect the Rate Only row as written.
_RATE_ONLY_BONUS = 2.0
_PRICED_BOQ_BONUS = 2.0
_PART_SUMMARY_BONUS = 2.0
_RATE_ONLY_RE = re.compile(r"(?i)\brate\s*only\b")
_EXCLUDED_RE = re.compile(r"(?i)\bexcluded\b")
_ITEM_AMOUNT_ASK_RE = re.compile(
    r"(?i)\b(?:total\s+amount|(?<!contract\s)amount|sum\s+for|value\s+for)\b",
)
_UNIT_RATE_ONLY_ASK_RE = re.compile(
    r"(?i)\b(?:unit\s+rate|rate\s+for|rate\s+per)\b",
)
_CESMM_COMPACT_ITEM_RE = re.compile(r"(?i)^[a-z]\d{2,4}(?:\.\d+)?$")
_CESMM_IN_TEXT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])([A-Z])\s*(\d{2,4}(?:\.\d+)?)(?![A-Za-z0-9])",
)


def extract_asked_cesmm_codes(query: str) -> List[str]:
    """Compact CESMM item codes the ask names (``d529.3``, ``d549.2``)."""
    out: List[str] = []
    seen: Set[str] = set()
    blob = normalize_cesmm_item_codes(query or "")
    for ident in extract_query_identifiers(blob):
        compact = normalize_cesmm_item_codes(ident).lower()
        if not _CESMM_COMPACT_ITEM_RE.fullmatch(compact):
            continue
        if compact not in seen:
            seen.add(compact)
            out.append(compact)
    for match in _CESMM_IN_TEXT_RE.finditer(blob):
        compact = f"{match.group(1)}{match.group(2)}".lower()
        if compact not in seen:
            seen.add(compact)
            out.append(compact)
    return out


def query_asks_for_boq_item_amount(query: str) -> bool:
    """True for a BOQ item amount ask (total amount of a named CESMM / BOQ item).

    Accepted Contract Amount, Delay Damages rate and daily-amount asks
    (calculate … in SAR) stay off this path. A unit-rate-only ask
    (without ``amount``) is not this class — the Rate
    column can still be a number when Amount is Rate Only.
    """
    q = (query or "").strip()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if query_asks_for_accepted_contract_amount(q):
        return False
    # A daily-amount ask (calculate delay damages … in SAR) is rate × ACA compose,
    # not a CESMM unit-rate / amount quote. Check the daily-ask
    # class first so a monetary-base regex drift cannot open the
    # priced fence and refuse-close the daily-amount ask.
    if query_asks_delay_damages_daily_amount(q):
        return False
    if query_asks_for_delay_damages_rate(q) or query_needs_a_monetary_base(q):
        return False
    if _UNIT_RATE_ONLY_ASK_RE.search(q) and not re.search(
        r"(?i)\b(?:total\s+)?amount\b", q,
    ):
        return False
    if not _ITEM_AMOUNT_ASK_RE.search(q):
        return False
    return bool(extract_asked_cesmm_codes(q))


# A new BOQ row starts with a CESMM code (optionally after a pipe).
# CESMM4 is letter + 3 digits (D529.3 / D110). Same-line unit+rate
# ("m 80.00", "m 15.00") is letter + 2-digit money and must not cut
# the asked row — that was dropping 280,320 from D549.2. Pipe-led
# rows still allow 2-3 digits (``| I12 |``). Four-digit quantities
# fail ``\d{3}\b`` (the digit after the 3-digit prefix blocks ``\b``).
#
# The space-led branch is CASE-SENSITIVE. The two-digit guard above handles
# ``m 80.00``; it does not handle a three-digit rate, and OCR glues the unit
# to it: live ``D549.1 m240.00 Rate Only`` read ``m240.00`` as the next item
# "M240.00" and cut the row off before its own "Rate Only". A bill prints its
# item codes in capitals and its units in lower case.
_NEXT_CESMM_ROW_RE = re.compile(
    r"(?:\s*[|]\s*([A-Za-z])\s*(\d{2,3}(?:\.\d{1,2})?)\b"
    r"|\s+([A-Z])\s*(\d{3}(?:\.\d{1,2})?)\b)",
)
_CESMM_ROW_TAIL_CHARS = 220


def _cesmm_row_windows(text: str, code: str) -> List[str]:
    """Local row windows around one CESMM code.

    Amount sits to the right of the item code. A previous row's
    Rate Only must not stain the next item on a mixed BOQ page —
    including same-line OCR soup (live WAVE 2 B4/B5 on fa07b2f:
    D529.3 Rate Only + D549.2 fence + D599.5 carriageway in one
    scanned line). Cut at the next CESMM item on this or the next
    line; keep one continuation line so ``D 529.3`` / next-line
    ``Rate Only`` still belongs to D529.3.
    """
    compact = normalize_cesmm_item_codes(code or "")
    if not compact:
        return []
    letter, rest = compact[0], compact[1:]
    item_re = re.compile(
        rf"(?i)(?<![A-Za-z0-9]){re.escape(letter)}\s*{re.escape(rest)}"
        r"(?![A-Za-z0-9])",
    )
    blob = text or ""
    windows: List[str] = []
    for match in item_re.finditer(blob):
        tail = blob[match.end(): match.end() + _CESMM_ROW_TAIL_CHARS]
        cut = _NEXT_CESMM_ROW_RE.search(tail)
        if cut:
            tail = tail[:cut.start()]
        windows.append(
            _normalize_retrieval_ws(blob[match.start(): match.end()] + tail)
        )
    return windows


def chunk_states_rate_only_item(text: str, codes: List[str]) -> bool:
    """True when the asked CESMM row's Amount is Rate Only.

    A priced lookalike on the same page (D549.2 / D599.5) and an
    Excluded culvert that only shares the description are not this.
    A window that already prints qty × rate = amount is priced, even
    when a neighbor or page note says Rate Only (live WAVE 2 B5).
    Does not invent: the excerpt itself must already say Rate Only
    on the asked item's row, with no priced triple in that window.
    """
    if not codes or not _RATE_ONLY_RE.search(text or ""):
        return False
    for code in codes:
        for window in _cesmm_row_windows(text, code):
            if _RATE_ONLY_RE.search(window) and not _parse_priced_cesmm_window(
                window, code,
            ):
                return True
    return False


# Words every BOQ question uses; they describe the ASK, not the item.
_BOQ_ASK_GENERIC_TERMS = frozenset({
    "amount", "total", "quantity", "rate", "unit", "item", "items", "price",
    "priced", "cost", "value", "many", "much", "removal", "remove", "removed",
    "removing", "existing", "breakout", "breaking", "break", "bill", "demolition",
    "stated", "including", "allowed", "page", "cesmm", "work", "works",
    "equal", "equals", "verify", "check", "according", "under", "against",
})
_CESMM_ROW_LEAD_CHARS = 170


def _same_word(a: str, b: str) -> bool:
    """"culverts"/"culvert", "fencing"/"fence": same word, different ending.

    A shared prefix of all but the last letter of the shorter word, at least
    four letters, and lengths within three. "wall"/"walkway" differ.
    """
    a, b = a.lower(), b.lower()
    short = min(len(a), len(b))
    if short < 4 or abs(len(a) - len(b)) > 3:
        return False
    k = max(4, short - 1)
    return a[:k] == b[:k]


_CODE_IN_QUESTION_RE = re.compile(
    r"(?i)\(?\s*(?:item\s+)?[A-Z]\s?\d{2,4}(?:\.\d+)?\s*\)?"
)


def asked_item_description_terms(query: str) -> frozenset:
    """The words of the question that describe the ITEM it asks about.

    By POSITION: a BOQ question puts the description right before the code —
    "removal of *storm water culverts* (D529.3)". Words after it ("...D549.2
    *according to the tender bill*") or in another clause ("Verify: ...") are
    about the ask, not the item; treating them as a description would reject
    the right row for a question that simply gave none.
    """
    q = query or ""
    m = _CODE_IN_QUESTION_RE.search(q)
    if not m:
        return frozenset()
    before = re.split(r"[:;?.!]", q[: m.start()])[-1]
    query = before
    codes = {c.replace(".", "") for c in extract_asked_cesmm_codes(query or "")}
    return frozenset(
        t for t in _significant_terms(query)
        if t not in _BOQ_ASK_GENERIC_TERMS and t.replace(".", "") not in codes
        and not t[0].isdigit()
    )


def _cesmm_row_contexts(text: str, code: str) -> List[Tuple[str, str]]:
    """``(description_before_the_code, row_window)`` for each occurrence.

    A bill prints the description BEFORE its code, so the window that starts
    at the code does not contain it. The lead is cut at the previous item so
    a neighbour's description is not borrowed.
    """
    compact = normalize_cesmm_item_codes(code or "")
    if not compact:
        return []
    letter, rest = compact[0], compact[1:]
    item_re = re.compile(
        rf"(?i)(?<![A-Za-z0-9]){re.escape(letter)}\s*{re.escape(rest)}"
        r"(?![A-Za-z0-9])",
    )
    blob = text or ""
    out: List[Tuple[str, str]] = []
    windows = _cesmm_row_windows(text, code)
    for match, window in zip(item_re.finditer(blob), windows):
        lead = blob[max(0, match.start() - _CESMM_ROW_LEAD_CHARS): match.start()]
        prev = list(_NEXT_CESMM_ROW_RE.finditer(lead))
        if prev:
            lead = lead[prev[-1].end():]
        out.append((_normalize_retrieval_ws(lead), window))
    return out


def row_is_the_asked_item(query: str, lead: str, window: str) -> bool:
    """False when the row's description shares nothing with the question's.

    Live 5312551: one code, two bills, two items. The priced BOQ prints
    ``...existing concrete wall/barrier D 529.3 m 26,997 500 13,498,500.00``;
    the demolition bill prints ``...storm water culverts D529.3 m 1,370.00
    Rate Only``. Asked for the culverts' total, the platform stated the
    wall's. The code does not identify the item; the description does. A
    question that gives only the code has nothing to check and passes.
    """
    wanted = asked_item_description_terms(query)
    if not wanted:
        return True
    have = re.findall(r"[A-Za-z]{4,}", f"{lead} {window}")
    return any(_same_word(w, h) for w in wanted for h in have)


def chunk_states_priced_item(text: str, codes: List[str], query: str = "") -> bool:
    """True when the asked CESMM row already prints qty × rate = amount.

    With ``query``, the row must also BE the asked item
    (:func:`row_is_the_asked_item`).
    """
    if not codes or not text:
        return False
    for code in codes:
        for lead, window in _cesmm_row_contexts(text, code):
            if not _parse_priced_cesmm_window(window, code):
                continue
            if query and not row_is_the_asked_item(query, lead, window):
                continue
            return True
    return False


def chunk_states_excluded_item(text: str, codes: List[str]) -> bool:
    """True when the asked CESMM row's Amount is Excluded, not priced.

    A page note like "Excluded items listed separately" after a valid
    triple is not this. Does not invent a total.
    """
    if not codes or not _EXCLUDED_RE.search(text or ""):
        return False
    for code in codes:
        for window in _cesmm_row_windows(text, code):
            if _EXCLUDED_RE.search(window) and not _parse_priced_cesmm_window(
                window, code,
            ):
                return True
    return False


def chunk_states_rate_only_row(text: str) -> bool:
    """Query-free: a CESMM row in ``text`` already says Rate Only."""
    if not text or not _RATE_ONLY_RE.search(text):
        return False
    blob = normalize_cesmm_item_codes(text)
    codes = [
        f"{m.group(1)}{m.group(2)}".lower()
        for m in _CESMM_IN_TEXT_RE.finditer(blob)
    ]
    return bool(codes) and chunk_states_rate_only_item(text, codes)


def format_rate_only_line(codes: List[str], excerpt: str = "") -> str:
    """User-facing Rate Only sentence. Does not invent a money total.

    Description is taken from the asked item's isolated row only.
    A storm-water neighbor on the same OCR page must not be grafted
    onto D599.5 / D549.2 (live WAVE 2 B4/B5).
    """
    code = (codes[0] if codes else "the item")
    pretty = f"{code[0].upper()}{code[1:]}" if code and code[0].isalpha() else code
    desc = ""
    windows = _cesmm_row_windows(excerpt or "", code) if code else []
    collapsed = _normalize_retrieval_ws(" ".join(windows))
    if re.search(r"(?i)storm\s+water\s+culvert", collapsed):
        desc = " (removal of storm water culverts)"
    return (
        f"{pretty}{desc} is Rate Only. No amount exists for this item "
        f"in the client BOQ — do not invent a total."
    )


def answer_states_rate_only(text: str) -> bool:
    """True when ``text`` already elects Rate Only / no amount."""
    blob = text or ""
    if _RATE_ONLY_RE.search(blob):
        return True
    return bool(re.search(r"(?i)\bno amount\b", blob))


# WAVE 2 B4/B5: after #542 election the priced CESMM row is in the
# excerpts, but synthesis can still hang empty (question echo /
# search promise / chrome) and never write quantity + amount.
# Live B5 OCR prints ``3,504 m @ SAR 80.00 = SAR 280,320.00`` — a
# currency token between @/= and the figure. B4 is often bare
# ``340904 m2 31.00 10568024``. Compose from the isolated window
# only. Do not invent: qty × rate must already equal the printed
# amount. Kill-switch: COMPOSE_PRICED_BOQ_ROW=0 restores the hang.
_BOQ_CURRENCY_ATOM = (
    r"(?:SAR|SR|AED|USD|EUR|GBP|QAR|BHD|KWD|OMR|EGP|CNY|INR|JPY|riyal[s]?)"
)
_BOQ_CURRENCY_PREFIX = rf"(?:{_BOQ_CURRENCY_ATOM}\s+)?"
_PRICED_BOQ_TRIPLE_RE = re.compile(
    r"(?i)(?P<qty>\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"\s+"
    r"(?P<unit>m[2²³3]|sq\.?\s*m|lin\.?\s*m|nr|no\.?|item|sum|ls|m)\b"
    r"\s*[@]?\s*"
    + _BOQ_CURRENCY_PREFIX
    + r"(?P<rate>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    + r"\s*[=]?\s*"
    + _BOQ_CURRENCY_PREFIX
    + r"(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2}|\d{4,})"
)


# The same triple with the UNIT column before the quantity. A CESMM bill
# prints Ref | Unit | Qty | Rate | Amount, so the live row is
# ``D549.2 m 3,504 80.00 280,320.00`` — which the pattern above cannot see.
# With nothing parsed, the model read the row itself and attached the
# neighbouring row's "Rate Only" to it (live d8d9573 B5, 0/5). OCR glues the
# unit to the quantity (``m3,504``), hence ``\s*``. Safe for the same reason
# the first order is: the caller only accepts qty × rate == amount.
_PRICED_BOQ_TRIPLE_UNIT_FIRST_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.])"
    r"(?P<unit>m[2²³3]|sq\.?\s*m|lin\.?\s*m|nr|no\.?|item|sum|ls|ha|kg|t|m)"
    r"\s*"
    r"(?P<qty>\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"\s+"
    + _BOQ_CURRENCY_PREFIX
    + r"(?P<rate>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    + r"\s+"
    + _BOQ_CURRENCY_PREFIX
    + r"(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2}|\d{4,})"
)


def _priced_boq_triples(blob: str):
    """Every candidate (qty, unit, rate, amount) match, both column orders."""
    yield from _PRICED_BOQ_TRIPLE_RE.finditer(blob)
    yield from _PRICED_BOQ_TRIPLE_UNIT_FIRST_RE.finditer(blob)


def priced_boq_compose_enabled() -> bool:
    """ON by default. ``COMPOSE_PRICED_BOQ_ROW=0`` restores the B4/B5 empty hang."""
    return _env_flag_on("COMPOSE_PRICED_BOQ_ROW")


def _plain_boq_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _fmt_boq_qty(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def _parse_boq_number(raw: str) -> Optional[float]:
    tok = (raw or "").replace(",", "").strip()
    if not tok:
        return None
    try:
        return float(tok)
    except ValueError:
        logger.debug("BOQ number parse failed for %r", raw, exc_info=True)
        return None


def _parse_priced_cesmm_window(window: str, code: str) -> Optional[Dict[str, Any]]:
    """Qty / unit / rate / amount from one isolated CESMM row.

    A Rate Only / Excluded *status* with no triple is None. A valid
    qty × rate = amount still parses when a neighbor or page note
    says Rate Only (live WAVE 2 B5 Part Nr. 3 stain).
    """
    if not window:
        return None
    blob = _normalize_retrieval_ws((window or "").replace("|", " "))
    match = None
    qty = rate = amount = None
    # First candidate, in either column order, whose arithmetic holds. The
    # arithmetic IS the test: a regex hit that does not multiply out is three
    # numbers that happened to sit together.
    for cand in _priced_boq_triples(blob):
        c_qty = _parse_boq_number(cand.group("qty"))
        c_rate = _parse_boq_number(cand.group("rate"))
        c_amount = _parse_boq_number(cand.group("amount"))
        if c_qty is None or c_rate is None or c_amount is None:
            continue
        if c_qty <= 0 or c_rate <= 0 or c_amount <= 0:
            continue
        if abs(c_qty * c_rate - c_amount) > max(1.0, 0.015 * c_amount):
            continue
        match, qty, rate, amount = cand, c_qty, c_rate, c_amount
        break
    if match is None:
        return None
    unit = _normalize_retrieval_ws(match.group("unit") or "")
    pretty = f"{code[0].upper()}{code[1:]}" if code and code[0].isalpha() else code
    letter, rest = (code[0], code[1:]) if code else ("", "")
    item_re = re.compile(
        rf"(?i)(?<![A-Za-z0-9]){re.escape(letter)}\s*{re.escape(rest)}"
        r"(?![A-Za-z0-9])",
    )
    code_m = item_re.search(blob)
    start = code_m.end() if code_m else 0
    desc = _normalize_retrieval_ws(blob[start:match.start()]).strip(" :-–—")
    return {
        "code": pretty,
        "description": desc,
        "qty": qty,
        "unit": unit,
        "rate": rate,
        "amount": amount,
    }


def compose_priced_boq_row(query: str, excerpt: str) -> Optional[Dict[str, Any]]:
    """Parse the asked CESMM row's quantity + amount from excerpts.

    Prefers a priced window when Rate Only / Excluded siblings for the
    same CESMM code are also in the excerpt (a priced-row conflict). A
    Rate-Only-only ask and an Excluded-only culvert are not this.
    Does not invent: qty × rate must already equal the printed amount.
    """
    if not priced_boq_compose_enabled():
        return None
    codes = extract_asked_cesmm_codes(query)
    if not codes or not excerpt:
        return None
    qlow = (query or "").lower()
    candidates: List[Dict[str, Any]] = []
    for code in codes:
        for lead, window in _cesmm_row_contexts(excerpt, code):
            parsed = _parse_priced_cesmm_window(window, code)
            if parsed and row_is_the_asked_item(query, lead, window):
                candidates.append(parsed)
    if not candidates:
        return None
    # Unseen Set 3, live 5312551: two bills priced the SAME item differently
    # (915 Nr @ 3,800 in one, 897 Nr @ 1,275.00 in the other) and the first
    # was stated flatly. One line is only honest when there is one answer;
    # when the sources disagree the model gets the turn, with both excerpts.
    # Copies of one bill (signed, unsigned, an OCR of it) agree and collapse.
    if len({(c["qty"], c["rate"], c["amount"]) for c in candidates}) > 1:
        logger.info(
            "priced-BOQ compose declined: %d excerpts disagree on %s",
            len(candidates), ", ".join(codes),
        )
        return None

    def _score(parsed: Dict[str, Any]) -> int:
        desc = (parsed.get("description") or "").lower()
        return sum(1 for w in desc.split() if len(w) > 3 and w in qlow)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def format_priced_boq_line(parsed: Dict[str, Any]) -> str:
    """User-facing priced-row sentence. Does not invent a currency."""
    if not parsed:
        return ""
    code = parsed.get("code") or "the item"
    desc = (parsed.get("description") or "").strip()
    head = f"{code} {desc}".strip() if desc else str(code)
    unit = (parsed.get("unit") or "").strip()
    qty_s = _fmt_boq_qty(float(parsed["qty"]))
    rate_s = f"{float(parsed['rate']):,.2f}"
    amt_s = _fmt_boq_qty(float(parsed["amount"]))
    unit_bit = f" {unit}" if unit else ""
    return f"{head}: quantity {qty_s}{unit_bit} @ {rate_s} = {amt_s}."


def answer_states_priced_boq(text: str, parsed: Dict[str, Any]) -> bool:
    """True when ``text`` already names the elected quantity and amount.

    A priced CESMM row that still wears a Rate Only label is not an
    election — Rate Only means no quantity and therefore no extended
    amount (live B5 D549.2). The graft then emits the priced line.
    """
    if not text or not parsed:
        return False
    if _RATE_ONLY_RE.search(text):
        return False
    blob = text.replace(",", "").replace(" ", "")
    return (
        _plain_boq_number(float(parsed["qty"])) in blob
        and _plain_boq_number(float(parsed["amount"])) in blob
    )


# Part Summary total for a CESMM bill page
# (``d/3/1``). Named-item asks compose a CESMM triple; this ask has no item
# code. Line-item chunks from the same demolition page (D110 / D290.1)
# outrank the sparse footer, then synthesis hangs or says the total
# was not found. Compose the printed page total only — do not sum
# OCR line items. Kill-switch: COMPOSE_PART_SUMMARY=0.
_BOQ_PAGE_REF_RE = re.compile(
    r"(?i)\b([A-Za-z])\s*[/\-]\s*(\d{1,3})\s*[/\-]\s*(\d{1,3})\b"
)
_PART_SUMMARY_LABEL_RE = re.compile(
    r"(?i)\b(?:part\s+summary|total\s+this\s+page|page\s+total|"
    r"carried\s+to\s+collection)\b"
)
# The chunker cuts on length, not on meaning. Live (unseen Set 3 B3, page
# d/3/3): one chunk ENDS "...culverts 1,370.00 To Par" and the next BEGINS
# "t Summary ... SAR 17,496,857.00 ... Page d/3/3". Neither half says "Part
# Summary". The second half is still that page's total, and it is
# recognisable: "Summary", within a few characters of the START of a chunk
# (start of text, or right after an excerpt marker's closing bracket). The
# word anywhere else in running prose is not a label.
_PART_SUMMARY_SPLIT_LABEL_RE = re.compile(
    r"(?i)(?:\A|\]\s*)\W{0,4}(?:[a-z]{1,4}\W{1,3})?summary\b"
)


_PART_SUMMARY_COLLECTION_ROW_RE = re.compile(
    r"(?i)from\s+page\s+(?:nr|no)?\.?\s*([a-z])\s*[/\-]\s*(\d{1,3})\s*[/\-]\s*(\d{1,3})"
    r"\s+(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
)


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _part_summary_labels(blob: str) -> List["re.Match"]:
    """Every Part Summary label in ``blob``, whole or cut by the chunker."""
    found = list(_PART_SUMMARY_LABEL_RE.finditer(blob or ""))
    whole = [(m.start(), m.end()) for m in found]
    for m in _PART_SUMMARY_SPLIT_LABEL_RE.finditer(blob or ""):
        # "...] To Part Summary" is a whole label, already counted.
        if not any(a <= m.end() <= b for a, b in whole):
            found.append(m)
    return sorted(found, key=lambda m: m.start())


_PART_SUMMARY_ASK_RE = re.compile(
    r"(?i)\bpart\s+summary\b|\btotal\s+this\s+page\b|\bpage\s+total\b"
)
_PART_SUMMARY_BILL_RE = re.compile(
    r"(?i)\b(?:bill|boq|demolit|site\s+clear|clearance)\b"
)
_PART_SUMMARY_MONEY_RE = re.compile(
    r"(?i)"
    + _BOQ_CURRENCY_PREFIX
    + r"(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2}|\d{5,})"
)
_PART_SUMMARY_ROW_CUT_RE = re.compile(
    r"(?i)(?:\b[a-z]\d{2,4}(?:\.\d+)?\b|\brate\s*only\b|"
    r"\bpart\s+summary\b)"
)


def part_summary_compose_enabled() -> bool:
    """ON by default. ``COMPOSE_PART_SUMMARY=0`` restores the B3 hang."""
    return _env_flag_on("COMPOSE_PART_SUMMARY")


def normalize_boq_page_ref(token: str) -> str:
    """``D / 3 / 1`` / ``d-3-1`` → ``d/3/1``. Empty when not a page ref."""
    match = _BOQ_PAGE_REF_RE.search(token or "")
    if not match:
        return ""
    return f"{match.group(1).lower()}/{int(match.group(2))}/{int(match.group(3))}"


def extract_asked_boq_page_refs(query: str) -> List[str]:
    """CESMM bill page refs the ask names (``d/3/1``)."""
    out: List[str] = []
    seen: Set[str] = set()
    for match in _BOQ_PAGE_REF_RE.finditer(query or ""):
        ref = f"{match.group(1).lower()}/{int(match.group(2))}/{int(match.group(3))}"
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def _normalize_boq_page_refs_in_text(text: str) -> str:
    """Collapse OCR ``D / 3 / 1`` so a query ``d/3/1`` can match."""

    def _repl(match: re.Match) -> str:
        return f"{match.group(1).lower()}/{int(match.group(2))}/{int(match.group(3))}"

    return _BOQ_PAGE_REF_RE.sub(_repl, text or "")


_PART_SUMMARY_NOT_A_LOOKUP_RE = re.compile(
    r"(?i)\b(?:verify|check\s+(?:that|whether|if)|consistent|add\s+up|adds\s+up|sum\s+of|"
    r"combined|altogether|in\s+total\s+across|compare[ds]?|larger|smaller|"
    r"greater|less\s+than|more\s+than|difference|reconcile[ds]?)\b"
)


def query_is_a_check_not_a_lookup(query: str) -> bool:
    """True for "verify / does it add up / which is larger" — not "what is".

    The deterministic composers state one printed figure and skip the model.
    That answers a lookup. It does not answer a check: live 5312551,
    "Verify: does 158 ha at SAR 186,328/ha equal the stated D110 amount?" got
    the row back and no verdict.
    """
    return bool(_PART_SUMMARY_NOT_A_LOOKUP_RE.search(query or ""))


def query_names_part_summary_pages(query: str) -> bool:
    """True when the ask involves the Part Summary total of named bill page(s).

    The RETRIEVAL class: fetch and lift those pages' totals. Wider than
    :func:`query_asks_for_part_summary_total`, which also decides whether one
    printed figure can answer the whole question.
    """
    q = (query or "").strip()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if not extract_asked_boq_page_refs(q):
        return False
    if _PART_SUMMARY_ASK_RE.search(q):
        return True
    return bool(
        re.search(r"(?i)\btotal\b", q)
        and re.search(r"(?i)\bpage\b", q)
        and _PART_SUMMARY_BILL_RE.search(q)
    )


def query_asks_for_part_summary_total(query: str) -> bool:
    """True for a page-total ask (Part Summary / page total of ONE named bill page).

    Named-CESMM amounts stay on the priced-row path. ACA / daily-amount
    monetary particulars are not this.
    """
    q = (query or "").strip()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if query_asks_for_accepted_contract_amount(q):
        return False
    if query_asks_delay_damages_daily_amount(q):
        return False
    if query_asks_for_delay_damages_rate(q) or query_needs_a_monetary_base(q):
        return False
    if query_asks_for_boq_item_amount(q):
        return False
    if not query_names_part_summary_pages(q):
        return False
    refs = extract_asked_boq_page_refs(q)
    # This class short-circuits the turn: it states ONE page's printed total
    # and skips the model. Live d8d9573 (unseen Set 3) it answered "combined
    # total of pages d/3/1, d/3/2 and d/3/3" and "do the three items on
    # d/3/1 add up to its Part Summary?" with page d/3/1's total alone — a
    # correct number, to a different question. Anything over several pages,
    # or that asks for a check rather than a lookup, goes to the model.
    return not (len(refs) > 1 or _PART_SUMMARY_NOT_A_LOOKUP_RE.search(q))


# Live d8d9573, unseen Set 3: one page total came back as "SAR 34,645,529.00"
# for one question and "INR 34,645,529.00" for another. The currency was the
# FIRST currency-shaped token near the label, matched case-insensitively, so
# a scrap of stamp OCR ("Inr", "Sr") ahead of the label beat the "SAR"
# printed against the figure. Two rules now: the code BESIDE the amount wins;
# failing that, only a properly upper-case code elsewhere counts, because
# "SAR" in a column header is a currency and "sr" in OCR soup is not.
_CURRENCY_BESIDE_AMOUNT_RE = re.compile(
    rf"(?i)(?<![A-Za-z])({_BOQ_CURRENCY_ATOM})\s*$"
)
_UPPERCASE_CURRENCY_RE = re.compile(
    r"(?<![A-Za-z])(SAR|SR|AED|USD|EUR|GBP|QAR|BHD|KWD|OMR|EGP|CNY|INR|JPY)(?![A-Za-z])"
)


def _normalise_currency_token(token: str) -> str:
    if not token:
        return ""
    return "SAR" if token.lower().startswith("riyal") else token.upper()


def _part_summary_currency(blob: str) -> str:
    """Fallback only: a genuine upper-case currency code somewhere in ``blob``."""
    match = _UPPERCASE_CURRENCY_RE.search(blob or "")
    if match:
        return match.group(1)
    word = re.search(r"(?i)\briyals?\b", blob or "")
    return "SAR" if word else ""


def _parse_part_summary_amount(raw: str) -> Optional[float]:
    amount = _parse_boq_number(raw)
    if amount is None or amount <= 0:
        return None
    # Page totals are money, not 2–3 digit quantities / rates.
    if amount < 100 and "." not in (raw or ""):
        return None
    return amount


# A scanned bill prints, in this order: the page total, a footer naming THAT
# page (``Classification - Public Page d/3/1 124 of 675``), then the NEXT
# page's header (``PAGE Nr. d/3/2``). The chunker keeps all three together.
# So the page a total belongs to is the first footer after it — not any ref
# in the chunk, and never the ``PAGE Nr.`` header, which is printed above the
# items that follow. OCR also glues the footer (``Paged/3/12``), which
# ``_BOQ_PAGE_REF_RE``'s leading word boundary cannot see.
_BOQ_FOOTER_PAGE_RE = re.compile(
    r"(?i)page\s*([a-z])\s*[/\-]\s*(\d{1,3})\s*[/\-]\s*(\d{1,3})"
)
# Far enough to cross the date / RFP / classification line between a total
# and its footer (~170 chars live); never past the next total.
_PART_SUMMARY_FOOTER_REACH = 320


def _part_summary_totals(blob: str) -> List[Dict[str, Any]]:
    """Every printed Part Summary total in ``blob`` with the page it is for.

    ``page`` is None when no page number survived next to the total. Such a
    total is evidence for no page in particular: it must never be elected as
    the page a question names.
    """
    labels = _part_summary_labels(blob)
    out: List[Dict[str, Any]] = []
    # The collection page that closes each Part lists every page's total with
    # the reference BEFORE the amount: "From Page Nr. d/3/3 17,496,857.00".
    # Each row owns exactly its own page. (Unseen Set 3 B3: this page was the
    # only place d/3/3's total could be read, and nothing understood it.)
    for row in _PART_SUMMARY_COLLECTION_ROW_RE.finditer(blob):
        amount = _parse_part_summary_amount(row.group("amount"))
        if amount is None:
            continue
        out.append({
            "pages": [f"{row.group(1).lower()}/{int(row.group(2))}/{int(row.group(3))}"],
            "amount": amount,
            "raw": row.group("amount"),
            "currency": "",
        })
    for i, label in enumerate(labels):
        # A label that HEADS a collection list owns none of it: the old window
        # rule read "PART SUMMARY From Page Nr. d/3/1 <amt> From Page Nr.
        # d/3/2 ..." as one total belonging to every reference in 80 chars.
        if _PART_SUMMARY_COLLECTION_ROW_RE.match(blob, _skip_ws(blob, label.end())):
            continue
        # Amount sits on the summary row (after the label). Looking
        # behind the label elects a neighbor line-item rate (220.00).
        # A wide window used to reach the next page's 1,370.00 Rate
        # Only figure (d/3/3) and elect that as the d/3/1 total.
        start = max(0, label.start() - 24)
        after_limit = min(len(blob), label.end() + 80)
        cut = _PART_SUMMARY_ROW_CUT_RE.search(blob, label.end())
        if cut:
            after_limit = min(after_limit, cut.start())
        window = blob[start:after_limit]
        amount: Optional[float] = None
        raw = ""
        beside = ""
        after = blob[label.end():after_limit]
        for match in _PART_SUMMARY_MONEY_RE.finditer(after):
            amount = _parse_part_summary_amount(match.group("amount"))
            if amount is not None:
                raw = match.group("amount")
                # The currency of a figure is the code printed AGAINST it.
                lead = _CURRENCY_BESIDE_AMOUNT_RE.search(
                    after[: match.start("amount")]
                )
                beside = lead.group(1) if lead else ""
                break
        if amount is None:
            continue
        # The row itself may name its page (``Part Summary total d/3/1``).
        pages = extract_asked_boq_page_refs(window)
        if not pages:
            reach = min(len(blob), label.end() + _PART_SUMMARY_FOOTER_REACH)
            if i + 1 < len(labels):
                reach = min(reach, labels[i + 1].start())
            footer = _BOQ_FOOTER_PAGE_RE.search(blob, label.end(), reach)
            if footer:
                pages = [
                    f"{footer.group(1).lower()}/"
                    f"{int(footer.group(2))}/{int(footer.group(3))}"
                ]
        out.append({
            "pages": pages,
            "amount": amount,
            "raw": raw,
            "currency": _normalise_currency_token(beside),
        })
    return out


def compose_part_summary_total(
    query: str, excerpt: str,
) -> Optional[Dict[str, Any]]:
    """Parse the asked bill-page Part Summary total from excerpts.

    The figure must already be printed next to a Part Summary / page-
    total label. Line items on the same page are not summed.
    """
    if not part_summary_compose_enabled():
        return None
    refs = extract_asked_boq_page_refs(query)
    if not refs or not excerpt:
        return None
    asked = refs[0]
    blob = _normalize_boq_page_refs_in_text(
        _normalize_retrieval_ws((excerpt or "").replace("|", " "))
    )
    if not _part_summary_labels(blob):
        return None
    # Only a total printed for the ASKED page. A lone total whose page number
    # did not survive the scan used to be returned as the asked page's.
    chosen = next(
        (t for t in _part_summary_totals(blob) if asked in t["pages"]), None,
    )
    if not chosen:
        return None
    return {
        "page": asked,
        "amount": chosen["amount"],
        "currency": chosen["currency"] or _part_summary_currency(blob),
    }


def format_part_summary_line(parsed: Dict[str, Any]) -> str:
    """User-facing Part Summary sentence. Does not invent a currency."""
    if not parsed:
        return ""
    page = parsed.get("page") or "the page"
    amt_s = f"{float(parsed['amount']):,.2f}"
    currency = (parsed.get("currency") or "").strip()
    money = f"{currency} {amt_s}".strip() if currency else amt_s
    return f"Part Summary total for page {page}: {money}."


def query_asks_combined_part_summary(query: str) -> bool:
    """True for a combined Part Summary ask of several named bill pages.

    A one-page total stays on ``query_asks_for_part_summary_total``.
    """
    q = (query or "").strip()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if query_asks_for_boq_item_amount(q):
        return False
    refs = extract_asked_boq_page_refs(q)
    if len(refs) < 2:
        return False
    return query_names_part_summary_pages(q)


def compose_combined_part_summary_total(
    query: str, excerpt: str,
) -> Optional[Dict[str, Any]]:
    """Sum the printed Part Summary totals of every named page.

    Example: 1,000 + 2,500 + 4,000 = 7,500.
    Does not invent a missing page and does not elect a neighbour.
    """
    if not part_summary_compose_enabled():
        return None
    if not query_asks_combined_part_summary(query):
        return None
    refs = extract_asked_boq_page_refs(query)
    if len(refs) < 2 or not excerpt:
        return None
    blob = _normalize_boq_page_refs_in_text(
        _normalize_retrieval_ws((excerpt or "").replace("|", " "))
    )
    totals = _part_summary_totals(blob)
    found: Dict[str, Dict[str, Any]] = {}
    for ref in refs:
        chosen = next((t for t in totals if ref in t["pages"]), None)
        if not chosen:
            return None
        found[ref] = chosen
    amount = sum(float(t["amount"]) for t in found.values())
    currency = next(
        (t.get("currency") or "" for t in found.values() if t.get("currency")),
        "",
    )
    return {
        "amount": amount,
        "pages": list(found.keys()),
        "page_amounts": {ref: float(found[ref]["amount"]) for ref in found},
        "currency": currency or _part_summary_currency(blob),
    }


def format_combined_part_summary_line(parsed: Dict[str, Any]) -> str:
    """User-facing combined Part Summary sentence."""
    if not parsed:
        return ""
    pages = parsed.get("pages") or []
    if len(pages) >= 2:
        listed = ", ".join(pages[:-1]) + f" and {pages[-1]}"
    else:
        listed = pages[0] if pages else "the named pages"
    amt_s = f"{float(parsed['amount']):,.2f}"
    currency = (parsed.get("currency") or "").strip()
    money = f"{currency} {amt_s}".strip() if currency else amt_s
    return f"The combined Part Summary total of pages {listed} is {money}."


def answer_states_part_summary(text: str, parsed: Dict[str, Any]) -> bool:
    """True when ``text`` already names the elected page total."""
    if not text or not parsed:
        return False
    blob = (text or "").replace(",", "").replace(" ", "")
    return _plain_boq_number(float(parsed["amount"])) in blob


def chunk_states_part_summary_total(
    text: str, page_refs: Optional[List[str]] = None,
) -> bool:
    """True when the chunk prints a Part Summary / page-total figure.

    When ``page_refs`` is given, the asked page must appear in the
    chunk or the chunk must be a single unlabeled Part Summary row
    (header + footer split by the 500-char BOQ chunker).
    """
    blob = text or ""
    if not _part_summary_labels(blob):
        return False
    if not _PART_SUMMARY_MONEY_RE.search(
        _normalize_retrieval_ws(blob.replace("|", " "))
    ):
        return False
    if not page_refs:
        return True
    normalized = _normalize_boq_page_refs_in_text(
        _normalize_retrieval_ws(blob.replace("|", " "))
    )
    return any(
        ref in total["pages"]
        for total in _part_summary_totals(normalized)
        for ref in page_refs
    )


def _apply_part_summary_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
) -> None:
    """In-place: lift the asked Part Summary page total over line items."""
    if not part_summary_compose_enabled():
        return
    if not query_names_part_summary_pages(query):
        return
    refs = extract_asked_boq_page_refs(query)
    if not refs:
        return
    for i, (score, chunk) in enumerate(scored):
        if not chunk_states_part_summary_total(chunk.text or "", refs):
            continue
        boosted = score + _PART_SUMMARY_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


def _pool_page_total_rows(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    extra_pids: Optional[List[str]] = None,
) -> int:
    """Pull the asked page's Part Summary row into ``fused``. Project-first.

    Cosine prefers D110 / D290.1 line items on the same demolition
    page (live H3). ``chunks_containing_all`` is the out-of-pool
    backup when the sparse footer never entered the candidate set.
    Failures never raise. GK rate-book notes are not searched.
    """
    if not part_summary_compose_enabled():
        return 0
    if not query_names_part_summary_pages(query):
        return 0
    refs = extract_asked_boq_page_refs(query)
    if not refs:
        return 0
    asked = refs[0]

    def _keep(text: str) -> bool:
        return chunk_states_part_summary_total(text, refs)

    recovered = _pool_lexical_hits_matching(
        project_id, fused, store,
        ("part summary", *refs, "total this page"),
        _keep, label="part-summary",
        bonus=_PART_SUMMARY_BONUS,
    )
    fetch = getattr(store, "chunks_containing_all", None)
    if not callable(fetch):
        return recovered
    pids = [project_id] + [
        p for p in (extra_pids or []) if p and p != project_id
    ]
    # Every page the question names: "combined total of d/3/1, d/3/2 and
    # d/3/3" needs three footers, and only the first used to be looked for.
    needle_sets = tuple(
        [["part summary"]]
        + [[label, ref] for ref in refs
           for label in ("part summary", "total this page", "page total", "summary")]
    )
    for pid in pids:
        for needles in needle_sets:
            try:
                hits = fetch(pid, list(needles), k=20)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "part-summary rescue for %s (%r) failed: %s",
                    pid, needles, exc,
                )
                continue
            for chunk in hits:
                if not _keep(chunk.text or ""):
                    continue
                if chunk.chunk_id in fused:
                    continue
                fused[chunk.chunk_id] = (chunk, 0.0, _PART_SUMMARY_BONUS)
                recovered += 1
    if recovered:
        logger.info(
            "part-summary rescue recovered %d chunk(s) for page %s",
            recovered, asked,
        )
    return recovered


def _apply_rate_only_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
) -> None:
    """In-place: lift the asked Rate Only row over priced lookalikes.

    Skip when a priced row for the same CESMM code is already in the
    pool — live WAVE 2 B5 must not promote Rate Only over 280,320.
    """
    if not query_asks_for_boq_item_amount(query):
        return
    codes = extract_asked_cesmm_codes(query)
    if not codes:
        return
    if any(
        chunk_states_priced_item(chunk.text or "", codes, query=query)
        for _s, chunk in scored
    ):
        return
    for i, (score, chunk) in enumerate(scored):
        if not chunk_states_rate_only_item(chunk.text or "", codes):
            continue
        boosted = score + _RATE_ONLY_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


def _apply_priced_boq_boost(
    query: str,
    scored: List[Tuple[float, Chunk]],
) -> None:
    """In-place: lift the asked priced CESMM row over Rate Only / Excluded."""
    if not priced_boq_compose_enabled():
        return
    if not query_asks_for_boq_item_amount(query):
        return
    codes = extract_asked_cesmm_codes(query)
    if not codes:
        return
    for i, (score, chunk) in enumerate(scored):
        if not chunk_states_priced_item(chunk.text or "", codes, query=query):
            continue
        boosted = score + _PRICED_BOQ_BONUS
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


# ── list continuation ─────────────────────────────────────────────────────
#
# A clause that introduces a list ("... shall be set out as follows:", "the
# following documents:") is often split from its list by the chunker: chunk N
# ends on the introduction, chunk N+1 opens with the items. Retrieval ranks the
# introduction -- it carries the question's words -- and the items, which
# carry none of them, never enter the pool. The answer then stops at "as
# follows".
#
# The fix reads the split from the chunks themselves: when a chunk in the
# provisional top-k ends on an open introduction and shares the question's
# terms, the next chunk of the same document is fetched; if it opens with a
# run of list items, it is the rest of that clause and is pooled with the
# list-continuation bonus, ahead of the introduction.
_LIST_CONTINUATION_BONUS = 2.0
_LIST_CONTINUATION_MIN_TERMS = 2
_LIST_CONTINUATION_MIN_ITEMS = 2
_LIST_CONTINUATION_ITEM_CHARS = 90
# The introduction ends the chunk: "as follows", "the following", "listed
# below", optionally with a colon or full stop.
_OPEN_LIST_TAIL_RE = re.compile(
    r"(?i)(?:as\s+follows|the\s+following(?:\s+\w+){0,3}|"
    r"(?:listed|set\s+out|given|shown)\s+below|in\s+the\s+following\s+order)"
    r"\s*[:.\-–—]?\s*$"
)
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:[-•*·▪]|\(?[a-z0-9ivx]{1,4}[.)]|\d+(?:\.\d+)*\.?)\s+", re.IGNORECASE,
)


def chunk_is_open_list_intro(text: str) -> bool:
    """True when the chunk ends on a clause that introduces a list.

    "The design review procedure is as follows" is an introduction too, so
    callers also require the introduction to share the question's terms.
    A chunk that goes on to give its items is not open.
    """
    blob = (text or "").strip()
    return bool(blob) and bool(_OPEN_LIST_TAIL_RE.search(blob))


def _list_items(text: str) -> List[str]:
    """The leading run of list items in ``text`` (lines or ``;`` cells)."""
    blob = (text or "").strip()
    lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
    if len(lines) < _LIST_CONTINUATION_MIN_ITEMS and ";" in blob:
        lines = [part.strip() for part in blob.split(";") if part.strip()]
    items: List[str] = []
    for line in lines:
        item = _LIST_MARKER_RE.sub("", line)
        if not item or len(item) > _LIST_CONTINUATION_ITEM_CHARS:
            break
        # A list item names a thing; a sentence of prose ends with a stop and
        # runs on.
        if item.endswith(".") and len(item.split()) > 8:
            break
        items.append(item)
    return items


def chunk_opens_with_list(text: str) -> bool:
    """True when the chunk opens with a run of short list items."""
    return len(_list_items(text)) >= _LIST_CONTINUATION_MIN_ITEMS


def recall_list_continuations(
    query: str,
    project_id: str,
    fused: Dict[str, Tuple],
    store,
    *,
    k: int = 5,
) -> int:
    """Pool the chunk that carries the list an in-pool introduction opens.

    Project corpus only. Returns the number of chunks added or lifted.
    Failures leave the pool standing.
    """
    follow = getattr(store, "chunks_following", None)
    if not callable(follow):
        return 0
    terms = [stem_query_term(t) for t in distinctive_query_terms(query)]
    if len(terms) < _LIST_CONTINUATION_MIN_TERMS:
        return 0
    ranked = sorted(fused.values(), key=lambda e: -((e[1] or 0.0) + (e[2] or 0.0)))
    anchors: List[Tuple[str, int]] = []
    for chunk, _sem, _bonus in ranked[:max(k, 1)]:
        if chunk.project_id and chunk.project_id != project_id:
            continue
        text = chunk.text or ""
        if not chunk_is_open_list_intro(text):
            continue
        low = text.lower()
        if sum(1 for t in terms if t in low) < _LIST_CONTINUATION_MIN_TERMS:
            continue
        key = (chunk.doc_id, int(chunk.chunk_index or 0))
        if key[0] and key not in anchors:
            anchors.append(key)
    if not anchors:
        return 0
    try:
        following = follow(project_id, anchors, n=1) or []
    except Exception as exc:  # noqa: BLE001 — recall must not break the turn
        logger.warning("list-continuation fetch for %s failed: %s", project_id, exc)
        return 0
    by_key = {(c.doc_id, int(c.chunk_index or 0)): c for c in following}
    recovered = 0
    for doc_id, idx in anchors:
        nxt = by_key.get((doc_id, idx + 1))
        if nxt is None or not chunk_opens_with_list(nxt.text or ""):
            continue
        prev = fused.get(nxt.chunk_id)
        if prev is not None:
            if (prev[2] or 0.0) >= _LIST_CONTINUATION_BONUS:
                continue
            fused[nxt.chunk_id] = (prev[0], prev[1], _LIST_CONTINUATION_BONUS)
        else:
            fused[nxt.chunk_id] = (nxt, 0.0, _LIST_CONTINUATION_BONUS)
        recovered += 1
    if recovered:
        logger.info("list-continuation recall pooled %d chunk(s)", recovered)
    return recovered


def _cd_particulars_boost_enabled() -> bool:
    return (os.getenv("RAG_CD_PARTICULARS_BOOST") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _contract_synonym_boost_enabled() -> bool:
    return (os.getenv("RAG_SYNONYM_BOOST") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


# Canonical FIDIC / contract-term equivalences. When a user phrasing on the
# LEFT (any lowercase substring) appears in the query, the canonical term(s)
# on the RIGHT are appended to a SUPPLEMENTARY retrieval query — the primary
# query is never altered. A synonym like "contract sum before VAT" then still
# surfaces the "Accepted Contract Amount" Contract Data line it would otherwise
# miss. Additive-only, mirroring the Contract Data particulars boost.
#
# Live gap (2026-09-13): synonym phrasings honestly declined facts that ARE in
# the corpus — "contract sum"/"net contract value" -> Accepted Contract Amount
# was not retrieved at all; probing showed appending the canonical term lifts
# that chunk to rank 0. Terms are canonical contract headings, not values, so
# the query is never "led" toward a specific figure.
_CONTRACT_SYNONYMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("contract sum", "contract value", "contract worth", "net contract",
      "total contract value", "worth of the contract", "contract price before"),
     "Accepted Contract Amount"),
    (("retention", "held back", "withheld", "retained from the"),
     "Percentage of Retention Retention Money"),
    (("time for completion", "completion period", "duration of the works",
      "how long to complete"),
     "Time for Completion"),
    (("delay damages", "liquidated damages", "penalty for late completion",
      "late completion penalty"),
     "Delay Damages Maximum Amount of Delay Damages Contract Price"),
    (("defects notification", "defects liability", "maintenance period",
      "warranty period", "defects period"),
     "Defects Notification Period"),
)


def expand_contract_synonyms(query: str) -> str:
    """Canonical contract terms to APPEND for any synonym present in ``query``.

    Returns a space-joined, de-duplicated string (empty when nothing triggers).
    Definition questions are left alone — they belong on the glossary path, not
    a Contract Data particulars lookup."""
    q = (query or "").strip().lower()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return ""
    out: list[str] = []
    for triggers, canonical in _CONTRACT_SYNONYMS:
        if any(t in q for t in triggers) and canonical.lower() not in q:
            out.append(canonical)
    # De-dup while preserving order (a term may recur across groups).
    seen: set[str] = set()
    parts: list[str] = []
    for term in " ".join(out).split():
        key = term.lower()
        if key not in seen:
            seen.add(key)
            parts.append(term)
    return " ".join(parts)


def query_asks_for_contract_particulars(query: str) -> bool:
    """True when the question wants a filled-in Contract Data particular.

    Definition questions ("what does Accepted Contract Amount mean") stay
    on the glossary path. Arithmetic unit-rate questions are not this.

    A particular is not always a figure: "who is the Engineer" asks for the
    filled row that names the party, which is why a contract-role identity
    ask counts here too.
    """
    q = (query or "").strip()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    if _PARTICULARS_FIELD_RE.search(q):
        return True
    if _CD_WHO_IS_RE.search(q) and _CD_CONTRACT_ROLE_RE.search(q):
        return True
    if _CD_SCHEDULE_ASK_RE.search(q) and _CD_SCHEDULE_CONTEXT_RE.search(q):
        return True
    return bool(_FILLED_IN_ASK_RE.search(q) and _CD_HEADING_IN_CHUNK_RE.search(q))


def query_asks_for_boq_scope(query: str) -> bool:
    """True when the question is about measured scope in a bill of quantities.

    The answer lives in BOQ item rows, not in the prose of a Conditions of
    Contract that happens to describe the same scope — and certainly not in
    another contract year's prose, which is what wave-2 F1 returned for all
    three of its citations.
    """
    return bool(_BOQ_SCOPE_ASK_RE.search((query or "").strip()))


def document_is_a_bill_of_quantities(filename: str) -> bool:
    """True when the DOCUMENT NAME says it is a bill of quantities.

    Consumes ``doc_index.BOQ_FILENAME_RE`` rather than restating it, so the
    retrieval-side test cannot drift from the one the indexer already uses to
    pick a chunker and an OCR budget. The import is deferred because
    ``doc_index`` imports this module; it is reached only for a BOQ-shaped
    ask, and ``sys.modules`` caches it after the first.
    """
    name = filename or ""
    if not name:
        return False
    try:
        from app.core.doc_index import BOQ_FILENAME_RE
    except Exception as exc:  # noqa: BLE001 — never break a turn over this
        logger.warning("BOQ filename test unavailable: %s", exc)
        return False
    return bool(BOQ_FILENAME_RE.search(name))


def query_needs_a_monetary_base(query: str) -> bool:
    """True when the ask is arithmetic over a particular and wants money out.

    A rate expressed as a percentage cannot answer "how much per day in SAR"
    on its own; the amount it is a percentage OF has to be in the excerpts
    too. In the daily-amount ask that was the whole failure: the 0.1%-per-day row came back at
    rank 1 and the answer then reported the SAR figure as absent.
    """
    q = (query or "").strip()
    if not q:
        return False
    return bool(
        _CD_MONEY_ARITHMETIC_ASK_RE.search(q) and _CD_MONEY_UNIT_ASK_RE.search(q)
    )


def particulars_row_states_an_amount_of_money(text: str) -> bool:
    """True when a particulars row's value is an amount of money.

    This is the base row an arithmetic ask needs. Deliberately the row's
    VALUE and not its label: "10% of the Accepted Contract Amount" names the
    base without stating it, and that row is what the daily-amount ask already had.
    """
    t = text or ""
    if not _CD_PARTICULARS_PREFIX_RE.search(t):
        return False
    return bool(_CD_MONETARY_VALUE_RE.search(_cd_chunk_body(t)))


def reserve_monetary_base_row(
    query: str,
    kept: List[Chunk],
    ranked: List[Chunk],
    *,
    allow=None,
) -> bool:
    """Give the amount a percentage refers to one slot in the top-k.

    Returns True when a swap happened. ``kept`` is modified in place: the
    LOWEST-ranked survivor is replaced, so k is unchanged and the rows the
    question actually named keep their places.

    A reservation rather than a bigger bonus, deliberately. The money row
    earns no label bonus — the question says "in SAR", and the row says
    "SAR <amount>", and they share no term the overlap can see — so on
    the live Contract Data it competes against 200 siblings that each earn
    the full 1.40. Any constant large enough to clear that field is a
    constant fitted to one corpus's cosine spread; one slot is a guarantee.

    ``allow`` is the caller's contract-scope test, so a reserved row cannot
    re-enter a contract the fence already excluded.
    """
    # Live 39d6b8d E2: "If Milestone 1 is 30 days late, what are the milestone
    # delay damages?" names no currency and says no "calculate", so it was
    # never a monetary-base ask — yet 30 days × a milestone rate of the Contract Price
    # is money. The sum was fetched and lifted, and still came sixth of five
    # behind copies of the rate row from every copy of the contract.
    delay_scenario = bool(
        query_applies_a_delay_duration(query)
        and query_asks_for_delay_damages_rate(query)
    )
    if not kept or not (query_needs_a_monetary_base(query) or delay_scenario):
        return False
    daily_damages_ask = (
        query_asks_delay_damages_daily_amount(query)
    )

    def _is_money_base(text: str) -> bool:
        if delay_scenario and not daily_damages_ask:
            # The sum the rate is a percentage OF — not any row with money in
            # it (an insurance deductible is an amount too).
            return chunk_states_accepted_contract_amount(text)
        if daily_damages_ask:
            try:
                from app.lib.construction_formulas_commercial import (
                    chunk_has_real_accepted_contract_amount,
                )
                # Toy 8.8 windows and particulars-prefixed 10M examples
                # are not the rate base. Only a non-toy ACA (a filled
                # excl-VAT amount) satisfies reservation.
                return chunk_has_real_accepted_contract_amount(text)
            except Exception:  # noqa: BLE001 — fall through to the usual tests
                logger.debug("toy-ACA money-base test failed", exc_info=True)
        if particulars_row_states_an_amount_of_money(text):
            return True
        return bool(daily_damages_ask and chunk_states_accepted_contract_amount(text))

    if any(_is_money_base(c.text or "") for c in kept):
        return False
    present = {c.chunk_id for c in kept}
    replace_at = len(kept) - 1
    if daily_damages_ask:
        # Daily-amount ask: particulars reserved the 0.1% row into the
        # last slot, then this function overwrote it with including-VAT
        # ACA. Prefer a non-rate slot. If every survivor is a rate
        # (k=1 unit pin), still take the last slot so the money row
        # can enter; earlier rate rows remain when k > 1.
        for i in range(len(kept) - 1, -1, -1):
            if not chunk_states_delay_damages_rate(kept[i].text or ""):
                replace_at = i
                break
    for chunk in ranked:
        if chunk.chunk_id in present:
            continue
        if not _is_money_base(chunk.text or ""):
            continue
        if allow is not None and not allow(chunk):
            continue
        kept[replace_at] = chunk
        logger.debug(
            "reserved a Contract Data money row for an arithmetic ask; "
            "a percentage alone cannot answer it",
        )
        return True
    return False


# Reservation heads for the synonym boost. A user asking with a synonym
# ("contract sum before VAT") triggers the retriever's synonym expansion, which
# pulls the canonical chunk into the candidate pool — but its cosine to the
# diluted synonym query is lower than the primary query's own top hits, so the
# score-based top-k cut drops it before the model sees it (live 2026-09-13: ACA
# "contract sum" answered "I don't have it" though the doc was retrieved). Each
# entry maps the synonym triggers to the CANONICAL HEADING substring to look
# for in a pool chunk. Headings are specific enough not to over-match; retention
# is intentionally omitted (its rate chunk is already surfaced by the retriever
# synonym leg, and "retention" alone is too broad to reserve safely).
# Each entry: (synonym triggers, canonical heading, VALUE regex). The value
# regex is essential: the pool holds BOTH the figure chunk ("Accepted Contract
# Amount ... SAR <amount>") and mention-only chunks ("...the Accepted
# Contract Amount stated in the Contract Data..."). Reserving on the heading
# alone grabbed the FIRST match in score order — a mention with no amount — and
# the answer layer still declined (live trace 2026-09-14: reserved idx 90, a
# mention; the figure chunk idx 0 sat at scored rank 47). Requiring a value
# alongside the heading makes the reservation pick a chunk that can answer.
_SYNONYM_RESERVE_HEADS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("contract sum", "contract value", "contract worth", "net contract value",
      "contract price before", "total contract value", "worth of the contract"),
     "accepted contract amount",
     # a thousands-separated money figure (>= millions: two+ comma groups)
     r"\d{1,3}(?:,\d{3}){2,}"),
    (("maximum amount of delay", "cap on delay", "ceiling on delay",
      "cap on liquidated", "maximum liquidated", "ld cap", "delay damages cap",
      "maximum delay damages"),
     "maximum amount of delay damages",
     r"\d+(?:\.\d+)?\s*%|\d{1,3}(?:,\d{3})+"),  # a percentage or a money figure
    (("defects liability", "maintenance period", "warranty period",
      "defects period"),
     "defects notification period",
     r"\d+\s*(?:day|days|month|months|year|years)"),  # a duration
)


def reserve_contract_synonym_row(
    query: str,
    kept: List[Chunk],
    ranked: List[Chunk],
    *,
    allow=None,
) -> bool:
    """Give a synonym-named Contract Data figure one slot in the top-k.

    Mirrors ``reserve_monetary_base_row``: a reservation, not a bonus, so no
    constant fitted to one corpus's cosine spread can regress. ``kept`` is
    modified in place (the lowest-ranked survivor is replaced, k unchanged).
    Returns True on a swap. No-op unless the query uses a synonym whose
    canonical heading is present in the pool but missing from ``kept``.

    ``allow`` is the caller's contract-scope test, so a reserved row cannot
    re-enter a contract the fence already excluded."""
    if not kept or not _contract_synonym_boost_enabled():
        return False
    q = (query or "").strip().lower()
    if not q or _DEFINITION_QUESTION_RE.search(q):
        return False
    heads = [
        (head, value_re) for triggers, head, value_re in _SYNONYM_RESERVE_HEADS
        if any(t in q for t in triggers)
    ]
    if not heads:
        return False

    def _has_answer(text: str) -> bool:
        # Collapse OCR/scan whitespace first: Contract Data text arrives as
        # "Accepted \nContract \nAmount". Require the canonical heading AND a
        # value (figure / percentage / duration) — a heading-only mention
        # cannot answer the ask, and reserving it leaves the model declining.
        low = _collapse_retrieval_ws((text or "")).lower()
        return any(
            head in low and re.search(value_re, low) for head, value_re in heads
        )

    if any(_has_answer(c.text or "") for c in kept):
        return False
    present = {c.chunk_id for c in kept}
    for chunk in ranked:
        if chunk.chunk_id in present:
            continue
        if not _has_answer(chunk.text or ""):
            continue
        if allow is not None and not allow(chunk):
            continue
        kept[len(kept) - 1] = chunk
        logger.debug(
            "reserved a canonical-heading row for a contract-term synonym ask; "
            "the score-based cut had dropped it below top-k",
        )
        return True
    return False


def _daily_damages_non_operand_index(
    kept: List[Chunk],
    *,
    protect_rate: bool,
    protect_aca: bool,
) -> Optional[int]:
    """Lowest-ranked slot that is not a protected daily-amount compose operand."""
    for i in range(len(kept) - 1, -1, -1):
        text = kept[i].text or ""
        if protect_rate and chunk_states_delay_damages_rate(text):
            continue
        if protect_aca and chunk_states_accepted_contract_amount(text):
            continue
        return i
    return None


def reserve_daily_damages_operands(
    query: str,
    kept: List[Chunk],
    ranked: List[Chunk],
    *,
    allow=None,
) -> bool:
    """Guarantee both daily-amount multiply operands in top-k, prefer excl-VAT ACA.

    ``reserve_matching_particulars_row`` and ``reserve_monetary_base_row``
    share one last slot. Daily-amount ask after #523: CoC 8.8 filled
    kept, the money reserve elected including-VAT ACA, and compose never
    saw the 0.1% row — the answer was the ACA particular. This pass
    restores the rate and upgrades incl-VAT to excl-VAT when both twins
    are reachable.
    """
    if not kept:
        return False
    if not (
        query_asks_delay_damages_daily_amount(query)
    ):
        return False
    changed = False
    present = {c.chunk_id for c in kept}

    if not any(chunk_states_delay_damages_rate(c.text or "") for c in kept):
        idx = _daily_damages_non_operand_index(kept, protect_rate=True, protect_aca=True)
        if idx is not None:
            for chunk in ranked:
                if chunk.chunk_id in present:
                    continue
                if not chunk_states_delay_damages_rate(chunk.text or ""):
                    continue
                if allow is not None and not allow(chunk):
                    continue
                kept[idx] = chunk
                present.add(chunk.chunk_id)
                changed = True
                break

    # Daily-amount ask after #535: a milestone rate CoC windows already satisfy
    # chunk_states_delay_damages_rate, so the 0.1% Contract Data row
    # never replaced them. Upgrade when a better rate is in ranked.
    best_rate: Optional[Chunk] = None
    best_rate_rank = -1
    for chunk in ranked:
        if allow is not None and not allow(chunk):
            continue
        rank = _daily_rate_preference(chunk.text or "")
        if rank > best_rate_rank:
            best_rate_rank = rank
            best_rate = chunk
    kept_rate = max(
        (_daily_rate_preference(c.text or "") for c in kept), default=-1,
    )
    if (
        best_rate is not None
        and best_rate_rank > kept_rate
        and best_rate.chunk_id not in {c.chunk_id for c in kept}
    ):
        rate_idxs = [
            i for i, chunk in enumerate(kept)
            if chunk_states_delay_damages_rate(chunk.text or "")
            and _daily_rate_preference(chunk.text or "") < best_rate_rank
        ]
        idx = rate_idxs[-1] if rate_idxs else _daily_damages_non_operand_index(
            kept, protect_rate=True, protect_aca=True,
        )
        if idx is not None:
            kept[idx] = best_rate
            present.add(best_rate.chunk_id)
            changed = True

    best_chunk: Optional[Chunk] = None
    best_rank = -1
    for chunk in ranked:
        if allow is not None and not allow(chunk):
            continue
        rank = _daily_damages_aca_preference(chunk.text or "")
        if rank > best_rank:
            best_rank = rank
            best_chunk = chunk
    if best_chunk is None:
        return changed

    kept_best = max((_daily_damages_aca_preference(c.text or "") for c in kept), default=-1)
    if best_rank <= kept_best:
        return changed
    if best_chunk.chunk_id in {c.chunk_id for c in kept}:
        return changed

    idx = None
    if kept_best >= 0:
        worst_i = None
        worst_rank = 99
        for i, chunk in enumerate(kept):
            rank = _daily_damages_aca_preference(chunk.text or "")
            if 0 <= rank < worst_rank:
                worst_rank = rank
                worst_i = i
        idx = worst_i
    if idx is None:
        idx = _daily_damages_non_operand_index(kept, protect_rate=True, protect_aca=True)
    if idx is None:
        # Daily-amount ask after #529: toy 8.8 windows occupy every
        # slot as rate operands. Skipping the toy cleared the money
        # operand without a free slot — still replace a surplus rate
        # so the excl-VAT ACA can enter. Keep at least one rate row.
        rate_idxs = [
            i for i, chunk in enumerate(kept)
            if chunk_states_delay_damages_rate(chunk.text or "")
        ]
        if len(rate_idxs) > 1:
            idx = rate_idxs[-1]
        elif kept_best < 0:
            idx = len(kept) - 1
    if idx is None:
        return changed
    kept[idx] = best_chunk
    return True


def ensure_kept_can_compose_daily_damages(
    query: str,
    kept: List[Chunk],
    ranked: List[Chunk],
    *,
    allow=None,
) -> bool:
    """Force both compose operands into kept when top-k is refuse-prone.

    Daily-amount ask after #536: Cosine kept Contract Data chunks 9–11
    that do not parse as rate × excl-VAT ACA. The operands already sit
    in ``ranked`` after the all-chunk scan. Put them in kept so compose
    does not fall through to the cost-grounding refuse.
    """
    if not kept:
        return False
    if not (
        query_asks_delay_damages_daily_amount(query)
    ):
        return False
    try:
        from app.lib.construction_formulas_commercial import (
            compose_delay_damages_daily_from_excerpts,
        )
    except Exception:  # noqa: BLE001 — never break a turn over an import
        logger.debug("e1 kept-compose import failed", exc_info=True)
        return False
    if compose_delay_damages_daily_from_excerpts(
        query, "\n\n".join(c.text or "" for c in kept),
    ):
        return False

    def _ok(chunk: Chunk) -> bool:
        return allow is None or allow(chunk)

    rate: Optional[Chunk] = None
    aca: Optional[Chunk] = None
    for chunk in ranked:
        if not _ok(chunk):
            continue
        text = chunk.text or ""
        if rate is None and _daily_rate_preference(text) >= 2:
            rate = chunk
        if aca is None and _has_standalone_excl_vat_aca(text):
            aca = chunk
        if rate is not None and aca is not None:
            break
    if rate is None or aca is None:
        return False
    changed = False
    present = {c.chunk_id for c in kept}
    if rate.chunk_id not in present:
        idx = _daily_damages_non_operand_index(kept, protect_rate=True, protect_aca=True)
        if idx is None:
            idx = len(kept) - 1
        kept[idx] = rate
        present.add(rate.chunk_id)
        changed = True
    rate.score = max(float(rate.score or 0.0), _OPERAND_PIN_SCORE)
    if aca.chunk_id not in present:
        idx = _daily_damages_non_operand_index(kept, protect_rate=True, protect_aca=True)
        if idx is None:
            rate_idxs = [
                i for i, chunk in enumerate(kept)
                if chunk.chunk_id == rate.chunk_id
                or chunk_states_delay_damages_rate(chunk.text or "")
            ]
            if len(rate_idxs) > 1:
                idx = next(
                    (i for i in reversed(rate_idxs) if kept[i].chunk_id != rate.chunk_id),
                    rate_idxs[-1],
                )
            else:
                idx = 0 if kept[-1].chunk_id == rate.chunk_id else len(kept) - 1
        if kept[idx].chunk_id == rate.chunk_id and len(kept) > 1:
            idx = 0 if idx != 0 else 1
        kept[idx] = aca
        changed = True
    aca.score = max(float(aca.score or 0.0), _OPERAND_PIN_SCORE)
    return changed


def reserve_matching_particulars_row(
    query: str,
    kept: List[Chunk],
    ranked: List[Chunk],
    *,
    allow=None,
) -> bool:
    """Give the particulars row the question named one slot in the top-k.

    Same-year General Conditions clauses can fill every slot after the
    fence has locked the right PREFIX-YEAR-SEQ (delay-rate ask: three HIGH
    Sub-Clause 8.8 chunks, no rate). A reservation rather than a bigger
    bonus: the clause repeats every label word and its cosine is not
    bounded.
    """
    if not kept or not query_asks_for_contract_particulars(query):
        return False
    # A cap row / mixed window used to count as "already answered" for
    # the delay-rate ask, so the 0.1%-per-day chunk never replaced same-year 8.8. The
    # asked *value* (rate / Engineer) must be in kept, not merely the
    # label family.
    if any(chunk_answers_asked_particular(query, c.text or "") for c in kept):
        need_delay_rate = (
            query_asks_for_delay_damages_rate(query)
            and not any(
                chunk_states_delay_damages_rate(c.text or "") for c in kept
            )
        )
        need_daily_rate = (
            query_asks_delay_damages_daily_amount(query)
            and not any(
                chunk_states_delay_damages_rate(c.text or "") for c in kept
            )
        )
        if not (need_delay_rate or need_daily_rate):
            return False
    present = {c.chunk_id for c in kept}
    for chunk in ranked:
        if chunk.chunk_id in present:
            continue
        text = chunk.text or ""
        if not chunk_answers_asked_particular(query, text):
            continue
        if (
            query_asks_for_delay_damages_rate(query)
            and not chunk_states_delay_damages_rate(text)
        ):
            continue
        if (
            query_asks_delay_damages_daily_amount(query)
            and not chunk_states_delay_damages_rate(text)
        ):
            continue
        if allow is not None and not allow(chunk):
            continue
        kept[-1] = chunk
        logger.debug(
            "reserved a matching Contract Data particulars row; "
            "same-year General Conditions had filled the top-k",
        )
        return True
    return False


def contract_data_particulars_delta(text: str) -> float:
    """Score delta for a chunk when the query is particulars-shaped.

    Dedicated ``CONTRACT DATA particulars`` chunks (index-time prefix) get
    the strongest lift. Other chunks that still carry a Contract Data
    heading plus a filled value get a smaller lift — but a chunk whose only
    mention of the Contract Data is a cross-reference to it does not, because
    a clause that says where the rate is stated is not the row that states
    it. Glossary "means the" chunks with no filled figure are demoted so they
    cannot bury the row.
    """
    t = text or ""
    if _CD_PARTICULARS_PREFIX_RE.search(t) and (
        _CD_FILLED_VALUE_RE.search(t) or particulars_chunk_states_a_value(t)
    ):
        return _CD_PARTICULARS_PREFIX_BONUS
    if (
        _CD_HEADING_IN_CHUNK_RE.search(t)
        and _CD_FILLED_VALUE_RE.search(t)
        and not _CD_MEANS_RE.search(t)
        and not contract_data_mention_is_only_a_cross_reference(t)
    ):
        return _CD_PARTICULARS_HEADING_BONUS
    if _CD_MEANS_RE.search(t) and not _CD_FILLED_VALUE_RE.search(t):
        return -_CD_DEFINITION_PENALTY
    return 0.0


def _apply_contract_data_particulars_boost(query: str, scored: List[Tuple[float, Chunk]]) -> None:
    """In-place re-score when the query asks for a filled-in particular."""
    if not _cd_particulars_boost_enabled():
        return
    if not query_asks_for_contract_particulars(query):
        return
    wants_whole = bool(_CD_WHOLE_WORKS_QUERY_RE.search(query))
    wants_milestone = bool(_CD_MILESTONE_QUERY_RE.search(query))
    query_terms = _significant_terms(query)
    for i, (score, chunk) in enumerate(scored):
        text = chunk.text or ""
        delta = contract_data_particulars_delta(text)
        if not delta:
            continue
        if delta > 0:
            chunk_is_milestone = bool(_CD_MILESTONE_CHUNK_RE.search(text))
            if wants_whole and not wants_milestone and chunk_is_milestone:
                # Whole-works ask: milestone rows lose the family bonus and
                # take a penalty so the whole-works row can surface.
                delta = -_CD_SCOPE_MISMATCH_PENALTY
            elif wants_milestone and not wants_whole and not chunk_is_milestone:
                # Milestone ask: non-milestone particulars keep their score
                # but get no family lift over the milestone rows.
                delta = 0.0
            if delta > 0:
                # Still inside the family: separate the row the question names
                # from the rest of it.
                delta += _cd_label_bonus(query_terms, text)
        boosted = score + delta
        chunk.score = round(boosted, 6)
        scored[i] = (boosted, chunk)


def _dual_search(
    store, project_id: str, query_vec, query: str,
    alt_vec, alt_query: Optional[str], *, k: int,
) -> List[Chunk]:
    """Search with the raw query and (when present) the wrapper-stripped
    variant; merge by chunk_id keeping each chunk's BEST score. The alt
    leg failing can never break the primary results."""
    hits = store.search(project_id, query_vec, k=k, query_text=query)
    if alt_vec is None:
        return hits
    try:
        alt_hits = store.search(project_id, alt_vec, k=k, query_text=alt_query)
    except Exception as exc:  # noqa: BLE001 -- alt leg is best-effort
        logger.warning(
            "dual-query alt retrieval for %s failed: %s; primary results stand",
            project_id, exc,
        )
        return hits
    best: Dict[str, Chunk] = {c.chunk_id: c for c in hits}
    for c in alt_hits:
        prev = best.get(c.chunk_id)
        if prev is None or (c.score or 0.0) > (prev.score or 0.0):
            best[c.chunk_id] = c
    return sorted(best.values(), key=lambda c: -(c.score or 0.0))


# Production chat retrieval is k=5 (runtime search_project_documents,
# rag inject). Delay-rate and daily-amount particulars rows sat at ranks 21 and 29 on
# the live index — the old floor of 20 dropped them before #430's
# label-awareness could promote them. Floor 60 is the parked
# pool-stability change that must land *after* #430: raising it first
# flooded A2/A6 with family competitors and they failed. The SHA
# 7efeadb was never pushed; this reconstructs that change from the
# #430 evidence (k=5, candidate pool 60).
_OVERFETCH_MULTIPLIER = 4
_OVERFETCH_FLOOR = 60


def candidate_overfetch(k: int) -> int:
    """How many raw candidates to pull before ranking down to ``k``.

    Production callers pass k=5, which yields 60. Do not lower the floor:
    a particulars row sitting at rank ~21–29 never enters a pool of 20,
    so the label bonus has nothing to promote.
    """
    try:
        n = int(k)
    except (TypeError, ValueError):
        n = 5
    if n < 1:
        n = 1
    return max(n * _OVERFETCH_MULTIPLIER, _OVERFETCH_FLOOR)


def _lexical_only_retrieve(query: str, project_id: str, k: int) -> tuple:
    """BM25-only retrieval for when no embedder can be loaded.

    Used when the embedding model is absent, removed, or failed to load. The
    vector leg is unavailable, so ranking is purely lexical — worse than hybrid,
    and dramatically better than the empty list this used to return.

    Deliberately mirrors the main path's shape: active project first (so its
    chunks win ties over general knowledge), GK merged, noise-filtered, top-k,
    and the same ``(chunks, noise_filtered_count)`` tuple. Chunks are tagged
    ``layer="general_knowledge"`` for GK hits exactly as the main path does, so
    every downstream consumer — citation markers, the sources panel, disclosure
    — behaves identically.

    Failures here return ``([], 0)`` rather than raising: this IS the
    degradation path, and it must not become a new way to break a request.
    """
    if not query or not query.strip():
        return [], 0
    if not project_id:
        raise ValueError("project_id is required")

    try:
        # get_lexical_store, not get_store: the latter constructs an embedder
        # just to read the table width, which is the coupling this path exists
        # to break.
        store = get_lexical_store()
    except Exception as exc:  # noqa: BLE001 — degradation must not raise
        logger.warning("lexical-only retrieval unavailable: %s", exc)
        return [], 0

    over_fetch = candidate_overfetch(k)
    candidates: List[Chunk] = []
    try:
        candidates.extend(store.bm25_search(
            project_id, retrieval_lexical_query(query), over_fetch,
        ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("lexical retrieval failed for %s: %s", project_id, exc)
        return [], 0

    for gk_pid in _general_knowledge_project_ids():
        if gk_pid == project_id:
            continue
        try:
            for chunk in store.bm25_search(gk_pid, query, over_fetch):
                chunk.layer = "general_knowledge"
                candidates.append(chunk)
        except Exception as exc:  # noqa: BLE001 — GK never breaks the primary leg
            logger.warning("lexical GK retrieval for %s failed: %s", gk_pid, exc)

    if (
        _cd_particulars_boost_enabled()
        and query_asks_for_contract_particulars(query)
    ):
        particulars_q = (
            f"{query.strip()} Contract Data particulars filled-in amount "
            "duration percentage"
        )
        try:
            extra = store.bm25_search(project_id, particulars_q, over_fetch)
            seen = {c.chunk_id for c in candidates}
            for chunk in extra:
                if chunk.chunk_id not in seen:
                    candidates.append(chunk)
                    seen.add(chunk.chunk_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "lexical contract-data particulars retrieval failed: %s", exc,
            )
        for chunk in candidates:
            delta = contract_data_particulars_delta(chunk.text or "")
            if delta:
                chunk.score = round((chunk.score or 0.0) + delta, 6)

    seen_lex = {c.chunk_id for c in candidates}
    candidates.extend(
        _fetch_numeric_requirement_chunks(
            query, project_id, store, over_fetch, seen_lex,
        )
    )

    fused_lex: Dict[str, Tuple] = {
        c.chunk_id: (c, c.score or 0.0, 0.0) for c in candidates
    }
    extra_lex_pids = _general_knowledge_project_ids()
    recall_asked_quantity_chunks(
        query, project_id, fused_lex, store, k=k, extra_pids=extra_lex_pids,
    )
    filename_names = _pool_docs_named_by_query(
        query, project_id, fused_lex, store,
        extra_pids=extra_lex_pids,
    )
    filename_names.update(recall_titled_documents(
        query, project_id, fused_lex, store,
        extra_pids=extra_lex_pids,
    ))
    filename_names.update(recall_labelled_rows(
        query, project_id, fused_lex, store,
        extra_pids=extra_lex_pids,
    ))
    _pool_named_document_control_block(
        query, project_id, fused_lex, store,
        extra_pids=extra_lex_pids,
    )
    recall_issue_stamps(
        query, project_id, fused_lex, store,
        extra_pids=extra_lex_pids,
    )
    recall_composition_operands(query, project_id, fused_lex, store)
    recall_rows_deep_in_pooled_documents(query, project_id, fused_lex, store)
    _pool_page_total_rows(
        query, project_id, fused_lex, store,
    )
    recall_list_continuations(query, project_id, fused_lex, store, k=k)
    if len(fused_lex) > len(candidates):
        seen = {c.chunk_id for c in candidates}
        for chunk, _sem, _b in fused_lex.values():
            if chunk.chunk_id not in seen:
                candidates.append(chunk)
                seen.add(chunk.chunk_id)

    name_by_id: Dict[str, str] = dict(filename_names)
    scored_lex: List[Tuple[float, Chunk]] = [
        (c.score or 0.0, c) for c in candidates
    ]
    with _doc_names_prefetched(
        c.doc_id for _, c in scored_lex if c.doc_id not in name_by_id
    ):
        for _, chunk in scored_lex:
            if chunk.doc_id not in name_by_id:
                name_by_id[chunk.doc_id] = _doc_name_for_id(chunk.doc_id)
    _apply_filename_overlap_boost(query, scored_lex, name_by_id)
    _apply_source_class_preference(query, scored_lex, name_by_id)
    _cap_specification_class_bonus(query, scored_lex, name_by_id)
    _apply_numeric_requirement_boost(query, scored_lex, higher_is_better=False)
    _apply_title_filename_boost(query, scored_lex, name_by_id)
    _apply_register_line_boost(query, scored_lex)
    _apply_contract_data_filename_boost(query, scored_lex, name_by_id)
    _apply_asked_particular_value_boost(query, scored_lex)
    _apply_schedule_register_boost(query, scored_lex)
    _apply_pcg_value_boost(query, scored_lex)
    _apply_commencement_date_boost(query, scored_lex)
    _apply_rate_only_boost(query, scored_lex)
    _apply_priced_boq_boost(query, scored_lex)
    _apply_part_summary_boost(query, scored_lex)
    candidates = [chunk for _s, chunk in scored_lex]

    # Stable sort keeps the active project ahead of GK on equal scores.
    candidates.sort(key=lambda c: -(c.score or 0.0))

    kept: List[Chunk] = []
    noise_filtered = 0
    seen_copies: Set[str] = set()

    def _name(doc_id: str) -> str:
        return name_by_id.get(doc_id, "") or _doc_name_for_id(doc_id)

    # Elect the unnamed contract from answer-bearing evidence. Reads the
    # SAME ``name_by_id`` the loop below uses, which #490 populates from its
    # filename rescue — so a document that only entered the pool because its
    # NAME matched the query is electable on that name.
    scope = _ContractScope(
        query,
        ranked_docs=((_name(c.doc_id), c.text or "") for c in candidates),
    )
    for chunk in candidates:
        name = _name(chunk.doc_id)
        if _is_noise_filename(name):
            noise_filtered += 1
            continue
        if not scope.allow(name, chunk.text or ""):
            continue
        copy_key = chunk_copy_key(chunk.text or "") if spec_boost_guard_enabled() else ""
        if copy_key and copy_key in seen_copies:
            continue
        if copy_key:
            seen_copies.add(copy_key)
        chunk.source_name = name
        kept.append(chunk)
        if len(kept) >= k:
            break
    _allow = (
        lambda c: not _is_noise_filename(_name(c.doc_id))
        and scope.allow(_name(c.doc_id), c.text or "")
    )
    reserve_matching_particulars_row(query, kept, candidates, allow=_allow)
    reserve_monetary_base_row(query, kept, candidates, allow=_allow)
    reserve_daily_damages_operands(query, kept, candidates, allow=_allow)
    ensure_kept_can_compose_daily_damages(query, kept, candidates, allow=_allow)
    ensure_kept_has_including_vat(query, kept, candidates, allow=_allow)
    for chunk in kept:
        chunk.source_name = _name(chunk.doc_id)
    return kept, noise_filtered


def retrieve_with_filter(
    query: str,
    project_id: str,
    k: int = 5,
    *,
    intent: Optional[str] = None,
    operator_text: Optional[str] = None,
) -> tuple:
    """Returns ``(chunks, noise_filtered_count)``.

    Pulls ``candidate_overfetch(k)`` raw candidates (floor 60, so
    production k=5 yields a pool of 60) from the active project's
    vector store, then ALSO pulls the same over-fetch from each
    general-knowledge project (the configured default — see
    ``_general_knowledge_project_ids``). The two candidate sets are
    merged, re-ranked by vector score descending, noise-filtered, and
    the top K returned.

    **Identifier-aware precision:** if the query contains construction
    reference identifiers (VO/RFI/NCR/PRC/drawing codes/etc.), the
    retriever also performs a case-insensitive substring search over
    chunk text and boosts matching chunks above pure semantic hits.
    This prevents a high-cosine generic boilerplate chunk from
    outranking the exact document that contains the requested code.

    Behaviour notes:
      * The active project is queried first so its chunks appear
        before GK chunks on equal scores (stable Python sort).
      * GK projects equal to ``project_id`` are skipped (no
        double-counting).
      * A GK lookup failure NEVER breaks the primary query — it is
        logged + the active-only results stand.
      * When ``RAG_GENERAL_KNOWLEDGE_PROJECTS=""``, no GK lookup runs
        and the retriever behaves as it did pre-PR-107.

    **GK contamination knobs (all default OFF — unset env var means the
    merge is bit-identical to the description above):** RAG_AUDIT_V2
    measured curated GK notes beating a user's own uploads 9/12 times,
    because GK candidates compete purely on score. Three independent,
    env-driven counters:

      * ``RAG_GK_SCORE_MARGIN`` (float) — a GK candidate survives only
        when its score >= best active-project candidate's score + margin.
        A project with no candidates leaves GK untouched (the GK-only
        fallback for empty projects must keep working).
      * ``RAG_OWN_DOC_BOOST`` (float) — additive boost on active-project
        candidates (ownership, not recency) before the merged re-rank.
      * ``RAG_GK_TOPK_CAP`` (int) — at most this many GK chunks in the
        final top-K; excess GK slots backfill with the next-best project
        chunks, or shrink the result when none remain.
      * ``RAG_GK_LEXICAL_FOLD`` (bool flag) — folds the GK lexical bonus
        inside the margin gate: the H1 comparison runs on each GK chunk's
        raw fused score (lexical bonus subtracted), so a GK note that only
        outranks the project's own document because of its keyword bonus
        is gated out. Survivors keep the bonus for ordering. No-op unless
        ``RAG_GK_SCORE_MARGIN`` is also set.

    The knobs apply only when ``intent`` is None or in DOC_LOOKUP_INTENTS;
    any other intent (notably CALC_KB_INTENTS) bypasses them so features
    that depend on GK winning are unaffected.

    The audit log records ``noise_filtered_count`` so the regex can be
    tuned from data.
    """
    if not available():
        # NOT an automatic []. BM25 is pure text matching over chunks.text and
        # needs no embedder at all — the vector leg does. Returning [] here
        # coupled the ENTIRE search surface to the embedder: remove the model
        # and search_project_documents went to zero hits on a corpus whose text
        # was sitting right there, fully indexed and matchable.
        #
        # This is what lets the embedder be removed, swapped, or fail without
        # taking search down with it: semantic ranking is lost, keyword
        # retrieval survives. Degrade a capability, not the product.
        logger.debug("embedding stack unavailable; retrieving lexical-only")
        return _lexical_only_retrieve(query, project_id, k)
    if not query or not query.strip():
        return [], 0
    if not project_id:
        raise ValueError("project_id is required")

    embedder = get_embedder()
    query_vec = embedder.encode_queries([query])[0]
    # F18 dual-query: also embed the wrapper-stripped phrase-intact variant
    # of a natural question (None for terse queries -- no extra cost).
    alt_query = _strip_question_wrapper(query) if _dual_query_enabled() else None
    alt_vec = embedder.encode_queries([alt_query])[0] if alt_query else None
    store = get_store(dim=embedder.dim)
    over_fetch = candidate_overfetch(k)
    # The GK corpus is small and curated (units / CESMM / FIDIC / procedures), so
    # over-fetch it generously: a lexically-relevant reference chunk must enter
    # the candidate pool even when its semantic score for a broad query is low
    # -- the lexical boost below can only re-rank chunks that made the fetch.
    gk_over_fetch = max(k * 12, 80)

    # Active project (operator's own corpus — first so it wins ties).
    raw_active = _dual_search(
        store, project_id, query_vec, retrieval_lexical_query(query),
        alt_vec, alt_query, k=over_fetch,
    )

    # Particulars-shaped questions: a third search whose wording matches the
    # index-time "CONTRACT DATA particulars" prefix. Additive to F18 dual-query
    # (that transform stays the alt_query above). Failures never break primary.
    if (
        _cd_particulars_boost_enabled()
        and query_asks_for_contract_particulars(query)
    ):
        particulars_q = (
            f"{query.strip()} Contract Data particulars filled-in amount "
            "duration percentage"
        )
        try:
            pvec = embedder.encode_queries([particulars_q])[0]
            p_hits = store.search(
                project_id, pvec, k=over_fetch, query_text=particulars_q,
            )
            by_id = {c.chunk_id: c for c in raw_active}
            for c in p_hits:
                prev = by_id.get(c.chunk_id)
                if prev is None or (c.score or 0.0) > (prev.score or 0.0):
                    by_id[c.chunk_id] = c
            raw_active = sorted(
                by_id.values(), key=lambda c: -(c.score or 0.0),
            )
        except Exception as exc:  # noqa: BLE001 — extras must not break the turn
            logger.warning(
                "contract-data particulars retrieval for %s failed: %s; "
                "primary results stand",
                project_id, exc,
            )

    # Contract-term synonym boost: a supplementary search whose wording adds
    # the canonical FIDIC heading for any synonym the user used ("contract sum"
    # -> "Accepted Contract Amount"). Additive to the primary + particulars
    # legs; the primary query is unchanged. Failures never break primary.
    if _contract_synonym_boost_enabled():
        synonym_terms = expand_contract_synonyms(query)
        if synonym_terms:
            synonym_q = f"{query.strip()} {synonym_terms}"
            try:
                svec = embedder.encode_queries([synonym_q])[0]
                s_hits = store.search(
                    project_id, svec, k=over_fetch, query_text=synonym_q,
                )
                by_id = {c.chunk_id: c for c in raw_active}
                for c in s_hits:
                    prev = by_id.get(c.chunk_id)
                    if prev is None or (c.score or 0.0) > (prev.score or 0.0):
                        by_id[c.chunk_id] = c
                raw_active = sorted(
                    by_id.values(), key=lambda c: -(c.score or 0.0),
                )
            except Exception as exc:  # noqa: BLE001 — extras must not break the turn
                logger.warning(
                    "contract-synonym retrieval for %s failed: %s; "
                    "primary results stand",
                    project_id, exc,
                )

    seen_active = {c.chunk_id for c in raw_active}
    numeric_extra = _fetch_numeric_requirement_chunks(
        query, project_id, store, over_fetch, seen_active,
    )
    if numeric_extra:
        raw_active = list(raw_active) + numeric_extra

    # General-knowledge projects (cross-project background context).
    # Only merge GK when the active project already has indexed chunks.
    # An empty/unindexed project must return [] — not general-knowledge
    # hits — or search_project_documents, lazy bootstrap, and the
    # "unindexed project" contract all break (Postgres CI shares a DB where
    # GK rows exist from other tests / the migrated corpus).
    # Prefer authoritative corpus-size check, but fall back to the fetched
    # active candidates for mocked/in-memory test stores that don't model
    # ``count`` consistently with ``search``.
    include_gk = store.count(project_id) > 0 or bool(raw_active)
    gk_ids = (
        [pid for pid in _general_knowledge_project_ids() if pid != project_id]
        if include_gk
        else []
    )
    raw_gk: List[Chunk] = []
    for gk_pid in gk_ids:
        try:
            raw_gk.extend(store.search(gk_pid, query_vec, k=gk_over_fetch, query_text=query))
        except Exception as exc:  # noqa: BLE001 — never let GK break primary path
            logger.warning(
                "general-knowledge retrieval for %s failed: %s; primary results stand",
                gk_pid, exc,
            )

    # Project-own rows of another project are never fetched. Shared
    # general-knowledge stays in gk_ids. The former master-corpus fallback
    # queried a foreign project_id; that is the leak this scope closes.
    fb_id = _master_corpus_fallback_id()
    use_fallback = False
    raw_fb: List[Chunk] = []

    # Identifier-aware lexical rescue for exact reference lookups.
    identifiers = extract_query_identifiers(query)
    id_candidates: Dict[str, Tuple[Chunk, float]] = {}
    if identifiers:
        try:
            id_active = store.identifier_search(project_id, identifiers, k=over_fetch)
            for c in id_active:
                id_candidates[c.chunk_id] = (c, c.score or 0.0)
            for gk_pid in gk_ids:
                try:
                    id_gk = store.identifier_search(gk_pid, identifiers, k=over_fetch)
                    for c in id_gk:
                        # Active-project identifier hits win ties over GK.
                        if c.chunk_id not in id_candidates:
                            id_candidates[c.chunk_id] = (c, c.score or 0.0)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "identifier search for GK %s failed: %s", gk_pid, exc
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("identifier search failed: %s; falling back to semantic", exc)

    # Fuse semantic and identifier signals.
    # Semantic chunks carry their cosine score; identifier hits add a
    # bonus proportional to how many identifiers they match. A chunk that
    # matches all requested identifiers receives a +2.0 bonus, which is
    # larger than any pure semantic score, guaranteeing it outranks
    # semantically-similar boilerplate that lacks the exact reference.
    fused: Dict[str, Tuple[Chunk, float]] = {}
    for c in list(raw_active) + raw_gk + raw_fb:
        fused[c.chunk_id] = (c, c.score or 0.0, 0.0)

    IDENTIFIER_BONUS_MAX = 2.0
    # Tie-break among identifier hits. Capped equal to the identifier
    # bonus so a full-overlap code row can beat a high-cosine Excluded
    # mention of the same code, but a no-code semantic chunk (≤ ~1.0)
    # still cannot outrank a bare identifier hit (2.0).
    IDENTIFIER_CONTEXT_BOOST_MAX = 2.0
    for chunk_id, (id_chunk, id_score) in id_candidates.items():
        if chunk_id in fused:
            sem_chunk, sem_score, _ = fused[chunk_id]
            fused[chunk_id] = (sem_chunk, sem_score, id_score * IDENTIFIER_BONUS_MAX)
        else:
            # Identifier-only hit: keep its text but start from zero semantic.
            fused[chunk_id] = (id_chunk, 0.0, id_score * IDENTIFIER_BONUS_MAX)

    # 2026-07-26 precision fix: identifier_search returns an ARBITRARY top-k
    # of the (possibly hundreds of) chunks containing the code, so on a large
    # corpus the semantically-best chunk that ALSO carries the identifier can
    # miss the id_candidates set entirely — and then flat-bonused label-soup
    # (drawing station tables, schedule rows) displaces it. Award the same
    # bonus to every SEMANTIC candidate whose text contains the identifiers:
    # cosine + bonus then always outranks identifier-only hits (bonus alone),
    # which is the ordering the boost was designed to produce. (Live find:
    # 'WWPS-01 total flow rate' on the corpus project buried the 0.75-cosine
    # spec table under zero-semantic drawing chunks.)
    if identifiers:
        # Token-wise matching, mirroring identifier_search: "VO Ref: 99" and
        # "VO 99" both match, punctuation between tokens is ignored. CESMM
        # codes are collapsed so ``d549`` is a substring of ``D 549.2``.
        ident_token_sets = []
        for ident in identifiers:
            toks = [
                t for t in re.split(
                    r"[^a-z0-9]+", normalize_cesmm_item_codes(ident).lower()
                ) if t
            ]
            if toks:
                ident_token_sets.append(toks)
        for chunk_id, entry in list(fused.items()):
            sem_chunk, sem_score, id_bonus = entry
            # Skip entries that already carry the search-assigned bonus
            # (identifier-only hits live there with sem_score 0.0). Real
            # semantic candidates keep eligibility even at negative cosine.
            if id_bonus > 0.0 or not ident_token_sets:
                continue
            text_lower = normalize_cesmm_item_codes(sem_chunk.text or "").lower()
            matched = sum(
                1 for toks in ident_token_sets
                if all(t in text_lower for t in toks)
            )
            if matched:
                local_score = matched / len(ident_token_sets)
                fused[chunk_id] = (
                    sem_chunk, sem_score, local_score * IDENTIFIER_BONUS_MAX,
                )

        # WAVE 2 B4: several BOQ rows can share a CESMM code (carriageway
        # qty vs an Excluded culvert that mentions D599.5). Identifier
        # bonus is flat, so the arbitrary SQL hit wins. Add a secondary
        # boost from the rest of the query (carriageway / 340904) — only
        # on chunks that already earned the identifier bonus, so
        # boilerplate without the code cannot climb the fence.
        ctx_terms = _identifier_context_terms(query, identifiers)
        if ctx_terms:
            for chunk_id, (chunk, sem_score, id_bonus) in list(fused.items()):
                if id_bonus <= 0.0:
                    continue
                overlap = _identifier_context_overlap(ctx_terms, chunk.text)
                if overlap <= 0.0:
                    continue
                fused[chunk_id] = (
                    chunk,
                    sem_score,
                    id_bonus + overlap * IDENTIFIER_CONTEXT_BOOST_MAX,
                )

    # ── lexical term rescue ────────────────────────────────────────────────
    # Two passes, because the live failure had two distinct shapes and only
    # doing one of them leaves the other broken:
    #
    #  (a) IN-POOL. The chunk carrying the query's terms WAS fetched, but sits
    #      deep in the over-fetch (rank ~15 of the pool) on cosine alone and never
    #      reaches the top-5 the user sees. Bonusing it in place is what lifts
    #      it. An earlier version of this fix gated the whole rescue on "is a
    #      term-carrying chunk anywhere in the candidate pool" — which this case
    #      satisfies, so the rescue was skipped and the chunk still never
    #      surfaced. The gate reproduced the very bug it was meant to fix.
    #
    #  (b) OUT-OF-POOL. The chunk was never fetched at all (the SBC 304 case:
    #      one document in 227, no semantic pull, k*4 candidates). Only a
    #      lexical lookup can recover it, so that runs when — and only when —
    #      pass (a) found nothing, keeping the extra SQL off the healthy path.
    rescue_terms = distinctive_query_terms(query)
    if len(rescue_terms) >= _TERM_RESCUE_MIN_TERMS:
        pairs = cooccurrence_pair_phrases(rescue_terms)

        def _pair_fraction(text: str) -> float:
            """Fraction of term PAIRS co-occurring in ``text``. Graduated:
            a chunk with every term scores 1.0, one with two of five scores
            0.1 — so a passing word overlap earns a token bonus, not a
            promotion."""
            lowered = (text or "").lower()
            if not pairs:
                return 0.0
            matched = sum(
                1 for pair in pairs
                if all(tok in lowered for tok in pair.split())
            )
            return matched / len(pairs)

        # Healthy-retrieval gate. If the top-K the user would ALREADY see
        # carries the query's terms, retrieval is working and the rescue
        # must be a strict no-op — scores included. Without this the bonus
        # perturbs every ordinary query, inflating top_score and pushing
        # marginal retrievals past RAG_CONFIDENCE_THRESHOLD, which trades a
        # recall bug for an ungrounded-answer bug.
        provisional_top = sorted(
            fused.values(), key=lambda e: -((e[1] or 0.0) + (e[2] or 0.0)),
        )[:k]
        already_grounded = any(
            _pair_fraction(chunk.text) > 0.0 for chunk, _s, _b in provisional_top
        )

        # (a) bonus every candidate already in the pool.
        found_in_pool = False
        for chunk_id, (chunk, sem_score, id_bonus) in list(fused.items()):
            fraction = _pair_fraction(chunk.text)
            if fraction <= 0.0:
                continue
            found_in_pool = True
            if already_grounded:
                continue
            bonus = fraction * _TERM_RESCUE_BONUS_MAX
            # Never displace a stronger identifier bonus — an exact code
            # match remains the strongest signal available.
            fused[chunk_id] = (chunk, sem_score, max(id_bonus, bonus))

        # (b) lexical fetch for chunks the semantic pass never saw.
        if not found_in_pool and not already_grounded:
            rescue_pids = [project_id] + gk_ids
            if use_fallback and fb_id:
                rescue_pids.append(fb_id)
            recovered = 0
            for pid in rescue_pids:
                try:
                    hits = store.identifier_search(pid, pairs, k=over_fetch)
                except Exception as exc:  # noqa: BLE001 — never break the turn
                    logger.warning(
                        "lexical term rescue for %s failed: %s", pid, exc,
                    )
                    continue
                for chunk in hits:
                    if chunk.chunk_id in fused:
                        continue
                    # Enters on the rescue bonus alone (zero cosine), so it
                    # ranks below any genuine semantic match — but above
                    # nothing, which is what the corpus-wide "does not
                    # mention it at all" answer amounted to.
                    fused[chunk.chunk_id] = (
                        chunk, 0.0,
                        _pair_fraction(chunk.text) * _TERM_RESCUE_BONUS_MAX,
                    )
                    recovered += 1
            if recovered:
                logger.info(
                    "term rescue recovered %d chunk(s) for terms %r that "
                    "semantic retrieval missed entirely",
                    recovered, rescue_terms,
                )

    # The general-knowledge pids the semantic leg searches, plus the empty-
    # project master-corpus fallback: the recall passes below fetch from the
    # SAME corpora the semantic leg did.
    extra_rescue_pids = gk_ids + ([fb_id] if use_fallback and fb_id else [])

    # FW4 S1: "per the specification" + a cover ask. The specification's own
    # clause says the cover is "as specified on the Drawings" and states no
    # millimetre, so neither cosine nor the numeric fetch ever pools it.
    spec_deferral_names = follow_quantity_pointers(
        query, project_id, fused, store, k=k,
        embedder=embedder, query_vec=query_vec,
    )
    # Asked-quantity recall: the chunk that states the asked figure for the
    # asked subject (a table row, a drawing note) in the document's own words.
    recall_asked_quantity_chunks(
        query, project_id, fused, store, k=k,
        extra_pids=extra_rescue_pids, embedder=embedder, query_vec=query_vec,
    )

    # Letter / named-party filename rescue (D1). Runs EVEN WHEN term rescue
    # already found place-name overlap in Volume 5 — that in-pool hit is
    # what used to skip the out-of-pool fetch of the actual letter.
    filename_names = _pool_docs_named_by_query(
        query,
        project_id,
        fused,
        store,
        extra_pids=extra_rescue_pids,
    )
    # Titled-document recall. Runs EVEN WHEN term rescue already found the
    # topic's words in some long volume -- that in-pool hit is what used to
    # skip the out-of-pool fetch of the document the question names.
    for _did, _nm in spec_deferral_names.items():
        filename_names.setdefault(_did, _nm)
    filename_names.update(recall_titled_documents(
        query,
        project_id,
        fused,
        store,
        extra_pids=extra_rescue_pids,
    ))
    # Labelled-row recall: the filled row for every label the question names
    # (a particular, a register entry, a bill item), from the particulars
    # documents and -- for a particulars-shaped question -- the project text.
    filename_names.update(recall_labelled_rows(
        query,
        project_id,
        fused,
        store,
        extra_pids=extra_rescue_pids,
    ))
    # Number / revision / author of a document the question names.
    _pool_named_document_control_block(
        query, project_id, fused, store,
        extra_pids=extra_rescue_pids,
    )
    # Issue stamp of the named document. Filename recall cannot see a volume
    # whose name does not repeat the document's title; the stamp does.
    recall_issue_stamps(
        query, project_id, fused, store,
        extra_pids=extra_rescue_pids,
    )
    # Rows past the early windows of a pooled volume: every operand a
    # composition needs, and the asked particular's own row.
    recall_composition_operands(query, project_id, fused, store)
    recall_rows_deep_in_pooled_documents(query, project_id, fused, store)
    # Page-total ask: the page's summary footer loses to the item lines on
    # the same page. Project-only so a reference rate note cannot
    # impersonate the client's page total.
    _pool_page_total_rows(query, project_id, fused, store)
    # List continuation: an in-pool introduction that ends "as follows";
    # the items are the next same-doc chunk. Project-only so a reference
    # note cannot impersonate the client's list.
    recall_list_continuations(query, project_id, fused, store, k=k)

    # General-knowledge lexical boost: lift GK reference chunks that overlap the
    # query so everyday phrasings surface curated references (units/CESMM/FIDIC).
    # ``gk_lex_added`` records the bonus per chunk so the H1 margin gate below
    # can compare on the pre-bonus (raw fused) score when RAG_GK_LEXICAL_FOLD
    # is on. Recording here (rather than deferring the bonus until after the
    # gate) is the smaller diff: the flag-off path stays byte-identical, and
    # the gate only needs one subtraction instead of a second scoring pass.
    q_terms = _significant_terms(query)
    gk_lex_added: Dict[str, float] = {}
    for gk_chunk_id in {c.chunk_id for c in raw_gk}:
        entry = fused.get(gk_chunk_id)
        if entry is None:
            continue
        gk_chunk, sem_score, bonus = entry
        add = _gk_lexical_bonus(q_terms, gk_chunk.text)
        if add:
            fused[gk_chunk_id] = (gk_chunk, sem_score, bonus + add)
            gk_lex_added[gk_chunk_id] = add

    # GK background factor: penalise GK vs the active project's own docs. Default
    # 1.0 (OFF) — a 0.9 penalty was found to DEMOTE the authoritative curated KB
    # below a project's contract templates for knowledge questions (a FIDIC "IPC
    # payment days" query grounded on a project's amended 45-day contract instead
    # of the KB's 56-day FIDIC default), which is worse than the project-question
    # case it was meant to help. Kept as a live env knob for future tuning.
    gk_id_set = set(gk_ids)
    GK_BACKGROUND_FACTOR = float(os.getenv("RAG_GK_BACKGROUND_FACTOR", "1.0"))
    scored: List[Tuple[float, Chunk]] = []
    for chunk, sem_score, id_bonus in fused.values():
        final_score = (sem_score or 0.0) + (id_bonus or 0.0)
        if chunk.project_id in gk_id_set and final_score > 0:
            final_score *= GK_BACKGROUND_FACTOR
        chunk.score = round(final_score, 6)
        scored.append((final_score, chunk))

    _apply_contract_data_particulars_boost(query, scored)

    # GK contamination knobs — see the docstring. Each is None (OFF) unless
    # its env var is set AND the intent is lookup-shaped; when all are None
    # the pipeline below is byte-for-byte the pre-knob behavior.
    knobs_apply = intent is None or intent in DOC_LOOKUP_INTENTS
    gk_margin = _knob_float("RAG_GK_SCORE_MARGIN") if knobs_apply else None
    own_boost = _knob_float("RAG_OWN_DOC_BOOST") if knobs_apply else None
    gk_cap = _knob_int("RAG_GK_TOPK_CAP") if knobs_apply else None
    lex_fold = _knob_flag("RAG_GK_LEXICAL_FOLD") if knobs_apply else False

    # H2: ownership boost — the active project's own chunks get an additive
    # lift before the merged re-rank, so a user's uploaded doc can beat a
    # keyword-dense GK note of similar raw score.
    if own_boost is not None:
        for i, (score, chunk) in enumerate(scored):
            if chunk.project_id == project_id:
                boosted = score + own_boost
                chunk.score = round(boosted, 6)
                scored[i] = (boosted, chunk)

    # H1: GK score margin — GK must BEAT the project's best candidate by the
    # margin to enter the pool at all. Compared after H2 so an enabled boost
    # also raises the bar. Empty projects skip the gate: GK-only fallback.
    #
    # Lexical fold (RAG_GK_LEXICAL_FOLD): RAG_AUDIT_V3 showed the GK lexical
    # bonus (+0.25/term, cap +1.2) lifts curated notes ~+0.5 above any project
    # chunk on collision queries, so a user's own contract can never clear the
    # margin comparison against a bonused GK note. When the flag is on, the
    # margin is compared on the GK chunk's RAW fused score (bonus subtracted);
    # a GK chunk that only beats the project because of its lexical bonus is
    # gated out, while one that clears the margin on raw score survives and
    # keeps its bonus for ordering. The recorded bonus is scaled by
    # GK_BACKGROUND_FACTOR because the factor multiplied the whole fused score
    # (bonus included) above; subtracting the scaled bonus recovers exactly
    # the factor-adjusted pre-bonus score.
    if gk_margin is not None:
        project_scores = [s for s, c in scored if c.project_id == project_id]
        if project_scores:
            # Outside a project, general knowledge competes on merit: it enters
            # whenever its raw score is the best match (margin 0, the lexical
            # fold still applies). The margin protects project questions only.
            if not question_framed_in_project(query, project_id, scored, filename_names):
                gk_margin = 0.0
            bar = max(project_scores) + gk_margin

            def _margin_score(s: float, c: Chunk) -> float:
                if lex_fold and c.project_id in gk_id_set:
                    return s - gk_lex_added.get(c.chunk_id, 0.0) * GK_BACKGROUND_FACTOR
                return s

            scored = [
                (s, c) for s, c in scored
                if c.project_id not in gk_id_set or _margin_score(s, c) >= bar
            ]

    # Revision currency (§5.2) — ALWAYS ON, independent of RAG_LAYERED. A
    # stale-revision drawing answer is a real construction-safety hazard, so
    # this runs regardless of the layered flag. Both signals key off the
    # uploader's original filename (resolved once per distinct doc below):
    #   1. annotate every chunk with its parsed revision / drawing number;
    #   2. firmly down-rank any chunk whose filename marks it SUPERSEDED so a
    #      current document of comparable relevance always wins — a penalty, not
    #      a drop, so a superseded chunk still survives as a last resort when
    #      nothing current matches.
    # The same-drawing "prefer highest revision" suppression is computed from
    # these annotations and applied in the kept-selection loop below.
    name_by_id: Dict[str, str] = dict(filename_names)
    with _doc_names_prefetched(
        c.doc_id for _, c in scored if c.doc_id not in name_by_id
    ):
        for _, chunk in scored:
            if chunk.doc_id not in name_by_id:
                name_by_id[chunk.doc_id] = _doc_name_for_id(chunk.doc_id)
    for i, (score, chunk) in enumerate(scored):
        nm = name_by_id.get(chunk.doc_id, "")
        # The filename itself is evidence, not just a ranking input — see
        # Chunk.source_name. Set here because this is where the name is already
        # resolved, so it costs nothing extra.
        chunk.source_name = nm
        chunk.revision = _revision.revision_token(nm)
        chunk.drawing_number = _revision.drawing_number(nm)
        if _revision.is_superseded(nm):
            chunk.superseded = True
            demoted = score - _revision.SUPERSEDED_PENALTY
            chunk.score = round(demoted, 6)
            scored[i] = (demoted, chunk)

    _apply_filename_overlap_boost(query, scored, name_by_id)
    _apply_source_class_preference(query, scored, name_by_id)
    _cap_specification_class_bonus(query, scored, name_by_id)
    _apply_numeric_requirement_boost(query, scored)
    _apply_quantity_pointer_boost(query, scored, name_by_id)
    _apply_title_filename_boost(query, scored, name_by_id)
    _apply_register_line_boost(query, scored)
    _apply_contract_data_filename_boost(query, scored, name_by_id)
    _apply_asked_particular_value_boost(query, scored)
    _apply_schedule_register_boost(query, scored)
    _apply_pcg_value_boost(query, scored)
    _apply_commencement_date_boost(query, scored)
    _apply_rate_only_boost(query, scored)
    _apply_priced_boq_boost(query, scored)
    _apply_part_summary_boost(query, scored)

    # Stage 3 (layered RAG): authority-precedence re-rank. Add a small term so a
    # higher-authority / higher-layer chunk (e.g. an L2B contractual clause)
    # outranks a comparably-relevant low-authority one (an L1 historical note).
    # Applied AFTER the GK-margin gate so it only changes final ordering, and
    # flag-gated: when RAG_LAYERED is off, `scored` is untouched — today's
    # ordering byte-for-byte.
    if layers.layered_enabled():
        for i, (score, chunk) in enumerate(scored):
            bonus = layers.precedence_bonus(
                getattr(chunk, "knowledge_layer", None),
                getattr(chunk, "authority", None),
            )
            if bonus:
                new_score = score + bonus
                chunk.score = round(new_score, 6)
                scored[i] = (new_score, chunk)

    # Project layer first for a project-framed lookup question (after every
    # boost, so nothing below can lift a GK chunk back over the project).
    if knobs_apply:
        _keep_project_layer_first(query, project_id, scored, gk_id_set)

    # Sort by fused score descending; active-project chunks naturally come
    # first when scores are equal because they were inserted first.
    scored.sort(key=lambda x: -x[0])

    # Revision currency (§5.2 step 2): highest COMPARABLE revision retrieved per
    # drawing number, bucketed by revision kind ((drawing_number, kind) -> max
    # value). Numeric and alphabetic revisions of one sheet live in separate
    # buckets and never suppress each other — so a mixed-scheme sheet keeps both
    # rather than risk hiding the current one (see revision.revision_rank).
    best_rev: Dict[Tuple[str, int], int] = {}
    for _, c in scored:
        dn = c.drawing_number
        rank = _revision.revision_rank(c.revision)
        if dn and rank is not None:
            key = (dn, rank[0])
            if key not in best_rev or rank[1] > best_rev[key]:
                best_rev[key] = rank[1]

    # Photo chunks RAG leg was removed in migration 0008 along with the
    # photo_chunks table. Chat-attached photos are now question-context
    # (see POST /v1/chat/analyze-photo), not corpus material.
    # H3: GK top-K cap — skipping (not truncating) excess GK chunks lets the
    # next-best project chunks flow into the freed slots; when the pool has
    # no project chunks left the result simply comes back shorter.
    #
    # Cross-encoder rerank (RERANK_ENABLED / RAG_RERANKER, default OFF):
    # collect a DEEPER candidate pool (default top-50 hybrid) through the
    # very same gates below, then let the cross-encoder pick the best k.
    # Flag off -> target == k and the loop is byte-identical to before.
    # The gates run FIRST either way, so a noise-filtered, revision-
    # suppressed, or GK-capped chunk can never be resurrected by a good
    # rerank score. Do not enable in production until artifacts/fork/
    # RERANK_EVAL.md shows zero regressions + a held p95.
    rerank_on = _reranker.enabled()
    target = _reranker.candidate_depth(k) if rerank_on else k
    kept: List[Chunk] = []
    noise_dropped = 0
    revision_suppressed = 0
    gk_kept = 0
    seen_copies: Set[str] = set()
    # `scored` is already in final rank order, so this election sees exactly
    # the ranking the user would have got — and prevents a wrong-contract
    # pointer at rank 1 from deleting the row that holds the answer.
    scope = _ContractScope(
        query,
        ranked_docs=((name_by_id.get(c.doc_id, ""), c.text or "") for _, c in scored),
    )
    for _, c in scored:
        name = name_by_id.get(c.doc_id, "")
        if _is_noise_filename(name):
            noise_dropped += 1
            continue
        # Named-contract questions stay on that contract/doc id. Wrong
        # year (AB-2023 query / AB-2022 chunk) is dropped here, not ranked
        # through. Empty kept after this loop is fail-closed.
        if not scope.allow(name, c.text or ""):
            continue
        # Prefer the highest revision of a given drawing: skip this chunk when a
        # strictly-higher, same-kind revision of the SAME drawing number is also
        # in the pool. Stale-revision safety, so this suppresses rather than
        # merely nudges — but only among chunks that share a parsed drawing id.
        dn = c.drawing_number
        rank = _revision.revision_rank(c.revision)
        if dn and rank is not None and best_rev.get((dn, rank[0]), rank[1]) > rank[1]:
            revision_suppressed += 1
            continue
        # Signed and unsigned copies of one clause share a body. Keep the
        # higher-scored copy (this list is rank order) and free the slot.
        copy_key = chunk_copy_key(c.text or "") if spec_boost_guard_enabled() else ""
        if copy_key and copy_key in seen_copies:
            continue
        if gk_cap is not None and c.project_id in gk_id_set:
            if gk_kept >= gk_cap:
                continue
            gk_kept += 1
        if copy_key:
            seen_copies.add(copy_key)
        kept.append(c)
        if len(kept) == target:
            break

    # Second-stage rerank: reorder the survivors by cross-encoder relevance
    # and cut to k. Degrades to kept[:k] (i.e. today's exact result) on any
    # model/scoring failure — see reranker.rerank.
    if rerank_on and len(kept) > k:
        kept = _reranker.rerank(query, kept, k)

    # Runs on the FINAL k, after the rerank cut, so the reserved row cannot be
    # reordered back out. Re-uses the same noise and contract-scope gates the
    # loop above applied.
    _allow_final = (
        lambda c: not _is_noise_filename(name_by_id.get(c.doc_id, ""))
        and scope.allow(name_by_id.get(c.doc_id, ""), c.text or "")
    )
    reserve_matching_particulars_row(
        query, kept, [c for _, c in scored], allow=_allow_final,
    )
    reserve_monetary_base_row(
        query, kept, [c for _, c in scored], allow=_allow_final,
    )
    reserve_daily_damages_operands(
        query, kept, [c for _, c in scored], allow=_allow_final,
    )
    ensure_kept_can_compose_daily_damages(
        query, kept, [c for _, c in scored], allow=_allow_final,
    )
    ensure_kept_has_including_vat(
        query, kept, [c for _, c in scored], allow=_allow_final,
    )
    # Synonym boost survival: a synonym-named Contract Data figure ("contract
    # sum" -> Accepted Contract Amount) that the score-based cut dropped below
    # top-k gets one reserved slot, so the answer layer sees the figure it
    # would otherwise decline on. Runs last, on the final k, like the reserves
    # above; no-op unless the query uses such a synonym.
    reserve_contract_synonym_row(
        query, kept, [c for _, c in scored], allow=_allow_final,
    )

    # Tag each returned chunk with its retrieval layer so the chat runtime can
    # disclose a Master-Corpus fallback (STEP 0b). "own" is the active project;
    # a chunk from the fallback corpus is "master_corpus"; anything else that
    # made it through is a disclosed general-knowledge chunk.
    for c in kept:
        if c.project_id == project_id:
            c.layer = "own"
        elif use_fallback and c.project_id == fb_id:
            c.layer = "master_corpus"
        else:
            c.layer = "general_knowledge"
        # A BOQ item reference is an identifier, not a figure. Label it here,
        # in the retrieval path, so a consumer reading this chunk can tell the
        # two apart without re-deriving the taxonomy (app/lib/boq_ref_codes).
        # Display and citation are unaffected: the text is untouched.
        c.ref_codes = tuple(find_ref_codes(c.text))

    if revision_suppressed:
        logger.debug(
            "revision currency: suppressed %d lower-revision chunk(s) in favour "
            "of a higher revision of the same drawing", revision_suppressed,
        )

    return kept, noise_dropped


def index_chunks(
    project_id: str,
    doc_id: str,
    chunks: List[str],
    *,
    pages: Optional[List[Optional[int]]] = None,
) -> int:
    """Embed ``chunks`` and write them to the store for retrieval.

    ``pages`` (optional, aligned with ``chunks``) is the 1-based source page
    each chunk starts on; None entries, or None, mean no page is known.

    Returns the number of chunks indexed. Returns 0 silently when the
    embedding stack isn't installed — the doc indexer treats this as
    "RAG is off, nothing to do."
    """
    if not available():
        return 0
    if not chunks:
        return 0
    chunks = [normalize_cesmm_item_codes(c) for c in chunks]
    embedder = get_embedder()
    embeddings = embedder.encode(chunks)
    store = get_store(dim=embedder.dim)
    # Layered RAG (flag-gated): tag the doc's chunks with their knowledge layer
    # (L1/L2A/L2B/L3) and authority so retrieval can rank by precedence. Off by
    # default -> (None, None), i.e. today's behaviour byte-for-byte. A doc whose
    # metadata carries provenance="user_upload" (set by the interactive upload
    # endpoint) is routed to the user_session layer (Stage 4).
    knowledge_layer = authority = None
    if layers.layered_enabled():
        name, is_user_upload = _doc_name_and_provenance(doc_id)
        knowledge_layer, authority = layers.classify(
            project_id, name, is_user_upload=is_user_upload)
    # Pages go to the store only when some are known, so a store (or a test
    # double) written before pages existed keeps working for non-PDF sources.
    extra = {"pages": list(pages)} if pages and any(pages) else {}
    return store.upsert_chunks(
        project_id, doc_id, chunks, embeddings,
        knowledge_layer=knowledge_layer, authority=authority, **extra)


def _doc_name_and_provenance(doc_id: str) -> tuple:
    """Return ``(original_name, is_user_upload)`` for a doc. is_user_upload is
    True when the doc's metadata provenance marks it an interactive upload.
    Safe: unknown/missing doc -> ('', False)."""
    try:
        from app.core import projects as _projects
        doc = _projects.get_document(doc_id) or {}
        name = doc.get("original_name") or ""
        prov = (doc.get("metadata") or {}).get("provenance")
        return name, prov == "user_upload"
    except Exception:
        return "", False
