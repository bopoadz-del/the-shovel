"""Conflicts between documents in the evidence of one answer (owner order C2/C3).

A general rule over the evidence set. No document, project, question or
figure is named here, and no order of precedence is built in.

C2 DETECT. For every sentence of the answer that states a value, the asked
thing is that sentence (plus the user's question when known). In each
evidence document, the value of the same kind whose sentence shares the most
content words with the asked thing is that document's value for it. When two
or more documents give values that differ after unit and format
normalisation, that is a conflict: every document's value is reported with
its document. The comparison is deterministic; no model decides it.

  * Kinds: a quantity with a unit (length, area, volume, mass, time,
    pressure, force, percent), money in one currency, a calendar date, a
    grade or class code, and -- only in a sentence with none of these -- a
    short labelled phrase (``Label: value,``).
  * Normalisation: units convert to one base per kind (``0.2 m`` is
    ``200 mm``; ``N/mm2`` is ``MPa``; ``12 weeks`` is ``84 days``); a month
    or a year is a range of days, so two durations agree when their ranges
    meet; money in words (``2.5 million``) is the amount; ``2027-03-15`` is
    ``15 March 2027``.
  * The answer anchors it: a conflict is reported only when one of the
    documents' values is the value the answer itself stated. A value the
    answer did not state is never turned into a conflict about something
    else.
  * The value must belong to the asked thing: the thing is the asked words
    (the question's, else the answer sentence's) that the document naming
    the answer's value best uses (a word naming only the value's kind, such
    as "date", is said by the value); another document's value counts only
    when its sentence names all of them. Two documents giving values for different
    things that share words (two elements, two sections) are no conflict.
  * A document that holds two different values of the kind for the asked
    thing, equally well matched, is left out (no claim on doubt).
  * A value is the sentence's own claim. Citation spans, retrieval keys and
    numbered source markers are blanked first (``app.lib.citation_spans``,
    the same rule the offline answer scorer uses), and a number glued to a
    letter, hyphen, slash or point (a drawing, clause or item code) is no
    quantity. A labelled phrase is a value only when it holds a word and no
    code: a bare number, a page, a list or row number, or a document code
    after a label is not a claim.
  * No thing, no conflict: when the asked words that name the thing are
    empty, nothing is compared.
  * The value's thing is read from its own clause: a passage is cut at a
    heading-path step, a semicolon or a line break; a clause holding ONE
    value is named by its whole clause, a clause holding several values by
    only the words within a few of each value (a table row or a page of
    text with no full stop is not one claim).
  * The question's thing: its content words less the kind of value (a kind
    word and the single word that runs straight into it, "concrete grade").
    A question that names only the kind names no thing: no conflict, since
    two values found for it may belong to two different elements. A
    trailing "what do X and Y each say?" clause asks what each source
    says; its words name the sources, not the thing, and are dropped
    before the thing is read.
  * An answer that already states every one of the differing values has
    told the user both; it is left as written -- unless it also says the
    documents agree (match, are consistent): then it has not told the user
    they differ, and the conflict statement leads.

C3 PRECEDENCE. A sentence in the evidence that states which document kind
takes precedence, prevails, governs or has priority (a pairwise statement
or an ordered list) is read for the kinds it names; a kind names a document
when every content word of the kind is in the document's name. When every
document in the conflict is ranked by one such statement, the top-ranked
value governs and the statement's document is the basis. Otherwise no
winner is named.

Never raises: on any failure, no conflict is reported.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_LOG = logging.getLogger(__name__)

# ── Words ───────────────────────────────────────────────────────────────────

_STOP = frozenset(["a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "and", "or", "as", "with", "from", "into", "within", "per", "is", "are", "was", "were", "be", "been", "being", "shall", "will", "would", "should", "must", "may", "can", "could", "do", "does", "did", "it", "its", "this", "that", "these", "those", "there", "their", "what", "which", "when", "where", "who", "whom", "whose", "why", "how", "long", "much", "many", "any", "all", "each", "every", "some", "specified", "required", "given", "stated", "state", "states", "value", "values", "amount", "number", "figure", "total"])


def _stem(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    changed = True
    while changed:
        changed = False
        for suf in ("ness", "ions", "ion", "ing", "ed", "es", "s", "e"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                changed = True
                break
    return w


#: Words that name a value KIND this module reads (a date, a grade or class,
#: a quantity's dimension, a sum of money). In a value's own sentence the
#: value says them; they never tell one thing from another.
_KIND_TERMS = frozenset(_stem(w) for w in (
    "date", "grade", "class", "length", "area", "volume", "mass", "weight", "time", "duration",
    "period", "pressure", "force", "percent", "percentage", "price", "cost", "sum"))


#: A trailing "what do X and Y each say?" clause asks what each SOURCE
#: says; its words name the sources, not the thing the values belong to.
_SOURCE_ASK_RE = re.compile(r"\b(?:what|which)\s+(?:do|does)\b", re.IGNORECASE)
_SOURCE_SAY_RE = re.compile(
    r"\b(?:each\s+)?(?:say|says|state|states|specify|specifies|give|gives|show|shows|indicate|indicates)\s*\??$",
    re.IGNORECASE)


def _drop_source_query(question: str) -> str:
    """The question with a trailing source query ("what do X and Y each
    say?") removed; unchanged when it asks no such thing."""
    text = question or ""
    for m in reversed(list(_SOURCE_ASK_RE.finditer(text))):
        if _SOURCE_SAY_RE.search(text[m.start():]):
            return text[: m.start()].rstrip()
    return text


def _question_thing(question: str) -> set[str]:
    """The thing the question names: its content words, less the words that
    only name the kind of value and the words of a trailing source query. A
    kind word ("grade", "date", "price") and the single word that runs
    straight into it ("concrete grade", "completion date") name the kind; a
    question made of nothing else names no thing, and two values found for
    it may be about two different elements."""
    question = _drop_source_query(question)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", question or "") if len(w) > 1]
    kind_phrase: set[str] = set()
    for i, w in enumerate(words):
        if _stem(w) not in _KIND_TERMS:
            continue
        j = i - 1
        if j >= 0 and words[j].lower() not in _STOP and _stem(words[j]) not in _KIND_TERMS:
            kind_phrase.add(_stem(words[j]))
    return _terms(question) - _KIND_TERMS - kind_phrase


#: A sentence that tells the user the documents agree, are consistent or
#: match. An answer that says so while it states two differing values has
#: not told the user they differ.
_AGREE_RE = re.compile(
    r"\b(?:agree|agrees|agreed|agreement|consistent|consistently|identical|match|matches|"
    r"no\s+(?:conflict|conflicts|discrepanc(?:y|ies)|difference|differences))\b", re.IGNORECASE)


#: A negation just before the agreement word turns it round ("do not
#: agree", "are not consistent", "never match"): that tells the user they differ.
_NEGATED_RE = re.compile(r"(?:\bnot|n't|\bnever|\bno\s+longer)(?:\s+\w+){0,2}\s*$", re.IGNORECASE)


def asserts_agreement(sentence: str) -> bool:
    """The sentence tells the user the documents agree: an agreement word
    (``_AGREE_RE``) that no negation stands just before."""
    s = sentence or ""
    return any(not _NEGATED_RE.search(s[: m.start()]) for m in _AGREE_RE.finditer(s))


def _claims_agreement(answer: str) -> bool:
    return any(asserts_agreement(s) for s in _sentences(answer))


#: Words a sentence uses to point at the evidence itself, not at a thing in
#: it ("the sources are consistent", "both documents match").
_EVIDENCE_TERMS = frozenset(_stem(w) for w in (
    "source", "document", "record", "evidence", "file", "reference", "both", "two", "they", "point",
    "between"))


def agreement_names_no_thing(sentence: str, documents: Iterable[str] = ()) -> bool:
    """An agreement sentence about no other thing: it states no typed value,
    and its words, less the agreement words, the words pointing at the
    evidence and the words of the documents' own names, are empty."""
    if any(v.kind != "text" for v in extract(sentence)):
        return False
    rest = _terms(_AGREE_RE.sub(" ", sentence or "")) - _EVIDENCE_TERMS
    for doc in documents:
        rest -= _terms(re.sub(r"\.\w{2,4}$", "", doc or ""))
    return not rest


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text or "")
            if w.lower() not in _STOP and len(w) > 1}


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\d])|\n+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


# ── Values ──────────────────────────────────────────────────────────────────

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_END = r"(?![A-Za-z0-9²³/])"

#: unit pattern -> (kind, low factor, high factor) into the kind's base unit.
_UNITS: list[tuple[str, str, float, float]] = [
    (r"N/mm(?:2|²)|MPa", "pressure", 1.0, 1.0),
    (r"kPa", "pressure", 1e-3, 1e-3),
    (r"GPa", "pressure", 1e3, 1e3),
    (r"kN", "force", 1.0, 1.0),
    (r"m(?:3|³)|cu\.?\s?m|cubic\s+met(?:re|er)s?", "volume", 1.0, 1.0),
    (r"m(?:2|²)|sq\.?\s?m|square\s+met(?:re|er)s?", "area", 1.0, 1.0),
    (r"mm|millimet(?:re|er)s?", "length", 1e-3, 1e-3),
    (r"cm|centimet(?:re|er)s?", "length", 1e-2, 1e-2),
    (r"km|kilomet(?:re|er)s?", "length", 1e3, 1e3),
    (r"m|met(?:re|er)s?", "length", 1.0, 1.0),
    (r"tonnes?|tons?|t", "mass", 1.0, 1.0),
    (r"kg|kilograms?", "mass", 1e-3, 1e-3),
    (r"(?:calendar\s+|working\s+)?days?", "time", 1.0, 1.0),
    (r"weeks?|wks?", "time", 7.0, 7.0),
    (r"months?", "time", 28.0, 31.0),
    (r"years?|yrs?", "time", 365.0, 366.0),
    (r"%|per\s?cent", "percent", 1.0, 1.0),
]
#: A number glued to an ASCII letter, a hyphen, a slash or a point is part of
#: a code (``IP-INF-053``, ``A-101``, ``3.2.4``), never a quantity.
_FREE = r"(?<![0-9A-Za-z_\-/.])"
_QTY_RE = re.compile(
    rf"{_FREE}(?P<num>{_NUM})\s?(?P<unit>" + "|".join(f"(?:{u})" for u, *_ in _UNITS) + rf"){_END}")
#: A range of one kind is one value covering both ends: "150-200 mm",
#: "150 to 200 mm", "between 150 and 200 mm", "1.5 m to 2 m". A value inside
#: it is the same value; a range is never read as its upper end alone.
_UNIT_ALT = "|".join(f"(?:{u})" for u, *_ in _UNITS)
_RANGE_RE = re.compile(
    rf"(?:(?P<between>\b(?i:between|from))\s+)?{_FREE}(?P<a>{_NUM})\s?(?P<ua>{_UNIT_ALT})?"
    rf"\s*(?P<sep>-|–|—|(?i:to|and))\s*(?P<b>{_NUM})\s?(?P<unit>{_UNIT_ALT}){_END}")
_UNIT_RES = [(re.compile(rf"^(?:{u})$", re.IGNORECASE if k in ("time", "percent") else 0), k, lo, hi)
             for u, k, lo, hi in _UNITS]

_CURRENCY = (r"AED|USD|EUR|GBP|SAR|QAR|KWD|OMR|BHD|INR|EGP|JOD|CAD|AUD|NZD|CHF|JPY|CNY|SGD|HKD|ZAR|"
             r"US\$|\$|£|€")
_MONEY_RE = re.compile(
    rf"(?P<cur>{_CURRENCY})\s?(?P<num>{_NUM})(?:\s?(?P<mult>(?i:million|mn|billion|bn|thousand)))?(?![A-Za-z0-9])")
_MULT = {"million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "thousand": 1e3}

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], start=1)}
_MON = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|" \
       r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_DATE_RES = [
    re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mon>{_MON})\.?,?\s+(?P<y>\d{{4}})\b"),
    re.compile(rf"\b(?P<mon>{_MON})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})\b"),
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b"),
    re.compile(r"\b(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})\b"),
]

_GRADE_RES = [
    re.compile(r"\b(?:grade|class)\s+(?P<g>[A-Z]{1,3}\s?\d{1,3}(?:/\d{1,3})?[A-Z]?)(?![\w/])", re.IGNORECASE),
    re.compile(r"\b(?P<g>[A-Z]{1,2}\d{2,3}/\d{2,3})(?![\w/])"),
]

#: ``Label: short value`` -- the value runs to the first comma, semicolon or full stop.
_LABELLED_RE = re.compile(r"^(?P<label>[^:]{2,80}):\s*(?P<val>[^,;.:]{2,80}?)\s*(?:[,;.]|$)")


@dataclass
class Value:
    raw: str          # as written
    kind: str         # group key: comparable values share it
    norm: Any         # normalised: (low, high) | iso date | canonical text
    start: int
    end: int


def _unit(unit: str) -> tuple[str, float, float] | None:
    for urx, kind, lo, hi in _UNIT_RES:
        if urx.match(unit):
            return kind, lo, hi
    return None


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _month(name: str) -> int:
    low = name.lower().rstrip(".")
    for full, i in _MONTHS.items():
        if full.startswith(low[:3]):
            return i
    return 0


def _valid_date(y: int, m: int, d: int) -> bool:
    import calendar

    return 1 <= m <= 12 and 1 <= d <= calendar.monthrange(y, m)[1]


_WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*[A-Za-z]")


def _claim_text(val: str) -> bool:
    """A labelled phrase that is a claim: it holds a word and no code. A bare
    number, a page or row number and a document or drawing code (a token
    mixing digits with letters, hyphens, slashes or points) are not."""
    tokens = [t.strip("()[]\"'*_,") for t in val.split()]
    if not any(_WORD_TOKEN_RE.fullmatch(t) for t in tokens):
        return False
    return not any(re.search(r"\d", t) and re.search(r"[A-Za-z\-/.]", t) for t in tokens)


def extract(sentence: str) -> list[Value]:
    """Typed values in one sentence, left to right; a span is used once.
    Citation spans are blanked first (same positions), so a figure inside a
    citation, a retrieval key or a numbered marker is never a value."""
    from app.lib.citation_spans import blank_citations

    sentence = blank_citations(sentence)
    found: list[Value] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def add(v: Value) -> None:
        if free(v.start, v.end):
            taken.append((v.start, v.end))
            found.append(v)

    for rx in _DATE_RES:
        for m in rx.finditer(sentence):
            gd = m.groupdict()
            y, d = int(gd["y"]), int(gd["d"])
            mo = _month(gd["mon"]) if gd.get("mon") else int(gd["m"])
            if rx is _DATE_RES[3] and d <= 12 and mo <= 12 and d != mo:
                continue  # dd/mm or mm/dd: ambiguous, no claim
            if _valid_date(y, mo, d):
                add(Value(m.group(0), "date", f"{y:04d}-{mo:02d}-{d:02d}", m.start(), m.end()))
    for m in _MONEY_RE.finditer(sentence):
        cur = m.group("cur").upper().replace("US$", "$")
        amt = _num(m.group("num")) * _MULT.get((m.group("mult") or "").lower(), 1.0)
        add(Value(m.group(0), f"money:{cur}", (amt, amt), m.start(), m.end()))
    for rx in _GRADE_RES:
        for m in rx.finditer(sentence):
            g = m.group("g")
            add(Value(g, "grade", re.sub(r"\s+", "", g).upper(), m.start("g"), m.end("g")))
    for m in _RANGE_RE.finditer(sentence):
        if m.group("sep").lower() == "and" and not m.group("between"):
            continue  # "A and B mm" is two values, not a range
        hit = _unit(m.group("unit"))
        first = _unit(m.group("ua")) if m.group("ua") else hit
        if hit and first and first[0] == hit[0]:
            a, b = _num(m.group("a")), _num(m.group("b"))
            ends = (a * first[1], a * first[2], b * hit[1], b * hit[2])
            add(Value(m.group(0), hit[0], (min(ends), max(ends)), m.start(), m.end()))
    for m in _QTY_RE.finditer(sentence):
        unit = m.group("unit")
        for urx, kind, lo, hi in _UNIT_RES:
            if urx.match(unit):
                n = _num(m.group("num"))
                add(Value(m.group(0), kind, (n * lo, n * hi), m.start(), m.end()))
                break
    found.sort(key=lambda v: v.start)
    if not found:
        lab = _LABELLED_RE.match(sentence)
        if lab and len(lab.group("val").split()) <= 6 and _claim_text(lab.group("val")):
            val = lab.group("val").strip()
            found.append(Value(val, "text", " ".join(val.split()).casefold(), lab.start("val"), lab.end("val")))
    return found


def same(a: Value, b: Value) -> bool:
    if a.kind != b.kind:
        return False
    if isinstance(a.norm, tuple):
        (alo, ahi), (blo, bhi) = a.norm, b.norm
        tol = 1e-9 * max(abs(ahi), abs(bhi), 1.0)
        return alo <= bhi + tol and blo <= ahi + tol
    return a.norm == b.norm


# ── Candidates in the evidence ──────────────────────────────────────────────

@dataclass
class _Cand:
    doc: str
    row: dict[str, Any]
    value: Value
    sent_terms: set[str]
    local_terms: set[str]
    sent_key: tuple[int, int]


def _row_doc(row: dict[str, Any]) -> str:
    return str(row.get("doc_name") or row.get("document") or "").strip()


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("snippet") or row.get("passage") or row.get("text") or "")


#: A passage "sentence" can be a whole table row, a heading path or a page of
#: text with no full stop. A value's claim is its own clause: the passage is
#: cut at a heading-path step, a semicolon or a line break. A clause holding
#: one value is named by its whole clause; a clause holding several values
#: (a long table row or enumeration) is named by the words near each value.
_CLAUSE_SPLIT_RE = re.compile(r"\s+>\s+|;\s*|\n+")
_WINDOW_WORDS = 12


def _clauses(text: str) -> list[str]:
    return [c.strip() for s in _sentences(text) for c in _CLAUSE_SPLIT_RE.split(s) if c.strip()]


def _window(sent: str, start: int, end: int) -> str:
    """The value's own neighbourhood: up to ``_WINDOW_WORDS`` words on each
    side, inside its clause."""
    before = sent[:start].split()[-_WINDOW_WORDS:]
    after = sent[end:].split()[:_WINDOW_WORDS]
    return " ".join(before) + " " + " ".join(after)


def _candidates(rows: list[dict[str, Any]]) -> list[_Cand]:
    out: list[_Cand] = []
    for ri, row in enumerate(rows):
        doc = _row_doc(row)
        if not doc:
            continue
        for si, sent in enumerate(_clauses(_row_text(row))):
            vals = extract(sent)
            if not vals:
                continue
            masked = sent
            for v in sorted(vals, key=lambda v: -v.start):
                masked = masked[: v.start] + " " * (v.end - v.start) + masked[v.end:]
            for i, v in enumerate(vals):
                lo = vals[i - 1].end if i else 0
                hi = vals[i + 1].start if i + 1 < len(vals) else len(sent)
                local = sent[lo: v.start] + " " + sent[v.end: hi]
                if len(vals) == 1:
                    near = _terms(masked)
                else:
                    near = _terms(_window(masked, v.start, v.end))
                out.append(_Cand(doc, row, v, near, _terms(local) & near, (ri, si)))
    return out


def _doc_values(cands: list[_Cand], query: set[str]) -> dict[str, tuple[int, _Cand]]:
    """Per document: its best-matched value for the asked thing, or nothing
    when two different values tie for it."""
    by_doc: dict[str, list[tuple[int, _Cand]]] = {}
    for c in cands:
        by_doc.setdefault(c.doc, []).append((len(c.sent_terms & query), c))
    chosen: dict[str, tuple[int, _Cand]] = {}
    for doc, scored in by_doc.items():
        best = max(s for s, _ in scored)
        if best < 1:
            continue
        top = [c for s, c in scored if s == best]
        # Two values in one sentence: the words next to each value decide.
        loc = max(len(c.local_terms & query) for c in top)
        top = [c for c in top if len(c.local_terms & query) == loc] if loc else top
        if all(same(top[0].value, c.value) for c in top[1:]):
            chosen[doc] = (best, top[0])
    return chosen


# ── Precedence (C3) ─────────────────────────────────────────────────────────

_PREC_RE = re.compile(r"\b(?:precedence|prevail|prevails|govern|governs|priority)\b", re.IGNORECASE)
_WIN_RE = re.compile(r"\b(?:shall\s+|will\s+|is\s+to\s+)?(?:take|takes|have|has)?\s*"
                     r"(?:precedence|prevail|prevails|govern|governs|priority)\b", re.IGNORECASE)
_PHRASE_RE = re.compile(r"[A-Z][\w-]*(?:\s+(?:(?:of|for)\s+(?:the\s+)?)?[A-Z][\w-]*)*")


def _phrases(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in _PHRASE_RE.finditer(text) if _terms(m.group(0))]


def _ranking(sentence: str) -> list[tuple[int, str]]:
    """(rank, kind phrase) from one precedence statement; empty if none."""
    if not _PREC_RE.search(sentence):
        return []
    if re.search(r"\border\b", sentence, re.IGNORECASE) and ":" in sentence:
        tail_at = sentence.index(":") + 1
        listed = [p for s, _e, p in _phrases(sentence[tail_at:])]
        return list(enumerate(listed)) if len(listed) >= 2 else []
    win = _WIN_RE.search(sentence)
    if not win:
        return []
    before = [p for p in _phrases(sentence) if p[1] <= win.start()]
    if not before:
        return []
    winner = before[-1][2]
    others = [p for _s, _e, p in _phrases(sentence) if p != winner]
    return [(0, winner)] + [(1, p) for p in others] if others else []


def _names(phrase: str, doc: str) -> bool:
    want = _terms(phrase)
    return bool(want) and want <= _terms(re.sub(r"\.\w{2,4}$", "", doc))


def _precedence(docs: list[str], rows: list[dict[str, Any]]) -> tuple[str | None, str | None, str, str]:
    """(governing document, basis document, winning kind, lower kind) or Nones."""
    verdicts = set()
    found: tuple[str | None, str | None, str, str] = (None, None, "", "")
    for row in rows:
        for sent in _sentences(_row_text(row)):
            ranking = _ranking(sent)
            if not ranking:
                continue
            rank_of: dict[str, tuple[int, str]] = {}
            for doc in docs:
                hits = [(r, p) for r, p in ranking if _names(p, doc)]
                if hits:
                    rank_of[doc] = min(hits)
            if len(rank_of) != len(docs):
                continue  # the statement does not cover every document in the conflict
            top = min(r for r, _p in rank_of.values())
            winners = [d for d in docs if rank_of[d][0] == top]
            if len(winners) != 1:
                continue
            loser = min((rank_of[d] for d in docs if d not in winners), default=(0, ""))[1]
            found = (winners[0], _row_doc(row), rank_of[winners[0]][1], loser)
            verdicts.add(winners[0])
    if len(verdicts) != 1:
        return None, None, "", ""
    return found


# ── Detection ───────────────────────────────────────────────────────────────

def _says_text(norm: str, text: str) -> bool:
    """A labelled phrase is said when it stands as whole words in the text
    (citation spans blanked), never as a piece of another word or number."""
    from app.lib.citation_spans import blank_citations

    hay = " ".join(blank_citations(text).split()).casefold()
    return bool(norm) and re.search(rf"(?<![\w-]){re.escape(norm)}(?![\w-])", hay) is not None


def _states_answer(c: _Cand, sent: str, typed: list[Value], kind: str) -> bool:
    if kind == "text":
        return _says_text(c.value.norm, sent)
    return any(same(c.value, a) for a in typed if a.kind == kind)


def _same_thing(kept: list[_Cand], sent: str, typed: list[Value], kind: str, asked: set[str],
                from_question: bool = False) -> list[_Cand]:
    """Only the values that belong to the asked thing. The thing is the asked
    words that every document stating the answer's value uses for it; a value
    from a sentence that does not name all of them is about something else
    that merely shares words (another element, section or revision)."""
    anchors = [c for c in kept if _states_answer(c, sent, typed, kind)]
    # The thing is named by the anchor that names the most asked words: a
    # second, loosely worded passage stating the same value never shrinks the
    # thing to a word other things share. A word that only names the value's
    # own kind ("date" for a date) is said by the value itself.
    named = [(len(asked & c.sent_terms), c) for c in anchors]
    best = max((n for n, _c in named), default=0)
    thing = set(asked) - _KIND_TERMS
    for n, c in named:
        if n == best:
            thing &= c.sent_terms
    need = _named_enough(len(set(asked) - _KIND_TERMS)) if from_question else 1
    if len(thing) < need:
        # One word shared with the question is not the thing the question
        # names: two values for two things that share a word are no conflict.
        return []
    return [c for c in kept if thing <= c.sent_terms]


def _named_enough(asked: int) -> int:
    """How many of the asked thing's words (``asked`` >= 1: the caller holds a
    non-empty question thing) a value's own words must name: a third of
    them, and at least two when the question names two or more."""
    return min(asked, max(2, -(-asked // 3)))


def _thing(kept: list[_Cand], asked: set[str]) -> list[str]:
    common = set(asked)
    for c in kept:
        common &= c.sent_terms
    return sorted(common)


def detect(answer: str, rows: Iterable[dict[str, Any]] | None, question: str | None = None) -> list[dict[str, Any]]:
    """Conflicts over the evidence for the values the answer states.

    Each: ``{"values": [{"value", "document", "row"}], "governs", "basis_document",
    "governing_kind", "lower_kind"}``. ``row`` is the evidence row the value
    came from (for its citation)."""
    try:
        rows = [r for r in (rows or []) if isinstance(r, dict)]
        if len({_row_doc(r) for r in rows if _row_doc(r)}) < 2 or not isinstance(answer, str):
            return []
        cands = _candidates(rows)
        if not cands:
            return []
        said = extract(answer)
        q_terms = _question_thing(question) if question else set()
        if question and not q_terms:
            return []  # the question names only the kind of value: no thing to compare
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, frozenset]] = set()
        for sent in _sentences(answer):
            avals = extract(sent)
            typed = [v for v in avals if v.kind != "text"]
            kinds = {v.kind for v in typed} or {"text"}
            masked = sent
            for v in sorted(typed, key=lambda v: -v.start):
                masked = masked[: v.start] + " " + masked[v.end:]
            query = q_terms | _terms(masked)
            if not query:
                continue
            for kind in sorted(kinds):
                chosen = _doc_values([c for c in cands if c.value.kind == kind], query)
                if len(chosen) < 2:
                    continue
                top = max(s for s, _c in chosen.values())
                kept = [c for s, c in chosen.values() if 2 * s >= top]
                if len(kept) < 2:
                    continue
                if kind == "text":
                    anchored = any(_says_text(c.value.norm, sent) for c in kept)
                else:
                    anchored = any(same(c.value, a) for c in kept for a in typed if a.kind == kind)
                if not anchored:
                    continue
                kept = _same_thing(kept, sent, typed, kind, q_terms or query, from_question=bool(q_terms))
                classes: list[_Cand] = []
                for c in kept:
                    if not any(same(c.value, k.value) for k in classes):
                        classes.append(c)
                if len(classes) < 2:
                    continue
                if kind == "text":
                    told = all(_says_text(c.value.norm, answer) for c in classes)
                else:
                    told = all(any(same(c.value, a) for a in said if a.kind == kind) for c in classes)
                if told and not _claims_agreement(answer):
                    continue  # the answer already states every differing value as differing
                key = (kind, frozenset(str(c.value.norm) for c in classes))
                if key in seen:
                    continue
                seen.add(key)
                order = {id(r): i for i, r in enumerate(rows)}
                kept.sort(key=lambda c: order.get(id(c.row), 0))
                gov, basis, gkind, lkind = _precedence([c.doc for c in kept], rows)
                if gov is not None:
                    gval = next(c.value for c in kept if c.doc == gov)
                    if not all(same(gval, c.value) for c in kept if _names(gkind, c.doc)):
                        gov = basis = None
                out.append({
                    "kind": kind,
                    "thing": _thing(kept, q_terms or query),
                    "values": [{"value": c.value.raw, "document": c.doc, "row": c.row, "typed": c.value}
                               for c in kept],
                    "governs": gov,
                    "basis_document": basis,
                    "governing_kind": gkind if gov else "",
                    "lower_kind": lkind if gov else "",
                })
        return out
    except Exception:  # noqa: BLE001 -- detection must never break an answer
        _LOG.exception("evidence_conflicts: failed; no conflict reported")
        return []
