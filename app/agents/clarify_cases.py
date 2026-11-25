"""Clarify-before-choosing: the exit mechanism's decision (PR 2 part b).

The owner's 15:03Z rule (19 Nov UTC): when what the product holds supports
more than one materially different answer to the question as asked, and the
question does not say which case is meant, the turn ends as ONE short
clarifying question that names the cases it actually found, each with its
source. The next turn answers the case the user picks.

This module is the DETECTION half and is deliberately pure: no retrieval, no
model, no I/O. ``clarifying_question`` takes the question as asked, the draft
answer, and the cases already read from the handed evidence, and returns the
clarifying question (or ``None`` when the turn should stand as written). It
is tested against recorded strings, the way ``missing_input`` is.

A ``Case`` is one value claim in the handed passages:

- ``value``: a number with its unit, or a clause / document-id / rate
  reference (read by the typed readers already in the product);
- ``subject``: the noun phrase the passage's OWN sentence, heading, or detail
  title names for that value -- read from the passage's own words, never from
  a list of known structures, elements, locations, depths, sizes, materials,
  or stages;
- ``source``: document name (and page/chunk) from the evidence records.

The value+subject+source EXTRACTION from evidence is the shared reader's job
(``citation_provenance``'s figure/sentence primitives) and is wired in the
caller. This module never lists a known subject; it only compares the subjects
it is handed. Passing cases in keeps the decision testable.

Bounds (each one a test, below):

1. One clarifying question per turn, never a chain.
2. Only when the cases lead to DIFFERENT answers (same value in every case:
   answer).
3. Never when the question already names the case.
4. Never to ask for something the documents state plainly (no invented
   question).
5. Options listed are only cases actually found, each with its source.
6. "all" / no choice is the caller's next turn (every case side by side), not
   this function's output -- the question is still one question.
7. A question that is merely broad is answered, not bounced back (no value
   conflict -> this detector never fires).

Case 2 vs case 3: different values for DIFFERENT subjects, question names
none -> ask (this mechanism). Different values for the SAME subject -> the
both-values statement (#453, on main). When the passages do not let the
product tell them apart, it behaves as case 2 (it asks).

Kill-switch: ``CLARIFY_CASES=0``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

FLAG_ENV = "CLARIFY_CASES"

#: A number token: plain, comma-grouped, or with one decimal part. Clause ids
#: such as "8.8.1" and ids such as "AB-2001-101" parse as MORE than one number
#: and are therefore compared as strings, never as floats.
_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")

#: Words that do not identify a subject.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or",
    "with", "without", "by", "from", "per", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "this", "that", "these", "those", "it", "its",
    "their", "our", "your", "his", "her", "which", "what", "when",
    "where", "who", "whom", "how", "not", "no", "yes", "as", "than",
    "then", "there", "here", "each", "any", "some", "all", "every", "one",
    "i", "me", "we", "us", "you", "he", "him", "she", "they", "them",
}

#: Month names and abbreviations. A calendar date ("Jan-25", "30-Jan-25")
#: matches the letter-number reference SHAPE but is not a clause or document
#: reference; these mark the letter-number codes that are dates so the reader
#: skips them. A month is a language category, not a project or test case.
_MONTH_WORDS = frozenset({
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
})


def _is_date_token(value: str) -> bool:
    """True when a letter-number token is a calendar date, not a reference."""
    lead = re.match(r"[A-Za-z]+", (value or "").strip())
    return bool(lead) and lead.group(0).lower() in _MONTH_WORDS


def enabled() -> bool:
    """ON by default; ``CLARIFY_CASES=0`` is the kill switch."""
    return str(os.getenv(FLAG_ENV, "1")).strip().lower() not in (
        "0", "false", "no", "off",
    )


@dataclass(frozen=True)
class Case:
    """One value claim in the handed passages."""

    subject: str
    value: str
    source: str
    kind: str = ""   # "measure" | "clause" | "code"


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _numbers(value: str) -> list[float]:
    out: list[float] = []
    for m in _NUMBER_RE.finditer(value or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


def _number_of(value: str) -> float | None:
    """The single float a value carries, else None when it has none, or more
    than one (a clause / document id compared as a string)."""
    nums = _numbers(value)
    return nums[0] if len(nums) == 1 else None


def _unit_of(value: str) -> str:
    """The non-numeric remainder of a value ("mm" in "200 mm"), normalised."""
    for m in _NUMBER_RE.finditer(value or ""):
        value = value[: m.start()] + " " + value[m.end() :]
    return _norm(value)


def _subject_tokens(subject: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{3,}", (subject or "").lower())
        if w not in _STOPWORDS
    }


def _same_subject(a: str, b: str) -> bool:
    """True only when two subjects are confidently the same phrase.

    Conservative: the default is "different" (which asks), because collapsing
    two different structures into one case is the exact per-case miss the
    standing order forbids. Same only when one token set is a subset of the
    other and they share a content token of length >= 4 (a near-identical
    phrase, not two phrases that merely share one word).
    """
    ta, tb = _subject_tokens(a), _subject_tokens(b)
    if not ta or not tb:
        return _norm(a) == _norm(b)
    smaller, larger = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not smaller <= larger:
        return False
    return any(len(t) >= 4 for t in smaller)


def _question_names_case(question: str, case: Case) -> bool:
    """Bound 3: the question already names its case. Conservative -- only
    treated as named when clearly so, so a vague ask still asks."""
    q = _norm(question)
    value = _norm(case.value)
    if value and value in q:
        return True
    tokens = {t for t in _subject_tokens(case.subject) if len(t) >= 4}
    return bool(tokens) and all(t in q for t in tokens)


def _text_commits(text: str, value: str) -> bool:
    """True when the draft states this case's value."""
    n = _number_of(value)
    if n is not None:
        for found in _numbers(text):
            if abs(found - n) <= 1e-3 * max(1.0, abs(found), abs(n)):
                return True
        return False
    return _norm(value) in _norm(text)


#: Prepositions that start a qualifier ("cover to reinforcement", "fall of
#: concrete"): the head noun is the last content word before the first one.
_PREPOSITIONS = frozenset({
    "of", "at", "in", "on", "for", "to", "with", "from", "during", "per",
    "above", "below", "between", "under", "over", "by", "into", "onto",
    "through", "across", "within",
})

#: Measurable-quantity nouns. A question naming one of these asks for a
#: measure (a number with a unit), not a class or a clause. This is a
#: language category (the things English speakers measure), never a list of
#: structures, materials, locations, projects or test cases.
_MEASURE_NOUNS = frozenset({
    "length", "rate", "thickness", "depth", "height", "width", "diameter",
    "temperature", "strength", "pressure", "head", "duration", "leakage",
    "area", "volume", "weight", "size", "bore", "cover", "fall", "spacing",
    "frequency", "slump", "distance", "level",
})

#: Value-kind words that take an "of" object: "what class of X" asks for the
#: class (a code) OF X, so X is the subject and the kind is "code".
_TYPE_OF = frozenset({"class", "grade", "type"})

#: The words that mark what kind of value is asked for, used to find the
#: subject of a "what <modifier> <kind>" question ("what concrete grades",
#: "what reinforcement grade").
_KIND_WORDS = frozenset(
    _MEASURE_NOUNS | _TYPE_OF
    | {"clause", "classes", "grades", "types", "covers", "lengths"}
)

#: Subject heads that do not narrow a question ("what is the value?" asks
#: for everything). A generic head means the filter keeps every case.
_GENERIC_HEADS = frozenset({
    "value", "figure", "number", "answer", "amount", "quantity",
    "requirement", "case",
    # a bare kind word ("which clause applies?") does not narrow the cases
    "clause", "class", "grade", "type",
})

#: Verbs that end the ask's noun phrase ("which clause APPLIES?", "what
#: strength MUST block walls have"). The noun phrase is the words before
#: the first one.
_VERB_WORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "must", "shall", "should", "will", "would", "may", "might",
    "can", "could", "do", "does", "did", "have", "has", "had",
    "applies", "apply", "applied", "governs", "govern", "fixes", "fix",
    "states", "state", "specifies", "specify", "requires", "require",
    "says", "say", "means", "mean", "reads", "read", "receives", "receive",
})

#: A clause value: numbers joined by dots ("8.8.1"), as opposed to a
#: letter-number code ("D400", "C35/45").
_CLAUSE_VALUE_RE = re.compile(r"^\d+(?:\.\d+){2,}$")


def _is_document_id(value: str) -> bool:
    """True for a document / contract / drawing id ("DOC-2024-001",
    "ABC-0000-XYZ-000001"): a letter-number code with two or more
    dash-separated segments. A document id is a SOURCE (already carried by
    ``Case.source``), not an answer value, so the reader skips it. Shape
    only, never a known id."""
    return (value or "").count("-") >= 2


def _head_noun(phrase: str) -> str:
    """The head noun of a short phrase: the last content word before the
    first preposition ("cover to reinforcement" -> "cover"; "fall of
    concrete" -> "fall"), or the last content word when there is none
    ("temperature limit" -> "limit")."""
    words = [w.strip(".,;:()\"'") for w in (phrase or "").lower().split()]
    idx = next((i for i, w in enumerate(words) if w in _PREPOSITIONS), len(words))
    for w in reversed(words[:idx]):
        if w not in _STOPWORDS and len(w) >= 3:
            return w
    return ""


def _thing_of(subject: str) -> tuple[str, set[str]]:
    """The thing a case's own phrase names: ``(head noun, words before it)``.

    A phrase the reader cut at the value's own kind word ("BLINDING CONCRETE:
    Grade" for a grade, "SLAB: Thickness" for a thickness) names its thing in
    the words before that word, so a trailing kind word is dropped and the
    head is read from what is left. The kind words are the module's existing
    language category, never a list of subjects.
    """
    words = [w for w in (subject or "").lower().split() if w.strip(".,;:()") not in _STOPWORDS]
    while words and words[-1].strip(".,;:()") in _KIND_WORDS:
        words.pop()
    rest = " ".join(words)
    return _head_noun(rest), _subject_tokens(rest)


#: Reporting verbs that make an ask a request to summarise / describe /
#: report the documents rather than to name one value. Such an ask is
#: answered by describing several different subjects, so its values are
#: never alternatives the detector is asked to disambiguate. A language
#: category (reporting verbs), never a list of structures, documents or
#: test cases.
_SUMMARY_VERBS = frozenset({
    "say", "says", "said", "state", "states", "stated",
    "describe", "describes", "described",
    "summarise", "summarize", "summarises", "summarizes",
    "explain", "explains", "explained",
    "list", "lists", "listed", "outline", "outlines", "outlined",
    "detail", "details", "detailed", "mention", "mentions", "mentioned",
    "discuss", "discusses", "discussed", "report", "reports", "reported",
})

#: Generic value words: naming one keeps an ask a value ask even when a
#: reporting verb also appears ("what value does the spec state?").
_GENERIC_VALUE_WORDS = frozenset({
    "value", "values", "figure", "figures", "number", "numbers",
    "amount", "amounts", "quantity", "quantities", "answer", "answers",
})


def _is_summary_ask(question: str) -> bool:
    """True when the ask is to report/summarise/describe, not to name a value.

    An imperative ("summarise the boundary fencing requirements") or a
    "what does X say about Y" ask is answered by describing the documents,
    never by one value, so the clarify-cases detector must stand down.
    Naming a value (a kind word is handled by the caller, a generic value
    word here) keeps it a value ask.
    """
    words = re.findall(r"[a-z]+", (question or "").lower())
    if not any(w in _SUMMARY_VERBS for w in words):
        return False
    return not any(w in _GENERIC_VALUE_WORDS for w in words)


#: Words that compare the asked thing with a stated figure ("is the rate
#: HIGHER than 5", "is the depth ABOVE 2 m"). In a yes/no ask they end the
#: noun phrase that names the thing. A language category, never a case.
_COMPARISON_WORDS = frozenset({
    "higher", "lower", "greater", "smaller", "larger", "bigger", "more",
    "less", "fewer", "above", "below", "over", "under", "exceed",
    "exceeds", "within", "between", "equal", "than", "least", "most",
    "enough", "sufficient", "same", "different",
})


def _yes_no_subject(q: str) -> str | None:
    """The thing a yes/no ask names: "is the <thing> <figure or comparison>".

    "is the cover 40 mm" names "cover"; "is the unit rate of X higher than
    5 USD/kg" names "unit rate of X". The phrase runs from after the
    auxiliary (and its article) to the first word that carries a digit or
    compares, so the asked thing narrows the cases the way a "what is the
    <thing>" ask does. An ask with no such opening reads nothing (None).
    """
    m = re.match(r"(?:is|are|was|were|does|do|did)\s+(?:the\s+|a\s+|an\s+)?(.+)$", q)
    if not m:
        return None
    raw = m.group(1).split()
    bares = [w.strip(".,;:()\"'") for w in raw]
    words: list[str] = []
    for i, bare in enumerate(bares):
        if (
            any(ch.isdigit() for ch in bare)
            or bare in _COMPARISON_WORDS
            or _compares_by_than(bares, i)
        ):
            break
        words.append(raw[i])
    return " ".join(words) or None


def _compares_by_than(bares: list[str], i: int) -> bool:
    """True when word ``i`` is a comparative because "than" follows it
    ("wider than", "thinner or thicker than"). English marks the
    comparison by its grammar, so no list of comparatives is needed."""
    nxt = bares[i + 1:i + 4]
    if nxt[:1] == ["than"]:
        return True
    return len(nxt) == 3 and nxt[0] in ("or", "and") and nxt[2] == "than"


def _question_asks(question: str) -> tuple[set[str], set[str], set[str]] | None:
    """What the question asks for, as ``(kinds, heads, qualifiers)``.

    ``kinds`` is a subset of {measure, clause, code}; ``heads`` is the set of
    head nouns to match against case subjects; ``qualifiers`` is the set of
    non-head content words the question's subject names (the words that tell
    one case of the same measure apart from another: "fresh concrete" vs
    "ambient"). Returns ``("none", set(), set())`` for a non-value ask (a
    party, an action, or a request to summarise/describe the documents:
    "who", "what must happen", "summarise the requirements"), and ``None``
    for a generic question that does not narrow the cases ("what is the
    value?").
    """
    q = (question or "").lower().strip().rstrip("?.!")
    if ":" in q:
        q = q.split(":")[-1].strip()

    # A party or an action is not a value to compare.
    if re.search(r"\b(?:who|whom)\b", q) or re.search(
        r"\bwhat (?:must|should|will|would|shall) happen\b", q
    ):
        return ("none", set(), set())

    kinds: set[str] = set()
    if re.search(r"\bhow (?:many|much|long|often|far|deep|high|thick|wide|fast)\b", q):
        kinds.add("measure")
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m\b|m2|m3|kg|kn|mpa|n/mm2|bar|pa|%|°c|°f)\b", q):
        kinds.add("measure")
    if re.search(r"\b(?:class|classes|grade|grades|type|types)\b", q):
        kinds.add("code")
    if re.search(r"\bclause\b", q):
        kinds.add("clause")
    for w in _MEASURE_NOUNS:
        if re.search(r"\b" + re.escape(w) + r"\b", q):
            kinds.add("measure")

    # A summary / description ask names no single value the evidence can
    # disagree on; it is answered by describing several subjects, so the
    # clarify-cases detector must not turn its values into alternatives.
    if not kinds and _is_summary_ask(q):
        return ("none", set(), set())

    # The subject phrase, from the question's own words.
    subj = None
    m = re.search(r"\bwhat(?:'s| is| are)\s+the\s+(.+)$", q)
    if m:
        subj = m.group(1)
    else:
        m = re.search(r"\b(?:what|which)\s+(\w+)\s+of\s+(.+)$", q)
        if m and m.group(1) in _TYPE_OF:
            subj = m.group(2)
        else:
            # "what <modifier> <kind>" ("what concrete grades", "what
            # reinforcement grade"): the modifier is the subject.
            m = re.search(
                r"\b(?:what|which)\s+(\w+)\s+("
                + "|".join(sorted(_KIND_WORDS))
                + r")\b", q,
            )
            if m:
                subj = m.group(1)
            else:
                m = re.search(r"\b(?:what|which)\s+(.+)$", q)
                if m:
                    subj = m.group(1)
                else:
                    subj = _yes_no_subject(q)

    heads: set[str] = set()
    qualifiers: set[str] = set()
    if subj:
        words = subj.split()
        cut_i = next(
            (i for i, w in enumerate(words) if w.strip(".,;:") in _VERB_WORDS),
            len(words),
        )
        subj = " ".join(words[:cut_i]).strip()
        h = _head_noun(subj)
        if h and h not in _GENERIC_HEADS:
            heads.add(h)
            qualifiers = _subject_tokens(subj) - {h}
    if not kinds and not heads:
        return None
    return kinds, heads, qualifiers


def _kind_of_value(value: str) -> str:
    """The kind of a value when the reader did not tag it: a dotted id is a
    clause, a figure (number with a unit) is a measure, anything else with a
    digit is a code. Reuses the product's figure reader, so no unit list is
    invented here."""
    if _CLAUSE_VALUE_RE.match(value or ""):
        return "clause"
    from app.agents.citation_provenance import _figure_mentions
    for m, _figure in _figure_mentions(value or ""):
        if m.group(0).strip() == (value or "").strip():
            return "measure"
    return "code"


def _case_kind(case: Case) -> str:
    """The case's kind, falling back to the value's shape when a caller
    built the ``Case`` by hand (the reader always tags it)."""
    return case.kind or _kind_of_value(case.value)


def _relevant(
    case: Case,
    kinds: set[str],
    heads: set[str],
    qualifiers: set[str] = frozenset(),
) -> bool:
    """True when the case could answer the question.

    Its kind must match; when the question names a subject, the case's head
    noun must match it -- read from the case's own phrase with a trailing kind
    word dropped, so a passage cut at the value's own label ("BLINDING
    CONCRETE: Grade") still names its thing; and when the question's subject
    names a qualifier too ("fresh concrete temperature", not just
    "temperature"), the case must share at least one of those qualifier words --
    so a same-measure but different-thing case ("ambient temperature") is not
    kept. A case with no qualifier of its own stays (conservative: the head
    alone still matches).
    """
    if kinds and _case_kind(case) not in kinds:
        return False
    if heads:
        c_head = _head_noun(case.subject)
        c_tokens = _subject_tokens(case.subject)
        if c_head not in heads:
            # A phrase the reader cut at the value's own kind word ("BLINDING
            # CONCRETE: Grade") names its thing in the words before it: read
            # the head from there before giving the case up.
            c_head, c_tokens = _thing_of(case.subject)
            if c_head not in heads:
                return False
        if qualifiers:
            c_quals = c_tokens - {c_head}
            if c_quals and not (c_quals & qualifiers):
                return False
    return True


def _draft_distinct_values(draft: str, kinds: set[str]) -> set[tuple]:
    """Distinct values the draft states, read with the same reader as the
    evidence (so a document id or date is not a value), restricted to the
    asked kinds."""
    vals: set[tuple] = set()
    for case in cases_from_passages([("", draft)]):
        if kinds and case.kind not in kinds:
            continue
        n = _number_of(case.value)
        vals.add((round(n, 6), _unit_of(case.value)) if n is not None
                 else _norm(case.value))
    return vals


def clarifying_question(question: str, draft: str, cases: list[Case]) -> str | None:
    """The ONE clarifying question for this turn, or None to stand as written.

    ``cases`` are the value claims read from the handed evidence. The question
    fires only when the evidence holds more than one materially different
    relevant value, the question names none of them, and the draft has
    silently committed to exactly one value (rather than answering several
    side by side).
    """
    if not enabled():
        return None
    if not (question or "").strip() or not (draft or "").strip():
        return None

    asks = _question_asks(question)
    if asks is not None and asks[0] == "none":
        return None  # a party or an action: not a value to compare
    kinds, heads, qualifiers = asks if asks is not None else (set(), set(), set())

    relevant = [c for c in cases if _relevant(c, kinds, heads, qualifiers)]
    if len(relevant) < 2:
        return None

    # Dedup by (subject, value, source), preserving first appearance.
    unique: list[Case] = []
    seen: set[tuple[str, str, str]] = set()
    for case in relevant:
        key = (_norm(case.subject), _norm(case.value), _norm(case.source))
        if key in seen:
            continue
        seen.add(key)
        unique.append(case)

    # Collapse by value: one value = one case, wherever it is repeated.
    groups: dict[tuple, list[Case]] = {}
    order: list[tuple] = []
    for case in unique:
        n = _number_of(case.value)
        key = (round(n, 6), _unit_of(case.value)) if n is not None else _norm(case.value)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(case)

    # Bound 2: same value in every case -> answer, never ask.
    if len(order) < 2:
        return None

    # Bound 3: the question already names one of the cases.
    if any(_question_names_case(question, c) for c in unique):
        return None

    # Same-subject conflict (different values, one subject) is the both-values
    # statement (#453). Only a subject that DIFFERS makes this mechanism ask.
    if all(_same_subject(unique[0].subject, c.subject) for c in unique[1:]):
        return None

    # Silent pick: the draft states exactly one relevant value, and that
    # value is one of the evidence's (so it picked one and did not already
    # answer several structures side by side).
    draft_vals = _draft_distinct_values(draft, kinds)
    committed = draft_vals & set(order)
    if len(draft_vals) != 1 or not committed:
        return None

    # Bound 5: list only the cases actually found, each with its source.
    options = []
    for c in unique:
        if c.subject and _norm(c.subject) != _norm(c.value):
            options.append(f"{c.value} in {c.source} ({c.subject})")
        else:
            options.append(f"{c.value} in {c.source}")
    return (
        "Retrieved context supports more than one answer: "
        + "; ".join(options)
        + ". Which one is meant?"
    )


# --- the reader: value claims read from the handed passages (pure) --------

#: A sentence ends at . ! ? followed by whitespace or the end of the
#: string, or at a newline. A period BETWEEN digits (a decimal "4.5", a
#: dotted clause "2.14.1", a coded id "D999.43") is part of the value,
#: not a sentence boundary -- splitting there turns the subject into a
#: fragment ("Depth to invert is 4.").
_SENTENCE_BREAK_RE = re.compile(r"[.!?](?=\s|$)|\n")

#: A clause / document reference the passage states, read by SHAPE only:
#: numbers joined by dots (8.8.1) or a letter-number code (C35/45, D400).
#: A code with two or more dash-separated segments is a document id and is
#: skipped downstream (``_is_document_id``). Never an enumerated id -- the
#: shape decides, so a new code is read the same way as an old one.
_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z]{1,6}[-/]?\d[A-Za-z0-9./-]*\d|\d+(?:\.\d+){2,})"
    r"(?![A-Za-z0-9])",
)


def _sentence_span(text: str, start: int) -> tuple[int, int]:
    """The sentence boundaries around ``start`` (split on . ! ? and newlines)."""
    t = text or ""
    lo = 0
    for m in _SENTENCE_BREAK_RE.finditer(t):
        if m.start() < start:
            lo = m.end()
            continue
        hi = m.end() if m.group() == "\n" else m.start() + 1
        return lo, hi
    return lo, len(t)


#: Copula / auxiliary words that link a value to its subject. They sit at
#: the end of the text before the value ("X is / shall be <value>") and are
#: dropped so the subject is the noun phrase, not the whole sentence.
_COPULA_WORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am",
    "shall", "will", "should", "would", "must", "may", "can", "could",
    "might", "has", "have", "had", "does", "do", "did", "to",
})

#: Leading words that are not part of a noun phrase (determiners, and the
#: prepositions a value-initial sentence leaves before its noun).
_LEAD_WORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "of", "to", "for", "at", "in", "on",
})


def _noun_phrase(fragment: str) -> str:
    """The noun phrase inside a pre/post-value fragment.

    Drops trailing copula/auxiliary words, then leading determiners and
    prepositions. Returns "" when nothing is left, so the caller falls back
    to the words on the other side of the value.
    """
    words = (fragment or "").split()
    while words and words[-1].strip(".,;:()'\"") in _COPULA_WORDS:
        words.pop()
    while words and words[0].lower() in _LEAD_WORDS:
        words.pop(0)
    return " ".join(words).strip(" .,;:()\"'")


def _subject_phrase(text: str, start: int, end: int) -> str:
    """The short noun phrase a value modifies, read from the passage's own
    words.

    In the handed sentences the value sits in predicate position ("X is /
    shall be <value>"), so the subject is the phrase immediately before it
    with the copula dropped -- not the whole sentence. When the value has no
    leading text ("<value> of the cover"), the words after it are the phrase
    instead. Never a known list of subjects.
    """
    t = text or ""
    lo, hi = _sentence_span(t, start)
    phrase = _noun_phrase(t[lo:start])
    if phrase:
        return phrase
    return _noun_phrase(t[end:hi])


def cases_from_passages(passages: Iterable[tuple[str, str]]) -> list[Case]:
    """The value claims read from ``(source, text)`` passages.

    One ``Case`` per value the passage states. A measure (a number with a
    unit) reuses the product's figure reader; a reference is read by shape
    (dotted clause ids and letter-number codes). The subject is the short
    noun phrase the value modifies (its own sentence words, never a known
    list of subjects), so two passages about different things are told apart
    by the passage's own words. Pure: no I/O, no model.
    """
    from app.agents.citation_provenance import _figure_mentions

    cases: list[Case] = []
    for source, text in passages:
        t = text or ""
        for m in _REFERENCE_RE.finditer(t):
            value = m.group(0).strip()
            if _is_date_token(value):
                continue
            if _is_document_id(value):
                continue
            kind = "clause" if _CLAUSE_VALUE_RE.match(value) else "code"
            cases.append(Case(
                subject=_subject_phrase(t, m.start(), m.end()),
                value=value, source=source or "", kind=kind,
            ))
        for m, _figure in _figure_mentions(t):
            value = m.group(0).strip()
            cases.append(Case(
                subject=_subject_phrase(t, m.start(), m.end()),
                value=value, source=source or "", kind="measure",
            ))
    return cases
