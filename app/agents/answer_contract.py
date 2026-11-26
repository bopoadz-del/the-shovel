"""The answer contract: the form of the text a user reads.

Owner order (BUILD 0, 15 Nov). Every rule is structural; none names a
question, a formula, a project or a document.

1. ANSWER FIRST. On a turn that computed, the first line carries a figure.
   Narration of what was searched never opens an answer, on any turn: a
   narration lead that is process only goes (one that reports an absence
   is the refusal itself and stays, rule 5).
2. LENGTH FOLLOWS THE QUESTION. Offers of further help, closing summaries,
   restatements of the question and "notes" blocks are padding and go. A
   note survives only when it is an assumption the result depends on: a line
   stating a value the working uses or an assumption with no value of its
   own. A closing summary goes only when it repeats figures already given.
3. NO MACHINERY IN THE TEXT. Status words about how a figure was produced
   ("deterministic", "certified" as a status, "pre-dispatch", the platform's
   calculator and pipeline names) and narration of the search ("the search
   returned", "retrieved documents") are removed. A sentence that is only
   machinery goes; a sentence that also carries a figure keeps the figure.
4. CITATIONS OUT OF THE TEXT. ``Source:`` lines and bracketed file citations
   become short numbered markers (``[1]``); each marker has one structured
   entry (document, page or section, passage, tier) that travels beside the
   answer. The ``N of M project documents indexed`` footer leaves the text
   and travels as structured ``coverage``.
5. Honesty is unchanged. Nothing here adds a figure, raises a tier or turns
   a refusal into an answer: a tier is copied from the turn's own Sources row
   or left ``None``; a refusal (an answer with no figure) keeps every sentence
   that is not machinery or padding. An answer that says a thing is absent and
   then states it with a cited figure loses only the absence sentence the
   cited figure overrides; an uncited figure overrides nothing.

Never raises: on any failure the text passes through unchanged.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_LOG = logging.getLogger(__name__)


@dataclass
class Contracted:
    """The answer text and what left it."""

    text: str
    #: One entry per marker: ``{"marker", "document", "section", "passage", "tier", "kind"}``.
    citations: list[dict[str, Any]] = field(default_factory=list)
    #: ``{"indexed": N, "total": M}`` when the coverage footer was on the text.
    coverage: dict[str, int] | None = None
    #: One entry per conflict between documents in the evidence (C2-C4):
    #: ``{"values": [{"value", "document", "marker"}], "governs", "basis_document"}``.
    conflicts: list[dict[str, Any]] = field(default_factory=list)


# ── Rule 4: citations and the coverage footer ───────────────────────────────

_COVERAGE_RE = re.compile(r"^[ \t]*(?P<n>\d+)[ \t]+of[ \t]+(?P<m>\d+)[ \t]+project documents indexed[ \t.]*$",
                          re.IGNORECASE | re.MULTILINE)

_SOURCE_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*•][ \t]+)?[*_]*[ \t]*(?:sources?|citations?|references?)[ \t]*[*_]*[ \t]*:[ \t]*[*_]*"
    r"[ \t]*(?P<body>.*?)[ \t]*[*_]*[ \t]*\.?[ \t]*$",
    re.IGNORECASE,
)

_FILE_EXTS = r"pdf|docx?|xlsx?|xlsm|csv|txt|md|pptx?|dwg|dxf|ifc|xer|xml|json|zip|rar|png|jpe?g"
#: A parenthesised or bracketed citation that names a file:
#: ``(Spec Section 03.pdf, p. 4)``, ``[Source: BOQ (Priced).xlsx]``.
_SOURCE_PREFIX_RE = re.compile(r"^[ 	]*sources?[ 	]*:", re.IGNORECASE)
_CITE_BODY_MAX = 260
_CLOSE = {"(": ")", "[": "]"}


def _group_end(line: str, start: int) -> int:
    """Index of the bracket closing the one at ``start``, or -1. Round and
    square brackets nest inside each other; a mismatched close ends the
    search (the group is not a citation)."""
    stack = [_CLOSE[line[start]]]
    for i in range(start + 1, min(len(line), start + 2 + _CITE_BODY_MAX)):
        ch = line[i]
        if ch in _CLOSE:
            stack.append(_CLOSE[ch])
        elif ch in ")]":
            if ch != stack.pop():
                return -1
            if not stack:
                return i
    return -1


def _names_a_file(body: str) -> bool:
    """True when the group's own text (not a group nested in it) carries a
    file extension: ``Bill (Priced).xlsx`` names a file; ``the bill [Bill.xlsx]
    shows`` does not, its nested group does."""
    depth = 0
    flat = []
    for ch in body:
        if ch in ")]":
            depth = max(depth - 1, 0)
        flat.append(ch if depth == 0 else " ")
        if ch in _CLOSE:
            depth += 1
    return bool(_FILE_NAME_END_RE.search("".join(flat)))


def _inline_citations(line: str, markers: "_Markers") -> str:
    """Every bracketed citation that names a file, as its marker. The group
    is read by its balanced brackets, so a file name that carries its own
    brackets (``Bill of Quantities (Priced).xlsx``, ``Drawings (6 of 7).pdf``)
    is one citation, not text the recogniser stops at."""
    out: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        end = _group_end(line, i) if ch in _CLOSE else -1
        if end > 0:
            inner = line[i + 1:end]
            labelled = bool(_SOURCE_PREFIX_RE.match(inner))
            body = _SOURCE_PREFIX_RE.sub("", inner, count=1).strip()
            if body and len(body) <= _CITE_BODY_MAX and _names_a_file(body):
                head = "".join(out).rstrip(" 	")
                out = [head, " " + markers.marker(body)]
                i = end + 1
                continue
            if labelled and len(body) <= _CITE_BODY_MAX:
                # A group the writer labelled as its source but that names no
                # file ("[source: Bill of Quantities, Bill No. 4]", "[source:
                # 1]") is still a citation, never prose: it becomes the marker
                # of the one Sources row its words name, or, when no single
                # row backs it, it leaves the text -- an attribution nothing
                # in the turn backs is not printed, and none is guessed.
                tag = markers.for_title(body)
                head = "".join(out).rstrip(" 	")
                out = [head, " " + tag] if tag else [head]
                i = end + 1
                continue
        out.append(ch)
        i += 1
    return "".join(out).rstrip()


_PAGE_RE = re.compile(r"(?:,|\s)\s*(?P<sec>(?:pp?\.|pages?|section|clause|§)\s*[\w.\-–, ]+?)\s*$", re.IGNORECASE)
_PARENS_TAIL_RE = re.compile(r"\((?P<inner>[^()]*)\)\s*$")
_FILE_NAME_END_RE = re.compile(rf"\.(?:{_FILE_EXTS})\b", re.IGNORECASE)
#: What may stand between a file name and the locator after it, in any
#: script: Latin, Arabic and full-width commas and semicolons, a colon, a
#: dash, a space. Punctuation only, no word of any language.
_LOCATOR_LEAD = " \t,;:،؛，；—–-"
#: The colons a label ends with; a file name holds none.
_LABEL_COLONS = ":："
#: Where a document named in words ends and its locator begins: a comma or
#: semicolon of any script, or a dash with space on both sides. Punctuation
#: only, no word of any language.
_TITLE_SPLIT_RE = re.compile(r"[ \t]*[,;،؛，；][ \t]*|[ \t]+[—–-][ \t]+")


def _entry_from_body(body: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One structured citation from a citation's text, filled from the turn's
    own Sources row when one names the same document. Nothing is invented:
    an unmatched citation carries what its text said and ``tier: None``."""
    from app.lib.source_labels import CALCULATOR_SUFFIX

    text = body.strip().strip("*_").strip()
    kind = "document"
    passage = ""
    low = text.lower()
    if CALCULATOR_SUFFIX in low:
        kind = "calculation"
        tail = _PARENS_TAIL_RE.search(text)
        if tail:
            passage = tail.group("inner").strip()
            text = text[: tail.start()].strip()
        text = re.sub(rf"\s*[—–-]+\s*{re.escape(CALCULATOR_SUFFIX)}\s*$", "", text, flags=re.IGNORECASE).strip()
    section = ""
    # A locator follows the document's name, never sits inside it: a file
    # name ("Headwall Sections.dwg", "Spec Section 03.pdf") stays whole, and
    # the page or section is looked for only after its extension.
    named = None
    for named in _FILE_NAME_END_RE.finditer(text):
        pass
    if named and kind == "document":
        # Read by structure, so it holds in every language. A file name
        # holds no colon, so a label ahead of it ("Source:", or the same
        # word in Arabic) ends at the last colon before the name. The name
        # ends at its extension, so what follows it, past the separating
        # punctuation, is its locator, whatever words it is written in.
        head = text[: named.end()]
        cut = max(head.rfind(c) for c in _LABEL_COLONS)
        section = text[named.end():].strip(_LOCATOR_LEAD).strip()
        text = head[cut + 1:].strip()
    elif kind == "document":
        sec = _PAGE_RE.search(text)
        if sec:
            section = sec.group("sec").strip()
            text = text[: sec.start()].strip().rstrip(",")
    entry: dict[str, Any] = {
        "document": text,
        "section": section,
        "passage": passage,
        "tier": None,
        "kind": kind,
    }
    key = _name_key(text)
    for row in rows:
        name = str(row.get("doc_name") or "")
        if key and name and (_name_key(name) == key or key in _name_key(name) or _name_key(name) in key):
            entry["document"] = name
            entry["section"] = section or str(row.get("page_or_section") or "")
            entry["passage"] = passage or str(row.get("snippet") or row.get("passage") or "")
            entry["tier"] = row.get("layer_label") or row.get("layer") or None
            break
    return entry


def _name_key(name: str) -> str:
    stem = re.sub(rf"\.(?:{_FILE_EXTS})$", "", (name or "").strip(), flags=re.IGNORECASE)
    # Letters and digits of any script: an Arabic name keys by its own
    # letters, not to the empty key every non-Latin name used to share.
    return re.sub(r"[\W_]+", "", stem.casefold())


@dataclass
class _Markers:
    rows: list[dict[str, Any]]
    entries: list[dict[str, Any]] = field(default_factory=list)

    def marker(self, body: str) -> str:
        entry = _entry_from_body(body, self.rows)
        if not entry["document"]:
            return ""
        for existing in self.entries:
            if (existing["document"], existing["section"], existing["passage"]) == (
                    entry["document"], entry["section"], entry["passage"]):
                return f"[{existing['marker']}]"
        entry["marker"] = len(self.entries) + 1
        self.entries.append(entry)
        return f"[{entry['marker']}]"

    def for_title(self, body: str) -> str:
        """The marker for a citation that names its document in words, not
        by file name: the one Sources row whose name holds those words. The
        whole text is tried first, then its lead before the first separator
        with the rest as the locator. A text with no word in it, or whose
        words fit no row or more than one document, has no marker ("")."""
        parts = _TITLE_SPLIT_RE.split(body.strip(), maxsplit=1)
        lead, rest = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")
        for title, section in ((body.strip(), ""), (lead, rest)):
            key = _name_key(title)
            if not title or not _doc_words(title):
                continue
            hits = [r for r in self.rows if key and key in _name_key(str(r.get("doc_name") or ""))]
            if len({str(r.get("doc_name")) for r in hits}) == 1:
                row = hits[0]
                return self.for_row(dict(row, page_or_section=section or row.get("page_or_section") or ""))
        return ""

    def for_row(self, row: dict[str, Any]) -> str:
        """The marker for one evidence row (its document and section)."""
        doc = str(row.get("doc_name") or row.get("document") or "")
        sec = str(row.get("page_or_section") or "")
        for existing in self.entries:
            if existing["document"] == doc and existing["section"] in ("", sec):
                return f"[{existing['marker']}]"
        entry = {"document": doc, "section": sec,
                 "passage": str(row.get("snippet") or row.get("passage") or ""),
                 "tier": row.get("layer_label") or row.get("layer") or None, "kind": "document",
                 "marker": len(self.entries) + 1}
        self.entries.append(entry)
        return f"[{entry['marker']}]"


def _doc_words(name: str) -> str:
    """A document's name as words for a sentence (its file extension off);
    empty when the name holds no word at all."""
    stem = re.sub(rf"\.(?:{_FILE_EXTS})$", "", (name or "").strip(), flags=re.IGNORECASE).strip()
    return stem if re.search(r"[^\W\d_]{2,}", stem) else ""


def _nameable(c: dict[str, Any]) -> bool:
    """A conflict is stated only when the line can name every value and every
    value's document in words; one that cannot is not emitted at all."""
    vals = c.get("values") or []
    return (len({str(v.get("value") or "").strip() for v in vals}) >= 2
            and len({v.get("document") for v in vals}) >= 2
            and all(str(v.get("value") or "").strip() and _doc_words(str(v.get("document") or "")) for v in vals))


def _conflict_lines(found: list[dict[str, Any]], markers: _Markers) -> tuple[list[str], list[dict[str, Any]]]:
    """Each conflict as one plain sentence (every value with its document by
    name and its marker; the governing value and why, or that no document
    says) and as a structured entry for the side panel. A conflict whose
    values or documents cannot all be named in words is not emitted."""
    lines: list[str] = []
    structured: list[dict[str, Any]] = []
    for c in found:
        if not _nameable(c):
            continue
        said: dict[tuple[str, str], str] = {}
        values = []
        for v in c["values"]:
            tag = markers.for_row(v["row"])
            k = (_doc_words(v["document"]), v["value"])
            said[k] = said.get(k, "") + ("" if tag in said.get(k, "") else tag)
            values.append({"value": v["value"], "document": v["document"], "marker": int(tag.strip("[]"))})
        line = "The documents differ on this: " + "; ".join(
            f"{doc} gives {val} {tags}" for (doc, val), tags in said.items()) + "."
        if c.get("governs"):
            gval = next(v["value"] for v in c["values"] if v["document"] == c["governs"])
            basis_row = next((r for r in markers.rows if str(r.get("doc_name") or "") == c["basis_document"]), None)
            # Each part is joined as it stands: a value or a kind phrase is
            # quoted exactly as the evidence gave it, never edited afterwards.
            words = [f"{gval} governs: the stated order of precedence puts the {c['governing_kind']}",
                     f"above the {c['lower_kind']}"]
            if basis_row:
                words.append(markers.for_row(basis_row))
            line += " " + " ".join(words) + "."
        else:
            line += " No document states which of them governs."
        lines.append(line)
        structured.append({"values": values, "governs": c.get("governs"),
                           "basis_document": c.get("basis_document")})
    return lines, structured


def _drop_superseded(body: str, found: list[dict[str, Any]]) -> str:
    """The sentences the leading conflict statement replaces.

    * A sentence that states only ONE of a conflict's differing values for
      that thing. Read by the typed values (``evidence_conflicts.extract``/
      ``same``, citation spans blanked); a sentence about another thing (none
      of the thing's words), a sentence stating two or more of the values and
      a sentence that also states a typed value which is none of the
      conflict's values (a result worked from it, a second quantity: news the
      conflict line does not carry) stay.
    * A sentence asserting that the documents agree
      (``evidence_conflicts.asserts_agreement``: an agreement word with no
      negation just before it) about that thing: it states one or more of the
      differing values, or uses a word of the thing, or names no other thing
      at all (``agreement_names_no_thing``: no typed value, and no word left
      beyond the agreement words, the words pointing at the evidence and the
      documents' own names, as in "the sources are consistent"). An
      agreement sentence about another thing stays."""
    from app.agents.evidence_conflicts import _terms, agreement_names_no_thing, asserts_agreement, extract, same

    rules = []
    for c in found:
        kind = c.get("kind")
        classes: list[Any] = []
        for v in c["values"]:
            # The value as detect typed it in its passage: its raw text alone
            # may not re-read as that kind (a grade code without its label).
            typed = v.get("typed")
            if typed is not None and not any(same(typed, k) for k in classes):
                classes.append(typed)
        if len(classes) >= 2:
            docs = [str(v.get("document") or "") for v in c["values"]]
            rules.append((kind, classes, set(c.get("thing") or []), docs))
    out: list[str] = []
    for line in body.split("\n"):
        lead = re.match(r"^[ \t]*(?:[-*•][ \t]+|\d+[.)][ \t]+)?", line).group(0)
        sentences = _sentences(line[len(lead):])
        kept = []
        for sent in sentences:
            vals = extract(sent)
            agrees = asserts_agreement(sent)
            gone = False
            for kind, classes, thing, docs in rules:
                hits = {i for i, k in enumerate(classes) for x in vals if x.kind == kind and same(x, k)}
                about = bool(thing & _terms(sent))
                # A typed value the conflict line does not carry (a result
                # worked from the value, a second quantity) is news, not a
                # restatement: only the competing value is superseded, so
                # the sentence holding it stays.
                more = any(not any(same(x, k) for k in classes) for x in vals)
                if len(hits) == 1 and about and not more:
                    gone = True
                elif agrees and (hits or about or agreement_names_no_thing(sent, docs)):
                    gone = True
            if not gone:
                kept.append(sent.strip())
        if not sentences or len(kept) == len(sentences):
            out.append(line)
        elif kept:
            out.append(lead + " ".join(kept))
    return "\n".join(out)


def _split_parts(body: str) -> list[str]:
    """A Source line can list several citations: split on ``;`` (and on ``,``
    between two file names), never inside a calculator's input list."""
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in body:
        depth += ch == "("
        depth -= ch == ")" and depth > 0
        if ch == ";" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _move_citations(text: str, markers: _Markers) -> str:
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        m = _SOURCE_LINE_RE.match(line)
        if not m:
            out.append(_inline_citations(line, markers))
            continue
        tags = [t for t in dict.fromkeys(markers.marker(p) for p in _split_parts(m.group("body"))) if t]
        if not tags:
            continue
        # The marker sits on the nearest line of the answer above the Source
        # line, once: a marker that line already carries is not repeated.
        for i in range(len(out) - 1, -1, -1):
            if out[i].strip():
                new = "".join(t for t in tags if t not in out[i])
                if new:
                    out[i] = out[i].rstrip() + " " + new
                break
        else:
            out.append("".join(tags))
    return "\n".join(out)


# ── Rule 3: machinery ───────────────────────────────────────────────────────

_MACHINE_TERMS = (
    r"\bdeterministic(?:ally)?\b",
    r"\bplatform(?:'s)?\s+calculator\b",
    r"\bcalculator\b",
    r"\bpre-?dispatch(?:ed)?\b",
    r"\b(?:answer|processing|retrieval|ingest(?:ion)?|rag)\s+pipeline\b",
    r"\b(?:the|a)\s+(?:\w+\s+)?(?:tool|block|hat)\s+(?:call|result|output|returned|ran)\b",
    r"\bcertified\s+(?:calculation|result|method|formula|figure|output|computation)s?\b",
    r"\b(?:retrieved|indexed|injected|provided)\s+(?:project\s+)?(?:documents?|excerpts?|passages?|chunks?|context)\b",
    r"\bsearch\s+(?:returned|found|results?|shows?|showed|surfaced|came\s+back)\b",
    r"\bI\s+(?:searched|looked\s+through|checked|queried|retrieved)\b[^.\n]*\b(?:documents?|index|corpus|files?|knowledge\s+base)\b",
    r"\b\d+\s+of\s+\d+\s+(?:project\s+)?documents?\s+(?:are\s+|were\s+)?indexed\b",
)
_MACHINE_RE = re.compile("|".join(_MACHINE_TERMS), re.IGNORECASE)
#: A phrase that only says how a figure was produced, inside a sentence that
#: also states the figure: removed, the figure stays.
_MACHINE_CLAUSE_RE = re.compile(
    r"(?:[ \t]*[,;(—–-][ \t]*|[ \t]+)(?:(?:computed|calculated|obtained|produced|derived|checked|verified)[ \t]+)?"
    r"(?:by|via|using|with|from|through)[ \t]+(?:the[ \t]+|a[ \t]+)?[^,.;()\n]*?"
    r"(?:" + "|".join(_MACHINE_TERMS) + r")[^,.;()\n]*[)]?",
    re.IGNORECASE,
)
_LEAD_IN_RE = re.compile(
    r"^(?:according\s+to|based\s+on|from|per|in)\s+(?:the\s+)?[^,]*?(?:" + "|".join(_MACHINE_TERMS) + r")[^,]*,\s*",
    re.IGNORECASE,
)
_OPEN_MARKS = {"(": ")", "[": "]", "“": "”"}
_CLOSE_MARKS = {v: k for k, v in _OPEN_MARKS.items()}


def _paired(text: str) -> set[str]:
    """The opening marks whose pairs close in ``text`` (a straight ``"``
    pairs with itself). A mark left open is not trusted to hold a sentence
    together: an unpaired mark is ordinary punctuation."""
    out = {o for o, c in _OPEN_MARKS.items() if text.count(o) == text.count(c)}
    if text.count('"') % 2 == 0:
        out.add('"')
    return out


def _sentences(text: str) -> list[str]:
    """The sentences of one line. A sentence ends at ``.``/``!``/``?``
    followed by space or the end; a point inside a figure (``12.5``) or a name
    (``p.12``) does not end it, and neither does one inside a paired bracket
    or quotation -- ``(e.g. "A", "B")`` is part of the sentence around it, so
    no rule that keeps or drops whole sentences can leave half of a quoted or
    bracketed clause standing as a sentence of its own."""
    paired = _paired(text)
    out: list[str] = []
    stack: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"' and '"' in paired:
            if stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append('"')
        elif ch in _OPEN_MARKS and ch in paired:
            stack.append(ch)
        elif ch in _CLOSE_MARKS and _CLOSE_MARKS[ch] in paired and stack and stack[-1] == _CLOSE_MARKS[ch]:
            stack.pop()
        # A stop ends the sentence; so does a closing mark right after one
        # (``"... stop." Then``).
        if ch in ".!?" or (i and text[i - 1] in ".!?" and (ch == '"' or ch in _CLOSE_MARKS)):
            j = i + 1
            while j < n and text[j] in ".!?":
                j += 1
            if not stack and (j == n or text[j].isspace()):
                if text[start:j].strip():
                    out.append(text[start:j])
                start = j
            i = j
            continue
        i += 1
    if text[start:].strip():
        out.append(text[start:])
    return out


_FIGURE_RE = re.compile(r"\d")
_MARKER_RE = re.compile(r"\[\d+\]")
#: A figure with the unit word that follows it: ``45 kN·m``, ``12.6 m³``, ``10%``.
_FIGURE_UNIT_RE = re.compile(r"(\d(?:[\d,.]*\d)?)[ \t]*([^\s\d,.;:!?()\[\]]*)")
#: A sentence that reports something absent is the refusal itself: it stays
#: as written even when it narrates the search.
_NEGATIVE_RE = re.compile(
    r"\b(?:no|not|none|nothing|neither|nor|cannot|can't|could\s+not|couldn't|unable|without|missing)\b",
    re.IGNORECASE,
)


def _figures(text: str) -> Counter:
    """Each figure with its unit word, counted. Citation markers are not figures."""
    return Counter((n.replace(",", ""), u) for n, u in _FIGURE_UNIT_RE.findall(_MARKER_RE.sub("", text)))


def _strip_machinery_sentence(sentence: str) -> str:
    """The sentence without its machinery. A machinery-only sentence goes,
    unless it reports something absent (that is the refusal itself). A
    sentence that also states a figure keeps every figure with its unit: if
    removing the machinery would lose one, the sentence stays as written."""
    if not _MACHINE_RE.search(sentence):
        return sentence
    rest = _MACHINE_RE.sub("", sentence)
    if not _FIGURE_RE.search(rest):
        return sentence if _NEGATIVE_RE.search(rest) else ""
    s = _LEAD_IN_RE.sub("", sentence.lstrip())
    s = _MACHINE_CLAUSE_RE.sub("", s)
    s = _MACHINE_RE.sub("", s)
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if _figures(rest) - _figures(s):
        return sentence
    return s


#: A word that says a thing came out of the search, used to describe it:
#: "the retrieved rows", "text retrieved from it". Such a sentence often
#: carries meaning a reader needs (what was absent, what the answer rests
#: on), so the word is rewritten in place rather than the sentence dropped.
_SEARCH_PARTICIPLE = r"(?:retrieved|fetched)"
#: Where a noun phrase after the participle ends: punctuation, a citation,
#: or a word that starts a clause or a prepositional phrase.
_NP_END = (
    r"(?=\s*(?:[\[\](),.;:!?\"“”—–]|$)|\s+(?:for|from|of|in|on|at|by|with|to|under|about|is|are|was|were|be|been|"
    r"came|come|comes|show|shows|showed|contain|contains|cover|covers|give|gives|hold|holds|list|lists|"
    r"and|or|but|that|which|where|so|as)\b)"
)
_NP_FUNCTION_WORD = (
    r"(?:for|from|of|in|on|at|by|with|to|under|about|out|is|are|was|were|be|been|and|or|but|that|which|it|them)\b"
)
#: Adjective use: the participle stands before its noun phrase, which never
#: opens on a function word.
_PARTICIPLE_NP_RE = re.compile(
    r"\b" + _SEARCH_PARTICIPLE + r"\s+(?!" + _NP_FUNCTION_WORD + r")"
    r"((?:[A-Za-z][\w'-]*\s+){0,2}?[A-Za-z][\w'-]*)" + _NP_END,
    re.IGNORECASE,
)
#: After its noun, pointing at a document the sentence already names (a
#: pronoun or a citation): ``the note retrieved from it``.
_PARTICIPLE_FROM_RE = re.compile(
    r"\b" + _SEARCH_PARTICIPLE + r"(?=\s+(?:from|out\s+of)\s+(?:it|them|this|that|these|those|\[))", re.IGNORECASE)
#: What may stand before an adjective use. After a form of "be" or "have"
#: the participle is a verb ("the sample was retrieved"), not the search.
_DETERMINER_BEFORE_RE = re.compile(
    r"(?:^|[.!?:;(\-–—]\s*|\b(?:the|these|those|this|that|any|all|no|its|their|our|of|in|from|among)\s+)$",
    re.IGNORECASE,
)
#: A search-unit noun is itself machinery; in the reader's words it is a document.
_SEARCH_UNIT_RE = re.compile(r"\b(?:excerpt|passage|chunk|snippet)(s?)\b|\bcontext\b", re.IGNORECASE)
#: Quoted text and bracketed spans (citations, file names) are the documents'
#: own words: they are never rewritten.
_VERBATIM_RE = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”|\[[^\]\n]*\]")


def _reader_unit(m: re.Match) -> str:
    """A search-unit noun as the reader names it: a document."""
    if m.group(0).lower() == "context":
        return "documents"
    return "document" + (m.group(1) or "")


def _reader_noun(phrase: str) -> str:
    return _SEARCH_UNIT_RE.sub(lambda m: _reader_unit(m), phrase)


def _participle_np(m: re.Match) -> str:
    """One adjective use of a search participle, in the reader's words."""
    if not _DETERMINER_BEFORE_RE.search(m.string[:m.start()]):
        # After a verb such as "was": the participle is a verb, not the search.
        return m.group(0)
    head = _reader_noun(m.group(1)) + " I read"
    # A participle that opened a sentence hands its capital to the noun.
    return head[0].upper() + head[1:] if m.group(0)[0].isupper() else head


def _rewrite_search_words(segment: str) -> str:
    segment = _PARTICIPLE_FROM_RE.sub("read", segment)
    return _PARTICIPLE_NP_RE.sub(lambda m: _participle_np(m), segment)


def _search_words_in_place(text: str) -> str:
    """Each search participle in the reader's words, everywhere outside
    quotes and brackets: ``not in the retrieved rows`` becomes ``not in the
    rows I read`` -- the scope of the claim stays, the machinery goes, and a
    negative sentence still says what was absent. A sentence is never
    dropped here and no figure is touched."""
    out: list[str] = []
    last = 0
    for m in _VERBATIM_RE.finditer(text):
        out.append(_rewrite_search_words(text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_rewrite_search_words(text[last:]))
    return "".join(out)


def _strip_machinery(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        if not _MACHINE_RE.search(line):
            out.append(line)
            continue
        lead = re.match(r"^[ \t]*(?:[-*•][ \t]+|\d+[.)][ \t]+)?", line).group(0)
        body = line[len(lead):]
        kept = [k for k in (_strip_machinery_sentence(s.strip()) for s in _sentences(body)) if k]
        new = lead + " ".join(kept)
        if _figures(_MACHINE_RE.sub("", line)) - _figures(new):
            # Removing the machinery would lose a figure: the line stays as written.
            out.append(line)
        elif kept:
            out.append(new)
    # What stays is in the reader's words: a search participle left in a kept
    # sentence (a refusal keeps its sentences) is rewritten, not kept.
    return "\n".join(_search_words_in_place(line) for line in out)


# ── Rule 2: padding ─────────────────────────────────────────────────────────

_OFFER_RE = re.compile(
    r"^(?:let\s+me\s+know|if\s+you(?:'d|\s+would)?\s+(?:like|want|wish|need\s+(?:anything|more|further|help))|"
    r"would\s+you\s+like|feel\s+free|i\s+can\s+also|i(?:'m|\s+am)\s+happy\s+to|happy\s+to\s+help|"
    r"shall\s+i|do\s+you\s+want\s+me|hope\s+this\s+helps)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(r"^(?:\*\*)?(?:in\s+summary|to\s+summari[sz]e|in\s+short|summary|bottom\s+line|overall)\b",
                         re.IGNORECASE)
_RESTATE_RE = re.compile(r"^(?:you\s+asked|you(?:'re|\s+are)\s+asking|your\s+question\s+(?:is|was|asks))\b",
                         re.IGNORECASE)
_NOTES_HEAD_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?[ \t]*(?:(?:one|two|three|four|a\s+few|some|\d+|a\s+couple\s+of)\s+)?"
    r"(?:notes?|caveats?|remarks?|considerations?)\b[^\n]*$",
    re.IGNORECASE,
)
_ASSUMPTION_RE = re.compile(r"\bassum(?:e|ed|es|ing|ption)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*•]|\d+[.)])[ \t]+")


def _working_numbers(lines: list[str]) -> set[str]:
    """Numbers the answer's own result and working use: the first figure line
    and any line that substitutes values into a formula (carries ``=``)."""
    nums: set[str] = set()
    first = next((ln for ln in lines if _FIGURE_RE.search(ln)), "")
    for ln in [first] + [ln for ln in lines if "=" in ln and _FIGURE_RE.search(ln)]:
        nums.update(n.replace(",", "") for n in _NUMBER_RE.findall(ln))
    return nums


def _changes_result(line: str, working: set[str], computed: bool = True) -> bool:
    """A note that stays: one that states a value the working uses, an
    assumption that states no value of its own, or -- on a turn that did not
    compute, where there is no working to judge it by -- any note carrying a
    figure. A note stating a value the working did not use (an unasked
    caveat on a calculation) and a figure-less remark go."""
    nums = {n.replace(",", "") for n in _NUMBER_RE.findall(line)}
    if nums & working:
        return True
    if nums and not computed:
        return True
    return bool(_ASSUMPTION_RE.search(line)) and not nums


def _drop_padding(text: str, computed: bool = True) -> str:
    lines = text.split("\n")
    working = _working_numbers(lines)
    out: list[str] = []
    in_notes = False
    for line in lines:
        bare = _BULLET_RE.sub("", line).strip().lstrip("*_ ").strip()
        if not line.strip():
            in_notes = False
            out.append(line)
            continue
        if _NOTES_HEAD_RE.match(line) and len(bare) <= 80:
            # A notes heading opens a block; its body stays only where it is
            # an assumption the result depends on.
            in_notes = True
            continue
        if in_notes and not _changes_result(line, working, computed):
            continue
        sentences = [s.strip() for s in _sentences(bare)] or [bare]
        kept = [s for s in sentences if not (_OFFER_RE.match(s) or _RESTATE_RE.match(s))]
        if not kept:
            continue
        if _SUMMARY_RE.match(bare) and out and not (_figures(bare) - _figures("\n".join(out))):
            # A closing summary goes only when every figure in it was already given.
            continue
        if len(kept) != len(sentences):
            lead = line[: len(line) - len(line.lstrip())] + (_BULLET_RE.match(line).group(0).strip() + " "
                                                            if _BULLET_RE.match(line) else "")
            line = lead + " ".join(kept)
        out.append(line)
    return "\n".join(out)


#: A sentence addressed to the reader that asks them for the next step:
#: an imperative to supply or direct something ("Upload the priced BOQ and
#: I'll price it", "point me at that file"), optionally after a condition
#: ("If the pipework is billed elsewhere, point me at that file and I'll pull
#: the line") or a "please"; or the writer's own offer made conditional on
#: the reader ("I can pull the line if you send the file").
#: An imperative to supply something counts only when the sentence names
#: the writer as the one who receives it or acts next ("... and I'll pull
#: the line", "send it to me"): "Send the notice to the Engineer" is an
#: instruction the answer gives, not a request for the writer.
_REQUEST_RE = re.compile(
    r"^(?:(?:if|once|when|where|should)\b[^.!?]{0,200}?,[ \t]*(?:then[ \t]+)?)?(?:please[ \t]+|just[ \t]+)?"
    r"(?:(?:point|tell|give|show)\s+me|let\s+me\s+know"
    r"|(?:upload|re-?upload|share|send|provide|attach|forward)\b[^.!?]*\b(?:me|i'll|i\s+(?:can|will|could))\b)\b"
    r"|^(?:i\s+can|i\s+could|i'll|i\s+will|i'd\s+be\s+able\s+to)\b[^.!?]*\bif\s+you\b",
    re.IGNORECASE,
)


def _is_request_paragraph(par: str) -> bool:
    """Every sentence of the paragraph asks the reader for a next step (or
    offers further help), and it states no figure and carries no citation
    marker -- it adds nothing the reader can use from this answer."""
    if _MARKER_RE.search(par) or _FIGURE_RE.search(par):
        return False
    sentences = [s.strip() for ln in par.split("\n")
                 for s in _sentences(_BULLET_RE.sub("", ln).strip().lstrip("*_ ").strip()) if s.strip()]
    return bool(sentences) and all(_REQUEST_RE.match(s) or _OFFER_RE.match(s) for s in sentences)


def _drop_closing_request(text: str) -> str:
    """Rule 2, every kind of answer: a closing paragraph that only asks the
    reader for a next step ("upload X", "point me at the file and I'll pull
    the line") and states no figure and no marker is padding and goes. The
    answer before it stays; an answer that is nothing but such a paragraph
    stays as written. A turn that ends on its own question for inputs a
    step still needs (``asking``) never reaches this rule."""
    pars = re.split(r"\n[ \t]*\n", text.strip())
    while len(pars) > 1 and _is_request_paragraph(pars[-1]):
        pars.pop()
    return "\n\n".join(pars)


# ── Rule 1: answer first ────────────────────────────────────────────────────

def _answer_first(text: str, computed: bool) -> str:
    """On a turn that computed, the first line carries a figure: a lead-in
    line with no figure goes ahead of nothing -- the first figure line moves
    to the top. A turn that did not compute is left in its order (a refusal
    or a direct statement may carry no figure at all)."""
    if not computed:
        return text
    lines = [ln for ln in text.split("\n")]
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    if not nonblank or _FIGURE_RE.search(lines[nonblank[0]]):
        return text
    fig = next((i for i in nonblank if _FIGURE_RE.search(lines[i])), None)
    if fig is None:
        return text
    head = lines.pop(fig)
    return "\n".join([head] + lines)


#: A sentence that opens on the writer's own search: "I searched ...",
#: "I looked through ...", "I reviewed ...". It tells the reader what was
#: done, not what is so.
_NARRATION_LEAD_RE = re.compile(
    r"^(?:\*\*)?I\s+(?:searched|looked(?:\s+(?:through|for|in|at|over))?|checked|reviewed|went\s+through|"
    r"scanned|queried)\b",
    re.IGNORECASE,
)


def _narration_first(text: str) -> str:
    """Narration of the search never opens an answer, on any kind of turn.

    The first sentence of the first line is read. When it narrates the
    writer's search, states no figure and reports nothing absent, it is
    process only and goes, so the sentence after it opens the answer.

    It stays when it states a figure (it is the answer already), when it
    reports something absent (that is the refusal itself, rule 5), when it
    introduces what follows (ends in ``:``), or when it is the whole
    answer."""
    lines = text.split("\n")
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is None:
        return text
    sentences = _sentences(lines[first].strip())
    if not sentences:
        return text
    lead = sentences[0].strip()
    if (not _NARRATION_LEAD_RE.match(lead) or _FIGURE_RE.search(_MARKER_RE.sub("", lead))
            or _NEGATIVE_RE.search(lead) or lead.rstrip("*_ ").endswith(":")):
        return text
    rest = " ".join(s.strip() for s in sentences[1:]).strip()
    if rest:
        lines[first] = rest
        return "\n".join(lines)
    if any(ln.strip() for ln in lines[first + 1:]):
        return "\n".join(lines[:first] + lines[first + 1:])
    return text


# ── Rule 5: an absence the answer itself overrides ──────────────────────────

#: Words that say nothing about WHAT is absent: function words, the places a
#: thing is looked for, and the verbs of having or showing it. What is left of
#: an absence clause names the thing itself.
_ABSENCE_FILLER = frozenset((
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "there", "here", "is", "are", "was",
    "were", "be", "been", "being", "has", "have", "had", "do", "does", "did", "for", "of", "on", "in", "at",
    "to", "by", "from", "with", "within", "into", "about", "any", "anything", "such", "so", "and", "or", "but",
    "nor", "than", "then", "you", "your", "we", "our", "i", "me", "my", "they", "their", "them", "which",
    "what", "where", "who", "whom", "whose", "how", "no", "not", "none", "nothing", "neither", "cannot",
    "can't", "could", "couldn't", "unable", "without", "missing", "file", "files", "record", "records",
    "document", "documents", "project", "projects", "excerpt", "excerpts", "context", "provided", "available",
    "uploaded", "indexed", "given", "shown", "stated", "listed", "included", "found", "contain", "contains",
    "contained", "give", "gives", "gave", "show", "shows", "showed", "state", "states", "list", "lists",
    "include", "includes", "hold", "holds", "held", "appear", "appears", "appeared", "find", "see", "seen",
    "yet", "currently",
))
_CLAUSE_END_RE = re.compile(r"\s[—–-]\s|[—–;:]")
_WORD_RE = re.compile(r"[a-z][a-z'’]+")


def _stem(word: str) -> str:
    word = word.replace("’", "'").removesuffix("'s")
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _absence_terms(sentence: str) -> set[str]:
    """What an absence sentence says is absent: the content words of its
    first clause (a remedy after a dash or semicolon is not the thing). Empty
    when the sentence states no absence, carries a figure or a marker, or
    names nothing."""
    plain = sentence.strip().strip("*_ ")
    if _FIGURE_RE.search(plain) or _MARKER_RE.search(plain):
        return set()
    clause = _CLAUSE_END_RE.split(plain, maxsplit=1)[0]
    if not _NEGATIVE_RE.search(clause):
        return set()
    return {_stem(w) for w in _WORD_RE.findall(clause.lower()) if w not in _ABSENCE_FILLER}


def _drop_overridden_absence(text: str) -> str:
    """An answer never says a thing is absent and then states it.

    A sentence stating that something is absent goes when a LATER sentence of
    the same answer states a figure, carries a citation marker and names every
    content word of what the absence sentence said was missing. The cited
    figure is the answer; the absence line it overrides only contradicts it.

    It stays when no later sentence states a cited figure, when the figure is
    uncited (nothing here turns an uncited figure into the answer, rule 5),
    when the cited sentence is about something else (a word of the absent
    thing is not in it), and when it names nothing. The answer is never left
    empty: the cited figure sentence itself is never dropped."""
    lines = text.split("\n")
    # Every sentence, in reading order, with the line it sits on.
    spans = [(i, s) for i, line in enumerate(lines) for s in _sentences(line)]
    drop: set[int] = set()
    for k, (_, s) in enumerate(spans):
        terms = _absence_terms(s)
        if not terms:
            continue
        for _, later in spans[k + 1:]:
            if not (_MARKER_RE.search(later) and _FIGURE_RE.search(_MARKER_RE.sub("", later))):
                continue
            named = {_stem(w) for w in _WORD_RE.findall(later.lower())}
            if terms <= named:
                drop.add(k)
                break
    out: list[str] = []
    for i, line in enumerate(lines):
        mine = [(k, s) for k, (j, s) in enumerate(spans) if j == i]
        if not any(k in drop for k, _ in mine):
            out.append(line)
            continue
        kept = " ".join(s.strip() for k, s in mine if k not in drop).strip()
        if kept:
            lead = line[: len(line) - len(line.lstrip())]
            out.append(lead + kept)
    return "\n".join(out)


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def apply(
    text: str,
    *,
    sources: Iterable[dict[str, Any]] | None = None,
    computed: bool = False,
    asking: bool = False,
    question: str | None = None,
    evidence: Iterable[dict[str, Any]] | None = None,
) -> Contracted:
    """The answer in contract form. ``sources`` are the turn's Sources rows
    (used only to fill a citation's page, passage and tier); ``computed`` is
    True when a calculation ran this turn; ``asking`` is True when the turn
    ends on its own question for inputs a step still needs -- that question
    is the user's next action, so its wording is left as asked (citations and
    the coverage footer still leave the text). ``question`` is the user's
    question when known: it sharpens which evidence values answer the same
    thing when documents are checked for conflicts (C2). ``evidence`` is
    every passage the turn was handed (pre-injected excerpts and tool
    passages): conflicts are checked over it as well as over ``sources``,
    since a Sources row names only what the answer cited and may carry no
    passage."""
    if not isinstance(text, str) or not text.strip():
        return Contracted(text=text)
    try:
        rows = [r for r in (sources or []) if isinstance(r, dict)]
        coverage = None
        m = _COVERAGE_RE.search(text)
        if m:
            coverage = {"indexed": int(m.group("n")), "total": int(m.group("m"))}
        body = _COVERAGE_RE.sub("", text)
        markers = _Markers(rows)
        body = _move_citations(body, markers)
        computed = computed or any(e["kind"] == "calculation" for e in markers.entries) or any(
            str(r.get("page_or_section") or "").lower() == "calculated" for r in rows)
        if not asking:
            body = _strip_machinery(body)
            body = _drop_padding(body, computed)
            body = _drop_closing_request(body)
            body = _answer_first(body, computed)
            body = _narration_first(body)
            body = _drop_overridden_absence(body)
        body = _tidy(body)
        from app.agents.evidence_conflicts import detect

        handed = [r for r in (evidence or []) if isinstance(r, dict)]
        # Read over the text the user reads (citations already markers).
        found = detect(body, rows + handed, question)
        structured: list[dict[str, Any]] = []
        if found:
            lines, structured = _conflict_lines(found, markers)
            if lines:
                # The conflict statement leads; a sentence asserting only one
                # of the differing values for that thing, or asserting that the
                # documents agree about it, is replaced by it.
                stated = [c for c in found if _nameable(c)]
                body = _tidy("\n".join(lines) + "\n\n" + _drop_superseded(body, stated))
        # A marker whose line went (padding, a dropped note) is not lost: it
        # rides on the last line, so every citation entry has its marker.
        lost = "".join(f"[{e['marker']}]" for e in markers.entries if f"[{e['marker']}]" not in body)
        if body and lost:
            body = body + " " + lost
        if not body or body == _tidy(text):
            # Nothing but machinery and padding: the original stands, so the
            # user is never left with an empty answer. Nothing to change: the
            # original stands byte for byte.
            return Contracted(text=text)
        return Contracted(text=body, citations=markers.entries, coverage=coverage, conflicts=structured)
    except Exception:  # noqa: BLE001 -- the contract must never break an answer
        _LOG.exception("answer_contract: failed; answer passed through")
        return Contracted(text=text)
