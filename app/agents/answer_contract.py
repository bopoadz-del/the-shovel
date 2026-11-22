"""The answer contract: the form of the text a user reads.

Owner order (BUILD 0, 15 Nov). Every rule is structural; none names a
question, a formula, a project or a document.

1. ANSWER FIRST. On a turn that computed, the first line carries a figure.
   Narration of what was searched never opens an answer.
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
5. THE LEAD IS THE ANSWER; THE REST IS DETAIL. On a turn that did not
   compute, the text the user reads first is the opening paragraph (a
   heading carries on into the paragraph under it) with the list it
   introduces, each item cut to its first sentence. Every later paragraph,
   and each cut item word for word, leaves the text and travels as
   structured ``details`` beside the answer. The budget is the answer's own
   shape, never a word count: nothing is deleted, and an answer that is one
   paragraph is untouched.
6. Honesty is unchanged. Nothing here adds a figure, raises a tier or turns
   a refusal into an answer: a tier is copied from the turn's own Sources row
   or left ``None``; a refusal (an answer with no figure) keeps every sentence
   that is not machinery or padding.

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
    #: Rule 5: the supporting text that left the lead, word for word, or None.
    details: str | None = None


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
#: ``(Spec Section 03.pdf, p. 4)``, ``[Source: BOQ.xlsx]``.
_INLINE_FILE_CITE_RE = re.compile(
    rf"[ \t]*[(\[](?:[ \t]*sources?[ \t]*:)?[ \t]*(?P<body>[^()\[\]\n]{{0,200}}?\.(?:{_FILE_EXTS})\b[^()\[\]\n]{{0,60}}?)[)\]]",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"(?:,|\s)\s*(?P<sec>(?:pp?\.|pages?|section|clause|§)\s*[\w.\-–, ]+?)\s*$", re.IGNORECASE)
_PARENS_TAIL_RE = re.compile(r"\((?P<inner>[^()]*)\)\s*$")
_FILE_NAME_END_RE = re.compile(rf"\.(?:{_FILE_EXTS})\b", re.IGNORECASE)


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
    start = named.end() if named else 0
    sec = _PAGE_RE.search(text, start) if kind == "document" else None
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
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


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
    return stem if re.search(r"[A-Za-z]{2,}", stem) else ""


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
      of the thing's words) and a sentence stating two or more of the values
      stay.
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
        sentences = _SENTENCE_RE.findall(line[len(lead):])
        kept = []
        for sent in sentences:
            vals = extract(sent)
            agrees = asserts_agreement(sent)
            gone = False
            for kind, classes, thing, docs in rules:
                hits = {i for i, k in enumerate(classes) for x in vals if x.kind == kind and same(x, k)}
                about = bool(thing & _terms(sent))
                if len(hits) == 1 and about:
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
            out.append(_INLINE_FILE_CITE_RE.sub(lambda c: " " + markers.marker(c.group("body")), line).rstrip())
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
#: A sentence ends at ``.``/``!``/``?`` followed by space or the end; a point
#: inside a figure (``12.5``) or a name (``p.12``) does not end it.
_SENTENCE_RE = re.compile(r"(?:[^.!?\n]|[.!?](?![.!?]*(?:\s|$)))+(?:[.!?]+(?=\s|$)|$)")
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


def _strip_machinery(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        if not _MACHINE_RE.search(line):
            out.append(line)
            continue
        lead = re.match(r"^[ \t]*(?:[-*•][ \t]+|\d+[.)][ \t]+)?", line).group(0)
        body = line[len(lead):]
        kept = [k for k in (_strip_machinery_sentence(s.strip()) for s in _SENTENCE_RE.findall(body)) if k]
        new = lead + " ".join(kept)
        if _figures(_MACHINE_RE.sub("", line)) - _figures(new):
            # Removing the machinery would lose a figure: the line stays as written.
            out.append(line)
        elif kept:
            out.append(new)
    return "\n".join(out)


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
        sentences = [s.strip() for s in _SENTENCE_RE.findall(bare)] or [bare]
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


# ── Rule 5: the lead and the details ────────────────────────────────────────

_PARA_SPLIT_RE = re.compile(r"\n[ \t]*\n")
_HEADING_RE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+\S.*|\*\*[^*\n]+\*\*:?[ \t]*)$")
_TABLE_RE = re.compile(r"^[ \t]*\|")
_FENCE_RE = re.compile(r"^[ \t]*(?:```|~~~)", re.MULTILINE)


def _nonblank(par: str) -> list[str]:
    return [ln for ln in par.split("\n") if ln.strip()]


def _is_list(par: str) -> bool:
    """A paragraph that is a list or a table: its first line is an item or a
    table row and every other line is one too or continues one (indented)."""
    lines = _nonblank(par)
    if not lines or not (_BULLET_RE.match(lines[0]) or _TABLE_RE.match(lines[0])):
        return False
    return all(_BULLET_RE.match(ln) or _TABLE_RE.match(ln) or ln[:1] in (" ", "\t") for ln in lines)


def _introduces_list(par: str) -> bool:
    last = _nonblank(par)[-1].rstrip()
    return last.endswith(":") or bool(_HEADING_RE.match(last))


def _cut_item(line: str) -> tuple[str, str | None]:
    """A list item cut to its first sentence, and the item word for word when
    it was cut (else None). Every marker the item carried stays on it."""
    m = _BULLET_RE.match(line)
    if not m:
        return line, None
    sentences = [x.strip() for x in _SENTENCE_RE.findall(line[m.end():]) if x.strip()]
    if len(sentences) < 2:
        return line, None
    first = sentences[0]
    rest = "".join(mk for mk in dict.fromkeys(_MARKER_RE.findall(" ".join(sentences[1:]))) if mk not in first)
    return line[: m.end()] + first + (" " + rest if rest else ""), line.strip()


def _lead_and_details(body: str) -> tuple[str, str | None]:
    """The lead (opening paragraph, headings carrying on into the paragraph
    under them, plus the list it introduces with items cut to their first
    sentence) and the details (each cut item and every later paragraph, word
    for word), or ``(body, None)`` when nothing would move. A body carrying a
    fenced block is left whole."""
    if _FENCE_RE.search(body):
        return body, None
    pars = [p for p in _PARA_SPLIT_RE.split(body) if p.strip()]
    if not pars:
        return body, None
    i = 1
    while i < len(pars) and all(_HEADING_RE.match(ln) for ln in _nonblank(pars[i - 1])):
        i += 1
    lead_pars = pars[:i]
    if i < len(pars) and _introduces_list(lead_pars[-1]) and _is_list(pars[i]):
        lead_pars.append(pars[i])
        i += 1
    cut_items: list[str] = []
    lead: list[str] = []
    for par in lead_pars:
        lines = []
        for ln in par.split("\n"):
            short, whole = _cut_item(ln)
            lines.append(short)
            if whole:
                cut_items.append(whole)
        lead.append("\n".join(lines))
    rest = pars[i:]
    if not cut_items and not rest:
        return body, None
    details = ([ "\n".join(cut_items)] if cut_items else []) + rest
    return "\n\n".join(lead), "\n\n".join(details)


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
            body = _answer_first(body, computed)
        body = _tidy(body)
        details: str | None = None
        if not asking and not computed:
            body, details = _lead_and_details(body)
        from app.agents.evidence_conflicts import detect

        handed = [r for r in (evidence or []) if isinstance(r, dict)]
        # Read over the whole text (citations already markers): lead and details.
        found = detect(body + ("\n\n" + details if details else ""), rows + handed, question)
        structured: list[dict[str, Any]] = []
        if found:
            lines, structured = _conflict_lines(found, markers)
            if lines:
                # The conflict statement leads; a sentence asserting only one
                # of the differing values for that thing, or asserting that the
                # documents agree about it, is replaced by it.
                stated = [c for c in found if _nameable(c)]
                body = _tidy("\n".join(lines) + "\n\n" + _drop_superseded(body, stated))
                if details:
                    details = _tidy(_drop_superseded(details, stated)) or None
        # A marker whose line went (padding, a dropped note) is not lost: it
        # rides on the last line, so every citation entry has its marker. A
        # marker that travels in the details is not lost.
        shown = body + "\n" + (details or "")
        lost = "".join(f"[{e['marker']}]" for e in markers.entries if f"[{e['marker']}]" not in shown)
        if body and lost:
            body = body + " " + lost
        if not body or (body == _tidy(text) and not details):
            # Nothing but machinery and padding: the original stands, so the
            # user is never left with an empty answer. Nothing to change: the
            # original stands byte for byte.
            return Contracted(text=text)
        return Contracted(text=body, citations=markers.entries, coverage=coverage, conflicts=structured,
                          details=details)
    except Exception:  # noqa: BLE001 -- the contract must never break an answer
        _LOG.exception("answer_contract: failed; answer passed through")
        return Contracted(text=text)
