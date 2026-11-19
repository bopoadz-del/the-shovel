"""Offline answer-against-its-sources score (measurement only, never a gate).

One row in, three numbers and a flag out, by fixed rules and no model:

(a) numbers   -- every number written in the answer is looked for in the text
                 of a chunk the answer cites, or in a tool result of that turn.
                 Reported as ``numbers_found`` / ``numbers_not_found``.
(b) citations -- every citation in the answer is checked against the chunks
                 handed to the model in that turn. Reported as
                 ``citations_handed`` / ``citations_not_handed``.
(c) overlap   -- for every citation, the share of the citing sentence's
                 content words that also occur in the cited chunks' text.
                 Reported as ``overlap_mean`` (0..1, None when nothing is
                 cited or no cited text is available).

Input (one JSON object per row, or a list of them)::

    {
      "id": "<row id>",
      "answer": "<answer text>",
      "handed": [{"chunk_id": "...", "doc_id": "...", "doc_name": "...",
                  "text": "<chunk text, optional>"}],
      "tool_results": ["<text of a tool result of that turn>", ...],
      "sources_list": ["<name of numbered source 1>", ...]
    }

``handed`` is the turn's ``rag_retrieval_trace`` ``chunks`` list; ``text`` is
filled from a ``{chunk_id: text}`` file (``--chunk-text``), never fetched
from a live service. ``sources_list`` resolves the numbered markers the
answer shows for the Sources panel: bare (``... 2.``) and bracketed (``[2]``,
the answer contract's form). A bracketed marker is never a figure; one the
panel does not number is counted in ``markers_unresolved``.

Output carries ids and counts only: no answer text, no document name, no
question text. The flag is ``low`` when any number is unfound, any citation
was not handed, or the overlap is under ``--overlap-floor``; it is a reading
aid for the table, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

# --- citations ---------------------------------------------------------------

# The citation-span rule lives in one place (app/lib/citation_spans.py) and is
# shared with the product's conflict check; this script only reads it.
_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from app.lib import citation_spans as _spans  # after the path fix

_NAMED_CITE_RE = _spans.NAMED_CITE_RE
_KEYED_CITE_RE = _spans.KEYED_CITE_RE
_MARKER_RE = _spans.MARKER_RE
_BRACKET_MARKER_RE = _spans.BRACKET_MARKER_RE
blank_named = _spans.blank_named
blank_markers = _spans.blank_markers
blank_citations = _spans.blank_citations

_ELLIPSIS = ("…", "...")


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def name_matches(cited: str, doc_name: str) -> bool:
    """A cited name matches a document name. A name shortened for display
    with an ellipsis matches on its two visible ends."""
    a, b = _fold(cited), _fold(doc_name)
    if not a or not b:
        return False
    if a == b:
        return True
    for mark in _ELLIPSIS:
        if mark in a:
            head, _, tail = a.partition(mark)
            return b.startswith(head.strip()) and b.endswith(tail.strip())
    return False


@dataclass
class Citation:
    kind: str          # "name" | "key" | "marker"
    target: str        # cited name, key value, or resolved source name
    sentence: str      # the sentence that carries it
    resolved: bool = True


def _sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans: split after . ! ? or a line break."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\[(\"])|\n+", text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _sentence_at(text: str, spans: list[tuple[int, int]], pos: int) -> str:
    for s, e in spans:
        if s <= pos <= e:
            return text[s:e]
    return ""


def find_citations(answer: str, sources_list: list[str] | None = None) -> list[Citation]:
    """Every citation the answer makes, with the sentence that makes it."""
    text = answer or ""
    spans = _sentences(text)
    out: list[Citation] = []
    for m in _NAMED_CITE_RE.finditer(text):
        name = m.group(1) if m.group(1) is not None else m.group(2)
        out.append(Citation("name", name.strip(), _sentence_at(text, spans, m.start())))
    for m in _KEYED_CITE_RE.finditer(text):
        out.append(Citation("key", m.group(1).strip(), _sentence_at(text, spans, m.start())))
    listed = list(sources_list or [])
    stripped = _strip_named(text)
    for m in _MARKER_RE.finditer(stripped):
        n = int(m.group(1))
        if not listed or not 1 <= n <= len(listed):
            continue  # a marker only exists when the Sources panel numbers it
        out.append(Citation("marker", listed[n - 1], _sentence_at(text, spans, m.start())))
    for m in _BRACKET_MARKER_RE.finditer(stripped):
        n = int(m.group(0)[1:-1])
        if not 1 <= n <= len(listed):
            continue  # counted by unresolved_markers, never a figure
        out.append(Citation("marker", listed[n - 1], _sentence_at(text, spans, m.start())))
    return out


def unresolved_markers(answer: str, sources_list: list[str] | None = None) -> int:
    """Bracketed markers the Sources panel does not number (a citation the
    reader cannot follow)."""
    listed = len(sources_list or [])
    return sum(1 for m in _BRACKET_MARKER_RE.finditer(_strip_named(answer or ""))
               if not 1 <= int(m.group(0)[1:-1]) <= listed)


def _strip_named(text: str) -> str:
    """Blank the named and keyed citation spans (same length, so positions
    still line up with the original text)."""
    return blank_named(text)


def cited_chunks(cite: Citation, handed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The handed chunks a citation points to (none when it was not handed)."""
    if cite.kind == "key":
        return [c for c in handed
                if cite.target in (str(c.get("chunk_id")), str(c.get("doc_id")))]
    return [c for c in handed if name_matches(cite.target, str(c.get("doc_name") or ""))]


# --- numbers -----------------------------------------------------------------

# A number not glued to a letter, hyphen or slash on either side (so ids such
# as "IP-INF-053" or "I233.3" are not figures). Thousands commas allowed.
_NUM_RE = re.compile(r"(?<![\w\-/.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w\-/])")


def numbers_in(text: str) -> list[float]:
    vals: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(1).replace(",", "") + (m.group(2) or "")
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    return vals


def answer_numbers(answer: str, sources_list: list[str] | None = None) -> list[float]:
    """The answer's figures: citation spans and numbered source markers are
    not figures."""
    return numbers_in(blank_citations(answer, len(sources_list or [])))


def _found(value: float, pool: Iterable[float]) -> bool:
    return any(abs(value - p) <= 1e-9 * max(1.0, abs(p)) for p in pool)


# --- overlap -----------------------------------------------------------------

_STOP = frozenset(
    "the and for are was were with that this from into onto upon over under than then "
    "these those their there which what when where while shall will would should could "
    "must have has had not but all any each per its it's is be by of to in on at as or "
    "an a no nor also only both such same other more most less least very can may "
    "source sources".split()
)
_WORD_RE = re.compile(r"[a-z][a-z]{2,}")


def content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").casefold()) if w not in _STOP}


def overlap(sentence: str, chunk_text: str) -> float | None:
    words = content_words(_strip_named(sentence))
    if not words:
        return None
    return len(words & content_words(chunk_text)) / len(words)


# --- the row score -----------------------------------------------------------

@dataclass
class RowScore:
    id: str
    numbers_found: int = 0
    numbers_not_found: int = 0
    citations_handed: int = 0
    citations_not_handed: int = 0
    markers_unresolved: int = 0
    overlap_mean: float | None = None
    text_available: bool = False
    flag: str = "ok"
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "numbers_found": self.numbers_found,
            "numbers_not_found": self.numbers_not_found,
            "citations_handed": self.citations_handed,
            "citations_not_handed": self.citations_not_handed,
            "markers_unresolved": self.markers_unresolved,
            "overlap_mean": None if self.overlap_mean is None else round(self.overlap_mean, 3),
            "text_available": self.text_available,
            "flag": self.flag,
            "reasons": self.reasons,
        }


def score_row(row: dict[str, Any], *, overlap_floor: float = 0.2) -> RowScore:
    answer = str(row.get("answer") or "")
    handed = list(row.get("handed") or [])
    tools = [str(t) for t in (row.get("tool_results") or [])]
    listed = [str(s) for s in (row.get("sources_list") or [])]
    out = RowScore(id=str(row.get("id") or ""))

    cites = find_citations(answer, listed)
    cited_text: list[str] = []
    overlaps: list[float] = []
    for cite in cites:
        hits = cited_chunks(cite, handed)
        if hits:
            out.citations_handed += 1
        else:
            out.citations_not_handed += 1
        text = "\n".join(str(c.get("text") or "") for c in hits)
        if text.strip():
            cited_text.append(text)
            ov = overlap(cite.sentence, text)
            if ov is not None:
                overlaps.append(ov)
    out.text_available = bool(cited_text)
    out.markers_unresolved = unresolved_markers(answer, listed)

    pool = [v for t in cited_text + tools for v in numbers_in(t)]
    for value in answer_numbers(answer, listed):
        if _found(value, pool):
            out.numbers_found += 1
        else:
            out.numbers_not_found += 1

    out.overlap_mean = sum(overlaps) / len(overlaps) if overlaps else None
    if out.numbers_not_found:
        out.reasons.append("number_not_found")
    if out.citations_not_handed:
        out.reasons.append("citation_not_handed")
    if out.overlap_mean is not None and out.overlap_mean < overlap_floor:
        out.reasons.append("low_overlap")
    out.flag = "low" if out.reasons else "ok"
    return out


# --- inputs ------------------------------------------------------------------

def attach_chunk_text(rows: list[dict[str, Any]], texts: dict[str, str]) -> None:
    """Fill each handed chunk's ``text`` from an offline ``{chunk_id: text}`` map."""
    for row in rows:
        for c in row.get("handed") or []:
            key = str(c.get("chunk_id"))
            if not c.get("text") and key in texts:
                c["text"] = texts[key]


def parse_sources_list(panel: str) -> list[str]:
    """Numbered source names from a Sources panel's text: a line holding only
    an integer N followed by the name line, integers running 1, 2, 3 ... in
    order. The panel's head is its title line, then its count line when the
    panel lists anything; that count is never an entry, even when the next
    line is not blank (a one-source panel with no coverage line, whose
    entries are headed by a confidence chip instead of a number)."""
    lines = [ln.strip() for ln in (panel or "").splitlines()]
    start = 2 if len(lines) > 1 and lines[1].isdigit() else 0
    names: list[str] = []
    for i in range(start, len(lines) - 1):
        ln = lines[i]
        if ln.isdigit() and int(ln) == len(names) + 1 and lines[i + 1] and not lines[i + 1].isdigit():
            names.append(lines[i + 1])
    return names


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rows", help="JSON file: one row object or a list of rows")
    ap.add_argument("--chunk-text", help="JSON file {chunk_id: text}, offline")
    ap.add_argument("--overlap-floor", type=float, default=0.2)
    args = ap.parse_args(argv)
    data = _load(args.rows)
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        if not row.get("sources_list") and isinstance(row.get("sources"), dict):
            row["sources_list"] = parse_sources_list(str(row["sources"].get("list") or ""))
    if args.chunk_text:
        attach_chunk_text(rows, _load(args.chunk_text))
    for row in rows:
        print(json.dumps(score_row(row, overlap_floor=args.overlap_floor).as_dict(),
                         separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
