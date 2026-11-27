"""Knowledge entries of the catalogue built from the knowledge itself.

Two sources, both read as they are; nothing here edits, adds to or
re-words their content:

* each entry of the curated construction knowledge base (the JSON the
  ``construction_advisor`` tool searches and evaluates) becomes one
  ``knowledge:<entry id>`` catalogue entry: its title is the name, its
  statement the ``for`` line, its provenance and credibility tier go into
  ``source`` verbatim. Its ``inputs`` and ``outputs`` are what the tool
  takes and returns for it, never more: an entry the tool evaluates (the
  knowledge base's own ``evaluation_refusal`` says which) offers its
  variables as inputs and the result as output; any other entry offers
  neither, whatever fields it holds;
* a knowledge file's sections are its own headings.

``source["sections"]`` names the parts a call to the entry's tool hands
back: for a knowledge base entry, the fields it holds content in among
those a search match carries (the knowledge base's own ``match_fields``
says which); for a knowledge file, its headings. A part the entry holds in
its file and a call does not return (a table, a checklist, a states list,
the variables) is not named. An entry of which a call returns no part
beyond its record has ``sections: []``.
"""
from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, List

#: The knowledge base fields that say what an entry IS (its identity and
#: its record), as opposed to what it HOLDS. They fill the entry's name,
#: code and source; a section is another field, returned with content.
RECORD_FIELDS = ("id", "title", "domain", "type", "provenance",
                 "credibility_tier", "needs_review", "applicability")

# An ATX heading as CommonMark reads it: at most three spaces of indent, one
# to six ``#``, then a space or the end of the line. ``#hashtag``, seven
# ``#`` and a line indented four spaces (code) are not headings.
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ 	](.*))?$")
# The optional closing run of ``#``: only after a space (or as the whole
# text). A ``#`` that touches the text ("C#") belongs to it.
_CLOSING = re.compile(r"(?:^|[ 	])#+[ 	]*$")
# A code fence: at most three spaces of indent, three or more of one fence
# character, then the rest of the line.
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def markdown_headings(text: str) -> List[str]:
    """Every ATX heading (``#`` to ``######``) of a markdown text that has
    text, in order, outside fenced code blocks. The one reading of a
    knowledge file's headings: its name is the first, its sections the
    rest. A code block ends only at a fence of the character that opened
    it, at least as long, with nothing after it. Setext headings (a line
    underlined with ``===`` or ``---``) are not read."""
    out: List[str] = []
    fence = ""  # the fence that opened the code block being passed over
    for line in (text or "").splitlines():
        f = _FENCE.match(line)
        if fence:
            if f and f.group(1).startswith(fence) and not f.group(2).strip():
                fence = ""
            continue
        # A backtick fence cannot carry a backtick after it (inline code).
        if f and not (f.group(1)[0] == "`" and "`" in f.group(2)):
            fence = f.group(1)
            continue
        m = _HEADING.match(line)
        heading = _CLOSING.sub("", (m.group(1) or "").strip(" 	")).strip(" 	") if m else ""
        if heading:
            out.append(heading)
    return out


def markdown_sections(text: str) -> List[str]:
    """Every heading of a markdown text after its first one (the first is
    the file's name), in order."""
    return markdown_headings(text)[1:]


def _held(value: Any) -> bool:
    return value not in (None, "", [], {})


def kb_sections(entry: Dict[str, Any]) -> List[str]:
    """The parts of an entry a call to the tool returns: the fields a match
    carries with content, in the entry's own order, leaving out its record
    fields. Empty when a call returns no part of it beyond its record."""
    from app.blocks import _knowledge

    returned = _knowledge.match_fields(entry)
    return [k for k in entry if k not in RECORD_FIELDS and _held(returned.get(k))]


def _kb_input(name: str, spec: Any) -> Dict[str, Any]:
    spec = spec if isinstance(spec, dict) else {}
    return {"name": name, "label": spec.get("desc") or name, "type": "number",
            "unit": spec.get("unit") or "", "range": None, "required": True, "basis": ""}


def kb_entry(entry: Dict[str, Any], file: str) -> Dict[str, Any]:
    """One catalogue entry for one knowledge base entry."""
    from app.agents.driver import catalogue
    from app.blocks import _knowledge

    # Everything the entry says of itself is read from what a call returns
    # of it; the variables are what an evaluation takes.
    returned = _knowledge.match_fields(entry)
    rule_id = str(returned["id"] or "")
    # What the tool does with the entry decides what the entry offers: values
    # in and a result out only where ``evaluate`` computes one.
    evaluated = _knowledge.evaluation_refusal(entry) is None
    variables = entry.get("variables") if evaluated and isinstance(entry.get("variables"), dict) else {}
    thresholds = returned["thresholds"] if isinstance(returned["thresholds"], dict) else {}
    outputs = [{"name": "result", "unit": thresholds.get("unit") or "", "main": True}] if evaluated else []
    out = catalogue._entry(
        "knowledge", rule_id, name=returned["title"] or rule_id,
        inputs=[_kb_input(k, v) for k, v in variables.items()],
        outputs=outputs,
        code={"tool": "construction_advisor", "rule_id": rule_id},
        source={"registry": "knowledge", "file": file, "entry": rule_id,
                "type": returned["type"] or "", "domain": returned["domain"] or "",
                "sections": kb_sections(entry),
                "provenance": copy.deepcopy(returned["provenance"] or {}),
                "credibility_tier": returned["credibility_tier"]})
    out["for"] = catalogue._one_line(returned["statement"])
    out["gaps"] = catalogue._gaps(out)
    return out


def kb_file() -> str:
    """The knowledge base file the ``construction_advisor`` tool reads,
    relative to the repository root."""
    from app.agents.driver import catalogue
    from app.blocks import _knowledge

    path = os.path.realpath(_knowledge._kb_path())
    root = os.path.realpath(catalogue._ROOT)
    # A file on another drive (Windows) has no relative path: its full path.
    inside = os.path.splitdrive(path)[0] == os.path.splitdrive(root)[0]
    return (os.path.relpath(path, root) if inside else path).replace(os.sep, "/")


def kb_entries() -> List[Dict[str, Any]]:
    """One catalogue entry per entry of the knowledge base, in its order.
    An entry without an id cannot be looked up or evaluated: it is left out."""
    from app.blocks import _knowledge

    file = kb_file()
    return [kb_entry(e, file) for e in _knowledge.load_knowledge()
            if isinstance(e, dict) and e.get("id")]
