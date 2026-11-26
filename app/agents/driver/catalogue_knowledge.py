"""Knowledge entries of the catalogue built from the knowledge itself.

Two sources, both read as they are; nothing here edits, adds to or
re-words their content:

* each entry of the curated construction knowledge base (the JSON the
  ``construction_advisor`` tool searches and evaluates) becomes one
  ``knowledge:<entry id>`` catalogue entry: its title is the name, its
  statement the ``for`` line, its variables the inputs, its provenance
  and credibility tier go into ``source`` verbatim;
* a knowledge file's sections are its own headings.

``source["sections"]`` names what the entry holds, so the model can tell
from a lookup whether the entry or file carries the part it needs (a
table, a checklist, a states list, a section heading) before it calls the
tool that reads it.
"""
from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, List

#: The knowledge base fields that say what an entry IS (its identity and
#: its record), as opposed to what it HOLDS. They fill the entry's name,
#: code and source; every other field with content is a section.
RECORD_FIELDS = ("id", "title", "domain", "type", "provenance",
                 "credibility_tier", "needs_review", "applicability")

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = ("```", "~~~")


def markdown_sections(text: str) -> List[str]:
    """Every heading of a markdown text after its first one (the first is
    the file's name), in order, outside fenced code blocks."""
    out: List[str] = []
    seen_title = False
    fenced = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(_FENCE):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = _HEADING.match(stripped)
        if not m or not m.group(2):
            continue
        if not seen_title:
            seen_title = True
            continue
        out.append(m.group(2))
    return out


def _held(value: Any) -> bool:
    return value not in (None, "", [], {})


def kb_sections(entry: Dict[str, Any]) -> List[str]:
    """The fields an entry holds content in, in the entry's own order,
    leaving out its record fields."""
    return [k for k, v in entry.items() if k not in RECORD_FIELDS and _held(v)]


def _kb_input(name: str, spec: Any) -> Dict[str, Any]:
    spec = spec if isinstance(spec, dict) else {}
    return {"name": name, "label": spec.get("desc") or name, "type": "number",
            "unit": spec.get("unit") or "", "range": None, "required": True, "basis": ""}


def kb_entry(entry: Dict[str, Any], file: str) -> Dict[str, Any]:
    """One catalogue entry for one knowledge base entry."""
    from app.agents.driver import catalogue

    rule_id = str(entry.get("id") or "")
    variables = entry.get("variables") if isinstance(entry.get("variables"), dict) else {}
    thresholds = entry.get("thresholds") if isinstance(entry.get("thresholds"), dict) else {}
    outputs = ([{"name": "result", "unit": thresholds.get("unit") or "", "main": True}]
               if _held(entry.get("expression")) else [])
    out = catalogue._entry(
        "knowledge", rule_id, name=entry.get("title") or rule_id,
        inputs=[_kb_input(k, v) for k, v in variables.items()],
        outputs=outputs,
        code={"tool": "construction_advisor", "rule_id": rule_id},
        source={"registry": "knowledge", "file": file, "entry": rule_id,
                "type": entry.get("type") or "", "domain": entry.get("domain") or "",
                "sections": kb_sections(entry),
                "provenance": copy.deepcopy(entry.get("provenance") or {}),
                "credibility_tier": entry.get("credibility_tier")})
    out["for"] = catalogue._one_line(entry.get("statement"))
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
