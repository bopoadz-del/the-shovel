"""An entry's provenance names the course when its own statement says it
comes from course material, and only then.

D5 (no answer names its course source): the knowledge tool hands the model
each entry's ``provenance.source``, and the refusal guard tells the model to
name "the source its provenance gives". On 24 Nov every entry whose statement
says "from course material" still carried a "... site-experience reference"
label, so the only source the model was handed never named the course.

The rule reads each entry's own statement; it names no entry.
"""
from __future__ import annotations

import re

import pytest

from app.blocks import _knowledge as kb

#: The statement's own declaration that the entry comes from the course.
_DECLARES_COURSE = re.compile(
    r"\b(?:from|owner-supplied)\s+course\s+material\b", re.IGNORECASE
)
#: A provenance source that names the course.
_NAMES_COURSE = re.compile(r"\bcourse\s+material\b", re.IGNORECASE)


def _mismatches(entries):
    """``(unnamed, unbacked)``: entries that declare the course but whose
    source does not name it, and entries whose source names the course but
    whose statement never says so."""
    unnamed, unbacked = [], []
    for e in entries:
        declares = bool(_DECLARES_COURSE.search(e.get("statement") or ""))
        source = (e.get("provenance") or {}).get("source") or ""
        names = bool(_NAMES_COURSE.search(source))
        if declares and not names:
            unnamed.append(e["id"])
        if names and not declares:
            unbacked.append(e["id"])
    return unnamed, unbacked


def _entry(statement, source):
    return {"id": "x", "statement": statement, "provenance": {"source": source}}


@pytest.mark.parametrize(
    "statement, source, want",
    [
        # Declares the course, source names it: clean.
        ("Area = a x b. Site-practice knowledge from course material.",
         "owner-supplied course material (pavements)", ([], [])),
        ("Uses the course factor (owner-supplied course material, module 2, p4).",
         "owner-supplied course material (structures)", ([], [])),
        # Declares the course, source does not name it: unnamed.
        ("Area = a x b. Site-practice knowledge from course material.",
         "pavements site-experience reference", (["x"], [])),
        ("Uses the course factor (owner-supplied course material, module 2, p4).",
         "", (["x"], [])),
        # Never declares the course, source claims it: unbacked.
        ("Volume = l x w x d, a general relation.",
         "owner-supplied course material (pavements)", ([], ["x"])),
        ("A taxonomy of methods; the course is not mentioned as its origin.",
         "Course material, module 9", ([], ["x"])),
        # Never declares the course, source does not claim it: clean.
        ("Volume = l x w x d, a general relation.",
         "pavements site-experience reference", ([], [])),
        ("A wearing course of asphalt is 40 mm.",
         "asphalt site-experience reference", ([], [])),
    ],
)
def test_mismatch_rule(statement, source, want):
    assert _mismatches([_entry(statement, source)]) == want


@pytest.mark.parametrize("side", [0, 1])
def test_shipped_knowledge_names_the_course_exactly_where_it_declares_it(side):
    entries = kb.load_knowledge()
    declaring = [e for e in entries if _DECLARES_COURSE.search(e.get("statement") or "")]
    assert declaring, "no entry declares the course; the rule would test nothing"
    assert _mismatches(entries)[side] == []


@pytest.mark.parametrize("pick", [0, -1])
def test_the_tool_result_carries_the_course_source(pick):
    """The loader's search result is what the chat tool hands the model: the
    course-declaring entry's result carries the course in its provenance."""
    declaring = [e for e in kb.load_knowledge()
                 if _DECLARES_COURSE.search(e.get("statement") or "")]
    entry = declaring[pick]
    hits = kb.search_knowledge(entry["title"], top_k=len(declaring) + 100)
    hit = next(h for h in hits if h.get("id") == entry["id"])
    assert _NAMES_COURSE.search(hit["provenance"]["source"])
