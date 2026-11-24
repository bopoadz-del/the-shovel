"""An entry's provenance names the course when the entry itself says it
comes from course material, and only then.

D5 (no answer names its course source): the knowledge tool hands the model
each entry's ``provenance.source``, and the refusal guard tells the model to
name "the source its provenance gives". On 24 Nov every entry whose statement
says "from course material" still carried a "... site-experience reference"
label, so the only source the model was handed never named the course.

An entry says so in its statement or in its own ``provenance.course_reference``
field (the page it was taken from). The field is a declaration even when the
statement prose is silent: four entries on 25 Nov cited a course page there
and still carried the site-experience label.

The rule reads each entry's own fields; it names no entry.
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


def _declares_course(entry):
    """The entry's statement or its own course reference says it comes from
    course material."""
    reference = (entry.get("provenance") or {}).get("course_reference") or ""
    return any(
        _DECLARES_COURSE.search(text)
        for text in (entry.get("statement") or "", reference)
    )


def _mismatches(entries):
    """``(unnamed, unbacked)``: entries that declare the course but whose
    source does not name it, and entries whose source names the course but
    whose statement never says so."""
    unnamed, unbacked = [], []
    for e in entries:
        declares = _declares_course(e)
        source = (e.get("provenance") or {}).get("source") or ""
        names = bool(_NAMES_COURSE.search(source))
        if declares and not names:
            unnamed.append(e["id"])
        if names and not declares:
            unbacked.append(e["id"])
    return unnamed, unbacked


def _entry(statement, source, reference=None):
    provenance = {"source": source}
    if reference is not None:
        provenance["course_reference"] = reference
    return {"id": "x", "statement": statement, "provenance": provenance}


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


@pytest.mark.parametrize(
    "reference, source, want",
    [
        # The statement is silent; the entry's own course reference declares
        # the course, so a site-experience source leaves it unnamed.
        ("owner-supplied course material, module 4, p12",
         "piling site-experience reference", (["x"], [])),
        ("from course material, part 2 p7 (retyped sheet)",
         "", (["x"], [])),
        # The same references with a source that names the course: clean,
        # and not unbacked although the statement never says so.
        ("owner-supplied course material, module 4, p12",
         "owner-supplied course material (piling)", ([], [])),
        ("from course material, part 2 p7 (retyped sheet)",
         "owner-supplied course material (asphalt)", ([], [])),
        # A reference field that does not say course material declares nothing.
        ("BS 8004 clause 6.4", "owner-supplied course material (piling)", ([], ["x"])),
        ("", "piling site-experience reference", ([], [])),
    ],
)
def test_course_reference_field_declares_the_course(reference, source, want):
    statement = "Capacity = k x A, a general relation."
    assert _mismatches([_entry(statement, source, reference)]) == want


@pytest.mark.parametrize("side", [0, 1])
def test_shipped_knowledge_names_the_course_exactly_where_it_declares_it(side):
    entries = kb.load_knowledge()
    declaring = [e for e in entries if _declares_course(e)]
    assert declaring, "no entry declares the course; the rule would test nothing"
    assert _mismatches(entries)[side] == []


@pytest.mark.parametrize("pick", [0, -1])
def test_the_tool_result_carries_the_course_source(pick):
    """The loader's search result is what the chat tool hands the model: the
    course-declaring entry's result carries the course in its provenance."""
    declaring = [e for e in kb.load_knowledge() if _declares_course(e)]
    entry = declaring[pick]
    hits = kb.search_knowledge(entry["title"], top_k=len(declaring) + 100)
    hit = next(h for h in hits if h.get("id") == entry["id"])
    assert _NAMES_COURSE.search(hit["provenance"]["source"])
