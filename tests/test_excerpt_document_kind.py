"""S-R11: the model is told what each excerpt is, and one rule says which
statement governs.

Before: the marker carried class, layer, page, rev and src only, so the
model could not tell a specification from a drawing or a contractor's
submission; a long file name was cut to its tail, losing the document number
at its head; and the prompt held two contradicting rules on differing values
("flag, do not arbitrate" against "a second number is a wrong answer").
Every test builds its own input; no real document, project or value is named.
"""
from __future__ import annotations

import re
from pathlib import Path
import pytest

from app.agents import citation_provenance as cp
from app.core.rag import evidence_actions as ea
from app.core.rag import inject
from app.core.rag.inject import (
    _truncate_source_name,
    annotate_document_kinds,
    chunk_marker,
    format_chunks_as_system_message,
)
from app.core.rag.vector_store import Chunk

_PROMPT = Path(__file__).resolve().parents[1] / "app/agents/configs/project-assistant.md"


def _chunk(doc, idx, *, name="", page=None, doc_type="", doc_role="", text="body text"):
    c = Chunk(chunk_id=f"p:{doc}:{idx}", project_id="p", doc_id=doc,
              chunk_index=idx, text=text, score=0.4)
    c.source_name, c.page, c.doc_type, c.doc_role = name, page, doc_type, doc_role
    return c


# --- the marker says what the document is ------------------------------------

@pytest.mark.parametrize("doc_type,doc_role,page,name", [
    ("specification", "baseline_schedule", 12, "Section 9 finishes.pdf"),
    ("method_statement", "weekly_report", 3, "MS lifting plan rev C.docx"),
])
def test_marker_carries_kind_role_and_page(doc_type, doc_role, page, name):
    out = chunk_marker(_chunk("d1", 4, name=name, page=page,
                              doc_type=doc_type, doc_role=doc_role))
    assert f" kind={doc_type} " in out and f" role={doc_role} " in out
    assert f" page={page} " in out
    # The parsers that read the marker still read it whole.
    head = out[:out.index("]")]
    assert cp._MARKER_SRC_RE.search(head).group(1) == name
    assert cp._MARKER_PAGE_RE.search(head).group(1) == str(page)
    assert out.startswith("[doc_id=d1 chunk=4 score=0.400 class=")
    assert head.rindex(" src=") > head.index(" kind=")


@pytest.mark.parametrize("doc_type,doc_role", [("document", "other"), ("", "")])
def test_a_column_default_is_not_printed_as_a_kind(doc_type, doc_role):
    out = chunk_marker(_chunk("d2", 0, name="x.pdf", doc_type=doc_type, doc_role=doc_role))
    assert " kind=" not in out and " role=" not in out


@pytest.mark.parametrize("doc_type", ["general notes", "bill of quantities"])
def test_a_spaced_kind_stays_one_attribute(doc_type):
    out = chunk_marker(_chunk("d3", 1, name="n.pdf", doc_type=doc_type))
    assert f" kind={doc_type.replace(' ', '_')} " in out


# --- a shortened name keeps its document number ------------------------------

@pytest.mark.parametrize("number,rest", [
    ("QQ-ZZ-104-0007-ARC-DWG-000213-C", "_General arrangement of the north podium levels 1 to 4.pdf"),
    ("K7.220.15", " Particular specification for mechanical ventilation works (exp. 9Mar27).pdf"),
])
def test_a_shortened_name_keeps_its_leading_document_number(number, rest):
    name = number + rest
    assert len(name) > inject._MAX_SOURCE_NAME_CHARS
    out = _truncate_source_name(name)
    assert len(out) <= inject._MAX_SOURCE_NAME_CHARS
    assert out.startswith(number + "…")
    assert name.endswith(out.split("…", 1)[1])


@pytest.mark.parametrize("name", [
    "Minutes of the coordination meeting held on site with all subcontractors present.pdf",
    "Report_without_a_document_number_at_its_head_and_long_enough_to_cut.pdf",
])
def test_a_name_without_a_number_keeps_its_tail(name):
    out = _truncate_source_name(name)
    assert out == "…" + name[-(inject._MAX_SOURCE_NAME_CHARS - 1):]


@pytest.mark.parametrize("name", ["short.pdf", "AB-12 notes.pdf"])
def test_a_name_that_fits_is_untouched(name):
    assert _truncate_source_name(name) == name


# --- the kind is read from the documents rows, in one batch ------------------

def _fake_kinds(monkeypatch, table):
    from app.core import projects
    calls = []

    def kinds(ids):
        calls.append(list(ids))
        return {i: table[i] for i in ids if i in table}
    monkeypatch.setattr(projects, "document_kinds", kinds)
    return calls


@pytest.mark.parametrize("table", [
    {"a": ("drawing", "other"), "b": ("specification", "baseline_schedule")},
    {"a": ("bill_of_quantities", "weekly_report")},
])
def test_annotate_reads_once_and_fills_only_missing(monkeypatch, table):
    calls = _fake_kinds(monkeypatch, table)
    held = _chunk("c", 0, doc_type="kept", doc_role="kept_role")
    chunks = [_chunk("a", 0), _chunk("a", 1), _chunk("b", 0), held]
    annotate_document_kinds(chunks)
    assert len(calls) == 1 and "c" not in calls[0]
    for c in chunks[:3]:
        assert (c.doc_type, c.doc_role) == table.get(c.doc_id, ("", ""))
    assert (held.doc_type, held.doc_role) == ("kept", "kept_role")


@pytest.mark.parametrize("exc", [RuntimeError("db down"), ValueError("bad id")])
def test_a_failed_read_hands_the_excerpt_without_a_kind(monkeypatch, exc):
    from app.core import projects

    def boom(ids):
        raise exc
    monkeypatch.setattr(projects, "document_kinds", boom)
    c = _chunk("a", 0, name="z.pdf")
    annotate_document_kinds([c])
    assert (c.doc_type, c.doc_role) == ("", "")
    assert " kind=" not in chunk_marker(c)


@pytest.mark.parametrize("doc_type", ["drawing", "specification"])
def test_first_pass_and_loop_excerpts_both_carry_the_kind(monkeypatch, doc_type):
    _fake_kinds(monkeypatch, {"a": (doc_type, "baseline_schedule")})
    first = format_chunks_as_system_message([_chunk("a", 2, name="f.pdf")], 1, "q")["content"]
    loop = ea.chunks_to_text([_chunk("a", 3, name="f.pdf")])
    for text in (first, loop):
        assert re.search(rf"\[doc_id=a chunk=\d score=[^\]]* kind={doc_type} role=baseline_schedule", text)


@pytest.fixture
def one_project(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import projects, users
    users.init_db()
    projects.init_db()
    users.ensure_user_exists("u-kind")
    return projects.create_project("Synthetic Kind Works", user_id="u-kind")["id"]


@pytest.mark.parametrize("name,role", [
    ("pour sequence programme.xer", "baseline_schedule"),
    ("site diary week 14.pdf", "weekly_report"),
])
def test_document_kinds_reads_the_stored_columns(one_project, name, role):
    from app.core import projects
    doc = projects.add_document(one_project, original_name=name, stored_as=name,
                                file_path=f"/tmp/{name}", size=1, role=role)
    got = projects.document_kinds([doc["id"], "no-such-doc"])
    assert got == {doc["id"]: (doc["doc_type"], doc["doc_role"])}
    assert doc["doc_role"] == role


# --- one rule in the prompt, and it names only what the marker prints --------

def test_the_prompt_holds_one_precedence_rule():
    text = _PROMPT.read_text(encoding="utf-8")
    assert "flagged, not arbitrated" not in text
    assert "Which statement governs: one rule" in text
    rule = text[text.index("8. **Which statement governs"):text.index("9. **")]
    for clause in ("Item-specific outranks general", "Different values: give both",
                   "Same value: say so", "exempt from the one-figure rule"):
        assert clause in rule
    one_fig = text[text.index("**One figure per quantity asked.**"):]
    assert "exception is rule 8" in one_fig.split("\n", 1)[0]


@pytest.mark.parametrize("doc_type,doc_role", [("drawing", "baseline_schedule"),
                                               ("specification", "weekly_report")])
def test_every_marker_field_the_rule_names_is_emitted(doc_type, doc_role):
    text = _PROMPT.read_text(encoding="utf-8")
    rule = text[text.index("8. **Which statement governs"):text.index("9. **")]
    named = set(re.findall(r"`(\w+)=`", rule))
    assert named == {"kind", "role", "page", "src"}
    out = chunk_marker(_chunk("d", 0, name="n.pdf", page=2, doc_type=doc_type, doc_role=doc_role))
    for field in named:
        assert f" {field}=" in out


@pytest.mark.parametrize("doc_type,doc_role", [("drawing", "baseline_schedule"),
                                               ("method_statement", "weekly_report")])
def test_chunk_kind_never_reaches_the_wire(doc_type, doc_role):
    d = _chunk("w", 0, doc_type=doc_type, doc_role=doc_role).to_dict()
    assert "doc_type" not in d and "doc_role" not in d
