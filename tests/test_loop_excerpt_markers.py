"""S-R5b: a chunk the evidence loop fetched is evidence of its own document.

The loop joined its fetches as bare text after the first pass's marked
excerpts, so (1) the provenance gate glued that text onto the previous
excerpt and stripped a correct attribution of the fetched document, and
(2) the sources panel, built from first-pass chunks only, listed a
first-pass document when the answer cited the fetched one. One defect: the
fetched chunks carried no excerpt marker. Every test builds its own input.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents import citation_provenance as cp
from app.agents.runtime import _build_sources_from_audit, _fold_loop_excerpts_into_audit
from app.core.rag import evidence_actions as ea
from app.core.rag.inject import chunk_marker


def _chunk(doc, idx, text, *, name="", score=0.5, page=None):
    return SimpleNamespace(
        doc_id=doc, chunk_index=idx, chunk_id=f"p:{doc}:{idx}", score=score,
        text=text, source_name=name, layer="own", page=page,
    )


def _first_pass(*chunks) -> str:
    return "EXCERPTS\n" + "\n\n".join(chunk_marker(c) for c in chunks)


# --- the marker rides on loop chunks -----------------------------------------

@pytest.mark.parametrize("doc,idx,name", [
    ("doc-a", 3, "Part A schedule.pdf"),
    ("doc-b", 0, "Volume 2 rates.xlsx"),
])
def test_loop_chunk_carries_the_first_pass_marker(doc, idx, name):
    c = _chunk(doc, idx, "row 4.2 concrete 12 m3", name=name)
    out = ea.chunks_to_text([c])
    assert out == chunk_marker(c)
    assert out.startswith(f"[doc_id={doc} chunk={idx} score=")
    assert f"src={name}]" in out


@pytest.mark.parametrize("item", ["plain evidence line", SimpleNamespace(text="no document named")])
def test_unmarkable_items_pass_through_as_text(item):
    out = ea.chunks_to_text([item])
    assert out == (item if isinstance(item, str) else item.text)
    assert "[doc_id=" not in out


@pytest.mark.parametrize("name,page", [("Bill 2 earthworks.pdf", 7), ("", None)])
def test_section_tagged_copy_keeps_what_the_marker_prints(name, page):
    tagged = ea._section_tagged(_chunk("doc-c", 5, "rows", name=name, page=page), "SECTION X")
    out = ea.chunks_to_text([tagged])
    assert out.startswith("[doc_id=doc-c chunk=5 score=")
    assert (f"src={name}]" in out) is bool(name)
    assert (f"page={page}" in out) is bool(page)
    assert "SECTION X\nrows" in out


# --- (1) provenance: the fetched document backs its own attribution ----------

def _merged_content(loop_name: str) -> str:
    first = _first_pass(_chunk("doc-w", 1, "forecast sheet line", name="Cost forecast.xlsx"))
    loop = ea.chunks_to_text([_chunk("doc-q", 9, "item 3.1 rate 45.00", name=loop_name)])
    return first + "\n" + loop


@pytest.mark.parametrize("loop_name", ["Priced bill part 3.pdf", "Bill of quantities vol 1.pdf"])
def test_fetched_chunk_is_its_own_evidence_record(loop_name):
    recs = cp.excerpt_records(_merged_content(loop_name))
    by_doc = {r.doc_id: r for r in recs}
    assert set(by_doc) == {"doc-w", "doc-q"}
    assert by_doc["doc-q"].source_name == loop_name
    assert "item 3.1 rate 45.00" in by_doc["doc-q"].text
    # the first-pass record no longer swallows the fetched text
    assert "item 3.1" not in by_doc["doc-w"].text


@pytest.mark.parametrize("content", ["", "a flat context with no markers"])
def test_excerpt_records_empty_without_markers(content):
    assert cp.excerpt_records(content) == []


@pytest.mark.parametrize("named,kept", [
    ("Priced bill part 3.pdf", True),     # fetched by the loop: backed
    ("Unrelated report 9.pdf", False),    # read by nothing: stripped
])
def test_gate_keeps_a_source_line_for_the_fetched_document(named, kept, monkeypatch):
    monkeypatch.setenv(cp.FLAG_ENV, "1")
    msg = {"role": "system", "content": _merged_content("Priced bill part 3.pdf")}
    answer = f"The rate is 45.00 per item.\n\nSource: {named}"
    out = cp.gate(answer, msg, [{"role": "user", "content": "what is the rate for item 3.1?"}])
    assert (f"Source: {named}" in out) is kept
    assert ("Unverified attribution removed" in out) is (not kept)


# --- (2) the sources panel lists the fetched document ------------------------

def _audit():
    return {"project_id": "p", "chunks": [
        {"doc_id": "doc-w", "chunk_index": 1, "score": 0.81, "layer": "own"},
    ]}


@pytest.mark.parametrize("loop_idx", [9, 2])
def test_fold_appends_loop_excerpts_once(loop_idx):
    audit = _audit()
    content = _first_pass(_chunk("doc-w", 1, "forecast", name="Cost forecast.xlsx")) + "\n" + \
        ea.chunks_to_text([_chunk("doc-q", loop_idx, "rate rows", name="Priced bill.pdf")])
    assert _fold_loop_excerpts_into_audit(audit, content) == 1
    assert _fold_loop_excerpts_into_audit(audit, content) == 0  # held already
    row = audit["chunks"][-1]
    assert (row["doc_id"], row["chunk_index"], row["via"]) == ("doc-q", loop_idx, "evidence_loop")
    assert row["score"] is None and row["source_name"] == "Priced bill.pdf"
    assert len(audit["chunks"]) == 2


@pytest.mark.parametrize("audit,content", [
    (None, "[doc_id=x chunk=1 score=0.1] t"),
    ({"chunks": "not a list"}, "[doc_id=x chunk=1 score=0.1] t"),
])
def test_fold_ignores_a_missing_or_malformed_audit(audit, content):
    assert _fold_loop_excerpts_into_audit(audit, content) == 0


@pytest.mark.parametrize("folded,expected", [(True, "doc-q"), (False, "doc-w")])
def test_panel_lists_the_cited_fetched_chunk(folded, expected, monkeypatch):
    names = {"doc-w": "Cost forecast.xlsx", "doc-q": "Priced bill.pdf"}
    monkeypatch.setattr("app.core.projects.get_document",
                        lambda did: {"original_name": names.get(did, "")})
    audit = _audit()
    if folded:
        content = ea.chunks_to_text([_chunk("doc-q", 9, "rate rows", name="Priced bill.pdf")])
        _fold_loop_excerpts_into_audit(audit, content)
    answer = "Item 3.1 is priced at 45.00 [source: Priced bill.pdf, chunk 9]."
    rows = _build_sources_from_audit(audit, answer)
    assert rows and rows[0]["doc_id"] == expected
    # folded: the panel lists exactly what the answer cited; not folded: the
    # cite matches nothing and the panel falls back to the first-pass chunk
    assert [r["doc_id"] for r in rows] == [expected]
