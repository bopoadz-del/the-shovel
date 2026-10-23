"""PR — _build_sources_from_audit derives Sources from agent citations.

Pre-PR: the SSE end event always returned the top-3 retrieved chunks by
score, ignoring what the agent actually cited in its text response.
With the gpt-oss model emitting "[source: <file>, chunk N]" markers,
the Sources panel now reflects what the operator can verify in the
answer rather than the top retrieval slice.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _build_sources_from_audit,
    _extract_cited_chunk_indexes,
    _normalise_filename,
)


def test_extract_single_citation():
    txt = "These points are listed in the procedure’s checklist [source: XYZ-406_HSE.pdf, chunk 65]."
    out = _extract_cited_chunk_indexes(txt)
    assert out == [("XYZ-406_HSE.pdf", 65)]


def test_extract_multiple_chunk_numbers():
    txt = "See the policy [source: Vendor PQ.pdf, chunks 16, 34, 55] for full detail."
    out = _extract_cited_chunk_indexes(txt)
    assert out == [
        ("Vendor PQ.pdf", 16),
        ("Vendor PQ.pdf", 34),
        ("Vendor PQ.pdf", 55),
    ]


def test_extract_filename_only_citation():
    txt = "Per the spec [source: Doc.pdf]."
    out = _extract_cited_chunk_indexes(txt)
    assert out == [("Doc.pdf", -1)]


def test_extract_handles_unicode_hyphens():
    """gpt-oss sometimes rewrites file names with non-breaking hyphens.
    The extractor returns whatever the model emitted; matching is then
    handled by _normalise_filename at the lookup step."""
    txt = "[source: XYZ‑1000.pdf, chunk 1]"  # non-breaking hyphen
    out = _extract_cited_chunk_indexes(txt)
    assert out[0][1] == 1
    assert "XYZ" in out[0][0]


def test_normalise_filename_collapses_dashes():
    a = "XYZ‑406_HSE Audit.PDF"  # non-breaking hyphen + uppercase ext
    b = "xyz-406_hse audit.pdf"
    assert _normalise_filename(a) == _normalise_filename(b)


def _make_audit(chunks: list[dict]) -> dict:
    return {"chunks": chunks}


def _stub_get_document(monkeypatch, original_name: str) -> None:
    """Stub app.core.projects.get_document without relying on sys.modules."""
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": original_name},
    )


def test_build_sources_uses_cited_chunks_when_present(monkeypatch):
    """When the agent's text cites a specific chunk, the Sources list
    returns THAT chunk, not the top-3 by score."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 0,  "score": 0.78},  # high-score scaffolding
        {"doc_id": "d1", "chunk_index": 4,  "score": 0.74},
        {"doc_id": "d1", "chunk_index": 18, "score": 0.77},
        {"doc_id": "d1", "chunk_index": 65, "score": 0.69},  # the actually-cited one
    ])
    # Stub the doc-name lookup so we don't need a real DB.
    _stub_get_document(monkeypatch, "XYZ-406_HSE.pdf")

    text = "Per the procedure [source: XYZ-406_HSE.pdf, chunk 65]."
    out = _build_sources_from_audit(audit, text)

    assert len(out) == 1
    assert (out[0]["chunk_index"], out[0]["page_or_section"]) == (65, "")
    assert out[0]["doc_name"] == "XYZ-406_HSE.pdf"
    # Score 0.69 → Medium, not High
    assert out[0]["confidence"] == "Medium"


def test_build_sources_falls_back_when_no_citations(monkeypatch):
    """When the agent text has no [source: ...] markers, behave like
    pre-PR-110: top-3 retrieved chunks by score."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 0,  "score": 0.78},
        {"doc_id": "d1", "chunk_index": 4,  "score": 0.74},
        {"doc_id": "d1", "chunk_index": 18, "score": 0.77},
        {"doc_id": "d1", "chunk_index": 65, "score": 0.69},
    ])
    _stub_get_document(monkeypatch, "x.pdf")

    out = _build_sources_from_audit(audit, "answer with no citations")
    assert len(out) == 3
    # Top 3 by score desc: 0.78, 0.77, 0.74 — chunks 0, 18, 4
    assert [c["chunk_index"] for c in out] == [0, 18, 4]
    assert all(c["page_or_section"] == "" for c in out)


def test_build_sources_falls_back_when_citation_doesnt_match(monkeypatch):
    """Citation that references a non-injected chunk → fall back, don't
    return an empty list."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 0, "score": 0.78},
    ])
    _stub_get_document(monkeypatch, "x.pdf")

    out = _build_sources_from_audit(audit, "see [source: other.pdf, chunk 99]")
    # Citation didn't match any injected chunk → fall back
    assert len(out) == 1
    assert (out[0]["chunk_index"], out[0]["page_or_section"]) == (0, "")


def test_build_sources_unicode_dash_matches(monkeypatch):
    """The agent rewrote the source name with a non-breaking hyphen.
    The lookup still resolves via _normalise_filename."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 65, "score": 0.69},
    ])
    _stub_get_document(monkeypatch, "XYZ-406_HSE.pdf")

    text = "see [source: XYZ‑406_HSE.pdf, chunk 65]"  # NBH
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 1
    assert (out[0]["chunk_index"], out[0]["page_or_section"]) == (65, "")


def test_build_sources_empty_audit_returns_empty():
    out = _build_sources_from_audit({"chunks": []}, "any text")
    assert out == []


# ── gpt-oss-120b inline-bracketed form: "Source: [filename], chunk N" ──
# The pilot model (gpt-oss:120b-cloud) emits the filename INSIDE brackets
# with the chunk number AFTER the bracket, inline mid-sentence — a form none
# of the prior patterns matched, so the Sources panel silently fell back to
# top-3-by-score instead of the chunks the answer actually cited (2026-06-30).

def test_extract_inline_bracketed_source_single_chunk():
    txt = "Trees must be protected (Source: [AB-2022-202 - Site Demolition Part 3], chunk 941)."
    out = _extract_cited_chunk_indexes(txt)
    assert any(idx == 941 for _, idx in out), out


def test_extract_inline_bracketed_source_chunk_range():
    txt = "See the schedule (Source: [AB-2022-202 - Site Demolition Part 2], chunks 1988-1990)."
    out = _extract_cited_chunk_indexes(txt)
    nums = {idx for _, idx in out}
    assert 1988 in nums, out  # at least the range start is captured


def test_build_sources_matches_inline_bracket_cite_by_chunk_index(monkeypatch):
    """The cited chunk index uniquely identifies the injected chunk even when
    gpt-oss truncated the filename with an ellipsis. The Sources panel must
    surface THAT chunk, not the top-3-by-score scaffolding."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 5,   "score": 0.80},   # high-score scaffolding
        {"doc_id": "d1", "chunk_index": 12,  "score": 0.78},
        {"doc_id": "d2", "chunk_index": 941, "score": 0.64},   # the actually-cited chunk
    ])
    _stub_get_document(
        monkeypatch,
        "AB-2022-202 - Site Demolition and Site Clearance Works Package 1 Volume 2 Specs Part 3.pdf",
    )
    text = "Trees must be protected (Source: [AB-2022-202 - Site Demolition … Part 3], chunk 941)."
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 1, out
    assert (out[0]["chunk_index"], out[0]["page_or_section"]) == (941, "")


# ── PR #111 — bracketless "Source:" prefix form (gpt-oss variant) ──────

def test_extract_bracketless_line_form_single():
    """gpt-oss sometimes emits 'Source: foo.pdf, chunk N.' on its own
    line at the end of the answer, no brackets."""
    txt = "Some answer body here.\nSource: XYZ-406_HSE.pdf, chunk 65."
    out = _extract_cited_chunk_indexes(txt)
    assert ("XYZ-406_HSE.pdf", 65) in out


def test_extract_bracketless_at_start():
    """Source: prefix at the very start of the string (no leading newline)."""
    txt = "Source: vendor.pdf, chunk 4."
    out = _extract_cited_chunk_indexes(txt)
    assert ("vendor.pdf", 4) in out


def test_extract_bracketless_multichunk():
    txt = "See for detail.\nSources: vendor.pdf, chunks 16, 34, 55."
    out = _extract_cited_chunk_indexes(txt)
    assert ("vendor.pdf", 16) in out
    assert ("vendor.pdf", 34) in out
    assert ("vendor.pdf", 55) in out


def test_build_sources_uses_bracketless_citation(monkeypatch):
    """End-to-end: agent emits bracketless 'Source:' form; Sources panel
    still surfaces the cited chunk, not the top retrieval slice."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 0,  "score": 0.78},
        {"doc_id": "d1", "chunk_index": 65, "score": 0.69},
    ])
    _stub_get_document(monkeypatch, "XYZ-406_HSE.pdf")

    text = "Answer body.\nSource: XYZ-406_HSE.pdf, chunk 65."
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 1
    assert (out[0]["chunk_index"], out[0]["page_or_section"]) == (65, "")


# ── PR #112 — [doc_id=X chunk=N] form (gpt-oss technical-precision style) ──

def test_extract_doc_id_form_with_separators():
    """gpt-oss sometimes emits [doc_id=d0c0000e, chunk 65, score 0.697]."""
    txt = "see [doc_id=d0c0000e, chunk 65, score 0.697] for detail"
    out = _extract_cited_chunk_indexes(txt)
    assert ("d0c0000e", 65) in out


def test_extract_doc_id_form_kv_separators():
    """The RAG-injection header style: [doc_id=X chunk=N score=Y]."""
    txt = "[doc_id=abc12345 chunk=4 score=0.81]"
    out = _extract_cited_chunk_indexes(txt)
    assert ("abc12345", 4) in out


def test_build_sources_uses_doc_id_citation(monkeypatch):
    """When the agent cites by [doc_id=X chunk=N], the panel surfaces
    THAT chunk by direct doc_id match — bypasses the filename match
    entirely (avoids name-mismatch fall-through)."""
    audit = _make_audit([
        {"doc_id": "d0c0000e", "chunk_index": 0,  "score": 0.78},
        {"doc_id": "d0c0000e", "chunk_index": 4,  "score": 0.74},
        {"doc_id": "d0c0000e", "chunk_index": 65, "score": 0.69},
    ])
    _stub_get_document(monkeypatch, "XYZ-406_HSE.pdf")

    text = "Per the procedure [doc_id=d0c0000e, chunk 65, score 0.697]."
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 1
    assert (out[0]["chunk_index"], out[0]["page_or_section"]) == (65, "")
    assert out[0]["doc_id"] == "d0c0000e"


def test_build_sources_doc_id_mismatch_falls_back(monkeypatch):
    """Unknown doc_id in citation → no match → fall back to top-3."""
    audit = _make_audit([
        {"doc_id": "a", "chunk_index": 1, "score": 0.5},
        {"doc_id": "a", "chunk_index": 2, "score": 0.6},
    ])
    _stub_get_document(monkeypatch, "x.pdf")

    text = "[doc_id=unknown chunk=99]"
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 2  # both chunks (only 2 injected, top-3 cap returns all)


# ── P0A: smart-quote / inline / prose citation hardening ────────────────

def test_extract_smart_quote_source_inline():
    """Model emits Source: “File Name.xlsx”, which says..."""
    txt = 'Source: "Diff BOQ Qty Vs Revised Qty.xlsx", which lists PVC pipe sizes.'
    out = _extract_cited_chunk_indexes(txt)
    assert any("Diff BOQ Qty Vs Revised Qty.xlsx" in fname for fname, _ in out)


def test_extract_smart_quote_source_with_chunk():
    txt = 'Source: “XYZ-302_Risk Management.pdf”, chunk 8.'
    out = _extract_cited_chunk_indexes(txt)
    assert ("XYZ-302_Risk Management.pdf", 8) in out


def test_extract_filename_with_spaces_and_punctuation(monkeypatch):
    """A filename containing spaces, commas (when no chunk suffix), and
    typographic quotes still resolves to the injected chunk."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 0, "score": 0.8},
    ])
    _stub_get_document(monkeypatch, "Diff BOQ Qty Vs Revised Qty.xlsx")

    text = 'Source: “Diff BOQ Qty Vs Revised Qty.xlsx”, which lists PVC sizes.'
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 1
    assert out[0]["doc_name"] == "Diff BOQ Qty Vs Revised Qty.xlsx"


def test_build_sources_emits_fallback_when_citation_unparseable(monkeypatch):
    """If the model mentions a filename but the formal parser cannot map it
    and the text does not match any injected doc_name, the top-3 retrieved
    chunks must still be emitted so the right panel never goes empty."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 0, "score": 0.9},
        {"doc_id": "d2", "chunk_index": 3, "score": 0.7},
        {"doc_id": "d3", "chunk_index": 5, "score": 0.6},
    ])
    _stub_get_document(monkeypatch, "Real Document.pdf")

    text = "The answer is based on some external document not in the project."
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 3
    assert [c["chunk_index"] for c in out] == [0, 3, 5]
    assert all(c["page_or_section"] == "" for c in out)


def test_build_sources_uses_filename_mention_fallback(monkeypatch):
    """If the model names an injected document in prose without a formal
    citation marker, the right panel should still surface that document's
    highest-scoring chunk."""
    audit = _make_audit([
        {"doc_id": "d1", "chunk_index": 0, "score": 0.9},
        {"doc_id": "d2", "chunk_index": 3, "score": 0.7},
    ])

    def doc_lookup(did):
        return {"original_name": "Diff BOQ Qty Vs Revised Qty.xlsx"} if did == "d1" else {"original_name": "Other.pdf"}

    monkeypatch.setattr("app.core.projects.get_document", doc_lookup)

    text = "The storm-water pipe sizes come from Diff BOQ Qty Vs Revised Qty.xlsx."
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 1
    assert out[0]["doc_name"] == "Diff BOQ Qty Vs Revised Qty.xlsx"
    assert (out[0]["chunk_index"], out[0]["page_or_section"]) == (0, "")


# ── P0B: source contract hardening ───────────────────────────────────────────

from app.agents.runtime import _clean_path_label, _sanitize_citation_labels


def test_clean_path_label_strips_windows_drive():
    assert _clean_path_label(r"X:\Example Drive\500-Design Management\XYZ-501.pdf") == "XYZ-501.pdf"


def test_clean_path_label_strips_unix_path():
    assert _clean_path_label("/home/user/projects/specs/XYZ-501.pdf") == "XYZ-501.pdf"


def test_clean_path_label_strips_network_share():
    assert _clean_path_label(r"\\server\share\folder\doc.pdf") == "doc.pdf"


def test_clean_path_label_leaves_basename_alone():
    assert _clean_path_label("XYZ-501.pdf") == "XYZ-501.pdf"


def test_sanitize_citation_labels_cleans_bracketed_windows_path():
    raw = "See [source: X:\\Example Drive\\XYZ-501.pdf, chunk 3] for details."
    cleaned = _sanitize_citation_labels(raw)
    assert "X:\\Example Drive" not in cleaned
    assert "[source: XYZ-501.pdf, chunk 3]" in cleaned


def test_sanitize_citation_labels_cleans_chinese_bracket_path():
    raw = "See 【source: X:\\Example Drive\\XYZ-501.pdf, chunk 3】 for details."
    cleaned = _sanitize_citation_labels(raw)
    assert "X:\\Example Drive" not in cleaned
    assert "【source: XYZ-501.pdf, chunk 3】" in cleaned


def test_sanitize_citation_labels_cleans_bracketless_source_line():
    raw = "Answer body.\nSource: \\server\\share\\folder\\doc.pdf, chunk 8."
    cleaned = _sanitize_citation_labels(raw)
    assert "\\server" not in cleaned
    assert "Source: doc.pdf, chunk 8" in cleaned


def test_extract_chinese_bracket_source():
    txt = "Per the procedure 【source: XYZ-406_HSE.pdf, chunk 65】."
    out = _extract_cited_chunk_indexes(txt)
    assert ("XYZ-406_HSE.pdf", 65) in out


def test_build_sources_cleans_raw_doc_name(monkeypatch):
    audit = {
        "project_id": "proj_x",
        "chunks": [
            {"doc_id": "d1", "chunk_index": 4, "chunk_id": "proj_x:d1:4", "score": 0.82},
        ],
    }
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": r"X:\Example Drive\XYZ-406_HSE.pdf"},
    )
    out = _build_sources_from_audit(audit, "citation text")
    assert len(out) == 1
    assert out[0]["doc_name"] == "XYZ-406_HSE.pdf"
    assert out[0]["project_id"] == "proj_x"
    assert out[0]["chunk_index"] == 4
    assert out[0]["chunk_id"] == "proj_x:d1:4"


# ── R4: Sources must match the answer; preview needs the owning project ──

DD23 = (
    "AB-2023-101_the client project II Infrastructure Package 1_"
    "Vol 1 - Conditions of Contract.pdf"
)
DD22 = "AB-2022-202 - Volume 1 - Conditions of Contract.pdf"


def _patch_doc_names(monkeypatch, mapping: dict[str, str]) -> None:
    def lookup(did):
        return {"original_name": mapping.get(did, "unknown.pdf")}

    monkeypatch.setattr("app.core.projects.get_document", lookup, raising=False)
    import app.core.projects as projects_mod
    monkeypatch.setattr(projects_mod, "get_document", lookup)


def test_extract_src_equals_citation():
    txt = (
        "Delay Damages are 0.1% per day "
        "[doc_id=ab12cd34 chunk=12 score=0.88 src=AB-2023-101_Vol 1.pdf]."
    )
    out = _extract_cited_chunk_indexes(txt)
    assert any("AB-2023-101" in fname for fname, _ in out), out


def test_build_sources_stamps_chunk_owner_project_id(monkeypatch):
    """Preview fetch uses the cited file's owner, not only the workspace."""
    _patch_doc_names(monkeypatch, {"d1": "cited_spec.pdf"})
    audit = {
        "project_id": "workspace",
        "chunks": [{
            "doc_id": "d1", "chunk_index": 1, "chunk_id": "c1",
            "project_id": "projects_folder", "score": 0.82,
            "layer": "master_corpus",
        }],
    }
    out = _build_sources_from_audit(
        audit, "Per the spec [source: cited_spec.pdf, chunk 1].",
    )
    assert len(out) == 1
    assert out[0]["doc_id"] == "d1"
    assert out[0]["project_id"] == "projects_folder"
    assert out[0]["layer"] == "master_corpus"


def test_sources_follow_contract_named_in_answer(monkeypatch):
    """Answer cites AB-2023-101 in prose; panel must not list AB-2022."""
    _patch_doc_names(monkeypatch, {"d23": DD23, "d22": DD22})
    audit = {
        "project_id": "proj_a",
        "user_message_preview": "What is the Time for Completion?",
        "identifier_miss": False,
        "threshold_fired": False,
        "chunks": [
            {"doc_id": "d22", "chunk_index": 4, "chunk_id": "c22",
             "project_id": "proj_a", "score": 0.91, "layer": "own"},
            {"doc_id": "d23", "chunk_index": 12, "chunk_id": "c23",
             "project_id": "proj_a", "score": 0.80, "layer": "own"},
        ],
    }
    text = "Per AB-2023-101, Time for Completion is 365 calendar days."
    out = _build_sources_from_audit(audit, text)
    names = [s["doc_name"] for s in out]
    assert names, "prose named AB-2023-101 — Sources must not be empty"
    assert all("AB-2023-101" in n for n in names), names
    assert not any("AB-2022" in n for n in names), names


# ── Hardening: equal-score ties must resolve deterministically ───────────
# Kimi K2 is pinned to temperature=1, so the answer text (and its citations)
# vary run-to-run. Retrieval is deterministic, so the panel must be too: two
# chunks that tie on score can never flip which one survives the top-N
# truncation. The tie-break is chunk_id asc.

def test_fallback_top3_is_deterministic_under_equal_scores(monkeypatch):
    """Equal-score chunks must land in the top-3 in a stable chunk_id order
    regardless of the order they arrive in the audit."""
    _stub_get_document(monkeypatch, "x.pdf")
    base = [
        {"doc_id": "d1", "chunk_index": 0, "chunk_id": "p:z", "score": 0.80},
        {"doc_id": "d2", "chunk_index": 1, "chunk_id": "p:a", "score": 0.80},
        {"doc_id": "d3", "chunk_index": 2, "chunk_id": "p:m", "score": 0.80},
        {"doc_id": "d4", "chunk_index": 3, "chunk_id": "p:q", "score": 0.80},
    ]
    text = "answer with no citations"
    forward = _build_sources_from_audit({"chunks": list(base)}, text)
    reverse = _build_sources_from_audit({"chunks": list(reversed(base))}, text)
    # Same set and same order irrespective of arrival order — chunk_id asc.
    assert [s["chunk_id"] for s in forward] == ["p:a", "p:m", "p:q"]
    assert [s["chunk_id"] for s in forward] == [s["chunk_id"] for s in reverse]


def test_mention_fallback_is_deterministic_under_equal_scores(monkeypatch):
    """Filename-mention fallback dedups by doc, keeping the first chunk seen.
    With equal scores the survivor must be chunk_id-stable, not arrival-order."""
    def doc_lookup(did):
        return {"original_name": "Spec.pdf"}
    monkeypatch.setattr("app.core.projects.get_document", doc_lookup)
    base = [
        {"doc_id": "d1", "chunk_index": 9, "chunk_id": "p:z", "score": 0.7},
        {"doc_id": "d1", "chunk_index": 1, "chunk_id": "p:a", "score": 0.7},
    ]
    text = "The detail comes from Spec.pdf."
    forward = _build_sources_from_audit({"chunks": list(base)}, text)
    reverse = _build_sources_from_audit({"chunks": list(reversed(base))}, text)
    assert len(forward) == 1
    assert forward[0]["chunk_id"] == "p:a"  # lowest chunk_id wins the tie
    assert forward[0]["chunk_id"] == reverse[0]["chunk_id"]


def test_sources_match_src_marker_to_injected_chunk(monkeypatch):
    _patch_doc_names(monkeypatch, {"d23": DD23, "d22": DD22})
    audit = {
        "project_id": "proj_a",
        "chunks": [
            {"doc_id": "d22", "chunk_index": 4, "score": 0.9, "layer": "own"},
            {"doc_id": "d23", "chunk_index": 12, "score": 0.7, "layer": "own"},
        ],
    }
    text = (
        "Time for Completion is 365 days "
        "(src=AB-2023-101_the client project II Infrastructure Package 1_"
        "Vol 1 - Conditions of Contract.pdf)."
    )
    out = _build_sources_from_audit(audit, text)
    assert len(out) == 1, out
    assert out[0]["doc_id"] == "d23"
    assert (out[0]["chunk_index"], out[0]["page_or_section"]) == (12, "")
