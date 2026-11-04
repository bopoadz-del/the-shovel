"""S3: operand resolution order -- user input, project document (cited), then
an honest ask. Never a built-in default (K-T5). Pure helpers, not yet wired
into the calculator; this pins the contract. Dotted import so CI-M selects it.
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas_quantities as q


# ── bulking_factor_from_text: the number must belong to the bulking word ─────
@pytest.mark.parametrize("text,expected", [
    ("Bulking factor for excavated material: 1.25", 1.25),
    ("Allow 25% swell on the arisings", 1.25),
    ("Retention 5%, advance recovery 15%", None),   # no bulking/swell word
    ("Excavate the raft 30 x 20 x 1.5 m", None),     # a dim is not a factor
])
def test_bulking_factor_from_text(text, expected):
    assert q.bulking_factor_from_text(text) == expected


# ── resolve_operand: user input wins, with user_input provenance ─────────────
@pytest.mark.parametrize("supplied,expected", [
    ({"waste_factor": 0.07}, (0.07, "user_input")),
    ({"waste_factor": ""}, (0.05, "project_document")),   # blank -> fall through to the doc
])
def test_resolve_operand_user_input_then_document(supplied, expected):
    docs = [{"document": "BOQ preamble", "text": "documented waste 5% on concrete", "chunk": 2}]
    assert q.resolve_operand("waste_factor", supplied, docs) == expected


# ── resolve_operand: a documented value is cited, never defaulted ────────────
@pytest.mark.parametrize("name,docs,expected_val", [
    ("waste_factor", [{"document": "Spec 02300", "text": "documented waste 5%"}], 0.05),
    ("bulking_factor", [{"document": "Method statement", "text": "bulking factor 1.25"}], 1.25),
])
def test_resolve_operand_from_project_document_is_cited(name, docs, expected_val):
    value, prov = q.resolve_operand(name, {}, docs)
    assert value == expected_val and prov == "project_document"
    hit = q.lookup_in_project_documents(name, docs)
    assert hit["document"] == docs[0]["document"]   # the citation names the doc


# ── resolve_operand: neither -> awaiting_inputs (None, None), never a default ─
@pytest.mark.parametrize("docs", [
    [],
    [{"document": "Drawing register", "text": "no waste or bulking stated here"}],
])
def test_resolve_operand_unresolved_is_none_never_default(docs):
    assert q.resolve_operand("waste_factor", {}, docs) == (None, None)


# ── wired path: a documented-waste raft ask resolves from project docs ───────
import app.lib.construction_formulas as cf  # noqa: E402 -- dotted, for CI-M


@pytest.mark.parametrize("docs,expect_waste,expect_cite", [
    ([{"document": "BOQ preamble", "text": "documented waste 5% on concrete"}], 0.05, "BOQ preamble"),
    ([{"document": "Spec 02300", "text": "waste factor 7% for in-situ concrete"}], 0.07, "Spec 02300"),
])
def test_resolve_concrete_volume_calc_cites_documented_waste(docs, expect_waste, expect_cite):
    token = cf.set_project_docs(docs)
    try:
        _name, out = q.resolve_concrete_volume_calc(
            "concrete_volume",
            {"length_m": 30, "width_m": 20, "thickness_m": 1.5,
             "text": "raft 30 x 20 x 1.5 m including documented waste"},
        )
        assert out.get("waste_factor") == expect_waste
        assert out.get("waste_factor_citation") == expect_cite
    finally:
        cf._PROJECT_DOCS.reset(token)


@pytest.mark.parametrize("chunks,fallback,expected", [
    ([{"document": "Spec 02300", "text": "waste 5%", "chunk_index": 3}], None,
     [{"document": "Spec 02300", "text": "waste 5%", "chunk": 3}]),
    ([{"document": "X"}], "retrieved blob stating waste 7%",
     [{"document": "retrieved project context", "text": "retrieved blob stating waste 7%", "chunk": None}]),
])
def test_project_docs_from_chunks(chunks, fallback, expected):
    assert cf.project_docs_from_chunks(chunks, fallback) == expected


@pytest.mark.parametrize("docs", [
    [],
    [{"document": "Drawing register", "text": "no waste stated"}],
])
def test_documented_waste_ask_with_no_document_stays_awaiting(docs):
    token = cf.set_project_docs(docs)
    try:
        _name, out = q.resolve_concrete_volume_calc(
            "concrete_volume",
            {"length_m": 30, "width_m": 20, "thickness_m": 1.5,
             "text": "raft 30 x 20 x 1.5 m including documented waste"},
        )
        assert "waste_factor" not in out   # required input, never invented
    finally:
        cf._PROJECT_DOCS.reset(token)
