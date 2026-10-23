"""Stage 2 / classify() — map a doc to (knowledge_layer, authority) at ingest.

Deterministic, filename + project-based. GK project -> shared_domain (or
company_rules for procedures/templates); any other project -> project_record;
user uploads -> user_session (Stage 4 wiring). Authority from doc-type keywords,
falling back to a layer-appropriate default.
"""
import pytest
from app.core.rag import layers as L


@pytest.fixture(autouse=True)
def _gk(monkeypatch):
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb")


def test_gk_reference_is_shared_domain():
    assert L.classify("curated_kb", "cesmm4_classification_guide.md") == (
        "shared_domain", "policy")


def test_gk_contract_reference_is_contractual():
    layer, auth = L.classify("curated_kb", "fidic_2017_administration.md")
    assert layer == "shared_domain"
    assert auth == "contractual"


def test_gk_procedure_is_company_rules():
    assert L.classify("curated_kb", "design_review_procedure.md") == (
        "company_rules", "policy")


def test_project_priced_boq_is_project_record_commercial():
    assert L.classify("example_infra_pack", "priced_boq_sewer.xlsx") == (
        "project_record", "commercial")


def test_project_contract_is_contractual():
    assert L.classify("example_infra_pack", "Contract_Conditions_Part2.pdf") == (
        "project_record", "contractual")


def test_project_drawing_is_design():
    assert L.classify("example_infra_pack", "GA-plan-L02.dwg") == (
        "project_record", "design")


def test_project_daily_report_is_operational():
    assert L.classify("example_infra_pack", "daily_site_report_2026-07-01.pdf") == (
        "project_record", "operational")


def test_project_unknown_defaults_operational():
    assert L.classify("example_infra_pack", "misc_file.bin") == (
        "project_record", "operational")


def test_user_upload_is_user_session():
    layer, auth = L.classify("some_project", "my_notes.pdf", is_user_upload=True)
    assert layer == "user_session"


def test_empty_docname_is_safe():
    layer, auth = L.classify("example_infra_pack", "")
    assert layer == "project_record"
    assert auth in L.AUTHORITIES


@pytest.mark.parametrize("name", [
    "XYZ-101_Change Control.pdf",
    "qrs_330 HSE Audit.docx",
    "ABC 204 - Site Logistics Form.xlsx",
])
def test_a_controlled_company_document_is_company_rules_by_its_code_shape(name):
    """No organisation's code prefix is listed: any leading controlled-document
    code followed by a title marks a company procedure/template/form."""
    assert L.classify("curated_kb", name) == ("company_rules", "policy")


def test_a_contract_id_is_not_a_controlled_document_code():
    layer, _auth = L.classify("curated_kb", "AB-2001-101_Vol 1.pdf")
    assert layer == "shared_domain"
