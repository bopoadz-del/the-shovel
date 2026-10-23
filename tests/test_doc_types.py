"""Tests for the document-type registry — Roadmap V2 · Epic 2."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import doc_types
from app.core import jwt_auth
from app.core import users as users_store
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


def _admin_headers():
    """Mutations on the global registry are admin-only; the dev key is not."""
    users_store.ensure_user_exists(
        "doc-types-admin", role="admin", email="doc-types-admin@local"
    )
    return {"Authorization": f"Bearer {jwt_auth.create_token('doc-types-admin')}"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    doc_types.remove_type("method_statement")  # drop any test-created type


# ── built-in classification ─────────────────────────────────────────────────

def test_builtins_are_loaded():
    names = {t["name"] for t in doc_types.list_types()}
    assert {"drawing", "schedule", "boq", "contract"} <= names


def test_classify_by_filename():
    assert doc_types.classify("Tower_Contract_FIDIC.pdf")["name"] == "contract"
    assert doc_types.classify("Baseline_Programme_P6.xer")["name"] == "schedule"


def test_classify_is_content_aware():
    # filename gives no hint — content keywords decide it
    r = doc_types.classify("scan001.pdf", "This Bill of Quantities lists unit rate items")
    assert r["name"] == "boq"
    assert any(m.startswith("content:") for m in r["matched_on"])


def test_unmatched_document_is_unrecognised_not_guessed():
    r = doc_types.classify("mystery.pdf", "lorem ipsum dolor sit amet")
    assert r["name"] == "unrecognised"
    assert r["needs_user_confirmation"] is True


# ── custom types (no code change) ───────────────────────────────────────────

def test_add_custom_type_and_classify_with_it():
    doc_types.add_type({
        "name": "method_statement",
        "match": {"filename": ["method statement", "rams"], "content": ["safe system of work"]},
        "expected_fields": ["activity"],
    })
    names = {t["name"] for t in doc_types.list_types()}
    assert "method_statement" in names
    assert doc_types.classify("Excavation Method Statement.pdf")["name"] == "method_statement"


# ── API ─────────────────────────────────────────────────────────────────────

def test_document_types_api(client):
    r = client.get("/v1/document-types", headers=H)
    assert r.status_code == 200
    assert len(r.json()["document_types"]) >= 8

    admin = _admin_headers()
    r = client.post("/v1/document-types", headers=admin, json={
        "name": "method_statement",
        "match": {"filename": ["method statement"], "extensions": [], "content": []},
        "expected_fields": [],
    })
    assert r.status_code == 201
    assert r.json()["source"] == "custom"

    r = client.post("/v1/document-types/classify", headers=H,
                     json={"filename": "Piling Method Statement.docx"})
    assert r.json()["name"] == "method_statement"

    r = client.delete("/v1/document-types/method_statement", headers=admin)
    assert r.status_code == 200
