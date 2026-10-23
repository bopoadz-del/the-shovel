"""Price an UNPRICED BOQ from the typed rate-card.

Two layers:
  1. ``boq_pricing.price_line_items`` -- pure pricing: an exact (cat+unit) hit,
     a category-median fallback, and a NO-RATE line (no invented rate).
  2. ``POST /v1/projects/{id}/price-boq`` -- extract an unpriced xlsx BOQ via
     boq_processor, price it, generate the formula-linked workbook, and persist
     + eager-index it. Plus the unknown-asset_type 400.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import projects as store
from app.lib import boq_pricing
from app.lib.boq_excel import evaluate_workbook_total
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}
_XLSX_MEDIA = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── unit: price_line_items ───────────────────────────────────────────────────
def test_price_line_items_exact_fallback_and_no_rate():
    """Exact cat+unit hit, category-median fallback, and a flagged NO-RATE line
    -- and NO rate is invented for the NO-RATE line."""
    # Buildings/Towers AED has Concrete at units m3 (exact) but not at unit
    # "no" (forces category-median fallback across Concrete's other units).
    items = [
        {"description": "Reinforced concrete C40 to columns", "quantity": 100, "unit": "m3"},
        {"description": "Concrete sundry item", "quantity": 5, "unit": "no"},
    ]
    priced, summary = boq_pricing.price_line_items(items, "Buildings/Towers", "AED")

    assert priced[0]["work_category"] == "Concrete"
    assert priced[0]["rate_source"] == "exact (cat+unit)"
    assert priced[0]["rate"] > 0

    assert priced[1]["work_category"] == "Concrete"
    # Fallback tiers are unit-commensurable since F10: the Concrete category
    # holds only m/m2/m3/t rates, so a per-number sundry line skips the
    # category tier (a per-m3 median has the wrong dimensions for a count)
    # and prices from the asset-wide count-class median instead.
    assert priced[1]["rate_source"] == "weak (asset median)"
    assert priced[1]["rate"] > 0

    assert summary["exact"] == 1
    assert summary["fallback"] == 1
    assert summary["no_rate"] == 0
    assert summary["grand_total"] > 0

    # NO RATE: a valid asset with a currency it does NOT carry -> empty lookup
    # -> every line flagged, and NO invented rate.
    no_items = [{"description": "Reinforced concrete C40", "quantity": 100, "unit": "m3"}]
    n_priced, n_summary = boq_pricing.price_line_items(no_items, "Buildings/Towers", "USD")
    assert n_priced[0]["rate_source"] == "NO RATE"
    assert n_priced[0]["rate"] == 0
    assert n_summary["no_rate"] == 1
    assert n_summary["grand_total"] == 0


def test_available_assets_reports_currencies():
    assets = boq_pricing.available_assets()
    assert "AED" in assets["Buildings/Towers"]
    assert "Infrastructure" in assets


# ── endpoint helpers ─────────────────────────────────────────────────────────
def _new_project(client, name="Price BOQ Project"):
    r = client.post("/v1/projects", json={"name": name, "client": "ACME"}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


def _unpriced_boq_xlsx() -> bytes:
    """A small UNPRICED BOQ: Description/Quantity/Unit filled, Rate/Amount blank."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["Description", "Quantity", "Unit", "Rate", "Amount"])
    rows = [
        ("Reinforced concrete C40 to raft foundation", 250, "m3", None, None),
        ("Blockwork masonry to internal walls", 1200, "m2", None, None),
        ("Ceramic floor tiling to finishes", 800, "m2", None, None),
        ("Gypsum plaster to ceilings", 950, "m2", None, None),
    ]
    for row in rows:
        ws.append(list(row))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _upload(client, pid, content: bytes, filename="unpriced_boq.xlsx"):
    files = {"file": (filename, content, _XLSX_MEDIA)}
    r = client.post(f"/v1/projects/{pid}/documents", files=files, headers=H)
    assert r.status_code == 201, r.text
    return r.json()["document"]


# ── endpoint: happy path ─────────────────────────────────────────────────────
@pytest.mark.usefixtures("dev_key_is_admin")
def test_price_boq_endpoint_prices_persists_and_indexes(client, monkeypatch):
    proj = _new_project(client)
    pid = proj["id"]
    doc = _upload(client, pid, _unpriced_boq_xlsx())

    before = {d["id"] for d in store.list_documents(pid)}

    scheduled: list[tuple[str, str]] = []
    import app.routers.exports as exports
    monkeypatch.setattr(
        exports.doc_index, "maybe_eager_index",
        lambda project_id, document_id: scheduled.append((project_id, document_id)),
    )

    r = client.post(
        f"/v1/projects/{pid}/price-boq",
        json={"document_id": doc["id"], "asset_type": "Buildings/Towers", "currency": "AED"},
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == _XLSX_MEDIA
    assert r.content[:2] == b"PK"  # xlsx is a zip

    # The generated workbook's LIVE formulas evaluate to a non-zero total.
    total = evaluate_workbook_total(io.BytesIO(r.content))
    assert total > 0, "priced BOQ should have non-zero amounts"

    # A new document landed in the project, and eager indexing was scheduled.
    after = store.list_documents(pid)
    new_docs = [d for d in after if d["id"] not in before]
    assert len(new_docs) == 1, new_docs
    assert "ESTIMATED from rate-card" in new_docs[0]["original_name"]
    assert scheduled == [(pid, new_docs[0]["id"])]


def test_price_boq_endpoint_unknown_asset_type_400(client):
    proj = _new_project(client, "Bad Asset Project")
    pid = proj["id"]
    doc = _upload(client, pid, _unpriced_boq_xlsx())

    r = client.post(
        f"/v1/projects/{pid}/price-boq",
        json={"document_id": doc["id"], "asset_type": "Spaceport"},
        headers=H,
    )
    assert r.status_code == 400, r.text
    assert "Buildings/Towers" in r.text  # reports the valid options
