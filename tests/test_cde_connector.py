"""P0 Aconex / CDE connector — interface, fake adapter, ingest, draft-and-post.

Live Oracle credentials are not required. Tests pin the fake adapter and
assert the real client fails closed when unconfigured.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.cde import (
    CdeNotConfiguredError,
    FakeCdeClient,
    connector_mode,
    get_cde_client,
    ingest_cde_document,
    post_rfi_draft,
    reset_cde_clients,
)
from app.core.cde.aconex import AconexCdeClient
from app.core.cde.types import CdeMailDraft
from app.main import app
from tests._dev_key import DEV_KEY
from tests._fixture_values import ENV_SECRET_03, ENV_SECRET_04

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture
def fake_cde(monkeypatch):
    monkeypatch.setenv("CDE_ADAPTER", "fake")
    monkeypatch.delenv("ACONEX_ENABLED", raising=False)
    reset_cde_clients()
    client = get_cde_client()
    assert isinstance(client, FakeCdeClient)
    yield client
    reset_cde_clients()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _new_project(client: TestClient, name: str = "CDE Corpus") -> dict:
    r = client.post("/v1/projects", json={"name": name, "client": "ACME"}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


# ── interface + fake adapter ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fake_lists_mail_documents_rfis_transmittals(fake_cde):
    mail = await fake_cde.list_mail("p1")
    docs = await fake_cde.list_documents("p1")
    rfis = await fake_cde.list_rfis("p1")
    transmittals = await fake_cde.list_transmittals("p1")
    assert mail and docs and rfis and transmittals
    assert any(m.mail_type == "RFI" for m in rfis)
    assert any("transmittal" in m.mail_type.lower() for m in transmittals)
    one = await fake_cde.get_mail("p1", rfis[0].id)
    assert one.id == rfis[0].id
    meta = await fake_cde.get_document("p1", docs[0].id)
    assert meta.filename.endswith(".txt")


@pytest.mark.asyncio
async def test_fake_post_rfi_is_recorded_not_a_fork_number(fake_cde):
    result = await fake_cde.post_rfi(
        "p1",
        CdeMailDraft(
            subject="Waterproofing upstand",
            body="Confirm 200mm upstand at wet areas.",
            mail_type="RFI",
        ),
    )
    assert result.vendor == "fake"
    assert result.status == "posted"
    assert result.cde_id.startswith("posted-")
    assert result.as_dict()["source_of_truth"] == "cde"
    assert fake_cde.posted[-1].cde_id == result.cde_id
    # Fork must not treat a local DRAFT-RFI as the register number.
    assert "DRAFT-RFI" not in result.reference
    assert not result.reference.startswith("RFI-000")


@pytest.mark.asyncio
async def test_unconfigured_client_fails_closed(monkeypatch):
    monkeypatch.delenv("CDE_ADAPTER", raising=False)
    monkeypatch.delenv("ACONEX_ENABLED", raising=False)
    monkeypatch.delenv("ACONEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ACONEX_CLIENT_ID", raising=False)
    reset_cde_clients()
    client = get_cde_client()
    with pytest.raises(CdeNotConfiguredError, match="not configured"):
        await client.list_documents("x")


def test_aconex_constructor_fails_closed_when_flag_off(monkeypatch):
    monkeypatch.setenv("ACONEX_ENABLED", "false")
    monkeypatch.setenv("ACONEX_ACCESS_TOKEN", ENV_SECRET_03)
    with pytest.raises(CdeNotConfiguredError):
        AconexCdeClient(access_token="tok")


@pytest.mark.asyncio
async def test_aconex_uses_documented_rest_paths(monkeypatch):
    monkeypatch.setenv("ACONEX_ENABLED", "true")
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        path = request.url.path
        if path.endswith("/mail") and request.method == "GET":
            return httpx.Response(
                200,
                text=(
                    "<MailList><Mail><MailId>m1</MailId>"
                    "<Subject>Query</Subject><MailType>RFI</MailType></Mail></MailList>"
                ),
            )
        if "/mail/" in path and request.method == "GET":
            return httpx.Response(
                200,
                text="<Mail><MailId>m1</MailId><Subject>Query</Subject></Mail>",
            )
        if path.endswith("/mail/") and request.method == "POST":
            assert "multipart/mixed" in request.headers.get("content-type", "")
            return httpx.Response(200, text="<Mail><MailId>new-9</MailId><MailNo>RFI-9</MailNo></Mail>")
        if path.endswith("/register") and request.method == "GET":
            return httpx.Response(
                200,
                text=(
                    "<DocumentSearch><Document><DocumentId>d1</DocumentId>"
                    "<title>Spec</title><filename>spec.txt</filename></Document>"
                    "</DocumentSearch>"
                ),
            )
        if path.endswith("/metadata"):
            return httpx.Response(
                200,
                text="<Document><DocumentId>d1</DocumentId><filename>spec.txt</filename></Document>",
            )
        if "/register/" in path and request.method == "GET":
            return httpx.Response(200, content=b"spec body")
        return httpx.Response(404, text="unexpected " + path)

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport, base_url="https://api.aconex.com"
    )
    acx = AconexCdeClient(access_token="tok", api_base="https://api.aconex.com", http=http)

    mail = await acx.list_mail("1879048400")
    assert mail[0].id == "m1"
    got = await acx.get_mail("1879048400", "m1")
    assert got.id == "m1"
    docs = await acx.list_documents("1879048400")
    assert docs[0].id == "d1"
    meta = await acx.get_document("1879048400", "d1")
    assert meta.filename == "spec.txt"
    blob = await acx.download_document("1879048400", "d1")
    assert blob == b"spec body"
    posted = await acx.post_rfi(
        "1879048400",
        CdeMailDraft(subject="Upstand", body="Confirm height", mail_type="RFI"),
    )
    assert posted.cde_id == "new-9"
    assert posted.reference == "RFI-9"
    assert posted.vendor == "aconex"

    paths = {p for _, p in seen}
    assert "/api/projects/1879048400/mail" in paths
    assert "/api/projects/1879048400/mail/m1" in paths
    assert "/api/projects/1879048400/mail/" in paths
    assert "/api/projects/1879048400/register" in paths
    assert "/api/projects/1879048400/register/d1/metadata" in paths
    assert "/api/projects/1879048400/register/d1" in paths
    await http.aclose()


# ── ingest into existing corpus ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_fake_cde_document_into_project_corpus(fake_cde, client, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("INDEX_ON_UPLOAD", "false")
    proj = _new_project(client)
    docs = await fake_cde.list_documents("cde-proj")
    result = await ingest_cde_document(
        proj["id"],
        docs[0],
        cde_project_id="cde-proj",
        client=fake_cde,
        eager_index=False,
    )
    assert result["status"] == "stored"
    document = result["document"]
    meta = document["metadata"]
    assert meta["source"] == "cde"
    assert meta["origin"] == "cde_cache"
    assert meta["cde_vendor"] == "fake"
    assert meta["cde_document_id"] == docs[0].id
    assert meta["system_of_record"] == "cde"
    detail = client.get(f"/v1/projects/{proj['id']}", headers=H).json()
    assert any(d["original_name"] == docs[0].filename for d in detail["documents"])


# ── draft-and-post through the interface ─────────────────────────────────────


@pytest.mark.asyncio
async def test_draft_and_post_rfi_records_on_fake_adapter(fake_cde):
    posted = await post_rfi_draft(
        "cde-proj",
        {
            "subject": "Clarification on waterproofing upstand",
            "question": "Confirm 200mm upstand at wet areas.",
            "rfi_number": "DRAFT-RFI",
        },
        client=fake_cde,
    )
    assert fake_cde.posted, "fake adapter must record the post"
    assert posted.cde_id == fake_cde.posted[-1].cde_id
    assert posted.raw.get("subject") == "Clarification on waterproofing upstand"
    assert posted.as_dict()["source_of_truth"] == "cde"


@pytest.mark.asyncio
async def test_construction_action_posts_through_interface(fake_cde):
    pytest.importorskip("app.containers.construction")
    from tests.conftest import is_construction_kit_enabled

    if not is_construction_kit_enabled():
        pytest.skip("requires CEREBRUM_DOMAIN_KITS=construction")
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    result = await container.cde_post_rfi(
        {
            "cde_project_id": "cde-proj",
            "subject": "Hold point",
            "body": "Confirm inspection at grid A.",
        },
        {},
    )
    assert result["status"] == "success"
    assert result["source_of_truth"] == "cde"
    assert fake_cde.posted


@pytest.mark.asyncio
async def test_orchestrator_routes_post_rfi_to_cde():
    from tests.conftest import is_construction_kit_enabled

    if not is_construction_kit_enabled():
        pytest.skip("requires CEREBRUM_DOMAIN_KITS=construction")
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    block = SmartOrchestratorBlock()
    result = await block.process({"user_message": "post this rfi to aconex"})
    queue = result.get("action_queue") or []
    assert queue[0] == "cde_post_rfi"


# ── connector endpoints stay honest ──────────────────────────────────────────


def test_connectors_mode_not_configured_by_default(client):
    proj = _new_project(client, "Flag Off")
    r = client.get(f"/v1/projects/{proj['id']}/connectors", headers=H)
    assert r.status_code == 200, r.text
    conn = r.json()["connectors"][0]
    assert conn["name"] == "aconex"
    assert conn["connected"] is False
    assert conn["mode"] == "not_configured"


def test_connectors_mode_flag_when_connected_without_oauth(client, monkeypatch):
    monkeypatch.delenv("ACONEX_ENABLED", raising=False)
    monkeypatch.delenv("ACONEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CDE_ADAPTER", raising=False)
    proj = _new_project(client, "Flag Only")
    posted = client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex",
        json={"connected": True},
        headers=H,
    )
    assert posted.status_code == 200, posted.text
    body = posted.json()
    assert body["aconex_connected"] is True
    assert body["mode"] == "flag"
    listed = client.get(f"/v1/projects/{proj['id']}/connectors", headers=H).json()
    conn = listed["connectors"][0]
    assert conn["connected"] is True
    assert conn["mode"] == "flag"
    assert "OAuth" in conn["note"] or "oauth" in conn["note"].lower() or "credentials" in conn["note"].lower()


def test_connectors_mode_oauth_when_enabled_and_flagged(client, monkeypatch):
    monkeypatch.setenv("ACONEX_ENABLED", "true")
    monkeypatch.setenv("ACONEX_ACCESS_TOKEN", ENV_SECRET_04)
    proj = _new_project(client, "OAuth Mode")
    client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex",
        json={"connected": True},
        headers=H,
    )
    listed = client.get(f"/v1/projects/{proj['id']}/connectors", headers=H).json()
    conn = listed["connectors"][0]
    assert conn["mode"] == "oauth"
    assert conn["connected"] is True


@pytest.mark.usefixtures("dev_key_is_admin")
def test_sync_and_post_fail_closed_when_not_configured(client, monkeypatch):
    monkeypatch.delenv("CDE_ADAPTER", raising=False)
    monkeypatch.delenv("ACONEX_ENABLED", raising=False)
    monkeypatch.delenv("ACONEX_ACCESS_TOKEN", raising=False)
    reset_cde_clients()
    proj = _new_project(client, "No Client")
    sync = client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex/sync",
        json={"cde_project_id": "1879"},
        headers=H,
    )
    assert sync.status_code == 409
    assert "not configured" in sync.json()["detail"].lower()
    post = client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex/rfi",
        json={
            "cde_project_id": "1879",
            "subject": "Query",
            "body": "Please confirm.",
        },
        headers=H,
    )
    assert post.status_code == 409


@pytest.mark.usefixtures("dev_key_is_admin")
def test_sync_fake_document_and_post_rfi_via_http(client, fake_cde, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("INDEX_ON_UPLOAD", "false")
    monkeypatch.setenv("ACONEX_PROJECT_ID", "cde-proj")
    proj = _new_project(client, "HTTP Sync")
    sync = client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex/sync",
        json={"cde_project_id": "cde-proj"},
        headers=H,
    )
    assert sync.status_code == 200, sync.text
    body = sync.json()
    assert body["stored"] >= 1
    assert body["source_of_truth"] == "cde"
    detail = client.get(f"/v1/projects/{proj['id']}", headers=H).json()
    names = [d["original_name"] for d in detail["documents"]]
    assert "cde_waterproofing_spec.txt" in names
    meta = next(
        d["metadata"]
        for d in detail["documents"]
        if d["original_name"] == "cde_waterproofing_spec.txt"
    )
    assert meta["origin"] == "cde_cache"
    assert meta["cde_vendor"] == "fake"

    posted = client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex/rfi",
        json={
            "cde_project_id": "cde-proj",
            "subject": "Upstand height",
            "question": "Confirm 200mm upstand.",
        },
        headers=H,
    )
    assert posted.status_code == 200, posted.text
    payload = posted.json()
    assert payload["status"] == "posted"
    assert payload["source_of_truth"] == "cde"
    assert fake_cde.posted
    assert payload["post"]["cde_id"] == fake_cde.posted[-1].cde_id


def test_connector_mode_helper_is_honest():
    assert connector_mode(False) == "not_configured"


@pytest.mark.asyncio
async def test_default_adapter_is_not_a_local_register(monkeypatch):
    monkeypatch.delenv("CDE_ADAPTER", raising=False)
    monkeypatch.setenv("ACONEX_ENABLED", "")
    reset_cde_clients()
    client = get_cde_client()
    assert client.vendor == "none"
    with pytest.raises(CdeNotConfiguredError):
        await client.post_rfi("x", CdeMailDraft(subject="s", body="b"))
