"""CM overlay on CDE events — poll / subscribe / inject via FakeCdeClient.

Live Oracle credentials are not required. Unconfigured clients fail closed.
inject() cites live CDE rows or stays silent; it never invents an RFI/claim number.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.cde import (
    CdeEvent,
    CdePayloadError,
    CdeEventKind,
    CdeNotConfiguredError,
    FakeCdeClient,
    classify_cde_event,
    event_poll_enabled,
    get_cde_client,
    inject,
    poll_cde_events,
    process_cde_events,
    reset_cde_clients,
    run_cm_overlay,
)
from app.core.cde.events import event_from_payload
from app.core.cde.poll import run_poll_once
from app.core.delay_advice import DelayKind, classify_delay
from app.main import app
from tests._dev_key import DEV_KEY

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


def _new_project(client: TestClient, name: str = "CDE CM") -> dict:
    r = client.post("/v1/projects", json={"name": name, "client": "ACME"}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


def _hold_point() -> CdeEvent:
    return CdeEvent(
        id="mail-100",
        source="mail",
        subject="Site query — hold point",
        body="Please confirm hold-point inspection for grid A.",
        mail_type="General",
        reference="MAIL-100",
    )


def _slip_notice() -> CdeEvent:
    return CdeEvent(
        id="mail-400",
        source="mail",
        subject="Notice — chiller delivery has slipped four weeks",
        body="Supplier advised the chiller delivery has slipped four weeks.",
        mail_type="General",
        reference="MAIL-400",
    )


# ── classify + overlay reuse existing helpers ─────────────────────────────


def test_rfi_row_is_rfi_from_cde_mail_type():
    event = CdeEvent(
        id="rfi-200",
        source="mail",
        subject="RFI — waterproofing upstand height",
        mail_type="RFI",
        reference="RFI-200",
    )
    assert classify_cde_event(event) == CdeEventKind.RFI


def test_clash_mail_reuses_message_wants_clash():
    event = CdeEvent(
        id="mail-500",
        source="mail",
        subject="Clash — duct vs beam at grid B2",
        body="Clash detected between supply duct and structural beam.",
        reference="MAIL-500",
    )
    assert classify_cde_event(event) == CdeEventKind.CLASH


def test_slip_notice_reuses_classify_delay_ask():
    event = _slip_notice()
    assert classify_cde_event(event) == CdeEventKind.DELAY
    assert classify_delay(event.text()) == DelayKind.ASK


def test_hold_point_is_not_a_cm_event():
    assert classify_cde_event(_hold_point()) == CdeEventKind.NONE
    assert run_cm_overlay(_hold_point()) is None


def test_overlay_on_slip_does_not_publish_a_claim():
    overlay = run_cm_overlay(_slip_notice())
    assert overlay is not None
    assert overlay["kind"] == "delay"
    assert overlay["delay_kind"] == DelayKind.ASK.value
    assert overlay["matched_template"] != "delay_to_claim"
    assert "do not assume" in overlay["delay_advice"].lower()
    assert overlay["cited"]["cde_id"] == "mail-400"
    assert overlay["cited"]["reference"] == "MAIL-400"


# ── inject cites live rows or stays silent ────────────────────────────────


def test_inject_silent_without_relevant_rows():
    assert inject([]) == ""
    assert inject([_hold_point()]) == ""


def test_inject_cites_live_cde_id_and_reference():
    text = inject([_slip_notice()])
    assert "[CDE-grounded CM]" in text
    assert "cde_id=mail-400" in text
    assert "reference=MAIL-400" in text
    assert "RFI-000" not in text
    assert "DRAFT-RFI" not in text
    assert "CLAIM-" not in text


def test_inject_cites_cde_id_only_when_reference_absent():
    event = CdeEvent(
        id="clash-live-9",
        source="mail",
        subject="Clash — pipe vs hanger on L02",
        body="Clash detected at hanger C-12.",
    )
    text = inject([event])
    assert "cde_id=clash-live-9" in text
    assert "reference=" not in text
    assert "RFI-" not in text


def test_inject_skips_row_without_cde_id():
    orphan = CdeEvent(
        id="",
        source="mail",
        subject="RFI — invented locally",
        mail_type="RFI",
        reference="RFI-0001",
    )
    assert inject([orphan]) == ""


def test_subscribe_payload_without_id_is_rejected():
    with pytest.raises(CdePayloadError, match="missing id"):
        event_from_payload({"subject": "RFI about delay", "rfi_number": "RFI-0001"})


# ── poll through FakeCdeClient ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_lists_mail_and_register(fake_cde):
    events = await poll_cde_events("cde-proj", client=fake_cde)
    ids = {e.id for e in events}
    assert "rfi-200" in ids
    assert "mail-400" in ids
    assert "mail-500" in ids
    assert "cde-doc-1" in ids
    assert "cde-doc-clash" in ids


@pytest.mark.asyncio
async def test_process_poll_runs_overlay_and_inject(fake_cde):
    result = await process_cde_events("cde-proj", client=fake_cde)
    assert result["status"] == "ok"
    assert result["source_of_truth"] == "cde"
    assert result["vendor"] == "fake"
    assert result["relevant"] >= 3
    kinds = {row["kind"] for row in result["overlays"]}
    assert {"rfi", "delay", "clash"} <= kinds
    inject_text = result["inject"]
    assert "cde_id=rfi-200" in inject_text
    assert "reference=RFI-200" in inject_text
    assert "cde_id=mail-400" in inject_text
    assert "cde_id=mail-500" in inject_text
    assert "RFI-000" not in inject_text
    cited_ids = {c["cde_id"] for c in result["cited"]}
    assert "rfi-200" in cited_ids
    assert result["ingest"] is None


@pytest.mark.asyncio
async def test_subscribe_accepts_live_row_and_does_not_number_it(fake_cde):
    result = await process_cde_events(
        "cde-proj",
        client=fake_cde,
        payloads=[
            {
                "id": "aconex-mail-77",
                "subject": "RFI — hold-point release date",
                "mail_type": "RFI",
                "reference": "RFI-77",
                "rfi_number": "DRAFT-RFI",
            }
        ],
    )
    assert result["relevant"] == 1
    assert result["overlays"][0]["cde_id"] == "aconex-mail-77"
    assert result["overlays"][0]["reference"] == "RFI-77"
    assert "DRAFT-RFI" not in result["inject"]
    assert "aconex-mail-77" in result["inject"]


@pytest.mark.asyncio
async def test_poll_fails_closed_when_not_configured(monkeypatch):
    monkeypatch.delenv("CDE_ADAPTER", raising=False)
    monkeypatch.delenv("ACONEX_ENABLED", raising=False)
    monkeypatch.delenv("ACONEX_ACCESS_TOKEN", raising=False)
    reset_cde_clients()
    with pytest.raises(CdeNotConfiguredError):
        await poll_cde_events("x")


@pytest.mark.asyncio
async def test_subscribe_fails_closed_when_not_configured(monkeypatch):
    monkeypatch.delenv("CDE_ADAPTER", raising=False)
    monkeypatch.delenv("ACONEX_ENABLED", raising=False)
    monkeypatch.delenv("ACONEX_ACCESS_TOKEN", raising=False)
    reset_cde_clients()
    with pytest.raises(CdeNotConfiguredError):
        await process_cde_events(
            "x",
            payloads=[{"id": "m1", "subject": "RFI", "mail_type": "RFI"}],
        )


def test_event_poll_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("CDE_EVENT_POLL_ENABLED", raising=False)
    assert event_poll_enabled() is False


@pytest.mark.asyncio
async def test_background_poll_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("CDE_EVENT_POLL_ENABLED", raising=False)
    assert await run_poll_once("cde-proj") is None


# ── HTTP + construction + orchestrator ────────────────────────────────────


def test_http_poll_events_with_fake(client, fake_cde, monkeypatch):
    monkeypatch.setenv("ACONEX_PROJECT_ID", "cde-proj")
    proj = _new_project(client, "HTTP Events")
    r = client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex/events",
        json={"cde_project_id": "cde-proj"},
        headers=H,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_of_truth"] == "cde"
    assert body["relevant"] >= 3
    assert "cde_id=rfi-200" in body["inject"]
    assert "RFI-000" not in body["inject"]


def test_http_subscribe_without_id_is_rejected(client, fake_cde, monkeypatch):
    monkeypatch.setenv("ACONEX_PROJECT_ID", "cde-proj")
    proj = _new_project(client, "No Id")
    r = client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex/events",
        json={"events": [{"subject": "RFI", "rfi_number": "RFI-0001"}]},
        headers=H,
    )
    assert r.status_code == 400
    assert "invent" in r.json()["detail"].lower()


def test_http_events_fail_closed_when_not_configured(client, monkeypatch):
    monkeypatch.delenv("CDE_ADAPTER", raising=False)
    monkeypatch.delenv("ACONEX_ENABLED", raising=False)
    monkeypatch.delenv("ACONEX_ACCESS_TOKEN", raising=False)
    reset_cde_clients()
    proj = _new_project(client, "No Client Events")
    r = client.post(
        f"/v1/projects/{proj['id']}/connectors/aconex/events",
        json={"cde_project_id": "1879"},
        headers=H,
    )
    assert r.status_code == 409
    assert "not configured" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_construction_action_polls_through_interface(fake_cde):
    pytest.importorskip("app.containers.construction")
    from tests.conftest import is_construction_kit_enabled

    if not is_construction_kit_enabled():
        pytest.skip("requires CEREBRUM_DOMAIN_KITS=construction")
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    result = await container.cde_poll_events(
        {"cde_project_id": "cde-proj"},
        {},
    )
    assert result["status"] == "ok"
    assert result["source_of_truth"] == "cde"
    assert result["action"] == "cde_poll_events"
    assert result["relevant"] >= 1


@pytest.mark.asyncio
async def test_orchestrator_routes_poll_to_cde():
    from tests.conftest import is_construction_kit_enabled

    if not is_construction_kit_enabled():
        pytest.skip("requires CEREBRUM_DOMAIN_KITS=construction")
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    block = SmartOrchestratorBlock()
    result = await block.process({"user_message": "poll aconex for new mail events"})
    queue = result.get("action_queue") or []
    assert queue[0] == "cde_poll_events"
