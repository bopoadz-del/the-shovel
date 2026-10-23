"""Subscribe / poll / ingest CDE mail + register events.

Poll walks the existing thin ``CdeClient``. Subscribe accepts CDE-shaped
rows that already have a live id. Neither path writes a Fork-owned RFI
number or grows a local register.

Document-register bytes may be cached for RAG via the existing ingest
helper. Mail events stay ephemeral — overlay + inject only.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.cde.cm_rules import inject, relevant_events, run_cm_overlay
from app.core.cde.errors import CdeError, CdeNotConfiguredError
from app.core.cde.events import CdeEvent, event_from_document, event_from_mail, events_from_payloads
from app.core.cde.factory import get_cde_client
from app.core.cde.ingest import sync_cde_documents
from app.core.cde.protocol import CdeClient

logger = logging.getLogger(__name__)


def _dedupe(events: list[CdeEvent]) -> list[CdeEvent]:
    seen: set[str] = set()
    out: list[CdeEvent] = []
    for event in events:
        if not event.id or event.id in seen:
            continue
        seen.add(event.id)
        out.append(event)
    return out


async def poll_cde_events(
    cde_project_id: str,
    *,
    client: Optional[CdeClient] = None,
    mailbox: str = "inbox",
) -> list[CdeEvent]:
    """List live CDE mail + register rows. Fail closed when unconfigured."""
    client = client or get_cde_client()
    events: list[CdeEvent] = []
    mail = await client.list_mail(cde_project_id, mailbox=mailbox)
    events.extend(event_from_mail(item) for item in mail)
    try:
        rfis = await client.list_rfis(cde_project_id)
        events.extend(event_from_mail(item) for item in rfis)
    except CdeError:
        logger.debug("CDE list_rfis failed; mail list still stands", exc_info=True)
    documents = await client.list_documents(cde_project_id)
    events.extend(event_from_document(item) for item in documents)
    return _dedupe(events)


async def process_cde_events(
    cde_project_id: str,
    *,
    client: Optional[CdeClient] = None,
    events: Optional[list[CdeEvent]] = None,
    payloads: Optional[list[dict[str, Any]]] = None,
    fork_project_id: Optional[str] = None,
    ingest_documents: bool = False,
    mailbox: str = "inbox",
) -> dict[str, Any]:
    """Poll and/or accept CDE rows, run the CM overlay, return inject text.

    ``payloads`` is the subscribe path: CDE-shaped rows with live ids.
    Polling still requires a configured client. Unconfigured + no payloads
    raises ``CdeNotConfiguredError`` (fail closed).
    """
    client = client or get_cde_client()
    accepted = events_from_payloads(payloads) if payloads else []
    polled: list[CdeEvent] = list(events or [])
    if events is None and not accepted:
        polled = await poll_cde_events(
            cde_project_id, client=client, mailbox=mailbox
        )
    elif events is None and accepted:
        # Subscribe-only: still need a configured client so this cannot
        # become a Fork-owned register when Aconex is off.
        if getattr(client, "vendor", "") == "none":
            raise CdeNotConfiguredError()
    merged = _dedupe(list(polled) + accepted)
    overlays = [row for ev in relevant_events(merged) if (row := run_cm_overlay(ev))]
    inject_text = inject(merged)

    ingest_result: Optional[dict[str, Any]] = None
    if ingest_documents and fork_project_id:
        ingest_result = await sync_cde_documents(
            fork_project_id,
            cde_project_id,
            client=client,
            eager_index=False,
        )

    cited = [row["cited"] for row in overlays]
    return {
        "status": "ok",
        "source_of_truth": "cde",
        "vendor": client.vendor,
        "cde_project_id": cde_project_id,
        "listed": len(merged),
        "relevant": len(overlays),
        "overlays": overlays,
        "cited": cited,
        "inject": inject_text,
        "ingest": ingest_result,
        "note": (
            "CM overlay ran on live CDE rows. The Fork did not allocate an "
            "RFI or claim number and does not keep a local register. "
            "Empty inject means there was nothing live to cite."
        ),
    }


# ── optional poll loop (default off) ──────────────────────────────────────
# Does not persist a register. Logs overlay + inject. Fail closed when the
# client is unconfigured or CDE_EVENT_POLL_ENABLED is off.

_task = None


def _poll_interval_seconds() -> int:
    import os

    raw = os.getenv("CDE_EVENT_POLL_INTERVAL_SECONDS", "300").strip()
    try:
        return max(30, int(raw))
    except ValueError:
        return 300


async def run_poll_once(cde_project_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """One poll pass. None when disabled or unconfigured (fail closed)."""
    from app.core.cde import config

    if not config.event_poll_enabled():
        return None
    cid = (cde_project_id or config.default_cde_project_id()).strip()
    if not cid:
        logger.info("CDE event poll skipped — no ACONEX_PROJECT_ID")
        return None
    client = get_cde_client()
    if getattr(client, "vendor", "") == "none":
        logger.info("CDE event poll skipped — client not configured")
        return None
    result = await process_cde_events(cid, client=client)
    logger.info(
        "CDE event poll vendor=%s listed=%s relevant=%s inject=%s",
        result.get("vendor"),
        result.get("listed"),
        result.get("relevant"),
        bool(result.get("inject")),
    )
    return result


def start_event_poller() -> None:
    """Spawn the poll loop. No-op unless CDE_EVENT_POLL_ENABLED is on."""
    import asyncio

    from app.core.cde import config

    global _task
    if _task is not None and not _task.done():
        return
    if not config.event_poll_enabled():
        return
    _task = asyncio.create_task(_poll_loop(), name="cde-event-poll")


async def stop_event_poller() -> None:
    import asyncio

    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except asyncio.CancelledError:
        # Expected: we cancelled the loop one line above because lifespan
        # is shutting down. Not re-raised so remaining teardown still runs.
        logger.debug("CDE event poller cancelled at shutdown")
    _task = None


async def _poll_loop() -> None:
    import asyncio

    interval = _poll_interval_seconds()
    while True:
        try:
            await run_poll_once()
        except Exception:  # noqa: BLE001 — one bad pass must not kill the loop
            logger.exception("CDE event poll pass failed")
        await asyncio.sleep(interval)
