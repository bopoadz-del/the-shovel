"""Aconex / CDE connector — vendor-neutral interface, Aconex adapter, fake adapter.

Aconex (or another CDE) is the system of record. The Fork pulls, drafts, and
posts; it caches documents for RAG only. No Fork-owned RFI / submittal / VO /
punch / transmittal registers.
"""

from app.core.cde.cm_rules import CdeEventKind, classify_cde_event, inject, run_cm_overlay
from app.core.cde.config import (
    aconex_enabled,
    connector_mode,
    connector_note,
    default_cde_project_id,
    event_poll_enabled,
    oauth_client_ready,
)
from app.core.cde.errors import CdeError, CdeNotConfiguredError, CdePayloadError
from app.core.cde.events import CdeEvent
from app.core.cde.factory import get_cde_client, reset_cde_clients
from app.core.cde.fake import FakeCdeClient
from app.core.cde.ingest import ingest_cde_bytes, ingest_cde_document, post_rfi_draft, sync_cde_documents
from app.core.cde.poll import poll_cde_events, process_cde_events
from app.core.cde.protocol import CdeClient
from app.core.cde.types import CdeDocument, CdeMail, CdeMailDraft, CdePostResult

__all__ = [
    "CdeClient",
    "CdeDocument",
    "CdeError",
    "CdeEvent",
    "CdeEventKind",
    "CdeMail",
    "CdeMailDraft",
    "CdeNotConfiguredError",
    "CdePayloadError",
    "CdePostResult",
    "FakeCdeClient",
    "aconex_enabled",
    "classify_cde_event",
    "connector_mode",
    "connector_note",
    "default_cde_project_id",
    "event_poll_enabled",
    "get_cde_client",
    "ingest_cde_bytes",
    "ingest_cde_document",
    "inject",
    "oauth_client_ready",
    "poll_cde_events",
    "post_rfi_draft",
    "process_cde_events",
    "reset_cde_clients",
    "run_cm_overlay",
    "sync_cde_documents",
]
