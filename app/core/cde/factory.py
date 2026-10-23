"""Resolve the CDE client. Fake for tests; Aconex when flagged+credentialed; else fail closed."""

from __future__ import annotations

from typing import Optional

from app.core.cde import config
from app.core.cde.aconex import AconexCdeClient
from app.core.cde.errors import CdeNotConfiguredError
from app.core.cde.fake import FakeCdeClient
from app.core.cde.protocol import CdeClient

_FAKE: Optional[FakeCdeClient] = None


class NotConfiguredCdeClient(CdeClient):
    """Every operation raises. Used when the real client is feature-flagged off."""

    vendor = "none"

    async def list_mail(self, cde_project_id: str, *, mailbox: str = "inbox"):
        raise CdeNotConfiguredError()

    async def get_mail(self, cde_project_id: str, mail_id: str):
        raise CdeNotConfiguredError()

    async def list_documents(self, cde_project_id: str):
        raise CdeNotConfiguredError()

    async def get_document(self, cde_project_id: str, document_id: str):
        raise CdeNotConfiguredError()

    async def download_document(self, cde_project_id: str, document_id: str):
        raise CdeNotConfiguredError()

    async def list_rfis(self, cde_project_id: str):
        raise CdeNotConfiguredError()

    async def list_transmittals(self, cde_project_id: str):
        raise CdeNotConfiguredError()

    async def post_mail(self, cde_project_id: str, draft):
        raise CdeNotConfiguredError()


def reset_cde_clients() -> None:
    """Drop the fake singleton. Tests call this between cases."""
    global _FAKE
    _FAKE = None


def get_cde_client() -> CdeClient:
    """Return the process CDE client.

    * ``CDE_ADAPTER=fake`` — in-memory adapter (tests). Same instance so posts persist.
    * ``ACONEX_ENABLED`` + credentials — live Oracle Aconex REST client.
    * otherwise — NotConfiguredCdeClient (fail closed, no local register).
    """
    adapter = config.requested_adapter()
    if adapter == "fake":
        global _FAKE
        if _FAKE is None:
            _FAKE = FakeCdeClient()
        return _FAKE
    if adapter == "aconex" or config.oauth_client_ready():
        if not config.oauth_client_ready():
            return NotConfiguredCdeClient()
        return AconexCdeClient()
    return NotConfiguredCdeClient()
