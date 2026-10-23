"""Vendor-neutral CDE client. Procore / ACC can implement the same door later."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.cde.types import CdeDocument, CdeMail, CdeMailDraft, CdePostResult


class CdeClient(ABC):
    """Thin CDE interface: list/get mail, documents, RFIs, transmittals; post a draft.

    Implementations MUST fail closed when the vendor is unreachable or
    unconfigured. They MUST NOT persist Fork-owned RFI / submittal / VO /
    punch / transmittal registers.
    """

    vendor: str = ""

    @abstractmethod
    async def list_mail(
        self, cde_project_id: str, *, mailbox: str = "inbox"
    ) -> list[CdeMail]:
        """List project mail (Oracle: GET /api/projects/{id}/mail)."""

    @abstractmethod
    async def get_mail(self, cde_project_id: str, mail_id: str) -> CdeMail:
        """Fetch one mail item."""

    @abstractmethod
    async def list_documents(self, cde_project_id: str) -> list[CdeDocument]:
        """List the document register (Oracle: GET /api/projects/{id}/register)."""

    @abstractmethod
    async def get_document(
        self, cde_project_id: str, document_id: str
    ) -> CdeDocument:
        """Fetch document-register metadata."""

    @abstractmethod
    async def download_document(
        self, cde_project_id: str, document_id: str
    ) -> bytes:
        """Download document bytes (Oracle: GET /api/projects/{id}/register/{docid})."""

    @abstractmethod
    async def list_rfis(self, cde_project_id: str) -> list[CdeMail]:
        """RFIs are CDE mail of that correspondence type — not a Fork log."""

    @abstractmethod
    async def list_transmittals(self, cde_project_id: str) -> list[CdeMail]:
        """Transmittals are CDE mail of that type — not a Fork register."""

    @abstractmethod
    async def post_mail(
        self, cde_project_id: str, draft: CdeMailDraft
    ) -> CdePostResult:
        """Post a mail/RFI draft. The CDE allocates the real number."""

    async def post_rfi(
        self, cde_project_id: str, draft: CdeMailDraft
    ) -> CdePostResult:
        """Post an RFI draft. Wrapper that pins mail_type=RFI."""
        pinned = CdeMailDraft(
            subject=draft.subject,
            body=draft.body,
            mail_type=draft.mail_type or "RFI",
            mail_type_id=draft.mail_type_id,
            extra=dict(draft.extra),
        )
        return await self.post_mail(cde_project_id, pinned)
