"""In-memory CDE adapter for tests. Never pretends to be a live Aconex instance."""

from __future__ import annotations

from app.core.cde.errors import CdeError
from app.core.cde.protocol import CdeClient
from app.core.cde.types import CdeDocument, CdeMail, CdeMailDraft, CdePostResult

_DEFAULT_SPEC = (
    "CDE cache sample — waterproofing membrane to extend 200mm above "
    "finished floor level at wet areas. This file is a RAG cache only; "
    "Aconex remains the system of record."
)


class FakeCdeClient(CdeClient):
    """Deterministic in-memory CDE. Posts are recorded, not numbered by Fork."""

    vendor = "fake"

    def __init__(self) -> None:
        self.mail: list[CdeMail] = [
            CdeMail(
                id="mail-100",
                subject="Site query — hold point",
                body="Please confirm hold-point inspection for grid A.",
                mail_type="General",
                correspondence_type="General",
                status="open",
                reference="MAIL-100",
                mailbox="inbox",
            ),
            CdeMail(
                id="rfi-200",
                subject="RFI — waterproofing upstand height",
                body="Confirm 200mm upstand at wet areas per spec.",
                mail_type="RFI",
                correspondence_type="RFI",
                status="open",
                reference="RFI-200",
                mailbox="inbox",
            ),
            CdeMail(
                id="trn-300",
                subject="Transmittal — architectural IFC set",
                body="Issued for construction, revision C.",
                mail_type="Transmittal",
                correspondence_type="Transmittal",
                status="issued",
                reference="TRN-300",
                mailbox="inbox",
            ),
            CdeMail(
                id="mail-400",
                subject="Notice — chiller delivery has slipped four weeks",
                body="Supplier advised the chiller delivery has slipped four weeks.",
                mail_type="General",
                correspondence_type="General",
                status="open",
                reference="MAIL-400",
                mailbox="inbox",
            ),
            CdeMail(
                id="mail-500",
                subject="Clash — duct vs beam at grid B2",
                body="Clash detected between supply duct and structural beam at grid B2.",
                mail_type="General",
                correspondence_type="General",
                status="open",
                reference="MAIL-500",
                mailbox="inbox",
            ),
        ]
        self.documents: list[CdeDocument] = [
            CdeDocument(
                id="cde-doc-1",
                title="Waterproofing specification excerpt",
                filename="cde_waterproofing_spec.txt",
                revision="C",
                doc_type="Specification",
                status="current",
            ),
            CdeDocument(
                id="cde-doc-clash",
                title="Clash report — MEP vs structure L03",
                filename="cde_clash_report.txt",
                revision="A",
                doc_type="Clash report",
                status="current",
            ),
        ]
        self.files: dict[str, bytes] = {
            "cde-doc-1": _DEFAULT_SPEC.encode("utf-8"),
            "cde-doc-clash": (
                "Clash report cache — supply duct vs beam at grid B2. "
                "Aconex remains the system of record."
            ).encode("utf-8"),
        }
        self.posted: list[CdePostResult] = []
        self._post_seq = 0

    def _mail_by_id(self, mail_id: str) -> CdeMail:
        for item in self.mail:
            if item.id == mail_id:
                return item
        raise CdeError(f"CDE mail '{mail_id}' not found")

    def _doc_by_id(self, document_id: str) -> CdeDocument:
        for item in self.documents:
            if item.id == document_id:
                return item
        raise CdeError(f"CDE document '{document_id}' not found")

    async def list_mail(
        self, cde_project_id: str, *, mailbox: str = "inbox"
    ) -> list[CdeMail]:
        box = (mailbox or "inbox").lower()
        return [m for m in self.mail if (m.mailbox or "inbox").lower() == box]

    async def get_mail(self, cde_project_id: str, mail_id: str) -> CdeMail:
        return self._mail_by_id(mail_id)

    async def list_documents(self, cde_project_id: str) -> list[CdeDocument]:
        return list(self.documents)

    async def get_document(
        self, cde_project_id: str, document_id: str
    ) -> CdeDocument:
        return self._doc_by_id(document_id)

    async def download_document(
        self, cde_project_id: str, document_id: str
    ) -> bytes:
        self._doc_by_id(document_id)
        data = self.files.get(document_id)
        if data is None:
            raise CdeError(f"CDE document '{document_id}' has no file bytes")
        return data

    async def list_rfis(self, cde_project_id: str) -> list[CdeMail]:
        return [m for m in self.mail if _is_rfi(m)]

    async def list_transmittals(self, cde_project_id: str) -> list[CdeMail]:
        return [m for m in self.mail if _is_transmittal(m)]

    async def post_mail(
        self, cde_project_id: str, draft: CdeMailDraft
    ) -> CdePostResult:
        self._post_seq += 1
        cde_id = f"posted-{self._post_seq}"
        # The CDE allocates the reference. Fork must not treat a local
        # DRAFT-RFI / RFI-0001 as source of truth.
        reference = f"{(draft.mail_type or 'MAIL').upper()}-CDE-{self._post_seq:04d}"
        result = CdePostResult(
            cde_id=cde_id,
            reference=reference,
            status="posted",
            vendor=self.vendor,
            raw={
                "subject": draft.subject,
                "body": draft.body,
                "mail_type": draft.mail_type,
                "cde_project_id": cde_project_id,
            },
        )
        self.posted.append(result)
        self.mail.append(
            CdeMail(
                id=cde_id,
                subject=draft.subject,
                body=draft.body,
                mail_type=draft.mail_type or "RFI",
                correspondence_type=draft.mail_type or "RFI",
                status="draft",
                reference=reference,
                mailbox="sentbox",
            )
        )
        return result


def _is_rfi(mail: CdeMail) -> bool:
    blob = f"{mail.mail_type} {mail.correspondence_type}".lower()
    return "rfi" in blob or "request for information" in blob


def _is_transmittal(mail: CdeMail) -> bool:
    blob = f"{mail.mail_type} {mail.correspondence_type}".lower()
    return "transmittal" in blob
