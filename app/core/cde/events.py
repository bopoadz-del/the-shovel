"""CDE mail + register rows as events. Cache/provenance only — not a Fork register.

A CDE event is a live row from the CDE (mail, RFI, transmittal, or document
register). Fork never allocates the id or reference. Subscribe/poll/ingest
carry these shapes; they must not grow an RFI / claim / VO / punch log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.cde.errors import CdePayloadError
from app.core.cde.types import CdeDocument, CdeMail


@dataclass(frozen=True)
class CdeEvent:
    """One live CDE mail or register row, ready for the CM overlay."""

    id: str
    source: str  # "mail" | "register"
    subject: str = ""
    body: str = ""
    title: str = ""
    reference: str = ""
    mail_type: str = ""
    correspondence_type: str = ""
    doc_type: str = ""
    status: str = ""
    filename: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        """Concatenated fields the existing CM classifiers already understand."""
        parts = [
            self.subject,
            self.body,
            self.title,
            self.mail_type,
            self.correspondence_type,
            self.doc_type,
            self.filename,
            self.reference,
        ]
        return " ".join(p for p in parts if p)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "subject": self.subject,
            "body": self.body,
            "title": self.title,
            "reference": self.reference,
            "mail_type": self.mail_type,
            "correspondence_type": self.correspondence_type,
            "doc_type": self.doc_type,
            "status": self.status,
            "filename": self.filename,
            "extra": dict(self.extra),
            "source_of_truth": "cde",
        }


def event_from_mail(mail: CdeMail) -> CdeEvent:
    return CdeEvent(
        id=str(mail.id or "").strip(),
        source="mail",
        subject=mail.subject or "",
        body=mail.body or "",
        reference=mail.reference or "",
        mail_type=mail.mail_type or "",
        correspondence_type=mail.correspondence_type or "",
        status=mail.status or "",
        extra=dict(mail.extra),
    )


def event_from_document(document: CdeDocument) -> CdeEvent:
    return CdeEvent(
        id=str(document.id or "").strip(),
        source="register",
        subject=document.title or "",
        title=document.title or "",
        doc_type=document.doc_type or "",
        status=document.status or "",
        filename=document.filename or "",
        extra=dict(document.extra),
    )


def event_from_payload(row: dict[str, Any]) -> CdeEvent:
    """Accept a CDE-shaped push (subscribe). Require a live CDE id.

    Rejects rows with no CDE id rather than inventing an RFI / claim number.
    Any ``rfi_number`` / ``fork_rfi_number`` on the payload is ignored.
    """
    if not isinstance(row, dict):
        raise CdePayloadError("CDE event payload must be an object")
    cid = str(row.get("id") or row.get("cde_id") or "").strip()
    if not cid:
        raise CdePayloadError(
            "CDE event is missing id — The Fork will not invent an RFI or "
            "claim number. Push the live CDE row or poll the CDE."
        )
    source = str(row.get("source") or "mail").strip().lower()
    if source not in {"mail", "register"}:
        source = "mail"
    extra = dict(row.get("extra") or {})
    # Local draft labels are not source of truth and must not be cited.
    extra.pop("rfi_number", None)
    extra.pop("fork_rfi_number", None)
    extra.pop("claim_number", None)
    return CdeEvent(
        id=cid,
        source=source,
        subject=str(row.get("subject") or ""),
        body=str(row.get("body") or ""),
        title=str(row.get("title") or ""),
        reference=str(row.get("reference") or ""),
        mail_type=str(row.get("mail_type") or ""),
        correspondence_type=str(row.get("correspondence_type") or ""),
        doc_type=str(row.get("doc_type") or ""),
        status=str(row.get("status") or ""),
        filename=str(row.get("filename") or ""),
        extra=extra,
    )


def events_from_payloads(rows: Optional[list[Any]]) -> list[CdeEvent]:
    out: list[CdeEvent] = []
    seen: set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            raise CdePayloadError("CDE event payload must be an object")
        event = event_from_payload(row)
        if event.id in seen:
            continue
        seen.add(event.id)
        out.append(event)
    return out
