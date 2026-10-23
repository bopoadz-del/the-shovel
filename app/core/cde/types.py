"""Vendor-neutral CDE record shapes. Cache/provenance only — not source of truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class CdeMail:
    """A mail item (RFI, transmittal, or general correspondence) in the CDE."""

    id: str
    subject: str
    body: str = ""
    mail_type: str = ""
    correspondence_type: str = ""
    status: str = ""
    reference: str = ""
    mailbox: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "body": self.body,
            "mail_type": self.mail_type,
            "correspondence_type": self.correspondence_type,
            "status": self.status,
            "reference": self.reference,
            "mailbox": self.mailbox,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class CdeDocument:
    """A document-register entry in the CDE. Bytes are fetched separately."""

    id: str
    title: str
    filename: str
    revision: str = ""
    doc_type: str = ""
    status: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "filename": self.filename,
            "revision": self.revision,
            "doc_type": self.doc_type,
            "status": self.status,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class CdeMailDraft:
    """A draft to post back to the CDE. The CDE allocates the real number."""

    subject: str
    body: str
    mail_type: str = "RFI"
    mail_type_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CdePostResult:
    """What the CDE returned after accepting a draft. Fork does not number this."""

    cde_id: str
    reference: str = ""
    status: str = "posted"
    vendor: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cde_id": self.cde_id,
            "reference": self.reference,
            "status": self.status,
            "vendor": self.vendor,
            "raw": dict(self.raw),
            "source_of_truth": "cde",
        }
