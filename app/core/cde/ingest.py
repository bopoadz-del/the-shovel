"""Pull CDE documents into the existing Fork project corpus (RAG cache only).

Reuses Drive-import storage: file_crypto.write_document → add_document →
maybe_eager_index. Provenance lives on document.metadata — there is no
parallel CDE store and no Fork-owned register.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any, Optional

from app.core import doc_index, file_crypto, projects as store
from app.core.cde.errors import CdeError, CdeNotConfiguredError
from app.core.cde.factory import get_cde_client
from app.core.cde.protocol import CdeClient
from app.core.cde.types import CdeDocument, CdeMailDraft, CdePostResult
from app.core.ingest_status import TEXT_BEARING_EXTS

logger = logging.getLogger(__name__)


def _data_dir() -> str:
    path = os.getenv("DATA_DIR", "./data")
    os.makedirs(path, exist_ok=True)
    return path


def _safe_filename(name: str) -> str:
    base = os.path.basename(str(name or "cde-document").replace("\\", "/")).strip()
    return base or "cde-document"


def ingest_cde_bytes(
    fork_project_id: str,
    *,
    original_name: str,
    payload: bytes,
    cde_document_id: str,
    vendor: str,
    cde_project_id: str,
    title: str = "",
    revision: str = "",
    extra: Optional[dict[str, Any]] = None,
    eager_index: bool = True,
) -> dict[str, Any]:
    """Store CDE bytes as a project document with CDE provenance metadata."""
    original_name = _safe_filename(original_name)
    _, ext = os.path.splitext(original_name.lower())
    if ext not in TEXT_BEARING_EXTS:
        raise CdeError(
            f"CDE file type '{ext}' is not ingestible: the knowledge base takes "
            f"text formats only (docs/INGEST_EXCLUSION_RULE.md)"
        )
    digest = hashlib.sha256(payload).hexdigest()
    existing = store.find_document_by_sha(fork_project_id, digest)
    if existing:
        meta = existing.get("metadata") or {}
        if meta.get("cde_document_id") == cde_document_id:
            return {"status": "exists", "document": existing, "skipped": True}
        return {
            "status": "exists",
            "document": existing,
            "skipped": True,
            "error": "DUPLICATE_SHA",
        }

    file_id = str(uuid.uuid4())[:8]
    stored_as = f"{file_id}_{original_name}"
    filepath = os.path.join(_data_dir(), stored_as)
    file_crypto.write_document(filepath, payload)

    metadata = {
        "source": "cde",
        "origin": "cde_cache",
        "cde_vendor": vendor,
        "cde_document_id": cde_document_id,
        "cde_project_id": cde_project_id,
        "cde_title": title,
        "cde_revision": revision,
        "system_of_record": "cde",
    }
    if extra:
        metadata["cde_extra"] = extra

    doc = store.add_document(
        fork_project_id,
        original_name,
        stored_as,
        filepath,
        len(payload),
        content_sha256=digest,
        metadata=metadata,
    )
    if eager_index:
        try:
            doc_index.maybe_eager_index(fork_project_id, doc["id"])
        except Exception:
            logger.warning(
                "CDE ingest index failed for %s / %s",
                fork_project_id,
                doc["id"],
                exc_info=True,
            )
    return {"status": "stored", "document": doc, "skipped": False}


async def ingest_cde_document(
    fork_project_id: str,
    document: CdeDocument,
    *,
    cde_project_id: str,
    client: Optional[CdeClient] = None,
    eager_index: bool = True,
) -> dict[str, Any]:
    client = client or get_cde_client()
    payload = await client.download_document(cde_project_id, document.id)
    return ingest_cde_bytes(
        fork_project_id,
        original_name=document.filename or f"{document.id}.bin",
        payload=payload,
        cde_document_id=document.id,
        vendor=client.vendor,
        cde_project_id=cde_project_id,
        title=document.title,
        revision=document.revision,
        extra=document.extra,
        eager_index=eager_index,
    )


async def sync_cde_documents(
    fork_project_id: str,
    cde_project_id: str,
    *,
    client: Optional[CdeClient] = None,
    eager_index: bool = True,
) -> dict[str, Any]:
    """Pull the CDE document register into the project corpus (cache)."""
    client = client or get_cde_client()
    documents = await client.list_documents(cde_project_id)
    stored: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for document in documents:
        try:
            result = await ingest_cde_document(
                fork_project_id,
                document,
                cde_project_id=cde_project_id,
                client=client,
                eager_index=eager_index,
            )
            if result.get("skipped"):
                skipped.append(result)
            else:
                stored.append(result)
        except (CdeError, CdeNotConfiguredError) as exc:
            errors.append({"cde_document_id": document.id, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the pull
            logger.warning("CDE ingest failed for %s: %s", document.id, exc, exc_info=True)
            errors.append({"cde_document_id": document.id, "error": str(exc)})
    return {
        "status": "ok",
        "vendor": client.vendor,
        "cde_project_id": cde_project_id,
        "listed": len(documents),
        "stored": len(stored),
        "skipped": len(skipped),
        "errors": errors,
        "documents": [row["document"] for row in stored],
        "source_of_truth": "cde",
    }


def draft_from_rfi_payload(payload: dict[str, Any]) -> CdeMailDraft:
    """Build a CDE mail draft from a chat / rfi_generator payload.

    Does not allocate a Fork RFI number. Any local ``rfi_number`` on the
    payload is treated as a draft label only and is not posted as SoR.
    """
    rfi = payload
    rfis = payload.get("rfis")
    if isinstance(rfis, list) and rfis and isinstance(rfis[0], dict):
        rfi = rfis[0]
    subject = (
        str(
            payload.get("subject")
            or rfi.get("subject")
            or payload.get("title")
            or "RFI draft"
        ).strip()
        or "RFI draft"
    )
    body = str(
        payload.get("body")
        or payload.get("question")
        or rfi.get("question")
        or rfi.get("body")
        or ""
    ).strip()
    if not body:
        raise CdeError("RFI draft has no question/body to post to the CDE")
    extra = {}
    local_label = rfi.get("rfi_number") or payload.get("rfi_number")
    if local_label:
        extra["fork_draft_label"] = str(local_label)
        extra["fork_draft_label_is_not_sor"] = True
    return CdeMailDraft(
        subject=subject,
        body=body,
        mail_type=str(payload.get("mail_type") or "RFI"),
        mail_type_id=payload.get("mail_type_id"),
        extra=extra,
    )


async def post_rfi_draft(
    cde_project_id: str,
    payload: dict[str, Any],
    *,
    client: Optional[CdeClient] = None,
) -> CdePostResult:
    """Post an RFI draft to the CDE. Chat is not the register."""
    client = client or get_cde_client()
    draft = draft_from_rfi_payload(payload)
    return await client.post_rfi(cde_project_id, draft)
