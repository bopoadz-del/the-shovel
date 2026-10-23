"""Aconex / CDE connector HTTP surface.

GET/POST ``/v1/projects/{id}/connectors`` stay honest about flag vs oauth vs
not_configured. Sync and RFI post go through the vendor-neutral CDE client.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from app.core import audit, privileges, projects as store
from app.core.cde import (
    CdeError,
    CdeNotConfiguredError,
    CdePayloadError,
    connector_mode,
    connector_note,
    default_cde_project_id,
    post_rfi_draft,
    process_cde_events,
    sync_cde_documents,
)
from app.dependencies import require_user
from app.routers.projects import _owned_or_404

router = APIRouter()


class ConnectorRequest(BaseModel):
    connected: bool = True


class CdeSyncRequest(BaseModel):
    cde_project_id: Optional[str] = None


class CdeRfiPostRequest(BaseModel):
    cde_project_id: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    question: Optional[str] = None
    rfis: Optional[list[Dict[str, Any]]] = None
    mail_type: str = "RFI"
    mail_type_id: Optional[str] = None


class CdeEventsRequest(BaseModel):
    """Poll the CDE and/or accept CDE-shaped rows (subscribe).

    ``events`` must already carry a live CDE id. This endpoint does not
    allocate a Fork RFI number and does not persist a register.
    """

    cde_project_id: Optional[str] = None
    events: Optional[list[Dict[str, Any]]] = None
    ingest_documents: bool = False
    mailbox: str = "inbox"


def _resolve_cde_project_id(explicit: Optional[str]) -> str:
    cid = (explicit or default_cde_project_id()).strip()
    if not cid:
        raise HTTPException(
            400,
            "cde_project_id is required (body or ACONEX_PROJECT_ID). "
            "That id is the CDE project, not a Fork register.",
        )
    return cid


def _cde_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CdeNotConfiguredError):
        return HTTPException(409, str(exc))
    if isinstance(exc, CdePayloadError):
        return HTTPException(400, str(exc))
    if isinstance(exc, CdeError):
        return HTTPException(502, str(exc))
    return HTTPException(502, f"CDE request failed: {exc}")


def connector_payload(project_id: str, proj: Dict[str, Any]) -> Dict[str, Any]:
    connected = bool(proj.get("aconex_connected"))
    mode = connector_mode(connected)
    return {
        "name": "aconex",
        "connected": connected,
        "mode": mode,
        "vendor": "aconex",
        "note": connector_note(mode),
    }


@router.post("/v1/projects/{project_id}/connectors/aconex")
async def connect_aconex(
    project_id: str, req: ConnectorRequest, auth: dict = Depends(require_user)
):
    """Set the Aconex / CDE feed flag for a project.

    The flag means "this project has a CDE feed" — a live OAuth client *or*
    a feed managed outside The Fork. It is not a local RFI register.
    """
    _owned_or_404(project_id, auth["user_id"])
    if not store.set_aconex(project_id, req.connected):
        raise HTTPException(404, f"Project '{project_id}' not found")
    await audit.arecord(
        "connector.aconex",
        project_id=project_id,
        connected=req.connected,
        user_id=auth["user_id"],
    )
    proj = store.get_project(project_id, user_id=auth["user_id"]) or {
        "aconex_connected": req.connected
    }
    payload = connector_payload(project_id, proj)
    return {
        "status": "ok",
        "project_id": project_id,
        "aconex_connected": req.connected,
        "mode": payload["mode"],
        "connector": payload,
        "readiness": store.compute_readiness(project_id),
    }


@router.get("/v1/projects/{project_id}/connectors")
async def list_connectors(project_id: str, auth: dict = Depends(require_user)):
    """Connector status for a project.

    ``mode`` is ``flag``, ``oauth``, or ``not_configured`` — never implied
    live OAuth from the boolean flag alone.
    """
    proj = _owned_or_404(project_id, auth["user_id"])
    return {
        "project_id": project_id,
        "connectors": [connector_payload(project_id, proj)],
    }


@router.post(
    "/v1/projects/{project_id}/connectors/aconex/sync",
)
async def sync_aconex(
    project_id: str,
    req: CdeSyncRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(require_user),
):
    """Pull CDE documents into the project corpus (RAG cache)."""
    _owned_or_404(project_id, auth["user_id"])
    privileges.raise_unless_may_add_to_project_rag(auth.get("role"))
    cde_project_id = _resolve_cde_project_id(req.cde_project_id)
    try:
        result = await sync_cde_documents(
            project_id,
            cde_project_id,
            eager_index=False,
        )
    except (CdeNotConfiguredError, CdeError) as exc:
        raise _cde_http_error(exc) from exc
    # Index after the response — same pattern as Drive import.
    for doc in result.get("documents") or []:
        doc_id = doc.get("id")
        if doc_id:
            from app.core import doc_index

            background_tasks.add_task(doc_index.maybe_eager_index, project_id, doc_id)
    if result.get("stored", 0) or result.get("skipped", 0):
        store.set_aconex(project_id, True)
    await audit.arecord(
        "connector.aconex.sync",
        project_id=project_id,
        cde_project_id=cde_project_id,
        stored=result.get("stored"),
        user_id=auth["user_id"],
    )
    result["readiness"] = store.compute_readiness(project_id)
    result["aconex_connected"] = True
    result["mode"] = connector_mode(True)
    return result


@router.post(
    "/v1/projects/{project_id}/connectors/aconex/rfi",
)
async def post_aconex_rfi(
    project_id: str,
    req: CdeRfiPostRequest,
    auth: dict = Depends(require_user),
):
    """Post an RFI draft to the CDE. The chat bubble is not the register."""
    _owned_or_404(project_id, auth["user_id"])
    cde_project_id = _resolve_cde_project_id(req.cde_project_id)
    payload = req.model_dump(exclude_none=True)
    payload.pop("cde_project_id", None)
    try:
        posted = await post_rfi_draft(cde_project_id, payload)
    except (CdeNotConfiguredError, CdeError) as exc:
        raise _cde_http_error(exc) from exc
    await audit.arecord(
        "connector.aconex.rfi_posted",
        project_id=project_id,
        cde_project_id=cde_project_id,
        cde_id=posted.cde_id,
        user_id=auth["user_id"],
    )
    return {
        "status": "posted",
        "source_of_truth": "cde",
        "cde_project_id": cde_project_id,
        "post": posted.as_dict(),
        "note": (
            "Posted to the CDE. The Fork did not allocate an RFI number "
            "and does not keep an RFI register."
        ),
    }


@router.post(
    "/v1/projects/{project_id}/connectors/aconex/events",
)
async def poll_aconex_events(
    project_id: str,
    req: CdeEventsRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(require_user),
):
    """Poll or accept CDE mail+register events and run the CM overlay.

    Inject cites live CDE rows or stays silent. No Fork-owned register.
    """
    _owned_or_404(project_id, auth["user_id"])
    if req.ingest_documents:
        privileges.raise_unless_may_add_to_project_rag(auth.get("role"))
    cde_project_id = _resolve_cde_project_id(req.cde_project_id)
    try:
        result = await process_cde_events(
            cde_project_id,
            payloads=req.events,
            fork_project_id=project_id if req.ingest_documents else None,
            ingest_documents=req.ingest_documents,
            mailbox=req.mailbox or "inbox",
        )
    except (CdeNotConfiguredError, CdeError) as exc:
        raise _cde_http_error(exc) from exc
    ingest = result.get("ingest") or {}
    for doc in ingest.get("documents") or []:
        doc_id = doc.get("id")
        if doc_id:
            from app.core import doc_index

            background_tasks.add_task(doc_index.maybe_eager_index, project_id, doc_id)
    await audit.arecord(
        "connector.aconex.events",
        project_id=project_id,
        cde_project_id=cde_project_id,
        listed=result.get("listed"),
        relevant=result.get("relevant"),
        user_id=auth["user_id"],
    )
    result["project_id"] = project_id
    return result
