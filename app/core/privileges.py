"""Admin-only blocks: the ones that reach past the app itself.

Shared by /v1/execute, /v1/chain, the orchestrator, MCP call_tool AND the
agent runtime, so the 403 gate cannot drift across surfaces. An agent is only
as safe as the blocks it may call: the runtime refuses these blocks to a
non-admin caller, and an agent that holds one cannot be pinned, delegated to
or chatted with by a non-admin.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Iterable, Optional

from fastapi import HTTPException

PRIVILEGED_BLOCKS = frozenset({
    "code",          # runs Python / JavaScript it is handed
    "sandbox",       # same, behind a policy
    "mcp_consumer",  # starts whatever command / args it is handed
    "local_drive",   # reads, lists and WRITES the data disk -- every user's uploads
    "web",           # fetches any URL from inside the network
    "webhook",       # sends a request to any URL from inside the network
    # Both fall back to a SERVER-WIDE token (GOOGLE_ACCESS_TOKEN /
    # ONEDRIVE_ACCESS_TOKEN) when the caller passes none, so a direct call
    # would browse whoever's drive that is. Users reach their OWN drive
    # through /v1/drive/*, which passes the per-user token and never goes
    # through this gate.
    "google_drive",
    "onedrive",
})

_caller_role: ContextVar[Optional[str]] = ContextVar("caller_role", default=None)


def set_caller_role(role: Optional[str]) -> None:
    _caller_role.set(role)


def caller_role() -> Optional[str]:
    return _caller_role.get()


# Owner ruling (docs/INGEST_EXCLUSION_RULE.md "Who adds to the knowledge
# base"): only the admin path adds to a project's knowledge base. A user's
# upload is stored and stays usable as a file, but is never indexed into the
# project RAG; the user layer is written only when the user explicitly asks
# through the LLM. Every route that would index into a project asks this.
PROJECT_RAG_ADMIN_ONLY_DETAIL = (
    "Only an admin adds documents to a project's knowledge base. "
    "The file was not indexed."
)


def caller_may_add_to_project_rag(role: Optional[str] = None) -> bool:
    """True only for an admin caller. An unknown caller is not: fails closed."""
    effective = role if role is not None else _caller_role.get()
    return effective == "admin"


def raise_unless_may_add_to_project_rag(role: Optional[str] = None) -> None:
    """403 for a route whose only job is adding documents to a project's RAG."""
    if not caller_may_add_to_project_rag(role):
        raise HTTPException(403, PROJECT_RAG_ADMIN_ONLY_DETAIL)


def privileged_forbidden_detail(block_name: str) -> str:
    return f"Block '{block_name}' executes arbitrary code and is admin-only."


def caller_may_use_block(block_name: Optional[str], role: Optional[str] = None) -> bool:
    """False for a privileged block unless the caller is an admin. An unknown
    caller is NOT an admin: this fails closed."""
    if not block_name or block_name not in PRIVILEGED_BLOCKS:
        return True
    effective = role if role is not None else _caller_role.get()
    return effective == "admin"


def raise_if_privileged_block(block_name: Optional[str], role: Optional[str] = None) -> None:
    if not caller_may_use_block(block_name, role):
        raise HTTPException(403, privileged_forbidden_detail(block_name))


def agent_is_privileged(agent: Any) -> bool:
    """True when any block the agent may call is admin-only."""
    return any(b in PRIVILEGED_BLOCKS for b in (getattr(agent, "allowed_blocks", None) or ()))


def caller_may_use_agent(agent: Any, role: Optional[str] = None) -> bool:
    if not agent_is_privileged(agent):
        return True
    effective = role if role is not None else _caller_role.get()
    return effective == "admin"


def agent_forbidden_detail(agent_name: str) -> str:
    return f"Agent '{agent_name}' is not available."


def raise_if_privileged_steps(steps: Optional[Iterable[Any]], role: Optional[str] = None) -> None:
    for step in steps or []:
        if isinstance(step, dict):
            name = step.get("block")
        else:
            name = getattr(step, "block", None)
        raise_if_privileged_block(name, role)


def collect_project_ids(obj: Any, found: Optional[set] = None) -> set:
    """Walk a request blob and collect nested ``project_id`` strings."""
    if found is None:
        found = set()
    if isinstance(obj, dict):
        pid = obj.get("project_id")
        if isinstance(pid, str) and pid.strip():
            found.add(pid.strip())
        for value in obj.values():
            collect_project_ids(value, found)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            collect_project_ids(value, found)
    return found


def raise_if_inaccessible_project_ids(auth: Optional[dict], *blobs: Any) -> None:
    """404 when any nested project_id is not readable by this caller.

    Same grant as chat/RAG (``get_project_accessible``): owner, approved
    shared, master_corpus. A miss is 404 so existence is not leaked.
    """
    from app.core import projects as projects_store

    uid = (auth or {}).get("user_id")
    for blob in blobs:
        for pid in collect_project_ids(blob):
            if projects_store.get_project_accessible(pid, uid) is None:
                raise HTTPException(404, "Project not found")


def collect_document_ids(obj: Any, found: Optional[set] = None) -> set:
    """Walk a request blob and collect nested ``document_id`` / ``document_ids``."""
    if found is None:
        found = set()
    if isinstance(obj, dict):
        did = obj.get("document_id")
        if isinstance(did, str) and did.strip():
            found.add(did.strip())
        ids = obj.get("document_ids")
        if isinstance(ids, (list, tuple)):
            for item in ids:
                if isinstance(item, str) and item.strip():
                    found.add(item.strip())
        for value in obj.values():
            collect_document_ids(value, found)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            collect_document_ids(value, found)
    return found


def raise_if_inaccessible_document_ids(auth: Optional[dict], *blobs: Any) -> None:
    """404 when any nested document_id is not on a project this caller can read.

    Missing and foreign ids use the same 404 so existence is not leaked.
    """
    from app.core import projects as projects_store

    uid = (auth or {}).get("user_id")
    for blob in blobs:
        for did in collect_document_ids(blob):
            doc = projects_store.get_document(did)
            if doc is None:
                raise HTTPException(404, "document not found")
            pid = doc.get("project_id")
            if projects_store.get_project_accessible(pid, uid) is None:
                raise HTTPException(404, "document not found")
