"""Copy selected pages of documents from one project into a NEW project.

Plan first, execute only with the plan's hash (the same plan/execute pattern
as the document-include and doc-typing ops). Additive only: the source
project's rows are read, never updated or deleted, and the target must not
exist yet.

A selection is ``{"doc_id": str, "pages": [int, ...] | "all"}``. A chunk is
taken when the page its text starts on (``chunks.page``) is selected; a chunk
with no page (a spreadsheet, a plain text file) can only be taken with
``"all"``. Chunks are copied with their stored text and vector, page, OCR
flag, layer and authority, so nothing is re-extracted or re-embedded.

The plan carries counts, ids and flags only, never a document name, so its
output can be printed in a public log.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np

ALL_PAGES = "all"


def _normalise_selections(selections: Any) -> List[Dict[str, Any]]:
    """Validate the selection list; raise ValueError naming the first fault."""
    if not isinstance(selections, list) or not selections:
        raise ValueError("selections must be a non-empty list")
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for i, sel in enumerate(selections):
        if not isinstance(sel, dict):
            raise ValueError(f"selection {i} is not an object")
        doc_id = sel.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"selection {i} has no doc_id")
        doc_id = doc_id.strip()
        if doc_id in seen:
            raise ValueError(f"selection {i} repeats a doc_id")
        seen.add(doc_id)
        pages = sel.get("pages")
        if pages == ALL_PAGES:
            out.append({"doc_id": doc_id, "pages": ALL_PAGES})
            continue
        if not isinstance(pages, list) or not pages:
            raise ValueError(f"selection {i} needs pages: a non-empty list or \"all\"")
        clean: List[int] = []
        for p in pages:
            if isinstance(p, bool) or not isinstance(p, int) or p < 1:
                raise ValueError(f"selection {i} has a page that is not a 1-based integer")
            clean.append(p)
        out.append({"doc_id": doc_id, "pages": sorted(set(clean))})
    return out


def _store():
    from app.core.rag.vector_store import get_store
    return get_store()


def plan_page_copy(
    source_project_id: str,
    target_project_id: str,
    selections: Any,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure what a copy would take. Writes nothing.

    Returns per-selection counts (chunks taken, pages found, pages missing),
    the reasons the copy is not eligible (if any) and ``plan_hash`` over the
    exact source chunk ids, the target id and the profile.
    """
    from app.core import projects as _projects

    sels = _normalise_selections(selections)
    if not (target_project_id or "").strip():
        raise ValueError("target project id is required")
    if profile is not None and not isinstance(profile, dict):
        raise ValueError("profile must be a JSON object")
    not_eligible: List[str] = []
    if _projects.get_project(source_project_id) is None:
        not_eligible.append("source project not found")
    target_exists = _projects.get_project(target_project_id) is not None
    if target_exists:
        not_eligible.append("target project already exists")
    if source_project_id == target_project_id:
        not_eligible.append("source and target are the same project")

    store = _store()
    rows_out: List[Dict[str, Any]] = []
    hash_parts: List[Any] = []
    total_chunks = 0
    total_pages = 0
    for i, sel in enumerate(sels):
        doc = _projects.get_document(sel["doc_id"])
        entry: Dict[str, Any] = {"selection": i, "doc_id": sel["doc_id"]}
        if doc is None or doc.get("project_id") != source_project_id:
            entry["found"] = False
            not_eligible.append(f"selection {i}: document not in the source project")
            rows_out.append(entry)
            continue
        chunks = store.doc_chunk_rows(source_project_id, sel["doc_id"])
        if sel["pages"] == ALL_PAGES:
            taken = chunks
            pages_found = sorted({c["page"] for c in taken if c["page"] is not None})
            pages_missing: List[int] = []
        else:
            wanted = set(sel["pages"])
            taken = [c for c in chunks if c["page"] in wanted]
            pages_found = sorted({c["page"] for c in taken})
            pages_missing = sorted(wanted - set(pages_found))
        layers = sorted({(c["knowledge_layer"] or "", c["authority"] or "") for c in taken})
        entry.update({
            "found": True,
            "doc_type": doc.get("doc_type"),
            "pages_selected": sel["pages"],
            "chunks_in_source": len(chunks),
            "chunks_taken": len(taken),
            "pages_found": pages_found,
            "pages_missing": pages_missing,
            "chunks_without_page": sum(1 for c in chunks if c["page"] is None),
            "ocr_chunks_taken": sum(1 for c in taken if c["ocr"]),
        })
        if not taken:
            not_eligible.append(f"selection {i}: no chunk starts on a selected page")
        if len(layers) > 1:
            not_eligible.append(f"selection {i}: taken chunks carry more than one layer/authority")
        total_chunks += len(taken)
        total_pages += len(pages_found)
        hash_parts.append([sel["doc_id"], [c["chunk_id"] for c in taken]])
        rows_out.append(entry)

    payload = json.dumps(
        {
            "source": source_project_id,
            "target": target_project_id,
            "taken": hash_parts,
            "profile": profile or {},
        },
        sort_keys=True,
        default=str,
    )
    return {
        "source_project_id": source_project_id,
        "target_project_id": target_project_id,
        "target_exists": target_exists,
        "selections": rows_out,
        "documents": len(sels),
        "chunks_taken": total_chunks,
        "pages_found": total_pages,
        "eligible": not not_eligible,
        "not_eligible": not_eligible,
        "plan_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def execute_page_copy(
    source_project_id: str,
    target_project_id: str,
    selections: Any,
    *,
    expected_plan_hash: Optional[str],
    reason: str,
    name: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Re-plan, then copy only if the live plan is eligible and its hash
    equals ``expected_plan_hash``. Otherwise ``refused`` says why and nothing
    is written."""
    from app.core import projects as _projects

    plan = plan_page_copy(source_project_id, target_project_id, selections, profile)
    out: Dict[str, Any] = {"plan": plan, "executed": False}
    if not plan["eligible"]:
        out["refused"] = "; ".join(plan["not_eligible"]) + "; nothing written"
        return out
    if not (reason or "").strip():
        out["refused"] = "reason is required with execute=true"
        return out
    if not expected_plan_hash or expected_plan_hash != plan["plan_hash"]:
        out["refused"] = "plan_hash does not match the live plan; nothing written"
        return out

    sels = _normalise_selections(selections)
    store = _store()
    _projects.create_project(
        (name or "").strip() or target_project_id,
        user_id=user_id,
        project_id=target_project_id,
    )
    copied: List[Dict[str, Any]] = []
    for sel in sels:
        src = _projects.get_document(sel["doc_id"]) or {}
        rows = store.doc_chunk_rows(source_project_id, sel["doc_id"], with_vectors=True)
        if sel["pages"] != ALL_PAGES:
            wanted = set(sel["pages"])
            rows = [r for r in rows if r["page"] in wanted]
        meta = dict(src.get("metadata") or {})
        meta["copied_from"] = {
            "project_id": source_project_id,
            "doc_id": sel["doc_id"],
            "pages": sel["pages"],
        }
        new_doc = _projects.add_document(
            target_project_id,
            src.get("original_name") or sel["doc_id"],
            stored_as=src.get("stored_as"),
            file_path=src.get("file_path"),
            size=int(src.get("size") or 0),
            role=src.get("doc_role"),
            content_sha256=src.get("content_sha256"),
            drive_md5=src.get("drive_md5"),
            metadata=meta,
        )
        n = store.upsert_chunks(
            target_project_id,
            new_doc["id"],
            [r["text"] for r in rows],
            np.vstack([r["embedding"] for r in rows]),
            knowledge_layer=rows[0]["knowledge_layer"],
            authority=rows[0]["authority"],
            pages=[r["page"] for r in rows],
            ocr_flags=[r["ocr"] for r in rows],
        )
        _projects.stamp_document_index(
            new_doc["id"],
            chunk_count=n,
            ingest_status=src.get("ingest_status") or _projects.INDEXED,
            extractor_version=src.get("extractor_version"),
            pages_in_file=src.get("pages_in_file"),
            pages_ocrd=src.get("pages_ocrd"),
        )
        copied.append({"source_doc_id": sel["doc_id"], "doc_id": new_doc["id"], "chunks": n})
    if profile:
        _projects.set_project_profile(target_project_id, profile)
    out.update({
        "executed": True,
        "project_id": target_project_id,
        "documents": copied,
        "chunks_copied": sum(c["chunks"] for c in copied),
        "pages_copied": plan["pages_found"],
    })
    return out
