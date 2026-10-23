"""HTTP surface for the RAG layer — one route, narrowly scoped.

``POST /v1/rag/search`` is what dashboards, debugging, and external
callers use to inspect retrieval against a project's indexed docs. The
chat block does its own in-process retrieval via ``retriever.retrieve()``
and does not go through this route — adding HTTP hops between blocks
would be silly. The route is here for visibility, not for plumbing.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import require_api_key

router = APIRouter()


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text to retrieve against")
    project_id: str = Field(..., min_length=1, description="Scope to one project")
    k: int = Field(5, ge=1, le=50, description="Number of chunks to return")
    # Eval hook: lets recall runners exercise the intent-gated GK knobs
    # (retriever.DOC_LOOKUP_INTENTS / CALC_KB_INTENTS). Chat does not go
    # through this route and keeps passing no intent.
    intent: Optional[str] = Field(
        None, description="Optional retrieval intent; gates GK ranking knobs",
    )


class RagSearchChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    score: float


class RagSearchResponse(BaseModel):
    chunks: List[RagSearchChunk]
    count: int
    available: bool
    fast_search: bool  # True when sqlite-vec is loaded; False = numpy fallback


def _searchable_project_or_404(project_id: str, auth: dict) -> str:
    """Tenancy gate shared by every RAG read surface.

    The same rule chat (``routers/chat.py``) and document search
    (``routers/doc_search.py``) already apply: owner -> admin-approved
    shared -> admin on PLATFORM projects only, via
    ``projects.get_project_accessible``. This route reached
    ``retriever.retrieve()`` without it, so a caller who was merely
    authenticated could post any project id and read its chunks back.

    404 for missing AND for foreign, deliberately: a 403 would confirm
    the project exists and make this route an id-enumeration oracle.
    Auth failure is 401 (require_api_key), not 404 — a 404 is membership
    or existence, never a bad JWT. Empty retrieval is 200 / count 0.

    Master-corpus membership covers the backing source id too (S13 /
    WATCH-1): a caller who can open ``master_corpus`` can search
    ``projects_folder`` (or whatever ``MASTER_CORPUS_SOURCE_PROJECT_ID``
    is). The picker still 404s the physical id.

    Returns the id to retrieve against — for the master-corpus alias that
    is its backing source project, matching what doc_search resolves.
    """
    from app.core import projects as projects_store

    if projects_store.get_project_accessible(project_id, auth.get("user_id")) is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return projects_store._master_corpus_source(project_id) or project_id


@router.post("/v1/rag/search", response_model=RagSearchResponse)
async def rag_search(
    req: RagSearchRequest,
    auth: dict = Depends(require_api_key),
) -> RagSearchResponse:
    """Search the project's indexed chunks for the query.

    Returns ``available=false`` when the embedding stack isn't installed
    (install via ``pip install -r requirements-rag.txt``); status code
    is still 200 so dashboards can render the empty state without
    treating it as an error.
    """
    from app.core.rag import retriever as _r
    from app.core.rag.vector_store import get_store
    from app.core.models import EMBEDDING_DIM

    # Access check FIRST: an unauthorised caller must not be able to tell
    # "no such project" from "the RAG stack is not installed" either.
    search_project_id = _searchable_project_or_404(req.project_id, auth)

    if not _r.available():
        return RagSearchResponse(
            chunks=[], count=0, available=False, fast_search=False,
        )

    try:
        # Off the event loop: a synchronous retrieve here stalled the single
        # worker past Render's 5 s health check (live 21 Sep 2026, 18:32).
        chunks = await asyncio.to_thread(
            _r.retrieve, req.query, search_project_id, k=req.k, intent=req.intent,
        )
    except ValueError as exc:
        # Caller-side error (e.g. empty project_id) — surface as 400
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        # Unexpected — log + 503 so dashboards can surface "retrieval down"
        raise HTTPException(status_code=503, detail=f"retrieval failed: {exc}")

    store = get_store(dim=EMBEDDING_DIM)
    return RagSearchResponse(
        chunks=[
            RagSearchChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                chunk_index=c.chunk_index,
                text=c.text,
                score=float(c.score or 0.0),
            )
            for c in chunks
        ],
        count=len(chunks),
        available=True,
        fast_search=store.fast_search,
    )


@router.get("/v1/rag/gk-status")
async def rag_gk_status(
    q: str = "FIDIC 2017 Golden Principles Contractor claim time-bar",
    project_id: str = "master_corpus",
    auth: dict = Depends(require_api_key),
):
    """Read-only diagnostic for general-knowledge (``RAG_GENERAL_KNOWLEDGE_PROJECTS``) grounding.

    Added to settle why FIDIC KB notes don't surface in chat. Answers:
      1. Is the GK corpus seeded?  -> ``gk_docs`` lists each GK doc by name.
      2. Does GK retrieve at all?  -> ``gk_only_hits`` searches the GK store
         directly (proves seeding + shows GK-internal ranking for the probe).
      3. What does chat actually get? -> ``merged_topk`` is exactly what
         ``retrieve_with_filter`` (the chat grounding path) returns for
         ``project_id`` — so we see whether/where GK chunks rank vs the
         active project's docs.
    """
    from app.core.rag import retriever as _r
    from app.core import projects as _projects

    # Same gate as /v1/rag/search: this diagnostic returns merged_topk,
    # which is exactly what the chat grounding path retrieves for
    # ``project_id`` — i.e. the project's own chunk text.
    project_id = _searchable_project_or_404(project_id, auth)

    out: dict = {
        "available": _r.available(),
        "gk_project_ids": [],
        "gk_docs": {},
        "probe_query": q,
        "active_project": project_id,
        "gk_only_hits": [],
        "merged_topk": [],
        "errors": [],
    }
    try:
        out["gk_project_ids"] = _r._general_knowledge_project_ids()
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"gk_ids: {e}")

    if not out["available"]:
        out["errors"].append("embedding stack unavailable")
        return out

    for pid in out["gk_project_ids"]:
        try:
            docs = _projects.list_documents(pid) or []
            out["gk_docs"][pid] = [
                {"doc_id": d.get("id"), "name": d.get("original_name") or d.get("name")}
                for d in docs
            ]
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"list_docs {pid}: {e}")

    # 1) GK-only direct search — proves seeding + GK-internal ranking.
    try:
        from app.core.rag.vector_store import get_store as _gs
        embedder = _r.get_embedder()
        qv = embedder.encode([q])[0]
        store = _gs(dim=embedder.dim)
        for pid in out["gk_project_ids"]:
            try:
                for c in store.search(pid, qv, k=8, query_text=q):
                    out["gk_only_hits"].append({
                        "project_id": c.project_id, "doc_id": c.doc_id,
                        "chunk_index": c.chunk_index,
                        "score": round(float(c.score or 0.0), 4),
                        "snippet": (c.text or "")[:100],
                    })
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"gk_search {pid}: {e}")
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"embed: {e}")

    # 2) What chat actually gets — merged active+GK top-k after filtering.
    try:
        chunks, _noise = _r.retrieve_with_filter(q, project_id, k=8)
        for c in chunks:
            out["merged_topk"].append({
                "project_id": c.project_id, "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "score": round(float(c.score or 0.0), 4),
                "snippet": (c.text or "")[:100],
            })
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"merged: {e}")

    return out
