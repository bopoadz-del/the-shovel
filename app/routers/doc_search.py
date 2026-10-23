"""Document search endpoint — HTTP layer for per-project full-text search.

Roadmap V2 · Stream C — Phase C3.

GET /v1/projects/{project_id}/documents/search
  q       : str      — search query (required, must not be empty/whitespace)
  top_k   : int = 5  — max results; clamped to [1, 25]

Authentication: Bearer JWT or API key (require_user dependency).
Ownership:      same 404-never-leak-existence pattern as projects.py.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.core import doc_index
from app.core import projects as store
from app.dependencies import require_user

router = APIRouter()


@router.get("/v1/projects/{project_id}/documents/search")
async def search_documents(
    project_id: str,
    q: str = "",
    top_k: int = 5,
    auth: dict = Depends(require_user),
):
    """Search indexed documents within a project.

    Returns ranked results plus a count of unsupported (skipped) documents.
    Never leaks project existence to unauthorized callers (404 for both missing
    and cross-tenant projects).
    """
    # Ownership check — scoped to the calling user.
    # Document search is read-only, so admin-approved platform projects (and
    # the master-corpus alias backed by one) are visible to non-owners.
    proj = store.get_project(project_id, user_id=auth["user_id"], include_admin_approved=True)
    if not proj:
        raise HTTPException(404, f"Project '{project_id}' not found")

    # Pilot: resolve the master-corpus alias so search queries the backing
    # source corpus (e.g. master_corpus -> projects_folder). Chat already
    # does this resolution in app.routers.agents; document search must match.
    search_project_id = store._master_corpus_source(project_id) or project_id

    # Validate query
    if not q or not q.strip():
        raise HTTPException(400, "Query parameter 'q' must not be empty")

    # Clamp top_k to valid range
    top_k = max(1, min(25, top_k))

    # Perform search against the resolved project id (lazy-builds index if needed)
    results = await doc_index.search_project_documents(search_project_id, q.strip(), top_k)

    # Count skipped/unsupported docs from the resolved index -- a COUNT over
    # the entry rows, never a load of the entries themselves.
    try:
        skipped_count = doc_index.skipped_count(search_project_id)
    except Exception:
        skipped_count = 0

    return {
        "project_id": project_id,
        "query": q,
        "results": results,
        "count": len(results),
        "skipped_unsupported": skipped_count,
    }
