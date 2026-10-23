"""Document-type registry API — Roadmap V2 · Epic 2.

List built-in types, register custom ones at runtime, and classify a document
by filename + content — all without a code change.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator

from app.core import doc_types
from app.dependencies import require_api_key, require_user

router = APIRouter()


class MatchRule(BaseModel):
    filename: List[str] = []
    extensions: List[str] = []
    content: List[str] = []


class DocumentTypeRequest(BaseModel):
    name: str
    description: str = ""
    match: MatchRule = MatchRule()
    expected_fields: List[str] = []


class ClassifyRequest(BaseModel):
    filename: str = ""
    content_sample: str = ""

    @model_validator(mode="after")
    def _needs_something_to_classify(self):
        # Both fields defaulted to "", so an empty body classified nothing
        # and answered 200 with whatever the fallback rule returns.
        if not (self.filename.strip() or self.content_sample.strip()):
            raise ValueError("filename or content_sample is required")
        return self


@router.get("/v1/document-types")
async def list_document_types(auth: dict = Depends(require_api_key)):
    """List every document type — built-in and custom."""
    return {"document_types": doc_types.list_types()}


@router.post("/v1/document-types", status_code=201)
async def add_document_type(
    req: DocumentTypeRequest, auth: dict = Depends(require_user)
):
    """Register a custom document type — extends the registry, no redeploy.

    Admin-only: the registry is process-global and affects classification for
    every tenant, so it must not be mutable by an arbitrary user.
    """
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    if not req.name.strip():
        raise HTTPException(400, "Document type 'name' is required")
    try:
        return doc_types.add_type({
            "name": req.name,
            "description": req.description,
            "match": req.match.model_dump(),
            "expected_fields": req.expected_fields,
        })
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/v1/document-types/{name}")
async def delete_document_type(name: str, auth: dict = Depends(require_user)):
    """Remove a custom document type (built-ins cannot be removed). Admin-only."""
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    if not doc_types.remove_type(name):
        raise HTTPException(
            404, f"Custom document type '{name}' not found "
                 f"(built-in types cannot be removed)"
        )
    return {"status": "deleted", "name": name}


@router.post("/v1/document-types/classify")
async def classify_document(
    req: ClassifyRequest, auth: dict = Depends(require_api_key)
):
    """Classify a document by filename + content sample."""
    return doc_types.classify(req.filename, req.content_sample)
