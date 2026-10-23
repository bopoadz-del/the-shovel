import os
import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.dependencies import require_api_key, block_instances, _create_block_instance
from app.blocks import BLOCK_REGISTRY
from app.core import file_crypto
from app.core.projects import create_ingestion_job
from app.worker.ingest_queue import enqueue_ingest

logger = logging.getLogger(__name__)
router = APIRouter()

# Cap + accepted formats are shared with /v1/projects/{id}/documents via
# app.core.upload_limits. These two routes previously carried DIFFERENT caps
# (10 MB here, 50 MB there) and different extension sets, so whether a file was
# accepted depended on which route the client happened to call.
from app.core import upload_limits

ALLOWED_UPLOAD_EXTENSIONS = upload_limits.ALLOWED_UPLOAD_EXTENSIONS

DATA_DIR = os.getenv("DATA_DIR", "./data")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except PermissionError:
    # Fallback to temp dir if DATA_DIR is not writable
    import tempfile
    DATA_DIR = tempfile.gettempdir()


@router.post("/upload")
async def upload_v1(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    auth: dict = Depends(require_api_key),
):
    """File upload endpoint (v1 API).

    Accepts validated files and stores them. Returns URL for processing.

    When ``project_id`` is provided, ALSO:
    - Registers the file as a document in the project (so it survives across
      chat sessions and is reachable via /v1/projects/<id>/documents).
    - Schedules indexing via doc_index.maybe_eager_index in the background so
      the chat can search the file content without the user having to
      re-upload it. This is what the previous "uploaded but not indexed" gap
      was — the indexer was there, we just weren't calling it.
    """
    try:
        # Validate file size
        max_size = upload_limits.max_upload_bytes()
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        if file_size > max_size:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {max_size} bytes")

        # Validate and sanitize filename
        original_name = (file.filename or "unknown").strip()
        if not original_name or original_name in (".", ".."):
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Prevent path traversal
        original_name = os.path.basename(original_name.replace("\\", "/"))
        _, ext = os.path.splitext(original_name.lower())
        # A file sent WITH a project becomes a project document: the one
        # project rule (TEXT_BEARING_EXTS) decides, exactly as on
        # POST /v1/projects/{id}/documents, and it is refused before anything
        # is written. Without a project it is a session/sandbox upload and
        # the sandbox list decides.
        from app.core import compressed
        from app.core.ingest_status import is_ingestible

        # Never opened, unpacked or indexed -- with or without a project.
        if compressed.is_compressed(original_name, compressed.head_of(file.file)):
            raise HTTPException(status_code=415, detail=compressed.COMPRESSED_UPLOAD_DETAIL)
        with_project = bool(project_id and project_id.strip())
        if not with_project and ext not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed")
        if with_project and not is_ingestible(original_name):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"File type '{ext}' is not ingestible into a project: the "
                    f"knowledge base takes text formats only "
                    f"(docs/INGEST_EXCLUSION_RULE.md). Upload it without a "
                    f"project to use it in a chat for the session."
                ),
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        filename = f"{file_id}_{original_name}"
        filepath = os.path.join(DATA_DIR, filename)

        # Save uploaded file — encrypted at rest iff DATA_ENCRYPTION_KEY is set
        # (opt-in; plaintext otherwise — see app/core/file_crypto.py). Streamed
        # in blocks so a large drawing set never sits in memory whole.
        try:
            file_size = file_crypto.write_document_stream(
                filepath, file.file, max_bytes=max_size,
            )
        except file_crypto.UploadTooLarge as exc:
            raise HTTPException(
                status_code=413, detail=f"File too large. Max size: {exc.limit} bytes",
            ) from exc

        # NB: uploaded files live under DATA_DIR, not on any static route —
        # the old ``/static/{filename}`` URL 404'd. Documents are reached
        # through the project documents API once indexed (see project_id
        # branch below), so we do not fabricate a browser URL here.
        response = {
            "filename": original_name,
            "stored_as": filename,
            "size": file_size,
        }

        # If a project_id was supplied, persist + index the file so future
        # questions can search it via doc_index, which retrieves through
        # app.core.rag.retriever (BM25 + vector RRF over pgvector). The actual
        # indexing runs as a FastAPI BackgroundTask so the upload request
        # returns immediately.
        if project_id and project_id.strip():
            try:
                from app.core import projects as projects_store
                from app.core import doc_index
                proj = projects_store.get_project(project_id, user_id=auth.get("user_id"))
                if proj is not None:
                    # Persist the file as a project document. Reuses the same
                    # storage path the upload already wrote to, so no copy.
                    doc = projects_store.add_document(
                        project_id=project_id,
                        original_name=original_name,
                        stored_as=filename,
                        file_path=filepath,
                        size=file_size,
                        role="user_upload",
                    )
                    response["project_id"] = project_id
                    response["document_id"] = doc.get("id") if isinstance(doc, dict) else None
                    response["indexed"] = True
                    response["indexing_status"] = "scheduled"
                    # Only the admin path adds to the project's knowledge base
                    # (owner ruling, app/core/privileges.py): a user's upload
                    # is stored, not indexed.
                    from app.core import privileges

                    if not privileges.caller_may_add_to_project_rag(auth.get("role")):
                        response["indexed"] = False
                        response["indexing_status"] = "not_indexed"
                        response["indexing_detail"] = privileges.PROJECT_RAG_ADMIN_ONLY_DETAIL
                    elif response["document_id"]:
                        # Queue and index under the id the document was STORED
                        # under. For the master-corpus alias that is the backing
                        # corpus — the alias has no project row, so anything
                        # keyed to it either violates a foreign key or indexes
                        # into a project the document does not live in.
                        storage_pid = projects_store.storage_project_id(project_id)
                        job = create_ingestion_job(storage_pid, response["document_id"])
                        ok = await enqueue_ingest(
                            storage_pid, response["document_id"], str(job.id),
                        )
                        if ok:
                            response["indexing_status"] = "queued"
                        elif background_tasks is not None:
                            background_tasks.add_task(
                                doc_index.maybe_eager_index,
                                storage_pid,
                                response["document_id"],
                            )
                else:
                    response["indexed"] = False
                    response["indexing_status"] = "skipped_project_not_found"
            except Exception as e:
                # Don't fail the upload if indexing scheduling fails —
                # the file is already stored. Surface the reason.
                logger.warning("Upload indexing failed: %s", e, exc_info=True)
                response["indexed"] = False
                response["indexing_status"] = f"error: {type(e).__name__}"
        else:
            response["indexed"] = False
            response["indexing_status"] = "no_project_id"

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.post("/v1/upload")
async def upload_v1_endpoint(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    auth: dict = Depends(require_api_key),
):
    """File upload endpoint (v1 API alias)."""
    return await upload_v1(file, project_id, background_tasks, auth)


# ---------------------------------------------------------------------------
# Document Engine Ingest Endpoint
# ---------------------------------------------------------------------------

async def _run_document_engine_block(saved_paths: dict, output_format: str = "json") -> dict:
    """Run the document_engine platform block on decrypted file paths."""
    if "document_engine" not in block_instances:
        block_instances["document_engine"] = _create_block_instance(BLOCK_REGISTRY["document_engine"])

    block = block_instances["document_engine"]
    result = await block.process({
        "pdf_path": saved_paths.get("pdf"),
        "docx_path": saved_paths.get("docx"),
        "xlsx_path": saved_paths.get("xlsx"),
    }, {"output_format": output_format})

    if result.get("status") == "error":
        raise HTTPException(500, result.get("error", "Document engine failed"))
    return result


class IngestResponse(BaseModel):
    status: str
    documents_parsed: int
    glossary_count: int
    requirements_count: int
    constraints_count: int
    schedule_targets_count: int
    equipment_specs_count: int
    risks_count: int
    output: dict


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    pdf: Optional[UploadFile] = File(None, description="PDF document (BOD, spec, drawing)"),
    docx: Optional[UploadFile] = File(None, description="Word document (RFP, contract)"),
    xlsx: Optional[UploadFile] = File(None, description="Excel file (schedule template, BOQ)"),
    output_format: str = "json",
    auth: dict = Depends(require_api_key),
):
    """Document Engine — Parse → Reason → Map pipeline.

    Upload technical documents and run the 3-layer reasoning pipeline:
    Layer 1 (Parse): Extract text, tables, glossary, figures
    Layer 2 (Reason): 8 semantic pipelines (requirements, constraints, risks, etc.)
    Layer 3 (Map): Structured output + downstream feeds for schedule/cost/risk engines

    Returns structured JSON/YAML consumable by downstream blocks.
    """
    from contextlib import ExitStack
    saved_paths = {}
    temp_files = []
    _stack = ExitStack()

    try:
        # --- Layer 0: Accept & persist uploads ---
        for file_obj, key in [(pdf, "pdf"), (docx, "docx"), (xlsx, "xlsx")]:
            if file_obj is None:
                continue

            # Validate extension
            original_name = (file_obj.filename or "unknown").strip()
            _, ext = os.path.splitext(original_name.lower())
            if ext not in (".pdf", ".docx", ".xlsx", ".doc", ".xls"):
                raise HTTPException(400, f"File type '{ext}' not supported for ingest")

            # Save to temp — encrypted at rest iff DATA_ENCRYPTION_KEY is set.
            file_id = str(uuid.uuid4())[:8]
            filename = f"{file_id}_{original_name}"
            filepath = os.path.join(DATA_DIR, filename)

            file_obj.file.seek(0)
            file_crypto.write_document(filepath, file_obj.file.read())

            saved_paths[key] = filepath
            temp_files.append(filepath)

        if not saved_paths:
            raise HTTPException(400, "No valid files provided. Upload at least one PDF/DOCX/XLSX")

        # The document_engine parsers (fitz / python-docx / openpyxl) read files
        # by raw path, so decrypt the stored files to plaintext temp paths for
        # the duration of the pipeline. open_plaintext is a no-op for plaintext
        # files (encryption disabled, or legacy files).
        saved_paths = {
            key: _stack.enter_context(file_crypto.open_plaintext(p))
            for key, p in saved_paths.items()
        }

        result = await _run_document_engine_block(saved_paths, output_format)

        if output_format == "yaml":
            import yaml as yaml_lib
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(
                content=yaml_lib.dump(result, sort_keys=False, allow_unicode=True),
                media_type="application/x-yaml",
                headers={"X-Documents-Parsed": str(result.get("documents_parsed", 0))},
            )

        return {
            "status": "success",
            "documents_parsed": result.get("documents_parsed", 0),
            "glossary_count": len(result.get("glossary", {})),
            "requirements_count": len(result.get("requirements", [])),
            "constraints_count": len(result.get("constraints", [])),
            "schedule_targets_count": len(result.get("schedule_targets", [])),
            "equipment_specs_count": len(result.get("equipment_specs", [])),
            "risks_count": len(result.get("risks", [])),
            "output": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ingest pipeline failed")
        raise HTTPException(status_code=500, detail="Ingest pipeline failed")
    finally:
        # Close any decrypt-to-temp copies created by open_plaintext.
        try:
            _stack.close()
        except Exception:
            # _stack holds decrypt-to-temp copies of CLIENT DOCUMENTS. A close
            # failure leaves plaintext on disk, which is what encryption at
            # rest exists to prevent -- and silence made that invisible.
            # Cleanup must not raise here (it would mask the real ingest
            # error being reported above), so log and continue.
            logger.warning(
                "could not close decrypt-to-temp stack; plaintext copies may "
                "remain on disk", exc_info=True,
            )
        # Cleanup temp files
        for tmp in temp_files:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                logger.warning(
                    "could not remove temp upload file %s; it may contain "
                    "client document data", tmp, exc_info=True,
                )


@router.post("/ingest-via-block")
async def ingest_via_block(
    pdf: Optional[UploadFile] = File(None, description="PDF document"),
    docx: Optional[UploadFile] = File(None, description="Word document"),
    xlsx: Optional[UploadFile] = File(None, description="Excel file"),
    output_format: str = "json",
    auth: dict = Depends(require_api_key),
):
    """Document Engine via platform block registry (dependencies auto-wired).

    Same pipeline as /ingest but routes through BLOCK_REGISTRY so platform
    blocks (pdf, ocr) are resolved and injected automatically.
    """
    from contextlib import ExitStack
    saved_paths = {}
    temp_files = []
    _stack = ExitStack()

    try:
        for file_obj, key in [(pdf, "pdf"), (docx, "docx"), (xlsx, "xlsx")]:
            if file_obj is None:
                continue
            original_name = (file_obj.filename or "unknown").strip()
            _, ext = os.path.splitext(original_name.lower())
            if ext not in (".pdf", ".docx", ".xlsx", ".doc", ".xls"):
                raise HTTPException(400, f"File type '{ext}' not supported")
            file_id = str(uuid.uuid4())[:8]
            filename = f"{file_id}_{original_name}"
            filepath = os.path.join(DATA_DIR, filename)
            # Encrypted at rest iff DATA_ENCRYPTION_KEY is set (opt-in).
            file_obj.file.seek(0)
            file_crypto.write_document(filepath, file_obj.file.read())
            saved_paths[key] = filepath
            temp_files.append(filepath)

        if not saved_paths:
            raise HTTPException(400, "No valid files provided")

        # The document_engine block + its fallback parsers read files by raw
        # path; decrypt the stored files to plaintext temp paths for the run.
        # open_plaintext is a no-op when files are plaintext.
        saved_paths = {
            key: _stack.enter_context(file_crypto.open_plaintext(p))
            for key, p in saved_paths.items()
        }

        result = await _run_document_engine_block(saved_paths, output_format)

        if output_format == "yaml":
            from fastapi.responses import PlainTextResponse
            import yaml as yaml_lib
            return PlainTextResponse(
                content=yaml_lib.dump(result, sort_keys=False, allow_unicode=True),
                media_type="application/x-yaml",
            )

        return {
            "status": "success",
            "documents_parsed": result.get("documents_parsed", 0),
            "platform_blocks_used": result.get("platform_blocks_used", []),
            "output": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ingest via block failed")
        raise HTTPException(status_code=500, detail="Ingest via block failed")
    finally:
        try:
            _stack.close()
        except Exception:
            # _stack holds decrypt-to-temp copies of CLIENT DOCUMENTS. A close
            # failure leaves plaintext on disk, which is what encryption at
            # rest exists to prevent -- and silence made that invisible.
            # Cleanup must not raise here (it would mask the real ingest
            # error being reported above), so log and continue.
            logger.warning(
                "could not close decrypt-to-temp stack; plaintext copies may "
                "remain on disk", exc_info=True,
            )
        for tmp in temp_files:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                logger.warning(
                    "could not remove temp upload file %s; it may contain "
                    "client document data", tmp, exc_info=True,
                )
