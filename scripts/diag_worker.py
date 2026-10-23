#!/usr/bin/env python3
"""Diagnostic for the Render ingestion worker."""
import os, sys, traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

print(f"PYTHON: {sys.executable}")
print(f"PWD: {Path.cwd()}")
print(f"DATA_DIR: {os.getenv('DATA_DIR')}")
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
print(f"RAG_EMBEDDING_MODEL: {os.getenv('RAG_EMBEDDING_MODEL')}")
print(f"RAG_VECTOR_NAMESPACE: {os.getenv('RAG_VECTOR_NAMESPACE')}")
print(f"DATABASE_URL set: {bool(os.getenv('DATABASE_URL'))}")
print(f"GDRIVE_SERVICE_ACCOUNT_JSON set: {bool(os.getenv('GDRIVE_SERVICE_ACCOUNT_JSON'))}")
print(f"GDRIVE_PROJECT_FOLDERS: {os.getenv('GDRIVE_PROJECT_FOLDERS')}")

mods = [
    "app.core.gdrive_service",
    "app.core.projects",
    "app.core.doc_index",
    "app.core.file_crypto",
    "app.core.rag.embeddings",
    "app.core.rag.vector_store",
]
for name in mods:
    try:
        __import__(name)
        print(f"IMPORT OK: {name}")
    except Exception as exc:
        print(f"IMPORT FAIL: {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()

# Try Drive list
try:
    from app.core import gdrive_service
    print(f"gdrive configured: {gdrive_service.is_configured()}")
    info = gdrive_service._load_service_account_info()
    print(f"gdrive SA info loaded: {bool(info)}")
    token = gdrive_service._mint_access_token()
    print(f"gdrive token minted: {bool(token)}")
    if token:
        fid = os.environ.get("FORK_DIAG_FOLDER_ID", "1ExampleFolderId001xxxxxxxxxxxxxx")
        files, err = gdrive_service.list_folder_files(fid, page_size=2)
        print(f"gdrive list {fid}: {len(files)} files, err={err}")
except Exception as exc:
    print(f"gdrive check FAIL: {type(exc).__name__}: {exc}")
    traceback.print_exc()

print("diag done")
