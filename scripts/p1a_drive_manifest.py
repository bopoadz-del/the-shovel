#!/usr/bin/env python3
"""P1a — enumerate the complete Drive source manifest for the clean rebuild.

Walks each configured Drive folder recursively and emits:
  {
    "generated_at": ISO timestamp,
    "service_account": "...",
    "folders": [
      {
        "project_id": "example_infra_pack",
        "folder_id": "1ExampleFolderId001xxxxxxxxxxxxxx",
        "folder_name": "the client project",
        "total_files": N,
        "subfolders": [
          {"path": "Design/...", "file_count": N, "folder_id": "..."}
        ],
        "files": [
          {"id", "name", "mimeType", "path", "size", "sha", "class"}
        ]
      }
    ],
    "scanned_images": [ ... ]   # when --classify-pdfs
  }

Every file carries exactly one ``class`` from the discovery-time source
classification (SPEC_CORPUS_FINAL_BUILD_10NOV P2): TEXT / SPREADSHEET /
SCANNED_IMAGE / NON_TEXT. ``--classify-pdfs`` downloads and samples each PDF to
separate a text-layer PDF (TEXT) from an image-only scan (SCANNED_IMAGE);
without it, PDFs are TEXT candidates and the scan split is deferred.

The manifest is committed before Phase 1b ingestion begins.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import gdrive_service, ingest_status


DEFAULT_FOLDERS = {
    "example_infra_pack": "1ExampleFolderId001xxxxxxxxxxxxxx",
}


def _project_folders() -> Dict[str, str]:
    """Return project_id -> Drive folder_id map."""
    parsed = gdrive_service.parse_project_folder_map()
    if parsed:
        return parsed
    return dict(DEFAULT_FOLDERS)


def _pdf_classify_max_mb() -> int:
    """PDF size ceiling for the --classify-pdfs sample download.

    A scanned drawing set can be hundreds of MB; downloading it whole just to
    sample one page is the OOM hazard the ingest path already guards against.
    Above the ceiling a PDF is left unsampled (class stays TEXT candidate).
    ``<= 0`` disables the ceiling.
    """
    raw = (os.getenv("MANIFEST_PDF_CLASSIFY_MAX_MB") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return 64
    return 64


def _sample_pdf_text_layer(file_id: str, size: int) -> Optional[bool]:
    """Download a PDF and report whether a sampled page set has a text layer.

    Returns True / False on a successful sample, None when it cannot be
    determined (download failed, or the file exceeds the size ceiling).
    """
    ceiling = _pdf_classify_max_mb()
    if ceiling > 0 and size > ceiling * 1024 * 1024:
        return None
    data, err = gdrive_service.download_file_bytes(file_id)
    if err or not data:
        print(
            f"[p1a] classify skip {file_id}: {err or 'empty body'}",
            file=sys.stderr,
        )
        return None
    from app.core import doc_index

    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return doc_index.pdf_has_text_layer(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _class_of_file(f: Dict[str, Any], classify_pdfs: bool) -> Tuple[str, Optional[bool]]:
    """Return ``(class, has_text_layer)`` for a walked Drive file dict."""
    name = f.get("name") or ""
    ext = Path(name).suffix.casefold()
    has_text_layer: Optional[bool] = None
    if classify_pdfs and ext == ".pdf":
        file_id = f.get("id") or ""
        if file_id:
            has_text_layer = _sample_pdf_text_layer(file_id, int(f.get("size") or 0))
    return ingest_status.classify_source_document(ext, has_text_layer=has_text_layer), has_text_layer


def _summarize_tree(
    files: List[Dict[str, Any]],
    classify_pdfs: bool,
) -> Dict[str, Any]:
    """Build per-subfolder counts and a flat file list from walk_folder output."""
    by_folder: Dict[str, Dict[str, Any]] = {}
    flat: List[Dict[str, Any]] = []
    scanned: List[Dict[str, Any]] = []
    for f in files:
        path = f.get("_drive_path", f.get("name", ""))
        parent = str(Path(path).parent).replace(".", "")
        if parent not in by_folder:
            by_folder[parent] = {"path": parent or "(root)", "file_count": 0, "folder_id": ""}
        by_folder[parent]["file_count"] += 1
        source_class, _has_text_layer = _class_of_file(f, classify_pdfs)
        row = {
            "id": f["id"],
            "name": f.get("name", ""),
            "mimeType": f.get("mimeType", ""),
            "path": path,
            "size": int(f.get("size") or 0),
            "sha": f.get("md5Checksum") or "",
            "class": source_class,
        }
        flat.append(row)
        if source_class == ingest_status.SOURCE_CLASS_SCANNED_IMAGE:
            scanned.append(row)
    return {
        "total_files": len(files),
        "subfolders": sorted(by_folder.values(), key=lambda x: x["path"]),
        "files": flat,
        "scanned_images": scanned,
    }


def _walk_one(
    folder_id: str,
    folder_name: str,
    project_id: str,
    classify_pdfs: bool,
) -> Dict[str, Any]:
    print(f"[p1a] walking {project_id} / {folder_name} ({folder_id}) ...", file=sys.stderr)
    t0 = time.monotonic()
    files, errors = gdrive_service.walk_folder(folder_id, max_depth=12, page_size=200)
    elapsed = time.monotonic() - t0
    print(
        f"[p1a] {project_id}: {len(files)} files, {len(errors)} errors in {elapsed:.1f}s",
        file=sys.stderr,
    )
    summary = _summarize_tree(files, classify_pdfs)
    return {
        "project_id": project_id,
        "folder_id": folder_id,
        "folder_name": folder_name,
        "walk_seconds": round(elapsed, 1),
        "errors": errors,
        **summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Drive source manifest for clean rebuild")
    ap.add_argument("--output", default="manifests/p1a_drive_manifest.json",
                    help="Output manifest path")
    ap.add_argument("--project", action="append", default=[],
                    help="Specific project_id to include (default: all configured)")
    ap.add_argument("--classify-pdfs", action="store_true",
                    help="Download and sample PDFs to split TEXT from SCANNED_IMAGE")
    args = ap.parse_args()

    if not gdrive_service.is_configured():
        print(
            "ERROR: GDRIVE_SERVICE_ACCOUNT_JSON is not set.",
            file=sys.stderr,
        )
        return 1

    folders = _project_folders()
    if args.project:
        folders = {k: v for k, v in folders.items() if k in args.project}

    if not folders:
        print("ERROR: no project folders configured.", file=sys.stderr)
        return 1

    results: List[Dict[str, Any]] = []
    for project_id, folder_id in folders.items():
        # Best-effort folder name lookup.
        folder_name = folder_id
        try:
            import httpx
            token = gdrive_service._mint_access_token()
            if token:
                with httpx.Client(timeout=30) as client:
                    resp = client.get(
                        f"https://www.googleapis.com/drive/v3/files/{folder_id}",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"fields": "name"},
                    )
                    if resp.status_code == 200:
                        folder_name = resp.json().get("name") or folder_id
        except Exception as exc:  # noqa: BLE001 — name is cosmetic
            print(
                f"[p1a] folder name lookup failed for {folder_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        results.append(_walk_one(folder_id, folder_name, project_id, args.classify_pdfs))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "service_account": _service_account_email(),
        "folders": results,
        "scanned_images": sorted(
            (row for folder in results for row in folder.get("scanned_images", [])),
            key=lambda r: (r.get("path", ""), r.get("name", "")),
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[p1a] manifest written to {out_path}", file=sys.stderr)
    return 0


def _service_account_email() -> str:
    info = gdrive_service._load_service_account_info()
    return info.get("client_email", "unknown") if info else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
