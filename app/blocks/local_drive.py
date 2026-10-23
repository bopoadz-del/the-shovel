"""Local Drive Block - sandboxed local filesystem access.

All paths are resolved relative to LOCAL_DRIVE_ROOT (default: DATA_DIR, itself
defaulting to ./data) and confined to it. The block cannot read, write, or
list anything outside that directory — absolute paths and ``..`` segments that
would escape the root are rejected.
"""

import os
from typing import Any, Dict, Optional

from app.core.universal_base import UniversalBlock
import logging

logger = logging.getLogger(__name__)


def _root() -> str:
    """The directory this block is confined to (resolved at call time)."""
    root = os.path.realpath(
        os.getenv("LOCAL_DRIVE_ROOT") or os.getenv("DATA_DIR", "./data")
    )
    try:
        os.makedirs(root, exist_ok=True)
    except OSError:
        logger.debug(
            "swallowed %s in _root() — continuing",
            "OSError", exc_info=True,
        )
    return root


def _safe_path(requested: str) -> Optional[str]:
    """Resolve ``requested`` inside the drive root.

    Returns the absolute path when it stays within the root, or None when the
    request would escape it. The request is treated as relative to the root;
    leading slashes are stripped and the realpath check catches ``..`` escapes
    and (on Windows) drive-absolute paths.
    """
    root = _root()
    rel = (requested or ".").lstrip("/\\")
    target = os.path.realpath(os.path.join(root, rel))
    if target == root or target.startswith(root + os.sep):
        return target
    return None


class LocalDriveBlock(UniversalBlock):
    """Local filesystem operations, confined to a configured root directory."""

    auto_validate = False
    name = "local_drive"
    version = "1.1"
    description = "Sandboxed local filesystem access: list, read, write files"
    layer = 4
    tags = ["integration", "storage", "local"]
    requires = []

    ui_schema = {
        "input": {
            "type": "file",
            "accept": ["*/*"],
            "placeholder": "Browse local files...",
            "multiline": False
        },
        "output": {
            "type": "list",
            "fields": [
                {"name": "files", "type": "array", "label": "Files"},
                {"name": "path", "type": "text", "label": "Path"}
            ]
        },
        "quick_actions": [
            {"icon": "📁", "label": "Browse Local", "prompt": "List local files"}
        ]
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """List, read, or write files within the configured drive root."""
        params = params or {}
        operation = params.get("operation", "list")
        # Accept the target from input_data or the common param keys.
        path = input_data if isinstance(input_data, str) else (
            params.get("folder_path") or params.get("path") or "."
        )

        try:
            if operation == "write":
                requested = params.get("file_path", "")
                target = _safe_path(requested)
                if target is None:
                    return {"status": "error", "operation": "write",
                            "error": f"Path escapes the allowed directory: {requested}"}
                content = params.get("content", "")
                parent = os.path.dirname(target)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(target, "w") as f:
                    f.write(content)
                return {"status": "success", "operation": "write",
                        "file_path": requested, "bytes_written": len(content)}

            elif operation == "read":
                requested = params.get("file_path", path)
                target = _safe_path(requested)
                if target is None:
                    return {"status": "error", "operation": "read",
                            "error": f"Path escapes the allowed directory: {requested}"}
                if not os.path.isfile(target):
                    return {"status": "error", "operation": "read",
                            "error": f"Not a file: {requested}"}
                with open(target, "r") as f:
                    content = f.read()
                return {"status": "success", "operation": "read",
                        "file_path": requested, "content": content}

            else:
                target = _safe_path(path)
                if target is None:
                    return {"status": "error", "operation": "list",
                            "error": f"Path escapes the allowed directory: {path}"}
                if not os.path.isdir(target):
                    return {"status": "error", "operation": "list",
                            "path": path, "error": f"Not a directory: {path}"}
                # Caller-tunable cap (default 500). The old hardcoded 20-cap
                # silently truncated every directory listing, so the sidebar
                # only ever showed the first ~20 entries alphabetically and
                # the user reported "doesn't show all the documents".
                try:
                    limit = int(params.get("limit", 500))
                except (TypeError, ValueError):
                    limit = 500
                limit = max(1, min(limit, 10000))

                entries = []
                with os.scandir(target) as it:
                    for entry in it:
                        try:
                            stat = entry.stat()
                            entries.append({
                                "name": entry.name,
                                "is_folder": entry.is_dir(),
                                "size_bytes": stat.st_size if entry.is_file() else 0,
                                "modified": int(stat.st_mtime),
                            })
                        except OSError:
                            continue
                # Folders first, then alpha (matches the Google Drive ordering).
                entries.sort(key=lambda e: (not e["is_folder"], e["name"].lower()))
                truncated = len(entries) > limit
                entries = entries[:limit]
                return {
                    "status": "success",
                    "operation": "list",
                    "path": path,
                    "files": entries,
                    "total": len(entries),
                    "truncated": truncated,
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}
