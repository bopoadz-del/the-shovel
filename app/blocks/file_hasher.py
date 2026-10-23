"""File Hasher Block - SHA256/MD5 hashing and file metadata extraction."""

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Any

from app.core.universal_base import UniversalBlock


class FileHasherBlock(UniversalBlock):
    """Generate file hashes and extract metadata."""

    name = "file_hasher"
    version = "1.0.0"
    description = "SHA256/MD5 hashing and metadata extraction for files"
    layer = 0
    tags = ["infrastructure", "security", "hash", "metadata"]
    requires = []

    default_config = {
        "default_algorithm": "sha256",
        "chunk_size": 8192
    }

    ui_schema = {
        "input": {
            "type": "text",
            "accept": None,
            "placeholder": "/path/to/file.pdf",
            "multiline": False
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "sha256", "type": "text", "label": "SHA256"},
                {"name": "md5", "type": "text", "label": "MD5"}
            ]
        },
        "quick_actions": [
            {"icon": "#️⃣", "label": "Hash File", "prompt": "/path/to/file.pdf"},
            {"icon": "✅", "label": "Verify Hash", "prompt": '{"action":"verify","file_path":"/path/to/file","expected_hash":"abc123"}'}
        ]
    }

    async def process(self, input_data: Any, params: dict = None) -> dict:
        """Route to appropriate hasher action."""
        params = params or {}
        # Default action is "hash" whether the input is a dict or a bare
        # path -- a dict without an explicit action used to fall through to
        # None and error (found by the decrypt-contract fence).
        action = (params.get("action")
                  or (input_data.get("action") if isinstance(input_data, dict) else None)
                  or "hash")
        handlers = {
            "hash": self.hash_file,
            "metadata": self.metadata,
            "compare": self.compare,
            "health_check": self.health_check,
        }
        handler = handlers.get(action)
        if not handler:
            return {"status": "error", "error": f"Unknown action: {action}"}
        return await handler(input_data, params)

    async def hash_file(self, input_data: Any, params: dict) -> dict:
        """Compute file hash(es)."""
        file_path = self._resolve_path(input_data, params)
        if not file_path:
            return {"status": "error", "error": "No file path provided"}

        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}
        if not path.is_file():
            return {"status": "error", "error": f"Path is not a file: {file_path}"}

        algorithms = params.get("algorithms", ["sha256"])
        chunk_size = self.config.get("chunk_size", 8192)
        results = {}

        hashers = {alg: hashlib.new(alg) for alg in algorithms if alg in hashlib.algorithms_available}

        try:
            # F30: platform uploads are encrypted at rest with a per-file
            # nonce, so hashing the STORED bytes gives a different digest
            # every time the same content is saved — which defeats the whole
            # point of a content hash (dedup, integrity, comparison). Hash
            # the PLAINTEXT; unencrypted files pass through untouched.
            from app.core.file_crypto import open_plaintext
            with open_plaintext(str(path)) as plain_path:
                with open(plain_path, "rb") as f:
                    while chunk := f.read(chunk_size):
                        for hasher in hashers.values():
                            hasher.update(chunk)

            for alg, hasher in hashers.items():
                results[alg] = hasher.hexdigest()

            return {
                "status": "success",
                "file": str(path),
                "size": path.stat().st_size,
                "hashes": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def metadata(self, input_data: Any, params: dict) -> dict:
        """Extract file metadata."""
        file_path = self._resolve_path(input_data, params)
        if not file_path:
            return {"status": "error", "error": "No file path provided"}

        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}

        stat = path.stat()
        mime, _ = mimetypes.guess_type(str(path))

        return {
            "status": "success",
            "file": str(path),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size": stat.st_size,
            "mime_type": mime or "application/octet-stream",
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "is_readable": os.access(path, os.R_OK)
        }

    async def compare(self, input_data: Any, params: dict) -> dict:
        """Compare two files by hash."""
        file_a = params.get("file_a") or (input_data.get("file_a") if isinstance(input_data, dict) else None)
        file_b = params.get("file_b") or (input_data.get("file_b") if isinstance(input_data, dict) else None)

        if not file_a or not file_b:
            return {"status": "error", "error": "Both file_a and file_b required"}

        hash_a = await self.hash_file({"file_path": file_a}, {"algorithms": ["sha256"]})
        hash_b = await self.hash_file({"file_path": file_b}, {"algorithms": ["sha256"]})

        if hash_a.get("status") == "error":
            return hash_a
        if hash_b.get("status") == "error":
            return hash_b

        match = hash_a["hashes"].get("sha256") == hash_b["hashes"].get("sha256")
        return {
            "status": "success",
            "match": match,
            "file_a": file_a,
            "file_b": file_b,
            "hash_a": hash_a["hashes"].get("sha256"),
            "hash_b": hash_b["hashes"].get("sha256")
        }

    async def health_check(self, input_data: Any = None, params: dict = None) -> dict:
        """Health check for file hasher."""
        return {
            "status": "success",
            "block": self.name,
            "version": self.version,
            "algorithms_available": list(hashlib.algorithms_available)
        }

    def _resolve_path(self, input_data: Any, params: dict) -> str | None:
        return params.get("file_path") or (input_data.get("file_path") if isinstance(input_data, dict) else (str(input_data) if isinstance(input_data, str) else None))

    def get_actions(self) -> dict[str, Any]:
        """Return all public methods for block registry."""
        return {
            "hash": self.hash_file,
            "metadata": self.metadata,
            "compare": self.compare,
            "health_check": self.health_check,
        }
