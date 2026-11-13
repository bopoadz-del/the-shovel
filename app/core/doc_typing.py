"""Document type and role at ingest, read from tables, not from code branches.

* ``doc_type`` comes from the document-type registry (``app.core.doc_types``,
  table ``config/document_types.yaml`` plus runtime custom types). An
  unmatched name keeps the column default ``document``.
* ``doc_role`` comes from a folder-name table: the nearest folder in the
  document's source path whose name is in the table gives the role. The
  table is configuration (``config/doc_role_folders.yaml``, or the file named
  by ``DOC_ROLE_FOLDERS_FILE``, or inline JSON in ``DOC_ROLE_FOLDERS``); the
  code holds only the role vocabulary the database constraint allows.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any

import yaml

from app.core import doc_types

logger = logging.getLogger(__name__)

#: Column default for a document the type table does not recognise.
DEFAULT_DOC_TYPE = "document"

#: Roles a folder table may assign. ``ck_documents_doc_role`` allows these
#: plus the readiness roles in ``app.core.projects``; alembic 0034 widens it.
FOLDER_ROLES = frozenset({"contract", "draft", "tender", "award", "reporting", "design"})

#: Metadata keys that carry a document's source path, nearest source first.
_PATH_KEYS = ("drive_path", "local_drive_path", "source_path")


def type_for(filename: str, content_sample: str = "") -> str:
    """The registry's type for this name (and content, when given)."""
    verdict = doc_types.classify(filename or "", content_sample or "")
    if verdict.get("needs_user_confirmation"):
        return DEFAULT_DOC_TYPE
    return str(verdict.get("name") or DEFAULT_DOC_TYPE)


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).casefold()


def _default_table_file() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "config", "doc_role_folders.yaml"))


def _raw_table() -> Mapping[str, Any]:
    inline = (os.getenv("DOC_ROLE_FOLDERS") or "").strip()
    if inline:
        try:
            data = json.loads(inline)
        except ValueError:
            logger.warning("DOC_ROLE_FOLDERS is not valid JSON; folder roles are off")
            return {}
        return data if isinstance(data, Mapping) else {}
    path = (os.getenv("DOC_ROLE_FOLDERS_FILE") or "").strip() or _default_table_file()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    table = data.get("folder_roles") if isinstance(data, Mapping) else None
    return table if isinstance(table, Mapping) else {}


def folder_role_table() -> dict[str, str]:
    """Normalised folder name -> role. Entries naming an unknown role are
    dropped with a warning, so configuration cannot break the constraint."""
    out: dict[str, str] = {}
    for folder, role in _raw_table().items():
        role_n = _norm(str(role))
        if role_n not in FOLDER_ROLES:
            logger.warning("doc role table: %r maps to unknown role %r; ignored", folder, role)
            continue
        key = _norm(str(folder))
        if key:
            out[key] = role_n
    return out


def source_path(metadata: Mapping[str, Any] | None) -> str:
    """The document's source path from its metadata, or ``""``."""
    if not isinstance(metadata, Mapping):
        return ""
    for key in _PATH_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def role_for_path(path: str, table: Mapping[str, str] | None = None) -> str | None:
    """Role of the nearest enclosing folder named in the table, else None.

    The last path segment is the file itself and is never read as a folder.
    """
    table = folder_role_table() if table is None else table
    if not table or not path:
        return None
    folders = [s for s in re.split(r"[\\/]+", path) if s][:-1]
    for folder in reversed(folders):
        role = table.get(_norm(folder))
        if role:
            return role
    return None
