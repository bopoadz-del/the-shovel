"""A document's KIND OF STANDING, decided at indexing from what is known.

The owner's ruling (19 Nov): when documents disagree, the answer states every
value and names each origin with its kind of standing in plain words; the user
decides. The kind is a LABEL for that reader. Nothing in the product ranks by
it or lets it pick a value.

The kind is derived by one general rule from two facts the product already
stores on every document row: ``doc_type`` (the type table,
``config/document_types.yaml``) and ``doc_role`` (the folder-name table,
``config/doc_role_folders.yaml``). The mapping is configuration
(``config/doc_standing.yaml``, or ``DOC_STANDING_FILE``, or inline JSON in
``DOC_STANDING``); the code holds only the vocabulary and the reader's words.
It is stored as ``metadata["standing"]`` on the document row, so a reader of
an excerpt reads it, never recomputes it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: The metadata key the kind is stored under.
METADATA_KEY = "standing"

#: A document no table entry matches.
OTHER = "other"

#: Kind -> the plain words a reader sees. Order is the owner's standing order
#: of document kinds, kept for the reader; no code ranks by it.
STANDING_KINDS: dict[str, str] = {
    "contract_agreement": "signed contract agreement and conditions",
    "specification": "specification",
    "contract_drawing": "contract drawing",
    "priced_boq": "priced bill of quantities",
    "contractor_submittal": "contractor's submittal",
    OTHER: "other document",
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _default_table_file() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "config", "doc_standing.yaml"))


def _raw_table() -> list[Any]:
    inline = (os.getenv("DOC_STANDING") or "").strip()
    if inline:
        try:
            data = json.loads(inline)
        except ValueError:
            logger.warning("DOC_STANDING is not valid JSON; every document is 'other'")
            return []
        return data if isinstance(data, list) else []
    path = (os.getenv("DOC_STANDING_FILE") or "").strip() or _default_table_file()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    table = data.get("standing") if isinstance(data, Mapping) else None
    return table if isinstance(table, list) else []


def _names(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(n for n in (_norm(v) for v in value) if n)


def standing_table() -> list[tuple[str, frozenset[str], frozenset[str]]]:
    """``(kind, doc_types, doc_roles)`` entries in table order. An entry that
    names an unknown kind, or no doc_type, is dropped with a warning; an empty
    ``doc_roles`` means any role."""
    out: list[tuple[str, frozenset[str], frozenset[str]]] = []
    for entry in _raw_table():
        if not isinstance(entry, Mapping):
            continue
        kind = _norm(entry.get("kind"))
        if kind not in STANDING_KINDS or kind == OTHER:
            logger.warning("doc standing table: unknown kind %r; entry ignored", entry.get("kind"))
            continue
        types = _names(entry.get("doc_types"))
        if not types:
            logger.warning("doc standing table: kind %r names no doc_types; entry ignored", kind)
            continue
        out.append((kind, types, _names(entry.get("doc_roles"))))
    return out


def standing_for(
    doc_type: str | None,
    doc_role: str | None,
    table: list[tuple[str, frozenset[str], frozenset[str]]] | None = None,
) -> str:
    """The kind of standing of a document with this type and role."""
    table = standing_table() if table is None else table
    t, r = _norm(doc_type), _norm(doc_role)
    for kind, types, roles in table:
        if t in types and (not roles or r in roles):
            return kind
    return OTHER


def label_for(kind: str | None) -> str:
    """The reader's words for a stored kind; an unknown or missing kind reads
    as 'other document', never as a guess."""
    return STANDING_KINDS.get(_norm(kind), STANDING_KINDS[OTHER])


def stored_standing(metadata: Mapping[str, Any] | None) -> str | None:
    """The kind stored on a document's metadata, or ``None`` if never set."""
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(METADATA_KEY)
    return str(value) if isinstance(value, str) and value else None


def with_standing(
    metadata: Mapping[str, Any] | None, doc_type: str | None, doc_role: str | None,
) -> dict[str, Any]:
    """A copy of ``metadata`` carrying the decided kind (the caller's own
    metadata is never mutated)."""
    out = dict(metadata) if isinstance(metadata, Mapping) else {}
    out[METADATA_KEY] = standing_for(doc_type, doc_role)
    return out
