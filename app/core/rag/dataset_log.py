"""Dataset logging of verified Q&A (item 2b).

A verified Q&A turn is appended as a JSON-lines record to a dataset file under
the project's data dir (never git): the question, the evidence plan the agentic
retrieval generated, the ids of the sources the turn retrieved/cited, the final
answer, the citation rows and the grounding verdict.

Purpose: training/eval material for a construction-specific small model later
(owner: separate project, post-market). The record is a general shape; nothing
here names a project, an item or a column, and no owner template values are
read from source.
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Default data-dir, mirroring the app-wide ``DATA_DIR`` convention.
_DEFAULT_DATA_DIR = "./data"

#: Source layers that count as the project's own documents for the verdict.
#: Under the owner's one-layer ruling (07:36Z) ``master_corpus`` IS the
#: general-knowledge corpus, so it is its own verdict class, not a project doc.
_PROJECT_LAYERS = frozenset({"own", "project", "project_corpus"})

#: Layer -> verdict class, mirroring ``app.core.rag.source_class``'s layer map
#: in the dataset-log verdict vocabulary (``project_document`` /
#: ``general_knowledge`` instead of ``project_corpus`` / ``knowledge_base``).
_LAYER_VERDICT = {
    "own": "project_document",
    "project": "project_document",
    "project_corpus": "project_document",
    "master_corpus": "master_corpus",
    "general_knowledge": "general_knowledge",
}

#: Strongest-first verdict precedence. Template is decided before layer (a
#: template can sit in any layer), exactly as ``source_class.classify`` does;
#: then the owner's order project > master_corpus > general_knowledge, with
#: ``other`` last for a row no layer names.
_VERDICT_PRECEDENCE = (
    "template",
    "project_document",
    "master_corpus",
    "general_knowledge",
    "other",
)

#: Identity/attribution keys kept on a citation row (never chunk text).
_CITATION_KEYS = (
    "doc_id", "document_id", "id", "chunk_id",
    "name", "title", "source_label", "layer", "origin", "page",
)

#: One lock guards the append so two turns on one process never interleave.
_APPEND_LOCK = threading.Lock()


@dataclass(frozen=True)
class DatasetRecord:
    """One verified Q&A for the dataset file.

    ``citations`` holds reduced citation rows (identity/attribution only);
    ``verdict`` is a single grounding verdict string, e.g. ``project_document``
    / ``general_knowledge`` / ``refused``.
    """

    question: str
    answer: str
    verdict: str
    evidence_plan: tuple[str, ...] = ()
    retrieved_ids: tuple[str, ...] = ()
    citations: tuple[dict[str, Any], ...] = ()
    logged_at: str = ""

    def with_timestamp(self, now: datetime | None = None) -> "DatasetRecord":
        """Return a copy carrying an ISO-8601 UTC ``logged_at`` stamp."""
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        return replace(self, logged_at=stamp)

    def as_jsonable(self) -> dict[str, Any]:
        """The record as a JSON-serialisable mapping (tuples become lists)."""
        data = asdict(self)
        data["evidence_plan"] = list(self.evidence_plan)
        data["retrieved_ids"] = list(self.retrieved_ids)
        data["citations"] = [dict(c) for c in self.citations]
        return data


def data_dir() -> Path:
    """The app data dir (``DATA_DIR`` env or the default)."""
    return Path(os.getenv("DATA_DIR", _DEFAULT_DATA_DIR))


def dataset_log_path(project_id: str, base: Path | None = None) -> Path:
    """The JSON-lines dataset file for one project (never under git)."""
    root = base if base is not None else data_dir()
    return root / "dataset" / str(project_id) / "qa.jsonl"


def reduced_citation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only a source row's identity/attribution keys (no chunk text)."""
    return {
        key: row[key]
        for key in _CITATION_KEYS
        if key in row and row[key] not in (None, "")
    }


def _row_verdict(row: Mapping[str, Any]) -> str:
    """One row's class in the verdict vocabulary.

    Template is decided first (a template can sit in any layer), mirroring
    ``source_class.classify``; the layer then maps through ``_LAYER_VERDICT``,
    and an unrecognised or absent layer is ``other`` (never a project document).
    """
    if str(row.get("source_class") or "").strip() == "template":
        return "template"
    layer = str(row.get("layer") or row.get("origin") or "").strip()
    return _LAYER_VERDICT.get(layer, "other")


def verdict_from_sources(sources: Sequence[Mapping[str, Any]] | None) -> str:
    """One grounding verdict from the citation rows.

    Empty rows -> ``refused``. Each row is classed in the verdict vocabulary:
    ``template`` (owner's template class), ``project_document`` (own/project
    layer), ``master_corpus``, ``general_knowledge``, or ``other``. The turn
    takes the strongest class present, in ``_VERDICT_PRECEDENCE`` order.
    """
    rows = [r for r in (sources or []) if r]
    if not rows:
        return "refused"
    classes = {_row_verdict(r) for r in rows}
    for candidate in _VERDICT_PRECEDENCE:
        if candidate in classes:
            return candidate
    return "other"


def append_dataset_record(
    project_id: str,
    record: DatasetRecord,
    base: Path | None = None,
) -> Path:
    """Append one record as a JSON line; creates the file's parents.

    Thread-safe within a process via a module lock. Returns the path written.
    """
    stamped = record if record.logged_at else record.with_timestamp()
    path = dataset_log_path(project_id, base=base)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(stamped.as_jsonable(), ensure_ascii=False, sort_keys=True)
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return path
