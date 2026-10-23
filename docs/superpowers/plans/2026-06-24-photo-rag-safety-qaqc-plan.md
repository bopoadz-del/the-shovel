# Photo RAG with Safety + QA/QC Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add construction-site photos as first-class RAG chunks on Render, populated by a YOLOv8n model fine-tuned locally on the construction-3-001.zip corpus for 6–7 safety + QA/QC classes (registry holds 33).

**Architecture:** Two-host split. Local PC owns training + batch inference (GPU). Render owns metadata storage + RAG serving (CPU-only, no PC runtime dependency). Connection is a one-shot HTTPS POST from PC after each training cycle.

**Tech Stack:** Python 3.12, FastAPI, ultralytics YOLOv8, Grounding DINO (HuggingFace `IDEA-Research/grounding-dino-tiny`), Label Studio (operator-side OSS labelling), Alembic, Postgres 16 with `BYTEA` photo storage, pytest with `pytest-asyncio` + `httpx.AsyncClient`.

## Global Constraints

- Python interpreter: `.venv/Scripts/python.exe` (Windows PowerShell)
- No emojis in any code, comment, commit message, log line, UI string, or test fixture (operator hard rule)
- **Subagents MUST stage only the files named in their task's commit step.** NEVER `git add -A`, `git add .`, or `git commit -am`. The working tree contains in-flight work from a parallel Claude session as of 2026-06-24 (see Task 2.3 guardrail); a blanket `add` would sweep that work into the wrong commit.
- Image block input key is `file_path` (NOT legacy `image_path`); `safety_detector` follows the same contract
- New image-block mode name is `safety_qaqc` (snake_case)
- Photo chunks live in a new dedicated `photo_chunks` table (parallel to `chunks`). The existing `chunks` table is NOT modified — `chunks.embedding` is NOT NULL and `chunks.text_search` is a generated tsvector, both of which break if photo rows with no embedding are inserted
- `project_id` stays `null` for construction-3-001.zip rows — never inferred from pixels (see `feedback-no-assumptions.md`)
- Class IDs in `safety_classes.json` are stable forever; once assigned, never reused
- Active subset of classes is renumbered 0..N-1 at training time via a class-map JSON stored alongside the weights
- `min_examples_required` default per class: 30 bboxes
- Training validation gate: refuse to ship if any active class has <30 examples, OR validation mAP@0.5 < 0.3 (override with `--force-low-quality`)
- Tests mock external services (Grounding DINO, ultralytics YOLO inference, Render HTTP). Real model invocation reserved for operator smoke tests.
- Each task ends with one commit
- Render service id: `srv-d8hdc6ek1jcs739rq5sg` — Alembic migrations auto-run on deploy
- Phase 3 (active-project upload pathway, Render-side runtime inference, active-learning loop, VLM captions) is OUT OF SCOPE for this plan
- Local PC venv has `requirements-cv.txt` installed (PyTorch 2.12 + CUDA 13 + ultralytics 8.4) — no new ML deps needed for training; Grounding DINO needs `pip install transformers huggingface_hub`

---

## File Structure

**Created in Phase 0:**
- `app/blocks/safety_classes.json` — 33-class registry (the data)
- `app/blocks/safety_classes.py` — loader + validator (the code)
- `alembic/versions/0006_photo_chunks_and_photos.py` — migration (creates two NEW tables; does NOT modify existing `chunks` table)
- `scripts/survey_photo_corpus.py` — Phase 0 survey tool
- `tests/test_safety_classes.py`
- `tests/test_migration_0006_photo_chunks.py`
- `tests/test_survey_photo_corpus.py`

**Created in Phase 1:**
- `scripts/prelabel_with_dino.py`
- `scripts/train_safety_qaqc.py`
- `tests/test_prelabel_with_dino.py`
- `tests/test_train_safety_qaqc.py`

**Created in Phase 2:**
- `app/blocks/safety_detector.py`
- `app/routers/admin_photos.py`
- `app/routers/photos.py`
- `scripts/infer_photo_metadata.py`
- `scripts/export_to_render.py`
- `tests/test_safety_detector.py`
- `tests/test_image_block_safety_qaqc_mode.py`
- `tests/test_construction_container_yolo_compose.py`
- `tests/test_infer_photo_metadata.py`
- `tests/test_admin_photos.py`
- `tests/test_photo_serving.py`
- `tests/test_retriever_photo_chunks.py`
- `tests/test_export_to_render.py`

**Modified in Phase 2:**
- `app/blocks/image.py` — add `safety_qaqc` mode
- `app/containers/construction/__init__.py` — add `_classes_to_hazards()` + `_classes_to_defects()` helpers; `safety_compliance_audit` + `qa_qc_inspection` use them when `safety_qaqc` output present
- `app/core/rag/retriever.py` — photo-chunk content adapter
- `app/main.py` — register the two new routers

---

## Phase 0 — Foundation (4 tasks)

### Task 0.1: Class registry (data + loader)

**Files:**
- Create: `app/blocks/safety_classes.json`
- Create: `app/blocks/safety_classes.py`
- Test: `tests/test_safety_classes.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `load_class_registry() -> List[ClassEntry]` — sorted by id ascending
  - `ClassEntry` dataclass: `id: int`, `name: str`, `category: Literal["safety","qaqc"]`, `definition: str`, `active: bool`, `weights_version: Optional[str]`, `min_examples_required: int`
  - `get_active_classes() -> List[ClassEntry]` — `active=True` only
  - `get_class_by_id(class_id: int) -> ClassEntry` — raises `KeyError` if missing
  - `get_class_by_name(name: str) -> ClassEntry` — raises `KeyError` if missing
  - `validate_registry(entries: List[ClassEntry]) -> None` — raises `ValueError` on duplicate id/name, missing fields, or active-without-weights

- [ ] **Step 1: Write the failing test**

```python
# tests/test_safety_classes.py
from app.blocks.safety_classes import (
    load_class_registry, get_active_classes, get_class_by_id, get_class_by_name,
    validate_registry, ClassEntry,
)
import pytest


def test_registry_loads_all_33_classes():
    entries = load_class_registry()
    assert len(entries) == 33
    assert entries[0].id == 0


def test_all_ids_unique():
    entries = load_class_registry()
    ids = [e.id for e in entries]
    assert len(set(ids)) == len(ids)


def test_all_names_unique():
    entries = load_class_registry()
    names = [e.name for e in entries]
    assert len(set(names)) == len(names)


def test_get_active_classes_returns_only_active():
    entries = get_active_classes()
    assert all(e.active for e in entries)
    assert all(e.weights_version for e in entries)


def test_get_class_by_id_known():
    e = get_class_by_id(0)
    assert e.name == "no_hardhat"


def test_get_class_by_id_unknown_raises():
    with pytest.raises(KeyError):
        get_class_by_id(999)


def test_get_class_by_name_known():
    e = get_class_by_name("concrete_crack")
    assert e.category == "qaqc"


def test_validate_rejects_duplicate_id():
    entries = [
        ClassEntry(id=0, name="a", category="safety", definition="", active=False, weights_version=None, min_examples_required=30),
        ClassEntry(id=0, name="b", category="safety", definition="", active=False, weights_version=None, min_examples_required=30),
    ]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_registry(entries)


def test_validate_rejects_active_without_weights():
    entries = [
        ClassEntry(id=0, name="a", category="safety", definition="", active=True, weights_version=None, min_examples_required=30),
    ]
    with pytest.raises(ValueError, match="active.*weights"):
        validate_registry(entries)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_safety_classes.py -v`
Expected: ImportError (module missing)

- [ ] **Step 3: Create the registry JSON**

Write `app/blocks/safety_classes.json` with exactly 33 entries. IDs 0–6 are active (V1 candidates from the spec). IDs 7–32 are placeholders. Definitions copy verbatim from spec.

```json
[
  {"id": 0, "name": "no_hardhat", "category": "safety", "definition": "Worker on-site, head visible, no hard hat", "active": true, "weights_version": "v1", "min_examples_required": 30},
  {"id": 1, "name": "no_high_vis_vest", "category": "safety", "definition": "Worker on-site, torso visible, no reflective vest/jacket", "active": true, "weights_version": "v1", "min_examples_required": 30},
  {"id": 2, "name": "fall_hazard_unprotected", "category": "safety", "definition": "Open edge / opening / elevated platform with no handrail, OR person at visible height with no harness", "active": true, "weights_version": "v1", "min_examples_required": 30},
  {"id": 3, "name": "concrete_crack", "category": "qaqc", "definition": "Visible crack in cured concrete surface (structural or surface)", "active": true, "weights_version": "v1", "min_examples_required": 30},
  {"id": 4, "name": "concrete_honeycomb", "category": "qaqc", "definition": "Voids / exposed-aggregate patches from poor compaction or vibration", "active": true, "weights_version": "v1", "min_examples_required": 30},
  {"id": 5, "name": "rebar_correct_inspection", "category": "qaqc", "definition": "Reinforcement bars correctly visible during pre-pour inspection (no defect)", "active": true, "weights_version": "v1", "min_examples_required": 30},
  {"id": 6, "name": "rebar_exposed_defect", "category": "qaqc", "definition": "Reinforcing steel visible at surface of cured concrete (cover defect)", "active": true, "weights_version": "v1", "min_examples_required": 30},
  {"id": 7, "name": "excavation_slope_unsafe", "category": "safety", "definition": "Excavation slope steeper than safe angle of repose for visible soil type", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 8, "name": "loose_sand_edge", "category": "safety", "definition": "Loose sand or spoil piled at or over excavation/trench edge", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 9, "name": "missing_jersey_barrier", "category": "safety", "definition": "Edge of pit / drop without jersey barrier where one is required", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 10, "name": "missing_lifeline", "category": "safety", "definition": "Worker at height without a visible lifeline anchor", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 11, "name": "non_safety_shoes", "category": "safety", "definition": "Worker footwear that does not appear to be steel-toe safety boot (colour-based heuristic; non-black assumed not safety)", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 12, "name": "faulty_scaffolding", "category": "safety", "definition": "Scaffold without diagonal bracing or with visibly simple/improvised construction", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 13, "name": "missing_toe_board", "category": "safety", "definition": "Scaffold top working platform without a toe board on the edge", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 14, "name": "no_scaffold_green_tag", "category": "safety", "definition": "Scaffold without a green inspection tag visible", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 15, "name": "unorganized_cables", "category": "safety", "definition": "Electrical cables sprawled / tangled / not bundled", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 16, "name": "cables_no_trunking", "category": "safety", "definition": "Electrical cables run without trunking / conduit where required", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 17, "name": "water_near_cable", "category": "safety", "definition": "Standing water or wet surface immediately adjacent to electrical cable (RELATIONAL - V2)", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 18, "name": "ceiling_opening_unsealed", "category": "safety", "definition": "Horizontal ceiling opening not closed by plywood (RELATIONAL - V2)", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 19, "name": "shaft_opening_unsealed", "category": "safety", "definition": "Elevator/services shaft opening not closed by scaffold pipes or plywood (RELATIONAL - V2)", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 20, "name": "rebar_no_plastic_cap", "category": "safety", "definition": "Vertical reinforcement bar end without plastic safety cap", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 21, "name": "uneven_tile_grouting", "category": "qaqc", "definition": "Tile grouting that is uneven, missing, or inconsistent in width", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 22, "name": "bulging_concrete", "category": "qaqc", "definition": "Concrete column or wall with visible bulging from formwork failure", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 23, "name": "uneven_plaster", "category": "qaqc", "definition": "Plaster work with visible unevenness, ridges, or surface waves", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 24, "name": "dirty_false_ceiling", "category": "qaqc", "definition": "False/suspended ceiling tiles visibly dirty or stained at handover", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 25, "name": "uncentered_electrical_socket", "category": "qaqc", "definition": "Electrical socket / switch plate visibly off-centre or misaligned", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 26, "name": "no_chamfer_on_edge", "category": "qaqc", "definition": "Concrete edge without a chamfer where one is required", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 27, "name": "column_misalignment", "category": "qaqc", "definition": "Column off-alignment with adjacent columns in the same line", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 28, "name": "kerb_misalignment", "category": "qaqc", "definition": "Kerb stones off-alignment with the road / kerb line", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 29, "name": "dirty_pre_cast_surface", "category": "qaqc", "definition": "Slab or foundation surface not clean before concrete casting", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 30, "name": "dirty_finished_room", "category": "qaqc", "definition": "Finished room or space left dirty at handover", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 31, "name": "open_excavation_no_barrier", "category": "safety", "definition": "Open excavation without any perimeter barrier", "active": false, "weights_version": null, "min_examples_required": 30},
  {"id": 32, "name": "missing_handrail", "category": "safety", "definition": "Stair / platform edge missing handrail", "active": false, "weights_version": null, "min_examples_required": 30}
]
```

- [ ] **Step 4: Write the loader module**

```python
# app/blocks/safety_classes.py
"""Construction safety + QA/QC class registry. Source of truth: safety_classes.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

_REGISTRY_PATH = Path(__file__).parent / "safety_classes.json"


@dataclass(frozen=True)
class ClassEntry:
    id: int
    name: str
    category: Literal["safety", "qaqc"]
    definition: str
    active: bool
    weights_version: Optional[str]
    min_examples_required: int


def _parse_entries(raw: list) -> List[ClassEntry]:
    out = []
    for r in raw:
        out.append(ClassEntry(
            id=int(r["id"]),
            name=str(r["name"]),
            category=r["category"],
            definition=str(r.get("definition", "")),
            active=bool(r["active"]),
            weights_version=r.get("weights_version"),
            min_examples_required=int(r.get("min_examples_required", 30)),
        ))
    return out


def validate_registry(entries: List[ClassEntry]) -> None:
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for e in entries:
        if e.id in seen_ids:
            raise ValueError(f"duplicate id {e.id}")
        if e.name in seen_names:
            raise ValueError(f"duplicate name {e.name}")
        if e.category not in ("safety", "qaqc"):
            raise ValueError(f"invalid category {e.category} for {e.name}")
        if e.active and not e.weights_version:
            raise ValueError(f"class {e.name} is active but has no weights_version")
        seen_ids.add(e.id)
        seen_names.add(e.name)


def load_class_registry() -> List[ClassEntry]:
    raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = sorted(_parse_entries(raw), key=lambda e: e.id)
    validate_registry(entries)
    return entries


def get_active_classes() -> List[ClassEntry]:
    return [e for e in load_class_registry() if e.active]


def get_class_by_id(class_id: int) -> ClassEntry:
    for e in load_class_registry():
        if e.id == class_id:
            return e
    raise KeyError(f"class id {class_id} not in registry")


def get_class_by_name(name: str) -> ClassEntry:
    for e in load_class_registry():
        if e.name == name:
            return e
    raise KeyError(f"class name {name!r} not in registry")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_safety_classes.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add app/blocks/safety_classes.json app/blocks/safety_classes.py tests/test_safety_classes.py
git commit -m "feat(safety): add 33-class registry for safety + QA/QC detection"
```

---

### Task 0.2: Alembic migration 0006 (photo_chunks + photos tables)

**Files:**
- Create: `alembic/versions/0006_photo_chunks_and_photos.py`
- Test: `tests/test_migration_0006_photo_chunks.py`

**Background — why two new tables, not modifications:**
The original spec wrongly named `doc_index` as the chunk-level table. In reality:
- `doc_index` = per-project JSONB blob (PK `project_id`), NOT chunks.
- `chunks` = per-chunk RAG table with `text + embedding vector(256) NOT NULL + text_search tsvector` (generated, immutable).

Adding kind-discriminated photo rows to `chunks` would require either embeddings for photos (out of scope V1: spec says "no vector embedding for photos in V1") or making `embedding` nullable (schema change with broader downstream impact). The clean path is a separate `photo_chunks` table that the retriever queries alongside `chunks`.

**Interfaces:**
- Consumes: nothing (creates new tables)
- Produces:
  - New `photo_chunks` table: `chunk_id TEXT PRIMARY KEY` (= sha256 for V1, one row per photo), `project_id TEXT NULL` (populated in Phase 3 only), `sha256 TEXT NOT NULL`, `caption TEXT NOT NULL`, `photo_metadata` (JSONB on Postgres / TEXT on SQLite), `created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP`, UNIQUE(sha256)
  - New `photos` table: `sha256 TEXT PRIMARY KEY`, `content_type TEXT NOT NULL`, `size_bytes INTEGER NOT NULL`, `bytes BYTEA NOT NULL` (Postgres) / `BLOB NOT NULL` (SQLite), `uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP`
  - Migration is reversible (downgrade drops both tables)
  - Existing `chunks` and `doc_index` tables are NOT modified

(FTS5 virtual table for SQLite photo BM25 + GIN index for Postgres are added in Task 2.7's `vector_store` extension, not in this migration.)

**Test-infrastructure note (from the prior attempt at this task):** the project's `alembic/env.py` reads `DATABASE_URL` at import time from `app.core.db`, so passing `cfg.set_main_option("sqlalchemy.url", ...)` is ignored. Also, migration `0001` is Postgres-only (uses `CREATE EXTENSION vector` and `pg_catalog` queries) so cannot be applied to a fresh SQLite DB. Use a `monkeypatch.setenv("DATABASE_URL", ...)` + `importlib.reload(app.core.db)` + pre-seeded `alembic_version` table + `command.stamp(cfg, "0005")` approach. Look at any other migration tests in `tests/` for the project's actual pattern and mirror it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_migration_0006_photo_chunks.py
"""Round-trip migration 0006 on a fresh in-memory SQLite database."""
import importlib
import os
import tempfile

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


@pytest.fixture
def sqlite_url(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    url = f"sqlite:///{db}"
    monkeypatch.setenv("DATABASE_URL", url)
    import app.core.db as db_mod
    importlib.reload(db_mod)
    yield url
    importlib.reload(db_mod)


@pytest.fixture
def stamped_sqlite_engine(sqlite_url):
    """Create a SQLite DB with alembic_version stamped at 0005 (skipping the
    Postgres-only early migrations). Lets us test forward-from-0005 without
    needing pgvector."""
    engine = create_engine(sqlite_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"))
        conn.execute(text("INSERT INTO alembic_version VALUES ('0005')"))
    return engine


@pytest.fixture
def alembic_cfg():
    return Config("alembic.ini")


def test_upgrade_creates_photo_chunks_and_photos_tables(stamped_sqlite_engine, alembic_cfg):
    command.upgrade(alembic_cfg, "0006")
    insp = inspect(stamped_sqlite_engine)
    tables = set(insp.get_table_names())
    assert "photo_chunks" in tables
    assert "photos" in tables
    pc_cols = {c["name"] for c in insp.get_columns("photo_chunks")}
    assert pc_cols >= {"chunk_id", "project_id", "sha256", "caption", "photo_metadata", "created_at"}
    photos_cols = {c["name"] for c in insp.get_columns("photos")}
    assert photos_cols >= {"sha256", "content_type", "size_bytes", "bytes", "uploaded_at"}


def test_upgrade_does_not_modify_chunks_or_doc_index(stamped_sqlite_engine, alembic_cfg):
    # The point of this migration: avoid touching existing tables.
    # Insert sentinel rows; verify they survive untouched.
    with stamped_sqlite_engine.begin() as conn:
        conn.execute(text("CREATE TABLE doc_index (project_id TEXT PRIMARY KEY, index_json TEXT, updated_at TEXT)"))
        conn.execute(text("INSERT INTO doc_index VALUES ('p1', '{}', '2026-06-24')"))
    command.upgrade(alembic_cfg, "0006")
    insp = inspect(stamped_sqlite_engine)
    doc_index_cols = {c["name"] for c in insp.get_columns("doc_index")}
    assert "kind" not in doc_index_cols  # explicitly NOT added
    assert "photo_metadata" not in doc_index_cols
    with stamped_sqlite_engine.begin() as conn:
        row = conn.execute(text("SELECT project_id FROM doc_index WHERE project_id = 'p1'")).fetchone()
    assert row is not None  # untouched


def test_downgrade_drops_both_new_tables(stamped_sqlite_engine, alembic_cfg):
    command.upgrade(alembic_cfg, "0006")
    command.downgrade(alembic_cfg, "0005")
    insp = inspect(stamped_sqlite_engine)
    tables = set(insp.get_table_names())
    assert "photo_chunks" not in tables
    assert "photos" not in tables
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_migration_0006_doc_index_photo.py -v`
Expected: alembic error — revision 0006 not found

- [ ] **Step 3: Inspect prior migration to learn dialect branching pattern**

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; print(Path('alembic/versions/0005_projects_origin.py').read_text())"`

Confirm: the file uses `op.get_bind().dialect.name` to branch SQLite vs Postgres.

- [ ] **Step 4: Write the migration**

```python
# alembic/versions/0006_photo_chunks_and_photos.py
"""photo_chunks + photos tables

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-24
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.create_table(
            "photo_chunks",
            sa.Column("chunk_id", sa.Text(), primary_key=True),
            sa.Column("project_id", sa.Text(), nullable=True),
            sa.Column("sha256", sa.Text(), nullable=False),
            sa.Column("caption", sa.Text(), nullable=False),
            sa.Column("photo_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("sha256", name="uq_photo_chunks_sha256"),
        )
        op.create_table(
            "photos",
            sa.Column("sha256", sa.Text(), primary_key=True),
            sa.Column("content_type", sa.Text(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("bytes", postgresql.BYTEA(), nullable=False),
            sa.Column("uploaded_at", sa.TIMESTAMP(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
    else:
        op.create_table(
            "photo_chunks",
            sa.Column("chunk_id", sa.Text(), primary_key=True),
            sa.Column("project_id", sa.Text(), nullable=True),
            sa.Column("sha256", sa.Text(), nullable=False),
            sa.Column("caption", sa.Text(), nullable=False),
            sa.Column("photo_metadata", sa.Text(), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("sha256", name="uq_photo_chunks_sha256"),
        )
        op.create_table(
            "photos",
            sa.Column("sha256", sa.Text(), primary_key=True),
            sa.Column("content_type", sa.Text(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("bytes", sa.LargeBinary(), nullable=False),
            sa.Column("uploaded_at", sa.TIMESTAMP(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )


def downgrade() -> None:
    op.drop_table("photos")
    op.drop_table("photo_chunks")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_migration_0006_doc_index_photo.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add alembic/versions/0006_photo_chunks_and_photos.py tests/test_migration_0006_photo_chunks.py
git commit -m "feat(db): migration 0006 creates photo_chunks + photos tables"
```

---

### Task 0.3: Grounding DINO survey script

**Files:**
- Create: `scripts/survey_photo_corpus.py`
- Test: `tests/test_survey_photo_corpus.py`

**Interfaces:**
- Consumes: `load_class_registry()` from Task 0.1
- Produces:
  - Command-line entry point: `python scripts/survey_photo_corpus.py <folder> <output_json>`
  - JSON shape: `{"folder": str, "total_images": int, "per_class": {class_name: {"detections": int, "photos_with_at_least_one": int}}, "per_photo": [{"filename": str, "detections_by_class": {class_name: int}}], "model": "IDEA-Research/grounding-dino-tiny", "box_threshold": 0.35, "text_threshold": 0.25}`
  - Detector wrapper: `detect_with_dino(image_path: Path, class_names: List[str]) -> List[Dict]` returns `[{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_survey_photo_corpus.py
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from scripts.survey_photo_corpus import survey_folder


@pytest.fixture
def fixture_folder(tmp_path: Path) -> Path:
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(tmp_path / name)
    return tmp_path


def _fake_dino(image_path, class_names):
    if image_path.name == "a.jpg":
        return [
            {"class": "no_hardhat", "confidence": 0.7, "bbox": [0, 0, 32, 32]},
            {"class": "concrete_crack", "confidence": 0.6, "bbox": [16, 16, 48, 48]},
        ]
    return [{"class": "no_hardhat", "confidence": 0.5, "bbox": [0, 0, 32, 32]}]


def test_survey_counts_per_class_and_per_photo(fixture_folder, tmp_path):
    out_json = tmp_path / "out.json"
    with patch("scripts.survey_photo_corpus.detect_with_dino", side_effect=_fake_dino):
        report = survey_folder(fixture_folder, out_json)

    assert report["total_images"] == 2
    assert report["per_class"]["no_hardhat"]["detections"] == 2
    assert report["per_class"]["no_hardhat"]["photos_with_at_least_one"] == 2
    assert report["per_class"]["concrete_crack"]["detections"] == 1
    assert report["per_class"]["concrete_crack"]["photos_with_at_least_one"] == 1

    persisted = json.loads(out_json.read_text())
    assert persisted == report
    assert len(persisted["per_photo"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_survey_photo_corpus.py -v`
Expected: ImportError — scripts.survey_photo_corpus not found

- [ ] **Step 3: Write the script**

```python
# scripts/survey_photo_corpus.py
"""Phase 0 — Grounding DINO survey of a folder of photos against the safety/QA-QC class registry.

Usage:
    python scripts/survey_photo_corpus.py <folder> <output_json>

Outputs per-class detection counts so the operator can decide which classes
have enough examples to ship as V1 active. Heavy dep (transformers + torch
+ ~1 GB model download) is only imported when the script actually runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.blocks.safety_classes import load_class_registry  # noqa: E402

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_BOX_THRESHOLD = 0.35
_TEXT_THRESHOLD = 0.25


def detect_with_dino(image_path: Path, class_names: List[str]) -> List[Dict]:
    """Run Grounding DINO on one image with the given open-vocab class prompts.
    Returns [{class, confidence, bbox: [x1,y1,x2,y2]}, ...]."""
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch

    if not hasattr(detect_with_dino, "_cache"):
        proc = AutoProcessor.from_pretrained(_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID).eval()
        detect_with_dino._cache = (proc, model)
    proc, model = detect_with_dino._cache

    image = Image.open(image_path).convert("RGB")
    prompt = ". ".join(name.replace("_", " ") for name in class_names) + "."
    inputs = proc(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=_BOX_THRESHOLD, text_threshold=_TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    name_lookup = {name.replace("_", " "): name for name in class_names}
    out = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        canon = name_lookup.get(label, label.replace(" ", "_"))
        out.append({"class": canon, "confidence": float(score), "bbox": [float(x) for x in box.tolist()]})
    return out


def survey_folder(folder: Path, output_json: Path) -> Dict:
    class_names = [c.name for c in load_class_registry()]
    per_class = {n: {"detections": 0, "photos_with_at_least_one": 0} for n in class_names}
    per_photo: List[Dict] = []

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    for img in images:
        detections = detect_with_dino(img, class_names)
        by_class: Dict[str, int] = {}
        for d in detections:
            by_class[d["class"]] = by_class.get(d["class"], 0) + 1
        for name, count in by_class.items():
            per_class[name]["detections"] += count
            per_class[name]["photos_with_at_least_one"] += 1
        per_photo.append({"filename": img.name, "detections_by_class": by_class})

    report = {
        "folder": str(folder),
        "total_images": len(images),
        "model": _MODEL_ID,
        "box_threshold": _BOX_THRESHOLD,
        "text_threshold": _TEXT_THRESHOLD,
        "per_class": per_class,
        "per_photo": per_photo,
    }
    output_json.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("output_json", type=Path)
    args = p.parse_args()
    survey_folder(args.folder, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_survey_photo_corpus.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/survey_photo_corpus.py tests/test_survey_photo_corpus.py
git commit -m "feat(safety): Grounding DINO survey script for Phase 0 class lock"
```

---

### Task 0.4 [OPERATOR-DRIVEN]: Run survey and lock V1 active class list

**This task is operator-driven. The subagent driver MUST STOP and surface this checkpoint to the operator rather than auto-running.**

**Operator actions:**

1. Extract the zip to a working folder:
   ```bash
   .venv/Scripts/python.exe -c "import zipfile; zipfile.ZipFile(r'X:/Example Drive/construction-3-001.zip').extractall(r'data/training/raw_photos')"
   ```

2. Install Grounding DINO deps in the local venv:
   ```bash
   .venv/Scripts/python.exe -m pip install "transformers>=4.40" "huggingface_hub>=0.23" torch
   ```

3. Run the survey:
   ```bash
   .venv/Scripts/python.exe scripts/survey_photo_corpus.py data/training/raw_photos data/training/photo_survey.json
   ```

4. Review `data/training/photo_survey.json`. For each class in `per_class`, check `photos_with_at_least_one`:
   - **≥30** → class stays `active: true`
   - **<30** → flip to `active: false` and set `weights_version: null`

   **Dataset honesty:** 30 examples per class is the activation FLOOR for V1, not a quality target. Industry rule-of-thumb for a robust YOLO fine-tune is 150–200 examples per class, and the ship gate of mAP@0.5 ≥ 0.3 is barely-functional. V1's real deliverable is the **end-to-end pipeline** (label → train → infer → export → RAG-serve), not a production-grade detector. Expect noticeable false positives and missed detections; Phase 3's active-learning loop is what closes the quality gap once the platform's upload pathway feeds more labelled data over time.

5. Edit `app/blocks/safety_classes.json` accordingly. Re-run `pytest tests/test_safety_classes.py` to confirm the registry still validates.

6. Commit:
   ```bash
   git add app/blocks/safety_classes.json data/training/photo_survey.json
   git commit -m "data(safety): lock V1 active class list from survey of construction-3-001"
   ```

7. Tell the next subagent: V1 active class list is locked, proceed to Phase 1.

---

## Phase 1 — Training Pipeline (4 tasks)

### Task 1.1: Grounding DINO pre-label script

**Files:**
- Create: `scripts/prelabel_with_dino.py`
- Test: `tests/test_prelabel_with_dino.py`

**Interfaces:**
- Consumes: `get_active_classes()` from Task 0.1; `detect_with_dino()` from Task 0.3
- Produces:
  - Command-line: `python scripts/prelabel_with_dino.py <folder> <output_dir>`
  - Output: one Label Studio JSON file at `<output_dir>/tasks.json` with shape:
    ```json
    [
      {
        "data": {"image": "/data/raw_photos/IMG-...jpg"},
        "predictions": [{
          "model_version": "grounding-dino-tiny",
          "result": [{
            "type": "rectanglelabels",
            "value": {"x": pct, "y": pct, "width": pct, "height": pct, "rectanglelabels": ["class_name"]},
            "from_name": "label", "to_name": "image", "image_rotation": 0,
            "original_width": int, "original_height": int,
            "score": float
          }]
        }]
      }
    ]
    ```
  - bbox coords converted from absolute pixels to percentages (Label Studio convention)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prelabel_with_dino.py
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from scripts.prelabel_with_dino import prelabel_folder


def _fake_dino(image_path, class_names):
    return [{"class": "no_hardhat", "confidence": 0.8, "bbox": [10, 20, 50, 60]}]


def _fake_active():
    from app.blocks.safety_classes import ClassEntry
    return [ClassEntry(id=0, name="no_hardhat", category="safety", definition="", active=True, weights_version="v1", min_examples_required=30)]


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    Image.new("RGB", (100, 100), color=(50, 50, 50)).save(tmp_path / "a.jpg")
    return tmp_path


def test_prelabel_emits_label_studio_json(folder, tmp_path):
    out_dir = tmp_path / "out"
    with patch("scripts.prelabel_with_dino.detect_with_dino", side_effect=_fake_dino), \
         patch("scripts.prelabel_with_dino.get_active_classes", side_effect=_fake_active):
        prelabel_folder(folder, out_dir)

    tasks = json.loads((out_dir / "tasks.json").read_text())
    assert len(tasks) == 1
    pred = tasks[0]["predictions"][0]["result"][0]
    assert pred["value"]["rectanglelabels"] == ["no_hardhat"]
    assert pred["value"]["x"] == pytest.approx(10.0)
    assert pred["value"]["y"] == pytest.approx(20.0)
    assert pred["value"]["width"] == pytest.approx(40.0)
    assert pred["value"]["height"] == pytest.approx(40.0)
    assert pred["original_width"] == 100
    assert pred["original_height"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prelabel_with_dino.py -v`
Expected: ImportError

- [ ] **Step 3: Write the script**

```python
# scripts/prelabel_with_dino.py
"""Phase 1a — Grounding DINO pre-label pass.

Usage:
    python scripts/prelabel_with_dino.py <folder> <output_dir>

Reads V1 active classes from the registry, runs Grounding DINO on each
image, and emits a Label Studio import file (tasks.json) at output_dir.
The operator then opens this file in Label Studio, corrects boxes, and
exports YOLO format for training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.blocks.safety_classes import get_active_classes  # noqa: E402
from scripts.survey_photo_corpus import detect_with_dino  # noqa: E402

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _to_label_studio_box(bbox_xyxy: List[float], width: int, height: int, label: str, score: float) -> Dict:
    x1, y1, x2, y2 = bbox_xyxy
    return {
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "image_rotation": 0,
        "original_width": width,
        "original_height": height,
        "value": {
            "x": x1 * 100.0 / width,
            "y": y1 * 100.0 / height,
            "width": (x2 - x1) * 100.0 / width,
            "height": (y2 - y1) * 100.0 / height,
            "rectanglelabels": [label],
        },
        "score": float(score),
    }


def prelabel_folder(folder: Path, output_dir: Path) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    active = get_active_classes()
    class_names = [c.name for c in active]

    tasks: List[Dict] = []
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    for img in images:
        with Image.open(img) as pil_img:
            w, h = pil_img.size
        detections = detect_with_dino(img, class_names)
        results = [
            _to_label_studio_box(d["bbox"], w, h, d["class"], d["confidence"])
            for d in detections
        ]
        tasks.append({
            "data": {"image": f"/data/raw_photos/{img.name}"},
            "predictions": [{"model_version": "grounding-dino-tiny", "result": results}],
        })

    (output_dir / "tasks.json").write_text(json.dumps(tasks, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    prelabel_folder(args.folder, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prelabel_with_dino.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/prelabel_with_dino.py tests/test_prelabel_with_dino.py
git commit -m "feat(safety): Grounding DINO pre-label script emits Label Studio JSON"
```

---

### Task 1.2 [OPERATOR-DRIVEN]: Label 208 photos in Label Studio

**This task is operator-driven. The subagent driver MUST STOP and surface this checkpoint.**

**Operator actions:**

1. Install Label Studio in a separate venv (it has heavy deps that should NOT pollute `.venv`):
   ```bash
   py -3.12 -m venv .venv-labelstudio
   .venv-labelstudio/Scripts/python.exe -m pip install label-studio
   ```

2. Run pre-label on the corpus:
   ```bash
   .venv/Scripts/python.exe scripts/prelabel_with_dino.py data/training/raw_photos data/training/labels_draft
   ```

3. Launch Label Studio:
   ```bash
   .venv-labelstudio/Scripts/python.exe -m label_studio start --port 8080
   ```
   Open `http://localhost:8080`, create a project, import `data/training/labels_draft/tasks.json`. Configure local-files serving for `data/training/raw_photos/`.

4. For every task: review the AI-drafted boxes. Add missing boxes. Delete wrong boxes. Adjust box positions. Class names are pre-populated; only edit if the AI chose wrong.

5. Export project as YOLO format to `data/training/labels_final/`. Structure:
   ```
   data/training/labels_final/
     images/{train,val}/*.jpg
     labels/{train,val}/*.txt
     classes.txt
     data.yaml
   ```
   Label Studio's "Export → YOLO" handles the 80/20 train/val split. Verify `data.yaml` lists exactly the V1 active classes.

6. Commit the dataset (or document it as gitignored if too large):
   ```bash
   git add data/training/labels_final/ -f
   git commit -m "data(safety): labelled V1 training set from construction-3-001"
   ```
   If labels_final exceeds 50 MB, add `data/training/labels_final/images/` to `.gitignore` and commit only labels + classes.txt + data.yaml.

7. Tell the next subagent: training set ready at `data/training/labels_final/`.

---

### Task 1.3: Training script

**Files:**
- Create: `scripts/train_safety_qaqc.py`
- Test: `tests/test_train_safety_qaqc.py`

**Interfaces:**
- Consumes: `get_active_classes()` from Task 0.1; YOLO format dataset from Task 1.2
- Produces:
  - Command-line: `python scripts/train_safety_qaqc.py --data data/training/labels_final/data.yaml --version 1 [--force-low-quality] [--epochs 50] [--imgsz 640]`
  - Outputs `data/models/safety_qaqc_v{N}.pt` + `data/models/safety_qaqc_v{N}_classmap.json` + `data/training/eval_v{N}.json`
  - `eval_v{N}.json` shape: `{"version": int, "model_grade": "shippable"|"experimental"|"failed", "mAP_0.5": float, "per_class": {class_name: {"precision": float, "recall": float, "mAP_0.5": float}}, "labels_per_active_class": {class_name: int}}`
  - `_classmap.json` shape: `{"yolo_class_id_to_registry_id": {0: 3, 1: 0, ...}, "registry_id_to_yolo_class_id": {3: 0, 0: 1, ...}}`
  - Validation gate: raises `SystemExit(1)` if any active class has <30 labels OR mAP@0.5 < 0.3, unless `--force-low-quality` set

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_safety_qaqc.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.train_safety_qaqc import train_and_eval, count_labels_per_class


def _fake_active():
    from app.blocks.safety_classes import ClassEntry
    return [
        ClassEntry(id=0, name="no_hardhat", category="safety", definition="", active=True, weights_version="v1", min_examples_required=30),
        ClassEntry(id=3, name="concrete_crack", category="qaqc", definition="", active=True, weights_version="v1", min_examples_required=30),
    ]


def _make_dataset(tmp_path: Path, label_counts: dict) -> Path:
    base = tmp_path / "dataset"
    (base / "images" / "train").mkdir(parents=True)
    (base / "labels" / "train").mkdir(parents=True)
    (base / "images" / "val").mkdir(parents=True)
    (base / "labels" / "val").mkdir(parents=True)
    classes = list(label_counts.keys())
    (base / "data.yaml").write_text(
        f"path: {base}\ntrain: images/train\nval: images/val\nnames: {classes}\n"
    )
    file_idx = 0
    for cls_idx, (cls_name, count) in enumerate(label_counts.items()):
        for _ in range(count):
            img = base / "images" / "train" / f"img_{file_idx}.jpg"
            img.write_bytes(b"fake jpg")
            lbl = base / "labels" / "train" / f"img_{file_idx}.txt"
            lbl.write_text(f"{cls_idx} 0.5 0.5 0.1 0.1\n")
            file_idx += 1
    return base


def test_count_labels_per_class(tmp_path):
    ds = _make_dataset(tmp_path, {"no_hardhat": 35, "concrete_crack": 40})
    counts = count_labels_per_class(ds / "data.yaml")
    assert counts == {"no_hardhat": 35, "concrete_crack": 40}


def test_aborts_when_class_under_budget(tmp_path):
    ds = _make_dataset(tmp_path, {"no_hardhat": 35, "concrete_crack": 10})
    with patch("scripts.train_safety_qaqc.get_active_classes", side_effect=_fake_active):
        with pytest.raises(SystemExit) as exc:
            train_and_eval(ds / "data.yaml", version=1, out_dir=tmp_path / "models", epochs=1, imgsz=64)
    assert "concrete_crack" in str(exc.value)


def test_marks_experimental_when_low_mAP(tmp_path):
    ds = _make_dataset(tmp_path, {"no_hardhat": 35, "concrete_crack": 40})
    fake_yolo = MagicMock()
    fake_yolo.train.return_value = None
    fake_yolo.val.return_value = MagicMock(
        box=MagicMock(map50=0.2,
                      maps=[0.2, 0.2],
                      p=[0.3, 0.3], r=[0.25, 0.25]),
        names={0: "no_hardhat", 1: "concrete_crack"},
    )
    fake_yolo.save = lambda p: Path(p).write_bytes(b"fake")
    with patch("scripts.train_safety_qaqc.get_active_classes", side_effect=_fake_active), \
         patch("scripts.train_safety_qaqc.YOLO", return_value=fake_yolo):
        with pytest.raises(SystemExit):
            train_and_eval(ds / "data.yaml", version=1, out_dir=tmp_path / "models", epochs=1, imgsz=64)


def test_ships_when_above_threshold(tmp_path):
    ds = _make_dataset(tmp_path, {"no_hardhat": 35, "concrete_crack": 40})
    fake_yolo = MagicMock()
    fake_yolo.train.return_value = None
    fake_yolo.val.return_value = MagicMock(
        box=MagicMock(map50=0.55,
                      maps=[0.55, 0.55],
                      p=[0.6, 0.6], r=[0.5, 0.5]),
        names={0: "no_hardhat", 1: "concrete_crack"},
    )
    fake_yolo.save = lambda p: Path(p).write_bytes(b"fake")
    with patch("scripts.train_safety_qaqc.get_active_classes", side_effect=_fake_active), \
         patch("scripts.train_safety_qaqc.YOLO", return_value=fake_yolo):
        train_and_eval(ds / "data.yaml", version=1, out_dir=tmp_path / "models", epochs=1, imgsz=64)
    eval_json = json.loads((tmp_path / "training" / "eval_v1.json").read_text())
    assert eval_json["model_grade"] == "shippable"
    assert (tmp_path / "models" / "safety_qaqc_v1.pt").exists()
    assert (tmp_path / "models" / "safety_qaqc_v1_classmap.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_safety_qaqc.py -v`
Expected: ImportError

- [ ] **Step 3: Write the script**

```python
# scripts/train_safety_qaqc.py
"""Phase 1c — Fine-tune YOLOv8n on the safety + QA/QC dataset.

Usage:
    python scripts/train_safety_qaqc.py --data data/training/labels_final/data.yaml --version 1

Refuses to ship if any active class has <30 labels OR validation mAP@0.5 < 0.3.
Override the mAP gate with --force-low-quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.blocks.safety_classes import get_active_classes  # noqa: E402

try:
    from ultralytics import YOLO  # noqa: E402
except ImportError:
    YOLO = None  # tests patch this; runtime use must have ultralytics installed


_MIN_EXAMPLES_PER_CLASS = 30
_MIN_mAP_50 = 0.3


def count_labels_per_class(data_yaml: Path) -> Dict[str, int]:
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg["names"]
    root = data_yaml.parent
    label_dirs = [root / "labels" / "train", root / "labels" / "val"]
    counter: Counter = Counter()
    for d in label_dirs:
        if not d.is_dir():
            continue
        for txt in d.glob("*.txt"):
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls_idx = int(parts[0])
                counter[names[cls_idx]] += 1
    return dict(counter)


def train_and_eval(
    data_yaml: Path,
    version: int,
    out_dir: Path,
    epochs: int = 50,
    imgsz: int = 640,
    force_low_quality: bool = False,
) -> Dict:
    if YOLO is None:
        raise RuntimeError("ultralytics not installed; run pip install -r requirements-cv.txt")

    active = get_active_classes()
    cfg = yaml.safe_load(data_yaml.read_text())
    yolo_names = cfg["names"]

    classmap = {
        "yolo_class_id_to_registry_id": {
            i: next(a.id for a in active if a.name == n) for i, n in enumerate(yolo_names)
        },
        "registry_id_to_yolo_class_id": {
            next(a.id for a in active if a.name == n): i for i, n in enumerate(yolo_names)
        },
    }

    counts = count_labels_per_class(data_yaml)
    under = [n for n in yolo_names if counts.get(n, 0) < _MIN_EXAMPLES_PER_CLASS]
    if under:
        raise SystemExit(f"classes under {_MIN_EXAMPLES_PER_CLASS}-example budget: {under}")

    out_dir.mkdir(parents=True, exist_ok=True)
    training_dir = out_dir.parent / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, seed=42, verbose=False)
    metrics = model.val(data=str(data_yaml))

    per_class = {}
    for i, name in enumerate(yolo_names):
        per_class[name] = {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "mAP_0.5": float(metrics.box.maps[i]),
        }

    overall_map = float(metrics.box.map50)
    if overall_map < _MIN_mAP_50 and not force_low_quality:
        eval_json = {
            "version": version, "model_grade": "experimental",
            "mAP_0.5": overall_map, "per_class": per_class,
            "labels_per_active_class": counts,
            "reject_reason": f"mAP@0.5 {overall_map:.3f} below floor {_MIN_mAP_50}",
        }
        (training_dir / f"eval_v{version}.json").write_text(json.dumps(eval_json, indent=2))
        raise SystemExit(f"model not shippable: mAP@0.5={overall_map:.3f}; use --force-low-quality to override")

    weights_path = out_dir / f"safety_qaqc_v{version}.pt"
    model.save(str(weights_path))
    (out_dir / f"safety_qaqc_v{version}_classmap.json").write_text(json.dumps(classmap, indent=2))

    eval_json = {
        "version": version,
        "model_grade": "shippable" if overall_map >= _MIN_mAP_50 else "experimental",
        "mAP_0.5": overall_map,
        "per_class": per_class,
        "labels_per_active_class": counts,
    }
    (training_dir / f"eval_v{version}.json").write_text(json.dumps(eval_json, indent=2))
    return eval_json


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("data/models"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--force-low-quality", action="store_true")
    args = p.parse_args()
    train_and_eval(args.data, args.version, args.out_dir, args.epochs, args.imgsz, args.force_low_quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_safety_qaqc.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/train_safety_qaqc.py tests/test_train_safety_qaqc.py
git commit -m "feat(safety): YOLOv8 training script with class-budget + mAP validation gates"
```

---

### Task 1.4 [OPERATOR-DRIVEN]: Train safety_qaqc_v1.pt

**This task is operator-driven. The subagent driver MUST STOP and surface this checkpoint.**

**Operator actions:**

1. Confirm CUDA is available:
   ```bash
   .venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
   ```

2. Train:
   ```bash
   .venv/Scripts/python.exe scripts/train_safety_qaqc.py --data data/training/labels_final/data.yaml --version 1
   ```
   Expect 10–30 min on a consumer GPU at 50 epochs.

3. Review `data/training/eval_v1.json`. Confirm:
   - `model_grade == "shippable"`
   - Each `per_class[X]["mAP_0.5"] >= 0.3`
   - `labels_per_active_class` matches expectations

4. If a class is unshippable: flip its `active: false` in `safety_classes.json`, re-export YOLO dataset from Label Studio with the reduced class list, re-train.

5. Commit the artefacts (or document gitignore if .pt too large):
   ```bash
   git add data/models/safety_qaqc_v1.pt data/models/safety_qaqc_v1_classmap.json data/training/eval_v1.json
   git commit -m "data(safety): trained V1 safety_qaqc weights"
   ```
   If `.pt` exceeds 50 MB, use git LFS or add `data/models/*.pt` to `.gitignore` and document where the weights live (Drive, S3, etc.) — but `_classmap.json` and `eval_v1.json` MUST commit.

6. Tell the next subagent: trained weights at `data/models/safety_qaqc_v1.pt`, classmap at `_classmap.json`. Proceed to Phase 2.

---

## Phase 2 — Inference + RAG Integration (9 tasks)

### Task 2.1: Safety detector block

**Files:**
- Create: `app/blocks/safety_detector.py`
- Test: `tests/test_safety_detector.py`

**Interfaces:**
- Consumes: `get_class_by_id()` from Task 0.1; weights + classmap from Task 1.4
- Produces:
  - `SafetyDetector(weights_path: Path, classmap_path: Path)` — loads model on init
  - `SafetyDetector.detect(file_path: Path, conf_threshold: float = 0.4) -> List[Dict]` returns `[{class_id: int (registry id), class: str, confidence: float, bbox: [x1,y1,x2,y2], category: "safety"|"qaqc"}, ...]`
  - Module-level `default_detector() -> Optional[SafetyDetector]` — reads `SAFETY_DETECTOR_WEIGHTS` env var, returns None if unset or weights missing (lets callers gracefully no-op)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_safety_detector.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def weights_and_classmap(tmp_path: Path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fake")
    classmap = tmp_path / "classmap.json"
    classmap.write_text(json.dumps({
        "yolo_class_id_to_registry_id": {"0": 0, "1": 3},
        "registry_id_to_yolo_class_id": {"0": 0, "3": 1},
    }))
    return weights, classmap


def test_detect_returns_registry_ids_and_categories(weights_and_classmap, tmp_path):
    weights, classmap = weights_and_classmap
    img = tmp_path / "img.jpg"
    img.write_bytes(b"fake")

    fake_result = MagicMock()
    fake_result.boxes.cls.tolist.return_value = [0, 1]
    fake_result.boxes.conf.tolist.return_value = [0.85, 0.72]
    fake_result.boxes.xyxy.tolist.return_value = [[10, 20, 30, 40], [50, 60, 70, 80]]
    fake_model = MagicMock(return_value=[fake_result])

    from app.blocks.safety_detector import SafetyDetector
    with patch("app.blocks.safety_detector.YOLO", return_value=fake_model):
        det = SafetyDetector(weights, classmap)
        out = det.detect(img, conf_threshold=0.4)

    assert len(out) == 2
    assert out[0] == {
        "class_id": 0, "class": "no_hardhat", "category": "safety",
        "confidence": 0.85, "bbox": [10.0, 20.0, 30.0, 40.0],
    }
    assert out[1]["class"] == "concrete_crack"
    assert out[1]["category"] == "qaqc"


def test_detect_filters_below_threshold(weights_and_classmap, tmp_path):
    weights, classmap = weights_and_classmap
    img = tmp_path / "img.jpg"
    img.write_bytes(b"fake")

    fake_result = MagicMock()
    fake_result.boxes.cls.tolist.return_value = [0, 1]
    fake_result.boxes.conf.tolist.return_value = [0.85, 0.35]
    fake_result.boxes.xyxy.tolist.return_value = [[10, 20, 30, 40], [50, 60, 70, 80]]
    fake_model = MagicMock(return_value=[fake_result])

    from app.blocks.safety_detector import SafetyDetector
    with patch("app.blocks.safety_detector.YOLO", return_value=fake_model):
        det = SafetyDetector(weights, classmap)
        out = det.detect(img, conf_threshold=0.4)

    assert len(out) == 1
    assert out[0]["class_id"] == 0


def test_default_detector_returns_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("SAFETY_DETECTOR_WEIGHTS", raising=False)
    from app.blocks.safety_detector import default_detector
    assert default_detector() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_safety_detector.py -v`
Expected: ImportError

- [ ] **Step 3: Write the module**

```python
# app/blocks/safety_detector.py
"""Fine-tuned YOLOv8 safety + QA/QC detector.

Loads weights produced by scripts/train_safety_qaqc.py. The class-id translation
layer maps the YOLO model's renumbered 0..N-1 IDs back to the stable registry IDs
via the classmap JSON saved alongside the weights.

Used by scripts/infer_photo_metadata.py during batch PC inference. Registered in
the block registry for Phase 3 runtime use; in V1, runtime callers receive None
from default_detector() unless SAFETY_DETECTOR_WEIGHTS is explicitly set.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from app.blocks.safety_classes import get_class_by_id

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


class SafetyDetector:
    def __init__(self, weights_path: Path, classmap_path: Path) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics not installed")
        self._model = YOLO(str(weights_path))
        cm = json.loads(Path(classmap_path).read_text())
        self._yolo_to_registry = {int(k): int(v) for k, v in cm["yolo_class_id_to_registry_id"].items()}

    def detect(self, file_path: Path, conf_threshold: float = 0.4) -> List[Dict]:
        results = self._model(str(file_path))
        if not results:
            return []
        r = results[0]
        cls_ids = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        boxes = r.boxes.xyxy.tolist()

        out: List[Dict] = []
        for yolo_cls, conf, box in zip(cls_ids, confs, boxes):
            if conf < conf_threshold:
                continue
            registry_id = self._yolo_to_registry[int(yolo_cls)]
            entry = get_class_by_id(registry_id)
            out.append({
                "class_id": registry_id,
                "class": entry.name,
                "category": entry.category,
                "confidence": float(conf),
                "bbox": [float(x) for x in box],
            })
        return out


def default_detector() -> Optional[SafetyDetector]:
    weights_env = os.getenv("SAFETY_DETECTOR_WEIGHTS")
    if not weights_env:
        return None
    weights = Path(weights_env)
    classmap = weights.with_name(weights.stem + "_classmap.json")
    if not weights.is_file() or not classmap.is_file():
        logger.warning("SAFETY_DETECTOR_WEIGHTS set but weights or classmap missing")
        return None
    try:
        return SafetyDetector(weights, classmap)
    except Exception:
        logger.exception("failed to load SafetyDetector")
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_safety_detector.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/blocks/safety_detector.py tests/test_safety_detector.py
git commit -m "feat(safety): SafetyDetector block wraps fine-tuned YOLOv8 weights"
```

---

### Task 2.2: Image block — `safety_qaqc` mode

**Files:**
- Modify: `app/blocks/image.py` (ImageBlock.execute — add mode dispatch)
- Test: `tests/test_image_block_safety_qaqc_mode.py`

**Interfaces:**
- Consumes: `SafetyDetector.detect()` from Task 2.1; existing `_pil_metadata` / `_tesseract_ocr` / `_yolo_detect` helpers in `image.py`
- Produces:
  - `ImageBlock.execute({"file_path": str}, {"mode": "safety_qaqc"})` returns `{"status": "success", "result": {"pil": ..., "ocr_text": str, "coco_objects": [...], "safety_qaqc": [...], "provider": "pil+tesseract+yolo+safety_qaqc"}}`
  - When no detector available: `safety_qaqc` field is `[]` and `provider` omits `+safety_qaqc`
  - Backward-compatible: any other mode (or no mode) keeps current behavior

- [ ] **Step 1: Write the failing test**

```python
# tests/test_image_block_safety_qaqc_mode.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.blocks.image import ImageBlock


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (100, 100), color=(120, 60, 30)).save(p)
    return p


@pytest.mark.asyncio
async def test_safety_qaqc_mode_includes_detector_output(image_path):
    fake = MagicMock()
    fake.detect.return_value = [{"class_id": 0, "class": "no_hardhat", "category": "safety", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    with patch("app.blocks.image.default_detector", return_value=fake):
        block = ImageBlock()
        result = await block.execute({"file_path": str(image_path)}, {"mode": "safety_qaqc"})

    assert result["status"] == "success"
    body = result["result"]
    assert body["pil"]["width"] == 100
    assert body["safety_qaqc"] == [{"class_id": 0, "class": "no_hardhat", "category": "safety", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    assert "safety_qaqc" in body["provider"]


@pytest.mark.asyncio
async def test_safety_qaqc_mode_no_detector(image_path):
    with patch("app.blocks.image.default_detector", return_value=None):
        block = ImageBlock()
        result = await block.execute({"file_path": str(image_path)}, {"mode": "safety_qaqc"})

    body = result["result"]
    assert body["safety_qaqc"] == []
    assert "safety_qaqc" not in body["provider"]


@pytest.mark.asyncio
async def test_default_mode_unchanged(image_path):
    block = ImageBlock()
    result = await block.execute({"file_path": str(image_path)}, {})
    assert result["status"] == "success"
    assert "safety_qaqc" not in result["result"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_image_block_safety_qaqc_mode.py -v`
Expected: AttributeError or AssertionError

- [ ] **Step 3: Modify `app/blocks/image.py`**

At the top of the file:
```python
from app.blocks.safety_detector import default_detector
```

In `ImageBlock.execute()`, after the existing PIL / OCR / YOLO path produces its result body, add this branch BEFORE the return:

```python
        mode = (params or {}).get("mode")
        if mode == "safety_qaqc":
            detector = default_detector()
            if detector is not None:
                detections = detector.detect(Path(file_path))
                body["safety_qaqc"] = detections
                body["provider"] = body.get("provider", "") + "+safety_qaqc"
            else:
                body["safety_qaqc"] = []
```

(Use Read tool first to find the precise insertion point in the existing `execute` method, then Edit to wire it in. The body dict must already contain `pil`, `ocr_text`, `coco_objects` from the default path.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_image_block_safety_qaqc_mode.py -v`
Expected: 3 passed

- [ ] **Step 5: Run existing image block tests to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -k "image" -v`
Expected: no new failures vs baseline

- [ ] **Step 6: Commit**

```bash
git add app/blocks/image.py tests/test_image_block_safety_qaqc_mode.py
git commit -m "feat(image): add safety_qaqc mode that delegates to SafetyDetector"
```

---

### Task 2.3: Construction container — compose YOLO output with hazard/defect verdict methods

**PRE-TASK GUARDRAIL (do this before Step 1):** As of 2026-06-24 the working tree had uncommitted changes in `app/containers/construction/__init__.py` and `documents.py` from a parallel Claude session: an `image_path` → `file_path` keyword rename in 5 image-block call sites, plus a `jetson_dispatch` stub, plus an accompanying test at `tests/test_construction_photo_detection.py`. Those changes are a genuine bug fix that this task BUILDS ON.

Run `git status --short app/containers/construction/ tests/test_construction_photo_detection.py` first:
- If the files show as `M` (modified) or `??` (untracked) — the parallel session's work is still uncommitted. **STOP and surface this to the operator.** Ask: "Is the parallel-session work on `image_path` → `file_path` rename + jetson_dispatch + test_construction_photo_detection.py ready to commit, or should I proceed with the implementer subagent assuming I'll commit it as part of this task?"
- If clean — proceed; the rename has already landed.

**Files:**
- Modify: `app/containers/construction/__init__.py` (add `_classes_to_hazards` + `_classes_to_defects`)
- Modify: `app/containers/construction/documents.py` (`safety_compliance_audit` + `qa_qc_inspection` use the new helpers when `safety_qaqc` output present)
- Test: `tests/test_construction_container_yolo_compose.py`

**Interfaces:**
- Consumes: `safety_qaqc` field in image-block output from Task 2.2
- Produces:
  - `_classes_to_hazards(safety_qaqc: List[Dict]) -> List[Dict]` returns hazard dicts in the same shape as existing `_parse_safety_hazards`: `[{"type": str, "severity": "critical"|"major"|"minor", "source": "yolo", "confidence": float}, ...]`
  - `_classes_to_defects(safety_qaqc: List[Dict]) -> List[Dict]` returns defect dicts in the same shape as `_parse_defects`: `[{"description": str, "severity": "major"|"minor", "source": "yolo", "confidence": float}, ...]`
  - Severity mapping: `no_hardhat / no_high_vis_vest / fall_hazard_unprotected` → critical; `concrete_crack / concrete_honeycomb / rebar_exposed_defect` → major; `rebar_correct_inspection` is not a defect (filtered out)
  - When image-block output contains `safety_qaqc` AND it's non-empty, `safety_compliance_audit` and `qa_qc_inspection` USE the YOLO-derived hazards/defects PLUS the legacy keyword-parsed ones (union, dedup by `type`/`description`)
  - When `safety_qaqc` absent or empty, behavior is unchanged (legacy keyword path only)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_construction_container_yolo_compose.py
from typing import Any, Dict
from unittest.mock import patch

import pytest

from app.containers.construction import ConstructionContainer


class _FakeImageBlock:
    def __init__(self, payload):
        self.payload = payload
    async def execute(self, input_data: Any, params: Dict = None):
        return {"status": "success", "result": self.payload}


def _container_with(payload):
    c = ConstructionContainer()
    c._dependencies = {"image": _FakeImageBlock(payload)}
    return c


@pytest.mark.asyncio
async def test_safety_audit_uses_yolo_classes_when_present():
    payload = {
        "description": "",
        "extracted_text": "",
        "safety_qaqc": [
            {"class_id": 0, "class": "no_hardhat", "category": "safety", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"class_id": 3, "class": "concrete_crack", "category": "qaqc", "confidence": 0.8, "bbox": [0, 0, 10, 10]},
        ],
    }
    c = _container_with(payload)
    out = await c.safety_compliance_audit({"file_path": "/tmp/x.jpg"}, {"audit_type": "general"})

    hazards = out["violations"]
    types = {h["type"] for h in hazards}
    assert "no_hardhat" in types
    sources = {h.get("source") for h in hazards}
    assert "yolo" in sources


@pytest.mark.asyncio
async def test_qaqc_uses_yolo_classes_when_present():
    payload = {
        "description": "",
        "extracted_text": "",
        "safety_qaqc": [
            {"class_id": 3, "class": "concrete_crack", "category": "qaqc", "confidence": 0.8, "bbox": [0, 0, 10, 10]},
            {"class_id": 5, "class": "rebar_correct_inspection", "category": "qaqc", "confidence": 0.7, "bbox": [0, 0, 10, 10]},
        ],
    }
    c = _container_with(payload)
    out = await c.qa_qc_inspection({"file_path": "/tmp/x.jpg"}, {"type": "concrete"})

    defects = out["defects"]
    descs = {d["description"] for d in defects}
    assert "concrete_crack" in descs
    assert "rebar_correct_inspection" not in descs  # correct-inspection is not a defect


@pytest.mark.asyncio
async def test_yolo_path_skipped_when_safety_qaqc_absent():
    payload = {
        "description": "missing PPE and concrete with crack",
        "extracted_text": "",
    }
    c = _container_with(payload)
    out = await c.safety_compliance_audit({"file_path": "/tmp/x.jpg"}, {"audit_type": "general"})
    # legacy keyword path still works; no source=yolo
    for h in out["violations"]:
        assert h.get("source") != "yolo"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_construction_container_yolo_compose.py -v`
Expected: at least 2 failures

- [ ] **Step 3: Add helpers + compose into `__init__.py`**

In `app/containers/construction/__init__.py`, add inside the `ConstructionContainer` class (next to the existing `_parse_safety_hazards` / `_parse_defects` methods):

```python
    _YOLO_SAFETY_SEVERITY = {
        "no_hardhat": ("critical", "no_hardhat"),
        "no_high_vis_vest": ("critical", "no_high_vis_vest"),
        "fall_hazard_unprotected": ("critical", "fall_hazard_unprotected"),
    }

    _YOLO_QAQC_DEFECT = {
        "concrete_crack": ("major", "concrete_crack"),
        "concrete_honeycomb": ("major", "concrete_honeycomb"),
        "rebar_exposed_defect": ("major", "rebar_exposed_defect"),
    }

    def _classes_to_hazards(self, safety_qaqc):
        out = []
        for entry in safety_qaqc or []:
            mapping = self._YOLO_SAFETY_SEVERITY.get(entry.get("class"))
            if not mapping:
                continue
            severity, hazard_type = mapping
            out.append({
                "type": hazard_type,
                "severity": severity,
                "source": "yolo",
                "confidence": float(entry.get("confidence", 0.0)),
            })
        return out

    def _classes_to_defects(self, safety_qaqc):
        out = []
        for entry in safety_qaqc or []:
            mapping = self._YOLO_QAQC_DEFECT.get(entry.get("class"))
            if not mapping:
                continue
            severity, desc = mapping
            out.append({
                "description": desc,
                "severity": severity,
                "source": "yolo",
                "confidence": float(entry.get("confidence", 0.0)),
            })
        return out
```

- [ ] **Step 4: Wire helpers into `safety_compliance_audit` (in `documents.py`)**

Use Read on `app/containers/construction/documents.py` around line 488 to find `safety_compliance_audit`. After the existing `hazards_found = self._parse_safety_hazards(desc)` line, add:

```python
        safety_qaqc = analysis.get("result", {}).get("safety_qaqc") or []
        yolo_hazards = self._classes_to_hazards(safety_qaqc)
        seen_types = {h["type"] for h in hazards_found}
        for yh in yolo_hazards:
            if yh["type"] not in seen_types:
                hazards_found.append(yh)
                seen_types.add(yh["type"])
```

Also: when calling the image block in this method, pass `{"mode": "safety_qaqc"}` in params so the YOLO path runs:

```python
analysis = await image_block.execute(
    {"file_path": photo_path},
    {"prompt": safety_prompts.get(audit_type, safety_prompts["general"]), "mode": "safety_qaqc"},
)
```

- [ ] **Step 5: Wire helpers into `qa_qc_inspection` (in `documents.py` around line 1671)**

Same pattern. After `defects = self._parse_defects(desc)`:

```python
        safety_qaqc = analysis.get("result", {}).get("safety_qaqc") or []
        yolo_defects = self._classes_to_defects(safety_qaqc)
        seen_descs = {d["description"] for d in defects}
        for yd in yolo_defects:
            if yd["description"] not in seen_descs:
                defects.append(yd)
                seen_descs.add(yd["description"])
```

And add `"mode": "safety_qaqc"` to the image-block params in this method.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_construction_container_yolo_compose.py tests/test_construction_photo_detection.py -v`
Expected: all tests pass — both new YOLO tests AND the existing parallel-session keyword-parsing tests

- [ ] **Step 7: Commit**

```bash
git add app/containers/construction/__init__.py app/containers/construction/documents.py tests/test_construction_container_yolo_compose.py
git commit -m "feat(construction): compose YOLO safety_qaqc output into hazard + defect verdicts"
```

---

### Task 2.4: Inference script — `infer_photo_metadata.py`

**Files:**
- Create: `scripts/infer_photo_metadata.py`
- Test: `tests/test_infer_photo_metadata.py`

**Interfaces:**
- Consumes: `ImageBlock` (via `app.dependencies.get_block_instance("image")`); accepts a folder of photos + an output JSONL path
- Produces:
  - Command-line: `python scripts/infer_photo_metadata.py <folder> <output_jsonl> [--source-zip NAME]`
  - JSONL row per photo matching the spec's `photo_metadata.jsonl` schema exactly (see spec section "`photo_metadata.jsonl` row schema")
  - `sha256` = SHA-256 of the raw file bytes
  - `caption` = templated from class lists (see spec)
  - `inference_failed: true` + `inference_error: str` for photos where inference raises

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer_photo_metadata.py
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from scripts.infer_photo_metadata import build_caption, run_inference


@pytest.fixture
def folder(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "a.jpg")
    Image.new("RGB", (50, 50)).save(tmp_path / "b.jpg")
    return tmp_path


def test_build_caption_safety_and_qaqc():
    qaqc = [
        {"class": "no_hardhat", "category": "safety"},
        {"class": "no_hardhat", "category": "safety"},
        {"class": "concrete_crack", "category": "qaqc"},
    ]
    caption = build_caption(qaqc)
    assert "1 safety issue" not in caption  # 2 safety entries, deduped or counted
    assert "no_hardhat" in caption
    assert "concrete_crack" in caption


def test_build_caption_empty():
    assert build_caption([]) == "Site photo (no detected violations or defects)."


@pytest.mark.asyncio
async def test_run_inference_writes_jsonl_per_photo(folder, tmp_path):
    fake_block = AsyncMock()
    fake_block.execute.return_value = {
        "status": "success",
        "result": {
            "pil": {"width": 100, "height": 100, "format": "JPEG", "file_size_bytes": 123, "dominant_channel": "red"},
            "ocr_text": "",
            "coco_objects": [],
            "safety_qaqc": [{"class_id": 0, "class": "no_hardhat", "category": "safety", "confidence": 0.9, "bbox": [0, 0, 10, 10]}],
        },
    }
    out_jsonl = tmp_path / "out.jsonl"
    with patch("scripts.infer_photo_metadata.get_block_instance", return_value=fake_block):
        await run_inference(folder, out_jsonl, source_zip="construction-3-001.zip")

    lines = out_jsonl.read_text().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["filename"] in ("a.jpg", "b.jpg")
    assert row["source_zip"] == "construction-3-001.zip"
    assert row["project_id"] is None
    assert "no_hardhat" in row["caption"]
    assert row["inference_failed"] is False
    expected_sha = hashlib.sha256((folder / row["filename"]).read_bytes()).hexdigest()
    assert row["sha256"] == expected_sha


@pytest.mark.asyncio
async def test_run_inference_marks_failure(folder, tmp_path):
    fake_block = AsyncMock()
    fake_block.execute.side_effect = ValueError("boom")
    out_jsonl = tmp_path / "out.jsonl"
    with patch("scripts.infer_photo_metadata.get_block_instance", return_value=fake_block):
        await run_inference(folder, out_jsonl, source_zip="z")

    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines()]
    assert all(r["inference_failed"] for r in rows)
    assert all("boom" in r["inference_error"] for r in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_infer_photo_metadata.py -v`
Expected: ImportError

- [ ] **Step 3: Write the script**

```python
# scripts/infer_photo_metadata.py
"""Phase 2a — Run PIL + Tesseract + COCO-YOLO + fine-tuned safety_qaqc YOLO over a folder of photos.

Usage:
    python scripts/infer_photo_metadata.py <folder> <output_jsonl> [--source-zip NAME]

Outputs one JSONL row per image matching the spec's photo_metadata schema.
project_id stays null because the zip has no confirmed project.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dependencies import get_block_instance, init_blocks  # noqa: E402

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def build_caption(safety_qaqc: List[Dict]) -> str:
    if not safety_qaqc:
        return "Site photo (no detected violations or defects)."
    safety_classes = [d["class"] for d in safety_qaqc if d.get("category") == "safety"]
    qaqc_classes = [d["class"] for d in safety_qaqc if d.get("category") == "qaqc"]
    parts: List[str] = []
    if safety_classes:
        parts.append(f"{len(safety_classes)} safety issue(s): " + ", ".join(safety_classes))
    if qaqc_classes:
        parts.append(f"{len(qaqc_classes)} QA/QC issue(s): " + ", ".join(qaqc_classes))
    return "Site photo showing " + "; ".join(parts) + "."


async def run_inference(folder: Path, output_jsonl: Path, source_zip: Optional[str]) -> None:
    await init_blocks()
    image_block = get_block_instance("image")

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    with output_jsonl.open("w", encoding="utf-8") as fp:
        for img in images:
            sha = hashlib.sha256(img.read_bytes()).hexdigest()
            row = {
                "sha256": sha,
                "filename": img.name,
                "source_zip": source_zip,
                "project_id": None,
                "inference_failed": False,
                "inference_error": None,
            }
            try:
                result = await image_block.execute({"file_path": str(img)}, {"mode": "safety_qaqc"})
                body = result.get("result", {})
                row["pil"] = body.get("pil", {})
                row["ocr_text"] = body.get("ocr_text", "")
                row["coco_objects"] = body.get("coco_objects", [])
                row["safety_qaqc"] = body.get("safety_qaqc", [])
                row["caption"] = build_caption(row["safety_qaqc"])
            except Exception as exc:
                row["inference_failed"] = True
                row["inference_error"] = f"{type(exc).__name__}: {exc}"
                row["pil"] = {}
                row["ocr_text"] = ""
                row["coco_objects"] = []
                row["safety_qaqc"] = []
                row["caption"] = "Site photo (inference failed)."
            fp.write(json.dumps(row) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("output_jsonl", type=Path)
    p.add_argument("--source-zip", default=None)
    args = p.parse_args()
    asyncio.run(run_inference(args.folder, args.output_jsonl, args.source_zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_infer_photo_metadata.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/infer_photo_metadata.py tests/test_infer_photo_metadata.py
git commit -m "feat(safety): batch inference script produces photo_metadata.jsonl"
```

---

### Task 2.5: Admin photo endpoints (bytes upload + metadata import)

**Files:**
- Create: `app/routers/admin_photos.py`
- Modify: `app/main.py` (mount the router)
- Test: `tests/test_admin_photos.py`

**Interfaces:**
- Consumes: existing admin auth dependency (find via `grep -r "require_admin\|admin_role\|admin_only" app/routers`); existing DB session dependency (find via `grep -rn "get_db_session\|get_session\|get_async_session" app/`)
- Produces:
  - `POST /v1/admin/photo-bytes/{sha256}` — multipart upload, body field name `file`, max 25 MB, content-type kept as-is. Idempotent: existing sha256 returns 200 with `{stored: false, sha256: ...}`.
  - `POST /v1/admin/photo-import` — body is text/plain JSONL stream. Validates each row has `sha256`. Rejects rows whose `sha256` is not present in `photos` table. Inserts into `photo_chunks` with `chunk_id=sha256`, `project_id=row.project_id` (null for V1 zip), `sha256=row.sha256`, `caption=row.caption`, `photo_metadata=json.dumps(row)` (or jsonb-cast on Postgres). Idempotent on `sha256` (UNIQUE constraint).
  - Response: `{inserted: N, skipped_duplicate: M, rejected_no_bytes: K, errors: [...]}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_admin_photos.py
import json
from io import BytesIO

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_photo_bytes_then_import_metadata(admin_client: AsyncClient, db_session):
    # admin_client + db_session fixtures provided by tests/conftest.py — see existing
    # tests/test_admin_*.py for the pattern.
    photo_bytes = b"\xff\xd8\xff\xe0" + b"jpeg-payload" * 100
    import hashlib
    sha = hashlib.sha256(photo_bytes).hexdigest()

    r = await admin_client.post(
        f"/v1/admin/photo-bytes/{sha}",
        files={"file": ("a.jpg", BytesIO(photo_bytes), "image/jpeg")},
    )
    assert r.status_code == 200
    assert r.json()["stored"] is True

    r2 = await admin_client.post(
        f"/v1/admin/photo-bytes/{sha}",
        files={"file": ("a.jpg", BytesIO(photo_bytes), "image/jpeg")},
    )
    assert r2.json()["stored"] is False

    jsonl = json.dumps({
        "sha256": sha, "filename": "a.jpg", "source_zip": "z",
        "project_id": None, "caption": "Site photo showing 1 safety issue(s): no_hardhat.",
        "safety_qaqc": [{"class": "no_hardhat"}], "pil": {}, "ocr_text": "",
        "coco_objects": [], "inference_failed": False, "inference_error": None,
    }) + "\n"
    r3 = await admin_client.post("/v1/admin/photo-import", content=jsonl, headers={"content-type": "text/plain"})
    body = r3.json()
    assert body["inserted"] == 1
    assert body["skipped_duplicate"] == 0
    assert body["rejected_no_bytes"] == 0


@pytest.mark.asyncio
async def test_import_rejects_when_bytes_missing(admin_client: AsyncClient):
    jsonl = json.dumps({"sha256": "deadbeef" * 8, "caption": "x"}) + "\n"
    r = await admin_client.post("/v1/admin/photo-import", content=jsonl, headers={"content-type": "text/plain"})
    assert r.json()["rejected_no_bytes"] == 1
    assert r.json()["inserted"] == 0


@pytest.mark.asyncio
async def test_import_idempotent_on_sha256(admin_client: AsyncClient):
    photo_bytes = b"\xff\xd8\xff" + b"x" * 100
    import hashlib
    sha = hashlib.sha256(photo_bytes).hexdigest()
    await admin_client.post(
        f"/v1/admin/photo-bytes/{sha}",
        files={"file": ("a.jpg", BytesIO(photo_bytes), "image/jpeg")},
    )
    jsonl = json.dumps({"sha256": sha, "caption": "first", "safety_qaqc": [], "filename": "a.jpg",
                        "source_zip": None, "project_id": None, "pil": {}, "ocr_text": "",
                        "coco_objects": [], "inference_failed": False, "inference_error": None}) + "\n"
    await admin_client.post("/v1/admin/photo-import", content=jsonl, headers={"content-type": "text/plain"})
    r2 = await admin_client.post("/v1/admin/photo-import", content=jsonl, headers={"content-type": "text/plain"})
    body = r2.json()
    assert body["inserted"] == 0
    assert body["skipped_duplicate"] == 1
```

- [ ] **Step 2: Inspect existing admin router to learn auth + DB patterns**

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; print(Path('app/routers/admin.py').read_text()[:3000])"`

Note: the import names + function signatures for admin-auth dep and DB session — use them verbatim in the new router.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_photos.py -v`
Expected: 404 — router not mounted

- [ ] **Step 4: Write the router**

```python
# app/routers/admin_photos.py
"""Admin endpoints for photo bytes + metadata import (Phase 2)."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

# Use whatever the existing admin router imports for auth + DB session.
# This block must be reconciled with the actual app/routers/admin.py imports.
from app.routers.admin import require_admin  # adjust if name differs
from app.dependencies import get_db_session  # adjust if name differs

router = APIRouter(prefix="/v1/admin", tags=["admin-photos"])
logger = logging.getLogger(__name__)

_MAX_PHOTO_BYTES = 25 * 1024 * 1024


@router.post("/photo-bytes/{sha256}")
async def upload_photo_bytes(
    sha256: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
    _: Any = Depends(require_admin),
) -> Dict[str, Any]:
    existing = await db.execute(text("SELECT 1 FROM photos WHERE sha256 = :s"), {"s": sha256})
    if existing.first() is not None:
        return {"stored": False, "sha256": sha256}

    data = await file.read()
    if len(data) > _MAX_PHOTO_BYTES:
        raise HTTPException(413, "photo exceeds 25 MB limit")

    import hashlib
    if hashlib.sha256(data).hexdigest() != sha256:
        raise HTTPException(400, "sha256 mismatch")

    await db.execute(
        text(
            "INSERT INTO photos (sha256, content_type, size_bytes, bytes) "
            "VALUES (:s, :c, :sz, :b)"
        ),
        {"s": sha256, "c": file.content_type or "image/jpeg", "sz": len(data), "b": data},
    )
    await db.commit()
    return {"stored": True, "sha256": sha256, "size_bytes": len(data)}


@router.post("/photo-import")
async def photo_import(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    _: Any = Depends(require_admin),
) -> Dict[str, Any]:
    body = (await request.body()).decode("utf-8")
    inserted = skipped = rejected = 0
    errors: List[str] = []
    for line_no, line in enumerate(body.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: bad JSON: {exc}")
            continue
        sha = row.get("sha256")
        if not sha:
            errors.append(f"line {line_no}: missing sha256")
            continue
        bytes_present = await db.execute(text("SELECT 1 FROM photos WHERE sha256 = :s"), {"s": sha})
        if bytes_present.first() is None:
            rejected += 1
            continue
        existing = await db.execute(
            text("SELECT 1 FROM photo_chunks WHERE sha256 = :s"),
            {"s": sha},
        )
        if existing.first() is not None:
            skipped += 1
            continue
        await db.execute(
            text(
                "INSERT INTO photo_chunks (chunk_id, project_id, sha256, caption, photo_metadata) "
                "VALUES (:cid, :p, :s, :c, :m)"
            ),
            {
                "cid": sha,
                "p": row.get("project_id"),
                "s": sha,
                "c": row.get("caption") or "Site photo.",
                "m": json.dumps(row),
            },
        )
        inserted += 1
    await db.commit()
    return {"inserted": inserted, "skipped_duplicate": skipped, "rejected_no_bytes": rejected, "errors": errors}
```

- [ ] **Step 5: Mount the router in `app/main.py`**

Locate the existing `app.include_router(...)` calls in `app/main.py`. Add:

```python
from app.routers import admin_photos
app.include_router(admin_photos.router)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_photos.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add app/routers/admin_photos.py app/main.py tests/test_admin_photos.py
git commit -m "feat(admin): photo-bytes + photo-import endpoints with idempotency + ordering"
```

---

### Task 2.6: Photo serving endpoint `GET /v1/photos/{sha256}`

**Files:**
- Create: `app/routers/photos.py`
- Modify: `app/main.py` (mount the new router)
- Test: `tests/test_photo_serving.py`

**Interfaces:**
- Consumes: `photos` table from Task 0.2; `doc_index` for project-scope check
- Produces:
  - `GET /v1/photos/{sha256}` returns the photo bytes with the original `content_type`
  - 404 when sha256 not in `photos` table
  - For V1: no project-scope check (the zip has `project_id=null`); add a TODO marker so Phase 3 can wire it
  - Sets `Cache-Control: public, max-age=86400`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_photo_serving.py
import hashlib
from io import BytesIO

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_serves_uploaded_photo(admin_client: AsyncClient, public_client: AsyncClient):
    payload = b"\xff\xd8\xff\xe0" + b"jpg" * 50
    sha = hashlib.sha256(payload).hexdigest()
    await admin_client.post(
        f"/v1/admin/photo-bytes/{sha}",
        files={"file": ("a.jpg", BytesIO(payload), "image/jpeg")},
    )
    r = await public_client.get(f"/v1/photos/{sha}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/jpeg")
    assert r.content == payload


@pytest.mark.asyncio
async def test_returns_404_for_unknown_sha(public_client: AsyncClient):
    r = await public_client.get("/v1/photos/" + "0" * 64)
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_photo_serving.py -v`
Expected: 404 — route missing

- [ ] **Step 3: Write the router**

```python
# app/routers/photos.py
"""Public photo bytes serving (Phase 2)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db_session

router = APIRouter(prefix="/v1/photos", tags=["photos"])


@router.get("/{sha256}")
async def get_photo(sha256: str, db: AsyncSession = Depends(get_db_session)) -> Response:
    # Phase 3 TODO: project-scope check against doc_index.project_id once
    # photos start being uploaded under specific projects via the platform UI.
    row = await db.execute(
        text("SELECT content_type, bytes FROM photos WHERE sha256 = :s"),
        {"s": sha256},
    )
    record = row.first()
    if record is None:
        raise HTTPException(404, "photo not found")
    return Response(
        content=record.bytes,
        media_type=record.content_type or "image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
```

- [ ] **Step 4: Mount in `app/main.py`**

```python
from app.routers import photos as photos_router
app.include_router(photos_router.router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_photo_serving.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add app/routers/photos.py app/main.py tests/test_photo_serving.py
git commit -m "feat(photos): GET /v1/photos/{sha256} serves bytes from photos table"
```

---

### Task 2.7: RAG retriever — add photo_chunks BM25 leg

**Files:**
- Modify: `app/core/rag/vector_store.py` — add a parallel BM25 leg over `photo_chunks` (Postgres tsquery on `to_tsvector('english', caption)`; SQLite via new `photo_chunks_fts` FTS5 virtual table built the same way as `chunks_fts`)
- Modify: `app/core/rag/retriever.py` — merge photo BM25 results into the returned chunk list with `kind` field
- Test: `tests/test_retriever_photo_chunks.py`

**Interfaces:**
- Consumes: `photo_chunks` rows from Task 2.5
- Produces:
  - Retrieval results include both text chunks (`kind="text"`) and photo chunks (`kind="photo"`)
  - Each photo chunk carries: `content` = caption + `"\nClasses: "` + comma-joined class names from photo_metadata, `kind="photo"`, `photo_url=f"/v1/photos/{sha256}"`, `thumbnail_url=photo_url + "?w=256"` (thumbnail handler isn't built in V1; URL is informational)
  - When the existing BM25 leg returns N text chunks and photo BM25 returns M photo chunks, the merged top_k drops the lowest-scoring across both sets (one global ranking)
  - SQLite path creates `photo_chunks_fts` lazily on first query (mirroring `chunks_fts` lazy init); Postgres uses an ad-hoc `to_tsvector` query (the GIN index added in Task 0.2 makes this fast)

- [ ] **Step 1: Inspect retriever to learn current shape**

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; print(Path('app/core/rag/retriever.py').read_text()[:4000])"`

Identify: the function that returns retrieval results, and the exact dict shape it produces. Adapt the test below to match. (The plan below uses placeholder field names that the implementer MUST replace with the actual retriever's field names.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_retriever_photo_chunks.py
import json

import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_retrieve_includes_photo_chunks_with_class_names(db_session, retriever):
    sha = "abc" * 21 + "a"
    await db_session.execute(text("INSERT INTO photos (sha256, content_type, size_bytes, bytes) VALUES (:s, 'image/jpeg', 10, :b)"), {"s": sha, "b": b"x" * 10})
    photo_metadata = json.dumps({
        "sha256": sha, "caption": "Site photo showing 1 safety issue(s): no_hardhat.",
        "safety_qaqc": [{"class": "no_hardhat", "confidence": 0.9}],
    })
    await db_session.execute(
        text(
            "INSERT INTO photo_chunks (chunk_id, project_id, sha256, caption, photo_metadata) "
            "VALUES (:cid, NULL, :s, :c, :m)"
        ),
        {"cid": sha, "s": sha, "c": "Site photo showing 1 safety issue(s): no_hardhat.", "m": photo_metadata},
    )
    await db_session.commit()

    results = await retriever.retrieve(query="no hardhat", top_k=10, project_id=None)
    photo_results = [r for r in results if r.get("kind") == "photo"]
    assert len(photo_results) >= 1
    assert "no_hardhat" in photo_results[0]["content"]
    assert photo_results[0]["photo_url"] == f"/v1/photos/{sha}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retriever_photo_chunks.py -v`
Expected: KeyError on `kind` or `photo_url` — adapter missing

- [ ] **Step 4: Add the photo_chunks BM25 leg**

In `app/core/rag/vector_store.py`, after the existing `_bm25_postgres` / `_bm25_sqlite` (FTS5) functions, add a parallel `_bm25_postgres_photos` / `_bm25_sqlite_photos`:

- Postgres version: `SELECT chunk_id, sha256, caption, photo_metadata, ts_rank_cd(to_tsvector('english', caption), q) AS score FROM photo_chunks c, plainto_tsquery('english', :q) AS q WHERE to_tsvector('english', caption) @@ q ORDER BY score DESC LIMIT :k`
- SQLite version: build `photo_chunks_fts` virtual table on first call via `_ensure_fts5_photos_sqlite` (mirror `_ensure_fts5_sqlite` exactly); then SELECT against it.

In `app/core/rag/retriever.py`, where the existing `retrieve_with_filter` (or equivalent top-level function) calls BM25, also call the photo BM25 leg, and merge results:

```python
def _photo_chunk_to_result(row) -> dict:
    meta = row.photo_metadata
    if isinstance(meta, str):
        meta = json.loads(meta)
    meta = meta or {}
    caption = meta.get("caption", row.caption)
    class_names = [d.get("class") for d in meta.get("safety_qaqc") or [] if d.get("class")]
    content = f"{caption}\nClasses: {', '.join(class_names)}" if class_names else caption
    return {
        "kind": "photo",
        "content": content,
        "photo_url": f"/v1/photos/{row.sha256}",
        "thumbnail_url": f"/v1/photos/{row.sha256}?w=256",
        "score": float(row.score),
        "sha256": row.sha256,
    }

# After existing BM25 returns text_results:
photo_rows = await _bm25_photos(query, top_k)
photo_results = [_photo_chunk_to_result(r) for r in photo_rows]
text_results = [{**r, "kind": "text"} for r in text_results]  # tag the existing leg
merged = sorted(text_results + photo_results, key=lambda r: -r["score"])[:top_k]
```

The exact attribute/field names depend on the retriever's existing dict shape — adapt to fit.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retriever_photo_chunks.py tests/ -k "retriever" -v`
Expected: 1 new passed + no regressions in existing retriever tests

- [ ] **Step 6: Commit**

```bash
git add app/core/rag/retriever.py tests/test_retriever_photo_chunks.py
git commit -m "feat(rag): photo chunks expose class names + photo_url in retrieval output"
```

---

### Task 2.8: Export script `export_to_render.py`

**Files:**
- Create: `scripts/export_to_render.py`
- Test: `tests/test_export_to_render.py`

**Interfaces:**
- Consumes: `photo_metadata.jsonl` from Task 2.4; admin endpoints from Task 2.5
- Produces:
  - Command-line: `python scripts/export_to_render.py <jsonl> <photos_dir> --base-url https://the-fork.onrender.com --token $TOKEN [--state-file state.json]`
  - For each row: PUT bytes first (`POST /v1/admin/photo-bytes/{sha256}`), then collect successful sha256s
  - After all bytes uploaded: stream JSONL of successful rows to `POST /v1/admin/photo-import`
  - Resumable via state file: tracks `uploaded_sha256s` set, skips ones already uploaded
  - Exponential backoff on 5xx (3 retries, 1s/2s/4s)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_to_render.py
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_uploads_bytes_then_metadata_in_order(tmp_path):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    img = photos_dir / "a.jpg"
    img.write_bytes(b"jpg-bytes")
    import hashlib
    sha = hashlib.sha256(img.read_bytes()).hexdigest()

    jsonl = tmp_path / "meta.jsonl"
    jsonl.write_text(json.dumps({"sha256": sha, "filename": "a.jpg", "caption": "c"}) + "\n")

    call_order = []
    async def fake_post(url, **kwargs):
        call_order.append(url)
        rv = AsyncMock()
        rv.status_code = 200
        rv.json = lambda: {"stored": True, "inserted": 1, "skipped_duplicate": 0, "rejected_no_bytes": 0, "errors": []}
        rv.raise_for_status = lambda: None
        return rv

    with patch("scripts.export_to_render.httpx.AsyncClient") as ac_cls:
        ac_cls.return_value.__aenter__.return_value.post = fake_post
        from scripts.export_to_render import run_export
        await run_export(jsonl, photos_dir, "https://render.test", "token", tmp_path / "state.json")

    assert any("/photo-bytes/" in u for u in call_order)
    assert any("/photo-import" in u for u in call_order)
    # bytes call comes before import call
    bytes_idx = next(i for i, u in enumerate(call_order) if "/photo-bytes/" in u)
    import_idx = next(i for i, u in enumerate(call_order) if "/photo-import" in u)
    assert bytes_idx < import_idx


@pytest.mark.asyncio
async def test_skips_already_uploaded_via_state_file(tmp_path):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    img = photos_dir / "a.jpg"
    img.write_bytes(b"jpg-bytes")
    import hashlib
    sha = hashlib.sha256(img.read_bytes()).hexdigest()
    jsonl = tmp_path / "meta.jsonl"
    jsonl.write_text(json.dumps({"sha256": sha, "filename": "a.jpg", "caption": "c"}) + "\n")

    state = tmp_path / "state.json"
    state.write_text(json.dumps({"uploaded_sha256s": [sha]}))

    call_order = []
    async def fake_post(url, **kwargs):
        call_order.append(url)
        rv = AsyncMock()
        rv.status_code = 200
        rv.json = lambda: {"stored": False, "inserted": 0, "skipped_duplicate": 1, "rejected_no_bytes": 0, "errors": []}
        rv.raise_for_status = lambda: None
        return rv

    with patch("scripts.export_to_render.httpx.AsyncClient") as ac_cls:
        ac_cls.return_value.__aenter__.return_value.post = fake_post
        from scripts.export_to_render import run_export
        await run_export(jsonl, photos_dir, "https://render.test", "token", state)

    assert not any("/photo-bytes/" in u for u in call_order)
    assert any("/photo-import" in u for u in call_order)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_export_to_render.py -v`
Expected: ImportError

- [ ] **Step 3: Write the script**

```python
# scripts/export_to_render.py
"""Phase 2b — Export photo bytes + metadata from PC to Render.

Usage:
    python scripts/export_to_render.py <jsonl> <photos_dir> --base-url ... --token ... [--state-file ...]

Uploads bytes first (idempotent), then streams metadata JSONL. Resumable via state file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List

import httpx

_RETRY_BACKOFFS = (1.0, 2.0, 4.0)


async def _post_with_retries(client: httpx.AsyncClient, url: str, **kwargs):
    last_exc = None
    for backoff in (0.0, *_RETRY_BACKOFFS):
        if backoff:
            await asyncio.sleep(backoff)
        try:
            r = await client.post(url, **kwargs)
            if r.status_code >= 500:
                continue
            r.raise_for_status()
            return r
        except (httpx.HTTPError,) as exc:
            last_exc = exc
    raise RuntimeError(f"giving up on {url}: {last_exc}")


async def run_export(jsonl: Path, photos_dir: Path, base_url: str, token: str, state_file: Path) -> None:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    state = {"uploaded_sha256s": []}
    if state_file.is_file():
        state = json.loads(state_file.read_text())
    uploaded = set(state.get("uploaded_sha256s", []))

    headers = {"Authorization": f"Bearer {token}"}
    successful_shas: List[str] = []

    async with httpx.AsyncClient(timeout=60, headers=headers) as client:
        for row in rows:
            sha = row["sha256"]
            if sha in uploaded:
                successful_shas.append(sha)
                continue
            img_path = photos_dir / row["filename"]
            if not img_path.is_file():
                continue
            with img_path.open("rb") as fp:
                files = {"file": (img_path.name, fp, "image/jpeg")}
                r = await _post_with_retries(client, f"{base_url}/v1/admin/photo-bytes/{sha}", files=files)
            if r.status_code == 200:
                uploaded.add(sha)
                successful_shas.append(sha)
                state["uploaded_sha256s"] = sorted(uploaded)
                state_file.write_text(json.dumps(state))

        successful_set = set(successful_shas)
        body = "\n".join(json.dumps(r) for r in rows if r["sha256"] in successful_set) + "\n"
        await _post_with_retries(
            client,
            f"{base_url}/v1/admin/photo-import",
            content=body,
            headers={"content-type": "text/plain", **headers},
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path)
    p.add_argument("photos_dir", type=Path)
    p.add_argument("--base-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--state-file", type=Path, default=Path("export_state.json"))
    args = p.parse_args()
    asyncio.run(run_export(args.jsonl, args.photos_dir, args.base_url, args.token, args.state_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_export_to_render.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/export_to_render.py tests/test_export_to_render.py
git commit -m "feat(safety): export script ships bytes + metadata to Render in order"
```

---

### Task 2.9 [OPERATOR-DRIVEN]: Run end-to-end on construction-3-001.zip

**This task is operator-driven. The subagent driver MUST STOP and surface this checkpoint.**

**Operator actions:**

1. Confirm migration 0006 ran on Render (auto-deploys with Phase 0):
   ```bash
   curl -H "Authorization: Bearer $ADMIN_TOKEN" https://the-fork.onrender.com/v1/admin/health
   ```
   (Or check Render dashboard: srv-d8hdc6ek1jcs739rq5sg latest deploy includes commit from Task 0.2)

2. Set the safety detector weights env var locally and run inference:
   ```powershell
   $env:SAFETY_DETECTOR_WEIGHTS = "data\models\safety_qaqc_v1.pt"
   .venv\Scripts\python.exe scripts\infer_photo_metadata.py data\training\raw_photos data\training\photo_metadata.jsonl --source-zip construction-3-001.zip
   ```
   Expect ~5–15 min for 208 photos depending on hardware.

3. Spot-check a few rows for sanity:
   ```bash
   .venv/Scripts/python.exe -c "import json; rows=[json.loads(l) for l in open('data/training/photo_metadata.jsonl')]; print('total:',len(rows)); print('failed:',sum(r['inference_failed'] for r in rows)); print('with detections:',sum(bool(r['safety_qaqc']) for r in rows)); print(rows[0])"
   ```

4. Export to Render:
   ```powershell
   $env:ADMIN_TOKEN = "<your admin bearer token>"
   .venv\Scripts\python.exe scripts\export_to_render.py data\training\photo_metadata.jsonl data\training\raw_photos --base-url https://the-fork.onrender.com --token $env:ADMIN_TOKEN
   ```
   First run uploads bytes + metadata. State file at `export_state.json` lets you resume if interrupted.

5. Verify retrieval works. Send a chat query that should match photo classes:
   ```bash
   curl -X POST https://the-fork.onrender.com/v1/execute -H "Authorization: Bearer $CB_KEY" -d '{"block":"chat","input_data":{"text":"show me photos with concrete cracks","project_id":null}}'
   ```
   Expect: response cites photo chunks. If response cites zero photos, check:
   - `SELECT COUNT(*) FROM doc_index WHERE kind='photo'` (should equal number of exported rows)
   - `SELECT COUNT(*) FROM photos` (should equal number of exported rows)
   - retriever query matches caption substring "concrete_crack" (case-sensitive depending on BM25 config)

6. Tell the team: V1 photo RAG is live on Render. Next phase (active-project upload pathway) gets its own spec when ready.

---

## Self-Review

(This section is for the plan author to verify the plan covers the spec; not a runtime step.)

1. **Spec coverage:** Every spec component is covered by a task:
   - `safety_classes.json` + loader → Task 0.1
   - Migration 0006 → Task 0.2
   - Survey script → Task 0.3 + checkpoint 0.4
   - Pre-label script → Task 1.1 + checkpoint 1.2
   - Training script → Task 1.3 + checkpoint 1.4
   - Safety detector module → Task 2.1
   - Image block extension → Task 2.2
   - Construction container composition → Task 2.3
   - Inference script → Task 2.4
   - Admin endpoints → Task 2.5
   - Photo serving → Task 2.6
   - Retriever change → Task 2.7
   - Export script → Task 2.8
   - End-to-end run → Task 2.9

2. **Placeholder scan:** Tasks 2.5 and 2.7 leave the auth/DB dependency import names and the retriever's exact field names to the implementer to reconcile against existing code (Step 2 in each: "inspect existing router"). This is necessary because the spec doesn't pin those — they're already-existing patterns the new code must match. Implementer must Read + Grep before writing.

3. **Type consistency:** `safety_qaqc` is the spec's field name for YOLO output and is used consistently across Tasks 2.2, 2.3, 2.4. `safety_qaqc` is the image-block mode name. `kind="photo"` is the doc_index discriminator everywhere. `photo_metadata` is the JSONB column name. Class IDs use the stable registry-side IDs in `safety_qaqc` entries, not the YOLO-side renumbered IDs. The classmap JSON name is consistently `safety_qaqc_v{N}_classmap.json`.

4. **Operator-driven gates:** 4 checkpoint tasks (0.4, 1.2, 1.4, 2.9). Subagent driver MUST stop at each and surface the operator's required actions; resume only after the operator confirms.
