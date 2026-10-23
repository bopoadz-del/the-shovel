# Layered RAG (4-layer + authority) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the project-coupled single-corpus RAG with the layered design in `docs/rag-deployment-plan.md` — knowledge lives in layers (L1 shared domain, L2A company/client rules, L2B live project record, L3 user/session) with cross-cutting authority scoring, and is structurally decoupled from workspace ("project") rows so deleting a workspace can never destroy a chunk.

**Architecture:** Add persisted `layer` + `authority` columns to `chunks`. Route ingestion into a layer by source. Retrieval stops keying off the caller's workspace `project_id`; it reads a fixed layer set (the shared Main corpus L1/L2A/L2B) plus the caller's own L3 upload layer, then re-ranks by authority precedence. Soften the `project_id` foreign key from `ON DELETE CASCADE` to `ON DELETE SET NULL` so knowledge outlives workspaces. Entire behavior is gated behind `RAG_LAYERED` (default OFF) so the cloud dev/demo stays byte-for-byte today until flipped.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic (Postgres prod / SQLite dev+test), pgvector, model2vec (256-dim), pytest (asyncio_mode=auto).

## Global Constraints

- **Flag-gated, default OFF.** `RAG_LAYERED` env (truthy: `1/true/yes/on`). When off, every new path is a strict no-op and retrieval/ingestion behave exactly as today. This ships to `main`/cloud without changing demo behavior; the pilot/on-prem env sets `RAG_LAYERED=1`.
- **No emojis** anywhere in repo, code, comments, commits, UI.
- **Never hard-delete a project/chunk in code paths.** Deletes stay soft (archive). NEVER-delete ids: `projects_folder, training_material, master_corpus, example_infra_pack, curated_kb, drive_archive, bc812f36, b5a0fed8`.
- **Dual DB.** Every schema change lands in BOTH `the_fork_schema.sql` (fresh-DB bootstrap) AND an Alembic migration (next rev = `0011`). ORM (`app/core/models.py::RagChunk`) omits FK constraints (SQLite convenience) — FK/cascade changes go in the Alembic migration + `the_fork_schema.sql` only.
- **Layer enum values (verbatim):** `shared_domain` (L1), `company_rules` (L2A), `project_record` (L2B), `user_session` (L3). Legacy retrieval tags (`own`/`general_knowledge`/`master_corpus`) map onto these but stay accepted.
- **Authority enum values (verbatim):** `contractual, design, commercial, operational, policy, historical, personal`. Precedence high→low in that order.
- **Embedding dim = 256** (`EMBEDDING_DIM`). Do not change.
- **Provider routing** Groq→Ollama only; no schema task calls an LLM.

---

## Stage roadmap (5 stages, each independently shippable)

| Stage | Deliverable | Ships |
|---|---|---|
| **1** | Persisted `layer`+`authority` columns; softened cascade; `Chunk` dataclass + `upsert_chunks`/`search` thread them; flag scaffold. Behavior unchanged (flag off). | This plan, full detail below |
| **2** | Layer + authority classification at ingest (`index_chunks(..., layer=, authority=)`); source→layer map. | Expanded at execution |
| **3** | Layer-aware retrieval + authority-precedence re-rank (L2B dominates project questions; contract beats note). Reuses existing boost pipeline in `retriever.retrieve_with_filter`. | Expanded at execution |
| **4** | User-upload layer routing: uploads → `user_session` keyed by owner, never Main; retrieval scope = Main layers + caller's L3; source disclosure labels. Workspaces own no knowledge. | Expanded at execution |
| **5** | Migration/backfill: re-tag `curated_kb`→L1, `example_infra_pack`/`drive_archive`→L2B, existing uploads→L3; `hidden_from_sidebar` flag on system rows; flip `RAG_LAYERED=1` on pilot env. | Expanded at execution |

Stages 2-5 have concrete file targets + acceptance criteria at the end of this doc; their bite-sized steps are written when each stage begins (keeps the plan honest — earlier stages settle exact signatures the later ones consume).

---

## Stage 1 — Persisted layer + authority columns, softened cascade, flag scaffold

### Task 1: Layer/authority constants + `RAG_LAYERED` flag

**Files:**
- Create: `app/core/rag/layers.py`
- Test: `tests/test_rag_layers.py`

**Interfaces:**
- Produces: `LAYERS` (frozenset[str]), `AUTHORITIES` (tuple[str,...] high→low), `DEFAULT_LAYER="shared_domain"`, `DEFAULT_AUTHORITY="operational"`, `layered_enabled() -> bool`, `authority_rank(name: str) -> int` (0 = highest precedence, larger = weaker; unknown → len(AUTHORITIES)).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_layers.py
from app.core.rag import layers as L

def test_layer_and_authority_vocab():
    assert L.LAYERS == frozenset(
        {"shared_domain", "company_rules", "project_record", "user_session"})
    assert L.AUTHORITIES == (
        "contractual", "design", "commercial", "operational",
        "policy", "historical", "personal")

def test_authority_rank_precedence():
    # contract beats meeting note (policy) beats a personal note
    assert L.authority_rank("contractual") < L.authority_rank("policy")
    assert L.authority_rank("policy") < L.authority_rank("personal")
    assert L.authority_rank("does-not-exist") == len(L.AUTHORITIES)

def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    assert L.layered_enabled() is False
    monkeypatch.setenv("RAG_LAYERED", "1")
    assert L.layered_enabled() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rag_layers.py -q`
Expected: FAIL (`ModuleNotFoundError: app.core.rag.layers`).

- [ ] **Step 3: Write minimal implementation**

```python
# app/core/rag/layers.py
"""Layered-RAG vocabulary + flag (docs/rag-deployment-plan.md).

Layers are the WHERE of knowledge; authority is the HOW-MUCH-IT-WINS.
Both are persisted per chunk (Alembic 0011) and default-inert until
RAG_LAYERED is set — see docs/superpowers/plans/2026-07-23-layered-rag.md.
"""
from __future__ import annotations

import os

# L1 shared domain, L2A company/client rules, L2B live project record,
# L3 user/session. Names are the persisted `chunks.layer` values.
LAYERS = frozenset({"shared_domain", "company_rules", "project_record", "user_session"})
DEFAULT_LAYER = "shared_domain"

# Cross-cutting authority, highest precedence first. contract beats meeting
# note; approved drawing beats draft; project BOQ beats generic benchmark.
AUTHORITIES = ("contractual", "design", "commercial", "operational",
               "policy", "historical", "personal")
DEFAULT_AUTHORITY = "operational"

# Legacy retrieval-time tags map onto persisted layers (back-compat).
LEGACY_LAYER_MAP = {
    "own": "project_record",
    "general_knowledge": "shared_domain",
    "master_corpus": "project_record",
    "user": "user_session",
}


def layered_enabled() -> bool:
    return str(os.getenv("RAG_LAYERED", "")).strip().lower() in {"1", "true", "yes", "on"}


def authority_rank(name: str) -> int:
    """0 = strongest. Unknown authority sorts weakest."""
    try:
        return AUTHORITIES.index(name)
    except ValueError:
        return len(AUTHORITIES)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rag_layers.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/core/rag/layers.py tests/test_rag_layers.py
git commit -m "feat(rag): layered-RAG vocabulary + RAG_LAYERED flag (stage 1/5)"
```

### Task 2: Persist `layer` + `authority` columns (schema + Alembic + ORM)

**Files:**
- Modify: `the_fork_schema.sql:192-209` (add two columns + index)
- Create: `alembic/versions/0011_chunks_layer_authority.py`
- Modify: `app/core/models.py:432-438` (RagChunk columns)
- Test: `tests/test_chunks_layer_columns.py`

**Interfaces:**
- Produces: `chunks.layer TEXT` (nullable, default `NULL`) and `chunks.authority TEXT` (nullable). `RagChunk.layer`, `RagChunk.authority` mapped columns. A partial index `idx_chunks_layer` on `(layer)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunks_layer_columns.py
from sqlalchemy import inspect
from app.core.models import RagChunk

def test_ragchunk_has_layer_and_authority():
    cols = {c.name for c in RagChunk.__table__.columns}
    assert "layer" in cols
    assert "authority" in cols

def test_layer_columns_are_nullable():
    by = {c.name: c for c in RagChunk.__table__.columns}
    assert by["layer"].nullable is True
    assert by["authority"].nullable is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunks_layer_columns.py -q`
Expected: FAIL (`AssertionError` — columns absent).

- [ ] **Step 3: Write minimal implementation**

`app/core/models.py` — add after `created_at` (line 438):

```python
    # Layered RAG (docs/rag-deployment-plan.md). Nullable so pre-migration
    # rows and the flag-off path read as unlayered. Values: see
    # app/core/rag/layers.py LAYERS / AUTHORITIES.
    layer: Mapped[str | None] = mapped_column(String, nullable=True)
    authority: Mapped[str | None] = mapped_column(String, nullable=True)
```

Add to `RagChunk.__table_args__` (after the existing `idx_chunks_doc` Index):

```python
        Index("idx_chunks_layer", "layer"),
```

`the_fork_schema.sql` — within `CREATE TABLE chunks (...)` add before the closing `)` / UNIQUE line:

```sql
    layer       TEXT,
    authority   TEXT,
```

and after the existing indexes:

```sql
CREATE INDEX idx_chunks_layer ON chunks (layer);
```

`alembic/versions/0011_chunks_layer_authority.py`:

```python
"""chunks: layer + authority columns (layered RAG stage 1)

Revision ID: 0011_chunks_layer_authority
Revises: 0010_projects_location
"""
from alembic import op
import sqlalchemy as sa

revision = "0011_chunks_layer_authority"
down_revision = "0010_projects_location"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("layer", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("authority", sa.Text(), nullable=True))
    op.create_index("idx_chunks_layer", "chunks", ["layer"])


def downgrade() -> None:
    op.drop_index("idx_chunks_layer", table_name="chunks")
    op.drop_column("chunks", "authority")
    op.drop_column("chunks", "layer")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunks_layer_columns.py -q`
Expected: PASS (2 tests).
Run (regression — SQLite bootstrap still builds the table): `python -m pytest tests/test_step0_retrieval_isolation.py -q`
Expected: PASS (unchanged).

- [ ] **Step 5: Commit**

```bash
git add the_fork_schema.sql alembic/versions/0011_chunks_layer_authority.py app/core/models.py tests/test_chunks_layer_columns.py
git commit -m "feat(rag): persist chunks.layer + chunks.authority (stage 1/5)"
```

### Task 3: Thread layer/authority through `Chunk` dataclass + `upsert_chunks` + `search`

**Files:**
- Modify: `app/core/rag/vector_store.py` — `Chunk` dataclass (~line 25 region), `upsert_chunks` (494), `search` (634) and the row→Chunk hydration.
- Test: `tests/test_vector_store_layer_roundtrip.py`

**Interfaces:**
- Consumes: `Chunk` already has `layer` (STEP 0). 
- Produces: `Chunk.authority: str | None`. `upsert_chunks(project_id, doc_id, chunks, embeddings, *, layer=None, authority=None)` — writes both columns for every row. `search(...)` returns `Chunk`s with `.layer`/`.authority` populated from the stored columns (falling back to the retrieval-time tag when the column is NULL).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_store_layer_roundtrip.py
import numpy as np
from app.core.rag.vector_store import VectorStore

def _emb(store, texts):
    return np.asarray(store._embed(texts) if hasattr(store, "_embed")
                      else [[0.0]*256 for _ in texts], dtype=np.float32)

def test_upsert_persists_layer_and_authority(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    store = VectorStore()
    vecs = np.ones((1, 256), dtype=np.float32)
    store.upsert_chunks("p1", "d1", ["hello world"], vecs,
                        layer="project_record", authority="contractual")
    got = store.search("p1", np.ones(256, dtype=np.float32), k=1)
    assert got and got[0].layer == "project_record"
    assert got[0].authority == "contractual"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vector_store_layer_roundtrip.py -q`
Expected: FAIL (`upsert_chunks() got an unexpected keyword argument 'layer'`).

- [ ] **Step 3: Write minimal implementation**

In `Chunk` dataclass add (next to the STEP 0 `layer` field):

```python
    authority: str | None = field(default=None, compare=False)
```

`upsert_chunks` signature → add keyword-only params and write them into each row's insert values (both the Postgres `insert().values(...)` path and the SQLite/ORM path). Example for the values dict per row:

```python
    def upsert_chunks(
        self,
        project_id: str,
        doc_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
        *,
        layer: Optional[str] = None,
        authority: Optional[str] = None,
    ) -> int:
        ...
        # per-row values (both backends): include the two columns
        row["layer"] = layer
        row["authority"] = authority
```

`search` row hydration → populate the fields, preferring the stored column, else the caller-supplied retrieval tag:

```python
        chunk = Chunk(
            chunk_id=row.chunk_id, project_id=row.project_id,
            doc_id=row.doc_id, chunk_index=row.chunk_index,
            text=row.text, score=score,
            layer=(getattr(row, "layer", None) or "own"),
            authority=getattr(row, "authority", None),
        )
```

(Match the actual local variable names in `search`; the SELECT must include the `layer, authority` columns.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vector_store_layer_roundtrip.py -q`
Expected: PASS.
Run (regression): `python -m pytest tests/test_step0_retrieval_isolation.py tests/test_chunks_layer_columns.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/rag/vector_store.py tests/test_vector_store_layer_roundtrip.py
git commit -m "feat(rag): thread layer/authority through upsert + search (stage 1/5)"
```

### Task 4: Soften the `project_id` cascade (schema + Alembic) — knowledge outlives workspaces

**Files:**
- Modify: `the_fork_schema.sql:194-195` (FK clause)
- Create: `alembic/versions/0012_chunks_project_fk_set_null.py`
- Test: `tests/test_chunks_survive_project_delete.py`

**Interfaces:**
- Produces: `chunks.project_id` becomes `NULL`-able with `ON DELETE SET NULL` (was `NOT NULL ... ON DELETE CASCADE`). A hard `DELETE FROM projects WHERE id=?` no longer deletes the project's chunks — it nulls their `project_id`; the chunk (text + embedding + layer + authority) survives.

- [ ] **Step 1: Write the failing test** (Postgres-only assertion; skips on SQLite where FK cascade isn't enforced the same way)

```python
# tests/test_chunks_survive_project_delete.py
import os, pytest
from sqlalchemy import text
from app.core.db import get_database_url, _engine_for_url

pg = get_database_url()
pytestmark = pytest.mark.skipif(
    not pg.startswith("postgres"),
    reason="cascade semantics are Postgres-specific; SQLite dev DB skips")

def test_chunk_project_fk_is_set_null():
    eng = _engine_for_url(pg)
    with eng.connect() as c:
        rule = c.execute(text("""
            SELECT rc.delete_rule
            FROM information_schema.referential_constraints rc
            JOIN information_schema.constraint_column_usage ccu
              ON rc.unique_constraint_name = ccu.constraint_name
            WHERE rc.constraint_name LIKE '%chunks_project%'
        """)).scalar()
        assert rule == "SET NULL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunks_survive_project_delete.py -q`
Expected: on Postgres FAIL (`CASCADE != SET NULL`); on local SQLite SKIP.

- [ ] **Step 3: Write minimal implementation**

`the_fork_schema.sql` chunks table — change:

```sql
    project_id  TEXT
                REFERENCES projects (id) ON DELETE SET NULL,
```

`alembic/versions/0012_chunks_project_fk_set_null.py`:

```python
"""chunks.project_id: CASCADE -> SET NULL (decouple knowledge from workspaces)

Revision ID: 0012_chunks_project_fk_set_null
Revises: 0011_chunks_layer_authority
"""
from alembic import op

revision = "0012_chunks_project_fk_set_null"
down_revision = "0011_chunks_layer_authority"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN project_id DROP NOT NULL")
    op.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_project_id_fkey")
    op.execute(
        "ALTER TABLE chunks ADD CONSTRAINT chunks_project_id_fkey "
        "FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE SET NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_project_id_fkey")
    op.execute(
        "ALTER TABLE chunks ADD CONSTRAINT chunks_project_id_fkey "
        "FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE"
    )
    op.execute("ALTER TABLE chunks ALTER COLUMN project_id SET NOT NULL")
```

Also relax ORM `project_id` to `nullable=True` in `app/core/models.py:433` so the ORM matches (`Mapped[str | None]`).

- [ ] **Step 4: Run test to verify it passes**

Run (against a Postgres test URL if available, else confirm SKIP locally): `python -m pytest tests/test_chunks_survive_project_delete.py -q`
Expected: PASS on Postgres / SKIP on SQLite.
Run (regression): `python -m pytest tests/test_step0_retrieval_isolation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add the_fork_schema.sql alembic/versions/0012_chunks_project_fk_set_null.py app/core/models.py tests/test_chunks_survive_project_delete.py
git commit -m "feat(rag): chunks.project_id CASCADE->SET NULL, decouple knowledge from workspaces (stage 1/5)"
```

### Task 5: Stage-1 gate — full suite green, migrations apply clean

- [ ] **Step 1:** Run the RAG + schema suites:
  `python -m pytest tests/test_rag_layers.py tests/test_chunks_layer_columns.py tests/test_vector_store_layer_roundtrip.py tests/test_chunks_survive_project_delete.py tests/test_step0_retrieval_isolation.py -q`
  Expected: all PASS/skip, zero fail.
- [ ] **Step 2:** Dry-run Alembic heads locally (SQLite): `python -m alembic heads` then `python -m alembic upgrade head` on a scratch DB URL; expected: `0012_chunks_project_fk_set_null (head)`, no error.
- [ ] **Step 3:** Confirm flag-off no-op: with `RAG_LAYERED` unset, `layered_enabled()` is False and no new column is required non-null. Commit any fixups.

---

## Stages 2-5 — concrete targets (bite-sized steps written at execution)

### Stage 2 — classification at ingest
- **Files:** `app/core/rag/layers.py` (add `classify(source_project_id, doc_name, doc_meta) -> tuple[layer, authority]`), `app/core/doc_index.py:1364,1667` (`_rag.index_chunks(...)` calls → pass `layer=`, `authority=`), wrapper `index_chunks` signature.
- **Map:** source `curated_kb`/GK → `shared_domain`; `example_infra_pack`/`drive_archive`/project docs → `project_record`; user upload → `user_session`; company templates/procedures → `company_rules`. Authority from doc-type keywords (contract/agreement→`contractual`; drawing/dwg→`design`; boq/rate/cost→`commercial`; method/procedure/policy→`policy`; report/log→`operational`; note/draft→`personal`; superseded/old→`historical`).
- **Acceptance:** a re-indexed doc lands with the expected `(layer, authority)`; flag-off path passes `None,None` (unchanged).

### Stage 3 — layer-aware retrieval + authority re-rank
- **Files:** `app/core/rag/retriever.py:413` (`retrieve_with_filter`) — when `layered_enabled()`, fetch the Main layer set + caller L3, then in the final re-rank add an authority-precedence term and an L2B-dominance boost for project-record questions; `tests/test_layered_retrieval.py`.
- **Acceptance:** for a project question, an L2B `contractual` chunk outranks an L1 `historical` chunk at equal cosine; flag-off reproduces today's ranking exactly (golden test).

### Stage 4 — user-upload layer + decoupled scope + disclosure
- **Files:** upload/index entrypoint (route uploads to `user_session` keyed by owner, id `user_<uid>` rather than the workspace), `retriever` scope = Main layers + `user_<caller>`, `app/agents/runtime.py` `_build_sources_from_audit` (label `user_session` as "Your upload"), `inject.py` audit.
- **Acceptance:** user A's upload retrievable in A's chats, absent in B's, never in Main; deleting the workspace leaves it retrievable (keyed by user, not project).

### Stage 5 — migration/backfill + cutover
- **Files:** `scripts/backfill_layers.py` (idempotent UPDATE by source project → layer/authority), `app/core/projects.py` (`hidden_from_sidebar` flag + list filter), Render env `RAG_LAYERED=1`.
- **Acceptance:** live corpus re-tagged (counts per layer reported), the 4 remaining system/eval rows hidden from sidebar yet still retrievable, grounded chat verified against Main + a test upload.

---

## Self-review

- **Spec coverage:** L1/L2A/L2B/L3 → layer enum (Task 1) + classification (Stage 2); authority scoring → authority enum + rank (Task 1) + re-rank (Stage 3); "project record dominates" → Stage 3 L2B boost; user/session memory + "never override project docs" → Stage 4 (L3 keyed by user, authority `personal` sorts weakest so it can't override `contractual`/`design`); decoupling from projects → Task 4 SET NULL + Stage 4 user-keying. Covered.
- **Placeholder scan:** none — every Stage-1 step has real code/commands. Stages 2-5 are explicitly deferred-detail with concrete file targets, per the staging note (not placeholders inside an active task).
- **Type consistency:** `layer`/`authority` are `str | None` in ORM, `Chunk`, and `upsert_chunks` throughout; `layered_enabled`/`authority_rank` names stable across tasks.
- **Scope:** Stage 1 is self-contained and shippable (columns + cascade + plumbing, all inert under flag-off). Good single-plan first increment; later stages layer on top.
