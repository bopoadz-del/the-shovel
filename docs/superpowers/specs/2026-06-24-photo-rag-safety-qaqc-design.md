# Photo RAG with Safety + QA/QC Detection — Design Spec

**Date:** 2026-06-24
**Status:** Brainstormed, awaiting operator review before plan
**Author:** Claude (via brainstorming skill, in collaboration with operator)

## Goal

Add construction site photos to The Fork's RAG so the platform can retrieve them by their visual content. V1 focus: detect six common safety violations and QA/QC defects in photos using a fine-tuned YOLOv8 model trained on the `construction-3-001.zip` corpus (208 files), then expose the resulting bbox + class metadata as first-class RAG chunks served by Render.

## Non-Goals (V1)

- Live video / camera streams
- Relational / multi-entity rules (water-next-to-cable, opening-not-closed-by-plywood)
- Subjective "quality degree" classes (uneven plaster, dirty false ceiling)
- Active-learning loop with operator-correction feedback
- Built-in labelling UI inside the platform
- Auto-redeploy of new fine-tuned weights to Render
- Project-context inference from photo visuals (NEVER — see `feedback-no-assumptions.md`)
- Vector embeddings for photo chunks (BM25 over caption + class names for V1)

## Architecture

**Two-host split:**

- **Local PC (operator's workstation, GPU available):** all training, all labelling, all batch inference. The PC owns the build pipeline end-to-end. The PC is NOT a runtime dependency — Render must function with the PC offline.
- **Render (cloud, CPU-only, srv-d8hdc6ek1jcs739rq5sg):** holds the produced metadata in Postgres, serves it as RAG chunks alongside existing document chunks. Never runs vision inference in V1.

**Connection:** one-shot HTTPS POST from PC to a new admin endpoint on Render after each training cycle.

**Why this split:** Render starter is $7/mo CPU-only; YOLO inference is ~100 ms/photo CPU and ~10 ms/photo GPU. No tunnel + no GPU on Render = clean and cheap. Operator's prior pushback against PC-tunnel dependencies (`feedback-agents-via-ollama` notwithstanding — that's LLM-specific) drove this choice. Phase 3 (live upload pathway) can revisit if real-time matters.

## Components (new)

| Component | Location | Purpose |
|---|---|---|
| `safety_classes.json` | `app/blocks/` | Class registry. All ~33 classes from operator's taxonomy. Each entry: `id` (int, stable), `name` (str), `category` (`safety`/`qaqc`), `definition` (str), `active` (bool), `weights_version` (str?), `min_examples_required` (int, default 30). V1 marks 6–10 active; rest are placeholders. |
| `survey_photo_corpus.py` | `scripts/` | Phase 0. Runs Grounding DINO (open-vocab detector) over a photo folder with class names as prompts. Outputs `data/training/photo_survey.json` with per-class detection counts. |
| `prelabel_with_dino.py` | `scripts/` | Phase 1a. Runs Grounding DINO on V1-active classes, exports Label Studio JSON for operator correction. |
| `train_safety_qaqc.py` | `scripts/` | Phase 1c. Thin wrapper over `ultralytics.YOLO.train()`. 80/20 split, fixed seed. Outputs `data/models/safety_qaqc_v{N}.pt` + `eval_v{N}.json` (mAP per class). |
| `infer_photo_metadata.py` | `scripts/` | Phase 2a. Runs PIL metadata + Tesseract OCR + COCO-YOLO (existing image.py) + fine-tuned safety/QA-QC YOLO on every photo. Outputs `data/training/photo_metadata.jsonl`. |
| `export_to_render.py` | `scripts/` | Phase 2b. Streams JSONL to Render admin endpoint with bearer auth, idempotent on SHA-256. |
| `app/blocks/safety_detector.py` | new module | Loads + serves the fine-tuned YOLO. Used by `infer_photo_metadata.py` on PC; registered in the block registry but unused at runtime in V1 (Phase 3 activation). |
| `app/routers/admin_photos.py` | new router | `POST /v1/admin/photo-import`: accepts JSONL of photo metadata, idempotent. Admin-auth gated. |
| Alembic migration `0006_photo_chunks_and_photos.py` | `alembic/versions/` | Creates `photo_chunks` table (`chunk_id` PK = sha256, `project_id` nullable, `sha256`, `caption`, `photo_metadata` JSONB, `created_at`) AND `photos` table (`sha256` PK, `content_type`, `size_bytes`, `bytes BYTEA`, `uploaded_at`). SQLite + Postgres branches. **Does NOT modify the existing `chunks` table** (chunks has NOT NULL `embedding` + `tsvector`; V1 photos have no embedding). |

## Components (extended)

- **`app/blocks/image.py`** — gains a `safety_qaqc` mode that delegates to `safety_detector`. Returns existing PIL + Tesseract + COCO output PLUS the fine-tuned model's bbox + class output. Uses the canonical `file_path` input key (the construction container's call sites were renamed from the legacy `image_path` key in a separate concurrent change — `safety_detector` must follow the same contract).
- **`app/containers/construction/__init__.py` and `documents.py`** — existing methods `safety_compliance_audit()` and `qa_qc_inspection()` already convert image-block output into hazard / defect lists with severity. Those methods stay. The new `safety_detector` produces typed class labels (e.g. `no_hardhat`) that compose into the same hazard / defect format these methods consume, replacing the brittle keyword-greping path on description text. Net effect: stronger detection signal flowing into the same downstream verdict logic.
- **`app/core/rag/retriever.py`** + **`app/core/rag/vector_store.py`** — current BM25 leg queries the `chunks` table. Extend so that after the existing BM25 over `chunks` runs, a parallel BM25 query runs over `photo_chunks` (caption text), and results from both merge into the same returned chunk list with `kind` set to `"text"` or `"photo"`. Photo chunks carry `photo_url=f"/v1/photos/{sha256}"` and the full `photo_metadata` blob. No embedding/vector leg for photos in V1. Postgres uses tsquery on `to_tsvector('english', caption)`; SQLite uses a new `photo_chunks_fts` FTS5 virtual table built the same way as `chunks_fts`.

## Data Flow

```
construction-3-001.zip
        |
        v  extract (Phase 0a)
data/training/raw_photos/
        |
        v  survey_photo_corpus.py        (Phase 0b)
data/training/photo_survey.json          ← per-class counts; locks V1 list
        |
        v  prelabel_with_dino.py         (Phase 1a, V1-active classes only)
data/training/labels_draft/              (Label Studio JSON)
        |
        v  operator corrects in Label Studio  (Phase 1b, manual)
data/training/labels_final/              (YOLO txt + images, 80/20 split)
        |
        v  train_safety_qaqc.py          (Phase 1c)
data/models/safety_qaqc_v1.pt
data/training/eval_v1.json
        |
        v  infer_photo_metadata.py       (Phase 2a)
data/training/photo_metadata.jsonl
        |
        v  export_to_render.py           (Phase 2b, HTTPS POST)
Render Postgres: photo_chunks rows + photos rows
        |
        v  RAG retrieval                 (Phase 2c, automatic)
queries matching class names return photo chunks with bbox metadata
```

### `photo_metadata.jsonl` row schema

```json
{
  "sha256": "abc123...",
  "filename": "IMG-20230523-WA0009.jpg",
  "source_zip": "construction-3-001.zip",
  "project_id": null,
  "pil": {
    "width": 4160,
    "height": 2336,
    "format": "JPEG",
    "file_size_bytes": 2451234,
    "dominant_channel": "red"
  },
  "ocr_text": "(empty or extracted text)",
  "coco_objects": [
    {"class": "person", "confidence": 0.91, "bbox": [x, y, w, h]}
  ],
  "safety_qaqc": [
    {"class_id": 0, "class": "no_hardhat", "confidence": 0.87, "bbox": [x, y, w, h], "category": "safety"},
    {"class_id": 4, "class": "concrete_honeycomb", "confidence": 0.74, "bbox": [x, y, w, h], "category": "qaqc"}
  ],
  "caption": "Site photo showing 1 worker without hard hat; concrete surface with honeycomb defect.",
  "inference_failed": false,
  "inference_error": null
}
```

**Why `project_id: null` for V1:** the zip has no confirmed project (operator hasn't bound it to one). Forcing a project_id would violate `feedback-no-assumptions.md`. Phase 3 populates this field from the active project when photos are uploaded through the platform UI.

**Caption generation:** templated from class labels + counts. Not a VLM call. Example template: `"Site photo showing {count_safety} safety issue(s): {safety_list}; {count_qaqc} QA/QC issue(s): {qaqc_list}."` If both lists are empty, caption is `"Site photo (no detected violations or defects)."`.

## Class Registry — V1 Activation Rules

`safety_classes.json` ships with all ~33 classes from the operator's taxonomy. Class IDs are **stable forever** — once assigned, an ID never gets reassigned even if a class is deprecated. The YOLO model only sees `active=true` classes; the active subset is renumbered 0..N-1 at training time via a class-id remap (stored in `data/models/safety_qaqc_v{N}_classmap.json`).

**Activation gate:** a class becomes `active=true` when survey shows ≥30 example bboxes. Below 30, the class stays in the registry as a placeholder; inference returns no detections for it.

**V1 active candidates (subject to survey confirmation):**

| ID | Name | Category |
|---|---|---|
| 0 | `no_hardhat` | safety |
| 1 | `no_high_vis_vest` | safety |
| 2 | `fall_hazard_unprotected` | safety |
| 3 | `concrete_crack` | qaqc |
| 4 | `concrete_honeycomb` | qaqc |
| 5 | `rebar_correct_inspection` | qaqc |
| 6 | `rebar_exposed_defect` | qaqc |

**Placeholder classes (active=false at V1):** all remaining 26 items from the operator's expanded list, IDs 7..32 reserved. Examples: `excavation_slope_unsafe`, `loose_sand_edge`, `missing_jersey_barrier`, `missing_lifeline`, `non_safety_shoes`, `faulty_scaffolding`, `missing_toe_board`, `no_scaffold_green_tag`, `unorganized_cables`, `cables_no_trunking`, `water_near_cable`, `ceiling_opening_unsealed`, `shaft_opening_unsealed`, `rebar_no_plastic_cap`, `uneven_tile_grouting`, `bulging_concrete`, `uneven_plaster`, `dirty_false_ceiling`, `uncentered_electrical_socket`, `no_chamfer_on_edge`, `column_misalignment`, `kerb_misalignment`, `dirty_pre_cast_surface`, `dirty_finished_room`. Final names + definitions locked during Phase 0 survey.

**Note on relational classes:** `water_near_cable`, `cables_no_trunking`, `ceiling_opening_unsealed`, `shaft_opening_unsealed` involve spatial reasoning between two or more detected objects. V1 keeps these as registered names only; the spec for activating them requires a separate predicate-rule layer on top of YOLO (out of scope for V1).

## RAG Integration on Render

**Migration `0006`:**

Creates two NEW tables; does NOT touch the existing `chunks` table (its `embedding` is NOT NULL + has a generated `text_search` tsvector — adding a kind-discriminated photo row would break both invariants).

```sql
CREATE TABLE photo_chunks (
    chunk_id        TEXT PRIMARY KEY,            -- = sha256 (one chunk per photo in V1)
    project_id      TEXT NULL,                   -- nullable; populated in Phase 3
    sha256          TEXT NOT NULL,               -- references photos.sha256
    caption         TEXT NOT NULL,
    photo_metadata  JSONB NOT NULL,              -- TEXT (JSON-encoded) on SQLite
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (sha256)
);

CREATE TABLE photos (
    sha256        TEXT PRIMARY KEY,
    content_type  TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    bytes         BYTEA NOT NULL,                -- BLOB / LargeBinary on SQLite
    uploaded_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Postgres branch additionally creates a GIN index on `to_tsvector('english', caption)` for BM25; SQLite branch creates a `photo_chunks_fts` FTS5 virtual table mirroring the `chunks_fts` pattern (handled by `vector_store._ensure_fts5_sqlite` extension in Task 2.7, not in the migration itself).

**Two endpoints (used in sequence by `export_to_render.py`):**

1. `POST /v1/admin/photo-bytes/{sha256}` — multipart upload. Stores the raw photo bytes in the `photos` table. Idempotent: existing `sha256` returns 200 without re-writing. Body limit: 25 MB per photo.
2. `POST /v1/admin/photo-import` — JSONL metadata.
   - Auth: bearer token, admin role only (uses existing admin middleware).
   - Body: JSONL stream of `photo_metadata` rows.
   - Behavior: idempotent on `sha256`. New rows inserted into `photo_chunks` with `chunk_id=sha256`, `project_id=row.project_id` (null for V1 zip), `sha256=row.sha256`, `caption=row.caption`, `photo_metadata=row` (full JSON blob). Existing rows with same `sha256` are skipped (no overwrite — operator must explicitly delete + re-import to update). Rejects rows whose `sha256` has no corresponding bytes in the `photos` table (ordering enforced).
   - Response: `{inserted: N, skipped_duplicate: M, rejected_no_bytes: K, errors: [...]}`

The exporter script uploads bytes for every photo first (skipping already-present sha256s), then posts the JSONL. Order matters because the metadata import refuses rows whose bytes don't yet exist on Render — preventing dangling references.

**Retriever change:** after the existing BM25 leg over the `chunks` table returns, a parallel BM25 query runs over `photo_chunks` (caption text). Results from both legs merge into a single returned chunk list with `kind` set to `"text"` or `"photo"`. BM25 score scale is consistent across both legs (same tsquery / FTS5 mechanism). No vector embedding for photos in V1.

**Citation format:** `[photo IMG-20230523-WA0009.jpg: no_hardhat (0.87), concrete_honeycomb (0.74)]` with a link to the photo (served via existing `/v1/files/{sha256}` endpoint — extended to serve the photo bytes, or a new `/v1/photos/{sha256}` if cleaner).

**Photo storage:** photos themselves live in a new `photos` table (defined in migration `0006`), keyed by `sha256` with `bytes BYTEA` (Postgres) / `BLOB` (SQLite). NOT object storage in V1 — keeps deployment simple; 208 photos × ~2 MB each = ~400 MB which fits the existing 1 GB disk plan. Phase 3 migrates to R2/S3 when corpus grows. Bytes are served back to the UI via `GET /v1/photos/{sha256}` (new endpoint, public-read with project-scope check against the `doc_index` row).

## Error Handling

- **Survey (Phase 0):** if Grounding DINO model/API unavailable, abort with a clear error pointing to install (`pip install groundingdino-py` or HuggingFace download instructions). Falls back to operator-only labelling if explicitly requested via `--no-prelabel`.
- **Training (Phase 1c):** validation gates that REFUSE to ship:
  - Any active class has fewer than 30 labelled examples → abort, list under-budget classes
  - Validation mAP@0.5 < 0.3 → ship the weights file but mark `model_grade: "experimental"` in metadata; export prevented unless `--force-low-quality` flag set
- **Inference (Phase 2a):** per-photo failures (corrupt image, PIL error, YOLO crash) are caught individually; failed photos exported with `inference_failed: true` + `inference_error: "ZeroDivisionError: ..."`. Pipeline does not abort on a single failure.
- **Export (Phase 2b):** retries with exponential backoff on 5xx; resumable (tracks `last_exported_sha256` in a state file); partial batches OK.

## Testing

| Test | Type | What it proves |
|---|---|---|
| `tests/test_safety_classes.py` | unit | Class registry parses; all IDs unique; active subset has weights_version set |
| `tests/test_label_format_roundtrip.py` | unit | Label Studio JSON → YOLO txt → re-import to LS keeps bboxes bit-identical |
| `tests/test_train_safety_qaqc_tinyset.py` | smoke | 5-photo, 1-class, 1-epoch training run produces a `.pt` file under 60s |
| `tests/test_infer_photo_metadata.py` | smoke | Run pipeline on 1 fixture image, assert output schema matches |
| `tests/test_admin_photo_import.py` | integration | POST JSONL → query `photo_chunks` → photo row present |
| `tests/test_retriever_finds_photos.py` | integration | Insert photo with `concrete_crack` in caption → query "show me cracks" → photo in top-K |

All tests mock the live LLM and live Grounding DINO calls. Real model invocation is reserved for manual smoke tests on the operator's PC.

## What Goes On Local PC vs Render

| Artifact | PC | Render |
|---|---|---|
| `construction-3-001.zip` (raw) | yes | no |
| Extracted photos | yes (`data/training/raw_photos/`) | no |
| Label Studio + labels | yes | no |
| Trained `.pt` weights | yes (`data/models/`) | no in V1 (Phase 3 may copy) |
| `photo_metadata.jsonl` | yes (build artifact) | yes (imported via admin endpoint) |
| Photo bytes (served as RAG sources) | yes (originals) | yes (uploaded with metadata) |

## Open Items for Operator (after spec lock, during implementation)

- Confirm Render disk has room for ~400 MB of photos (current plan: 1 GB starter — should be fine)
- Decide whether the new `photos` table sits in the existing Postgres or a separate schema
- Confirm Label Studio is installable on the PC (free OSS, `pip install label-studio`) or prefer Roboflow / CVAT
- During Phase 0 survey, decide final V1 class list once detection counts are visible

## Execution Postscript — 2026-06-24

The implementation diverged from the brainstormed spec during execution; recording here so future sessions and reviewers see ground truth rather than the original aspirations:

**Source corpus pivot.** Operator's 206-photo `construction-3-001.zip` was insufficient as the sole training corpus (sample read showed dominant PPE compliance + zero visible concrete defects). Operator approved pivot to public-dataset augmentation, one labelled dataset per V1 class. License gating was operator-waived ("don't worry about license").

**V1 shipped 5 classes, not 7.** `fall_hazard_unprotected` and `rebar_exposed_defect` were deferred because clean public sources within the 1 GB disk budget could not be located in the available time. V1 active classes:
- `no_hardhat` (id 0) — `keremberke/hard-hat-detection` (HF)
- `no_high_vis_vest` (id 1) — `keremberke/construction-safety-object-detection` (HF, only ~32 NO-vest examples in source)
- `concrete_crack` (id 3) — Ultralytics `crack-seg.zip` (polygon→bbox)
- `concrete_honeycomb` (id 4) — `jdkuhnke/HiC` HiCIS/web subset (GitHub API)
- `rebar_correct_inspection` (id 5) — `tsrobcvai/ROI-1555` (HF, LabelMe polygon)

**Multi-round training pattern.** Operator requested "set by set stop at 1 giga, train and delete" disk-management. Implemented as `scripts/run_safety_qaqc_round.py` round driver: merge → train → save weights → delete. **Side effect:** each round produces an independent class-subset model (YOLO re-init head when class count changes). The final V_final round retrained from yolov8n.pt on all 5 simultaneously — that is the canonical V1 model.

**Class swap for V2.** Per operator 2026-06-24, `rebar_exposed_defect` (id 6) is permanently inactive; ID 6 stays reserved per the stable-ID rule. `bulging_concrete` (id 22) takes its slot in the V2 active set. V2 will train 7 classes (the 5 V1 + `fall_hazard_unprotected` + `bulging_concrete`).

**Held-out test split (new).** `scripts/merge_training_corpus.py --test-per-class N` reserves N images per class as never-seen test set; `scripts/run_safety_qaqc_round.py --test-per-class N` runs `model.val()` on it after training. Added 2026-06-24 after the operator pointed out we should have done this from V1.

## Phase Sequencing

| Phase | What ships | Operator action required |
|---|---|---|
| 0 | survey tool + class registry + Render migration | Run survey on PC, review counts, lock V1 classes |
| 1 | pre-label tool + training script + Label Studio workflow | Run pre-label on PC, correct in Label Studio, run training |
| 2 | inference script + export script + admin endpoint + retriever change | Run inference on PC, run export to Render, verify RAG retrieval |
| 3 (separate spec) | active-project upload pathway, optional Render-side runtime inference, active-learning loop, VLM caption pass | Out of scope for this design |

Each phase is independently shippable. Operator decides whether to gate phases behind manual review or auto-advance.
