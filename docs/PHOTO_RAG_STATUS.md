> **SUPERSEDED (2026-09-28).** Historical snapshot. Not a source of open items or current status. Current state lives in the fleet standing-order loop and the live measurements.

# Photo RAG — Status (2026-06-24)

> **SUPERSEDED (2026-07-18).** The `photo_chunks` table and the photo-RAG BM25
> leg described below were **removed in Alembic migration 0008**. Chat-attached
> photos are now question-context (`POST /v1/chat/analyze-photo`), not corpus
> material, and site-photo analysis runs through the Safety Observation AI
> (YOLO-World v2 ONNX). This document is retained for history only; the
> Phase-2 platform-integration tasks below did not ship as written.

Single source of truth for what shipped, what's running, what's deferred.

## Shipped

### Foundation (Phase 0, plan complete)

- **Class registry** — `app/blocks/safety_classes.json` (33 entries) + `app/blocks/safety_classes.py` loader. Stable IDs 0–32; `active` flag flips per round.
- **Alembic migration `0006_photo_chunks_and_photos.py`** — creates two new tables, modifies nothing existing.
- **Grounding DINO survey script** — `scripts/survey_photo_corpus.py` (used once for Phase 0 reconnaissance; produced findings that drove the pivot to public-dataset augmentation).

### Trained models (in `data/models/`)

| File | Classes | Epochs | Val mAP@50 | Notes |
|---|---|---|---|---|
| `safety_qaqc_v1_r1.pt` | 3 | ~2 (truncated) | ~0.236 | First incremental attempt; obsolete |
| `safety_qaqc_v1_r2.pt` | 2 | 10 | 0.272 | Different classes than r1; obsolete |
| `safety_qaqc_v1_r3.pt` | **5** | **20** | **0.517** | **Production V1** |

**V_final per-class on val set:**

| Class | mAP@50 | Notes |
|---|---|---|
| `no_hardhat` | 0.924 | Strong — production-ready |
| `no_high_vis_vest` | 0.533 | Decent |
| `concrete_crack` | 0.497 | Decent |
| `concrete_honeycomb` | 0.276 | Weak — only 9 val imgs; known FP on rocky ground (HiC training had whole-image labels, not per-bbox) |
| `rebar_correct_inspection` | 0.355 | Weak — domain gap (trained on aerial UAV, our corpus is ground-level) |

**V_final real-world test on operator's 206 photos** (conf=0.25):
- 66/206 photos got ≥1 detection
- Counts: `no_hardhat`=4, `no_high_vis_vest`=54, `concrete_crack`=2, `concrete_honeycomb`=73, `rebar`=0
- Confirmed false positive: top honeycomb detection (0.975) was rocky outdoor ground (no concrete in frame)

### Scripts shipped

| Script | Purpose |
|---|---|
| `scripts/survey_photo_corpus.py` | Grounding DINO survey (Phase 0) |
| `scripts/merge_training_corpus.py` | External per-class data → unified YOLO train/val (+ optional held-out test) |
| `scripts/run_safety_qaqc_round.py` | Driver: merge → train → save weights → delete data (+ optional held-out eval) |
| `scripts/test_model_on_corpus.py` | Run a model on a folder of photos; produce JSONL + summary |
| `scripts/compare_models.py` | Side-by-side per-class metrics for N weight files on same dataset |
| `scripts/compare_corpus_detections.py` | Delta two corpus-detection JSONLs (new/lost detections, conf shifts) |

### Platform code shipped

- `app/blocks/safety_detector.py` — loads a YOLO `.pt`, maps YOLO indices to registry IDs, returns structured detections. `default_detector()` reads `SAFETY_DETECTOR_WEIGHTS` env var. 5 tests passing.

## In flight

- **V2 data agent** `abc8c6ab97ac3b17b` — re-fetching the 5 working V_final sources after the static `redownload_v1_sources.py` script failed (since-deleted).

## Pending

- **V2 training** — `python scripts/run_safety_qaqc_round.py --round 4 --epochs 20 --test-per-class 25` once data lands.
- **V2 vs V_final A/B** — `compare_models.py` on same held-out test (V2 will be the first round with held-out test).
- **V2 real-world delta** — `compare_corpus_detections.py` between V_final and V2 on operator's 206.
- **Phase 2 platform integration** (not started): Task 2.4 (`infer_photo_metadata.py`), Task 2.5 (admin endpoints on Render), Task 2.6 (`GET /v1/photos/{sha256}`), Task 2.7 (retriever photo_chunks BM25 leg), Task 2.8 (`export_to_render.py`).

## Deferred

- **`fall_hazard_unprotected`** (class id 2) — first round agent mapped PPE-absence images to it (semantically wrong, would pollute hardhat/vest classes); operator dropped from V2. Future: need real harness or edge-protection labelled data.
- **`rebar_exposed_defect`** (class id 6) — permanently retired by operator 2026-06-24 ("forget the rebar exposed pick another"). ID 6 stays reserved per the stable-ID rule.
- **`bulging_concrete`** (class id 22) — Roboflow source had real bboxes but they're gated behind an API key; only placeholder whole-image boxes were public. Operator dropped from V2 to avoid the same FP issue as honeycomb.

## Known issues

1. **Placeholder bbox labels train weak localization.** HiC honeycomb data has whole-image labels; model learned "rough texture = honeycomb" instead of "honeycomb on cured concrete." Same risk for any future class sourced from image-classification datasets — must filter for real per-instance bboxes.
2. **`rebar_correct_inspection` domain gap.** Public sources are mostly aerial UAV views; operator's corpus is ground-level. Detector gets 0 hits on the 206. Needs ground-level rebar data to fix.
3. **`tsrobcvai/ROI-1555` HF dataset is now gated** (was public during V_final). V2 agent will pick an alternative.
4. **Some V2 agents went down rabbit holes** — Open Images full dataset, CODEBRIM 7-12 GB. Solved by tighter agent instructions with explicit "don't go there" warnings.

## Operator-driven preferences captured this session

- Augment training corpus with public datasets per class — operator's 206-photo corpus alone is insufficient.
- Don't worry about dataset licenses for V1 (research/training context).
- "Set by set stop at 1 GB, train and delete, keep going" — incremental rounds with disk cap.
- Keep going autonomously during long background tasks — don't prompt for "keep going" check-ins.
- Reserve per-class held-out test set for honest measurement (added in V2).
- `rebar_exposed_defect` permanently dropped; `bulging_concrete` was the proposed swap, then dropped due to label quality.
