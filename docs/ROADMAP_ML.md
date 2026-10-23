# Plan: Applied ML → RAG → PyTorch (Sequenced Rollout)

## Context

The Fork already does a lot that *looks* like ML — `learning_engine` fits linear regressions on correction pairs via `numpy.linalg.lstsq`, `vector_search` uses scikit-learn's `TfidfVectorizer`, `chat` routes between cloud LLM providers — but the actual learning loop is open. Hydration (just merged into `learning_engine` via the writeback to `projects.set_fact` / `agent_memory.set_agent_fact` / `_record_pattern`) put rich training data on disk; nothing trains on it yet.

The previous version of this plan addressed five named weaknesses (chat router as model registry, orchestrator with learned classifier, image block with local CV tier, learning engine as spine, Dockerfile with CUDA). That work remains the right substrate. This rewrite re-organizes it through the **Applied ML → PyTorch → LLM Apps** skill lens the user requested, locks in the user's three direction-setting answers, and adds an explicit ignore list so we don't drift into the buffet of techniques the field offers.

**User direction (locked):**

| Question | Answer |
|---|---|
| Hardware target | **Cloud / workstation GPU** — full PyTorch on tap, no quantization-first thinking, no edge constraints |
| Iteration scope | **All three sequenced** — Applied ML lands this PR; RAG and PyTorch are planned but follow-on |
| First model | **Chat-routing classifier** — replace `smart_orchestrator`'s 39-keyword regex with a learned dispatcher |

**Intended outcome:** the data hydration is already producing (patterns, corrections, conversation history) starts driving a real classifier this PR. By PR 3, the system has persistent RAG over project docs and a small fine-tuned model for the construction domain. Each PR ships standalone; nothing here requires Orin / arm64 / edge runtime (still deferred), but PR 3 explicitly bakes in **Orin-portability** so the future re-port is a swap, not a rewrite (see the "Orin-portability invariant" inside PR 3 below).

**Companion action (PR 1 commit):** drop this file into the repo at `docs/ROADMAP_ML.md` so it survives container reclamation and is reviewable alongside the code. The plan-file copy at `/root/.claude/plans/lets-get-real-for-precious-dream.md` stays as the working draft; the repo copy is the canonical artifact. Add to PR 1's first commit.

---

## PR 1 — Applied ML: learned chat routing (this iteration)

**The smallest valuable slice.** Pure scikit-learn, TF-IDF features, no embeddings, no GPU. Proves the loop end-to-end: data on disk → trained model → routing decision → feedback → retrain. Once this loop exists, swapping in better representations (PR 2) and bigger models (PR 3) is a backend change, not a redesign.

### Critical files

- **`app/blocks/smart_orchestrator.py`** — owns the existing 39-keyword regex. Gets a new optional `routing_mode` ("keyword" default, "learned" opt-in). In learned mode, consults `learning_engine` via the registry; falls back to keyword when confidence < threshold so we never regress on a query the regex would have handled.
- **`app/blocks/learning_engine.py`** — grows three operations alongside the existing 13: `train_router`, `predict_route`, `route_metrics`. Pattern mirrors `hydrate`/`hydration_latest`/`hydration_history` — thin dispatcher delegating to heavy logic in `app/core/learning/router.py`.
- **`app/core/learning/router.py`** (new) — sibling to the just-created `app/core/learning/hydration.py`. Functions: `collect_training_data()`, `train()`, `predict(text)`, `evaluate()`. Model is `sklearn.pipeline.Pipeline([TfidfVectorizer, LogisticRegression(class_weight='balanced')])`. Persisted via `joblib.dump` to `$DATA_DIR/learning/router_model.joblib`; metadata (train date, sample count, accuracy, label distribution) lives in `learning_engine`'s state JSON under a new `"models"` key.
- **`app/core/learning/data_extraction.py`** (new) — reads `agent_memory.db` conversations + `learning_engine._state["patterns"]` and produces labeled `(message, block_name, label_quality, source)` rows. Labels come from two sources: (a) historical `_op` field on saved assistant messages (which block actually ran — marked `label_quality="auto"`), (b) explicit user corrections (W4-style feedback, optional this PR — marked `label_quality="corrected"`).

  **Label noise is acknowledged and tracked.** The `_op` field records what the keyword router *did* dispatch, not what the *correct* block was. Training on the keyword router's outputs without separating these means the model can inherit its mistakes — especially on novel phrasings the regex botches. The `label_quality` field is the lever: PR 1 trains on `"auto"` + `"corrected"` at equal weight (because we don't have enough `"corrected"` yet to train on alone), but `train_router` accepts a `prefer_corrected: bool` param that, once sample volume allows, fits only on `"corrected"` or weights them higher. **Caveat to document in the README and the `route_metrics` response: "not yet self-correcting — the model inherits the keyword router's bias proportional to its training mix; corrections accumulate to fix this over time."**

  **Synthetic data — conditional, not banned.** When the labeled set is too small (rule of thumb: <100 examples total, or <10 examples for any single class), generate paraphrases via the existing chat block: feed each real example with the prompt "produce 5 distinct paraphrases that preserve intent and domain vocabulary." Mark synthetic rows with `source="paraphrase"` and `label_quality="auto"` in the training data so they're filterable, downweight them with `sample_weight=0.3` at fit time (lower than the 0.5 I first wrote — paraphrases of `auto`-labeled data are doubly noisy: noisy label × noisy paraphrase), and **never include them in the holdout set** (paraphrase contamination would inflate accuracy by giving the model near-copies of train rows at test time). Drop synthetic rows automatically once real per-class sample count crosses the threshold.
- **Probability calibration is required, not optional.** Wrap the fitted classifier in `sklearn.calibration.CalibratedClassifierCV(method="sigmoid", cv=5)` — sigmoid (Platt scaling) is correct for the data volumes we expect (a few hundred samples per class); isotonic only outperforms it at >1000/class. Without calibration, the "confidence < threshold → fallback to keyword router" decision is reading raw softmax outputs that don't correspond to actual probabilities — the fallback threshold becomes a vibe, not a signal. With calibration, `predict_proba()[0] > 0.6` means roughly "60% chance the label is correct" in the empirical sense.
- **Joblib ↔ state JSON integrity.** Two files can drift: the joblib model artifact and the metadata in `_state["models"]["router"]`. At load time, `predict_route` checks: (a) joblib file exists at the recorded path, (b) the file's mtime/sha256 matches the recorded fingerprint, (c) the recorded label set matches the model's `.classes_`. Any mismatch → log warning, force keyword-router fallback, surface in `route_metrics` as `model_loaded: false, reason: "..."`. No silent "model exists" claims when only metadata survives.
- **Loaded model lives on the block instance, not reloaded per request.** The learning_engine block follows a singleton-ish lifecycle (one instance per registry lookup); the joblib model is cached on first `predict_route` call and reused thereafter. `train_router` invalidates the cache when it rewrites the artifact. This avoids 10–50ms of joblib deserialization per chat turn.
- **No coupling to `vector_search.py`'s global state.** We use `TfidfVectorizer` from `sklearn.feature_extraction.text` directly — instantiated locally inside `router.py`'s Pipeline. We do NOT import `vector_search`'s shared vectorizer (which holds a fitted vocabulary keyed to that block's domain); reusing it would silently couple routing accuracy to whatever last touched `vector_search`. One vectorizer per concern.
- **`app/blocks/__init__.py`** — no new block; this PR adds operations, not surface area. Same architectural choice as the hydration merge.
- **Tests** — `tests/test_router_ml.py`:
  - `test_train_and_predict_happy_path` — train on synthetic 5-class corpus, assert top-1 accuracy > 80% on holdout, assert predict returns `{block, confidence, fallback_used}`.
  - `test_calibrated_probabilities` — Brier score on holdout < 0.25 (calibration sanity check, not a tight bound).
  - `test_holdout_excludes_paraphrases` — assert no paraphrase-sourced rows leak into holdout.
  - `test_label_quality_field_present` — every training row has `label_quality ∈ {"auto","corrected"}`.
  - `test_fallback_when_model_absent` — delete the joblib file; assert smart_orchestrator falls back to keyword routing and surfaces `model_loaded: false`.
  - `test_fallback_when_metadata_drifts` — write the joblib but corrupt the recorded fingerprint; assert integrity check trips and falls back.
  - **`test_no_regression_with_keyword_fallback`** — feed sentences the keyword router gets right but the model is uncertain about (force low-confidence prediction); assert fallback kicks in and the keyword router's correct answer is returned. This is the testable form of the "never regress on what worked" promise.
  - `test_model_cached_on_instance` — call predict twice in a row, assert joblib is loaded from disk once (mock `joblib.load` to count calls).

### Reuse, don't rebuild

- `TfidfVectorizer` is already in `app/blocks/vector_search.py` — confirm the same import path works; do not create a parallel one.
- `learning_engine._save_state` already handles JSON persistence; the new `"models"` key piggy-backs on it.
- `agent_memory.list_conversations` + `list_messages` are the read-side primitives for `data_extraction.py`. No new SQL.
- The existing pickle-vs-joblib choice has precedent: `formula_executor` uses no model artifacts; `vector_search` keeps its TF-IDF matrix in memory. We introduce joblib for the first real artifact — file under `$DATA_DIR/learning/` so the existing `isolated_data_dir` test fixture isolates it.

### Acceptance for this PR

- `pytest tests/test_router_ml.py` green.
- `curl POST /v1/execute {"block":"learning_engine","input_data":{"operation":"train_router"}}` returns `{status:"success", samples_used, accuracy, label_distribution}`.
- `curl POST /v1/execute {"block":"smart_orchestrator","input_data":{"text":"how many guys on site?","routing_mode":"learned"}}` returns the same answer the keyword router would have for an in-vocabulary query, and a confidence-aware fallback for novel phrasing.
- `learning_engine` state file shows `"models": {"router": {trained_at, sample_count, accuracy, labels}}`.
- Net new dependencies: **zero** (sklearn + numpy + joblib all already installed).

---

## PR 2 — LLM Apps: persistent RAG (real vector DB + embeddings)

**Promoted from "in-memory TF-IDF only" to a persistent retrieval layer.** The chat router stops being a pure proxy; it consults indexed project docs before answering. The PR 1 chat-routing classifier is upgraded from TF-IDF features to embedding features (drop-in via the same `Pipeline` interface).

### Critical files (planned, not implemented this iteration)

- **`app/core/rag/`** (new package) — `embeddings.py` wraps `sentence-transformers/all-MiniLM-L6-v2` (small enough to keep CPU-OK; GPU when available). `vector_store.py` uses `sqlite-vec` (single dependency, single SQLite file, no Docker, no service). `retriever.py` exposes `retrieve(query, project_id, k=5) -> List[Chunk]`.
- **`app/blocks/chat.py`** — pre-LLM step: call `retriever.retrieve()`, inject top-k chunks into the system prompt under "Relevant project context:". Behavior is opt-in via request param so existing chat behavior is unchanged for callers that don't pass it.
- **`app/blocks/doc_index.py`** — already indexes documents; extends to also produce embeddings and write to `sqlite-vec` alongside the existing index. Idempotent per fingerprint (same pattern as today).
- **`app/core/learning/router.py`** — swap `TfidfVectorizer` for `EmbeddingVectorizer` (sklearn-compatible wrapper around sentence-transformers). Re-train. PR 1 logreg classifier becomes more accurate without code change downstream.

### New dependencies

`sentence-transformers`, `sqlite-vec`. Both pip-installable, no external service. ~100MB for the MiniLM model (cached to `~/.cache/huggingface/`).

**Platform-wheel caveat (worth flagging now):** `sqlite-vec` ships pre-compiled wheels for x86_64 Linux, macOS Intel, and Windows. **For Apple Silicon (M1/M2) and arm64 Linux (including the future Orin), pip falls back to building from source** — needs `sqlite3-dev` and a C compiler. The Dockerfile's CPU image (already in the prior plan's Workstream 5) covers x86 fine; arm64 builds need the toolchain. We surface this in `requirements-rag.txt` with a comment, not by switching libraries. Alternative (FAISS) has worse footprint and a separate index file; sqlite-vec wins on simplicity for our scale.

### Acceptance criteria (when this PR lands)

- Chat answer cites the document it retrieved from.
- Router classifier accuracy improves over PR 1's TF-IDF baseline (measure via the same holdout).
- Indexed documents are queryable through `POST /v1/rag/search` (new route).

---

## PR 3 — PyTorch: domain fine-tune + local CV tier

**The deep-learning layer.** Two distinct workstreams; can ship together or separately.

### Orin-portability invariant (applies to all PR 3 code)

The hardware-target answer ("cloud GPU") sets the **training** environment, not the deployment ceiling. The Orin is still the endgame — when it arrives, the only PR 3 artifact that should need touching is the inference stack (quantize + TensorRT-compile), not the training code or the model formats. To make that re-port cheap:

- **No hardcoded `.to("cuda")`, no hardcoded `device=0`.** Every device reference goes through a single helper: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Models load to `device` once; tensors follow.
- **Save in HuggingFace format only.** LoRA adapters as `adapter_model.safetensors` + `adapter_config.json`. Full fine-tunes (if any) as `safetensors`. These are trivially quantizable later via `bitsandbytes` (4-bit / 8-bit) or convertible to GGUF for `llama.cpp`-class runtimes — both of which fit the Orin path.
- **YOLO weights kept in `ultralytics`-native `.pt` format**, not torch-script. Ultralytics has a documented TensorRT export path; torch-script does not survive that conversion cleanly.
- **No vLLM, no Triton, no Ray Serve in PR 3.** The ignore list already excludes them; restating here because the convenience of `vllm.LLM(...)` is real and tempting on a cloud GPU. Plain `transformers.pipeline` is the chosen inference path so the Orin port is a swap, not a rewrite.
- **Document the quantization step that DOES NOT ship in PR 3.** A one-page `docs/EDGE_PORT.md` (added in PR 3) names the future commands: `python -m bitsandbytes.convert_4bit ...`, `yolo export model=best.pt format=engine`, the exact files to ship to the Orin. The point is to make the future work obvious, not to do it now.

### 3a — Fine-tune a small instruction-tuned model on construction Q&A

- Base model: `Qwen2.5-3B-Instruct` or `Llama-3.2-3B-Instruct` (both fit comfortably on a workstation GPU; permissive licenses).
- Method: **LoRA** via `peft` — full fine-tune is overkill for the data we have. Training data: exported corrections + hydration patterns + a curated seed set. Script: `scripts/finetune_router.py` (new), uses `transformers` + `accelerate`.
- Output: HuggingFace-format LoRA adapter at `$DATA_DIR/learning/adapters/construction_v1/`. Chat block grows a `local_model` capability flag; when enabled, loads base+adapter via `transformers.AutoModelForCausalLM`.

### 3b — Local CV tier on site photos

- Already planned in the prior "Workstream 3" (tiered image pipeline). YOLOv8-nano via `ultralytics` for PPE/people detection, Tesseract OCR for text-bearing images, Claude Vision only as fallback.
- New: a small CNN head fine-tuned on the user's actual labeled site photos (collected via the feedback loop from PR 1's groundwork). PyTorch + `torchvision`. ResNet-18 backbone, frozen until accuracy plateaus, then unfrozen.

### New dependencies (this PR)

`torch`, `transformers`, `accelerate`, `peft`, `ultralytics`, `torchvision`. Heavy — adds ~3GB to the install. Gated behind a `requirements-ml.txt` so dev installs stay slim.

### Acceptance criteria (when this PR lands)

- `python scripts/finetune_router.py --epochs 3 --lora-r 16` completes on a GPU workstation and writes an adapter.
- Loading the adapter at chat startup answers a construction-specific test prompt with domain-grounded language (not a generic LLM answer).
- YOLO tier-1 detects PPE in a sample site photo in <500ms on CPU, no Claude Vision call.

---

## What to IGNORE — explicit, scoped to "cloud GPU + chat-routing-first + sequenced"

These are real techniques the field uses; they're being deliberately deferred or skipped for this codebase right now.

### Skip from "Applied ML"

- **The sklearn buffet** — Decision trees, SVM, KNN, Naive Bayes. We commit to LogisticRegression as the baseline and (only if it underperforms) one boosting library (XGBoost OR LightGBM, not both). The breadth-first sklearn tour is a teaching exercise, not a product decision.
- **t-SNE, UMAP, PCA-for-visualization** — No customer is asking for dimension-reduction plots. PCA as a preprocessing step inside a pipeline is fine when it earns its place; the visualization end is noise.
- **GridSearchCV / RandomizedSearchCV** — Premature hyperparameter tuning. With <10k labeled samples, defaults win.
- **Cross-validation when N is small** — Use a single deterministic train/holdout split until we have >5k labeled samples. K-fold then.
- **ROC-AUC obsession** — Multi-class chat routing isn't a binary problem; macro-F1 + per-class precision/recall is the right scoreboard. ROC curves go in a notebook, not the dashboard.

### Skip from "PyTorch"

- **Writing networks from scratch** — No custom CNNs, no custom transformer blocks. Always start from a pretrained checkpoint via `transformers` or `torchvision`. The "build it from scratch" curriculum is for learning the field, not shipping.
- **RNN / LSTM / GRU** — Transformers replaced them. Chat-routing isn't sequential at the token level for our purposes anyway.
- **Mixed-precision training, distributed training, FSDP** — Single workstation GPU is the target. Don't optimize for scale we don't need.
- **Reinforcement learning, RLHF, DPO** — Not on the menu. The signal we have is corrections, not preference pairs.
- **Quantization (GGUF / AWQ / GPTQ / bitsandbytes 4-bit)** — Cloud GPU target. Quantization re-enters when the Orin does (deferred per the prior plan).

### Skip from "LLM Applications"

- **vLLM, TGI, Triton serving** — Overkill until we have a fine-tuned model AND serve enough traffic to justify a dedicated inference server. For PR 3 the `transformers` pipeline is fine.
- **ONNX export / TensorRT** — Same reason. Edge deployment is deferred.
- **Multi-agent frameworks (LangGraph, CrewAI, AutoGen)** — The existing agent/swarm block already does function-calling. Adding a second framework adds confusion, not capability.
- **Prompt-versioning systems (Promptfoo, LangSmith)** — Until we have prompt-quality metrics that someone other than the author looks at, this is YAGNI. The PR 1 router gives us actual routing accuracy; that's a better signal than prompt diff trackers.
- **Full LLM evaluation harnesses (lm-eval-harness, HELM)** — We're not publishing benchmarks. Custom F1 on our own holdout is the only metric that matters here.
- **MLflow / Weights & Biases / Neptune** — One TensorBoard log per training run is enough until we have enough runs to need a tracker.

### Skip permanently (architectural choices, not deferrals)

- **Big custom embedding training** — Use pretrained sentence-transformers. Training embeddings from scratch on our data is dollars-per-1%-improvement territory.
- **Building a feature store** — We have one project, one chat path, one classifier. A feature store is six PRs from now if ever.
- **Adversarial robustness, model cards as a deliverable, fairness metrics** — Important in domains where they apply (lending, healthcare). Construction chat routing doesn't trigger these. We will not pretend otherwise.

---

## Known limitations & honest caveats (PR 1)

These are written into `route_metrics` output and the README so the system never overclaims:

- **Not yet self-correcting.** PR 1's classifier learns from the keyword router's historical decisions; until the `"corrected"` label fraction grows, the model can only equal — not exceed — the regex's ceiling on misrouted novel phrasings. The plan to fix this is mechanical (W4-style feedback buttons → `label_quality="corrected"` rows → `prefer_corrected=true` retrain), not architectural.
- **Data-volume floor.** With <40 real labeled samples total, `train_router` returns `{status: "insufficient_data", samples_used, threshold_needed}` and does not train. The fallback chain (paraphrase-augment → train → calibrate) only activates above that floor.
- **Pagination not implemented for `agent_memory.list_conversations`.** PR 1's training set is small enough to fit in memory; if conversations grow to tens of thousands of long histories, `data_extraction.py` will need batching. Flagged as a follow-up, not a launch blocker.
- **No online learning.** The model is static between explicit `train_router` calls. A nightly cron (or piggyback on the existing hydration scheduler) is the right time to retrain — added when `"corrected"` rows accumulate, not in PR 1.

## Architectural invariants (carry across all three PRs)

These were established by the recently-merged hydration work and must hold:

1. **All learning state on `learning_engine`.** New training artifacts (router model, embeddings index metadata, fine-tune adapters) register through `learning_engine._state` or via paths under `$DATA_DIR/learning/`. We do not create parallel state stores.
2. **Every writeback is best-effort.** Failures don't abort the parent pass; they get recorded in a `skipped` list. Same pattern as `_writeback_for_next_chat`.
3. **`isolated_data_dir` fixture pattern.** Every new test that touches the filesystem uses the pattern we just established — fresh `DATA_DIR` per test, module-level `_initialized` flags reset. Apply to `tests/test_router_ml.py` and onward.
4. **No new top-level block when an operation will do.** Hydration moved from standalone block to operation on `learning_engine`. Router training follows the same path. New blocks only when the surface area genuinely warrants one (PR 2's RAG retriever is a candidate; PR 3's local model loader is another).

---

## Verification (PR 1 specifically)

End-to-end check before merge:

1. `pytest tests/test_router_ml.py tests/test_hydration.py tests/test_learning_engine.py -v` — all green.
2. Seed `agent_memory.db` with 50 conversations across 5 blocks (boq_processor, spec_analyzer, drawing_qto, primavera_parser, chat).
3. `curl POST /v1/execute {"block":"learning_engine","input_data":{"operation":"train_router"}}` — assert `samples_used >= 40`, `accuracy >= 0.7` on holdout.
4. `curl POST /v1/execute {"block":"learning_engine","input_data":{"operation":"route_metrics"}}` — assert per-label precision/recall present.
5. `curl POST /v1/execute {"block":"smart_orchestrator","input_data":{"text":"extract BOQ totals","routing_mode":"learned"}}` — assert `routed_to: "boq_processor"`, `confidence > 0.5`.
6. `curl POST /v1/execute {"block":"smart_orchestrator","input_data":{"text":"gibberish that no block handles","routing_mode":"learned"}}` — assert `fallback_used: true`, `routed_to` matches the keyword fallback.
7. State JSON inspection: `cat $DATA_DIR/learning/learning_engine_state.json | jq '.models.router'` — confirm train_date, sample_count, accuracy, labels populated.

---

## When the Orin arrives (still deferred)

Unchanged from the prior plan. Re-enable the previously-listed quantization/ONNX/arm64/JetPack work then; nothing in PRs 1–3 above blocks or invalidates it. The fine-tune adapter from PR 3a transfers directly via a quantization step; the YOLO model from PR 3b transfers via TensorRT compilation.
