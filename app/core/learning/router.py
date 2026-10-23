"""Learned chat-routing classifier — sklearn LogisticRegression on TF-IDF.

Predicts a ``smart_orchestrator`` action name from a user message. Sits on
top of the existing keyword router as an optional second tier:
``smart_orchestrator`` consults this when ``routing_mode="learned"``, and
falls back to the keyword regex when confidence is below threshold or the
model is not loaded.

The "learning" here is honest:

- Day-1 seeds come from ``smart_orchestrator.ACTION_PATTERNS`` itself —
  each keyword variant becomes a synthetic training example. The
  classifier starts at parity with the keyword router, no better. This is
  what makes PR 1 ship without waiting for production traffic.
- Runtime examples accumulate via ``smart_orchestrator``'s
  ``_record_pattern`` hook under ``category="routing_decisions"``. Each
  real dispatch becomes a training row for the next retrain.
- User corrections (when they exist) carry ``label_quality="corrected"``
  and can be weighted higher or used exclusively via the
  ``prefer_corrected`` switch once volume permits.

The model artifact (joblib) is the persistent learning state; metadata
lives on ``learning_engine._state["models"]["router"]``. Integrity-checked
on load: file present, sha256 matches, label set matches model.classes_.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────

# Below this total real sample count, train_router refuses and asks for more
# data. The synthetic-seed bootstrap means we're always above the floor on
# day 1, but the floor exists for the case where someone resets state.
_MIN_TOTAL_SAMPLES = 40

# Below this per-class count, paraphrase augmentation kicks in.
_MIN_PER_CLASS_BEFORE_PARAPHRASE = 10

# Confidence threshold: below this, smart_orchestrator's learned-mode dispatch
# falls back to the keyword regex. The plan calls this out as the "never
# regress" lever; the value is conservative on purpose.
DEFAULT_CONFIDENCE_THRESHOLD = 0.45


def _model_path() -> str:
    base = os.getenv("DATA_DIR", "./data")
    d = os.path.join(base, "learning")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "router_model.joblib")


# ── Feature vectorizers ───────────────────────────────────────────────────


class EmbeddingVectorizer:
    """Sklearn-compatible wrapper around the RAG embedder (PR 2.5).

    Deliberately **stateless** — the embedder isn't pickled into the
    joblib artifact. fit/transform reach into the process-cached
    embedder via :func:`app.core.rag.embeddings.get_embedder`, so the
    saved Pipeline stays a few kilobytes even though the underlying
    model is ~25 MB. Trade-off: the loading code path needs the RAG
    deps available; the integrity check at predict time will surface
    a clean error if they aren't.

    Implements the sklearn estimator protocol via duck-typing rather
    than inheriting from ``BaseEstimator``/``TransformerMixin``: those
    are heavyweight when we only need fit/transform/fit_transform and
    we don't want sklearn pulled into joblib's pickle graph for the
    wrapper itself.
    """

    # Tell sklearn we're a transformer. CalibratedClassifierCV inspects
    # this on the upstream stage during cross-validation.
    _estimator_type = "transformer"

    def fit(self, X, y=None):
        # Touch the embedder at train time so availability errors surface
        # here rather than on first predict. Stateless otherwise.
        from app.core.rag.embeddings import get_embedder
        get_embedder()
        return self

    def transform(self, X):
        from app.core.rag.embeddings import get_embedder
        return get_embedder().encode(list(X))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {}

    def set_params(self, **params):
        return self


# ── Training row schema ───────────────────────────────────────────────────


@dataclass
class TrainingRow:
    """One labeled example.

    - ``label_quality`` is the lever for self-correction: "auto" rows came
      from the keyword router or paraphrasing (potentially noisy); "corrected"
      rows came from a user explicitly fixing a routing mistake.
    - ``source`` is finer-grained: "seed_keyword" (from ACTION_PATTERNS),
      "runtime" (real dispatch logged via smart_orchestrator), "paraphrase"
      (word-swap augmentation), "correction" (user feedback).
    - ``weight`` defaults to 1.0; paraphrases come in at 0.3 because they're
      doubly-noisy (noisy label × noisy paraphrase).
    """

    text: str
    label: str
    label_quality: str  # "auto" | "corrected"
    source: str         # "seed_keyword" | "runtime" | "paraphrase" | "correction"
    weight: float = 1.0


# ── Data collection ───────────────────────────────────────────────────────


def collect_training_data(
    prefer_corrected: bool = False,
    augment_paraphrases: bool = True,
) -> List[TrainingRow]:
    """Build the full training set from all available sources.

    When ``prefer_corrected`` is set and at least ``_MIN_TOTAL_SAMPLES``
    correction rows exist, the seed + runtime "auto" rows are dropped —
    the model retrains purely on user-validated data. Until then, we mix.
    """
    rows: List[TrainingRow] = []
    rows.extend(_seed_data_from_keyword_router())
    rows.extend(_runtime_data_from_patterns())

    corrected = [r for r in rows if r.label_quality == "corrected"]
    if prefer_corrected and len(corrected) >= _MIN_TOTAL_SAMPLES:
        rows = corrected

    if augment_paraphrases:
        rows = _paraphrase_augment(rows)

    return rows


def _seed_data_from_keyword_router() -> List[TrainingRow]:
    """One row per (action, keyword) pair from ``ACTION_PATTERNS``.

    Each keyword becomes a positive training example for its action. This
    is the day-1 bootstrap that lets PR 1 ship without waiting for data.
    """
    from app.blocks.smart_orchestrator import ACTION_PATTERNS

    rows: List[TrainingRow] = []
    for action, keywords in ACTION_PATTERNS:
        for kw in keywords:
            # Skip pure file-extension keywords (".xlsx", ".dwg") — they're
            # routing signals at file-attach time, not natural language.
            # Including them would teach the model to predict actions from
            # dots, which is brittle.
            if kw.startswith("."):
                continue
            rows.append(TrainingRow(
                text=kw,
                label=action,
                label_quality="auto",
                source="seed_keyword",
                weight=1.0,
            ))
    return rows


def _runtime_data_from_patterns() -> List[TrainingRow]:
    """Read accumulated ``routing_decisions`` patterns from learning_engine.

    Each pattern observation is a JSON blob with shape
    ``{"text": "...", "action": "...", "score": float, "corrected": bool}``.
    Patterns are added by smart_orchestrator's _record_pattern hook on every
    learned-mode dispatch. Corrections (when the user marks a routing
    decision wrong) flip ``label_quality`` to "corrected" — see the
    ``record_correction`` flow on learning_engine for the UI hook (PR 1
    ships the storage, not the UI buttons; W4 will).
    """
    from app.blocks import BLOCK_REGISTRY

    cls = BLOCK_REGISTRY.get("learning_engine")
    if cls is None:
        # Virgin / no-kit boots do not register the construction-kit block.
        # Independently labeled rows still live on LearningEngineBlock when
        # a test (or opt-in) wrote them — read that process-local state
        # instead of dropping every training row.
        from app.blocks.learning_engine import LearningEngineBlock

        cls = LearningEngineBlock
    # shared_instance() — uses the same _state the writeback paths populated,
    # so train_router sees patterns hydration just recorded without a re-load.
    le = cls.shared_instance()

    if "patterns" not in le._state:
        return []

    rows: List[TrainingRow] = []
    for project_id, by_category in le._state["patterns"].items():
        for obs in by_category.get("routing_decisions", []):
            blob = obs.get("observation")
            if not blob:
                continue
            try:
                parsed = json.loads(blob) if isinstance(blob, str) else blob
            except (ValueError, TypeError):
                continue
            text = parsed.get("text") or parsed.get("message")
            action = parsed.get("action") or parsed.get("label")
            if not text or not action:
                continue
            from app.core.learning.trainable import is_independently_labeled

            trainable = parsed.get("trainable")
            if trainable is None:
                trainable = obs.get("trainable")
            if trainable is None:
                trainable = is_independently_labeled(obs.get("source"), parsed)
            if not trainable:
                continue
            rows.append(TrainingRow(
                text=text,
                label=action,
                label_quality="corrected" if parsed.get("corrected") else "auto",
                source="correction" if parsed.get("corrected") else "runtime",
                weight=1.0,
            ))
    return rows


def _paraphrase_augment(
    rows: List[TrainingRow],
    target_per_class: int = _MIN_PER_CLASS_BEFORE_PARAPHRASE,
) -> List[TrainingRow]:
    """Deterministic word-swap paraphrasing — no LLM call, no chat dependency.

    For any class below ``target_per_class``, generate paraphrases by:

    1. Prepending a question template ("how do I", "can you", "I need to")
    2. Wrapping with imperative qualifiers ("please", "now")
    3. Synonymizing a small set of action verbs ("extract" → "pull",
       "estimate" → "calculate", etc.)

    Paraphrases carry ``source="paraphrase"`` and ``weight=0.3`` — far less
    than real examples — and are stripped from holdout splits in ``train()``.

    This is intentionally low-quality: the goal is to get class counts
    above the floor for ``CalibratedClassifierCV(cv=...)`` to fit, not to
    produce great training data. Real runtime data replaces these.
    """
    SYNONYMS = {
        "extract": ["pull", "get", "grab", "retrieve"],
        "estimate": ["calculate", "compute", "work out"],
        "analyze": ["review", "check", "examine"],
        "process": ["handle", "deal with", "work on"],
        "generate": ["create", "produce", "make"],
        "optimize": ["improve", "tune", "refine"],
        "schedule": ["plan", "timetable", "sequence"],
        "check": ["verify", "validate", "audit"],
    }
    TEMPLATES = [
        "how do I {}",
        "can you {}",
        "please {}",
        "I need to {}",
        "{} please",
        "help me {}",
    ]

    by_label: Dict[str, List[TrainingRow]] = {}
    for r in rows:
        by_label.setdefault(r.label, []).append(r)

    augmented: List[TrainingRow] = list(rows)
    for label, label_rows in by_label.items():
        if len(label_rows) >= target_per_class:
            continue
        deficit = target_per_class - len(label_rows)
        new_rows: List[TrainingRow] = []
        # Cycle through seed rows and apply templates / synonyms until we
        # close the gap. Deduplicate by exact text to avoid degenerate copies.
        seen = {r.text for r in label_rows}
        for seed in label_rows:
            if len(new_rows) >= deficit:
                break
            for template in TEMPLATES:
                candidate = template.format(seed.text)
                if candidate not in seen:
                    new_rows.append(TrainingRow(
                        text=candidate,
                        label=label,
                        label_quality="auto",
                        source="paraphrase",
                        weight=0.3,
                    ))
                    seen.add(candidate)
                    if len(new_rows) >= deficit:
                        break
            # Synonym swap on the leading verb
            words = seed.text.split()
            if words and words[0].lower() in SYNONYMS:
                for syn in SYNONYMS[words[0].lower()]:
                    candidate = " ".join([syn] + words[1:])
                    if candidate not in seen:
                        new_rows.append(TrainingRow(
                            text=candidate,
                            label=label,
                            label_quality="auto",
                            source="paraphrase",
                            weight=0.3,
                        ))
                        seen.add(candidate)
                        if len(new_rows) >= deficit:
                            break
        augmented.extend(new_rows)
    return augmented


# ── Model load / save (with integrity check) ──────────────────────────────


_MODEL_CACHE: Optional[Tuple[str, Any]] = None  # (sha256, Pipeline)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model(expected_sha256: Optional[str] = None, expected_labels: Optional[List[str]] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Return (pipeline, reason_unloaded). ``pipeline`` is None when any
    integrity check fails; ``reason_unloaded`` is a human-readable string."""
    global _MODEL_CACHE
    path = _model_path()
    if not os.path.exists(path):
        return None, f"model file not found at {path}"

    current_sha = _sha256(path)
    if expected_sha256 and current_sha != expected_sha256:
        return None, f"sha256 mismatch (file: {current_sha[:12]}, expected: {expected_sha256[:12]})"

    if _MODEL_CACHE is not None and _MODEL_CACHE[0] == current_sha:
        pipeline = _MODEL_CACHE[1]
    else:
        try:
            import joblib  # local import — joblib not always at module-import time
            pipeline = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            return None, f"joblib load failed: {type(exc).__name__}: {exc}"
        _MODEL_CACHE = (current_sha, pipeline)

    if expected_labels:
        actual = set(getattr(pipeline, "classes_", []))
        expected = set(expected_labels)
        if actual != expected:
            return None, f"label set mismatch (model: {len(actual)} classes, expected: {len(expected)})"

    return pipeline, None


def invalidate_model_cache() -> None:
    """Called by ``train()`` after rewriting the joblib so the next predict
    call picks up the new artifact."""
    global _MODEL_CACHE
    _MODEL_CACHE = None


# ── Training ──────────────────────────────────────────────────────────────


def _build_vectorizer(feature_mode: str):
    """Construct the first stage of the Pipeline.

    ``"tfidf"`` (default) is the PR 1 baseline — no extra dependencies,
    pure sklearn. ``"embeddings"`` swaps in :class:`EmbeddingVectorizer`
    which wraps the RAG embedder; requires ``requirements-rag.txt`` to
    be installed (or ``RAG_EMBEDDING_MODEL=fake`` for tests).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if feature_mode == "tfidf":
        return TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True,
            stop_words=None,  # domain terms; don't strip
        )
    if feature_mode == "embeddings":
        return EmbeddingVectorizer()
    raise ValueError(
        f"unknown feature_mode {feature_mode!r}; use 'tfidf' or 'embeddings'"
    )


def train(
    prefer_corrected: bool = False,
    min_samples: int = _MIN_TOTAL_SAMPLES,
    feature_mode: str = "tfidf",
) -> Dict[str, Any]:
    """Fit + calibrate the classifier, persist to disk, return metrics.

    Returns ``{"status": "success", ...}`` on success or
    ``{"status": "insufficient_data", "samples_used": N, "threshold_needed": _MIN_TOTAL_SAMPLES}``
    when the training set is too small. ``feature_mode`` selects the
    first Pipeline stage — see :func:`_build_vectorizer`.
    """
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    import numpy as np

    rows = collect_training_data(prefer_corrected=prefer_corrected, augment_paraphrases=True)
    if len(rows) < min_samples:
        return {
            "status": "insufficient_data",
            "samples_used": len(rows),
            "threshold_needed": min_samples,
        }

    # Holdout split MUST exclude paraphrase rows — including them would
    # inflate accuracy by giving the model near-copies at test time.
    real_rows = [r for r in rows if r.source != "paraphrase"]
    paraphrase_rows = [r for r in rows if r.source == "paraphrase"]

    if len(real_rows) < 10:
        # Without enough real rows to hold any out, skip evaluation but
        # still train. This is the cold-start case.
        train_rows = rows
        holdout_rows: List[TrainingRow] = []
    else:
        # Stratify if every real-class has ≥2 samples; else simple split.
        real_labels = [r.label for r in real_rows]
        label_counts: Dict[str, int] = {}
        for lab in real_labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1
        can_stratify = all(c >= 2 for c in label_counts.values())
        train_real, holdout_rows = train_test_split(
            real_rows,
            test_size=0.2,
            random_state=42,
            stratify=real_labels if can_stratify else None,
        )
        train_rows = train_real + paraphrase_rows

    X_train = [r.text for r in train_rows]
    y_train = [r.label for r in train_rows]
    w_train = np.array([r.weight for r in train_rows])

    # LogisticRegression needs at least two classes. Without this guard,
    # sklearn raises a long traceback that masks the real problem (e.g.
    # prefer_corrected=True with corrections for only one action). The
    # graceful return tells the caller what to do.
    if len(set(y_train)) < 2:
        return {
            "status": "insufficient_classes",
            "samples_used": len(rows),
            "single_class": next(iter(set(y_train)), None),
            "remediation": (
                "Training requires labeled examples for at least 2 classes. "
                "If prefer_corrected=True, accumulate corrections for additional "
                "actions before retraining."
            ),
        }

    # CalibratedClassifierCV needs cv ≤ min_class_count. With 40 actions and
    # 3-10 keywords per class plus paraphrases, min_class_count is typically 6+.
    # We compute it defensively.
    class_counts: Dict[str, int] = {}
    for lab in y_train:
        class_counts[lab] = class_counts.get(lab, 0) + 1
    min_class = min(class_counts.values())
    cv_folds = max(2, min(5, min_class))

    pipeline = Pipeline([
        ("vec", _build_vectorizer(feature_mode)),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0),
            method="sigmoid",
            cv=cv_folds,
        )),
    ])

    # sklearn's Pipeline doesn't pass sample_weight through unless we name
    # the final step. The calibrated wrapper doesn't accept sample_weight
    # either — it would propagate to the base LR. We accept this limitation
    # in PR 1: paraphrase down-weighting is honored by the threshold-based
    # holdout exclusion above; full weighting through the calibration
    # wrapper would require a custom estimator. Documented in the plan.
    pipeline.fit(X_train, y_train)
    invalidate_model_cache()

    # Evaluation on holdout — only when we kept some real rows aside
    accuracy = None
    per_class: Dict[str, Any] = {}
    brier = None
    if holdout_rows:
        X_h = [r.text for r in holdout_rows]
        y_h = [r.label for r in holdout_rows]
        y_pred = pipeline.predict(X_h)
        accuracy = float((y_pred == np.array(y_h)).mean())
        report = classification_report(y_h, y_pred, output_dict=True, zero_division=0)
        per_class = {
            k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"], "support": int(v["support"])}
            for k, v in report.items()
            if isinstance(v, dict) and k not in ("macro avg", "weighted avg", "accuracy")
        }
        # Brier score for top-1 — one-vs-rest binarization
        try:
            proba = pipeline.predict_proba(X_h)
            class_idx = {c: i for i, c in enumerate(pipeline.classes_)}
            top1_proba = np.array([proba[i, class_idx[y_pred[i]]] for i in range(len(y_pred))])
            correct = np.array([1 if y_pred[i] == y_h[i] else 0 for i in range(len(y_pred))])
            brier = float(brier_score_loss(correct, top1_proba))
        except Exception as exc:  # noqa: BLE001
            logger.warning("brier score computation failed: %s", exc)

    # Persist
    path = _model_path()
    joblib.dump(pipeline, path)
    sha = _sha256(path)
    label_distribution: Dict[str, int] = {}
    for lab in y_train:
        label_distribution[lab] = label_distribution.get(lab, 0) + 1

    metadata = {
        "trained_at": time.time(),
        "model_path": path,
        "sha256": sha,
        "feature_mode": feature_mode,
        "samples_used": len(rows),
        "real_samples": len(real_rows),
        "paraphrase_samples": len(paraphrase_rows),
        "holdout_size": len(holdout_rows),
        "accuracy": accuracy,
        "brier_score": brier,
        "per_class_metrics": per_class,
        "label_distribution": label_distribution,
        "labels": sorted(set(y_train)),
        "cv_folds": cv_folds,
        "prefer_corrected": prefer_corrected,
        "min_class_size": min_class,
    }

    return {
        "status": "success",
        **{k: v for k, v in metadata.items() if k != "per_class_metrics"},
        "per_class_metrics": per_class,
    }


# ── Prediction ────────────────────────────────────────────────────────────


def predict(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Predict one routing action with calibrated confidence.

    Returns ``{"action": str, "confidence": float, "model_loaded": bool,
    "fallback_recommended": bool, "reason": Optional[str], "top_k": [...]}``.

    ``metadata`` is the ``models.router`` slot from learning_engine state.
    When provided, used for integrity verification (sha256 + label set).
    When None, the integrity check is skipped (file presence only).
    """
    if not text or not text.strip():
        return {
            "action": None,
            "confidence": 0.0,
            "model_loaded": False,
            "fallback_recommended": True,
            "reason": "empty input",
            "top_k": [],
        }

    expected_sha = (metadata or {}).get("sha256")
    expected_labels = (metadata or {}).get("labels")
    pipeline, reason = _load_model(expected_sha, expected_labels)
    if pipeline is None:
        return {
            "action": None,
            "confidence": 0.0,
            "model_loaded": False,
            "fallback_recommended": True,
            "reason": reason,
            "top_k": [],
        }

    try:
        proba = pipeline.predict_proba([text])[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("predict_proba failed: %s", exc)
        return {
            "action": None,
            "confidence": 0.0,
            "model_loaded": True,
            "fallback_recommended": True,
            "reason": f"predict failed: {type(exc).__name__}: {exc}",
            "top_k": [],
        }

    classes = list(pipeline.classes_)
    ranked = sorted(zip(classes, proba), key=lambda t: t[1], reverse=True)
    top_k = [{"action": c, "confidence": float(p)} for c, p in ranked[:5]]
    top_action, top_conf = ranked[0]
    fallback = float(top_conf) < confidence_threshold

    return {
        "action": top_action,
        "confidence": float(top_conf),
        "model_loaded": True,
        "fallback_recommended": fallback,
        "reason": (
            f"confidence {top_conf:.3f} < threshold {confidence_threshold:.2f}"
            if fallback else None
        ),
        "top_k": top_k,
    }


def evaluate(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the metrics from the last train. Pure read; no recompute."""
    if not metadata:
        return {"status": "no_model", "reason": "no router metadata in learning_engine state"}
    return {
        "status": "success",
        "trained_at": metadata.get("trained_at"),
        "accuracy": metadata.get("accuracy"),
        "brier_score": metadata.get("brier_score"),
        "samples_used": metadata.get("samples_used"),
        "real_samples": metadata.get("real_samples"),
        "paraphrase_samples": metadata.get("paraphrase_samples"),
        "labels": metadata.get("labels"),
        "label_distribution": metadata.get("label_distribution"),
        "per_class_metrics": metadata.get("per_class_metrics"),
        "caveat": (
            "Not yet self-correcting — the model inherits the keyword router's "
            "bias proportional to its training mix. Corrections accumulate to "
            "fix this over time; see prefer_corrected in train_router."
        ),
    }
