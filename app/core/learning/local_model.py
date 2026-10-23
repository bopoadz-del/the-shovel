"""Local fine-tuned model loader (PR 3a — code-only scaffolding).

Loads a base model + LoRA adapter trained via ``scripts/finetune_router.py``
for the chat block to use when ``use_local_model=true`` is passed. Same
lazy-import + process-cache pattern as ``app/core/rag/embeddings.py``
so callers can gate via :func:`available` without pulling torch into
the chat block's import graph.

Orin-portability invariant (binding):

* No hardcoded ``.to("cuda")`` — every device reference goes through
  :func:`_device` which honors GPU availability.
* HuggingFace format only. Adapter loads via ``peft.PeftModel.from_pretrained``
  reading ``adapter_model.safetensors`` + ``adapter_config.json``. When
  the Orin port runs, ``bitsandbytes.convert_4bit`` or a GGUF conversion
  operates on the same files — no format change needed.
* No vLLM / TGI / Triton. Inference is ``transformers.pipeline`` —
  swap-the-runtime is a future PR, not a redesign.

Verification status: the lazy-import gating, cache invalidation, and
graceful-fallback are covered in ``tests/test_local_model.py``. The
end-to-end model load against a real base+adapter pair has NOT been
exercised in CI — see ``docs/EDGE_PORT.md`` and the PR 3a description
for the verification steps you run on a GPU host.
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Base model and adapter paths. Override via env so deployments can
# point at the artifact paths their training run produced without
# editing code.
_DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _base_model_name() -> str:
    """Resolve the base model ID.

    Precedence (highest first):
    1. ``LOCAL_BASE_MODEL`` env — pins a concrete HF model ID
    2. ``LOCAL_MODEL_ALIAS`` env → resolved via model_registry
    3. Default alias ``construction_v1`` → resolved via model_registry
    4. Historical fallback ``Qwen/Qwen2.5-3B-Instruct`` when the
       registry is missing or the alias is unknown (don't break existing
       deploys on an import error)

    """
    override = os.getenv("LOCAL_BASE_MODEL")
    if override:
        return override
    alias = os.getenv("LOCAL_MODEL_ALIAS") or "construction_v1"
    try:
        from app.core.learning.model_registry import resolve_base_model
        return resolve_base_model(alias)
    except Exception:
        return _DEFAULT_BASE_MODEL


def _adapter_dir() -> str:
    """Where the LoRA adapter lives. Defaults to
    ``$DATA_DIR/learning/adapters/construction_v1`` (matches the default
    ``finetune_router.py`` output_dir)."""
    if override := os.getenv("LOCAL_ADAPTER_DIR"):
        return override
    base = os.getenv("DATA_DIR", "./data")
    return os.path.join(base, "learning", "adapters", "construction_v1")


def _device():
    """Device selector — the one place in the file that says "cuda"."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Prompt template (SINGLE SOURCE OF TRUTH for train + inference) ────────
#
# Critical: training in scripts/finetune_router.py wraps each example in
# this exact Alpaca-style template. Inference MUST use the same wrapper or
# the model sees a different prompt shape than it trained on, silently
# degrading response quality. Both sides import from here so they cannot
# drift.

_TEMPLATE_INSTRUCTION_PREFIX = "### Instruction:\n"
_TEMPLATE_RESPONSE_PREFIX = "\n\n### Response:\n"


def format_prompt(instruction: str, response: str = "") -> str:
    """Wrap an instruction (and optional response) in the training template.

    With ``response``, returns a full training example (used by the trainer
    at fit time). Without one, returns the prompt suffix used at inference
    time — the model continues from after ``### Response:\\n``.
    """
    if response:
        return f"{_TEMPLATE_INSTRUCTION_PREFIX}{instruction}{_TEMPLATE_RESPONSE_PREFIX}{response}"
    return f"{_TEMPLATE_INSTRUCTION_PREFIX}{instruction}{_TEMPLATE_RESPONSE_PREFIX}"


# ── Module-level cache ────────────────────────────────────────────────────


_PIPELINE_CACHE: Optional[Any] = None
_LOAD_LOCK = Lock()


def available() -> bool:
    """True when the local-model stack is functional in this process.

    Three conditions all required:

    1. ``transformers`` + ``peft`` + ``torch`` are importable (deps in
       ``requirements-ml.txt`` are installed)
    2. ``_adapter_dir()`` exists and contains an ``adapter_config.json``
       (otherwise there's nothing to load even if the libs are present)
    3. We're not in test mode forcing the unavailable path (callers can
       set ``LOCAL_MODEL_UNAVAILABLE=1`` to force the fallback for tests)
    """
    if os.getenv("LOCAL_MODEL_UNAVAILABLE") == "1":
        return False
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.warning("local-model stack imports unavailable", exc_info=True)
        return False
    adapter_dir = _adapter_dir()
    if not os.path.isdir(adapter_dir):
        return False
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        return False
    return True


def get_pipeline() -> Optional[Any]:
    """Return the process-cached ``transformers.pipeline`` for chat.

    Loads the base model + LoRA adapter on first call (slow — ~10-30s on
    GPU, longer on CPU); subsequent calls return the cache. Returns
    ``None`` when :func:`available` is False — callers must handle this
    rather than calling unguarded.
    """
    global _PIPELINE_CACHE
    if not available():
        return None
    with _LOAD_LOCK:
        if _PIPELINE_CACHE is not None:
            return _PIPELINE_CACHE
        try:
            _PIPELINE_CACHE = _build_pipeline()
        except Exception as exc:  # noqa: BLE001
            # Real load failures (out-of-memory, corrupt adapter, etc.)
            # bubble up as "unavailable" to the chat block's fallback path.
            # Logged once so operators can find the cause; cache stays
            # None so we don't re-attempt on every chat turn.
            logger.exception("local model load failed; falling back: %s", exc)
            _PIPELINE_CACHE = None
        return _PIPELINE_CACHE


def _build_pipeline():
    """Construct the actual pipeline. Separated so :func:`get_pipeline`
    can wrap with try/except + caching cleanly."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    base_name = _base_model_name()
    adapter_dir = _adapter_dir()
    device = _device()

    logger.info("loading base model %s on %s", base_name, device)
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype="auto",  # bf16/fp16 on CUDA, fp32 on CPU — let transformers decide
    )
    # PeftModel.from_pretrained reads adapter_config.json + adapter_model.safetensors
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.to(device)
    model.eval()

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,  # Orin-portability: device, not "cuda" literal
    )


def invalidate_cache() -> None:
    """Drop the cached pipeline. Used by tests and by operators after
    swapping an adapter — saves a process restart."""
    global _PIPELINE_CACHE
    with _LOAD_LOCK:
        _PIPELINE_CACHE = None


def generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> Optional[str]:
    """Run one generation through the cached pipeline. Returns None when
    the model isn't loadable (callers should fall back to cloud chat).

    Wraps ``prompt`` in the same Alpaca-style template the trainer used at
    fit time (see :func:`format_prompt`). Without this wrap the model sees
    a different prompt shape than it trained on, silently producing
    worse-quality responses. Reviewer note: this was the load-bearing
    correctness fix on PR #24 review.
    """
    pipe = get_pipeline()
    if pipe is None:
        return None
    # Wrap with the training template — MUST match what finetune_router.py
    # fed to tokenize() during training, or the adapter underperforms.
    wrapped = format_prompt(prompt)
    try:
        out = pipe(
            wrapped,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),  # transformers rejects 0.0
            return_full_text=False,
        )
        if isinstance(out, list) and out:
            text = (out[0].get("generated_text") or "").strip()
            # Defense-in-depth: if the adapter ever overruns EOS and starts
            # producing a fresh "### Instruction:" block, cut at the next
            # turn boundary so we don't return a hallucinated follow-up.
            cut = text.find("### Instruction:")
            if cut > 0:
                text = text[:cut].rstrip()
            return text
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("local model generation failed; falling back: %s", exc)
        return None
