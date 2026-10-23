#!/usr/bin/env python
"""Download the RAG embedder weights into the image at BUILD time.

Run by the Dockerfile so the container ships with the model already present.

Why this exists
---------------
Only the embedding LIBRARIES were ever installed into the image; the weights
were not. The first embed call therefore resolved the model name against
huggingface.co at RUNTIME (``snapshot_download``), and with ephemeral
containers and no persistent HF cache that happened on every deploy, restart
and scale event — putting a third-party host in the boot path of retrieval.

The failure was silent, which is what made it costly: ``doc_index`` wraps its
RAG hook in try/except, so a failed download left documents stored, registered
and listed while indexed with ZERO chunks — permanently unsearchable, with no
failed upload and nothing in the UI.

Contract
--------
* Uses the SAME backend selection as ``Embedder.__init__`` (sentence-
  transformers first, model2vec second) so the cache is written by the loader
  that will later read it.
* Verifies the model loads with the Hub disabled before declaring success, so
  a build cannot pass while leaving the runtime unable to work offline.
* Exits non-zero on any failure — a broken build beats a silent empty index.
"""
from __future__ import annotations

import os
import sys
import time


# Hub 429s on anonymous CI (compose --build has no GHA layer cache). The
# huggingface_hub client already retries each HEAD 5× with 1–8s gaps; a
# burst still exhausts that and fails the image. An outer loop waits longer
# between full load attempts so a rate-limit window can clear.
_RETRY_DELAYS_S = (20, 40, 80)


def _retryable_prefetch_error(exc: BaseException) -> bool:
    """True for Hub rate-limit / connect storms — not missing-model errors."""
    blob = f"{exc}".lower()
    return any(
        marker in blob
        for marker in (
            "429",
            "too many requests",
            "rate limit",
            "couldn't connect to 'https://huggingface.co'",
            "could not connect to huggingface",
        )
    )


def _load(model_name: str):
    """Construct the model through the app's backend precedence."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        from model2vec import StaticModel

        return StaticModel.from_pretrained(model_name), "model2vec"
    return SentenceTransformer(model_name), "sentence_transformers"


def _load_with_retry(model_name: str, load=_load):
    """Call ``load``; retry Hub 429 / connect failures with long backoff."""
    attempts = len(_RETRY_DELAYS_S) + 1
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return load(model_name)
        except Exception as exc:  # noqa: BLE001 — classified below
            last = exc
            if attempt >= attempts - 1 or not _retryable_prefetch_error(exc):
                raise
            wait = _RETRY_DELAYS_S[attempt]
            print(
                f"prefetch_embedder: retryable Hub error "
                f"({type(exc).__name__}: {exc}); sleeping {wait}s "
                f"[{attempt + 1}/{attempts}]",
                flush=True,
            )
            time.sleep(wait)
    raise last  # pragma: no cover — loop always raises or returns


def main(argv: list[str]) -> int:
    model_name = (
        argv[1] if len(argv) > 1 else os.getenv("RAG_EMBEDDING_MODEL", "")
    ).strip()
    if not model_name or model_name == "fake":
        print(
            f"prefetch_embedder: nothing to fetch for model={model_name!r}",
            flush=True,
        )
        return 0

    home = os.getenv("HF_HOME", "(default)")
    print(
        f"prefetch_embedder: fetching {model_name!r} into HF_HOME={home}",
        flush=True,
    )
    model, backend = _load_with_retry(model_name)
    print(f"prefetch_embedder: fetched via {backend}", flush=True)

    # Prove the cache is complete by re-loading with the Hub switched OFF —
    # exactly the runtime configuration. Without this a build could succeed on
    # a partially-populated cache and still reach for the network in prod.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    offline_model, _ = _load(model_name)

    # Probe the dimension the way Embedder does, so a checkpoint that loads but
    # cannot encode is caught here rather than on the first user question.
    encode = getattr(offline_model, "encode", None)
    if encode is None:
        raise RuntimeError(f"{model_name} exposes no encode(); cannot embed")
    vectors = encode(["dimension probe"])
    dim = int(getattr(vectors, "shape", [0, 0])[1])
    if dim <= 0:
        raise RuntimeError(f"{model_name} produced a non-positive dim {dim}")

    print(
        f"prefetch_embedder: OK — {model_name} dim={dim} loads offline",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as exc:  # noqa: BLE001 — build must fail loudly
        print(f"prefetch_embedder: FAILED — {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
