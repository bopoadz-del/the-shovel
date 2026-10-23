#!/usr/bin/env python
"""Measure RERANK_ENABLED off vs on against B4/B5/A5 exact tests.

Writes artifacts/fork/RERANK_EVAL.md. Does not enable the flag in
production. Golden-set live sweep is a separate chat-path gate and is
recorded as UNPRODUCED when this process cannot reach that corpus.
"""
from __future__ import annotations

import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "fork" / "RERANK_EVAL.md"
PYTHON = sys.executable

# WAVE 2 B4 / B5 + Contract Data A5 exact nodes.
NODES = [
    (
        "B4",
        "tests/test_rag_identifier_retrieval.py::"
        "test_d5995_carriageway_outranks_excluded_culvert",
    ),
    (
        "B5",
        "tests/test_rag_identifier_retrieval.py::"
        "test_ocr_spaced_d5492_is_retrievable_as_compact",
    ),
    (
        "B5_chat",
        "tests/test_rag_identifier_retrieval.py::"
        "test_chat_path_retrieves_ocr_spaced_d5492_without_reindex",
    ),
    (
        "A5",
        "tests/test_contract_data_qa_retrieval.py::"
        "test_retrieval_returns_particulars_row_not_only_defined_term",
    ),
]

REPEATS = 11


def _p95(samples: list[float]) -> float:
    if not samples:
        return float("nan")
    if len(samples) == 1:
        return samples[0]
    ordered = sorted(samples)
    idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[idx]


def _cross_encoder_status() -> tuple[str, str]:
    """Return (status, detail). status is loaded | unavailable."""
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except Exception as exc:
        return "unavailable", f"sentence_transformers import failed: {exc}"
    try:
        from app.core.rag import reranker

        model = reranker._get_model()
        if model is None:
            return "unavailable", "reranker._get_model() returned None (load failed)"
        return "loaded", f"{type(model).__name__} {reranker.DEFAULT_MODEL}"
    except Exception as exc:
        return "unavailable", f"model load failed: {exc}"


def _run_node(node: str, env: dict[str, str]) -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PYTHON, "-m", "pytest", node, "-q", "--tb=line"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    detail = tail[-1] if tail else f"exit {proc.returncode}"
    return ok, elapsed, detail


def _measure(flag_on: bool, model_note: str) -> dict:
    env = os.environ.copy()
    env["RAG_EMBEDDING_MODEL"] = "fake"
    env["ENV"] = env.get("ENV") or "development"
    env["RERANK_ENABLED"] = "true" if flag_on else "false"
    env.pop("RAG_RERANKER", None)
    rows = []
    for name, node in NODES:
        times: list[float] = []
        passes = 0
        last_detail = ""
        for _ in range(REPEATS):
            ok, elapsed, detail = _run_node(node, env)
            times.append(elapsed)
            last_detail = detail
            if ok:
                passes += 1
        rows.append(
            {
                "name": name,
                "node": node,
                "passes": passes,
                "n": REPEATS,
                "ok": passes == REPEATS,
                "median_s": statistics.median(times),
                "p95_s": _p95(times),
                "min_s": min(times),
                "max_s": max(times),
                "last_detail": last_detail,
            }
        )
    return {
        "flag_on": flag_on,
        "model_note": model_note,
        "rows": rows,
        "all_ok": all(r["ok"] for r in rows),
        "p95_s": _p95([r["p95_s"] for r in rows]),
    }


def _fmt(row: dict) -> str:
    verdict = "PASS" if row["ok"] else "FAIL"
    return (
        f"| {row['name']} | {verdict} {row['passes']}/{row['n']} | "
        f"{row['median_s']:.3f} | {row['p95_s']:.3f} | "
        f"{row['min_s']:.3f} | {row['max_s']:.3f} |"
    )


def _section(title: str, run: dict | None, reason: str | None = None) -> list[str]:
    lines = [f"## {title}", ""]
    if run is None:
        lines.append(f"**UNPRODUCED** — {reason}")
        lines.append("")
        return lines
    lines.append(f"Flag: `RERANK_ENABLED={'true' if run['flag_on'] else 'false'}`")
    lines.append(f"Cross-encoder: {run['model_note']}")
    lines.append(f"Suite verdict: {'PASS' if run['all_ok'] else 'FAIL'}")
    lines.append(f"Repeats per node: {REPEATS}")
    lines.append("")
    lines.append("| node | verdict | median s | p95 s | min s | max s |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(_fmt(r) for r in run["rows"])
    lines.append("")
    return lines


def main() -> int:
    model_status, model_detail = _cross_encoder_status()
    off = _measure(False, "not consulted (flag off)")
    if model_status == "loaded":
        on = _measure(True, model_detail)
        on_reason = None
    else:
        # Still run the flag-ON suite: the hook degrades to cosine order.
        # That is a real conservation measurement, not a cross-encoder run.
        on_degrade = _measure(True, f"degrade-to-cosine ({model_detail})")
        on = on_degrade
        on_reason = model_detail

    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = [
        "# RERANK_EVAL",
        "",
        "Dormant cross-encoder over hybrid top-50 candidates in `doc_index`.",
        "Flag stays **`RERANK_ENABLED=false`**. Do not enable in production.",
        "",
        "Gate to flip later: zero regressions on `tests/golden_set.yaml` +",
        "B4/B5/A5 exact, and p95 latency budget held versus the flag-OFF run",
        "(no separate numeric p95 budget is published in-repo; hold means",
        "ON p95 must not exceed OFF p95 on these exact nodes).",
        "",
        f"Eval host: `{sys.version.split()[0]}`. Embedder: `RAG_EMBEDDING_MODEL=fake`.",
        f"Repeats: {REPEATS} subprocess pytest invocations per node per flag.",
        "Times are wall-clock per invocation (includes pytest startup).",
        "",
    ]
    body = []
    body += _section("Run 1 — RERANK_ENABLED=false (default)", off)
    if model_status == "loaded":
        body += _section("Run 2 — RERANK_ENABLED=true (cross-encoder loaded)", on)
    else:
        body += _section(
            "Run 2 — RERANK_ENABLED=true (cross-encoder)",
            None,
            f"real cross-encoder ON run UNPRODUCED: {on_reason}",
        )
        body += _section(
            "Run 2b — RERANK_ENABLED=true, model unavailable (degrade-to-cosine)",
            on,
        )
    body += [
        "## tests/golden_set.yaml",
        "",
        "**UNPRODUCED** — `scripts/golden_set_gate.py` drives live",
        "`POST /v1/agents/{agent}/chat/stream` against the fixture project.",
        "This branch is not deployed (deploy SHA HOLD, live `567147a`).",
        "A prod sweep would measure the live SHA, not this change. A local",
        "sweep needs that corpus plus a funded chat key on this process.",
        "",
        "## Production",
        "",
        "`RERANK_ENABLED` remains `false`. Do not flip on.",
        "",
    ]
    OUT.write_text("\n".join(head + body), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0 if off["all_ok"] and on["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
