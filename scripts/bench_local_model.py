"""Local-model bench harness — STEP 3 of the on-prem sovereignty program.

Characterises a locally-served Ollama model on the referees that decide whether
it can drive The Fork's agent loop on-prem. The make-or-break one is **tool-call
reliability**: the cloud models (Groq Scout, Kimi) pass it; small local models
often emit prose where a tool call is required, which silently breaks every
deliverable workflow. This harness measures that directly against Ollama's
native /api/chat tools API, plus a grounding (no-fabrication) probe and latency.

It is deliberately dependency-light (stdlib only) and hits a LOCAL Ollama, so it
runs air-gapped. It does NOT need the full app. For the full golden-set /
calc-intact / EOT / fresh-upload referees, point the app at the model under the
on-prem profile and run scripts/golden_set_gate.py etc. (documented in
LOCAL_MODEL_DECISION.md) — those need the app + a corpus.

Usage:
    python scripts/bench_local_model.py qwen2.5:7b-instruct [--runs 5]
    python scripts/bench_local_model.py qwen2.5:3b-instruct qwen2.5:7b-instruct

Env: OLLAMA_URL (default http://localhost:11434).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

# A representative deliverable tool the agent loop relies on. If the model
# won't emit a structured call for this, it can't run the workflow on-prem.
_WBS_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_wbs",
        "description": "Generate a construction Work Breakdown Structure (WBS) "
                       "for a project. Call this whenever the user asks to create, "
                       "build, or draft a WBS or work breakdown.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_type": {"type": "string", "description": "e.g. villa, tower, road"},
                "storeys": {"type": "integer", "description": "number of storeys"},
            },
            "required": ["project_type"],
        },
    },
}

# Prompts that MUST trigger a tool call (deliverable intents).
_TOOLCALL_PROMPTS = [
    "Create a WBS for a 3-storey villa.",
    "Draft a work breakdown structure for a highway project.",
    "Build me a WBS for a 20-storey residential tower.",
]

# Grounding probe: the answer must come from the supplied context, not training
# data. The context states a deliberately non-standard value; a grounded model
# reproduces it, a fabricating one gives the textbook number.
_GROUNDING_CONTEXT = (
    "AUTHORITATIVE REFERENCE CONTEXT — answer using ONLY this.\n"
    "Per the project's amended particular conditions, the defects notification "
    "period for this contract is 512 days from the taking-over certificate."
)
_GROUNDING_Q = "What is the defects notification period for this contract, in days?"
_GROUNDING_EXPECT = "512"


def _post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat(model: str, messages: list, tools: list | None = None) -> dict:
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    return _post("/api/chat", payload)


def bench_toolcalls(model: str, runs: int) -> dict:
    """Fraction of deliverable prompts for which the model emits a well-formed
    tool_call (correct name + parseable args)."""
    ok = 0
    total = 0
    latencies = []
    for prompt in _TOOLCALL_PROMPTS:
        for _ in range(runs):
            total += 1
            t0 = time.monotonic()
            try:
                resp = _chat(
                    model,
                    [{"role": "user", "content": prompt}],
                    tools=[_WBS_TOOL],
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    [error] {prompt!r}: {exc}")
                continue
            latencies.append(time.monotonic() - t0)
            msg = resp.get("message", {})
            calls = msg.get("tool_calls") or []
            good = False
            for c in calls:
                fn = (c.get("function") or {})
                if fn.get("name") == "generate_wbs":
                    args = fn.get("arguments")
                    if isinstance(args, dict):
                        good = True
                    elif isinstance(args, str):
                        try:
                            json.loads(args)
                            good = True
                        except json.JSONDecodeError:
                            good = False
            if good:
                ok += 1
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else None
    return {"passed": ok, "total": total,
            "rate": round(ok / total, 3) if total else 0.0,
            "avg_latency_s": avg_lat}


def bench_grounding(model: str, runs: int) -> dict:
    """Fraction of runs where the model reproduces the context's non-standard
    figure (512) rather than fabricating a textbook value."""
    ok = 0
    for _ in range(runs):
        try:
            resp = _chat(model, [
                {"role": "system", "content": _GROUNDING_CONTEXT},
                {"role": "user", "content": _GROUNDING_Q},
            ])
        except Exception as exc:  # noqa: BLE001
            print(f"    [error] grounding: {exc}")
            continue
        ans = (resp.get("message", {}).get("content") or "")
        if _GROUNDING_EXPECT in ans:
            ok += 1
    return {"passed": ok, "total": runs,
            "rate": round(ok / runs, 3) if runs else 0.0}


def bench(model: str, runs: int) -> dict:
    print(f"\n=== {model} (runs={runs}) ===")
    tc = bench_toolcalls(model, runs)
    print(f"  tool-call reliability: {tc['passed']}/{tc['total']} "
          f"= {tc['rate']*100:.0f}%   avg latency {tc['avg_latency_s']}s")
    gr = bench_grounding(model, runs)
    print(f"  grounding (no-fabrication): {gr['passed']}/{gr['total']} "
          f"= {gr['rate']*100:.0f}%")
    return {"model": model, "tool_calls": tc, "grounding": gr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="emit JSON summary")
    args = ap.parse_args()
    results = [bench(m, args.runs) for m in args.models]
    if args.json:
        print("\n" + json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
