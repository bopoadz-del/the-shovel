"""ONE-USER UI battery runner (per ORDERS_02NOV_0127Z / 0148Z).

Drives the LIVE chat UI as a single logged-in user against the project that
already holds the owner's real documents (saved + embedded), asks each battery
question, and records -- per question -- the route the product actually took,
the answer, and the Why-this-answer inputs + sources, as PASS/FAIL against the
question's declared expected route.

CODE ONLY. The battery questions, expected answers and any figures are NEVER in
this repo (client-content scanner): they are read at run time from a file the
caller exports from FLEET_OPS/Drive (``--battery`` / ``BATTERY_FILE``), and the
results are written back to Drive as ``BATTERY_RESULT_<HHMM>Z`` by the caller.

The route is OBSERVED from the SSE turn, never guessed. The observation has two
pure, unit-tested pieces: ``turn_from_events`` (raw SSE events -> a normalized
turn) and ``classify_route`` (normalized turn -> one of the five route labels).
The network driving lives in ``run_battery`` and is intentionally thin.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any, Iterable

# The five routes the battery distinguishes (ORDERS_02NOV_0127Z).
ROUTE_PROJECT_RAG = "project_rag_llm"
ROUTE_GENERAL_RAG = "general_rag_llm"
ROUTE_CALCULATOR = "llm_calculator"
ROUTE_ASKS_INPUTS = "asks_for_inputs"
ROUTE_TOOL_ON_DOC = "tool_on_document"

#: Tool names whose use means the turn ran a tool over a document/deliverable.
_DOC_TOOLS = frozenset({
    "fetch_document", "drawing_qto", "drawing_takeoff", "generate_wbs",
    "deliver_schedule", "schedule_export", "boq_parse",
})
#: run_calculation dispatch tool.
_CALC_TOOL = "construction_calc"
#: Retrieved-source classes -> which RAG the answer stood on.
_PROJECT_CLASSES = frozenset({"project_corpus", "this_contract", "project_document"})
_GENERAL_CLASSES = frozenset({"master_corpus", "knowledge_base", "template", "reference"})
#: The product's own missing-input request phrasing (no figure invented).
_ASK_RE = re.compile(
    r"(what values should i use|calculator needs the|please provide|could you (?:provide|confirm)|"
    r"need(?:s)? the .*?(?:value|amount|rate|dimensions))",
    re.IGNORECASE,
)


def turn_from_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Collapse the raw SSE events of one assistant turn into the observable
    signals the route is read from: the tool names dispatched, the retrieved
    sources' classes, the per-figure provenance, and the final answer text."""
    tools: list[str] = []
    sources: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    answer = ""
    for evt in events:
        if not isinstance(evt, dict):
            continue
        etype = evt.get("type")
        if etype == "tool_call":
            tool = evt.get("tool")
            if isinstance(tool, str) and tool:
                tools.append(tool)
        elif etype == "end":
            if isinstance(evt.get("content"), str):
                answer = evt["content"]
            if isinstance(evt.get("sources"), list):
                sources = [s for s in evt["sources"] if isinstance(s, dict)]
            if isinstance(evt.get("provenance"), list):
                provenance = [p for p in evt["provenance"] if isinstance(p, dict)]
    return {"tools": tools, "sources": sources, "provenance": provenance, "answer": answer}


def _source_classes(sources: list[dict[str, Any]]) -> set[str]:
    return {str(s.get("source_class") or "") for s in sources}


def _asks_for_inputs(turn: dict[str, Any]) -> bool:
    """The product asked for a missing input rather than answering: its reply
    matches the input-request phrasing and no calculator figure was credited."""
    answer = str(turn.get("answer") or "")
    credited_calc = any(
        isinstance(p, dict) and p.get("source") == "calculator" and p.get("inputs")
        for p in turn.get("provenance") or []
    )
    return bool(_ASK_RE.search(answer)) and not credited_calc


def classify_route(turn: dict[str, Any]) -> str:
    """The route the turn actually took, from observable signals only.

    Precedence: an input-request wins (even if a calculator was attempted and
    came back empty); then a calculator run; then a tool over a document; then
    the retrieved-source class (project corpus vs general). An answer with no
    tool and no retrieved source falls back to general_rag_llm (general LLM
    knowledge)."""
    tools = turn.get("tools") or []
    if _asks_for_inputs(turn):
        return ROUTE_ASKS_INPUTS
    if _CALC_TOOL in tools:
        return ROUTE_CALCULATOR
    if any(t in _DOC_TOOLS for t in tools):
        return ROUTE_TOOL_ON_DOC
    classes = _source_classes(turn.get("sources") or [])
    if classes & _PROJECT_CLASSES:
        return ROUTE_PROJECT_RAG
    if classes & _GENERAL_CLASSES:
        return ROUTE_GENERAL_RAG
    return ROUTE_GENERAL_RAG


def _provenance_digest(provenance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The inputs + sources the Why-this-answer panel would show, flattened for
    the result record (no raw figures beyond what the product already returned)."""
    out: list[dict[str, Any]] = []
    for p in provenance or []:
        if not isinstance(p, dict):
            continue
        row = {"figure": p.get("figure"), "source": p.get("source")}
        if isinstance(p.get("inputs"), list):
            row["inputs"] = [
                {"name": i.get("name"), "source": i.get("source")}
                for i in p["inputs"] if isinstance(i, dict)
            ]
        out.append(row)
    return out


def _stream_turn(base_url: str, token: str, project_id: str, question: str,
                 timeout: float = 120.0) -> list[dict[str, Any]]:
    """POST one question to the live chat SSE endpoint and collect its events.
    conversation id is ws-{projectId}-{unix_ms} (extra path segments 404)."""
    conv = f"ws-{project_id}-{int(time.time() * 1000)}"
    body = json.dumps({"conversation_id": conv, "message": question}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/stream", data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "Accept": "text/event-stream"},
    )
    events: list[dict[str, Any]] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except ValueError:
                continue
    return events


def run_battery(questions: list[dict[str, Any]], base_url: str, token: str,
                project_id: str) -> list[dict[str, Any]]:
    """Ask each battery question against the live project and judge the route.

    Each question is {"question": str, "route_expected": str}. The expected
    ANSWER is never hard-compared here (figures are the owner's, live): the
    result carries the answer + provenance for human/grader review, and the
    machine PASS/FAIL is on the route the product took."""
    results: list[dict[str, Any]] = []
    for q in questions:
        text, expected = q.get("question", ""), q.get("route_expected", "")
        try:
            turn = turn_from_events(_stream_turn(base_url, token, project_id, text))
            observed = classify_route(turn)
            results.append({
                "question": text, "route_expected": expected, "route_observed": observed,
                "answer": turn["answer"], "provenance": _provenance_digest(turn["provenance"]),
                "sources": sorted(_source_classes(turn["sources"])),
                "route_pass": observed == expected,
            })
        except Exception as exc:  # noqa: BLE001 — one bad turn must not abort the battery
            results.append({"question": text, "route_expected": expected,
                            "route_observed": None, "error": f"{type(exc).__name__}: {exc}",
                            "route_pass": False})
    return results


def main() -> int:
    battery_file = os.environ.get("BATTERY_FILE")
    base_url = os.environ.get("SHOVEL_BASE_URL", "https://theshovel.ai")
    token = os.environ.get("CEREBRUM_TOKEN", "")
    project_id = os.environ.get("SHOVEL_PROJECT_ID", "")
    if not (battery_file and token and project_id):
        print("set BATTERY_FILE (Drive-exported questions), CEREBRUM_TOKEN, SHOVEL_PROJECT_ID")
        return 2
    with open(battery_file, encoding="utf-8") as fh:
        questions = json.load(fh)
    results = run_battery(questions, base_url, token, project_id)
    passed = sum(1 for r in results if r.get("route_pass"))
    print(json.dumps({"total": len(results), "route_pass": passed, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
