"""ONE-USER UI battery runner (per ORDERS_02NOV_0127Z / 0148Z).

Drives the LIVE chat UI as a single logged-in user against the project that
already holds the owner's real documents (saved + embedded), asks each battery
question, and records -- per question -- the route the product actually took,
the answer, and the Why-this-answer inputs + sources, as PASS/FAIL against the
question's declared expected route.

CODE ONLY. The battery questions, expected answers and any figures are NEVER in
this repo (client-content scanner): they are read at run time from BATTERY_FILE
(the .github/workflows/ui-battery.yml dispatch input, exported from the DECISION
to Drive), and the per-question result is uploaded as a workflow artifact for
the caller to post to Drive as ``BATTERY_RESULT_<HHMM>Z``.

Live access uses the SAME repository secrets as smoke-live.yml / eval-battery.yml
(FORK_TOKEN > FORK_API_KEY > FORK_EMAIL+FORK_PASSWORD, FORK_BASE_URL), resolved
through scripts/fork_cli.resolve_auth -- no token is ever handed to a lane or
read from disk. The project is resolved via GET /v1/projects (by PROJECT_NAME,
else the most-indexed project), never a hardcoded id.

The route is OBSERVED from the SSE turn, never guessed. The observation has two
pure, unit-tested pieces: ``turn_from_events`` (raw SSE events -> a normalized
turn) and ``classify_route`` (normalized turn -> one of the five route labels);
``_pick_project`` is the third tested piece. The network driving is thin.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Iterable

# The five routes the battery distinguishes (ORDERS_02NOV_0127Z).
ROUTE_PROJECT_RAG = "project_rag_llm"
ROUTE_GENERAL_RAG = "general_rag_llm"
ROUTE_CALCULATOR = "llm_calculator"
ROUTE_ASKS_INPUTS = "asks_for_inputs"
ROUTE_TOOL_ON_DOC = "tool_on_document"
#: No tool, no retrieved source, no provenance -> nothing to attribute the
#: answer to. Never silently passed as general RAG; it is a battery FAIL.
ROUTE_UNCLASSIFIED = "unclassified"

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
    the retrieved-source class (project corpus vs general). A turn with no tool,
    no retrieved source AND no provenance has nothing to attribute the answer to
    -- it is "unclassified" (a FAIL), never silently passed as general RAG, so a
    silent/empty answer cannot slip through."""
    tools = turn.get("tools") or []
    sources = turn.get("sources") or []
    if _asks_for_inputs(turn):
        return ROUTE_ASKS_INPUTS
    if _CALC_TOOL in tools:
        return ROUTE_CALCULATOR
    if any(t in _DOC_TOOLS for t in tools):
        return ROUTE_TOOL_ON_DOC
    classes = _source_classes(sources)
    if classes & _PROJECT_CLASSES:
        return ROUTE_PROJECT_RAG
    if classes & _GENERAL_CLASSES:
        return ROUTE_GENERAL_RAG
    if not sources and not (turn.get("provenance") or []):
        return ROUTE_UNCLASSIFIED
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


def _indexed_count(project: dict[str, Any]) -> int:
    """Best-effort count of indexed/embedded documents on a project row, across
    the field names the projects listing has used."""
    for key in ("indexed_documents", "indexed_document_count", "embedded_documents",
                "document_count", "documents", "chunks"):
        val = project.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, list):
            return len(val)
    return 0


def _pick_project(projects: list[dict[str, Any]], project_name: str | None) -> dict[str, Any] | None:
    """Resolve the battery's project with NO hardcoded id: by exact name when a
    project_name is given, else the project with the most indexed documents.
    Returns the project row, or None when the list is empty."""
    rows = [p for p in (projects or []) if isinstance(p, dict)]
    if not rows:
        return None
    if project_name:
        named = [p for p in rows if str(p.get("name") or "") == project_name]
        if named:
            return named[0]
    return max(rows, key=_indexed_count)


def _auth_base():
    """(bearer credential, base_url) from the FORK_* env, resolved exactly as
    scripts/fork_cli.py / smoke-live.yml do (FORK_TOKEN > FORK_API_KEY >
    FORK_EMAIL+FORK_PASSWORD login). Imported lazily so the pure route-
    classification helpers import without httpx present."""
    from types import SimpleNamespace

    from scripts import fork_cli  # lazy: keeps classify_route import-light
    base = (os.environ.get("FORK_BASE_URL") or "https://theshovel.ai").rstrip("/")
    cred = fork_cli.resolve_auth(SimpleNamespace(
        token=None, api_key=None, email=None, password=None, base=base))
    return cred, base


def resolve_project(client, base: str, cred: str, project_name: str | None) -> str:
    """The project id to run against, chosen by _pick_project over GET /v1/projects."""
    resp = client.get(f"{base}/v1/projects", headers={"Authorization": f"Bearer {cred}"},
                      timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("projects") if isinstance(data, dict) else data
    chosen = _pick_project(rows if isinstance(rows, list) else [], project_name)
    if not chosen:
        raise RuntimeError("no projects available to run the battery against")
    return str(chosen.get("id") or chosen.get("project_id") or "")


def _stream_turn(client, base: str, cred: str, project_id: str, question: str,
                 agent: str = "project-assistant", timeout: float = 150.0) -> list[dict[str, Any]]:
    """POST one question to the live chat SSE endpoint and collect its events."""
    events: list[dict[str, Any]] = []
    url = f"{base}/v1/agents/{agent}/chat/stream"
    with client.stream("POST", url, json={"message": question, "project_id": project_id},
                       headers={"Authorization": f"Bearer {cred}",
                                "Accept": "text/event-stream"}, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            line = (line or "").strip()
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


def run_battery(questions: list[dict[str, Any]], base: str, cred: str,
                project_id: str, client) -> list[dict[str, Any]]:
    """Ask each battery question against the live project and judge the route.

    The machine PASS/FAIL is on the ROUTE the product took (observed, not
    guessed); the full answer + provenance ride in the result for review but are
    written only to the artifact file, never to stdout (they may carry the
    owner's figures)."""
    results: list[dict[str, Any]] = []
    for q in questions:
        text, expected = q.get("question", ""), q.get("route_expected", "")
        try:
            turn = turn_from_events(_stream_turn(client, base, cred, project_id, text))
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
    import httpx  # lazy: only the live run needs it

    battery_file = os.environ.get("BATTERY_FILE")
    result_file = os.environ.get("RESULT_FILE", "battery_result.json")
    project_name = os.environ.get("PROJECT_NAME") or None
    if not battery_file:
        print("::error::set BATTERY_FILE to the Drive-exported questions JSON")
        return 2
    with open(battery_file, encoding="utf-8") as fh:
        questions = json.load(fh)
    cred, base = _auth_base()  # exits/raises with the smoke-live ::error if creds absent
    with httpx.Client() as client:
        project_id = resolve_project(client, base, cred, project_name)
        results = run_battery(questions, base, cred, project_id, client)
    # The artifact carries the full result (answer + provenance) for review.
    with open(result_file, "w", encoding="utf-8") as fh:
        json.dump({"project_id": project_id, "results": results}, fh, indent=2)
    # Stdout (public Actions log) carries ONLY route PASS/FAIL per question —
    # never the answers, which may contain the owner's figures.
    passed = sum(1 for r in results if r.get("route_pass"))
    for r in results:
        verdict = "PASS" if r.get("route_pass") else "FAIL"
        print(f"[{verdict}] {r.get('route_expected')} -> {r.get('route_observed')}  "
              f"{r.get('question')}")
    print(f"route_pass {passed}/{len(results)}; full per-question result in {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
