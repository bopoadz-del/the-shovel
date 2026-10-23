#!/usr/bin/env python
"""Feature-matrix sweep runner (TASK G — pre-pilot validation).

Drives every prompt in tests/feature_matrix_manifest.yaml through the NORMAL
chat path (POST /v1/agents/{agent}/chat/stream — the same endpoint the browser
and fork_cli use, so nothing bypasses the smart_orchestrator routing gate) and
judges each turn on three mechanical oracles:

  routing:    the `route` SSE event's action matches `expect_route`, or a
              tool_call names something in `expect_tool`. A matched route with
              zero tool calls on a synthetic-agent-tool feature is TOOL_SKIP
              (routing FAIL subtype).
  execution:  no error events, no HTTP failure, answer_chars >= min_chars.
              A clean turn whose answer lands below 30% of min_chars is
              additionally tagged NO_OUTPUT (execution_subtype) — it produced
              ~nothing and should be triaged as fixture/wiring (class 5),
              not read as a quality FAIL.
  structure:  named regex/heuristic checks over the answer text. A check that
              cannot be evaluated mechanically returns SKIP, never PASS.

Verdicts: PASS / PARTIAL (structure FAILs only) / FAIL (routing or execution)
/ BLOCKED (fixture missing — `needs` unmet, detected from the error text, or
pre-marked `blocked: true` in the manifest).

Construction correctness is NOT certified here: every answer is written
verbatim to review_pack/<action>_<n>.md for human review.

Auth (env only, same contract as fork_cli / rag_recall_eval — values are
never printed or stored): FORK_TOKEN or FORK_API_KEY, or
FORK_EMAIL+FORK_PASSWORD (logs in via /v1/users/login). Target from
FORK_BASE_URL (default prod).

Usage:
  python scripts/feature_matrix_sweep.py --dry-run
  python scripts/feature_matrix_sweep.py --list-checks
  python scripts/feature_matrix_sweep.py                      # full sweep
  python scripts/feature_matrix_sweep.py --action generate_wbs
  python scripts/feature_matrix_sweep.py --class must --delay 30
  python scripts/feature_matrix_sweep.py --start 12           # resume a park
  python scripts/feature_matrix_sweep.py --report             # regen matrix

Artifacts (all incremental / kill-safe):
  feature_matrix_results.jsonl   one line per run, appended as it happens
  review_pack/<action>_<n>.md    verbatim answer + oracle evidence
  FEATURE_MATRIX.md              regenerated from the jsonl by --report
  feature_matrix_sweep_state.json  written when the sweep parks itself
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import httpx

# PyYAML ships in the app requirements; degrade with a clear message rather
# than a traceback if someone runs this from a bare interpreter.
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "feature_matrix_manifest.yaml"
DEFAULT_BASE = os.getenv("FORK_BASE_URL", "https://the-fork-jn3t.onrender.com")
REVIEW_DIR = ROOT / "review_pack"
RESULTS_JSONL = ROOT / "feature_matrix_results.jsonl"
REPORT_MD = ROOT / "FEATURE_MATRIX.md"
STATE_FILE = ROOT / "feature_matrix_sweep_state.json"

# Retry contract for live turns: 5xx / 429 / connection errors get ONE retry
# after a long pause (these are heavy LLM turns on a rate-limited tier; a
# quick hammer-retry just deepens the hole). 4xx is deterministic — no retry.
RETRY_WAIT_S = 60
FALLBACK_WINDOW = 10  # rolling runs over which --max-fallback-pct is judged


class ManifestError(ValueError):
    """Manifest fails validation (missing prompts, unknown check names, ...)."""


# ── structure checks ─────────────────────────────────────────────────────────
# Each check takes the full answer text and returns "PASS", "FAIL" or "SKIP".
# SKIP means "cannot be evaluated mechanically on this answer" — it never
# counts as a pass. Keep these honest and simple: they gate PARTIAL vs PASS,
# nothing more; a human reads the review_pack either way.

STRUCTURE_CHECKS: dict = {}


def _check(fn):
    STRUCTURE_CHECKS[fn.__name__] = fn
    return fn


_LIST_ROW_RE = re.compile(r"^\s*(?:\|\s*\S|[-*•]\s+\S|\d{1,3}[.)]\s+\S)", re.M)
_UNIT_WORD = r"(?:m3|m³|m2|m²|mm|cm|km|kg|t|tonnes?|nr|no\.?|ls|lm|sqm|cum|each|m)"
_MONTH = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
          r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
          r"Nov(?:ember)?|Dec(?:ember)?")
_DATE_RE = re.compile(
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"          # 15/03/2027
    r"|\d{4}-\d{2}-\d{2}"                      # 2027-03-15
    r"|(?:" + _MONTH + r")\.?\s+\d{4}"         # March 2027
    r"|Q[1-4][\s-]?20\d{2}"                    # Q3 2027
    r"|\b20\d{2}\b",                           # bare year (weakest)
    re.I,
)


@_check
def cites_sources(text: str) -> str:
    """A source/document/drawing reference or citation marker is present."""
    if re.search(r"\[\d+\]|\[source|\(source|source[s]?\s*:", text, re.I):
        return "PASS"
    if re.search(r"\.(pdf|xlsx?|docx?|dwg|dxf|xer|ifc|csv)\b", text, re.I):
        return "PASS"
    if re.search(r"\baccording to\b|\bas per the\b|\breferenced (in|from)\b", text, re.I):
        return "PASS"
    if re.search(r"\b(document|drawing|specification|report)\b[^\n]{0,60}"
                 r"(titled|named|no\.?|number|ref)", text, re.I):
        return "PASS"
    return "FAIL"


@_check
def has_numbers(text: str) -> str:
    """At least 3 distinct numeric values."""
    return "PASS" if len(set(re.findall(r"\d+(?:[.,]\d+)?", text))) >= 3 else "FAIL"


@_check
def has_currency_or_units(text: str) -> str:
    """A currency or measurement unit token appears."""
    if re.search(r"\b(SAR|USD|AED|m3|m³|m2|m²|kg|t|mm|m)\b", text):
        return "PASS"
    return "FAIL"


@_check
def boq_rows_have_qty_unit(text: str) -> str:
    """At least 3 table-ish lines carrying both a quantity and a unit."""
    rows = 0
    for line in text.splitlines():
        # "table-ish" = markdown pipe row, tab-separated, or aligned columns
        if not re.search(r"\||\t| {2,}", line):
            continue
        if re.search(r"\d", line) and re.search(r"\b" + _UNIT_WORD + r"\b", line, re.I):
            rows += 1
    return "PASS" if rows >= 3 else "FAIL"


@_check
def has_activities_and_durations(text: str) -> str:
    """At least 5 activity lines with a day/week/month duration."""
    n = sum(1 for line in text.splitlines()
            if re.search(r"\d+\s*(?:working\s+)?(day|week|month)s?\b", line, re.I)
            or re.search(r"\((?:Day|Week|Month)\s*\d+\s*-\s*\d+\)", line, re.I))
    return "PASS" if n >= 5 else "FAIL"


@_check
def has_period_buckets(text: str) -> str:
    """At least 3 distinct month/period labels (Jan.., Month 3, M4, Q2, Week 7)."""
    labels = set(m.group(0).lower()[:3] for m in re.finditer(r"\b(?:" + _MONTH + r")\b", text))
    labels |= set(m.group(0).lower().replace(" ", "")
                  for m in re.finditer(r"\b(?:month|period|week|wk)\s*[-#]?\s*\d{1,2}\b"
                                       r"|\bM\d{1,2}\b|\bQ[1-4]\b", text, re.I))
    return "PASS" if len(labels) >= 3 else "FAIL"


@_check
def milestones_have_dates(text: str) -> str:
    """At least 2 date-like strings on lines that mention milestone words."""
    words = re.compile(r"milestone|completion|handover|award|mobili[sz]|"
                       r"commencement|substantial|energi[sz]|start|finish", re.I)
    n = sum(1 for line in text.splitlines()
            if words.search(line) and _DATE_RE.search(line))
    return "PASS" if n >= 2 else "FAIL"


@_check
def scurve_cumulative_nondecreasing(text: str) -> str:
    """If a cumulative %/value series is present it must be non-decreasing.

    SKIP when no series is found — absence of an extractable series is a
    presentation question, not a correctness failure we can judge here.
    """
    series: list = []
    for line in text.splitlines():
        if re.search(r"cumulative|cum\.|s[- ]?curve", line, re.I):
            series.extend(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", line))
    if len(series) < 3:
        return "SKIP"
    return "PASS" if all(a <= b + 1e-9 for a, b in zip(series, series[1:])) else "FAIL"


@_check
def has_line_items(text: str) -> str:
    """At least 5 bullet / numbered / table rows."""
    return "PASS" if len(_LIST_ROW_RE.findall(text)) >= 5 else "FAIL"


@_check
def has_sections(text: str) -> str:
    """At least 3 markdown headings or numbered section headings."""
    n = len(re.findall(r"^#{1,6}\s+\S", text, re.M))
    n += len(re.findall(r"^(?:\*\*)?\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{2,}", text, re.M))
    n += len(re.findall(r"^\*\*[^*\n]{3,}\*\*:?\s*$", text, re.M))
    return "PASS" if n >= 3 else "FAIL"


@_check
def mentions_standard_or_grade(text: str) -> str:
    """A standards body, concrete class (C40) or the word grade appears."""
    if re.search(r"\b(ASTM|ACI|BS|EN|SASO|SBC|ISO)\b", text):
        return "PASS"
    if re.search(r"\bC\d{2}(?:/\d{2})?\b", text) or re.search(r"\bgrade\b", text, re.I):
        return "PASS"
    return "FAIL"


@_check
def has_checklist_items(text: str) -> str:
    """At least 8 checkbox / bullet / numbered checklist or table rows."""
    n = len(re.findall(r"^\s*(?:[-*•]\s*\[[ xX]?\]|\[[ xX]?\]|[-*•]|\d{1,3}[.)])\s+\S",
                       text, re.M))
    # Markdown table rows that look like numbered checklist items (| 1.1 | Item | ...).
    n += len(re.findall(r"^\s*\|\s*\d{1,3}(?:\.\d+)*\s*\|.*\|\s*$", text, re.M))
    return "PASS" if n >= 8 else "FAIL"


@_check
def cites_standards(text: str) -> str:
    """At least 2 distinct standards references (ASTM C150, ACI 318, BS EN 206...)."""
    refs = set(re.findall(
        r"\b(?:ASTM\s?[A-Z]?\s?\d+|ACI\s?\d+(?:\.\d+)?|BS\s?EN\s?\d+|BS\s?\d+|"
        r"EN\s?\d+|IEC\s?\d+|IEEE\s?\d+|NFPA\s?\d+|ISO\s?\d+|SBC\s?\d+|"
        r"SASO\s?[A-Z0-9]+|NEMA\s?[A-Z0-9]+)\b", text))
    return "PASS" if len(refs) >= 2 else "FAIL"


@_check
def references_drawing(text: str) -> str:
    """A drawing number pattern, or the word drawing with an identifier nearby."""
    if re.search(r"\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6})*-\d{2,}\b", text):
        return "PASS"
    if re.search(r"\bdrawing[s]?\b[^\n]{0,60}\d", text, re.I):
        return "PASS"
    return "FAIL"


@_check
def shows_working_or_rule(text: str) -> str:
    """An equation, a numeric operation, or an explicit rule statement."""
    if re.search(r"=\s*[\d(]", text) or re.search(r"\d\s*[x×*/÷]\s*\d", text, re.I):
        return "PASS"
    if re.search(r"\b(formula|equation|rule of thumb|calculated as|"
                 r"per\s+(?:ACI|BS|ASTM|EN|CIRIA)\b)", text, re.I):
        return "PASS"
    return "FAIL"


@_check
def lists_documents(text: str) -> str:
    """At least 3 filename-ish entries, or 3 list rows naming document types."""
    if len(re.findall(r"\.(pdf|xlsx?|docx?|dwg|dxf|xer|ifc|csv|rvt)\b", text, re.I)) >= 3:
        return "PASS"
    doc_word = re.compile(r"\b(drawing|plan|report|spec(?:ification)?|boq|bill|"
                          r"schedule|manual|contract|method statement|register)\b", re.I)
    n = sum(1 for line in text.splitlines()
            if _LIST_ROW_RE.match(line) and doc_word.search(line))
    return "PASS" if n >= 3 else "FAIL"


# ── manifest ─────────────────────────────────────────────────────────────────

def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict:
    """Parse + validate the manifest. Raises ManifestError on any defect so a
    typo in a structure-check name dies at load time, not mid-sweep."""
    if yaml is None:
        raise ManifestError("PyYAML is not installed (pip install pyyaml)")
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "features" not in raw:
        raise ManifestError("manifest must be a mapping with a `features` list")

    defaults = raw.get("defaults") or {}
    problems: list = []
    features: list = []
    seen: set = set()

    for i, feat in enumerate(raw["features"] or []):
        action = feat.get("action")
        if not action:
            problems.append(f"features[{i}]: missing `action`")
            continue
        if action in seen:
            problems.append(f"features[{i}]: duplicate action `{action}`")
        seen.add(action)

        prompts = feat.get("prompts") or []
        if not prompts:
            problems.append(f"{action}: no prompts")
        for j, p in enumerate(prompts):
            if not (p.get("text") or "").strip():
                problems.append(f"{action}.prompts[{j}]: empty text")
            if not p.get("expect_route"):
                problems.append(f"{action}.prompts[{j}]: missing expect_route")

        for name in feat.get("structure") or []:
            if name not in STRUCTURE_CHECKS:
                problems.append(f"{action}: unknown structure check `{name}`")

        features.append({
            "action": action,
            "feature": feat.get("feature", ""),
            "class": feat.get("class", "coverage"),
            "handler": feat.get("handler", ""),
            "project": feat.get("project", defaults.get("project")),
            "min_chars": int(feat.get("min_chars", defaults.get("min_chars", 0))),
            "agent": feat.get("agent", defaults.get("agent", "project-assistant")),
            "structure": list(feat.get("structure") or []),
            "needs": feat.get("needs", ""),
            "blocked": bool(feat.get("blocked", False)),
            "prompts": [{"text": p["text"],
                         "expect_route": list(p.get("expect_route") or []),
                         "expect_tool": list(p.get("expect_tool") or [])}
                        for p in prompts],
        })

    if problems:
        raise ManifestError("manifest validation failed:\n  - " + "\n  - ".join(problems))
    return {"defaults": defaults, "features": features}


def _resolve_fixture_projects(manifest: dict, base: str, headers: dict) -> None:
    """Replace canonical fixture names in the manifest with live project ids.

    Fixture projects are disposable and recreated by scripts/seed_fixtures.py;
    hard-coding their ids in the manifest is a defect. We query the project
    list once, map names to ids, and mark any unresolved fixture as blocked.
    """
    names_to_resolve = {
        f["project"] for f in manifest["features"]
        if isinstance(f.get("project"), str) and f["project"].startswith("FIXTURE —")
    }
    if not names_to_resolve:
        return

    try:
        with httpx.Client(base_url=base, headers=headers, timeout=30) as client:
            r = client.get("/v1/projects")
            r.raise_for_status()
            name_to_id = {p.get("name"): p.get("id")
                          for p in r.json().get("projects", []) or []}
    except httpx.HTTPError as exc:
        print(f"[fixture-resolve] warning: could not list projects: {exc}",
              file=sys.stderr)
        name_to_id = {}

    for feat in manifest["features"]:
        name = feat.get("project")
        if not isinstance(name, str) or not name.startswith("FIXTURE —"):
            continue
        resolved = name_to_id.get(name)
        if resolved:
            feat["project"] = resolved
            feat["_resolved_from"] = name
        else:
            feat["blocked"] = True
            feat["needs"] = (feat.get("needs") or "") + f" (fixture project {name!r} missing — run scripts/seed_fixtures.py)"
            print(f"[fixture-resolve] {feat['action']}: {name!r} not found -> BLOCKED",
                  file=sys.stderr)


def build_plan(manifest: dict, action: str | None, klass: str | None,
               runs_per_prompt: int) -> list:
    """Flatten features into an ordered run plan. Plan indices are stable for
    a given (manifest, --action, --class) so --start can resume a parked sweep;
    --start/--end are applied by the caller AFTER this."""
    plan = []
    idx = 0
    for feat in manifest["features"]:
        if action and feat["action"] != action:
            continue
        if klass and feat["class"] != klass:
            continue
        for p_i, prompt in enumerate(feat["prompts"]):
            for r_i in range(runs_per_prompt):
                plan.append({"plan_index": idx, "feature": feat,
                             "prompt_index": p_i, "run_index": r_i,
                             "prompt": prompt})
                idx += 1
    return plan


# ── auth / transport ─────────────────────────────────────────────────────────

def _auth_header(base: str) -> dict:
    """Env-only credentials, same contract as fork_cli. The value is used in
    the header and never echoed anywhere."""
    tok = os.getenv("FORK_TOKEN") or os.getenv("FORK_API_KEY")
    if not tok:
        email, pw = os.getenv("FORK_EMAIL"), os.getenv("FORK_PASSWORD")
        if not (email and pw):
            sys.exit("No credentials: set FORK_TOKEN / FORK_API_KEY / "
                     "FORK_EMAIL+FORK_PASSWORD")
        r = httpx.post(f"{base}/v1/users/login",
                       json={"email": email, "password": pw}, timeout=30)
        r.raise_for_status()
        tok = r.json()["token"]
        print("[auth] logged in via FORK_EMAIL", file=sys.stderr)
    return {"Authorization": f"Bearer {tok}"}


def stream_prompt(client: httpx.Client, base: str, agent: str, headers: dict,
                  message: str, project_id: str | None,
                  conversation_id: str) -> dict:
    """One chat turn over SSE. Mirrors fork_cli's request shape exactly:
    same endpoint, same payload keys, same Bearer header, same timeouts."""
    url = f"{base}/v1/agents/{agent}/chat/stream"
    body: dict = {"message": message}
    if project_id:
        body["project_id"] = project_id
    body["conversation_id"] = conversation_id

    out = {
        "http_status": None, "http_error": None,
        "route": None, "tool_calls": [], "tool_results": [],
        "answer": "", "errors": [], "end": None,
        "first_token_s": None, "total_s": None,
        "served_model": None, "fallback_used": False,
        "sources_count": 0, "n_events": 0,
    }
    t0 = time.monotonic()
    answer_parts: list = []
    pending_tool: tuple | None = None  # (name, started_at) for duration_ms

    with client.stream("POST", url, json=body, headers=headers,
                       timeout=httpx.Timeout(10, read=600)) as r:
        out["http_status"] = r.status_code
        if r.status_code != 200:
            r.read()
            out["http_error"] = f"HTTP {r.status_code}: {r.text[:300]}"
            return out
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            out["n_events"] += 1
            try:
                evt = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            # Any event carrying a truthy fallback_used marks the run degraded
            if isinstance(evt, dict) and evt.get("fallback_used"):
                out["fallback_used"] = True
            etype = evt.get("type", "?")

            if etype == "token":
                if out["first_token_s"] is None:
                    out["first_token_s"] = round(time.monotonic() - t0, 2)
                answer_parts.append(evt.get("content", ""))
            elif etype == "route":
                out["route"] = {k: v for k, v in evt.items() if k != "type"}
            elif etype == "tool_call":
                name = evt.get("tool") or evt.get("name") or "?"
                pending_tool = (name, time.monotonic())
                out["tool_calls"].append({
                    "name": name,
                    "args": evt.get("args_preview") or evt.get("arguments")
                            or evt.get("args") or "",
                    "id": evt.get("id"),
                })
            elif etype == "tool_result":
                name = evt.get("tool") or evt.get("name") or "?"
                dur_ms = None
                if pending_tool and pending_tool[0] == name:
                    dur_ms = round((time.monotonic() - pending_tool[1]) * 1000)
                    pending_tool = None
                ok = bool(evt.get("ok", True))
                out["tool_results"].append({
                    "name": name, "ok": ok, "duration_ms": dur_ms,
                    "error": None if ok else (evt.get("summary") or evt.get("error") or "")[:300],
                })
            elif etype == "error":
                out["errors"].append(str(evt.get("message", ""))[:500])
                break
            elif etype == "end":
                out["end"] = {k: v for k, v in evt.items()
                              if k not in ("type", "sources", "exports")}
                out["served_model"] = evt.get("model") or out["served_model"]
                out["sources_count"] = len(evt.get("sources") or [])
                # Capture the DELIVERABLE offers. These were dropped, so a
                # feature whose whole product is a downloadable workbook
                # recorded exports=null and was then judged purely on how
                # chatty its summary was -- see judge_execution.
                out["exports"] = list(evt.get("exports") or [])
                out["exports_count"] = len(out["exports"])
                break
            elif etype == "sources":
                # some paths emit sources as their own event instead of on end
                out["sources_count"] = max(
                    out["sources_count"],
                    len(evt.get("sources") or evt.get("items") or []))
            # start / heartbeat: nothing to record beyond the event count

    out["answer"] = "".join(answer_parts)
    out["total_s"] = round(time.monotonic() - t0, 2)
    return out


def execute_with_retry(client, base, agent, headers, message, project_id,
                       conversation_id) -> dict:
    """5xx / 429 / connection errors get one retry after RETRY_WAIT_S; the
    second failure is recorded as EXECUTION-FAIL evidence, not raised."""
    last: dict | None = None
    for attempt in range(2):
        if attempt:
            print(f"    retryable failure - waiting {RETRY_WAIT_S}s and retrying once",
                  file=sys.stderr)
            time.sleep(RETRY_WAIT_S)
        try:
            res = stream_prompt(client, base, agent, headers, message,
                                project_id, conversation_id)
        except httpx.HTTPError as exc:
            last = {"http_status": None,
                    "http_error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "route": None, "tool_calls": [], "tool_results": [],
                    "answer": "", "errors": [], "end": None,
                    "first_token_s": None, "total_s": None,
                    "served_model": None, "fallback_used": False,
                    "sources_count": 0, "n_events": 0}
            continue
        status = res.get("http_status")
        if status is not None and (status >= 500 or status == 429):
            last = res
            continue
        return res
    return last  # type: ignore[return-value]


# ── oracles ──────────────────────────────────────────────────────────────────

# BLOCKED heuristic: only consulted when the feature declares `needs` and
# execution already failed — a fixture gap, not a platform bug. Includes the
# platform's honest-refusal phrasings (the fabrication-kill answers: "I don't
# have that information … upload the register and I'll extract it") so an
# honest no-data answer on a fixture-dependent feature triages as the fixture
# gap it is, not as a broken feature.
_BLOCKED_RE = re.compile(
    r"(no|not|missing|cannot|can't|couldn'?t|unable to)\s+"
    r"(find|found|locate|available|access)"
    r"|please upload|no (such )?([\w-]+ ){0,2}(file|document|drawing|model|programme|"
    r"schedule|bim|ifc|xer)|not (been )?(uploaded|indexed|provided)"
    r"|don'?t have (that|this|the) (information|data)"
    r"|does not (contain|include)"
    r"|upload (it|one|the [\w ]{1,30}) and I'?ll"
    r"|would need [\w ]{0,40}(uploaded|indexed|register|log|document)",
    re.I,
)


def judge_routing(feature: dict, prompt: dict, res: dict) -> tuple:
    """Returns (verdict, subtype). Route match OR expected-tool match passes;
    a matched route with zero tool calls on a synthetic-agent-tool handler is
    TOOL_SKIP — the router said the right thing but no real work ran."""
    route_action = (res.get("route") or {}).get("action")
    tool_names = [tc["name"] for tc in res.get("tool_calls") or []]
    matched_route = route_action in prompt["expect_route"]
    matched_tool = bool(prompt["expect_tool"]) and any(
        n in prompt["expect_tool"] for n in tool_names)
    if matched_route or matched_tool:
        if matched_route and not tool_names and \
                "synthetic agent tool" in (feature.get("handler") or ""):
            return "FAIL", "TOOL_SKIP"
        return "PASS", None
    return "FAIL", "ROUTE_MISS"


def judge_execution(feature: dict, res: dict) -> tuple:
    """Returns (verdict, reasons)."""
    reasons = []
    if res.get("http_error"):
        reasons.append(res["http_error"])
    for err in res.get("errors") or []:
        reasons.append(f"error event: {err}")
    # A turn that shipped a DELIVERABLE is not judged on inline verbosity.
    #
    # Verified live 2026-08-03: generate_wbs answered in 233 chars --
    #   "Schedule built: 204 activities over 688 working days (44 on the
    #    critical path) ... workbook ready to download"
    # -- against min_chars 1500, so it scored FAIL/NO_OUTPUT. But POSTing the
    # offered endpoint returned HTTP 200 and a valid 37KB xlsx: 5 sheets
    # (L2 Schedule, Cost Loading, Manpower Histogram, Milestones, Summary),
    # 210 rows of real CPM data. The feature worked perfectly; the oracle was
    # measuring the wrong artifact, and 19 of 23 sweep FAILs were this class.
    #
    # min_chars still applies to answer-only features (exports empty), where a
    # short answer really is a husk.
    if res.get("exports"):
        return ("PASS" if not reasons else "FAIL"), reasons
    if len(res.get("answer") or "") < feature["min_chars"]:
        reasons.append(f"answer_chars {len(res.get('answer') or '')} "
                       f"< min_chars {feature['min_chars']}")
    return ("PASS" if not reasons else "FAIL"), reasons


# A short answer and a NEAR-EMPTY answer are different failures. The 2026-07
# results audit (docs/PILOT_FOLLOWUP.md on the retrieval-fixes branch) showed
# most `answer_chars < min_chars` FAILs were 60-250-char husks — a feature that
# produced essentially nothing (unrouted, fixture-absent, or a silently
# swallowed block error), not a correct-but-terse answer. Conflating the two
# made the headline pass-rate untriageable. Threshold per that audit: below
# 30% of min_chars with no error event = NO_OUTPUT.
_NO_OUTPUT_FRACTION = 0.3


def execution_subtype(feature: dict, res: dict) -> "str | None":
    """``NO_OUTPUT`` when the turn 'succeeded' but produced almost nothing.

    Only claims NO_OUTPUT when there is no explicit error signal — an HTTP
    failure or error event already names the real cause and must stay the
    headline reason.
    """
    if res.get("http_error") or (res.get("errors") or []):
        return None
    # A turn that produced a downloadable deliverable produced OUTPUT, however
    # terse its summary. Calling that "NO_OUTPUT" is the exact misread that
    # tagged generate_wbs (233 chars + a working 37KB workbook) as producing
    # essentially nothing.
    if res.get("exports"):
        return None
    min_chars = int(feature.get("min_chars") or 0)
    if min_chars <= 0:
        return None
    answer_len = len(res.get("answer") or "")
    if answer_len < max(1, int(min_chars * _NO_OUTPUT_FRACTION)):
        return "NO_OUTPUT"
    return None


def judge_structure(feature: dict, answer: str) -> dict:
    return {name: STRUCTURE_CHECKS[name](answer) for name in feature["structure"]}


def run_verdict(feature: dict, routing: str, execution: str,
                exec_reasons: list, structure: dict, res: dict) -> str:
    if feature.get("blocked"):
        return "BLOCKED"
    if execution == "FAIL" and feature.get("needs"):
        blob = " ".join(exec_reasons) + " " + (res.get("answer") or "")[:2000]
        if _BLOCKED_RE.search(blob):
            return "BLOCKED"
    if routing == "FAIL" or execution == "FAIL":
        return "FAIL"
    if any(v == "FAIL" for v in structure.values()):
        return "PARTIAL"
    return "PASS"


def triage_class(record: dict) -> str:
    """1 routing miss / 2 block crash / 3 structural gap / 4 fixture gap /
    5 no output (ran 'clean' but produced ~nothing — likely fixture-absent or
    a silently swallowed block error; triage as wiring, not answer quality)."""
    v = record["verdict"]
    if v == "BLOCKED":
        return "4"
    if v == "FAIL":
        if record["routing"] == "FAIL":
            return "1"
        if record.get("execution_subtype") == "NO_OUTPUT":
            return "5"
        return "2"
    if v == "PARTIAL":
        return "3"
    return ""


# ── evidence ─────────────────────────────────────────────────────────────────

def write_evidence(record: dict, res: dict) -> Path:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = REVIEW_DIR / f"{record['action']}_{record['evidence_n']}.md"
    lines = [
        f"# {record['action']} — prompt {record['prompt_index'] + 1}, "
        f"run {record['run_index'] + 1}",
        "",
        f"- verdict: {record['verdict']}",
        f"- prompt: {record['prompt']}",
        f"- project: {record['project']}",
        f"- conversation_id: {record['conversation_id']}",
        f"- route event: `{json.dumps(res.get('route'))}`",
        f"- tool_calls: {json.dumps(res.get('tool_calls'))}",
        f"- tool_results: {json.dumps(res.get('tool_results'))}",
        f"- routing oracle: {record['routing']}"
        + (f" ({record['routing_subtype']})" if record.get("routing_subtype") else ""),
        f"- execution oracle: {record['execution']}"
        + (f" ({record['execution_subtype']})" if record.get("execution_subtype") else "")
        + (f" — {'; '.join(record['execution_reasons'])}" if record["execution_reasons"] else ""),
        f"- structure: {json.dumps(record['structure'])}",
        f"- first_token_s: {res.get('first_token_s')}  total_s: {res.get('total_s')}",
        f"- served_model: {res.get('served_model')}  "
        f"fallback_used: {res.get('fallback_used')}",
        f"- answer_chars: {len(res.get('answer') or '')}  "
        f"sources: {res.get('sources_count')}  events: {res.get('n_events')}",
    ]
    if res.get("http_error"):
        lines.append(f"- http_error: {res['http_error']}")
    if res.get("errors"):
        lines.append(f"- error events: {json.dumps(res['errors'])}")
    lines += ["", "## Answer (verbatim)", "", res.get("answer") or "(no answer text)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def append_result(record: dict) -> None:
    # One JSON line per run, flushed immediately — a killed sweep keeps every
    # finished row and --report / --start work off whatever landed.
    with open(RESULTS_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()


# ── report ───────────────────────────────────────────────────────────────────

_SEVERITY = {"FAIL": 3, "BLOCKED": 2, "PARTIAL": 1, "PASS": 0}


def generate_report() -> None:
    if not RESULTS_JSONL.exists():
        sys.exit(f"no results yet: {RESULTS_JSONL} not found")
    # Re-runs of the same (action, prompt, run) overwrite older rows: last wins.
    rows: dict = {}
    with open(RESULTS_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[(rec["action"], rec["prompt_index"], rec["run_index"])] = rec

    by_action: dict = {}
    for rec in rows.values():
        by_action.setdefault(rec["action"], []).append(rec)

    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "BLOCKED": 0}
    no_output_features = 0
    table = []
    for action in sorted(by_action, key=lambda a: min(r["ts"] for r in by_action[a])):
        recs = sorted(by_action[action],
                      key=lambda r: (r["prompt_index"], r["run_index"]))
        verdict = max((r["verdict"] for r in recs), key=lambda v: _SEVERITY[v])
        counts[verdict] += 1
        worst = max((r for r in recs if r["verdict"] == verdict),
                    key=lambda r: _SEVERITY[r["verdict"]])
        if verdict == "FAIL" and triage_class(worst) == "5":
            no_output_features += 1
        latencies = [r["total_s"] for r in recs if r.get("total_s")]
        latency = f"{max(latencies):.0f}s" if latencies else "-"
        routed = ", ".join(sorted({str(r.get("route_action")) for r in recs}))
        prompts = "<br>".join(
            (r["prompt"][:70] + ("..." if len(r["prompt"]) > 70 else ""))
            for r in recs)
        evidence = "<br>".join(f"[{Path(r['evidence']).name}]({r['evidence']})"
                               for r in recs if r.get("evidence"))
        table.append((action, recs[0].get("class", ""), prompts, routed,
                      verdict, latency, evidence, triage_class(worst)))

    lines = [
        "# FEATURE_MATRIX — TASK G sweep results",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"from {RESULTS_JSONL.name} ({len(rows)} runs, {len(by_action)} features).",
        "",
        f"PASS {counts['PASS']} | PARTIAL {counts['PARTIAL']} | "
        f"FAIL {counts['FAIL']}"
        + (f" (of which NO_OUTPUT {no_output_features})" if no_output_features else "")
        + f" | BLOCKED {counts['BLOCKED']}",
        "",
        "Triage classes: 1 = routing miss, 2 = block crash / HTTP failure, "
        "3 = structural gap, 4 = fixture gap, 5 = no output (ran clean but "
        "produced ~nothing — triage as fixture/wiring, not answer quality).",
        "",
        "| action | class | prompt(s) | routed action | verdict | latency | evidence | triage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        lines.append("| " + " | ".join(str(c) for c in row[:2])
                     + " | " + row[2] + " | " + " | ".join(str(c) for c in row[3:]) + " |")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_MD} - features: {len(by_action)}  "
          f"PASS={counts['PASS']} PARTIAL={counts['PARTIAL']} "
          f"FAIL={counts['FAIL']} BLOCKED={counts['BLOCKED']}")


# ── sweep ────────────────────────────────────────────────────────────────────

def run_sweep(args, manifest: dict) -> int:
    plan = build_plan(manifest, args.action, args.klass, args.runs_per_prompt)
    lo, hi = args.start, (args.end or len(plan))
    plan = [p for p in plan if lo <= p["plan_index"] < hi]
    if not plan:
        sys.exit("empty plan after filters")

    headers = _auth_header(args.base)
    expected_model = args.expected_model  # optional substring; env-driven
    fallback_window: deque = deque(maxlen=FALLBACK_WINDOW)
    print(f"sweep: {len(plan)} runs  base={args.base}  delay={args.delay}s  "
          f"max_fallback_pct={args.max_fallback_pct}")
    print("=" * 76)

    with httpx.Client() as client:
        for n, item in enumerate(plan):
            feat = item["feature"]
            prompt = item["prompt"]
            if n and args.delay:
                time.sleep(args.delay)

            # Fresh conversation per prompt-run: the endpoint does not require
            # continuity and a shared history is exactly the contamination
            # vector this sweep must exclude (poisoned-history routing drift).
            conversation_id = f"fmx-{uuid.uuid4().hex[:12]}"
            evidence_n = item["prompt_index"] * args.runs_per_prompt + item["run_index"] + 1

            print(f"[{item['plan_index']:>3}] {feat['action']} "
                  f"p{item['prompt_index'] + 1} r{item['run_index'] + 1}: "
                  f"{prompt['text'][:60]}")

            if feat["blocked"]:
                res = {"route": None, "tool_calls": [], "tool_results": [],
                       "answer": "", "errors": [], "end": None,
                       "first_token_s": None, "total_s": None,
                       "served_model": None, "fallback_used": False,
                       "sources_count": 0, "n_events": 0,
                       "http_status": None,
                       "http_error": "pre-marked blocked in manifest (fixture missing)"}
            else:
                res = execute_with_retry(client, args.base, feat["agent"],
                                         headers, prompt["text"],
                                         feat["project"], conversation_id)

            if expected_model and res.get("served_model") and \
                    expected_model.lower() not in res["served_model"].lower():
                res["fallback_used"] = True

            routing, subtype = judge_routing(feat, prompt, res)
            execution, exec_reasons = judge_execution(feat, res)
            exec_subtype = execution_subtype(feat, res) if execution == "FAIL" else None
            structure = judge_structure(feat, res.get("answer") or "")
            verdict = run_verdict(feat, routing, execution, exec_reasons,
                                  structure, res)

            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "plan_index": item["plan_index"],
                "action": feat["action"], "class": feat["class"],
                "prompt_index": item["prompt_index"],
                "run_index": item["run_index"],
                "prompt": prompt["text"], "project": feat["project"],
                "conversation_id": conversation_id,
                "expect_route": prompt["expect_route"],
                "expect_tool": prompt["expect_tool"],
                "route_action": (res.get("route") or {}).get("action"),
                "route": res.get("route"),
                "tool_calls": [tc["name"] for tc in res.get("tool_calls") or []],
                "tool_results": res.get("tool_results"),
                "routing": routing, "routing_subtype": subtype,
                "execution": execution, "execution_reasons": exec_reasons,
                "execution_subtype": exec_subtype,
                "structure": structure, "verdict": verdict,
                "answer_chars": len(res.get("answer") or ""),
                "first_token_s": res.get("first_token_s"),
                "total_s": res.get("total_s"),
                "served_model": res.get("served_model"),
                "fallback_used": bool(res.get("fallback_used")),
                "sources_count": res.get("sources_count"),
                "http_error": res.get("http_error"),
                "errors": res.get("errors"),
                "evidence_n": evidence_n,
            }
            # posix-style relative path so the markdown links in
            # FEATURE_MATRIX.md work on any viewer, including GitHub
            record["evidence"] = write_evidence(record, res).relative_to(ROOT).as_posix()
            append_result(record)

            print(f"      {verdict:8s} routing={routing}"
                  + (f"({subtype})" if subtype else "")
                  + f" exec={execution} chars={record['answer_chars']}"
                  + f" total={record['total_s']}s"
                  + (" FALLBACK" if record["fallback_used"] else ""))

            # Park the sweep when the serving tier is degraded: judging the
            # matrix against a fallback model would mislabel routing/structure
            # gaps as platform failures. State on disk makes --start resumable.
            fallback_window.append(1 if record["fallback_used"] else 0)
            if len(fallback_window) == FALLBACK_WINDOW:
                pct = 100.0 * sum(fallback_window) / FALLBACK_WINDOW
                if pct > args.max_fallback_pct:
                    next_start = item["plan_index"] + 1
                    state = {
                        "stopped_at": record["ts"],
                        "reason": f"fallback_pct {pct:.0f}% > "
                                  f"{args.max_fallback_pct}% over last "
                                  f"{FALLBACK_WINDOW} runs",
                        "next_start": next_start,
                        "filters": {"action": args.action, "class": args.klass,
                                    "runs_per_prompt": args.runs_per_prompt},
                    }
                    STATE_FILE.write_text(json.dumps(state, indent=2),
                                          encoding="utf-8")
                    print("=" * 76)
                    print(f"PARKED: {state['reason']}")
                    print(f"state written to {STATE_FILE.name}; resume with "
                          f"--start {next_start} (same filters)")
                    return 3

    print("=" * 76)
    print(f"sweep complete: {len(plan)} runs -> {RESULTS_JSONL.name}; "
          f"run --report to regenerate {REPORT_MD.name}")
    return 0


# ── dry-run / list-checks ────────────────────────────────────────────────────

def dry_run(args, manifest: dict) -> int:
    plan = build_plan(manifest, args.action, args.klass, args.runs_per_prompt)
    lo, hi = args.start, (args.end or len(plan))
    plan = [p for p in plan if lo <= p["plan_index"] < hi]

    feats = {p["feature"]["action"]: p["feature"] for p in plan}
    by_class: dict = {}
    for f in feats.values():
        by_class[f["class"]] = by_class.get(f["class"], 0) + 1
    n_prompts = len({(p["feature"]["action"], p["prompt_index"]) for p in plan})
    used_checks = sorted({c for f in feats.values() for c in f["structure"]})
    needy = [f["action"] for f in feats.values() if f["needs"]]

    print(f"manifest: {args.manifest}")
    print(f"features: {len(feats)} ("
          + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items())) + ")")
    print(f"prompts: {n_prompts}   planned runs: {len(plan)} "
          f"(runs-per-prompt={args.runs_per_prompt})")
    # load_manifest already died if any referenced check lacks an
    # implementation, so reaching this line proves the registry is complete.
    print(f"structure checks referenced: {len(used_checks)} (all implemented)")
    print(f"features with fixture needs (candidate BLOCKED): {len(needy)}: "
          + ", ".join(needy))
    print("-" * 76)
    for p in plan:
        f = p["feature"]
        print(f"[{p['plan_index']:>3}] {f['class']:8s} {f['action']:32s} "
              f"p{p['prompt_index'] + 1} r{p['run_index'] + 1}  "
              f"{p['prompt']['text'][:70]}")
    print("-" * 76)
    print("dry-run OK - no network calls made")
    return 0


def list_checks() -> int:
    for name in sorted(STRUCTURE_CHECKS):
        doc = (STRUCTURE_CHECKS[name].__doc__ or "").strip().splitlines()[0]
        print(f"{name:36s} {doc}")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="TASK G feature-matrix sweep vs the live chat path")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--action", help="run a single feature by action name")
    ap.add_argument("--class", dest="klass", choices=["must", "coverage"],
                    help="run only one class of features")
    ap.add_argument("--start", type=int, default=0,
                    help="first plan index to run (resume point after a park)")
    ap.add_argument("--end", type=int, default=0,
                    help="stop before this plan index (0 = all)")
    ap.add_argument("--runs-per-prompt", type=int, default=1)
    ap.add_argument("--delay", type=float, default=20,
                    help="seconds between prompts — heavy LLM turns on a "
                         "rate-limited tier need room (default 20)")
    ap.add_argument("--max-fallback-pct", type=float, default=40,
                    help="park the sweep when the rolling 10-run fallback "
                         "rate exceeds this (default 40)")
    ap.add_argument("--expected-model",
                    default=os.getenv("FORK_EXPECTED_MODEL") or None,
                    help="substring of the expected primary model; a served "
                         "model that does not contain it counts as fallback. "
                         "Unset = only explicit fallback_used flags count.")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the manifest and print the run plan; no network")
    ap.add_argument("--list-checks", action="store_true",
                    help="print implemented structure checks")
    ap.add_argument("--report", action="store_true",
                    help="regenerate FEATURE_MATRIX.md from the results jsonl")
    ap.add_argument("--auto-seed", action="store_true",
                    help="run scripts/seed_fixtures.py for missing self-contained "
                         "fixtures before the sweep")
    args = ap.parse_args()

    if args.list_checks:
        return list_checks()
    if args.report:
        generate_report()
        return 0

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        sys.exit(str(exc))

    # Dry-run validates the manifest/plan with NO network calls — so it must
    # not demand credentials either (fixture-project resolution is a live
    # lookup and is skipped; unresolved fixtures show under their canonical
    # names). Same contract as golden_set_gate --dry-run.
    if args.dry_run:
        return dry_run(args, manifest)

    # Resolve canonical fixture names to live project ids before any planning.
    headers = _auth_header(args.base)
    _resolve_fixture_projects(manifest, args.base, headers)

    unresolved = [
        (f["action"], f.get("project"))
        for f in manifest["features"]
        if f.get("blocked") and isinstance(f.get("project"), str) and f["project"].startswith("FIXTURE —")
    ]
    if unresolved:
        print(f"[pre-sweep] {len(unresolved)} fixture-bound features unresolved:")
        for action, name in unresolved:
            print(f"  - {action}: {name}")
        if args.auto_seed:
            seeder = ROOT / "scripts" / "seed_fixtures.py"
            print(f"[pre-sweep] auto-seeding via {seeder} ...")
            subprocess.run(
                [sys.executable, str(seeder), "--base", args.base],
                check=False,
            )
            # Re-resolve after seeding.
            _resolve_fixture_projects(manifest, args.base, headers)
            still_unresolved = [
                (f["action"], f.get("project"))
                for f in manifest["features"]
                if f.get("blocked") and isinstance(f.get("project"), str) and f["project"].startswith("FIXTURE —")
            ]
            if still_unresolved:
                print(f"[pre-sweep] {len(still_unresolved)} fixture-bound features still blocked "
                      f"(need FIXTURES_DIR files): {', '.join(a for a, _ in still_unresolved)}")
        else:
            print("[pre-sweep] run with --auto-seed to create self-contained fixtures, "
                  "or set FIXTURES_DIR and run scripts/seed_fixtures.py for dir-based fixtures.")

    return run_sweep(args, manifest)


if __name__ == "__main__":
    sys.exit(main())
