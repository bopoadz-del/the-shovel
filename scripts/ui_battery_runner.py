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
#: The battery's own target (ORDERS_02NOV_1508Z/1545Z): a blank PROJECT_NAME
#: resolves here, and the seed step creates + indexes it from the committed
#: synthetic fixtures, so the battery needs no owner setup and never grades
#: the general corpus.
DEFAULT_BATTERY_PROJECT = "battery-t1"
SEED_FIXTURES_DIR = "tests/battery/fixtures"
SEED_POLL_ATTEMPTS = 90
SEED_POLL_SLEEP_S = 2.0

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


# ── S2: question-intent layer ───────────────────────────────────────────────
# The route is driven by what was ASKED first, with sources/tools as
# confirmation only. Q2 and Q17 produce the same answer shape (a grounded
# contract figure plus an offer to compute) but differ only in the question:
# Q2 asks what the contract states (a lookup, fully answerable), Q17 says
# "calculate" (a computation whose operand is missing). No rule over sources or
# tools can separate them; the question can. Structural only -- no per-question
# strings (ORDERS_03NOV_0152Z / 0345Z / 0420Z).
INTENT_LOOKUP = "lookup"
INTENT_COMPUTE = "compute"
INTENT_DOC_OP = "doc_op"
INTENT_UNSPECIFIED = "unspecified"

# A document OPERATION: an imperative verb over a document/deliverable artefact
# ("summarise the specification", "build the WBS").
_DOC_OP_RE = re.compile(
    r"\b(?:summari[sz]e|build|generate|create|draft|prepare|export|compile)\b"
    r".{0,40}\b(?:document|spec(?:ification)?|report|wbs|work[\s-]?breakdown|"
    r"schedule|programme|program|gantt|drawing)\b",
    re.IGNORECASE | re.DOTALL,
)
# A COMPUTATION: an explicit calculate verb, a digit-flanked arithmetic
# operator, a "per ... at ... rate/amount/times" formula, or a quantity a
# calculator produces from operands (load capacity, carbon footprint, a priced
# take-off, a payment certificate). "lux", a bare "rate", or a "requirement"
# are NOT arithmetic -- those stay lookups.
_COMPUTE_RE = re.compile(
    r"\b(?:calculate|compute|work\s+out)\b"
    r"|\d\s*[x*×/]\s*\d"
    r"|\bper\s+\w+\b.{0,60}\b(?:times|multiplied|rate|amount)\b"
    r"|\btimes\b.{0,30}\b(?:amount|rate|total|sum)\b"
    r"|\b(?:load\s+capacity|carbon\s+footprint|footprint|priced\s+take[\s-]?off|take[\s-]?off)\b"
    r"|\bissue\s+a\b.{0,30}\b(?:certificate|valuation)\b"
    r"|\bpayment\s+certificate\b",
    re.IGNORECASE | re.DOTALL,
)
# A LOOKUP: an interrogative asking what a value IS (from a document/standard).
_LOOKUP_RE = re.compile(
    r"^\s*(?:what|who|which|when|where|how\s+many|does|is|are|list|name)\b",
    re.IGNORECASE,
)


def classify_intent(message: str) -> str:
    """The intent CLASS of the question, by structure. Document-operation beats
    computation beats lookup; anything else is unspecified (the existing
    source/tool precedence then decides)."""
    raw = message or ""
    if _DOC_OP_RE.search(raw):
        return INTENT_DOC_OP
    if _COMPUTE_RE.search(raw):
        return INTENT_COMPUTE
    if _LOOKUP_RE.search(raw) or raw.strip().endswith("?"):
        return INTENT_LOOKUP
    return INTENT_UNSPECIFIED


def _arithmetic_was_performed(turn: dict[str, Any]) -> bool:
    """A calculator actually produced a figure this turn (a credited calculator
    provenance entry, or the construction_calc tool ran)."""
    if _CALC_TOOL in (turn.get("tools") or []):
        return True
    return any(
        isinstance(p, dict) and p.get("source") == "calculator"
        for p in turn.get("provenance") or []
    )


def classify_route(turn: dict[str, Any], question: str = "") -> str:
    """The route the turn actually took. The QUESTION'S INTENT decides first;
    sources and tools confirm. For an UNSPECIFIED question the old
    source/tool precedence is used unchanged.

    ``question`` is the full operator question, not a truncated preview -- a
    classifier reading a preview would be a silent defect.
    """
    tools = turn.get("tools") or []
    sources = turn.get("sources") or []
    classes = _source_classes(sources)

    # R8 (ORDERS_03NOV_0455Z): intent RESOLVES a conflict among the signals the
    # product actually produced; it never SUPPLIES a route the product's own
    # signals do not support. Otherwise route_observed becomes a function of the
    # question, the comparison against route_expected is a tautology, and the
    # score measures nothing (it would turn ~5 rows green with no product change,
    # including a turn whose calculator fabricated an operand). Each intent
    # branch therefore only resolves in favour of a signal that is present; if it
    # is not, the branch falls through to the real precedence below.
    intent = classify_intent(question)
    if intent == INTENT_DOC_OP:
        # only if a document operation actually happened
        if any(t in _DOC_TOOLS for t in tools):
            return ROUTE_TOOL_ON_DOC
        # else: fall through and report what it really did
    if intent == INTENT_COMPUTE:
        if _arithmetic_was_performed(turn):
            return ROUTE_CALCULATOR
        if _asks_for_inputs(turn):  # the product must have ACTUALLY asked
            return ROUTE_ASKS_INPUTS
        # else: fall through
    if intent == INTENT_LOOKUP:
        # A lookup answered from the project stands on the project, even if the
        # answer also offers to compute something (Q2); a doc tool having run
        # does not make a lookup a document operation (Q1). But a lookup with no
        # retrieved CLASS (an empty-string "class" is not a source) is not a
        # general-RAG answer -- it falls through to the real precedence.
        if classes & _PROJECT_CLASSES:
            return ROUTE_PROJECT_RAG
        if classes & _GENERAL_CLASSES:  # NOT `if sources` -- [""] is not a class
            return ROUTE_GENERAL_RAG
        # else: fall through

    # ── UNSPECIFIED / fell-through: the pre-intent precedence, untouched ──
    if _asks_for_inputs(turn):
        return ROUTE_ASKS_INPUTS
    if _CALC_TOOL in tools:
        return ROUTE_CALCULATOR
    if any(t in _DOC_TOOLS for t in tools):
        return ROUTE_TOOL_ON_DOC
    if classes & _PROJECT_CLASSES:
        return ROUTE_PROJECT_RAG
    if classes & _GENERAL_CLASSES:
        return ROUTE_GENERAL_RAG
    if not sources and not (turn.get("provenance") or []):
        return ROUTE_UNCLASSIFIED
    return ROUTE_GENERAL_RAG


# ── R6: grounding gate (ORDERS_03NOV_0310Z / 0420Z) ──────────────────────────
# A correct route is necessary but not sufficient: the answer must also be
# GROUNDED in a source of the kind the route claims, or the row fails. This is
# measurement only -- no product code is involved.
_PROV_PROJECT = "project_document"
_PROV_GENERAL = "general_knowledge"
_PROV_CALC = "calculator"
_PROV_USER = "user_input"


def _grounded(expected: str, provenance: list[dict[str, Any]],
              sources: list[str]) -> bool:
    """True when the answer is grounded in the kind of source the expected route
    claims. asks_for_inputs may carry only the user's own figures (or none);
    tool_on_document is the one class evidenced by the sources panel."""
    kinds = {p.get("source") for p in (provenance or []) if isinstance(p, dict)}
    if expected == ROUTE_PROJECT_RAG:
        return _PROV_PROJECT in kinds
    if expected == ROUTE_GENERAL_RAG:
        return _PROV_GENERAL in kinds
    if expected == ROUTE_CALCULATOR:
        return _PROV_CALC in kinds
    if expected == ROUTE_ASKS_INPUTS:
        return kinds <= {_PROV_USER}
    if expected == ROUTE_TOOL_ON_DOC:
        return _PROV_PROJECT in kinds or "project_corpus" in (sources or [])
    return True  # no other class appears in t1.json


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
    # Auto-pick (no name match): never silently land on the GENERAL/master
    # corpus. A blank PROJECT_NAME must resolve to a real project holding the
    # owner's documents -- otherwise the project_rag questions fail by
    # construction and the score is meaningless (observed 2026-11-02: the
    # general corpus had the most indexed docs, so it won the pick and the
    # battery read 4/20 against master_corpus). Exclude the general corpora
    # (an explicit project_name above still honours a deliberate choice); if
    # none remain, return None so resolve_project fails loudly.
    real = [p for p in rows if not _is_general_corpus(p)]
    if not real:
        return None
    return max(real, key=_indexed_count)


def _is_general_corpus(project: dict[str, Any]) -> bool:
    """True when this project row is a general/master corpus (not a real
    document project), by id or name against :data:`_GENERAL_CLASSES`."""
    ident = str(project.get("id") or project.get("project_id") or "").lower()
    name = str(project.get("name") or "").lower()
    return ident in _GENERAL_CLASSES or name in _GENERAL_CLASSES


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
        raise RuntimeError(
            "no real project to run the battery against: only the general/master "
            "corpus is visible to this login. Set PROJECT_NAME to the owner's "
            "project, or index the owner's documents into this account, so the "
            "battery does not silently grade the general corpus.")
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
            observed = classify_route(turn, text)
            src_classes = sorted(_source_classes(turn["sources"]))
            prov = _provenance_digest(turn["provenance"])
            route_pass = observed == expected
            # Grounding reads the RAW provenance (its "source" kinds), not the
            # flattened digest.
            grounded = _grounded(expected, turn["provenance"], src_classes)
            results.append({
                "question": text, "route_expected": expected, "route_observed": observed,
                "answer": turn["answer"], "provenance": prov,
                "sources": src_classes,
                "route_pass": route_pass,
                "grounded": grounded,
                "pass": route_pass and grounded,
            })
        except Exception as exc:  # noqa: BLE001 — one bad turn must not abort the battery
            results.append({"question": text, "route_expected": expected,
                            "route_observed": None, "error": f"{type(exc).__name__}: {exc}",
                            "route_pass": False, "grounded": False, "pass": False})
    return results


def battery_exit_code(passed: int, total: int) -> int:
    """0 only when EVERY question passed (and at least one ran); else 1. A
    battery that scores < total is a FAILED acceptance run -- the workflow must
    go red, never finish green at a partial score (ORDERS_02NOV_1508Z finding A).
    An empty battery (total == 0) is a setup error, so also non-zero."""
    return 0 if total > 0 and passed == total else 1


def main() -> int:
    import httpx  # lazy: only the live run needs it

    battery_file = os.environ.get("BATTERY_FILE")
    result_file = os.environ.get("RESULT_FILE", "battery_result.json")
    # A blank PROJECT_NAME targets the battery's own seeded project, never
    # an implicit "whatever has the most documents" (ORDERS_02NOV_1545Z).
    project_name = os.environ.get("PROJECT_NAME") or DEFAULT_BATTERY_PROJECT
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
    total = len(results)
    route_passed = sum(1 for r in results if r.get("route_pass"))
    grounded = sum(1 for r in results if r.get("grounded"))
    passed = sum(1 for r in results if r.get("pass"))
    for r in results:
        verdict = "PASS" if r.get("pass") else "FAIL"
        marks = f"route={'Y' if r.get('route_pass') else 'n'} grounded={'Y' if r.get('grounded') else 'n'}"
        print(f"[{verdict}] {r.get('route_expected')} -> {r.get('route_observed')}  "
              f"[{marks}]  {r.get('question')}")
    # R6: the headline is BOTH numbers; the gate keys on `pass` (route AND grounded).
    print(f"routes {route_passed}/{total}; grounded {grounded}/{total}; pass {passed}/{total}; "
          f"full per-question result in {result_file}")
    code = battery_exit_code(passed, total)
    if code:
        print(f"::error::battery acceptance FAILED: pass {passed}/{total} "
              f"(routes {route_passed}/{total}, grounded {grounded}/{total}) -- red until 20/20")
    return code


# ── seed: the battery provisions its own target (ORDERS_02NOV_1545Z) ─────────

def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _existing_doc_names(client: Any, base: str, pid: str, cred: str) -> set[str]:
    """Names already attached to ``pid``, so a re-run uploads only the rest.

    Best-effort: a listing failure must not wedge the seed, so it falls back to
    "upload everything" (and logs why) rather than silently skipping files.
    """
    try:
        resp = client.get(
            f"{base}/v1/projects/{pid}/documents?limit=200",
            headers=_bearer(cred), timeout=30,
        )
        resp.raise_for_status()
        docs = resp.json().get("documents", []) or []
    except Exception as exc:  # noqa: BLE001 — logged, not swallowed; safe fallback
        print(f"[seed] could not list existing documents ({exc}); uploading all")
        return set()
    names: set[str] = set()
    for d in docs:
        if isinstance(d, dict):
            name = d.get("original_name") or d.get("name") or d.get("filename")
            if name:
                names.add(str(name))
    return names


def _seed_decision(projects: list[dict[str, Any]] | None,
                   name: str) -> tuple[str, dict[str, Any] | None]:
    """('found', row) when a project with exactly ``name`` exists, else
    ('create', None). Pure, so idempotence (find — never duplicate) is
    testable without a server."""
    rows = [p for p in (projects or []) if isinstance(p, dict)]
    named = [p for p in rows if str(p.get("name") or "") == name]
    return ("found", named[0]) if named else ("create", None)


def _fixtures_needed(row: dict[str, Any] | None) -> bool:
    """True when the target project holds zero indexed documents — only then
    does the seed upload the committed fixtures (idempotent on re-run)."""
    return _indexed_count(row or {}) == 0


def _wait_until_indexed(fetch_rows, name: str, attempts: int, sleep) -> dict[str, Any]:
    """Poll the projects listing until ``name`` shows at least one indexed
    document; after ``attempts`` tries raise a CLEAR error (never a silent
    pass against an empty project). ``fetch_rows``/``sleep`` are injectable."""
    for _ in range(attempts):
        _, row = _seed_decision(fetch_rows(), name)
        if row is not None and _indexed_count(row) > 0:
            return row
        sleep()
    raise RuntimeError(
        f"battery seed: project {name!r} did not reach an indexed state in time. "
        "The fixtures uploaded but never indexed — only an ADMIN's upload is "
        "indexed into project RAG, so check the battery account's role and the "
        "service logs."
    )


_SEED_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _fixture_paths(fixtures_dir: "pathlib.Path") -> list["pathlib.Path"]:
    """The document fixtures to seed, sorted — ONLY files whose extension is a
    known document type (``_SEED_CONTENT_TYPES``).

    The fixtures directory also holds repo-control files such as ``.gitattributes``
    (which marks the binaries). Those are not documents: ``.gitattributes`` has an
    empty extension, and the project-documents endpoint correctly rejects an
    empty/unknown type with 415 "not ingestible". Filtering by known document
    extension skips any such control file by construction, not by name.
    """
    if not fixtures_dir.is_dir():
        return []
    return sorted(
        p for p in fixtures_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SEED_CONTENT_TYPES
    )


def seed_main() -> int:
    """Provision the battery's target before a run: promote the battery
    account to admin via the role-management endpoint (master-key principal —
    the value is never printed), find-or-create the target project, upload the
    committed fixtures when it holds nothing indexed, and poll to indexed.
    Idempotent on every path."""
    import pathlib

    import httpx  # lazy: only the live seed needs it

    master = os.environ.get("CEREBRUM_MASTER_KEY") or ""
    if not master:
        print("::error::Missing CEREBRUM_MASTER_KEY repository secret — the "
              "admin principal the battery seed step promotes and uploads with.")
        return 2
    name = os.environ.get("PROJECT_NAME") or DEFAULT_BATTERY_PROJECT
    fixtures_dir = pathlib.Path(os.environ.get("BATTERY_FIXTURES_DIR") or SEED_FIXTURES_DIR)
    cred, base = _auth_base()
    with httpx.Client() as client:
        me = client.get(f"{base}/v1/users/me", headers=_bearer(cred), timeout=30)
        me.raise_for_status()
        who = me.json()
        if who.get("role") != "admin":
            # require_user reads the role from the DB per request, so the same
            # credential carries admin immediately after the promotion.
            resp = client.post(
                f"{base}/v1/admin/users/{who['user_id']}/role",
                json={"role": "admin"}, headers=_bearer(master), timeout=30,
            )
            resp.raise_for_status()
            print(f"[seed] battery account promoted to admin (changed={resp.json().get('changed')})")
        else:
            print("[seed] battery account already admin")

        def fetch_rows() -> list[dict[str, Any]]:
            resp = client.get(f"{base}/v1/projects", headers=_bearer(cred), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("projects") if isinstance(data, dict) else data
            return rows if isinstance(rows, list) else []

        decision, row = _seed_decision(fetch_rows(), name)
        if decision == "create":
            resp = client.post(f"{base}/v1/projects", json={"name": name},
                               headers=_bearer(cred), timeout=30)
            resp.raise_for_status()
            print(f"[seed] created project {name!r}")
            _, row = _seed_decision(fetch_rows(), name)
        else:
            print(f"[seed] found project {name!r}")

        if _fixtures_needed(row):
            pid = str((row or {}).get("id") or (row or {}).get("project_id") or "")
            files = _fixture_paths(fixtures_dir)
            if not files:
                print(f"::error::battery seed: no fixtures found in {fixtures_dir}")
                return 2
            # Converge on re-run: upload only the fixtures the project does not
            # already have. A run that failed partway (e.g. file 1 uploaded,
            # file 2 rejected) left battery-t1 half-made; without this the retry
            # re-sends file 1 (duplicate) instead of finishing file 2.
            existing = _existing_doc_names(client, base, pid, cred)
            for path in files:
                if path.name in existing:
                    print(f"[seed] already present, skipping {path.name}")
                    continue
                ctype = _SEED_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
                with open(path, "rb") as fh:
                    resp = client.post(
                        f"{base}/v1/projects/{pid}/documents",
                        files={"file": (path.name, fh, ctype)},
                        headers=_bearer(cred), timeout=120,
                    )
                if resp.status_code >= 400:
                    # raise_for_status prints only the status line; the server's
                    # body carries WHICH guard rejected the upload (e.g. a 415's
                    # "not ingestible" vs "compressed folder" detail). Surface it
                    # so a rejection is diagnosable from the run log, not opaque.
                    print(
                        f"::error::battery seed: POST documents {path.name} "
                        f"(sent {ctype}) -> {resp.status_code}: {resp.text[:400]}"
                    )
                    resp.raise_for_status()
                print(f"[seed] uploaded {path.name}")
            _wait_until_indexed(fetch_rows, name, SEED_POLL_ATTEMPTS,
                                lambda: time.sleep(SEED_POLL_SLEEP_S))
            print("[seed] fixtures indexed")
        else:
            print("[seed] project already indexed; nothing to upload")
    return 0


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(seed_main() if _sys.argv[1:2] == ["seed"] else main())
