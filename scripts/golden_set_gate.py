#!/usr/bin/env python
"""Golden-set pilot gate (TASK 4e).

Drives every query in tests/golden_set.yaml through the NORMAL chat path
(POST /v1/agents/{agent}/chat/stream — the same endpoint the browser and
fork_cli use, so nothing bypasses the smart_orchestrator routing gate) and
judges each answer mechanically:

  PASS iff
    - every regex in `answer_expect` matches the final answer text, and
    - no regex in `answer_reject` matches, and
    - when `retrieval_expect` is set, a cited source (the end event's
      `sources` list) has a doc_name or doc_id containing it
      (case-insensitive substring), and
    - the turn executed (HTTP 200, no error events, non-empty answer).

All regexes are compiled with re.IGNORECASE | re.MULTILINE.

Pilot bar: >= 90% PASS of the full golden set (28 queries -> >= 26/28).
The process exits 0 iff the bar is met — both after a sweep and in --report
mode — so this script IS the gate.

Auth (env only, same contract as feature_matrix_sweep / fork_cli — values
are never printed or stored): FORK_TOKEN or FORK_API_KEY, or
FORK_EMAIL+FORK_PASSWORD (logs in via /v1/users/login). Target from
FORK_BASE_URL (alias FORK_BASE) (default prod).

Named fixtures (``FIXTURE — BOQ`` and the other ``FIXTURE — *`` projects)
are resolved by display name. A missing named fixture is self-seeded via
``scripts/seed_fixtures.py --synthetic``; the unresolved display name is
never sent as ``project_id``. Seed failure is fatal. ``master_corpus``
is not self-seeded.

Usage:
  python scripts/golden_set_gate.py --dry-run          # validate, no network
  python scripts/golden_set_gate.py                    # full gate run
  python scripts/golden_set_gate.py --only fresh_eot_notice_period
  python scripts/golden_set_gate.py --start 12         # resume a killed run
  python scripts/golden_set_gate.py --report           # regen report + gate

Artifacts (all incremental / kill-safe):
  golden_set_results.jsonl   one line per query, appended as it happens
  GOLDEN_SET_REPORT.md       regenerated from the jsonl after a sweep or by
                             --report; every failure itemized with the
                             verbatim answer excerpt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_SET = ROOT / "tests" / "golden_set.yaml"
DEFAULT_BASE = (
    os.getenv("FORK_BASE_URL")
    or os.getenv("FORK_BASE")
    or "https://the-fork-jn3t.onrender.com"
)
RESULTS_JSONL = ROOT / "golden_set_results.jsonl"
REPORT_MD = ROOT / "GOLDEN_SET_REPORT.md"
MASTER_CORPUS_ALIAS = "master_corpus"

PASS_BAR_PCT = 90  # the pilot gate: >= 90% of the golden set must PASS

# Same retry contract as feature_matrix_sweep: 5xx / 429 / connection errors
# get ONE retry after a long pause (heavy LLM turns on a rate-limited tier);
# 4xx is deterministic — no retry.
RETRY_WAIT_S = 60

REGEX_FLAGS = re.IGNORECASE | re.MULTILINE

# Keep the stored answer bounded so the jsonl stays greppable; excerpts in
# the report come from this field.
MAX_ANSWER_STORED = 6000


class GoldenSetError(ValueError):
    """Golden set fails validation (missing prompt, bad regex, dup id, ...)."""


class FixtureUnresolvedError(GoldenSetError):
    """A named golden fixture is missing and could not be seeded."""


# ── golden set ───────────────────────────────────────────────────────────────

def load_golden_set(path: Path | str = DEFAULT_GOLDEN_SET) -> dict:
    """Parse + validate the golden set. Raises GoldenSetError on any defect
    so a typo in a regex dies at load time, not mid-run against prod."""
    if yaml is None:
        raise GoldenSetError("PyYAML is not installed (pip install pyyaml)")
    path = Path(path)
    if not path.exists():
        raise GoldenSetError(f"golden set not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "queries" not in raw:
        raise GoldenSetError("golden set must be a mapping with a `queries` list")

    defaults = raw.get("defaults") or {}
    problems: list = []
    queries: list = []
    seen_ids: set = set()

    def _compile_all(patterns, where: str) -> list:
        out = []
        for p in patterns:
            if not isinstance(p, str) or not p.strip():
                problems.append(f"{where}: empty or non-string pattern")
                continue
            try:
                re.compile(p, REGEX_FLAGS)
            except re.error as exc:
                problems.append(f"{where}: bad regex {p!r} ({exc})")
                continue
            out.append(p)
        return out

    for i, q in enumerate(raw["queries"] or []):
        qid = q.get("id")
        where = f"queries[{i}]" + (f" ({qid})" if qid else "")
        if not qid:
            problems.append(f"{where}: missing `id`")
            continue
        if qid in seen_ids:
            problems.append(f"{where}: duplicate id")
        seen_ids.add(qid)
        if not (q.get("prompt") or "").strip():
            problems.append(f"{where}: missing/empty `prompt`")
        if not q.get("feature"):
            problems.append(f"{where}: missing `feature`")
        if not q.get("project"):
            problems.append(f"{where}: missing `project`")
        expect = q.get("answer_expect")
        if not expect or not isinstance(expect, list):
            problems.append(f"{where}: `answer_expect` must be a non-empty list")
            expect = []
        reject = q.get("answer_reject") or []
        if not isinstance(reject, list):
            problems.append(f"{where}: `answer_reject` must be a list")
            reject = []
        queries.append({
            "id": qid,
            "feature": q.get("feature", ""),
            "project": q.get("project", ""),
            "agent": q.get("agent", defaults.get("agent", "project-assistant")),
            "prompt": (q.get("prompt") or "").strip(),
            "retrieval_expect": (q.get("retrieval_expect") or "").strip(),
            "answer_expect": _compile_all(expect, f"{where}.answer_expect"),
            "answer_reject": _compile_all(reject, f"{where}.answer_reject"),
        })

    if problems:
        raise GoldenSetError("golden set validation failed:\n  - "
                             + "\n  - ".join(problems))
    if not queries:
        raise GoldenSetError("golden set has no queries")
    return {"defaults": defaults, "queries": queries}


def pass_bar(n_queries: int) -> int:
    """Minimum PASS count for the gate: >= 90% of the full set."""
    return math.ceil(n_queries * PASS_BAR_PCT / 100.0)


# ── auth / transport (same env contract as feature_matrix_sweep) ─────────────

def _auth_header(base: str) -> dict:
    """Env-only credentials; the value is used in the header, never echoed."""
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
    """One chat turn over SSE. Mirrors feature_matrix_sweep's client but KEEPS
    the end event's sources list — the retrieval oracle needs doc names."""
    url = f"{base}/v1/agents/{agent}/chat/stream"
    body: dict = {"message": message, "conversation_id": conversation_id}
    if project_id:
        body["project_id"] = project_id

    out = {
        "http_status": None, "http_error": None,
        "route": None, "tool_calls": [], "answer": "", "errors": [],
        "sources": [], "served_model": None, "fallback_used": False,
        "first_token_s": None, "total_s": None, "n_events": 0,
    }
    t0 = time.monotonic()
    answer_parts: list = []

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
                out["tool_calls"].append(
                    evt.get("tool") or evt.get("name") or "?")
            elif etype == "sources":
                out["sources"] = _normalize_sources(
                    evt.get("sources") or evt.get("items") or [])
            elif etype == "error":
                out["errors"].append(str(evt.get("message", ""))[:500])
                break
            elif etype == "end":
                out["served_model"] = evt.get("model") or out["served_model"]
                srcs = _normalize_sources(evt.get("sources") or [])
                if srcs:
                    out["sources"] = srcs
                break
            # start / heartbeat / tool_result: nothing the oracles need

    out["answer"] = "".join(answer_parts)
    out["total_s"] = round(time.monotonic() - t0, 2)
    return out


def _normalize_sources(raw: list) -> list:
    """Keep just what the retrieval oracle and the report need."""
    out = []
    for s in raw or []:
        if not isinstance(s, dict):
            continue
        out.append({"doc_id": str(s.get("doc_id") or ""),
                    "doc_name": str(s.get("doc_name") or ""),
                    "score": s.get("score")})
    return out


def execute_with_retry(client, base, agent, headers, message, project_id,
                       conversation_id) -> dict:
    """5xx / 429 / connection errors get one retry after RETRY_WAIT_S; the
    second failure is recorded as FAIL evidence, not raised."""
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
                    "route": None, "tool_calls": [], "answer": "",
                    "errors": [], "sources": [], "served_model": None,
                    "fallback_used": False, "first_token_s": None,
                    "total_s": None, "n_events": 0}
            continue
        status = res.get("http_status")
        if status is not None and (status >= 500 or status == 429):
            last = res
            continue
        return res
    return last  # type: ignore[return-value]


# ── oracle ───────────────────────────────────────────────────────────────────

def judge(entry: dict, res: dict) -> dict:
    """Mechanical verdict for one query. Returns a dict with `verdict`
    (PASS/FAIL) and itemized evidence for the report."""
    answer = res.get("answer") or ""
    reasons: list = []

    if res.get("http_error"):
        reasons.append(f"execution: {res['http_error']}")
    for err in res.get("errors") or []:
        reasons.append(f"execution: error event: {err}")
    if not answer.strip():
        reasons.append("execution: empty answer")

    expect_misses = [p for p in entry["answer_expect"]
                     if not re.search(p, answer, REGEX_FLAGS)]
    for p in expect_misses:
        reasons.append(f"answer_expect not matched: {p!r}")

    reject_hits = [p for p in entry["answer_reject"]
                   if re.search(p, answer, REGEX_FLAGS)]
    for p in reject_hits:
        m = re.search(p, answer, REGEX_FLAGS)
        reasons.append(f"answer_reject matched: {p!r} -> {m.group(0)!r}")

    retrieval_ok = None
    if entry["retrieval_expect"]:
        needle = entry["retrieval_expect"].lower()
        retrieval_ok = any(
            needle in (s.get("doc_name") or "").lower()
            or needle in (s.get("doc_id") or "").lower()
            for s in res.get("sources") or [])
        if not retrieval_ok:
            cited = ", ".join(
                (s.get("doc_name") or s.get("doc_id") or "?")
                for s in res.get("sources") or []) or "(no sources cited)"
            reasons.append(
                f"retrieval_expect {entry['retrieval_expect']!r} not in "
                f"cited sources: {cited}")

    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "expect_misses": expect_misses,
        "reject_hits": reject_hits,
        "retrieval_ok": retrieval_ok,
    }


# ── artifacts ────────────────────────────────────────────────────────────────

def append_result(record: dict) -> None:
    # One JSON line per query, flushed immediately — a killed run keeps every
    # finished row and --report / --start work off whatever landed.
    with open(RESULTS_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()


def load_results() -> dict:
    """Last-wins per query id, so re-runs of single queries update the gate."""
    rows: dict = {}
    if not RESULTS_JSONL.exists():
        return rows
    with open(RESULTS_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[rec.get("id")] = rec
    return rows


def _excerpt(text: str, limit: int = 800) -> str:
    text = (text or "").strip()
    if not text:
        return "(no answer text)"
    return text[:limit] + ("\n[... truncated ...]" if len(text) > limit else "")


def generate_report(golden: dict) -> int:
    """Regenerate GOLDEN_SET_REPORT.md from the jsonl. Returns the process
    exit code: 0 iff the >=90% bar is met over the FULL golden set (queries
    without a result count as not passed)."""
    rows = load_results()
    queries = golden["queries"]
    n = len(queries)
    bar = pass_bar(n)

    passed = [q["id"] for q in queries
              if rows.get(q["id"], {}).get("verdict") == "PASS"]
    failed = [q for q in queries
              if rows.get(q["id"]) and rows[q["id"]]["verdict"] != "PASS"]
    missing = [q["id"] for q in queries if q["id"] not in rows]
    met = len(passed) >= bar

    lines = [
        "# GOLDEN_SET_REPORT — TASK 4e pilot gate",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"from {RESULTS_JSONL.name}.",
        "",
        f"## Score: {len(passed)}/{n} PASS",
        "",
        f"Gate bar: >= {PASS_BAR_PCT}% of the golden set must PASS "
        f"(>= {bar}/{n}). **Gate {'MET' if met else 'NOT MET'}.**",
        "",
    ]
    if missing:
        lines += [f"Queries with no recorded result (count as not passed): "
                  f"{', '.join(missing)}", ""]

    lines += [
        "| id | feature | project | verdict | retrieval | latency |",
        "|---|---|---|---|---|---|",
    ]
    for q in queries:
        rec = rows.get(q["id"])
        verdict = rec["verdict"] if rec else "NO RESULT"
        retr = "-"
        if rec and rec.get("retrieval_ok") is not None:
            retr = "hit" if rec["retrieval_ok"] else "MISS"
        elif q["retrieval_expect"]:
            retr = "?"
        latency = f"{rec['total_s']:.0f}s" if rec and rec.get("total_s") else "-"
        lines.append(f"| {q['id']} | {q['feature']} | {q['project']} "
                     f"| {verdict} | {retr} | {latency} |")
    lines.append("")

    if failed or missing:
        lines += ["## Failures", ""]
        for q in failed:
            rec = rows[q["id"]]
            lines += [
                f"### {q['id']} ({q['feature']})",
                "",
                f"- prompt: {q['prompt']}",
                f"- expected (all must match): "
                + "; ".join(f"`{p}`" for p in q["answer_expect"]),
            ]
            if q["answer_reject"]:
                lines.append("- rejected (none may match): "
                             + "; ".join(f"`{p}`" for p in q["answer_reject"]))
            if q["retrieval_expect"]:
                lines.append(f"- retrieval_expect: `{q['retrieval_expect']}`")
            lines.append("- what failed:")
            for reason in rec.get("reasons") or ["(no reasons recorded)"]:
                lines.append(f"  - {reason}")
            srcs = rec.get("sources") or []
            if srcs:
                lines.append("- cited sources: "
                             + "; ".join((s.get("doc_name") or s.get("doc_id") or "?")
                                         for s in srcs))
            lines += ["", "What came back (verbatim excerpt):", "", "```",
                      _excerpt(rec.get("answer", "")), "```", ""]
        for qid in missing:
            q = next(x for x in queries if x["id"] == qid)
            lines += [f"### {qid} ({q['feature']})", "",
                      f"- prompt: {q['prompt']}",
                      "- no result recorded (query never ran or the run was "
                      "killed before it finished)", ""]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_MD.name}: {len(passed)}/{n} PASS "
          f"(bar >= {bar}/{n}) -> gate {'MET' if met else 'NOT MET'}")
    return 0 if met else 1


# ── named-fixture resolve / self-seed ────────────────────────────────────────

def collect_named_fixtures(queries: list) -> list:
    """Canonical project names that are not the master_corpus alias."""
    names: list = []
    seen: set = set()
    for q in queries:
        value = (q.get("project") or "").strip()
        if not value or value == MASTER_CORPUS_ALIAS:
            continue
        if value in seen:
            continue
        seen.add(value)
        names.append(value)
    return names


def lookup_live_project_id(client, value: str, *, base: str,
                           headers: dict | None = None) -> str | None:
    """Return the live project id for a canonical name or id, or None."""
    r = client.get(f"{base}/v1/projects", headers=headers, timeout=30)
    r.raise_for_status()
    for p in (r.json().get("projects", []) or []):
        if p.get("name") == value or p.get("id") == value:
            pid = p.get("id")
            if pid:
                return str(pid)
    return None


def _default_named_fixture_seeder(name: str, *, base: str, headers: dict,
                                  **_kw):
    from scripts.seed_fixtures import ensure_named_fixture
    return ensure_named_fixture(
        name, base=base, headers=headers, synthetic=True,
    )


def resolve_golden_project(
    client,
    value: str,
    *,
    base: str,
    headers: dict | None = None,
    cache: dict | None = None,
    seed_missing: bool = True,
    lookup=None,
    seeder=None,
) -> str:
    """Resolve a golden ``project`` field to a live project id.

    ``master_corpus`` passes through. Named fixtures are looked up by
    display name. If a seedable fixture is missing, the seed path runs
    (``seed_fixtures --synthetic``) instead of sending the unresolved
    display name as ``project_id`` (that 404s). Seed failure is fatal.
    """
    if not value or value == MASTER_CORPUS_ALIAS:
        return value
    cache = cache if cache is not None else {}
    cached = cache.get(value)
    if cached and cached != value:
        return cached
    if cached == value:
        # A prior bug cached the display name as the id — refuse it.
        cache.pop(value, None)

    lookup_fn = lookup or lookup_live_project_id
    found = None
    try:
        found = lookup_fn(client, value, base=base, headers=headers or {})
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        print(
            f"[fixture] project lookup for {value!r} failed ({exc}); "
            f"will try seed",
            file=sys.stderr,
        )
        found = None

    if found:
        from scripts.seed_fixtures import fixture_key_for_name
        if found == value and fixture_key_for_name(value):
            found = None
        else:
            cache[value] = found
            return found

    if seed_missing:
        from scripts.seed_fixtures import fixture_key_for_name
        if fixture_key_for_name(value):
            print(
                f"[fixture] {value!r} missing — seeding via "
                f"scripts/seed_fixtures.py --synthetic",
                file=sys.stderr,
            )
            seeder_fn = seeder or _default_named_fixture_seeder
            try:
                result = seeder_fn(
                    value, base=base, headers=headers or {}, client=client,
                )
            except FixtureUnresolvedError:
                raise
            except Exception as exc:
                raise FixtureUnresolvedError(
                    f"seed of {value!r} failed: {exc}"
                ) from exc
            pid = (result or {}).get("project_id")
            if not pid or pid == value:
                raise FixtureUnresolvedError(
                    f"seed of {value!r} returned no live project_id"
                )
            confirmed = None
            try:
                confirmed = lookup_fn(
                    client, value, base=base, headers=headers or {},
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                confirmed = None
            resolved = confirmed or str(pid)
            if resolved == value:
                raise FixtureUnresolvedError(
                    f"seed of {value!r} still unresolved; refusing to use "
                    f"the display name as project_id"
                )
            cache[value] = resolved
            return resolved

    raise FixtureUnresolvedError(
        f"project {value!r} is not on the target and is not a "
        f"self-seedable fixture — refusing to send the display name "
        f"as project_id"
    )


def preflight_named_fixtures(
    client,
    queries: list,
    *,
    base: str,
    headers: dict | None = None,
    cache: dict | None = None,
    seed_missing: bool = True,
    lookup=None,
    seeder=None,
) -> dict:
    """Resolve/seed every non-master_corpus project before chat turns.

    Returns the name -> project_id cache. Raises FixtureUnresolvedError
    if any named fixture is missing and seed fails. A missing display
    name is never stored as a project_id.
    """
    cache = cache if cache is not None else {}
    for name in collect_named_fixtures(queries):
        resolve_golden_project(
            client, name,
            base=base, headers=headers, cache=cache,
            seed_missing=seed_missing, lookup=lookup, seeder=seeder,
        )
    return cache


# ── run ──────────────────────────────────────────────────────────────────────

def run_gate(args, golden: dict) -> int:
    queries = golden["queries"]
    plan = [(i, q) for i, q in enumerate(queries)]
    if args.only:
        plan = [(i, q) for i, q in plan if q["id"] == args.only]
        if not plan:
            sys.exit(f"no query with id {args.only!r}")
    plan = [(i, q) for i, q in plan if i >= args.start]
    if not plan:
        sys.exit("empty plan after filters")

    headers = _auth_header(args.base)
    print(f"golden-set gate: {len(plan)} queries  base={args.base}  "
          f"delay={args.delay}s  bar>={pass_bar(len(queries))}/{len(queries)}")
    print("=" * 76)

    _project_cache: dict = {}

    with httpx.Client() as client:
        try:
            preflight_named_fixtures(
                client, [q for _, q in plan],
                base=args.base, headers=headers, cache=_project_cache,
            )
        except FixtureUnresolvedError as exc:
            print(f"[fixture] FATAL: {exc}", file=sys.stderr)
            return 1

        for n, (idx, entry) in enumerate(plan):
            if n and args.delay:
                time.sleep(args.delay)
            # Fresh conversation per query: shared history is exactly the
            # contamination vector the gate must exclude.
            conversation_id = f"gsg-{uuid.uuid4().hex[:12]}"
            print(f"[{idx:>2}] {entry['id']}: {entry['prompt'][:60]}")

            resolved_pid = resolve_golden_project(
                client, entry["project"],
                base=args.base, headers=headers, cache=_project_cache,
            )
            res = execute_with_retry(client, args.base, entry["agent"],
                                     headers, entry["prompt"],
                                     resolved_pid, conversation_id)
            oracle = judge(entry, res)

            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "index": idx,
                "id": entry["id"],
                "feature": entry["feature"],
                "project": entry["project"],
                "prompt": entry["prompt"],
                "conversation_id": conversation_id,
                "verdict": oracle["verdict"],
                "reasons": oracle["reasons"],
                "expect_misses": oracle["expect_misses"],
                "reject_hits": oracle["reject_hits"],
                "retrieval_ok": oracle["retrieval_ok"],
                "route_action": (res.get("route") or {}).get("action"),
                "tool_calls": res.get("tool_calls"),
                "sources": res.get("sources"),
                "answer": (res.get("answer") or "")[:MAX_ANSWER_STORED],
                "answer_chars": len(res.get("answer") or ""),
                "served_model": res.get("served_model"),
                "fallback_used": bool(res.get("fallback_used")),
                "first_token_s": res.get("first_token_s"),
                "total_s": res.get("total_s"),
                "http_error": res.get("http_error"),
                "errors": res.get("errors"),
            }
            append_result(record)

            print(f"     {oracle['verdict']:4s} chars={record['answer_chars']}"
                  f" total={record['total_s']}s"
                  + (" FALLBACK" if record["fallback_used"] else ""))
            for reason in oracle["reasons"]:
                print(f"       - {reason}")
            # Kill-safe: the next index to resume from is always printed, so
            # a Ctrl-C'd run restarts with --start N without bookkeeping.
            print(f"     (resume point: --start {idx + 1})", file=sys.stderr)

    print("=" * 76)
    return generate_report(golden)


def dry_run(args, golden: dict) -> int:
    queries = golden["queries"]
    n = len(queries)
    features = sorted({q["feature"] for q in queries})
    n_retr = sum(1 for q in queries if q["retrieval_expect"])
    n_rej = sum(1 for q in queries if q["answer_reject"])
    print(f"golden set: {args.golden_set}")
    print(f"queries: {n}   features: {len(features)}   "
          f"with retrieval_expect: {n_retr}   with answer_reject: {n_rej}")
    print(f"gate bar: >= {PASS_BAR_PCT}% -> >= {pass_bar(n)}/{n} PASS")
    print("-" * 76)
    for i, q in enumerate(queries):
        marks = ("R" if q["retrieval_expect"] else "-") + \
                ("X" if q["answer_reject"] else "-")
        print(f"[{i:>2}] {marks} {q['feature']:28s} {q['id']:28s} "
              f"{q['prompt'][:50]}")
    print("-" * 76)
    print("dry-run OK - no network calls made")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="TASK 4e golden-set pilot gate vs the live chat path")
    ap.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--only", help="run a single query by id")
    ap.add_argument("--start", type=int, default=0,
                    help="first query index to run (resume point)")
    ap.add_argument("--delay", type=float, default=20,
                    help="seconds between queries — heavy LLM turns on a "
                         "rate-limited tier need room (default 20)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the golden set and print the plan; no network")
    ap.add_argument("--report", action="store_true",
                    help="regenerate GOLDEN_SET_REPORT.md from the results "
                         "jsonl and exit with the gate verdict")
    args = ap.parse_args()

    try:
        golden = load_golden_set(args.golden_set)
    except GoldenSetError as exc:
        sys.exit(str(exc))

    if args.dry_run:
        return dry_run(args, golden)
    if args.report:
        return generate_report(golden)
    return run_gate(args, golden)


if __name__ == "__main__":
    sys.exit(main())
