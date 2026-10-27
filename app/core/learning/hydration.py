"""Nightly hydration — the learning engine's "sleep on it" pass.

Lives under ``app/core/learning/`` because hydration IS learning: it reads
the day's conversations + files and writes back into the surfaces tomorrow's
chat actually consults. Exposed publicly as the ``hydrate`` operation on
:class:`app.blocks.learning_engine.LearningEngineBlock`; the routes under
``/v1/hydration/*`` and the nightly scheduler both call into this module
through that block.

What one pass does, per project that had activity yesterday (UTC by default):

1. Read every conversation message in the window from ``agent_memory.db``.
2. Summarize the day as four-section markdown. The heuristic summarizer
   (``_heuristic_project_summary``, real signals: top keywords, friction
   patterns) is the DEFAULT — the nightly pass spends no model call unless
   ``HYDRATION_LLM_SUMMARY`` is opted in. When opted in, ChatBlock produces
   the summary and the heuristic is the fallback if no LLM is reachable.
3. **Close the loop**: write recurring topics + friction signals back to
   ``projects.set_fact`` (read by the chat router via
   ``project_memory.build_memory_context``) AND ``agent_memory.set_agent_fact``
   (read by the runtime agent path at ``app/agents/runtime.py:548``), and
   record each friction signal as a pattern on the learning engine so it
   accumulates a corpus over time. This is what makes hydration "learning"
   instead of just an operator daily report.
4. Persist the row to ``hydration.db`` for the operator-facing endpoints.

Hydration reads conversations only. It never discovers, attaches, downloads
or re-indexes documents: the app reads an original only when a person asks
(an admin runs the ingest, or a user opens a cited document), and only the
admin path adds to a project's knowledge base (docs/INGEST_EXCLUSION_RULE.md).

Failures inside one project must not abort the whole run — each project is
isolated, errors are captured per-project in the row's ``facts`` payload.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.offload import off_loop_in


logger = logging.getLogger(__name__)


_MAX_MESSAGES_PER_PROJECT = 400
_SUMMARY_MAX_TOKENS = 600


def _llm_summary_enabled() -> bool:
    """Whether hydration may spend a model call on its day summaries.

    OFF by default. Hydration runs unattended every night and the LLM
    summary is a nicety, not load-bearing — the heuristic path produces the
    same four-section shape from real signals. Defaulting to an LLM call
    meant the nightly pass billed the provider on a schedule with no user in
    the loop (one global summary every night even on a zero-activity day,
    plus one per active project). The capability stays; it is now opt-in via
    ``HYDRATION_LLM_SUMMARY`` so a fresh deploy never spends unasked.
    """
    return os.getenv("HYDRATION_LLM_SUMMARY", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ── Public entry points ───────────────────────────────────────────────────


async def run(
    target_date: Optional[str] = None,
    project_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute one full hydration pass. ``target_date`` (YYYY-MM-DD) overrides
    the default of "yesterday UTC". ``project_ids`` overrides auto-detection
    from the conversation store — useful for forced re-runs."""
    from app.core import hydration_store

    target_date = target_date or _yesterday_utc_iso()
    window_start, window_end = _utc_day_bounds(target_date)

    if project_ids:
        pids: List[str] = [str(p) for p in project_ids if p]
    else:
        pids = await off_loop_in("background", _projects_active_in_window, window_start, window_end)

    # Drop project_ids with no row in the projects table before doing any
    # (expensive) per-project work: hydration_runs.project_id is a FK to
    # projects, so an orphan pid — a deleted or never-persisted project still
    # referenced by an old conversation — would raise ForeignKeyViolation on
    # Postgres (PYTHON-FASTAPI-7). Skip silently; a deleted project has nothing
    # to hydrate.
    pids = await off_loop_in("background", _filter_existing_projects, pids)

    results_per_project: List[Dict[str, Any]] = []
    global_errors: List[str] = []

    for pid in pids:
        try:
            row = await _hydrate_project(pid, target_date, window_start, window_end)
            results_per_project.append(row)
        except Exception as exc:  # noqa: BLE001 — never abort the whole pass
            logger.exception("hydration: project %s failed", pid)
            global_errors.append(f"{pid}: {type(exc).__name__}: {exc}")

    global_facts = {
        "projects_processed": len(results_per_project),
        "project_ids": [r["project_id"] for r in results_per_project],
        "per_project_message_counts": {
            r["project_id"]: r.get("messages_seen", 0)
            for r in results_per_project
        },
        "errors": global_errors,
    }
    global_summary_md, global_provider = await _summarize_global(
        results_per_project, target_date
    )
    if global_provider in ("offline_template", "unavailable", "error", "disabled"):
        global_summary_md = await off_loop_in("background", 
            _heuristic_global_summary, target_date, results_per_project, global_errors
        )
    await off_loop_in("background", 
        hydration_store.record_run,
        run_date=target_date,
        scope="global",
        project_id=None,
        summary_md=global_summary_md,
        facts=global_facts,
        provider=global_provider,
    )

    return {
        "status": "success",
        "run_date": target_date,
        "projects_processed": len(results_per_project),
        "errors": global_errors,
        "summary_md": global_summary_md,
    }


def get_latest(scope: str, project_id: Optional[str] = None) -> Dict[str, Any]:
    """Latest run for the given scope/project. ``{status: 'empty'}`` when none."""
    from app.core import hydration_store

    if scope not in ("global", "project"):
        return {"status": "error", "error": f"invalid scope: {scope!r}"}
    if scope == "project" and not project_id:
        return {"status": "error", "error": "project scope requires project_id"}
    row = hydration_store.get_latest(scope, project_id)
    if not row:
        return {"status": "empty", "scope": scope, "project_id": project_id}
    return {"status": "success", **row}


def list_history(
    scope: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    from app.core import hydration_store

    rows = hydration_store.list_history(scope=scope, project_id=project_id, limit=limit)
    return {"status": "success", "count": len(rows), "runs": rows}


# ── per-project work ──────────────────────────────────────────────────────


async def _hydrate_project(
    project_id: str,
    run_date: str,
    window_start: str,
    window_end: str,
) -> Dict[str, Any]:
    from app.core import hydration_store

    messages = await off_loop_in("background", _collect_project_messages, project_id, window_start, window_end)
    if len(messages) > _MAX_MESSAGES_PER_PROJECT:
        messages = messages[-_MAX_MESSAGES_PER_PROJECT:]

    summary_md, provider = await _summarize_project(project_id, messages, run_date)
    if provider in ("offline_template", "unavailable", "error", "disabled"):
        summary_md = await off_loop_in("background", _heuristic_project_summary, project_id, run_date, messages)

    # ── Close the loop: write back to surfaces the next chat will read.
    # Failures here are non-fatal — the operator-facing row is still useful
    # even if e.g. the learning_engine's storage is read-only.
    writeback_summary: Dict[str, Any] = {}
    try:
        writeback_summary = await off_loop_in("background", 
            _writeback_for_next_chat, project_id, messages, run_date
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("hydration writeback for project %s failed: %s", project_id, exc)
        writeback_summary = {"writeback_error": f"{type(exc).__name__}: {exc}"}

    facts = {
        "messages_seen": len(messages),
        "writeback": writeback_summary,
    }
    await off_loop_in("background", 
        hydration_store.record_run,
        run_date=run_date,
        scope="project",
        project_id=project_id,
        summary_md=summary_md,
        facts=facts,
        provider=provider,
    )
    return {
        "project_id": project_id,
        "messages_seen": len(messages),
        "summary_md": summary_md,
    }


# ── The actual learning step: writeback to surfaces chat consults ─────────


def _writeback_for_next_chat(
    project_id: str,
    messages: List[Dict[str, Any]],
    run_date: str,
) -> Dict[str, Any]:
    """Persist what we learned into stores tomorrow's chat will pull as
    priors. Two parallel surfaces, kept in lockstep so both code paths benefit:

    * ``projects.set_fact`` — keyed facts on the project itself, picked up by
      ``project_memory.build_memory_context`` which the chat router injects
      as a system message at conversation start.
    * ``agent_memory.set_agent_fact`` — agent-scoped facts, picked up by
      ``app/agents/runtime.py``'s message-building loop for any agent
      answering on this project.

    Also records each friction signal as a pattern via the learning engine
    so observations accumulate across runs — future work (tier promotion of
    chat-routing decisions, etc.) can mine this corpus.
    """
    import json as _json

    topics = [w for w, _c in _top_keywords(messages, n=8)]
    friction = _user_friction_signals(messages)
    asks = _top_user_asks(messages, n=5)

    written: Dict[str, Any] = {
        "project_facts": 0,
        "agent_facts": 0,
        "patterns_recorded": 0,
        "skipped": [],
    }

    payload = {"topics": topics, "friction": friction, "asks": asks, "run_date": run_date}

    try:
        from app.core import projects as projects_store

        if topics:
            projects_store.set_fact(
                project_id, "hydration:topics", ", ".join(topics),
                source_document="hydration", confidence=0.5,
            )
            written["project_facts"] += 1
        if friction:
            projects_store.set_fact(
                project_id, "hydration:friction", "; ".join(friction),
                source_document="hydration", confidence=0.5,
            )
            written["project_facts"] += 1
        projects_store.set_fact(
            project_id, "hydration:last_run", run_date,
            source_document="hydration", confidence=1.0,
        )
        written["project_facts"] += 1
    except Exception as exc:  # noqa: BLE001
        written["skipped"].append(f"project_facts: {type(exc).__name__}: {exc}")

    try:
        from app.core import agent_memory

        agent_memory.set_agent_fact(
            agent_name="chat",
            project_id=project_id,
            key="hydration:last_brief",
            value=_json.dumps(payload, ensure_ascii=False),
        )
        written["agent_facts"] += 1
    except Exception as exc:  # noqa: BLE001
        written["skipped"].append(f"agent_facts: {type(exc).__name__}: {exc}")

    try:
        from app.blocks import BLOCK_REGISTRY

        cls = BLOCK_REGISTRY.get("learning_engine")
        if cls is not None and friction:
            # shared_instance() — hydration friction writeback runs alongside
            # smart_orchestrator._record_routing_decision, both touching
            # _record_pattern. The per-instance state lock keeps them coherent.
            le = cls.shared_instance()
            for signal in friction:
                result = le._record_pattern(  # type: ignore[attr-defined]
                    {"project_id": project_id, "category": "friction",
                     "observation": signal, "source": "hydration",
                     "run_date": run_date}, {},
                )
                if isinstance(result, dict) and result.get("status") == "success":
                    written["patterns_recorded"] += 1
    except Exception as exc:  # noqa: BLE001
        written["skipped"].append(f"learning_engine: {type(exc).__name__}: {exc}")

    return written


def _top_user_asks(messages: List[Dict[str, Any]], n: int = 5) -> List[str]:
    """First line of each unique user message, trimmed. Stable across runs."""
    seen: set = set()
    out: List[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        text = (m.get("content") or "").strip().replace("\n", " ")
        key = text[:60].lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text[:160] + ("…" if len(text) > 160 else ""))
        if len(out) >= n:
            break
    return out


# ── Summarization (delegates to ChatBlock) ────────────────────────────────

async def _summarize_project(
    project_id: str,
    messages: List[Dict[str, Any]],
    run_date: str,
) -> tuple[str, str]:
    prompt = _build_project_summary_prompt(project_id, run_date, messages)
    return await _call_chat(prompt, max_tokens=_SUMMARY_MAX_TOKENS)


async def _summarize_global(
    per_project: List[Dict[str, Any]],
    run_date: str,
) -> tuple[str, str]:
    prompt = _build_global_summary_prompt(run_date, per_project)
    return await _call_chat(prompt, max_tokens=_SUMMARY_MAX_TOKENS)


# ── helpers (module-level so tests can monkey-patch them) ─────────────────

def _yesterday_utc_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")


def _utc_day_bounds(date_iso: str) -> tuple[str, str]:
    """Return (start, end) ISO-8601 timestamps for the UTC calendar day."""
    d = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = d.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (d + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return start, end


def _projects_active_in_window(window_start: str, window_end: str) -> List[str]:
    """Distinct project_ids whose conversations were updated in the window."""
    from app.core import agent_memory

    convs = agent_memory.list_conversations()
    seen: List[str] = []
    for c in convs:
        updated = c.get("updated_at") or c.get("created_at") or ""
        if window_start <= updated < window_end:
            pid = c.get("project_id")
            if pid and pid not in seen:
                seen.append(pid)
    return seen


def _filter_existing_projects(pids: List[str]) -> List[str]:
    """Keep only project_ids that have a row in the ``projects`` table.

    Pids arrive from conversation activity (see ``_projects_active_in_window``)
    or an explicit forced-rerun list, either of which can reference a project
    that was deleted or never persisted. ``hydration_runs.project_id`` is a FK
    to ``projects`` — hydrating an orphan raises ForeignKeyViolation on Postgres
    (SQLite enforces it too when foreign_keys is ON). The master-corpus alias is
    resolved to its backing project so the virtual pilot project is retained.
    """
    if not pids:
        return pids
    from app.core.db import SessionLocal
    from app.core.models import Project
    from app.core.projects import _master_corpus_source

    kept: List[str] = []
    skipped: List[str] = []
    with SessionLocal() as session:
        for pid in pids:
            source_id = _master_corpus_source(pid) or pid
            if session.get(Project, source_id) is not None:
                kept.append(pid)
            else:
                skipped.append(pid)
    if skipped:
        logger.info(
            "hydration: skipping %d project(s) with no projects-table row "
            "(deleted or never persisted): %s",
            len(skipped),
            skipped,
        )
    return kept


def _collect_project_messages(
    project_id: str, window_start: str, window_end: str
) -> List[Dict[str, Any]]:
    """All messages across the project's conversations whose timestamp falls
    inside the window. Returned in oldest-first order across conversations."""
    from app.core import agent_memory

    convs = agent_memory.list_conversations(project_id=project_id)
    out: List[Dict[str, Any]] = []
    for c in convs:
        msgs = agent_memory.get_messages(c["id"], limit=500)
        for m in msgs:
            ts = m.get("created_at") or ""
            if window_start <= ts < window_end:
                out.append({
                    "conversation_id": c["id"],
                    "role": m.get("role"),
                    "content": m.get("content") or "",
                    "created_at": ts,
                })
    out.sort(key=lambda m: m.get("created_at") or "")
    return out


def _build_project_summary_prompt(
    project_id: str,
    run_date: str,
    messages: List[Dict[str, Any]],
) -> str:
    transcript = "\n".join(
        f"[{m['role']}] {m['content']}"[:600] for m in messages
    ) or "(no user activity)"
    return (
        "You are a hydration agent producing operator-facing 'lessons learned' "
        f"for project '{project_id}' on {run_date}.\n\n"
        "Read the transcript below and respond in markdown with EXACTLY these sections:\n"
        "## What users asked for\n"
        "## Where they hit friction\n"
        "## Recurring patterns or themes\n"
        "## Lessons learned for the platform\n\n"
        "Be terse. Bullet points. Quote short user phrases when illustrative. "
        "Do NOT invent activity that is not in the transcript. If the transcript "
        "is empty, say 'No user activity in window' under each section.\n\n"
        "TRANSCRIPT:\n"
        f"{transcript}\n"
    )


def _build_global_summary_prompt(
    run_date: str, per_project: List[Dict[str, Any]]
) -> str:
    rollup = "\n\n".join(
        f"### Project {p['project_id']}\n"
        f"- Messages: {p.get('messages_seen', 0)}\n"
        f"{p.get('summary_md', '')[:1500]}"
        for p in per_project
    ) or "(no projects had activity)"
    return (
        f"You are producing the global hydration rollup for {run_date}.\n\n"
        "Below are per-project summaries already written. Produce a single "
        "tenant-wide markdown brief with EXACTLY these sections:\n"
        "## Activity at a glance\n"
        "## Cross-project lessons learned\n"
        "## Platform-level action items\n\n"
        "Be terse. Use bullet points. Do NOT repeat per-project detail verbatim — "
        "look for the cross-cutting signal.\n\n"
        "PER-PROJECT SUMMARIES:\n"
        f"{rollup}\n"
    )


# ── Heuristic summaries (used when ChatBlock has no model to call) ─────────

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "for", "in", "on",
    "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "this", "that", "these",
    "those", "it", "its", "i", "you", "we", "they", "he", "she", "them", "us",
    "my", "your", "our", "their", "his", "her", "what", "when", "where", "why",
    "how", "which", "who", "whom", "can", "could", "should", "would", "will",
    "shall", "may", "might", "must", "not", "no", "yes", "so", "than", "then",
    "also", "just", "very", "any", "some", "all", "each", "more", "most", "less",
    "least", "out", "up", "down", "into", "about", "over", "under",
}

_WORD_RX = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _top_keywords(messages: List[Dict[str, Any]], n: int = 8) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for m in messages:
        if m.get("role") != "user":
            continue
        for tok in _WORD_RX.findall(m.get("content") or ""):
            t = tok.lower()
            if t in _STOPWORDS:
                continue
            counter[t] += 1
    return counter.most_common(n)


def _user_friction_signals(messages: List[Dict[str, Any]]) -> List[str]:
    """Cheap heuristics that flag likely friction: repeated questions, short
    user messages followed by long retries, explicit complaint markers."""
    signals: List[str] = []
    complaint_rx = re.compile(
        r"\b(error|broken|doesn'?t work|not working|wrong|fail(ed|s)?|"
        r"missing|empty|why|stuck|help)\b",
        re.IGNORECASE,
    )
    user_msgs = [m for m in messages if m.get("role") == "user"]
    complaints = sum(1 for m in user_msgs if complaint_rx.search(m.get("content") or ""))
    if complaints:
        signals.append(f"{complaints} user message(s) used complaint/error language")

    # Repeated near-duplicate questions (same first 40 chars) — a sign the
    # user asked twice because the first answer didn't land.
    prefixes = [
        (m.get("content") or "")[:40].strip().lower()
        for m in user_msgs
        if (m.get("content") or "").strip()
    ]
    dup_count = sum(c - 1 for c in Counter(prefixes).values() if c > 1)
    if dup_count:
        signals.append(f"{dup_count} user message(s) appear to repeat earlier asks")
    return signals


def _heuristic_project_summary(
    project_id: str,
    run_date: str,
    messages: List[Dict[str, Any]],
) -> str:
    """Structured non-LLM summary. Reads real signals from the day's data so
    the row is useful even when no model was reachable."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]

    asks_section = "_No user activity in window._"
    if user_msgs:
        # First 5 unique user asks, trimmed
        seen = set()
        lines = []
        for m in user_msgs:
            t = (m.get("content") or "").strip().replace("\n", " ")
            key = t[:60].lower()
            if not t or key in seen:
                continue
            seen.add(key)
            lines.append(f"- {t[:160]}{'…' if len(t) > 160 else ''}")
            if len(lines) >= 5:
                break
        asks_section = "\n".join(lines)

    keywords = _top_keywords(messages)
    themes_section = (
        "\n".join(f"- `{w}` × {c}" for w, c in keywords)
        if keywords else "_No content to analyze._"
    )

    friction = _user_friction_signals(messages)
    friction_section = (
        "\n".join(f"- {s}" for s in friction)
        if friction else "_No obvious friction signals._"
    )

    lessons: List[str] = []
    if not messages:
        lessons.append("Project was idle today — nothing to learn.")
    if friction:
        lessons.append("Friction signals detected — review the asks above for retry patterns.")
    if not lessons:
        lessons.append("No notable lessons for the platform today.")

    return (
        f"# {project_id} — {run_date} (heuristic; no LLM reached)\n\n"
        f"_Stats: {len(user_msgs)} user msg, {len(asst_msgs)} assistant msg._\n\n"
        f"## What users asked for\n{asks_section}\n\n"
        f"## Where they hit friction\n{friction_section}\n\n"
        f"## Recurring patterns or themes\n{themes_section}\n\n"
        f"## Lessons learned for the platform\n"
        + "\n".join(f"- {l}" for l in lessons)
        + "\n"
    )


def _heuristic_global_summary(
    run_date: str,
    per_project: List[Dict[str, Any]],
    errors: List[str],
) -> str:
    if not per_project:
        return (
            f"# Global hydration — {run_date} (heuristic; no LLM reached)\n\n"
            "## Activity at a glance\n- No projects had user activity today.\n\n"
            "## Cross-project lessons learned\n- N/A.\n\n"
            "## Platform-level action items\n- None.\n"
        )
    # Sort projects by message volume for "busiest"
    busiest = sorted(
        per_project, key=lambda p: p.get("messages_seen", 0), reverse=True
    )[:5]
    activity_lines = [
        f"- **{p['project_id']}**: {p.get('messages_seen', 0)} msg"
        for p in busiest
    ]
    err_section = (
        "\n".join(f"- `{e}`" for e in errors[:10])
        if errors else "- No project-level failures.\n"
    )
    return (
        f"# Global hydration — {run_date} (heuristic; no LLM reached)\n\n"
        f"_Totals: {len(per_project)} projects active, "
        f"{len(errors)} project errors._\n\n"
        f"## Activity at a glance\n" + "\n".join(activity_lines) + "\n\n"
        f"## Cross-project lessons learned\n"
        f"- Per-project detail lives in the project-scoped rows; this rollup is "
        f"heuristic and surfaces volume + error signal only.\n\n"
        f"## Platform-level action items\n{err_section}\n"
    )


async def _call_chat(prompt: str, max_tokens: int = 600) -> tuple[str, str]:
    """Invoke the ChatBlock and return (text, provider). The ChatBlock's
    fallback chain (DeepSeek → OpenRouter → offline template) guarantees
    a response so hydration never crashes on a missing model.

    No-ops with provider ``"disabled"`` unless ``HYDRATION_LLM_SUMMARY`` is
    opted in: the model is never called on the nightly path by default, and
    the caller falls through to the heuristic summary. The ChatBlock is not
    even imported in the disabled case, so there is no model spend."""
    if not _llm_summary_enabled():
        return ("_LLM summary disabled (HYDRATION_LLM_SUMMARY off); heuristic used._",
                "disabled")

    from app.blocks import BLOCK_REGISTRY

    cls = BLOCK_REGISTRY.get("chat")
    if cls is None:
        return ("_Chat block not available; no summary produced._", "unavailable")
    block = cls()
    try:
        resp = await block.execute(
            {"text": prompt},
            {"max_tokens": max_tokens, "temperature": 0.2},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("hydration: chat call failed: %s", exc)
        return (f"_Summarizer failed: {type(exc).__name__}_", "error")

    text = ""
    if isinstance(resp, dict):
        text = (
            resp.get("response")
            or resp.get("text")
            or resp.get("message")
            or ""
        )
    provider = (resp.get("provider") if isinstance(resp, dict) else None) or "unknown"
    return (text or "_(empty response)_", provider)
