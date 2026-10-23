"""Project Reasoner — Reasoning Engine Plan 5.

The LLM agent. One turn: UNDERSTAND + PLAN (one LLM call returning plan JSON)
-> EXECUTE (PlanExecutor runs the steps) -> DELIVER (one LLM call writing the
answer from the executed results).
"""

import json
import logging
import os
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

# Shown when the planner cannot produce a JSON plan AND there is no project
# context to answer from. Deliberately user-facing — no planner internals.
_NO_PLAN_NO_CONTEXT_MESSAGE = (
    "I could not build a structured project plan for this question, but I "
    "also could not confirm an answer from the indexed project sources. "
    "Please narrow the question or provide a specific document or reference."
)

# Shown when the plan built and executed but writing the final answer failed
# (LLM error or empty reply). Never exposes the raw exception.
_DELIVER_FAILED_MESSAGE = (
    "I found relevant project context, but I could not generate the final "
    "written answer for this turn. Please retry or narrow the question."
)


def _build_sources_from_excerpts(excerpts: list) -> list:
    """Project-doc excerpts -> the clean source shape the chat path uses
    (``doc_name``/``doc_id``/``score``).

    The raw ``snippet`` is deliberately NOT carried through: chunk text can
    embed raw Drive/Windows paths (``X:\\Example Drive\\…``), and the chat path's
    structured sources don't expose snippets either."""
    out = []
    for e in excerpts or []:
        if not isinstance(e, dict):
            continue
        name = e.get("filename") or e.get("document_id")
        if not name:
            continue
        out.append({
            "doc_name": name,
            "doc_id": e.get("document_id"),
            "score": e.get("score"),
        })
    return out

from app.core.universal_base import UniversalBlock
from app.core.plan_executor import PlanExecutor
from app.prompts.reasoner_system import build_reasoner_prompt
from app.schemas.execution_plan import ExecutionPlan
from app.schemas.project_session import ProjectSession


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of an LLM reply (it may add prose or fences).
    Tries the whole string first, then decodes the first object starting at
    the first '{' — trailing prose (even prose containing a '}') is ignored.
    Raises ValueError when there is no parsable object."""
    if not isinstance(text, str):
        raise ValueError("no JSON object in LLM reply")
    try:
        return json.loads(text)
    except ValueError:
        logger.debug(
            "swallowed %s in _extract_json() — continuing",
            "ValueError", exc_info=True,
        )
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in LLM reply")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        raise ValueError("no JSON object in LLM reply")
    return obj


# Per-step bound for the DELIVER prompt: every step's result is included, but
# no single step may exceed this, so nothing is silently dropped.
_DELIVER_STEP_LIMIT = 2500


def _render_step_results(run) -> str:
    """Format this turn's step results (step + computed output) for the
    DELIVER prompt. Each output is bounded individually."""
    lines = []
    for i, sr in enumerate(run.step_results, 1):
        if sr.status == "success":
            rendered = json.dumps(sr.output, default=str)
            if len(rendered) > _DELIVER_STEP_LIMIT:
                rendered = rendered[:_DELIVER_STEP_LIMIT] + "… (truncated)"
            lines.append(
                f"{i}. {sr.type} (-> {sr.output_key or 'default'}): "
                f"OK\n   output: {rendered}"
            )
        else:
            lines.append(
                f"{i}. {sr.type} (-> {sr.output_key or 'default'}): "
                f"ERROR — {sr.error}"
            )
    return "\n".join(lines) if lines else "(no steps executed)"


class ProjectReasonerBlock(UniversalBlock):
    name = "project_reasoner"
    version = "1.0.0"
    description = (
        "Reasoning agent: UNDERSTAND -> PLAN -> EXECUTE -> DELIVER over a "
        "project session."
    )
    layer = 3
    tags = ["domain", "construction", "reasoning", "agent", "llm"]
    requires = []

    default_config = {"model": None}  # None -> active provider default_model

    ui_schema = {
        "input": {
            "type": "text",
            "placeholder": "Ask anything about your project...",
            "multiline": True,
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "answer", "type": "markdown", "label": "Answer"},
                {"name": "understanding", "type": "text", "label": "Understood as"},
            ],
        },
        "quick_actions": [
            {"icon": "", "label": "Critical path", "prompt": "What is the critical path?"},
            {"icon": "⏱️", "label": "Compress", "prompt": "How can I finish 2 weeks sooner?"},
        ],
    }

    async def _call_llm(self, prompt: str) -> str:
        """Active-LLM-provider call. Overridden by test doubles.

        Routes via app.agents.runtime._llm_config — Kimi primary, Groq fallback.
        """
        from app.agents.runtime import _llm_config  # local import: avoid cycle at module load
        cfg = _llm_config()
        api_key = os.getenv(cfg["env_key"])
        if not api_key:
            raise RuntimeError(f"{cfg['env_key']} not configured")
        # The active provider's default model unless a real one is pinned.
        model = self.config.get("model") or cfg["default_model"]
        # Honor a reasoning provider's fixed temperature (Kimi k2.6 rejects any
        # temperature but 1 with HTTP 400).
        from app.agents.runtime import _llm_http_timeout
        temperature = cfg.get("fixed_temperature", 0.2)
        async with httpx.AsyncClient(timeout=_llm_http_timeout()) as client:
            resp = await client.post(
                cfg["url"],
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"{cfg['provider']} API error (HTTP {resp.status_code})"
                )
            return resp.json()["choices"][0]["message"]["content"]

    async def _fallback_answer(self, request: str, excerpts: list) -> tuple:
        """Direct answer used when the planner can't produce a JSON plan.

        With project excerpts, answer grounded in them and return their
        sources. Without context, return a controlled message and no sources.
        Never raises and never exposes planner internals or raw LLM/tool
        output to the caller.
        """
        if not excerpts:
            return _NO_PLAN_NO_CONTEXT_MESSAGE, []

        excerpt_block = "\n\n".join(
            f"[{i}] {e.get('filename') or e.get('document_id') or 'source'}: "
            f"{(e.get('snippet') or '').strip()}"
            for i, e in enumerate(excerpts, 1)
            if isinstance(e, dict)
        )
        prompt = (
            "Answer the user's question about their project using ONLY the "
            "document excerpts below. If the excerpts do not contain the "
            "answer, say you could not confirm it in the indexed project "
            "sources. Cite filenames where relevant. Do not invent figures.\n\n"
            f"QUESTION:\n{request}\n\n"
            f"DOCUMENT EXCERPTS:\n{excerpt_block}"
        )
        try:
            answer = await self._call_llm(prompt)
        except Exception as e:                              # noqa: BLE001
            logger.warning(
                "project_reasoner: fallback answer call failed: %s", e
            )
            return _NO_PLAN_NO_CONTEXT_MESSAGE, []

        answer = (answer or "").strip()
        if not answer:
            return _NO_PLAN_NO_CONTEXT_MESSAGE, []
        return answer, _build_sources_from_excerpts(excerpts)

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}
        request = data.get("request") or data.get("text") \
            or params.get("request") \
            or (str(input_data) if not isinstance(input_data, dict) else "")
        session: ProjectSession = data.get("session") or params.get("session")
        project_id = data.get("project_id") or params.get("project_id")

        request = request or ""
        if not request.strip():
            return {"status": "error", "error": "No request provided"}
        if session is None:
            return {"status": "error", "error": "No session provided"}

        session.add_message("user", request)

        # Look up top-k relevant snippets from this project's indexed
        # documents so the planner can ground its steps in the actual
        # uploaded files. Silent no-op when no project_id is given or the
        # project has nothing indexed yet — the reasoner falls back to
        # session-state-only planning, matching the old behavior.
        excerpts: list = []
        if project_id:
            try:
                from app.core.doc_index import search_project_documents
                excerpts = await search_project_documents(
                    project_id, request, top_k=5
                )
            except Exception:
                excerpts = []

        # ── UNDERSTAND + PLAN ────────────────────────────────────────────
        try:
            plan_reply = await self._call_llm(
                build_reasoner_prompt(session, request, excerpts)
            )
            plan = ExecutionPlan.model_validate(_extract_json(plan_reply))
        except Exception as e:                              # noqa: BLE001
            # Small models often answer general/conversational questions in
            # prose instead of emitting plan JSON, so the parse/validate
            # fails here. Degrade gracefully instead of surfacing the raw
            # planner error: answer directly from the project excerpts when
            # we have them (with sources), else a controlled message.
            logger.warning(
                "project_reasoner: plan build failed, degrading to direct "
                "answer: %s", e,
            )
            answer, sources = await self._fallback_answer(request, excerpts)
            session.add_message("assistant", answer)
            return {
                "status": "success",
                "answer": answer,
                "understanding": "",
                "plan": None,
                "execution": None,
                "sources": sources,
                "degraded": True,
                # Grounding evidence for the router's cost-grounding gate. NOT
                # returned to the client (the router pops it) — the confidential
                # snippet text is used only to decide whether a cost figure in
                # the answer traces to retrieved rate-semantic evidence.
                "_grounding": {"excerpts": excerpts, "results_text": ""},
            }

        # ── EXECUTE ──────────────────────────────────────────────────────
        run = await PlanExecutor().run(plan, session)

        # ── DELIVER ──────────────────────────────────────────────────────
        # Build the prompt from THIS turn's step results, not the whole
        # accumulated session blob — so the answer sees exactly what was just
        # computed and no step's output is silently truncated away.
        deliver_prompt = (
            f"You planned and executed steps for this request:\n{request}\n\n"
            f"UNDERSTANDING: {plan.understanding}\n"
            f"EXECUTION STATUS: {run.status}\n"
            f"STEP RESULTS (from this turn):\n"
            f"{_render_step_results(run)}\n\n"
            f"Write a clear, concise answer for the user from these results. "
            f"If the status is error or partial, explain what is missing."
        )
        deliver_sources: list = []
        try:
            answer = (await self._call_llm(deliver_prompt) or "").strip()
        except Exception as e:                              # noqa: BLE001
            logger.warning("project_reasoner: deliver step failed: %s", e)
            answer = ""
        if not answer:
            # The plan built and executed, but writing the final answer failed
            # (LLM error or empty reply). Never surface the raw exception or a
            # blank answer — return a controlled message and preserve any
            # retrieved sources so the turn still carries evidence.
            answer = _DELIVER_FAILED_MESSAGE
            deliver_sources = _build_sources_from_excerpts(excerpts)

        session.add_message("assistant", answer)

        status = run.status
        return {
            "status": status,
            "answer": answer,
            "understanding": plan.understanding,
            "plan": plan.model_dump(mode="json"),
            "execution": run.model_dump(mode="json"),
            "sources": deliver_sources,
            # Grounding evidence for the router's cost-grounding gate (see the
            # fallback return above). Retrieved excerpts are classified per
            # rate-semantics; this turn's executed step results are
            # authoritative computed figures that ground the delivered answer.
            "_grounding": {
                "excerpts": excerpts,
                "results_text": _render_step_results(run),
            },
        }
