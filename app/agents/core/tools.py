"""The tools a turn offers and the dispatch of one tool call.

Moved from the Agent class in app/agents/runtime.py by
scripts/move_agent_methods.py (F-DRIVER Phase A). Each function is still
an Agent method (the class binds it); runtime names are read at call
time, so this module does not import runtime when it is imported.
"""
from __future__ import annotations

from typing import Any


def tool_definitions(self, project_id: str | None = None) -> list[dict[str, Any]]:
    """Build DeepSeek-style tool definitions.

    Includes one tool per allowed block, plus synthetic tools:
    - ``remember_fact`` — always available.
    - ``search_project_documents`` — only when ``project_id`` is set.
    - ``delegate_to_agent`` — only when ``self.can_delegate``.
    """

    from app.agents.runtime import (  # noqa: F401 -- read at call time
        BLOCK_REGISTRY, _FILE_TOOL_SCHEMAS, _construction_calc_tool_schema,
    )
    from app.core.privileges import caller_may_use_block

    tools = []
    for block_name in self.allowed_blocks:
        block_class = BLOCK_REGISTRY.get(block_name)
        if not block_class or not caller_may_use_block(block_name):
            continue  # a tool the caller would be refused is not offered
        # File-consuming blocks get a typed schema with required file_path.
        override = _FILE_TOOL_SCHEMAS.get(block_name)
        if override:
            tools.append({
                "type": "function",
                "function": {
                    "name": block_name,
                    "description": override["description"],
                    "parameters": {
                        "type": "object",
                        "properties": override["properties"],
                        "required": override["required"],
                    },
                },
            })
            continue
        description = (getattr(block_class, "description", "") or f"Block: {block_name}")[:1024]
        tools.append({
            "type": "function",
            "function": {
                "name": block_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "description": "Input for the block — string, dict, or chain output.",
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional block-specific parameters (e.g. {'action': 'auto_pipeline'}).",
                        },
                    },
                    "required": [],
                },
            },
        })

    # ── synthetic tool: remember_fact (always available) ─────────────────
    tools.append({
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Persist a fact you should remember in future turns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short identifier for the fact."},
                    "value": {"type": "string", "description": "The fact value to remember."},
                },
                "required": ["key", "value"],
            },
        },
    })

    # ── synthetic tool: search_project_documents (project-scoped) ────────
    if project_id:
        tools.append({
            "type": "function",
            "function": {
                "name": "search_project_documents",
                "description": "Search inside this project's documents (including imported Drive files).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for."},
                        # Some providers (Groq/llama-3.3-70b in particular) emit numeric tool
                        # args as strings — declaring this as ["integer","string"] avoids the
                        # provider-side tool_use_failed validator rejecting the call. The
                        # Python side at _run_tool_call coerces with `top_k or 5`, so a
                        # string here works at runtime.
                        "top_k": {"type": ["integer", "string"], "description": "Max number of results (default 5)."},
                    },
                    "required": ["query"],
                },
            },
        })

    # ── synthetic tool: list_project_documents (project-scoped) ──────────
    # The document REGISTER, deterministically from the project store.
    # Without this, "list the documents in this project / what do we have"
    # can only be answered by hoping a semantic search surfaces a register
    # that does not exist as a chunk — verified failing on the golden set
    # (pilot_document_metadata, 2026-08-02): the model honestly reported
    # the RAG context held no document list.
    if project_id:
        tools.append({
            "type": "function",
            "function": {
                "name": "list_project_documents",
                "description": (
                    "List this project's document register: each document's "
                    "filename, type, and size. Use this when the user asks "
                    "WHAT documents/files the project has. To search INSIDE "
                    "document content use search_project_documents."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": ["integer", "string"], "description": "Max documents to return (default 100, newest first)."},
                    },
                    "required": [],
                },
            },
        })

    # ── synthetic tool: fetch_document (project-scoped) ──────────────────
    # Complements search_project_documents: fetches ONE specific document's
    # content by document_id or filename. This is what makes "work on the
    # attached file" actionable — the attachment note gives the agent a
    # concrete document_id, and this tool reads that document directly
    # instead of hoping a semantic query happens to surface it.
    if project_id:
        tools.append({
            "type": "function",
            "function": {
                "name": "fetch_document",
                "description": (
                    "Fetch the content of ONE specific project document by "
                    "document_id or filename. Use this when the user refers to a "
                    "specific or attached file. For open-ended questions across "
                    "the corpus use search_project_documents instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string", "description": "The document id (preferred when known, e.g. from an attachment note or a prior search result)."},
                        "filename": {"type": "string", "description": "The document's original filename (used when the id is not known)."},
                    },
                    "required": [],
                },
            },
        })

    # ── synthetic tool: generate_wbs (when construction is allowed) ──────
    # Exposed as a top-level tool with an explicit param schema so the
    # agent never has to guess the params shape. The generic `construction`
    # tool stayed advertised with "input/params" only, and the agent kept
    # emitting empty `action` fields, retrying, and eventually escaping to
    # delegate_to_agent (which hit the iteration cap). This direct tool
    # eliminates that ambiguity.
    if "construction" in self.allowed_blocks:
        tools.append({
            "type": "function",
            "function": {
                "name": "generate_wbs",
                "description": (
                    "Generate a CPM-validated Work Breakdown Structure / schedule. "
                    "Returns an activity list with ES/EF/LS/LF/total_float per activity, "
                    "phase tree, and assumptions. CALL ONCE — the tool is deterministic "
                    "and re-calling with the same params returns the same large result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brief": {
                            "type": "string",
                            "description": "Project brief / scope description (from RFP, BOD, conversation)."
                        },
                        "target_count": {
                            # Some providers (Groq/llama-3.x/llama-4-scout) emit integer
                            # tool args as strings. Declaring ["integer","string"] keeps
                            # the strict tool-use validator happy; we coerce in
                            # _run_tool_call before passing to ConstructionContainer.
                            "type": ["integer", "string"],
                            "description": "Target number of activities (default 200, clamped to [20, 1000]).",
                        },
                        "project_type": {
                            "type": "string",
                            "enum": ["data_center", "solar_plant", "wind_farm", "building", "infrastructure"],
                            "description": "Project type — determines the WBS template scaffold.",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Schedule start date in ISO format (YYYY-MM-DD). Optional — defaults to today.",
                        },
                        "duration_overrides": {
                            "type": "object",
                            "additionalProperties": {"type": ["integer", "string"]},
                            "description": (
                                "Activity-name substring → working days "
                                "(e.g. {\"slab\": 6}). Parsed from the user "
                                "message when omitted — 'use N days per slab "
                                "and re-run' is applied even if this is empty."
                            ),
                        },
                    },
                    "required": ["brief"],
                },
            },
        })
        # ── synthetic tool: commissioning_checklist ──────────────────────
        tools.append({
            "type": "function",
            "function": {
                "name": "commissioning_checklist",
                "description": (
                    "Generate a systems commissioning / testing & commissioning "
                    "checklist with the REAL test standards and acceptance "
                    "criteria per system (HVAC: ASHRAE/AHRI; electrical: "
                    "IEEE/BS 7671; fire: NFPA; plumbing/lift/facade/BMS too). "
                    "Call this whenever the user asks to produce/generate a "
                    "commissioning or T&C checklist — do NOT invent test "
                    "standards yourself; this tool returns the authoritative ones."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "systems": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Systems to commission. Map the user's wording to "
                                "these keys: waterproofing, hvac, electrical, fire, "
                                "plumbing, elevator, facade, bms. For torch-applied "
                                "membrane / pre-backfill waterproofing use "
                                "waterproofing — do not default to HVAC."
                            ),
                        },
                    },
                    "required": ["systems"],
                },
            },
        })
        # ── synthetic tool: construction_calc (deterministic formulas) ───
        tools.append(_construction_calc_tool_schema())
        # ── synthetic tool: cash_flow_forecast (S-curve) ────────────────
        # Same reason as generate_wbs: the generic `construction` tool's
        # input/params shape lets the model narrate an S-curve instead of
        # calling cash_flow_forecast (pinned construction-pm, 12-month
        # cash flow). A typed top-level tool is the deterministic lever;
        # K2 rejects forced tool_choice.
        tools.append({
            "type": "function",
            "function": {
                "name": "cash_flow_forecast",
                "description": (
                    "Build a monthly S-curve / cash-flow forecast from a "
                    "contract value and duration. CALL THIS immediately "
                    "when the user asks for an S-curve, cash flow, spend "
                    "curve, or monthly drawdown. Do not invent monthly "
                    "percents in prose — this tool is deterministic."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contract_value": {
                            "type": ["number", "string"],
                            "description": "Contract / project value (numeric).",
                        },
                        "duration_months": {
                            "type": ["integer", "string"],
                            "description": "Forecast horizon in months (e.g. 12).",
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "Original user request. Used to parse "
                                "figures when contract_value / duration "
                                "are omitted."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        })
        # ── synthetic tool: payment_certificate (IPC) ───────────────────
        # Forced tool_choice / the generic construction envelope emit
        # ``{}`` and ask2 (measured works / MOS / contract sum) never
        # reached the container. A typed top-level tool + NL coercion
        # is the same lever as cash_flow_forecast.
        tools.append({
            "type": "function",
            "function": {
                "name": "payment_certificate",
                "description": (
                    "Issue an Interim Payment Certificate from the "
                    "operator's figures (measured works, retention, "
                    "materials on site / MOS, contract sum). CALL THIS "
                    "when the user asks for an IPC / payment certificate. "
                    "Do not invent figures — this tool parses the ask."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contract_value": {
                            "type": ["number", "string"],
                            "description": "Accepted contract amount / contract sum.",
                        },
                        "gross_valuation": {
                            "type": ["number", "string"],
                            "description": "This-period gross / certified work.",
                        },
                        "measured_works": {
                            "type": ["number", "string"],
                            "description": "Value of measured works this period.",
                        },
                        "materials_on_site": {
                            "type": ["number", "string"],
                            "description": "Materials on site (MOS) this period.",
                        },
                        "retention_percent": {
                            "type": ["number", "string"],
                            "description": "Retention percent (e.g. 5 or 10).",
                        },
                        "work_done_percent": {
                            "type": ["number", "string"],
                            "description": "Percent complete when no gross is given.",
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "Original user request. Used to parse "
                                "figures when the numeric fields are omitted."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        })
        # ── synthetic tool: resource_histogram ───────────────────────────
        # Same reason as cash_flow_forecast: pinned construction-pm will
        # otherwise call primavera_parser (the `xer` intent map) and
        # invent W1–W4 crew rows. A typed top-level tool + predispatch
        # is the deterministic lever.
        tools.append({
            "type": "function",
            "function": {
                "name": "resource_histogram",
                "description": (
                    "Build a time-phased manpower / activity histogram "
                    "from a Primavera P6 .xer. CALL THIS when the user "
                    "asks for a resource, manpower, labor, or crew "
                    "histogram. Uses real TASKRSRC labor hours when "
                    "present; otherwise returns CPM activity-count week "
                    "buckets — never invent man-hours. Prefer the .xer "
                    "the user named. Do NOT call primavera_parser for a "
                    "histogram."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "schedule_file": {
                            "type": "string",
                            "description": (
                                "The .xer original_name "
                                "(e.g. resource_loaded.xer). Must be a "
                                "project document name."
                            ),
                        },
                        "period_unit": {
                            "type": "string",
                            "enum": ["week", "month"],
                            "description": "Bucket size. Default week.",
                        },
                    },
                    "required": [],
                },
            },
        })
        # ── synthetic tool: look_ahead ────────────────────────────────────
        tools.append({
            "type": "function",
            "function": {
                "name": "look_ahead",
                "description": (
                    "Build a 3–4 week look-ahead from a Primavera P6 "
                    ".xer, OR from activities listed in the message "
                    "(pass them as `activities`). CALL THIS when the user asks for a look-ahead "
                    "/ lookahead programme. Returns activities overlapping "
                    "the window with ES/EF, remaining duration, total "
                    "float, critical flag, WBS, and name. Never invent "
                    "activities. Prefer the .xer the user named. Do NOT "
                    "call primavera_parser for a look-ahead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "schedule_file": {
                            "type": "string",
                            "description": (
                                "The .xer original_name. Must be a "
                                "project document name."
                            ),
                        },
                        "weeks": {
                            "type": "number",
                            "description": "Look-ahead length in weeks (default 3).",
                        },
                        "days": {
                            "type": "number",
                            "description": "Look-ahead length in calendar days (overrides weeks).",
                        },
                        "as_of": {
                            "type": "string",
                            "description": (
                                "Reference date YYYY-MM-DD for the window "
                                "start. Pass the date the user states as "
                                "today ('Today is 21 September', "
                                "'as of today, 21 September', "
                                "'today's date is 21 September', "
                                "'as at 21 September', '21/09', "
                                "'21/09/2026'). Omit only when no date is "
                                "stated; the tool then uses the real clock."
                            ),
                        },
                        "activities": {
                            "type": "array",
                            "description": (
                                "Activities the USER listed in the message. "
                                "Pass them when the message contains the "
                                "activities and no .xer is named — then no "
                                "schedule file is needed. Never invent rows."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "start": {"type": "string", "description": "YYYY-MM-DD"},
                                    "finish": {"type": "string", "description": "YYYY-MM-DD"},
                                },
                            },
                        },
                    },
                    "required": [],
                },
            },
        })
        # ── synthetic tool: evm_calculate ────────────────────────────────
        tools.append({
            "type": "function",
            "function": {
                "name": "evm_calculate",
                "description": (
                    "Classic Earned Value Management. REQUIRES Planned "
                    "Value (PV), Earned Value (EV), and Actual Cost (AC). "
                    "BAC optional for EAC/ETC/VAC. Returns SPI, CPI, SV, "
                    "CV. Do not invent actuals — refuse if PV/EV/AC missing. "
                    "Prefer this over progress_tracker for true EVM."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pv": {"type": "number", "description": "Planned Value (BCWS)"},
                        "ev": {"type": "number", "description": "Earned Value (BCWP)"},
                        "ac": {"type": "number", "description": "Actual Cost (ACWP)"},
                        "bac": {"type": "number", "description": "Budget at Completion (optional)"},
                        "bcws": {"type": "number"},
                        "bcwp": {"type": "number"},
                        "acwp": {"type": "number"},
                    },
                    "required": [],
                },
            },
        })
        # ── synthetic tool: procurement_list_generator ───────────────────
        # Same reason as generate_wbs / cash_flow_forecast: the generic
        # `construction` tool's input/params shape lets the model say
        # "procurement_list_generator is not in my toolkit" and fall
        # through to construction_calc (live Phase 2, second ask).
        tools.append({
            "type": "function",
            "function": {
                "name": "procurement_list_generator",
                "description": (
                    "Build a prioritised procurement / buy list from BOQ "
                    "line items or discrete quantities. CALL THIS when "
                    "the user asks for a procurement list, material list, "
                    "purchase list, or what materials to buy. Do not "
                    "invent line items in prose and do not use "
                    "construction_calc for this deliverable. Empty BOQ / "
                    "quantities returns an honest empty list."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "quantities": {
                            "type": "object",
                            "description": (
                                "Discrete item → {quantity, unit} map "
                                "(e.g. {\"Rebar\": {\"quantity\": 3.2, "
                                "\"unit\": \"t\"}})."
                            ),
                        },
                        "boq": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "BOQ / estimate line items.",
                        },
                        "budget": {
                            "type": ["number", "string"],
                            "description": "Optional budget for variance.",
                        },
                        "location": {
                            "type": "string",
                            "description": "Rate-lookup location.",
                        },
                        "project_type": {
                            "type": "string",
                            "description": "Project type for rate lookup.",
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "Original user request. Used when "
                                "quantities / boq are omitted."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        })
        # ── synthetic tool: rfi_generator ────────────────────────────────
        # Same reason as generate_wbs / cash_flow_forecast: the generic
        # `construction` tool's input/params shape lets the model say
        # "rfi_generator is not in my toolkit" and fall through to
        # construction_calc or write RFI prose (live Phase 2).
        tools.append({
            "type": "function",
            "function": {
                "name": "rfi_generator",
                "description": (
                    "Draft a Request for Information (RFI) from an "
                    "engineering clarification, drawing/spec issue, or "
                    "follow-on question. CALL THIS when the user asks "
                    "to draft / raise / create an RFI or a request for "
                    "information. Do not invent the RFI in prose and "
                    "do not use construction_calc for this deliverable."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issues": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Runnable issues to turn into RFIs "
                                "({description, type, severity})."
                            ),
                        },
                        "drawing_ref": {
                            "type": "string",
                            "description": "Drawing / spec reference.",
                        },
                        "project_name": {
                            "type": "string",
                            "description": "Project name on the RFI.",
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "Original user request. Used to draft "
                                "the question when issues are omitted."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        })

    # ── synthetic tool: delegate_to_agent (delegating agents only) ───────
    if self.can_delegate:
        tools.append({
            "type": "function",
            "function": {
                "name": "delegate_to_agent",
                "description": (
                    "Hand off a sub-task to a specialist agent and receive its answer. "
                    "Use when another agent is better suited to part of the request."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string", "description": "Name of the specialist agent to delegate to."},
                        "message": {"type": "string", "description": "The task / question for that agent."},
                    },
                    "required": ["agent_name", "message"],
                },
            },
        })

    return tools


async def _run_tool_call(
    self,
    tool_call: dict[str, Any],
    api_key: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    _depth: int = 0,
    _call_stack: list[str] | None = None,
    user_message: str | None = None,
    history: list | None = None,
) -> dict[str, Any]:
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        BLOCK_REGISTRY, _FILE_CONSUMING_BLOCKS, _HAT_CALC_ALIASES, _HAT_CONTAINER_ALIASES,
        _auto_validate, _create_block_instance, _dispatch_args, _ipc_args_from_ask, _off_loop,
        _resolve_block_file_input, block_instances, json,
    )
    fn = tool_call.get("function") or {}
    name = fn.get("name") or ""
    raw_args = fn.get("arguments") or "{}"
    try:
        # Presentation params (char_offset) are removed here, not later:
        # they choose which window of the SERIALIZED result comes back,
        # and the block reads the whole file either way. The window
        # itself is read off the tool call by _requested_char_offset.
        args = _dispatch_args(raw_args)
    except json.JSONDecodeError:
        return {
            "name": name,
            "ok": False,
            "result": {
                "status": "error",
                "error": f"Invalid JSON args: {raw_args[:200]}",
                "hint": "Re-issue the tool call with valid JSON arguments.",
            },
        }

    _call_stack = _call_stack or [self.name]

    # Retrieval-shaped manifest actions are the search tool by another
    # name; rewrite and fall through to the search branch below.
    if name in ("search_documents", "fidic_clause_lookup", "standards_lookup"):
        name = "search_project_documents"

    # Tools owned by a package (app.agents.base, app.agents.hats.<hat>)
    # answer from their own module; see app.agents.core.tool_registry.
    from app.agents.core import tool_registry as _tool_registry

    _spec = _tool_registry.get(name)
    if _spec is not None:
        return await _spec.handler(_tool_registry.ToolCall(
            agent=self, name=name, args=args, tool_call=tool_call, api_key=api_key,
            project_id=project_id, conversation_id=conversation_id, depth=_depth,
            call_stack=_call_stack, user_message=user_message, history=history,
        ))

    # ── declared hat-manifest actions -> real implementations ───────────
    # Calculator aliases: the declared name runs the registry calculator
    # it always meant. The result carries `aliased_to` so the trace shows
    # the real machinery that produced the numbers.
    if name in ("concrete_maturity_calculator", "grout_pressure_calculator",
                "thermal_crack_assessor", "mix_design_validator",
                "crane_planner", "critical_path_calc", "float_analysis",
                "tender_evaluator", "supplier_scoring",
                "compliance_gate_checker", "ncr_tracker"):
        from app.lib import construction_formulas as _cf
        calc = _HAT_CALC_ALIASES[name]
        result = _cf.run_calculation(calc, args.get("params") or args)
        payload = result if isinstance(result, dict) else {"result": result}
        return {
            "name": name,
            "ok": isinstance(result, dict) and result.get("status") != "error",
            "result": {"aliased_to": f"construction_calc:{calc}", **payload},
        }

    # Container aliases: the declared name runs the construction
    # container route that implements it (same gate as generate_wbs:
    # the agent must hold the construction block).
    if name in ("interim_certificate_generator", "schedule_analysis",
                "baseline_compare", "eot_claim_assessor",
                "dispute_timeline_builder", "variation_order_generator",
                "boq_cost_analyzer", "drawing_qto_extract",
                "milestone_tracker", "rate_card_lookup", "itp_generator",
                "po_generator"):
        if "construction" not in self.allowed_blocks:
            return {
                "name": name, "ok": False,
                "result": {"status": "error",
                           "error": "construction container not in agent's allowed_blocks"},
            }
        route = _HAT_CONTAINER_ALIASES[name]
        try:
            from app.dependencies import get_block_instance
            block = get_block_instance("construction")
            params = dict(args.get("params") or {})
            if route == "payment_certificate":
                params = _ipc_args_from_ask(user_message or "", params)
            params["action"] = route
            _alias_input = args.get("input") or args
            if route == "payment_certificate":
                if isinstance(_alias_input, dict):
                    _alias_input = _ipc_args_from_ask(
                        user_message or "", _alias_input,
                    )
                elif not _alias_input:
                    _alias_input = _ipc_args_from_ask(user_message or "", {})
            # F43: container file actions need the same original-name ->
            # stored-path resolution as file-schema blocks (dicts only).
            if project_id and isinstance(_alias_input, dict):
                _alias_input = (await _off_loop(_resolve_block_file_input, project_id, _alias_input))
            if project_id and isinstance(params, dict):
                params = (await _off_loop(_resolve_block_file_input, project_id, params))
                params["action"] = route
            result = await block.process(_alias_input, params)
        except Exception as e:  # noqa: BLE001 — a tool error is a payload, not a crash
            result = {"status": "error", "error": f"{route} failed: {e}"}
        payload = result if isinstance(result, dict) else {"result": result}
        return {
            "name": name,
            "ok": isinstance(result, dict) and result.get("status") != "error",
            "result": {"aliased_to": f"construction:{route}", **payload},
        }

    if name not in BLOCK_REGISTRY:
        return {
            "name": name,
            "ok": False,
            "result": {
                "status": "error",
                "error": f"Unknown block: {name}",
                "hint": "Choose a tool from the provided tool list.",
            },
        }
    if name not in self.allowed_blocks:
        return {
            "name": name,
            "ok": False,
            "result": {
                "status": "error",
                "error": f"Block '{name}' not in agent's allowed_blocks.",
                "hint": "This tool is not available to you; choose another.",
            },
        }
    from app.core.privileges import caller_may_use_block, privileged_forbidden_detail
    if not caller_may_use_block(name):
        return {
            "name": name,
            "ok": False,
            "result": {
                "status": "error",
                "error": privileged_forbidden_detail(name),
                "hint": "This tool is not available to this user; answer without it.",
            },
        }

    instance = block_instances.get(name) or _create_block_instance(name)
    # File-consuming blocks get an explicit `file_path` schema (see
    # _FILE_TOOL_SCHEMAS above in tool_definitions). When the LLM
    # responds to that schema it emits `{"file_path": "<name>"}` at
    # the top level — NOT nested under input/params. Detect that
    # shape and synthesize the block's expected envelope.
    if name in _FILE_CONSUMING_BLOCKS and "file_path" in args and "input" not in args:
        block_input = {"file_path": args.get("file_path")}
        block_params = {k: v for k, v in args.items() if k != "file_path"}
    else:
        block_input = args.get("input")
        block_params = args.get("params") or {}
    # A block that needs its arguments shaped has an adapter in its
    # owner's package (tool_registry.block_args); it may answer itself.
    _adapter = _tool_registry.get_block_args(name)
    if _adapter is not None:
        _adapted = await _adapter.adapt(_tool_registry.ToolCall(
            agent=self, name=name, args=args, tool_call=tool_call, api_key=api_key,
            project_id=project_id, conversation_id=conversation_id, depth=_depth,
            call_stack=_call_stack, user_message=user_message, history=history,
        ), block_input, block_params)
        if isinstance(_adapted, dict):
            return _adapted
        block_input, block_params = _adapted
    # File-consuming blocks: the LLM typically supplies just the filename
    # (e.g. 'Infra-1 - Demolition BOQ.pdf') because that is what the
    # user said. The block then calls os.path.exists on a bare filename
    # which always fails on the deployed disk, producing
    # 'File not found: <name>'. Resolve to the absolute file_path of the
    # uploaded document for this project before dispatch.
    if name in _FILE_CONSUMING_BLOCKS and project_id:
        block_input = (await _off_loop(_resolve_block_file_input, project_id, block_input))
        block_params = (await _off_loop(_resolve_block_file_input, project_id, block_params))
    try:
        result = await instance.execute(block_input, block_params)
        envelope = {"name": name, "ok": True, "result": result}
        await _auto_validate(envelope)
        return envelope
    except Exception as e:
        return {
            "name": name,
            "ok": False,
            "result": {
                "status": "error",
                "error": str(e),
                "hint": "The tool failed; retry with different input or proceed without it.",
            },
        }
