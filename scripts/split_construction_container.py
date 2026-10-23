#!/usr/bin/env python3
"""Mechanically split app/containers/construction.py into a package."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

SRC = Path("app/containers/construction.py")
OUT = Path("app/containers/construction")
FILE_LINE_LIMIT = 1800
BODY_LIMIT = 1785  # placement budget (AST line counts; emitted files are slightly shorter)
METADATA_LINES = 55  # class attrs + docstring moved to documents.py


def body_budget(mod: str) -> int:
    if mod == "init":
        return 1660  # facade methods only; class metadata lives in documents.py
    if mod == "documents":
        return BODY_LIMIT - METADATA_LINES
    return BODY_LIMIT

FACADE = {
    "_resolve_block",
    "route",
    "get_actions",
    "_status",
    "health_check",
    "orchestrate",
    "jetson_dispatch",
    "formula_execute",
    "bim_extract",
    "learn",
}

CHAT = {"chat"}
QTO = {"drawing_qto"}

# Methods that must never be relocated by the balancer.
PINNED: dict[str, str] = {
    "chat": "chat",
    "drawing_qto": "qto",
    "boq_process": "boq",
    "primavera_parse": "schedule",
}

# Preferred home for each method (balancer may move to satisfy MAX_LINES).
PREFERRED = {
    "chat": CHAT,
    "qto": QTO,
    "init": FACADE,
    "schedule": {
        "parse_primavera_schedule",
        "_parse_xer_file",
        "_parse_xml_schedule",
        "_calculate_cpm",
        "_analyze_delays",
        "_analyze_schedule_risks",
        "_generate_recovery_options",
        "_extract_milestones",
        "_generate_schedule_recommendations",
        "_assess_delay_impact",
        "_calculate_duration_days",
        "_calculate_date_diff",
        "analyze_schedule_risk",
        "resource_histogram",
        "_calculate_labor_histogram",
        "_identify_resource_peaks",
        "_identify_resource_conflicts",
        "_suggest_resource_leveling",
        "_breakdown_by_trade",
        "_calculate_cost_histogram",
        "forensic_delay_analysis",
        "_run_time_impact_analysis",
        "_run_windows_analysis",
        "_group_events_into_windows",
        "_run_collapsed_as_built",
        "_run_impacted_as_planned",
        "_analyze_critical_path_changes",
        "_analyze_concurrency",
        "_parse_event_date",
        "_apportion_delay",
        "generate_wbs",
        "_build_wbs_activities",
        "_attach_cpm_to_activities",
        "primavera_parse",
        "_process_schedule",
        "_get_primavera_parser_block",
        "_score_schedule",
        "progress_tracker",
        "warranty_maintenance_schedule",
        "_get_maintenance_tasks",
        "claims_builder",
        "_generate_claim_narrative",
        "_calculate_prolongation_costs",
        "_build_causation_link",
        "_check_eot_entitlement",
        "_identify_concurrent_delays",
        "_list_claim_documents",
        "_compile_evidence_list",
        "_anticipate_defenses",
        "daily_site_report",
        "_fetch_weather",
        "_analyze_site_photo",
        "_extract_activities_from_voice",
        "_extract_location_from_context",
        "_extract_issues_from_voice",
        "_extract_manpower_from_voice",
        "_extract_equipment_from_photos",
        "_generate_daily_narrative",
        "_extract_safety_observations",
        "_extract_quality_observations",
        "_extract_material_deliveries",
        "_generate_next_day_plan",
        "commissioning_checklist",
        "_generate_hvac_commissioning",
        "_generate_electrical_commissioning",
        "_generate_fire_commissioning",
        "_generate_plumbing_commissioning",
        "_generate_elevator_commissioning",
        "_generate_facade_commissioning",
        "_generate_bms_commissioning",
        "_estimate_commissioning_duration",
        "_list_commissioning_docs",
        "_generate_training_requirements",
        "_add_days",
        "_add_weeks",
        "_add_months",
        "_subtract_weeks",
        "_days_between",
    },
    "boq": {
        "boq_process",
        "extract_quantities",
        "estimate_costs",
        "generate_cost_estimate",
        "payment_certificate",
        "procurement_list_generator",
        "_build_procurement_item",
        "_classify_procurement_item",
        "_calculate_order_date",
        "_group_by_category",
        "_generate_procurement_recommendations",
        "procurement_optimizer",
        "_generate_procurement_insights",
        "_identify_consolidation",
        "procurement_analysis",
        "_process_bill_of_materials",
        "_lookup_unit_cost",
        "_get_historical_benchmark_block",
        "_calculate_quantities",
        "_estimate_costs",
        "_estimate_carbon",
        "change_order_impact",
        "_analyze_change_type",
        "_detect_trade_from_text",
        "_calculate_co_cost_impact",
        "variation_order_manager",
        "_categorize_variation",
        "_calculate_variation_price",
        "_calculate_cumulative_variations",
        "_determine_approval_workflow",
        "_extract_variation_clauses",
        "_check_time_bar",
        "_generate_vo_document",
        "_list_vo_documents",
        "_identify_vo_risks",
        "sympy_reason",
        "spec_analyze",
        "benchmark_lookup",
        "recommend",
        "carbon_footprint_calculator",
        "generate_carbon_report",
        "extract_measurements",
        "risk_register_auto_populate",
        "_create_risk_item",
        "rfi_generator",
        "_map_rfi_discipline",
        "submittal_log_generator",
        "_create_submittal_item",
        "_group_submittals_by_type",
        "tender_bid_analysis",
        "_score_price",
        "_analyze_unit_prices",
        "_assess_bidder_risk",
        "_identify_qualification_gaps",
        "_identify_bid_clarifications",
        "_generate_negotiation_strategy",
        "value_engineering",
        "_find_value_engineering_alternatives",
        "_build_ve_scenarios",
        "_select_optimal_scenario",
        "_group_ve_by_category",
        "_generate_ve_roadmap",
        "_identify_ve_approvals",
        "cash_flow_forecast",
        "esg_sustainability_report",
        "_calculate_environmental_metrics",
        "_calculate_social_metrics",
        "_calculate_governance_metrics",
        "_score_environmental",
        "_score_social",
        "_score_governance",
        "_check_certification_eligibility",
        "_map_to_sdgs",
        "_generate_esg_recommendations",
        "_generate_stakeholder_narrative",
        "_suggest_bundling",
        "_identify_procurement_risks",
    },
    "documents": {
        "process_document",
        "_looks_like_file",
        "_get_or_create_cache_key",
        "_classify_document",
        "_process_drawing",
        "_process_drawing_page",
        "_process_report",
        "_process_ifc",
        "_process_site_photo",
        "_download_file",
        "_process_image",
        "_extract_drawing_number",
        "_extract_revision",
        "_calculate_confidence",
        "_analyse_text_only",
        "_process_office_document",
        "qa_qc_inspection",
        "generate_construction_report",
        "_detect_risks_from_drawing",
        "_extract_measurements_advanced",
        "_extract_tables_advanced",
        "_extract_annotations",
        "_extract_specs_advanced",
        "_extract_title_block",
        "_extract_scale",
        "_detect_disciplines",
        "as_built_deviation_report",
        "_compare_as_built_to_design",
        "_compare_measurement_sets",
        "track_progress",
        "_compare_photo_to_bim",
        "_query_bim_location",
        "_parse_defects",
        "_calculate_severity",
        "_check_compliance",
        "_generate_recommendations",
        "auto_pipeline",
        "process_contract",
        "_extract_obligations",
        "_categorize_obligation",
        "_assess_obligation_priority",
        "_assess_contract_risks",
        "_extract_financial_terms",
        "_generate_contract_summary",
        "process_specification_full",
        "_get_spec_analyzer_block",
        "_split_csi_divisions",
        "_topup_keyword_matches",
        "bim_analysis",
        "_get_bim_extractor_block",
        "bim_clash_detection",
        "_normalize_block_clash",
        "_group_clashes_by_discipline",
        "_generate_coordination_agenda",
        "safety_compliance_audit",
        "_analyze_safety_photo",
        "_parse_safety_hazards",
        "_generate_safety_recommendations",
        "intelligent_workflow",
        "_build_intelligent_chain",
        "_suggest_next_action",
        "_extract_key_findings",
        "_consolidate_results",
        "om_manual_generator",
        "_add_years_str",
        "_group_equipment_by_system",
        "_map_system_dependencies",
        "_generate_equipment_maintenance",
        "_generate_startup_procedures",
        "_generate_normal_operation",
        "_generate_shutdown_procedures",
        "_generate_emergency_procedures",
        "_generate_seasonal_operation",
        "_generate_troubleshooting_guide",
        "_generate_spare_parts_list",
        "_extract_training_needs",
        "_create_maintenance_matrix",
        "_generate_daily_tasks",
        "_generate_weekly_tasks",
        "_generate_monthly_tasks",
        "_generate_quarterly_tasks",
        "_generate_annual_tasks",
        "digital_twin_sync",
        "_get_platform_config",
        "_generate_initial_sync_operations",
        "_generate_update_operations",
        "_generate_delta_operations",
        "_check_twin_data_quality",
        "_transform_for_platform",
        "_generate_api_payloads",
        "_generate_sync_recommendations",
        "_detect_project_type_from_brief",
        "analyze_spec_section",
        "progress_tracking",
        "qa_inspection",
        "_process_specification",
    },
}

INIT_ATTRS = {
    "name",
    "version",
    "description",
    "layer",
    "tags",
    "requires",
    "default_config",
    "ui_schema",
}

MODULE_IMPORTS = {
    "chat": textwrap.dedent(
        """\
        import logging
        from typing import Any, Dict

        logger = logging.getLogger(__name__)
        """
    ),
    "qto": textwrap.dedent(
        """\
        from typing import Any, Dict
        """
    ),
    "boq": textwrap.dedent(
        """\
        import logging
        import re
        from datetime import datetime, timedelta, timezone
        from typing import Any, Dict, List, Optional, Tuple

        from app.core.construction_types import Measurement, SpecItem, RiskItem

        from .helpers import _parse_money_str, _safe_float, _safe_iso_date

        logger = logging.getLogger(__name__)
        """
    ),
    "schedule": textwrap.dedent(
        """\
        import logging
        import math
        import uuid
        from datetime import datetime, timedelta, timezone
        from typing import Any, Dict, List, Optional, Tuple

        from .helpers import _safe_float, _safe_iso_date

        logger = logging.getLogger(__name__)
        """
    ),
    "documents": textwrap.dedent(
        """\
        import logging
        import os
        import re
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        from typing import Any, Dict, List, Optional, Tuple

        from app.core.construction_types import Measurement, SpecItem, RiskItem

        from .helpers import _parse_money_str, _safe_float, _safe_iso_date

        logger = logging.getLogger(__name__)
        """
    ),
    "init": textwrap.dedent(
        """\
        import logging
        from typing import Any, Dict, List, Optional, Tuple

        from app.core.universal_base import UniversalContainer

        from .boq import ConstructionBoqMixin
        from .chat import ConstructionChatMixin
        from .documents import ConstructionDocumentsMixin
        from .qto import ConstructionQtoMixin
        from .schedule import ConstructionScheduleMixin

        logger = logging.getLogger(__name__)
        """
    ),
}

MIXIN_NAMES = {
    "chat": "ConstructionChatMixin",
    "qto": "ConstructionQtoMixin",
    "boq": "ConstructionBoqMixin",
    "schedule": "ConstructionScheduleMixin",
    "documents": "ConstructionDocumentsMixin",
}


def preferred_module(name: str) -> str:
    for mod, names in PREFERRED.items():
        if name in names:
            return mod
    raise KeyError(name)


def balance_assignments(sizes: dict[str, int]) -> dict[str, str]:
    """Assign every method to a module; keep each pool under its line budget."""
    assignment: dict[str, str] = {}
    if "_WBS_TEMPLATES" in sizes:
        assignment["_WBS_TEMPLATES"] = "schedule"
    if "_DEFECT_KEYWORDS" in sizes:
        assignment["_DEFECT_KEYWORDS"] = "documents"

    load = {m: 0 for m in ("boq", "schedule", "documents", "chat", "qto", "init")}

    def cap(mod: str) -> int:
        return body_budget(mod)

    def place(name: str, mod: str) -> None:
        assignment[name] = mod
        load[mod] += sizes[name]

    fixed: list[tuple[str, str]] = []
    pending: list[str] = []
    for name in sizes:
        if name in assignment:
            fixed.append((name, assignment[name]))
            continue
        if name in PINNED:
            fixed.append((name, PINNED[name]))
        elif name in CHAT:
            fixed.append((name, "chat"))
        elif name in QTO:
            fixed.append((name, "qto"))
        elif name in FACADE:
            fixed.append((name, "init"))
        else:
            pending.append(name)

    for name, mod in fixed:
        place(name, mod)

    pending.sort(key=lambda n: sizes[n], reverse=True)
    for name in pending:
        pref = preferred_module(name)
        order = [pref] + sorted(
            ("boq", "schedule", "documents", "init"),
            key=lambda m: load[m],
        )
        for mod in order:
            if load[mod] + sizes[name] <= cap(mod):
                place(name, mod)
                break
        else:
            raise SystemExit(
                f"cannot place {name} ({sizes[name]} lines); loads={dict(load)}"
            )

    # Keep chat/qto thin — only canonical actions belong there.
    for name, assigned in list(assignment.items()):
        if name in PINNED:
            assignment[name] = PINNED[name]
            continue
        if assigned == "chat" and name not in CHAT:
            assignment[name] = "init"
        elif assigned == "qto" and name not in QTO:
            assignment[name] = "init"

    load = {m: 0 for m in ("boq", "schedule", "documents", "chat", "qto", "init")}
    for name, mod in assignment.items():
        load[mod] += sizes[name]

    # Rebalance among the four implementation modules (never chat/qto).
    pools = ("boq", "schedule", "documents", "init")
    for _ in range(200):
        changed = False
        for mod in pools:
            while load[mod] > cap(mod):
                movable = [
                    n
                    for n, m in assignment.items()
                    if m == mod and n not in FACADE and n not in PINNED
                    and n not in {"_WBS_TEMPLATES", "_DEFECT_KEYWORDS"}
                ]
                movable.sort(key=lambda n: sizes[n])
                moved = False
                for name in movable:
                    for target in pools:
                        if target == mod:
                            continue
                        if load[target] + sizes[name] <= cap(target):
                            load[mod] -= sizes[name]
                            load[target] += sizes[name]
                            assignment[name] = target
                            moved = True
                            changed = True
                            break
                    if moved:
                        break
                if not moved:
                    raise SystemExit(
                        f"cannot rebalance {mod}: {load[mod]} lines (max {cap(mod)})"
                    )
        if not changed:
            break

    return assignment


def main() -> None:
    src_text = SRC.read_text()
    lines = src_text.splitlines()
    tree = ast.parse(src_text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ConstructionContainer")

    helpers_block = "\n".join(lines[20:107]).rstrip() + "\n"

    sizes: dict[str, int] = {}
    nodes_by_name: dict[str, ast.AST] = {}
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sizes[node.name] = node.end_lineno - node.lineno + 1
            nodes_by_name[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            sizes[node.target.id] = node.end_lineno - node.lineno + 1
            nodes_by_name[node.target.id] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_DEFECT_KEYWORDS":
                    sizes["_DEFECT_KEYWORDS"] = node.end_lineno - node.lineno + 1
                    nodes_by_name["_DEFECT_KEYWORDS"] = node

    assignment = balance_assignments(sizes)

    buckets: dict[str, list[str]] = {k: [] for k in ["chat", "qto", "boq", "schedule", "documents", "init"]}

    for node in cls.body:
        chunk = ast.get_source_segment(src_text, node) or "\n".join(
            lines[node.lineno - 1 : node.end_lineno]
        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            buckets[assignment[node.name]].append(chunk)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n in INIT_ATTRS for n in names):
                buckets["documents"].insert(0, chunk)
            elif any(n == "_DEFECT_KEYWORDS" for n in names):
                buckets["documents"].append(chunk)
            else:
                raise ValueError(f"unassigned attribute: {names}")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            buckets[assignment[node.target.id]].append(chunk)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            buckets["documents"].insert(0, chunk)
        else:
            raise ValueError(f"unhandled node: {type(node)}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "helpers.py").write_text(
        '"""Shared helpers for the construction container package."""\n\n'
        "import re\n"
        "from datetime import datetime\n"
        "from typing import Any, Optional\n\n"
        + helpers_block
    )

    def _normalize_method_chunk(chunk: str) -> str:
        """Class methods in the source are indented 4 spaces; keep one level."""
        lines = chunk.splitlines()
        if lines and lines[0].startswith("    ") and not lines[0].startswith("        "):
            return "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
        return chunk

    for mod in ("chat", "qto", "boq", "schedule", "documents"):
        mixin = MIXIN_NAMES[mod]
        body = "\n".join(_normalize_method_chunk(c) for c in buckets[mod])
        content = (
            f'"""Construction container — {mod} submodule."""\n\n'
            + MODULE_IMPORTS[mod]
            + f"\n\nclass {mixin}:\n"
            + textwrap.indent(body, "    ")
            + "\n"
        )
        path = OUT / f"{mod}.py"
        path.write_text(content)
        n = len(content.splitlines())
        if n > 1800:
            raise SystemExit(f"{path} has {n} lines (>1800)")

    init_body = "\n".join(_normalize_method_chunk(c) for c in buckets["init"])
    init_content = (
        '"""Construction Container - Full AEC Industry Domain Container v3.1"""\n\n'
        + MODULE_IMPORTS["init"]
        + "\n\nclass ConstructionContainer(\n"
        + "    ConstructionDocumentsMixin,\n"
        + "    ConstructionBoqMixin,\n"
        + "    ConstructionScheduleMixin,\n"
        + "    ConstructionChatMixin,\n"
        + "    ConstructionQtoMixin,\n"
        + "    UniversalContainer,\n"
        + "):\n"
        + textwrap.indent(init_body, "    ")
        + "\n"
    )
    init_path = OUT / "__init__.py"
    init_path.write_text(init_content)
    if len(init_content.splitlines()) > 1800:
        raise SystemExit(f"{init_path} has {len(init_content.splitlines())} lines (>1800)")

    print("Split complete:")
    for p in sorted(OUT.glob("*.py")):
        print(f"  {p}: {len(p.read_text().splitlines())} lines")


if __name__ == "__main__":
    main()
