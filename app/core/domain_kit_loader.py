"""Load domain kits from env + installed registry into BLOCK_REGISTRY specs."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Dict, List, Tuple

from app.core.domain_kit_registry import enabled_kit_ids, load_registry

logger = logging.getLogger(__name__)

# Blocks bundled with each store kit (must match manifest.json blocks list).
_KIT_BLOCK_SPECS: Dict[str, List[Tuple[str, str, str]]] = {
    "construction": [
        ("construction", "app.containers.construction", "ConstructionContainer"),
        ("construction_advisor", "app.blocks.construction_advisor", "ConstructionAdvisorBlock"),
        ("boq_processor", "app.blocks.boq_processor", "BOQProcessorBlock"),
        ("bim", "app.blocks.bim", "BIMBlock"),
        ("bim_extractor", "app.blocks.bim_extractor", "BIMExtractorBlock"),
        ("drawing_qto", "app.blocks.drawing_qto", "DrawingQTOBlock"),
        ("primavera_parser", "app.blocks.primavera_parser", "PrimaveraParserBlock"),
        ("spec_analyzer", "app.blocks.spec_analyzer", "SpecAnalyzerBlock"),
        ("formula_executor_v2", "app.blocks.formula_executor_v2", "FormulaExecutorV2Block"),
        ("project_reasoner", "app.blocks.project_reasoner", "ProjectReasonerBlock"),
        ("sympy_reasoning", "app.blocks.sympy_reasoning", "SymPyReasoningBlock"),
        ("smart_orchestrator", "app.blocks.smart_orchestrator", "SmartOrchestratorBlock"),
        ("project_dashboard", "app.blocks.project_dashboard", "ProjectDashboardBlock"),
        ("learning_engine", "app.blocks.learning_engine", "LearningEngineBlock"),
        ("recommendation_template", "app.blocks.recommendation_template", "RecommendationTemplateBlock"),
        ("historical_benchmark", "app.blocks.historical_benchmark", "HistoricalBenchmarkBlock"),
        # L2 Schedule Engine roadmap-named shims (delegate to existing schedule code)
        ("scope_extractor", "app.blocks.scope_extractor", "ScopeExtractorBlock"),
        ("schedule_generator", "app.blocks.schedule_generator", "ScheduleGeneratorBlock"),
        ("cpm_engine", "app.blocks.cpm_engine", "CPMEngineBlock"),
        ("manpower_planner", "app.blocks.manpower_planner", "ManpowerPlannerBlock"),
        ("fasttrack_analyzer", "app.blocks.fasttrack_analyzer", "FastTrackAnalyzerBlock"),
        ("schedule_excel_writer", "app.blocks.schedule_excel_writer", "ScheduleExcelWriterBlock"),
    ],
}


def _env_kit_ids() -> list[str]:
    raw = os.getenv("CEREBRUM_DOMAIN_KITS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def active_kit_ids() -> list[str]:
    """Union of CEREBRUM_DOMAIN_KITS env and persisted install registry."""
    ids = set(_env_kit_ids())
    ids.update(enabled_kit_ids())
    return sorted(ids)


def is_virgin_boot() -> bool:
    """Virgin Fork: generic blocks only unless a domain kit is enabled."""
    if os.getenv("CEREBRUM_VIRGIN", "true").strip().lower() in ("0", "false", "no"):
        return False
    return len(active_kit_ids()) == 0


def kit_block_specs() -> List[Tuple[str, str, str]]:
    specs: List[Tuple[str, str, str]] = []
    for kit_id in active_kit_ids():
        kit_specs = _KIT_BLOCK_SPECS.get(kit_id)
        if not kit_specs:
            logger.warning("domain kit '%s' has no block spec map — skipped", kit_id)
            continue
        specs.extend(kit_specs)
    return specs


def verify_installed_containers() -> None:
    """Log warnings when registry references a container class that cannot import."""
    for kit_id, record in load_registry().get("kits", {}).items():
        class_path = record.get("container_class", "")
        if not class_path or "." not in class_path:
            continue
        module_path, class_name = class_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
            getattr(mod, class_name)
        except Exception as exc:
            logger.warning(
                "installed kit '%s' container %s failed to import: %s",
                kit_id,
                class_path,
                exc,
            )
