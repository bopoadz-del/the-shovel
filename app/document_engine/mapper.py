"""Document Mapper — Layer 3 pragmatic structuring."""
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class StructuredDocument:
    project_metadata: Dict[str, Any] = field(default_factory=dict)
    glossary: Dict[str, str] = field(default_factory=dict)
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    schedule_targets: List[Dict[str, Any]] = field(default_factory=list)
    equipment_specs: List[Dict[str, Any]] = field(default_factory=list)
    wbs_mapping: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    risks: List[Dict[str, Any]] = field(default_factory=list)
    downstream: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), sort_keys=False, allow_unicode=True)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentMapper:
    """Transform reasoned output into downstream-consumable structure."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ontology = config.get("ontology", {})
        self.wbs_dict = config.get("wbs_dictionary", {})

    def map_to_structured(self, reasoned: Any) -> StructuredDocument:
        doc = StructuredDocument()

        doc.project_metadata = {
            "generated_at": datetime.now().isoformat(),
            "engine_version": self.config.get("version", "1.0.0"),
            "engine_name": self.config.get("name", "Document Reasoning Engine"),
        }

        doc.glossary = getattr(reasoned, "glossary", {})
        doc.requirements = getattr(reasoned, "requirements", [])
        doc.constraints = getattr(reasoned, "constraints", [])
        doc.schedule_targets = getattr(reasoned, "schedule_targets", [])
        doc.equipment_specs = getattr(reasoned, "equipment_specs", [])
        doc.wbs_mapping = getattr(reasoned, "wbs_mapping", {})
        doc.risks = getattr(reasoned, "risks", [])

        doc.downstream = self._build_downstream(reasoned)
        return doc

    def _build_downstream(self, reasoned: Any) -> Dict[str, Any]:
        downstream = {
            "schedule_engine": {},
            "cost_engine": {},
            "risk_engine": {},
        }

        # ------------------------------------------------------------------
        # Schedule Engine Feed
        # ------------------------------------------------------------------
        activities = []
        for spec in getattr(reasoned, "equipment_specs", []):
            equip = spec.get("equipment", "").lower()
            lead_time = spec.get("lead_time_days", 0)
            wbs = self.ontology.get(equip, {}).get("wbs", "3.0")
            activities.append({
                "activity_name": f"Procure {spec.get('equipment', 'Equipment')}",
                "wbs_code": wbs,
                "duration": lead_time,
                "predecessor": "2.0 Design Complete",
                "type": "procurement",
            })
        downstream["schedule_engine"]["procurement_activities"] = activities

        milestones = []
        for target in getattr(reasoned, "schedule_targets", []):
            year = target.get("year", "")
            month = target.get("month", "")
            if year and month:
                try:
                    date_obj = datetime.strptime(f"{month} {year}", "%B %Y")
                    milestones.append({
                        "name": target.get("context", "Milestone")[:80],
                        "target_date": date_obj.strftime("%Y-%m-%d"),
                    })
                except ValueError:
                    logger.warning(
                        "swallowed %s in _build_downstream() — continuing",
                        "ValueError", exc_info=True,
                    )
        downstream["schedule_engine"]["milestones"] = milestones

        # ------------------------------------------------------------------
        # Cost Engine Feed
        # ------------------------------------------------------------------
        cost_buckets = {}
        for code, items in getattr(reasoned, "wbs_mapping", {}).items():
            cost_buckets[code] = {
                "wbs_description": self.wbs_dict.get(code, ""),
                "item_count": len(items),
                "categories": list({i.get("category", "unknown") for i in items}),
            }
        downstream["cost_engine"]["wbs_buckets"] = cost_buckets

        # ------------------------------------------------------------------
        # Risk Engine Feed
        # ------------------------------------------------------------------
        downstream["risk_engine"]["identified_risks"] = [
            {
                "category": r.get("type", "unknown"),
                "description": r.get("context", "")[:200],
                "severity": r.get("severity", "medium"),
                "keyword": r.get("keyword", ""),
            }
            for r in getattr(reasoned, "risks", [])
        ]

        return downstream
