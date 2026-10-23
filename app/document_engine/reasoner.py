"""Document Reasoner — Layer 2 semantic extraction (the brain)."""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ReasonedOutput:
    glossary: Dict[str, str] = field(default_factory=dict)
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    schedule_targets: List[Dict[str, Any]] = field(default_factory=list)
    equipment_specs: List[Dict[str, Any]] = field(default_factory=list)
    diagrams: List[Dict[str, Any]] = field(default_factory=list)
    wbs_mapping: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    risks: List[Dict[str, Any]] = field(default_factory=list)


class DocumentReasoner:
    """8 reasoning pipelines over tagged document chunks."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.patterns = config.get("patterns", {})
        self.ontology = config.get("ontology", {})
        self.wbs_dict = config.get("wbs_dictionary", {})

    def reason(self, documents: List[Any]) -> ReasonedOutput:
        output = ReasonedOutput()

        # Aggregate corpus
        all_text = ""
        all_tables: List[List[List[str]]] = []
        all_figures: List[Dict[str, Any]] = []

        for doc in documents:
            if hasattr(doc, "text"):
                all_text += doc.text + "\n"
            if hasattr(doc, "tables"):
                all_tables.extend(doc.tables)
            if hasattr(doc, "figures"):
                all_figures.extend(doc.figures)
            if hasattr(doc, "glossary"):
                output.glossary.update(doc.glossary)

        # Merge table text into corpus for richer extraction
        for table in all_tables:
            for row in table:
                all_text += " | ".join(str(c) for c in row if c) + "\n"

        # Run 8 pipelines
        output.requirements = self._extract_requirements(all_text)
        output.constraints = self._extract_constraints(all_text)
        output.schedule_targets = self._extract_schedule_targets(all_text)
        output.equipment_specs = self._extract_equipment_specs(all_text)
        output.diagrams = self._interpret_diagrams(all_figures, all_text)
        output.wbs_mapping = self._map_wbs(all_text, output.requirements, output.constraints)
        output.risks = self._identify_risks(all_text, output.constraints, output.schedule_targets)

        return output

    # ------------------------------------------------------------------
    # Pipeline 1: Glossary (already populated by PDF parser, augmented here)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pipeline 2: Requirement Mapping
    # ------------------------------------------------------------------
    def _extract_requirements(self, text: str) -> List[Dict[str, Any]]:
        reqs = []
        obligation_patterns = self.patterns.get(
            "obligation", [r"\b(SHALL|MUST|REQUIRED|WILL BE REQUIRED)\b"]
        )

        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            if len(sent) < 10:
                continue
            for pattern in obligation_patterns:
                match = re.search(pattern, sent, re.IGNORECASE)
                if match:
                    category = self._categorize_requirement(sent)
                    reqs.append({
                        "text": sent.strip(),
                        "category": category,
                        "obligation_type": match.group(1).upper(),
                    })
                    break
        return reqs

    def _categorize_requirement(self, text: str) -> str:
        text_lower = text.lower()
        if any(k in text_lower for k in ["electrical", "power", "voltage", "mw", "kw", "generator", "switchgear"]):
            return "electrical"
        elif any(k in text_lower for k in ["mechanical", "chiller", "cooling", "pcw", "tcs", "cfm", "crac"]):
            return "mechanical"
        elif any(k in text_lower for k in ["security", "access", "badge", "layer", "cctv"]):
            return "security"
        elif any(k in text_lower for k in ["it", "network", "fiber", "cable", "anr", "rack"]):
            return "it"
        elif any(k in text_lower for k in ["schedule", "milestone", "date", "completion", "rfs"]):
            return "schedule"
        else:
            return "general"

    # ------------------------------------------------------------------
    # Pipeline 3: Constraint Extraction
    # ------------------------------------------------------------------
    def _extract_constraints(self, text: str) -> List[Dict[str, Any]]:
        constraints = []
        constraint_patterns = self.patterns.get("constraint", [])
        for pattern in constraint_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groupdict()
                constraints.append({
                    "raw": match.group(0),
                    "value": groups.get("value", ""),
                    "unit": groups.get("unit", ""),
                    "operator": groups.get("operator", ""),
                    "context": text[max(0, match.start() - 60) : match.end() + 60].strip(),
                })
        return constraints

    # ------------------------------------------------------------------
    # Pipeline 4: Schedule Targets
    # ------------------------------------------------------------------
    def _extract_schedule_targets(self, text: str) -> List[Dict[str, Any]]:
        targets = []
        schedule_patterns = self.patterns.get("schedule_target", [])
        for pattern in schedule_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groupdict()
                target = {
                    "raw": match.group(0),
                    "month": groups.get("month") or "",
                    "year": groups.get("year") or "",
                    "context": text[max(0, match.start() - 80) : match.end() + 80].strip(),
                }
                # Duration-shaped target ("completion within 14 months") --
                # normalise to days so the schedule engine can consume it.
                qty, unit = groups.get("qty"), groups.get("unit")
                if qty and unit:
                    per = {"day": 1, "week": 7, "month": 30}[unit.lower().rstrip("s")]
                    target["duration_days"] = int(qty) * per
                    target["duration_raw"] = f"{qty} {unit}"
                targets.append(target)
        return targets

    # ------------------------------------------------------------------
    # Pipeline 5: Equipment Specs
    # ------------------------------------------------------------------
    def _extract_equipment_specs(self, text: str) -> List[Dict[str, Any]]:
        specs = []
        equip_patterns = self.patterns.get("equipment_lead_time", [])
        for pattern in equip_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groupdict()
                equip_name = groups.get("equipment", "unknown")
                try:
                    if groups.get("weeks"):
                        days = int(groups["weeks"]) * 7
                    else:
                        days = int(groups.get("days") or 0)
                except ValueError:
                    days = 0
                specs.append({
                    "equipment": equip_name.capitalize(),
                    "lead_time_days": days,
                    "context": text[max(0, match.start() - 50) : match.end() + 50].strip(),
                })

        # Augment with ontology defaults for missing equipment
        found_names = {s["equipment"].lower() for s in specs}
        for term, meta in self.ontology.items():
            if meta.get("type") == "equipment" and term.lower() not in found_names:
                specs.append({
                    "equipment": term.capitalize(),
                    "lead_time_days": meta.get("lead_time", 0),
                    "source": "ontology_default",
                })
        return specs

    # ------------------------------------------------------------------
    # Pipeline 6: Diagram Interpretation
    # ------------------------------------------------------------------
    def _interpret_diagrams(self, figures: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        diagrams = []
        tier_patterns = [
            (r"Tier\s+(\d+)", "tier"),
            (r"Layer\s+(\d+)", "layer"),
            (r"Tier\s+(\d+)\s*[=:]\s*(.+?)(?:\n|$)", "tier_definition"),
        ]

        for fig in figures:
            diagram = {
                "figure_id": fig.get("id"),
                "caption": fig.get("caption"),
                "tiers": [],
                "layers": [],
            }
            caption = fig.get("caption", "")
            for pattern, dtype in tier_patterns:
                for match in re.finditer(pattern, caption, re.IGNORECASE):
                    if dtype == "tier_definition":
                        diagram["tiers"].append({
                            "tier": match.group(1),
                            "description": match.group(2).strip(),
                        })
                    else:
                        diagram["tiers"].append({"tier": match.group(1)})
            diagrams.append(diagram)

        # Section / cross-reference detection
        section_pattern = r"(?:Section|§)\s+(\d+(?:\.\d+)*)"
        for match in re.finditer(section_pattern, text):
            diagrams.append({
                "type": "section_reference",
                "section": match.group(1),
                "context": text[max(0, match.start() - 30) : match.end() + 30].strip(),
            })

        return diagrams

    # ------------------------------------------------------------------
    # Pipeline 7: WBS Mapping
    # ------------------------------------------------------------------
    def _map_wbs(self, text: str, requirements: List[Dict[str, Any]], constraints: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        mapping: Dict[str, List[Dict[str, Any]]] = {code: [] for code in self.wbs_dict.keys()}

        # Map requirements → WBS by category
        for req in requirements:
            category = req["category"]
            target_code = self._wbs_code_for_category(category)
            if target_code:
                # setdefault: a config whose wbs_dictionary lacks the target
                # code must degrade to an extra mapping bucket, not a
                # KeyError that kills the whole reasoning pass (found by the
                # W4 planted-RFP probe with a minimal config).
                mapping.setdefault(target_code, []).append({
                    "type": "requirement",
                    "text": req["text"][:200],
                    "category": category,
                })

        # Map ontology terms → WBS
        for term, meta in self.ontology.items():
            wbs = meta.get("wbs")
            if wbs and wbs in mapping:
                mapping[wbs].append({
                    "type": "ontology_term",
                    "term": term,
                    "category": meta.get("category", "unknown"),
                })

        # Map constraints → WBS by unit type
        for con in constraints:
            unit = con.get("unit", "").lower()
            target_code = None
            if unit in ["mw", "kw", "v", "%"]:
                target_code = "7.0"  # MEP Systems
            elif unit in ["ft"]:
                target_code = "5.0"  # Building Construction
            if target_code:
                mapping.setdefault(target_code, []).append({
                    "type": "constraint",
                    "raw": con.get("raw", ""),
                    "unit": unit,
                })

        return mapping

    def _wbs_code_for_category(self, category: str) -> Optional[str]:
        mapping = {
            "electrical": "7.0",
            "mechanical": "7.0",
            "it": "8.0",
            "security": "9.0",
            "schedule": "1.0",
            "general": "2.0",
        }
        return mapping.get(category.lower())

    # ------------------------------------------------------------------
    # Pipeline 8: Risk Identification
    # ------------------------------------------------------------------
    def _identify_risks(self, text: str, constraints: List[Dict[str, Any]], schedule_targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        risks = []
        risk_patterns = self.patterns.get(
            "risk",
            [r"\b(winter|supply chain|Indigenous|TFO|blast radius|load.shed|permit delay|weather|labour shortage)\b"],
        )

        for pattern in risk_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                keyword = match.group(0)
                risks.append({
                    "type": "keyword_match",
                    "keyword": keyword,
                    "severity": self._infer_severity(keyword, text),
                    "context": text[max(0, match.start() - 80) : match.end() + 80].strip(),
                })

        # Risk from external schedule dependencies
        for target in schedule_targets:
            ctx = target.get("context", "").lower()
            if any(k in ctx for k in ["tfo", "permit", "indigenous", "supply"]):
                risks.append({
                    "type": "schedule_dependency",
                    "keyword": "external_dependency",
                    "severity": "high",
                    "context": target["context"],
                })

        # Risk from tight constraints (≤3% blast radius → high risk)
        for con in constraints:
            op = con.get("operator", "")
            val = con.get("value", "")
            unit = con.get("unit", "").lower()
            try:
                numeric = float(val)
            except ValueError:
                continue
            if unit == "%" and op in ["≤", "<"] and numeric <= 5:
                risks.append({
                    "type": "tight_constraint",
                    "keyword": f"{op}{numeric}%",
                    "severity": "high",
                    "context": con.get("context", ""),
                })
            if unit == "mw" and numeric >= 50:
                risks.append({
                    "type": "scale_constraint",
                    "keyword": f"{numeric} MW",
                    "severity": "medium",
                    "context": con.get("context", ""),
                })

        return risks

    def _infer_severity(self, keyword: str, context: str) -> str:
        high = ["blast radius", "winter", "supply chain", "permit delay", "labour shortage"]
        medium = ["tfo", "indigenous", "load shed", "weather"]
        klow = keyword.lower()
        if any(h in klow for h in high):
            return "high"
        if any(m in klow for m in medium):
            return "medium"
        return "low"
