"""Construction Container - Full AEC Industry Domain Container v3.1"""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.universal_base import UniversalContainer

from .boq import ConstructionBoqMixin
from .chat import ConstructionChatMixin
from .documents import ConstructionDocumentsMixin
from .qto import ConstructionQtoMixin
from .schedule import ConstructionScheduleMixin

logger = logging.getLogger(__name__)


class ConstructionContainer(
    ConstructionDocumentsMixin,
    ConstructionBoqMixin,
    ConstructionScheduleMixin,
    ConstructionChatMixin,
    ConstructionQtoMixin,
    UniversalContainer,
):
    def _looks_like_file(self, input_data: Any, params: Dict) -> bool:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        return any(k in data or k in p for k in ["file_path", "content", "filename", "file", "url"])
    async def _get_or_create_cache_key(self, file_path: str, doc_type: str) -> str:
        from app.blocks import BLOCK_REGISTRY
        hasher_block = BLOCK_REGISTRY.get("file_hasher")
        if hasher_block and os.path.exists(file_path):
            try:
                hasher_instance = hasher_block()
                hash_result = await hasher_instance.execute(
                    {"file_path": file_path}, {"action": "hash_file"}
                )
                if hash_result.get("status") == "success":
                    return f"construction:doc:{doc_type}:{hash_result.get('sha256', '')}"
            except Exception:
                logger.warning(
                    "swallowed %s in _get_or_create_cache_key() — continuing",
                    "Exception", exc_info=True,
                )
        if os.path.exists(file_path):
            return f"construction:doc:{doc_type}:{os.path.getmtime(file_path)}:{os.path.getsize(file_path)}"
        import hashlib
        path_hash = hashlib.md5(str(file_path).encode(), usedforsecurity=False).hexdigest()
        return f"construction:doc:{doc_type}:missing:{path_hash}"
    async def _classify_document(self, file_path: str) -> str:
        name = Path(file_path).name.lower()
        if any(x in name for x in [".ifc", ".bim", "model"]):
            return "bim"
        if any(x in name for x in [".xer", ".xml", "schedule", "primavera", "p6"]):
            return "schedule"
        if any(x in name for x in ["contract", "agreement", "terms", "conditions", "legal"]):
            return "contract"
        if any(x in name for x in ["spec", "specification", "masterformat", "csi"]):
            return "specification"
        if any(x in name for x in ["bom", "bill", "materials", "takeoff", "quantity"]):
            return "bom"
        if any(x in name for x in ["report", "inspection", "test", "certificate"]):
            return "report"
        if any(x in name for x in [".jpg", ".png", ".jpeg", "photo", "site", "image"]):
            return "image"
        if any(x in name for x in ["change order", "variation", "vo", "co", "claim"]):
            return "change_order"
        if any(x in name for x in ["safety", "audit", "inspection", "hazard"]):
            return "safety_audit"
        return "drawing"
    def _process_drawing_page(self, page, page_num: int, pre_extracted_text: str = "") -> Dict:
        # Use pre-extracted text if available, otherwise extract from page
        if pre_extracted_text:
            raw_text = pre_extracted_text[:8000]  # Use provided text
            text_dict = None  # No dict structure available from pre-extracted
        else:
            text_dict = page.get_text("dict")
            raw_text = page.get_text()[:8000]
    
        return {
            "page_number": page_num + 1,
            "raw_text": raw_text[:8000],
            "measurements": self._extract_measurements_advanced(raw_text, text_dict or {}),
            "tables": self._extract_tables_advanced(page),
            "annotations": self._extract_annotations(page),
            "specs": self._extract_specs_advanced(raw_text),
            "image_count": len(page.get_images()),
            "rotation": page.rotation,
            "cropbox": [page.cropbox.x0, page.cropbox.y0, page.cropbox.x1, page.cropbox.y1]
        }
    async def _process_report(self, input_data: Any, params: Dict) -> Dict:
        """Parse the report through document_engine so its raw text reaches
        the LLM (the report classifier covers RFI logs, daily/weekly reports,
        progress reports — free-form prose, no specific schema to extract).
        """
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        file_path = (
            data.get("file_path") if isinstance(data, dict) else None
        ) or p.get("file_path") or (input_data if isinstance(input_data, str) else None)
        if not file_path:
            return {"status": "error", "doc_type": "report", "error": "No file_path provided"}
        block = self._resolve_block("document_engine")
        if block is None:
            return {"status": "error", "doc_type": "report", "error": "document_engine block unavailable"}
        ext = os.path.splitext(file_path)[1].lower()
        engine_params = dict(p)
        if ext == ".pdf":
            engine_params["pdf_path"] = file_path
        elif ext in (".docx", ".doc"):
            engine_params["docx_path"] = file_path
        elif ext in (".xlsx", ".xls"):
            engine_params["xlsx_path"] = file_path
        else:
            engine_params["pdf_path"] = file_path  # best-guess fallback
        result = await block.process({}, engine_params)
        if isinstance(result, dict):
            result.setdefault("doc_type", "report")
        return result
    async def _process_ifc(self, input_data: Any, params: Dict) -> Dict:
        # Route to the real BIM analysis pipeline. The dispatcher in process_document
        # places the file path on params["file_path"], which bim_analysis honours.
        return await self.bim_analysis(input_data, params)
    async def _process_site_photo(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        file_path = data.get("file_path") or p.get("file_path")
        return await self._process_image(file_path, p)
    async def _download_file(self, url: str) -> str:
        import uuid
        import httpx
        ext = url.split('?')[0].split('.')[-1] or 'pdf'
        path = f"/tmp/{uuid.uuid4().hex[:8]}_{url.split('/')[-1] or 'download'}.{ext}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                with open(path, "wb") as f:
                    f.write(response.content)
            return path
        except Exception:
            return ""
    async def _process_image(self, file_path: str, params: Dict) -> Dict:
        ocr_block = self.get_dep("ocr")
        if ocr_block:
            try:
                ocr_result = await ocr_block.execute({"file_path": file_path}, {})
                text = ocr_result.get("result", {}).get("text", "")
                measurements = self._extract_measurements_advanced(text, {})
                specs = self._extract_specs_advanced(text)
                from app.core.confidence import assess_extraction_confidence
                image_result = {
                    "status": "success",
                    "file_name": Path(file_path).name,
                    "source": "ocr",
                    "text": text[:2000],
                    "measurements": measurements,
                    "specifications": specs,
                }
                image_result["confidence"] = assess_extraction_confidence(
                    image_result,
                    expected_fields=["text", "measurements", "specifications"],
                    ocr_quality=ocr_result.get("result", {}).get("quality"),
                )
                return image_result
            except Exception as e:
                return {"status": "error", "error": f"Image OCR failed: {str(e)}"}
        return {"status": "error", "error": "OCR block not available for image processing"}
    def _extract_drawing_number(self, filename: str) -> str:
        import re
        m = re.search(r'[A-Z]{1,3}[-_]?\d{3,6}', filename, re.IGNORECASE)
        return m.group(0).upper() if m else ""
    def _extract_revision(self, filename: str) -> str:
        # Single source of truth with the retrieval revision-currency parser
        # (app.core.rag.revision) so a filename yields the SAME revision at
        # ingest and at retrieval — including multi-character revisions
        # ('Rev 10' -> '10', not '1').
        from app.core.rag.revision import revision_token
        return revision_token(filename)
    def _calculate_confidence(self, result: Dict) -> Dict:
        """Measured extraction confidence (Roadmap V2 · Epic 1).

        Derived from real signals — text recovered, field coverage, OCR
        quality — not a hardcoded constant.
        """
        from app.core.confidence import assess_extraction_confidence
        return assess_extraction_confidence(
            result,
            expected_fields=[
                "drawing_number", "revision", "scale", "title_block",
                "detected_disciplines", "measurements", "specifications",
            ],
            ocr_quality=result.get("ocr_quality"),
        )
    def _extract_obligations(self, text: str) -> List[Dict]:
        obligations = []
        obligation_patterns = [
            (r'(?:contractor|builder)[\s\w]{0,50}(?:shall|must|will|agrees to)[\s\w]{0,100}(?:\.)', "contractor_obligation"),
            (r'(?:employer|owner|client)[\s\w]{0,50}(?:shall|must|will|agrees to)[\s\w]{0,100}(?:\.)', "employer_obligation"),
            (r'(?:both parties|each party)[\s\w]{0,50}(?:shall|must|will)[\s\w]{0,100}(?:\.)', "mutual_obligation"),
            (r'(?:architect|engineer|supervisor)[\s\w]{0,50}(?:shall|must|will)[\s\w]{0,100}(?:\.)', "consultant_obligation"),
        ]
        for pattern, obl_type in obligation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                obligations.append({
                    "type": obl_type,
                    "text": match.group(0),
                    "category": self._categorize_obligation(match.group(0)),
                    "priority": self._assess_obligation_priority(match.group(0))
                })
        return obligations[:20]
    def _categorize_obligation(self, text: str) -> str:
        if any(w in text.lower() for w in ["safety", "health", "protect"]):
            return "safety"
        if any(w in text.lower() for w in ["insurance", "indemnif", "liability"]):
            return "risk"
        if any(w in text.lower() for w in ["payment", "invoice", "cost"]):
            return "financial"
        if any(w in text.lower() for w in ["quality", "defect", "warranty", "guarantee"]):
            return "quality"
        if any(w in text.lower() for w in ["time", "schedule", "milestone", "delay", "completion"]):
            return "schedule"
        return "general"
    def _assess_obligation_priority(self, text: str) -> str:
        if any(w in text.lower() for w in ["shall", "must", "required", "mandatory"]):
            return "high"
        if any(w in text.lower() for w in ["will", "agrees to", "responsible for"]):
            return "medium"
        return "low"
    def _assess_contract_risks(self, clauses: Dict, contract_type: str) -> Dict:
        score = 100
        critical = []
        warnings = []
        recommendations = []
    
        if not clauses.get("payment_terms", {}).get("found"):
            score -= 20
            critical.append("Payment terms not clearly defined")
            recommendations.append("Add explicit payment schedule with milestones")
        if not clauses.get("liquidated_damages", {}).get("found"):
            score -= 15
            warnings.append("No liquidated damages clause")
            recommendations.append("Consider adding LDs for late completion protection")
        if not clauses.get("termination", {}).get("found"):
            score -= 10
            warnings.append("Termination clause missing or unclear")
        if not clauses.get("force_majeure", {}).get("found"):
            score -= 10
            warnings.append("No force majeure clause")
            recommendations.append("Add force majeure for unforeseen delays (weather, pandemic, etc.)")
        if not clauses.get("dispute_resolution", {}).get("found"):
            score -= 10
            warnings.append("No dispute resolution mechanism")
    
        risk_level = "low" if score > 80 else "medium" if score > 60 else "high"
        return {"score": max(0, score), "level": risk_level, "critical": critical, "warnings": warnings, "recommendations": recommendations}
    def _extract_financial_terms(self, text: str) -> Dict:
        terms = {}
        value_match = re.search(r'(?:contract (?:value|sum|price|amount)|total)[\s:]*[$\u20ac£]?[\s]*(\d[\d,\.]*)', text, re.IGNORECASE)
        if value_match:
            terms["contract_value"] = value_match.group(1)
        advance_match = re.search(r'(?:advance|mobilization)[\s\w]{0,30}(\d+)%', text, re.IGNORECASE)
        if advance_match:
            terms["advance_payment"] = f"{advance_match.group(1)}%"
        retention_match = re.search(r'(?:retention|retainage)[\s\w]{0,30}(\d+)%', text, re.IGNORECASE)
        if retention_match:
            terms["retention"] = f"{retention_match.group(1)}%"
        currency_match = re.search(r'(?:currency|in|amounts)[\s\w]{0,20}(USD|EUR|GBP|AED|SAR|QAR)', text, re.IGNORECASE)
        if currency_match:
            terms["currency"] = currency_match.group(1)
        return terms
    def _generate_contract_summary(self, clauses: Dict, financial: Dict) -> str:
        summary_parts = []
        if clauses.get("payment_terms", {}).get("found"):
            summary_parts.append("Payment terms defined")
        else:
            summary_parts.append("⚠️ Payment terms unclear")
        if clauses.get("liquidated_damages", {}).get("found"):
            summary_parts.append("LDs apply")
        if financial.get("contract_value"):
            summary_parts.append(f"Value: {financial['contract_value']}")
        return " | ".join(summary_parts)
    def _resolve_block(self, name: str):
        """Resolve a block by name — dependency injection first, then the
        platform singleton (pre-wired with its own deps), then a bare
        class-instantiation as last resort.

        Returning bare class-instantiations is what broke delegation chains
        like construction._process_report → document_engine: the fresh
        document_engine had no `pdf`/`ocr` wired and silently returned
        empty content. Going through `get_block_instance` reuses the
        singleton that the dependency container has already wired with its
        own `requires`.
        """
        block = self.get_dep(name)
        if block is not None:
            return block
        # Prefer the platform's wired singleton.
        try:
            from app.dependencies import get_block_instance
            return get_block_instance(name)
        except KeyError:
            logger.warning(
                "swallowed %s in _resolve_block() — continuing",
                "KeyError", exc_info=True,
            )
        except Exception:
            logger.warning(
                "swallowed %s in _resolve_block() — continuing",
                "Exception", exc_info=True,
            )
        # Last resort — bare class, no deps wired.
        from app.blocks import BLOCK_REGISTRY
        block_cls = BLOCK_REGISTRY.get(name)
        return block_cls() if block_cls else None
    def _get_primavera_parser_block(self):
        """Resolve the primavera_parser block — dependency injection first, registry fallback."""
        return self._resolve_block("primavera_parser")
    def _parse_xml_schedule(self, file_path: str) -> Dict:
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
        
            activities = []
            for activity in root.findall('.//Activity') if root else []:
                act_data = {
                    "id": activity.findtext('ID', ''),
                    "name": activity.findtext('Name', ''),
                    "start": activity.findtext('Start', ''),
                    "finish": activity.findtext('Finish', ''),
                    "duration": activity.findtext('Duration', 0),
                    "total_float": activity.findtext('TotalFloat', 0),
                    "percent_complete": activity.findtext('PercentComplete', 0)
                }
                activities.append(act_data)
        
            return {
                "status": "success",
                "file_type": "xml",
                "activities": activities
            }
        except Exception as e:
            return {"status": "error", "error": f"XML parse failed: {str(e)}"}
    def _calculate_cpm(self, schedule_data: Dict) -> Dict:
        activities = schedule_data.get("activities", [])
    
        if not activities:
            return {"critical_path": [], "project_duration_days": 0, "average_float": 0}
    
        critical_activities = [a for a in activities if a.get("total_float", 999) <= 0.1]
    
        if not critical_activities:
            min_float = min((a.get("total_float", 999) for a in activities), default=0)
            critical_activities = [a for a in activities if a.get("total_float", 999) <= min_float + 0.1]
    
        duration = 0
        if critical_activities:
            try:
                start_dates = [datetime.fromisoformat(a.get("start", "").replace('Z', '+00:00')) for a in critical_activities if a.get("start")]
                finish_dates = [datetime.fromisoformat(a.get("finish", "").replace('Z', '+00:00')) for a in critical_activities if a.get("finish")]
                if start_dates and finish_dates:
                    duration = (max(finish_dates) - min(start_dates)).days
            except Exception:
                duration = sum(a.get("duration", 0) for a in critical_activities) / 8
    
        floats = [a.get("total_float", 0) for a in activities if a.get("total_float", 999) < 999]
        avg_float = sum(floats) / len(floats) if floats else 0
        near_critical = [a for a in schedule_data.get("activities", []) if 0 < a.get("total_float", 999) < 5]
    
        return {
            "critical_path": [a["id"] for a in critical_activities],
            "critical_path_activities": critical_activities,
            "critical_path_duration": duration,
            "critical_count": len(critical_activities),
            "near_critical_count": len(near_critical),
            "near_critical_activities": near_critical[:10],
            "average_float": avg_float,
            "project_duration_days": duration,
            "driving_paths": []
        }
    def _extract_milestones(self, schedule_data: Dict) -> List[Dict]:
        milestones = []
        for act in schedule_data.get("activities", [])[:100]:
            name = act.get("name", "").lower()
            if any(k in name for k in ["milestone", "substantial completion", "practical completion", "handover", "start", "finish"]):
                milestones.append({"id": act.get("id"), "name": act.get("name"), "date": act.get("start") or act.get("finish")})
        return milestones
    def _generate_schedule_recommendations(self, cpm: Dict, delay_analysis: Optional[Dict]) -> List[str]:
        recs = []
        if cpm.get("average_float", 999) < 2:
            recs.append("Schedule is tightly constrained - consider adding buffers")
        if delay_analysis and delay_analysis.get("total_delay_days", 0) > 7:
            recs.append("Significant delays detected - implement recovery plan immediately")
        return recs
    def _assess_delay_impact(self, delays: List[Dict], total_delay: int) -> str:
        return "critical" if total_delay > 14 else "moderate" if total_delay > 7 else "minor"
    def _calculate_duration_days(self, start: str, finish: str) -> int:
        try:
            s = datetime.fromisoformat(start.replace('Z', '+00:00'))
            f = datetime.fromisoformat(finish.replace('Z', '+00:00'))
            return max(0, (f - s).days)
        except Exception:
            return 0
    def _calculate_date_diff(self, date1: str, date2: str) -> int:
        try:
            d1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            d2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
            return max(0, (d2 - d1).days)
        except Exception:
            return 0
    def _get_spec_analyzer_block(self):
        """Resolve the spec_analyzer block — dependency injection first, registry fallback."""
        return self._resolve_block("spec_analyzer")
    @staticmethod
    def _topup_keyword_matches(full_text: str, keywords: list, existing: list = None) -> list:
        """Scan full_text for keyword hits and UNION with block-derived results.

        Restores the coverage of the old _extract_testing_requirements /
        _extract_qaqc helpers, whose bare-word matching was broader than the
        spec_analyzer block's compliance keyword/pattern sets. For each keyword
        found (word-boundary, case-insensitive) produces a useful entry: the
        matched keyword plus a short surrounding-context snippet. Results are
        merged with `existing` (the block's flags) and deduplicated, preserving
        the block's entries first.
        """
        merged = list(existing or [])
        seen = {str(e).strip().lower() for e in merged}
        text = full_text or ""
        for kw in keywords:
            # word-boundary, case-insensitive — \b around the literal keyword
            m = re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE)
            if not m:
                continue
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            snippet = " ".join(text[start:end].split())
            entry = f"{kw}: {snippet}" if snippet else kw
            if entry.strip().lower() not in seen:
                seen.add(entry.strip().lower())
                merged.append(entry)
        return merged
    async def analyze_spec_section(self, input_data: Any, params: Dict) -> Dict:
        return await self.process_specification_full(input_data, params)
    def _get_historical_benchmark_block(self):
        """Resolve the historical_benchmark block — DI first, registry fallback."""
        return self._resolve_block("historical_benchmark")
    # YOLO class -> (severity, hazard_type) for safety_compliance_audit
    _YOLO_SAFETY_SEVERITY = {
        "no_hardhat": ("critical", "no_hardhat"),
        "no_high_vis_vest": ("critical", "no_high_vis_vest"),
        "fall_hazard_unprotected": ("critical", "fall_hazard_unprotected"),
    }
    # YOLO class -> (severity, defect_label) for qa_qc_inspection
    _YOLO_QAQC_DEFECT = {
        "concrete_crack": ("major", "concrete_crack"),
        "concrete_honeycomb": ("major", "concrete_honeycomb"),
        "rebar_exposed_defect": ("major", "rebar_exposed_defect"),
        "bulging_concrete": ("major", "bulging_concrete"),
    }

    def _classes_to_hazards(self, safety_qaqc: List[Dict]) -> List[Dict]:
        """Map fine-tuned-YOLO safety_qaqc output to hazard dicts that match
        the shape _parse_safety_hazards produces. Drops classes that aren't
        in the safety severity map (e.g. concrete defects, rebar inspection)."""
        out = []
        for entry in safety_qaqc or []:
            mapping = self._YOLO_SAFETY_SEVERITY.get(entry.get("class"))
            if not mapping:
                continue
            severity, hazard_type = mapping
            out.append({
                "type": hazard_type,
                "description": f"YOLO: {entry.get('class')}",
                "severity": severity,
                "source": "yolo",
                "confidence": float(entry.get("confidence", 0.0)),
            })
        return out

    def _classes_to_defects(self, safety_qaqc: List[Dict]) -> List[Dict]:
        """Map fine-tuned-YOLO safety_qaqc output to defect dicts matching
        _parse_defects' shape. Drops non-defect classes (PPE, correct rebar)."""
        out = []
        for entry in safety_qaqc or []:
            mapping = self._YOLO_QAQC_DEFECT.get(entry.get("class"))
            if not mapping:
                continue
            severity, label = mapping
            out.append({
                "keyword": entry.get("class"),
                "description": label,
                "severity": severity,
                "source": "yolo",
                "confidence": float(entry.get("confidence", 0.0)),
            })
        return out

    def _parse_safety_hazards(self, text: str) -> List[Dict]:
        hazards = []
        hazard_patterns = [
            (r'miss(?:ing)?\s*(?:PPE|helmet|harness|vest)', 'missing_ppe', 'critical'),
            (r'exposed\s*(?:edge|opening|hole)', 'fall_hazard', 'critical'),
            (r'trip\s*hazard', 'trip_hazard', 'major'),
            (r'(?:no|missing)\s*guardrail', 'missing_guardrail', 'critical'),
            (r'improper\s*storage', 'improper_storage', 'minor'),
        ]
        for pattern, h_type, severity in hazard_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                hazards.append({
                    "type": h_type,
                    "description": match.group(0),
                    "severity": severity,
                    "context": text[max(0, match.start()-30):match.end()+30]
                })
        return hazards
    def _generate_safety_recommendations(self, violations: List[Dict]) -> List[str]:
        if not violations:
            return ["Continue current safety practices", "Document compliance for audit trail"]
    
        recs = []
        types = set(v.get("type") for v in violations)
    
        if "missing_ppe" in types:
            recs.append("Immediate: Enforce mandatory PPE - hard hats, vests, safety boots")
        if "fall_hazard" in types or "missing_guardrail" in types:
            recs.append("Critical: Install guardrails/harnesses before work continues")
        if "trip_hazard" in types:
            recs.append("Clean and organize work area - remove trip hazards")
    
        return recs
    async def _compare_as_built_to_design(
        self, as_built_path: str, design_path: str, tolerance_mm: float
    ):
        """Real dimension comparison: extract measurements from BOTH files via the
        drawing_qto block, then compare like-typed dimensions. Returns None when
        comparable dimensions cannot be extracted from one or both files so the
        caller emits an honest 'needs comparable dimensions' error — NEVER a
        fabricated deviation. (The prior implementation returned a hardcoded
        'Column grid A-1, 18mm' record for every file pair.)"""
        qto = None
        try:
            qto = self._resolve_block("drawing_qto")
        except Exception:  # noqa: BLE001
            qto = None
        if qto is None:
            return None
        try:
            ab = await qto.process({"file_path": as_built_path})
            dg = await qto.process({"file_path": design_path})
        except Exception:  # noqa: BLE001
            return None
        ab_m = (ab or {}).get("measurements") or []
        dg_m = (dg or {}).get("measurements") or []
        if not ab_m or not dg_m:
            return None
        return self._compare_measurement_sets(ab_m, dg_m, tolerance_mm)
    def _group_submittals_by_type(self, submittals: List[Dict]) -> Dict:
        grouped: Dict = {}
        for s in submittals:
            t = s.get("type", "Other")
            grouped.setdefault(t, 0)
            grouped[t] += 1
        return grouped
    def _add_days(self, date_str: str, days: int) -> str:
        try:
            return (
                datetime.strptime(date_str[:10], "%Y-%m-%d") + timedelta(days=days)
            ).strftime("%Y-%m-%d")
        except Exception:
            return date_str
    async def carbon_footprint_calculator(self, input_data: Any, params: Dict) -> Dict:
        """Calculate embodied carbon footprint. Delegates to generate_carbon_report."""
        return await self.generate_carbon_report(input_data, params)

    async def jetson_dispatch(self, input_data: Any, params: Dict) -> Dict:
        """Edge dispatch: validate and stage the job, honestly pending hardware.

        The Orin/Jetson processor has not shipped (docs/EDGE_PORT.md), so no
        executor exists yet. Rather than a bare not-implemented error, accept
        the job shape, validate it, and return the staged dispatch plan with
        ``status: pending_hardware`` — when the board arrives only the
        executor swaps in; callers and tests already see the real contract.
        """
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        task = p.get("task") or data.get("task")
        payload = p.get("payload") or data.get("payload") or {}
        if not task:
            return {
                "status": "error",
                "error": "jetson_dispatch requires a 'task' (e.g. safety_scan, "
                         "photo_inference) to stage an edge job",
            }
        return {
            "status": "pending_hardware",
            "action": "jetson_dispatch",
            "staged_job": {
                "task": task,
                "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
                "target": p.get("target", data.get("target", "jetson-orin")),
            },
            "reason": "edge processor not yet installed — job validated and "
                      "staged, no executor available (see docs/EDGE_PORT.md)",
        }
    def _get_maintenance_tasks(self, system_type: str) -> List[tuple]:
        tasks = {
            "mechanical": [
                ("monthly", "Clean and inspect air filters"),
                ("quarterly", "Service AHUs and FCUs"),
                ("annually", "Full HVAC system service and re-commission"),
            ],
            "electrical": [
                ("monthly", "Inspect electrical panels and check for faults"),
                ("annually", "Thermographic survey of electrical distribution"),
            ],
            "plumbing": [
                ("quarterly", "Test backflow preventers and strainers"),
                ("annually", "Full system flush and legionella risk assessment"),
            ],
            "vertical_transport": [
                ("monthly", "Lift/escalator maintenance contract visit"),
                ("annually", "Full statutory inspection by approved inspector"),
            ],
            "fire_protection": [
                ("monthly", "Test fire alarms and emergency lighting"),
                ("quarterly", "Inspect sprinkler heads and test pumps"),
                ("annually", "Full fire system service and certification"),
            ],
            "waterproofing": [
                ("annually", "Inspect roof membrane and drains"),
                ("5_yearly", "Full waterproofing condition survey"),
            ],
        }
        return tasks.get(system_type, [("annually", "General inspection and service")])
    def _get_bim_extractor_block(self):
        """Resolve the bim_extractor block — dependency injection first, registry fallback."""
        return self._resolve_block("bim_extractor")
    async def health_check(self, input_data: Any, params: Dict) -> Dict:
        """Return container health and available action status."""
        available_deps = {}
        for dep_name in ["pdf", "ocr", "image", "voice"]:
            dep = self.get_dep(dep_name)
            available_deps[dep_name] = dep is not None

        all_actions = list(self.get_actions().keys())

        return {
            "status": "success",
            "action": "health_check",
            "container": self.name,
            "version": self.version,
            "total_actions": len(all_actions),
            "actions": all_actions,
            "dependencies": available_deps,
            "all_deps_available": all(available_deps.values()),
        }

    async def _handle_procedure_action(
        self, action: str, input_data: Any, params: Dict, handlers: Dict
    ) -> Dict:
        """Honest procedure routing: delegate when a real handler exists, else metadata."""
        from app.core.procedure_actions import (
            is_procedure_action,
            normalize_rfi_issues,
            procedure_metadata,
        )

        if not is_procedure_action(action):
            return {
                "status": "error",
                "error": f"Unknown procedure action: {action}",
            }
        data = input_data if isinstance(input_data, dict) else {}
        p = dict(params or {})
        meta = procedure_metadata(action, project_id=data.get("project_id") or p.get("project_id"))
        delegate = meta.get("delegate_action")
        needs_file = delegate in {"qa_qc_inspection", "drawing_qto", "bim_analysis"}
        has_file = bool(
            data.get("file_path") or p.get("file_path") or data.get("spec_file") or p.get("spec_file")
        )
        # Gate rfi_generator: only delegate when runnable issues exist.
        # Normalize singular `issue` → issues list; otherwise keep procedure metadata.
        if delegate == "rfi_generator":
            issues = normalize_rfi_issues(data, p)
            if not issues:
                return meta
            # Ensure the generator sees the normalized list shape.
            if isinstance(input_data, dict):
                data = dict(data)
                data["issues"] = issues
                input_data = data
            p["issues"] = issues
            params = p
        if delegate and delegate in handlers and not (needs_file and not has_file):
            result = await handlers[delegate](input_data, params)
            if isinstance(result, dict):
                result = dict(result)
                result["procedure_context"] = {
                    "orchestrator_action": action,
                    "procedure_id": meta.get("procedure_id"),
                    "execution_mode": "delegated",
                    "delegate_action": delegate,
                }
            return result
        return meta

    def _detect_trade_from_text(self, text: str) -> str:
        trades = ["concrete", "steel", "electrical", "plumbing", "hvac", "masonry", "finishes", "fire protection"]
        return next((t for t in trades if t in text.lower()), "general")
    async def analyze_schedule_risk(self, input_data: Any, params: Dict) -> Dict:
        return await self.parse_primavera_schedule(input_data, params)
    def _extract_tables_advanced(self, page) -> List[Dict]:
        """Extract ruled/whitespace-aligned tables from a drawing sheet.

        A drawing carries most of its numbers in tables — rebar and bar-bending
        schedules, door/window schedules, legends, revision history. Until now
        this returned ``[]`` unconditionally, so `_process_drawing_page` fed an
        empty ``tables`` list into every drawing ingest and those quantities
        were never seen by retrieval.

        ``[]`` now means "this page has no table", which is a different and
        true statement. The two ways it can be empty for a non-page reason —
        a PyMuPDF too old to have ``find_tables()``, or a detector error — are
        logged rather than passed off as an empty page.
        """
        finder = getattr(page, "find_tables", None)
        if finder is None:
            logger.warning(
                "PyMuPDF on this host has no Page.find_tables(); drawing tables "
                "cannot be extracted (need PyMuPDF >= 1.23). Reporting no tables "
                "for page %s.", getattr(page, "number", "?"),
            )
            return []
        try:
            found = finder()
        except Exception:
            logger.warning(
                "table detection raised on page %s — reporting no tables for it",
                getattr(page, "number", "?"), exc_info=True,
            )
            return []

        tables: List[Dict] = []
        for idx, table in enumerate(getattr(found, "tables", None) or []):
            try:
                raw_rows = table.extract() or []
            except Exception:
                logger.warning(
                    "could not extract table %s on page %s — skipping that table",
                    idx, getattr(page, "number", "?"), exc_info=True,
                )
                continue
            rows = [
                ["" if cell is None else str(cell).replace("\n", " ").strip()
                 for cell in row]
                for row in raw_rows
            ]
            rows = [r for r in rows if any(c for c in r)]
            if not rows:
                continue
            header, body = (rows[0], rows[1:]) if len(rows) > 1 else ([], rows)
            bbox = list(getattr(table, "bbox", None) or [])
            tables.append({
                "table_index": idx,
                "page": (getattr(page, "number", -1) or 0) + 1,
                "bbox": [round(float(v), 2) for v in bbox] if bbox else None,
                "header": header,
                "rows": body,
                "row_count": len(body),
                "column_count": max((len(r) for r in rows), default=0),
                "kind": self._classify_drawing_table(header, body),
            })
        return tables

    def _classify_drawing_table(self, header: List[str], rows: List[List[str]]) -> str:
        """Name the table so downstream consumers can pick the one they need.

        Classifies off the header first, then the first body row, because CAD
        exports frequently emit the caption as row 0 with no true header.
        """
        probe = " ".join(header or []).upper()
        if not probe and rows:
            probe = " ".join(rows[0]).upper()
        checks = (
            ("rebar_schedule", ("BAR MARK", "BAR SCHEDULE", "REBAR", "BAR BENDING", "BBS")),
            ("door_schedule", ("DOOR SCHEDULE", "DOOR TYPE", "DOOR NO", "DOOR REF")),
            ("window_schedule", ("WINDOW SCHEDULE", "WINDOW TYPE", "WINDOW NO")),
            ("revision_history", ("REVISION", "REV DATE", "REV. DATE", "AMENDMENT")),
            ("legend", ("LEGEND", "SYMBOL", "KEY PLAN")),
            ("general_notes", ("GENERAL NOTES", "NOTES:")),
            ("quantities", ("QUANTITY", "QTY", "UNIT", "AREA (", "VOLUME")),
            ("finishes_schedule", ("FINISH SCHEDULE", "ROOM FINISH", "FLOOR FINISH")),
        )
        for kind, needles in checks:
            if any(n in probe for n in needles):
                return kind
        return "unclassified"

    def _extract_annotations(self, page) -> List[Dict]:
        """Extract the PDF annotation layer from a drawing sheet.

        On an issued drawing these are the consultant's markups: revision
        clouds, review comments, stamps, dimension callouts added after
        plotting. They are the review trail, and returning ``[]``
        unconditionally meant every ingest silently discarded it.
        """
        try:
            annots = list(page.annots() or [])
        except Exception:
            logger.warning(
                "could not read the annotation layer of page %s — reporting none",
                getattr(page, "number", "?"), exc_info=True,
            )
            return []

        out: List[Dict] = []
        for annot in annots:
            try:
                atype = getattr(annot, "type", None)
                # PyMuPDF reports (int_code, "Square"); keep the readable half.
                kind = atype[1] if isinstance(atype, (list, tuple)) and len(atype) > 1 \
                    else str(atype)
                info = dict(getattr(annot, "info", None) or {})
                rect = getattr(annot, "rect", None)
                out.append({
                    "type": kind,
                    "content": (info.get("content") or "").strip() or None,
                    "author": (info.get("title") or "").strip() or None,
                    "subject": (info.get("subject") or "").strip() or None,
                    "modified": info.get("modDate") or None,
                    "page": (getattr(page, "number", -1) or 0) + 1,
                    "rect": [round(float(v), 2) for v in
                             (rect.x0, rect.y0, rect.x1, rect.y1)] if rect else None,
                })
            except Exception:
                logger.warning(
                    "skipping an unreadable annotation on page %s",
                    getattr(page, "number", "?"), exc_info=True,
                )
                continue
        return out
    def _extract_specs_advanced(self, text: str) -> List[Dict]:
        specs = []
        grade_pattern = r'\b(C\d{2,3}|M\d{2,3}|S\d{2,3}|Grade\s+\d+)\b'
        for match in re.finditer(grade_pattern, text):
            specs.append({
                "type": "grade",
                "value": match.group(1),
                "context": text[max(0, match.start()-30):match.end()+30]
            })
        return specs
    def _extract_title_block(self, sheet_data: Dict) -> Dict:
        """Read the title block of a drawing sheet from its extracted text.

        The title block is the drawing's identity — number, title, revision,
        scale, who drew and checked it. `_process_drawing_page` puts sheet 1's
        text in ``raw_text``; this reads the labelled fields out of it.

        Deliberately generic (label-anchored), unlike the char-geometry
        extractor in ``app/blocks/drawing_qto.py``, which is tuned to one
        client's sheet layout and needs the per-character stream this caller
        does not have.

        Returns every field as a key so callers can address them
        unconditionally; a field that was not on the sheet is ``None``, never
        an invented value. ``fields_found`` says how many were actually read,
        so "nothing on this sheet" is distinguishable from "extraction did
        not run".
        """
        fields = [
            "drawing_number", "drawing_title", "revision", "scale", "date",
            "drawn_by", "checked_by", "approved_by", "project_name",
            "client", "sheet_number", "discipline",
        ]
        result: Dict[str, Any] = {f: None for f in fields}
        result["fields_found"] = 0

        text = (sheet_data or {}).get("raw_text") or ""
        if not text.strip():
            logger.debug("no raw text on the sheet — title block left empty")
            return result

        # Title blocks sit at the end of a CAD text stream far more often than
        # the start, so prefer the LAST match of each label.
        # DELIBERATE LIMITATION, and the reason every separator below is
        # `[ \t]*` and never `\s*`: a value is only read from the SAME LINE as
        # its label. Some CAD title blocks do put the value on the next line,
        # and those are missed here — but allowing the gap to cross a newline
        # makes `DRAWN BY:` (empty) capture the following `CHECKED BY` label as
        # the drafter's name, which is a fabricated field on a client drawing.
        # Missing a field is recoverable; inventing one is not.
        patterns = {
            "drawing_number": r"(?:DRAWING|DWG|SHEET)[ \t]*(?:NO\.?|NUMBER|#)[ \t]*[:\-.]?[ \t]*([A-Z0-9][A-Z0-9\-_/\.]{3,})",
            "drawing_title": r"(?:DRAWING[ \t]*TITLE|SHEET[ \t]*TITLE|TITLE)[ \t]*[:\-.]?[ \t]*([^\r\n]{3,80})",
            "revision": r"\bREV(?:ISION)?\.?[ \t]*(?:NO\.?)?[ \t]*[:\-.]?[ \t]*([A-Z]?\d{1,2}[A-Z]?|[A-Z])(?=[ \t]*$|\W)",
            "scale": r"\bSCALE[ \t]*[:\-.]?[ \t]*(1[ \t]*:[ \t]*\d{1,5}|N\.?T\.?S\.?|NOT[ \t]*TO[ \t]*SCALE|AS[ \t]*SHOWN)",
            "date": r"\bDATE[ \t]*[:\-.]?[ \t]*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})",
            # `(?:[ \t]*BY)?` must be greedy and inside the label, not a
            # separate alternative: with `(?:DRAWN|DRAWN[ \t]*BY)` the regex
            # takes the shorter branch first and captures "BY" as the name.
            "drawn_by": r"\b(?:DRAWN|DRAFTED|DES(?:IGNED)?)(?:[ \t]*BY)?[ \t]*[:\-.]?[ \t]*([A-Z][A-Za-z0-9 .&'\-]{1,40})",
            "checked_by": r"\bCHECKED(?:[ \t]*BY)?[ \t]*[:\-.]?[ \t]*([A-Z][A-Za-z0-9 .&'\-]{1,40})",
            "approved_by": r"\b(?:APPROVED|AUTHORISED|AUTHORIZED)(?:[ \t]*BY)?[ \t]*[:\-.]?[ \t]*([A-Z][A-Za-z0-9 .&'\-]{1,40})",
            "project_name": r"\bPROJECT[ \t]*(?:NAME|TITLE)?[ \t]*[:\-.]?[ \t]*([^\r\n]{3,80})",
            "client": r"\b(?:CLIENT|EMPLOYER|OWNER)[ \t]*[:\-.]?[ \t]*([^\r\n]{3,80})",
            "sheet_number": r"\b(?:SHEET|SH\.?)[ \t]*[:\-.]?[ \t]*(\d{1,3}[ \t]*(?:OF|/)[ \t]*\d{1,3})",
        }
        for field, pattern in patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if not matches:
                continue
            value = matches[-1].group(1).strip(" :-.\t")
            # A label immediately followed by the next label read as a value.
            if value and not re.fullmatch(r"(?i)(no|title|by|date|rev|scale)\.?", value):
                result[field] = value
                result["fields_found"] += 1

        # Discipline is derivable, not a labelled field: take it from the
        # drawing number prefix when the sheet uses the usual convention.
        number = result["drawing_number"]
        if number:
            head = re.split(r"[\-_/]", number)[0].upper()
            disciplines = {
                "A": "architectural", "AR": "architectural",
                "S": "structural", "ST": "structural",
                "C": "civil", "CV": "civil",
                "M": "mechanical", "ME": "mechanical",
                "E": "electrical", "EL": "electrical",
                "P": "plumbing", "PL": "plumbing",
                "L": "landscape", "FP": "fire protection",
                "INF": "infrastructure",
            }
            if head in disciplines:
                result["discipline"] = disciplines[head]
                result["fields_found"] += 1

        if not result["fields_found"]:
            logger.info(
                "no title-block fields matched on the sheet (%d chars of text) — "
                "the sheet may be a scan, or use an unlabelled title block",
                len(text),
            )
        return result
    def _extract_scale(self, text: str) -> Optional[str]:
        scale_match = re.search(r'\b\d+\s*:\s*\d+\b', text)
        return scale_match.group(0) if scale_match else None
    def _detect_disciplines(self, text: str) -> List[str]:
        disciplines = []
        disc_patterns = {
            "architectural": ["plan", "elevation", "section", "detail"],
            "structural": ["rebar", "rc", "concrete", "steel", "beam", "column", "slab"],
            "mep": ["electrical", "plumbing", "hvac", "mechanical", "fire", "lighting"],
            "civil": ["grading", "drainage", "utility", "road", "pavement"]
        }
        text_lower = text.lower()
        for disc, keywords in disc_patterns.items():
            if any(kw in text_lower for kw in keywords):
                disciplines.append(disc)
        return disciplines
    def _normalize_block_clash(self, clash: Dict, model_file: str) -> Dict:
        """Normalize a bim_extractor clash into the container clash shape so the
        thin shaping helpers (by-discipline grouping, coordination agenda) work."""
        category = clash.get("category", "unknown")
        return {
            "clash_id": f"CLASH-{abs(hash((model_file, clash.get('element_a'), clash.get('element_b')))) % 100000:05d}",
            "type": clash.get("type", "name_duplicate"),
            "description": clash.get("description", ""),
            "severity": clash.get("severity", "warning"),
            "involved_disciplines": [category],
            "category": category,
            "element_a": clash.get("element_a"),
            "element_b": clash.get("element_b"),
            "model_file": model_file,
        }
    def _group_clashes_by_discipline(self, clashes: List[Dict]) -> Dict:
        result = {}
        for clash in clashes:
            for disc in clash.get("involved_disciplines", ["unknown"]):
                result.setdefault(disc, []).append(clash)
        return result
    def _generate_coordination_agenda(self, clashes: List[Dict]) -> List[str]:
        return [f"Review {c['description']} ({c['clash_id']})" for c in clashes[:10]]
    @staticmethod
    def _wmo_conditions(code) -> str:
        """Map a WMO weather-interpretation code to a human label. 'unknown' when
        the code is absent/unrecognised — never guesses 'sunny'."""
        m = {
            0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "fog", 48: "depositing rime fog",
            51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
            56: "freezing drizzle", 57: "dense freezing drizzle",
            61: "slight rain", 63: "rain", 65: "heavy rain",
            66: "freezing rain", 67: "heavy freezing rain",
            71: "slight snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
            80: "slight rain showers", 81: "rain showers", 82: "violent rain showers",
            85: "snow showers", 86: "heavy snow showers",
            95: "thunderstorm", 96: "thunderstorm with hail",
            99: "thunderstorm with heavy hail",
        }
        try:
            return m.get(int(code), "unknown")
        except (ValueError, TypeError):
            return "unknown"

    @staticmethod
    def _weather_impact(precip, wind) -> str:
        """Construction-work impact from real precipitation (mm) + max wind (km/h).
        'unknown' when neither figure is available — never assumes 'favorable'."""
        if precip is None and wind is None:
            return "unknown"
        try:
            p = float(precip) if precip is not None else 0.0
            w = float(wind) if wind is not None else 0.0
        except (ValueError, TypeError):
            return "unknown"
        if p >= 10.0 or w >= 40.0:
            return "adverse"      # heavy rain or high wind — likely work stoppage
        if p >= 1.0 or w >= 25.0:
            return "marginal"
        return "favorable"

    async def _fetch_weather(self, location: str, date: str) -> Dict:
        """Real daily weather for a site via Open-Meteo (free, keyless). Geocodes the
        location, then reads the daily record for `date` (archive API for dates >5 days
        past, forecast API otherwise). Returns status:'unavailable' with a reason on any
        failure — geocode miss, bad date, no record, or network error. NEVER fabricates
        a figure."""
        import httpx
        from datetime import datetime, timezone as _tz

        def unavailable(reason: str) -> Dict:
            return {"location": location, "date": date,
                    "status": "unavailable", "reason": reason, "source": "open-meteo"}

        if not location:
            return unavailable("no site location provided")
        try:
            d = datetime.strptime(date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return unavailable(f"invalid date '{date}' (expected YYYY-MM-DD)")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                geo = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1, "format": "json"},
                )
                results = (geo.json() or {}).get("results") or []
                if not results:
                    return unavailable(f"could not geocode location '{location}'")
                lat, lon = results[0].get("latitude"), results[0].get("longitude")
                days_ahead = (d - datetime.now(_tz.utc).date()).days
                base = ("https://archive-api.open-meteo.com/v1/archive"
                        if days_ahead < -5
                        else "https://api.open-meteo.com/v1/forecast")
                wx = await client.get(base, params={
                    "latitude": lat, "longitude": lon,
                    "daily": ("temperature_2m_max,temperature_2m_min,"
                              "precipitation_sum,windspeed_10m_max,weathercode"),
                    "timezone": "auto", "start_date": date, "end_date": date,
                })
                daily = (wx.json() or {}).get("daily") or {}
                if not (daily.get("time") or []):
                    return unavailable("no weather record for that date/location")

                def first(key):
                    seq = daily.get(key) or []
                    return seq[0] if seq else None

                hi, lo = first("temperature_2m_max"), first("temperature_2m_min")
                precip, wind = first("precipitation_sum"), first("windspeed_10m_max")
                code = first("weathercode")
                return {
                    "location": location, "date": date, "status": "success",
                    "latitude": lat, "longitude": lon,
                    "temperature_high": hi, "temperature_low": lo,
                    "conditions": self._wmo_conditions(code),
                    "wind_speed": f"{wind} km/h" if wind is not None else None,
                    "precipitation": f"{precip}mm" if precip is not None else None,
                    "impact": self._weather_impact(precip, wind),
                    "source": "open-meteo",
                }
        except Exception as exc:
            return unavailable(f"weather service error: {exc}")
    def _run_construction_detector(self, photo_path: str) -> Dict:
        """Run the committed YOLO-Worldv2 ONNX over one photo.

        Returns ``{"detections": [...], "detector": "safety_world_v2"|"unavailable"}``.

        conf_threshold is 0.05, NOT the detector default of 0.25, because the
        report's low-confidence tier starts at 0.05 — at the default the whole
        0.05-0.25 band would be discarded inside the detector and could never
        be surfaced as "possible X -- low confidence".

        `detector: "unavailable"` when SAFETY_WORLD_WEIGHTS is unset or the
        model cannot load. The report must say the photo path was unavailable
        rather than print an empty section that reads as "nothing found on
        site" — an absent detector and a clean site are not the same claim.
        """
        from app.blocks.safety_world_detector import default_detector
        try:
            detector = default_detector()
        except Exception:
            logger.exception("construction detector failed to load for %s", photo_path)
            return {"detections": [], "detector": "unavailable"}
        if detector is None:
            return {"detections": [], "detector": "unavailable"}
        try:
            from app.containers.construction.photo_observations import LOW_CONF_THRESHOLD
            return {
                "detections": detector.detect(Path(photo_path), conf_threshold=LOW_CONF_THRESHOLD),
                "detector": "safety_world_v2",
            }
        except Exception:
            logger.exception("construction detector raised on %s", photo_path)
            return {"detections": [], "detector": "unavailable"}

    async def _analyze_site_photo(self, photo_path: str) -> Dict:
        """Per-photo analysis feeding the daily report's photo consumers.

        CONTRACT (the three _extract_*_from_photos consumers read this shape):
          photo                 filename, the evidence reference in the report
          activities_detected   generic `image` block objects (pre-existing)
          safety_compliance     compliant | issues_found | unknown
          headcount_estimate    int  (pre-existing)
          progress_indicators   str  (pre-existing)
          detections            [{class, confidence, bbox}] from the
                                construction detector's 33-prompt vocabulary
          detector              safety_world_v2 | unavailable

        The first five keys are unchanged — `_extract_safety_observations`
        already reads `safety_compliance` and must keep working. `detections`
        and `detector` are added for the quality and equipment consumers, which
        need the class and confidence the generic block does not provide.
        """
        detector_result = await asyncio.to_thread(self._run_construction_detector, photo_path)
        analysis = {
            "photo": Path(photo_path).name,
            "activities_detected": [],
            # "unknown" not "compliant": with no generic-block result there is
            # no basis for either claim. _extract_safety_observations treats
            # anything other than "compliant" as an observation, so this must
            # never default to a value that invents one.
            "safety_compliance": "unknown",
            "headcount_estimate": 0,
            "progress_indicators": "",
            **detector_result,
        }
        image_block = self.get_dep("image")
        if image_block:
            try:
                result = await image_block.execute(
                    {"file_path": photo_path},
                    {"prompt": "Identify: trade/work activity, equipment, materials, safety conditions, progress indicators, headcount estimate"}
                )
                objects = result.get("objects", [])
                analysis.update({
                    "activities_detected": objects,
                    "safety_compliance": "compliant" if not any("hazard" in str(o).lower() for o in objects) else "issues_found",
                    "headcount_estimate": result.get("people_count", 0),
                    "progress_indicators": result.get("description", "")[:200],
                })
            except Exception:
                logger.warning(
                    "swallowed %s in _analyze_site_photo() — continuing",
                    "Exception", exc_info=True,
                )
        return analysis
    def _extract_activities_from_voice(self, transcriptions: List[Dict]) -> List[Dict]:
        activities = []
        combined_text = " ".join([t.get("text", "") for t in transcriptions])
        activity_patterns = [
            (r'(?:poured|placed|cast)\s+(\d+)\s*(?:m3|cubic)\s+(?:of\s+)?concrete', "concrete_pour"),
            (r'(?:erected|installed)\s+(?:steel|column|beam)', "steel_erection"),
            (r'(?:block|masonry|brick)\s+(?:work|laid|installed)', "masonry_work"),
            (r'(?:formwork|shuttering)\s+(?:stripped|removed)', "formwork_stripping"),
            (r'(?:rebar|steel)\s+(?:fixing|installation)', "rebar_fixing"),
            (r'(?:excavation|digging|earth)', "earthwork"),
            (r'(?:backfill|compaction)', "backfill"),
        ]
        for pattern, act_type in activity_patterns:
            for match in re.finditer(pattern, combined_text, re.IGNORECASE):
                activities.append({
                    "type": act_type,
                    "description": match.group(0),
                    "location": self._extract_location_from_context(match.start(), combined_text),
                    "quantity": match.group(1) if match.groups() else "unknown",
                    "percent_complete": "ongoing"
                })
        return activities
    def _extract_location_from_context(self, position: int, text: str) -> str:
        before = text[max(0, position-50):position]
        m = re.search(r'(?:at|in|near)\s+([A-Za-z0-9\s]+)', before, re.IGNORECASE)
        return m.group(1).strip() if m else "site"
    def _extract_issues_from_voice(self, transcriptions: List[Dict]) -> List[Dict]:
        issues = []
        combined_text = " ".join([t.get("text", "") for t in transcriptions])
        issue_patterns = [
            (r'(?:delay|held up|waiting)', "delay"),
            (r'(?:clarification|question|need to know)', "clarification_needed"),
            (r'(?:safety|hazard|unsafe)', "safety_issue"),
            (r'(?:defect|quality|rework)', "quality_issue"),
        ]
        for pattern, issue_type in issue_patterns:
            for match in re.finditer(pattern, combined_text, re.IGNORECASE):
                issues.append({
                    "type": issue_type,
                    "description": match.group(0),
                    "context": combined_text[max(0, match.start()-30):match.end()+30]
                })
        return issues
    def _extract_manpower_from_voice(self, transcriptions: List[Dict]) -> Dict:
        return {"total": 0, "by_trade": {}, "absent": 0}
    def _extract_equipment_from_photos(self, photo_analysis: List[Dict]) -> List[Dict]:
        """Plant and temporary works seen in the photo set.

        Only crane / ladder / scaffolding exist in the detector's vocabulary,
        so that is all this can return. Excavators, telehandlers and dumpers
        have no prompt string and are invisible at any confidence — a
        vocabulary limit registered in docs/KNOWN_INCOMPLETE.md, not something to
        paper over by guessing from context.
        """
        from app.containers.construction.photo_observations import equipment_from_photos
        return equipment_from_photos(photo_analysis)
    def _generate_daily_narrative(self, date: str, activities: List, issues: List, weather: Dict, manpower: Dict) -> str:
        parts = []
        parts.append(f"DAILY SITE REPORT - {date}")
        if weather.get("status") == "unavailable":
            parts.append(f"Weather: unavailable ({weather.get('reason', 'no data')})")
        elif weather.get("conditions") is not None:
            parts.append(f"Weather: {weather.get('conditions')}, High: {weather.get('temperature_high')}°C")
        else:
            parts.append("Weather: not requested")
        parts.append("")
        parts.append("MANPOWER:")
        parts.append(f"Total: {manpower.get('total', 0)} workers present")
        for trade, count in manpower.get("by_trade", {}).items():
            parts.append(f"  - {trade}: {count}")
        parts.append("")
        parts.append("WORK COMPLETED:")
        for act in activities[:5]:
            parts.append(f"• {act['description']} at {act.get('location', 'site')}")
        if not activities:
            parts.append("• General site activities ongoing")
        parts.append("")
        if issues:
            parts.append("ISSUES/CONSTRAINTS:")
            for issue in issues:
                parts.append(f"⚠ {issue.get('description')}")
            parts.append("")
        parts.append(f"Next Day: Continue ongoing activities pending resolution of identified issues")
        return "\n".join(parts)
    def _extract_safety_observations(self, photo_analysis: List[Dict], transcriptions: List[Dict]) -> List[Dict]:
        """Safety observations from photo analysis.

        Only `issues_found` produces an observation. It previously fired on
        anything that was not "compliant", which meant `unknown` — the value
        used when no analysis could be performed at all — emitted "Safety
        issues detected" for a photo nothing had looked at. That is a
        fabricated observation on a client-facing report, and the reason the
        default is now explicit rather than inferred from inequality.
        """
        obs = []
        for p in photo_analysis:
            if p.get("safety_compliance") == "issues_found":
                obs.append({
                    "source": "photo",
                    "photo": p.get("photo"),
                    "observation": "Safety issues detected in photo analysis",
                })
        return obs
    def _extract_quality_observations(self, photo_analysis: List[Dict]) -> List[Dict]:
        """Workmanship observations from the 13 quality prompts.

        Observations, never verdicts: "crack in concrete wall detected (0.62)"
        records what the camera saw. Whether it is a defect, and what it means,
        is an engineer's judgement made with the operator in the loop.
        """
        from app.containers.construction.photo_observations import quality_observations
        return quality_observations(photo_analysis)
    def _extract_material_deliveries(self, transcriptions: List[Dict]) -> List[Dict]:
        return []
    def _generate_next_day_plan(self, activities: List[Dict], issues: List[Dict]) -> List[str]:
        return ["Continue ongoing activities"] + [f"Resolve: {i.get('description')}" for i in issues[:3]]
    def _select_optimal_scenario(self, scenarios: Dict, cost_priority: bool = True) -> Dict:
        if cost_priority:
            return scenarios.get("aggressive") if scenarios.get("aggressive", {}).get("savings_percent", 0) > 0.15 else scenarios.get("conservative")
        return scenarios.get("carbon_optimized", scenarios.get("conservative"))
    def _group_ve_by_category(self, alternatives: List[Dict]) -> Dict:
        result = {}
        for a in alternatives:
            cat = a.get("original", "unknown")
            result.setdefault(cat, []).append(a)
        return result
    def _generate_ve_roadmap(self, scenario: Dict) -> List[str]:
        return ["Identify affected BOQ items", "Obtain engineer approval", "Update specifications", "Issue variation order"]
    def _identify_ve_approvals(self, scenario: Dict) -> List[str]:
        return ["Engineer", "Client"] if scenario.get("risk_level") != "low" else ["Engineer"]
    def _generate_hvac_commissioning(self) -> List[Dict]:
        return [
            {"test": "Air Balancing", "standard": "ASHRAE 111", "witness_required": True, "acceptance_criteria": "±10% of design"},
            {"test": "Chiller Performance", "standard": "AHRI 550/590", "witness_required": True, "acceptance_criteria": "Within 5% of spec"},
            {"test": "Pump Performance", "standard": "HI 40.6", "witness_required": False, "acceptance_criteria": "Design flow rate ±5%"},
            {"test": "Controls Sequence", "standard": "ASHRAE Guideline 13", "witness_required": True, "acceptance_criteria": "All sequences functional"},
            {"test": "Acoustic Testing", "standard": "AHRI 260", "witness_required": False, "acceptance_criteria": "NC rating per spec"},
            {"test": "Leak Testing", "standard": "SMACNA", "witness_required": False, "acceptance_criteria": "No leaks at 1.5x working pressure"},
            {"test": "Energy Metering Verification", "standard": "IPMVP", "witness_required": True, "acceptance_criteria": "±2% accuracy"},
        ]
    def _generate_electrical_commissioning(self) -> List[Dict]:
        return [
            {"test": "Insulation Resistance", "standard": "IEEE 43", "witness_required": False, "acceptance_criteria": ">1 MΩ"},
            {"test": "Continuity Testing", "standard": "BS 7671", "witness_required": False, "acceptance_criteria": "R1+R2 < design"},
            {"test": "Earth Fault Loop", "standard": "BS 7671", "witness_required": True, "acceptance_criteria": "Zs < tabulated"},
            {"test": "RCD Testing", "standard": "BS 7671", "witness_required": True, "acceptance_criteria": "Trip time < 300ms"},
            {"test": "Load Bank Test", "standard": "IEEE 450", "witness_required": True, "acceptance_criteria": "Full load 4 hours"},
            {"test": "Power Quality", "standard": "IEEE 519", "witness_required": False, "acceptance_criteria": "THD < 5%"},
            {"test": "Generator Auto-Start", "standard": "NFPA 110", "witness_required": True, "acceptance_criteria": "Start < 10 seconds"},
        ]
    def _generate_fire_commissioning(self) -> List[Dict]:
        return [
            {"test": "Sprinkler Flow Test", "standard": "NFPA 13", "witness_required": True, "acceptance_criteria": "Design density achieved"},
            {"test": "Fire Pump Performance", "standard": "NFPA 20", "witness_required": True, "acceptance_criteria": "Rated flow and pressure"},
            {"test": "Alarm Device Function", "standard": "NFPA 72", "witness_required": True, "acceptance_criteria": "100% devices tested"},
            {"test": "Smoke Detector Sensitivity", "standard": "NFPA 72", "witness_required": False, "third_party_required": True, "acceptance_criteria": "Within listed range"},
            {"test": "Door Holder Release", "standard": "NFPA 80", "witness_required": False, "acceptance_criteria": "All doors close on alarm"},
            {"test": "Stair Pressurization", "standard": "NFPA 92", "witness_required": True, "acceptance_criteria": "50 Pa minimum"},
        ]
    def _generate_plumbing_commissioning(self) -> List[Dict]:
        return [
            {"test": "Water Pressure Test", "standard": "IPC", "witness_required": False, "acceptance_criteria": "No leaks at 1.5x working pressure"},
            {"test": "Drainage Flow Test", "standard": "IPC", "witness_required": False, "acceptance_criteria": "Free flow, no blockages"}
        ]
    def _generate_elevator_commissioning(self) -> List[Dict]:
        return [
            {"test": "Safety Gear Test", "standard": "EN 81", "witness_required": True, "acceptance_criteria": "Functional"},
            {"test": "Load Test", "standard": "EN 81", "witness_required": True, "acceptance_criteria": "Rated load ±5%"}
        ]
    def _generate_facade_commissioning(self) -> List[Dict]:
        return [
            {"test": "Water Tightness", "standard": "ASTM E331", "witness_required": True, "acceptance_criteria": "No leakage at test pressure"},
            {"test": "Air Infiltration", "standard": "ASTM E283", "witness_required": False, "acceptance_criteria": "Within spec"}
        ]

    def _generate_waterproofing_commissioning(self) -> List[Dict]:
        """Pre-backfill hold-point checks for torch-applied SBS / below-grade WP."""
        return [
            {"test": "Substrate condition", "standard": "Manufacturer method statement", "witness_required": True, "hold_point": True, "acceptance_criteria": "Concrete cured, dry, free of laitance and standing water"},
            {"test": "Primer application", "standard": "Manufacturer TDS", "witness_required": True, "acceptance_criteria": "Even coat, tack window observed before torching"},
            {"test": "Membrane type and thickness", "standard": "Project spec / TDS", "witness_required": True, "acceptance_criteria": "Torch-applied SBS as specified (typical 4 mm); mill certs on site"},
            {"test": "Lap width and torching", "standard": "Manufacturer TDS", "witness_required": True, "acceptance_criteria": "Laps typically 100 mm, continuous bleed of bitumen, no fish-mouths"},
            {"test": "Holiday / spark test", "standard": "ASTM D4787 / manufacturer", "witness_required": True, "hold_point": True, "acceptance_criteria": "No holidays on completed sheet before protection"},
            {"test": "Protection board before backfill", "standard": "Project spec", "witness_required": True, "hold_point": True, "acceptance_criteria": "Protection in place; no plant tracking on bare membrane"},
            {"test": "Backfill release", "standard": "ITP hold point", "witness_required": True, "hold_point": True, "acceptance_criteria": "Engineer / QC sign-off recorded before backfill"},
        ]

    def _generate_reservoir_commissioning(self) -> List[Dict]:
        """First wet-test checks for a PWPS / reservoir — not a membrane holiday test."""
        return [
            {"test": "Pre-fill cleanliness", "standard": "Project spec / PWPS method statement", "witness_required": True, "hold_point": True, "acceptance_criteria": "Debris, laitance and construction residue removed; outlets clear"},
            {"test": "Concrete strength — C21-OPC cubes", "standard": "Cast in-situ concrete spec", "witness_required": True, "hold_point": True, "acceptance_criteria": "Cubes meet C21-OPC; mix and pour volumes recorded (planned vs actual)"},
            {"test": "Blinding extent and volume", "standard": "Pour record / IFC", "witness_required": True, "acceptance_criteria": "Blinding as poured; shortfall vs planned volume noted before fill"},
            {"test": "Inlet and outlet isolation", "standard": "PWPS O&M / method statement", "witness_required": True, "hold_point": True, "acceptance_criteria": "Valves isolated; no backflow into live mains during first wet test"},
            {"test": "First wet test / watertightness", "standard": "BS 8007 / project spec", "witness_required": True, "hold_point": True, "acceptance_criteria": "No visible leakage during hold; seepage within spec"},
            {"test": "Fill and drawdown cycle", "standard": "Project commissioning plan", "witness_required": True, "acceptance_criteria": "Controlled fill; drawdown rate and residual volume recorded"},
            {"test": "Level gauges and instrumentation", "standard": "Manufacturer / PWPS spec", "witness_required": False, "acceptance_criteria": "Gauges reading; alarms proved if installed"},
        ]
    def _generate_bms_commissioning(self) -> List[Dict]:
        return [
            {"test": "Point-to-Point Checkout", "standard": "ASHRAE Guideline 13", "witness_required": False, "acceptance_criteria": "100% points verified"},
            {"test": "Sequence Verification", "standard": "ASHRAE Guideline 13", "witness_required": True, "acceptance_criteria": "All sequences functional"}
        ]
    def _estimate_commissioning_duration(self, systems: List[str], equipment_count: int) -> int:
        base_weeks = len(systems) * 2
        return base_weeks + (equipment_count // 10)
    def _add_weeks(self, date_str: str, weeks: int) -> Optional[str]:
        try:
            d = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return (d + timedelta(weeks=weeks)).isoformat()
        except Exception:
            return None
    def _list_commissioning_docs(self, systems: List[str]) -> List[str]:
        return [f"{s}_commissioning_report.pdf" for s in systems]
    def _generate_training_requirements(self, systems: List[str]) -> List[Dict]:
        return [{"system": s, "training": f"Operator training for {s}"} for s in systems]
    # NOTE: the former _calculate_labor_histogram / _identify_resource_peaks /
    # _identify_resource_conflicts / _suggest_resource_leveling / _breakdown_by_trade /
    # _calculate_cost_histogram cluster was deleted (V1). It fabricated an all-zero
    # 26-week histogram with a hardcoded 30/20/15/15/20 trade split (it read
    # early_start/resources.labor keys the parser never produces). resource_histogram
    # now delegates to the hand-verified app.lib.pm_computations.resource_histogram
    # (real TASKRSRC time-phasing) or returns an honest error.
    def _generate_claim_narrative(self, events: List[Dict], delay_analysis: Dict, entitlement: Dict) -> Dict:
        total_delay = delay_analysis.get("total_delay_days", 0)
        exec_summary = f"""EXTENSION OF TIME CLAIM

    The Contractor has encountered delays totaling {total_delay} calendar days due to circumstances beyond our control and for which the Contract provides entitlement to Extension of Time and associated costs.

    Key Events:
    """
        for i, event in enumerate(events[:5], 1):
            exec_summary += f"{i}. {event.get('description', 'Unknown event')} ({event.get('delay_days', 0)} days)\n"
    
        full_narrative = f"""BACKGROUND
    The Contractor has been progressing the Works in accordance with the Approved Programme when the following delay events occurred:

    {chr(10).join([f"Event {i+1}: {e.get('description')} on {e.get('date')}" for i, e in enumerate(events)])}

    CONTRACTUAL ENTITLEMENT
    Under Clause {entitlement.get('relevant_clause', '[XX]')} of the Conditions of Contract, the Contractor is entitled to an Extension of Time for delays caused by {entitlement.get('entitlement_basis', '[compensable delay events]')}.

    CAUSATION ANALYSIS
    {delay_analysis.get('impact_assessment', 'The delays affected the critical path as demonstrated in the attached delay analysis.')}

    DELAY QUANTIFICATION
    Total Extension of Time Sought: {total_delay} days
    """
        return {
            "executive_summary": exec_summary,
            "full_narrative": full_narrative,
            "word_count": len(full_narrative.split())
        }
    def _build_causation_link(self, events: List[Dict], delay_analysis: Dict) -> List[Dict]:
        linkages = []
        for event in events:
            linkages.append({
                "event": event.get("description"),
                "date": event.get("date"),
                "cause": event.get("cause", "Employer Risk Event"),
                "effect": f"Delay of {event.get('delay_days')} days to {event.get('affected_activity', 'critical path')}",
                "mitigation_attempted": event.get("mitigation", "None possible"),
                "concurrent": event.get("concurrent", False),
                "compensable": event.get("compensable", True)
            })
        return linkages
    def _identify_concurrent_delays(self, events: List[Dict]) -> List[Dict]:
        return [e for e in events if e.get("concurrent", False)]
    def _list_claim_documents(self, events: List[Dict]) -> List[str]:
        return ["Delay notices", "Schedule analysis", "Daily reports", "Photos"]
    def _compile_evidence_list(self, events: List[Dict]) -> List[Dict]:
        return [{"event": e.get("description"), "evidence": e.get("evidence", [])} for e in events]
    def _anticipate_defenses(self, events: List[Dict]) -> List[str]:
        return ["Mitigation efforts were reasonable"]
    def _score_schedule(self, duration: int, all_durations: List[int]) -> float:
        if not all_durations or duration <= 0:
            return 50
        avg = sum(all_durations) / len(all_durations)
        min_d = min(all_durations)
        if duration == min_d:
            return 100
        elif duration <= avg:
            return 80
        elif duration <= avg * 1.1:
            return 60
        return 40
    def _identify_qualification_gaps(self, bid: Dict) -> List[str]:
        return []
    def _identify_bid_clarifications(self, bid: Dict) -> List[str]:
        return []
    def _extract_variation_clauses(self, contract_data: Dict) -> Dict:
        return {
            "clause_reference": None,
            "note": "Variation clause extraction not implemented; provide clause_reference manually.",
        }
    def _check_time_bar(self, existing: List[Dict], new_vo: Dict) -> Dict:
        event_date = new_vo.get("event_date")
        notice_date = new_vo.get("notice_date")
        if event_date and notice_date:
            days_elapsed = self._days_between(event_date, notice_date)
            return {"at_risk": days_elapsed > 14, "days_elapsed": days_elapsed, "mitigation": "Immediate notice recommended" if days_elapsed > 10 else None}
        return {
            "at_risk": None,
            "days_elapsed": None,
            "note": "event_date and notice_date required to assess time-bar risk",
        }
    def _days_between(self, date1: str, date2: str) -> int:
        try:
            d1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            d2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
            return abs((d2 - d1).days)
        except Exception:
            return 0
    def _generate_vo_document(
        self,
        vo_number: str,
        description: str,
        pricing: Dict,
        vo_type: str,
        lines: Optional[List[Dict]] = None,
    ) -> str:
        parts = [
            f"Variation Order {vo_number}",
            f"Type: {vo_type}",
            f"Description: {description}",
        ]
        if lines:
            parts.append("Lines:")
            for line in lines:
                if not isinstance(line, dict):
                    parts.append(f"  {line}")
                    continue
                parts.append(
                    f"  {line.get('kind')} {line.get('quantity')} "
                    f"{line.get('unit')} {line.get('description')} "
                    f"@ {line.get('rate')} = {line.get('amount')}"
                )
        net = pricing.get("total")
        parts.append(f"Net: {net}")
        parts.append(f"Total: {net}")
        return "\n".join(parts)
    def _list_vo_documents(self, vo_data: Dict) -> List[str]:
        docs = vo_data.get("supporting_documents") or vo_data.get("attachments") or []
        if isinstance(docs, list):
            return [str(d) for d in docs if d]
        if docs:
            return [str(docs)]
        return []
    def _identify_vo_risks(self, vo_data: Dict, cumulative: Dict) -> List[str]:
        risks = []
        if cumulative.get("approaching_cap"):
            risks.append("Approaching contract variation cap")
        if "notice_given" in vo_data and not vo_data.get("notice_given"):
            risks.append("Notice not given - time bar risk")
        return risks
    def _run_time_impact_analysis(self, baseline: Dict, updated: Dict, events: List[Dict]) -> Dict:
        impacted_durations = []
        for event in events:
            activity = next((a for a in baseline.get("activities", []) if a["id"] == event.get("activity_id")), None)
            if activity:
                original_duration = activity.get("duration", 0)
                delay = event.get("delay_days", 0)
                impacted_durations.append({"activity": activity["id"], "original": original_duration, "delay_added": delay, "new_duration": original_duration + delay, "critical": activity.get("critical", False)})
        critical_delays = [d for d in impacted_durations if d["critical"]]
        total_delay = sum(d["delay_added"] for d in critical_delays)
        return {"method": "Time Impact Analysis", "total_delay_days": total_delay, "impacted_activities": len(impacted_durations), "critical_path_impacts": critical_delays, "methodology_notes": "Delays inserted into baseline CPM, network recalculated"}
    def _run_windows_analysis(self, baseline: Dict, updated: Dict, events: List[Dict]) -> Dict:
        windows = self._group_events_into_windows(events)
        window_results = []
        cumulative_delay = 0
        for window in windows:
            window_delay = sum(e.get("delay_days", 0) for e in window["events"] if e.get("critical", False))
            cumulative_delay += window_delay
            window_results.append({"period": window["period"], "events_count": len(window["events"]), "this_period_delay": window_delay, "cumulative_delay": cumulative_delay, "float_consumed": window_delay * 0.5})
        return {"method": "Windows Analysis", "total_delay_days": cumulative_delay, "windows_analyzed": len(window_results), "window_details": window_results, "methodology_notes": "Schedule divided into time windows, delay apportioned per period"}
    def _group_events_into_windows(self, events: List[Dict]) -> List[Dict]:
        """Bucket events into calendar-month windows by their `date` field.

        Previously this binned every 5 sorted events into one "window"
        regardless of date — "Month 2" started at event 5, not day 31.
        On a real claim with unevenly-distributed events that silently
        mixed calendar months.
        """
        sorted_events = sorted(events, key=lambda x: str(x.get("date", "")))
        buckets: Dict[str, List[Dict]] = {}
        order: List[str] = []
        for event in sorted_events:
            date_str = str(event.get("date", "")).strip()
            if not date_str:
                continue
            # Take the YYYY-MM prefix as the bucket key. Tolerates full ISO
            # ('2026-03-15T...') or date-only ('2026-03-15') formats.
            key = date_str[:7]
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(event)
        return [
            {"period": key, "events": buckets[key]}
            for key in order
        ]
    def _run_impacted_as_planned(self, baseline: Dict, updated: Dict, events: List[Dict]) -> Dict:
        """Impacted As-Planned (IAP) method — real implementation.

        Take the baseline schedule duration and add the delay events one
        at a time. Predicted impacted duration = baseline + sum(delay_days)
        for events affecting critical activities. Compare against the
        as-built duration to identify model bias.
        """
        baseline_duration = baseline.get("project_duration", 0)
        as_built_duration = updated.get("project_duration", 0)

        # Only critical-path events add to the predicted impacted duration.
        critical_events = [e for e in events if e.get("critical", False)]
        added_delay = sum(int(e.get("delay_days", 0)) for e in critical_events)
        predicted_duration = baseline_duration + added_delay
        model_bias = predicted_duration - as_built_duration

        return {
            "method": "Impacted As-Planned",
            "total_delay_days": added_delay,
            "impacted_activities": len(critical_events),
            "critical_path_impacts": [
                {
                    "event": e.get("description") or e.get("id", "(unnamed)"),
                    "date": e.get("date"),
                    "delay_days": int(e.get("delay_days", 0)),
                }
                for e in critical_events
            ],
            "baseline_duration": baseline_duration,
            "predicted_impacted_duration": predicted_duration,
            "as_built_duration": as_built_duration,
            "model_bias_days": model_bias,
            "methodology_notes": (
                "IAP inserts each critical-event delay into the baseline. "
                "Predicted duration is compared to the as-built; "
                "model_bias_days > 0 means the model over-predicts delay "
                "(contractor likely recovered some), < 0 means under-predicts."
            ),
        }
    def _analyze_critical_path_changes(self, baseline: Dict, updated: Dict) -> Dict:
        baseline_acts = baseline.get("activities", [])
        updated_acts = updated.get("activities", [])
        baseline_critical_ids = {a["id"] for a in baseline_acts if a.get("critical")}
        updated_critical_ids = {a["id"] for a in updated_acts if a.get("critical")}
        added = sorted(updated_critical_ids - baseline_critical_ids)
        removed = sorted(baseline_critical_ids - updated_critical_ids)
        unchanged = sorted(baseline_critical_ids & updated_critical_ids)
        return {
            "baseline_critical_count": len(baseline_critical_ids),
            "updated_critical_count": len(updated_critical_ids),
            "critical_path_unchanged": len(removed) == 0 and len(added) == 0,
            "newly_critical": added[:50],
            "no_longer_critical": removed[:50],
            "unchanged_critical": unchanged[:50],
        }
    def _parse_event_date(self, val):
        """Parse a delay-event date string into a ``date`` object, or None."""
        if not val:
            return None
        s = str(val).strip()
        if not s:
            return None
        # Accept full ISO datetimes ('2026-03-15T08:30:00Z') as well as
        # date-only ('2026-03-15').
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except Exception:
            logger.warning(
                "swallowed %s in _parse_event_date() — continuing",
                "Exception", exc_info=True,
            )
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except Exception:
            return None
    def _apportion_delay(self, total_days: int, events: List[Dict], concurrency: Dict) -> Dict:
        compensable = sum(e.get("delay_days", 0) for e in events if e.get("compensable") and e.get("excusable"))
        non_excusable = sum(e.get("delay_days", 0) for e in events if not e.get("excusable"))
        concurrent = concurrency.get("concurrent_days", 0)
        return {"total_delay": total_days, "compensable_days": compensable, "non_compensable_days": non_excusable, "concurrent_days": concurrent, "contractor_entitlement": max(0, compensable - concurrent), "contractor_responsible": non_excusable, "shared_delay": min(compensable, non_excusable)}
    def _add_months(self, start_date_str: str, months: int) -> str:
        try:
            start = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            new_month = ((start.month - 1 + months) % 12) + 1
            new_year = start.year + ((start.month - 1 + months) // 12)
            return f"{new_year}-{new_month:02d}"
        except Exception:
            return f"Month+{months}"
    def _subtract_weeks(self, date_str: str, weeks: int) -> str:
        try:
            d = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return (d - timedelta(weeks=weeks)).isoformat()
        except Exception:
            return "ASAP"
    def _identify_consolidation(self, plan: List[Dict]) -> List[Dict]:
        return []
    def _suggest_bundling(self, plan: List[Dict], suppliers: List[Dict]) -> List[Dict]:
        return []
    def _identify_procurement_risks(self, plan: List[Dict]) -> List[Dict]:
        return []
    def _score_governance(self, metrics: Dict) -> float:
        score = 70
        if metrics.get("anti_corruption"):
            score += 15
        if metrics.get("ethics_training", 0) > 90:
            score += 10
        return min(100, score)
    def _generate_esg_recommendations(self, scores: Dict, env: Dict, social: Dict) -> List[str]:
        recs = []
        if scores["environmental"] < 70:
            recs.append("Improve waste diversion and recycled content targets")
        if social.get("ltifr", 0) > 2:
            recs.append("Strengthen safety training and monitoring")
        return recs
    def _generate_stakeholder_narrative(self, scores: Dict, env: Dict, social: Dict) -> str:
        return f"This project demonstrates {'strong' if scores['overall'] >= 70 else 'moderate'} ESG performance with overall score {scores['overall']:.1f}."
    def _add_years_str(self, date_str: Optional[str], years: int) -> str:
        if not date_str:
            return "TBD"
        try:
            d = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return f"{d.year + years}-{d.month:02d}-{d.day:02d}"
        except Exception:
            return "TBD"
    def _group_equipment_by_system(self, equipment: List[Dict]) -> List[Dict]:
        systems = {}
        for equip in equipment:
            system_type = equip.get("system_type", "General")
            if system_type not in systems:
                systems[system_type] = []
            systems[system_type].append(equip)
        return [{"name": k, "description": f"{k} System", "equipment": v} for k, v in systems.items()]
    def _map_system_dependencies(self, systems: List[Dict]) -> List[Dict]:
        return []
    def _generate_equipment_maintenance(self, equip: Dict) -> Dict:
        category = equip.get("category", "general")
        schedules = {
            "hvac_equipment": {"daily": ["Check operation", "Check for unusual noise"], "monthly": ["Filter inspection", "Belt tension check"], "quarterly": ["Coil cleaning", "Motor bearing check"], "annually": ["Full service", "Performance testing"]},
            "pump": {"weekly": ["Visual inspection", "Leak check"], "monthly": ["Vibration check", "Seal inspection"], "annually": ["Impeller inspection", "Motor service"]},
            "electrical_panel": {"monthly": ["Temperature check", "Torque connections"], "annually": ["IR testing", "Breaker testing"]}
        }
        return schedules.get(category, schedules["hvac_equipment"])
    def _generate_startup_procedures(self, systems: List[Dict]) -> List[str]:
        return [f"Startup procedure for {s['name']}" for s in systems]
    def _generate_normal_operation(self, systems: List[Dict]) -> List[str]:
        return [f"Normal operation for {s['name']}" for s in systems]
    def _generate_shutdown_procedures(self, systems: List[Dict]) -> List[str]:
        return [f"Shutdown procedure for {s['name']}" for s in systems]
    def _generate_emergency_procedures(self, systems: List[Dict]) -> List[str]:
        return [f"Emergency procedure for {s['name']}" for s in systems]
    def _generate_seasonal_operation(self, systems: List[Dict]) -> List[str]:
        return [f"Seasonal operation for {s['name']}" for s in systems]
    def _generate_daily_tasks(self, equipment: List[Dict]) -> List[str]:
        return []
    def _generate_weekly_tasks(self, equipment: List[Dict]) -> List[str]:
        return []
    def _generate_monthly_tasks(self, equipment: List[Dict]) -> List[str]:
        return ["Inspect visible equipment"]
    def _generate_quarterly_tasks(self, equipment: List[Dict]) -> List[str]:
        return []
    def _generate_annual_tasks(self, equipment: List[Dict]) -> List[str]:
        return ["Annual service"]
    def _create_maintenance_matrix(self, equipment: List[Dict]) -> List[Dict]:
        return []
    def _generate_troubleshooting_guide(self, equipment: List[Dict]) -> List[Dict]:
        return []
    def _generate_spare_parts_list(self, equipment: List[Dict]) -> List[Dict]:
        return []
    def _extract_training_needs(self, equipment: List[Dict]) -> List[str]:
        return []
    def _transform_for_platform(self, data: Dict, platform: str) -> Dict:
        transformed = {"project_id": data.get("project_id"), "elements": []}
        for element in data.get("elements", []):
            twin_element = {"id": element.get("guid", element.get("id")), "name": element.get("name"), "type": element.get("category", "Generic"), "geometry": element.get("geometry"), "properties": element.get("properties", {}), "relationships": element.get("relationships", [])}
            if platform == "bim360":
                twin_element["objectId"] = twin_element.pop("id")
                twin_element["externalId"] = twin_element["objectId"]
            elif platform == "azure":
                twin_element["$dtId"] = twin_element.pop("id")
                twin_element["$metadata"] = {"$model": f"dtmi:construction:{twin_element['type']};1"}
            transformed["elements"].append(twin_element)
        return transformed
    def _generate_initial_sync_operations(self, data: Dict, platform: str) -> List[Dict]:
        return [{"operation": "CREATE", "type": "element", "target_id": element.get("id"), "properties": element.get("properties", {}), "geometry": element.get("geometry") if platform != "azure" else None, "relationships": element.get("relationships", [])} for element in data.get("elements", [])]
    def _generate_delta_operations(self, data: Dict, platform: str) -> List[Dict]:
        operations = []
        for element in data.get("elements", []):
            change_type = element.get("change_type", "UPDATE")
            if change_type == "ADD":
                operations.append({"operation": "CREATE", "type": "element", "target_id": element.get("id"), "properties": element.get("properties", {})})
            elif change_type == "DELETE":
                operations.append({"operation": "DELETE", "type": "element", "target_id": element.get("id")})
            else:
                operations.append({"operation": "UPDATE", "type": "property_update", "target_id": element.get("id"), "changed_properties": element.get("changed_properties", []), "timestamp": element.get("timestamp")})
        return operations
    def _generate_update_operations(self, data: Dict, platform: str) -> List[Dict]:
        return self._generate_delta_operations(data, platform)
    def _get_platform_config(self, platform: str, project_id: str) -> Dict:
        configs = {
            "bim360": {"format": "Forge JSON", "geometry_format": "SVF", "property_sets": ["Identity Data", "Phasing", "Structural"], "rate_limits": "1000 calls/minute"},
            "azure": {"format": "JSON-LD", "model_repo_required": True, "twin_lifecycle": "Full DTDL support", "query_language": "Digital Twins Query Language"},
            "aveva": {"format": "AVEVA E3D / Unified", "integration": "AVEVA Connect", "data_types": ["Equipment", "Piping", "Structural"]},
            "nvidia_omniverse": {"format": "USD", "connector": "Revit/Omniverse", "real_time": True, "physics_simulation": True}
        }
        return configs.get(platform, {"format": "Generic JSON", "note": "Platform-specific configuration required"})
    def _check_twin_data_quality(self, data: Dict) -> Dict:
        elements = data.get("elements", [])
        checks = {
            "total_elements": len(elements),
            "with_geometry": len([e for e in elements if e.get("geometry")]),
            "with_properties": len([e for e in elements if e.get("properties")]),
            "with_relationships": len([e for e in elements if e.get("relationships")]),
            "unique_ids": len(set(e.get("id") for e in elements)),
            "duplicate_ids": len(elements) - len(set(e.get("id") for e in elements)),
            "missing_geometry": [e.get("id") for e in elements if not e.get("geometry")][:10]
        }
        checks["completeness_score"] = (checks["with_geometry"] / len(elements) * 100) if elements else 0
        return checks
    def _generate_api_payloads(self, operations: List[Dict], platform: str) -> List[Dict]:
        return [{"platform": platform, "operation": op} for op in operations[:5]]
    def _generate_sync_recommendations(self, quality: Dict, platform: str) -> List[str]:
        recs = []
        if quality.get("duplicate_ids", 0) > 0:
            recs.append("Resolve duplicate element IDs before sync")
        if quality.get("completeness_score", 100) < 80:
            recs.append("Add missing geometry to incomplete elements")
        return recs
    def _suggest_next_action(self, results: List[Dict], original_goal: str) -> Dict:
        """Suggest logical next step based on completed workflow"""
        completed_actions = [r["step"] for r in results]
        last_result = results[-1] if results else {}
    
        if "extract_quantities" in completed_actions and "procurement_optimizer" not in completed_actions:
            return {"suggested_action": "procurement_optimizer", "reason": "Quantities calculated - ready to source materials", "confidence": 0.95}
    
        if "parse_primavera_schedule" in completed_actions and "cash_flow_forecast" not in completed_actions:
            return {"suggested_action": "cash_flow_forecast", "reason": "Schedule loaded - can now project cash requirements", "confidence": 0.90}
    
        if "qa_qc_inspection" in completed_actions and last_result.get("status") == "success":
            defects = last_result.get("key_findings", {}).get("defects_found", 0)
            if defects > 0:
                # Back to generate_construction_report now that it runs and is
                # routed. It was briefly retargeted to process_document while
                # the action was parked, which was a weak suggestion — parsing
                # a document is a restart, not the next step after an
                # inspection. This names the action that actually produces the
                # formal report.
                return {"suggested_action": "generate_construction_report", "reason": f"{defects} defects found - generate formal QA report", "confidence": 0.88}
    
        if "forensic_delay_analysis" in completed_actions:
            return {"suggested_action": "claims_builder", "reason": "Delay analysis complete - prepare formal claim submission", "confidence": 0.92}
    
        if "process_specification_full" in completed_actions:
            return {"suggested_action": "submittal_log_generator", "reason": "Specifications parsed - extract all required submittals", "confidence": 0.85}
    
        if "tender_bid_analysis" in completed_actions:
            return {"suggested_action": "process_contract", "reason": "Bid selected - prepare contract with identified risks", "confidence": 0.80}
    
        if "carbon_footprint_calculator" in completed_actions:
            return {"suggested_action": "esg_sustainability_report", "reason": "Carbon calculated - generate full ESG disclosure", "confidence": 0.85}
    
        return {"suggested_action": "process_document", "reason": "Consolidate all findings into formal report", "confidence": 0.75}
    def _extract_key_findings(self, result: Dict) -> Dict:
        """Extract summary data from result for chaining"""
        return {
            "status": result.get("status"),
            "metrics": result.get("summary", {}),
            "risks_found": len(result.get("risks", [])) if isinstance(result.get("risks"), list) else 0,
            "cost_impact": result.get("cost_impact") or result.get("total_cost") or result.get("grand_total"),
            "schedule_impact": result.get("schedule_impact", {}).get("days", 0) if isinstance(result.get("schedule_impact"), dict) else 0,
            "defects_found": result.get("defects_found", 0),
            "approval_status": result.get("approval_status") or result.get("pass_fail")
        }
    def _consolidate_results(self, results: List[Dict]) -> Dict:
        """Create unified summary from multiple workflow steps"""
        total_cost_impact = sum([
            r.get("key_findings", {}).get("cost_impact", 0) or 0 
            for r in results 
            if isinstance(r.get("key_findings", {}).get("cost_impact"), (int, float))
        ])
    
        total_schedule_impact = sum([
            r.get("key_findings", {}).get("schedule_impact", 0) 
            for r in results
        ])
    
        all_risks = []
        for r in results:
            if "risk" in r.get("step", ""):
                all_risks.extend(r.get("result", {}).get("risks", []))
    
        return {
            "workflow_steps_completed": len(results),
            "total_cost_impact_usd": total_cost_impact,
            "total_schedule_impact_days": total_schedule_impact,
            "risks_identified": len(all_risks),
            "critical_issues": len([r for r in results if r.get("status") == "error"]),
            "success_rate": len([r for r in results if r.get("status") == "success"]) / len(results) if results else 0
        }
    async def _process_specification(self, file_path: str, params: Dict) -> Dict:
        # Route to the real specification pipeline.
        p = dict(params or {})
        p["file_path"] = file_path
        return await self.process_specification_full({"file_path": file_path}, p)
    async def _process_schedule(self, file_path: str, params: Dict) -> Dict:
        # Route to the real schedule parser (Primavera P6 / XER).
        p = dict(params or {})
        p["file_path"] = file_path
        return await self.parse_primavera_schedule({"file_path": file_path}, p)
    # NOTE: qa_inspection (a redundant wrapper that only defaulted `type` and
    # delegated to qa_qc_inspection) was deleted (W1) — it was never in the
    # dispatch handlers, so nothing reached it; qa_qc_inspection is the real,
    # reachable action and already handles the `type`/`trade` default.
    def _parse_defects(self, description: str) -> List[Dict[str, Any]]:
        """Keyword-extract defect candidates from an inspection description.

        Returns a list of {keyword, description, severity}. This is a
        keyword pass over the LLM-produced inspection text — not a vision
        model. Trades the call-site AttributeError for an honest, bounded
        capability. Empty list means "no defect keywords matched" (which
        is NOT the same as "no defects exist in the image").
        """
        if not description:
            return []
        text = description.lower()
        defects: List[Dict[str, Any]] = []
        seen = set()
        for trade_keywords in self._DEFECT_KEYWORDS.values():
            for kw, label, severity in trade_keywords:
                if kw in text and label not in seen:
                    seen.add(label)
                    defects.append({
                        "keyword": kw,
                        "description": label,
                        "severity": severity,
                    })
        return defects
    def _calculate_severity(self, defects: List[Dict[str, Any]]) -> float:
        """Aggregate defect severities into a 0-100 score (lower is better)."""
        weights = {"critical": 25.0, "major": 10.0, "minor": 3.0}
        score = sum(weights.get(d.get("severity", "minor"), 3.0) for d in defects)
        return round(min(score, 100.0), 1)
    def _check_compliance(
        self, defects: List[Dict[str, Any]], inspection_type: str
    ) -> Dict[str, Any]:
        """Roll up defects into a compliance verdict.

        Severities map to status: any 'critical' → 'non_compliant';
        any 'major' → 'conditional'; otherwise → 'compliant'.
        """
        severities = {d.get("severity") for d in defects}
        if "critical" in severities:
            status = "non_compliant"
        elif "major" in severities:
            status = "conditional"
        else:
            status = "compliant"
        return {
            "status": status,
            "inspection_type": inspection_type,
            "issues": [d["description"] for d in defects],
            "_note": (
                "Compliance status is a roll-up of keyword-matched defect "
                "severities — not a verdict against a specific code clause."
            ),
        }
    def _generate_recommendations(
        self, defects: List[Dict[str, Any]], inspection_type: str
    ) -> List[Dict[str, Any]]:
        """Per-defect recommendation lines."""
        actions = {
            "critical": "STOP work in this area. Engineer review required before any further activity.",
            "major": "Schedule repair before next concrete pour / progress step. Document with photos.",
            "minor": "Add to punch list; address before practical completion.",
        }
        recs = []
        for d in defects:
            severity = d.get("severity", "minor")
            recs.append({
                "defect": d.get("description", ""),
                "severity": severity,
                "action": actions.get(severity, actions["minor"]),
            })
        return recs
    async def track_progress(self, input_data: Any, params: Dict) -> Dict:
        """Compare as-built photos against BIM/design drawings"""
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        bim_file = data.get("bim_file") or p.get("bim_file")
        photo_files = data.get("photos") or p.get("photos", [])
        location = p.get("location", "unknown")
    
        if not isinstance(photo_files, list):
            photo_files = [photo_files] if photo_files else []
    
        results = []
        for photo in photo_files:
            comparison = await self._compare_photo_to_bim(photo, bim_file or "", location)
            results.append(comparison)
    
        # Refuse rather than report 0%. With no photographs, or no model
        # elements to compare against, the old code divided len(results) into
        # a zero numerator and returned progress_percentage 0 with
        # status:"success" — indistinguishable, on screen, from a genuine
        # finding that nothing has been built. That is the expensive direction
        # to be wrong in on a progress claim.
        if not photo_files:
            return {"status": "error", "error": (
                "No photographs supplied — progress cannot be compared against "
                "the model. Provide `photos`.")}
        # Refuse if no photograph was actually analysed. Otherwise a missing
        # image block yields zero detections on every photo, which flows
        # through as "0% built, high delay risk" — a manufactured finding, and
        # the most damaging one this action can produce.
        if not any(r.get("detection_ran") for r in results):
            reasons = {r.get("detection_error") for r in results if r.get("detection_error")}
            return {"status": "error", "error": (
                "No photograph could be analysed, so progress cannot be assessed: "
                + "; ".join(sorted(f for f in reasons if f))),
                "photos_analyzed": len(photo_files)}

        expected_total = sum(r.get("elements_expected", 0) for r in results)
        if not expected_total:
            return {"status": "error", "error": (
                "No model elements were resolved for this location, so nothing "
                "can be compared. " + (getattr(self, "_last_bim_query_error", None)
                                       or "Provide a `bim_file` IFC model.")),
                "photos_analyzed": len(photo_files)}

        # Element-level, not photo-level. Counting photos whose confidence
        # cleared 0.7 answers "how many pictures looked right", which is not a
        # progress measure; the model elements matched across all photographs
        # are.
        matched_ids, missing_ids = set(), set()
        for r in results:
            for el in r.get("matched", []):
                matched_ids.add(str(el.get("id") or el.get("name")))
            for el in r.get("missing", []):
                missing_ids.add(str(el.get("id") or el.get("name")))
        missing_ids -= matched_ids          # seen in any photo counts as seen
        observed = len(matched_ids) + len(missing_ids)

        return {
            "status": "success",
            "location": location,
            "photos_analyzed": len(photo_files),
            "progress_percentage": round(len(matched_ids) / observed * 100, 1) if observed else 0.0,
            "elements_found": len(matched_ids),
            "elements_missing": len(missing_ids),
            "basis": (
                f"{len(matched_ids)} of {observed} model elements identified across "
                f"{len(photo_files)} photograph(s) at {location}. Visual "
                "identification only — not a verified installation record."
            ),
            "deviations": [d for r in results for d in r.get("deviations", [])],
            "details": results,
            "delay_risk": self._assess_delay_risk(results)
        }

    def _assess_delay_risk(self, results: List[Dict]) -> Dict:
        """Schedule risk implied by what the photographs did and did not show.

        Returns an explicit `not_assessed` when there was nothing to compare.
        A risk function that answers "low" when it has no data is worse than
        one that refuses, because "low" is the answer a reader acts on.
        """
        expected = sum(r.get("elements_expected", 0) for r in results or [])
        if not expected:
            return {
                "level": "not_assessed",
                "reason": "No model elements were resolved, so no shortfall can be measured.",
                "elements_expected": 0,
            }
        matched = sum(len(r.get("matched", [])) for r in results)
        found_ratio = matched / expected
        if found_ratio >= 0.9:
            level, reason = "low", "Nearly all expected elements were identified."
        elif found_ratio >= 0.7:
            level, reason = "medium", "A minority of expected elements were not identified."
        else:
            level, reason = "high", "Most expected elements were not identified."
        return {
            "level": level,
            "reason": reason + (
                " Absence in a photograph is not proof of absence on site — "
                "coverage, occlusion and camera angle all produce the same signal."
            ),
            "elements_expected": expected,
            "elements_identified": matched,
            "identified_ratio": round(found_ratio, 3),
        }
    # NOTE: progress_tracking (a dead legacy method that only returned an
    # honest-error redirect to progress_tracker) was deleted (W1) — it was never
    # dispatched; progress_tracker is the real, reachable progress action.
    async def generate_construction_report(self, input_data: Any, params: Dict) -> Dict:
        """Generate comprehensive construction document report"""
        doc_result = await self.process_document(input_data, params)
    
        if doc_result.get("status") != "success":
            return doc_result
    
        # `.get`, not `[...]`. process_document dispatches by document type and
        # each branch returns a different key set: the drawing branch supplies
        # doc_type / detected_disciplines / total_pages / measurements / tables,
        # but the specification, report and contract branches do not. Indexing
        # them raised KeyError: 'doc_type' on every non-drawing document, which
        # is why this action was never routed.
        return {
            "status": "success",
            "report_type": "construction_analysis",
            "summary": {
                "document": doc_result.get("file_name"),
                "type": doc_result.get("doc_type", "unknown"),
                "disciplines": doc_result.get("detected_disciplines", []),
                "pages": doc_result.get("total_pages"),
                "measurements_found": len(doc_result.get("measurements") or []),
                "tables_found": len(doc_result.get("tables") or []),
            },
            "cost_summary": doc_result.get("cost_estimate"),
            "recommendations": self._generate_doc_recommendations(doc_result),
            "raw": doc_result if params.get("include_raw") else None
        }

    def _generate_doc_recommendations(self, doc_result: Dict) -> List[str]:
        """What to do next about THIS document, derived from what was found.

        Every line is conditional on a field the parse actually produced. No
        default advice: a document that yields nothing to say gets an explicit
        "nothing actionable was extracted" rather than filler, because generic
        recommendations under a construction-analysis heading read as findings.
        """
        recs: List[str] = []
        doc_type = doc_result.get("doc_type", "unknown")

        confidence = doc_result.get("confidence")
        if isinstance(confidence, dict):
            overall = confidence.get("overall")
            if isinstance(overall, (int, float)) and overall < 0.7:
                caveats = "; ".join(confidence.get("caveats") or []) or "low field coverage"
                recs.append(
                    f"Extraction confidence is {overall:.0%} ({caveats}) — verify "
                    "this document against the source before relying on figures below."
                )

        risks = doc_result.get("auto_risks") or []
        if risks:
            recs.append(
                f"{len(risks)} risk(s) were flagged from the document — review and "
                "transfer any that are live to the risk register."
            )
        if doc_result.get("revision"):
            recs.append(
                f"Confirm revision {doc_result['revision']} is the current issue "
                "before using this for construction."
            )
        elif doc_type == "drawing":
            recs.append(
                "No revision was found in the title block — confirm the issue "
                "status before use."
            )
        if not doc_result.get("scale") and doc_type == "drawing":
            recs.append(
                "No drawing scale was detected — do not scale quantities off this "
                "sheet without confirming it."
            )
        if doc_result.get("measurements"):
            recs.append(
                f"{len(doc_result['measurements'])} measurement(s) extracted — "
                "cross-check against the BOQ before pricing."
            )
        if doc_result.get("grade_requirements") or doc_result.get("standards_referenced"):
            recs.append(
                "Specified grades/standards were found — confirm the procured "
                "materials and submittals match them."
            )
        if doc_result.get("compliance_flags"):
            recs.append(
                f"{len(doc_result['compliance_flags'])} compliance flag(s) raised "
                "— resolve before submittal approval."
            )
        if doc_result.get("cost_estimate"):
            recs.append(
                "A cost estimate was derived from this document — treat it as "
                "indicative until the rates are confirmed against the contract."
            )

        if not recs:
            recs.append(
                f"Nothing actionable was extracted from this {doc_type}. It parsed "
                "successfully but produced no measurements, specifications, risks "
                "or compliance flags to act on."
            )
        return recs

    async def spec_analyze(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to SpecAnalyzerBlock: extract grades, materials, compliance."""
        block = self._resolve_block("spec_analyzer")
        if block is None:
            return {"status": "error", "error": "spec_analyzer block unavailable"}
        return await block.process(input_data, params)
    async def sympy_reason(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to SymPyReasoningBlock: variance analysis + recommendations."""
        block = self._resolve_block("sympy_reasoning")
        if block is None:
            return {"status": "error", "error": "sympy_reasoning block unavailable"}
        return await block.process(input_data, params)
    async def orchestrate(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to SmartOrchestratorBlock: keyword → action routing."""
        block = self._resolve_block("smart_orchestrator")
        if block is None:
            return {"status": "error", "error": "smart_orchestrator block unavailable"}
        return await block.process(input_data, params)
    async def formula_execute(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to FormulaExecutorV2Block: LLM code-gen formula execution."""
        block = self._resolve_block("formula_executor_v2")
        if block is None:
            return {"status": "error", "error": "formula_executor_v2 block unavailable"}
        return await block.process(input_data, params)

    async def construction_calc(self, input_data: Any, params: Dict) -> Dict:
        """Run the deterministic formula registry.

        The smart-orchestrator hat is told to call ``construction`` with
        ``params.action=construction_calc``. Without this alias ``route()``
        returned ``Unknown action: construction_calc`` and leftover L6
        (14.5 × 3.2 × 1.75 bank volume) never emitted 81.2 m³.
        """
        from app.lib import construction_formulas as _cf

        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        calc = (
            p.get("calculation") or p.get("name") or p.get("calculator")
            or data.get("calculation") or data.get("name") or data.get("calculator")
        )
        nested = p.get("params") if isinstance(p.get("params"), dict) else None
        if nested is None:
            nested = data.get("params") if isinstance(data.get("params"), dict) else None
        skip = {
            "action", "calculation", "name", "calculator", "params",
            "block", "unit", "formula",
        }
        flat = {k: v for k, v in {**data, **p}.items() if k not in skip}
        calc_params = dict(nested or {})
        for k, v in flat.items():
            calc_params.setdefault(k, v)

        aliases = {
            "length": "length_m", "width": "width_m", "depth": "depth_m",
            "height": "depth_m", "height_m": "depth_m",
            "thickness": "thickness_m",
            "l": "length_m", "w": "width_m", "d": "depth_m",
        }
        for src, dst in aliases.items():
            if src in calc_params and dst not in calc_params:
                calc_params[dst] = calc_params[src]

        # Live UI pack E4: leftover L6 aliased unnamed L×W×D to
        # excavation_volume (bank 900, waste missing). A concrete/raft ask
        # pins concrete_volume and the documented 5% waste → 945 m3.
        from app.lib.construction_formulas_quantities import (
            resolve_concrete_volume_calc,
            resolve_fw_calc,
        )
        text_blob = " ".join(
            str(x) for x in (
                p.get("text"), data.get("text"),
                p.get("formula"), data.get("formula"),
                calc,
            ) if x
        )
        orig_calc = None if calc is None else str(calc)
        calc, calc_params = resolve_concrete_volume_calc(
            orig_calc, calc_params, text_blob,
        )
        calc, calc_params = resolve_fw_calc(
            calc, calc_params, text_blob, original_name=orig_calc,
        )

        if not calc:
            formula = str(p.get("formula") or data.get("formula") or "")
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", formula)]
            if len(nums) >= 3:
                calc = "excavation_volume"
                calc_params.setdefault("length_m", nums[0])
                calc_params.setdefault("width_m", nums[1])
                calc_params.setdefault("depth_m", nums[2])
            else:
                keys = {str(k).lower() for k in calc_params}
                has_l = any("length" in k or k in {"l"} for k in keys)
                has_w = any("width" in k or k in {"w"} for k in keys)
                has_d = any("depth" in k or "height" in k or k in {"d"} for k in keys)
                if has_l and has_w and has_d:
                    calc = "excavation_volume"

        for junk in ("formula", "unit", "block", "action"):
            calc_params.pop(junk, None)

        if not calc:
            return {
                "status": "error",
                "error": "construction_calc needs a calculation name "
                         "(e.g. excavation_volume) or length/width/depth.",
                "available": _cf.available_calculations(),
            }
        return _cf.run_calculation(str(calc), calc_params)
    async def bim_extract(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to BIMExtractorBlock: IFC element + quantity extraction."""
        block = self._resolve_block("bim_extractor")
        if block is None:
            return {"status": "error", "error": "bim_extractor block unavailable"}
        return await block.process(input_data, params)
    async def learn(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to LearningEngineBlock: record corrections + promote tiers."""
        block = self._resolve_block("learning_engine")
        if block is None:
            return {"status": "error", "error": "learning_engine block unavailable"}
        return await block.process(input_data, params)
    async def recommend(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to RecommendationTemplateBlock: rule-based recommendations."""
        block = self._resolve_block("recommendation_template")
        if block is None:
            return {"status": "error", "error": "recommendation_template block unavailable"}
        return await block.process(input_data, params)
    @staticmethod
    def _detect_project_type_from_brief(brief: str) -> str:
        """Keyword heuristic to infer project_type from a free-text brief."""
        text = (brief or "").lower()
        if any(k in text for k in ("data center", "datacenter", "data hall")):
            return "data_center"
        if any(k in text for k in ("solar", "photovoltaic", " pv ", "pv plant", "pv farm")):
            return "solar_plant"
        if "wind" in text:
            return "wind_farm"
        if any(k in text for k in (
            "office", "residential", "hospital", "school", "tower",
            # F39: "two-storey villa" fell through to the data_center
            # default -- a data-hall WBS silently applied to a house.
            "villa", "apartment", "house", "hotel", "mall", "warehouse",
            "mosque", "clinic", "storey", "story building",
        )):
            return "building"
        if any(k in text for k in (
            "road", "bridge", "highway", "rail", "tunnel", "utility corridor",
            "community", "gated", "boulevard", "reservoir", "access-gated",
            "milestone 1", "pwps",
        )):
            return "infrastructure"
        # Unrecognised brief: the GENERIC building template, never the
        # data-center one (that default stamped hyperscale WBS items onto
        # unrelated projects).
        return "building"
    async def _status(self, input_data: Any, params: Dict) -> Dict:
        # Discover available actions by calling the active route() with a
        # sentinel — its handlers dict is the source of truth. Cheaper than
        # duplicating the action list here (which inevitably drifts).
        actions: List[str] = []
        try:
            # route() builds its handlers dict on every call; trigger it with
            # an unknown action and pull known_actions out of the error.
            sentinel = await self.route("__list__", None, {})
            actions = sentinel.get("known_actions", []) if isinstance(sentinel, dict) else []
        except Exception:
            actions = []
        return {
            "status": "success",
            "container": self.name,
            "version": self.version,
            "actions_available": actions,
            "action_count": len(actions),
        }
    async def route(self, action: str, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        action = data.get("action") or p.get("action") or action

        if not action:
            return {"status": "error", "error": "No action specified"}

        # Workflow-template step types (payment_cert, float_analysis, …)
        # resolve to canonical ConstructionContainer actions.
        from app.core.cm_step_aliases import resolve_action
        action = resolve_action(action)

        handlers = {
            "chat": self.chat,
            "process_document": self.process_document,
            "qa_qc_inspection": self.qa_qc_inspection,
            "wir_form": self.wir_form,
            "inspection_request": self.wir_form,
            "job_requisition": self.job_requisition,
            "safety_briefing": self.safety_briefing,
            "rfp_draft": self.rfp_draft,
            "rfp_management": self.rfp_draft,
            "extract_quantities": self.extract_quantities,
            "estimate_costs": self.estimate_costs,
            "progress_tracker": self.progress_tracker,
            "evm_calculate": self.evm_calculate,
            "look_ahead": self.look_ahead,
            "bim_analysis": self.bim_analysis,
            "parse_primavera_schedule": self.parse_primavera_schedule,
            "process_contract": self.process_contract,
            "process_specification_full": self.process_specification_full,
            "change_order_impact": self.change_order_impact,
            "rfi_generator": self.rfi_generator,
            "cde_post_rfi": self.cde_post_rfi,
            "cde_poll_events": self.cde_poll_events,
            "safety_compliance_audit": self.safety_compliance_audit,
            "carbon_footprint_calculator": self.carbon_footprint_calculator,
            "procurement_list_generator": self.procurement_list_generator,
            "as_built_deviation_report": self.as_built_deviation_report,
            "warranty_maintenance_schedule": self.warranty_maintenance_schedule,
            "risk_register_auto_populate": self.risk_register_auto_populate,
            "submittal_log_generator": self.submittal_log_generator,
            "payment_certificate": self.payment_certificate,
            "bim_clash_detection": self.bim_clash_detection,
            "daily_site_report": self.daily_site_report,
            "value_engineering": self.value_engineering,
            "commissioning_checklist": self.commissioning_checklist,
            "resource_histogram": self.resource_histogram,
            "claims_builder": self.claims_builder,
            "tender_bid_analysis": self.tender_bid_analysis,
            "variation_order_manager": self.variation_order_manager,
            "forensic_delay_analysis": self.forensic_delay_analysis,
            "cash_flow_forecast": self.cash_flow_forecast,
            "procurement_optimizer": self.procurement_optimizer,
            "procurement_analysis": self.procurement_analysis,
            "esg_sustainability_report": self.esg_sustainability_report,
            "om_manual_generator": self.om_manual_generator,
            "digital_twin_sync": self.digital_twin_sync,
            "intelligent_workflow": self.intelligent_workflow,
            "auto_pipeline": self.auto_pipeline,
            "health_check": self.health_check,
            # Generative scheduling
            "generate_wbs": self.generate_wbs,
            # Week-1 Intelligence Blocks
            "boq_process": self.boq_process,
            "spec_analyze": self.spec_analyze,
            "sympy_reason": self.sympy_reason,
            # Week-2 Domain Blocks
            "drawing_qto": self.drawing_qto,
            "primavera_parse": self.primavera_parse,
            "orchestrate": self.orchestrate,
            # Week-3 Intelligence Blocks
            "jetson_dispatch": self.jetson_dispatch,
            "formula_execute": self.formula_execute,
            # Hat prompt names the synthetic calculator as a container action.
            "construction_calc": self.construction_calc,
            "bim_extract": self.bim_extract,
            # LLM often names the standalone block; the container action is
            # bim_extract. Live leftover-hat L2 UI: "bim_extractor is not a
            # known action" then stall.
            "bim_extractor": self.bim_extract,
            # Week-4 Intelligence Blocks
            "learn": self.learn,
            "benchmark_lookup": self.benchmark_lookup,
            "recommend": self.recommend,
            # Short-name aliases — older callers + the now-removed initial route()
            # used these names. Keep them mapped so existing integrations keep
            # working, even though the canonical names are above.
            "cost_estimate": self.generate_cost_estimate,
            "analyze_spec": self.analyze_spec_section,
            "schedule_risk": self.analyze_schedule_risk,
            "contract_review": self.process_contract,
            "safety_audit": self.safety_compliance_audit,
            "carbon_report": self.generate_carbon_report,
            "procurement": self.procurement_analysis,
            "status": self._status,
            # Audit 2026-08-12. Three public actions existed that route() could
            # not dispatch. Only ONE of them was safe to wire.
            #
            # `extract_measurements` is real and runs: it pulls measurements out
            # of a drawing/document and feeds `extract_quantities` (which takes a
            # measurements LIST, so the two are pipeline stages, not twins).
            #
            # `generate_construction_report` and `track_progress` were unroutable
            # because they were unrunnable: between them they called four helpers
            # that had never been defined here, and the photo/BIM chain under
            # track_progress also read a key (`elements`) the BIM extractor has
            # never returned. All four helpers are now implemented and the chain
            # is fixed, so both are wired. tests/test_construction_actions_
            # reachable.py enforces the rule that made this safe: parking broken
            # code is allowed, routing it is not.
            "extract_measurements": self.extract_measurements,
            "generate_construction_report": self.generate_construction_report,
            "track_progress": self.track_progress,
        }

        handler = handlers.get(action)
        if not handler:
            from app.core.procedure_actions import is_procedure_action

            if is_procedure_action(action):
                return await self._handle_procedure_action(
                    action, input_data, params, handlers
                )
            return {
                "status": "error",
                "error": f"Unknown action: {action}",
                "known_actions": sorted(handlers.keys()),
            }

        result = await handler(input_data, params)

        # WRITE side of the construction learning loop (W6). Every container
        # action passes through here, so this is the one place that sees a
        # completed action together with its name and result.
        #
        # `learning_capture` described itself as "the wiring" but had no call
        # site, so FORK_LEARNING_CAPTURE=1 recorded nothing. Capturing actuals
        # is invisible to users and changes no answer, so it can run from day
        # one and accumulate the dataset; the READ side stays disconnected
        # until it can state its provenance in the answer.
        try:
            from app.core import learning_capture

            learning_capture.capture_from_action(action, result, params)
        except Exception:  # noqa: BLE001 — learning must never break an action
            logger.warning("learning capture failed for %s", action, exc_info=True)

        return result

    def get_actions(self) -> Dict[str, Any]:
        return {
            "chat": self.chat,
            "process_document": self.process_document,
            "qa_qc_inspection": self.qa_qc_inspection,
            "wir_form": self.wir_form,
            "inspection_request": self.wir_form,
            "job_requisition": self.job_requisition,
            "safety_briefing": self.safety_briefing,
            "rfp_draft": self.rfp_draft,
            "rfp_management": self.rfp_draft,
            "extract_quantities": self.extract_quantities,
            "estimate_costs": self.estimate_costs,
            "progress_tracker": self.progress_tracker,
            "evm_calculate": self.evm_calculate,
            "look_ahead": self.look_ahead,
            "bim_analysis": self.bim_analysis,
            "parse_primavera_schedule": self.parse_primavera_schedule,
            "process_contract": self.process_contract,
            "process_specification_full": self.process_specification_full,
            "change_order_impact": self.change_order_impact,
            "rfi_generator": self.rfi_generator,
            "cde_post_rfi": self.cde_post_rfi,
            "cde_poll_events": self.cde_poll_events,
            "safety_compliance_audit": self.safety_compliance_audit,
            "carbon_footprint_calculator": self.carbon_footprint_calculator,
            "procurement_list_generator": self.procurement_list_generator,
            "as_built_deviation_report": self.as_built_deviation_report,
            "warranty_maintenance_schedule": self.warranty_maintenance_schedule,
            "risk_register_auto_populate": self.risk_register_auto_populate,
            "submittal_log_generator": self.submittal_log_generator,
            "payment_certificate": self.payment_certificate,
            "bim_clash_detection": self.bim_clash_detection,
            "daily_site_report": self.daily_site_report,
            "value_engineering": self.value_engineering,
            "commissioning_checklist": self.commissioning_checklist,
            "resource_histogram": self.resource_histogram,
            "claims_builder": self.claims_builder,
            "tender_bid_analysis": self.tender_bid_analysis,
            "variation_order_manager": self.variation_order_manager,
            "forensic_delay_analysis": self.forensic_delay_analysis,
            "cash_flow_forecast": self.cash_flow_forecast,
            "procurement_optimizer": self.procurement_optimizer,
            "procurement_analysis": self.procurement_analysis,
            "esg_sustainability_report": self.esg_sustainability_report,
            "om_manual_generator": self.om_manual_generator,
            "digital_twin_sync": self.digital_twin_sync,
            "intelligent_workflow": self.intelligent_workflow,
            "auto_pipeline": self.auto_pipeline,
            "health_check": self.health_check,
            # Generative scheduling
            "generate_wbs": self.generate_wbs,
            # Week-1 Intelligence Blocks
            "boq_process": self.boq_process,
            "spec_analyze": self.spec_analyze,
            "sympy_reason": self.sympy_reason,
            # Week-2 Domain Blocks
            "drawing_qto": self.drawing_qto,
            "primavera_parse": self.primavera_parse,
            "orchestrate": self.orchestrate,
            # Week-3 Intelligence Blocks
            "jetson_dispatch": self.jetson_dispatch,
            "formula_execute": self.formula_execute,
            "construction_calc": self.construction_calc,
            "bim_extract": self.bim_extract,
            "bim_extractor": self.bim_extract,
            # Week-4 Intelligence Blocks
            "learn": self.learn,
            "benchmark_lookup": self.benchmark_lookup,
            "recommend": self.recommend,
        }
