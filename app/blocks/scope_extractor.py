"""Scope Extractor block — roadmap-named shim over the document engine.

The L2 Schedule Engine roadmap expected ``app/blocks/scope_extractor.py``. The
platform extracts scope, requirements, constraints, and schedule targets via
``app.blocks.document_engine.DocumentEngineBlock``. This block is a thin
wrapper that focuses the output into a schedule-scope shape.
"""

from typing import Any, Dict

from app.core.universal_base import UniversalBlock


class ScopeExtractorBlock(UniversalBlock):
    auto_validate = False
    name = "scope_extractor"
    version = "1.0.0"
    description = "Extract schedule-relevant scope from RFP/BOD/spec documents"
    layer = 3
    tags = ["domain", "construction", "schedule", "scope", "documents"]
    requires = ["pdf", "ocr"]

    default_config = {
        "output_format": "json",
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Extract scope from one or more documents.

        Accepts a file path in ``file_path`` / ``pdf_path`` / ``docx_path`` /
        ``xlsx_path`` (in either ``input_data`` or ``params``), or a brief in
        ``brief`` / ``text`` when no file is available.
        """
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        import os

        file_path = (
            data.get("file_path")
            or params.get("file_path")
            or data.get("pdf_path")
            or params.get("pdf_path")
            or data.get("docx_path")
            or params.get("docx_path")
            or data.get("xlsx_path")
            or params.get("xlsx_path")
        )
        brief = (
            data.get("brief")
            or params.get("brief")
            or data.get("text")
            or params.get("text")
            or ""
        )

        try:
            from app.blocks.document_engine import DocumentEngineBlock

            engine = DocumentEngineBlock()
            if file_path:
                ext = os.path.splitext(file_path)[1].lower()
                doc_input: Dict[str, Any] = {}
                if ext == ".pdf":
                    doc_input["pdf_path"] = file_path
                elif ext in (".docx", ".doc"):
                    doc_input["docx_path"] = file_path
                elif ext in (".xlsx", ".xls"):
                    doc_input["xlsx_path"] = file_path
                else:
                    doc_input["text"] = file_path
                doc_result = await engine.process(
                    doc_input,
                    params={"action": "extract_scope"},
                )
            else:
                doc_result = await engine.process(
                    {"text": brief},
                    params={"action": "extract_scope"},
                )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"Scope extraction failed: {exc}"}

        if doc_result.get("status") != "success":
            return doc_result

        # Focus the document-engine output into a schedule-scope shape.
        # F37: this shim used to read keys the engine never emits
        # ("summary", downstream.schedule_engine.procurement_lead_times,
        # .target_milestones) so every field came back empty while the
        # engine had ACTUALLY extracted equipment specs and targets -- a
        # wrap-around green. Map the keys the engine really returns.
        downstream = doc_result.get("downstream") or {}
        schedule_feed = downstream.get("schedule_engine") or {}
        requirements = doc_result.get("requirements", [])
        constraints = doc_result.get("constraints", [])
        schedule_targets = doc_result.get("schedule_targets", [])
        equipment = [
            s for s in doc_result.get("equipment_specs", [])
            if s.get("source") != "ontology_default"
        ]
        risks = doc_result.get("risks", [])
        n_docs = doc_result.get("documents_parsed", 0)
        scope = {
            "status": "success",
            "summary": (
                f"{len(requirements)} requirements, {len(constraints)} "
                f"constraints, {len(schedule_targets)} schedule targets, "
                f"{len(equipment)} equipment lead times, {len(risks)} risks "
                f"from {n_docs} document(s)"
            ),
            "requirements": requirements,
            "constraints": constraints,
            "schedule_targets": schedule_targets,
            "equipment_lead_times": equipment,
            "target_milestones": (
                schedule_feed.get("target_milestones")
                or [t for t in schedule_targets if t.get("month") or t.get("year")]
            ),
            "procurement_activities": schedule_feed.get("procurement_activities", []),
            "risks": risks,
            "documents_parsed": n_docs,
        }
        if file_path:
            scope["source_file"] = file_path
        return scope
