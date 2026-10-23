"""Shared schedule-feed extraction: run document_engine over RFP/BOD documents
and distil the schedule feed (real equipment lead times + target milestones).

One execution surface: both the `schedule-from-document` export endpoint and the
`extract_document` predefined-reasoning plan step call THIS — neither duplicates
the document_engine invocation or the distillation rules.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

_EXT_KEY = {".pdf": "pdf_path", ".docx": "docx_path", ".doc": "docx_path",
            ".xlsx": "xlsx_path", ".xls": "xlsx_path"}


async def extract_schedule_feed(document_ids: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (procurement_lead_times, target_milestones) distilled from the
    given documents via document_engine. Ontology-default lead times are dropped
    (never inject a fabricated default as a real lead time). Milestones are
    deduped. Documents that fail to parse are skipped, not fatal.
    """
    from app.blocks.document_engine import DocumentEngineBlock
    from app.core import projects as projects_store

    lead_map: Dict[str, int] = {}
    milestones: List[Dict[str, Any]] = []
    seen_ms: set = set()
    for did in document_ids or []:
        doc = projects_store.get_document(did)
        if not doc or not doc.get("file_path"):
            continue
        ext = os.path.splitext(doc.get("original_name") or doc["file_path"])[1].lower()
        key = _EXT_KEY.get(ext, "pdf_path")
        try:
            res = await DocumentEngineBlock().process({key: doc["file_path"]})
        except Exception:
            continue
        if not isinstance(res, dict) or res.get("status") != "success":
            continue
        for spec in res.get("equipment_specs") or []:
            if spec.get("source") == "ontology_default":
                continue
            equip = spec.get("equipment")
            try:
                days = int(spec.get("lead_time_days") or 0)
            except (TypeError, ValueError):
                days = 0
            if equip and days > 0:
                lead_map[equip] = max(lead_map.get(equip, 0), days)
        sched = (res.get("downstream") or {}).get("schedule_engine") or {}
        for m in sched.get("milestones") or []:
            k = (m.get("name"), m.get("target_date"))
            if k in seen_ms:
                continue
            seen_ms.add(k)
            milestones.append({"name": m.get("name"), "target_date": m.get("target_date")})
    lead_times = [{"equipment": k, "lead_time_days": v} for k, v in lead_map.items()]
    return lead_times, milestones
