"""Tests for construction container YOLO compose (Plan Task 2.3).

Verifies that when image_block returns safety_qaqc detections, they get
mapped into the existing hazard / defect dict shapes and composed with
the text-keyword path (dedup by type/description)."""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from app.containers.construction import ConstructionContainer


class _FakeImageBlock:
    """Returns a canned image-block result with controlled description + safety_qaqc."""

    def __init__(self, description: str = "", safety_qaqc=None) -> None:
        self.description = description
        self.safety_qaqc = safety_qaqc or []
        self.last_input: Dict[str, Any] = {}
        self.last_params: Dict[str, Any] = {}

    async def execute(self, input_data: Any, params: Dict = None) -> Dict:
        self.last_input = dict(input_data or {})
        self.last_params = dict(params or {})
        return {
            "status": "success",
            "result": {
                "description": self.description,
                "extracted_text": self.description,
                "metadata": {},
                "detections": [],
                "summary_by_class": {},
                "safety_qaqc": self.safety_qaqc,
            },
        }


def _container_with(image_block):
    c = ConstructionContainer()
    c._dependencies = {"image": image_block}
    return c


@pytest.mark.asyncio
async def test_safety_audit_adds_yolo_classes():
    block = _FakeImageBlock(
        description="",
        safety_qaqc=[
            {"class_id": 0, "class": "no_hardhat", "category": "safety",
             "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"class_id": 3, "class": "concrete_crack", "category": "qaqc",
             "confidence": 0.8, "bbox": [0, 0, 10, 10]},  # NOT a safety class — filtered out
        ],
    )
    c = _container_with(block)
    out = await c.safety_compliance_audit({"file_path": "/tmp/x.jpg"}, {"audit_type": "general"})

    types = {h["type"] for h in out["violations"]}
    assert "no_hardhat" in types
    # concrete_crack is a QAQC class; should not surface as a safety hazard
    assert not any(h.get("type") == "concrete_crack" for h in out["violations"])
    sources = {h.get("source") for h in out["violations"]}
    assert "yolo" in sources
    # image block was called with mode=safety_qaqc
    assert block.last_params.get("mode") == "safety_qaqc"


@pytest.mark.asyncio
async def test_qaqc_inspection_adds_yolo_defects():
    block = _FakeImageBlock(
        description="",
        safety_qaqc=[
            {"class_id": 3, "class": "concrete_crack", "category": "qaqc",
             "confidence": 0.8, "bbox": [0, 0, 10, 10]},
            {"class_id": 5, "class": "rebar_correct_inspection", "category": "qaqc",
             "confidence": 0.7, "bbox": [0, 0, 10, 10]},  # NOT a defect — filtered
            {"class_id": 0, "class": "no_hardhat", "category": "safety",
             "confidence": 0.9, "bbox": [0, 0, 10, 10]},  # NOT a defect — filtered
        ],
    )
    c = _container_with(block)
    out = await c.qa_qc_inspection({"file_path": "/tmp/x.jpg"}, {"type": "concrete"})

    descs = {d["description"] for d in out["defects"]}
    assert "concrete_crack" in descs
    assert "rebar_correct_inspection" not in descs
    assert "no_hardhat" not in descs
    assert block.last_params.get("mode") == "safety_qaqc"


@pytest.mark.asyncio
async def test_no_yolo_output_keeps_text_keyword_path_working():
    """When image_block returns no safety_qaqc, the legacy keyword-text
    parsing path still produces hazards/defects unchanged."""
    block = _FakeImageBlock(
        description="Missing PPE detected at corner.",
        safety_qaqc=[],
    )
    c = _container_with(block)
    out = await c.safety_compliance_audit({"file_path": "/tmp/x.jpg"}, {"audit_type": "general"})

    # text path produced a hazard via missing_ppe keyword
    types = {h["type"] for h in out["violations"]}
    assert "missing_ppe" in types
    # none of them have source=yolo (came from text path)
    for h in out["violations"]:
        assert h.get("source") != "yolo"
