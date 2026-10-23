"""Regression tests for construction photo-based safety and QA/QC detection.

These tests stub the image block so they run without Tesseract or
ultralytics, but they verify that the construction container correctly:

1. Passes a file path (not the legacy ``image_path`` key) to the image block.
2. Parses safety hazards and QA/QC defects from the returned description.
3. Returns the correct overall verdicts/severity.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from app.containers.construction import ConstructionContainer


class _FakeImageBlock:
    """Returns a canned description containing known hazard/defect keywords."""

    def __init__(self, description: str) -> None:
        self.description = description
        self.last_input: Dict[str, Any] = {}

    async def execute(self, input_data: Any, params: Dict = None) -> Dict:
        self.last_input = dict(input_data or {})
        return {
            "status": "success",
            "result": {
                "description": self.description,
                "extracted_text": self.description,
                "metadata": {},
                "detections": [],
                "summary_by_class": {},
            },
        }


@pytest.fixture
def container_with_photo(monkeypatch):
    """Container whose image-block dependency returns a hazard/defect description."""
    container = ConstructionContainer()
    description = (
        "Site photo shows missing PPE and an exposed edge near the slab. "
        "Concrete surface has a crack and honeycombing."
    )
    fake_image = _FakeImageBlock(description)
    monkeypatch.setattr(container, "_dependencies", {"image": fake_image})
    return container, fake_image


@pytest.mark.asyncio
async def test_safety_audit_detects_hazards_and_uses_file_path(container_with_photo):
    container, fake_image = container_with_photo

    result = await container.safety_compliance_audit(
        {"file_path": "/tmp/site.jpg"}, {"audit_type": "general"}
    )

    assert result["status"] == "success"
    assert result["overall_compliance"] == "fail"
    assert result["violations_found"] >= 2
    types = {v["type"] for v in result["violations"]}
    assert "missing_ppe" in types
    assert "fall_hazard" in types
    assert fake_image.last_input.get("file_path") == "/tmp/site.jpg"
    assert "image_path" not in fake_image.last_input


@pytest.mark.asyncio
async def test_qaqc_inspection_detects_defects_and_uses_file_path(container_with_photo):
    container, fake_image = container_with_photo

    result = await container.qa_qc_inspection(
        {"file_path": "/tmp/concrete.jpg"}, {"type": "concrete"}
    )

    assert result["status"] == "success"
    assert result["pass_fail"] == "FAIL"
    assert result["defects_found"] >= 2
    labels = {d["description"] for d in result["defects"]}
    assert "Crack visible" in labels
    assert "Honeycombing / segregation" in labels
    assert fake_image.last_input.get("file_path") == "/tmp/concrete.jpg"
    assert "image_path" not in fake_image.last_input
