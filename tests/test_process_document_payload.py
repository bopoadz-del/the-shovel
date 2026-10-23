"""process_document must actually read the document, not just return a shape.

Audit bar 3, 2026-08-12. `process_document` is the entry point for the whole
Documents capability — twelve container actions route through or beside it — and
it had NO payload coverage. Replacing its entire body with

    return {"status": "success", "file_name": "", "doc_type": "unknown",
            "detected_disciplines": [], "total_pages": 0, "measurements": [],
            "tables": [], "specifications": [], "confidence": {}}

left 148 tests green.

Every existing reference to `process_document` in tests/ is a string in a
routing table, an entry in a feature manifest, or an anti-hijack guard that
wires `object()` in its place. Those prove the ROUTER reaches it. None proves it
reads a file. So a document parser could have stopped parsing and the suite
would have stayed green.

These tests assert extracted VALUES that can only come from opening the file.
"""
from __future__ import annotations

import pytest

from scripts.synthetic_fixtures import build_spec_section_txt


@pytest.fixture(scope="module")
def spec_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("docpayload") / "synthetic_spec_section_03_concrete.txt"
    build_spec_section_txt(path)
    return path


@pytest.fixture
def container():
    from app.blocks.spec_analyzer import SpecAnalyzerBlock
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    # spec_analyzer registers via CEREBRUM_DOMAIN_KITS=construction; wire it
    # directly so this runs in the virgin profile too.
    c.wire("spec_analyzer", SpecAnalyzerBlock())
    return c


@pytest.mark.asyncio
async def test_process_document_extracts_values_from_the_file(container, spec_file):
    """The concrete grade is in the file and nowhere else.

    C32/40 cannot appear in the output unless the document was opened, read and
    parsed — which is exactly what a shape-only assertion cannot establish.
    """
    import json

    result = await container.process_document({"file_path": str(spec_file)}, {})

    assert result.get("status") == "success", result
    blob = json.dumps(result, default=str)
    assert "C32/40" in blob or "32/40" in blob, (
        "the concrete grade from the specification did not reach the output — "
        f"process_document returned keys {sorted(result)} and produced no grade"
    )
    # Something must have been extracted; an all-empty success is the failure
    # mode this file exists to catch.
    populated = [
        k for k, v in result.items()
        if not k.startswith("_") and isinstance(v, (list, dict)) and len(v) > 0
    ]
    assert populated, f"every extracted field is empty: {sorted(result)}"


@pytest.mark.asyncio
async def test_process_document_reports_the_file_it_read(container, spec_file):
    result = await container.process_document({"file_path": str(spec_file)}, {})
    assert result.get("file_name") == spec_file.name, (
        f"expected file_name {spec_file.name!r}, got {result.get('file_name')!r}"
    )


@pytest.mark.asyncio
async def test_process_document_refuses_honestly_without_a_file(container):
    """No file must be an explicit error, never an empty success.

    An empty success here is the shape that makes a gutted parser invisible.
    """
    result = await container.process_document({}, {})
    assert result.get("status") == "error"
    assert "No file provided" in result.get("error", "")


@pytest.mark.asyncio
async def test_route_reaches_the_same_payload(container, spec_file):
    """The router path must produce the parsed document too, not just the
    direct call — routing tests cover the dispatch, this covers the result."""
    import json

    result = await container.route(
        "process_document", {"file_path": str(spec_file)}, {"action": "process_document"}
    )
    assert result.get("status") == "success", result
    assert "32/40" in json.dumps(result, default=str)
