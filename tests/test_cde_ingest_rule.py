"""CDE ingest obeys the ingest rules (docs/INGEST_EXCLUSION_RULE.md).

Text formats, site photos and spreadsheets all pass the CDE format gate
(spreadsheets are working documents — owner ruling 2026-11-06 — they are
only excluded from RAG chunking downstream, in ingest_status.classify).
"""
from __future__ import annotations

import pytest

from app.core.cde.ingest import CdeError, ingest_cde_bytes


@pytest.mark.parametrize("filename", ["plan.dwg", "drone.mp4"])
def test_ingest_cde_bytes_refuses_a_non_ingestible_format(filename):
    with pytest.raises(CdeError):
        ingest_cde_bytes(
            "proj-1",
            original_name=filename,
            payload=b"\x00" * 64,
            cde_document_id="cde-1",
            vendor="acontex",
            cde_project_id="cde-proj",
            eager_index=False,
        )


@pytest.mark.parametrize("filename", ["notes.txt", "site.jpg", "boq.xlsx"])
def test_ingest_cde_bytes_text_photos_and_spreadsheets_pass_the_gate(monkeypatch, tmp_path, filename):
    """Accepted formats clear the gate (store errors come later)."""
    import app.core.projects as store

    seen = {}

    def _fake_find(project_id, digest):
        seen["digest"] = digest
        return None

    def _fake_add(*args, **kwargs):
        seen["stored"] = True
        return {"id": "doc-1"}

    monkeypatch.setattr(store, "find_document_by_sha", _fake_find)
    monkeypatch.setattr(store, "add_document", _fake_add)

    out = ingest_cde_bytes(
        "proj-1",
        original_name=filename,
        payload=b"probe content",
        cde_document_id="cde-1",
        vendor="acontex",
        cde_project_id="cde-proj",
        eager_index=False,
    )
    assert "doc-1" in str(out)
    assert seen.get("stored") is True
