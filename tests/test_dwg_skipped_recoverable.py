"""A .dwg ingest is a recoverable skip, not a zero-chunk failure.

Decision A: do not parse DWG. Record the file so a later pass can retry it.
"""
from __future__ import annotations

import importlib

from app.core import ingest_status as ist


def test_dwg_index_is_skipped_recoverable_not_zero_chunk(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from app.core import doc_index, file_crypto
    from app.core import projects as projects_mod

    importlib.reload(doc_index)
    projects_mod._initialized = False
    projects_mod.init_db()

    path = tmp_path / "site_plan.dwg"
    file_crypto.write_document(str(path), b"AC1032 not a parsed drawing")
    proj = projects_mod.create_project("DWG skip")
    doc = projects_mod.add_document(
        proj["id"], "site_plan.dwg", file_path=str(path), size=path.stat().st_size,
    )

    result = doc_index.index_document(proj["id"], doc["id"])

    assert result.get("error") != "ZERO_CHUNK"
    assert result.get("status") != "error"
    assert result.get("skip_reason") == "skipped_recoverable"
    row = projects_mod.get_document(doc["id"])
    assert row["ingest_status"] == ist.UNSUPPORTED_TYPE
    assert row["ingest_status"] != ist.ZERO_CHUNK
    assert row["ingest_status_reason"] == "dwg:recoverable"
    assert ist.is_open(row["ingest_status"], row["ingest_status_reason"])
    saved = doc_index._load_index(proj["id"])
    assert saved["documents"] == []
    assert saved["skipped"][0]["reason"] == "skipped_recoverable"

    from scripts.p1b_ingest_drive_server import tally_key_for_index_result

    assert tally_key_for_index_result(result) == "skipped_recoverable"
    assert tally_key_for_index_result(
        {"status": "error", "error": "ZERO_CHUNK"}
    ) == "zero_chunk"
