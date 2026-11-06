"""Extractor-registry contracts (mutation-gate killer file for text_extractors)."""
from __future__ import annotations

from app.core.text_extractors import extractor_for, formats


def test_ifc_extractor_reads_the_census(monkeypatch, tmp_path):
    import app.core.doc_index as doc_index

    monkeypatch.setattr(doc_index, "_ifc_census", lambda path, name: ["LINE-1", "LINE-2"])
    path = str(tmp_path / "model.ifc")
    text, meta = extractor_for(".ifc").read(path, "model.ifc", False)
    assert "LINE-1" in text
    assert "LINE-2" in text
    assert not meta


def test_photos_are_not_registered():
    registry = formats()
    assert ".jpg" not in registry
    assert ".jpeg" not in registry
    assert ".pdf" in registry
    assert ".xlsx" in registry
