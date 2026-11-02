"""The battery seed uploads only DOCUMENT fixtures, not repo-control files.

The 096360e seeded battery failed with 415 "File type '' is not ingestible"
because the seed's fixture loop iterated every file in tests/battery/fixtures/,
including the ``.gitattributes`` that marks the binaries (empty extension, which
the project-documents endpoint correctly rejects). ``_fixture_paths`` now returns
only files whose extension is a known document type, so any control file is
skipped by construction (not by name).
"""
from __future__ import annotations

import pytest

from scripts.ui_battery_runner import _fixture_paths


@pytest.mark.parametrize("junk_name", [".gitattributes", "README.md", "notes.txt"])
def test_non_document_files_are_skipped(tmp_path, junk_name):
    """Whatever non-document file sits beside the fixtures -- a dotfile, a
    markdown readme, a stray text file the seed has no content-type for -- is
    excluded; only the real document fixtures are selected, sorted."""
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (tmp_path / "b.xlsx").write_bytes(b"PK\x03\x04")
    (tmp_path / junk_name).write_text("not a document")

    got = [p.name for p in _fixture_paths(tmp_path)]

    assert got == ["a.pdf", "b.xlsx"], got


def test_a_missing_fixtures_dir_is_empty(tmp_path):
    assert _fixture_paths(tmp_path / "does-not-exist") == []
