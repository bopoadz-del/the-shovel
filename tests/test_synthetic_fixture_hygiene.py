"""Committed fixtures stay small and synthetic.

Client documents used to be committed as fixtures; they were replaced by
files generated inside the tests (``tests/_synthetic_fixtures.py``). These
checks keep that true: no fixture over 200 KB, and the one generated binary
that is still committed (the legacy .ppt sample) is exactly what its
generator writes.
"""
from __future__ import annotations

from pathlib import Path

from tests._synthetic_fixtures import build_legacy_ppt

FIXTURES = Path(__file__).parent / "fixtures"
MAX_FIXTURE_BYTES = 200 * 1024


def test_no_fixture_exceeds_200_kb():
    big = sorted(
        (str(p.relative_to(FIXTURES)), p.stat().st_size)
        for p in FIXTURES.rglob("*")
        if p.is_file() and p.stat().st_size > MAX_FIXTURE_BYTES
    )
    assert not big, f"fixtures over {MAX_FIXTURE_BYTES} bytes: {big}"


def test_committed_ppt_sample_is_the_generated_one(tmp_path):
    fresh = build_legacy_ppt(
        tmp_path / "sample.ppt",
        title="FORMATPROBE ppt",
        body="reinforced concrete pour sequence for level three.",
    )
    committed = FIXTURES / "formats" / "sample.ppt"
    assert committed.read_bytes() == fresh.read_bytes(), (
        "tests/fixtures/formats/sample.ppt drifted from build_legacy_ppt; "
        "regenerate it rather than committing an application-saved file"
    )
